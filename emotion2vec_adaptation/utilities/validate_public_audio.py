"""Exercise the complete adaptation lifecycle on real cached public recordings.

Run ``python emotion2vec_adaptation/utilities/validate_public_audio.py``. Configuration is
read from config/public_validation.yaml; normal training YAML files are unchanged.
The small public fixture checks software behavior, not accuracy or generalization:
these public recordings may have appeared in the released model's training data.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime
import gc
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
from uuid import uuid4
from zoneinfo import ZoneInfo

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
import yaml

from adaptation.audio import AudioDataset, AudioReader, AudioSettings, collate_audio
from adaptation.backbone import state_fingerprint
from utilities.config import PROJECT_DIR, load_settings, resolve_path, validate_settings
from adaptation.data import assert_disjoint_splits, read_manifest, read_unlabeled
from adaptation.head import HeadTrainer, load_backbone
from utilities.hardware import device_environment, resolve_precision, training_autocast
from adaptation.inference import AdaptedEmotionModel, run_prediction
from adaptation.pretrain import ContinualPretrainer
from utilities.public_fixture import create_public_fixture
from utilities.runtime import initialize_runtime, write_json
from adaptation.ssl import MaskedTeacherObjective
from utilities.smoke_test import monitoring_artifacts, release_parity


def _settings() -> tuple[dict, dict]:
    raw = yaml.safe_load((PROJECT_DIR / "config/public_validation.yaml").read_text(encoding="utf-8"))
    keys = {"output_root", "device", "precision", "cpu_threads", "head_epochs", "ssl_initial_epochs", "ssl_batch_size",
            "predictor_hidden", "run_release_parity", "parity_atol", "optional_utterance_weight"}
    if not isinstance(raw, dict) or set(raw) != {"public_validation", "fixture"}:
        raise ValueError("public_validation.yaml needs public_validation and fixture mappings")
    settings = raw["public_validation"]
    if not isinstance(settings, dict) or set(settings) != keys:
        raise ValueError("Unexpected/missing public_validation settings")
    if settings["device"] not in {"auto", "cpu", "cuda", "mps"} or settings["precision"] not in {"auto", "float32", "bfloat16"}:
        raise ValueError("Invalid public validation device or precision")
    for key in ("cpu_threads", "head_epochs", "ssl_initial_epochs", "ssl_batch_size", "predictor_hidden"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"public_validation.{key} must be a positive integer")
    if not isinstance(settings["run_release_parity"], bool) or not 0 < settings["parity_atol"] < 0.01:
        raise ValueError("Invalid public validation parity settings")
    if not 0 < settings["optional_utterance_weight"] <= 1:
        raise ValueError("optional_utterance_weight must be positive and at most one")
    return settings, raw["fixture"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _history(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise AssertionError(f"No training history in {path}")
    for row in rows:
        if "gradient_norm" in row and (not np.isfinite(row["gradient_norm"]) or row["gradient_norm"] <= 0):
            raise AssertionError("SSL must have finite, positive student gradient norms")
        if "loss" in row and not np.isfinite(row["loss"]):
            raise AssertionError("SSL loss is not finite")
    return rows


@contextmanager
def _observe_real_resume(checks: dict):
    """Invoke real loaders, then verify restored tensors before the next step.

    This does not substitute a fake model, optimizer or checkpoint. It observes
    the actual runtime restore so a successful resumed step alone cannot hide an
    accidentally reinitialized EMA teacher or Adam moments.
    """
    objective_loader = MaskedTeacherObjective.load_state_dict
    optimizer_loader = torch.optim.AdamW.load_state_dict

    def check_objective(instance, state, *args, **kwargs):
        result = objective_loader(instance, state, *args, **kwargs)
        for component in ("student", "teacher", "predictor"):
            expected = {name.removeprefix(component + "."): value for name, value in state.items()
                        if name.startswith(component + ".")}
            actual = getattr(instance, component).state_dict()
            if not expected or state_fingerprint(expected) != state_fingerprint(actual):
                raise AssertionError(f"Resume did not exactly restore {component} weights")
        checks["student_teacher_predictor_restored_exactly"] = True
        return result

    def check_optimizer(instance, state, *args, **kwargs):
        result = optimizer_loader(instance, state, *args, **kwargs)
        restored = instance.state_dict()
        if not state["state"] or set(state["state"]) != set(restored["state"]):
            raise AssertionError("Resume optimizer state is empty or incomplete")
        steps = []
        for key, expected in state["state"].items():
            actual = restored["state"][key]
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if not torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu()):
                    raise AssertionError(f"Resume optimizer {name} changed during restore")
            steps.append(int(actual["step"]))
        if min(steps) < 1:
            raise AssertionError("Resume optimizer moments came from an untrained initialization")
        checks.update(optimizer_moments_restored_exactly=True, optimizer_state_entries=len(steps),
                      restored_optimizer_step_min=min(steps), restored_optimizer_step_max=max(steps))
        return result

    with patch.object(MaskedTeacherObjective, "load_state_dict", check_objective), \
            patch.object(torch.optim.AdamW, "load_state_dict", check_optimizer):
        yield
    if not checks.get("student_teacher_predictor_restored_exactly") or not checks.get("optimizer_moments_restored_exactly"):
        raise AssertionError("Actual teacher/optimizer resume loaders were not called")


def _compare_encoder_weights(model_directory: Path, encoder_path: Path) -> dict:
    original = torch.load(model_directory / "model.pt", weights_only=True, map_location="cpu", mmap=True)["model"]
    artifact = torch.load(encoder_path, weights_only=True, map_location="cpu", mmap=True)
    exported = artifact["encoder_state"]
    frozen_count, changed_transformers = 0, 0
    for name, value in exported.items():
        before = original["d2v_model." + name]
        if not torch.isfinite(value).all():
            raise AssertionError(f"Exported encoder contains non-finite values: {name}")
        if name.startswith("modality_encoders.AUDIO.local_encoder."):
            frozen_count += 1
            if not torch.equal(before, value):
                raise AssertionError(f"Frozen CNN changed during SSL: {name}")
        if name.startswith("blocks.") and not torch.equal(before, value):
            changed_transformers += 1
    if not frozen_count or not changed_transformers:
        raise AssertionError("Expected unchanged CNN weights and updated transformer weights")
    return {"cnn_tensors_bitwise_unchanged": frozen_count, "transformer_tensors_updated": changed_transformers,
            "all_exported_tensors_finite": True, "encoder_fingerprint": artifact["fingerprint"]}


def _optional_utterance_check(cfg: dict, records, weight: float) -> dict:
    """Test optional extra-token loss on genuine variable-duration speech."""
    dataset = AudioDataset(records[:2], AudioReader(AudioSettings(**cfg["audio"])))
    batch = collate_audio([dataset[0], dataset[1]])
    lengths = (~batch["padding_mask"]).sum(dim=1).tolist()
    if lengths[0] == lengths[1]:
        raise AssertionError("Public optional SSL check needs a variable-duration batch")
    device = initialize_runtime(cfg["runtime"])
    backbone = load_backbone(cfg, device)
    objective_settings = {name: cfg["pretrain"][name] for name in (
        "mask_probability", "mask_length", "ema_decay", "target_top_layers", "predictor_hidden", "freeze_frontend")}
    objective_settings["utterance_weight"] = weight
    objective = MaskedTeacherObjective(backbone, **objective_settings).train()
    with training_autocast(device, cfg["runtime"]["precision"]):
        loss, stats = objective(batch["sources"].to(device), batch["padding_mask"].to(device))
    loss.backward()
    teacher_is_frozen = not objective.teacher.training and all(p.grad is None for p in objective.teacher.parameters())
    gradients = [p.grad for p in objective.student.parameters() if p.grad is not None]
    if not teacher_is_frozen or not gradients or not all(torch.isfinite(value).all() for value in gradients):
        raise AssertionError("Optional utterance objective has invalid teacher/student gradients")
    if not any(value.abs().sum() > 0 for value in gradients) or not np.isfinite(stats["loss"]) or stats["utterance_loss"] <= 0:
        raise AssertionError("Optional utterance loss/gradients are zero or non-finite")
    return {"objective_settings": objective_settings, "device": str(device),
            "resolved_precision": resolve_precision(device, cfg["runtime"]["precision"]),
            "samples_per_clip": lengths, "teacher_eval_and_no_gradients": True,
            "student_gradients_finite_nonzero": True, **stats}


def run_public_validation() -> Path:
    settings, fixture_settings = _settings()
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%dT%H%M%S")
    run = resolve_path(settings["output_root"]) / f"{stamp}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "real_public_audio": True, "synthetic_audio": False,
              "purpose": "Functional validation only; no model-quality or unseen-data claim.",
              "exposure_caveat": "The released encoder may have been trained on this public corpus.",
              "network_connections_blocked": True, "settings": settings, "output_directory": str(run)}
    write_json(run / "validation_report.json", report)
    try:
        with patch("socket.socket.connect", side_effect=AssertionError("Public validation forbids network connections")):
            dataset_root = create_public_fixture(run / "dataset", fixture_settings)
            report["source_provenance"] = _read_json(dataset_root / "source_provenance.json")
            cfg = load_settings("head")
            cfg["model"]["encoder_checkpoint"] = None
            cfg["runtime"].update(device=settings["device"], precision=settings["precision"],
                                  cpu_threads=settings["cpu_threads"], num_workers=0, seed=7)
            cfg["monitoring"].update(enabled=True, plot_every_steps=1, log_every_steps=1)
            cfg["labels"] = fixture_settings["labels"]
            cfg["data"].update(labeled_manifest=str(dataset_root / "labeled.csv"),
                                unlabeled_source=str(dataset_root / "unlabeled.csv"), max_train_clips=None,
                                audit_cache=str(run / "audio_audit.jsonl"))  # keep the shared cache clean
            cfg["head"].update(output_dir=str(run / "baseline_head"), epochs=settings["head_epochs"],
                                batch_size=8, patience=settings["head_epochs"])
            validate_settings(cfg)
            device = initialize_runtime(cfg["runtime"])
            report["actual_runtime"] = {"device": str(device), "requested_precision": settings["precision"],
                                        "resolved_precision": resolve_precision(device, settings["precision"]),
                                        "environment": device_environment(device)}
            labeled = read_manifest(dataset_root / "labeled.csv", labeled=True, labels=cfg["labels"])
            holdouts = [row for row in labeled if row.split in {"validation", "test"}]
            ssl_candidates = read_unlabeled(dataset_root / "unlabeled.csv")
            ssl_records = read_unlabeled(dataset_root / "unlabeled.csv", exclude_records=holdouts)
            assert_disjoint_splits({"ssl_train": ssl_records, "heldout": holdouts})
            if fixture_settings["include_holdout_probe"] and len(ssl_records) >= len(ssl_candidates):
                raise AssertionError("The deliberately inserted held-out audio copy was not excluded")
            lengths = [sf.info(row.path).duration for row in ssl_records]
            if len(set(lengths)) < 2:
                raise AssertionError("Public SSL fixture lacks variable clip lengths")
            report["data_checks"] = {"labeled_clips": len(labeled), "ssl_candidates": len(ssl_candidates),
                                      "ssl_accepted": len(ssl_records), "heldout_clips": len(holdouts),
                                      "heldout_probe_excluded": len(ssl_records) < len(ssl_candidates),
                                      "unlabeled_durations_seconds": lengths,
                                      "speaker_path_content_disjoint": True}
            if settings["run_release_parity"]:
                print("Public validation: original benchmark Predictor parity", flush=True)
                report["original_predictor_parity"] = release_parity(
                    resolve_path(cfg["model"]["directory"]), labeled[0].path, cfg["audio"], settings["parity_atol"])

            print("Public validation: baseline frozen-encoder classification head", flush=True)
            baseline_run = HeadTrainer(cfg).run()
            baseline_bundle = _read_json(baseline_run / "model_bundle.json")
            report["baseline_head"] = {"output": str(baseline_run), "frozen_encoder_fingerprint_verified": True,
                                       "monitoring": monitoring_artifacts(baseline_run),
                                       "test_metrics_functional_only": _read_json(baseline_run / "test_metrics.json")}
            gc.collect()

            ssl_cfg = load_settings("pretrain")
            for name in ("model", "runtime", "audio", "data", "labels"):
                ssl_cfg[name] = deepcopy(cfg[name])
            ssl_cfg["pretrain"].update(output_dir=str(run / "ssl_initial"), epochs=settings["ssl_initial_epochs"],
                                       batch_size=settings["ssl_batch_size"], predictor_hidden=settings["predictor_hidden"],
                                       max_steps=None, resume_checkpoint=None, save_training_state=True,
                                       keep_checkpoints=1,  # exercises checkpoint rotation with one extra encoder copy
                                       freeze_frontend=True, utterance_weight=0.0,
                                       validation_fraction=0.0, validation_manifest=None,
                                       # functional check only: the gates need more audio than this fixture has
                                       probe_steps=0, warmup_steps=0, downstream_every_steps=0)
            ssl_cfg["monitoring"].update(enabled=True, plot_every_steps=1, log_every_steps=1)
            validate_settings(ssl_cfg)
            print("Public validation: complete default SSL epoch and save teacher/optimizer", flush=True)
            initial_run = ContinualPretrainer(ssl_cfg).run()
            initial_history = _history(initial_run / "history.jsonl")
            if len(initial_history) < 2:
                raise AssertionError("Public validation requires at least two real-audio SSL optimizer steps")
            actual_records = read_manifest(initial_run / "unlabeled_train.csv", labeled=False)
            if {row.path for row in actual_records} != {row.path for row in ssl_records}:
                raise AssertionError("Actual SSL training records differ from leakage-audited fixture")
            state = torch.load(initial_run / "last_training.pt", map_location="cpu", weights_only=True, mmap=True)
            if not state["completed_epoch"] or state["step"] != len(initial_history):
                raise AssertionError("Saved training state is not at the expected epoch boundary")
            del state
            gc.collect()

            resume_cfg = deepcopy(ssl_cfg)
            resume_cfg["pretrain"].update(output_dir=str(run / "ssl_resumed"),
                                          epochs=settings["ssl_initial_epochs"] + 1,
                                          resume_checkpoint=str(initial_run / "last_training.pt"))
            resume_checks = {}
            print("Public validation: restore actual teacher, predictor and Adam moments", flush=True)
            with _observe_real_resume(resume_checks):
                resumed_run = ContinualPretrainer(resume_cfg).run()
            resumed_history = _history(resumed_run / "history.jsonl")
            if resumed_history[0]["step"] != initial_history[-1]["step"] + 1 or resumed_history[0]["epoch"] != settings["ssl_initial_epochs"] + 1:
                raise AssertionError("Resumed SSL did not continue step/epoch counters")
            encoder_path = resumed_run / "encoder.pt"
            weights_check = _compare_encoder_weights(resolve_path(cfg["model"]["directory"]), encoder_path)
            if weights_check["encoder_fingerprint"] == baseline_bundle["encoder_fingerprint"]:
                raise AssertionError("Continual pretraining did not change the exported encoder")
            report["default_ssl"] = {"initial_output": str(initial_run), "resumed_output": str(resumed_run),
                                     "initial_history": initial_history, "resumed_history": resumed_history,
                                     "resume_checks": resume_checks, "weight_checks": weights_check,
                                     "monitoring": {"initial": monitoring_artifacts(initial_run),
                                                    "resumed": monitoring_artifacts(resumed_run)},
                                     "objective_settings": {name: ssl_cfg["pretrain"][name] for name in (
                                         "mask_probability", "mask_length", "ema_decay", "target_top_layers",
                                         "predictor_hidden", "freeze_frontend", "utterance_weight",
                                         "batch_size", "learning_rate", "weight_decay")},
                                     "actual_holdout_exclusion_verified": True}
            gc.collect()

            print("Public validation: optional extra-token loss and real padded-batch gradients", flush=True)
            report["optional_utterance_objective"] = _optional_utterance_check(
                ssl_cfg, ssl_records, settings["optional_utterance_weight"])
            gc.collect()

            adapted_cfg = deepcopy(cfg)
            adapted_cfg["model"]["encoder_checkpoint"] = str(encoder_path)
            adapted_cfg["head"].update(output_dir=str(run / "adapted_head"), epochs=1, patience=1)
            print("Public validation: replacement encoder -> fresh classification head", flush=True)
            adapted_run = HeadTrainer(adapted_cfg).run()
            test_record = next(row for row in labeled if row.split == "test")
            model = AdaptedEmotionModel(adapted_run / "model_bundle.json", device)
            prediction = model.predict_file(test_record.path)
            if not np.isclose(sum(prediction["scores"].values()), 1, atol=1e-6):
                raise AssertionError("Exported single-file prediction does not sum to one")
            del model
            gc.collect()
            prediction_cfg = load_settings("predict")
            prediction_cfg["runtime"] = deepcopy(cfg["runtime"])
            prediction_cfg["predict"].update(bundle=str(adapted_run / "model_bundle.json"),
                                              source=str(adapted_run / "test.csv"),
                                              output_csv=str(run / "predictions.csv"))
            prediction_path = run_prediction(prediction_cfg)
            with prediction_path.open(newline="", encoding="utf-8") as handle:
                predicted_rows = list(csv.DictReader(handle))
            if len(predicted_rows) != sum(row.split == "test" for row in labeled):
                raise AssertionError("Batch prediction omitted public test clips")
            for row in predicted_rows:
                values = [float(row["score_" + label]) for label in cfg["labels"]]
                if not np.isfinite(values).all() or not np.isclose(sum(values), 1, atol=1e-6):
                    raise AssertionError("Batch prediction contains invalid probabilities")
            report["adapted_head_and_inference"] = {"output": str(adapted_run), "single_prediction": prediction,
                                                    "monitoring": monitoring_artifacts(adapted_run),
                                                    "batch_prediction_csv": str(prediction_path),
                                                    "batch_prediction_count": len(predicted_rows),
                                                    "test_metrics_functional_only": _read_json(adapted_run / "test_metrics.json")}
            gc.collect()

            stale = deepcopy(baseline_bundle)
            stale["head_checkpoint"] = str(baseline_run / "head.pt")
            stale["encoder_checkpoint"] = str(encoder_path)
            stale_path = run / "intentionally_mismatched_bundle.json"
            write_json(stale_path, stale)
            try:
                AdaptedEmotionModel(stale_path, device)
            except ValueError as error:
                if "mismatch" not in str(error).lower():
                    raise
                report["stale_head_rejected"] = {"passed": True, "error": str(error)}
            else:
                raise AssertionError("Old head was silently accepted with replacement encoder")
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(run / "validation_report.json", report)
        print(f"Public validation report: {run / 'validation_report.json'}", flush=True)
    return run


if __name__ == "__main__":
    run_public_validation()

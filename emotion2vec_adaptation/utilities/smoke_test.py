"""Validate both pipelines against the cached large model using synthetic audio.

Run ``python emotion2vec_adaptation/utilities/smoke_test.py``. Edit config/smoke.yaml for
controls; there are no command-line arguments. A new, uniquely named directory
contains synthetic fixtures, all model artifacts and a JSON validation receipt.
Nothing is downloaded, and a socket guard rejects attempted network connections.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import gc
import importlib.util
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
import torch
from torch.nn import functional as F
import yaml

from adaptation.audio import AudioReader, AudioSettings
from adaptation.backbone import EmotionBackbone
from utilities.config import PROJECT_DIR, REPOSITORY_ROOT, load_settings, resolve_path, validate_settings
from adaptation.data import read_manifest
from adaptation.head import HeadTrainer
from utilities.hardware import resolve_precision
from adaptation.inference import AdaptedEmotionModel
from adaptation.pretrain import ContinualPretrainer
from utilities.runtime import write_json
from utilities.generate_demo import create_demo_dataset


def release_parity(model_directory: Path, audio_path: Path, audio_settings: dict, atol: float) -> dict:
    """Check our embedding path against the repository's existing SER predictor."""
    device = torch.device("cpu")
    reader = AudioReader(AudioSettings(**audio_settings))
    backbone = EmotionBackbone.load(model_directory, device)
    checkpoint = torch.load(model_directory / "model.pt", weights_only=True, map_location="cpu", mmap=True)
    state = checkpoint["model"]
    wave = reader.read(audio_path)
    with torch.no_grad():
        logits = F.linear(backbone.embed(wave.unsqueeze(0)), state["proj.weight"], state["proj.bias"])
        probabilities = logits.softmax(dim=-1)[0].cpu().numpy()
    del checkpoint, state, backbone
    gc.collect()

    source = REPOSITORY_ROOT / "emotion_recognition_benchmark/scripts/models/emotion2vec.py"
    spec = importlib.util.spec_from_file_location("existing_emotion2vec_predictor", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    predictor = module.Predictor(model_directory, device)
    expected = predictor(str(audio_path))
    maximum_error = float(np.max(np.abs(probabilities - expected)))
    if not np.allclose(probabilities, expected, atol=atol, rtol=0):
        raise AssertionError(f"Existing benchmark predictor parity failed: max error {maximum_error}")
    del predictor
    gc.collect()
    return {"passed": True, "max_probability_absolute_error": maximum_error, "atol": atol,
            "classes": int(len(probabilities)), "audio_path": str(audio_path),
            "reference_device": "cpu", "reference_precision": "float32"}


def monitoring_artifacts(run: Path) -> dict:
    """Verify durable logs, finite metrics and the completed live plot snapshot."""
    paths = ("logs/train.log", "logs/metrics.jsonl", "monitoring/curves.png", "monitoring/dashboard.html")
    sizes = {}
    for name in paths:
        path = run / name
        if not path.is_file() or not path.stat().st_size:
            raise AssertionError(f"Training monitoring artifact is absent/empty: {path}")
        sizes[name] = path.stat().st_size
    if not (run / "monitoring/curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise AssertionError("Live training curve is not a valid PNG snapshot")
    dashboard = (run / "monitoring/dashboard.html").read_text(encoding="utf-8")
    if "completed" not in dashboard or "curves.png" not in dashboard:
        raise AssertionError("Live dashboard did not finish with a completed plot snapshot")
    rows = [json.loads(line) for line in (run / "logs/metrics.jsonl").read_text().splitlines()]
    if not rows or not any(row["event"] == "train_step" for row in rows) or not any(row["event"] == "epoch" for row in rows):
        raise AssertionError("Durable monitoring metrics lack step or epoch records")
    for row in rows:
        if not all(np.isfinite(value) for value in row["metrics"].values()):
            raise AssertionError("Monitoring emitted a non-finite scalar")
    return {"passed": True, "artifact_sizes_bytes": sizes, "metric_events": len(rows),
            "dashboard": str(run / "monitoring/dashboard.html"), "curves": str(run / "monitoring/curves.png")}


def _settings() -> dict:
    path = PROJECT_DIR / "config/smoke.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    keys = {"output_root", "cpu_threads", "head_epochs", "ssl_max_steps", "run_release_parity", "parity_atol"}
    if not isinstance(raw, dict) or set(raw) != {"smoke"} or not isinstance(raw["smoke"], dict) or set(raw["smoke"]) != keys:
        raise ValueError("config/smoke.yaml must contain exactly the documented smoke settings")
    settings = raw["smoke"]
    for name in ("cpu_threads", "head_epochs", "ssl_max_steps"):
        if isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1:
            raise ValueError(f"smoke.{name} must be a positive integer")
    if not isinstance(settings["run_release_parity"], bool) or not 0 < settings["parity_atol"] < 0.01:
        raise ValueError("Invalid smoke parity settings")
    return settings


def run_smoke() -> Path:
    """Run baseline head, unlabeled SSL, replacement head and inference checks."""
    settings = _settings()
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%dT%H%M%S")
    run = resolve_path(settings["output_root"]) / f"{stamp}_{uuid4().hex[:8]}"
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "synthetic_audio_only": True,
              "purpose": "Functional validation only; these results make no emotion-recognition accuracy claim.",
              "network_connections_blocked": True, "settings": settings, "output_directory": str(run)}
    write_json(run / "validation_report.json", report)
    try:
        with patch("socket.socket.connect", side_effect=AssertionError("Smoke test forbids network connections")):
            dataset_root = create_demo_dataset(run / "dataset")
            head_cfg = load_settings("head")
            head_cfg["model"]["encoder_checkpoint"] = None
            head_cfg["runtime"].update(device="cpu", precision="float32", cpu_threads=settings["cpu_threads"], num_workers=0, seed=7)
            head_cfg["monitoring"].update(enabled=True, plot_every_steps=1, log_every_steps=1)
            head_cfg["labels"] = ["neutral", "happy"]
            head_cfg["data"].update(labeled_manifest=str(dataset_root / "labeled.csv"),
                                     unlabeled_source=str(dataset_root / "unlabeled.csv"), max_train_clips=4,
                                     audit_cache=str(run / "audio_audit.jsonl"))  # keep the shared cache clean
            head_cfg["head"].update(output_dir=str(run / "baseline_head"), epochs=settings["head_epochs"],
                                     batch_size=4, patience=settings["head_epochs"])
            validate_settings(head_cfg)
            report["actual_runtime"] = {"device": "cpu", "resolved_precision": resolve_precision("cpu", "float32")}
            torch.set_num_threads(settings["cpu_threads"])
            records = read_manifest(dataset_root / "labeled.csv", labeled=True, labels=head_cfg["labels"])
            audio_path = records[0].path
            if settings["run_release_parity"]:
                print("Smoke: comparing with existing benchmark Predictor", flush=True)
                report["original_predictor_parity"] = release_parity(
                    resolve_path(head_cfg["model"]["directory"]), audio_path,
                    head_cfg["audio"], settings["parity_atol"],
                )

            print("Smoke: training baseline head on frozen cached-large features", flush=True)
            baseline_run = HeadTrainer(head_cfg).run()
            base_bundle = json.loads((baseline_run / "model_bundle.json").read_text())
            report["baseline_head"] = {"output": str(baseline_run), "encoder_fingerprint": base_bundle["encoder_fingerprint"],
                                       "monitoring": monitoring_artifacts(baseline_run),
                                       "test_metrics_synthetic_only": json.loads((baseline_run / "test_metrics.json").read_text())}
            gc.collect()

            ssl_cfg = load_settings("pretrain")
            for section in ("model", "runtime", "labels", "data", "audio"):
                ssl_cfg[section] = deepcopy(head_cfg[section])
            ssl_cfg["pretrain"].update(output_dir=str(run / "continued_encoder"), epochs=1, batch_size=1,
                                        max_steps=settings["ssl_max_steps"], predictor_hidden=64,
                                        save_training_state=False, resume_checkpoint=None, keep_checkpoints=0,
                                        validation_fraction=0.0, validation_manifest=None,
                                        # plumbing only: the gates are covered by unit tests and need real data
                                        probe_steps=0, warmup_steps=0, downstream_every_steps=0)
            ssl_cfg["monitoring"].update(enabled=True, plot_every_steps=1, log_every_steps=1)
            validate_settings(ssl_cfg)
            print("Smoke: performing standalone unlabeled encoder adaptation", flush=True)
            ssl_run = ContinualPretrainer(ssl_cfg).run()
            encoder_path = ssl_run / "encoder.pt"
            metadata = json.loads((ssl_run / "encoder_metadata.json").read_text())
            if metadata["initial_fingerprint"] != base_bundle["encoder_fingerprint"]:
                raise AssertionError("SSL did not initialize from the same cached encoder as baseline head")
            if metadata["fingerprint"] == metadata["initial_fingerprint"]:
                raise AssertionError("SSL optimizer step did not change any encoder weight")
            history = [json.loads(line) for line in (ssl_run / "history.jsonl").read_text().splitlines()]
            if not history or any(not np.isfinite(row["loss"]) or row["gradient_norm"] <= 0 for row in history):
                raise AssertionError("SSL did not produce finite losses and positive student gradients")
            report["unlabeled_ssl"] = {"output": str(ssl_run), "metadata": metadata, "history": history,
                                       "monitoring": monitoring_artifacts(ssl_run),
                                       "encoder_weights_changed": True, "training_state_saved": False}
            gc.collect()

            adapted_cfg = deepcopy(head_cfg)
            adapted_cfg["model"]["encoder_checkpoint"] = str(encoder_path)
            adapted_cfg["head"].update(output_dir=str(run / "adapted_head"), epochs=1, patience=1)
            print("Smoke: training a fresh head using the replacement encoder", flush=True)
            adapted_run = HeadTrainer(adapted_cfg).run()
            model = AdaptedEmotionModel(adapted_run / "model_bundle.json", torch.device("cpu"))
            prediction = model.predict_file(audio_path)
            if set(prediction["scores"]) != set(head_cfg["labels"]) or not np.isclose(sum(prediction["scores"].values()), 1, atol=1e-6):
                raise AssertionError("Exported model did not return a normalized distribution over configured labels")
            del model
            gc.collect()
            report["adapted_head"] = {"output": str(adapted_run), "prediction": prediction,
                                      "monitoring": monitoring_artifacts(adapted_run),
                                      "test_metrics_synthetic_only": json.loads((adapted_run / "test_metrics.json").read_text())}

            # A head trained on the original encoder must not silently accept a
            # replacement encoder whose feature distribution has changed.
            mismatched = deepcopy(base_bundle)
            mismatched["head_checkpoint"] = str(baseline_run / "head.pt")
            mismatched["encoder_checkpoint"] = str(encoder_path)
            mismatch_path = run / "intentionally_mismatched_bundle.json"
            write_json(mismatch_path, mismatched)
            try:
                AdaptedEmotionModel(mismatch_path, torch.device("cpu"))
            except ValueError as error:
                if "mismatch" not in str(error).lower():
                    raise
                report["mismatched_encoder_rejected"] = {"passed": True, "error": str(error)}
            else:
                raise AssertionError("An incompatible original-head/replacement-encoder pair was accepted")
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_json(run / "validation_report.json", report)
        print(f"Validation report: {run / 'validation_report.json'}", flush=True)
    return run


if __name__ == "__main__":
    run_smoke()

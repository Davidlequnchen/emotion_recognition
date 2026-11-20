"""End-to-end training checks using tiny native networks and generated WAVs."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptation.backbone import state_fingerprint
from utilities.config import load_settings, validate_settings
from adaptation.head import ClassificationHead, HeadTrainer, load_backbone
from adaptation.inference import AdaptedEmotionModel, run_prediction
from adaptation.pretrain import ContinualPretrainer
from adaptation.pretrain import (encoder_learning_rate, evaluate_unlabeled, select_unlabeled_budget,
                                 unlabeled_validation_split)
from adaptation.downstream import DownstreamProbe, effective_rank
from adaptation.head import labeled_splits
from adaptation.audio import AudioDataset, AudioReader, AudioSettings, collate_audio
from adaptation.data import read_manifest
from adaptation.ssl import MaskedTeacherObjective
from utilities.runtime import classification_metrics
from test_models import tiny_backbone


def fixture_settings(root: Path) -> dict:
    """Each split has distinct calls and nonidentical synthetic clips per class."""
    cfg = load_settings("head")
    cfg["runtime"].update(device="cpu", cpu_threads=2, num_workers=0, seed=19)
    cfg["monitoring"].update(enabled=False, console_log=False, progress_bar=False, plot_every_steps=100)
    cfg["audio"].update(min_seconds=0.001, max_seconds=0.1)
    cfg["labels"] = ["neutral", "sad"]
    cfg["data"].update(labeled_manifest=str(root / "manifest.csv"), max_train_clips=4,
                       audit_cache=str(root / "audio_audit.jsonl"))
    cfg["head"].update(output_dir=str(root / "head_run"), epochs=2, patience=2,
                       batch_size=2, learning_rate=0.01)
    rng = np.random.default_rng(15)
    rows = []
    for split in ("train", "validation", "test"):
        for class_index, label in enumerate(cfg["labels"]):
            for instance in range(2):
                clip_id = f"{split}_{label}_{instance}"
                wave = rng.normal(size=96 + 8 * instance).astype("float32") * 0.1
                wave += np.sin(np.arange(len(wave)) * (0.1 + 0.15 * class_index)).astype("float32") * 0.2
                sf.write(root / f"{clip_id}.wav", wave, 16000)
                rows.append(dict(path=f"{clip_id}.wav", label=label, group=f"{split}_{label}",
                                 split=split, clip_id=clip_id))
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split", "clip_id"])
        writer.writeheader()
        writer.writerows(rows)
    return cfg


class ConfigurationTests(unittest.TestCase):
    def test_default_labels_use_the_published_named_classes_including_other(self):
        cfg = load_settings("head")
        tokens = Path(__file__).resolve().parents[2] / "emotion_recognition_benchmark/models/emotion2vec_plus_large/tokens.txt"
        original = [token.split("/")[-1] for token in tokens.read_text().split()]
        self.assertIn("other", original)
        self.assertEqual(cfg["labels"], [label for label in original if label != "<unk>"])
        head = ClassificationHead(1024, len(cfg["labels"]))
        logits = head(torch.randn(2, 1024))
        self.assertEqual(logits.shape, (2, 8))
        torch.nn.functional.cross_entropy(logits, torch.tensor([cfg["labels"].index("other"), 0])).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))

    def test_stage_yaml_files_load_with_validated_settings(self):
        for stage in ("head", "pretrain", "predict", "labels"):
            with self.subTest(stage=stage):
                validate_settings(load_settings(stage))

    def test_misspelled_config_rejected_before_weights_are_loaded(self):
        cfg = load_settings("head")
        cfg["head"]["learnng_rate"] = cfg["head"].pop("learning_rate")
        with self.assertRaisesRegex(ValueError, "learnng_rate"):
            validate_settings(cfg)

    def test_layer_fusion_accepts_only_the_two_plan_options(self):
        cfg = load_settings("head")
        self.assertEqual(cfg["head"]["layer_fusion"], "last")
        cfg["head"]["layer_fusion"] = "mean"
        with self.assertRaisesRegex(ValueError, "layer_fusion"):
            validate_settings(cfg)

    def test_option_b_head_mixes_blocks_and_trains_the_layer_weights(self):
        head = ClassificationHead(16, 2, hidden_dim=4, num_layers=3)
        torch.testing.assert_close(head.layer_weights(), torch.full((3,), 1 / 3))
        logits = head(torch.randn(5, 3, 16))
        self.assertEqual(tuple(logits.shape), (5, 2))
        logits.sum().backward()
        self.assertGreater(float(head.layer_logits.grad.abs().sum()), 0)
        with self.assertRaisesRegex(ValueError, "Option B"):
            head(torch.randn(5, 16))

    def test_scale_settings_are_validated(self):
        for section, key, value, message in (("pretrain", "bucket_by_duration", "yes", "bucket_by_duration"),
                                             ("pretrain", "save_every_steps", -1, "save_every_steps"),
                                             ("pretrain", "keep_checkpoints", 1.5, "keep_checkpoints"),
                                             ("data", "audit_cache", 3, "audit_cache")):
            cfg = load_settings("pretrain")
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, message):
                validate_settings(cfg)
        cfg = load_settings("pretrain")
        cfg["data"]["audit_cache"] = None  # memory only
        validate_settings(cfg)

    def test_boolean_is_not_a_batch_size(self):
        cfg = load_settings("head")
        cfg["head"]["batch_size"] = True
        with self.assertRaisesRegex(ValueError, "head.batch_size"):
            validate_settings(cfg)


class MetricTests(unittest.TestCase):
    def test_metrics_use_all_configured_classes_and_true_label_rows(self):
        predictions = torch.tensor([0, 1, 1, 1])
        logits = torch.nn.functional.one_hot(predictions, num_classes=3).float()
        metrics = classification_metrics(logits, torch.tensor([0, 0, 1, 2]), ["a", "b", "c"])
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.assertEqual(metrics["support"], {"a": 2, "b": 1, "c": 1})
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["uar"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 0.5) / 3)


class HeadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_audio_to_frozen_features_to_trained_head_and_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = fixture_settings(root)
            cfg["monitoring"]["enabled"] = True
            torch.manual_seed(12)
            backbone = tiny_backbone().eval()
            original_fingerprint = backbone.fingerprint
            initial_head_states = []

            class RecordingHead(ClassificationHead):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    initial_head_states.append(deepcopy(self.state_dict()))

            with patch("adaptation.head.load_backbone", return_value=backbone), \
                 patch("adaptation.head.ClassificationHead", RecordingHead):
                output = HeadTrainer(cfg).run()

            self.assertEqual(backbone.fingerprint, original_fingerprint)
            self.assertTrue(all(parameter.grad is None for parameter in backbone.parameters()))
            self.assertTrue(all(not parameter.requires_grad for parameter in backbone.parameters()))
            artifact = torch.load(output / "head.pt", map_location="cpu", weights_only=True)
            self.assertNotEqual(state_fingerprint(artifact["state"]), state_fingerprint(initial_head_states[0]))
            self.assertEqual(artifact["encoder_fingerprint"], original_fingerprint)
            self.assertEqual(artifact["labels"], cfg["labels"])
            self.assertEqual(artifact["embedding_dim"], 16)
            self.assertIn(artifact["best_epoch"], (1, 2))
            history_rows = [json.loads(row) for row in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual(artifact["selection_metric"], "uar")
            self.assertEqual(artifact["selection_score"], max(row["uar"] for row in history_rows))
            counts = json.loads((output / "split_counts.json").read_text())
            self.assertEqual(counts["validation"], {"neutral": 2, "sad": 2})
            features = torch.load(output / "features.pt", weights_only=True)
            self.assertEqual(features["encoder_fingerprint"], original_fingerprint)
            self.assertEqual(set(features["splits"]), {"train", "validation", "test"})
            for embeddings, targets in features["splits"].values():
                self.assertEqual(embeddings.shape, (4, 16))
                self.assertEqual(targets.shape, (4,))
                self.assertFalse(embeddings.requires_grad)
            self.assertEqual(len(set(features["source_digest"].values())), 3)
            bundle = json.loads((output / "model_bundle.json").read_text())
            self.assertEqual(bundle["encoder_fingerprint"], original_fingerprint)
            self.assertEqual(bundle["head_checkpoint"], "head.pt")
            test_metrics = json.loads((output / "test_metrics.json").read_text())
            self.assertEqual(sum(test_metrics["support"].values()), 4)
            self.assertTrue(0 <= test_metrics["macro_f1"] <= 1)
            history = [json.loads(row) for row in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual(len(history), 2)
            self.assertTrue(all("validation_loss" in row for row in history))
            self.assertTrue(all("train_accuracy" in row and "validation_accuracy" in row for row in history))
            self.assertTrue((output / "logs/train.log").is_file())
            self.assertTrue((output / "logs/metrics.jsonl").is_file())
            self.assertTrue((output / "monitoring/curves.png").is_file())
            self.assertTrue((output / "monitoring/dashboard.html").is_file())

    def test_option_b_caches_every_block_and_predicts_through_the_same_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = fixture_settings(root)
            cfg["head"].update(layer_fusion="weighted", hidden_dim=4, epochs=3, patience=3)
            torch.manual_seed(12)
            backbone = tiny_backbone().eval()
            with patch("adaptation.head.load_backbone", return_value=backbone):
                output = HeadTrainer(cfg).run()
            features = torch.load(output / "features.pt", weights_only=True)
            self.assertEqual(features["layer_fusion"], "weighted")
            for embeddings, _ in features["splits"].values():
                self.assertEqual(tuple(embeddings.shape), (4, 3, 16))
            artifact = torch.load(output / "head.pt", map_location="cpu", weights_only=True)
            self.assertEqual((artifact["layer_fusion"], artifact["num_layers"]), ("weighted", 3))
            weights = json.loads((output / "layer_weights.json").read_text())["weights"]
            self.assertEqual(len(weights), 3)
            self.assertAlmostEqual(sum(weights), 1, places=3)
            self.assertNotEqual(weights, [round(1 / 3, 4)] * 3)  # the separate learning rate moved them
            bundle = json.loads((output / "model_bundle.json").read_text())
            self.assertEqual(bundle["layer_fusion"], "weighted")
            with patch("adaptation.inference.load_backbone", return_value=backbone):
                model = AdaptedEmotionModel(output / "model_bundle.json", torch.device("cpu"))
                prediction = model.predict_file(root / "test_neutral_0.wav")
            self.assertAlmostEqual(sum(prediction["scores"].values()), 1, places=6)
            bundle["layer_fusion"] = "last"
            (output / "edited_bundle.json").write_text(json.dumps(bundle))
            with patch("adaptation.inference.load_backbone", return_value=backbone), \
                 self.assertRaisesRegex(ValueError, "layer_fusion"):
                AdaptedEmotionModel(output / "edited_bundle.json", torch.device("cpu"))

    def test_macro_selection_refuses_a_validation_split_without_some_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = fixture_settings(root)
            rows = list(csv.DictReader((root / "manifest.csv").open()))
            rows = [row for row in rows if not (row["split"] == "validation" and row["label"] == "sad")]
            with (root / "manifest.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch("adaptation.head.load_backbone", return_value=tiny_backbone().eval()):
                with self.assertRaisesRegex(ValueError, "no clips of \\['sad'\\]"):
                    HeadTrainer(cfg).run()
                cfg["head"].update(selection_metric="loss", output_dir=str(root / "loss_run"))
                self.assertTrue((HeadTrainer(cfg).run() / "head.pt").is_file())

    def test_replacement_encoder_load_checks_architecture_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = load_settings("head")
            cfg["model"]["encoder_checkpoint"] = str(root / "encoder.pt")
            original = tiny_backbone().eval()
            artifact = {"format_version": 1, "model_config_hash": original.model_config_hash,
                        "encoder_state": original.encoder_state_dict(), "fingerprint": original.fingerprint}
            torch.save(artifact, root / "encoder.pt")
            replacement = tiny_backbone().eval()
            with patch("adaptation.head.EmotionBackbone.load", return_value=replacement):
                loaded = load_backbone(cfg, torch.device("cpu"))
            self.assertEqual(loaded.fingerprint, original.fingerprint)

            artifact["fingerprint"] = "different encoder content"
            torch.save(artifact, root / "encoder.pt")
            with patch("adaptation.head.EmotionBackbone.load", return_value=tiny_backbone().eval()):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    load_backbone(cfg, torch.device("cpu"))

            artifact["model_config_hash"] = "different architecture"
            torch.save(artifact, root / "encoder.pt")
            with patch("adaptation.head.EmotionBackbone.load", return_value=tiny_backbone().eval()):
                with self.assertRaisesRegex(ValueError, "architecture"):
                    load_backbone(cfg, torch.device("cpu"))


class PretrainingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def pretrain_settings(self, root):
        head_cfg = fixture_settings(root)
        cfg = load_settings("pretrain")
        for key in ("runtime", "audio", "labels", "data", "monitoring"):
            cfg[key] = deepcopy(head_cfg[key])
        cfg["data"]["unlabeled_source"] = str(root / "manifest.csv")
        cfg["pretrain"].update(output_dir=str(root / "ssl_full"), batch_size=2,
                               epochs=2, target_top_layers=2, predictor_hidden=8,
                               learning_rate=0.001, mask_length=2, utterance_weight=0.1,
                               validation_fraction=0.0, validation_manifest=None,
                               probe_steps=0, warmup_steps=0, downstream_every_steps=0)
        return cfg, head_cfg

    @staticmethod
    def initial_model_factory():
        torch.manual_seed(315)
        state = tiny_backbone().encoder_state_dict()

        def factory(*args, **kwargs):
            backbone = tiny_backbone().eval()
            backbone.load_encoder(state)
            return backbone

        return factory

    def test_ssl_validation_is_group_disjoint_fixed_mask_and_preserves_training_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.pretrain_settings(root)
            cfg["pretrain"]["validation_fraction"] = 0.5
            cfg["monitoring"]["enabled"] = True
            all_rows = read_manifest(root / "manifest.csv", labeled=False)
            training = [row for row in all_rows if row.split == "train"]
            holdouts = [row for row in all_rows if row.split != "train"]
            train, validation = unlabeled_validation_split(training, holdouts, cfg)
            self.assertEqual(len(train), 2)
            self.assertEqual(len(validation), 2)
            self.assertFalse({row.group for row in train} & {row.group for row in validation})
            reader = AudioReader(AudioSettings(**cfg["audio"]))
            loader = torch.utils.data.DataLoader(AudioDataset(validation, reader), batch_size=2, collate_fn=collate_audio)
            objective = MaskedTeacherObjective(tiny_backbone(), target_top_layers=2, predictor_hidden=8).train()
            before = torch.get_rng_state().clone()
            first = evaluate_unlabeled(objective, loader, cfg, torch.device("cpu"))
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            second = evaluate_unlabeled(objective, loader, cfg, torch.device("cpu"))
            self.assertEqual(first, second)
            self.assertTrue(objective.training)
            self.assertFalse(objective.teacher.training)
            with patch("adaptation.head.EmotionBackbone.load", side_effect=self.initial_model_factory()):
                cfg["pretrain"]["epochs"] = 1
                output = ContinualPretrainer(cfg).run()
            epoch = json.loads((output / "epoch_history.jsonl").read_text().splitlines()[0])
            self.assertTrue(0 <= epoch["validation_loss"] < float("inf"))
            self.assertTrue((output / "unlabeled_validation.csv").is_file())
            self.assertTrue((output / "monitoring/curves.png").is_file())

    def test_encoder_rate_is_zero_during_the_probe_then_warms_up(self):
        settings = {"probe_steps": 4, "warmup_steps": 2, "learning_rate": 1e-5}
        self.assertEqual([encoder_learning_rate(step, settings) for step in range(1, 8)],
                         [0.0, 0.0, 0.0, 0.0, 0.5e-5, 1e-5, 1e-5])
        self.assertEqual(encoder_learning_rate(1, {**settings, "probe_steps": 0, "warmup_steps": 0}), 1e-5)

    def gated_settings(self, root):
        cfg, head_cfg = self.pretrain_settings(root)
        cfg["pretrain"].update(validation_fraction=0.5, epochs=2, warmup_steps=0)
        return cfg, head_cfg

    def test_failed_probe_stops_the_run_and_leaves_the_encoder_untouched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.gated_settings(root)
            cfg["pretrain"].update(probe_steps=1, probe_min_gain=0.99)  # an unreachable bar
            created = []
            factory = self.initial_model_factory()
            with patch("adaptation.head.EmotionBackbone.load", side_effect=lambda *a, **k: created.append(factory()) or created[-1]):
                with self.assertRaisesRegex(RuntimeError, "not learnable"):
                    ContinualPretrainer(cfg).run()
            report = json.loads((root / "ssl_full/probe.json").read_text())
            self.assertFalse(report["passed"])
            self.assertLessEqual(report["constant_loss"], report["zero_loss"] + 1e-9)
            self.assertEqual(created[0].fingerprint, factory().fingerprint)
            self.assertFalse((root / "ssl_full/encoder.pt").exists())

    def test_probe_steps_train_only_the_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.gated_settings(root)
            cfg["pretrain"].update(probe_steps=2, max_steps=2)
            factory = self.initial_model_factory()
            with patch("adaptation.head.EmotionBackbone.load", side_effect=factory), \
                 patch.object(ContinualPretrainer, "_probe", return_value={"passed": True}):
                output = ContinualPretrainer(cfg).run()
            encoder = torch.load(output / "encoder.pt", weights_only=True)
            self.assertEqual(encoder["fingerprint"], encoder["initial_fingerprint"])
            history = [json.loads(row) for row in (output / "history.jsonl").read_text().splitlines()]
            self.assertEqual([row["learning_rate"] for row in history], [0.0, 0.0])

    def test_downstream_check_keeps_the_best_encoder_and_stops_on_a_drop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.gated_settings(root)
            cfg["pretrain"].update(probe_steps=0, downstream_every_steps=1, downstream_max_drop=0.05, epochs=3)
            scripted = iter([0.80, 0.85, 0.70, 0.90])
            def evaluate(probe, backbone):
                return {"downstream_uar": next(scripted), "downstream_macro_f1": 0.5,
                        "embedding_cosine_to_initial_median": 1.0, "embedding_cosine_to_initial_min": 1.0,
                        "embedding_spread": 1.0, "embedding_effective_rank": 2.0}
            with patch("adaptation.head.EmotionBackbone.load", side_effect=self.initial_model_factory()), \
                 patch.object(DownstreamProbe, "evaluate", evaluate):
                output = ContinualPretrainer(cfg).run()
            metadata = json.loads((output / "encoder_metadata.json").read_text())
            self.assertEqual(metadata["status"], "stopped_by_downstream_check")
            self.assertEqual(metadata["steps"], 2)  # baseline 0.80, step 1 0.85 (best), step 2 0.70 (drop 0.10)
            self.assertAlmostEqual(metadata["best_downstream_uar"], 0.85)
            best = torch.load(output / "encoder_best.pt", weights_only=True)
            self.assertEqual(best["step"], 1)
            self.assertAlmostEqual(best["downstream"]["downstream_uar"], 0.85)
            rows = [json.loads(row) for row in (output / "downstream_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1, 2])

    def test_downstream_probe_uses_its_own_random_numbers_and_never_the_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, head_cfg = self.pretrain_settings(root)
            splits = labeled_splits(head_cfg)
            probe = DownstreamProbe(splits, head_cfg, torch.device("cpu"))
            self.assertEqual(sum(len(clips) for clips in probe.clips.values()), 8)  # train 4 + validation 4, no test
            backbone = self.initial_model_factory()()
            before = torch.get_rng_state().clone()
            first = probe.evaluate(backbone)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(first, probe.evaluate(backbone))
            self.assertAlmostEqual(first["embedding_cosine_to_initial_median"], 1.0, places=5)
            self.assertTrue(0 <= first["downstream_uar"] <= 1)
        self.assertAlmostEqual(effective_rank(torch.eye(4)), 3.0, places=4)  # centring removes one direction

    def test_unlabeled_budget_is_deterministic_and_does_not_reintroduce_holdouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.pretrain_settings(root)
            rows = read_manifest(root / "manifest.csv", labeled=False)
            train = [row for row in rows if row.split == "train"]
            first = select_unlabeled_budget(train, 3, seed=9)
            second = select_unlabeled_budget(list(reversed(train)), 3, seed=9)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertTrue(set(select_unlabeled_budget(train, 2, seed=9)) <= set(first))  # nested budgets
            self.assertTrue(all(row.split == "train" for row in first))

    def test_epoch_resume_matches_continuous_training_and_export_to_head_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, head_cfg = self.pretrain_settings(root)
            factory = self.initial_model_factory()
            with patch("adaptation.head.EmotionBackbone.load", side_effect=factory):
                full = ContinualPretrainer(cfg).run()
                first_cfg = deepcopy(cfg)
                first_cfg["pretrain"].update(output_dir=str(root / "ssl_first"), epochs=1)
                first = ContinualPretrainer(first_cfg).run()
                resumed_cfg = deepcopy(cfg)
                resumed_cfg["pretrain"].update(output_dir=str(root / "ssl_resumed"),
                                                resume_checkpoint=str(first / "last_training.pt"))
                resumed = ContinualPretrainer(resumed_cfg).run()
                continuous_state = torch.load(full / "encoder.pt", weights_only=True)
                resumed_state = torch.load(resumed / "encoder.pt", weights_only=True)
                self.assertEqual(continuous_state["fingerprint"], resumed_state["fingerprint"])
                self.assertNotEqual(continuous_state["fingerprint"], continuous_state["initial_fingerprint"])
                self.assertEqual(continuous_state["step"], 4)
                for name in continuous_state["encoder_state"]:
                    torch.testing.assert_close(continuous_state["encoder_state"][name],
                                               resumed_state["encoder_state"][name], rtol=0, atol=0)
                full_training = torch.load(full / "last_training.pt", weights_only=True)
                resumed_training = torch.load(resumed / "last_training.pt", weights_only=True)
                self.assertTrue(resumed_training["completed_epoch"])
                self.assertEqual(state_fingerprint(full_training["objective"]),
                                 state_fingerprint(resumed_training["objective"]))
                self.assertTrue(any(name.startswith("teacher.") for name in resumed_training["objective"]))
                self.assertTrue(any(name.startswith("predictor.") for name in resumed_training["objective"]))
                audit = json.loads((full / "data_audit.json").read_text())
                self.assertEqual(audit["training_clips"], 4)
                self.assertEqual(audit["heldout_clips"], 8)

                head_cfg["model"]["encoder_checkpoint"] = str(resumed / "encoder.pt")
                head_output = HeadTrainer(head_cfg).run()
                model = AdaptedEmotionModel(head_output / "model_bundle.json", torch.device("cpu"))
                prediction = model.predict_file(root / "test_neutral_0.wav")
                self.assertIn(prediction["label"], head_cfg["labels"])
                self.assertEqual(set(prediction["scores"]), set(head_cfg["labels"]))
                self.assertAlmostEqual(sum(prediction["scores"].values()), 1, places=6)

                prediction_cfg = deepcopy(head_cfg)
                prediction_cfg["predict"] = {"bundle": str(head_output / "model_bundle.json"),
                                             "source": str(root / "manifest.csv"),
                                             "output_csv": str(root / "predictions.csv")}
                prediction_csv = run_prediction(prediction_cfg)
                with prediction_csv.open(newline="") as handle:
                    predictions = list(csv.DictReader(handle))
                self.assertEqual(len(predictions), 12)  # Inference retains held-out rows.
                self.assertEqual({row["clip_id"] for row in predictions},
                                 {f"{split}_{label}_{instance}" for split in ("train", "validation", "test")
                                  for label in head_cfg["labels"] for instance in (0, 1)})

                # A newly selected encoder may have the same architecture but an
                # incompatible representation space. Even editing the bundle's
                # encoder hash cannot make an old head a validated pair.
                stale = json.loads((head_output / "model_bundle.json").read_text())
                first_encoder = torch.load(first / "encoder.pt", weights_only=True)
                stale.update(encoder_checkpoint=str(first / "encoder.pt"),
                             encoder_fingerprint=first_encoder["fingerprint"])
                stale_path = head_output / "stale_bundle.json"
                stale_path.write_text(json.dumps(stale))
                with self.assertRaisesRegex(ValueError, "Head/encoder mismatch"):
                    AdaptedEmotionModel(stale_path, torch.device("cpu"))

    def test_mid_epoch_resume_is_bit_identical_and_changed_data_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.pretrain_settings(root)
            # 4 clips of 96/104 samples in batches of 3: two batches per epoch, the first cropped at a seeded offset
            cfg["pretrain"].update(batch_size=3, save_every_steps=1, keep_checkpoints=2)
            factory = self.initial_model_factory()
            with patch("adaptation.head.EmotionBackbone.load", side_effect=factory):
                full = ContinualPretrainer(cfg).run()
                continuous = torch.load(full / "encoder.pt", weights_only=True)
                self.assertEqual(continuous["step"], 4)
                self.assertNotEqual(continuous["fingerprint"], continuous["initial_fingerprint"])
                # every save keeps an encoder-only checkpoint; only the newest keep_checkpoints remain
                kept = sorted(path.name for path in (full / "checkpoints").iterdir())
                self.assertEqual(kept, ["encoder_step_00000003.pt", "encoder_step_00000004.pt"])
                self.assertEqual(torch.load(full / "checkpoints" / kept[-1], weights_only=True)["fingerprint"],
                                 continuous["fingerprint"])
                history = [json.loads(row) for row in (full / "history.jsonl").read_text().splitlines()]
                self.assertGreater(history[0]["cropped_fraction"], 0)  # 96 + 2 x 104 samples cropped to 96
                full_training = torch.load(full / "last_training.pt", weights_only=True)
                for interrupted_at in (1, 3):  # inside epoch 1 and inside epoch 2
                    with self.subTest(interrupted_at=interrupted_at):
                        first_cfg = deepcopy(cfg)
                        first_cfg["pretrain"].update(output_dir=str(root / f"stopped_{interrupted_at}"),
                                                     max_steps=interrupted_at)
                        first = ContinualPretrainer(first_cfg).run()
                        state = torch.load(first / "last_training.pt", weights_only=True)
                        self.assertEqual((state["epoch"], state["batches_consumed"], state["completed_epoch"]),
                                         ((interrupted_at + 1) // 2, 1, False))
                        resumed_cfg = deepcopy(cfg)
                        resumed_cfg["pretrain"].update(output_dir=str(root / f"resumed_{interrupted_at}"),
                                                       resume_checkpoint=str(first / "last_training.pt"))
                        resumed = ContinualPretrainer(resumed_cfg).run()
                        rows = [json.loads(row) for row in (resumed / "history.jsonl").read_text().splitlines()]
                        self.assertEqual([row["step"] for row in rows], list(range(interrupted_at + 1, 5)))
                        resumed_state = torch.load(resumed / "encoder.pt", weights_only=True)
                        for name in continuous["encoder_state"]:
                            torch.testing.assert_close(continuous["encoder_state"][name],
                                                       resumed_state["encoder_state"][name], rtol=0, atol=0)
                        resumed_training = torch.load(resumed / "last_training.pt", weights_only=True)
                        self.assertEqual(state_fingerprint(full_training["objective"]),
                                         state_fingerprint(resumed_training["objective"]))
                        epochs = [json.loads(row) for row in (resumed / "epoch_history.jsonl").read_text().splitlines()]
                        full_epochs = [json.loads(row) for row in (full / "epoch_history.jsonl").read_text().splitlines()]
                        self.assertEqual(epochs[-1]["train_loss"], full_epochs[-1]["train_loss"])
                # Replace source contents while preserving filenames and rows.
                changed = root / "train_neutral_0.wav"
                sf.write(changed, np.linspace(-0.1, 0.2, 96).astype("float32"), 16000)
                resumed_cfg["pretrain"]["output_dir"] = str(root / "changed_data_resume")
                with self.assertRaisesRegex(ValueError, "data/objective/settings"):
                    ContinualPretrainer(resumed_cfg).run()

    def test_step_metrics_audit_reuse_and_holdout_copies_found_after_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg, _ = self.pretrain_settings(root)
            # a directory source: the training clips plus a byte copy of a held-out clip under an unknown call
            for row in read_manifest(root / "manifest.csv", labeled=False):
                if row.split == "train" or row.clip_id == "test_sad_1":
                    folder = root / "unlabeled" / (row.group if row.split == "train" else "unrecognized_call")
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / row.path.name).write_bytes(row.path.read_bytes())
            cfg["data"]["unlabeled_source"] = str(root / "unlabeled")
            cfg["pretrain"].update(max_steps=1, epochs=1)
            with patch("adaptation.head.EmotionBackbone.load", side_effect=self.initial_model_factory()):
                first = ContinualPretrainer(cfg).run()
                cfg["pretrain"]["output_dir"] = str(root / "second")
                second = ContinualPretrainer(cfg).run()
            audit = json.loads((first / "data_audit.json").read_text())
            self.assertEqual((audit["training_clips"], audit["holdout_copies_dropped"]), (4, 1))
            self.assertGreater(audit["audio_audit"]["files_hashed"], 0)
            self.assertNotIn("unrecognized_call", (first / "unlabeled_train.csv").read_text())
            reused = json.loads((second / "data_audit.json").read_text())["audio_audit"]
            self.assertEqual(reused["files_hashed"], 0)  # the second run reads no file for its checks
            first_total = audit["audio_audit"]["files_hashed"] + audit["audio_audit"]["files_reused"]
            self.assertEqual(reused["files_reused"], first_total)
            row = json.loads((first / "history.jsonl").read_text().splitlines()[0])
            for key in ("data_wait_seconds", "clips_per_second", "compute_clips_per_second", "process_peak_rss_gib",
                        "cropped_fraction"):
                self.assertGreaterEqual(row[key], 0, key)
            self.assertGreaterEqual(row["compute_clips_per_second"], row["clips_per_second"])

if __name__ == "__main__":
    unittest.main()

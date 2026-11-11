"""Offline report import regressions; fixtures never call models or APIs."""
import csv
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("offline_refresh_reports", PROJECT / "scripts/evaluation/refresh_reports.py")
refresh_reports = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refresh_reports
spec.loader.exec_module(refresh_reports)


class OfflineRefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.model = Path(self.tmp.name), "fixture-model"
        self.entry = {"name": "Fixture API", "loader": "forbidden", "api": True,
                      "api_model": "fixture-v1", "training_data": [], "training_source": "not published"}
        config = self.root / "config"
        config.mkdir()
        (config / "models.yaml").write_text(yaml.safe_dump({self.model: self.entry}))
        (config / "evaluation.yaml").write_text("exclude: []\n")
        (config / "llm.yaml").write_text("prompts: config/prompts.yaml\n")
        (config / "prompts.yaml").write_text(yaml.safe_dump({"labels": ["angry", "neutral"],
                "system": "Choose {labels}.", "user": "Classify among {labels}."}))
        self.paths = refresh_reports.reporting_paths(self.root)
        self.paths.__enter__()
        self.protocol = refresh_reports.reporting.current_protocol(self.entry, ["angry", "neutral"])

    def tearDown(self):
        self.paths.__exit__(None, None, None)
        self.tmp.cleanup()

    def artifact(self, key="crema_d"):
        folder = self.root / "outputs" / self.model
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, (gold, answer, vote) in enumerate([("angry", "angry", "neutral"),
                ("angry", "neutral", "neutral"), ("neutral", "neutral", "neutral"),
                ("neutral", "angry", "angry or neutral")]):
            rows.append({"audio_path": f"data/{key}/clip{i}.wav", "acted_emotion": gold,
                "prediction": answer, "prediction_all_classes": answer, "listener_vote": vote,
                "correct": "Yes" if answer == gold else "No", "correct_acted_or_listener": "YES",
                "p_angry": int(answer == "angry"), "p_neutral": int(answer == "neutral"), "p_invalid": 0})
        meta = {"model": self.model, "model_name": "Fixture API", "key": key, "dataset": "Saved fixture scope",
            "language": "English", "classes": ["angry", "neutral"], "labels": ["angry", "neutral"],
            "model_classes": ["angry", "neutral", "invalid"], "accept": {}, "clips": 4, "total_clips": 4,
            "ground_truth": "acted_emotion", "chance": .5, "date": "2026-11-10", "inference_protocol": self.protocol,
            "notes": ["InternalCo gateway; wav24k; /Users/private/person/responses.jsonl",
                      "Up to 10 clips evaluated concurrently.", "Literal JSON string responseSchema."],
            "runtime": {"hardware": "Original client", "device": "cpu", "seconds": 12.5, "ms_per_clip": 3125},
            "metrics": {"accuracy": .5, "unweighted_accuracy": .5, "macro_f1": .5,
                        "accuracy_all_classes": .5, "accuracy_acted_or_listener": 1.,
                        "listeners_vs_acted": .333333, "model_vs_listeners": .666667}}
        path = folder / f"{key}_predictions.json"
        self.save(path, meta, rows)
        return path, meta, rows

    def save(self, path, meta, rows):
        path.write_text(json.dumps(meta))
        with path.with_suffix(".csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_bom_yes_flags_and_both_confusion_rules(self):
        path, _, _ = self.artifact()
        result = refresh_reports.validate_pair(path, self.model, self.entry)
        self.assertEqual(result.metrics["accuracy"], .5)
        self.assertEqual(result.metrics["weighted_f1"], .5)
        self.assertEqual(result.metrics["accuracy_acted_or_listener"], 1.)
        np.testing.assert_array_equal(result.matrix, [[1, 1], [1, 1]])
        np.testing.assert_array_equal(result.lenient_matrix, [[2, 0], [0, 2]])
        self.assertEqual(result.audit["probability_columns"], "one-hot answer encoding")

    def test_validation_failures_precede_all_report_writes(self):
        for problem in ("count", "duplicate", "metric", "weighted_metric", "flag", "protocol", "mass"):
            with self.subTest(problem=problem):
                path, meta, rows = self.artifact()
                if problem == "count": meta.update(clips=5, total_clips=5)
                if problem == "duplicate": rows[-1]["audio_path"] = rows[0]["audio_path"]
                if problem == "metric": meta["metrics"]["accuracy"] = .6
                if problem == "weighted_metric": meta["metrics"]["weighted_f1"] = .6
                if problem == "flag": rows[0]["correct"] = "No"
                if problem == "protocol": meta["inference_protocol"] = self.protocol | {"prompt_hash": "wrong"}
                if problem == "mass": rows[0].update(p_angry=0, p_neutral=0, p_invalid=1,
                                                     prediction_all_classes="invalid")
                self.save(path, meta, rows)
                with self.assertRaises(ValueError):
                    refresh_reports.refresh([self.model], root=self.root)
                self.assertFalse((self.root / "reports").exists())

    def test_composite_labels_match_existing_resolved_truth_f1(self):
        path, meta, _ = self.artifact()
        classes = ["angry", "happy", "surprised"]
        rows = [{"audio_path": f"data/crema_d/composite{i}.wav", "acted_emotion": truth,
                 "prediction": answer, "prediction_all_classes": answer, "correct": "Yes" if i == 0 else "No",
                 **{f"p_{label}": int(label == answer) for label in classes}}
                for i, (truth, answer) in enumerate([("excited", "surprised"),
                                                    ("excited", "angry"), ("angry", "happy")])]
        meta.update(classes=classes, labels=["excited", "angry"], model_classes=classes,
                    accept={"excited": ["happy", "surprised"]}, clips=3, total_clips=3, chance=round(5 / 9, 6),
                    metrics={"accuracy": round(1 / 3, 6), "accuracy_all_classes": round(1 / 3, 6),
                             "unweighted_accuracy": .25, "macro_f1": round(1 / 3, 6)})
        meta.pop("inference_protocol")
        self.save(path, meta, rows)
        result = refresh_reports.validate_pair(path, self.model, self.entry | {"api": False})
        self.assertEqual(result.metrics["unweighted_accuracy"], .25)
        self.assertAlmostEqual(result.metrics["macro_f1"], 1 / 3)
        self.assertAlmostEqual(result.metrics["weighted_f1"], 1 / 3)
        np.testing.assert_array_equal(result.matrix, [[1, 0, 1], [0, 1, 0]])

    def test_weighted_f1_uses_true_support_and_counts_invalids_as_errors(self):
        # 90 neutral, 10 angry: majority-only accuracy is 90%, but weighted F1 is 81/95.
        # Five invalid neutral answers stay in true support; excluding them would incorrectly inflate F1.
        for invalid in (0, 5):
            with self.subTest(invalid=invalid):
                path, meta, _ = self.artifact()
                pairs = [("neutral", "neutral")] * (90 - invalid) + [("neutral", "invalid")] * invalid \
                        + [("angry", "neutral")] * 10
                rows = [{"audio_path": f"data/crema_d/imbalanced{i}.wav", "acted_emotion": gold,
                         "prediction": answer if answer != "invalid" else "other answer",
                         "prediction_all_classes": answer, "correct": str(gold == answer),
                         **{f"p_{label}": int(label == answer) for label in meta["model_classes"]}}
                        for i, (gold, answer) in enumerate(pairs)]
                neutral_f1 = 2 * (90 - invalid) / (190 - invalid)
                meta.update(clips=100, total_clips=100,
                            metrics={"accuracy": (90 - invalid) / 100, "accuracy_all_classes": (90 - invalid) / 100,
                                     "unweighted_accuracy": round((90 - invalid) / 180, 6),
                                     "macro_f1": round(neutral_f1 / 2, 6)})
                self.save(path, meta, rows)
                result = refresh_reports.validate_pair(path, self.model, self.entry)
                self.assertNotIn("weighted_f1", result.metadata["metrics"])
                self.assertIn("weighted_f1", result.audit["derived_metrics"])
                self.assertAlmostEqual(result.metrics["weighted_f1"], .9 * neutral_f1)
                np.testing.assert_array_equal(result.resolved_support, [10, 90])
                self.assertEqual(result.other, bool(invalid))
                if not invalid:
                    self.assertAlmostEqual(result.metrics["weighted_f1"], 81 / 95)
                # A newly saved weighted metric is checked as strictly as the pre-existing metrics.
                meta["metrics"]["weighted_f1"] = round(result.metrics["weighted_f1"], 6)
                self.save(path, meta, rows)
                refresh_reports.validate_pair(path, self.model, self.entry)

    def test_weighted_f1_uses_composite_resolved_support(self):
        path, meta, _ = self.artifact()
        classes = ["angry", "happy", "surprised"]
        pairs = [("excited", "surprised"), ("excited", "surprised"), ("excited", "angry"),
                 ("angry", "happy"), ("angry", "angry")]
        rows = [{"audio_path": f"data/crema_d/composite{i}.wav", "acted_emotion": gold,
                 "prediction": answer, "prediction_all_classes": answer,
                 "correct": str(answer in (["happy", "surprised"] if gold == "excited" else [gold])),
                 **{f"p_{label}": int(label == answer) for label in classes}}
                for i, (gold, answer) in enumerate(pairs)]
        meta.update(classes=classes, labels=["angry", "excited"], model_classes=classes,
                    accept={"excited": ["happy", "surprised"]}, clips=5, total_clips=5, chance=round(8 / 15, 6),
                    metrics={"accuracy": .6, "accuracy_all_classes": .6,
                             "unweighted_accuracy": round(7 / 12, 6), "macro_f1": .5})
        meta.pop("inference_protocol")
        self.save(path, meta, rows)
        result = refresh_reports.validate_pair(path, self.model, self.entry | {"api": False})
        # Correct excited -> surprised; incorrect excited -> first accepted class happy.
        np.testing.assert_array_equal(result.resolved_support, [2, 1, 2])
        np.testing.assert_array_equal(result.f1, [.5, 0, 1])
        self.assertAlmostEqual(result.metrics["weighted_f1"], .6)

    def test_evaluation_and_offline_refresh_agree_without_a_model_or_audio_files(self):
        module = refresh_reports.reporting
        classes, labels = ["angry", "neutral"], ["angry", "neutral", "invalid"]
        pairs = [("neutral", "neutral"), ("neutral", "neutral"), ("neutral", "invalid"),
                 ("neutral", "angry"), ("angry", "neutral")]
        clips = [{"audio": self.root / f"data/crema_d/fake{i}.wav", "acted": gold, "listener": ""}
                 for i, (gold, _) in enumerate(pairs)]
        lookup = {str(clip["audio"]): np.array([int(label == answer) for label in labels])
                  for clip, (_, answer) in zip(clips, pairs)}
        predictor = mock.Mock(side_effect=lambda audio: lookup[audio])
        predictor.labels, predictor.protocol = labels, self.protocol
        predictor.notes, predictor.versions, predictor.on_cpu = [], {}, 0
        # Mock does not pretend to expose optional loader APIs such as gave_up or extra.
        del predictor.gave_up, predictor.extra
        ds = types.SimpleNamespace(key="crema_d", name="Fixture", language="English", classes=classes,
                                   clips=clips, accept={}, ground_truth="fixture truth", listener="",
                                   notes=[], caveats=[], subset="")
        (self.root / "reports" / self.model).mkdir(parents=True)
        with mock.patch.object(module, "plot_confusion"), mock.patch.object(module, "hardware", return_value="Fixture CPU"):
            module.evaluate(ds, predictor, self.model, self.entry, "", module.torch.device("cpu"), None)
        path = self.root / "outputs" / self.model / "crema_d_predictions.json"
        metadata = json.loads(path.read_text())
        self.assertAlmostEqual(metadata["metrics"]["weighted_f1"], 16 / 35, places=6)
        result = refresh_reports.validate_pair(path, self.model, self.entry)
        self.assertAlmostEqual(result.metrics["weighted_f1"], 16 / 35)
        self.assertAlmostEqual(result.metrics["macro_f1"], 2 / 7)
        self.assertEqual(result.metrics["accuracy"], .4)

    def test_listener_vocabulary_can_exceed_candidates_without_mapping_or_dropping_votes(self):
        path, meta, rows = self.artifact()
        rows[0]["listener_vote"] = rows[1]["listener_vote"] = "frustrated"
        rows[1]["correct_acted_or_listener"] = "No"
        meta["metrics"].update(accuracy_acted_or_listener=.75, model_vs_listeners=.333333)
        self.save(path, meta, rows)
        result = refresh_reports.validate_pair(path, self.model, self.entry)
        self.assertEqual(result.ds.classes, ["angry", "neutral"])
        self.assertEqual(result.records[0]["listener_vote"], "frustrated")
        self.assertAlmostEqual(result.metrics["listeners_vs_acted"], 1 / 3)
        self.assertAlmostEqual(result.metrics["model_vs_listeners"], 1 / 3)
        self.assertEqual(result.metrics["accuracy_acted_or_listener"], .75)
        self.assertEqual(result.lenient_rows, ["angry", "neutral"])
        np.testing.assert_array_equal(result.lenient_matrix, [[2, 1], [0, 1]])

    def test_later_bad_pair_leaves_no_partial_refresh(self):
        self.artifact()
        path, meta, rows = self.artifact("meld")
        meta["metrics"]["macro_f1"] = .9
        self.save(path, meta, rows)
        with mock.patch.object(refresh_reports, "render_report") as render:
            with self.assertRaises(ValueError): refresh_reports.refresh([self.model], root=self.root)
        render.assert_not_called()
        self.assertFalse((self.root / "reports").exists())

    def test_offline_refresh_preserves_inputs_runtime_and_note_privacy(self):
        path, _, _ = self.artifact()
        before = [path.read_bytes(), path.with_suffix(".csv").read_bytes()]
        module = refresh_reports.reporting
        with mock.patch.object(module, "plot_confusion") as figures, \
             mock.patch.object(module, "evaluate", side_effect=AssertionError("inference forbidden")), \
             mock.patch.object(module, "main", side_effect=AssertionError("evaluation main forbidden")), \
             mock.patch.object(module, "loader_note", side_effect=AssertionError("model import forbidden")):
            reports = refresh_reports.refresh([self.model], root=self.root)
        self.assertEqual(figures.call_count, 2)
        text = reports[0].read_text()
        for expected in ("Original client", "not measured by this refresh", "24 kHz",
                         "not individual request latency", "Metrics can be recomputed"):
            self.assertIn(expected, text)
        self.assertNotIn("InternalCo", text)
        self.assertNotIn("/Users/private", text)
        self.assertTrue((reports[0].parent / "crema_d_offline_audit.json").exists())
        self.assertTrue((self.root / "reports/summary.json").exists())
        self.assertEqual([path.read_bytes(), path.with_suffix(".csv").read_bytes()], before)
        self.assertIn("Weighted F1", text)
        per_model = json.loads((self.root / "reports" / self.model / "summary.json").read_text())
        self.assertEqual(per_model["datasets"][0]["metrics"]["weighted_f1"], .5)
        comparison = json.loads((self.root / "reports/summary.json").read_text())
        self.assertEqual(comparison["results"][self.model]["crema_d"]["weighted_f1"], .5)
        self.assertIn("## Weighted F1", (self.root / "reports/summary.md").read_text())

    def test_all_models_and_audit_overlay_remain_valid_only_for_unchanged_sources(self):
        path, _, _ = self.artifact()
        before = [path.read_bytes(), path.with_suffix(".csv").read_bytes()]
        with mock.patch.object(refresh_reports.reporting, "plot_confusion"):
            refresh_reports.refresh(None, root=self.root)
        module = refresh_reports.reporting
        self.assertTrue((self.root / "reports/derived_metrics.json").exists())
        # A single index suffices; individual import audits are not needed to display derived scores.
        (self.root / "reports" / self.model / "crema_d_offline_audit.json").unlink()
        self.assertEqual(module.best_runs(self.model, entry=self.entry)["crema_d"]["metrics"]["weighted_f1"], .5)
        self.assertEqual([path.read_bytes(), path.with_suffix(".csv").read_bytes()], before)
        # The JSON can change without changing its labels/metrics. Its old audit must still be rejected.
        path.write_bytes(before[0] + b"\n")
        self.assertNotIn("weighted_f1", module.best_runs(self.model, entry=self.entry)["crema_d"]["metrics"])
        path.write_bytes(before[0])
        self.assertEqual(module.best_runs(self.model, entry=self.entry)["crema_d"]["metrics"]["weighted_f1"], .5)
        csv_path = path.with_suffix(".csv")
        csv_path.write_bytes(before[1] + b"\n")
        self.assertNotIn("weighted_f1", module.best_runs(self.model, entry=self.entry)["crema_d"]["metrics"])

    def test_invalid_audit_does_not_supply_weighted_f1(self):
        path, meta, _ = self.artifact()
        audit_path = self.root / "reports" / self.model / "crema_d_offline_audit.json"
        audit_path.parent.mkdir(parents=True)
        for value in ("{bad JSON", "[]", json.dumps({"model": self.model, "key": "crema_d", "clips": 4,
                "recomputed_metrics": {"weighted_f1": True}, "source_artifacts": []})):
            with self.subTest(audit=value):
                audit_path.write_text(value)
                self.assertEqual(refresh_reports.reporting.load_reported_metrics(path, meta), meta["metrics"])
        index_path = self.root / "reports/derived_metrics.json"
        for value in ("{bad JSON", "[]", '{"runs": []}'):
            with self.subTest(index=value):
                index_path.write_text(value)
                self.assertEqual(refresh_reports.reporting.load_reported_metrics(path, meta), meta["metrics"])

    def test_optional_outputs_only_dispatch_to_plot_and_viewer_entrypoints(self):
        self.artifact()
        plots, viewer = mock.Mock(), mock.Mock()
        with mock.patch.object(refresh_reports, "ROOT", self.root.resolve()), \
             mock.patch.object(refresh_reports.reporting, "plot_confusion"), \
             mock.patch.dict(sys.modules, {"plot_results": types.SimpleNamespace(main=plots),
                                           "make_viewer": types.SimpleNamespace(main=viewer)}):
            refresh_reports.refresh([self.model], root=self.root, plots=True, viewer=True)
        plots.assert_called_once_with()
        viewer.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

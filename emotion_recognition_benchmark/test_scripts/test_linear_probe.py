"""CPU-only regression tests for frozen, group-disjoint linear probes.

Run without model weights, audio, network, or a GPU:
    python -m unittest discover -s test_scripts -p 'test_linear_probe.py' -v

All feature stores and exported artifacts are synthetic and temporary.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "analysis"))
sys.path.insert(0, str(PROJECT / "scripts" / "llm_eval"))

import linear_probe as probe
from probe_core import fit_linear_probe, make_group_splits, score_predictions
from probe_metadata import infer_groups
from prompts import configured_protocol


CLASSES = ["angry", "happy", "sad"]
MODEL = "fixture_model"
MODEL_ID = "fixture/audio-model"
PROTOCOL = {"label_scope": "dataset", "labels": CLASSES,
            "prompt_hash": "0123456789ab", "model_id": MODEL_ID}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    return {str(path.relative_to(root)): file_hash(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


class TemporaryFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ser_linear_probe_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def feature_store(self, protocol=None, model=MODEL, dataset="ravdess", rows=None, vectors=None):
        protocol = dict(protocol or PROTOCOL)
        folder = self.root / "features" / model / dataset / protocol["prompt_hash"]
        folder.mkdir(parents=True)
        if rows is None:
            rows = [{"row": index, "audio_path": f"data/{dataset}/clip_{index}.wav",
                     "pooling": "audio", "tokens": 4 + index} for index in range(6)]
        if vectors is None:
            vectors = np.arange(len(rows) * 2 * 3, dtype=np.float16).reshape(len(rows), 2, 3)
        meta = {"model": model, "hf_repo": protocol.get("model_id", MODEL_ID),
                "prompts": protocol["prompt_hash"], "labels": protocol["labels"],
                "label_scope": protocol["label_scope"], "layers": [2, 4], "num_layers": 4,
                "row_shape": [2, 3], "dtype": "float16", "pooling": "audio-token mean"}
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        self.write_rows(folder, rows)
        (folder / "hidden.f16").write_bytes(np.asarray(vectors, dtype=np.float16).tobytes())
        return folder

    @staticmethod
    def write_rows(folder, rows):
        (folder / "index.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def update_meta(folder, **changes):
        path = folder / "meta.json"
        meta = json.loads(path.read_text())
        meta.update(changes)
        path.write_text(json.dumps(meta), encoding="utf-8")

    @staticmethod
    def read_store(folder, protocol=None, **kwargs):
        return probe.read_feature_store(folder, dict(protocol or PROTOCOL), MODEL, **kwargs)


class StrictFeatureStoreTests(TemporaryFixture):
    def test_load_is_readonly_and_preserves_inputs(self):
        folder = self.feature_store()
        before = snapshot(folder)
        vectors, rows, meta = self.read_store(folder)
        self.assertIsInstance(vectors, np.memmap)
        self.assertFalse(vectors.flags.writeable)
        self.assertEqual(vectors.shape, (6, 2, 3))
        self.assertEqual(vectors.dtype, np.dtype("float16"))
        self.assertEqual([row["row"] for row in rows], list(range(6)))
        self.assertEqual(meta["layers"], [2, 4])
        np.testing.assert_array_equal(vectors[0], np.arange(6).reshape(2, 3))
        self.assertEqual(snapshot(folder), before)

    def test_rejects_wrong_dtype_shape_layer_or_protocol_metadata(self):
        cases = [
            {"dtype": "float32"}, {"row_shape": [3, 2]}, {"row_shape": [2, 0]},
            {"layers": [2, 2]}, {"layers": [2]}, {"layers": [2, 5]},
            {"layers": [[2], 4]}, {"num_layers": "4"},
            {"prompts": "different_hash"}, {"labels": ["angry", "sad"]},
            {"label_scope": "global"}, {"model": "another_model"},
            {"hf_repo": "another/repository"},
        ]
        for index, changes in enumerate(cases):
            with self.subTest(changes=changes):
                original = self.root
                self.root = original / f"case_{index}"
                try:
                    store = self.feature_store()
                finally:
                    self.root = original
                self.update_meta(store, **changes)
                before = snapshot(store)
                with self.assertRaises(ValueError):
                    self.read_store(store)
                self.assertEqual(snapshot(store), before)

    def test_rejects_noncontiguous_duplicate_or_aliased_index_rows(self):
        for problem in ("row", "duplicate_audio", "alias_audio", "malformed_json"):
            with self.subTest(problem=problem):
                rows = [{"row": i, "audio_path": f"data/ravdess/clip_{i}.wav",
                         "pooling": "audio", "tokens": 3} for i in range(3)]
                if problem == "row":
                    rows[1]["row"] = 2
                elif problem == "duplicate_audio":
                    rows[1]["audio_path"] = rows[0]["audio_path"]
                elif problem == "alias_audio":
                    rows[1]["audio_path"] = "data/ravdess/./clip_0.wav"
                original = self.root
                self.root = original / problem
                try:
                    store = self.feature_store(rows=rows)
                finally:
                    self.root = original
                if problem == "malformed_json":
                    with (store / "index.jsonl").open("a") as stream:
                        stream.write('{"row":')
                before = snapshot(store)
                with self.assertRaises(ValueError):
                    self.read_store(store)
                self.assertEqual(snapshot(store), before)

    def test_rejects_partial_or_unindexed_binary_rows_without_repairing(self):
        for change in ("partial", "extra_row", "missing_row"):
            with self.subTest(change=change):
                original = self.root
                self.root = original / change
                try:
                    store = self.feature_store()
                finally:
                    self.root = original
                path = store / "hidden.f16"
                data = path.read_bytes()
                path.write_bytes(data + b"\x00" if change == "partial" else
                                 data + bytes(12) if change == "extra_row" else data[:-12])
                before = snapshot(store)
                with self.assertRaises(ValueError):
                    self.read_store(store)
                self.assertEqual(snapshot(store), before)

    def test_audio_pooling_requirement_can_only_be_relaxed_explicitly(self):
        store = self.feature_store()
        rows = [json.loads(line) for line in (store / "index.jsonl").read_text().splitlines()]
        rows[1]["pooling"] = "prompt"
        self.write_rows(store, rows)
        with self.assertRaises(ValueError):
            self.read_store(store)
        vectors, loaded, _ = self.read_store(store, require_audio_pooling=False)
        self.assertEqual(loaded[1]["pooling"], "prompt")
        self.assertEqual(len(vectors), len(rows))

    def test_rejects_nonfinite_features(self):
        vectors = np.ones((6, 2, 3), dtype=np.float16)
        vectors[2, 0, 1] = np.nan
        store = self.feature_store(vectors=vectors)
        with self.assertRaises(ValueError):
            self.read_store(store)


class GroupSplitTests(unittest.TestCase):
    def fixtures(self):
        triples = [(f"clip_{group}_{label}_{repeat}", label, f"speaker_{group}")
                   for group in range(6) for label in CLASSES for repeat in range(2)]
        return tuple(map(list, zip(*triples)))

    def test_splits_are_deterministic_group_disjoint_and_row_order_independent(self):
        paths, labels, groups = self.fixtures()
        first = make_group_splits(paths, labels, groups, 42, group_counts=[2, 2, 2])
        second = make_group_splits(paths[::-1], labels[::-1], groups[::-1], 42,
                                   group_counts=[2, 2, 2])
        self.assertEqual(first, second)
        group_splits = {}
        for path, label, group in zip(paths, labels, groups):
            group_splits.setdefault(group, set()).add(first[path])
        self.assertTrue(all(len(values) == 1 for values in group_splits.values()))
        for split in ("train", "validation", "test"):
            self.assertEqual({label for path, label in zip(paths, labels) if first[path] == split},
                             set(CLASSES))
            self.assertEqual(sum(next(iter(values)) == split for values in group_splits.values()), 2)

    def test_two_speakers_cannot_fall_back_to_random_clip_splitting(self):
        with self.assertRaises(ValueError):
            make_group_splits(["a", "b", "c", "d"], ["happy", "sad"] * 2,
                              ["speaker1"] * 2 + ["speaker2"] * 2, seed=42)

    def test_duplicate_paths_and_classes_in_too_few_groups_are_rejected(self):
        paths, labels, groups = self.fixtures()
        with self.assertRaises(ValueError):
            make_group_splits([paths[0], *paths[1:-1], paths[0]], labels, groups, 42)
        rare_labels = ["rare" if group in ("speaker_0", "speaker_1") and label == "sad" else label
                       for label, group in zip(labels, groups)]
        with self.assertRaises(ValueError):
            make_group_splits(paths, rare_labels, groups, 42)


class GroupIdentityTests(TemporaryFixture):
    def test_ravdess_actor_is_shared_across_emotion_and_repetition(self):
        paths = ["data/ravdess/audio/Actor_01/03-01-01-01-01-01-01.wav",
                 "data/ravdess/audio/Actor_01/03-01-03-01-01-02-01.wav"]
        result = infer_groups("ravdess", paths, self.root)
        self.assertEqual({row["group"] for row in result.values()}, {"ravdess:speaker:01"})
        with self.assertRaises(ValueError):
            infer_groups("ravdess", [paths[0].replace("Actor_01", "Actor_02")], self.root)

    def test_tess_known_typo_does_not_create_a_third_speaker(self):
        paths = ["data/tess/OA_bite_neutral.wav", "data/tess/OAF_food_fear.wav",
                 "data/tess/YAF_food_fear.wav"]
        result = infer_groups("tess", paths, self.root)
        self.assertEqual(result[paths[0]]["group"], result[paths[1]]["group"])
        self.assertEqual(len({row["group"] for row in result.values()}), 2)

    def test_meld_records_dialogue_identity_and_its_scope_limitation(self):
        paths = ["data/meld/audio/dia1_utt0.wav", "data/meld/audio/dia1_utt1.wav",
                 "data/meld/audio/dia2_utt0.wav"]
        result = infer_groups("meld", paths, self.root)
        self.assertEqual(result[paths[0]]["group"], result[paths[1]]["group"])
        self.assertEqual(result[paths[0]]["group_kind"], "dialogue")
        self.assertIn("does not guarantee speaker independence", result[paths[0]]["source"])

    def test_thai_metadata_groups_different_microphones_by_actor(self):
        metadata = self.root / "data" / "thai_ser" / "labels.csv"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("audio_id,actor_id\nmic1.flac,007\nmic2.flac,007\n", encoding="utf-8")
        paths = ["data/thai_ser/audio/mic1.flac", "data/thai_ser/audio/mic2.flac"]
        result = infer_groups("thai_ser", paths, self.root)
        self.assertEqual({row["group"] for row in result.values()}, {"thai_ser:speaker:007"})
        with self.assertRaises(ValueError):
            infer_groups("thai_ser", ["data/thai_ser/audio/missing.flac"], self.root)

    def test_parent_traversal_cannot_bypass_dataset_identity_validation(self):
        with self.assertRaises(ValueError):
            infer_groups("ravdess", ["data/ravdess/../tess/OAF_food_fear.wav"], self.root)


class LinearFitTests(unittest.TestCase):
    @staticmethod
    def arrays():
        train = np.array([[-3, 0], [-2, 1], [-1, 0], [0, 3], [1, 2], [0, 1],
                          [3, 0], [2, -1], [1, 0]], dtype=np.float64)
        labels = ["sad"] * 3 + ["angry"] * 3 + ["happy"] * 3
        validation = train[[0, 3, 6]] + [0.1, -0.1]
        return train, labels, validation, ["sad", "angry", "happy"]

    def test_scaler_and_classifier_fit_only_training_rows(self):
        train, truth, validation, val_truth = self.arrays()
        fitted = fit_linear_probe(train, truth, validation + 100, val_truth,
                                  ["sad", "angry", "happy"], [.01, 1, 10])
        scaler = fitted["model"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
        np.testing.assert_allclose(scaler.var_, train.var(axis=0))
        self.assertEqual(scaler.n_samples_seen_, len(train))
        classifier = fitted["model"].named_steps["classifier"]
        self.assertEqual(classifier.solver, "lbfgs")
        self.assertEqual(classifier.coef_.shape, (3, 2))

    def test_C_selection_uses_validation_uar_with_stable_ties(self):
        train, truth, validation, val_truth = self.arrays()
        fitted = fit_linear_probe(train, truth, validation, val_truth,
                                  ["sad", "angry", "happy"], [.01, .1, 1, 10])
        search = fitted["validation_search"]
        winner = max(search, key=lambda row: row["validation_uar"])
        self.assertEqual(fitted["selected_C"], winner["C"])
        self.assertEqual(fitted["model"].named_steps["classifier"].C, winner["C"])
        self.assertEqual([row["C"] for row in search], [.01, .1, 1, 10])

    def test_probability_columns_match_explicit_nonlexical_class_order(self):
        train, truth, validation, val_truth = self.arrays()
        order = ["sad", "angry", "happy"]
        fitted = fit_linear_probe(train, truth, validation, val_truth, order, [1])
        probabilities = fitted["model"].predict_proba(validation)[:, fitted["probability_column_indices"]]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(np.asarray(order)[probabilities.argmax(axis=1)].tolist(), val_truth)
        metrics = score_predictions(val_truth, probabilities, order)
        self.assertEqual(metrics["classes"], order)
        self.assertEqual(metrics["confusion_matrix"], np.eye(3, dtype=int).tolist())
        self.assertEqual(metrics["uar"], 1.0)
        self.assertLess(metrics["log_loss"], 0.7)

    def test_missing_training_class_and_nonfinite_values_are_rejected(self):
        train, truth, validation, val_truth = self.arrays()
        with self.assertRaises(ValueError):
            fit_linear_probe(train[:6], truth[:6], validation, val_truth, CLASSES, [1])
        train[0, 0] = np.inf
        with self.assertRaises(ValueError):
            fit_linear_probe(train, truth, validation, val_truth, CLASSES, [1])

    def test_probability_validation_and_class_order_affect_actual_scoring(self):
        order = ["sad", "angry", "happy"]
        probabilities = np.array([[.7, .2, .1], [.1, .8, .1], [.1, .1, .8]])
        metrics = score_predictions(order, probabilities, order)
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["brier_score"], (.14 + .06 + .06) / 3)
        for invalid in (probabilities * .5, probabilities[:, :2], np.full((3, 3), np.nan)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score_predictions(order, invalid, order)


class PipelineIntegrationTests(TemporaryFixture):
    def project_fixture(self, models=(MODEL, "fixture_second")):
        config_dir = self.root / "config"
        config_dir.mkdir()
        prompts = config_dir / "prompts.yaml"
        prompts.write_text(yaml.safe_dump({"labels": CLASSES,
            "system": "Choose one emotion from {labels}.",
            "user": "Listen to the audio and answer with {labels}."}), encoding="utf-8")
        llm = config_dir / "llm.yaml"
        llm.write_text(yaml.safe_dump({"prompts": "config/prompts.yaml"}), encoding="utf-8")
        registry = {model: {"llm": True, "name": model, "loader": "fake_unused",
                            "hf_repo": f"fixture/{model}", "training_data": []}
                    for model in models}
        (config_dir / "models.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
        paths, labels, rows = [], [], []
        for speaker in range(1, 7):
            for label_index, label in enumerate(CLASSES):
                for repeat in (1, 2):
                    paths.append(f"data/ravdess/audio/Actor_{speaker:02d}/"
                                 f"03-01-0{label_index + 1}-01-01-0{repeat}-{speaker:02d}.wav")
                    labels.append(label)
                    rows.append({"row": len(rows), "audio_path": paths[-1],
                                 "pooling": "audio", "tokens": 12})
        vectors = np.zeros((len(rows), 2, 3), dtype=np.float16)
        for index, label in enumerate(labels):
            vectors[index, 1, CLASSES.index(label)] = 4
        for model in models:
            protocol = configured_protocol(registry[model], CLASSES, config=llm, root=self.root)
            self.feature_store(protocol=protocol, model=model, rows=rows, vectors=vectors)
            output = self.root / "outputs" / model
            output.mkdir(parents=True)
            metadata = {"key": "ravdess", "dataset": "Fixture RAVDESS", "model": model,
                        "clips": len(rows), "total_clips": len(rows), "classes": CLASSES,
                        "labels": CLASSES, "accept": {}, "inference_protocol": protocol}
            (output / "ravdess_predictions.json").write_text(json.dumps(metadata), encoding="utf-8")
            with (output / "ravdess_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["audio_path", "acted_emotion", "prediction",
                    "prediction_all_classes", "correct", *[f"token_p_{label}" for label in CLASSES]])
                writer.writeheader()
                for path, label in zip(paths, labels):
                    writer.writerow({"audio_path": path, "acted_emotion": label, "prediction": label,
                        "prediction_all_classes": label, "correct": True,
                        **{f"token_p_{name}": 1.0 if name == label else 0.0 for name in CLASSES}})
        config = {"models": list(models), "datasets": ["ravdess"], "seeds": [42], "layers": "all",
                  "c_values": [.01, 1], "class_weight": "balanced", "max_iter": 1000,
                  "train_validation_test": [.6, .2, .2], "group_counts": {"ravdess": [2, 2, 2]},
                  "require_current_protocol": True, "require_audio_pooling": True,
                  "export_model_parameters": True}
        return config, paths, labels

    @staticmethod
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def test_two_models_reuse_frozen_splits_and_export_selected_predictions(self):
        config, paths, _ = self.project_fixture()
        before = snapshot(self.root)
        output = self.root / "exports"
        result = probe.run(config, self.root, output_dir=output)
        self.assertIsInstance(result, dict)
        self.assertTrue((output / "summary.json").is_file())
        self.assertTrue((output / "summary.csv").is_file())
        manifest = self.read_csv(output / "splits" / "ravdess" / "seed_42.csv")
        self.assertEqual({row["audio_path"] for row in manifest}, set(paths))
        assignment = {row["audio_path"]: row["split"] for row in manifest}
        paired = []
        for model in config["models"]:
            base = output / model / "ravdess" / "seed_42"
            selected = json.loads((base / "selected.json").read_text())
            self.assertEqual(selected["layer"], 4)
            self.assertEqual(selected["C"], .01)
            self.assertEqual(len(selected["manifest_sha256"]), 64)
            predictions = self.read_csv(base / "selected_predictions.csv")
            self.assertEqual({row["audio_path"] for row in predictions}, set(paths))
            self.assertEqual({row["audio_path"]: row["split"] for row in predictions}, assignment)
            paired.append({row["audio_path"]: row["split"] for row in predictions})
            for row in predictions:
                probabilities = [float(row[f"probe_p_{label}"]) for label in CLASSES]
                self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
                self.assertIn(row["prediction"], CLASSES)
                self.assertTrue(row["group"])
            for layer in (2, 4):
                directory = base / f"layer_{layer}"
                for filename in ("predictions.csv", "metrics.json", "config.json", "model_parameters.json"):
                    self.assertTrue((directory / filename).is_file(), str(directory / filename))
                layer_config = json.loads((directory / "config.json").read_text())
                self.assertEqual(layer_config["manifest_sha256"], selected["manifest_sha256"])
                self.assertEqual(layer_config["scaler_fit"], "train_only")
                self.assertEqual(len(layer_config["validation_search"]), len(config["c_values"]))
                provenance = layer_config["provenance"]
                self.assertEqual(len(provenance["feature_sha256"]), 64)
                self.assertTrue(provenance["coverage"]["complete"])
            coverage = json.loads((output / model / "ravdess" / "coverage.json").read_text())
            self.assertEqual(coverage["prediction_clips"], len(paths))
            self.assertEqual(coverage["feature_rows"], len(paths))
            self.assertEqual(coverage["pooling_counts"], {"audio": len(paths)})
            self.assertEqual(coverage["missing_audio_paths"], [])
        self.assertEqual(*paired)
        self.assertEqual({name: file_hash(self.root / name) for name in before}, before)
        self.assertFalse(any(path.suffix == ".f16" for path in output.rglob("*")))

    def test_existing_output_is_never_overwritten(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        output = self.root / "exports"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("previous result", encoding="utf-8")
        with self.assertRaises((ValueError, FileExistsError)):
            probe.run(config, self.root, output_dir=output)
        self.assertEqual(sentinel.read_text(), "previous result")

    def test_test_feature_perturbations_cannot_change_layer_C_or_training_parameters(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        first_output = self.root / "first_export"
        first = probe.run(config, self.root, output_dir=first_output)
        self.assertFalse(first["failed"])
        manifest = self.read_csv(first_output / "splits" / "ravdess" / "seed_42.csv")
        test_paths = {row["audio_path"] for row in manifest if row["split"] == "test"}
        store = next((self.root / "features" / MODEL / "ravdess").iterdir())
        indices = [json.loads(line) for line in (store / "index.jsonl").read_text().splitlines()]
        data = np.fromfile(store / "hidden.f16", dtype=np.float16).reshape(len(indices), 2, 3)
        for row in indices:
            if row["audio_path"] in test_paths:
                data[row["row"]] = np.array([[100, -100, 50], [-40, 30, 100]], dtype=np.float16)
        (store / "hidden.f16").write_bytes(data.tobytes())
        second_output = self.root / "second_export"
        second = probe.run(config, self.root, output_dir=second_output)
        self.assertFalse(second["failed"])
        base = Path(MODEL) / "ravdess" / "seed_42"
        first_selected = json.loads((first_output / base / "selected.json").read_text())
        second_selected = json.loads((second_output / base / "selected.json").read_text())
        for key in ("layer", "C", "validation_uar", "manifest_sha256"):
            self.assertEqual(first_selected[key], second_selected[key])
        self.assertNotEqual(first_selected["test_accuracy"], second_selected["test_accuracy"])
        for layer in (2, 4):
            relative = base / f"layer_{layer}" / "model_parameters.json"
            self.assertEqual(json.loads((first_output / relative).read_text()),
                             json.loads((second_output / relative).read_text()))

    def test_incomplete_model_is_failed_without_shrinking_other_models_manifest(self):
        config, paths, _ = self.project_fixture()
        second_model = config["models"][1]
        store = next((self.root / "features" / second_model / "ravdess").iterdir())
        rows = [json.loads(line) for line in (store / "index.jsonl").read_text().splitlines()]
        self.write_rows(store, rows[:-1])
        data_path = store / "hidden.f16"
        data_path.write_bytes(data_path.read_bytes()[:-12])
        output = self.root / "exports"
        result = probe.run(config, self.root, output_dir=output)
        self.assertEqual(result["status"], "partial")
        self.assertEqual({row["model"] for row in result["results"]}, {MODEL})
        self.assertTrue(any(row.get("model") == second_model and "coverage" in row["reason"]
                            for row in result["failed"]))
        manifest = self.read_csv(output / "splits" / "ravdess" / "seed_42.csv")
        self.assertEqual(len(manifest), len(paths))
        coverage = json.loads((output / second_model / "ravdess" / "coverage.json").read_text())
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missing_audio_paths"], [paths[-1]])
        self.assertEqual(coverage["extra_audio_paths"], [])

    def test_default_tess_and_official_test_only_meld_are_explicitly_skipped(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        config["datasets"] = ["tess", "meld"]
        output = self.root / "exports"
        result = probe.run(config, self.root, output_dir=output)
        self.assertEqual({row["dataset"] for row in result["skipped"]}, {"tess", "meld"})
        self.assertFalse(result["results"])
        self.assertFalse(result["failed"])
        reasons = {row["dataset"]: row["reason"] for row in result["skipped"]}
        self.assertIn("two speakers", reasons["tess"])
        self.assertIn("official test", reasons["meld"])

    def test_dry_run_validates_without_writing_exports_or_changing_inputs(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        before = snapshot(self.root)
        output = self.root / "exports"
        result = probe.run(config, self.root, output_dir=output, dry_run=True)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(len(result["ready"]), 1)
        self.assertFalse(result["failed"])
        self.assertFalse(output.exists())
        self.assertEqual(snapshot(self.root), before)

    def test_explicit_meld_dialogue_manifest_is_labeled_diagnostic_only(self):
        config, old_paths, labels = self.project_fixture(models=(MODEL,))
        output = self.root / "outputs" / MODEL
        metadata = json.loads((output / "ravdess_predictions.json").read_text())
        metadata["key"] = "meld"
        metadata["dataset"] = "Fixture MELD official test"
        paths = []
        for index, path in enumerate(old_paths):
            speaker = int(Path(path).parent.name.removeprefix("Actor_"))
            paths.append(f"data/meld/audio/dia{speaker}_utt{index % 6}.wav")
        rows = [{"row": index, "audio_path": path, "pooling": "audio", "tokens": 12}
                for index, path in enumerate(paths)]
        vectors = np.zeros((len(rows), 2, 3), dtype=np.float16)
        for index, label in enumerate(labels):
            vectors[index, 1, CLASSES.index(label)] = 4
        self.feature_store(protocol=metadata["inference_protocol"], model=MODEL, dataset="meld",
                           rows=rows, vectors=vectors)
        (output / "meld_predictions.json").write_text(json.dumps(metadata), encoding="utf-8")
        with (output / "meld_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_path", "acted_emotion", "prediction"])
            writer.writeheader()
            for path, label in zip(paths, labels):
                writer.writerow({"audio_path": path, "acted_emotion": label, "prediction": label})
        groups = infer_groups("meld", paths, self.root)
        manifest = self.root / "meld_dialogues.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_path", "group", "split"])
            writer.writeheader()
            for index, path in enumerate(paths):
                writer.writerow({"audio_path": path, "group": groups[path]["group"],
                                 "split": ("train", "validation", "test")[index // 12]})
        config["datasets"] = ["meld"]
        config["split_manifests"] = {"meld": str(manifest)}
        config["allow_test_only_diagnostics"] = True
        destination = self.root / "exports"
        result = probe.run(config, self.root, output_dir=destination)
        self.assertFalse(result["failed"])
        self.assertEqual({row["evaluation_scope"] for row in result["results"]},
                         {"diagnostic_only_official_test_dialogues"})
        predictions = self.read_csv(destination / MODEL / "meld" / "seed_42" / "selected_predictions.csv")
        self.assertEqual({row["group_kind"] for row in predictions}, {"dialogue"})

    def test_explicit_manifest_cannot_rename_a_speaker_to_hide_leakage(self):
        config, paths, _ = self.project_fixture(models=(MODEL,))
        groups = infer_groups("ravdess", paths, self.root)
        manifest = self.root / "supplied.csv"
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_path", "group", "split"])
            writer.writeheader()
            for index, path in enumerate(paths):
                writer.writerow({"audio_path": path, "group": groups[path]["group"] + f":fake_{index}",
                                 "split": ("train", "validation", "test")[index % 3]})
        config["split_manifests"] = {"ravdess": str(manifest)}
        result = probe.run(config, self.root, output_dir=self.root / "exports")
        self.assertFalse(result["results"])
        self.assertTrue(any("group identity" in row["reason"] for row in result["failed"]))

    def test_sampled_predictions_are_not_reported_as_complete_corpus_probes(self):
        config, paths, _ = self.project_fixture(models=(MODEL,))
        metadata_path = self.root / "outputs" / MODEL / "ravdess_predictions.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["total_clips"] = len(paths) + 1
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        result = probe.run(config, self.root, output_dir=self.root / "exports", dry_run=True)
        self.assertTrue(result["failed"])
        self.assertFalse(result["ready"])

    def test_disabling_current_prompt_check_does_not_allow_global_label_protocol(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        config["require_current_protocol"] = False
        metadata_path = self.root / "outputs" / MODEL / "ravdess_predictions.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["inference_protocol"]["label_scope"] = "global"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        result = probe.run(config, self.root, output_dir=self.root / "exports", dry_run=True)
        self.assertTrue(result["failed"])
        self.assertFalse(result["ready"])

    def test_pairing_rejects_different_composite_label_semantics(self):
        config, _, _ = self.project_fixture()
        metadata_path = self.root / "outputs" / config["models"][1] / "ravdess_predictions.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["accept"] = {"excited": ["happy", "sad"]}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        result = probe.run(config, self.root, output_dir=self.root / "exports")
        self.assertTrue(result["failed"])
        self.assertFalse(result["results"])

    def test_nonconverged_selected_fit_is_not_a_formal_result(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        config["max_iter"] = 1
        config["c_values"] = [1]
        result = probe.run(config, self.root, output_dir=self.root / "exports")
        self.assertTrue(result["failed"])
        self.assertFalse(result["results"])
        self.assertTrue(any("converg" in row["reason"] for row in result["failed"]))

    def test_jl_excited_is_excluded_and_retains_accepted_labels_without_happy_relabeling(self):
        config, _, _ = self.project_fixture(models=(MODEL,))
        candidates = ["angry", "happy", "neutral", "sad", "surprised"]
        prompt_path = self.root / "config" / "prompts.yaml"
        prompt = yaml.safe_load(prompt_path.read_text())
        prompt["labels"] = candidates
        prompt_path.write_text(yaml.safe_dump(prompt), encoding="utf-8")
        registry = yaml.safe_load((self.root / "config" / "models.yaml").read_text())
        protocol = configured_protocol(registry[MODEL], candidates,
                                       config=self.root / "config" / "llm.yaml", root=self.root)
        rows, truth = [], []
        primary = ["angry", "happy", "neutral", "sad", "excited"]
        for speaker in ("female1", "female2", "male1", "male2"):
            for label in primary:
                rows.append({"row": len(rows), "audio_path": f"data/jl_corpus/audio/{speaker}_{label}_10a_1.wav",
                             "pooling": "audio", "tokens": 12})
                truth.append(label)
        vectors = np.zeros((len(rows), 2, 3), dtype=np.float16)
        for index, label in enumerate(truth):
            if label in primary[:3]:
                vectors[index, 1, primary.index(label)] = 4
            elif label == "excited":
                vectors[index] = 100
        self.feature_store(protocol=protocol, model=MODEL, dataset="jl_corpus", rows=rows, vectors=vectors)
        output = self.root / "outputs" / MODEL
        metadata = {"key": "jl_corpus", "model": MODEL, "classes": candidates, "clips": len(rows),
                    "total_clips": len(rows), "accept": {"excited": ["happy", "surprised"]},
                    "inference_protocol": protocol}
        (output / "jl_corpus_predictions.json").write_text(json.dumps(metadata), encoding="utf-8")
        with (output / "jl_corpus_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_path", "acted_emotion", "prediction"])
            writer.writeheader()
            for row, label in zip(rows, truth):
                writer.writerow({"audio_path": row["audio_path"], "acted_emotion": label,
                                 "prediction": "happy" if label == "excited" else label})
        config["datasets"] = ["jl_corpus"]
        config["group_counts"] = {"jl_corpus": [2, 1, 1]}
        destination = self.root / "exports"
        result = probe.run(config, self.root, output_dir=destination)
        self.assertFalse(result["failed"])
        self.assertTrue(result["results"])
        manifest_dir = destination / "splits" / "jl_corpus"
        manifest = json.loads((manifest_dir / "seed_42.json").read_text())
        self.assertEqual(manifest["classes"], primary[:4])
        self.assertEqual(manifest["inference_candidates"], candidates)
        excluded = self.read_csv(manifest_dir / "excluded_targets.csv")
        self.assertEqual(len(excluded), 4)
        self.assertEqual({row["target"] for row in excluded}, {"excited"})
        self.assertEqual({row["accepted_labels"] for row in excluded}, {"happy|surprised"})
        predictions = self.read_csv(destination / MODEL / "jl_corpus" / "seed_42" / "selected_predictions.csv")
        self.assertEqual(len(predictions), 16)
        self.assertEqual({row["y_true"] for row in predictions}, set(primary[:4]))
        self.assertFalse(any("_excited_" in row["audio_path"] for row in predictions))
        parameters = json.loads((destination / MODEL / "jl_corpus" / "seed_42" /
                                 "layer_4" / "model_parameters.json").read_text())
        self.assertEqual(parameters["classes"], primary[:4])


if __name__ == "__main__":
    unittest.main()

"""Offline regression checks for dataset-scoped audio-LLM choices and result reuse.

Run in the emotion_recognition_llm environment, without model weights or API keys:
    python -m unittest discover -s test_scripts -p 'test_dataset_label_choices.py' -v

All outputs live in TemporaryDirectory; model construction and API requests are fakes.
"""
import contextlib
import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
for folder in ("common", "models", "llm_eval", "evaluation", "viewer"):
    sys.path.insert(0, str(PROJECT / "scripts" / folder))

import audio_llm
import data_loader
import evaluate as evaluation
import gpt_realtime
import make_viewer
import plot_results
import prompts
from eval_datasets import EvalSet, accepted
from parsing import parse_label
from prediction_io import read_prediction_rows
from training_provenance import (EXPOSURE_LEVELS, training_hatch, training_marker,
                                 with_training_provenance)

LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
FOUR = ["angry", "happy", "neutral", "sad"]
SIX = LABELS[:-1]


class GenerationConfig(SimpleNamespace):
    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class Tokenizer:
    """Each distinct spelling is one token; no transformer/tokenizer downloads."""

    def __init__(self):
        self.tokens = {}

    def encode(self, text, add_special_tokens=False):
        if text not in self.tokens:
            self.tokens[text] = len(self.tokens)
        return [self.tokens[text]]

    def decode(self, tokens, **kwargs):
        names = {value: key for key, value in self.tokens.items()}
        return "".join(names[int(t)] for t in tokens)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ser_label_choices_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompt_path = self.root / "prompts.yaml"
        self.write_prompts()
        self.config = self.root / "llm.yaml"
        self.config.write_text(yaml.safe_dump({
            "prompts": "prompts.yaml", "retries_invalid": 1, "max_new_tokens": 8,
            "nan_retries": 0, "save_label_probs": True, "save_hidden_states": True,
            "hidden_layers": 1, "retries": 0, "timeout_s": 1,
        }))
        self.enter = contextlib.ExitStack()
        self.addCleanup(self.enter.close)
        self.enter.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def write_prompts(self, suffix=""):
        self.prompt_path.write_text(yaml.safe_dump({
            "system": "Choose exactly one label: {labels}." + suffix,
            "user": "Listen to this clip and choose: {labels}.", "labels": LABELS,
        }))

    def dataset(self, classes=FOUR, key="fixture", truths=None, accept=None):
        truths = truths or [classes[0], classes[-1]]
        clips = [{"audio": self.root / "data" / key / f"{i}.wav", "acted": label,
                  "listener": ""} for i, label in enumerate(truths)]
        return EvalSet(key, "Fixture", "English", list(classes), clips, "fixture labels", accept=accept or {})

    def local(self):
        self.enter.enter_context(mock.patch.object(audio_llm, "ROOT", self.root))
        self.enter.enter_context(mock.patch.object(audio_llm, "CONFIG", self.config))
        self.enter.enter_context(mock.patch.dict(sys.modules, {"transformers": SimpleNamespace(__version__="fake")}))
        processor = SimpleNamespace(tokenizer=Tokenizer())
        model = SimpleNamespace(generation_config=GenerationConfig(eos_token_id=None, pad_token_id=None),
                                config=SimpleNamespace(audio_token_id=3))
        family = SimpleNamespace(build=mock.Mock(return_value=(model, processor)), DESCRIPTION="offline fake")
        entry = {"key": "fake_local", "hf_repo": "test/local", "cache_file": self.root / "responses.jsonl",
                 "features_dir": self.root / "features"}
        predict = audio_llm.Predictor(self.root / "weights", torch.device("cpu"), entry, family)
        return predict, family

    def realtime(self):
        self.enter.enter_context(mock.patch.object(gpt_realtime, "ROOT", self.root))
        self.enter.enter_context(mock.patch.object(gpt_realtime, "CONFIG", self.config))
        self.enter.enter_context(mock.patch.object(gpt_realtime, "find_key", return_value="fake-key"))
        self.enter.enter_context(mock.patch.dict(sys.modules, {"websockets": SimpleNamespace(__version__="fake")}))
        return gpt_realtime.Predictor({"key": "fake_api", "api_model": "test/realtime",
                                      "cache_file": self.root / "responses_api.jsonl"})


class PromptChoicesTests(Fixture):
    def test_four_six_seven_choices_keep_canonical_order(self):
        for choices in (FOUR, SIX, LABELS):
            with self.subTest(choices=choices):
                filled = prompts.load_prompts(self.prompt_path, labels=list(reversed(choices)))
                self.assertEqual(filled["labels"], choices)
                for name in ("system", "user"):
                    self.assertIn(", ".join(choices), filled[name])
                    self.assertNotIn("{labels}", filled[name])

    def test_invalid_dataset_choices_are_rejected(self):
        for choices in ([], ["angry", "angry"], ["excited"], ["angry", "other"]):
            with self.subTest(choices=choices), self.assertRaises(SystemExit):
                prompts.load_prompts(self.prompt_path, labels=choices)

    def test_protocol_hash_separates_scope_prompt_and_local_reminder(self):
        filled = prompts.load_prompts(self.prompt_path, labels=FOUR)
        protocol = prompts.prompt_protocol(filled)
        self.assertEqual(protocol["label_scope"], "dataset")
        self.assertEqual(protocol["labels"], FOUR)
        self.assertEqual(protocol, prompts.prompt_protocol(prompts.load_prompts(self.prompt_path, labels=FOUR[::-1])))
        self.assertNotEqual(protocol["prompt_hash"], prompts.prompt_protocol(filled, local=True)["prompt_hash"])
        local_hash = prompts.prompt_protocol(filled, local=True)["prompt_hash"]
        with mock.patch.object(prompts, "REMINDER", "Choose one of {labels}."):
            self.assertNotEqual(local_hash, prompts.prompt_protocol(filled, local=True)["prompt_hash"])
            self.assertEqual(protocol, prompts.prompt_protocol(filled))
        self.assertNotEqual(protocol["prompt_hash"], prompts.prompt_protocol(
            prompts.load_prompts(self.prompt_path, labels=SIX))["prompt_hash"])
        legacy = hashlib.sha256(json.dumps(filled, sort_keys=True).encode()).hexdigest()[:12]
        self.assertNotEqual(protocol["prompt_hash"], legacy)
        self.write_prompts(" Follow the supplied choices.")
        self.assertNotEqual(protocol["prompt_hash"], prompts.prompt_protocol(
            prompts.load_prompts(self.prompt_path, labels=FOUR))["prompt_hash"])

    def test_jl_excited_is_a_scoring_rule_not_an_output_label(self):
        ds = self.dataset(["angry", "happy", "neutral", "sad", "surprised"],
                          key="jl_corpus", truths=["excited"], accept={"excited": ["happy", "surprised"]})
        filled = prompts.load_prompts(self.prompt_path, labels=ds.classes)
        self.assertNotIn("excited", filled["labels"])
        self.assertNotIn("excited", filled["system"] + filled["user"])
        self.assertEqual(accepted(ds, "excited"), {"happy", "surprised"})
        self.assertEqual(parse_label("surprise", filled["labels"]), "surprised")


class LocalPredictorTests(Fixture):
    def test_set_dataset_changes_prompt_probabilities_not_output_schema(self):
        predict, family = self.local()
        for classes in (FOUR, SIX, LABELS):
            with self.subTest(classes=classes):
                predict.set_dataset(self.dataset(classes))
                self.assertEqual(predict.choices, classes)
                self.assertEqual(predict.labels, LABELS + ["invalid"])
                self.assertEqual(set(predict.label_ids), set(classes))
                self.assertIn(", ".join(classes), predict.system)
                self.assertIn(", ".join(classes), predict.user)
                self.assertIn(", ".join(classes), predict.reminder)
                found = predict.label_probs(torch.zeros(256))
                self.assertEqual(set(found["label_probs"]), set(classes))
                self.assertAlmostEqual(sum(found["label_probs"].values()), 1, places=5)
        family.build.assert_called_once()

    def test_ground_truth_changes_do_not_change_the_question(self):
        predict, _ = self.local()
        ds = self.dataset()
        predict.set_dataset(ds)
        question = (predict.system, predict.user, predict.reminder, predict.prompt_hash)
        ds.clips[0]["acted"], ds.clips[0]["listener"] = "sad", "fearful"
        predict.set_dataset(ds)
        self.assertEqual(question, (predict.system, predict.user, predict.reminder, predict.prompt_hash))

    def test_excluded_answer_is_invalid_then_uses_scoped_retry(self):
        predict, _ = self.local()
        predict.set_dataset(self.dataset())
        with mock.patch.object(predict, "generate", side_effect=[("fearful", {}, True), ("happy", {}, True)]) as gen:
            answers, label, _, finite = predict.ask(np.zeros(16))
        self.assertEqual(answers, ["fearful", "happy"])
        self.assertEqual(label, "happy")
        self.assertTrue(finite)
        retry = gen.call_args_list[1].args[1]
        self.assertEqual(retry, [("fearful", predict.reminder)])
        self.assertNotIn("fearful", predict.reminder)

    def test_cache_and_features_do_not_reuse_another_protocol(self):
        predict, _ = self.local()
        ds = self.dataset()
        predict.set_dataset(ds)
        first_hash = predict.prompt_hash
        rel = "data/fixture/0.wav"
        record = {"audio_path": rel, "hf_repo": predict.hf_repo, "prompts": first_hash, "label": "happy"}
        legacy_hash = hashlib.sha256(json.dumps(prompts.load_prompts(self.prompt_path), sort_keys=True).encode()).hexdigest()[:12]
        predict.cache_file.write_text("\n".join(json.dumps(r) for r in [record,
            record | {"prompts": legacy_hash, "label": "sad"},
            record | {"hf_repo": "different/model", "label": "sad"}]) + "\n")
        predict.set_dataset(ds)
        self.assertEqual(predict.cache[rel]["label"], "happy")
        old_folder = predict.features_dir / ds.key
        old_folder.mkdir(parents=True, exist_ok=True)
        sentinel = old_folder / "legacy-kept.txt"
        sentinel.write_text("old features stay untouched")
        predict.layers, predict.num_layers = [1], 1
        found = {"hidden": np.array([[1, 2]], dtype=np.float16), "pooling": "audio", "tokens": 1}
        predict.save_features(rel, found)
        self.assertTrue((old_folder / first_hash / "meta.json").exists())
        self.assertEqual(sentinel.read_text(), "old features stay untouched")
        self.write_prompts(" A revised instruction.")
        predict.set_dataset(ds)
        self.assertNotEqual(predict.prompt_hash, first_hash)
        self.assertNotIn(rel, predict.cache)
        predict.save_features(rel, found)
        self.assertTrue((old_folder / predict.prompt_hash / "meta.json").exists())
        self.assertTrue((old_folder / first_hash / "meta.json").exists())


class RealtimePredictorTests(Fixture):
    def test_scoped_api_request_and_cache_resume_without_network(self):
        predict = self.realtime()
        ds = self.dataset()
        predict.set_dataset(ds)
        first_hash = predict.prompt_hash
        with mock.patch.object(gpt_realtime, "prepare_audio", return_value=("fake-audio", None, None)), \
                mock.patch.object(gpt_realtime, "request", side_effect=[
                    ("fearful", {}, "test/snapshot"), ("happy", {}, "test/snapshot")]) as request:
            result = predict(ds.clips[0]["audio"])
            self.assertEqual(request.call_count, 2)
            for call in request.call_args_list:
                self.assertIn(", ".join(FOUR), call.args[1])
                self.assertIn(", ".join(FOUR), call.args[2])
                self.assertNotIn("fearful", call.args[1] + call.args[2])
            self.assertEqual(result.shape, (8,))
            self.assertEqual(predict.labels, LABELS + ["invalid"])
            self.assertEqual(predict.labels[result.argmax()], "happy")
            predict.set_dataset(ds)
            np.testing.assert_array_equal(predict(ds.clips[0]["audio"]), result)
            self.assertEqual(request.call_count, 2)
            predict.set_dataset(self.dataset(SIX))
            self.assertNotEqual(predict.prompt_hash, first_hash)
            self.assertFalse(predict.cache)


class PreviewChoicesTests(Fixture):
    def test_preview_items_filter_choices_for_each_dataset(self):
        datasets = {f"choices_{len(classes)}": self.dataset(classes, key=f"choices_{len(classes)}")
                    for classes in (FOUR, SIX, LABELS)}
        with mock.patch.object(data_loader, "ROOT", self.root), \
                mock.patch.object(data_loader, "select_datasets", return_value=(list(datasets), {})), \
                mock.patch.object(data_loader, "LOADERS", {key: mock.Mock(return_value=ds) for key, ds in datasets.items()}), \
                mock.patch.object(data_loader, "prepare_audio", return_value=("fake-audio", 16000, 16000)):
            items = list(data_loader.items({"prompts": "prompts.yaml", "audio_format": "wav16k", "limit": None}))
        self.assertEqual(len(items), 6)
        for item in items:
            classes = datasets[item["dataset"]].classes
            self.assertEqual(item["choices"], classes)
            self.assertIn(", ".join(classes), item["system"])
            self.assertIn(", ".join(classes), item["user"])

    def test_preview_rejects_unfiltered_choices(self):
        ds = self.dataset()
        with mock.patch.object(data_loader, "prepare_audio") as prepare, self.assertRaises(ValueError):
            list(data_loader.dataset_items(ds, ds.clips, prompts.load_prompts(self.prompt_path), "wav16k"))
        prepare.assert_not_called()


class EvaluationReuseTests(Fixture):
    def setUp(self):
        super().setUp()
        self.enter.enter_context(mock.patch.object(evaluation, "ROOT", self.root))
        self.enter.enter_context(mock.patch.object(evaluation, "LLM_CONFIG", self.config))
        self.enter.enter_context(mock.patch.object(evaluation, "SUBSETS", {}))
        self.ds = self.dataset()
        self.enter.enter_context(mock.patch.object(evaluation, "dataset", return_value=self.ds))
        self.entry = {"name": "Fake local", "loader": "audio_llm", "llm": True, "hf_repo": "test/local",
                      "training_data": [], "training_source": "not published"}
        self.protocol = evaluation.current_protocol(self.entry, self.ds)

    def write_run(self, protocol=None, folder="outputs/fake", stem="fixture"):
        out = self.root / folder
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{stem}_predictions.json"
        meta = {"key": "fixture", "model": Path(folder).name, "dataset": self.ds.name,
                "ground_truth": "acted_emotion", "language": self.ds.language,
                "clips": len(self.ds.clips), "date": "2026-01-01",
                "classes": list(self.ds.classes), "labels": ["angry", "sad"], "metrics": {"accuracy": .5},
                "model_classes": LABELS + ["invalid"]}
        if protocol is not None:
            meta["inference_protocol"] = protocol
        path.write_text(json.dumps(meta))
        with open(path.with_suffix(".csv"), "w", newline="") as fh:
            columns = ["audio_path", "acted_emotion", "prediction", "prediction_all_classes"] + [f"p_{lab}" for lab in meta["model_classes"]]
            writer = csv.DictWriter(fh, columns)
            writer.writeheader()
            for clip in self.ds.clips:
                row = {"audio_path": clip["audio"].relative_to(self.root).as_posix(), "acted_emotion": clip["acted"],
                       "prediction": "happy", "prediction_all_classes": "happy"}
                row.update({f"p_{lab}": int(lab == "happy") for lab in meta["model_classes"]})
                writer.writerow(row)
        return path

    def test_current_protocol_uses_dataset_classes_and_model_identity(self):
        self.assertEqual(self.protocol["labels"], FOUR)
        self.assertEqual(self.protocol["model_id"], self.entry["hf_repo"])
        other = evaluation.current_protocol(self.entry | {"hf_repo": "different/model"}, self.ds)
        self.assertNotEqual(self.protocol, other)
        self.assertIsNone(evaluation.current_protocol(self.entry | {"llm": False}, self.ds))

    def test_finished_and_earlier_runs_require_protocol_match(self):
        for saved in (None, self.protocol | {"prompt_hash": "old"}, self.protocol | {"model_id": "other"}, self.protocol):
            with self.subTest(saved=saved):
                path = self.write_run(saved)
                finished = evaluation.finished_run("fake", "fixture", None, protocol=self.protocol)
                reused = evaluation.earlier_run("fake", "fixture", None, protocol=self.protocol)
                if saved == self.protocol:
                    self.assertEqual(finished, path)
                    self.assertIsNotNone(reused)
                    self.assertEqual(reused["scores"].shape, (2, 8))
                else:
                    self.assertIsNone(finished)
                    self.assertIsNone(reused)

    def test_ser_keeps_legacy_finished_result_reuse(self):
        path = self.write_run()
        self.assertEqual(evaluation.finished_run("fake", "fixture", None), path)
        self.assertIsNotNone(evaluation.earlier_run("fake", "fixture", None))

    def test_archive_reuse_ignores_incompatible_main_run(self):
        self.write_run(self.protocol | {"prompt_hash": "old"})
        good = self.write_run(self.protocol, folder="outputs/archive/fake", stem="fixture_full")
        reused = evaluation.earlier_run("fake", "fixture", None, protocol=self.protocol)
        self.assertEqual(reused["csv"], good.with_suffix(".csv"))

    def test_imported_bom_yes_no_results_keep_scores_and_viewer_correctness(self):
        path = self.write_run(self.protocol)
        csv_path = path.with_suffix(".csv")
        rows = read_prediction_rows(csv_path)
        for i, row in enumerate(rows):
            label = "angry" if i == 0 else "happy"
            row.update(prediction=label, prediction_all_classes=label,
                       correct="Yes" if i == 0 else "No")
            row.update({f"p_{lab}": int(lab == label) for lab in LABELS + ["invalid"]})
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        original = csv_path.read_bytes()
        reused = evaluation.earlier_run("fake", "fixture", None, protocol=self.protocol)
        self.assertEqual(reused["scores"][:, :4].argmax(1).tolist(), [0, 3])
        run = make_viewer.load_run(csv_path, json.loads(path.read_text()))
        self.assertEqual(run["criteria"][0]["accuracy"], .5)
        self.assertEqual(len(run["rows"]), 2)
        picked, counts = make_viewer.pick_clips("fixture", {"fake": run}, 1, 1, 7)
        self.assertEqual(len(picked), 2)
        self.assertEqual(counts["fake"], (1, 1))
        audio = self.ds.clips[0]["audio"]
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.touch()
        with mock.patch.object(make_viewer, "ROOT", self.root), \
                mock.patch.object(make_viewer.sf, "info", return_value=SimpleNamespace(duration=1.0)):
            clip = make_viewer.make_clip(audio.relative_to(self.root).as_posix(), {"fake": run})
        self.assertTrue(clip["models"]["fake"]["correct"]["correct"])
        self.assertEqual(clip["models"]["fake"]["score_kind"], "answer_indicator")
        self.assertEqual(csv_path.read_bytes(), original)

    def test_imported_unknown_correctness_flag_is_rejected(self):
        csv_path = self.root / "invalid.csv"
        csv_path.write_text("audio_path,correct\na.wav,maybe\n", encoding="utf-8-sig")
        with self.assertRaisesRegex(ValueError, "invalid.csv:2, correct"):
            read_prediction_rows(csv_path)

    def test_main_summary_excludes_larger_run_with_old_protocol(self):
        old = self.write_run()
        old.write_text(json.dumps(json.loads(old.read_text()) | {"clips": 100}))
        self.write_run(self.protocol, stem="fixture_sample2")
        selected = evaluation.best_runs("fake", entry=self.entry)
        self.assertEqual(selected["fixture"]["stem"], "fixture_sample2")
        self.assertEqual(evaluation.best_runs("fake", exclude=["fixture"], entry=self.entry), {})
        ser = evaluation.best_runs("fake", entry=self.entry | {"llm": False})
        self.assertEqual(ser["fixture"]["stem"], "fixture")

    def test_evaluate_sets_dataset_before_inference_and_saves_protocol(self):
        events = []
        expected = self.protocol

        class FakePredictor:
            labels = LABELS + ["invalid"]
            versions, notes, on_cpu = {}, [], 0

            def set_dataset(self, ds):
                events.append(("dataset", ds.key, list(ds.classes)))
                self.protocol = {key: value for key, value in expected.items() if key != "model_id"}

            def __call__(self, path):
                events.append(("infer", path))
                return np.array([int(lab == "angry") for lab in self.labels])

        (self.root / "reports" / "fake" / "figures").mkdir(parents=True)
        with mock.patch.object(evaluation, "plot_confusion"), mock.patch.object(evaluation, "hardware", return_value="fake CPU"):
            evaluation.evaluate(self.ds, FakePredictor(), "fake", self.entry, "", torch.device("cpu"), None)
        self.assertEqual(events[0], ("dataset", self.ds.key, FOUR))
        saved = json.loads((self.root / "outputs" / "fake" / "fixture_predictions.json").read_text())
        self.assertEqual(saved["inference_protocol"], expected)

    def test_shared_score_does_not_promote_runner_up_for_ser(self):
        out = self.root / "outputs" / "fake"
        out.mkdir(parents=True)
        with open(out / "fixture_predictions.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, ["acted_emotion", "prediction", "p_fearful", "p_angry", "p_happy", "p_neutral", "p_sad"])
            writer.writeheader()
            writer.writerow({"acted_emotion": "angry", "prediction": "fearful", "p_fearful": .6, "p_angry": .4})
            writer.writerow({"acted_emotion": "sad", "prediction": "sad", "p_sad": 1})
            writer.writerow({"acted_emotion": "fearful", "prediction": "fearful", "p_fearful": 1})
        clips, accuracy, uar = evaluation.shared_scores("fake", {"stem": "fixture"}, FOUR)
        self.assertEqual(clips, 2)
        self.assertEqual(accuracy, .5)
        self.assertEqual(uar, .5)

    def test_reused_run_keeps_its_own_dataset_notes_and_versions(self):
        self.write_run(self.protocol)
        reused = evaluation.earlier_run("fake", "fixture", None, protocol=self.protocol)
        reused["meta"].update(notes=["Allowed choices: angry, happy, neutral, sad"], versions={"backend": "saved"})
        predict = evaluation.Reused(reused["meta"] | {"notes": ["Wrong previous dataset choices"],
                                                      "versions": {"backend": "previous"}})
        (self.root / "reports" / "fake" / "figures").mkdir(parents=True)
        with mock.patch.object(evaluation, "plot_confusion"), mock.patch.object(evaluation, "hardware", return_value="fake CPU"):
            evaluation.evaluate(self.ds, predict, "fake", self.entry, "", torch.device("cpu"), None, reuse=reused)
        saved = json.loads((self.root / "outputs" / "fake" / "fixture_predictions.json").read_text())
        self.assertEqual(saved["notes"], reused["meta"]["notes"])
        self.assertEqual(saved["versions"], {"backend": "saved"})

    def viewer_config(self):
        config = self.root / "config"
        config.mkdir()
        (config / "llm.yaml").write_text(self.config.read_text())
        (config / "models.yaml").write_text(yaml.safe_dump({"fake": self.entry, "fake_ser": self.entry | {"llm": False}}))
        (config / "plots.yaml").write_text(yaml.safe_dump({"metrics": ["accuracy"]}))
        return config

    def test_plot_loader_filters_protocols_before_choosing_largest_run(self):
        self.viewer_config()
        old = self.write_run()
        old.write_text(json.dumps(json.loads(old.read_text()) | {"clips": 100}))
        self.write_run(self.protocol, stem="fixture_sample2")
        self.write_run(folder="outputs/fake_ser")
        selected = plot_results.load_results("all", "all", [], root=self.root)
        self.assertEqual(selected["fake"]["fixture"]["clips"], 2)
        self.assertEqual(selected["fake"]["fixture"]["inference_protocol"], self.protocol)
        self.assertIn("fake_ser", selected)

    def test_viewer_rebuilds_summary_only_from_compatible_metadata(self):
        self.viewer_config()
        old = self.write_run()
        old.write_text(json.dumps(json.loads(old.read_text()) | {"clips": 100}))
        good = self.write_run(self.protocol, stem="fixture_sample2")
        registry = {"fake": self.entry}
        with mock.patch.object(make_viewer, "ROOT", self.root):
            old_meta = json.loads(old.read_text())
            good_meta = json.loads(good.read_text())
            self.assertFalse(make_viewer.current_run(old_meta, registry))
            self.assertFalse(make_viewer.current_run(good_meta | {"inference_protocol": self.protocol | {"model_id": "old/model"}}, registry))
            metas = {path.with_suffix(".csv"): meta for path, meta in [(old, old_meta), (good, good_meta)]
                     if make_viewer.current_run(meta, registry)}
            best = make_viewer.best_metadata(metas)
            chosen = {path: metas[path] for path in best.values()}
            summary = make_viewer.current_summary("fake", chosen, self.entry)
        self.assertEqual(summary["datasets"][0]["clips"], 2)
        self.assertEqual(summary["datasets"][0]["inference_protocol"], self.protocol)
        self.assertIsNone(summary["figure"])

    def test_viewer_only_embeds_plots_matching_current_results_manifest(self):
        config = self.viewer_config()
        self.write_run(self.protocol)
        folder = self.root / "reports" / "figures"
        folder.mkdir(parents=True)
        (folder / "accuracy_by_dataset.png").write_bytes(b"fake image, embed is mocked")
        with mock.patch.object(make_viewer, "ROOT", self.root), \
                mock.patch.object(make_viewer, "MODELS_CONFIG", config / "models.yaml"), \
                mock.patch.object(make_viewer, "embed_image", return_value="embedded"):
            self.assertEqual(make_viewer.load_plots(), [])
            selected = plot_results.load_results("all", "all", [], root=self.root)
            manifest = {"fingerprint": plot_results.results_fingerprint(selected), "metrics": ["accuracy"],
                        "presentation_fingerprint": plot_results.presentation_fingerprint(
                            yaml.safe_load((config / "plots.yaml").read_text()))}
            (folder / "results_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(make_viewer.load_plots()[0]["by_dataset"], "embedded")
            (folder / "results_manifest.json").write_text(json.dumps(manifest | {"fingerprint": "old"}))
            self.assertEqual(make_viewer.load_plots(), [])

    def test_viewer_rejects_figure_with_stale_dataset_order(self):
        config = self.viewer_config()
        self.write_run(self.protocol)
        folder = self.root / "reports" / "figures"
        folder.mkdir(parents=True)
        (folder / "accuracy_by_dataset.png").write_bytes(b"mock image")
        settings = yaml.safe_load((config / "plots.yaml").read_text())
        selected = plot_results.load_results("all", "all", [], root=self.root)
        manifest = {"fingerprint": plot_results.results_fingerprint(selected), "metrics": ["accuracy"],
                    "presentation_fingerprint": plot_results.presentation_fingerprint(settings)}
        (folder / "results_manifest.json").write_text(json.dumps(manifest))
        with mock.patch.object(make_viewer, "ROOT", self.root), \
                mock.patch.object(make_viewer, "MODELS_CONFIG", config / "models.yaml"), \
                mock.patch.object(make_viewer, "embed_image", return_value="embedded"):
            self.assertEqual(len(make_viewer.load_plots()), 1)
            (config / "plots.yaml").write_text(yaml.safe_dump(settings | {"dataset_order": ["fixture"]}))
            self.assertEqual(make_viewer.load_plots(), [])

    def test_viewer_rebuilds_ser_shared_scores_from_original_predictions(self):
        chosen = {}
        for key, classes, prediction in (("six", SIX, "fearful"), ("four", FOUR, "angry")):
            path = self.root / f"{key}_predictions.csv"
            with open(path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, ["acted_emotion", "prediction", "p_fearful", "p_angry"])
                writer.writeheader()
                writer.writerow({"acted_emotion": "angry", "prediction": prediction,
                                 "p_fearful": .6, "p_angry": .4})
            chosen[path] = {"key": key, "dataset": key, "classes": classes, "clips": 1,
                            "ground_truth": "acted_emotion", "metrics": {"accuracy": int(prediction == "angry")}}
        with mock.patch.object(make_viewer, "ROOT", self.root):
            summary = make_viewer.current_summary("fake_ser", chosen, self.entry | {"llm": False})
        self.assertEqual(summary["shared_emotions"], FOUR)
        shared = {row["key"]: row for row in summary["shared"]}
        self.assertEqual(shared["six"]["accuracy"], 0)
        self.assertEqual(shared["four"]["accuracy"], 1)

    def test_registry_corrects_stale_plot_flags_both_ways_without_changing_inference(self):
        self.viewer_config()
        path = self.write_run(self.protocol)
        original = json.loads(path.read_text()) | {"cache_fingerprint": "unchanged"}
        for previous, confirmed in ((True, False), (False, True)):
            with self.subTest(previous=previous, confirmed=confirmed):
                stale = original | {"in_training_data": previous, "training_source": "obsolete source"}
                path.write_text(json.dumps(stale))
                before = path.read_bytes()
                entry = self.entry | {"training_data": ["fixture"] if confirmed else []}
                results = plot_results.load_results("all", "all", [], registry={"fake": entry}, root=self.root)
                run = results["fake"]["fixture"]
                self.assertEqual(run["in_training_data"], confirmed)
                self.assertEqual(run["training_status"], "confirmed_exposure" if confirmed else "not_established")
                for field in ("metrics", "inference_protocol", "cache_fingerprint"):
                    self.assertEqual(run[field], stale[field])
                self.assertEqual(path.read_bytes(), before)
                # Historical booleans cannot alter the effective figure fingerprint once the registry is fixed.
                fingerprint = plot_results.results_fingerprint(results)
                path.write_text(json.dumps(stale | {"in_training_data": not previous}))
                reloaded = plot_results.load_results("all", "all", [], registry={"fake": entry}, root=self.root)
                self.assertEqual(plot_results.results_fingerprint(reloaded), fingerprint)

    def test_viewer_corrects_current_and_summary_only_flags_both_ways(self):
        path = self.write_run(self.protocol)
        for previous, confirmed in ((True, False), (False, True)):
            with self.subTest(previous=previous, confirmed=confirmed), mock.patch.object(make_viewer, "ROOT", self.root):
                meta = json.loads(path.read_text()) | {"in_training_data": previous}
                entry = self.entry | {"training_data": ["fixture"] if confirmed else []}
                current = make_viewer.current_summary("fake", {path.with_suffix(".csv"): meta}, entry)
                row = current["datasets"][0]
                self.assertEqual(row["in_training_data"], confirmed)
                self.assertEqual(row["metrics"], meta["metrics"])
                self.assertEqual(row["inference_protocol"], self.protocol)
                summary_path = self.root / "old_summary.json"
                summary_path.write_text(json.dumps({"datasets": [meta], "shared": [meta], "figure": None}))
                fallback = make_viewer.load_summary(summary_path, [], entry)
                for group in ("datasets", "shared"):
                    self.assertEqual(fallback[group][0]["in_training_data"], confirmed)
                    self.assertEqual(fallback[group][0]["metrics"], meta["metrics"])

    def test_registry_change_invalidates_embedded_plots_without_changing_protocol(self):
        config = self.viewer_config()
        self.write_run(self.protocol)
        positive = self.entry | {"training_data": ["fixture"]}
        (config / "models.yaml").write_text(yaml.safe_dump({"fake": positive}))
        folder = self.root / "reports" / "figures"
        folder.mkdir(parents=True)
        (folder / "accuracy_by_dataset.png").write_bytes(b"mock image")
        old = plot_results.load_results("all", "all", [], root=self.root)
        manifest = {"fingerprint": plot_results.results_fingerprint(old), "metrics": ["accuracy"],
                    "presentation_fingerprint": plot_results.presentation_fingerprint(
                        yaml.safe_load((config / "plots.yaml").read_text()))}
        (folder / "results_manifest.json").write_text(json.dumps(manifest))
        with mock.patch.object(make_viewer, "ROOT", self.root), \
                mock.patch.object(make_viewer, "MODELS_CONFIG", config / "models.yaml"), \
                mock.patch.object(make_viewer, "embed_image", return_value="embedded"):
            self.assertEqual(len(make_viewer.load_plots()), 1)
            (config / "models.yaml").write_text(yaml.safe_dump({"fake": self.entry}))
            self.assertEqual(make_viewer.load_plots(), [])
            new = plot_results.load_results("all", "all", [], root=self.root)
            self.assertEqual(new["fake"]["fixture"]["metrics"], old["fake"]["fixture"]["metrics"])
            self.assertEqual(new["fake"]["fixture"]["inference_protocol"], self.protocol)
            self.assertNotEqual(plot_results.results_fingerprint(new), manifest["fingerprint"])

    def test_pure_summary_refresh_uses_registry_and_unknown_is_not_an_unseen_claim(self):
        path = self.write_run(self.protocol)
        meta = json.loads(path.read_text()) | {"in_training_data": True,
                "metrics": {"accuracy": .5, "unweighted_accuracy": .5, "macro_f1": .4}}
        path.write_text(json.dumps(meta))
        before = path.read_bytes()
        (self.root / "reports" / "fake").mkdir(parents=True)
        report_path = self.root / "reports" / "fake" / "fixture.md"
        before_note = "# Saved report\n\nAccuracy: 50.0%\n\n## Caveats\n\n- "
        after_note = "\n- Dataset-specific note stays intact.\n"
        report_path.write_text(before_note + "**Not in Fake local's training data** (obsolete source).\n"
                               "Obsolete wrapped training explanation." + after_note)
        for confirmed in (False, True):
            entry = self.entry | {"llm": False, "training_data": ["fixture"] if confirmed else []}
            with self.subTest(confirmed=confirmed), mock.patch.object(evaluation, "LOADERS", {"fixture": None}):
                evaluation.write_summary("fake", entry, "")
                evaluation.write_comparison({"fake": entry})
                self.assertEqual(evaluation.refresh_training_caveats("fake", entry), 1)
                updated = report_path.read_text()
                self.assertEqual(updated, before_note + evaluation.training_caveat("fixture", entry, "") + after_note)
                self.assertEqual(evaluation.refresh_training_caveats("fake", entry), 0)
                self.assertEqual(report_path.read_text(), updated)
                self.assertNotIn("Obsolete wrapped", updated)
                individual = json.loads((self.root / "reports" / "fake" / "summary.json").read_text())
                overall = json.loads((self.root / "reports" / "summary.json").read_text())
                self.assertEqual(individual["datasets"][0]["in_training_data"], confirmed)
                self.assertEqual(individual["datasets"][0]["metrics"], meta["metrics"])
                self.assertEqual(overall["results"]["fake"]["fixture"]["in_training_data"], confirmed)
                self.assertEqual(overall["results"]["fake"]["fixture"]["accuracy"], .5)
                self.assertEqual(path.read_bytes(), before)
                report = (self.root / "reports" / "fake" / "summary.md").read_text()
                self.assertIn("not verified unseen", report)
                self.assertNotIn("*no*: it is not", report)
                caveat = evaluation.training_caveat("fixture", entry, "")
                self.assertNotIn("**Not in", caveat)
                self.assertIn(EXPOSURE_LEVELS["confirmed_exposure" if confirmed else "not_established"]["label"], caveat)

    def test_missing_registry_fields_remain_readable_without_trusting_old_positive(self):
        record = {"key": "custom", "metrics": {"accuracy": .7}, "in_training_data": True,
                  "inference_protocol": self.protocol, "training_evidence": ["obsolete"]}
        for entry in (None, {}, {"name": "old fixture"}):
            current = with_training_provenance(record, entry)
            self.assertFalse(current["in_training_data"])
            self.assertEqual(current["training_status"], "not_established")
            self.assertEqual(current["metrics"], record["metrics"])
            self.assertEqual(current["inference_protocol"], self.protocol)
            self.assertNotIn("training_evidence", current)
        self.assertTrue(record["in_training_data"])
        evidence = [{"source": "provided by registry", "stage": "backbone"}]
        current = with_training_provenance(record, {"training_data": ["custom"], "training_evidence": evidence})
        self.assertEqual(current["training_evidence"], evidence)
        current["training_evidence"][0]["stage"] = "changed in presentation copy"
        self.assertEqual(evidence[0]["stage"], "backbone")

    def test_four_evidence_levels_survive_plot_viewer_and_report_metadata(self):
        statuses = list(EXPOSURE_LEVELS)
        detail = {f"grade{i}": {"status": status, "scope": "upstream_teacher_lineage",
                               "reason": f"Audit reason for {status}", "evidence": ["https://example.org/source"]}
                  for i, status in enumerate(statuses)}
        entry = self.entry | {"training_data": list(detail), "training_exposure": detail}
        chosen, originals = {}, {}
        for i, status in enumerate(statuses):
            path = self.write_run(self.protocol, stem=f"grade{i}")
            meta = json.loads(path.read_text()) | {"key": f"grade{i}", "dataset": f"Grade {i}",
                    "in_training_data": status == "not_established",
                    "metrics": {"accuracy": .5, "unweighted_accuracy": .5, "macro_f1": .4}}
            path.write_text(json.dumps(meta))
            chosen[path.with_suffix(".csv")] = meta
            originals[path] = path.read_bytes()
        registry = {"fake": entry}
        config = self.viewer_config()
        selected = plot_results.load_results("all", "all", [], registry=registry, root=self.root)["fake"]
        with mock.patch.object(make_viewer, "ROOT", self.root):
            viewer = make_viewer.current_summary("fake", chosen, entry)
        self.assertEqual([r["training_status"] for r in viewer["datasets"]], statuses)
        self.assertEqual([r["training_status"] for r in viewer["shared"]], statuses)
        (self.root / "reports" / "fake").mkdir(parents=True)
        with mock.patch.object(evaluation, "LOADERS", dict.fromkeys(detail)):
            evaluation.write_summary("fake", entry, "")
            evaluation.write_comparison(registry)
        report = json.loads((self.root / "reports" / "fake" / "summary.json").read_text())
        comparison = (self.root / "reports" / "summary.md").read_text()
        for i, status in enumerate(statuses):
            key = f"grade{i}"
            for row in (selected[key], viewer["datasets"][i], report["datasets"][i]):
                self.assertEqual(row["training_status"], status)
                self.assertEqual(row["in_training_data"], i < 2)
                self.assertEqual(row["training_reason"], detail[key]["reason"])
                self.assertEqual(row["metrics"], {"accuracy": .5, "unweighted_accuracy": .5, "macro_f1": .4})
            self.assertIn(EXPOSURE_LEVELS[status]["label"], evaluation.training_caveat(key, entry, ""))
            self.assertIn("50.0%" + EXPOSURE_LEVELS[status]["marker"], comparison)
        for path, before in originals.items():
            self.assertEqual(path.read_bytes(), before)

    def test_possible_and_unknown_change_figure_fingerprint_even_when_both_flags_are_false(self):
        self.viewer_config()
        path = self.write_run(self.protocol)
        before = path.read_bytes()
        results = []
        for status in ("possible_exposure", "not_established"):
            entry = self.entry | {"training_exposure": {"fixture": {"status": status}}}
            results.append(plot_results.load_results("all", "all", [], registry={"fake": entry}, root=self.root))
            row = results[-1]["fake"]["fixture"]
            self.assertFalse(row["in_training_data"])
            self.assertEqual(row["inference_protocol"], self.protocol)
            self.assertEqual(row["metrics"], {"accuracy": .5})
        self.assertNotEqual(plot_results.results_fingerprint(results[0]), plot_results.results_fingerprint(results[1]))
        self.assertEqual(path.read_bytes(), before)

    def test_training_patterns_are_unified_and_pending_models_have_no_score(self):
        statuses = list(EXPOSURE_LEVELS)
        existing = [f"model{i}" for i in range(4)]
        models = existing + ["gpt_realtime", "gemini"]
        names = {model: model for model in models}
        colours = dict(zip(models, plot_results.COLOURS))
        results = {model: {"fixture": {"key": "fixture", "dataset": "Fixture", "classes": FOUR,
                    "clips": 2, "metrics": {"accuracy": .5}, "training_status": status,
                    "in_training_data": status == "not_established"}}
                   for model, status in zip(existing, statuses)}
        before = json.dumps(results, sort_keys=True)
        plt = plot_results.setup_matplotlib()
        captured = []
        def capture(figure, _plt, stem):
            captured.append(figure)
            return stem
        try:
            with mock.patch.object(plot_results, "save", side_effect=capture):
                plot_results.per_dataset_figure(plt, "accuracy", results, models, names, colours,
                                                ["fixture"], 4, self.root)
                plot_results.overview_figure(plt, "accuracy", results, models, names, colours,
                                            ["fixture"], self.root)
            panels, overview = captured
            patches = panels.axes[0].patches
            self.assertEqual(len(patches), 4)  # no dummy zero bars for reserved model slots
            self.assertEqual([p.get_hatch() or "" for p in patches], ["///", "///", "", ""])
            self.assertEqual([p.get_height() for p in patches], [50.0] * 4)
            self.assertLess(patches[1].get_x() - (patches[0].get_x() + patches[0].get_width()), .2)
            self.assertAlmostEqual(patches[0].get_width(), plot_results.PANEL_LAYOUT["bar_width"])
            labels = [" ".join(t.get_text().split()) for t in panels.legends[0].texts]
            self.assertEqual(labels.count(plot_results.TRAINING_LABEL), 1)
            self.assertIn("gpt_realtime (pending)", labels)
            self.assertIn("gemini (pending)", labels)
            self.assertFalse(any("upstream" in label.lower() or "possible" in label.lower() for label in labels))
            self.assertEqual(len(panels.texts), 1)  # title only; no explanatory footer
            texts = [t.get_text() for t in overview.axes[0].texts]
            self.assertEqual(texts.count("50.0†"), 2)
            self.assertEqual(texts.count("50.0"), 2)
            self.assertEqual(texts.count("—"), 2)
            self.assertFalse(any("0.0" == value for value in texts))
            self.assertEqual(json.dumps(results, sort_keys=True), before)
            panels.canvas.draw()
            renderer = panels.canvas.get_renderer()
            numbers = [t for t in panels.axes[0].texts if t.get_text() == "50"]
            self.assertEqual(len(numbers), 4)  # panel display rounds percentages, without rounding bar heights
            for first, second in zip(numbers, numbers[1:]):
                self.assertFalse(first.get_window_extent(renderer).overlaps(second.get_window_extent(renderer)))
            overview.canvas.draw()
        finally:
            for figure in captured:
                plt.close(figure)

    def test_weighted_f1_is_preserved_in_viewer_and_plots_without_filling_legacy_scores(self):
        self.viewer_config()
        path = self.write_run(self.protocol)
        original = json.loads(path.read_text())
        for metrics in ({"accuracy": .5, "weighted_f1": .72}, {"accuracy": .5}):
            with self.subTest(metrics=metrics), mock.patch.object(make_viewer, "ROOT", self.root):
                metadata = original | {"metrics": metrics}
                path.write_text(json.dumps(metadata))
                before = path.read_bytes()
                selected = plot_results.load_results("all", "all", [], registry={"fake": self.entry}, root=self.root)
                summary = make_viewer.current_summary("fake", {path.with_suffix(".csv"): metadata}, self.entry)
                self.assertEqual(selected["fake"]["fixture"]["metrics"], metrics)
                self.assertEqual(summary["datasets"][0]["metrics"], metrics)
                self.assertEqual(path.read_bytes(), before)

    def test_result_metric_switch_updates_only_the_selected_explanation_and_formula(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js is required to exercise the viewer's metric switch")
        template = (PROJECT / "scripts/viewer/templates/viewer.html").read_text()
        implementation = "const METRICS =" + template.split("const METRICS =", 1)[1].split("function modelResults", 1)[0]
        harness = r'''
const assert = require("node:assert/strict");
const state = { metric: "accuracy" };
function el(tag, props = {}, ...children) {
  return { tag, ...props, children,
    replaceChildren(...next) { this.children = next; } };
}
const document = {
  createElementNS(namespaceURI, tag) {
    return { tag, namespaceURI, children: [], attrs: {},
      append(...children) { this.children.push(...children); },
      setAttribute(name, value) { this.attrs[name] = value; } };
  },
};
let activeSwitcher;
function segmented(container, options, metric, onPick) {
  activeSwitcher = container;
  container.options = options;
  container.metric = metric;
  container.pick = onPick;
}
const pct = value => typeof value === "number" ? `${100 * value}%` : "–";
const levelValue = value => el("span", { textContent: pct(value) });
const td = value => el("td", { textContent: value });
const partOfTraining = () => false;
const trainingMarker = () => "";
const trainingLabel = () => "";
const byDatasetType = items => [["Fixture", items]];
const resultsTable = (columns, groups, label) => ({ columns, groups, label });
const legend = () => el("div", { className: "legend" });
'''
        assertions = r'''
const models = [{ name: "Fixture model", summary: { datasets: [
  { key: "fixture", dataset: "Fixture", emotions: 2, clips: 10,
    metrics: { accuracy: .8, unweighted_accuracy: .7, macro_f1: .65, weighted_f1: .75 } },
] } }];
const view = comparison(models);
const explanation = view.find(item => item.className === "metric-note");
assert.equal(explanation.attrs["aria-live"], "polite");
const text = item => typeof item === "string" ? item
  : (item.textContent || "") + (item.children || []).map(text).join("");
const descendants = item => typeof item === "string" ? []
  : [item, ...(item.children || []).flatMap(descendants)];
const expected = {
  accuracy: { label: "Accuracy equals", mathCount: 1, fractionCount: 1, sumCount: 0 },
  unweighted_accuracy: { label: "UAR equals", mathCount: 2, fractionCount: 2, sumCount: 1 },
  macro_f1: { label: "Macro F1 equals", mathCount: 2, fractionCount: 4, sumCount: 1 },
  weighted_f1: { label: "Weighted F1 equals", mathCount: 2, fractionCount: 4, sumCount: 1 },
};
assert.deepEqual(activeSwitcher.options.map(([key]) => key), Object.keys(expected));
for (const [metric, specification] of Object.entries(expected)) {
  activeSwitcher.pick(metric);
  assert.equal(state.metric, metric);
  assert.equal(activeSwitcher.metric, metric);
  const formulaNodes = explanation.children.filter(item => item.className === "metric-formula");
  assert.equal(formulaNodes.length, 1);
  const nodes = descendants(formulaNodes[0]);
  const maths = nodes.filter(item => item.tag === "math");
  assert.equal(maths.length, specification.mathCount);
  assert.ok(maths[0].attrs.alttext.startsWith(specification.label));
  for (const math of maths) {
    assert.equal(math.namespaceURI, "http://www.w3.org/1998/Math/MathML");
    assert.equal(math.attrs.display, "block");
    assert.equal(math.attrs["aria-label"], math.attrs.alttext);
    assert.ok(math.attrs.alttext.length > 20);
  }
  assert.equal(nodes.filter(item => item.tag === "mfrac").length, specification.fractionCount);
  const sums = nodes.filter(item => item.tag === "munderover");
  assert.equal(sums.length, specification.sumCount);
  for (const sum of sums) {
    assert.deepEqual(sum.children.map(item => item.tag), ["mo", "mrow", "mi"]);
    assert.equal(text(sum.children[0]), "∑");
    assert.equal(text(sum.children[1]), "k=1");
    assert.equal(text(sum.children[2]), "K");
  }
  if (metric !== "accuracy") assert.ok(nodes.some(item => item.tag === "msub"));
  const firstFraction = nodes.find(item => item.tag === "mfrac");
  if (metric === "accuracy") {
    assert.equal(text(firstFraction.children[0]), "correct clips");
    assert.equal(text(firstFraction.children[1]), "N");
  } else if (metric === "weighted_f1") {
    assert.equal(text(firstFraction.children[0]), "nk");
    assert.equal(text(firstFraction.children[1]), "N");
  } else {
    assert.equal(text(firstFraction.children[0]), "1");
    assert.equal(text(firstFraction.children[1]), "K");
  }
  assert.ok(text(explanation).includes("Invalid answers count as wrong"));
  for (const [other, otherSpecification] of Object.entries(expected)) {
    if (other !== metric) assert.ok(!maths.some(math => math.attrs.alttext.startsWith(otherSpecification.label)));
  }
}
assert.ok(text(explanation).includes("Class support"));
assert.ok(text(explanation).includes("precision"));
assert.ok(text(explanation).includes("recall"));
assert.ok(text(explanation).includes("resolved truth"));
console.log("Metric explanation switching: PASS");
'''
        result = subprocess.run([node, "-"], input=harness + implementation + assertions,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_weighted_f1_charts_keep_zero_and_missing_distinct_without_accuracy_chance(self):
        models = ["scored", "zero", "legacy"]
        names = dict(zip(models, models))
        colours = dict(zip(models, plot_results.COLOURS))
        results = {model: {"fixture": {"key": "fixture", "dataset": "Fixture", "classes": FOUR,
                    "clips": 2, "metrics": metrics, "in_training_data": False}}
                   for model, metrics in zip(models, ({"weighted_f1": .72}, {"weighted_f1": 0.0},
                                                      {"macro_f1": .5}))}
        plt = plot_results.setup_matplotlib()
        captured = []
        def capture(figure, _plt, stem):
            captured.append(figure)
            return stem
        try:
            with mock.patch.object(plot_results, "save", side_effect=capture):
                plot_results.per_dataset_figure(plt, "weighted_f1", results, models, names, colours,
                                                ["fixture"], 1, self.root)
                plot_results.overview_figure(plt, "weighted_f1", results, models, names, colours,
                                            ["fixture"], self.root)
            panels, overview = captured
            self.assertEqual([patch.get_height() for patch in panels.axes[0].patches], [72.0, 0.0])
            self.assertEqual(len(panels.axes[0].lines), 0)
            self.assertNotIn("chance level", [text.get_text() for text in panels.legends[0].texts])
            self.assertEqual(panels.axes[0].get_ylabel(), "Weighted F1 (%)")
            matrix = overview.axes[0].collections[0].get_array()  # vector QuadMesh, also retained in SVG
            self.assertEqual(matrix.shape, (1, 3))
            self.assertTrue(matrix.mask[0, 2])
            self.assertEqual([text.get_text() for text in overview.axes[0].texts], ["72.0", "0.0", "—"])
            self.assertNotIn("Chance\nreference", [tick.get_text() for tick in overview.axes[0].get_xticklabels()])
        finally:
            for figure in captured:
                plt.close(figure)

    def test_viewer_and_plots_use_audited_weighted_f1_only_for_matching_source_files(self):
        self.viewer_config()
        path = self.write_run(self.protocol)
        metadata = json.loads(path.read_text())
        audit_path = self.root / "reports/fake/fixture_offline_audit.json"
        audit_path.parent.mkdir(parents=True)
        audit_path.write_text(json.dumps({"model": "fake", "key": "fixture", "clips": metadata["clips"],
            "recomputed_metrics": {"weighted_f1": .72}, "source_artifacts": [
                {"path": source.relative_to(self.root).as_posix(),
                 "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                for source in (path, path.with_suffix(".csv"))]}))
        before = path.read_bytes()
        for changed in (False, True):
            if changed:
                with path.with_suffix(".csv").open("a") as output:
                    output.write("\n")
            with self.subTest(changed=changed), mock.patch.object(make_viewer, "ROOT", self.root):
                selected = plot_results.load_results("all", "all", [], registry={"fake": self.entry}, root=self.root)
                summary = make_viewer.current_summary("fake", {path.with_suffix(".csv"): metadata}, self.entry)
                for metrics in (selected["fake"]["fixture"]["metrics"], summary["datasets"][0]["metrics"]):
                    if changed:
                        self.assertNotIn("weighted_f1", metrics)
                    else:
                        self.assertEqual(metrics["weighted_f1"], .72)
                    self.assertEqual(metrics["accuracy"], metadata["metrics"]["accuracy"])
                self.assertEqual(path.read_bytes(), before)

    def test_placeholder_order_stays_fixed_when_results_arrive_without_fabricating_records(self):
        registry = {"ser": {"name": "SER"}, "gpt_realtime": {"name": "GPT Realtime"},
                    "gemini": {"name": "Gemini"}}
        placeholders = {"gpt_realtime": "GPT Realtime", "gemini": "Gemini 3.8 Flash"}
        results = {"ser": {"fixture": {"model_name": "SER", "metrics": {"accuracy": .5}}}}
        before = json.dumps(results, sort_keys=True)
        models, names = plot_results.chart_models(results, registry, placeholders=placeholders)
        self.assertEqual(models, list(registry))
        self.assertEqual(names["gemini"], "Gemini 3.8 Flash")
        self.assertEqual(json.dumps(results, sort_keys=True), before)
        results["gemini"] = {"fixture": {"model_name": "Gemini 3.8 Flash", "metrics": {"accuracy": .6}}}
        self.assertEqual(plot_results.chart_models(results, registry, placeholders=placeholders)[0], models)
        self.assertEqual(plot_results.chart_models(results, registry, wanted=["ser"],
                                                    placeholders=placeholders)[0], ["ser"])

    def test_sixteen_dataset_grid_keeps_a_two_row_legend_below_all_panels(self):
        models = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper",
                  "vox_profile_wavlm", "qwen2_5_omni_7b", "qwen2_audio_7b", "gemma4_e4b",
                  "gpt_realtime", "gemini"]
        names = dict(zip(models, ["emotion2vec+ large", "MERaLiON-SER-v1", "Melati", "Vox-Profile Whisper",
                    "Vox-Profile WavLM", "Qwen2.5-Omni 7B", "Qwen2-Audio 7B Instruct", "Gemma 4 E4B",
                    "GPT Realtime", "Gemini 3.8 Flash"]))
        datasets = ["meld", "emotiontalk", "dailytalk", "multidialog", "crema_d", "ravdess", "savee", "jl_corpus",
                    "tess", "mead", "mess", "mcae_spps", "thai_ser", "nemo", "emovo", "emozionalmente"]
        colours = dict(zip(models, plot_results.COLOURS))
        # Synthetic artist fixtures only; no prediction files or real missing-dataset results are written.
        results = {m: {key: {"key": key, "dataset": f"Fixture {i}", "classes": FOUR, "clips": 2,
                            "metrics": {"accuracy": 1.0 if mi == 2 else .64,
                                        "macro_f1": .99 if mi == 2 else .61}, "in_training_data": False}
                       for i, key in enumerate(datasets)} for mi, m in enumerate(models[:-2])}
        plt = plot_results.setup_matplotlib()
        captured = []
        def capture(figure, _plt, stem):
            captured.append(figure)
            return stem
        try:
            for metric, entries in (("accuracy", 12), ("macro_f1", 11)):
                with self.subTest(metric=metric), mock.patch.object(plot_results, "save", side_effect=capture):
                    plot_results.per_dataset_figure(plt, metric, results, models, names, colours,
                                                    datasets, 4, self.root)
                    figure = captured[-1]
                    self.assertEqual(len(figure.axes), 16)
                    self.assertEqual(len({round(ax.get_position().x0, 5) for ax in figure.axes}), 4)
                    self.assertEqual(len({round(ax.get_position().y0, 5) for ax in figure.axes}), 4)
                    self.assertTrue(all(ax.axison for ax in figure.axes))
                    self.assertEqual([ax.get_title() for ax in figure.axes],
                                     [plot_results.panel_title(key, {}) for key in datasets])
                    legend = figure.legends[0]
                    self.assertEqual(legend._ncols, 6)
                    self.assertEqual(len(legend.texts), entries)
                    self.assertEqual((entries + legend._ncols - 1) // legend._ncols, 2)
                    self.assertTrue(all("\n" not in text.get_text() for text in legend.texts))
                    figure.canvas.draw()
                    renderer = figure.canvas.get_renderer()
                    for ax in figure.axes:
                        title_box = ax.title.get_window_extent(renderer)
                        numbers = [text for text in ax.texts if text.get_text().rstrip("*").isdigit()]
                        for text in numbers:
                            self.assertFalse(title_box.overlaps(text.get_window_extent(renderer)))
                        for first, second in zip(numbers, numbers[1:]):
                            self.assertFalse(first.get_window_extent(renderer).overlaps(
                                             second.get_window_extent(renderer)))
                    for ax in figure.axes[:4]:
                        self.assertFalse(ax.title.get_window_extent(renderer).overlaps(
                                         figure._suptitle.get_window_extent(renderer)))
                    baselines = {round(text.get_transform().transform(text.get_position())[1], 3)
                                 for text in legend.texts}
                    self.assertEqual(len(baselines), 2)  # two actual text lines, not two rows of wrapped items
                    box = legend.get_window_extent(figure.canvas.get_renderer())
                    self.assertGreaterEqual(box.x0, 0)
                    self.assertLessEqual(box.x1, figure.bbox.width)
                    self.assertGreaterEqual(box.y0, 0)
                    self.assertLess(box.y1, min(ax.bbox.y0 for ax in figure.axes))
                    self.assertEqual([bool(ax.get_yticklabels()) for ax in figure.axes],
                                     [i % 4 == 0 for i in range(16)])
                    self.assertEqual(len(figure.texts), 1)  # no explanatory footer under the model legend
        finally:
            for figure in captured:
                plt.close(figure)

    def test_dataset_order_and_layout_are_configured_without_losing_future_results(self):
        settings = yaml.safe_load((PROJECT / "config/plots.yaml").read_text())
        expected = ["meld", "emotiontalk", "dailytalk", "multidialog", "crema_d", "ravdess", "savee", "jl_corpus",
                    "tess", "mead", "mess", "mcae_spps", "thai_ser", "nemo", "emovo", "emozionalmente"]
        self.assertEqual(settings["dataset_order"], expected)
        results = {"fixture": dict.fromkeys(reversed(expected), {})}
        self.assertEqual(plot_results.chart_datasets(results, settings["dataset_order"]), expected)
        results["fixture"]["future_dataset"] = {}
        self.assertEqual(plot_results.chart_datasets(results, settings["dataset_order"]),
                         expected + ["future_dataset"])
        result_fingerprint = plot_results.results_fingerprint(results)
        fingerprint = plot_results.presentation_fingerprint(settings)
        reordered = settings | {"dataset_order": expected[1:] + expected[:1]}
        restyled = settings | {"layout": settings["layout"] | {"panel_height": 2.25}}
        self.assertNotEqual(fingerprint, plot_results.presentation_fingerprint(reordered))
        self.assertNotEqual(fingerprint, plot_results.presentation_fingerprint(restyled))
        self.assertEqual(result_fingerprint, plot_results.results_fingerprint(results))
        with self.assertRaises(ValueError):
            plot_results.chart_datasets(results, ["meld", "meld"])
        with self.assertRaises(ValueError):
            plot_results.panel_layout({"bar_width": .8, "bar_spacing": .7})

    def test_dataset_titles_are_one_line_without_changing_formal_scope(self):
        cases = {
            "crema_d": ("CREMA-D", "American English", "CREMA-D (English)"),
            "mcae_spps": ("MCAE-SPPS", "Mandarin Chinese", "MCAE-SPPS (Mandarin)"),
            "thai_ser": ("THAI-SER (high-agreement subset)", "Thai", "THAI-SER (Thai)"),
            "mead": ("MEAD (8,000-clip subset)", "English", "MEAD (English)"),
            "jl_corpus": ("JL-Corpus (five main emotions)", "English (New Zealand)", "JL-Corpus (English)"),
            "savee": ("SAVEE", "British English", "SAVEE (English)"),
            "emozionalmente": ("Emozionalmente", "Italian", "Emozionalmente (Italian)"),
            "dailytalk": ("DailyTalk (validation)", "English", "DailyTalk (English)"),
            "emotiontalk": ("EmotionTalk (test)", "Mandarin", "EmotionTalk (Mandarin)"),
            "multidialog": ("MultiDialog (gold_both)", "English", "MultiDialog (English)"),
        }
        models, names, colours = ["fixture"], {"fixture": "Fixture"}, {"fixture": plot_results.COLOURS[0]}
        results = {"fixture": {key: {"key": key, "dataset": full, "language": language, "classes": FOUR,
                           "clips": 2, "metrics": {"accuracy": .54}, "in_training_data": False}
                    for key, (full, language, _) in cases.items()}}
        before = json.dumps(results, sort_keys=True)
        plt = plot_results.setup_matplotlib()
        captured = []
        def capture(figure, _plt, stem):
            captured.append(figure)
            return stem
        try:
            with mock.patch.object(plot_results, "save", side_effect=capture):
                plot_results.per_dataset_figure(plt, "accuracy", results, models, names, colours,
                                                list(cases), 4, self.root)
                plot_results.overview_figure(plt, "accuracy", results, models, names, colours,
                                            list(cases), self.root)
            panels, overview = captured
            panels.canvas.draw()
            renderer = panels.canvas.get_renderer()
            for ax, (_, (_, _, expected)) in zip(panels.axes, cases.items()):
                self.assertEqual(ax.get_title(), expected)
                self.assertNotIn("\n", ax.get_title())
                box = ax.title.get_window_extent(renderer)
                self.assertGreaterEqual(box.x0, ax.bbox.x0 - 4)
                self.assertLessEqual(box.x1, ax.bbox.x1 + 4)
            self.assertEqual([label.get_text() for label in overview.axes[0].get_yticklabels()],
                             [plot_results.DATASET_DISPLAY[key][0] for key in cases])
            self.assertEqual(json.dumps(results, sort_keys=True), before)
        finally:
            for figure in captured:
                plt.close(figure)


class ViewerExampleCoverageTests(Fixture):
    DATASET_KEYS = ("crema_d", "mcae_spps", "thai_ser", "nemo", "mess", "ravdess", "tess",
                    "jl_corpus", "mead", "meld", "emovo", "emozionalmente", "savee",
                    "emotiontalk", "dailytalk", "multidialog")

    def example_datasets(self, keys):
        return [{"key": key, "dataset": key, "runs": {"fixture": {"total": 1}}, "clips": [{
            "file": f"{key}.wav", "path": self.root / "data" / key / f"{key}.wav",
            "labels": {"emotion": "happy"},
            "models": {"fixture": {"predictions": {"prediction": "happy"}}},
        }]} for key in keys]

    def test_all_mode_keeps_all_sixteen_datasets_and_audio_paths(self):
        datasets = self.example_datasets(self.DATASET_KEYS)
        selected = make_viewer.site_datasets(datasets, "all")
        self.assertEqual([dataset["key"] for dataset in selected], list(self.DATASET_KEYS))
        self.assertEqual(selected, datasets)
        self.assertEqual([dataset["clips"][0]["path"] for dataset in selected],
                         [self.root / "data" / key / f"{key}.wav" for key in self.DATASET_KEYS])
        settings = yaml.safe_load((PROJECT / "config" / "viewer.yaml").read_text())
        self.assertEqual(settings["site_audio"], "all")

    def test_public_mode_filters_restricted_and_unknown_datasets(self):
        keys = self.DATASET_KEYS + ("unknown_dataset",)
        datasets = self.example_datasets(keys)
        selected = make_viewer.site_datasets(datasets, "public")
        allowed = ["crema_d", "mcae_spps", "thai_ser", "nemo", "mess", "ravdess", "tess",
                   "jl_corpus", "emozionalmente"]
        self.assertEqual([dataset["key"] for dataset in selected], allowed)
        self.assertEqual(selected, [dataset for dataset in datasets if dataset["key"] in allowed])
        self.assertEqual([dataset["key"] for dataset in datasets], list(keys))

    def test_site_audio_rejects_invalid_modes(self):
        datasets = self.example_datasets(self.DATASET_KEYS)
        for mode in ("private", "ALL", "", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                make_viewer.site_datasets(datasets, mode)


class FigureExportTests(unittest.TestCase):
    def assert_vector_svg(self, path):
        import xml.etree.ElementTree as ET
        artwork = ET.parse(path).getroot()
        self.assertFalse(any(element.tag.endswith("}image") for element in artwork.iter()),
                         "SVG must contain plotted vector artwork, not an embedded PNG")
        self.assertTrue(any(element.tag.endswith("}path") for element in artwork.iter()))
        return artwork

    def test_comparison_figures_export_png_and_vector_svg_without_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            results = {"fixture": {"meld": {"key": "meld", "dataset": "MELD", "classes": FOUR,
                        "clips": 2, "metrics": {"accuracy": .5}, "training_status": "not_established"}}}
            plt = plot_results.setup_matplotlib()
            arguments = (plt, "accuracy", results, ["fixture"], {"fixture": "Fixture"},
                         {"fixture": plot_results.COLOURS[0]}, ["meld"])
            plot_results.per_dataset_figure(*arguments, 1, out)
            plot_results.overview_figure(*arguments, out)
            self.assertEqual(len(list(out.glob("*.png"))), 2)
            self.assertEqual(len(list(out.glob("*.svg"))), 2)
            self.assertFalse(list(out.glob("*.pdf")))
            for path in out.glob("*.svg"):
                self.assert_vector_svg(path)

    def test_confusions_and_summary_grid_keep_vector_paths_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            ds = SimpleNamespace(classes=["angry", "neutral"], accept={})
            paths = [out / "first.png", out / "second.png"]
            for path, matrix in zip(paths, (np.array([[3, 1], [1, 2]]), np.array([[2, 2], [0, 3]]))):
                evaluation.plot_confusion(matrix, ds, ds.classes, "Fixture", "Saved predictions", "Truth", path)
                self.assertTrue(path.is_file())
                self.assert_vector_svg(path.with_suffix(".svg"))
            evaluation.stitch(paths, out / "summary.png", columns=2)
            artwork = self.assert_vector_svg(out / "summary.svg")
            identifiers = [element.attrib["id"] for element in artwork.iter() if "id" in element.attrib]
            self.assertEqual(len(identifiers), len(set(identifiers)))
            self.assertTrue((out / "summary.png").is_file())
            self.assertFalse(list(out.glob("*.pdf")))
            paths[0].with_suffix(".svg").unlink()
            evaluation.stitch(paths, out / "summary.png", columns=2)
            self.assertFalse((out / "summary.svg").exists())  # no stale grid or fake SVG wrapper


if __name__ == "__main__":
    unittest.main()

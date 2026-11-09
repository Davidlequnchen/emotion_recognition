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
            manifest = {"fingerprint": plot_results.results_fingerprint(selected), "metrics": ["accuracy"]}
            (folder / "results_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(make_viewer.load_plots()[0]["by_dataset"], "embedded")
            (folder / "results_manifest.json").write_text(json.dumps(manifest | {"fingerprint": "old"}))
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


if __name__ == "__main__":
    unittest.main()

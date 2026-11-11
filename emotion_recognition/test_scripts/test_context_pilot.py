"""Context experiment invariants, with metadata fixtures only: no torch, processor or model weights."""
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("context_pilot", PROJECT / "scripts/evaluation/context_pilot.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def utterance(key, dialogue, speaker, start, gold="neutral", duration=1.0):
    return {"dataset": "emotiontalk", "id": key, "dialogue": dialogue, "speaker": speaker,
            "speaker_key": speaker, "start": start, "end": start + duration, "duration": duration,
            "transcript": "A normal sentence.", "split": "train", "turn_index": int(start) + 1,
            "audio": f"data/emotiontalk/{key}.wav", "gold": gold, "caption": "SECRET_CAPTION"}


class ContextPilotTests(unittest.TestCase):
    def test_weighted_f1_preserves_class_support_and_invalid_errors(self):
        rows = [{"gold": "neutral", "prediction": "neutral"} for _ in range(9)]
        rows += [{"gold": "angry", "prediction": "invalid"}]
        result = pilot.scores(rows, ["neutral", "angry"])
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["per_class"]["angry"]["support"], 1)
        self.assertAlmostEqual(result["macro_f1"], .5)
        self.assertAlmostEqual(result["weighted_f1"], .9)
        self.assertIsNone(pilot.scores([], ["neutral", "angry"])["weighted_f1"])

    def setUp(self):
        self.config = yaml.safe_load((PROJECT / "config/context_pilot.yaml").read_text())
        self.protocol = {"labels": self.config["labels"], "prompts": self.config["prompts"]}
        self.case = {"target": utterance("target", "d0", "A", 5),
                     "history": utterance("history", "d0", "B", 2),
                     "unrelated": utterance("donor", "d1", "B", 2), "gold": "SECRET_GOLD"}

    def test_four_conditions_keep_target_last_and_hide_annotations(self):
        expected_audio_count = {"H0": 1, "H1": 2, "H2": 1, "H3": 2}
        for condition, count in expected_audio_count.items():
            with self.subTest(condition=condition):
                messages = pilot.build_messages(self.case, condition, self.protocol)
                audio = [x["audio"] for message in messages for x in message["content"] if x["type"] == "audio"]
                self.assertEqual(len(audio), count)
                self.assertEqual(audio[-1], self.case["target"]["audio"])
                text = json.dumps(messages)
                self.assertNotIn("SECRET_GOLD", text)
                self.assertNotIn("SECRET_CAPTION", text)
                self.assertNotIn('"gold"', text)
        for condition in ("H0", "H1", "H3"):
            self.assertNotIn(self.case["history"]["transcript"],
                             json.dumps(pilot.build_messages(self.case, condition, self.protocol)))
        self.assertIn(self.case["history"]["transcript"],
                      json.dumps(pilot.build_messages(self.case, "H2", self.protocol)))

    def test_audio_history_control_changes_audio_only(self):
        true = pilot.build_messages(self.case, "H1", self.protocol)
        unrelated = pilot.build_messages(self.case, "H3", self.protocol)
        true[1]["content"][1]["audio"] = "removed"
        unrelated[1]["content"][1]["audio"] = "removed"
        self.assertEqual(true, unrelated)

    def test_selection_is_balanced_reproducible_causal_and_donor_gold_independent(self):
        config = copy.deepcopy(self.config)
        config["samples"] = 7
        config["selection"]["max_per_dialogue"] = 1
        rows = []
        for i, lab in enumerate(config["labels"]):
            rows += [utterance(f"history{i}", f"d{i}", "B", 0),
                     utterance(f"target{i}", f"d{i}", "A", 2, lab)]
        rows += [utterance(f"extra{i}", f"extra{i}", "B", 0) for i in range(3)]
        selected, stats = pilot.select_cases(rows, config)
        again, _ = pilot.select_cases(list(reversed(rows)), config)
        self.assertEqual([c["target"]["id"] for c in selected], [c["target"]["id"] for c in again])
        self.assertEqual(set(stats["selected_by_gold"].values()), {1})
        selected_ids = {c["target"]["id"] for c in selected}
        for case in selected:
            self.assertLessEqual(case["history"]["end"], case["target"]["start"])
            self.assertNotIn("gold", case["history"])
            self.assertNotEqual(case["unrelated"]["dialogue"], case["target"]["dialogue"])
            self.assertEqual(case["unrelated"]["speaker"], case["history"]["speaker"])
            self.assertNotIn(case["unrelated"]["id"], selected_ids)
        # Change only first-turn/donor emotions: target membership and H3 assignment must stay the same.
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["start"] == 0:
                row["gold"] = "fearful"
        updated, _ = pilot.select_cases(changed, config)
        self.assertEqual([(c["target"]["id"], c["unrelated"]["id"]) for c in selected],
                         [(c["target"]["id"], c["unrelated"]["id"]) for c in updated])

    def test_selection_rejects_nontrain_and_excessive_duration(self):
        foreign = utterance("test", "d", "A", 0)
        foreign["split"] = "test"
        with self.assertRaisesRegex(ValueError, "only train"):
            pilot.select_cases([foreign], self.config)
        with self.assertRaisesRegex(ValueError, "Not enough eligible"):
            pilot.select_cases([utterance("long", "d", "A", 2, duration=20)], self.config)

    def test_manifest_hash_covers_history_text_and_audio_identity(self):
        payload = {"protocol": self.protocol, "cases": [self.case], "selection_audit": {}}
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "manifest.json"
            pilot.write_json(file, payload | {"prepared_hash": pilot.digest(payload)})
            self.assertEqual(pilot.load_manifest(file)["prepared_hash"], pilot.digest(payload))
            payload["cases"][0]["history"]["transcript"] = "Changed history"
            pilot.write_json(file, payload | {"prepared_hash": "wrong"})
            with self.assertRaisesRegex(ValueError, "changed"):
                pilot.load_manifest(file)

    def test_torn_final_journal_line_is_recoverable_but_middle_corruption_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "responses.jsonl"
            file.write_bytes(b'{"id": 1}\n{"id":')
            rows, valid = pilot.read_responses(file)
            self.assertEqual(rows, [{"id": 1}])
            self.assertEqual(valid, b'{"id": 1}\n')
            file.write_bytes(b'{"id":\n{"id": 2}\n')
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                pilot.read_responses(file)

    def test_analysis_uses_four_condition_intersection_and_rejects_mixed_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            second = copy.deepcopy(self.case)
            second["target"]["id"] = "unfinished"
            second["gold"] = "happy"
            first = copy.deepcopy(self.case)
            first["gold"] = "neutral"
            payload = {"protocol": self.protocol, "cases": [first, second], "selection_audit": {}}
            manifest = root / "manifest.json"
            pilot.write_json(manifest, payload | {"prepared_hash": pilot.digest(payload)})
            execution = {"prepared_hash": pilot.digest(payload), "device": "fixture"}
            execution["execution_hash"] = pilot.digest(execution)
            pilot.write_json(root / "execution.json", execution)
            rows = []
            for case, conditions in [(first, pilot.CONDITIONS), (second, ("H0",))]:
                for condition in conditions:
                    messages = pilot.build_messages(case, condition, self.protocol)
                    rows.append({"id": case["target"]["id"], "condition": condition,
                        "prediction": "neutral", "answers": ["neutral"], "seconds": 0.1,
                        "execution_hash": execution["execution_hash"],
                        "request_hash": pilot.digest([execution["execution_hash"], case["target"]["id"], condition, messages])})
            journal = root / "responses.jsonl"
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
            summary = pilot.analyze(manifest, root)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["paired_complete_targets"], 1)
            self.assertTrue(all(s["accuracy"] == 1.0 for s in summary["metrics_on_common_targets"].values()))
            rows[0]["request_hash"] = "stale"
            journal.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, "request identity"):
                pilot.analyze(manifest, root)

    def test_interrupted_run_resumes_completed_requests_without_loading_real_libraries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = copy.deepcopy(self.case)
            case["gold"] = "neutral"
            protocol = self.protocol | {"model_dir": "models/fake", "config": {"device": "mps"}}
            payload = {"protocol": protocol, "cases": [case], "selection_audit": {}}
            manifest = payload | {"prepared_hash": pilot.digest(payload)}
            path = root / "manifest.json"
            pilot.write_json(path, manifest)
            calls = []

            def predict(messages):
                calls.append(messages)
                if len(calls) == 2:
                    raise RuntimeError("simulated out of memory")
                return {"prediction": "neutral", "answers": ["neutral"], "seconds": .1}

            modules = {"omni_context": types.SimpleNamespace(OmniContextPredictor=lambda *args: predict),
                       "torch": types.SimpleNamespace(),
                       "audio_llm": types.SimpleNamespace(dtype_for=lambda device: "bfloat16")}
            with mock.patch.dict(sys.modules, modules), mock.patch.object(pilot, "load_manifest", return_value=manifest), \
                    mock.patch.object(pilot.importlib.metadata, "version", return_value="fixture"):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    pilot.run(path, "mps")
                summary = pilot.run(path, "mps")
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["paired_complete_targets"], 1)
                self.assertEqual(len(calls), 5)  # one successful request is reused; only failure is retried
                self.assertTrue(all(r["invalid"] == 0 for r in summary["metrics_on_common_targets"].values()))
                pilot.run(path, "mps")
                self.assertEqual(len(calls), 5)  # a complete rerun only analyzes, without constructing the model


if __name__ == "__main__":
    unittest.main()

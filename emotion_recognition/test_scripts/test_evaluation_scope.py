"""Run the actual finished_run function in isolation: no torch/model imports or dataset loading.

python -m unittest discover -s test_scripts -p test_evaluation_scope.py -v
"""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "common"))
from eval_datasets import default_evaluation_subset


class FinishedScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="finished_scope_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "outputs" / "fake"
        self.folder.mkdir(parents=True)
        self.protocol = {"label_scope": "dataset", "labels": ["happy", "sad"], "prompt_hash": "test", "model_id": "fake"}
        source = PROJECT / "scripts" / "evaluation" / "evaluate.py"
        function = next(node for node in ast.parse(source.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "finished_run")
        self.dataset = mock.Mock(side_effect=AssertionError("finished_run must not reload or extract datasets"))
        self.namespace = {"ROOT": self.root, "SUBSETS": {}, "sys": sys, "json": json,
                          "default_evaluation_subset": default_evaluation_subset, "dataset": self.dataset}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), self.namespace)
        self.finished_run = self.namespace["finished_run"]

    def write_finished(self, key, scope, clips=10):
        path = self.folder / f"{key}_predictions.json"
        path.write_text(json.dumps({"subset": scope, "clips": clips, "inference_protocol": self.protocol}))
        path.with_suffix(".csv").touch()
        return path

    def test_three_canonical_dialogue_results_resume_without_config_subset_entries(self):
        for key, scope, clips in [("emotiontalk", "split:test", 1929),
                                  ("dailytalk", "split:validation", 1197),
                                  ("multidialog", "cohort:gold_both", 7672)]:
            with self.subTest(key=key):
                path = self.write_finished(key, scope, clips)
                before = path.read_bytes()
                self.assertEqual(self.finished_run("fake", key, None, self.protocol), path)
                self.assertEqual(path.read_bytes(), before)
        self.dataset.assert_not_called()

    def test_different_split_cohort_primary_or_clip_subset_still_refused(self):
        for key, scope in [("emotiontalk", "split:train"), ("dailytalk", "split:train"),
                           ("multidialog", "cohort:all"), ("multidialog", "cohort:gold_both;splits:test_freq"),
                           ("emotiontalk", "split:test;primary_dyadic"), ("dailytalk", "split:validation;clip_ids")]:
            with self.subTest(key=key, scope=scope):
                self.write_finished(key, scope)
                with self.assertRaisesRegex(SystemExit, "another subset/split/cohort"):
                    self.finished_run("fake", key, None, self.protocol)

    def test_original_dataset_subset_rules_unchanged(self):
        self.namespace["SUBSETS"] = {"thai_ser": "high_agreement", "mead": "stratified_8000"}
        for key, scope in [("meld", ""), ("thai_ser", "high_agreement"), ("mead", "stratified_8000")]:
            with self.subTest(key=key):
                path = self.write_finished(key, scope)
                self.assertEqual(self.finished_run("fake", key, None), path)
                self.write_finished(key, "different")
                with self.assertRaises(SystemExit):
                    self.finished_run("fake", key, None)

    def test_protocol_and_completion_checks_are_preserved(self):
        path = self.write_finished("emotiontalk", "split:test", 1929)
        self.assertIsNone(self.finished_run("fake", "emotiontalk", None, self.protocol | {"prompt_hash": "old"}))
        path.with_suffix(".csv").unlink()
        self.assertIsNone(self.finished_run("fake", "emotiontalk", None, self.protocol))


if __name__ == "__main__":
    unittest.main()

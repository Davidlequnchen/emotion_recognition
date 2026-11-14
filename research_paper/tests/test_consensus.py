"""Synthetic offline checks; no models, audio or remote requests."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import consensus


def runs_for(pairs, classes=None, accept=None):
    result = []
    for at in range(len(pairs[0][1])):
        rows = {f"data/crema_d/{i}.wav": {"gold": gold, "prediction": votes[at]}
                for i, (gold, votes) in enumerate(pairs)}
        result.append(consensus.Run(f"m{at}", "crema_d", Path(f"m{at}.json"), {},
                                    classes or ["angry", "neutral"], accept or {}, rows, []))
    return result


class ConsensusTests(unittest.TestCase):
    def test_filters_retain_invalids_and_reject_ties(self):
        runs = runs_for([("angry", ["angry"] * 3), ("neutral", ["neutral", "neutral", "angry"]),
                         ("neutral", ["angry", "angry", "neutral"]), ("angry", ["angry", "neutral", None]),
                         ("angry", ["angry", None, None])])
        rows = {r["rule"]: r for r in consensus.analyze_group("crema_d", "fixture", runs, sorted(runs[0].rows))}
        self.assertEqual((rows["unanimous"]["accepted"], rows["unanimous"]["accepted_correct"]), (1, 1))
        self.assertEqual(rows["unanimous"]["coverage"], .2)
        self.assertEqual(rows["unanimous"]["conditional_precision"], 1)
        self.assertEqual((rows["majority"]["accepted"], rows["majority"]["accepted_correct"]), (3, 2))
        self.assertAlmostEqual(rows["majority"]["conditional_precision"], 2 / 3)
        self.assertEqual(rows["majority"]["accepted_correct_fraction"], .4)
        self.assertEqual(rows["majority"]["samples_with_vote_ties"], 1)
        self.assertEqual(rows["majority"]["samples_with_invalid_votes"], 2)

    def test_composite_all_correct_is_not_literal_unanimity(self):
        runs = runs_for([("excited", ["happy", "surprised", "happy"])], ["happy", "surprised"],
                        {"excited": ["happy", "surprised"]})
        rows = {r["rule"]: r for r in consensus.analyze_group("crema_d", "fixture", runs, list(runs[0].rows))}
        self.assertEqual(rows["unanimous"]["all_models_individually_correct"], 1)
        self.assertEqual(rows["unanimous"]["unanimous_and_correct"], 0)
        self.assertIsNone(rows["unanimous"]["conditional_precision"])
        self.assertEqual(rows["majority"]["accepted_correct"], 1)

    def test_no_acceptance_is_undefined_precision_not_zero(self):
        runs = runs_for([("angry", [None, None])])
        for row in consensus.analyze_group("crema_d", "fixture", runs, list(runs[0].rows)):
            self.assertEqual(row["coverage"], 0)
            self.assertIsNone(row["conditional_precision"])
            self.assertEqual(row["accepted_correct_fraction"], 0)

    def test_matched_intersection_and_truth_rules(self):
        runs = runs_for([("angry", ["angry", "angry"]), ("neutral", ["neutral", "neutral"])])
        runs[1].rows.pop("data/crema_d/1.wav")
        available = {r.model: {"crema_d": r} for r in runs}
        models = [r.model for r in runs]
        _, common = consensus.prepare_cohort("crema_d", models, available)
        self.assertEqual(common, ["data/crema_d/0.wav"])
        runs[1].rows[common[0]]["gold"] = "neutral"
        with self.assertRaisesRegex(ValueError, "ground-truth mismatch"):
            consensus.prepare_cohort("crema_d", models, available)
        runs[1].rows[common[0]]["gold"] = "angry"
        runs[1].accept = {"excited": ["angry"]}
        with self.assertRaisesRegex(ValueError, "candidate/accepted-label"):
            consensus.prepare_cohort("crema_d", models, available)

    def artifact(self, root, model, protocol=None, dataset="crema_d"):
        path = root / "outputs" / model / f"{dataset}_predictions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"model": model, "key": dataset, "clips": 2, "total_clips": 2, "classes": ["angry", "neutral"],
                "ground_truth": "acted_emotion", "metrics": {"accuracy": .01}}
        if protocol is not None:
            meta["inference_protocol"] = protocol
        path.write_text(json.dumps(meta))
        with path.with_suffix(".csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["audio_path", "acted_emotion", "prediction", "correct"])
            writer.writeheader()
            writer.writerows({"audio_path": f"data/{dataset}/{i}.wav", "acted_emotion": label,
                              "prediction": label, "correct": "False"} for i, label in enumerate(["angry", "neutral"]))
        return path

    def test_protocol_filter_and_duplicate_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.artifact(root, "api", {"old": True})
            registry = {"api": {"api": True}}
            with mock.patch.object(consensus, "configured_protocol", return_value={"current": True}):
                selected, excluded = consensus.load_runs(root, registry)
            self.assertEqual(selected, {})
            self.assertEqual(excluded[0]["reason"], "incompatible_inference_protocol")
            with path.with_suffix(".csv").open("a") as stream:
                stream.write("data/crema_d/0.wav,angry,angry,False\n")
            with mock.patch.object(consensus, "configured_protocol", return_value={"old": True}):
                with self.assertRaisesRegex(ValueError, "duplicate audio_path"):
                    consensus.load_runs(root, registry)

    def test_reports_keep_all_datasets_partial_members_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "emotion_recognition"
            paper = Path(directory) / "research_paper"
            paths = [self.artifact(root, m) for m in ("emotion2vec_plus_large", "gemini-3.8-flash")]
            sources = [p for pair in paths for p in (pair, pair.with_suffix(".csv"))]
            before = {p: p.read_bytes() for p in sources}
            (root / "config").mkdir()
            registry = {m: {} for spec in consensus.GROUPS.values() for m in spec["models"]}
            (root / "config/models.yaml").write_text(yaml.safe_dump(registry))
            rows = consensus.run(root, paper)
            self.assertEqual(len(rows), 16 * 3 * 2)
            selected = next(r for r in rows if r["dataset"] == "crema_d" and r["group"] == "Selected3")
            self.assertEqual(selected["model_count"], 2)
            self.assertFalse(selected["complete"])
            self.assertEqual(selected["accepted_correct"], 2)  # stale False correctness flags were not trusted.
            ser = next(r for r in rows if r["dataset"] == "crema_d" and r["group"] == "SER5")
            self.assertIsNone(ser["common_samples"])
            self.assertEqual(ser["status"], "pending")
            self.assertEqual({p.relative_to(paper).as_posix() for p in paper.rglob('*') if p.is_file()},
                             {"analysis_report/consensus_zh.md", "analysis_report/consensus_en.md", "output/consensus.csv",
                              "output/consensus_thresholds.csv", "output/consensus_by_class.csv",
                              "output/consensus_baselines.csv", "output/consensus_manifest.json"})
            with (paper / "output/consensus_thresholds.csv").open() as stream:
                thresholds = list(csv.DictReader(stream))
            self.assertEqual(len(thresholds), 1)
            self.assertEqual((thresholds[0]["min_votes"], thresholds[0]["accepted_errors"]), ("2", "0"))
            manifest = json.loads((paper / "output/consensus_manifest.json").read_text())
            self.assertEqual(manifest["selected_run_count"], 2)
            self.assertEqual(len(manifest["sources"]), 4)
            self.assertFalse(any(c["eligible"] for c in manifest["cohorts"] if c["group"] in consensus.EXTRA_GROUPS))
            for p, content in before.items():
                self.assertEqual(p.read_bytes(), content)
            report = (paper / "analysis_report/consensus_zh.md").read_text()
            self.assertIn("2/3*", report)
            self.assertIn("MultiDialog", report)
            self.assertIn("暂无成员结果", report)

    def test_model_completion_is_distinct_from_consensus_group_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "emotion_recognition"
            paper = Path(directory) / "research_paper"
            for dataset in consensus.DATASETS:
                self.artifact(root, "gemini-3.8-flash", dataset=dataset)
            self.artifact(root, "emotion2vec_plus_large")
            (root / "config").mkdir()
            registry = {m: {} for spec in consensus.GROUPS.values() for m in spec["models"]}
            (root / "config/models.yaml").write_text(yaml.safe_dump(registry))
            rows = consensus.run(root, paper)
            for language in ("zh", "en"):
                report = (paper / "analysis_report" / f"consensus_{language}.md").read_text()
                self.assertIn("Gemini 3.8 Flash 16/16", report)
                self.assertIn("2/3*", report)
                self.assertIn("1/3*", report)
            report = (paper / "analysis_report/consensus_zh.md").read_text()
            self.assertIn("精选模型组 0", report)
            self.assertIn("待补：emotion2vec+, Qwen2.5-Omni", report)
            self.assertIn("待其他模型", report)
            available = [row for row in rows if row["group"] == "Selected3" and row["rule"] == "unanimous"]
            self.assertEqual(len(available), 16)
            self.assertTrue(all("gemini-3.8-flash" in json.loads(row["members"]) for row in available))

    def test_partial_sample_is_not_reported_as_completed_dataset(self):
        sample = consensus.Run("gemini-3.8-flash", "crema_d", Path("unused"), {"total_clips": 10},
                               ["angry", "neutral"], {}, {"a": {}, "b": {}}, [])
        selected = {"gemini-3.8-flash": {"crema_d": sample}}
        zh = consensus.progress_sentence(selected, "zh")
        en = consensus.progress_sentence(selected, "en")
        self.assertIn("Gemini 3.8 Flash 0/16（另有 1 个数据集仅有部分样本）", zh)
        self.assertIn("Gemini 3.8 Flash 0/16 (plus 1 dataset with partial samples)", en)

    def test_no_comparable_models_exports_pending_reports_and_empty_supplements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "emotion_recognition"
            paper = Path(directory) / "research_paper"
            self.artifact(root, "gemini-3.8-flash")
            (root / "config").mkdir()
            registry = {m: {} for spec in consensus.GROUPS.values() for m in spec["models"]}
            (root / "config/models.yaml").write_text(yaml.safe_dump(registry))
            rows = consensus.run(root, paper)
            self.assertTrue(all(row["status"] == "pending" for row in rows))
            for name in ("thresholds", "by_class", "baselines"):
                with (paper / "output" / f"consensus_{name}.csv").open() as stream:
                    reader = csv.DictReader(stream)
                    self.assertTrue(reader.fieldnames)
                    self.assertEqual(list(reader), [])
            manifest = json.loads((paper / "output/consensus_manifest.json").read_text())
            self.assertEqual(manifest["selected_run_count"], 1)
            self.assertTrue(all(c["cohort_sha256"] is None for c in manifest["cohorts"]))

    def test_fixed_four_audio_group_is_complete_without_claiming_five_model_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "emotion_recognition"
            paper = Path(directory) / "research_paper"
            for model in consensus.EXTRA_GROUPS["AudioLLM4"]["models"]:
                self.artifact(root, model)
            (root / "config").mkdir()
            registry = {m: {} for spec in consensus.GROUPS.values() for m in spec["models"]}
            (root / "config/models.yaml").write_text(yaml.safe_dump(registry))
            rows = consensus.run(root, paper)
            partial = next(r for r in rows if r["dataset"] == "crema_d" and r["group"] == "AudioLLM5")
            self.assertEqual(partial["model_count"], 4)
            self.assertFalse(partial["complete"])
            self.assertEqual(json.loads(partial["missing_members"]), ["gemma4_e4b"])
            with (paper / "output/consensus_thresholds.csv").open() as stream:
                fixed = [r for r in csv.DictReader(stream) if r["group"] == "AudioLLM4"]
            self.assertEqual([r["min_votes"] for r in fixed], ["3", "4"])
            self.assertTrue(all(r["complete"] == "True" and r["target_count"] == "4" for r in fixed))
            self.assertTrue(all(r["accepted"] == "2" and r["accepted_correct"] == "2" for r in fixed))

    def test_path_aliases_are_rejected(self):
        for path in ("../a.wav", "/a.wav", "data/../a.wav", "data//a.wav", "C:/a.wav", "data\\a.wav"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                consensus.canonical_path(path)


if __name__ == "__main__":
    unittest.main()

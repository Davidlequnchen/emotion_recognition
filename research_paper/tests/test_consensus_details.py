"""Small, offline checks of threshold and matched-cohort diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from consensus_details import build_details


def job(records, classes=("angry", "neutral", "sad"), accept=None):
    runs = []
    for index in range(len(records[0][1])):
        rows = {f"data/test/{at}.wav": {"gold": gold, "prediction": votes[index]}
                for at, (gold, votes) in enumerate(records)}
        runs.append(SimpleNamespace(model=f"m{index}", rows=rows,
                                    classes=list(classes), accept=accept or {}))
    fields = {"dataset": "test", "group": "fixture", "model_count": len(runs),
              "target_count": len(runs), "complete": True,
              "members": '["fixture"]', "missing_members": "[]", "sources": "{}"}
    return fields, runs, list(runs[0].rows)


class ConsensusDetailsTests(unittest.TestCase):
    def test_all_strict_majority_thresholds_and_fixed_invalid_denominator(self):
        fixture = job([("angry", ["angry"] * 5),
                       ("angry", ["angry"] * 4 + ["neutral"]),
                       ("neutral", ["angry"] * 3 + [None] * 2),
                       ("angry", ["angry"] * 2 + [None] * 3)])
        rows, _, _ = build_details([fixture])
        self.assertEqual([r["min_votes"] for r in rows], [3, 4, 5])
        self.assertEqual([r["accepted"] for r in rows], [3, 2, 1])
        self.assertEqual([r["accepted_correct"] for r in rows], [2, 2, 1])
        self.assertEqual(rows[0]["accepted_errors"], 1)
        self.assertEqual(rows[0]["coverage"], 3 / 4)
        self.assertEqual(rows[0]["conditional_precision"], 2 / 3)
        self.assertEqual(rows[0]["selective_risk"], 1 / 3)
        self.assertEqual(rows[0]["accepted_correct_fraction"], 1 / 2)

    def test_composite_gold_and_absent_predicted_class(self):
        fixture = job([("excited", ["happy", "surprised", "happy"])],
                      classes=("happy", "surprised", "sad"),
                      accept={"excited": ["happy", "surprised"]})
        rows, classes, baselines = build_details([fixture])
        self.assertEqual([r["accepted_correct"] for r in rows], [1, 0])
        self.assertTrue(all(r["correct"] == 1 for r in baselines))
        self.assertIsNone(rows[1]["conditional_precision"])
        self.assertIsNone(rows[1]["selective_risk"])
        majority = {r["predicted_class"]: r for r in classes if r["min_votes"] == 2}
        self.assertEqual(set(majority), {"happy", "surprised", "sad"})
        self.assertEqual(majority["happy"]["share_of_accepted"], 1)
        self.assertEqual(majority["sad"]["accepted"], 0)
        self.assertEqual(majority["sad"]["coverage"], 0)
        self.assertEqual(majority["sad"]["share_of_accepted"], 0)
        self.assertIsNone(majority["sad"]["conditional_precision"])
        self.assertTrue(all(r["share_of_accepted"] is None for r in classes if r["min_votes"] == 3))

    def test_ties_and_all_invalid_are_rejected(self):
        fixture = job([("angry", ["angry", "neutral"]),
                       ("angry", [None, None]),
                       ("angry", ["angry", None])])
        rows, _, _ = build_details([fixture])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["min_votes"], 2)
        self.assertEqual(rows[0]["accepted"], 0)
        self.assertIsNone(rows[0]["conditional_precision"])

    def test_baselines_use_common_cohort_and_union_denominator_is_distinct(self):
        fields, runs, common = job([("angry", ["angry", "angry"]),
                                    ("neutral", ["angry", None])])
        runs[0].rows["data/test/outside.wav"] = {"gold": "neutral", "prediction": "neutral"}
        rows, classes, baselines = build_details([(fields, runs, common)])
        self.assertEqual(rows[0]["common_samples"], 2)
        self.assertEqual(rows[0]["union_samples"], 3)
        self.assertEqual(rows[0]["coverage"], 1 / 2)
        self.assertEqual(rows[0]["accepted_fraction_of_union"], 1 / 3)
        self.assertEqual([(r["correct"], r["accuracy"], r["invalid_predictions"])
                          for r in baselines], [(1, 1 / 2, 0), (1, 1 / 2, 1)])
        self.assertEqual(len({r["cohort_sha256"] for r in rows + classes + baselines}), 1)
        reordered, _, _ = build_details([(fields, runs, common[::-1])])
        self.assertEqual(rows, reordered)
        runs[0].rows[common[0]]["gold"] = "neutral"
        changed, _, _ = build_details([(fields, runs, common)])
        self.assertNotEqual(rows[0]["cohort_sha256"], changed[0]["cohort_sha256"])


if __name__ == "__main__":
    unittest.main()

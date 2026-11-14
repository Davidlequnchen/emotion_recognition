"""Small, explicit scoring oracles for saved-prediction domain analysis."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cross_dataset_analysis import balanced_weights, cluster_identity, constant_baselines, score_run


def _run(gold, predictions, *, metrics=None):
    return SimpleNamespace(
        model="synthetic", dataset="jl_like",
        classes=["angry", "happy", "neutral", "sad", "surprised"],
        accept={"excited": ["happy", "surprised"]},
        metadata={"labels": ["angry", "happy", "neutral", "sad", "excited"],
                  "metrics": metrics or {}},
        rows={f"clip{i}": {"gold": g, "prediction": p}
              for i, (g, p) in enumerate(zip(gold, predictions))},
    )


class CrossDatasetAnalysisTests(unittest.TestCase):
    def test_composite_and_invalid_metrics_match_hand_resolved_truth(self):
        run = _run(["excited", "excited", "excited", "angry", "neutral", "sad", "happy"],
                   ["surprised", "happy", "sad", None, "neutral", "angry", "happy"])
        metrics, gold, correct, invalid, predictions = score_run(run, list(run.rows))
        np.testing.assert_array_equal(correct, [1, 1, 0, 0, 1, 0, 1])
        np.testing.assert_array_equal(invalid, [False, False, False, True, False, False, False])
        self.assertEqual(predictions[3], None)
        self.assertEqual(list(gold[:3]), ["excited"] * 3)
        self.assertAlmostEqual(metrics["accuracy"], 4 / 7)
        # UAR uses five original gold rows: 0, 1, 1, 0, and 2/3 recall.
        self.assertAlmostEqual(metrics["uar"], (0 + 1 + 1 + 0 + 2 / 3) / 5)
        # Accepted excited predictions resolve to the predicted class; a wrong
        # excited prediction resolves to happy. F1 = [0, .8, 1, 0, 1].
        self.assertAlmostEqual(metrics["macro_f1"], 2.8 / 5)
        self.assertAlmostEqual(metrics["weighted_f1"], 4.4 / 7)

    def test_saved_metric_mismatch_is_rejected_for_each_metric(self):
        run = _run(["happy", "sad", "neutral"], ["happy", None, "sad"])
        computed = score_run(run, list(run.rows))[0]
        mapping = {"accuracy": "accuracy", "uar": "unweighted_accuracy",
                   "macro_f1": "macro_f1", "weighted_f1": "weighted_f1"}
        run.metadata["metrics"] = {saved: computed[key] for key, saved in mapping.items()}
        score_run(run, list(run.rows))  # Exact saved metrics are accepted.
        for key, saved in mapping.items():
            with self.subTest(metric=key):
                run.metadata["metrics"][saved] += .02
                with self.assertRaisesRegex(ValueError, "recomputed " + key):
                    score_run(run, list(run.rows))
                run.metadata["metrics"][saved] -= .02

    def test_constant_baseline_counts_accepted_labels_not_raw_gold_frequency(self):
        run = _run(["excited", "excited", "happy", "neutral", "sad"], [None] * 5)
        baseline = constant_baselines(run, list(run.rows))
        self.assertEqual(baseline["best_constant_label"], "happy")
        self.assertAlmostEqual(baseline["best_constant_accuracy"], 3 / 5)
        self.assertAlmostEqual(baseline["neutral_accuracy"], 1 / 5)
        self.assertNotEqual(baseline["best_constant_accuracy"], 1 / 5)  # Raw happy prevalence.

    def test_weights_balance_dataset_and_native_gold_without_prediction_groups(self):
        dataset = np.array(["large"] * 6 + ["small"] * 4)
        gold = np.array(["major"] * 5 + ["minor"] + ["a", "b", "b", "c"])
        weights = balanced_weights(dataset, gold)
        for key in ("large", "small"):
            use = dataset == key
            self.assertAlmostEqual(weights[use].sum(), 1)
            labels = set(gold[use])
            for label in labels:
                self.assertAlmostEqual(weights[use & (gold == label)].sum(), 1 / len(labels))
        # Source clustering does not split the same EmotionTalk source by
        # dialogue, gold emotion, or a model's successful/failed prediction.
        a = {"dataset": "emotiontalk", "source_group_id": "pair1", "group": "dialogue1"}
        b = {"dataset": "emotiontalk", "source_group_id": "pair1", "group": "dialogue2"}
        self.assertEqual(cluster_identity(a), cluster_identity(b))
        self.assertEqual(cluster_identity({"dataset": "other", "group": "other:speaker:x"}),
                         "other:speaker:x")


if __name__ == "__main__":
    unittest.main()

"""Synthetic checks for native-scale intensity estimands, not real-model runs."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import intensity_analysis as analysis


def _spec(kind):
    return {"kind": kind, "scale": "native-test-scale", "source": "published synthetic metadata",
            "url": "https://example.invalid/native", "notes": "Known native values only.",
            "levels": [1, 2, 3], "pair_rule": "Same actor, emotion and text key."}


def _fixture(records):
    """records: (gold, level, prediction, case-key, group, available, applicable)."""
    inventory, strengths, predictions = [], [], {}
    for i, (gold, level, prediction, case, group, available, applicable) in enumerate(records):
        path = f"clip{i}"
        inventory.append({"dataset": "test", "audio_path": path, "gold": gold,
                          "duration_seconds": float(2 ** (i % 5)), "group": group})
        strengths.append({"dataset": "test", "audio_path": path, "gold": gold,
                          "intensity_available": available, "intensity_applicable": applicable,
                          "intensity_level": level, "intensity_name": f"level {level}",
                          "intensity_pair_key": case})
        predictions[path] = {"gold": gold, "prediction": prediction}
    selected = {"model": {"test": SimpleNamespace(rows=predictions, accept={})}}
    return selected, inventory, strengths


def _fake_fit(predictor, correct, strata, clusters, weights=None, covariates=None):
    return {"n": len(predictor), "nclusters": len(set(clusters)), "slope_pp": 1.,
            "ci_lower_pp": None, "ci_upper_pp": None, "stderr_pp": None,
            "pvalue": None, "note": "Synthetic regression capture, no inference."}


class IntensityAnalysisTests(unittest.TestCase):
    def test_paired_difference_class_balance_cluster_sandwich_and_pp_units(self):
        values = np.array([1.] * 30 + [0.] * 10 + [-1.] * 4 + [0.] * 4)
        gold = np.array(["common"] * 40 + ["rare"] * 8)
        groups = np.array([f"actor{i // 2}" for i in range(48)])
        result = analysis.paired_difference(values, gold, groups)
        # Class means are .75 and -.5; a clip-weighted mean would be 26/48.
        estimate = (.75 - .5) / 2
        self.assertAlmostEqual(result["difference_pp"], 100 * estimate)
        self.assertNotAlmostEqual(result["difference_pp"], 100 * values.mean())
        self.assertEqual((result["n_pairs"], result["nclusters"]), (48, 24))
        weights = np.array([1 / 80] * 40 + [1 / 16] * 8)
        scores = [np.sum(weights[groups == group] * (values[groups == group] - estimate))
                  for group in set(groups)]
        variance = 24 / 23 * np.sum(np.square(scores)) / weights.sum() ** 2
        margin_pp = 100 * t.ppf(.975, 23) * np.sqrt(variance)
        self.assertAlmostEqual(result["ci_lower_pp"], 100 * estimate - margin_pp)
        self.assertAlmostEqual(result["ci_upper_pp"], 100 * estimate + margin_pp)
        order = np.arange(len(values))[::-1]
        reordered = analysis.paired_difference(values[order], gold[order], ["new-" + g for g in groups[order]])
        for key in ("difference_pp", "ci_lower_pp", "ci_upper_pp"):
            self.assertAlmostEqual(result[key], reordered[key])
        small = analysis.paired_difference([1, 0, -1], ["a", "a", "b"], ["p", "p", "q"])
        self.assertEqual(small["nclusters"], 2)
        self.assertIsNone(small["ci_lower_pp"])

    def test_continuous_regression_keeps_native_values_and_excludes_unknown_neutral(self):
        records = [
            ("sad", 1.25, "sad", "", "a", True, True),
            ("sad", 2.5, "happy", "", "b", True, True),
            ("happy", 4.75, "happy", "", "c", True, True),
            ("sad", 7.125, None, "", "d", True, True),
            ("neutral", 9., "neutral", "", "e", True, False),
            ("sad", None, "sad", "", "f", False, False),
        ]
        selected, inventory, strengths = _fixture(records)
        with patch.object(analysis, "SCALES", {"test": _spec("perceived_continuous_mean")}), \
                patch.object(analysis, "fixed_effect_linear", side_effect=_fake_fit) as fit:
            result = analysis.analyse(selected, inventory, strengths)
        scope = result["scope_rows"][0]
        self.assertEqual((scope["n_total"], scope["n_known"], scope["n_non_neutral_known"]), (6, 5, 4))
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            np.testing.assert_array_equal(call.args[0], [1.25, 2.5, 4.75, 7.125])
            np.testing.assert_array_equal(call.args[1], [1, 0, 1, 0])  # Invalid stays wrong.
            np.testing.assert_array_equal(call.args[2], ["sad", "sad", "happy", "sad"])
            np.testing.assert_allclose(call.args[4], [1 / 6, 1 / 6, 1 / 2, 1 / 6])
        self.assertIsNone(fit.call_args_list[0].kwargs["covariates"])
        np.testing.assert_array_equal(fit.call_args_list[1].kwargs["covariates"], [0, 1, 2, 3])
        self.assertEqual([row["level_label"] for row in result["level_rows"]], ["Q1", "Q2", "Q3", "Q4"])
        self.assertEqual(sum(row["n"] for row in result["level_rows"]), 4)
        self.assertEqual(sum(row["invalid"] for row in result["level_rows"]), 1)
        self.assertEqual(result["matched_rows"], [])
        self.assertTrue(all(row["unit"] == "per listener-rating point" for row in result["effect_rows"]))

    def test_matched_keys_require_all_native_levels_and_average_repeated_takes(self):
        records = [
            ("sad", 1, "happy", "complete-a", "actor-a", True, True),
            ("sad", 1, "sad", "complete-a", "actor-a", True, True),
            ("sad", 2, "sad", "complete-a", "actor-a", True, True),
            ("sad", 3, "sad", "complete-a", "actor-a", True, True),
            ("sad", 3, "sad", "complete-a", "actor-a", True, True),
            ("sad", 1, "happy", "missing-middle", "actor-c", True, True),
            ("sad", 3, "sad", "missing-middle", "actor-c", True, True),
            ("happy", 1, "happy", "complete-b", "actor-b", True, True),
            ("happy", 2, "happy", "complete-b", "actor-b", True, True),
            ("happy", 3, "sad", "complete-b", "actor-b", True, True),
        ]
        selected, inventory, strengths = _fixture(records)
        with patch.object(analysis, "SCALES", {"test": _spec("acted_ordinal")}), \
                patch.object(analysis, "fixed_effect_linear", side_effect=_fake_fit):
            result = analysis.analyse(selected, inventory, strengths)
        matched = result["matched_rows"][0]
        self.assertEqual((matched["n_pairs"], matched["nclusters"]), (2, 2))
        # One sad-key difference of 1-.5 and one happy-key difference of 0-1.
        self.assertAlmostEqual(matched["difference_pp"], -25.)
        self.assertEqual((matched["low_level"], matched["high_level"]), (1, 3))
        self.assertIsNone(matched["ci_lower_pp"])

    def test_matched_key_cannot_cross_gold_class_or_source_cluster(self):
        for middle_gold, middle_group in (("happy", "actor"), ("sad", "different-actor")):
            with self.subTest(gold=middle_gold, group=middle_group):
                records = [("sad", 1, "sad", "bad", "actor", True, True),
                           (middle_gold, 2, middle_gold, "bad", middle_group, True, True),
                           ("sad", 3, "sad", "bad", "actor", True, True)]
                selected, inventory, strengths = _fixture(records)
                with patch.object(analysis, "SCALES", {"test": _spec("acted_ordinal")}), \
                        patch.object(analysis, "fixed_effect_linear", side_effect=_fake_fit), \
                        self.assertRaisesRegex(ValueError, "spans emotions or source clusters"):
                    analysis.analyse(selected, inventory, strengths)


if __name__ == "__main__":
    unittest.main()

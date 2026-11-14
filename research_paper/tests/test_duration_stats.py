"""Known-design checks for the duration association and cluster uncertainty."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from duration_stats import fixed_effect_linear, fixed_effect_slope


class DurationStatsTests(unittest.TestCase):
    def test_weighted_demeaning_matches_full_dummy_wls_and_cluster_sandwich(self):
        duration = np.array([1, 2, 3, 5, 2, 4, 6, 8, 1, 3, 4, 7], dtype=float)
        y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
        strata = np.repeat(np.arange(3), 4)
        clusters = np.tile(np.arange(4), 3)
        w = np.array([1, 2, 1, 3, 2, 1, 4, 1, 1, 3, 2, 2], dtype=float)
        design = np.column_stack([np.eye(3)[strata], np.log2(duration)])
        coefficient = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        residual = y - design @ coefficient
        scores = np.array([design[clusters == g].T @ (w[clusters == g] * residual[clusters == g])
                           for g in range(4)])
        inverse = np.linalg.inv(design.T @ (w[:, None] * design))
        covariance = 4 / 3 * 11 / 8 * inverse @ (scores.T @ scores) @ inverse
        se = np.sqrt(covariance[-1, -1])
        result = fixed_effect_slope(duration, y, strata, clusters, w)
        self.assertEqual(result["status"], "ok")
        self.assertEqual((result["n"], result["nstrata"], result["nclusters"], result["effective_df"]),
                         (12, 3, 4, 3))
        self.assertAlmostEqual(result["slope"], coefficient[-1], places=12)
        self.assertAlmostEqual(result["stderr_pp"], 100 * se, places=12)
        self.assertAlmostEqual(result["ci_lower_pp"], 100 * (coefficient[-1] - t.ppf(.975, 3) * se), places=12)

    def test_simpson_reversal_is_removed_by_stratum_intercepts(self):
        duration = np.array([1, 2, 8, 16.])
        y = np.array([.8, .9, .1, .2])
        pooled = fixed_effect_slope(duration, y, ["pooled"] * 4, range(4))
        adjusted = fixed_effect_slope(duration, y, ["a", "a", "b", "b"], range(4))
        self.assertLess(pooled["slope"], 0)
        self.assertAlmostEqual(adjusted["slope"], .1)
        self.assertAlmostEqual(adjusted["slope_pp"], 10)

    def test_doubling_units_order_identity_labels_and_weight_scale(self):
        duration = np.array([1, 2, 4, 8, 3, 6, 12, 24.])
        y = np.array([0, .1, .2, .3, .3, .4, .5, .6])
        strata = np.array(["a"] * 4 + ["b"] * 4)
        clusters = np.array(["p", "q", "r", "s"] * 2)
        weights = np.array([1, 2, 3, 4, 2, 1, 2, 1.])
        result = fixed_effect_slope(duration, y, strata, clusters, weights)
        order = np.array([5, 1, 7, 3, 0, 2, 6, 4])
        remapped = fixed_effect_slope(duration[order] * 1000, y[order], strata[order],
                                     np.array(["renamed-" + c for c in clusters[order]]), weights[order] * 7)
        self.assertAlmostEqual(result["slope_pp"], 10)
        for key in ("slope_pp", "stderr_pp", "ci_lower_pp", "ci_upper_pp"):
            self.assertAlmostEqual(result[key], remapped[key], places=10)

    def test_empty_constant_and_uncertainty_unavailable(self):
        self.assertEqual(fixed_effect_slope([], [], [], [])["status"], "not_identified")
        result = fixed_effect_slope([1, 1, 8, 8], [0, 1, 0, 1], ["a", "a", "b", "b"], range(4))
        self.assertEqual(result["status"], "not_identified")
        self.assertIsNone(result["slope_pp"])
        for strata, clusters in [(["a"] * 3, ["one"] * 3), (["a", "a", "b"], range(3))]:
            result = fixed_effect_slope([1, 2, 4], [0, 1, 1], strata, clusters)
            self.assertEqual(result["status"], "variance_unavailable")
            self.assertIsNotNone(result["slope"])
            self.assertIsNone(result["ci_lower_pp"])

    def test_invalid_inputs_are_rejected_instead_of_filtered(self):
        for duration, y, weights in [([0, 1], [0, 1], None), ([1, np.nan], [0, 1], None),
                                     ([1, 2], [-.1, 1], None), ([1, 2], [0, np.inf], None),
                                     ([1, 2], [0, 1], [0, 1]), ([1, 2], [0, 1], [1, np.nan])]:
            with self.subTest(duration=duration, y=y, weights=weights), self.assertRaises(ValueError):
                fixed_effect_slope(duration, y, ["a", "a"], ["p", "q"], weights)
        with self.assertRaises(ValueError):
            fixed_effect_slope([1, 2], [0, 1], ["a"], ["p", "q"])
        with self.assertRaises(ValueError):
            fixed_effect_slope([1, 2], [0, 1], ["a", None], ["p", "q"])

    def test_covariate_adjustment_matches_full_dummy_wls_and_cluster_sandwich(self):
        predictor = np.array([-2, 0, 1, 3, -1, 2, 0, 1, 3, 2, -2, 0.])
        covariate = np.log2([1, 2, 3, 5, 2, 4, 6, 8, 1, 3, 4, 7])
        y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1.])
        strata = np.repeat(np.arange(3), 4)
        clusters = np.tile(np.arange(4), 3)
        w = np.array([1, 2, 1, 3, 2, 1, 4, 1, 1, 3, 2, 2.])
        design = np.column_stack([np.eye(3)[strata], predictor, covariate])
        coefficient = np.linalg.lstsq(design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
        residual = y - design @ coefficient
        scores = np.array([design[clusters == g].T @ (w[clusters == g] * residual[clusters == g])
                           for g in range(4)])
        inverse = np.linalg.inv(design.T @ (w[:, None] * design))
        covariance = 4 / 3 * 11 / 7 * inverse @ (scores.T @ scores) @ inverse
        se = np.sqrt(covariance[-2, -2])
        result = fixed_effect_linear(predictor, y, strata, clusters, w, covariates=covariate)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["residual_df"], 7)
        self.assertAlmostEqual(result["slope"], coefficient[-2], places=12)
        self.assertAlmostEqual(result["stderr_pp"], 100 * se, places=12)
        self.assertAlmostEqual(result["ci_upper_pp"], 100 * (coefficient[-2] + t.ppf(.975, 3) * se), places=12)
        matrix_result = fixed_effect_linear(predictor, y, strata, clusters, w * 11,
                                            covariates=covariate[:, None] * 1000)
        for key in ("slope", "slope_pp", "stderr_pp", "ci_lower_pp", "ci_upper_pp", "pvalue"):
            self.assertAlmostEqual(result[key], matrix_result[key], places=10)
        self.assertIn("per unit of supplied predictor", result["note"])
        self.assertNotIn("duration", result["note"])

    def test_covariate_collinearity_and_invalid_inputs(self):
        predictor = np.arange(12, dtype=float)
        strata = np.repeat(np.arange(3), 4)
        clusters = np.tile(np.arange(4), 3)
        y = np.tile([0, 1, 1, 0.], 3)
        for covariate in (predictor * 2, predictor * 2 + 1e-12 * np.sin(predictor),
                          np.ones(len(predictor))):
            with self.subTest(covariate=covariate):
                result = fixed_effect_linear(predictor, y, strata, clusters, covariates=covariate)
                self.assertEqual(result["status"], "not_identified")
                self.assertIsNone(result["slope"])
                self.assertIsNone(result["ci_lower_pp"])
        for covariate in ([1, 2], np.ones((12, 1, 1)), np.full(12, np.nan), 1):
            with self.subTest(covariate=covariate), self.assertRaises(ValueError):
                fixed_effect_linear(predictor, y, strata, clusters, covariates=covariate)
        with self.assertRaises(ValueError):
            fixed_effect_linear([np.inf, 0], [0, 1], ["a", "a"], ["p", "q"])


if __name__ == "__main__":
    unittest.main()

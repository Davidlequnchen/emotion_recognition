"""Checks for the intensity figure inputs: benchmark model style and weakest/strongest contrasts."""
import ast
from pathlib import Path
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research_paper" / "scripts"))
import intensity_analysis as analysis
import intensity_plot as plot


class BenchmarkStyleTest(unittest.TestCase):
    def test_completed_gemma_has_no_preliminary_suffix(self):
        self.assertEqual(plot._label("gemma4_e4b", {"gemma4_e4b"}), "Gemma 4 E4B")

    def test_models_follow_benchmark_order_names_and_colours(self):
        registry = yaml.safe_load((ROOT / "emotion_recognition/config/models.yaml").read_text())
        source = (ROOT / "emotion_recognition/scripts/viewer/plot_results.py").read_text()
        colours = next(ast.literal_eval(node.value) for node in ast.parse(source).body
                       if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "COLOURS")
        keys = [key for key in registry if key in plot.MODELS]
        self.assertEqual(keys, list(plot.MODELS))
        for i, key in enumerate(registry):
            if key in plot.MODELS:
                self.assertEqual(plot.MODELS[key], (registry[key]["name"], colours[i % len(colours)]))


def _level(dataset, model, level_id, level, n, accuracy, neutral=0.0):
    return {"dataset": dataset, "model": model, "level_id": level_id, "level": level,
            "level_label": f"L{level_id}", "n": n, "accuracy": accuracy, "predicted_neutral_rate": neutral}


class ContrastRowsTest(unittest.TestCase):
    def test_small_top_level_is_skipped_and_slope_scaled_by_native_distance(self):
        levels = [_level("d", "m", 0, 1., 50, .2, .6), _level("d", "m", 1, 2., 60, .5, .3),
                  _level("d", "m", 2, 3., 12, 1., 0.)]
        effects = [{"dataset": "d", "model": "m", "fit": "class_balanced", "slope_pp": 99.,
                    "ci_lower_pp": 1., "ci_upper_pp": 2., "nclusters": 30},
                   {"dataset": "d", "model": "m", "fit": "class_balanced_duration_adjusted", "slope_pp": 10.,
                    "ci_lower_pp": -2., "ci_upper_pp": 22., "nclusters": 30}]
        row, = analysis.contrast_rows(levels, effects, [])
        self.assertEqual((row["low_label"], row["high_label"], row["level_span"]), ("L0", "L1", 1.))
        self.assertAlmostEqual(row["observed_change_pp"], 30.)
        self.assertEqual((row["low_neutral_rate"], row["high_neutral_rate"]), (.6, .3))
        self.assertEqual((row["adjusted_change_pp"], row["adjusted_ci_lower_pp"], row["adjusted_ci_upper_pp"]),
                         (10., -2., 22.))
        self.assertIsNone(row["matched_difference_pp"])

    def test_continuous_span_uses_quartile_medians_and_keeps_withheld_interval(self):
        levels = [_level("c", "m", i, level, 100, acc) for i, (level, acc) in
                  enumerate(((3.5, .1), (4.0, .2), (4.6, .3), (5.5, .4)))]
        effects = [{"dataset": "c", "model": "m", "fit": "class_balanced_duration_adjusted", "slope_pp": 5.,
                    "ci_lower_pp": None, "ci_upper_pp": None, "nclusters": 6}]
        matched = [{"dataset": "c", "model": "m", "n_pairs": 3, "difference_pp": 4., "ci_lower_pp": None,
                    "ci_upper_pp": None, "nclusters": 2}]
        row, = analysis.contrast_rows(levels, effects, matched)
        self.assertAlmostEqual(row["level_span"], 2.)
        self.assertAlmostEqual(row["adjusted_change_pp"], 10.)
        self.assertIsNone(row["adjusted_ci_lower_pp"])
        self.assertEqual((row["matched_pairs"], row["matched_difference_pp"]), (3, 4.))

    def test_single_displayed_level_gives_no_contrast(self):
        levels = [_level("d", "m", 0, 1., 50, .2), _level("d", "m", 1, 2., 5, .9)]
        self.assertEqual(analysis.contrast_rows(levels, [], []), [])


if __name__ == "__main__":
    unittest.main()

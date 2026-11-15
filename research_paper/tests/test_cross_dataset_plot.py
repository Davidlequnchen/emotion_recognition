"""Smoke and size checks for the cross-dataset figures and the report's splitting checks."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cross_dataset_plot import ANALYSIS_TO_DISPLAY, BINS, plot_analysis  # noqa: E402
from cross_dataset_report import group_checks  # noqa: E402

SLIDE_WIDTH, SLIDE_HEIGHT = 12.13, 5.3          # content box of the deck (inches); figures must fit unscaled


def _png_inches(path):
    with open(path, "rb") as stream:
        header = stream.read(24)
    width, height = struct.unpack(">II", header[16:24])
    return width / 300, height / 300


def _synthetic():
    datasets = [{"dataset": "meld", "name": "MELD", "language": "English", "conversation": True, "median_seconds": 2.5},
                {"dataset": "thai_ser", "name": "THAI-SER", "language": "Thai", "conversation": False, "median_seconds": 4.5}]
    inventory = [{"dataset": d["dataset"], "duration_seconds": s} for d in datasets for s in (0.5, 1.5, 2.5, 4, 7, 12, 31)]
    models = ("emotion2vec_plus_large", "gpt_realtime")
    groups = [{"model": m, "grouping": g, "level": level, "accuracy": .5 + i / 10, "uar": .4, "macro_f1": .3}
              for m in models for g, levels in (("context", ("conversational", "non_conversational")),
                                                 ("language", ("English", "non_English")))
              for i, level in enumerate(levels)]
    effects = [{"model": m, "fit": f, "cluster_kind": "dataset", "slope_pp": 2, "ci_lower_pp": -1, "ci_upper_pp": 5}
               for m in models for f in ("pooled", "balanced_within_dataset_gold")]
    dataset_effects = [{"dataset": d["dataset"], "model": m, "slope_pp": 30, "ci_lower_pp": 1, "ci_upper_pp": 60}
                       for d in datasets for m in models]
    bins = [{"dataset": d["dataset"], "model": m, "bin_id": b, "n": 60, "correct": 30}
            for d in datasets for m in models for b in range(8)]
    return inventory, datasets, groups, bins, effects, dataset_effects


class CrossDatasetPlotTests(unittest.TestCase):
    def test_every_figure_is_written_and_fits_the_slide_unscaled(self):
        self.assertEqual(len(ANALYSIS_TO_DISPLAY), 8)  # every analysis bin maps to a display bin
        self.assertEqual(max(ANALYSIS_TO_DISPLAY), len(BINS) - 1)
        with tempfile.TemporaryDirectory() as folder:
            paths = plot_analysis(*_synthetic(), folder)
            names = {Path(p).name for p in paths}
            for stem in ("duration_distributions", "performance_by_setting", "performance_by_language",
                         "duration_effects", "duration_within_dataset_slopes", "accuracy_by_duration"):
                self.assertIn(f"{stem}.png", names)
                self.assertIn(f"{stem}.svg", names)
                width, height = _png_inches(Path(folder) / f"{stem}.png")
                self.assertLessEqual(width, SLIDE_WIDTH, stem)
                self.assertLessEqual(height, SLIDE_HEIGHT, stem)


class GroupCheckTests(unittest.TestCase):
    def test_language_gap_within_acted_and_setting_gap_without_exposure(self):
        datasets = {"meld": {"language": "English"}, "crema_d": {"language": "English"},
                    "nemo": {"language": "Polish"}}
        rows = [{"model": "m", "dataset": "meld", "accuracy": .40, "uar": .30, "training_status": "not_established"},
                {"model": "m", "dataset": "crema_d", "accuracy": .90, "uar": .80, "training_status": "probable_exposure"},
                {"model": "m", "dataset": "nemo", "accuracy": .50, "uar": .60, "training_status": "not_established"}]
        check = group_checks(rows, datasets, "m")
        self.assertEqual(check["language_acted"], "+40.0 / +20.0")      # CREMA-D − nEMO; MELD is conversational
        self.assertEqual(check["setting_all"], "-30.0 / -40.0")          # MELD − mean(CREMA-D, nEMO)
        self.assertEqual(check["setting_clean"], "-10.0 / -30.0")        # MELD − nEMO (CREMA-D exposed)
        self.assertEqual(check["kept"], "1 + 1")
        self.assertEqual(group_checks(rows[:2], datasets, "m"), {})    # incomplete coverage gives no check


if __name__ == "__main__":
    unittest.main()

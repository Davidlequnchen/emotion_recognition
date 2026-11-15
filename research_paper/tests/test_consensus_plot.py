"""Offline checks of the consensus figures on synthetic threshold rows; no models, audio or saved predictions."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import consensus_plot


def row(dataset, group, k, m, coverage, accuracy, complete=True):
    accepted = round(100 * coverage)
    return {"dataset": dataset, "group": group, "min_votes": str(k), "model_count": str(m), "complete": str(complete),
            "coverage": str(coverage), "conditional_precision": "" if accuracy is None else str(accuracy),
            "accepted": str(accepted), "accepted_correct": str(round(accepted * (accuracy or 0))),
            "accepted_errors": "0", "common_samples": "100"}


ROWS = [row("meld", "SER5", 3, 5, .7, .44), row("meld", "SER5", 4, 5, .34, .48), row("meld", "SER5", 5, 5, .11, .54),
        row("meld", "LLM3", 2, 3, .79, .56), row("meld", "LLM3", 3, 3, .21, .67),
        row("crema_d", "LLM3", 2, 2, .22, .94, complete=False),
        row("crema_d", "All10", 9, 9, 0, None, complete=False),
        row("crema_d", "AudioLLM4", 3, 4, .55, .34), row("crema_d", "AudioLLM4", 4, 4, .13, .94)]
CLASS_ROWS = [{"dataset": "meld", "group": "SER5", "min_votes": "5", "model_count": "5", "complete": "True",
               "predicted_class": "neutral", "share_of_accepted": ".06"}]


class ConsensusPlotTests(unittest.TestCase):
    def test_records_keep_partial_groups_and_reject_minority_thresholds(self):
        records = consensus_plot._records(ROWS)
        self.assertFalse(records["crema_d", "LLM3"][0]["complete"])
        self.assertIsNone(records["crema_d", "All10"][0]["conditional_precision"])
        self.assertEqual([r["min_votes"] for r in records["meld", "SER5"]], [3, 4, 5])
        with self.assertRaisesRegex(ValueError, "strict-majority"):
            consensus_plot._records([row("meld", "SER5", 2, 5, .9, .4)])
        self.assertEqual(consensus_plot.vote_label(records["meld", "SER5"][0]), "≥3/5")
        self.assertEqual(consensus_plot.vote_label(records["meld", "SER5"][2]), "5/5")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is not installed")
    def test_paper_and_slide_figures_are_written(self):
        pending = {("LLM3", "crema_d"): ["Gemma 4"], ("All10", "crema_d"): ["Gemma 4"]}
        with tempfile.TemporaryDirectory() as directory:
            paper = consensus_plot.plot_thresholds(ROWS, Path(directory) / "paper", pending=pending)
            self.assertEqual(sorted(p.name for p in paper), sorted(
                f"{name}.{suffix}" for suffix in ("png", "svg") for name in
                ("consensus_tradeoff_by_dataset", "consensus_meld_tradeoff", "consensus_unanimous_by_dataset",
                 "consensus_tradeoff_extra_groups")))
            slides = consensus_plot.plot_slides(ROWS, CLASS_ROWS, Path(directory) / "slides", pending=pending)
            self.assertEqual(len(slides), 8)
            self.assertTrue(all(p.stat().st_size > 10_000 for p in paper + slides))
            svg = (Path(directory) / "paper/consensus_tradeoff_by_dataset.svg").read_text()
            self.assertIn("Gemma 4 pending", svg)          # the legend explains hollow/dashed marks
            self.assertIn("Coverage (% of clips labelled)", svg)
            heat = (Path(directory) / "paper/consensus_unanimous_by_dataset.svg").read_text()
            self.assertIn("94*", heat)                     # partial value marked
            self.assertIn("—", heat)                       # nothing accepted: undefined, not 0%


if __name__ == "__main__":
    unittest.main()

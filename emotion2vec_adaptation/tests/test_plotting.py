"""Validate publication exports, scientific labels and the shared plotting style."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.plotting import (
    PLOT_CONFIG, load_plot_settings, prepare_plot_cache, render_training_figures, validate_plot_settings,
)


def series_from(values):
    def series(*names):
        value = next((values[name] for name in names if name in values), None)
        return ([], []) if value is None else ([1, 2, 3, 4], value)
    return series


class PlottingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name).resolve()
        cls.settings = load_plot_settings()
        prepare_plot_cache()
        import matplotlib

        before = matplotlib.rcParams.copy()
        cls.head = render_training_figures("head", cls.root / "head", series_from({
            "train_loss": [1.1, 0.8, 0.6, 0.5], "validation_loss": [1.2, 0.9, 0.8, 0.7],
            "train_accuracy": [0.4, 0.6, 0.8, 1.0], "validation_accuracy": [0.3, 0.5, 0.7, 0.8],
            "train_macro_f1": [0.35, 0.55, 0.75, 0.95], "validation_macro_f1": [0.25, 0.45, 0.65, 0.75],
        }))
        cls.ssl = render_training_figures("pretrain", cls.root / "ssl", series_from({
            "loss": [0.8, 0.65, 0.7, 0.5], "loss_ema": [0.8, 0.785, 0.7765, 0.74885],
            "validation_loss": [0.9, 0.85, 0.8, 0.75], "target_std": [0.9, 0.91, 0.92, 0.93],
            "prediction_std": [0.1, 0.2, 0.3, 0.4], "gradient_norm": [2.0, 1.7, 1.6, 1.5],
            "learning_rate": [1e-5, 1e-5, 1e-5, 1e-5],
        }))
        cls.rc_unchanged = before == matplotlib.rcParams.copy()

    def test_benchmark_typography_color_palette_and_export_defaults(self):
        self.assertEqual(self.settings["exports"], {"dpi": 300, "png": True, "svg": True, "pdf": False})
        for name in ("tick_size", "label_size", "title_size", "legend_size", "overview_title_size"):
            self.assertGreaterEqual(self.settings["typography"][name], 16)
        self.assertEqual(self.settings["colors"]["training"], "#0072B2")
        self.assertEqual(self.settings["colors"]["validation"], "#D55E00")
        self.assertFalse(self.settings["axes"]["top_spine"])
        self.assertFalse(self.settings["axes"]["right_spine"])
        self.assertEqual(self.settings["axes"]["grid_axis"], "y")
        self.assertTrue(self.rc_unchanged)

    def test_standalone_and_overview_pngs_have_physical_dpi_and_size_metadata(self):
        for folder, manifest in (("head", self.head), ("ssl", self.ssl)):
            for name, figure in manifest["figures"].items():
                with Image.open(self.root / folder / figure["files"]["png"]) as image:
                    self.assertEqual(list(image.size), figure["size_pixels"])
                    self.assertEqual(list(image.size), [2640, 1740])
                    self.assertAlmostEqual(image.info["dpi"][0], 300, delta=0.01)
                self.assertTrue((self.root / folder / figure["files"]["svg"]).is_file())
                self.assertFalse((self.root / folder / "figures" / f"{name}.pdf").exists())
            with Image.open(self.root / folder / manifest["overview"]["file"]) as image:
                self.assertEqual(list(image.size), [3300, 2250])
            saved = json.loads((self.root / folder / "figures/plot_manifest.json").read_text())
            self.assertEqual(saved["style_config"], str(PLOT_CONFIG))
            self.assertIn(saved["font_family"], {"Arial", "DejaVu Sans"})
            self.assertEqual(saved["dpi"], 300)
            self.assertEqual(list((self.root / folder).rglob(".*.tmp")), [])

    def test_svg_text_is_editable_and_scores_have_consistent_percentage_units(self):
        for name, label in (("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1 (%)")):
            path = self.root / "head/figures" / f"{name}.svg"
            root = ET.fromstring(path.read_text())
            texts = ["".join(node.itertext()) for node in root.iter("{http://www.w3.org/2000/svg}text")]
            self.assertGreater(len(texts), 10)
            self.assertIn(label, texts)
            self.assertIn("100", texts)
            self.assertIn("Epoch", texts)
            self.assertNotIn("0.25", texts)
            self.assertIn("1", texts)
            self.assertIn("4", texts)
        self.assertEqual(self.ssl["figures"]["loss"]["ylabel"], "Mean squared error")
        self.assertEqual(self.ssl["figures"]["gradient_norm"]["ylabel"], "L2 gradient norm")

    def test_invalid_or_misspelled_plot_settings_fail_before_rendering(self):
        for section, key, value in (
            ("typography", "tick_size", 0), ("exports", "dpi", False), ("exports", "svg", False),
            ("colors", "training", "not-a-color"), ("axes", "score_unit", "mixed"),
            ("axes", "max_x_ticks", 1), ("axes", "scientific_format", 123),
            ("lines", "marker", "invalid"), ("canvas", "standalone_size", [True, 6]),
        ):
            settings = deepcopy(self.settings)
            settings[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate_plot_settings(settings)
        settings = deepcopy(self.settings)
        settings["typography"]["tick_szie"] = 16
        with self.assertRaises(ValueError):
            validate_plot_settings(settings)

    def test_font_cache_is_local_lazy_and_respects_existing_user_settings(self):
        fake_os = SimpleNamespace(environ={})
        fake_sys = SimpleNamespace(modules={})
        with patch("utilities.plotting.os", fake_os), patch("utilities.plotting.sys", fake_sys), \
                patch("utilities.plotting.PROJECT_DIR", self.root / "cache_test"):
            prepare_plot_cache()
            self.assertEqual(Path(fake_os.environ["MPLCONFIGDIR"]), self.root / "cache_test/.cache/matplotlib")
            self.assertTrue(Path(fake_os.environ["MPLCONFIGDIR"]).is_dir())
            fake_os.environ = {"MPLCONFIGDIR": "/custom/matplotlib", "XDG_CACHE_HOME": "/custom/cache"}
            prepare_plot_cache()
            self.assertEqual(fake_os.environ["MPLCONFIGDIR"], "/custom/matplotlib")
            self.assertEqual(fake_os.environ["XDG_CACHE_HOME"], "/custom/cache")
            fake_os.environ = {}
            fake_sys.modules = {"matplotlib": object()}
            prepare_plot_cache()
            self.assertEqual(fake_os.environ, {})


if __name__ == "__main__":
    unittest.main()

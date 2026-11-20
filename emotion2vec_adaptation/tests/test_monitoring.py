"""Exercise durable logging and live snapshots independently of training math."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.monitoring import (
    MONITORING_DEFAULTS, RunMonitor, validate_monitoring_settings,
)


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "run"
        self.quiet = {"console_log": False, "progress_bar": False}

    def test_settings_defaults_and_invalid_values(self):
        defaults = validate_monitoring_settings()
        self.assertEqual(defaults, MONITORING_DEFAULTS)
        defaults["enabled"] = False
        self.assertTrue(MONITORING_DEFAULTS["enabled"])
        for override in ({"enable": True}, {"enabled": "true"}, {"progress_bar": 1},
                         {"console_log": None}, {"plot_every_steps": 0}, {"log_every_steps": True},
                         {"max_plot_points": 3}, {"max_plot_points": False},
                         {"refresh_seconds": 1.5}, {"loss_smoothing": 1},
                         {"loss_smoothing": float("nan")}, {"loss_smoothing": -0.1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_monitoring_settings(override)

    def test_disabled_monitor_creates_nothing_and_preserves_iterable(self):
        values = [1, 2]
        with patch("utilities.monitoring.tqdm") as progress:
            with RunMonitor(self.root, "head", {"enabled": False}) as monitor:
                self.assertIs(monitor.progress(values, "training"), values)
                monitor.log("disabled message")
                monitor.record({"loss": 1.0}, step=1)
            progress.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_progress_bar_switch_and_total(self):
        with patch.object(RunMonitor, "_render"), patch("utilities.monitoring.tqdm") as progress:
            with RunMonitor(self.root, "head", self.quiet) as monitor:
                values = [1, 2]
                self.assertIs(monitor.progress(values, "disabled"), values)
                progress.assert_not_called()
            with RunMonitor(self.root / "bar", "head", {**self.quiet, "progress_bar": True}) as monitor:
                bar = monitor.progress(values, "head epoch", total=2)
                self.assertIs(bar, progress.return_value)
                self.assertEqual(progress.call_args.kwargs["total"], 2)
                self.assertEqual(progress.call_args.kwargs["desc"], "head epoch")
            progress.return_value.close.assert_called_once()

    def test_json_metrics_detach_scalars_and_log_frequency(self):
        with patch.object(RunMonitor, "_render") as render:
            with RunMonitor(self.root, "pretrain", {**self.quiet, "log_every_steps": 3,
                                                   "plot_every_steps": 2}) as monitor:
                monitor.log("Start loading local encoder")
                for step in range(1, 4):
                    monitor.record({"loss": torch.tensor(1 / step, requires_grad=True),
                                    "gradient_norm": 1.5, "lr": 1e-5}, step=step, epoch=1)
                monitor.record({"train_loss": 0.6, "validation_loss": 0.8, "global_step": 3}, epoch=1)
                x, values = monitor._series("validation_loss")
                self.assertEqual((x, values), ([3], [0.8]))
                self.assertIsInstance(monitor._records[0]["metrics"]["loss"], float)
            # Initial, first step, second step, epoch, and final close snapshot.
            self.assertEqual(render.call_count, 5)
        rows = [json.loads(line) for line in (self.root / "logs/metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["event"], "train_step")
        self.assertEqual(rows[0]["stage"], "pretrain")
        self.assertEqual(rows[-1]["event"], "epoch")
        self.assertEqual(rows[-1]["step"], 3)
        self.assertEqual(rows[0]["schema_version"], 1)
        self.assertTrue(rows[0]["timestamp"].endswith("+00:00"))
        log = (self.root / "logs/train.log").read_text()
        self.assertIn("Start loading local encoder", log)
        self.assertIn("step 1, epoch 1", log)
        self.assertNotIn("step 2, epoch 1", log)
        self.assertIn("step 3, epoch 1", log)

    def test_invalid_metrics_are_rejected_before_json_write(self):
        with patch.object(RunMonitor, "_render"):
            with RunMonitor(self.root, "head", self.quiet) as monitor:
                for value in (float("nan"), float("inf"), "bad", torch.ones(2), None):
                    with self.subTest(value=value), self.assertRaises(ValueError):
                        monitor.record({"loss": value}, epoch=1)
                with self.assertRaises(ValueError):
                    monitor.record({"loss": 1.0})
                with self.assertRaises(ValueError):
                    monitor.record({"loss": 1.0}, step=-1)
        self.assertEqual((self.root / "logs/metrics.jsonl").read_text(), "")

    @patch("utilities.monitoring.render_training_figures", return_value={"figures": {"loss": {"title": "Classification loss", "files": {"png": "figures/loss.png", "svg": "figures/loss.svg"}}}})
    def test_head_dashboard_links_and_refresh(self, renderer):
        with RunMonitor(self.root, "head", {**self.quiet, "refresh_seconds": 2}) as monitor:
            monitor.record({"train_loss": 1.2, "validation_loss": 1.4, "train_accuracy": 0.5,
                            "validation_accuracy": 0.4, "train_macro_f1": 0.45,
                            "validation_macro_f1": 0.38}, epoch=1)
            monitor.record({"train_loss": 0.7, "validation_loss": 0.9, "train_accuracy": 0.75,
                            "validation_accuracy": 0.65, "train_macro_f1": 0.7,
                            "validation_macro_f1": 0.6}, epoch=2)
        document = (self.root / "monitoring/dashboard.html").read_text()
        self.assertIn('http-equiv="refresh" content="2"', document)
        self.assertIn("completed", document)
        self.assertIn("validation_macro_f1", document)
        self.assertIn("../logs/train.log", document)
        self.assertIn("curves.png?v=", document)
        self.assertIn("../figures/loss.svg", document)
        self.assertIn("SVG (editable)", document)
        self.assertEqual(renderer.call_count, 4)
        monitor.close()  # Closing twice is safe.
        with self.assertRaises(RuntimeError):
            monitor.record({"train_loss": 0.5}, epoch=3)

    @patch("utilities.monitoring.render_training_figures", return_value={"figures": {}})
    def test_ssl_dashboard_plots_optional_metrics_and_ema(self, renderer):
        with RunMonitor(self.root, "pretrain", {**self.quiet, "max_plot_points": 4}) as monitor:
            monitor.record({"loss": 1.2, "target_std": 0.7, "prediction_std": 0.4,
                            "gradient_norm": 2.0, "learning_rate": 1e-5}, step=1)
            monitor.record({"loss": 0.8, "target_std": 0.75, "prediction_std": 0.45,
                            "gradient_norm": 1.8, "learning_rate": 1e-5}, step=2)
            for step in range(3, 11):
                monitor.record({"loss": 0.9 - step * 0.05}, step=step)
            monitor.record({"train_loss": 1.0, "validation_loss": 1.1, "global_step": 10}, epoch=1)
        document = (self.root / "monitoring/dashboard.html").read_text()
        self.assertIn("validation_loss", document)
        self.assertIn("Chart decimated for display; full metrics logs retained.", document)

    @patch("utilities.monitoring.render_training_figures", return_value={"figures": {}})
    def test_context_logs_full_failure_and_propagates_exception(self, renderer):
        with self.assertRaisesRegex(RuntimeError, "example training error"):
            with RunMonitor(self.root, "head", self.quiet):
                raise RuntimeError("example training error")
        log = (self.root / "logs/train.log").read_text()
        self.assertIn("Traceback (most recent call last)", log)
        self.assertIn("example training error", log)
        self.assertIn("failed", (self.root / "monitoring/dashboard.html").read_text())

    def test_console_logging_switch_and_append_preserve_previous_metrics(self):
        output = StringIO()
        with patch.object(RunMonitor, "_render"), redirect_stdout(output):
            with RunMonitor(self.root, "head", {**self.quiet, "console_log": True}) as monitor:
                monitor.record({"train_loss": 1.0}, epoch=1)
            with RunMonitor(self.root, "head", self.quiet) as monitor:
                self.assertEqual(len(monitor._records), 1)
                monitor.record({"train_loss": 0.8}, epoch=2)
        self.assertIn("epoch 1", output.getvalue())
        self.assertNotIn("epoch 2", output.getvalue())
        self.assertEqual(len((self.root / "logs/metrics.jsonl").read_text().splitlines()), 2)

    def test_plot_cache_is_bounded_but_full_metrics_and_true_ema_are_preserved(self):
        settings = {**self.quiet, "max_plot_points": 8, "plot_every_steps": 1000,
                    "log_every_steps": 1000, "loss_smoothing": 0.5}
        with patch.object(RunMonitor, "_render"):
            with RunMonitor(self.root, "pretrain", settings) as monitor:
                expected_ema = None
                for step in range(1, 102):
                    loss = float(step % 7)
                    expected_ema = loss if expected_ema is None else (expected_ema + loss) * 0.5
                    monitor.record({"loss": loss}, step=step, epoch=step // 10 + 1)
                    if step % 10 == 0:
                        monitor.record({"validation_loss": loss + 0.5, "global_step": step},
                                       epoch=step // 10)
                    self.assertLessEqual(len(monitor._records), 16)
                self.assertEqual(monitor._event_count, 111)
                self.assertEqual(monitor._records[0]["event_index"], 1)
                self.assertEqual(monitor._records[-1]["event_index"], 111)
                self.assertTrue(monitor._chart_decimated)
                x, values = monitor._series("loss")
                self.assertLessEqual(len(x), 8)
                self.assertEqual((x[0], x[-1]), (1, 101))
                self.assertAlmostEqual(monitor._loss_ema, expected_ema)
        rows = [json.loads(line) for line in (self.root / "logs/metrics.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 111)
        self.assertEqual([row["event_index"] for row in rows], list(range(1, 112)))
        self.assertAlmostEqual(rows[-1]["metrics"]["loss_ema"], expected_ema)
        with patch.object(RunMonitor, "_render"):
            with RunMonitor(self.root, "pretrain", settings) as monitor:
                self.assertEqual(monitor._event_count, 111)
                self.assertLessEqual(len(monitor._records), 16)
                self.assertAlmostEqual(monitor._loss_ema, expected_ema)
                monitor.record({"loss": 2.0}, step=102)
        last = json.loads((self.root / "logs/metrics.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["event_index"], 112)
        self.assertAlmostEqual(last["metrics"]["loss_ema"], (expected_ema + 2.0) * 0.5)

    def test_head_series_ignore_batch_accuracy_and_preserve_epoch_values(self):
        with patch.object(RunMonitor, "_render") as render:
            with RunMonitor(self.root, "head", self.quiet) as monitor:
                monitor.record({"loss": 0.8, "accuracy": 1.0}, step=1, epoch=1)
                monitor.record({"loss": 0.7, "accuracy": 0.9}, step=2, epoch=1)
                self.assertEqual(render.call_count, 1)  # Only initial placeholder, no redundant batch redraw.
                monitor.record({"train_loss": 0.75, "validation_accuracy": 0.5}, epoch=1)
                self.assertEqual(monitor._series("validation_accuracy", "accuracy"), ([1], [0.5]))
                self.assertEqual(monitor._event_count, 3)


if __name__ == "__main__":
    unittest.main()

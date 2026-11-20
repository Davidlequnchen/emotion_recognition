"""Terminal progress, durable scalar metrics and a server-free live dashboard.

Every record is flushed to logs/metrics.jsonl. Human-readable output goes to
logs/train.log, while monitoring/dashboard.html refreshes atomic PNG snapshots.
No logger handlers or matplotlib global backends are changed by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from html import escape
import json
import math
import numbers
import os
from pathlib import Path
import sys
import tempfile
import traceback

from tqdm import tqdm

from .plotting import render_training_figures

MONITORING_DEFAULTS = {
    "enabled": True,
    "progress_bar": True,
    "console_log": True,
    "log_every_steps": 10,
    "plot_every_steps": 20,
    "max_plot_points": 2000,
    "refresh_seconds": 5,
    "loss_smoothing": 0.9,
}


def validate_monitoring_settings(settings: Mapping | None = None) -> dict:
    """Fill optional defaults and reject misspelled or ill-typed YAML values."""
    if settings is None:
        settings = {}
    if not isinstance(settings, Mapping):
        raise ValueError("monitoring must be a YAML mapping")
    unknown = set(settings) - set(MONITORING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown monitoring settings: {sorted(unknown)}")
    result = {**MONITORING_DEFAULTS, **settings}
    for name in ("enabled", "progress_bar", "console_log"):
        if not isinstance(result[name], bool):
            raise ValueError(f"monitoring.{name} must be a YAML boolean")
    for name in ("log_every_steps", "plot_every_steps", "refresh_seconds"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"monitoring.{name} must be a positive integer")
    point_limit = result["max_plot_points"]
    if isinstance(point_limit, bool) or not isinstance(point_limit, int) or point_limit < 4:
        raise ValueError("monitoring.max_plot_points must be an integer >= 4")
    smoothing = result["loss_smoothing"]
    if (isinstance(smoothing, bool) or not isinstance(smoothing, numbers.Real)
            or not math.isfinite(smoothing) or not 0 <= smoothing < 1):
        raise ValueError("monitoring.loss_smoothing must be finite and in [0, 1)")
    result["loss_smoothing"] = float(smoothing)
    return result


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _scalar(value, name):
    """Detach scalar tensors before serialization; retain no computation graph."""
    if hasattr(value, "detach") and hasattr(value, "numel"):
        if value.numel() != 1:
            raise ValueError(f"Metric {name!r} must be a scalar, not a tensor array")
        value = value.detach().cpu().item()
    elif hasattr(value, "item") and not isinstance(value, numbers.Real):
        value = value.item()
    if isinstance(value, bool):
        return value
    if not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"Metric {name!r} must be a finite numeric scalar")
    return int(value) if isinstance(value, numbers.Integral) else float(value)


def _atomic_text(path: Path, text: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.stem}-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _downsample(values: list, budget: int) -> list:
    """Keep evenly spaced items including the first and last without numpy."""
    if len(values) <= budget:
        return list(values)
    if budget == 0:
        return []
    if budget == 1:
        return [values[-1]]
    return [values[round(index * (len(values) - 1) / (budget - 1))] for index in range(budget)]


class RunMonitor:
    """Monitor one head or SSL run without touching model/optimizer behavior.

    ``record(metrics, step=n, epoch=e)`` emits a train_step event. Omitting step
    emits an epoch event; for SSL its step coordinate uses the last observed
    training step. Metric JSONL rows contain schema_version, timestamp, stage,
    event_index, event, step, epoch and a nested metrics mapping of finite scalars.

    Epoch events always render. SSL step events render on the first event and
    every plot_every_steps; head batch events are logged without redrawing its
    epoch-only chart. update_plot=True and close() force a current snapshot.
    ``loss_smoothing`` is the previous-value weight in an exponential average.
    The plotting cache holds at most twice max_plot_points events and is compacted
    incrementally. Each plotted series is also capped; JSONL retains every event.
    """

    def __init__(self, run_dir: Path, stage: str, settings: Mapping | None = None):
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("Monitoring stage must be a nonempty string")
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.stage = stage
        self.settings = validate_monitoring_settings(settings)
        self.enabled = self.settings["enabled"]
        self.log_dir = self.run_dir / "logs"
        self.live_dir = self.run_dir / "monitoring"
        self.dashboard_path = self.live_dir / "dashboard.html"
        self.curves_path = self.live_dir / "curves.png"
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.log_path = self.log_dir / "train.log"
        self._records = []
        self._bars = []
        self._closed = False
        self._failed = False
        self._last_step = None
        self._step_records = 0
        self._event_count = 0
        self._chart_decimated = False
        self._loss_ema = None
        self._log_handle = self._metrics_handle = None
        if not self.enabled:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.live_dir.mkdir(parents=True, exist_ok=True)
        if self.metrics_path.exists():
            # Stream old logs too: resuming a long run must not materialize an
            # entire metrics JSONL in RAM merely to draw its dashboard.
            with self.metrics_path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("schema_version") != 1 or row.get("stage") != stage:
                        raise ValueError(f"Existing metrics belong to a different format/stage: {self.metrics_path}")
                    self._event_count += 1
                    row["event_index"] = self._event_count
                    self._update_loss_ema(row)
                    self._records.append(row)
                    self._compact_cache()
                    if row["step"] is not None:
                        self._last_step = row["step"]
                    self._step_records += int(row["event"] == "train_step")
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._metrics_handle = self.metrics_path.open("a", encoding="utf-8", buffering=1)
        try:
            self.log(f"Monitoring {stage}; dashboard: {self.dashboard_path}")
            self._render("running")
        except BaseException:
            self._log_handle.close()
            self._metrics_handle.close()
            self._closed = True
            raise

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, exception_traceback):
        try:
            if exception is not None:
                self.log_exception(exception)
        finally:
            self.close()
        return False

    def progress(self, iterable, description: str, total: int | None = None):
        """Wrap an iterable in a terminal bar only when enabled in YAML."""
        if not self.enabled or not self.settings["progress_bar"]:
            return iterable
        bar = tqdm(iterable, desc=description, total=total, dynamic_ncols=True,
                   leave=True, file=sys.stderr)
        self._bars.append(bar)
        return bar

    def log(self, message: str) -> None:
        """Flush one human-readable message to the dedicated log and console."""
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("Cannot log to a closed RunMonitor")
        line = f"[{_timestamp()}] [{self.stage}] {message}"
        self._log_handle.write(line + "\n")
        self._log_handle.flush()
        if self.settings["console_log"]:
            tqdm.write(line, file=sys.stdout)
            sys.stdout.flush()

    def log_exception(self, error: BaseException) -> None:
        """Retain the full traceback and mark the dashboard as failed."""
        self._failed = True
        self.log("Training failed:\n" + "".join(traceback.format_exception(type(error), error, error.__traceback__)))

    def record(self, metrics: Mapping, step: int | None = None, epoch: int | None = None,
               update_plot: bool = False) -> None:
        if not self.enabled:
            return
        if self._closed:
            raise RuntimeError("Cannot record to a closed RunMonitor")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("Monitoring metrics must be a nonempty mapping")
        for name, value in (("step", step), ("epoch", epoch)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"Monitoring {name} must be a nonnegative integer or null")
        if step is None and epoch is None:
            raise ValueError("Monitoring record requires a step or epoch")
        if any(not isinstance(key, str) or not key.strip() for key in metrics):
            raise ValueError("Metric keys must be nonempty strings")
        scalar_metrics = {key: _scalar(value, key) for key, value in metrics.items()}
        event = "train_step" if step is not None else "epoch"
        if step is not None:
            self._last_step = step
            self._step_records += 1
        coordinate = step if step is not None else (
            None if self.stage == "head" else scalar_metrics.get("global_step", self._last_step)
        )
        self._event_count += 1
        row = {"schema_version": 1, "timestamp": _timestamp(), "stage": self.stage,
               "event_index": self._event_count,
               "event": event, "step": coordinate, "epoch": epoch, "metrics": scalar_metrics}
        self._update_loss_ema(row)
        self._metrics_handle.write(json.dumps(row, allow_nan=False) + "\n")
        self._metrics_handle.flush()
        self._records.append(row)
        self._compact_cache()
        first_step = event == "train_step" and self._step_records == 1
        if event == "epoch" or first_step or step % self.settings["log_every_steps"] == 0:
            position = f"epoch {epoch}" if event == "epoch" else f"step {step}" + (f", epoch {epoch}" if epoch is not None else "")
            values = ", ".join(f"{name}={value:.6g}" for name, value in scalar_metrics.items())
            self.log(f"{position}: {values}")
        if (update_plot or event == "epoch" or (self.stage != "head" and
                (first_step or step % self.settings["plot_every_steps"] == 0))):
            self._render("running")

    def _update_loss_ema(self, row: dict) -> None:
        """Compute EMA on every original step, before any plot-cache sampling."""
        if self.stage == "head" or row["event"] != "train_step":
            return
        metrics = row["metrics"]
        loss = metrics.get("loss", metrics.get("train_loss"))
        if loss is None:
            return
        beta = self.settings["loss_smoothing"]
        self._loss_ema = loss if self._loss_ema is None else beta * self._loss_ema + (1 - beta) * loss
        metrics["loss_ema"] = self._loss_ema

    def _compact_cache(self) -> None:
        """Bound plot memory while retaining early/late step and epoch events."""
        limit = self.settings["max_plot_points"]
        if len(self._records) <= 2 * limit:
            return
        epochs = [row for row in self._records if row["event"] == "epoch"]
        steps = [row for row in self._records if row["event"] == "train_step"]
        epoch_budget = min(len(epochs), limit // 2)
        step_budget = min(len(steps), limit - epoch_budget)
        epoch_budget += min(len(epochs) - epoch_budget, limit - epoch_budget - step_budget)
        retained = _downsample(epochs, epoch_budget) + _downsample(steps, step_budget)
        self._records = sorted(retained, key=lambda row: row["event_index"])
        self._chart_decimated = True

    def _series(self, *names):
        points = []
        for index, row in enumerate(self._records, 1):
            # Per-batch head accuracy belongs to the step log. Mixing it into
            # validation accuracy creates many unrelated points at one epoch.
            if self.stage == "head" and row["event"] != "epoch":
                continue
            value = next((row["metrics"][name] for name in names if name in row["metrics"]), None)
            if value is None:
                continue
            coordinate = row["epoch"] if self.stage == "head" else row["step"]
            if coordinate is None:
                coordinate = row["step"] if self.stage == "head" else row["epoch"]
            points.append((index if coordinate is None else coordinate, value))
        if len(points) > self.settings["max_plot_points"]:
            points = _downsample(points, self.settings["max_plot_points"])
            self._chart_decimated = True
        return [point[0] for point in points], [point[1] for point in points]

    def _render(self, status: str) -> None:
        manifest = render_training_figures(self.stage, self.run_dir, self._series)
        updated = _timestamp()
        latest = self._records[-1] if self._records else None
        table = "" if latest is None else "".join(
            f"<tr><td>{escape(key)}</td><td>{value:.6g}</td></tr>" for key, value in latest["metrics"].items()
        )
        position = "Waiting for first metrics" if latest is None else (
            f"Latest event: {latest['event']} · step {latest['step']} · epoch {latest['epoch']}"
        )
        chart_note = ("Chart decimated for display; full metrics logs retained."
                      if self._chart_decimated else "Full scalar metrics are retained in the JSONL log.")
        figure_links = "".join(
            f"<li><strong>{escape(figure['title'])}</strong>: " + " · ".join(
                f'<a href="../{escape(path)}">{format_name.upper()}{" (editable)" if format_name == "svg" else ""}</a>'
                for format_name, path in figure["files"].items()
            ) + "</li>" for figure in manifest["figures"].values()
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{self.settings['refresh_seconds']}">
<title>{escape(self.stage)} training monitor</title>
<style>body{{font:15px system-ui,sans-serif;background:#f1f5f9;color:#172033;margin:24px auto;max-width:1300px;padding:0 20px}}
h1{{font-size:25px;margin-bottom:8px}}p{{color:#475569}}img{{width:100%;height:auto;background:white;border:1px solid #e2e8f0;border-radius:8px}}
table{{border-collapse:collapse;background:white;min-width:320px}}td{{padding:7px 16px;border-bottom:1px solid #e2e8f0}}td:last-child{{text-align:right;font-variant-numeric:tabular-nums}}
a{{color:#2563eb}}.status{{font-weight:700}}code{{overflow-wrap:anywhere}}</style></head>
<body><h1>{escape(self.stage)} training monitor</h1>
<p>Status: <span class="status">{escape(status)}</span> · updated {escape(updated)} · refresh {self.settings['refresh_seconds']}s</p>
<p><code>{escape(str(self.run_dir))}</code></p><p>{escape(position)} · {self._event_count} recorded events</p>
<p>{escape(chart_note)} Plot cache: {len(self._records)} events; each curve displays at most {self.settings['max_plot_points']} points.</p>
<p><a href="../logs/train.log">Training log</a> · <a href="../logs/metrics.jsonl">Scalar metrics JSONL</a></p>
<a href="curves.png?v={escape(updated)}"><img src="curves.png?v={escape(updated)}" alt="Live training curves; click for full resolution"></a>
<h2>Standalone figures</h2><p>300-dpi PNG and editable vector SVG; click the overview for full resolution.</p><ul>{figure_links}</ul>
<p><a href="../figures/plot_manifest.json">Figure style and export metadata</a></p><h2>Latest metrics</h2><table>{table}</table>
</body></html>"""
        _atomic_text(self.dashboard_path, document)

    def close(self) -> None:
        """Flush final curves and release only resources owned by this monitor."""
        if self._closed:
            return
        if not self.enabled:
            self._closed = True
            return
        try:
            for bar in self._bars:
                bar.close()
            status = "failed" if self._failed else "completed"
            self.log(f"Monitoring {status}; {self._event_count} scalar events recorded")
            self._render(status)
        finally:
            self._log_handle.close()
            self._metrics_handle.close()
            self._closed = True

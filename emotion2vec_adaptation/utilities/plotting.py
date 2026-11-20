"""Benchmark-style training figures with editable vector and 300-dpi exports.

Rendering is separate from monitoring state and training objectives. All visual
choices come from config/plots.yaml. Each metric has its own standalone figure;
the dashboard overview uses the same labels, colors and readable typography.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

import yaml

from .config import PROJECT_DIR


PLOT_CONFIG = PROJECT_DIR / "config/plots.yaml"
PLOT_SCHEMA = {
    "typography": {"families", "base_size", "tick_size", "label_size", "title_size", "legend_size", "overview_title_size", "title_weight"},
    "colors": {"training", "validation", "ema", "raw", "target", "prediction", "ink", "grid", "background"},
    "canvas": {"standalone_size", "overview_size", "padding", "panel_wspace", "panel_hspace"},
    "axes": {"line_width", "label_padding", "title_padding", "tick_length", "tick_width", "max_x_ticks", "max_y_ticks", "score_unit", "score_tick_count", "score_padding", "value_top_padding", "zero_baseline", "grid_axis", "grid_alpha", "grid_width", "top_spine", "right_spine", "scientific_format"},
    "lines": {"width", "raw_width", "raw_alpha", "marker_size", "marker", "validation_marker", "max_markers", "training_style", "validation_style", "ema_style"},
    "legend": {"location", "frame", "columns", "handle_length"},
    "exports": {"dpi", "png", "svg", "pdf"},
}


def prepare_plot_cache() -> None:
    """Choose local font caches lazily; preserve an existing user environment."""
    if "matplotlib" in sys.modules:
        return
    cache = PROJECT_DIR / ".cache"
    for variable, directory in (("MPLCONFIGDIR", cache / "matplotlib"), ("XDG_CACHE_HOME", cache)):
        if variable not in os.environ:
            directory.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(directory)


def load_plot_settings(path: Path | None = None) -> dict:
    path = Path(path or PLOT_CONFIG)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"plots"}:
        raise ValueError("plots.yaml must contain only the plots mapping")
    settings = value["plots"]
    validate_plot_settings(settings)
    return settings


def validate_plot_settings(settings: dict) -> None:
    if not isinstance(settings, dict) or set(settings) != set(PLOT_SCHEMA):
        raise ValueError(f"plots must contain exactly {sorted(PLOT_SCHEMA)}")
    for section, keys in PLOT_SCHEMA.items():
        values = settings[section]
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError(f"plots.{section} must contain exactly {sorted(keys)}")
    for key in ("standalone_size", "overview_size"):
        values = settings["canvas"][key]
        if not isinstance(values, list) or len(values) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"plots.canvas.{key} must contain two positive inch dimensions")
    boolean_keys = {"axes": ("zero_baseline", "top_spine", "right_spine"), "legend": ("frame",), "exports": ("png", "svg", "pdf")}
    for section, keys in boolean_keys.items():
        for key in keys:
            if not isinstance(settings[section][key], bool):
                raise ValueError(f"plots.{section}.{key} must be a YAML boolean")
    for section, keys in {
        "typography": ("base_size", "tick_size", "label_size", "title_size", "legend_size", "overview_title_size"),
        "canvas": ("padding", "panel_wspace", "panel_hspace"),
        "axes": ("line_width", "label_padding", "title_padding", "tick_length", "tick_width", "grid_width", "score_padding", "value_top_padding"),
        "lines": ("width", "raw_width", "marker_size"), "legend": ("handle_length",),
    }.items():
        for key in keys:
            value = settings[section][key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"plots.{section}.{key} must be a positive finite number")
    for section, keys in {"axes": ("max_x_ticks", "max_y_ticks", "score_tick_count"), "lines": ("max_markers",), "legend": ("columns",), "exports": ("dpi",)}.items():
        for key in keys:
            value = settings[section][key]
            minimum = 2 if section == "axes" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"plots.{section}.{key} must be an integer >= {minimum}")
    for section, key in (("axes", "grid_alpha"), ("lines", "raw_alpha")):
        if isinstance(settings[section][key], bool) or not isinstance(settings[section][key], (int, float)) or not 0 <= settings[section][key] <= 1:
            raise ValueError(f"plots.{section}.{key} must be in [0, 1]")
    if settings["axes"]["score_unit"] not in {"percent", "fraction"} or settings["axes"]["grid_axis"] not in {"x", "y", "both"}:
        raise ValueError("Invalid plots axes score_unit or grid_axis")
    if not isinstance(settings["typography"]["families"], list) or not settings["typography"]["families"] or any(not isinstance(v, str) or not v for v in settings["typography"]["families"]):
        raise ValueError("plots.typography.families must contain font names")
    if any(not isinstance(value, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", value) is None for value in settings["colors"].values()):
        raise ValueError("plots.colors must contain six-digit hex colors")
    if settings["typography"]["title_weight"] not in {"normal", "bold"}:
        raise ValueError("plots.typography.title_weight must be normal or bold")
    if settings["legend"]["location"] not in ("best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center"):
        raise ValueError("Invalid plots.legend.location")
    if not settings["exports"]["png"] or not settings["exports"]["svg"]:
        raise ValueError("Training figures require both PNG and editable SVG exports")
    for key in ("training_style", "validation_style", "ema_style"):
        if settings["lines"][key] not in {"-", "--", "-.", ":"}:
            raise ValueError(f"Invalid plots.lines.{key}")
    for key in ("marker", "validation_marker"):
        if settings["lines"][key] not in {"o", "s", "D", "^", "v", "x", "+"}:
            raise ValueError(f"Invalid plots.lines.{key}")
    try:
        if not isinstance(settings["axes"]["scientific_format"], str):
            raise TypeError("scientific_format is not a string")
        settings["axes"]["scientific_format"] % 0.001
    except (TypeError, ValueError) as error:
        raise ValueError("plots.axes.scientific_format must be a numeric format such as %.1e") from error


@dataclass
class PlotLine:
    label: str
    role: str
    x: list
    y: list


@dataclass
class MetricPlot:
    name: str
    title: str
    ylabel: str
    lines: list[PlotLine]
    score: bool = False
    scientific: bool = False


def _metric_plots(stage, series, settings):
    def line(label, role, *keys):
        x, y = series(*keys)
        return PlotLine(label, role, x, y)

    if stage == "head":
        unit = "%" if settings["axes"]["score_unit"] == "percent" else "0–1"
        return [
            MetricPlot("loss", "Classification loss", "Cross-entropy loss", [line("Training", "training", "train_loss"), line("Validation", "validation", "validation_loss", "val_loss")]),
            MetricPlot("accuracy", "Classification accuracy", f"Accuracy ({unit})", [line("Training", "training", "train_accuracy"), line("Validation", "validation", "validation_accuracy", "accuracy")], score=True),
            MetricPlot("macro_f1", "Macro-F1 score", f"Macro-F1 ({unit})", [line("Training", "training", "train_macro_f1", "train_f1"), line("Validation", "validation", "validation_macro_f1", "macro_f1")], score=True),
        ]
    raw_key = "loss" if series("loss")[0] else "train_loss"
    unit = "%" if settings["axes"]["score_unit"] == "percent" else "0–1"
    return [
        # a loss that stays on the constant-prediction line means the target is not being learned
        MetricPlot("loss", "Masked prediction loss", "Mean squared error", [line("Training (raw)", "raw", raw_key), line("Training (EMA)", "ema", "loss_ema"), line("Validation", "validation", "validation_loss", "val_loss"), line("Constant prediction", "target", "constant_loss")]),
        # target_std is not plotted: normalisation pins it, so it cannot reveal collapse
        MetricPlot("downstream_uar", "Downstream check", f"Validation UAR ({unit})", [line("Linear head on the current encoder", "validation", "downstream_uar")], score=True),
        MetricPlot("gradient_norm", "Gradient norm", "L2 gradient norm", [line("Before clipping", "training", "gradient_norm", "grad_norm")]),
        MetricPlot("learning_rate", "Optimizer learning rate", "Learning rate", [line("AdamW", "ema", "learning_rate", "lr")], scientific=True),
    ]


def _style_axis(axis, plot, stage, settings):
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator

    axes, colors, typography = settings["axes"], settings["colors"], settings["typography"]
    axis.set_title(plot.title, loc="left", pad=axes["title_padding"])
    axis.set_xlabel("Epoch" if stage == "head" else "Training step", labelpad=axes["label_padding"])
    axis.set_ylabel(plot.ylabel, labelpad=axes["label_padding"])
    axis.spines["top"].set_visible(axes["top_spine"])
    axis.spines["right"].set_visible(axes["right_spine"])
    axis.grid(axis=axes["grid_axis"], color=colors["grid"], linewidth=axes["grid_width"], alpha=axes["grid_alpha"])
    axis.set_axisbelow(True)
    axis.tick_params(length=axes["tick_length"], width=axes["tick_width"], labelsize=typography["tick_size"])
    positions = sorted({int(value) for line in plot.lines for value in line.x})
    if positions:
        low, high = min(positions), max(positions)
        if high - low < axes["max_x_ticks"]:
            ticks = list(range(low, high + 1))
        else:
            locator = MaxNLocator(nbins=axes["max_x_ticks"] - 1, integer=True, min_n_ticks=1)
            ticks = [int(value) for value in locator.tick_values(low, high) if low <= value <= high]
        axis.xaxis.set_major_locator(FixedLocator(ticks))
        padding = max(0.5, (high - low) * 0.04)
        axis.set_xlim(low - padding, high + padding)
    else:
        axis.set_xlim(0, 1)
        axis.xaxis.set_major_locator(FixedLocator([0, 1]))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    if plot.score:
        maximum = 100 if axes["score_unit"] == "percent" else 1
        padding = maximum * axes["score_padding"]
        axis.set_ylim(-padding, maximum + padding)
        axis.yaxis.set_major_locator(FixedLocator([index * maximum / (axes["score_tick_count"] - 1) for index in range(axes["score_tick_count"])]))
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.0f" if maximum == 100 else "%.2g"))
    else:
        axis.yaxis.set_major_locator(MaxNLocator(nbins=axes["max_y_ticks"]))
        if axes["zero_baseline"]:
            values = [value for line in plot.lines for value in line.y]
            axis.set_ylim(0, max(values, default=1) * (1 + axes["value_top_padding"]) or 1)
        if plot.scientific:
            axis.yaxis.set_major_formatter(FormatStrFormatter(axes["scientific_format"]))
    if axis.lines:
        legend = settings["legend"]
        axis.legend(loc=legend["location"], frameon=legend["frame"], ncol=legend["columns"], handlelength=legend["handle_length"])
    else:
        axis.text(0.5, 0.5, "Waiting for metrics", transform=axis.transAxes, ha="center", va="center", color=colors["ink"])


def _draw(axis, plot, stage, settings):
    styles = settings["lines"]
    for line in plot.lines:
        if not line.x:
            continue
        raw = line.role == "raw"
        scale = 100 if plot.score and settings["axes"]["score_unit"] == "percent" else 1
        linestyle = styles.get(f"{line.role}_style", "-")
        axis.plot(line.x, [value * scale for value in line.y], label=line.label,
                  color=settings["colors"][line.role], linestyle=linestyle,
                  linewidth=styles["raw_width"] if raw else styles["width"],
                  alpha=styles["raw_alpha"] if raw else 1,
                  marker=styles["validation_marker"] if line.role == "validation" else (styles["marker"] if not raw or len(line.x) == 1 else None),
                  markersize=styles["marker_size"], markevery=max(1, math.ceil(len(line.x) / styles["max_markers"])),
                  solid_capstyle="round")
    _style_axis(axis, plot, stage, settings)


def _save(figure, path: Path, dpi: int) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=path.suffix, delete=False) as handle:
            temporary = Path(handle.name)
        metadata = {"Date": None} if path.suffix == ".svg" else ({"CreationDate": None, "ModDate": None} if path.suffix == ".pdf" else None)
        figure.savefig(temporary, format=path.suffix[1:], dpi=dpi, facecolor=figure.get_facecolor(), metadata=metadata)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def render_training_figures(stage, run_dir, series_provider, settings=None) -> dict:
    """Export standalone paper figures and the live overview, returning metadata."""
    settings = settings or load_plot_settings()
    validate_plot_settings(settings)
    prepare_plot_cache()
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in settings["typography"]["families"] if name in installed), "DejaVu Sans")
    typography, canvas, colors = settings["typography"], settings["canvas"], settings["colors"]
    rc = {"font.family": family, "font.size": typography["base_size"], "axes.titlesize": typography["title_size"],
          "axes.titleweight": typography["title_weight"], "axes.labelsize": typography["label_size"],
          "xtick.labelsize": typography["tick_size"], "ytick.labelsize": typography["tick_size"],
          "legend.fontsize": typography["legend_size"], "text.color": colors["ink"], "axes.labelcolor": colors["ink"],
          "axes.titlecolor": colors["ink"], "xtick.color": colors["ink"], "ytick.color": colors["ink"],
          "axes.edgecolor": colors["ink"], "axes.linewidth": settings["axes"]["line_width"],
          "figure.facecolor": colors["background"], "axes.facecolor": colors["background"], "svg.fonttype": "none",
          "svg.hashsalt": "emotion2vec-training-benchmark-style", "pdf.fonttype": 42}
    root = Path(run_dir)
    folder, live = root / "figures", root / "monitoring"
    folder.mkdir(parents=True, exist_ok=True)
    live.mkdir(parents=True, exist_ok=True)
    dpi = settings["exports"]["dpi"]
    formats = [name for name in ("png", "svg", "pdf") if settings["exports"][name]]
    plots = _metric_plots(stage, series_provider, settings)
    manifest = {"stage": stage, "style_config": str(PLOT_CONFIG), "style_config_sha256": hashlib.sha256(PLOT_CONFIG.read_bytes()).hexdigest(),
                "font_family": family, "font_sizes_pt": typography, "dpi": dpi, "settings": settings, "figures": {}}
    with matplotlib.rc_context(rc):
        for plot in plots:
            figure = Figure(figsize=canvas["standalone_size"], layout="constrained")
            FigureCanvasAgg(figure)
            figure.get_layout_engine().set(w_pad=canvas["padding"], h_pad=canvas["padding"])
            _draw(figure.subplots(), plot, stage, settings)
            paths = {name: str(Path("figures") / f"{plot.name}.{name}") for name in formats}
            for name, path in paths.items():
                _save(figure, root / path, dpi)
            manifest["figures"][plot.name] = {"title": plot.title, "ylabel": plot.ylabel, "xlabel": "Epoch" if stage == "head" else "Training step",
                                                  "files": paths, "size_inches": canvas["standalone_size"],
                                                  "size_pixels": [round(value * dpi) for value in canvas["standalone_size"]]}
            figure.clear()
        overview = Figure(figsize=canvas["overview_size"], layout="constrained")
        FigureCanvasAgg(overview)
        overview.get_layout_engine().set(w_pad=canvas["padding"], h_pad=canvas["padding"], wspace=canvas["panel_wspace"], hspace=canvas["panel_hspace"])
        grid = overview.add_gridspec(2, 2)
        axes = [overview.add_subplot(grid[0, :]), overview.add_subplot(grid[1, 0]), overview.add_subplot(grid[1, 1])] if stage == "head" else [overview.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]
        for axis, plot in zip(axes, plots):
            _draw(axis, plot, stage, settings)
        overview.suptitle("Frozen-encoder classification" if stage == "head" else "Continual encoder pre-training", fontsize=typography["overview_title_size"], fontweight=typography["title_weight"])
        _save(overview, live / "curves.png", dpi)
        overview.clear()
    manifest["overview"] = {"file": "monitoring/curves.png", "size_inches": canvas["overview_size"],
                              "size_pixels": [round(value * dpi) for value in canvas["overview_size"]]}
    manifest_path = folder / "plot_manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    return manifest

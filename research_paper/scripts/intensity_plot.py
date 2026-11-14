"""Plot classification diagnostics against published emotion-strength metadata."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import textwrap

from cross_dataset_plot import MODELS


DATASETS = (
    ("crema_d", "CREMA-D"), ("ravdess", "RAVDESS"), ("mead", "MEAD"),
    ("thai_ser", "THAI-SER"), ("mcae_spps", "MCAE-SPPS"), ("dailytalk", "DailyTalk"),
)
MODELS_COMPLETE = {key: value for key, value in MODELS.items() if key != "gemma4_e4b"}
INK, GRID = "#000000", "#dddddd"


def _number(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _save(fig, plt, folder, name):
    paths = [folder / f"{name}.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            fig.savefig(path, dpi=220, facecolor="white", metadata={"Creator": "intensity_plot.py"})
    finally:
        plt.close(fig)
    return paths


def _axes(ax, *, forest=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(axis="both", colors=INK, labelcolor=INK)
    (ax.xaxis if forest else ax.yaxis).grid(True, color=GRID, linewidth=.7, zorder=0)
    if forest:
        ax.axvline(0, color=INK, linewidth=1.0, linestyle="--", zorder=1)


def _strength_accuracy(plt, rows, scopes, destination):
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D

    by_dataset = defaultdict(list)
    for row in rows:
        if row["model"] in MODELS_COMPLETE:
            by_dataset[row["dataset"]].append(row)
    fig, axes = plt.subplots(3, 2, figsize=(23, 21))
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.18, wspace=.15, hspace=.53)
    for ax, (dataset, name) in zip(axes.flat, DATASETS):
        _axes(ax)
        scope = scopes.get(dataset, {})
        scale = str(scope.get("scale_name", "Published strength scale"))
        size = scope.get("n_non_neutral_known")
        scope_text = f"; n = {int(size):,} non-neutral clips" if size not in (None, "") else ""
        ax.set_title(name + "\n" + textwrap.fill(scale + scope_text, 66),
                     fontsize=17, fontweight="bold", color=INK, pad=11)
        available = by_dataset[dataset]
        levels = {}
        for row in available:
            value = _number(row.get("level"))
            if value is not None:
                levels[int(row["level_id"])] = (value, str(row["level_label"]))
        ordered = sorted(levels)
        positions = [levels[level][0] for level in ordered]
        labels = [levels[level][1] for level in ordered]
        ax.set_xticks(positions, labels, fontsize=16)
        if positions:
            span = max(positions) - min(positions)
            margin = max(.15, .10 * span)
            ax.set_xlim(min(positions) - margin, max(positions) + margin)
        ax.set_ylim(0, 105)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_xlabel("Listener-rating quartile (display only)" if scope.get("scale_kind") == "continuous"
                      else "Published strength level", fontsize=17, color=INK)
        for model, (_label, color) in MODELS_COMPLETE.items():
            selected = {int(row["level_id"]): row for row in available if row["model"] == model}
            values = []
            for level in ordered:
                row = selected.get(level)
                score = _number(row.get("accuracy")) if row else None
                values.append(100 * score if row and int(row.get("n", 0)) >= 20 and score is not None else math.nan)
            if not any(math.isfinite(value) for value in values):
                continue
            line, = ax.plot(positions, values, color=color, marker="o", markersize=7,
                            markeredgecolor=INK, markeredgewidth=.6, linewidth=2, zorder=3)
            if model == "qwen2_audio_7b":
                line.set_path_effects([path_effects.Stroke(linewidth=3, foreground=INK), path_effects.Normal()])
        if not positions:
            ax.text(.5, .5, "No eligible strength metadata", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color=INK)
    fig.suptitle("Emotion strength and emotion-classification accuracy", fontsize=26, fontweight="bold", y=.98)
    fig.text(.5, .94, "NON-NEUTRAL clips only; each dataset keeps its own published strength scale.",
             fontsize=19, color=INK, ha="center")
    fig.text(.5, .90, "These models predict emotion labels, not strength. Points need at least 20 clips.",
             fontsize=18, color=INK, ha="center")
    fig.supylabel("Emotion-classification accuracy (%)", fontsize=20, x=.014)
    handles = [Line2D([0], [0], color=color, marker="o", markeredgecolor=INK, linewidth=2,
                      markersize=6, label=label) for label, color in MODELS_COMPLETE.values()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.54, .075), ncol=5,
               frameon=False, fontsize=16, columnspacing=1.4, handlelength=1.8)
    fig.text(.075, .025, "MCAE-SPPS quartiles visualize continuous listener ratings; they are not native intensity classes.\n"
             "Unknown strength and neutral samples are excluded. Gemma has no eligible completed strength dataset.",
             fontsize=16, color=INK, linespacing=1.4)
    return _save(fig, plt, destination, "emotion_intensity_accuracy")


def _forest_point(ax, row, y, color, *, marker="o", value_key="slope_pp"):
    value = _number(row.get(value_key))
    if value is None:
        return False
    low, high = (_number(row.get(key)) for key in ("ci_lower_pp", "ci_upper_pp"))
    clusters = _number(row.get("nclusters"))
    if clusters is not None and clusters < 20:
        low = high = None
    if low is not None and high is not None:
        ax.plot((low, high), (y, y), color=color, linewidth=2.2, zorder=2)
        ax.plot((low, high), (y, y), color=INK, marker="|", markersize=8, linestyle="none", zorder=3)
    ax.plot(value, y, marker=marker, markersize=8, color=color, markeredgecolor=INK,
            markeredgewidth=.7, linestyle="none", zorder=4)
    return True


def _strength_associations(plt, rows, scopes, destination):
    from matplotlib.lines import Line2D

    index = {(r["dataset"], r["model"], r["fit"]): r for r in rows}
    models = list(MODELS_COMPLETE)
    fig, axes = plt.subplots(3, 2, figsize=(24, 23))
    fig.subplots_adjust(left=.15, right=.98, top=.84, bottom=.14, wspace=.48, hspace=.38)
    for ax, (dataset, name) in zip(axes.flat, DATASETS):
        _axes(ax, forest=True)
        scope = scopes.get(dataset, {})
        ax.set_title(name, fontsize=21, color=INK, fontweight="bold", pad=13)
        ax.set_yticks(range(len(models)), [MODELS_COMPLETE[model][0] for model in models], fontsize=15)
        ax.invert_yaxis()
        unit = "per listener-rating point" if scope.get("scale_kind") == "continuous" else "per native ordinal step"
        available = False
        for y, model in enumerate(models):
            color = MODELS_COMPLETE[model][1]
            for delta, fit, marker in ((-.16, "class_balanced", "o"),
                                       (.16, "class_balanced_duration_adjusted", "s")):
                row = index.get((dataset, model, fit))
                if row is not None:
                    unit = row.get("unit", unit)
                    available |= _forest_point(ax, row, y + delta, color, marker=marker)
        ax.set_xlabel(f"Accuracy change (pp {unit})", fontsize=16, color=INK)
        if not available:
            ax.text(.5, .5, "Association not identifiable", transform=ax.transAxes,
                    ha="center", va="center", fontsize=16, color=INK)
    fig.suptitle("Is stronger annotated emotion associated with easier classification?",
                 fontsize=26, fontweight="bold", y=.98)
    fig.text(.5, .94, "Non-neutral clips; class-balanced models. Squares additionally adjust for clip duration.",
             ha="center", fontsize=19, color=INK)
    fig.text(.5, .90, "Native scales differ: coefficients belong to their own dataset and are never pooled.",
             ha="center", fontsize=18, color=INK)
    handles = [Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=8, label="Class balanced"),
               Line2D([0], [0], marker="s", color=INK, linestyle="none", markersize=8,
                      label="Class balanced + duration adjustment")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.57, .066), ncol=2,
               frameon=False, fontsize=17, handletextpad=.4, columnspacing=2)
    fig.text(.15, .022, "Intervals are withheld with fewer than 20 source clusters.\n"
             "DailyTalk: 93 dialogues but only 2 actors; intervals are conditional, not evidence for new speakers.\n"
             "Positive coefficients concern classification accuracy, not the model's ability to estimate emotion intensity.",
             fontsize=16, color=INK, linespacing=1.4)
    return _save(fig, plt, destination, "emotion_intensity_associations")


def _strength_matched(plt, rows, destination):
    available_keys = {r["dataset"] for r in rows}
    datasets = [(key, name) for key, name in DATASETS if key in available_keys]
    if not datasets:
        return []
    columns = 2
    nrows = math.ceil(len(datasets) / columns)
    models = list(MODELS_COMPLETE)
    index = {(r["dataset"], r["model"]): r for r in rows}
    fig, axes = plt.subplots(nrows, columns, figsize=(24, 6.3 * nrows + 4.2), squeeze=False)
    fig.subplots_adjust(left=.15, right=.98, top=.83, bottom=.15, wspace=.48, hspace=.32)
    for ax, (dataset, name) in zip(axes.flat, datasets):
        n_pairs = sorted({int(r["n_pairs"]) for r in rows if r["dataset"] == dataset})
        size = f"{n_pairs[0]:,}" if len(n_pairs) == 1 else f"{min(n_pairs):,}–{max(n_pairs):,}"
        ax.set_title(f"{name}: {size} matched pairs", fontsize=21, fontweight="bold", color=INK, pad=13)
        if dataset == "mead" and max(n_pairs) < 20:
            ax.set_axis_off()
            support = f"Only {size} take keys" + (" / 2 actors" if max(n_pairs) == 3 else "")
            ax.text(.5, .55, support + ";\ninsufficient support.\nTranscript match unverified.",
                    transform=ax.transAxes, ha="center", va="center", color=INK, fontsize=21, linespacing=1.5)
            continue
        _axes(ax, forest=True)
        ax.set_yticks(range(len(models)), [MODELS_COMPLETE[model][0] for model in models], fontsize=15)
        ax.invert_yaxis()
        for y, model in enumerate(models):
            row = index.get((dataset, model))
            if row is not None:
                _forest_point(ax, row, y, MODELS_COMPLETE[model][1], value_key="difference_pp")
        ax.set_xlabel("High − low strength: accuracy difference (percentage points)", fontsize=16, color=INK)
    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)
    fig.suptitle("Emotion classification on matched higher- and lower-strength recordings",
                 fontsize=25, fontweight="bold", y=.98)
    fig.text(.5, .935, "Positive values mean more correct emotion labels at the higher annotated strength.",
             ha="center", fontsize=19, color=INK)
    fig.text(.5, .89, "Matching rules differ; MEAD text equivalence is unverified. Associations, not interventions.",
             ha="center", fontsize=18, color=INK)
    fig.text(.15, .035, "Native strength definitions and matching rules differ across datasets; estimates are shown separately.\n"
             "This analysis tests emotion-classification sensitivity, not prediction of a strength label.",
             fontsize=16, color=INK, linespacing=1.4)
    return _save(fig, plt, destination, "emotion_intensity_matched")


def plot_intensity(level_rows, effect_rows, matched_rows, scope_rows, output_dir):
    """Export native-scale intensity diagnostics as PNG/SVG figure pairs.

    Level accuracies are fractions; regression coefficients and matched
    differences are percentage points. The caller computes all statistics and
    reference-label/strength restrictions. Plotting performs no fitting or I/O
    on source audio, metadata, predictions or model weights.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    scopes = {row["dataset"]: row for row in scope_rows}
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Helvetica Neue", "Helvetica", "Arial") if name in fonts), "DejaVu Sans")
    with plt.rc_context({"font.family": family, "font.size": 15, "axes.titlesize": 18,
                         "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 16, "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "svg.fonttype": "none",
                         "svg.hashsalt": "emotion-intensity-native-v1", "savefig.facecolor": "white"}):
        paths = _strength_accuracy(plt, level_rows, scopes, destination)
        paths += _strength_associations(plt, effect_rows, scopes, destination)
        paths += _strength_matched(plt, matched_rows, destination)
    return paths

"""Plot saved cross-dataset diagnostics without reading predictions or audio."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path


DATASETS = (
    ("meld", "MELD"), ("emotiontalk", "EmotionTalk"),
    ("dailytalk", "DailyTalk"), ("multidialog", "MultiDialog"),
    ("crema_d", "CREMA-D"), ("ravdess", "RAVDESS"),
    ("savee", "SAVEE"), ("jl_corpus", "JL-Corpus"),
    ("tess", "TESS"), ("mead", "MEAD"), ("mess", "MESS"),
    ("mcae_spps", "MCAE-SPPS"), ("thai_ser", "THAI-SER"),
    ("nemo", "nEMO"), ("emovo", "EMOVO"),
    ("emozionalmente", "Emozionalmente"),
)
MODELS = {
    "emotion2vec_plus_large": ("emotion2vec+ large", "#0072B2"),
    "meralion_ser_v1": ("MERaLiON-SER-v1", "#E69F00"),
    "melati_emotion": ("Melati", "#009E73"),
    "vox_profile_whisper": ("Vox-Profile Whisper", "#CC79A7"),
    "vox_profile_wavlm": ("Vox-Profile WavLM", "#56B4E9"),
    "qwen2_5_omni_7b": ("Qwen2.5-Omni", "#D55E00"),
    "qwen2_audio_7b": ("Qwen2-Audio", "#F0E442"),
    "gemma4_e4b": ("Gemma 4", "#665191"),
    "gpt_realtime": ("GPT Realtime", "#332288"),
    "gemini-3.8-flash": ("Gemini 3.8 Flash", "#AA4499"),
}
BINS = ("0–1", "1–2", "2–3", "3–5", "5–10", "10–15", "15–30", "≥30")
EDGES = (0, 1, 2, 3, 5, 10, 15, 30, math.inf)
INK, GRID = "#000000", "#dddddd"


def _number(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _save(fig, plt, destination, name):
    paths = [destination / f"{name}.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            fig.savefig(path, dpi=220, facecolor="white", metadata={"Creator": "cross_dataset_plot.py"})
    finally:
        plt.close(fig)
    return paths


def _axes(ax, *, horizontal=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(axis="both", colors=INK, labelcolor=INK)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, linewidth=.7, zorder=0)


def _duration_distributions(plt, inventory, datasets, destination):
    counts = defaultdict(lambda: [0] * len(BINS))
    for row in inventory:
        duration = _number(row.get("duration_seconds"))
        if duration is None or duration < 0:
            raise ValueError("Audio inventory durations must be finite and nonnegative")
        at = next(i for i in range(len(BINS)) if EDGES[i] <= duration < EDGES[i + 1])
        counts[row["dataset"]][at] += 1
    fig, axes = plt.subplots(4, 4, figsize=(23, 18.5), sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.10, top=.82, wspace=.16, hspace=.50)
    for index, (ax, (key, name)) in enumerate(zip(axes.flat, DATASETS)):
        meta = datasets.get(key, {})
        values = counts[key]
        total = sum(values)
        median = _number(meta.get("median_seconds"))
        suffix = f"median {median:.2f} s; n = {total:,}" if median is not None else f"n = {total:,}"
        ax.set_title(f"{name}\n{suffix}", fontsize=17, color=INK, fontweight="bold", pad=10)
        _axes(ax)
        ax.set_ylim(0, 115)
        ax.set_yticks((0, 25, 50, 75, 100))
        if index % 4:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(range(len(BINS)), BINS, rotation=45, ha="right")
        if not total:
            ax.text(.5, .5, "No audio inventory", transform=ax.transAxes, ha="center", color=INK, fontsize=15)
            continue
        shares = [100 * value / total for value in values]
        color = "#D55E00" if meta.get("conversation") in (True, "True", "true", 1) else "#0072B2"
        ax.bar(range(len(BINS)), shares, width=.76, color=color, edgecolor=INK, linewidth=.7, zorder=3)
        for i, (count, share) in enumerate(zip(values, shares)):
            if count:
                ax.text(i, share + 1.6, "<1" if share < 1 else f"{share:.0f}",
                        ha="center", va="bottom", color=INK, fontsize=15)
    fig.suptitle("How long are the evaluated audio clips?", fontsize=25, fontweight="bold", y=.98)
    fig.text(.5, .93, "Physical file duration, measured from audio headers on the exact evaluated clips",
             ha="center", color=INK, fontsize=18)
    fig.text(.5, .895, "Orange: conversational datasets. Blue: other datasets. Every dataset contributes one panel.",
             ha="center", color=INK, fontsize=17)
    fig.supxlabel("Clip duration (seconds; bins have different widths)", fontsize=20, y=.057)
    fig.supylabel("Share of evaluated clips (%)", fontsize=20, x=.014)
    fig.text(.065, .022, "Heights show percentages per bin, not probability density. Labels round to whole percentages.",
             color=INK, fontsize=15)
    return _save(fig, plt, destination, "duration_distributions")


def _language_context(plt, rows, destination):
    from matplotlib.patches import Patch

    index = {(r["model"], r["grouping"], r["level"]): r for r in rows}
    models = [model for model in MODELS if model != "gemma4_e4b" and any(key[0] == model for key in index)]
    panels = (("context", "conversational", "non_conversational", "Conversational (4 datasets)", "Other (12 datasets)"),
              ("language", "English", "non_English", "English (10 datasets)", "Non-English (6 datasets)"))
    fig, axes = plt.subplots(3, 2, figsize=(24, 24))
    fig.subplots_adjust(left=.16, right=.975, bottom=.13, top=.84, wspace=.48, hspace=.42)
    for at, (metric, title) in enumerate((("accuracy", "Accuracy"), ("uar", "UAR"), ("macro_f1", "Macro-F1"))):
        for column, (grouping, first, second, first_label, second_label) in enumerate(panels):
            ax = axes[at, column]
            _axes(ax, horizontal=True)
            ax.set_xlim(0, 110)
            ax.set_xticks((0, 25, 50, 75, 100))
            ax.set_yticks(range(len(models)), [MODELS[m][0] for m in models], fontsize=16)
            ax.invert_yaxis()
            ax.set_title(f"{title}: {'conversation type' if column == 0 else 'language'}", fontsize=21,
                         color=INK, fontweight="bold", pad=48)
            for y, model in enumerate(models):
                color = MODELS[model][1]
                for delta, level, hatch, alpha in ((-.22, first, None, .95), (.22, second, "///", .40)):
                    row = index.get((model, grouping, level))
                    value = _number(row.get(metric)) if row else None
                    if value is None:
                        continue
                    ax.barh(y + delta, 100 * value, height=.38, color=color, edgecolor=INK,
                            hatch=hatch, linewidth=.65, alpha=alpha, zorder=3)
                    ax.text(100 * value + 1.1, y + delta, f"{100 * value:.1f}", ha="left", va="center",
                            fontsize=15, color=INK)
            ax.set_xlabel("Mean dataset score (%)", fontsize=18, color=INK)
            handles = [Patch(facecolor="#0072B2", edgecolor=INK, label=first_label),
                       Patch(facecolor="#0072B2", edgecolor=INK, alpha=.40, hatch="///", label=second_label)]
            ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2,
                      frameon=False, fontsize=15, borderaxespad=0, columnspacing=1.2)
    fig.suptitle("Performance changes with the dataset group", fontsize=27, fontweight="bold", y=.98)
    fig.text(.5, .945, "Every dataset has equal weight; panels show the same nine complete models.",
             ha="center", fontsize=20, color=INK)
    fig.text(.16, .025, "These are descriptive comparisons, not causal effects of language or conversation.\n"
             "Gemma 4 is excluded here because only its MELD result is available. Class distributions and recording sources differ.",
             fontsize=17, color=INK, linespacing=1.5)
    return _save(fig, plt, destination, "language_context_performance")


def _duration_effects(plt, rows, destination):
    from matplotlib.lines import Line2D

    selected = {(r["model"], r["fit"]): r for r in rows
                if r.get("cluster_kind") == "dataset" and r.get("fit") in ("pooled", "balanced_within_dataset_gold")}
    models = [model for model in MODELS if any(key[0] == model for key in selected)]
    fig, ax = plt.subplots(figsize=(17, 12))
    fig.subplots_adjust(left=.23, right=.95, top=.81, bottom=.26)
    _axes(ax, horizontal=True)
    ax.axvline(0, color=INK, linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(range(len(models)), [MODELS[m][0] + (" (MELD only)" if m == "gemma4_e4b" else "")
                                     for m in models], fontsize=17)
    ax.invert_yaxis()
    for y, model in enumerate(models):
        for offset, fit, marker in ((-.15, "pooled", "o"), (.15, "balanced_within_dataset_gold", "s")):
            row = selected.get((model, fit))
            slope = _number(row.get("slope_pp")) if row else None
            if slope is None:
                continue
            low, high = (_number(row.get(key)) for key in ("ci_lower_pp", "ci_upper_pp"))
            color = MODELS[model][1]
            if low is not None and high is not None:
                ax.plot((low, high), (y + offset, y + offset), color=color, linewidth=2.5, zorder=2)
                ax.plot((low, high), (y + offset, y + offset), marker="|", color=INK,
                        markersize=9, linestyle="none", zorder=3)
            ax.plot(slope, y + offset, marker=marker, color=color, markeredgecolor=INK,
                    markeredgewidth=.8, markersize=9, linestyle="none", zorder=4)
    ax.set_xlabel("Accuracy change per doubling of original clip duration (percentage points)", fontsize=19, color=INK)
    handles = [Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=9, label="Pooled: no dataset/class adjustment"),
               Line2D([0], [0], marker="s", color=INK, linestyle="none", markersize=9,
                      label="Within dataset × gold class; dataset/class balanced")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.57, .09), fontsize=16,
               frameon=False, ncol=1)
    fig.suptitle("Does longer audio predict higher accuracy?", fontsize=27, fontweight="bold", y=.98)
    fig.text(.5, .925, "Compare the raw association with an analysis controlling dataset and reference emotion.",
             ha="center", fontsize=18, color=INK)
    fig.text(.5, .88, "Intervals cluster by dataset. Gemma has one dataset and no across-dataset interval.",
             ha="center", fontsize=17, color=INK)
    fig.text(.23, .032, "Positive = accuracy rises with duration; negative = it falls. These are observational associations.\n"
             "Balanced estimates give equal weight to each dataset and to each gold class within it.",
             fontsize=16, color=INK, linespacing=1.5)
    return _save(fig, plt, destination, "duration_effects")


def _within_dataset(plt, rows, destination):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    index = {(r["dataset"], r["model"]): _number(r.get("slope_pp")) for r in rows}
    models = list(MODELS)
    values = np.array([[index.get((key, model), np.nan) if index.get((key, model)) is not None else np.nan
                        for model in models] for key, _ in DATASETS], dtype=float)
    finite = np.abs(values[np.isfinite(values)])
    bound = max(5, math.ceil(float(finite.max()) / 5) * 5) if len(finite) else 5
    cmap = LinearSegmentedColormap.from_list("duration_light", ["#9ecae1", "#ffffff", "#fdae6b"])
    cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(22, 16.8))
    fig.subplots_adjust(left=.14, right=.90, top=.80, bottom=.11)
    plotted = ax.imshow(values, cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), aspect="auto")
    headers = {"emotion2vec_plus_large": "emotion2vec+\nlarge", "meralion_ser_v1": "MERaLiON-\nSER-v1",
               "vox_profile_whisper": "Vox-Profile\nWhisper", "vox_profile_wavlm": "Vox-Profile\nWavLM",
               "qwen2_5_omni_7b": "Qwen2.5-\nOmni", "qwen2_audio_7b": "Qwen2-\nAudio",
               "gpt_realtime": "GPT\nRealtime", "gemini-3.8-flash": "Gemini 3.8\nFlash"}
    ax.set_xticks(range(len(models)), [headers.get(m, MODELS[m][0]) for m in models], fontsize=15)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=12, colors=INK)
    ax.set_yticks(range(len(DATASETS)), [name for _, name in DATASETS], fontsize=17)
    ax.tick_params(axis="y", length=0, pad=10, colors=INK)
    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(DATASETS), 1), minor=True)
    ax.grid(which="minor", color=INK, linewidth=.45)
    ax.tick_params(which="minor", length=0)
    for y in range(len(DATASETS)):
        for x in range(len(models)):
            value = values[y, x]
            ax.text(x, y, f"{value:+.1f}" if np.isfinite(value) else "n/a", ha="center", va="center",
                    color=INK, fontsize=16)
    colorbar = fig.colorbar(plotted, ax=ax, fraction=.030, pad=.025)
    colorbar.set_label("Accuracy change per duration doubling (pp)", color=INK, fontsize=18, labelpad=15)
    colorbar.ax.tick_params(labelsize=15, colors=INK)
    fig.suptitle("Duration associations within each dataset", fontsize=26, fontweight="bold", y=.98)
    fig.text(.5, .93, "Reference-emotion adjustment and equal class weights; values are percentage points per doubling.",
             ha="center", fontsize=18, color=INK)
    fig.text(.5, .885, "Blue: lower accuracy on longer clips. Orange: higher accuracy. n/a: unavailable or unidentifiable.",
             ha="center", fontsize=17, color=INK)
    fig.text(.14, .03, "Each cell uses its own dataset; this does not compare longer and shorter datasets.\n"
             "Cells are point estimates, not significance tests. Gemma 4 is available on MELD only.\n"
             "Slopes are local: narrow duration ranges can yield large slopes. Do not extrapolate.",
             fontsize=16, color=INK, linespacing=1.5)
    return _save(fig, plt, destination, "duration_within_dataset_slopes")


def _accuracy_by_duration(plt, rows, destination):
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    index = {(r["dataset"], r["model"], int(r["bin_id"])): r for r in rows}
    fig, axes = plt.subplots(4, 4, figsize=(23, 18), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.99, bottom=.18, top=.84, wspace=.16, hspace=.34)
    for at, (ax, (key, name)) in enumerate(zip(axes.flat, DATASETS)):
        _axes(ax)
        ax.set_title(name, color=INK, fontsize=18, fontweight="bold", pad=11)
        ax.set_ylim(0, 105)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_xticks(range(len(BINS)), BINS, rotation=45, ha="right")
        ax.set_xlim(-.35, len(BINS) - .65)
        if at % 4:
            ax.tick_params(axis="y", labelleft=False)
        for model, (_label, color) in MODELS.items():
            values = []
            for bin_id in range(len(BINS)):
                row = index.get((key, model, bin_id))
                value = _number(row.get("accuracy")) if row else None
                values.append(100 * value if row and int(row.get("n", 0)) >= 50 and value is not None else math.nan)
            if not any(math.isfinite(value) for value in values):
                continue
            line, = ax.plot(range(len(BINS)), values, color=color, marker="o", linewidth=1.8,
                            markersize=5, markeredgecolor=INK, markeredgewidth=.45, zorder=3)
            if model == "qwen2_audio_7b":
                line.set_path_effects([path_effects.Stroke(linewidth=2.8, foreground=INK), path_effects.Normal()])
    fig.suptitle("Observed accuracy by clip-duration bin", fontsize=26, fontweight="bold", y=.98)
    fig.text(.5, .935, "Only bins with at least 50 clips are shown; missing bins break the line.",
             ha="center", fontsize=18, color=INK)
    fig.text(.5, .905, "Raw bin accuracy can change because class composition changes. Use the adjusted slopes for that control.",
             ha="center", fontsize=17, color=INK)
    handles = [Line2D([0], [0], color=color, marker="o", markeredgecolor=INK, linewidth=2, markersize=6,
                      label=label + (" (MELD only)" if model == "gemma4_e4b" else ""))
               for model, (label, color) in MODELS.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .075), ncol=5,
               frameon=False, fontsize=15, columnspacing=1.3, handlelength=1.8)
    fig.supxlabel("Original clip duration (seconds; categorical bins)", fontsize=20, y=.135)
    fig.supylabel("Accuracy in each duration bin (%)", fontsize=20, x=.014)
    fig.text(.065, .025, "Lines connect observed bins; no model is rerun and no smooth duration-response curve is assumed.",
             fontsize=16, color=INK)
    return _save(fig, plt, destination, "accuracy_by_duration")


def plot_analysis(inventory, dataset_rows, group_rows, bin_rows, effect_rows, dataset_effect_rows, output_dir):
    """Export five PNG/SVG figure pairs from the caller's computed diagnostics.

    All input rates are fractions. Effect slopes and bounds are percentage points
    per doubling of the original duration. This function only plots supplied
    estimates; it never fits models or reads source artifacts.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Helvetica Neue", "Helvetica", "Arial") if name in fonts), "DejaVu Sans")
    datasets = {row["dataset"]: row for row in dataset_rows}
    with plt.rc_context({"font.family": family, "font.size": 15, "axes.titlesize": 18,
                         "axes.labelsize": 19, "xtick.labelsize": 15, "ytick.labelsize": 15,
                         "legend.fontsize": 15, "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "svg.fonttype": "none",
                         "svg.hashsalt": "cross-dataset-analysis-v1", "savefig.facecolor": "white"}):
        paths = _duration_distributions(plt, inventory, datasets, destination)
        paths += _language_context(plt, group_rows, destination)
        paths += _duration_effects(plt, effect_rows, destination)
        paths += _within_dataset(plt, dataset_effect_rows, destination)
        paths += _accuracy_by_duration(plt, bin_rows, destination)
    return paths

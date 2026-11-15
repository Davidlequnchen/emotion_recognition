"""Figures of the native-intensity analysis, styled like the benchmark figures.

Single source of the intensity figures in analysis_report/figures/ (PNG at 300 dpi plus SVG). The same files are
used by the paper report and by research_paper/presentation/intensity/. Model colours, order and names follow the
benchmark figures (emotion_recognition/scripts/viewer/plot_results.py with config/models.yaml); fonts are sized for
a slide at the figure's own size (smallest text 16 pt). Figures carry no titles, subtitles or notes: captions,
slide bullets and speaker notes explain them.

  emotion_intensity_change        heat map, model x dataset: observed accuracy change, weakest -> strongest level
  emotion_intensity_associations  heat map, same layout: class- and duration-adjusted change; white = interval has 0
  emotion_intensity_matched       heat map, same layout: same actor/sentence/emotion, strong minus weak take
  emotion_intensity_accuracy      accuracy at every native level, one panel per dataset
  emotion_intensity_neutral       share of emotional clips answered "neutral", weakest vs strongest level
  emotion_intensity_forest        adjusted per-step slopes with 95% intervals (secondary)

All statistics come from intensity_analysis.py (contrast_rows, level_rows, effect_rows); plotting fits nothing.
Re-draw from the saved analysis without re-reading predictions (from the repository root):
    python research_paper/scripts/intensity_plot.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path


DATASETS = (
    ("crema_d", "CREMA-D"), ("ravdess", "RAVDESS"), ("mead", "MEAD"),
    ("thai_ser", "THAI-SER"), ("mcae_spps", "MCAE-SPPS"), ("dailytalk", "DailyTalk"),
)
# Benchmark order, names and colours: config/models.yaml order and names, plot_results.py COLOURS
# (checked by tests/test_intensity_plot.py).
MODELS = {
    "emotion2vec_plus_large": ("emotion2vec+ large", "#0072B2"),
    "meralion_ser_v1": ("MERaLiON-SER-v1", "#E69F00"),
    "melati_emotion": ("Melati", "#009E73"),
    "vox_profile_whisper": ("Vox-Profile Whisper", "#CC79A7"),
    "vox_profile_wavlm": ("Vox-Profile WavLM", "#56B4E9"),
    "qwen2_5_omni_7b": ("Qwen2.5-Omni 7B", "#D55E00"),
    "qwen2_audio_7b": ("Qwen2-Audio 7B Instruct", "#F0E442"),
    "gemma4_e4b": ("Gemma 4 E4B", "#999999"),
    "gpt_realtime": ("GPT Realtime", "#332288"),
    "gemini-3.8-flash": ("Gemini 3.8 Flash", "#AA4499"),
}
# Results explicitly designated as an unfinished re-run. All current runs are final;
# absence from one intensity scale alone does not make a completed model preliminary.
PRELIMINARY = set()
RATED = {"mcae_spps", "dailytalk"}   # intensity rated by listeners/annotators; the others are acted levels
INK, EDGE, GRID, PENDING_FILL = "#111111", "#444444", "#d9d9d9", "#eef0f3"
LIMIT = 40          # colour scale of all change heat maps: -40 ... +40 percentage points (values beyond are clipped)
MATCHED_MIN_PAIRS = 20
DPI = 300


def _number(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _label(model, present):
    name = MODELS[model][0]
    if model in PRELIMINARY:
        return name + (" (prelim.)" if model in present else " (pending)")
    return name


def _save(fig, plt, folder, name):
    paths = [folder / f"{name}.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            fig.savefig(path, dpi=DPI if path.suffix == ".png" else None, bbox_inches="tight", pad_inches=0.04,
                        facecolor="white", metadata={"Creator": "intensity_plot.py"} if path.suffix == ".png"
                        else {"Creator": "intensity_plot.py", "Date": None})
    finally:
        plt.close(fig)
    return paths


def _spines(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(EDGE)
    ax.tick_params(axis="both", colors=EDGE, labelcolor=INK)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def _signed(value, digits=0):
    text = f"{value:+.{digits}f}"
    if text.lstrip("+-").strip("0.") == "":
        return "0"
    return text.replace("-", "−")


def _text_colour(cmap, norm, value):
    import numpy as np
    rgb = np.array(cmap(norm(value))[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    luminance = float(linear @ [.2126, .7152, .0722])
    return "white" if luminance < .30 else INK


def _change_cmap(plt):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    # Diverging, centred at 0: blue = more correct at the stronger level, red = fewer. Endpoints trimmed.
    return LinearSegmentedColormap.from_list("intensity_change", plt.get_cmap("RdBu")(np.linspace(.06, .94, 256)))


def _heatmap(plt, models, present, datasets, cells, *, uncertainty, destination, name, bar_label):
    """cells[(model, dataset)] = (value, certain); certain=False draws a white cell with the value in brackets.

    Columns are grouped into acted levels and rated intensity (listener or annotator ratings), with a small gap.
    """
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    cmap, norm = _change_cmap(plt), Normalize(-LIMIT, LIMIT)
    column, row, gap, label_w, head_h, right_w = 1.20, 0.34, 0.14, 2.95, 0.74, 1.58
    groups = [[d for d in datasets if d[0] not in RATED], [d for d in datasets if d[0] in RATED]]
    groups = [g for g in groups if g]
    x0, positions = 0.0, {}
    for g, group in enumerate(groups):
        for dataset, _ in group:
            positions[dataset] = x0
            x0 += column
        x0 += gap
    cells_w = x0 - gap
    width, height = label_w + cells_w + right_w, head_h + row * len(models) + 0.04
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((label_w / width, 0.04 / height, cells_w / width, row * len(models) / height))
    ax.set_xlim(0, cells_w)
    ax.set_ylim(len(models), 0)
    for i, model in enumerate(models):
        for dataset, _ in datasets:
            x = positions[dataset]
            value, certain = cells.get((model, dataset), (None, False))
            if model not in present:
                fill, text, colour = PENDING_FILL, "\u2014", INK
            elif value is None:
                fill, text, colour = "white", "", INK
            elif certain:
                fill, text, colour = cmap(norm(value)), _signed(value), _text_colour(cmap, norm, value)
            else:
                fill, text, colour = "white", f"({_signed(value)})", INK
            ax.add_patch(Rectangle((x, i), column, 1, facecolor=fill, edgecolor=GRID, linewidth=1.0))
            ax.text(x + column / 2, i + .5, text, ha="center", va="center", fontsize=16, color=colour)
    ax.set_xticks([positions[d] + column / 2 for d, _ in datasets], [label for _, label in datasets], fontsize=16)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4)
    ax.set_yticks([i + .5 for i in range(len(models))], [_label(m, present) for m in models], fontsize=16)
    ax.tick_params(axis="y", length=0, pad=25)
    for i, model in enumerate(models):   # model colour key, as in the benchmark legend (x data units are inches)
        ax.add_patch(Rectangle((-.29, i + .2), .19, .6, facecolor=MODELS[model][1], edgecolor=INK,
                               linewidth=.8, clip_on=False))
    for group in groups:
        start, stop = positions[group[0][0]], positions[group[-1][0]] + column
        title = "Rated intensity" if group[0][0] in RATED else "Acted levels"
        ax.annotate(title, xy=((start + stop) / 2, 0), xytext=(0, 33), textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, fontweight="bold", color=INK, annotation_clip=False)
        ax.annotate("", xy=(start + .04, 0), xytext=(stop - .04, 0), textcoords="data", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": EDGE, "linewidth": 1.2,
                                "shrinkA": 0, "shrinkB": 0, "connectionstyle": "arc3"})
    ax.spines[:].set_visible(False)
    bar_x = (label_w + cells_w + .22) / width
    bar_bottom = (0.04 + row * len(models) * (.42 if uncertainty else 0)) / height
    bar_height = row * len(models) * (.58 if uncertainty else 1.0) / height
    bar = fig.add_axes((bar_x, bar_bottom, .2 / width, bar_height))
    colourbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=bar, orientation="vertical",
                             extend="both", ticks=range(-LIMIT, LIMIT + 1, 20))
    colourbar.ax.tick_params(labelsize=16, colors=EDGE, labelcolor=INK, length=3)
    colourbar.ax.set_yticklabels([_signed(t) for t in range(-LIMIT, LIMIT + 1, 20)])
    colourbar.set_label(bar_label, fontsize=16, color=INK, labelpad=4)
    colourbar.outline.set_edgecolor(EDGE)
    colourbar.solids.set_rasterized(False)
    if uncertainty:
        key = fig.add_axes((bar_x - .04 / width, .04 / height, (right_w - .2) / width, row * len(models) * .30 / height))
        key.set_axis_off()
        key.set_xlim(0, right_w - .2)
        key.set_ylim(0, 1)
        key.add_patch(Rectangle((0, .66), .62, .30, facecolor="white", edgecolor=EDGE, linewidth=1.0))
        key.text(.31, .81, "( )", ha="center", va="center", fontsize=16, color=INK)
        key.text(0, .58, "CI includes 0\nor no CI", ha="left", va="top", fontsize=16, color=INK, linespacing=1.1)
    return _save(fig, plt, destination, name)


def _certain(low, high):
    low, high = _number(low), _number(high)
    return low is not None and high is not None and (low > 0 or high < 0)


def _change_maps(plt, models, present, contrasts, destination):
    paths = []
    observed = {(r["dataset"], r["model"]): r for r in contrasts}
    cells = {(m, d): (_number(r.get("observed_change_pp")), True) for (d, m), r in observed.items()}
    paths += _heatmap(plt, models, present, DATASETS, cells, uncertainty=False, destination=destination,
                      name="emotion_intensity_change", bar_label="Accuracy change (pp)")
    cells = {(m, d): (_number(r.get("adjusted_change_pp")), _certain(r.get("adjusted_ci_lower_pp"),
                                                                   r.get("adjusted_ci_upper_pp")))
             for (d, m), r in observed.items()}
    paths += _heatmap(plt, models, present, DATASETS, cells, uncertainty=True, destination=destination,
                      name="emotion_intensity_associations", bar_label="Adjusted change (pp)")
    pairs = {}
    for row in contrasts:
        if _number(row.get("matched_pairs")) is not None:
            pairs[row["dataset"]] = max(pairs.get(row["dataset"], 0), int(float(row["matched_pairs"])))
    matched_sets = tuple((key, name) for key, name in DATASETS if pairs.get(key, 0) >= MATCHED_MIN_PAIRS)
    if matched_sets:
        cells = {(m, d): (_number(r.get("matched_difference_pp")), _certain(r.get("matched_ci_lower_pp"),
                                                                          r.get("matched_ci_upper_pp")))
                 for (d, m), r in observed.items() if d in dict(matched_sets)}
        paths += _heatmap(plt, models, present, matched_sets, cells, uncertainty=True, destination=destination,
                          name="emotion_intensity_matched", bar_label="Strong − weak take (pp)")
    return paths


def _legend_handles(models, present, *, line=True):
    from matplotlib.lines import Line2D
    handles = []
    for model in models:
        colour = MODELS[model][1]
        missing = model not in present
        handles.append(Line2D([0], [0], color=colour if line and not missing else "none", marker="o",
                              markersize=8, markerfacecolor="white" if missing else colour,
                              markeredgecolor=colour if missing else INK, markeredgewidth=1.2 if missing else .7,
                              linewidth=2.2, label=_label(model, present)))
    return handles


def _levels(level_rows, dataset, min_n):
    """Displayed strata of a dataset in native order: [(level_id, label)], each with at least min_n clips."""
    found = {}
    for row in level_rows:
        if row["dataset"] == dataset and int(row["n"]) >= min_n:
            found[int(row["level_id"])] = str(row["level_label"]).replace("ECSS level ", "level ")
    return sorted(found.items())


def _accuracy(plt, models, present, level_rows, min_n, destination):
    import matplotlib.patheffects as path_effects

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.3), sharey=True)
    fig.subplots_adjust(left=.088, right=.995, top=.93, bottom=.08, wspace=.10, hspace=.56)
    for k, (ax, (dataset, name)) in enumerate(zip(axes.flat, DATASETS)):
        _spines(ax)
        levels = _levels(level_rows, dataset, min_n)
        ids = [level for level, _ in levels]
        ax.set_xticks(range(len(levels)), [label for _, label in levels], fontsize=16)
        ax.set_xlim(-.3, len(levels) - .7)
        ax.set_ylim(0, 100)
        ax.set_yticks((0, 25, 50, 75, 100))
        ax.set_title(name, fontsize=18, fontweight="bold", color=INK, pad=6)
        for model in models:
            rows = {int(r["level_id"]): r for r in level_rows if r["dataset"] == dataset and r["model"] == model}
            values = [100 * float(rows[i]["accuracy"]) if i in rows and rows[i]["accuracy"] not in (None, "")
                      else math.nan for i in ids]
            if not any(math.isfinite(v) for v in values):
                continue
            line, = ax.plot(range(len(ids)), values, color=MODELS[model][1], marker="o", markersize=7,
                            markeredgecolor=INK, markeredgewidth=.6, linewidth=2.2, zorder=3)
            if model == "qwen2_audio_7b":     # the pale yellow line needs a dark outline on white
                line.set_path_effects([path_effects.Stroke(linewidth=3.6, foreground=INK), path_effects.Normal()])
        if k % 3 == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=17, color=INK)
    fig.legend(handles=_legend_handles(models, present), loc="center left", bbox_to_anchor=(1.0, .5),
               frameon=False, fontsize=16, handlelength=1.8, labelspacing=.55, borderaxespad=0)
    return _save(fig, plt, destination, "emotion_intensity_accuracy")


def _neutral(plt, models, present, contrasts, destination):
    from matplotlib.lines import Line2D

    index = {(r["dataset"], r["model"]): r for r in contrasts}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.0, 4.1), sharey=True)
    fig.subplots_adjust(left=.215, right=.985, top=.908, bottom=.255, wspace=.24)
    for k, (ax, (dataset, name)) in enumerate(zip(axes, DATASETS)):
        _spines(ax, grid_axis="x")
        ax.set_title(name, fontsize=18, fontweight="bold", color=INK, pad=6)
        ax.set_xlim(-6, 106)
        ax.set_xticks((0, 50, 100))
        ax.set_ylim(len(models) - .5, -.5)
        for i, model in enumerate(models):
            row = index.get((dataset, model))
            low = _number(row.get("low_neutral_rate")) if row else None
            high = _number(row.get("high_neutral_rate")) if row else None
            if low is None or high is None:
                continue
            colour = MODELS[model][1]
            ax.plot((100 * low, 100 * high), (i, i), color=colour, linewidth=2.6, zorder=2,
                    solid_capstyle="round")
            ax.plot(100 * low, i, marker="o", markersize=9, markerfacecolor="white", markeredgecolor=INK,
                    markeredgewidth=1.3, linestyle="none", zorder=3)
            ax.plot(100 * high, i, marker="o", markersize=9, markerfacecolor=colour, markeredgecolor=INK,
                    markeredgewidth=.8, linestyle="none", zorder=4)
        if k == 0:
            ax.set_yticks(range(len(models)), [_label(m, present) for m in models], fontsize=16)
            ax.tick_params(axis="y", length=0)
    fig.supxlabel("Emotional clips answered “neutral” (%)", fontsize=17, color=INK, x=.6, y=.10)
    handles = [Line2D([0], [0], marker="o", markersize=9, markerfacecolor="white", markeredgecolor=INK,
                      markeredgewidth=1.3, linestyle="none", label="weakest level"),
               Line2D([0], [0], marker="o", markersize=9, markerfacecolor=INK, markeredgecolor=INK,
                      linestyle="none", label="strongest level")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.6, -.005), ncol=2, frameon=False,
               fontsize=16, columnspacing=2.0, handletextpad=.3)
    return _save(fig, plt, destination, "emotion_intensity_neutral")


def _forest(plt, models, present, effect_rows, destination):
    from matplotlib.ticker import MaxNLocator

    index = {(r["dataset"], r["model"]): r for r in effect_rows if r["fit"] == "class_balanced_duration_adjusted"}
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(12.0, 4.2), sharey=True)
    fig.subplots_adjust(left=.215, right=.985, top=.91, bottom=.17, wspace=.24)
    for k, (ax, (dataset, name)) in enumerate(zip(axes, DATASETS)):
        _spines(ax, grid_axis="x")
        ax.axvline(0, color=INK, linewidth=1.0, linestyle="--", zorder=1)
        ax.set_title(name, fontsize=18, fontweight="bold", color=INK, pad=6)
        ax.set_ylim(len(models) - .5, -.5)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        unit = "pp / step"
        for i, model in enumerate(models):
            row = index.get((dataset, model))
            value = _number(row.get("slope_pp")) if row else None
            if value is None:
                continue
            if "listener" in str(row.get("unit", "")):
                unit = "pp / point"
            low, high = _number(row.get("ci_lower_pp")), _number(row.get("ci_upper_pp"))
            if (_number(row.get("nclusters")) or 0) >= 20 and low is not None and high is not None:
                ax.plot((low, high), (i, i), color=MODELS[model][1], linewidth=2.4, zorder=2)
                ax.plot((low, high), (i, i), color=INK, marker="|", markersize=9, linestyle="none", zorder=3)
            ax.plot(value, i, marker="o", markersize=8.5, color=MODELS[model][1], markeredgecolor=INK,
                    markeredgewidth=.8, linestyle="none", zorder=4)
        ax.set_xlabel(unit, fontsize=16, color=INK, labelpad=2)
        if k == 0:
            ax.set_yticks(range(len(models)), [_label(m, present) for m in models], fontsize=16)
            ax.tick_params(axis="y", length=0)
    return _save(fig, plt, destination, "emotion_intensity_forest")


def plot_intensity(data, output_dir):
    """Export the intensity figures as PNG (300 dpi) / SVG pairs; returns the written paths.

    `data` is the analysis dictionary of intensity_analysis.py (or its saved JSON): level accuracies and rates are
    fractions; contrast, effect and matched values are percentage points. No statistics are computed here.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    level_rows, contrasts = data["level_rows"], data.get("contrast_rows", [])
    present = {row["model"] for row in level_rows}
    models = [m for m in MODELS] + sorted(present - set(MODELS))
    for model in models:
        MODELS.setdefault(model, (model, "#777777"))
    min_n = int(data.get("manifest", {}).get("parameters", {}).get("display_min_level_n", 20))
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Helvetica Neue", "Helvetica", "Arial") if name in fonts), "DejaVu Sans")
    with plt.rc_context({"font.family": family, "font.size": 16, "axes.titlesize": 18, "axes.titleweight": "bold",
                         "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                         "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": EDGE,
                         "ytick.color": EDGE, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.edgecolor": EDGE, "axes.linewidth": 1.0, "svg.fonttype": "none",
                         "svg.hashsalt": "emotion-intensity-native-v2", "savefig.facecolor": "white"}):
        paths = _change_maps(plt, models, present, contrasts, destination)
        paths += _accuracy(plt, models, present, level_rows, min_n, destination)
        paths += _neutral(plt, models, present, contrasts, destination)
        paths += _forest(plt, models, present, data["effect_rows"], destination)
    return paths


if __name__ == "__main__":
    paper = Path(__file__).resolve().parents[1]
    saved = json.loads((paper / "output" / "intensity_analysis.json").read_text(encoding="utf-8"))
    for path in plot_intensity(saved, paper / "analysis_report" / "figures"):
        print(path.relative_to(paper))

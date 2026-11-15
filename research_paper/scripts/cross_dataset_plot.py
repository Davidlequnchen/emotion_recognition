"""Plot saved cross-dataset diagnostics without reading predictions or audio.

The figures are drawn at slide size with every font at 16 pt or larger, so one PNG (300 dpi) and SVG
serve the Markdown reports, a paper (scaled to the column width) and the slide deck in
research_paper/presentation/cross_dataset/. Figures carry no titles, subtitles or notes: the report
text, paper captions and slide bullets explain them. Model colours, display names and the dataset
order match the benchmark figures (emotion_recognition/scripts/viewer/plot_results.py with
config/plots.yaml and config/models.yaml).
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path


# Benchmark dataset order: conversational first, then English acted, then the other languages.
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
# Registry order and display names (config/models.yaml) with the benchmark colours (plot_results.COLOURS).
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
# Display bins: the analysis bins 10–15, 15–30 and >=30 s are pooled into one >=10 s bin (3.1% of all clips),
# so the edge labels stay readable at 16 pt in small panels. The analysis CSVs keep all eight bins.
BINS = ("0–1", "1–2", "2–3", "3–5", "5–10", "≥10")
EDGES = (0, 1, 2, 3, 5, 10, math.inf)
ANALYSIS_TO_DISPLAY = (0, 1, 2, 3, 4, 5, 5, 5)
EDGE_LABELS = ("1", "2", "3", "5", "10")
MIN_BIN_N = 50                                   # bins with fewer clips are not drawn
LABEL_CAP = 50                                   # heat-map cell labels show ≥50 / ≤−50 beyond this (pp)
INK, GRID, MISSING = "#000000", "#d9d9d9", "#e8e8e8"
CONVERSATIONAL, ACTED = "#D55E00", "#0072B2"     # setting colours of the duration panels
# Diverging scale for signed changes: blue = negative, white = 0, orange = positive.
DIVERGING = ("#08519c", "#3f8fc9", "#a9cfe8", "#ffffff", "#fdc896", "#f08a3e", "#a63603")
FS, FS_AXIS = 16, 17                             # minimum font size; axis labels


def _number(value):
    if value in (None, ""):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _save(fig, plt, destination, name):
    paths = [destination / f"{name}.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            png = path.suffix == ".png"
            fig.savefig(path, dpi=300 if png else None, facecolor="white", bbox_inches="tight", pad_inches=.04,
                        metadata={"Creator": "cross_dataset_plot.py"} if png else {"Creator": "cross_dataset_plot.py", "Date": None})
    finally:
        plt.close(fig)
    return paths


def _axes(ax, *, horizontal=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.tick_params(axis="both", colors=INK, labelcolor=INK, labelsize=FS)
    (ax.xaxis if horizontal else ax.yaxis).grid(True, color=GRID, linewidth=.7, zorder=0)
    ax.set_axisbelow(True)


def _coverage(dataset_effect_rows):
    coverage = defaultdict(list)
    for row in dataset_effect_rows:
        coverage[row["model"]].append(row["dataset"])
    return coverage


def _label(model, coverage=None):
    """Display name; a model with partial coverage names its datasets (or counts them)."""
    name = MODELS.get(model, (model, None))[0]
    present = (coverage or {}).get(model)
    if present is None or len(present) >= len(DATASETS):
        return name
    names = dict(DATASETS)
    return f"{name} ({', '.join(names.get(d, d) for d in present)} only)" if len(present) <= 2 else \
        f"{name} ({len(present)}/{len(DATASETS)} datasets)"


def _diverging(bound):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("cross_dataset_diverging", DIVERGING)
    return cmap, TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)


def _ink_on(rgb):
    """White or black text, whichever has the higher WCAG contrast on the cell colour."""
    def linear(c):
        return c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4
    luminance = .2126 * linear(rgb[0]) + .7152 * linear(rgb[1]) + .0722 * linear(rgb[2])
    return "white" if 1.05 / (luminance + .05) > (luminance + .05) / .05 else INK


def _signed(value, decimals=0):
    text = f"{value:+.{decimals}f}"
    if float(text) == 0:
        text = f"{0:.{decimals}f}"
    return text.replace("-", "−")


def _duration_bin(duration):
    return next(i for i in range(len(BINS)) if EDGES[i] <= duration < EDGES[i + 1])


def _bin_position(seconds):
    """Median position on the categorical bin axis, interpolated linearly inside its bin."""
    at = _duration_bin(seconds)
    if math.isinf(EDGES[at + 1]):
        return at
    return at - .5 + (seconds - EDGES[at]) / (EDGES[at + 1] - EDGES[at])


def _bin_axis(ax, bottom):
    ax.set_xlim(-.6, len(BINS) - .4)
    ax.set_xticks([i + .5 for i in range(len(EDGE_LABELS))])
    ax.set_xticklabels(EDGE_LABELS if bottom else [])
    ax.tick_params(axis="x", length=4)


def _duration_distributions(plt, inventory, datasets, destination):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    counts = defaultdict(lambda: [0] * len(BINS))
    for row in inventory:
        duration = _number(row.get("duration_seconds"))
        if duration is None or duration < 0:
            raise ValueError("Audio inventory durations must be finite and nonnegative")
        counts[row["dataset"]][_duration_bin(duration)] += 1
    fig, axes = plt.subplots(4, 4, figsize=(8.3, 5.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.995, bottom=.13, top=.89, wspace=.07, hspace=.58)
    for index, (ax, (key, name)) in enumerate(zip(axes.flat, DATASETS)):
        meta = datasets.get(key, {})
        values = counts[key]
        total = sum(values)
        _axes(ax)
        ax.set_title(name, fontsize=FS, fontweight="bold", color=INK, pad=3)
        ax.set_ylim(0, 100)
        ax.set_yticks((0, 50, 100))
        _bin_axis(ax, index >= 12)
        if not total:
            ax.text(.5, .5, "no audio", transform=ax.transAxes, ha="center", va="center", color=INK, fontsize=FS)
            continue
        conversational = meta.get("conversation") in (True, "True", "true", 1)
        ax.bar(range(len(BINS)), [100 * value / total for value in values], width=.78,
               color=CONVERSATIONAL if conversational else ACTED, linewidth=0, zorder=3)
        median = _number(meta.get("median_seconds"))
        if median is not None:
            ax.axvline(_bin_position(median), color=INK, linewidth=1.5, zorder=4)
    handles = [Patch(facecolor=CONVERSATIONAL, label="Conversational"), Patch(facecolor=ACTED, label="Acted"),
               Line2D([0], [0], color=INK, linewidth=1.5, label="Median")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, 1.005), ncol=3, frameon=False,
               fontsize=FS, handlelength=1.3, columnspacing=1.6, borderaxespad=0)
    fig.supxlabel("Clip duration (s)", fontsize=FS_AXIS, y=.012)
    fig.supylabel("Share of clips (%)", fontsize=FS_AXIS, x=.005)
    return _save(fig, plt, destination, "duration_distributions")


def _group_performance(plt, rows, grouping, levels, headers, destination, name):
    """Equal-dataset group means (Accuracy, UAR) and the difference between the two groups, per model.

    Geometry is in inches (data units = inches), so cells keep their size when models are added.
    """
    from matplotlib.patches import Rectangle

    index = {(r["model"], r["grouping"], r["level"]): r for r in rows}
    models = [m for m in MODELS if all((m, grouping, level) in index for level in levels)]
    if not models:
        return []
    metrics = (("accuracy", "Accuracy (%)"), ("uar", "UAR (%)"))
    # Rows shrink a little for a tenth model, so the figure keeps its slide size (fonts stay at 16 pt on the slide).
    label_width, level_width, gap_width, block_gap, row_height = 2.45, .86, .78, .22, min(.44, 3.95 / len(models))
    lines = max(label.count("\n") for label in headers.values())
    header = .85 + .25 * lines
    columns, x = [], 0.                           # (left edge, width, metric, kind)
    for metric, _ in metrics:
        for kind, width in ((levels[0], level_width), (levels[1], level_width), ("gap", gap_width)):
            columns.append((x, width, metric, kind))
            x += width
        x += block_gap
    content = x - block_gap
    cmap, norm = _diverging(30)
    width, height = label_width + content + .03, header + row_height * len(models) + .05
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=label_width / width, right=1 - .03 / width, top=1 - header / height, bottom=.05 / height)
    for y, model in enumerate(models):
        top = y * row_height
        first, second = (index[(model, grouping, level)] for level in levels)
        for left, cell, metric, kind in columns:
            # Δ is the difference of the displayed one-decimal means, so every row adds up as printed.
            a, b = round(100 * float(first[metric]), 1), round(100 * float(second[metric]), 1)
            if kind == "gap":
                value = round(a - b, 1)
                rgb = cmap(norm(max(-30, min(30, value))))
                ax.add_patch(Rectangle((left, top), cell, row_height, facecolor=rgb, edgecolor="white", linewidth=1.5))
                ax.text(left + cell / 2, top + row_height / 2, _signed(value, 1), ha="center", va="center",
                        fontsize=FS, color=_ink_on(rgb[:3]), fontweight="bold")
            else:
                ax.add_patch(Rectangle((left, top), cell, row_height, facecolor="white", edgecolor=GRID, linewidth=.8))
                ax.text(left + cell / 2, top + row_height / 2, f"{a if kind == levels[0] else b:.1f}",
                        ha="center", va="center", fontsize=FS, color=INK)
        ax.add_patch(Rectangle((-.17, top + .04), .1, row_height - .08, facecolor=MODELS[model][1],
                               edgecolor="none", clip_on=False))
    ax.set_xlim(0, content)
    ax.set_ylim(row_height * len(models), 0)
    ax.set_yticks([(y + .5) * row_height for y in range(len(models))], [MODELS[m][0] for m in models], fontsize=FS)
    ax.tick_params(axis="y", length=0, pad=19, colors=INK)
    ax.set_xticks([left + cell / 2 for left, cell, _, _ in columns], [headers[kind] for *_, kind in columns],
                  fontsize=FS)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4, colors=INK)
    ax.spines[:].set_visible(False)
    for block, (_, label) in enumerate(metrics):
        left = columns[3 * block][0]
        right = columns[3 * block + 2][0] + columns[3 * block + 2][1]
        ax.annotate(label, xy=((left + right) / 2, 0), xytext=(0, 34 + 18 * lines), textcoords="offset points",
                    ha="center", va="bottom", fontsize=FS, fontweight="bold", color=INK, annotation_clip=False)
        ax.annotate("", xy=(left + .04, 0), xytext=(right - .04, 0), textcoords="data", annotation_clip=False,
                    arrowprops={"arrowstyle": "-", "color": INK, "linewidth": 1.2, "shrinkA": 0, "shrinkB": 0})
    return _save(fig, plt, destination, name)


def _duration_effects(plt, rows, destination):
    """Per model: the slope over all clips pooled next to the slope within each dataset (primary fit)."""
    selected = {(r["model"], r["fit"]): r for r in rows
                if r.get("cluster_kind") == "dataset" and r.get("fit") in ("pooled", "balanced_within_dataset_gold")}
    # Only models with an across-dataset interval: one dataset has no between-dataset contrast to pool.
    models = [m for m in MODELS if all(_number(selected.get((m, f), {}).get("ci_lower_pp")) is not None
                                       for f in ("pooled", "balanced_within_dataset_gold"))]
    if not models:
        return []
    bounds = [abs(_number(selected[(m, f)][k])) for m in models for f in ("pooled", "balanced_within_dataset_gold")
              for k in ("ci_lower_pp", "ci_upper_pp")]
    bound = math.ceil(max(bounds) + 1.5)
    height = min(.43, 3.85 / len(models)) * len(models) + .95
    fig, axes = plt.subplots(1, 2, figsize=(7.4, height), sharey=True)
    fig.subplots_adjust(left=.34, right=.985, top=1 - .42 / height, bottom=.72 / height, wspace=.09)
    for ax, (fit, title) in zip(axes, (("pooled", "All clips pooled"),
                                        ("balanced_within_dataset_gold", "Within each dataset"))):
        _axes(ax, horizontal=True)
        ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
        ax.set_xlim(-bound, bound)
        step = 10 if bound > 12 else 5
        ticks = range(-step * (bound // step), step * (bound // step) + 1, step)
        ax.set_xticks(list(ticks), [_signed(t) for t in ticks])
        ax.set_title(title, fontsize=FS, fontweight="bold", color=INK, pad=6)
        for y, model in enumerate(models):
            row = selected[(model, fit)]
            slope, low, high = (_number(row[k]) for k in ("slope_pp", "ci_lower_pp", "ci_upper_pp"))
            ax.plot((low, high), (y, y), color=INK, linewidth=1.8, solid_capstyle="butt", zorder=2)
            ax.plot((low, high), (y, y), color=INK, marker="|", markersize=10, markeredgewidth=1.8,
                    linestyle="none", zorder=2)
            ax.plot(slope, y, marker="o", markersize=10, color=MODELS[model][1], markeredgecolor=INK,
                    markeredgewidth=1.1, linestyle="none", zorder=3)
    axes[0].set_yticks(range(len(models)), [MODELS[m][0] for m in models], fontsize=FS)
    axes[0].set_ylim(len(models) - .5, -.5)
    axes[0].tick_params(axis="y", length=0, pad=6)
    axes[1].tick_params(axis="y", length=0)
    fig.supxlabel("Accuracy change per doubling of clip duration (pp)", fontsize=FS, x=.66, y=.005)
    return _save(fig, plt, destination, "duration_effects")


def _within_dataset(plt, rows, destination):
    """Models × datasets heat map of the within-dataset slope; bold where the 95% interval excludes 0."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.patches import Rectangle

    index = {(r["dataset"], r["model"]): r for r in rows}
    models = [m for m in MODELS if any(key[1] == m for key in index)]
    bound = 20
    cmap, norm = _diverging(bound)
    width, height = 11.45, .34 * len(models) + 1.25
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=2.35 / width, right=1 - .95 / width, top=1 - 1.2 / height, bottom=.02)
    boundaries = []
    for x, (key, _) in enumerate(DATASETS):
        for y, model in enumerate(models):
            row = index.get((key, model))
            value = _number(row.get("slope_pp")) if row else None
            if value is None:
                ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=MISSING, edgecolor="white", linewidth=1.5))
                ax.text(x, y, "—", ha="center", va="center", fontsize=FS, color=INK)
                continue
            rgb = cmap(norm(max(-bound, min(bound, value))))
            ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, facecolor=rgb, edgecolor="white", linewidth=1.5))
            low, high = _number(row.get("ci_lower_pp")), _number(row.get("ci_upper_pp"))
            excludes = low is not None and high is not None and (low > 0 or high < 0)
            # Labels beyond ±50 pp (only RAVDESS, a 3.5–3.9 s corpus) are capped so they fit the cell at 16 pt.
            label = "≥50" if value >= LABEL_CAP else "≤−50" if value <= -LABEL_CAP else _signed(value)
            ax.text(x, y, label, ha="center", va="center", fontsize=FS, color=_ink_on(rgb[:3]),
                    fontweight="bold" if excludes else "normal")
    conversational = {key for key, _ in DATASETS[:4]}
    english_acted = {"crema_d", "ravdess", "savee", "jl_corpus", "tess", "mead", "mess"}
    for x in range(1, len(DATASETS)):
        groups = [DATASETS[x - 1][0], DATASETS[x][0]]
        if (groups[0] in conversational) != (groups[1] in conversational) or \
                (groups[0] in english_acted) != (groups[1] in english_acted):
            boundaries.append(x - .5)
    for x in boundaries:
        ax.plot((x, x), (-.5, len(models) - .5), color=INK, linewidth=2.2, solid_capstyle="butt", zorder=5)
    for y, model in enumerate(models):
        ax.add_patch(Rectangle((-.5 - .2, y - .42), .12, .84, facecolor=MODELS[model][1], edgecolor="none",
                               clip_on=False))
    ax.set_xlim(-.5, len(DATASETS) - .5)
    ax.set_ylim(len(models) - .5, -.5)
    ax.set_xticks(range(len(DATASETS)), [name for _, name in DATASETS], fontsize=FS, rotation=40,
                  ha="left", rotation_mode="anchor")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4, colors=INK)
    ax.set_yticks(range(len(models)), [MODELS[m][0] for m in models], fontsize=FS)
    ax.tick_params(axis="y", length=0, pad=16, colors=INK)
    ax.spines[:].set_visible(False)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, ax=ax, fraction=.022, pad=.012, aspect=14)
    colorbar.set_ticks([-20, -10, 0, 10, 20])
    colorbar.set_ticklabels(["≤ −20", "−10", "0", "+10", "≥ +20"])
    colorbar.ax.tick_params(labelsize=FS, colors=INK, length=3)
    colorbar.outline.set_edgecolor(INK)
    colorbar.solids.set_rasterized(False)
    colorbar.set_label("pp per doubling", fontsize=FS, color=INK, labelpad=6)
    return _save(fig, plt, destination, "duration_within_dataset_slopes")


def _accuracy_by_duration(plt, rows, destination, coverage):
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as path_effects

    pooled = defaultdict(lambda: [0, 0])
    for row in rows:
        at = ANALYSIS_TO_DISPLAY[int(row["bin_id"])]
        pooled[(row["dataset"], row["model"], at)][0] += int(row.get("n") or 0)
        pooled[(row["dataset"], row["model"], at)][1] += int(row.get("correct") or 0)
    models = [m for m in MODELS if any(key[1] == m for key in pooled)]
    fig, axes = plt.subplots(4, 4, figsize=(11.45, 5.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.765, bottom=.12, top=.955, wspace=.07, hspace=.5)
    for at, (ax, (key, name)) in enumerate(zip(axes.flat, DATASETS)):
        _axes(ax)
        ax.set_title(name, color=INK, fontsize=FS, fontweight="bold", pad=3)
        ax.set_ylim(0, 100)
        ax.set_yticks((0, 50, 100))
        _bin_axis(ax, at >= 12)
        for model in models:
            values = []
            for bin_id in range(len(BINS)):
                n, correct = pooled.get((key, model, bin_id), (0, 0))
                values.append(100 * correct / n if n >= MIN_BIN_N else math.nan)
            if not any(math.isfinite(value) for value in values):
                continue
            line, = ax.plot(range(len(BINS)), values, color=MODELS[model][1], marker="o", linewidth=2,
                            markersize=4.5, markeredgecolor=INK, markeredgewidth=.4, zorder=3)
            if model == "qwen2_audio_7b":            # yellow needs an outline on white
                line.set_path_effects([path_effects.Stroke(linewidth=3.2, foreground=INK), path_effects.Normal()])
    handles = []
    for model in models:
        handle = Line2D([0], [0], color=MODELS[model][1], marker="o", markeredgecolor=INK, linewidth=2.4,
                        markersize=7, label=_label(model, coverage).replace(" (", "\n("))
        if model == "qwen2_audio_7b":
            handle.set_path_effects([path_effects.Stroke(linewidth=3.6, foreground=INK), path_effects.Normal()])
        handles.append(handle)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(.775, .54), ncol=1, frameon=False,
               fontsize=FS, handlelength=1.6, handletextpad=.5, labelspacing=.42, borderaxespad=0)
    fig.supxlabel("Clip duration (s)", fontsize=FS_AXIS, x=.42, y=.005)
    fig.supylabel("Accuracy (%)", fontsize=FS_AXIS, x=.0)
    return _save(fig, plt, destination, "accuracy_by_duration")


def plot_analysis(inventory, dataset_rows, group_rows, bin_rows, effect_rows, dataset_effect_rows, output_dir):
    """Export the cross-dataset PNG (300 dpi) / SVG figure pairs from the caller's computed diagnostics.

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
    coverage = _coverage(dataset_effect_rows)
    with plt.rc_context({"font.family": family, "font.size": FS, "axes.titlesize": FS,
                         "axes.labelsize": FS_AXIS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS, "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": INK, "axes.linewidth": 1.0, "svg.fonttype": "none",
                         "svg.hashsalt": "cross-dataset-analysis-v2", "savefig.facecolor": "white"}):
        paths = _duration_distributions(plt, inventory, datasets, destination)
        paths += _group_performance(plt, group_rows, "context", ("conversational", "non_conversational"),
                                    {"conversational": "Conv.", "non_conversational": "Acted", "gap": "Δ"},
                                    destination, "performance_by_setting")
        paths += _group_performance(plt, group_rows, "language", ("English", "non_English"),
                                    {"English": "English", "non_English": "Non-\nEnglish", "gap": "Δ"},
                                    destination, "performance_by_language")
        paths += _duration_effects(plt, effect_rows, destination)
        paths += _within_dataset(plt, dataset_effect_rows, destination)
        paths += _accuracy_by_duration(plt, bin_rows, destination, coverage)
    return paths

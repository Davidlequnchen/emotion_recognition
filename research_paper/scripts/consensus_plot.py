"""Paper and slide figures of the model-consensus analysis; reads saved threshold rows only (no models or APIs).

plot_thresholds(rows, folder, pending)  paper figures, PNG (300 dpi) and SVG, in the benchmark's style:
    consensus_tradeoff_by_dataset    accuracy vs coverage of the four main groups, one panel per dataset (4 x 4)
    consensus_meld_tradeoff          MELD, identical cohort, every vote threshold labelled
    consensus_unanimous_by_dataset   heat map: unanimous accuracy and coverage per dataset and main group
    consensus_tradeoff_extra_groups  the 4 x 4 grid for the appendix groups (Selected3, SER+API7, AudioLLM4)
plot_slides(rows, class_rows, folder, pending)  the same content drawn at its size on a 16:9 slide (PNG, 300 dpi,
    no text under 16 pt), split into 2 x 4 grids, plus a CREMA-D example and an appendix class-share heat map.

Coverage = accepted / common clips; accuracy of agreed labels = correct accepted / accepted (undefined, never 0%,
when nothing is accepted). Each group keeps one colour (Okabe-Ito, colour-blind safe) and one marker shape.
Hollow markers and dashed lines: the group voted without some members on that dataset (`pending`, e.g. Gemma 4);
the heat maps add * to those values. No titles, subtitles or footnotes inside the figures: captions, the reports
and the slides explain them.
"""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

# Dataset order, names and languages of the benchmark figures (emotion_recognition/config/plots.yaml)
DATASETS = (
    ("meld", "MELD", "English"), ("emotiontalk", "EmotionTalk", "Mandarin"),
    ("dailytalk", "DailyTalk", "English"), ("multidialog", "MultiDialog", "English"),
    ("crema_d", "CREMA-D", "English"), ("ravdess", "RAVDESS", "English"),
    ("savee", "SAVEE", "English"), ("jl_corpus", "JL-Corpus", "English"),
    ("tess", "TESS", "English"), ("mead", "MEAD", "English"),
    ("mess", "MESS", "English"), ("mcae_spps", "MCAE-SPPS", "Mandarin"),
    ("thai_ser", "THAI-SER", "Thai"), ("nemo", "nEMO", "Polish"),
    ("emovo", "EMOVO", "Italian"), ("emozionalmente", "Emozionalmente", "Italian"),
)
# One colour and marker per group in every figure and slide: label, colour, marker, members (paper legend)
GROUPS = {
    "SER5": ("SER5", "#0072B2", "o", "five SER models"),
    "LLM3": ("LLM3", "#E69F00", "^", "three open audio LLMs"),
    "AudioLLM5": ("AudioLLM5", "#009E73", "D", "LLM3 + GPT Realtime + Gemini"),
    "All10": ("All10", "#CC79A7", "s", "all ten models"),
    "Selected3": ("Selected3", "#D55E00", "v", "emotion2vec+, Gemini, Qwen2.5-Omni"),
    "SER_API7": ("SER+API7", "#56B4E9", "p", "SER5 + GPT Realtime + Gemini"),
    "AudioLLM4": ("AudioLLM4", "#332288", "h", "GPT Realtime, Gemini, Qwen2.5-Omni, Qwen2-Audio"),
}
MAIN = ("SER5", "LLM3", "AudioLLM5", "All10")
EXTRA = ("Selected3", "SER_API7", "AudioLLM4")
X_LABEL, Y_LABEL = "Coverage (% of clips labelled)", "Accuracy of agreed labels (%)"
GRID_COLOUR = "#d9d9d9"   # grid lines only, never text
# Heat-map fills: light yellow-green-blue (darker = higher); dark text stays readable on every cell
HEAT = ("#FFFFFF", "#EDF8B1", "#C7E9B4", "#7FCDBB")
# Paper: benchmark look (Helvetica Neue, black text). Slide: the deck's Calibri and ink, drawn at slide size.
PAPER = {"fonts": ("Helvetica Neue", "Helvetica", "Arial"), "ink": "#000000", "title": 18, "tick": 16,
         "label": 21, "legend": 17, "annot": 16, "cell": 16, "marker": 8.5, "line": 2.2}
SLIDE = {"fonts": ("Calibri", "Helvetica Neue", "Arial"), "ink": "#1B2A41", "title": 17, "tick": 16,
         "label": 18, "legend": 17, "annot": 16, "cell": 16, "marker": 7.5, "line": 2.0}
CALIBRI = Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts")


def _rate(value):
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Consensus plot rates must be finite fractions on [0, 1]")
    return number


def _flag(value):
    return value in (True, "True", "true", 1, "1")


def _records(rows):
    """(dataset, group) -> threshold rows sorted by votes; accepts analysis dictionaries or CSV rows."""
    result, seen = defaultdict(list), set()
    for original in rows:
        group = original.get("group")
        if group not in GROUPS:
            continue
        m, threshold = int(original["model_count"]), int(original["min_votes"])
        if str(original["min_votes"]) not in (str(threshold), str(float(threshold))):
            raise ValueError("Consensus vote thresholds must be integers")
        if not m // 2 < threshold <= m:
            raise ValueError("Consensus vote thresholds must be strict-majority integers")
        key = original["dataset"], group, threshold
        if key in seen:
            raise ValueError(f"Duplicated consensus threshold: {key}")
        seen.add(key)
        row = dict(original, min_votes=threshold, model_count=m, complete=_flag(original.get("complete", True)),
                   coverage=_rate(original.get("coverage")),
                   conditional_precision=_rate(original.get("conditional_precision")))
        for field in ("accepted", "accepted_correct", "accepted_errors", "common_samples"):
            if original.get(field) not in (None, ""):
                row[field] = int(original[field])
        result[original["dataset"], group].append(row)
    return {key: sorted(value, key=lambda row: row["min_votes"]) for key, value in result.items()}


def vote_label(row):
    k, m = row["min_votes"], row["model_count"]
    return f"{k}/{m}" if k == m else f"≥{k}/{m}"


def _setup(plt, profile):
    """rc settings: the first installed font of the profile, dark text only, editable SVG text."""
    from matplotlib import font_manager
    if "Calibri" in profile["fonts"] and CALIBRI.is_dir():
        for name in ("Calibri.ttf", "Calibrib.ttf"):
            if (CALIBRI / name).is_file():
                font_manager.fontManager.addfont(str(CALIBRI / name))
    installed = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in profile["fonts"] if name in installed), "DejaVu Sans")
    ink = profile["ink"]
    return plt.rc_context({
        "font.family": family, "font.size": profile["tick"], "axes.titlesize": profile["title"],
        "axes.labelsize": profile["label"], "xtick.labelsize": profile["tick"], "ytick.labelsize": profile["tick"],
        "legend.fontsize": profile["legend"], "text.color": ink, "axes.labelcolor": ink, "axes.titlecolor": ink,
        "xtick.color": ink, "ytick.color": ink, "axes.edgecolor": ink, "axes.linewidth": 1.0,
        "svg.fonttype": "none", "svg.hashsalt": "emotion-consensus-v3", "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig, plt, destination, name, formats):
    paths = [destination / f"{name}.{suffix}" for suffix in formats]
    try:
        for path in paths:
            fig.savefig(path, dpi=300, facecolor="white",
                        metadata={"Creator": "consensus_plot.py"} if path.suffix == ".svg" else {"Software": None})
    finally:
        plt.close(fig)
    return paths


def _partial(rows, group, dataset, pending):
    return any(not row["complete"] for row in rows) or (group, dataset) in pending


def _draw(ax, rows, group, profile, *, partial):
    """One group: line through its vote thresholds; hollow markers and a dashed line if members are missing."""
    _, colour, marker, _ = GROUPS[group]
    usable = [r for r in rows if r["coverage"] is not None and r["conditional_precision"] is not None]
    xs = [100 * r["coverage"] for r in usable]
    ys = [100 * r["conditional_precision"] for r in usable]
    if not usable:
        return usable
    ax.plot(xs, ys, color=colour, linewidth=profile["line"], linestyle=(0, (3.5, 2)) if partial else "-",
            zorder=3, solid_capstyle="round")
    ax.plot(xs, ys, linestyle="none", marker=marker, markersize=profile["marker"],
            markerfacecolor="white" if partial else colour, markeredgecolor=colour if partial else profile["ink"],
            markeredgewidth=1.7 if partial else .8, zorder=4)
    return usable


def _style_axes(ax, profile, *, limits=(-3, 104), ticks=(0, 25, 50, 75, 100), ylimits=None, yticks=None):
    ax.set_xlim(*limits)
    ax.set_ylim(*(ylimits or limits))
    ax.set_xticks(ticks)
    ax.set_yticks(yticks or ticks)
    ax.grid(True, color=GRID_COLOUR, linewidth=.8, zorder=0)
    ax.tick_params(axis="both", length=3.5, pad=2, colors=profile["ink"])
    ax.spines[["top", "right"]].set_visible(False)


def _legend_handles(groups, records, pending, profile, *, describe):
    """Group entries (* when any panel lacks members) and one entry explaining hollow/dashed marks."""
    from matplotlib.lines import Line2D
    partial = {g for g in groups for (dataset, group), rows in records.items()
               if group == g and _partial(rows, g, dataset, pending)}
    handles = []
    for g in groups:
        label, colour, marker, members = GROUPS[g]
        text = label + ("*" if g in partial else "") + (f" ({members})" if describe else "")
        handles.append(Line2D([], [], color=colour, linewidth=profile["line"], marker=marker,
                              markersize=profile["marker"] + .5, markerfacecolor=colour,
                              markeredgecolor=profile["ink"], markeredgewidth=.8, label=text))
    if partial:
        names = sorted({name for (group, _), missing in pending.items() if group in partial for name in missing})
        who = ", ".join(names) if names else "Members"
        text = f"*{who} pending: voted with the available members" if describe else f"*{who} pending (available members)"
        handles.append(Line2D([], [], color=profile["ink"], linewidth=profile["line"] - .4, linestyle=(0, (3.5, 2)),
                              marker="o", markersize=profile["marker"] + .5, markerfacecolor="white",
                              markeredgecolor=profile["ink"], markeredgewidth=1.5, label=text))
    return handles


def _grid(plt, records, groups, datasets, profile, pending, *, size, columns, margins, gaps, legend_columns,
          describe, language=True, tick_step=25):
    """Accuracy vs coverage, one panel per dataset with shared axes; legend below the grid."""
    width, height = size
    left, right, top, bottom = margins
    rows_count = math.ceil(len(datasets) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=size, sharex=True, sharey=True, squeeze=False)
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height,
                        bottom=bottom / height, wspace=gaps[0], hspace=gaps[1])
    ticks = tuple(range(0, 101, tick_step))
    for ax, (dataset, name, lang) in zip(axes.flat, datasets):
        ax.set_title(f"{name} ({lang})" if language else name, fontweight="bold", pad=5)
        _style_axes(ax, profile, ticks=ticks)
        drawn = [_draw(ax, records.get((dataset, g), []), g, profile,
                       partial=_partial(records.get((dataset, g), []), g, dataset, pending)) for g in groups]
        if not any(drawn):
            ax.text(.5, .5, "No results yet", transform=ax.transAxes, ha="center", va="center",
                    fontsize=profile["annot"])
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)
    centre = (left + (width - right)) / 2 / width
    fig.supxlabel(X_LABEL, fontsize=profile["label"], x=centre, y=(bottom - .36) / height, va="top")
    fig.supylabel(Y_LABEL, fontsize=profile["label"], x=.1 / width, ha="left")
    shown = {key: rows for key, rows in records.items() if key[0] in {d for d, _, _ in datasets}}
    handles = _legend_handles(groups, shown, pending, profile, describe=describe)
    fig.legend(handles=handles, loc="lower center", ncol=min(legend_columns, len(handles)), frameon=False,
               bbox_to_anchor=(centre, .02 / height), borderaxespad=0, borderpad=.1,
               columnspacing=1.2, handlelength=2.0, handletextpad=.45, labelspacing=.35)
    return fig


def _annotate(ax, rows_by_group, groups, profile, *, fontsize):
    """Label every vote threshold next to its marker, avoiding markers, lines, other labels and the panel edge.

    The label box has a thin edge in the group's colour (the text stays dark) and a leader line to its point.
    """
    from matplotlib.transforms import Bbox
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = ax.get_window_extent(renderer).padded(-3)
    scale = fig.dpi / 72
    points, markers, lines = [], [], []
    for group, rows in zip(groups, rows_by_group):
        pixels = [ax.transData.transform((100 * r["coverage"], 100 * r["conditional_precision"])) for r in rows]
        points += [(group, row, px) for row, px in zip(rows, pixels)]
        markers += [Bbox.from_extents(x - 6 * scale, y - 6 * scale, x + 6 * scale, y + 6 * scale) for x, y in pixels]
        for (x0, y0), (x1, y1) in zip(pixels, pixels[1:]):
            steps = max(1, int(math.hypot(x1 - x0, y1 - y0) / (4 * scale)))
            lines += [Bbox.from_extents(x0 + (x1 - x0) * t / steps - 2 * scale, y0 + (y1 - y0) * t / steps - 2 * scale,
                                        x0 + (x1 - x0) * t / steps + 2 * scale, y0 + (y1 - y0) * t / steps + 2 * scale)
                      for t in range(steps + 1)]
    offsets = [(dx, dy) for radius in (14, 24, 36) for dx, dy in
               ((radius, 0), (-radius, 0), (0, radius), (0, -radius), (radius * .75, radius * .75),
                (-radius * .75, radius * .75), (radius * .75, -radius * .75), (-radius * .75, -radius * .75))]
    labels = []

    def overlap(box, others):
        return sum(max(0, min(box.x1, b.x1) - max(box.x0, b.x0)) * max(0, min(box.y1, b.y1) - max(box.y0, b.y0))
                   for b in others)
    for group, row, _ in sorted(points, key=lambda item: (item[2][0], -item[2][1])):
        x, y = 100 * row["coverage"], 100 * row["conditional_precision"]
        best = None
        for dx, dy in offsets:
            note = ax.annotate(vote_label(row), (x, y), xytext=(dx, dy), textcoords="offset points",
                               ha="center", va="center", fontsize=fontsize, zorder=6)
            note.update_positions(renderer)
            box = note.get_window_extent(renderer).padded(3 * scale)
            note.remove()
            outside = max(0, extent.x0 - box.x0) + max(0, box.x1 - extent.x1) + \
                max(0, extent.y0 - box.y0) + max(0, box.y1 - extent.y1)
            penalty = (3 * overlap(box, labels) + 2 * overlap(box, markers) + .6 * overlap(box, lines)
                       + 1e4 * outside + 4 * math.hypot(dx, dy) * scale)
            if best is None or penalty < best[0]:
                best = (penalty, dx, dy, Bbox.from_bounds(*box.bounds))
        _, dx, dy, box = best
        ax.annotate(vote_label(row), (x, y), xytext=(dx, dy), textcoords="offset points", ha="center",
                    va="center", fontsize=fontsize, color=profile["ink"], zorder=6,
                    bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": GROUPS[group][1],
                          "linewidth": 1.3},
                    arrowprops={"arrowstyle": "-", "color": GROUPS[group][1], "lw": 1.2, "shrinkA": 0,
                                "shrinkB": 4})
        labels.append(box)


def _meld(plt, records, groups, profile, pending, *, size, margins, legend_columns, describe, handlelength=2.0):
    """MELD, where all main groups are complete on the same clips: every threshold labelled."""
    width, height = size
    left, right, top, bottom = margins
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height)
    rows_by_group = []
    for g in groups:
        rows = records.get(("meld", g), [])
        rows_by_group.append(_draw(ax, rows, g, profile, partial=_partial(rows, g, "meld", pending)))
    values = [(100 * r["coverage"], 100 * r["conditional_precision"]) for rows in rows_by_group for r in rows]
    if values:
        xmax = min(100, 10 * math.ceil((max(x for x, _ in values) + 6) / 10))
        ylo = max(0, 5 * math.floor((min(y for _, y in values) - 3) / 5))
        yhi = min(104, 10 * math.ceil((max(y for _, y in values) + 4) / 10))
        _style_axes(ax, profile, limits=(0, xmax), ticks=tuple(range(0, xmax + 1, 10)),
                    ylimits=(ylo, yhi), yticks=tuple(range(10 * math.ceil(ylo / 10), yhi + 1, 10)))
        _annotate(ax, rows_by_group, groups, profile, fontsize=profile["annot"])
    else:
        _style_axes(ax, profile)
        ax.text(.5, .5, "No results yet", transform=ax.transAxes, ha="center", fontsize=profile["annot"])
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    meld_records = {key: rows for key, rows in records.items() if key[0] == "meld"}
    handles = _legend_handles(groups, meld_records, pending, profile, describe=describe)
    fig.legend(handles=handles, loc="lower center", ncol=min(legend_columns, len(handles)), frameon=False,
               bbox_to_anchor=(.5, .02 / height), borderaxespad=0, columnspacing=1.2, handlelength=handlelength,
               handletextpad=.5, labelspacing=.35)
    return fig


def _pick(rows, which):
    if which == "majority":
        return rows[0] if rows else None
    return next((r for r in rows if r["min_votes"] == r["model_count"]), None)


def _heat_colour(value, top):
    """Piecewise-linear light ramp; top is the value mapped to the darkest fill."""
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("consensus_heat", HEAT)
    return ramp(min(1.0, max(0.0, value / top)))


def _heatmap(plt, groups, datasets, profile, *, size, margins, blocks, language=True):
    """Annotated heat map. blocks: [(title, value of the darkest fill, getter(dataset, group) -> (value, partial))].

    value None = nothing accepted (accuracy undefined, shown as —); "n/a" = no result for that group.
    """
    from matplotlib.patches import Rectangle
    width, height = size
    left, right, top, bottom = margins
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([left / width, bottom / height, 1 - (left + right) / width, 1 - (top + bottom) / height])
    gap = .35
    ax.set_xlim(0, len(groups) * len(blocks) + gap * (len(blocks) - 1))
    ax.set_ylim(len(datasets), 0)
    ax.axis("off")
    row_inches = (height - top - bottom) / len(datasets)

    def above(inches):                                   # data y of a point this many inches above the table
        return -inches / row_inches
    cell, title = profile["cell"] / 72, profile["legend"] / 72
    bar = (.03, .10)
    label_y = above(bar[1] + .05 + .55 * cell)
    title_y = above(bar[1] + .05 + 1.1 * cell + .07 + .55 * title)
    for b, (heading, darkest, getter) in enumerate(blocks):
        x0 = b * (len(groups) + gap)
        ax.text(x0 + len(groups) / 2, title_y, heading, ha="center", va="center", fontsize=profile["legend"],
                fontweight="bold", clip_on=False)
        for j, g in enumerate(groups):
            label, colour, _, _ = GROUPS[g]
            values = [getter(d, g) for d, _, _ in datasets]
            ax.text(x0 + j + .5, label_y, label + ("*" if any(p for _, p in values) else ""), ha="center",
                    va="center", fontsize=profile["cell"], fontweight="bold", clip_on=False)
            ax.add_patch(Rectangle((x0 + j + .06, above(bar[1])), .88, above(bar[0]) - above(bar[1]),
                                   facecolor=colour, edgecolor="none", clip_on=False))
            for i, (value, partial) in enumerate(values):
                empty = value is None or value == "n/a"
                ax.add_patch(Rectangle((x0 + j, i), 1, 1, facecolor="#FFFFFF" if empty else
                                       _heat_colour(value, darkest), edgecolor="white", linewidth=1.5))
                text = ("n/a" if value == "n/a" else "—") if empty else f"{100 * value:.0f}" + ("*" if partial else "")
                ax.text(x0 + j + .5, i + .5, text, ha="center", va="center", fontsize=profile["cell"])
    for i, (_, name, lang) in enumerate(datasets):
        ax.text(-.12, i + .5, f"{name} ({lang})" if language else name, ha="right", va="center",
                fontsize=profile["cell"])
    return fig


def _unanimous_blocks(records, pending):
    def getter(metric):
        def get(dataset, group):
            rows = records.get((dataset, group), [])
            row = _pick(rows, "unanimous")
            if row is None:
                return "n/a", False
            return row[metric], _partial(rows, group, dataset, pending)
        return get
    return [("Accuracy of agreed labels (%)", 1.0, getter("conditional_precision")),
            ("Coverage (% of clips labelled)", .5, getter("coverage"))]


def plot_thresholds(threshold_rows, output_dir, pending=None):
    """Write the four paper figures (PNG 300 dpi + SVG); returns their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _records(threshold_rows)
    pending = dict(pending or {})
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    formats = ("png", "svg")
    paths = []
    with _setup(plt, PAPER):
        grid = {"size": (17.6, 12.9), "columns": 4, "margins": (1.0, .12, .42, 1.5), "gaps": (.07, .36),
                "legend_columns": 3, "describe": True}
        fig = _grid(plt, records, MAIN, DATASETS, PAPER, pending, **grid)
        paths += _save(fig, plt, destination, "consensus_tradeoff_by_dataset", formats)
        fig = _meld(plt, records, MAIN, PAPER, pending, size=(10.5, 7.6), margins=(1.05, .25, .2, 1.7),
                    legend_columns=2, describe=True)
        paths += _save(fig, plt, destination, "consensus_meld_tradeoff", formats)
        fig = _heatmap(plt, MAIN, DATASETS, PAPER, size=(13.4, 9.2), margins=(2.75, .1, .85, .05),
                       blocks=_unanimous_blocks(records, pending))
        paths += _save(fig, plt, destination, "consensus_unanimous_by_dataset", formats)
        fig = _grid(plt, records, EXTRA, DATASETS, PAPER, pending, **(grid | {"legend_columns": 2}))
        paths += _save(fig, plt, destination, "consensus_tradeoff_extra_groups", formats)
    return paths


def _concept(plt, records, profile, *, size):
    """Problem statement: one group on one dataset; stricter vote rules move left (fewer clips) and up."""
    from matplotlib.patches import FancyArrowPatch
    width, height = size
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.95 / width, right=1 - .2 / width, top=1 - .15 / height, bottom=.85 / height)
    usable = _draw(ax, records.get(("crema_d", "SER5"), []), "SER5", profile, partial=False)
    _style_axes(ax, profile, limits=(0, 100), ticks=(0, 25, 50, 75, 100), ylimits=(50, 103),
                yticks=(50, 60, 70, 80, 90, 100))
    for row in usable:
        ax.annotate(vote_label(row), (100 * row["coverage"], 100 * row["conditional_precision"]), xytext=(8, 10),
                    textcoords="offset points", fontsize=profile["label"], fontweight="bold", ha="left",
                    va="center", color=profile["ink"])
    if len(usable) >= 2:
        ax.add_patch(FancyArrowPatch((70, 62), (16, 86), connectionstyle="arc3,rad=-.2", arrowstyle="-|>",
                                     mutation_scale=20, color=profile["ink"], linewidth=1.6))
        ax.text(50, 60.5, "stricter vote rule", fontsize=profile["label"], ha="center", va="center")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    return fig


def _class_blocks(class_rows, label):
    """Share of unanimous labels predicted as `label`, per dataset and group (n/a if the class is absent)."""
    shares = {}
    for row in class_rows:
        if row["group"] in GROUPS and int(row["min_votes"]) == int(row["model_count"]) and \
                row["predicted_class"] == label:
            value = row.get("share_of_accepted")
            shares[row["dataset"], row["group"]] = (_rate(value), not _flag(row.get("complete", True)))

    def get(dataset, group):
        return shares.get((dataset, group), ("n/a", False))
    return [(f"Share of agreed labels that are {label} (%)", 1.0, get)]


def plot_slides(threshold_rows, class_rows, output_dir, pending=None):
    """Slide-size PNGs (300 dpi; text 16-18 pt as placed on a 13.33 x 7.5 in slide); returns their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = _records(threshold_rows)
    pending = dict(pending or {})
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    with _setup(plt, SLIDE):
        fig = _concept(plt, records, SLIDE, size=(6.6, 4.9))
        paths += _save(fig, plt, destination, "concept_crema_d", ("png",))
        grid = {"size": (12.13, 4.62), "columns": 4, "margins": (.78, .1, .3, 1.12), "gaps": (.13, .42),
                "legend_columns": 5, "describe": False, "tick_step": 50}
        for groups, stem in ((MAIN, "tradeoff_main"), (EXTRA, "tradeoff_extra")):
            for half in (0, 1):
                fig = _grid(plt, records, groups, DATASETS[8 * half:8 * half + 8], SLIDE, pending, **grid)
                paths += _save(fig, plt, destination, f"{stem}_{half + 1}", ("png",))
        fig = _meld(plt, records, MAIN, SLIDE, pending, size=(6.75, 5.2), margins=(.85, .15, .1, 1.18),
                    legend_columns=4, describe=False, handlelength=1.5)
        paths += _save(fig, plt, destination, "meld_tradeoff", ("png",))
        fig = _heatmap(plt, MAIN, DATASETS, SLIDE, size=(12.13, 5.0), margins=(2.45, .05, .74, .02),
                       blocks=_unanimous_blocks(records, pending))
        paths += _save(fig, plt, destination, "unanimous_heatmap", ("png",))
        fig = _heatmap(plt, MAIN, DATASETS, SLIDE, size=(7.4, 5.0), margins=(2.45, .05, .74, .02),
                       blocks=_class_blocks(class_rows, "neutral"))
        paths += _save(fig, plt, destination, "neutral_share_heatmap", ("png",))
    return paths

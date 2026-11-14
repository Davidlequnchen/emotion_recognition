"""Readable consensus plots and benchmark-style bars; no model or API calls."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path


# Keep the same dataset order, languages, palette and slim bars as the benchmark figures.
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
GROUPS = {
    "SER5": (5, "Five SER models", "#0072B2"),
    "Selected3": (3, "Selected mixed models", "#D55E00"),
    "SER_API7": (7, "SER + APIs", "#009E73"),
    "All10": (10, "All ten models", "#CC79A7"),
    "AudioLLM4": (4, "Four audio LLM/API models", "#332288"),
}
MELD_GROUPS = ("SER5", "Selected3", "SER_API7", "All10")
RULE_COLOURS = {3: "#56B4E9", 4: "#0072B2", 5: "#332288"}
AUDIO_RULE_COLOURS = {3: "#AA4499", 4: "#332288"}
INK, GRID = "#000000", "#d9d9d9"


def _rate(value):
    if value in (None, ""):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError("Consensus plot rates must be finite fractions on [0, 1]")
    return number


def _records(rows):
    """Accept analysis dictionaries or CSV dictionaries without changing their meaning."""
    result = defaultdict(list)
    seen = set()
    for original in rows:
        group = original.get("group")
        if group not in GROUPS:
            continue
        m = GROUPS[group][0]
        if int(original.get("model_count", 0)) != m:
            continue
        complete = original.get("complete", True)
        if complete not in (True, "True", "true", 1):
            continue
        threshold = int(original["min_votes"])
        if str(original["min_votes"]) not in (str(threshold), str(float(threshold))):
            raise ValueError("Consensus vote thresholds must be integers")
        if not m // 2 < threshold <= m:
            raise ValueError("Consensus vote thresholds must be strict-majority integers")
        key = original["dataset"], group, threshold
        if key in seen:
            raise ValueError(f"Duplicated consensus threshold: {key}")
        seen.add(key)
        row = dict(original)
        row.update(min_votes=threshold, model_count=m,
                   coverage=_rate(original.get("coverage")),
                   conditional_precision=_rate(original.get("conditional_precision")))
        for field in ("accepted", "accepted_correct", "accepted_errors", "common_samples"):
            if original.get(field) not in (None, ""):
                row[field] = int(original[field])
        result[original["dataset"], group].append(row)
    return {key: sorted(value, key=lambda row: row["min_votes"]) for key, value in result.items()}


def _save(fig, plt, destination, name):
    paths = [destination / f"{name}.{suffix}" for suffix in ("png", "svg")]
    try:
        for path in paths:
            fig.savefig(path, dpi=220, facecolor="white", metadata={"Creator": "consensus_plot.py"})
    finally:
        plt.close(fig)
    return paths


def _axes(ax, *, bars=False, labels=True):
    ax.set_ylim(0, 118 if bars else 105)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.tick_params(axis="both", colors=INK, labelcolor=INK, length=4)
    if not labels:
        ax.tick_params(axis="y", labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK)
    ax.spines[["left", "bottom"]].set_linewidth(1)
    if bars:
        ax.yaxis.grid(True, color=GRID, linewidth=.8, zorder=0)
    else:
        ax.set_xlim(0, 105)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.grid(True, color=GRID, linewidth=.8, zorder=0)


def _vote_label(row):
    k, m = row["min_votes"], row["model_count"]
    return f"{k}/{m}" if k == m else f"≥{k}/{m}"


def _annotate_votes(ax, rows, *, fontsize=15):
    """Keep black vote labels inside each panel and separate from one another."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extent = ax.get_window_extent(renderer).padded(-4)
    occupied = []
    # Unanimity is normally at upper left; label it first, then place looser rules.
    for row in sorted(rows, key=lambda r: -r["min_votes"]):
        x, y = 100 * row["coverage"], 100 * row["conditional_precision"]
        candidates = [(9, -18), (9, 10), (-9, -18), (-9, 10),
                      (9, -38), (9, 30), (-9, -38), (-9, 30),
                      (30, -22), (-30, -22), (0, 36), (0, -42)]
        best, best_penalty = None, float("inf")
        for dx, dy in candidates:
            annotation = ax.annotate(_vote_label(row), (x, y), xytext=(dx, dy),
                                     textcoords="offset points", ha="left" if dx > 0 else "right" if dx < 0 else "center",
                                     va="center", fontsize=fontsize, color=INK, zorder=6,
                                     bbox={"facecolor": "white", "edgecolor": "none", "pad": .3},
                                     arrowprops={"arrowstyle": "-", "color": INK, "lw": .8,
                                                 "shrinkA": 2, "shrinkB": 5})
            annotation.update_positions(renderer)
            annotation.update_bbox_position_size(renderer)
            box = annotation.get_bbox_patch().get_window_extent(renderer).padded(3)
            overlap = sum(max(0, min(box.x1, b.x1) - max(box.x0, b.x0)) *
                          max(0, min(box.y1, b.y1) - max(box.y0, b.y0)) for b in occupied)
            outside = max(0, extent.x0 - box.x0) + max(0, box.x1 - extent.x1) + \
                      max(0, extent.y0 - box.y0) + max(0, box.y1 - extent.y1)
            penalty = overlap + 1000 * outside
            annotation.remove()
            if penalty < best_penalty:
                best, best_penalty = (dx, dy, Bbox.from_bounds(*box.bounds)), penalty
            if penalty == 0:
                break
        dx, dy, box = best
        ax.annotate(_vote_label(row), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha="left" if dx > 0 else "right" if dx < 0 else "center", va="center",
                    fontsize=fontsize, color=INK, zorder=6,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .3},
                    arrowprops={"arrowstyle": "-", "color": INK, "lw": .8, "shrinkA": 2, "shrinkB": 5})
        occupied.append(box)


def _curve(ax, rows, color):
    usable = [r for r in rows if r["coverage"] is not None and r["conditional_precision"] is not None]
    ax.plot([100 * r["coverage"] for r in usable], [100 * r["conditional_precision"] for r in usable],
            color=color, linewidth=2.2, zorder=3)
    for row in usable:
        ax.plot(100 * row["coverage"], 100 * row["conditional_precision"],
                marker="s" if row["min_votes"] == row["model_count"] else "o", color=color,
                markersize=8, markeredgecolor=INK, markeredgewidth=.6, linestyle="none", zorder=4)
    return usable


def _ser5_overview(plt, records, destination):
    fig, axes = plt.subplots(4, 4, figsize=(19.4, 17.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.07, right=.99, bottom=.125, top=.85, wspace=.16, hspace=.30)
    for i, (ax, (dataset, name, language)) in enumerate(zip(axes.flat, DATASETS)):
        ax.set_title(f"{name} ({language})", fontsize=17, fontweight="bold", pad=10, color=INK)
        _axes(ax, labels=i % 4 == 0)
        usable = _curve(ax, records.get((dataset, "SER5"), []), GROUPS["SER5"][2])
        if not usable:
            ax.text(.5, .5, "Results pending", transform=ax.transAxes, ha="center", fontsize=15, color=INK)
    fig.suptitle("Five SER models: what changes when more votes must agree?", fontsize=24, fontweight="bold", y=.985)
    fig.text(.5, .935, "Each point is a different vote rule on the same samples: ≥3/5, ≥4/5 or 5/5.",
             ha="center", fontsize=18, color=INK)
    fig.text(.5, .905, "Square = all five agree. More coverage means more labels are accepted automatically.",
             ha="center", fontsize=17, color=INK)
    fig.supxlabel("Samples auto-labelled (%) = accepted samples / common samples", fontsize=20, y=.075)
    fig.supylabel("Correct auto-labels (%) = correct labels / accepted labels", fontsize=20, x=.012)
    fig.text(.07, .028, "Each dataset uses the same five SER models. Lines connect discrete vote rules; they are not fitted curves.",
             fontsize=15, color=INK)
    for ax, (dataset, _, _) in zip(axes.flat, DATASETS):
        usable = [r for r in records.get((dataset, "SER5"), []) if r["conditional_precision"] is not None]
        _annotate_votes(ax, usable)
    return _save(fig, plt, destination, "consensus_accuracy_coverage")


def _meld_details(plt, records, destination):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.81, bottom=.22, wspace=.17, hspace=.25)
    for ax, group in zip(axes.flat, MELD_GROUPS):
        m, name, color = GROUPS[group]
        rows = records.get(("meld", group), [])
        _axes(ax)
        ax.set_title(f"{name} ({m} models)", fontsize=20, fontweight="bold", pad=12, color=INK)
        usable = _curve(ax, rows, color)
        if rows:
            unanimous = next((row for row in rows if row["min_votes"] == m), None)
            if unanimous is not None:
                total = unanimous.get("common_samples", "?")
                accepted = unanimous.get("accepted", "?")
                correct = unanimous.get("accepted_correct", "?")
                errors = unanimous.get("accepted_errors", "?")
                ax.text(.03, .96, f"Unanimity: {accepted:,} accepted, {errors:,} errors\n"
                        f"{correct:,} correct; pool = {total:,} samples", transform=ax.transAxes,
                        fontsize=15, color=INK, ha="left", va="top")
        if not usable:
            ax.text(.5, .5, "Results pending", transform=ax.transAxes, ha="center", fontsize=16, color=INK)
    fig.suptitle("MELD: compare the vote rules within each model group", fontsize=25, fontweight="bold", y=.985)
    fig.text(.5, .93, "Every point is labelled with its required matching votes; square = all models agree.",
             ha="center", fontsize=18, color=INK)
    fig.text(.5, .895, "Colours identify model groups, not stronger or weaker confidence guarantees.",
             ha="center", fontsize=18, color=INK)
    pools = {r["common_samples"] for (dataset, _), rows in records.items() if dataset == "meld"
             for r in rows if "common_samples" in r}
    denominator = f"{next(iter(pools)):,} shared samples" if len(pools) == 1 else "each group's common samples"
    fig.supxlabel(f"Samples auto-labelled (%) = accepted / {denominator}", fontsize=20, y=.155)
    fig.supylabel("Correct auto-labels (%) = correct / accepted labels", fontsize=20, x=.015)
    fig.text(.09, .035, "Mixed3: emotion2vec+ + Gemini + Qwen2.5-Omni.  SER+API7: five SER + GPT + Gemini.\n"
             "All10: SER+API7 + Qwen2.5-Omni + Qwen2-Audio + Gemma 4.\n"
             "A group that accepts fewer samples can still make a larger fraction of errors on those samples.",
             fontsize=15, color=INK, linespacing=1.4)
    for ax, group in zip(axes.flat, MELD_GROUPS):
        usable = [r for r in records.get(("meld", group), []) if r["conditional_precision"] is not None]
        _annotate_votes(ax, usable, fontsize=16)
    return _save(fig, plt, destination, "consensus_meld_accuracy_coverage")


def _bar_panels(plt, records, destination, *, metric, by_votes=False, vote_group="SER5"):
    from matplotlib.patches import Patch

    accuracy = metric == "conditional_precision"
    label = "Correct auto-labels (%)" if accuracy else "Samples auto-labelled (%)"
    m, group_name, _ = GROUPS[vote_group]
    title = (f"{group_name}: accuracy at each vote rule" if accuracy else f"{group_name}: coverage at each vote rule") \
        if by_votes else ("All models agree: accuracy of the auto-labels" if accuracy else "All models agree: share of samples auto-labelled")
    fig, axes = plt.subplots(4, 4, figsize=(18.8, 13.5 if by_votes else 14.2), squeeze=False)
    fig.subplots_adjust(left=.065, right=.99, bottom=.19 if by_votes else .235, top=.84, wspace=.15, hspace=.40)
    if by_votes:
        colours = AUDIO_RULE_COLOURS if vote_group == "AudioLLM4" else RULE_COLOURS
        entries = [(k, f"≥{k}/{m} matching votes" if k < m else f"{m}/{m}: all {m} agree", colours[k])
                   for k in range(m // 2 + 1, m + 1)]
    else:
        names = {"SER5": "SER5", "Selected3": "Mixed3", "SER_API7": "SER+API7", "All10": "All10",
                 "AudioLLM4": "AudioLLM4"}
        entries = [(key, f"{names[key]}: all {spec[0]} agree", spec[2]) for key, spec in GROUPS.items()]
    positions = [i * .70 for i in range(len(entries))]
    for i, (dataset, name, language) in enumerate(DATASETS):
        ax = axes.flat[i]
        _axes(ax, bars=True, labels=i % 4 == 0)
        ax.set_title(f"{name} ({language})", fontsize=17, fontweight="bold", pad=9, color=INK)
        ax.set_xticks([])
        ax.set_xlim(-.48, positions[-1] + .48)
        for position, (entry, _name, color) in zip(positions, entries):
            if by_votes:
                row = next((r for r in records.get((dataset, vote_group), []) if r["min_votes"] == entry), None)
            else:
                m = GROUPS[entry][0]
                row = next((r for r in records.get((dataset, entry), []) if r["min_votes"] == m), None)
            value = row.get(metric) if row is not None else None
            if value is None:
                ax.text(position, 3, "—" if row is not None else "n/a", ha="center", va="bottom",
                        fontsize=15, color=INK)
                continue
            ax.bar(position, 100 * value, width=.56, color=color, edgecolor=INK, linewidth=1,
                   zorder=3)
            ax.text(position, 100 * value + 1.8, f"{100 * value:.0f}", ha="center", va="bottom",
                    fontsize=15.5, color=INK)
    fig.suptitle(title, fontsize=24, fontweight="bold", y=.985)
    denominator = "Correct predictions / auto-labelled samples" if accuracy else "Auto-labelled samples / all common samples"
    rule = f"Each colour is a different vote rule on the same {GROUPS[vote_group][0]} models." if by_votes else \
           "Each colour is a fixed model group; every model in that group must agree."
    fig.text(.5, .93, denominator + "; values are percentages.", ha="center", fontsize=18, color=INK)
    fig.text(.5, .90, rule, ha="center", fontsize=17, color=INK)
    fig.supylabel(label, fontsize=20, x=.012)
    handles = [Patch(facecolor=color, edgecolor=INK, linewidth=1, label=name) for _, name, color in entries]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .105 if by_votes else .14),
               ncol=3, frameon=False, fontsize=16, columnspacing=1.6,
               handlelength=1.4, handletextpad=.45)
    note = "n/a = group incomplete.  — = no samples accepted, so accuracy is undefined.  Bar labels round to whole percentages."
    fig.text(.065, .065 if by_votes else .10, note, fontsize=15, color=INK)
    if not by_votes:
        fig.text(.065, .025, "Mixed3 = emotion2vec+ + Gemini + Qwen2.5-Omni.  SER+API7 = five SER + GPT + Gemini.\n"
                 "All10 adds Qwen2.5-Omni, Qwen2-Audio and Gemma 4 to SER+API7.\n"
                 "AudioLLM4 = GPT + Gemini + Qwen2.5-Omni + Qwen2-Audio; Gemma 4 is excluded.", fontsize=15,
                 color=INK, linespacing=1.35)
    elif vote_group == "AudioLLM4":
        fig.text(.065, .025, "Members: GPT Realtime + Gemini + Qwen2.5-Omni + Qwen2-Audio. Gemma 4 is not included.\n"
                 "A stricter rule reduces automatic acceptance; its observed accuracy need not increase on every dataset.",
                 fontsize=15, color=INK, linespacing=1.35)
    else:
        fig.text(.065, .025, "A stricter rule reduces automatic acceptance; its observed accuracy need not increase on every dataset.",
                 fontsize=15, color=INK)
    name = f"consensus_{vote_group.lower()}_{'accuracy' if accuracy else 'coverage'}_by_votes" if by_votes else \
           f"consensus_unanimous_{'accuracy' if accuracy else 'coverage'}_by_dataset"
    return _save(fig, plt, destination, name)


def plot_thresholds(threshold_rows, output_dir):
    """Export eight PNG/SVG figure pairs using black, large text and explicit rules.

    Coverage is accepted/common samples; accuracy is correct/accepted labels.
    The overview isolates SER5 so all three thresholds can be labelled clearly.
    The MELD figure separates the four complete model groups into four panels.
    Two unanimous bars compare fixed groups; separate SER5 and AudioLLM4 bars
    compare vote rules. AudioLLM4 excludes Gemma and always requires its four members.
    Incomplete fixed groups are shown as unavailable, never as zero accuracy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    records = _records(threshold_rows)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Helvetica Neue", "Helvetica", "Arial") if name in fonts), "DejaVu Sans")
    with plt.rc_context({
        "font.family": family, "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 20,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 16,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK,
        "svg.fonttype": "none", "svg.hashsalt": "emotion-consensus-readable-v2",
        "savefig.facecolor": "white",
    }):
        paths = _ser5_overview(plt, records, destination)
        paths += _meld_details(plt, records, destination)
        for metric in ("conditional_precision", "coverage"):
            paths += _bar_panels(plt, records, destination, metric=metric)
            paths += _bar_panels(plt, records, destination, metric=metric, by_votes=True)
            paths += _bar_panels(plt, records, destination, metric=metric, by_votes=True, vote_group="AudioLLM4")
    return paths

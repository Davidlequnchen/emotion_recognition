"""Schematic figures of the deck (no audio, no models): writes fig/*.png and assets/worked_example.json.

Figures: CTC trellis with the best path (illustrative scores), pause score p for a slow and a fast speaker, the
length cost L(d), the worked example of the method document (section 4.6) with the DP and a greedy rule, and the
alignment-confidence ranges from the README. The worked example comes from tests/test_dp.py, so the figure shows
exactly the data the unit test checks.
Run (any folder):  python preparation_tools/presentation/scripts/make_figures.py
"""
import json
import sys

import matplotlib
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from style import (ACCENT, ACCENT2, ASSETS, BAD, DECK, FILL_BLUE, FILL_ORANGE, FILL_RED, GOOD, GRID, INK, TOOLS,
                   plt, save)

sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "tests"))
from segmenter.boundaries import best_partition, length_cost, partition_cost  # noqa: E402

LO, HI, TARGET = 2.0, 15.0, 7.0   # the length window of the design agreed on 2026-11-13 (one window, 2-15 s)
import test_dp  # noqa: E402  (worked example of doc section 4.6)

print("Versions: numpy", np.__version__, "| matplotlib", matplotlib.__version__)
print("Output:", (DECK / "fig").relative_to(TOOLS.parent))
ASSETS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 1) CTC trellis (illustrative scores)
def ctc_trellis():
    """Classic CTC trellis for the word YES: rows blank,Y,blank,E,blank,S,blank; 12 frames of 20 ms.
    Scores are made up for illustration; the path is the true best path for these scores (Viterbi below)."""
    print("Step 1: CTC trellis")
    rows = ["–", "Y", "–", "E", "–", "S", "–"]
    labels = ["blank", "Y", "blank", "E", "blank", "S", "blank"]
    T = 12
    truth = [0, 0, 1, 1, 2, 3, 3, 3, 4, 5, 5, 6]               # where the speech 'really' is, per frame
    rng = np.random.default_rng(3)
    prob = np.full((len(rows), T), 0.04)
    for t, r in enumerate(truth):
        prob[r, t] = rng.uniform(0.62, 0.9)
        for other in (r - 1, r + 1):                            # some mass on the neighbours
            if 0 <= other < len(rows):
                prob[other, t] = max(prob[other, t], rng.uniform(0.08, 0.25))
    # blank rows share one 'blank' score per frame (same output unit)
    blank = prob[[0, 2, 4, 6]].max(axis=0)
    for r in (0, 2, 4, 6):
        prob[r] = blank
    logp = np.log(prob)
    # Viterbi over the expanded CTC sequence: stay, move one row, or skip a blank between different letters
    S = len(rows)
    score = np.full((S, T), -np.inf)
    back = np.zeros((S, T), dtype=int)
    score[0, 0], score[1, 0] = logp[0, 0], logp[1, 0]
    for t in range(1, T):
        for s in range(S):
            options = [(score[s, t - 1], s)]
            if s >= 1:
                options.append((score[s - 1, t - 1], s - 1))
            if s >= 2 and rows[s] != "–" and rows[s] != rows[s - 2]:
                options.append((score[s - 2, t - 1], s - 2))
            best, src = max(options)
            score[s, t], back[s, t] = best + logp[s, t], src
    s = S - 1 if score[S - 1, T - 1] >= score[S - 2, T - 1] else S - 2   # end on the last blank or last letter
    path = [s]
    for t in range(T - 1, 0, -1):
        s = back[s, t]
        path.append(s)
    path = path[::-1]

    fig = plt.figure(figsize=(6.6, 4.75))
    ax = fig.add_axes([0.2, 0.17, 0.77, 0.62])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#FFFFFF", "#9CC3E6", "#3D85C6"])
    ax.imshow(prob, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="upper")
    for x in np.arange(-0.5, T, 1):
        ax.axvline(x, color="white", lw=1.5)
    for y in np.arange(-0.5, S, 1):
        ax.axhline(y, color="white", lw=1.5)
    ax.plot(range(T), path, color=ACCENT2, lw=3.5, marker="o", ms=9, mfc=ACCENT2, mec="white", mew=1.5, zorder=3)
    ax.set_yticks(range(S), labels, fontsize=17)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab != "blank":
            tick.set_fontweight("bold")
            tick.set_fontsize(19)
    ax.set_xticks(range(0, T, 2), [f"{20 * t}" for t in range(0, T, 2)])
    ax.set_xlabel("time (ms): one wav2vec 2.0 frame every 20 ms")
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    # letter spans from the path (consecutive frames on the same letter row)
    spans = {}
    for t, s in enumerate(path):
        if rows[s] != "–":
            spans.setdefault(s, []).append(t)
    top = fig.add_axes([0.2, 0.8, 0.77, 0.15], sharex=ax)
    top.set_xlim(ax.get_xlim())
    top.set_ylim(0, 1)
    top.axis("off")
    for s, frames in spans.items():
        a, b = frames[0] - 0.45, frames[-1] + 0.45
        top.add_patch(FancyBboxPatch((a, 0.12), b - a, 0.62, boxstyle="round,pad=0,rounding_size=0.15",
                                     fc=FILL_ORANGE, ec=ACCENT2, lw=1.5))
        top.text((a + b) / 2, 0.43, rows[s], ha="center", va="center", fontsize=19, fontweight="bold")
    top.text(-0.45, 0.43, "letter spans", ha="right", va="center", fontsize=17)
    save(fig, "ctc_trellis")
    return {"path_rows": [labels[s] for s in path]}


# ---------------------------------------------------------------- 2) relative pause score p
def pause_score():
    print("Step 2: pause score p for a slow and a fast speaker")
    gaps = np.linspace(0, 1.6, 400)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.95, bottom=0.19)
    for typical, color, name, dy in ((0.6, ACCENT, "slow speaker", 0.0), (0.2, ACCENT2, "fast speaker", 0.0)):
        p = np.minimum(gaps / (2 * typical), 1)
        ax.plot(gaps, p, color=color, lw=3.2)
        ax.plot([typical], [0.5], "o", color=color, ms=10, mec="white", mew=1.5, zorder=3)
        ax.plot([2 * typical], [1.0], "s", color=color, ms=9, mec="white", mew=1.5, zorder=3)
    # labels with a short line sample in the series colour (text stays dark)
    for (x, y, color, label) in ((0.47, 1.2, ACCENT2, "fast speaker: typical pause 0.2 s,\nfull marks from 0.4 s"),
                                 (0.66, 0.08, ACCENT, "slow speaker: typical pause 0.6 s\n(p = 0.5), full marks from 1.2 s")):
        ax.plot([x - 0.1, x - 0.02], [y + 0.07, y + 0.07], color=color, lw=3.2)
        ax.text(x, y + 0.07, label, ha="left", va="center", fontsize=17, linespacing=1.15)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1.42)
    ax.set_xlabel("gap between two words (s)")
    ax.set_ylabel("pause score p")
    ax.set_yticks([0, 0.5, 1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    save(fig, "pause_score")


# ---------------------------------------------------------------- 3) length cost L(d)
def length_cost_plot():
    print("Step 3: length cost L(d), window 2-15 s, with the price of one poor cut")
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.18)
    d = np.linspace(LO, HI, 400)
    ax.axvspan(0, LO, color=FILL_RED, lw=0)
    ax.axvspan(HI, 17, color=FILL_RED, lw=0)
    for x in (LO / 2, (HI + 17) / 2):
        ax.text(x, 0.75, "∞", ha="center", va="center", fontsize=22, color=INK)
    ax.text(8.5, 1.5, "outside 2–15 s: not allowed (∞)", ha="center", va="bottom", fontsize=17)
    ax.plot(d, ((d - TARGET) / TARGET) ** 2, color=ACCENT, lw=3.2)
    poor = 0.8
    cross = TARGET + TARGET * poor ** 0.5
    ax.plot([LO, HI], [poor, poor], color=ACCENT2, lw=2, ls=(0, (5, 3)))
    ax.text(2.3, poor + 0.04, "one poor cut (s = 0.2) = 0.8", ha="left", va="bottom", fontsize=17)
    ax.plot([cross], [poor], "o", color=ACCENT2, ms=10, mec="white", mew=1.5, zorder=4)
    ax.text(cross + 0.25, poor - 0.06, f"{cross:.1f} s", ha="left", va="top", fontsize=17, fontweight="bold")
    marks = [(2, "0.51", (0.25, 0.0), "left"), (4, "0.18", (0.25, 0.0), "left"),
             (10, "0.18", (-0.3, 0.03), "right"), (15, "1.31", (-0.3, 0.0), "right")]
    for x, lab, (dx, dy), ha in marks:
        y = length_cost(x, TARGET, LO, HI)
        ax.plot([x], [y], "o", color=ACCENT, ms=9, mec="white", mew=1.5, zorder=3)
        ax.text(x + dx, y + dy, lab, ha=ha, va="bottom", fontsize=17)
    ax.annotate("0 at 7 s", xy=(7, 0), xytext=(7, 0.36), ha="center", fontsize=17,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 1.45)
    ax.set_xticks([0, 2, 4, 7, 10, 15])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("segment length d (s)")
    ax.set_ylabel("length price L(d)")
    save(fig, "length_cost")
    return round(cross, 2)


# ---------------------------------------------------------------- 4) worked example (doc section 4.6)
def greedy(times, start, end, step=7.0):
    """'Cut at the first pause after 7 s': from each cut, take the first candidate at least 7 s later."""
    cuts, last = [], start
    for t in times:
        if t >= last + step and t < end:
            cuts.append(t)
            last = t
    return cuts


def worked_example():
    print("Step 4: worked example (tests/test_dp.py, doc section 4.6)")
    times, scores = test_dp.worked_example()
    cuts, total, length_total, cut_total = best_partition(times, scores, 1.0, TARGET, LO, HI)
    dp_cuts = [times[i] for i in cuts]
    a = [times.index(t) for t in (7.4, 15.3, 23.0, 29.6)]
    b = a + [times.index(34.2)]
    inner = times[1:-1]
    g_cuts = greedy(inner, 0.0, 40.0)
    g_idx = [times.index(t) for t in g_cuts]
    cost = lambda idx: dict(zip(("total", "length", "cut"),  # noqa: E731
                                (round(x, 4) for x in partition_cost(times, scores, idx, 1.0, TARGET, LO, HI))))
    # runner-up: the cheapest partition that differs from the DP answer (brute force over 3-6 cuts)
    from itertools import combinations
    ranked = sorted((partition_cost(times, scores, list(c), 1.0, TARGET, LO, HI)[0], c)
                    for r in range(3, 7) for c in combinations(range(1, len(times) - 1), r))
    runner = next((v, c) for v, c in ranked if list(c) != cuts)
    result = {
        "window_s": [LO, HI],
        "runner_up": {"cuts": [times[i] for i in runner[1]], "total": round(runner[0], 4)},
        "candidates": len(inner),
        "dp": {"cuts": dp_cuts, "total": round(total, 4), "length": round(length_total, 4), "cut": round(cut_total, 4)},
        "A": cost(a),
        "B": cost(b),
        "greedy": {"cuts": g_cuts, "scores": [round(scores[i], 4) for i in g_idx], **cost(g_idx)},
        "scores": {f"{t:.1f}": round(s, 4) for t, s in zip(times[1:-1], scores[1:-1])},
    }
    print(f"  DP cuts {dp_cuts}, total {total:.3f}; A {result['A']}; B {result['B']}; greedy {result['greedy']}")

    strong = {t: s for t, _, _, _, s in test_dp.STRONG}
    fig = plt.figure(figsize=(12.1, 4.5))
    ax = fig.add_axes([0.105, 0.42, 0.88, 0.56])
    for t, s in zip(times[1:-1], scores[1:-1]):
        is_strong = t in strong
        color = ACCENT if t in dp_cuts else ("#9CC3E6" if not is_strong else "#5DA5DA")
        if t == 21.8:
            color = ACCENT2
        ax.bar(t, s, width=0.42 if is_strong else 0.3, color=color, lw=0)
    for t, s in zip(times[1:-1], scores[1:-1]):
        if t in strong and t not in (21.8, 29.6, 34.2):
            ax.text(t, s + 0.04, f"{strong[t]:.2f}", ha="center", va="bottom", fontsize=17,
                    fontweight="bold" if t in dp_cuts else "normal")
    notes = [(29.6, 0.99, (30.3, 1.2), "“?” pause, 0.99: cut", "left"),
             (21.8, 0.845, (19.6, 1.2), "sigh, 0.84: skipped for 23.0", "right"),
             (34.2, 0.62, (33.6, 0.98), "0.62: not worth\nan extra cut", "left")]
    for x, y, (tx, ty), text, ha in notes:
        ax.annotate(text, xy=(x, y + 0.02), xytext=(tx, ty), ha=ha, va="center", fontsize=17,
                    fontweight="bold" if x == 29.6 else "normal",
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT2 if x == 21.8 else INK, lw=1.4))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("score s")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    seg = fig.add_axes([0.105, 0.17, 0.88, 0.2])
    seg.set_ylim(0, 1)
    seg.set_xlim(0, 40)
    bounds = [0.0, *dp_cuts, 40.0]
    for x0, x1 in zip(bounds, bounds[1:]):
        seg.add_patch(Rectangle((x0 + 0.06, 0.08), x1 - x0 - 0.12, 0.84, fc=FILL_BLUE, ec=ACCENT, lw=1.6))
        seg.text((x0 + x1) / 2, 0.5, f"{x1 - x0:.1f} s", ha="center", va="center", fontsize=17)
    seg.text(-0.6, 0.5, "segments", ha="right", va="center", fontsize=18, fontweight="bold")
    seg.set_yticks([])
    seg.spines["left"].set_visible(False)
    seg.set_xticks(range(0, 41, 5))
    seg.set_xlabel("time in the turn (s)")
    save(fig, "worked_example")
    (ASSETS / "worked_example.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("  wrote assets/worked_example.json")
    return result


# ---------------------------------------------------------------- 5) alignment confidence (README, YAML comment)
def confidence():
    print("Step 5: alignment confidence ranges (preparation_tools/README.md, config comment)")
    conds = ["correct transcript", "same audio, 8 kHz μ-law", "30% of words replaced", "wrong transcript"]
    mean = [(0.90, 0.95), (0.88, 0.94), (0.63, 0.74), (0.27, 0.36)]
    weak = [(0.0, 0.02), (0.0, 0.02), (0.18, 0.28), (0.51, 0.63)]
    fig = plt.figure(figsize=(12.1, 3.9))
    axes = [fig.add_axes([0.235, 0.22, 0.35, 0.76]), fig.add_axes([0.635, 0.22, 0.35, 0.76])]
    for ax, data, thr, title, keep_left in ((axes[0], mean, 0.5, "mean word confidence per turn", True),
                                           (axes[1], weak, 0.4, "share of weak words (< 0.3)", False)):
        for k, (lo, hi) in enumerate(data):
            y = len(conds) - 1 - k
            ok = (lo >= thr) if keep_left else (hi <= thr)
            color = GOOD if ok else BAD
            ax.plot([lo, hi], [y, y], color=color, lw=9, solid_capstyle="round")
            if lo > 0:
                ax.text((lo + hi) / 2, y + 0.2, f"{lo:.2f}–{hi:.2f}", va="bottom", ha="center", fontsize=17)
            else:
                ax.text(hi + 0.04, y, f"≤ {hi:.2f}", va="center", ha="left", fontsize=17)
        ax.plot([thr, thr], [-0.6, len(conds) - 0.3], color=INK, lw=1.6, ls=(0, (4, 3)))
        ax.text(thr, len(conds) - 0.25, f"threshold {thr}", ha="center", va="bottom", fontsize=17)
        ax.set_xlim(0, 1.18)
        ax.set_ylim(-0.6, len(conds) + 0.25)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xlabel(title)
        ax.set_yticks(range(len(conds)), conds[::-1] if keep_left else [""] * len(conds))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    save(fig, "confidence")


if __name__ == "__main__":
    ctc_trellis()
    pause_score()
    cross = length_cost_plot()
    result = worked_example()
    result["poor_cut_equals_length_s"] = cross
    (ASSETS / "worked_example.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    confidence()
    print("Done.")

"""Are the boundary-score weights critical? Re-run the cut choice over a grid of weights (no models, no audio).

For every long English turn of the integration run (synthetic DailyTalk calls), the review JSON stores each word gap with its p, q, v,
whether it may be cut, and where the cut would go (middle of the quiet part). This script recomputes s = w_p p + w_q q + w_v v
for a grid of weights (normalised to sum 1), re-runs the dynamic programme (segmenter/boundaries.py, window 2-15 s,
target 7 s, lambda 1) and measures: share of cuts in the word gap at an utterance join (joins rebuilt with the
integration seed) or after a sentence end, number of segments, segment-length range, and agreement with the
cuts of the default weights.
Also the three single cues (pause only, punctuation only, silence only).
Writes assets/sensitivity.json and fig/sensitivity.png. Needs numpy, matplotlib, soundfile, PyYAML and the local
DailyTalk copy (for the joins); the integration outputs in ../local/integration/.
Run:  python preparation_tools/presentation/scripts/sensitivity.py
"""
import importlib.util
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from style import ACCENT, ACCENT2, ASSETS, GOOD, GRID, INK, TOOLS, plt, save

sys.path.insert(0, str(TOOLS))
from segmenter.boundaries import best_partition  # noqa: E402

OUT = TOOLS / "local" / "integration" / "output"
TURNS = []                       # the long English turns, filled in main()
LO, HI, TARGET, LAM = 2.0, 15.0, 7.0, 1.0
DEFAULT = (0.5, 0.3, 0.2)
GRID_P, GRID_Q, GRID_V = (0.3, 0.4, 0.5, 0.6, 0.7), (0.0, 0.1, 0.2, 0.3, 0.4, 0.5), (0.1, 0.2, 0.3)
SINGLE = {"pause only": (1, 0, 0), "punctuation only": (0, 1, 0), "silence only": (0, 0, 1)}


def rebuild_joins():
    """Utterance joins of the long English turns: the integration script's own make_inputs() is run into a
    temporary folder (same seed); every rebuilt transcript must equal the saved input."""
    spec = importlib.util.spec_from_file_location("integ", TOOLS / "tests" / "integration_dailytalk.py")
    integ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(integ)
    real = integ.OUT
    with tempfile.TemporaryDirectory() as tmp:
        integ.OUT = Path(tmp)
        try:
            truth = integ.make_inputs()
            for turn_id in truth:
                rebuilt = (Path(tmp) / "input" / "turns" / f"{turn_id}.txt").read_text(encoding="utf-8").strip()
                saved = (TOOLS / "local" / "integration" / "input" / "turns" / f"{turn_id}.txt").read_text(encoding="utf-8").strip()
                assert rebuilt == saved, f"{turn_id}: rebuilt transcript differs from the saved input"
        finally:
            integ.OUT = real
    return {t: v["joins"] for t, v in truth.items() if v["kind"] == "long_en"}


def run(weights, reviews, joins):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    cuts_all, lengths, at_join, at_sentence = {}, [], 0, 0
    for t in TURNS:
        rv = reviews[t]
        cands = [g for g in rv["gaps"] if g["candidate"]]
        start, end = rv["span_s"]
        times = [start] + [g["cut_time"] for g in cands] + [end]
        scores = [0.0] + [w[0] * g["p"] + w[1] * g["q"] + w[2] * g["v"] for g in cands] + [1.0]
        res = best_partition(times, scores, LAM, TARGET, LO, HI)
        if res is None:
            cuts_all[t] = None
            continue
        idx = res[0]
        chosen = [cands[i - 1] for i in idx]
        cuts_all[t] = [round(g["cut_time"], 3) for g in chosen]
        bounds = [start] + [g["cut_time"] for g in chosen] + [end]
        lengths += [b - a for a, b in zip(bounds, bounds[1:])]
        for g in chosen:
            if any(g["start"] <= b and a <= g["end"] for a, b in joins[t]):
                at_join += 1
            elif g["q"] >= 1.0:
                at_sentence += 1
    n_cuts = sum(len(c) for c in cuts_all.values() if c)
    return {"weights": [round(x, 4) for x in w], "cuts": cuts_all, "n_cuts": n_cuts, "n_segments": len(lengths),
            "at_join": at_join, "at_sentence_end": at_sentence,
            "good_share": round((at_join + at_sentence) / n_cuts, 4) if n_cuts else None,
            "under_3s": sum(x < 3 for x in lengths), "min_len": round(min(lengths), 3), "max_len": round(max(lengths), 3)}


def agreement(a, b):
    """Share of cuts in a that are also in b (same gap), over all turns."""
    same = total = 0
    for t in TURNS:
        ca, cb = set(a["cuts"][t] or []), set(b["cuts"][t] or [])
        same += len(ca & cb)
        total += len(ca)
    return round(same / total, 4) if total else None


def main():
    print("Step 1: rebuild the utterance joins and load the review JSONs of the long English turns")
    joins = rebuild_joins()
    TURNS.extend(sorted(joins))
    reviews = {t: json.loads((OUT / "review" / f"{t}.json").read_text(encoding="utf-8")) for t in TURNS}
    print("Step 2: default weights, grid, single cues (window 2-15 s, target 7 s, lambda 1)")
    default = run(DEFAULT, reviews, joins)
    grid = []
    for wp, wq, wv in itertools.product(GRID_P, GRID_Q, GRID_V):
        r = run((wp, wq, wv), reviews, joins)
        r["raw"] = [wp, wq, wv]
        r["agree_default"] = agreement(r, default)
        grid.append(r)
    single = {}
    for name, w in SINGLE.items():
        r = run(w, reviews, joins)
        r["agree_default"] = agreement(r, default)
        single[name] = r
    shares = [g["good_share"] for g in grid]
    agrees = [g["agree_default"] for g in grid]
    summary = {
        "settings": len(grid), "default": default,
        "good_share_min_max": [min(shares), max(shares)],
        "settings_as_good_as_default": sum(x >= default["good_share"] for x in shares),
        "under_3s_max": max(g["under_3s"] for g in grid),
        "agree_default_min_median": [min(agrees), float(np.median(agrees))],
        "segments_min_max": [min(g["n_segments"] for g in grid), max(g["n_segments"] for g in grid)],
        "length_min": min(g["min_len"] for g in grid), "length_max": max(g["max_len"] for g in grid),
        "single": {k: {kk: v[kk] for kk in ("n_segments", "n_cuts", "at_join", "at_sentence_end", "good_share", "under_3s",
                                            "min_len", "max_len", "agree_default")}
                   for k, v in single.items()},
    }
    print("  default:", {k: default[k] for k in ("n_segments", "n_cuts", "at_join", "at_sentence_end", "min_len", "max_len")})
    print("  grid:", {k: v for k, v in summary.items() if k not in ("default", "single")})
    for k, v in summary["single"].items():
        print(f"  {k}: {v}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "sensitivity.json").write_text(json.dumps({"summary": summary, "grid": grid, "single": single}, indent=1),
                                             encoding="utf-8")

    print("Step 3: figure")
    fig, axes = plt.subplots(1, 3, figsize=(12.1, 2.7), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.985, top=0.87, bottom=0.27, wspace=0.12)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("g", ["#FFFFFF", "#9CC3E6", "#2B6CB0"])
    for ax, wv in zip(axes, GRID_V):
        m = np.array([[next(g["agree_default"] for g in grid if g["raw"] == [wp, wq, wv]) for wp in GRID_P] for wq in GRID_Q])
        ax.imshow(m * 100, cmap=cmap, vmin=50, vmax=100, origin="lower", aspect="auto")
        for i in range(len(GRID_Q)):
            for j in range(len(GRID_P)):
                ax.text(j, i, f"{m[i, j] * 100:.0f}", ha="center", va="center", fontsize=16, color=INK if m[i, j] < 0.9 else "white")
        ax.set_xticks(range(len(GRID_P)), [f"{x:.1f}" for x in GRID_P])
        ax.set_yticks(range(len(GRID_Q)), [f"{x:.1f}" for x in GRID_Q])
        ax.set_title(f"silence weight {wv}", fontsize=17)
        ax.set_xlabel("pause weight")
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        if wv == DEFAULT[2]:
            j, i = GRID_P.index(DEFAULT[0]), GRID_Q.index(DEFAULT[1])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ACCENT2, lw=3))
    axes[0].set_ylabel("punctuation weight")
    save(fig, "sensitivity")
    print("Done.")


if __name__ == "__main__":
    main()

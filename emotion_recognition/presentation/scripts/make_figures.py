"""Statistics and figures of the presentation: writes assets/stats.json and PNG/SVG pairs in assets/fig/.

Reads: outputs/<model>/<dataset>_predictions.csv, reports/summary.json, the audio headers in data/<dataset>/,
data/crema_d/processedResults/tabulatedVotes.csv and data/mcae_spps/Corpus information.xlsx.
Run (from the repository root or anywhere):  python emotion_recognition/presentation/scripts/make_figures.py
"""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # no window, files only
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import soundfile as sf
from matplotlib.patches import Patch

from common import (ASSETS, CREMA_CODE, DATASET_NAMES, DATASETS, EMOTIONS, ER, MODEL_NAMES, MODELS, ORDER,
                    crema_votes, load_predictions, load_summary, mcae_info, parse_meta, training_membership)

# ---------------------------------------------------------------- settings
FIG = ASSETS / "fig"
DPI = 300
SEEN_C, UNSEEN_C = "#eb6834", "#2a78d6"          # orange = dataset in the model's training data, blue = not
GRID_C, GRID_W = "#333333", 0.6                  # heatmap cell lines (pt)
BORDER_W = 1.2                                   # heatmap outer border (pt)
BAR_EDGE, BAR_EDGE_W = "black", 1.0              # edge line on every bar
PAIR_MODELS = ["emotion2vec_plus_large", "meralion_ser_v1"]  # the two top models, shown side by side
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def save_figure(fig, name):
    """Export the plotted objects in both raster and vector formats."""
    fig.savefig(FIG / f"{name}.png", dpi=DPI)
    fig.savefig(FIG / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def r4(x):
    """Numbers in stats.json are rounded to 4 decimals."""
    return round(float(x), 4)


print("Versions: numpy", np.__version__, "| matplotlib", matplotlib.__version__, "| soundfile", sf.__version__)
print("Project folder:", ER.name, "| output:", FIG.relative_to(ER))
FIG.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- Step 1: inputs
print("Step 1: load predictions of the 5 models, summary.json and listener data")
data = {ds: load_predictions(ds) for ds in DATASETS}
summary = load_summary()
res = summary["results"]
SEEN = training_membership(summary)
votes = crema_votes()
crema = data["crema_d"]
for r in crema:
    r.update(votes[r["stem"]])  # A, D, F, H, N, S, numResponses (voice only)
mcae_table = mcae_info()
mcae = data["mcae_spps"]
for r in mcae:
    r.update(mcae_table[r["stem"]])  # vote columns, rr, sentence content
for ds in DATASETS:
    print(f"  {ds}: {len(data[ds])} clips")
print("  training data:", {m: sorted(s) for m, s in SEEN.items()})

# ---------------------------------------------------------------- Step 2: per-dataset statistics
print("Step 2: per-dataset statistics (reading audio headers)")
stats = {"datasets": {}, "models": {}}
for ds in DATASETS:
    rows = data[ds]
    meta = [parse_meta(ds, r["stem"], r["audio_path"]) for r in rows]
    infos = [sf.info(str(ER / r["audio_path"])) for r in rows]
    dur = np.array([i.frames / i.samplerate for i in infos])
    srs = Counter(i.samplerate for i in infos)
    if ds == "jl_corpus":
        n_sent = len({m["sentence"] for m in meta})
    elif ds == "mcae_spps":
        n_sent = len({r["sentence content"] for r in rows})
    else:
        n_sent = len({m["sentence_id"] for m in meta})
    counts = Counter(r["acted_emotion"] for r in rows)
    speakers = len({m["speaker"] for m in meta})
    stats["datasets"][ds] = {
        "name": DATASET_NAMES[ds],
        "clips_evaluated": len(rows),
        "speakers": speakers,
        "emotions": [e for e in ORDER if e in counts],
        "per_emotion_counts": {e: counts[e] for e in ORDER if e in counts},
        "sentence_count": n_sent,
        "sample_rate_hz": {str(k): v for k, v in srs.most_common()},
        "mean_duration_s": r4(dur.mean()),
        "total_duration_h": round(float(dur.sum() / 3600), 2),
    }
    print(f"  {ds}: {len(rows)} clips, {speakers} speakers, {n_sent} sentences, {dict(srs)}")
stats["datasets"]["mess"]["note"] = ("No transcripts; sentence ids (V/T/M + number) from file names. Every clip is "
                                     "RMS-normalised to about -26.9 dBFS, so loudness carries no emotion cue.")
stats["datasets"]["tess"]["note"] = "Sentence = carrier phrase 'Say the word ___'; count is the number of target words."
stats["datasets"]["jl_corpus"]["note"] = ("Only the 5 primary emotions (1,200 clips) are evaluated; excited counts as "
                                          "correct when predicted happy or surprised.")

# ---------------------------------------------------------------- Step 3: CREMA-D actor vs listeners
print("Step 3: CREMA-D acted vs listener (voice-only) statistics")
acted = np.array([r["acted_emotion"] for r in crema])
lv = np.array([r["listener_vote"] for r in crema])           # "a or b" when the listener majority is a tie
nresp = np.array([r["numResponses"] for r in crema])
tie = np.array([" or " in v for v in lv])
agree = lv == acted
code = {v: k for k, v in CREMA_CODE.items()}
acted_votes = np.array([r[code[r["acted_emotion"]]] for r in crema])
perceived = np.where(tie, "tie", lv)
cr_rows = [e for e in EMOTIONS if e in set(acted)]
cr_cols = cr_rows + ["tie"]
mat = np.array([[int(np.sum((acted == a) & (perceived == p))) for p in cr_cols] for a in cr_rows])
crema_stats = {
    "clips": len(crema),
    "raters_per_clip_mean": r4(nresp.mean()),
    "raters_per_clip_range": [int(nresp.min()), int(nresp.max())],
    "listener_majority_equals_acted": r4(agree.mean()),
    "listener_majority_equals_acted_or_in_tie": r4(np.mean([a in l.split(" or ") for a, l in zip(acted, lv)])),
    "majority_is_tie": r4(tie.mean()),
    "per_acted_emotion": {},
    "zero_listeners_chose_acted": r4((acted_votes == 0).mean()),
    "mean_share_of_votes_for_acted": r4((acted_votes / nresp).mean()),
}
for i, e in enumerate(cr_rows):
    sel = acted == e
    nontie = mat[i, :-1]
    crema_stats["per_acted_emotion"][e] = {
        "clips": int(sel.sum()),
        "listener_majority_equals_acted": r4((lv[sel] == e).mean()),
        "tie": r4(tie[sel].mean()),
        "most_common_listener_label": cr_rows[int(np.argmax(nontie))],
    }
crema_stats["acted_vs_perceived_matrix"] = {"rows_acted": cr_rows, "cols_perceived": cr_cols, "counts": mat.tolist()}
# Do the models follow the actor or the listeners when the two disagree (clear, non-tie majority)?
dis = [r for r, t, a in zip(crema, tie, agree) if not t and not a]
crema_stats["when_listeners_disagree_with_actor"] = {"clips": len(dis), "models": {
    m: {"predicts_acted": r4(np.mean([r[f"pred_{m}"] == r["acted_emotion"] for r in dis])),
        "predicts_listener_majority": r4(np.mean([r[f"pred_{m}"] == r["listener_vote"] for r in dis]))}
    for m in MODELS}}
stats["crema_d_listeners"] = crema_stats
print(f"  listener majority = actor on {agree.mean() * 100:.1f}% of {len(crema)} clips")

# ---------------------------------------------------------------- Step 4: MCAE-SPPS recognition rate
print("Step 4: MCAE-SPPS listener recognition rate")
rr = np.array([r["rr"] for r in mcae])
mcae_acted = np.array([r["acted_emotion"] for r in mcae])
stats["mcae_spps_listeners"] = {
    "raters_per_clip": 40,
    "recognition_rate": {"mean": r4(rr.mean()), "min": r4(rr.min()), "q1": r4(np.quantile(rr, .25)),
                         "median": r4(np.median(rr)), "q3": r4(np.quantile(rr, .75)), "max": r4(rr.max())},
    "per_emotion_mean": {e: r4(rr[mcae_acted == e].mean()) for e in EMOTIONS},
    "clips_with_rate_1.0": int((rr == 1).sum()),
    "clips_with_rate_le_0.5": int((rr <= 0.5).sum()),
    "note": "The corpus kept only clips whose intended emotion was the listeners' most frequent choice (min 47.5%).",
}
print(f"  mean recognition rate {rr.mean() * 100:.1f}%")

# ---------------------------------------------------------------- Step 5: model accuracy
print("Step 5: per-model accuracy and seen vs unseen means")
for m in MODELS:
    acc = {ds: r4(res[m][ds]["accuracy"]) for ds in DATASETS}
    seen = [ds for ds in DATASETS if ds in SEEN[m]]
    unseen = [ds for ds in DATASETS if ds not in SEEN[m]]
    stats["models"][m] = {
        "name": MODEL_NAMES[m], "accuracy": acc,
        "unweighted_accuracy": {ds: r4(res[m][ds]["unweighted_accuracy"]) for ds in DATASETS},
        "datasets_in_training_data": seen,
        "mean_accuracy_in_training_data": r4(np.mean([acc[d] for d in seen])) if seen else None,
        "mean_accuracy_not_in_training_data": r4(np.mean([acc[d] for d in unseen])),
    }
(ASSETS / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print("  wrote assets/stats.json")

# ---------------------------------------------------------------- figure helpers
def row_percent(counts):
    """Each row as % of its total (rows with no clips stay 0)."""
    tot = counts.sum(axis=1, keepdims=True)
    return np.divide(counts * 100.0, tot, out=np.zeros(counts.shape), where=tot > 0)


def cm_counts(rows_data, m, cols):
    """Confusion counts: rows = acted emotions present (ORDER), columns = cols."""
    acted_set = {r["acted_emotion"] for r in rows_data}
    rows = [e for e in ORDER if e in acted_set]
    c = np.zeros((len(rows), len(cols)), dtype=int)
    for r in rows_data:
        p = r[f"pred_{m}"]
        if p in cols:
            c[rows.index(r["acted_emotion"]), cols.index(p)] += 1
    return rows, c


def draw_cm(ax, pct, rows, cols, xlabel="Predicted", ylabel="Acted", fs=None):
    """Blues heatmap with the % in every cell, thin dark lines between cells and a dark border."""
    n = max(len(rows), len(cols))
    fs = fs or (14 if n <= 5 else (13 if n == 6 else 12))
    im = ax.pcolormesh(np.arange(len(cols) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5,
                       pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = pct[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=fs,
                    color="white" if v >= 55 else ("#222222" if v >= 0.5 else "#9a9a9a"))
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(rows)), rows)
    # cell lines: minor ticks on the cell edges, grid drawn on them
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color=GRID_C, linewidth=GRID_W)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ax.spines.values():  # dark border around the whole matrix
        s.set_visible(True)
        s.set_color(GRID_C)
        s.set_linewidth(BORDER_W)
    return im


def add_colorbar(fig, im, **kw):
    cb = fig.colorbar(im, **kw)
    cb.solids.set_rasterized(False)
    cb.set_label("% of acted clips", fontsize=13)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_edgecolor(GRID_C)
    return cb


# ---------------------------------------------------------------- Step 6: confusion matrices
print("Step 6: confusion matrices of the two top models (single and side by side)")
for ds in DATASETS:
    rows_data = data[ds]
    predicted = {r[f"pred_{m}"] for r in rows_data for m in PAIR_MODELS}
    own = {r["acted_emotion"] for r in rows_data} - {"excited"}
    cols = [e for e in EMOTIONS if e in predicted | own]
    mats = {}
    for m in PAIR_MODELS:
        rows, c = cm_counts(rows_data, m, cols)
        pct = row_percent(c)
        mats[m] = (rows, pct)
        fig, ax = plt.subplots(figsize=(5.8, 5.2))
        im = draw_cm(ax, pct, rows, cols)
        add_colorbar(fig, im, ax=ax, fraction=0.046, pad=0.03)
        fig.tight_layout()
        save_figure(fig, f"cm_{m}_{ds}")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    for ax, m in zip(axes, PAIR_MODELS):
        rows, pct = mats[m]
        im = draw_cm(ax, pct, rows, cols, ylabel="Acted" if m == PAIR_MODELS[0] else "")
        tag = "  (seen in training)" if ds in SEEN[m] else "  (not in training)"
        ax.set_title(f"{MODEL_NAMES[m]}\naccuracy {res[m][ds]['accuracy'] * 100:.1f}%{tag}", fontsize=14)
    add_colorbar(fig, im, ax=axes, fraction=0.035, pad=0.02)
    save_figure(fig, f"cm_pair_{ds}")
    print(f"  {ds}: columns {cols}")

# ---------------------------------------------------------------- Step 7: CREMA-D actor vs listeners matrix
print("Step 7: CREMA-D acted vs listener-majority matrix")
fig, ax = plt.subplots(figsize=(7.0, 5.6))
im = draw_cm(ax, row_percent(mat), cr_rows, cr_cols, xlabel="Listener majority (voice only)", ylabel="Acted")
add_colorbar(fig, im, ax=ax, fraction=0.046, pad=0.03)
fig.tight_layout()
save_figure(fig, "crema_acted_vs_listener")

# ---------------------------------------------------------------- Step 8: MCAE-SPPS recognition histogram
print("Step 8: MCAE-SPPS recognition-rate histogram")
fig, ax = plt.subplots(figsize=(8, 4.8))
bins = np.arange(0.4625, 1.0376, 0.025)  # one bar per possible share k/40 (from 19/40 to 40/40)
hvals, _, _ = ax.hist(rr, bins=bins, color=UNSEEN_C, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_W)
ax.set_ylim(0, hvals.max() * 1.18)
mean = rr.mean()
ax.axvline(mean, color="#333333", linestyle="--", linewidth=1.5)
ax.text(mean - 0.008, hvals.max() * 1.09, f"mean {mean * 100:.0f}% (≈{mean * 40:.0f} of 40 listeners)",
        fontsize=14, color="#333333", ha="right")
n_half = int(np.isclose(rr, 0.5).sum())  # clips recognised by exactly 20 of 40 listeners
ax.annotate(f"{n_half} clips were recognised\nby only half of the listeners", xy=(0.5, n_half + 4),
            xytext=(0.515, hvals.max() * 0.74), ha="left", va="bottom", fontsize=13, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.2))
ax.set_xlim(bins[0], 1.0125)
ax.set_xlabel("Share of the 40 listeners who chose the intended emotion")
ax.set_ylabel("Number of clips")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout()
save_figure(fig, "mcae_recognition_hist")
print(f"  {n_half} clips at 50%, mean {mean * 100:.1f}%")

# ---------------------------------------------------------------- Step 9: seen vs unseen bars
print("Step 9: mean accuracy on datasets in vs not in each model's training data")
fig, ax = plt.subplots(figsize=(9, 5.2))
w = 0.36
for i, m in enumerate(MODELS):
    s = stats["models"][m]
    if s["mean_accuracy_in_training_data"] is not None:
        bars = [(i - w / 2, s["mean_accuracy_in_training_data"], SEEN_C),
                (i + w / 2, s["mean_accuracy_not_in_training_data"], UNSEEN_C)]
    else:
        bars = [(i, s["mean_accuracy_not_in_training_data"], UNSEEN_C)]
    for x, v, c in bars:
        ax.bar(x, v * 100, width=w, color=c, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_W)
        ax.text(x, v * 100 + 1.5, f"{v * 100:.1f}", ha="center", va="bottom", fontsize=14)
TICK = {"emotion2vec_plus_large": "emotion2vec+\nlarge", "meralion_ser_v1": "MERaLiON-\nSER-v1", "melati_emotion": "Melati",
        "vox_profile_whisper": "Vox-Profile\nWhisper", "vox_profile_wavlm": "Vox-Profile\nWavLM"}
ax.set_xticks(range(len(MODELS)), [TICK[m] for m in MODELS])
ax.set_ylabel("Mean accuracy (%)")
ax.set_ylim(0, 105)
ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(handles=[Patch(facecolor=SEEN_C, edgecolor=BAR_EDGE, label="datasets in its training data"),
                   Patch(facecolor=UNSEEN_C, edgecolor=BAR_EDGE, label="datasets not in its training data")],
          loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.tight_layout()
save_figure(fig, "seen_vs_unseen")

# ---------------------------------------------------------------- Step 10: emotion counts
print("Step 10: clips per acted emotion, one panel per dataset")
fig, axes = plt.subplots(2, 3, figsize=(8, 6))
for ax, ds in zip(axes.flat, DATASETS):
    c = stats["datasets"][ds]["per_emotion_counts"]
    ks = list(c)
    ax.bar(range(len(ks)), [c[k] for k in ks], color=UNSEEN_C, edgecolor=BAR_EDGE, linewidth=BAR_EDGE_W, width=0.75)
    ax.set_xticks(range(len(ks)), ks, fontsize=11, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_title(f"{DATASET_NAMES[ds]} (n={stats['datasets'][ds]['clips_evaluated']:,})", fontsize=12.5)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
for ax in axes[:, 0]:
    ax.set_ylabel("Clips", fontsize=12)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save_figure(fig, "emotion_counts")
print(f"Done: assets/stats.json and {len(list(FIG.glob('*.png')))} PNG/SVG pairs in assets/fig/")

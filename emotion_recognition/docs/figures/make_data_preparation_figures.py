"""
Figures for docs/data_preparation_call_centre_audio.md (writes docs/figures/data_prep_*.png and .svg).

The "customer turn" is simulated: the six CREMA-D sample clips in data/samples/ (ODbL) are trimmed and joined with
pauses of chosen lengths, with faint noise added, so that every step of the segmentation can be shown on real speech.
The call-level figure uses made-up probabilities and is marked as illustrative.

Run from the emotion_recognition/ folder:  python docs/figures/make_data_preparation_figures.py
"""
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # emotion_recognition/
SAMPLES = ROOT / "data" / "samples"
OUT = ROOT / "docs" / "figures"
SR = 16000
MIN_PAUSE = 0.3      # a pause at least this long is a cut point (s)
SEED = 0

# The simulated turn: (clip, pause after it in seconds); "HAP_short" is the first 0.6 s of the happy clip ("okay")
TURN = [("NEU", 0.2), ("ANG", 0.8), ("DIS", 0.15), ("FEA", 0.7), ("HAP_short", 0.4), ("SAD", 0.1), ("HAP", 0.1), ("NEU", 0.0)]

GREEN, RED, GREY, BLUE, INK = "#2e8b57", "#c0392b", "#9aa4b1", "#0072b2", "#1b2a41"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})


def save_figure(fig, name):
    """Export the same original artwork as PNG and vector SVG."""
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    plt.close(fig)


def build_turn():
    """Join the trimmed sample clips with the pauses in TURN; return the waveform and the true speech intervals."""
    rng = np.random.default_rng(SEED)
    parts, speech, t = [], [], 0.0
    for name, pause in TURN:
        clip, _ = librosa.load(SAMPLES / f"1001_DFA_{name.split('_')[0]}_XX.wav", sr=SR)
        clip, _ = librosa.effects.trim(clip, top_db=35)
        if name.endswith("_short"):
            clip = clip[: int(0.6 * SR)]
        clip = clip / np.max(np.abs(clip)) * 0.8
        parts += [clip, np.zeros(int(pause * SR))]
        speech.append((t, t + len(clip) / SR))
        t += len(clip) / SR + pause
    wav = np.concatenate(parts)
    return wav + rng.normal(0, 0.003, len(wav)), speech


def vad(wav, top_db=35, frame=0.025, hop=0.010):
    """Energy VAD for the illustration: frames within top_db of the loudest frame are speech; returns intervals (s)."""
    rms = librosa.feature.rms(y=wav, frame_length=int(frame * SR), hop_length=int(hop * SR))[0]
    db = 20 * np.log10(rms + 1e-9)
    on = db > db.max() - top_db
    intervals, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        if not v and start is not None:
            intervals.append((start * hop, i * hop)); start = None
    if start is not None:
        intervals.append((start * hop, len(on) * hop))
    merged = [list(intervals[0])]                     # join speech separated by gaps shorter than 0.1 s (within words)
    for a, b in intervals[1:]:
        if a - merged[-1][1] < 0.15:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= 0.25]  # ignore clicks and breaths shorter than 0.25 s


def pause_cuts(intervals):
    """Cut points in the middle of every pause of at least MIN_PAUSE seconds."""
    return [(b + a2) / 2 for (a, b), (a2, b2) in zip(intervals, intervals[1:]) if a2 - b >= MIN_PAUSE]


def waveform(ax, wav, title):
    t = np.arange(len(wav)) / SR
    ax.plot(t, wav, color=GREY, linewidth=0.5)
    ax.set_xlim(0, t[-1]); ax.set_ylim(-1, 1.25); ax.set_yticks([])
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK)


def fig_vad(wav, iv):
    print("Figure 1: VAD")
    fig, ax = plt.subplots(figsize=(12, 3.2))
    waveform(ax, wav, "VAD marks where someone is speaking; the gaps are pauses")
    for a, b in iv:
        ax.axvspan(a, b, color=GREEN, alpha=0.18, linewidth=0)
    k = 0
    for (a, b), (a2, b2) in zip(iv, iv[1:]):
        gap = a2 - b
        if gap >= MIN_PAUSE:
            ax.annotate(f"pause {gap:.1f} s", ((b + a2) / 2, 1.08 if k % 2 == 0 else 0.86), ha="center", fontsize=11, color=RED)
            k += 1
    ax.set_xlabel("time (s)")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=GREEN, alpha=0.3)], labels=["speech (VAD)"], loc="lower right", frameon=False)
    fig.tight_layout(); save_figure(fig, "data_prep_vad")


def fig_fixed_vs_pause(wav, iv):
    print("Figure 2: fixed windows vs cutting at pauses")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True)
    total = len(wav) / SR
    waveform(axes[0], wav, "Fixed 4 s windows: cuts fall in the middle of words")
    for c in np.arange(4, total, 4):
        inside = any(a < c < b for a, b in iv)
        axes[0].axvline(c, color=RED if inside else GREEN, linewidth=2.2)
        if inside:
            axes[0].annotate("cuts speech", (c, 1.05), ha="center", fontsize=10, color=RED)
    waveform(axes[1], wav, "Cut at pauses ≥ 0.3 s and merge pieces < 2 s: every segment starts and ends in a pause")
    cuts = pause_cuts(iv)
    edges = [0.0] + cuts + [total]
    pieces = list(zip(edges, edges[1:]))
    # merge pieces shorter than 2 s with the next piece (the short "okay")
    merged = []
    for a, b in pieces:
        if merged and (merged[-1][1] - merged[-1][0] < 2.0):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    for c in cuts:
        kept = any(abs(c - b) < 1e-6 for _, b in merged[:-1])
        axes[1].axvline(c, color=GREEN if kept else GREY, linewidth=2.2 if kept else 1.2, linestyle="-" if kept else ":")
    for i, (a, b) in enumerate(merged):
        axes[1].annotate(f"segment {i + 1}: {b - a:.1f} s", ((a + b) / 2, -0.95), ha="center", fontsize=11, color=GREEN, fontweight="bold")
    axes[1].set_xlabel("time (s)")
    fig.tight_layout(); save_figure(fig, "data_prep_fixed_vs_pause")


def bar(ax, x, w, y, label, color, h=0.55, text_color="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12", facecolor=color, edgecolor="black", linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11, color=text_color, fontweight="bold")


def fig_length_rules():
    print("Figure 3: length rules")
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(-6.5, 30); ax.set_ylim(-0.4, 5.2); ax.axis("off")
    rows = [
        (4.3, "Short piece (< 2 s)", [(0, 1.2, "", RED), (1.6, 5.0, "5.0 s", BLUE)], [(0, 6.6, "merged: 6.6 s", GREEN)]),
        (2.8, "Good piece (3–10 s)", [(0, 6.0, "6.0 s", BLUE)], [(0, 6.0, "kept: 6.0 s", GREEN)]),
        (1.3, "Long piece (> 15 s)", [(0, 22.0, "22 s, no long pause", RED)], [(0, 7.2, "7.2 s", GREEN), (7.4, 7.2, "7.2 s", GREEN), (14.8, 7.2, "7.2 s", GREEN)]),
    ]
    for y, name, before, after in rows:
        ax.text(-6.3, y + 0.28, name, va="center", fontsize=12, fontweight="bold", color=INK)
        for x, w, lab, col in before:
            bar(ax, x * 0.55, w * 0.55, y, lab, col)
        ax.annotate("", xy=(15.2, y + 0.28), xytext=(13.0, y + 0.28), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
        for x, w, lab, col in after:
            bar(ax, 15.6 + x * 0.6, w * 0.6, y, lab, col)
    ax.text(0.33, 4.3 + 0.75, "1.2 s", ha="center", fontsize=11, color=RED, fontweight="bold")
    ax.text(0, 0.35, "before", fontsize=12, color=GREY); ax.text(15.6, 0.35, "after", fontsize=12, color=GREY)
    ax.text(0, -0.25, "Long pieces are split at their longest pauses; without any pause, every 8–10 s with 0.5 s overlap.", fontsize=11, color=INK)
    fig.tight_layout(); save_figure(fig, "data_prep_length_rules")


def fig_pipeline():
    print("Figure 4: pipeline")
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 3.6); ax.axis("off")
    steps = [("Call\nrecording", "#e3e8ef", INK), ("Customer turns\n(diarization)", "#e3e8ef", INK), ("Quality\nfilter", "#eef4fa", INK),
             ("VAD:\nfind pauses", "#eef4fa", INK), ("Cut, merge,\nsplit", "#eef4fa", INK), ("16 kHz mono\n+ metadata", "#eef4fa", INK)]
    w, gap = 1.75, 0.38
    for i, (t, f, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.9), w, 1.2, boxstyle="round,pad=0,rounding_size=0.1", facecolor=f, edgecolor=INK, linewidth=1))
        ax.text(x + w / 2, 2.5, t, ha="center", va="center", fontsize=12, color=c)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 2.5), xytext=(x + w + 0.03, 2.5), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    xe = 0.1 + 5 * (w + gap) + w / 2
    for xo, t, col in [(-3.6, "Labelling pool: 3–10 s\nhuman labels, A–D, pseudo-labels", GREEN), (-0.5, "Pre-training pool: 5–30 s\noption E, random crops", "#8e4ec6")]:
        ax.add_patch(FancyBboxPatch((xe + xo - 1.25, 0.1), 3.1, 1.05, boxstyle="round,pad=0,rounding_size=0.1", facecolor="white", edgecolor=col, linewidth=1.6))
        ax.text(xe + xo + 0.3, 0.62, t, ha="center", va="center", fontsize=11, color=col, fontweight="bold")
        ax.annotate("", xy=(xe + xo + 0.3, 1.17), xytext=(xe, 1.88), arrowprops=dict(arrowstyle="->", color=col, lw=1.4, connectionstyle="angle,angleA=0,angleB=90"))
    fig.tight_layout(); save_figure(fig, "data_prep_pipeline")


def fig_aggregation():
    print("Figure 5: from segments to a call-level result (illustrative values)")
    rng = np.random.default_rng(SEED)
    n = 40
    t = np.arange(n) * 15 + 7.5                       # one segment every 15 s of a 10-minute call
    anger = np.clip(0.08 + 0.75 * np.exp(-((t - 330) / 70) ** 2) + rng.normal(0, 0.05, n), 0, 1)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.bar(t, anger, width=9, color=[RED if a >= 0.5 else GREY for a in anger], edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, color=INK, linestyle="--", linewidth=1)
    ax.set_xlim(0, 610); ax.set_ylim(0, 1.05)
    ax.set_xlabel("time in the call (s)"); ax.set_ylabel("P(angry) per segment")
    share = (anger >= 0.5).mean()
    ax.set_title(f"Call-level summary: {share:.0%} of segments angry · peak {anger.max():.2f} at {t[anger.argmax()] / 60:.1f} min · calm at the end"
                 "   (illustrative values)", loc="left", fontsize=12, color=INK)
    fig.tight_layout(); save_figure(fig, "data_prep_aggregation")


if __name__ == "__main__":
    print("Step 1: build the simulated customer turn from data/samples/")
    wav, truth = build_turn()
    print(f"  {len(wav) / SR:.1f} s, {len(truth)} spoken pieces")
    print("Step 2: VAD")
    iv = vad(wav)
    print(f"  {len(iv)} speech intervals, cuts at {[round(c, 1) for c in pause_cuts(iv)]} s")
    fig_vad(wav, iv)
    fig_fixed_vs_pause(wav, iv)
    fig_length_rules()
    fig_pipeline()
    fig_aggregation()
    print(f"Done: 5 figures in {OUT.relative_to(ROOT)}/")

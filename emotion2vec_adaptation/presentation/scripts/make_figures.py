"""Figures for the deck, drawn at slide size: python scripts/make_figures.py
->  img/data_scale.png, img/ema_weights.png, img/block_weights.png

data_scale: audio hours on a log axis. Published numbers: 262 h (emotion2vec paper, Table 1, self-supervised
pre-training) and 42,526 h (emotion2vec+ large model card, filtered pseudo-labelled fine-tuning data). Internal
numbers are a REFERENCE, not a measurement: 30 batches (the real count is above 30) x 100K or 200K clips; the bar
spans the 2-10 s clip-length limits and the dot assumes a 5 s mean clip length (illustration only).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from style import ACCENT, ACCENT2, DARK_ORANGE, FILL_BLUE, FILL_ORANGE, GRID, INK, LINK, TEACHER, save

ORIGINAL_SSL_H = 262          # emotion2vec paper, Table 1
PLUS_LARGE_FT_H = 42_526      # emotion2vec+ large model card, fine-tuning data
BATCHES = 30                  # reference only; the real number of batches is above 30
LIMITS_S = (2, 10)            # clip-length limits of the internal data
MEAN_S = 5                    # assumed mean, illustration only


def hours(clips, seconds):
    return clips * seconds / 3600


def data_scale():
    rows = []
    for per_batch in (100_000, 200_000):
        clips = BATCHES * per_batch
        lo, hi, mid = hours(clips, LIMITS_S[0]), hours(clips, LIMITS_S[1]), hours(clips, MEAN_S)
        rows.append((per_batch, lo, hi, mid))
        print(f"  {BATCHES} x {per_batch:,} clips: {lo:,.0f}-{hi:,.0f} h; at {MEAN_S} s {mid:,.0f} h "
              f"= {mid / ORIGINAL_SSL_H:.1f}x {ORIGINAL_SSL_H} h = {100 * mid / PLUS_LARGE_FT_H:.1f}% of {PLUS_LARGE_FT_H:,} h")

    fig = plt.figure(figsize=(12.1, 3.9))
    ax = fig.add_axes([0.31, 0.2, 0.635, 0.76])
    base, top = 100, 100_000
    ys = [3, 2, 1, 0]
    labels = ["emotion2vec: self-supervised\npre-training (paper)",
              "emotion2vec+ large: pseudo-label\nfine-tuning (model card)",
              f"Internal reference:\n{BATCHES} batches × 100K clips",
              f"Internal reference:\n{BATCHES} batches × 200K clips"]
    h = 0.56
    ax.barh(ys[0], ORIGINAL_SSL_H - base, left=base, height=h, color=ACCENT, edgecolor=ACCENT)
    ax.barh(ys[1], PLUS_LARGE_FT_H - base, left=base, height=h, color=ACCENT, edgecolor=ACCENT)
    ax.text(ORIGINAL_SSL_H * 1.12, ys[0], f"{ORIGINAL_SSL_H} h", va="center", fontsize=19, fontweight="bold")
    ax.text(PLUS_LARGE_FT_H * 1.08, ys[1], f"{PLUS_LARGE_FT_H:,} h", va="center", fontsize=19, fontweight="bold")
    for y, (per_batch, lo, hi, mid) in zip(ys[2:], rows):
        ax.barh(y, hi - lo, left=lo, height=h, color=FILL_ORANGE, edgecolor=ACCENT2, linewidth=1.6)
        ax.plot([mid], [y], "o", ms=13, color=DARK_ORANGE, mec="white", mew=1.5, zorder=3)
        ax.text(lo / 1.08, y, f"{lo:,.0f}–{hi:,.0f} h", va="center", ha="right", fontsize=18)
        ax.text(hi * 1.08, y, f"● {mid:,.0f} h at 5 s", va="center", fontsize=19, fontweight="bold",
                color=DARK_ORANGE)
    ax.set_xscale("log")
    ax.set_xlim(base, top)
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=18, linespacing=1.05)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("audio hours (log scale)", fontsize=18)
    ax.grid(axis="x", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for t in ax.get_yticklabels():
        t.set_color(INK)
    save(fig, "data_scale")


EMA_DECAY = 0.999             # config/pretrain.yaml ema_decay


def ema_weights():
    """Weight of the student from j optimizer steps ago in the EMA teacher: (1 - tau) * tau**j (initial copy ignored)."""
    tau = EMA_DECAY
    j = np.arange(0, 5001)
    w = (1 - tau) * tau ** j * 1000            # in units of 1e-3
    horizon = round(1 / (1 - tau))             # 1,000 steps
    share = 1 - tau ** horizon                 # weight carried by the last `horizon` steps
    print(f"  EMA tau={tau}: time constant {horizon} steps; last {horizon} steps carry {100 * share:.0f}% of the weight")

    fig = plt.figure(figsize=(5.9, 3.55))
    ax = fig.add_axes([0.155, 0.22, 0.775, 0.72])
    ax.fill_between(j[:horizon + 1], w[:horizon + 1], color="#EDE5F5", lw=0)
    ax.plot(j, w, color=TEACHER, lw=2.6)
    ax.axvline(horizon, color=TEACHER, lw=1.4, ls=(0, (4, 3)))
    ax.text(horizon + 120, 0.86, f"last {horizon:,} steps:\n{100 * share:.0f}% of the weight", fontsize=18,
            color="#4B2A6E", fontweight="bold", va="top")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000", "4,000", "5,000"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("student steps ago", fontsize=18)
    ax.set_ylabel("weight (×10⁻³)", fontsize=18)
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    save(fig, "ema_weights")


# Option B mock test (CREMA-D at 8 kHz, 6 classes): learned softmax weights of the 12 blocks (layer_weights.json),
# values supplied with the mock-test results on 2026-11-17. Blocks 1-4: prenet, 5-12: shared Transformer blocks.
BLOCK_WEIGHTS = [0.0609, 0.0607, 0.063, 0.0629, 0.0693, 0.0719, 0.1132, 0.1287, 0.1183, 0.1027, 0.0861, 0.0622]


def block_weights():
    w = np.array(BLOCK_WEIGHTS)
    print(f"  block weights sum {w.sum():.4f}; max block {int(w.argmax()) + 1} ({w.max():.4f})")
    fig = plt.figure(figsize=(5.25, 3.75))
    ax = fig.add_axes([0.175, 0.2, 0.8, 0.74])
    x = np.arange(1, 13)
    colors = [FILL_BLUE] * 4 + [ACCENT] * 8
    ax.bar(x, w, width=0.72, color=colors, edgecolor=ACCENT, linewidth=1.3, zorder=2)
    ax.axhline(1 / 12, color=INK, lw=1.5, ls=(0, (4, 3)), zorder=3)
    ax.text(0.45, 1 / 12 + 0.004, "start: 1/12", fontsize=17, va="bottom", ha="left")
    ax.text(2.5, 0.137, "prenet", fontsize=17, ha="center", color=LINK, fontweight="bold")
    ax.text(8.5, 0.137, "shared", fontsize=17, ha="center", color=LINK, fontweight="bold")
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels([str(i) for i in x], fontsize=16)
    ax.set_xlim(0.4, 12.6)
    ax.set_ylim(0, 0.15)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_xlabel("encoder block", fontsize=18)
    ax.set_ylabel("learned weight", fontsize=18)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    save(fig, "block_weights")


if __name__ == "__main__":
    data_scale()
    ema_weights()
    block_weights()

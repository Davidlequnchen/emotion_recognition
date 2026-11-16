"""Figures and numbers from the integration run on synthetic DailyTalk calls (public data, CC BY-SA 4.0).

Reads (all local, nothing is downloaded):
  preparation_tools/local/integration/input/turns/   the synthetic turns (git-ignored, from tests/integration_dailytalk.py)
  preparation_tools/local/integration/output/        review/*.json, the group manifests and call_timeline.csv
  emotion_recognition/data/dailytalk/                 to rebuild the utterance joins with the integration seed
Writes: fig/vad_silero.png, fig/compare_methods.png, fig/gap_zoom.png, fig/listen_<turn>.png and
assets/data_stats.json (Silero pieces, method comparison, examples, the call used on the context slide).

Method comparison on the 5 long English turns (no alignment model; Silero VAD runs on the CPU for its own segments):
  Silero default (100 ms) and 300 ms silence rule  speech chunks of get_speech_timestamps, cut between two chunks
  fixed 7 s chop                                   cuts at 7, 14, 21 ... s
  greedy "first pause after 7 s"                   first allowed word gap at least 7 s after the previous cut
  our method                                       the cuts the pipeline wrote (whole-turn dynamic programming)
Every cut is located against the aligned words of the review JSON: inside a word, or in a word gap (at an
utterance join, after a sentence end, or elsewhere), and gets that gap's boundary score s (0 inside a word).
Needs the segmenter's environment (torch, torchaudio, silero-vad, soundfile, PyYAML). Run from anywhere:
  python preparation_tools/presentation/scripts/make_data_figures.py
"""
import csv
import importlib.util
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

from style import (ACCENT, ACCENT2, ASSETS, BAD, FILL_BLUE, FILL_GREEN, FILL_ORANGE, FILL_RED, GOOD, GRID, INK, TOOLS,
                   WAVE, plt, save)

sys.path.insert(0, str(TOOLS))
from segmenter import audio_io, vad  # noqa: E402

INTEG = TOOLS / "local" / "integration"
TURNS = INTEG / "input" / "turns"
OUT = INTEG / "output"
VAD_TURN = "d1562_long_en"        # Silero's own segments, and the method comparison figure
ZOOM_TURN = "d418_long_en"        # one chosen cut at an utterance join, close up
LISTEN_TURNS = ["d418_long_en", "d1562_long_en"]   # whole turns with playable clips: cuts at joins; cuts at sentence ends
FIXED_STEP, GREEDY_STEP = 7.0, 7.0


def load_integration_module():
    spec = importlib.util.spec_from_file_location("integration_dailytalk", TOOLS / "tests" / "integration_dailytalk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rebuild_truth(integ):
    """Run the integration script's own make_inputs() into a temporary folder (same seed, same turns), so the
    utterance joins are known; check that every rebuilt transcript equals the saved input."""
    real = integ.OUT
    with tempfile.TemporaryDirectory() as tmp:
        integ.OUT = Path(tmp)
        try:
            truth = integ.make_inputs()
            for turn_id in truth:
                rebuilt = (Path(tmp) / "input" / "turns" / f"{turn_id}.txt").read_text(encoding="utf-8").strip()
                saved = (TURNS / f"{turn_id}.txt").read_text(encoding="utf-8").strip()
                if rebuilt != saved:
                    raise ValueError(f"{turn_id}: rebuilt transcript differs from the saved input; joins would be wrong")
        finally:
            integ.OUT = real
    return truth


def review(turn_id):
    return json.loads((OUT / "review" / f"{turn_id}.json").read_text(encoding="utf-8"))


def read_turn(turn_id):
    path = next(p for p in TURNS.glob(f"{turn_id}.*") if p.suffix in (".wav", ".mp3"))
    data, rate, _ = audio_io.read_audio(path)
    return data, rate, audio_io.analysis_copy(data, rate)


def silero_pieces(probs, n_samples, min_silence_ms):
    from silero_vad.utils_vad import get_speech_timestamps_from_probs
    stamps = get_speech_timestamps_from_probs(list(probs), sampling_rate=16000, min_silence_duration_ms=min_silence_ms,
                                              return_seconds=True, audio_length_samples=n_samples)
    return [(s["start"], s["end"]) for s in stamps]


def locate(t, rv, joins):
    """Where a cut at time t falls: ('word' | 'join' | 'sentence' | 'gap' | 'edge', boundary score)."""
    for w in rv["words"]:
        if w["start"] < t < w["end"]:
            return "word", 0.0
    for g in rv["gaps"]:
        if g["start"] <= t <= g["end"]:
            if any(g["start"] <= b and a <= g["end"] for a, b in joins):
                return "join", g["s"]
            return ("sentence" if g["q"] >= 1.0 else "gap"), g["s"]
    return "edge", 0.0


def method_cuts(name, rv, probs, n_samples, start, end):
    """(pieces [(a, b)], cut times) for one method on one turn."""
    if name in ("silero_100", "silero_300"):
        pieces = silero_pieces(probs, n_samples, 100 if name == "silero_100" else 300)
        cuts = [(a2 + b1) / 2 for (a1, b1), (a2, b2) in zip(pieces, pieces[1:])]
        return pieces, cuts
    if name == "fixed":
        cuts = list(np.arange(start + FIXED_STEP, end - 1e-6, FIXED_STEP))
    elif name == "greedy":
        cuts, last = [], start
        for g in rv["gaps"]:
            if g["candidate"] and g["cut_time"] >= last + GREEDY_STEP and g["cut_time"] < end:
                cuts.append(g["cut_time"])
                last = g["cut_time"]
    else:                                             # our method: the cuts the pipeline wrote
        cuts = list(rv["dp"]["cuts_s"])
    bounds = [start, *cuts, end]
    return list(zip(bounds, bounds[1:])), cuts


METHODS = [("silero_100", "Silero, default (100 ms)"), ("silero_300", "Silero, 300 ms silence rule"),
           ("fixed", "fixed 7 s chop"), ("greedy", "greedy: first pause after 7 s"),
           ("ours", "our method: whole-turn DP")]


def frame_steps(probs):
    edges = np.arange(len(probs) + 1) * vad.FRAME_S
    return np.repeat(edges, 2)[1:-1], np.repeat(probs, 2)


# ---------------------------------------------------------------- figures
def fig_vad_silero(probs, pieces100, pieces300, duration):
    print(f"Figure: Silero's own segments ({VAD_TURN})")
    fig = plt.figure(figsize=(12.1, 3.45))
    ax = fig.add_axes([0.205, 0.6, 0.7, 0.37])
    x, y = frame_steps(probs)
    ax.fill_between(x, 0, y, color="#BBD5EC", lw=0)
    ax.plot(x, y, color=ACCENT, lw=1.0)
    ax.axhline(0.5, color=INK, lw=1.6, ls=(0, (5, 3)))
    ax.axhline(0.35, color=ACCENT2, lw=1.6, ls=(0, (2, 2)))
    ax.text(duration + 0.6, 0.62, "start 0.5", va="center", ha="left", fontsize=16)
    ax.text(duration + 0.6, 0.23, "end 0.35", va="center", ha="left", fontsize=16)
    ax.set_xlim(0, duration)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("p(speech)", labelpad=8)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    seg = fig.add_axes([0.205, 0.2, 0.7, 0.36])
    seg.set_xlim(0, duration)
    seg.set_ylim(0, 2)
    for yc, label, pieces in ((1.5, f"Silero, 100 ms: {len(pieces100)}", pieces100), (0.5, f"Silero, 300 ms: {len(pieces300)}", pieces300)):
        for a, b in pieces:
            seg.add_patch(Rectangle((a, yc - 0.33), b - a, 0.66, fc=FILL_ORANGE, ec=ACCENT2, lw=1.2))
        seg.text(-0.8, yc, label, ha="right", va="center", fontsize=17, fontweight="bold")
    seg.set_yticks([])
    seg.spines["left"].set_visible(False)
    seg.set_xticks(range(0, int(duration) + 1, 5))
    seg.set_xlabel("time in the turn (s)")
    save(fig, "vad_silero")


def fig_compare(turn_pieces, rv, joins, duration):
    print(f"Figure: five methods on one turn ({VAD_TURN})")
    fig = plt.figure(figsize=(12.1, 2.75))
    ax = fig.add_axes([0.255, 0.3, 0.73, 0.68])
    ax.set_xlim(0, duration)
    n = len(METHODS)
    ax.set_ylim(0, n)
    for a, b in joins:
        ax.axvspan(a, b, color=FILL_GREEN, lw=0)
    for k, (key, label) in enumerate(METHODS):
        yc = n - k - 0.5
        pieces, cuts = turn_pieces[key]
        ours = key == "ours"
        for a, b in pieces:
            short = b - a < 3
            ax.add_patch(Rectangle((a + 0.05, yc - 0.32), max(b - a - 0.1, 0.05), 0.64,
                                   fc=FILL_RED if short else (FILL_BLUE if ours else "#FFFFFF"),
                                   ec=BAD if short else (ACCENT if ours else INK), lw=1.3))
        for c in cuts:
            where, _ = locate(c, rv, joins)
            if where == "word":
                ax.plot([c], [yc + 0.43], marker="v", ms=11, color=BAD, zorder=4)
        ax.text(-0.6, yc, label, ha="right", va="center", fontsize=16, fontweight="bold" if ours else "normal")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(range(0, int(duration) + 1, 5))
    ax.text(-0.6, -0.55, "time in the turn (s)", ha="right", va="center", fontsize=17)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(fc=FILL_RED, ec=BAD, label="piece under 3 s"),
               Line2D([], [], marker="v", ls="", ms=11, color=BAD, label="cut inside a word"),
               Patch(fc=FILL_GREEN, ec=FILL_GREEN, label="utterance join")]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(0.99, -0.01), ncol=3, fontsize=16, frameon=False,
               handletextpad=0.4, columnspacing=1.2)
    save(fig, "compare_methods")


def fig_gap_zoom(turn_id, data, rate, probs):
    rv = review(turn_id)
    cut = rv["dp"]["cuts_s"][0]
    print(f"Figure: chosen cut at {cut} s in {turn_id}, close up")
    gap = min(rv["gaps"], key=lambda g: abs(g["cut_time"] - cut))
    a0, a1 = cut - 1.45, cut + 1.45
    words = [w for w in rv["words"] if w["end"] > a0 and w["start"] < a1]
    mono = data.mean(axis=1)
    t = np.arange(len(mono)) / rate
    keep = (t >= a0) & (t <= a1)
    fig = plt.figure(figsize=(6.6, 4.75))
    ax = fig.add_axes([0.03, 0.62, 0.94, 0.34])
    ax.axvspan(gap["start"], gap["end"], color=FILL_GREEN, lw=0)
    ax.plot(t[keep], mono[keep], color=WAVE, lw=0.5)
    peak = np.abs(mono[keep]).max()
    for w in words:
        if w["start"] >= a0 and w["end"] <= a1:          # label only words that are fully in the window
            ax.text((w["start"] + w["end"]) / 2, peak * 1.28, w["text"], ha="center", va="bottom", fontsize=17)
    ax.set_xlim(a0, a1)
    ax.set_ylim(-peak * 1.1, peak * 1.9)
    ax.axis("off")
    ax.text(0.5 * (gap["start"] + gap["end"]), -peak * 1.05, f"word gap {gap['length']:.2f} s", ha="center", va="top", fontsize=17)
    bx = fig.add_axes([0.03, 0.17, 0.94, 0.33])
    centres = vad.frame_centres(probs)
    inside = (centres >= a0) & (centres <= a1)
    in_gap = (centres >= gap["start"]) & (centres <= gap["end"])
    gap_min = probs[in_gap].min()
    quiet = in_gap & (probs <= gap_min + vad.QUIET_TOLERANCE)       # the quiet part of the gap
    colors = np.where(quiet, "#1D6B3F", np.where(in_gap, GOOD, "#9CC3E6"))
    heights = np.maximum(probs, np.where(in_gap, 0.025, 0.0))       # near-zero frames stay visible
    bx.bar(centres[inside], heights[inside], width=vad.FRAME_S * 0.82, color=colors[inside], lw=0)
    qs = centres[quiet]
    bx.plot([qs.min() - vad.FRAME_S / 2, qs.max() + vad.FRAME_S / 2], [0.12, 0.12], color="#1D6B3F", lw=3)
    bx.text(qs.min() + 0.04, 0.16, "quiet part", ha="left", va="bottom", fontsize=17)
    bx.axvline(gap["cut_time"], color=ACCENT, lw=2.4, ls=(0, (5, 3)))
    bx.plot([gap["cut_time"]], [0.03], marker="^", ms=12, color=ACCENT2, clip_on=False, zorder=4)
    bx.set_xlim(a0, a1)
    bx.set_ylim(0, 1.05)
    bx.spines["left"].set_visible(False)
    bx.set_yticks([])
    ticks = np.arange(np.ceil(a0 * 2) / 2, a1, 0.5)
    bx.set_xticks(ticks, [f"{v:.1f}" for v in ticks])
    bx.set_xlabel("time in the turn (s); bars: p(speech) per 32 ms")
    bx.text(gap["cut_time"] + 0.06, 0.72, "cut: middle of\nthe quiet part", ha="left", va="center", fontsize=17,
            fontweight="bold", linespacing=1.1)
    position = (gap["cut_time"] - gap["start"]) / (gap["end"] - gap["start"])
    save(fig, "gap_zoom")
    return {"turn": turn_id, "cut_s": gap["cut_time"], "gap_s": gap["length"], "p": gap["p"], "q": gap["q"],
            "v": gap["v"], "s": gap["s"], "quiet_frames": int(quiet.sum()), "cut_position_in_gap": round(float(position), 3),
            "words": [w["text"] for w in words]}


def fig_listen(turn_id, data, rate, probs, joins):
    print(f"Figure: listening example {turn_id}")
    rv = review(turn_id)
    cuts = rv["dp"]["cuts_s"]
    start, end = rv["span_s"]
    segs = list(zip([start, *cuts], [*cuts, end]))
    duration = data.shape[0] / rate
    mono = data.mean(axis=1)
    step = max(1, rate // 2000)
    t = np.arange(0, len(mono), step) / rate
    fig = plt.figure(figsize=(12.1, 3.8))
    left, width = 0.115, 0.875
    ax = fig.add_axes([left, 0.6, width, 0.38])
    for a, b in joins:
        ax.axvspan(a, b, color=FILL_GREEN, lw=0)
    ax.plot(t, mono[::step], color=WAVE, lw=0.4)
    for c in cuts:
        ax.axvline(c, color=ACCENT, lw=2.4)
    ax.set_xlim(0, duration)
    lim = np.abs(mono).max() * 1.05
    ax.set_ylim(-lim, lim * 1.55)
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.text(-0.4, 0, "audio", ha="right", va="center", fontsize=17, fontweight="bold")
    notes = []
    for c in cuts:
        where, s = locate(c, rv, joins)
        notes.append({"cut_s": c, "where": where, "s": s})
        label = {"join": "join", "sentence": "sentence end", "gap": "pause"}.get(where, where)
        ax.text(c + 0.15, lim * 1.3, label, ha="left", va="center", fontsize=16, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"), zorder=5)
    vx = fig.add_axes([left, 0.44, width, 0.14])
    x, y = frame_steps(probs)
    vx.fill_between(x, 0, y, color="#BBD5EC", lw=0)
    for c in cuts:
        vx.axvline(c, color=ACCENT, lw=2.4)
    vx.set_xlim(0, duration)
    vx.set_ylim(0, 1)
    vx.axis("off")
    vx.text(-0.4, 0.5, "VAD", ha="right", va="center", fontsize=17, fontweight="bold")
    wx = fig.add_axes([left, 0.33, width, 0.09])
    for w in rv["words"]:
        wx.add_patch(Rectangle((w["start"], 0.1), max(w["end"] - w["start"], 0.03), 0.8, fc=INK, lw=0))
    for c in cuts:
        wx.axvline(c, color=ACCENT, lw=2.4)
    wx.set_xlim(0, duration)
    wx.set_ylim(0, 1)
    wx.axis("off")
    wx.text(-0.4, 0.5, "words", ha="right", va="center", fontsize=17, fontweight="bold")
    sx = fig.add_axes([left, 0.17, width, 0.14])
    for a, b in segs:
        sx.add_patch(Rectangle((a + 0.08, 0.05), b - a - 0.16, 0.9, fc=FILL_BLUE, ec=ACCENT, lw=1.6))
        sx.text((a + b) / 2, 0.5, f"{b - a:.1f} s", ha="center", va="center", fontsize=17)
    sx.set_xlim(0, duration)
    sx.set_ylim(0, 1)
    sx.set_yticks([])
    sx.spines["left"].set_visible(False)
    sx.set_xticks(range(0, int(duration) + 1, 5))
    sx.set_xlabel("time in the turn (s)")
    sx.text(-0.4, 0.5, "segments", ha="right", va="center", fontsize=17, fontweight="bold")
    save(fig, f"listen_{turn_id}")
    return {"turn": turn_id, "duration_s": round(duration, 3), "segments_s": [[round(a, 3), round(b, 3)] for a, b in segs],
            "joins": len(joins), "cuts": notes, "axes_left": left, "axes_width": width}


def main():
    import soundfile
    import torch
    print(f"Versions: torch {torch.__version__}, silero-vad {version('silero-vad')}, soundfile {soundfile.__version__}, "
          f"numpy {np.__version__}")
    torch.set_num_threads(2)
    ASSETS.mkdir(parents=True, exist_ok=True)

    print("Step 1: rebuild the synthetic calls in a temporary folder (integration seed) to know the utterance joins")
    integ = load_integration_module()
    truth = rebuild_truth(integ)
    long_turns = sorted(t for t, v in truth.items() if v["kind"] == "long_en")
    for t in long_turns:
        print(f"  {t}: {truth[t]['duration']:.1f} s, {len(truth[t]['joins'])} joins")

    print("Step 2: Silero probabilities (the pipeline's own functions) and the five methods on the long English turns")
    model = vad.load_vad()
    params_16k = sum(p.numel() for n, p in model.named_parameters() if n.startswith("_model."))
    stats = {"silero_vad_version": version("silero-vad"), "silero_16k_parameters": params_16k, "turns": {}}
    totals = {k: {"pieces": [], "cuts": 0, "word": 0, "join": 0, "sentence": 0, "scores": []} for k, _ in METHODS}
    cache = {}
    for t in long_turns:
        data, rate, wave = read_turn(t)
        probs = vad.speech_probabilities(model, wave)
        cache[t] = (data, rate, probs)
        rv = review(t)
        start, end = rv["span_s"]
        joins = truth[t]["joins"]
        per = {}
        for key, _ in METHODS:
            pieces, cuts = method_cuts(key, rv, probs, wave.numel(), start, end)
            per[key] = (pieces, cuts)
            tot = totals[key]
            tot["pieces"] += [b - a for a, b in pieces]
            tot["cuts"] += len(cuts)
            for c in cuts:
                where, s = locate(c, rv, joins)
                tot["scores"].append(s)
                tot["word"] += where == "word"
                tot["join"] += where == "join"
                tot["sentence"] += where == "sentence"
        stats["turns"][t] = {key: {"pieces": [[round(a, 3), round(b, 3)] for a, b in v[0]], "cuts": [round(c, 3) for c in v[1]]}
                             for key, v in per.items()}
        if t == VAD_TURN:
            vad_pieces = per
    positions = []
    for t in long_turns:
        rv = review(t)
        for c in rv["dp"]["cuts_s"]:
            g = min(rv["gaps"], key=lambda g_: abs(g_["cut_time"] - c))
            positions.append((g["cut_time"] - g["start"]) / (g["end"] - g["start"]))
    stats["cut_position_in_gap"] = [round(min(positions), 3), round(max(positions), 3)]
    print(f"  cut position within the chosen gap (0 = start, 1 = end): {min(positions):.2f}-{max(positions):.2f}")
    table = {}
    for key, label in METHODS:
        tot = totals[key]
        p = np.array(tot["pieces"])
        table[key] = {"label": label, "pieces": int(len(p)), "median_s": round(float(np.median(p)), 2),
                      "min_s": round(float(p.min()), 2), "max_s": round(float(p.max()), 2),
                      "under_3s": int((p < 3).sum()), "over_15s": int((p > 15).sum()), "cuts": tot["cuts"],
                      "cuts_inside_word": tot["word"], "cuts_at_join": tot["join"], "cuts_at_sentence_end": tot["sentence"],
                      "join_or_sentence_share": round((tot["join"] + tot["sentence"]) / tot["cuts"], 4) if tot["cuts"] else None,
                      "mean_s_at_cuts": round(float(np.mean(tot["scores"])), 3) if tot["scores"] else None}
        print(f"  {label}: {table[key]}")
    stats["methods"] = table
    stats["long_turns"] = {t: {"duration_s": round(truth[t]["duration"], 3), "joins": len(truth[t]["joins"])} for t in long_turns}

    print("Step 3: figures")
    data, rate, probs = cache[VAD_TURN]
    dur = data.shape[0] / rate
    fig_vad_silero(probs, vad_pieces["silero_100"][0], vad_pieces["silero_300"][0], dur)
    fig_compare(vad_pieces, review(VAD_TURN), truth[VAD_TURN]["joins"], dur)
    stats["vad_turn"] = {"turn": VAD_TURN, "duration_s": round(dur, 3),
                         **{k: len(v[0]) for k, v in vad_pieces.items()}}
    data, rate, probs = cache[ZOOM_TURN]
    stats["zoom"] = fig_gap_zoom(ZOOM_TURN, data, rate, probs)
    stats["listen"] = {}
    for t in LISTEN_TURNS:
        data, rate, probs = cache[t]
        stats["listen"][t] = fig_listen(t, data, rate, probs, truth[t]["joins"])
    timeline = list(csv.DictReader(open(OUT / "call_timeline.csv", newline="", encoding="utf-8")))
    stats["context_call"] = [{k: r[k] for k in ("order_in_call", "segment_id", "group", "reason", "duration_s",
                                                 "gap_to_prev_s", "prev_group", "next_group")}
                             for r in timeline if r["call_id"] == "dailytalk_d1562"]
    (ASSETS / "data_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print("  wrote assets/data_stats.json")
    print("Done.")


if __name__ == "__main__":
    main()

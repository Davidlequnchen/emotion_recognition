#!/usr/bin/env python3
"""Integration check on synthetic calls built from the local public DailyTalk copy (CC BY-SA 4.0).

Each synthetic call is one DailyTalk dialogue; its "customer" is the speaker with the most audio. The call has three
customer turns, in this order (turns.csv gives call_id, turn_index 0/2/4 and the call time, with a fixed gap for
the removed agent turn in between):
  1. a long English turn: consecutive utterances of the customer joined with known pauses (long and short, like a
     slow speaker), 25-35 s; one call has it as MP3;
  2. a Spanish-text turn over the next utterance(s): long (> 10 s) in some calls, short in the others;
  3. a short English turn: the next single utterance.
The full pipeline runs on them. Reported: groups and counts, segment lengths, distance of every cut to the nearest
utterance join, energy at the cut, text and audio coverage, alignment confidences, and whether the prev/next links
and call-time gaps of every call are correct. Outputs go to preparation_tools/local/integration/ (git-ignored).
Run from anywhere; not part of unittest.
"""
import copy
import csv
import json
import platform
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DAILYTALK = ROOT.parent / "emotion_recognition" / "data" / "dailytalk"
OUT = ROOT / "local" / "integration"
N_CALLS = 5                  # DailyTalk dialogues with the most audio from one speaker
LENGTH_RANGE = (25.0, 35.0)  # seconds per long English turn
SPANISH_LONG_MIN = 11.0      # seconds of audio under a long Spanish-text turn
LONG_PAUSE = (0.9, 1.4)      # inserted between some utterances ...
SHORT_PAUSE = (0.15, 0.35)   # ... and shorter ones between the others
LONG_SHARE = 0.4
AGENT_GAP = 4.0              # call time between two customer turns (the removed agent turn)
SEED = 7
MIN_FREE_MEMORY = 25         # percent; skip the model run below this (macOS)
CPU_THREADS = 4
SPANISH_LONG = ("Hola, buenos días. Llamo porque me cobraron dos veces en mi cuenta este mes. "
                "Ya hablé con ustedes la semana pasada y nadie me ayudó con el problema. "
                "Quiero saber por qué pasó esto y cuándo me van a devolver el dinero, por favor.")
SPANISH_SHORT = "Sí, gracias. Está bien, señora."
EXPECTED = {"long_en": ("final", "cut"), "es": ("non_english", "whole_language"),
            "short_en": ("final", "whole_short")}


def free_memory_percent():
    if platform.system() != "Darwin":
        return 100
    text = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True).stdout
    return int(re.search(r"free percentage: (\d+)%", text).group(1))


def load_utterances():
    with open(DAILYTALK / "utterances.csv", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: (int(r["dialogue_id"]), int(r["turn_index"])))
    return rows


def read_mono(path):
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), rate


def build_turn(utterances, rng, target):
    """Concatenate utterances with known pauses until target seconds.
    Returns (audio, rate, text, joins [(start, end)], number of utterances used)."""
    pieces, texts, joins, t, rate = [], [], [], 0.0, None
    for row in utterances:
        audio, rate = read_mono(DAILYTALK / row["audio_path"])
        if pieces:
            pause = rng.uniform(*(LONG_PAUSE if rng.random() < LONG_SHARE else SHORT_PAUSE))
            noise = np.random.default_rng(len(pieces)).standard_normal(int(pause * rate)).astype(np.float32)
            pieces.append(noise * 10 ** (-60 / 20))                    # quiet room noise, -60 dBFS
            joins.append((t, t + pause))
            t += pause
        pieces.append(audio)
        texts.append(row["text"].strip())
        t += len(audio) / rate
        if t >= target:
            break
    return np.concatenate(pieces), rate, " ".join(texts), joins, len(texts)


def pick_dialogues(rows):
    """The N_CALLS dialogues whose busiest speaker has the most audio: (dialogue id, that speaker's utterances)."""
    audio = {}
    for r in rows:
        key = (r["dialogue_id"], r["speaker_id"])
        audio[key] = audio.get(key, 0.0) + float(r["duration_s"])
    best = {}
    for (dialogue, speaker), seconds in audio.items():
        if seconds > best.get(dialogue, ("", 0.0))[1]:
            best[dialogue] = (speaker, seconds)
    chosen = sorted(best, key=lambda d: -best[d][1])[:N_CALLS]
    return [(d, [r for r in rows if r["dialogue_id"] == d and r["speaker_id"] == best[d][0]]) for d in chosen]


def make_inputs():
    rows = load_utterances()
    rng = random.Random(SEED)
    turns_dir = OUT / "input" / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    truth, csv_rows = {}, []
    for k, (dialogue, utterances) in enumerate(pick_dialogues(rows)):
        call_id, start, long_spanish = f"dailytalk_d{dialogue}", 0.0, k % 2 == 0
        n_es = 1                                     # utterances under the Spanish text, taken from the end
        while long_spanish and sum(float(r["duration_s"]) for r in utterances[-1 - n_es:-1]) < SPANISH_LONG_MIN:
            n_es += 1
        plan = [("long_en", utterances[:-1 - n_es], rng.uniform(*LENGTH_RANGE), None),
                ("es", utterances[-1 - n_es:-1], 1e9, SPANISH_LONG if long_spanish else SPANISH_SHORT),
                ("short_en", utterances[-1:], 1e9, None)]
        for order, (kind, chosen, target, text) in enumerate(plan):
            audio, rate, en_text, joins, n = build_turn(chosen, rng, target)
            turn_id = f"d{dialogue}_{kind}"
            suffix = ".mp3" if kind == "long_en" and k == N_CALLS - 1 else ".wav"   # one MP3 turn
            sf.write(str(turns_dir / f"{turn_id}{suffix}"), audio, rate, format="MP3" if suffix == ".mp3" else "WAV",
                     **({} if suffix == ".mp3" else {"subtype": "PCM_16"}))
            (turns_dir / f"{turn_id}.txt").write_text((text or en_text) + "\n", encoding="utf-8")
            duration = len(audio) / rate
            truth[turn_id] = {"call_id": call_id, "order": order, "kind": kind, "joins": joins,
                              "duration": duration, "start": start}
            csv_rows.append([turn_id, call_id, order * 2, f"{start:.3f}"])
            print(f"  {call_id} turn {order}: {turn_id}{suffix}, {n} utterances, {duration:.1f} s, "
                  f"{len(joins)} joins ({sum(b - a >= 0.9 for a, b in joins)} long)")
            start += duration + AGENT_GAP
    with open(OUT / "input" / "turns.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["turn_id", "call_id", "turn_index", "turn_start_in_call_s"])
        writer.writerows(csv_rows)
    return truth


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check_outputs(truth, rows):
    """Per turn: expected group, every word once, exact clip slices, and where the cuts of long turns land.
    Returns the number of turns that did not reach their expected group."""
    from segmenter.textnorm import tokenize
    out = OUT / "output"
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    stats = {"cuts": 0, "at_join": 0, "in_inserted": 0, "levels": [], "distance": [], "long_joins": 0,
             "long_joins_cut": 0}
    unexpected = 0
    print("Checks per turn")
    for turn_id, segs in sorted(by_turn.items()):
        source = next(p for p in (OUT / "input" / "turns").glob(f"{turn_id}.*") if p.suffix in (".wav", ".mp3"))
        data, rate = sf.read(str(source), dtype="float64", always_2d=True)
        text = (OUT / "input" / "turns" / f"{turn_id}.txt").read_text(encoding="utf-8").strip()
        pieces = [(out / r["transcript_path"]).read_text(encoding="utf-8").strip() for r in segs]
        # every word exactly once, in order
        assert [w.text for w in tokenize(" ".join(pieces))] == [w.text for w in tokenize(text)], turn_id
        # clips are exact, contiguous slices of the source
        position = round(float(segs[0]["segment_start_in_turn_s"]) * rate)
        for r in segs:
            clip, clip_rate = sf.read(str(out / r["audio_path"]), dtype="float64", always_2d=True)
            assert clip_rate == rate
            if source.suffix == ".wav":
                assert np.array_equal(clip, data[position:position + len(clip)]), turn_id
            position += len(clip)
        assert abs(position / rate - float(segs[-1]["segment_end_in_turn_s"])) < 1e-3
        lengths = [float(r["duration_s"]) for r in segs]
        got = (segs[0]["group"], segs[0]["segmentation_status"])
        expected = EXPECTED[truth[turn_id]["kind"]]
        unexpected += got != expected
        line = (f"  {turn_id}: {got[0]}/{got[1]}, {len(segs)} segments {lengths}"
                + ("" if got == expected else f"  (expected {expected[0]}/{expected[1]}: {segs[0]['reason']})"))
        if truth[turn_id]["kind"] == "long_en" and len(segs) > 1:
            joins = truth[turn_id]["joins"]
            review = json.loads((out / segs[0]["review_path"]).read_text(encoding="utf-8"))
            mono = data.mean(axis=1)
            speech_rms = np.sqrt(np.mean(mono[np.abs(mono) > 1e-3] ** 2))
            notes = []
            for r in segs[1:]:
                cut = float(r["segment_start_in_turn_s"])
                gap = min(review["gaps"], key=lambda g: abs(g["cut_time"] - cut))   # the chosen word gap
                at_join = any(gap["start"] <= b and a <= gap["end"] for a, b in joins)
                window = mono[max(0, round((cut - 0.01) * rate)):round((cut + 0.01) * rate)]
                level = 20 * np.log10(np.sqrt(np.mean(window ** 2)) / speech_rms + 1e-12)
                stats["cuts"] += 1
                stats["at_join"] += at_join
                stats["in_inserted"] += any(a <= cut <= b for a, b in joins)
                stats["levels"].append(level)
                stats["distance"].append(min(0.0 if a <= cut <= b else min(abs(cut - a), abs(cut - b))
                                             for a, b in joins))
                where = "join" if at_join else f"pause inside an utterance (q={gap['q']})"
                notes.append(f"{cut:.2f}s {where}, gap {gap['length']:.2f}s, s={gap['s']:.2f}, {level:.0f} dB")
            long_joins = [(a, b) for a, b in joins if b - a >= LONG_PAUSE[0]]
            cuts = [float(r["segment_start_in_turn_s"]) for r in segs[1:]]
            stats["long_joins"] += len(long_joins)
            stats["long_joins_cut"] += sum(any(a - 0.5 <= c <= b + 0.5 for c in cuts) for a, b in long_joins)
            line += "\n      " + "\n      ".join(notes)
        print(line)
    if stats["cuts"]:
        print(f"Cuts in long English turns: {stats['cuts']}; in the word gap at an utterance join: "
              f"{stats['at_join']} (inside the inserted pause itself: {stats['in_inserted']}); others at natural "
              f"pauses inside an utterance. Distance to the inserted pause: median {np.median(stats['distance']):.2f} s, "
              f"max {max(stats['distance']):.2f} s. Level at the cut (20 ms vs speech RMS): median "
              f"{np.median(stats['levels']):.0f} dB, max {max(stats['levels']):.0f} dB. Long inserted pauses "
              f"with a cut within 0.5 s: {stats['long_joins_cut']} of {stats['long_joins']}.")
    return unexpected


def group_stats(rows, cfg):
    """Turns and files per group, and the lengths of the final segments."""
    for group in cfg["paths"]["groups"]:
        mine = [r for r in rows if r["group"] == group]
        print(f"  {group}: {len({r['turn_id'] for r in mine})} turns, {len(mine)} files")
    lengths = sorted(float(r["duration_s"]) for r in rows if r["group"] == "final" and int(r["segment_count"]) > 1)
    if lengths:
        inside = sum(cfg["segmentation"]["min_s"] <= d <= cfg["segmentation"]["max_s"] for d in lengths)
        print(f"  final segments of cut turns: {len(lengths)}, {lengths[0]:.2f}-{lengths[-1]:.2f} s, median "
              f"{np.median(lengths):.2f} s, mean {np.mean(lengths):.2f} s; inside the window: {inside} of {len(lengths)}")


def check_links(truth, cfg):
    """Every call in call_timeline.csv: items in turn order, names sorted in time order, prev/next links that
    cross the groups correctly, and call-time gaps (0 inside a turn, AGENT_GAP between turns).
    Returns the number of calls with a problem."""
    out = OUT / "output"
    timeline = read_csv(out / "call_timeline.csv")
    in_manifests = sum(len(read_csv(out / folder / "manifest.csv")) for folder in cfg["paths"]["groups"].values())
    assert in_manifests == len(timeline), (in_manifests, len(timeline))
    calls = {}
    for r in timeline:
        calls.setdefault(r["call_id"], []).append(r)
    bad = 0
    for call_id, items in calls.items():
        problems = []
        if [int(r["order_in_call"]) for r in items] != list(range(len(items))):
            problems.append("order_in_call")
        if [truth[r["turn_id"]]["order"] for r in items] != sorted(truth[r["turn_id"]]["order"] for r in items):
            problems.append("turn order")
        if [r["segment_id"] for r in items] != sorted(r["segment_id"] for r in items):
            problems.append("segment ids not in time order")
        for k, r in enumerate(items):
            prev, nxt = (items[k - 1] if k else None), (items[k + 1] if k + 1 < len(items) else None)
            if (r["prev_segment_id"], r["prev_group"]) != ((prev["segment_id"], prev["group"]) if prev else ("", "")):
                problems.append(f"prev link of {r['segment_id']}")
            if (r["next_segment_id"], r["next_group"]) != ((nxt["segment_id"], nxt["group"]) if nxt else ("", "")):
                problems.append(f"next link of {r['segment_id']}")
            if prev:
                expected = 0.0 if prev["turn_id"] == r["turn_id"] else AGENT_GAP
                if abs(float(r["gap_to_prev_s"]) - expected) > 0.003:
                    problems.append(f"gap before {r['segment_id']}: {r['gap_to_prev_s']} (expected {expected})")
        runs = []
        for r in items:
            if runs and runs[-1][0] == r["group"]:
                runs[-1][1] += 1
            else:
                runs.append([r["group"], 1])
        order = " -> ".join(f"{g} x{n}" if n > 1 else g for g, n in runs)
        crossings = sum(1 for r in items if r["next_group"] and r["next_group"] != r["group"])
        print(f"  {call_id}: {len(items)} items: {order}; {crossings} links cross groups; "
              + ("OK" if not problems else "PROBLEMS: " + "; ".join(problems)))
        bad += bool(problems)
    print(f"Links: {len(calls) - bad} of {len(calls)} calls correct ({len(timeline)} items in call_timeline.csv, "
          f"{in_manifests} in the group manifests)")
    return bad


def alignment_stats():
    confidences, means = [], []
    for path in sorted((OUT / "output" / "review").glob("*.json")):
        review = json.loads(path.read_text(encoding="utf-8"))
        if review["alignment"].get("mean_confidence") is not None:
            means.append(review["alignment"]["mean_confidence"])
            confidences += [w["confidence"] for w in review["words"] if not w["wildcard"]]
    if confidences:
        q = np.percentile(confidences, [1, 5, 10, 50])
        print(f"Alignment: {len(means)} turns, mean confidence per turn {min(means):.2f}-{max(means):.2f}; "
              f"word confidence p1 {q[0]:.2f}, p5 {q[1]:.2f}, p10 {q[2]:.2f}, median {q[3]:.2f}")


def main():
    print(f"Step 0: free memory {free_memory_percent()}% (need {MIN_FREE_MEMORY}%)")
    if free_memory_percent() < MIN_FREE_MEMORY:
        print("Skipped: not enough free memory for the model run.")
        return 2
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)                                    # only the git-ignored integration folder
    print(f"Step 1: building synthetic calls from {DAILYTALK}")
    truth = make_inputs()
    cfg = copy.deepcopy(yaml.safe_load((ROOT / "config" / "segment_turns.yaml").read_text(encoding="utf-8")))
    cfg["paths"].update(input_dir="local/integration/input/turns", turns_csv="local/integration/input/turns.csv",
                        output_dir="local/integration/output")
    cfg["cpu_threads"] = CPU_THREADS
    print("Step 2: running the pipeline")
    from segmenter.pipeline import run
    rows, failures = run(cfg, ROOT)
    print("Step 3: checking outputs")
    unexpected = check_outputs(truth, rows)
    print("Groups")
    group_stats(rows, cfg)
    alignment_stats()
    print("Context links per call")
    bad_calls = check_links(truth, cfg)
    return 1 if failures or unexpected or bad_calls else 0


if __name__ == "__main__":
    sys.exit(main())

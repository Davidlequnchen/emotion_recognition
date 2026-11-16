"""Turn segmenter: customer turns (audio + transcript) in, labelling-ready segments out (doc §2-§6).

Every turn ends in exactly one of four groups (folder names in the YAML, same layout in each):
  final             English turns <= keep_whole_max_s exported whole, and longer English turns cut by the dynamic
                    programme (or kept whole when "no cut" is cheapest). Only this group is labelled.
  non_english       turns in a language without an alignment model (es, mixed), any length, exported whole
  alignment_issues  long English turns whose alignment failed or failed the checks, or with no text; whole
  boundary_issues   long English turns with no partition inside min_s-max_s; whole

segmentation_status: whole_short, whole_dp, cut (final); whole_language (non_english); whole_alignment,
whole_no_text (alignment_issues); whole_boundary (boundary_issues).

Turns are processed call by call. After the last turn of a call, every exported item of that call (all groups
together, in call order) gets its order and prev/next links, and the call's rows are appended to the group
manifests and to call_timeline.csv (crash-safe per call).
"""
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import time

import torch

from . import audio_io, vad
from .alignment import Aligner
from .boundaries import assign_words, best_partition, boundary_score, typical_pause
from .language import detect_language
from .textnorm import has_punctuation, tokenize

GROUPS = ["final", "non_english", "alignment_issues", "boundary_issues"]
LINK_FIELDS = ["order_in_call", "prev_segment_id", "prev_group", "next_segment_id", "next_group",
               "gap_to_prev_s", "gap_to_next_s"]
MANIFEST_FIELDS = [
    "segment_id", "group", "reason", "call_id", "customer_id", "turn_id", "turn_index", "turn_order",
    "segment_index", "segment_count", *LINK_FIELDS,
    "audio_path", "transcript_path", "review_path", "source_audio", "source_audio_sha256", "source_transcript",
    "source_transcript_sha256", "source_sample_rate", "source_channels", "clip_subtype",
    "turn_start_in_call_s", "segment_start_in_turn_s", "segment_end_in_turn_s", "segment_start_in_call_s",
    "segment_end_in_call_s", "call_time_status", "duration_s", "speech_duration_s", "speech_ratio",
    "low_speech_ratio", "language", "segmentation_status", "alignment_model", "alignment_status",
    "alignment_mean_confidence", "alignment_unaligned_share", "start_boundary_score", "end_boundary_score",
    "word_count", "segmentation_version",
]
TIMELINE_FIELDS = [
    "call_id", "order_in_call", "segment_id", "group", "reason", "turn_id", "turn_order", "segment_index",
    "segment_count", "audio_path", "transcript_path", "language", "duration_s", "segment_start_in_turn_s",
    "segment_end_in_turn_s", "turn_start_in_call_s", "segment_start_in_call_s", "segment_end_in_call_s",
    "call_time_status", "gap_to_prev_s", "gap_to_next_s", "prev_segment_id", "prev_group", "next_segment_id",
    "next_group",
]
TUNED_SECTIONS = ["language", "vad", "alignment", "segmentation", "scoring"]


def settings_version(cfg):
    """Name from the YAML plus a short hash of every setting that changes the cuts."""
    tuned = json.dumps({key: cfg[key] for key in TUNED_SECTIONS}, sort_keys=True)
    return f"{cfg['settings_version']}+{hashlib.sha256(tuned.encode()).hexdigest()[:8]}"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


# ---------------------------------------------------------------------------------------------------------------
# One turn: group, cut points and segments
# ---------------------------------------------------------------------------------------------------------------

def trim_span(edges, total, vad_cfg):
    """Turn span without long silence at the very start and end (keeps a margin). No trim without speech.

    Used for the keep-whole decision; the exported audio is trimmed only for aligned long turns and never into an
    aligned word (segment_turn), because VAD can miss quiet speech, crying or sighs at the edges."""
    if not vad_cfg["trim_edges"] or edges is None:
        return 0.0, total
    first, last = edges
    start = first - vad_cfg["trim_margin_s"] if first > vad_cfg["trim_min_silence_s"] else 0.0
    end = last + vad_cfg["trim_margin_s"] if total - last > vad_cfg["trim_min_silence_s"] else total
    return max(0.0, start), min(total, end)


def alignment_quality(words, cfg):
    """(problems, mean confidence of spelled words, share of words that are wildcards or below the word threshold).
    problems is a list of short texts; empty means the turn passed the check (doc §2.4)."""
    spelled = [w.confidence for w in words if not w.wildcard]
    mean_conf = round(sum(spelled) / len(spelled), 4) if spelled else 0.0
    weak = sum(1 for w in words if w.wildcard or w.confidence < cfg["word_min_confidence"])
    share = round(weak / len(words), 4)
    problems = []
    if mean_conf < cfg["turn_min_mean_confidence"]:
        problems.append(f"mean confidence {mean_conf:.2f} < {cfg['turn_min_mean_confidence']:g}")
    if share > cfg["turn_max_unaligned_share"]:
        problems.append(f"{share:.0%} of words weak or unspelled > {cfg['turn_max_unaligned_share']:.0%}")
    return problems, mean_conf, share


def choose_cuts(words, probs, transcript, start, end, cfg):
    """Gap scores and the cheapest partition. Returns (cut list, review dict, status);
    status is "cut", "whole_dp" (no cut is cheapest) or "whole_boundary" (no partition fits the window)."""
    sc, seg = cfg["scoring"], cfg["segmentation"]
    gaps = [max(0.0, words[k + 1].start - words[k].end) for k in range(len(words) - 1)]
    typical = typical_pause(gaps, sc["typical_pause_min_gap_s"], sc["typical_pause_min_count"],
                            sc["typical_pause_fallback_s"])
    punctuated = has_punctuation(transcript)
    weights = (sc["weight_pause"], sc["weight_text"], sc["weight_silence"])
    plain = (sc["no_punctuation_weight_pause"], sc["no_punctuation_weight_silence"])
    weak = [w.wildcard or w.confidence < cfg["alignment"]["word_min_confidence"] for w in words]

    gap_rows, candidates = [], []
    for k, gap in enumerate(gaps):
        a, b = words[k].end, words[k + 1].start
        v, cut_time = vad.gap_silence(probs, a, b)
        s, p = boundary_score(gap, typical, words[k].punct, v, punctuated, weights, plain)
        row = {"after_word": k, "start": a, "end": b, "length": round(gap, 4), "p": round(p, 4),
               "q": words[k].punct, "v": round(v, 4), "s": round(s, 4), "cut_time": round(cut_time, 4),
               "candidate": not (weak[k] or weak[k + 1])}
        if not row["candidate"]:
            row["excluded"] = "next to a low-confidence or unspelled word"
        gap_rows.append(row)
        if row["candidate"]:
            candidates.append(row)

    times = [start] + [r["cut_time"] for r in candidates] + [end]
    scores = [0.0] + [r["s"] for r in candidates] + [1.0]
    limits = [seg["min_s"], seg["max_s"]]
    result = best_partition(times, scores, seg["cut_weight_lambda"], seg["target_s"], *limits)
    review = {"typical_pause_s": round(typical, 4), "punctuated": punctuated,
              "weights": list(weights) if punctuated else [plain[0], 0.0, plain[1]], "gaps": gap_rows}
    if result is None:
        review["dp"] = {"feasible": False, "limits_s": limits}
        return [], review, "whole_boundary"
    cut_idx, total, length_total, cut_total = result
    cuts = [candidates[i - 1] for i in cut_idx]
    review["dp"] = {"feasible": True, "limits_s": limits, "cuts_s": [c["cut_time"] for c in cuts],
                    "total_cost": round(total, 4), "length_cost": round(length_total, 4),
                    "cut_cost": round(cut_total, 4)}
    return cuts, review, "cut" if cuts else "whole_dp"


def analyse_audio(data, rate, vad_model):
    """(mono 16 kHz analysis copy, VAD speech probability per 32 ms frame) of one turn."""
    wave = audio_io.analysis_copy(data, rate)
    return wave, vad.speech_probabilities(vad_model, wave)


def segment_turn(data, rate, transcript, cfg, vad_model, aligners, analysis=None):
    """Decide the group and the segments of one turn. Returns (segments, info); segment times are seconds in the
    turn. Order of the checks: language (any length) -> short English turn -> text -> alignment -> partition.
    analysis: optional (wave, probs) from analyse_audio on the same audio, so a caller that re-runs one turn with
    other settings (explorer/) does not compute them again; computed here when None."""
    total = data.shape[0] / rate
    wave, probs = analysis if analysis is not None else analyse_audio(data, rate, vad_model)
    lang_cfg, vad_cfg, seg_cfg = cfg["language"], cfg["vad"], cfg["segmentation"]
    language, es_cues, en_cues = detect_language(
        transcript, lang_cfg["spanish_share_min"], lang_cfg["english_share_max"], lang_cfg["min_cue_words"],
        lang_cfg["default"])
    words = tokenize(transcript)
    # Decision span: edge silence (if trimming is on) does not count towards the keep-whole limit.
    decision_start, decision_end = trim_span(vad.speech_edges(probs, vad_cfg["speech_threshold"]), total, vad_cfg)
    start, end = 0.0, total                           # exported span: the whole turn unless trimmed below
    info = {"duration_s": round(total, 4), "language": language, "language_cues": {"es": es_cues, "en": en_cues},
            "group": "final", "reason": "", "alignment": {"model": "", "status": "not_needed"}, "words": []}
    cuts = []

    if language not in lang_cfg["alignment_models"]:  # 1) non-English: whole, whatever the length
        info.update(group="non_english", status="whole_language", reason=f"language: {language}")
        info["alignment"]["status"] = "skipped_language"
    elif decision_end - decision_start <= seg_cfg["keep_whole_max_s"]:   # 2) short English turn: whole
        info["status"] = "whole_short"
    elif not words:                                   # 3) long turn without text
        info.update(group="alignment_issues", status="whole_no_text", reason="alignment: empty transcript")
        info["alignment"]["status"] = "no_text"
    else:                                             # 4) forced alignment and its turn-level check (doc §2.4)
        model_key = lang_cfg["alignment_models"][language]
        aligned, reason = aligners[model_key].align(wave, words)
        info["alignment"] = {"model": model_key, "status": "failed", "reason": reason}
        problems = [reason]
        if aligned:
            info["words"] = words
            problems, mean_conf, share = alignment_quality(words, cfg["alignment"])
            info["alignment"].update(status="failed" if problems else "ok", mean_confidence=mean_conf,
                                     unaligned_share=share, reason="; ".join(problems) or "ok")
        if problems:
            info.update(group="alignment_issues", status="whole_alignment",
                        reason="alignment: " + "; ".join(problems))
        else:                                         # 5) dynamic programme
            if vad_cfg["trim_edges"]:                 # trim edge silence, but never into an aligned word
                start = min(decision_start, max(0.0, words[0].start - vad_cfg["trim_margin_s"]))
                end = max(decision_end, min(total, words[-1].end + vad_cfg["trim_margin_s"]))
            cuts, review, info["status"] = choose_cuts(words, probs, transcript, start, end, cfg)
            info.update(review)
            if info["status"] == "whole_boundary":
                info.update(group="boundary_issues",
                            reason=f"no partition within {seg_cfg['min_s']:g}-{seg_cfg['max_s']:g} s")

    bounds = [start] + [c["cut_time"] for c in cuts] + [end]
    edges_scores = [None] + [c["s"] for c in cuts] + [None]
    samples = [round(t * rate) for t in bounds]
    if any(b <= a for a, b in zip(samples, samples[1:])):
        raise AssertionError("segment bounds are not increasing")
    if cuts:                                          # output check: every word in exactly one segment
        assignment = assign_words(words, bounds)
        parts = [[w for w, k in zip(words, assignment) if k == s] for s in range(len(cuts) + 1)]
        texts = [transcript[g[0].char_start:g[-1].char_end] for g in parts]
        counts = [len(g) for g in parts]
    else:
        texts, counts = [transcript.strip()], [len(words)]
    segments = []
    for k in range(len(bounds) - 1):
        a, b = samples[k] / rate, samples[k + 1] / rate
        speech = vad.speech_seconds(probs, a, b, vad_cfg["speech_threshold"])
        segments.append({"index": k, "start_sample": samples[k], "end_sample": samples[k + 1],
                         "start_s": a, "end_s": b, "duration_s": b - a, "speech_s": speech,
                         "speech_ratio": speech / (b - a), "start_score": edges_scores[k],
                         "end_score": edges_scores[k + 1], "text": texts[k], "word_count": counts[k]})
    info["span_s"] = [round(start, 4), round(end, 4)]
    info["decision_span_s"] = [round(decision_start, 4), round(decision_end, 4)]
    return segments, info


# ---------------------------------------------------------------------------------------------------------------
# Inputs: turns, calls and turn order
# ---------------------------------------------------------------------------------------------------------------

def list_turns(input_dir, extensions):
    """Audio files directly in input_dir and in its direct subfolders (one subfolder per call).
    Returns one dict per turn; turn ids (file stems) must be unique across the whole input."""
    input_dir = Path(input_dir)
    folders = [input_dir] + sorted(p for p in input_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    turns, seen = [], {}
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if not (path.is_file() and path.suffix.lower() in extensions):
                continue
            if path.stem in seen:
                raise ValueError(f"Turn id {path.stem!r} appears twice ({seen[path.stem]} and {path}); "
                                 "audio file names must be unique across all calls")
            seen[path.stem] = path
            turns.append({"turn_id": path.stem, "folder": "" if folder == input_dir else folder.name,
                          "audio": path, "transcript": path.with_suffix(".txt"),
                          "source": path.relative_to(input_dir).as_posix()})
    return turns


def read_turns_csv(path):
    if not path or not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as stream:
        return {row["turn_id"].strip(): {k: (v or "").strip() for k, v in row.items() if k}
                for row in csv.DictReader(stream)}


def safe_name(call_id):
    """A call id as a folder name: letters, digits, '.', '_' and '-' only; anything else becomes '_'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(call_id).strip()).strip("._") or "call"


def order_turns(members):
    """Sort the turns of one call in place: by turn_index if every turn has one, else by turn_start_in_call_s,
    else by file name. Returns the key used."""
    for key in ("turn_index", "turn_start_in_call_s"):
        if all(t["meta"].get(key) for t in members):
            members.sort(key=lambda t: (float(t["meta"][key]), t["source"]))
            return key
    members.sort(key=lambda t: t["source"])
    return "file name"


def group_into_calls(turns, turns_csv_rows):
    """call_id per turn: turns.csv, else the subfolder name, else the turn itself (its own call).
    Returns ({call_id: {"folder", "order_by", "turns"}} sorted by call id, counts per source)."""
    sources = {"turns.csv": 0, "folder": 0, "own call": 0}
    calls = {}
    for turn in turns:
        turn["meta"] = turns_csv_rows.get(turn["turn_id"], {})
        if turn["meta"].get("call_id"):
            turn["call_id"], source = turn["meta"]["call_id"], "turns.csv"
        elif turn["folder"]:
            turn["call_id"], source = turn["folder"], "folder"
        else:
            turn["call_id"], source = turn["turn_id"], "own call"
        sources[source] += 1
        calls.setdefault(turn["call_id"], []).append(turn)
    result, folders = {}, {}
    for call_id in sorted(calls):
        folder = safe_name(call_id)
        if folder in folders:
            raise ValueError(f"Call ids {folders[folder]!r} and {call_id!r} give the same folder name {folder!r}")
        folders[folder] = call_id
        members = calls[call_id]
        order_by = order_turns(members)
        for order, turn in enumerate(members):
            turn["turn_order"], turn["call_folder"] = order, folder
        result[call_id] = {"folder": folder, "order_by": order_by, "turns": members}
    return result, sources


def segment_id(call_folder, turn_order, segment_index):
    """<call>_t<turn order, 3 digits>_s<segment index, 2 digits>: name order = time order within a call."""
    return f"{call_folder}_t{turn_order:03d}_s{segment_index:02d}"


# ---------------------------------------------------------------------------------------------------------------
# Export and context links
# ---------------------------------------------------------------------------------------------------------------

def export_turn(turn, data, rate, subtype, segments, info, cfg, version, output_dir, written):
    """Write the clips, transcripts and review JSON of one turn into its group folder; return its manifest rows
    (without the context links, which are added per call). Every written path, and a call folder created here,
    is appended to `written`, so the caller can remove a half-exported turn. Paths are relative to output_dir."""
    meta = turn["meta"]
    folder = f"{cfg['paths']['groups'][info['group']]}/{turn['call_folder']}"
    if not (output_dir / folder).exists():
        (output_dir / folder).mkdir(parents=True)
        written.append(output_dir / folder)
    turn_start = meta.get("turn_start_in_call_s", "")
    review_rel = f"review/{turn['turn_id']}.json"
    audio_sha = file_sha256(turn["audio"])
    text_sha = file_sha256(turn["transcript"])
    turn_rows = []
    for seg in segments:
        seg_id = segment_id(turn["call_folder"], turn["turn_order"], seg["index"])
        clip_rel, text_rel = f"{folder}/{seg_id}.wav", f"{folder}/{seg_id}.txt"
        written.append(output_dir / clip_rel)
        clip_subtype = audio_io.write_clip(output_dir / clip_rel, data[seg["start_sample"]:seg["end_sample"]],
                                           rate, subtype)
        written.append(output_dir / text_rel)
        (output_dir / text_rel).write_text(seg["text"] + "\n", encoding="utf-8")
        turn_rows.append({
            "segment_id": seg_id, "group": info["group"], "reason": info["reason"], "call_id": turn["call_id"],
            "customer_id": meta.get("customer_id", ""), "turn_id": turn["turn_id"],
            "turn_index": meta.get("turn_index", ""), "turn_order": turn["turn_order"],
            "segment_index": seg["index"], "segment_count": len(segments),
            "audio_path": clip_rel, "transcript_path": text_rel, "review_path": review_rel,
            "source_audio": turn["source"], "source_audio_sha256": audio_sha,
            "source_transcript": Path(turn["source"]).with_suffix(".txt").as_posix(),
            "source_transcript_sha256": text_sha,
            "source_sample_rate": rate, "source_channels": data.shape[1], "clip_subtype": clip_subtype,
            "turn_start_in_call_s": turn_start,
            "segment_start_in_turn_s": fmt(seg["start_s"]), "segment_end_in_turn_s": fmt(seg["end_s"]),
            "segment_start_in_call_s": fmt(float(turn_start) + seg["start_s"]) if turn_start else "",
            "segment_end_in_call_s": fmt(float(turn_start) + seg["end_s"]) if turn_start else "",
            "call_time_status": "resolved" if turn_start else "unresolved",
            "duration_s": fmt(seg["duration_s"]), "speech_duration_s": fmt(seg["speech_s"]),
            "speech_ratio": fmt(seg["speech_ratio"]),
            "low_speech_ratio": int(seg["speech_ratio"] < cfg["vad"]["low_speech_ratio"]),
            "language": info["language"], "segmentation_status": info["status"],
            "alignment_model": info["alignment"]["model"], "alignment_status": info["alignment"]["status"],
            "alignment_mean_confidence": fmt(info["alignment"].get("mean_confidence")),
            "alignment_unaligned_share": fmt(info["alignment"].get("unaligned_share")),
            "start_boundary_score": fmt(seg["start_score"]), "end_boundary_score": fmt(seg["end_score"]),
            "word_count": seg["word_count"], "segmentation_version": version})
    review = {"turn_id": turn["turn_id"], "call_id": turn["call_id"], "turn_order": turn["turn_order"],
              "source_audio": turn["source"], "segmentation_version": version,
              **{k: v for k, v in info.items() if k != "words"},
              "words": [{"i": i, "text": w.text, "units": w.units, "start": w.start, "end": w.end,
                         "confidence": w.confidence, "wildcard": w.wildcard, "q": w.punct}
                        for i, w in enumerate(info["words"])],
              "segments": [{"segment_id": row["segment_id"], "path": row["audio_path"],
                            **{k: v for k, v in seg.items() if k != "text"}}
                           for row, seg in zip(turn_rows, segments)]}
    written.append(output_dir / review_rel)
    (output_dir / review_rel).write_text(json.dumps(review, indent=1, ensure_ascii=False), encoding="utf-8")
    return turn_rows


def call_gap(earlier, later):
    """Seconds of call time between two exported items (usually the agent's turn), or "" if a time is unknown."""
    if not (earlier and later and earlier["segment_end_in_call_s"] and later["segment_start_in_call_s"]):
        return ""
    return fmt(float(later["segment_start_in_call_s"]) - float(earlier["segment_end_in_call_s"]))


def link_call(rows):
    """Add order_in_call, prev/next segment and group, and call-time gaps to the rows of one call (in call order,
    all groups together). Failed turns have no rows, so the links skip them."""
    for k, row in enumerate(rows):
        prev = rows[k - 1] if k > 0 else None
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        row.update(order_in_call=k,
                   prev_segment_id=prev["segment_id"] if prev else "", prev_group=prev["group"] if prev else "",
                   next_segment_id=nxt["segment_id"] if nxt else "", next_group=nxt["group"] if nxt else "",
                   gap_to_prev_s=call_gap(prev, row), gap_to_next_s=call_gap(row, nxt))
    return rows


# ---------------------------------------------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------------------------------------------

def load_models(cfg, root):
    """Silero VAD (CPU) and one Aligner per language in language.alignment_models. Returns (vad_model, aligners)."""
    device = cfg["device"]
    if (device == "cuda" and not torch.cuda.is_available()) or (device == "mps" and not torch.backends.mps.is_available()):
        raise ValueError(f"Device {device!r} is not available here; set device: cpu")
    if cfg.get("cpu_threads"):
        torch.set_num_threads(int(cfg["cpu_threads"]))
    vad_model = vad.load_vad()
    print(f"Step 2: Silero VAD loaded (JIT, CPU); torch threads {torch.get_num_threads()}")
    aligners = {}
    for language, key in cfg["language"]["alignment_models"].items():
        t0 = time.time()
        aligners[key] = Aligner(key, Path(root) / cfg["paths"]["weights_dir"], device,
                                cfg["alignment"]["emission_chunk_s"], cfg["alignment"]["emission_context_s"])
        print(f"        alignment model for {language}: {key} ({aligners[key].parameters / 1e6:.1f}M parameters, "
              f"{device}, loaded offline in {time.time() - t0:.1f} s)")
    return vad_model, aligners


def run(cfg, root):
    """Segment every turn, call by call, into the four group folders; write review/, the group manifests,
    call_timeline.csv and (if needed) failures.csv. Returns (rows of all exported items, failures)."""
    paths = cfg["paths"]
    groups = paths["groups"]
    input_dir, output_dir = root / paths["input_dir"], root / paths["output_dir"]
    turns_csv = root / paths["turns_csv"] if paths.get("turns_csv") else None
    version = settings_version(cfg)
    print(f"Settings version: {version}")

    # 1) inputs: turns, their calls and their order in the call
    turns = list_turns(input_dir, {e.lower() for e in cfg["audio_extensions"]})
    if not turns:
        raise ValueError(f"No audio files with extensions {cfg['audio_extensions']} in {input_dir}")
    csv_rows = read_turns_csv(turns_csv)
    calls, sources = group_into_calls(turns, csv_rows)
    print(f"Step 1: {len(turns)} turns in {input_dir} -> {len(calls)} calls; call id from turns.csv "
          f"{sources['turns.csv']}, from the subfolder {sources['folder']}, own call {sources['own call']}")
    if sources["own call"]:
        print(f"  WARNING: {sources['own call']} turns are in no call (not in turns.csv, not in a call subfolder); "
              "each becomes its own call, so context links cannot be built for them")
    unresolved = sum(1 for t in turns if not t["meta"].get("turn_start_in_call_s"))
    if unresolved:
        print(f"  {unresolved} turns have no turn_start_in_call_s: their call times and gaps stay empty")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output folder {output_dir} is not empty; choose a new output_dir or empty it.")
    for sub in ["review", *(groups[g] for g in GROUPS)]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    # 2) models: Silero VAD (CPU) and the alignment model(s)
    vad_model, aligners = load_models(cfg, root)

    # 3) call by call; the manifests and the timeline get a call's rows only after its last turn
    print("Step 3: segmenting turns, call by call")
    rows, turn_info, failures = [], [], []
    with contextlib.ExitStack() as stack:
        def open_csv(path, fields):
            stream = stack.enter_context(open(path, "w", newline="", encoding="utf-8"))
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            return stream, writer
        outputs = {g: open_csv(output_dir / groups[g] / "manifest.csv", MANIFEST_FIELDS) for g in GROUPS}
        outputs["timeline"] = open_csv(output_dir / "call_timeline.csv", TIMELINE_FIELDS)

        for call_id, call in calls.items():
            print(f"  call {call_id}: {len(call['turns'])} turns, ordered by {call['order_by']}")
            call_rows = []
            for turn in call["turns"]:
                t0, written = time.time(), []
                try:                                  # one bad file must not stop a large batch
                    if not turn["transcript"].is_file():
                        raise FileNotFoundError(f"missing transcript {turn['transcript'].name}")
                    transcript = turn["transcript"].read_text(encoding="utf-8")
                    data, rate, subtype = audio_io.read_audio(turn["audio"])
                    segments, info = segment_turn(data, rate, transcript, cfg, vad_model, aligners)
                    turn_rows = export_turn(turn, data, rate, subtype, segments, info, cfg, version, output_dir,
                                            written)
                except Exception as exc:
                    for path in reversed(written):    # no half-exported turn: remove its files (and new folder)
                        if path.is_dir():
                            if not any(path.iterdir()):
                                path.rmdir()
                        else:
                            path.unlink(missing_ok=True)
                    failures.append((turn["turn_id"], call_id, f"{type(exc).__name__}: {exc}"))
                    print(f"    FAILED {turn['turn_id']}: {type(exc).__name__}: {exc}")
                    continue
                call_rows += turn_rows
                turn_info.append((info["group"], info["status"]))
                lengths = ", ".join(f"{seg['duration_s']:.1f}" for seg in segments)
                print(f"    {turn['turn_id']}: {info['duration_s']:.1f} s, {info['language']}, "
                      f"{info['group']}/{info['status']} -> {len(segments)} segment(s) [{lengths}] "
                      f"in {time.time() - t0:.1f} s" + (f"; {info['reason']}" if info["reason"] else ""))
            link_call(call_rows)
            for row in call_rows:
                outputs[row["group"]][1].writerow(row)
                outputs["timeline"][1].writerow(row)
            for stream, _ in outputs.values():
                stream.flush()
            rows += call_rows

    # 4) failures and summary
    if failures:
        with open(output_dir / "failures.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["turn_id", "call_id", "error"])
            writer.writerows(failures)
    print_summary(len(turns), len(calls), turn_info, failures, rows, output_dir)
    return rows, failures


def print_summary(n_turns, n_calls, turn_info, failures, rows, output_dir):
    print("Run summary")
    print(f"  turns read: {n_turns} in {n_calls} calls, failed: {len(failures)}")
    for group in GROUPS:
        statuses = {}
        for g, status in turn_info:
            if g == group:
                statuses[status] = statuses.get(status, 0) + 1
        items = sum(1 for r in rows if r["group"] == group)
        detail = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
        print(f"  {group}: {sum(statuses.values())} turns, {items} files" + (f" ({detail})" if detail else ""))
    durations = sorted(float(r["duration_s"]) for r in rows if r["group"] == "final")
    if durations:
        bins = [(0, 2), (2, 3), (3, 5), (5, 7), (7, 10), (10, 15), (15, float("inf"))]
        counts = ", ".join(f"{lo}-{hi if hi != float('inf') else ''} s: {sum(lo <= d < hi for d in durations)}"
                           for lo, hi in bins)
        print(f"  final segment length: min {durations[0]:.2f} s, median {median(durations):.2f} s, "
              f"max {durations[-1]:.2f} s; {counts}")
    low = [r for r in rows if r["low_speech_ratio"]]
    print(f"  low_speech_ratio: {len(low)} files ({sum(1 for r in low if r['group'] == 'final')} in final)")
    for turn_id, call_id, reason in failures:
        print(f"  failed: {turn_id} (call {call_id}): {reason}")
    print(f"  output: {output_dir}")

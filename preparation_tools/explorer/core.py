"""Segment Explorer core: run the turn segmenter on one uploaded audio file and its transcript, cache the expensive
parts, and build the page data and the self-contained export. The web server is in app.py; nothing here needs a
network or a browser, so the unit tests (tests/test_explorer.py) run it with fake models.

The decision always comes from segmenter.pipeline.segment_turn, the same function segment_turns.py runs. Decoding,
the 16 kHz analysis copy, the VAD probabilities and the forced alignment are computed once per upload and then
reused, so a re-run with other settings only redoes the language check, the gap scores and the dynamic programme.
"""
from collections import OrderedDict
import base64
import copy
import hashlib
import html
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from segmenter import audio_io, vad
from segmenter.pipeline import alignment_quality, analyse_audio, choose_cuts, segment_turn, settings_version
from segmenter.textnorm import tokenize

# The settings the page can change; everything else comes unchanged from config/segment_turns.yaml.
EDITABLE = [
    ("segmentation", "keep_whole_max_s"), ("segmentation", "target_s"), ("segmentation", "min_s"),
    ("segmentation", "max_s"), ("segmentation", "cut_weight_lambda"),
    ("scoring", "weight_pause"), ("scoring", "weight_text"), ("scoring", "weight_silence"),
    ("scoring", "no_punctuation_weight_pause"), ("scoring", "no_punctuation_weight_silence"),
]
WEIGHTS = {"weight_pause", "weight_text", "weight_silence", "no_punctuation_weight_pause",
           "no_punctuation_weight_silence"}
AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}
LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "mixed": "mixed English and Spanish"}
PEAKS_PER_SECOND = 100       # waveform resolution sent to the page (10 ms bins) ...
MAX_PEAK_BINS = 200_000      # ... but never more bins than this (very long files)


# ---------------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------------

def editable_values(cfg):
    """{"section.key": value} of the settings the page can change."""
    return {f"{section}.{key}": cfg[section][key] for section, key in EDITABLE}


def apply_settings(base_cfg, values):
    """A copy of base_cfg with the given editable values. Returns (cfg, names of the values that differ from
    base_cfg, in EDITABLE order). Raises ValueError for unknown names or values that make no sense."""
    cfg, defaults = copy.deepcopy(base_cfg), editable_values(base_cfg)
    for name, value in (values or {}).items():
        if name not in defaults:
            raise ValueError(f"Unknown setting {name!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a number of at least 0")
        if value == defaults[name]:
            continue                                  # keep the YAML value as it is (1.0 stays 1.0: same hash)
        if isinstance(value, float) and value.is_integer():
            value = int(value)                        # 8.0 -> 8, as YAML would write it
        section, key = name.split(".")
        cfg[section][key] = value
    seg, sc = cfg["segmentation"], cfg["scoring"]
    if not 0 < seg["min_s"] < seg["max_s"]:
        raise ValueError("Need 0 < min_s < max_s")
    if seg["target_s"] <= 0:
        raise ValueError("target_s must be above 0")
    if any(sc[key] > 1 for key in WEIGHTS):
        raise ValueError("Score weights must be between 0 and 1")
    changed = [name for name, value in editable_values(cfg).items() if value != defaults[name]]
    return cfg, changed


def yaml_snippet(cfg, base_cfg, changed):
    """The changed values as YAML to paste into config/segment_turns.yaml by hand ("" when nothing changed)."""
    if not changed:
        return ""
    lines = [f"# Segment Explorer: values that differ from config/segment_turns.yaml "
             f"({settings_version(base_cfg)} -> {settings_version(cfg)})"]
    for section in ("segmentation", "scoring"):
        keys = [name.split(".")[1] for name in changed if name.startswith(section + ".")]
        if keys:
            lines.append(f"{section}:")
            lines += [f"  {key}: {cfg[section][key]!r}  # YAML: {base_cfg[section][key]!r}" for key in keys]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Waveform and display data
# ---------------------------------------------------------------------------------------------------------------

def waveform_peaks(samples, rate, per_second=PEAKS_PER_SECOND, max_bins=MAX_PEAK_BINS):
    """Minimum and maximum of a mono signal per bin, as integers in -1000..1000 of the file's own peak.
    Returns {"bin_s", "peak", "min", "max"}; bin k covers [k, k + 1) x bin_s seconds."""
    samples = np.asarray(samples, dtype=np.float32).ravel()
    if samples.size == 0:
        return {"bin_s": 1.0 / per_second, "peak": 0.0, "min": [], "max": []}
    bins = max(1, min(max_bins, math.ceil(samples.size / rate * per_second)))
    size = math.ceil(samples.size / bins)             # samples per bin
    bins = math.ceil(samples.size / size)
    blocks = np.pad(samples, (0, bins * size - samples.size), mode="edge").reshape(bins, size)
    peak = float(np.abs(samples).max())
    scale = 1000.0 / peak if peak > 0 else 0.0
    return {"bin_s": size / rate, "peak": round(peak, 6),
            "min": np.round(blocks.min(axis=1) * scale).astype(int).tolist(),
            "max": np.round(blocks.max(axis=1) * scale).astype(int).tolist()}


def decision_text(info, cfg, segment_count):
    """One plain sentence: what the pipeline decided for this turn and why."""
    seg, status = cfg["segmentation"], info["status"]
    duration, language = info["duration_s"], info["language"]
    window = f"{seg['min_s']:g}-{seg['max_s']:g} s"
    if status == "whole_language":
        cues = info["language_cues"]
        return (f"The transcript looks {LANGUAGE_NAMES.get(language, language)} ({cues['es']} Spanish and "
                f"{cues['en']} English cue words), so the turn goes whole to non_english/, whatever its length. "
                "It was not aligned or cut.")
    if status == "whole_short":
        span = info["decision_span_s"][1] - info["decision_span_s"][0]
        return (f"English turn of {span:.2f} s, at most keep_whole_max_s ({seg['keep_whole_max_s']:g} s): it goes "
                "whole to final/ without alignment or cutting.")
    if status == "whole_no_text":
        return "Long English turn with an empty transcript: it goes whole to alignment_issues/."
    if status == "whole_alignment":
        return f"Long English turn that failed the alignment check ({info['reason']}): it goes whole to alignment_issues/."
    if status == "whole_boundary":
        return f"Long English turn with no partition that keeps every piece within {window}: it goes whole to boundary_issues/."
    if status == "whole_dp":
        return (f"Long English turn ({duration:.2f} s): the dynamic programme found that no cut is cheapest, so it "
                "stays whole in final/.")
    return f"Long English turn ({duration:.2f} s) cut by the dynamic programme into {segment_count} segments within {window}; they go to final/."


def display_details(info, transcript, analysis, cfg, aligners, display_alignment=True):
    """Words and gap scores for the page. When the pipeline aligned and scored the turn, these are its own.
    Otherwise (short or non-English turn, failed alignment check) they come from an extra run for display only,
    which never changes the group or the segments. Returns a dict with words, gaps, scores, display_only and note."""
    out = {"words": info["words"], "gaps": info.get("gaps", []), "display_only": False, "note": "",
           "scores": {k: info[k] for k in ("typical_pause_s", "punctuated", "weights") if k in info},
           "alignment": None}
    if "gaps" in info:                                # the dynamic programme ran: nothing extra
        return out
    words, status = info["words"], info["status"]
    shown = "The words and gap scores below are"
    if status == "whole_short":
        why = "The pipeline kept this turn whole because it is short, so it did not align or score it."
    elif status == "whole_language":
        why = ("The pipeline kept this turn whole because it is not English, so it did not align or score it. "
               "The English alignment model was used on non-English text here, so the word times are rough.")
    elif words:                                       # aligned, but failed the turn-level check
        why = "The turn failed the alignment check, so the pipeline did not score its gaps or run the dynamic programme."
        shown = "The words are the pipeline's alignment; the gap scores below are"
    elif status == "whole_alignment":
        why = f"Forced alignment failed ({info['alignment'].get('reason', '')}), so there are no word times."
    else:
        why = "The transcript is empty, so there are no words to show."
    if not words and transcript.strip() and display_alignment and status in ("whole_short", "whole_language"):
        models = cfg["language"]["alignment_models"]
        key = models.get(info["language"]) or models.get(cfg["language"]["default"]) or next(iter(models.values()))
        words = tokenize(transcript)
        ok, reason = aligners[key].align(analysis[0], words)
        if not ok:
            out.update(display_only=True, note=f"{why} An extra alignment for display failed: {reason}.")
            return out
        problems, mean_conf, share = alignment_quality(words, cfg["alignment"])
        out["alignment"] = {"model": key, "mean_confidence": mean_conf, "unaligned_share": share}
    if not words:
        out["note"] = why
        return out
    start, end = info["span_s"]
    _, review, _ = choose_cuts(words, analysis[1], transcript, start, end, cfg)
    out.update(words=words, gaps=review["gaps"], display_only=True,
               scores={k: review[k] for k in ("typical_pause_s", "punctuated", "weights")},
               note=f"{why} {shown} for display only and are not part of the pipeline decision; no cut is shown.")
    return out


def build_payload(upload, transcript_name, segments, info, details, cfg, base_cfg, changed, timings):
    """Everything the page draws, as JSON-ready data (also embedded in the exported HTML)."""
    weak_below = cfg["alignment"]["word_min_confidence"]
    chosen = set()
    if info["status"] == "cut":                       # a cut follows the last word of every segment but the last
        chosen = {int(k) for k in np.cumsum([s["word_count"] for s in segments])[:-1] - 1}
    words = [{"i": i, "text": w.text, "start": w.start, "end": w.end, "confidence": w.confidence,
              "wildcard": w.wildcard, "weak": bool(w.wildcard or w.confidence < weak_below), "q": w.punct}
             for i, w in enumerate(details["words"])]
    gaps = [{**g, "chosen": g["after_word"] in chosen} for g in details["gaps"]]
    low_below = cfg["vad"]["low_speech_ratio"]
    rows = [{"index": s["index"], "start_s": round(s["start_s"], 4), "end_s": round(s["end_s"], 4),
             "duration_s": round(s["duration_s"], 4), "text": s["text"], "word_count": s["word_count"],
             "start_score": s["start_score"], "end_score": s["end_score"], "speech_s": round(s["speech_s"], 3),
             "speech_ratio": round(s["speech_ratio"], 4), "low_speech": bool(s["speech_ratio"] < low_below)}
            for s in segments]
    return {
        "tool": "Segment Explorer",
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "files": {"audio": upload["name"], "transcript": transcript_name, "audio_bytes": len(upload["raw"]),
                  "sample_rate": upload["rate"], "channels": upload["channels"], "subtype": upload["subtype"],
                  "mime": upload["mime"]},
        "duration_s": info["duration_s"], "language": info["language"], "language_cues": info["language_cues"],
        "group": info["group"], "status": info["status"], "reason": info["reason"],
        "decision": decision_text(info, cfg, len(segments)),
        "alignment": info["alignment"], "display_alignment": details["alignment"],
        "dp": info.get("dp"), "scores": details["scores"],
        "span_s": info["span_s"], "decision_span_s": info["decision_span_s"],
        "words": words, "gaps": gaps, "display_only": details["display_only"], "display_note": details["note"],
        "segments": rows,
        "vad": {"frame_s": vad.FRAME_S, "threshold": cfg["vad"]["speech_threshold"],
                "probs": np.round(upload["probs"], 3).tolist()},
        "peaks": upload["peaks"],
        "settings": {"version": settings_version(cfg), "yaml_version": settings_version(base_cfg),
                     "values": editable_values(cfg), "defaults": editable_values(base_cfg), "changed": changed,
                     "yaml_snippet": yaml_snippet(cfg, base_cfg, changed),
                     "word_min_confidence": weak_below, "low_speech_ratio": low_below},
        "timings": timings,
    }


# ---------------------------------------------------------------------------------------------------------------
# Self-contained export
# ---------------------------------------------------------------------------------------------------------------

LIVE_BLOCK = re.compile(r"<!--LIVE-->.*?<!--/LIVE-->", re.S)
DATA_TAG = '<script id="explorer-data" type="application/json">null</script>'
AUDIO_TAG = '<script id="explorer-audio" type="text/plain"></script>'


def build_export_html(template, payload, audio_bytes, mime):
    """One HTML file that works offline: the page without its run/upload/re-run controls, the result as JSON and
    the original audio as a base64 data URI. Network access is switched off by its Content-Security-Policy."""
    page = LIVE_BLOCK.sub("", template)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    audio = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"
    title = f"<title>Segment Explorer: {html.escape(payload['files']['audio'])}</title>"
    for old, new in ((DATA_TAG, f'<script id="explorer-data" type="application/json">{data}</script>'),
                     (AUDIO_TAG, f'<script id="explorer-audio" type="text/plain">{audio}</script>'),
                     ("connect-src 'self'", "connect-src 'none'"),
                     ("<title>Segment Explorer</title>", title)):
        if page.count(old) != 1:
            raise ValueError(f"page.html must contain {old!r} exactly once")
        page = page.replace(old, new)
    return page


def export_name(audio_name):
    """File name of the export, e.g. segment_explorer_d418_long_en.html."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(audio_name).stem).strip("._") or "turn"
    return f"segment_explorer_{stem}.html"


# ---------------------------------------------------------------------------------------------------------------
# One explorer session: uploads, cache and runs
# ---------------------------------------------------------------------------------------------------------------

class CachedAligner:
    """Wraps one alignment model for one upload: a word list is aligned once, later runs replay the result.
    The audio of an upload never changes, so the letter units of the words are a complete cache key."""

    def __init__(self, aligner):
        self.aligner, self.memo, self.runs = aligner, {}, 0

    def align(self, wave, words):
        key = tuple(tuple(w.units) for w in words)
        if key not in self.memo:
            self.runs += 1
            ok, reason = self.aligner.align(wave, words)
            self.memo[key] = ok, reason, [(w.start, w.end, w.confidence) for w in words]
        ok, reason, times = self.memo[key]
        for word, (start, end, confidence) in zip(words, times):
            word.start, word.end, word.confidence = start, end, confidence
        return ok, reason


class Explorer:
    """Holds the YAML settings, the models (loaded once, on first use) and the last few uploads in memory.
    load_models: a function returning (vad_model, {model key: aligner}), as segmenter.pipeline.load_models."""

    def __init__(self, base_cfg, load_models, tmp_dir, keep_uploads=3, display_alignment=True):
        self.base_cfg, self.load_models, self.tmp_dir = base_cfg, load_models, Path(tmp_dir)
        self.keep_uploads, self.display_alignment = keep_uploads, display_alignment
        self.models, self.uploads = None, OrderedDict()

    def get_models(self):
        if self.models is None:
            self.models = self.load_models()
        return self.models

    def add_audio(self, name, raw):
        """Decode one uploaded file and compute its analysis copy, VAD probabilities and waveform peaks.
        Returns (audio id, upload dict). The bytes go to a temporary file only for decoding, deleted at once."""
        ext = Path(name).suffix.lower()
        allowed = {e.lower() for e in self.base_cfg["audio_extensions"]} & set(AUDIO_TYPES)
        if ext not in allowed:
            raise ValueError(f"Audio must be one of {', '.join(sorted(allowed))} (got {ext or 'no extension'})")
        audio_id = hashlib.sha256(raw).hexdigest()[:16]
        if audio_id in self.uploads:
            self.uploads.move_to_end(audio_id)
            self.uploads[audio_id]["name"] = name
            return audio_id, self.uploads[audio_id]
        vad_model, aligners = self.get_models()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        path = self.tmp_dir / f"upload_{audio_id}{ext}"
        t0 = time.time()
        try:
            path.write_bytes(raw)
            data, rate, subtype = audio_io.read_audio(path)
        finally:
            path.unlink(missing_ok=True)
        if data.shape[0] == 0:
            raise ValueError("The audio file has no samples")
        t1 = time.time()
        wave, probs = analyse_audio(data, rate, vad_model)
        t2 = time.time()
        upload = {"name": name, "raw": raw, "mime": AUDIO_TYPES[ext], "rate": rate, "channels": data.shape[1],
                  "subtype": subtype, "frames": data.shape[0], "wave": wave, "probs": probs,
                  "peaks": waveform_peaks(data.mean(axis=1), rate),
                  "aligners": {key: CachedAligner(a) for key, a in aligners.items()},
                  "timings": {"decode_s": round(t1 - t0, 3), "vad_s": round(t2 - t1, 3)}}
        self.uploads[audio_id] = upload
        while len(self.uploads) > self.keep_uploads:
            self.uploads.popitem(last=False)
        return audio_id, upload

    def run(self, audio_id, transcript, transcript_name="", values=None):
        """segment_turn on a cached upload with the YAML settings plus the given values. Returns the page data."""
        if audio_id not in self.uploads:
            raise KeyError(audio_id)
        upload = self.uploads[audio_id]
        cfg, changed = apply_settings(self.base_cfg, values)
        transcript = transcript.replace("\r\n", "\n").replace("\r", "\n")   # as Path.read_text in the batch
        analysis = (upload["wave"], upload["probs"])
        frames = np.empty((upload["frames"], 0))      # segment_turn needs only the length once analysis is given
        runs_before = sum(a.runs for a in upload["aligners"].values())
        t0 = time.time()
        segments, info = segment_turn(frames, upload["rate"], transcript, cfg, None, upload["aligners"], analysis)
        details = display_details(info, transcript, analysis, cfg, upload["aligners"], self.display_alignment)
        aligned = sum(a.runs for a in upload["aligners"].values()) - runs_before
        timings = {**upload["timings"], "run_s": round(time.time() - t0, 3),
                   "alignment": "computed" if aligned else ("cached" if details["words"] else "not run")}
        return build_payload(upload, transcript_name, segments, info, details, cfg, self.base_cfg, changed, timings)

    def clear(self):
        self.uploads.clear()

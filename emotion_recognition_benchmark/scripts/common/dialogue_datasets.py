"""Metadata-only access to audited dialogue audio; no model imports or audio decoding.

load_dialogue_records returns scoring records (including gold). completed_history returns
only context-safe fields: it never returns gold, votes, captions or JSON sidecar paths.
DailyTalk has numeric turn order, not published session timestamps: its start/end stay None.
"""
import csv
import json
import math
import re
from pathlib import Path


DATA = Path(__file__).resolve().parents[2] / "data"
CLASSES = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
LABELS = {**{label: label for label in CLASSES}, "anger": "angry", "disgust": "disgusted",
          "fear": "fearful", "happiness": "happy", "no emotion": "neutral", "none": "neutral",
          "sadness": "sad", "surprise": "surprised"}
CONTEXT_FIELDS = ("dataset", "id", "audio", "speaker", "speaker_key", "dialogue", "start", "end",
                  "duration", "transcript", "split", "turn_index", "time_basis")


def canonical_emotion(value):
    try:
        return LABELS[str(value).strip().lower()]
    except KeyError:
        raise ValueError(f"Unknown dialogue emotion: {value!r}") from None


def _splits(dataset, split):
    aliases = {"val": "validation", "dev": "validation", "valid": "validation"}
    values = [split] if isinstance(split, str) or split is None else list(split)
    values = {aliases.get(str(value).lower(), str(value).lower()) for value in values}
    if values & {"all", "none"}:
        return None
    allowed = {"train", "validation", "test"} if dataset == "emotiontalk" else {"train", "validation", "unassigned"}
    if not values or not values <= allowed:
        raise ValueError(f"{dataset}: split must be one of {sorted(allowed)} or 'all'; got {split!r}. "
                         + ("Public ECSS provides no DailyTalk test split." if dataset == "dailytalk" else ""))
    return values


def _path(root, relative):
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"Dataset path must stay relative: {relative!r}")
    path = root / value
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; restore the dataset with scripts/common/setup_datasets.py {root.name}")
    return path


def _number(value, description):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {description}")
    return value


def _plain_text(value):
    # Both bracket forms encode human acoustic/stage annotations in the released transcripts.
    return " ".join(re.sub(r"\[[^\]]*\]|【[^】]*】", " ", value or "").split())


def load_dialogue_records(dataset, split="train", *, primary_only=False, data_root=None):
    """Return records with id/audio(Path)/gold/speaker/dialogue/start/end/duration/transcript/split.

    EmotionTalk supports train/validation/test/all; DailyTalk supports ECSS train/validation,
    all original audio, or its 14 unassigned originals. val/dev alias validation; split may
    also be a sequence. data_root is the parent data/ directory, not the dataset folder.
    EmotionTalk gold/timestamps come from original Audio/json, never
    captions or an inferred majority vote. primary_only excludes incomplete/single-speaker
    dialogues but does not itself require a target to have past history.
    """
    if dataset not in ("emotiontalk", "dailytalk"):
        raise ValueError(f"Unsupported dialogue dataset: {dataset!r}")
    selected = _splits(dataset, split)
    root = (Path(data_root) if data_root is not None else DATA) / dataset
    index = _path(root, "utterances.csv")
    with index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    speaker_counts = {}
    if dataset == "emotiontalk":
        for row in rows:
            speaker_counts.setdefault(row["dialogue_id"], set()).add(row["speaker_id"])
    records, seen = [], set()
    for row in rows:
        actual_split = row["split"] if dataset == "emotiontalk" else row["ecss_split"] or "unassigned"
        actual_split = "validation" if actual_split == "val" else actual_split
        if selected is not None and actual_split not in selected:
            continue
        dialogue = row["dialogue_id"]
        if primary_only and ((dataset == "emotiontalk" and len(speaker_counts[dialogue]) != 2)
                             or (dataset == "dailytalk" and dialogue in {"175", "1451"})):
            continue
        audio = _path(root, row["audio_path"])
        duration = _number(row["duration_s"], f"duration of {audio}")
        if duration <= 0:
            raise ValueError(f"Nonpositive duration for {audio}")
        if dataset == "emotiontalk":
            sidecar = _path(root, row["json_path"])
            if sidecar.stem != row["clip_id"]:
                raise ValueError(f"EmotionTalk index/JSON ID mismatch: {row['clip_id']}")
            original = json.loads(sidecar.read_text(encoding="utf-8"))
            gold = canonical_emotion(original["emotion_result"])
            if row.get("emotion") and canonical_emotion(row["emotion"]) != gold:
                raise ValueError(f"EmotionTalk index/original gold mismatch: {row['clip_id']}")
            start = _number(original["paragraphs"]["startTime"], "startTime")
            end = _number(original["paragraphs"]["endTime"], "endTime")
            if end < start:
                raise ValueError(f"EmotionTalk end before start: {row['clip_id']}")
            speaker = str(original["speaker_id"])
            if row.get("speaker_id") != speaker:
                raise ValueError(f"EmotionTalk index/original speaker mismatch: {row['clip_id']}")
            transcript = _plain_text(row["text_without_markup"] if "text_without_markup" in row
                                     else original.get("content", row.get("text", "")))
            record = {"id": row["clip_id"], "gold": gold, "speaker": speaker,
                      "speaker_key": row.get("speaker_group_key") or f"{row['group_id']}:{speaker}",
                      "start": start, "end": end, "turn_index": int(row["utterance_index"]),
                      "transcript": transcript, "time_basis": "published_timestamp",
                      "label_source": "Audio/json emotion_result"}
        else:
            gold_name = row["ecss_emotion"] or row["original_emotion"]
            record = {"id": row["audio_id"], "gold": canonical_emotion(gold_name),
                      "speaker": row["speaker_id"], "speaker_key": row["speaker_id"],
                      "start": None, "end": None, "turn_index": int(row["turn_index"]),
                      "transcript": _plain_text(row["text"]), "time_basis": "turn_order",
                      "label_source": "ECSS category" if row["ecss_emotion"] else "DailyTalk original category"}
        if audio.stem != record["id"]:
            raise ValueError(f"{dataset} index/audio ID mismatch: {record['id']}")
        if record["id"] in seen:
            raise ValueError(f"Duplicate {dataset} utterance ID: {record['id']}")
        seen.add(record["id"])
        records.append(record | {"dataset": dataset, "audio": audio, "dialogue": dialogue,
                                 "duration": duration, "split": actual_split})
    return sorted(records, key=lambda r: (r["dialogue"], r["start"] if r["start"] is not None
                                         else r["turn_index"], r["turn_index"], r["id"]))


def context_record(record):
    """A safe audio/text context view, also for unrelated-dialogue control turns; no gold or annotations."""
    return {key: record[key] for key in CONTEXT_FIELDS if key in record}


def completed_history(target, records, max_turns=None):
    """Completed, same-split past only; returns chronological context records with no annotations.

    Timestamped audio uses end <= target.start exactly, with no 1ms tolerance. Ordered-only
    DailyTalk uses turn_index < target.turn_index; no artificial session times are invented.
    max_turns keeps the nearest N qualifying turns, still in chronological order.
    """
    if max_turns is not None and (not isinstance(max_turns, int) or max_turns < 0):
        raise ValueError("max_turns must be a nonnegative integer or None")
    candidates = [r for r in records if r.get("dataset") == target.get("dataset")
                  and r["dialogue"] == target["dialogue"] and r["split"] == target["split"]
                  and r["id"] != target["id"]]
    if target.get("start") is not None:
        candidates = [r for r in candidates if r.get("start") is not None and r.get("end") is not None
                      and r["start"] <= r["end"] <= target["start"]]
        candidates.sort(key=lambda r: (r["start"], r["end"], r["id"]))
    else:
        candidates = [r for r in candidates if r["turn_index"] < target["turn_index"]]
        candidates.sort(key=lambda r: (r["turn_index"], r["id"]))
    if max_turns is not None:
        candidates = candidates[-max_turns:] if max_turns else []
    return [context_record(record) for record in candidates]


def select_records(records, subset=None):
    """Select explicit clip IDs in stable corpus order; fail on unknown IDs instead of silently dropping them."""
    if subset is None:
        return records
    wanted = {subset} if isinstance(subset, str) else set(subset)
    missing = wanted - {record["id"] for record in records}
    if missing:
        raise ValueError(f"Subset IDs absent from selected split: {sorted(missing)}")
    return [record for record in records if record["id"] in wanted]

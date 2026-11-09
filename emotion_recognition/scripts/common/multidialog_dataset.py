"""Read the retained MultiDialog test index without downloading or running models.

The default is the conservative ``gold_both`` cohort: complete conversations
whose two actors both occur in the authors' gold-actor list, after whole-dialogue
text/duration QC. These labels are
performed TopicalChat script emotions, not per-utterance listener consensus.
"""
import csv
import json
from pathlib import Path, PurePosixPath


SOURCE = "https://huggingface.co/datasets/IVLLab/MultiDialog"
LICENCE = "CC BY-NC 4.0"
CREDIT = ("Park et al. Let's Go Real Talk: Spoken Dialogue Model for Face-to-Face Conversation. "
          "ACL 2024. https://doi.org/10.18653/v1/2024.acl-long.860")
DEFAULT_SPLITS = ("test_freq", "test_rare")
GOLD_ACTORS = frozenset("abcefgijk")
EMOTIONS = frozenset(("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"))
RAW_LABELS = {
    "Neutral": "neutral", "Curious to dive deeper": "neutral", "Happy": "happy",
    "Sad": "sad", "Angry": "angry", "Fear": "fearful", "Fearful": "fearful",
    "Disgusting": "disgusted", "Disgusted": "disgusted",
    "Surprising": "surprised", "Surprised": "surprised",
}


def canonical_emotion(raw):
    """Apply the paper's Curious->Neutral merge and spelling normalisation."""
    if raw not in RAW_LABELS:
        raise ValueError("Unknown MultiDialog emotion: " + repr(raw))
    return RAW_LABELS[raw]


def _flag(value):
    if value not in ("True", "False"):
        raise ValueError("Invalid boolean in MultiDialog index: " + repr(value))
    return value == "True"


def _audio_path(dataset_dir, relative):
    parts = PurePosixPath(relative).parts
    if (not parts or parts[0] != "audio" or relative.startswith("/")
            or "\\" in relative or ":" in relative
            or any(part.startswith(".") for part in parts)):
        raise ValueError("Unsafe MultiDialog audio path: " + repr(relative))
    path = dataset_dir.joinpath(*parts)
    if not path.resolve().is_relative_to(dataset_dir):
        raise ValueError("MultiDialog audio path escapes dataset directory")
    return path


def load_multidialog_records(dataset_dir, *, splits=DEFAULT_SPLITS, cohort="gold_both"):
    """Return indexed records with absolute ``path`` and canonical ``emotion``.

    ``dataset_dir`` is ``data/multidialog``, not the parent data directory.
    ``cohort`` is ``gold_both`` (default, including QC), ``gold_actor`` (QC-passing
    target actors, possibly lower-quality partners), or ``all`` (raw audit view,
    including QC-excluded conversations). The index is absent on a fresh
    clone until data restoration; in that case return an empty list. No network,
    audio decoding, model import or inference happens here.
    """
    if cohort not in ("all", "gold_actor", "gold_both"):
        raise ValueError("Unknown MultiDialog cohort: " + repr(cohort))
    if isinstance(splits, str):
        splits = (splits,)
    selected = set(splits)
    if not selected or not selected <= set(DEFAULT_SPLITS):
        raise ValueError("Only retained test_freq/test_rare splits are available")
    dataset_dir = Path(dataset_dir).resolve()
    index = dataset_dir / "utterances.csv"
    if not index.is_file():
        return []
    records = []
    with index.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in selected or row["issues"]:
                continue
            gold_actor = _flag(row["gold_actor"])
            gold_both = _flag(row["dialogue_gold_both"])
            complete = _flag(row["dialogue_complete"])
            qc_excluded = _flag(row["qc_dialogue_excluded"])
            if cohort == "gold_actor" and not (gold_actor and complete and not qc_excluded):
                continue
            if cohort == "gold_both" and not _flag(row["baseline_eligible"]):
                continue
            emotion = canonical_emotion(row["raw_emotion"])
            if row["emotion"] != emotion or emotion not in EMOTIONS:
                raise ValueError("MultiDialog canonical label disagrees with raw label")
            path = _audio_path(dataset_dir, row["audio_path"])
            if not path.is_file():
                raise FileNotFoundError("Restore MultiDialog audio before evaluation: " + str(path))
            records.append({
                "path": path, "audio_path": row["audio_path"], "emotion": emotion,
                "raw_emotion": row["raw_emotion"], "clip_id": row["clip_id"],
                "dialogue_id": row["dialogue_id"], "conv_id": row["conv_id"],
                "turn_index": int(row["turn_index"]), "speaker_id": row["speaker_id"],
                "role": row["role"], "text": row["text"], "split": row["split"],
                "duration_s": float(row["duration_s"]), "gold_actor": gold_actor,
                "dialogue_gold_both": gold_both, "dialogue_complete": complete,
                "qc_dialogue_excluded": qc_excluded, "qc_clip_flags": row["qc_clip_flags"],
                "context_eligible": _flag(row["context_eligible"]),
                "past_3_audio_paths": json.loads(row["past_3_audio_paths"]),
            })
    return records

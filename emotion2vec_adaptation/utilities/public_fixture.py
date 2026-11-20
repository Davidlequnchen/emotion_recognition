"""Build a small local integration fixture from cached real RAVDESS speech.

Nothing is downloaded. Original corpus files are read only, emotion labels come
from the corpus filename convention, and full clips retain their source duration.
The fixture tests software behavior; its small metrics are not benchmark results.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import soundfile as sf

from .config import resolve_path
from adaptation.data import ClipRecord, write_manifest


RAVDESS_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgusted", "08": "surprised",
}
SPEECH_NAME = re.compile(r"03-01-(0[1-8])-0[12]-0[12]-0[12]-(0[1-9]|1\d|2[0-4])")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _properties(path: Path) -> dict:
    info = sf.info(str(path))
    return {"sample_rate": info.samplerate, "channels": info.channels,
            "frames": info.frames, "seconds": info.frames / info.samplerate}


def _validate_settings(settings: dict) -> None:
    expected = {"dataset", "source_directory", "labels", "speakers_per_split",
                "clips_per_class_per_speaker", "unlabeled_speakers", "unlabeled_clips_per_speaker",
                "min_seconds", "max_seconds", "include_holdout_probe", "include_8khz_stereo_probe"}
    if not isinstance(settings, dict) or set(settings) != expected:
        raise ValueError(f"public fixture settings must contain exactly {sorted(expected)}")
    if settings["dataset"] != "ravdess":
        raise ValueError("The public fixture currently supports cached RAVDESS speech only")
    labels = settings["labels"]
    if (not isinstance(labels, list) or not labels or len(labels) != len(set(labels))
            or not set(labels).issubset(RAVDESS_LABELS.values())):
        raise ValueError("public fixture labels must be unique RAVDESS emotion names")
    for name in ("speakers_per_split", "clips_per_class_per_speaker", "unlabeled_speakers",
                 "unlabeled_clips_per_speaker"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"public fixture {name} must be a positive integer")
    for name in ("min_seconds", "max_seconds"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"public fixture {name} must be a finite number")
    if not 0 < settings["min_seconds"] < settings["max_seconds"]:
        raise ValueError("public fixture requires 0 < min_seconds < max_seconds")
    for name in ("include_holdout_probe", "include_8khz_stereo_probe"):
        if not isinstance(settings[name], bool):
            raise ValueError(f"public fixture {name} must be a YAML boolean")


def create_public_fixture(directory: Path, settings: dict) -> Path:
    """Copy deterministic, speaker-disjoint real-speech subsets into a new folder.

    Output contains labeled.csv, unlabeled.csv, source_provenance.json and audio/.
    All supplied filename emotions retain their original meaning: calm remains
    calm when requested, rather than silently being relabeled neutral. Unlabeled
    CSV rows contain no class labels even though provenance retains source facts.

    The optional heldout probe intentionally copies a validation recording under
    a different declared group, testing byte-content exclusions. The optional
    8 kHz stereo probe resamples one independent unlabeled recording without
    cropping, testing the normal audio reader's telephony/stereo preprocessing.
    """
    _validate_settings(settings)
    source_root = resolve_path(settings["source_directory"])
    if not source_root.is_dir():
        raise ValueError(f"Cached RAVDESS audio directory does not exist: {source_root}")
    labels = settings["labels"]
    by_speaker = defaultdict(lambda: defaultdict(list))
    for source in sorted(source_root.glob("Actor_*/*.wav")):
        match = SPEECH_NAME.fullmatch(source.stem)
        if match is None:
            continue
        code, speaker = match.groups()
        if source.parent.name != f"Actor_{speaker}":
            raise ValueError(f"RAVDESS actor filename/folder mismatch: {source}")
        label = RAVDESS_LABELS[code]
        if label not in labels:
            continue
        duration = sf.info(str(source)).duration
        if settings["min_seconds"] <= duration <= settings["max_seconds"]:
            by_speaker[speaker][label].append(source.resolve())
    per_class = settings["clips_per_class_per_speaker"]
    speakers = sorted(speaker for speaker, buckets in by_speaker.items()
                      if all(len(buckets[label]) >= per_class for label in labels)
                      and sum(map(len, buckets.values())) >= settings["unlabeled_clips_per_speaker"])
    supervised_speakers = settings["speakers_per_split"] * 3
    needed = supervised_speakers + settings["unlabeled_speakers"]
    if len(speakers) < needed:
        raise ValueError(f"Cached RAVDESS has only {len(speakers)} eligible speakers; need {needed} "
                         "for the requested complete clips, classes and independent splits")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    provenance, labeled, unlabeled = [], [], []

    def materialize(source, split, label, speaker, probe=None, transform=False):
        group = "public_probe_unknown_group" if probe else f"ravdess_actor_{speaker}"
        clip_id = f"{probe}_{source.stem}" if probe else source.stem
        category = "unlabeled" if label is None else "labeled"
        destination = root / "audio" / category / split / f"{clip_id}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        transformations = []
        if transform:
            import torch
            from torchaudio.functional import resample

            samples, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
            mono = torch.from_numpy(samples.mean(axis=1))
            mono_8khz = resample(mono, sample_rate, 8000).numpy()
            stereo = np.stack([mono_8khz, mono_8khz], axis=1)
            sf.write(str(destination), stereo, 8000, subtype="PCM_16")
            transformations = ["sinc resample full clip to 8000 Hz",
                               "mean original channels, duplicate mono into two channels", "PCM_16 encode"]
        else:
            shutil.copyfile(source, destination)
        original_label = RAVDESS_LABELS[source.stem.split("-")[2]]
        provenance.append({
            "fixture_path": str(destination.relative_to(root)), "source_path": str(source),
            "source_relative_path": str(source.relative_to(source_root)), "source_sha256": _digest(source),
            "fixture_sha256": _digest(destination), "source_audio": _properties(source),
            "fixture_audio": _properties(destination), "source_label": original_label,
            "manifest_label": label, "source_speaker": speaker, "group": group, "split": split,
            "transformations": transformations, "intentional_heldout_copy": bool(probe),
        })
        return ClipRecord(destination, label, group, split, clip_id)

    per_split = settings["speakers_per_split"]
    for offset, split in enumerate(("train", "validation", "test")):
        for speaker in speakers[offset * per_split:(offset + 1) * per_split]:
            for label in labels:
                for source in by_speaker[speaker][label][:per_class]:
                    labeled.append(materialize(source, split, label, speaker))
    for speaker in speakers[supervised_speakers:needed]:
        candidates = sorted(source for label in labels for source in by_speaker[speaker][label])
        for source in candidates[:settings["unlabeled_clips_per_speaker"]]:
            transform = settings["include_8khz_stereo_probe"] and not unlabeled
            unlabeled.append(materialize(source, "train", None, speaker, transform=transform))
    if settings["include_holdout_probe"]:
        heldout = next(row for row in labeled if row.split == "validation")
        source = source_root / f"Actor_{heldout.clip_id.split('-')[-1]}" / f"{heldout.clip_id}.wav"
        unlabeled.append(materialize(source, "train", None, heldout.clip_id.split("-")[-1],
                                     probe="heldout_copy_probe"))
    write_manifest(labeled, root / "labeled.csv")
    write_manifest(unlabeled, root / "unlabeled.csv")
    manifest = {
        "dataset": "RAVDESS", "source_directory": str(source_root), "synthetic_audio": False,
        "source_url": "https://zenodo.org/records/1188976", "license": "CC BY-NC-SA 4.0",
        "credit": "Livingstone and Russo (2018), The Ryerson Audio-Visual Database of Emotional Speech and Song",
        "label_definition": "Acted source emotion from the third filename field; not listener-adjudicated emotion",
        "purpose": "Local software integration validation; not internal data and not a new benchmark estimate",
        "settings": settings, "labeled_clips": len(labeled), "unlabeled_candidate_clips": len(unlabeled),
        "expected_holdout_exclusions": int(settings["include_holdout_probe"]), "recordings": provenance,
    }
    (root / "source_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return root

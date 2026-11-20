"""Create a small synthetic audio fixture, never real internal training data.

Run ``python emotion2vec_adaptation/utilities/generate_demo.py`` without arguments. The
generated emotion labels are arbitrary test values and have no acoustic meaning.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from utilities.config import REPOSITORY_ROOT


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split", "clip_id"])
        writer.writeheader()
        writer.writerows(rows)


def _write_wave(path: Path, rng: np.random.Generator) -> None:
    """Three seconds of tone, harmonics and noise; contains no recorded speech."""
    rate = 16000
    time = np.arange(3 * rate, dtype=np.float64) / rate
    fundamental = rng.uniform(150, 350)
    phase = rng.uniform(0, 2 * np.pi)
    modulation = 0.6 + 0.3 * np.sin(2 * np.pi * rng.uniform(1.5, 4) * time)
    signal = modulation * (0.18 * np.sin(2 * np.pi * fundamental * time + phase)
                           + 0.05 * np.sin(2 * np.pi * fundamental * 2 * time))
    signal += rng.normal(0, 0.012, len(time))
    fade = 160
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal.astype(np.float32), rate, subtype="PCM_16")


def create_demo_dataset(directory: Path | None = None) -> Path:
    """Create exclusive labeled/unlabeled fixtures and return their root.

    labeled.csv has 12 clips: two per class per explicit split. Distinct customer
    groups span neither split nor supervised/unlabeled fixtures. unlabeled.csv
    has four separate clips marked train. Every manifest path is root-relative.
    Existing directories are rejected so no curated data can be overwritten.
    """
    root = (Path(directory) if directory is not None else REPOSITORY_ROOT / "curated_dataset" / "demo")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261116)
    labeled = []
    for split in ("train", "validation", "test"):
        for customer in range(1, 3):
            group = f"demo_{split}_customer_{customer:02d}"
            for label in ("neutral", "happy"):
                clip_id = f"{group}_{label}"
                relative = Path("audio/labeled") / split / f"{clip_id}.wav"
                _write_wave(root / relative, rng)
                labeled.append({"path": relative.as_posix(), "label": label, "group": group,
                                "split": split, "clip_id": clip_id})
    unlabeled = []
    for customer in range(1, 3):
        group = f"demo_unlabeled_customer_{customer:02d}"
        for index in range(1, 3):
            clip_id = f"{group}_{index:02d}"
            relative = Path("audio/unlabeled") / group / f"{clip_id}.wav"
            _write_wave(root / relative, rng)
            unlabeled.append({"path": relative.as_posix(), "label": "", "group": group,
                              "split": "train", "clip_id": clip_id})
    _write_csv(root / "labeled.csv", labeled)
    _write_csv(root / "unlabeled.csv", unlabeled)
    _write_csv(root / "clips.csv", [{**row, "label": ""} for row in labeled])
    votes = [{"clip_id": row["clip_id"], "task_id": task, "annotation_id": task * 10 + annotator,
              "annotator_id": annotator, "lead_time": 1.0, "decision": row["label"],
              "status": "accepted", "emotion": row["label"]}
             for task, row in enumerate(labeled, 1) for annotator in (1, 2)]
    (root / "normalized.json").write_text(json.dumps(votes, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    manifest = {
        "synthetic": True, "contains_recorded_speech": False,
        "labels_are_arbitrary_test_values": True, "seed": 20261116,
        "sample_rate": 16000, "seconds_per_clip": 3,
        "labels": ["neutral", "happy"], "labeled_clips": len(labeled), "unlabeled_clips": len(unlabeled),
        "purpose": "Software integration validation only; not an emotion recognition benchmark.",
    }
    (root / "demo_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Synthetic integration fixture\n\n"
        "These 16 WAV files contain generated tones, harmonics and noise, with no recorded speech. "
        "The neutral/happy labels are arbitrary software test values; resulting classification "
        "metrics provide no evidence of emotion recognition quality.\n\n"
        "labeled.csv contains 12 clips in customer-disjoint train/validation/test splits. "
        "unlabeled.csv contains four other training clips. clips.csv and normalized.json "
        "demonstrate joining unanimously accepted Label Studio votes. Audio paths are relative "
        "to these CSV files. demo_manifest.json records generation settings.\n\n"
        "The generator refuses an existing destination. Delete this disposable fixture explicitly "
        "or choose a new destination when generating another copy.\n",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    print(f"Created synthetic validation fixture: {create_demo_dataset()}")

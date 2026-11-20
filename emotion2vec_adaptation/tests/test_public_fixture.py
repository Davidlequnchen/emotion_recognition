"""Verify real cached speech fixtures without downloading or loading a model."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptation.audio import AudioReader, AudioSettings
from utilities.config import REPOSITORY_ROOT
from adaptation.data import read_manifest, read_unlabeled, split_records
from utilities.public_fixture import create_public_fixture


SETTINGS = {
    "dataset": "ravdess", "source_directory": "emotion_recognition_benchmark/data/ravdess/audio",
    "labels": ["neutral", "happy"], "speakers_per_split": 2, "clips_per_class_per_speaker": 2,
    "unlabeled_speakers": 2, "unlabeled_clips_per_speaker": 2, "min_seconds": 3.0,
    "max_seconds": 12.0, "include_holdout_probe": True, "include_8khz_stereo_probe": True,
}


@unittest.skipUnless((REPOSITORY_ROOT / SETTINGS["source_directory"]).is_dir(),
                     "Cached RAVDESS data unavailable; fixtures never download data")
class PublicFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = create_public_fixture(Path(cls.temp.name) / "public_fixture", SETTINGS)
        cls.provenance = json.loads((cls.root / "source_provenance.json").read_text())

    def test_real_labels_and_groups_are_preserved_in_complete_disjoint_splits(self):
        labeled = read_manifest(self.root / "labeled.csv", labels=SETTINGS["labels"])
        splits = split_records(labeled)
        self.assertEqual({name: len(rows) for name, rows in splits.items()},
                         {"train": 8, "validation": 8, "test": 8})
        for split, rows in splits.items():
            self.assertEqual(len({row.group for row in rows}), 2)
            self.assertEqual({row.label for row in rows}, {"neutral", "happy"})
            for row in rows:
                self.assertEqual(row.path.stem.split("-")[2], "01" if row.label == "neutral" else "03")
                self.assertEqual(row.group, f"ravdess_actor_{row.path.stem.split('-')[-1]}")
                self.assertTrue(row.path.is_relative_to(self.root))

    def test_renamed_holdout_copy_is_removed_and_format_probe_decodes(self):
        labeled = read_manifest(self.root / "labeled.csv")
        heldout = [row for row in labeled if row.split != "train"]
        candidates = read_unlabeled(self.root / "unlabeled.csv")
        training = read_unlabeled(self.root / "unlabeled.csv", exclude_records=heldout)
        self.assertEqual((len(candidates), len(training)), (5, 4))
        removed = set(candidates) - set(training)
        self.assertEqual({row.group for row in removed}, {"public_probe_unknown_group"})
        format_probe = next(row for row in self.provenance["recordings"]
                            if row["transformations"])
        self.assertEqual(format_probe["fixture_audio"]["sample_rate"], 8000)
        self.assertEqual(format_probe["fixture_audio"]["channels"], 2)
        wave = AudioReader(AudioSettings()).read(self.root / format_probe["fixture_path"])
        self.assertGreater(wave.numel(), 3 * 16000)
        self.assertEqual(wave.ndim, 1)

    def test_provenance_matches_unchanged_sources_and_actual_copies(self):
        self.assertFalse(self.provenance["synthetic_audio"])
        self.assertEqual(len(self.provenance["recordings"]), 29)
        for row in self.provenance["recordings"]:
            source = Path(row["source_path"])
            copied = self.root / row["fixture_path"]
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), row["source_sha256"])
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), row["fixture_sha256"])
            if not row["transformations"]:
                self.assertEqual(row["source_sha256"], row["fixture_sha256"])
            self.assertAlmostEqual(row["source_audio"]["seconds"], row["fixture_audio"]["seconds"], delta=1/8000)
        with self.assertRaises(FileExistsError):
            create_public_fixture(self.root, SETTINGS)

    def test_invalid_fixture_settings_fail_without_creating_destination(self):
        destination = self.root.parent / "invalid_fixture"
        with self.assertRaisesRegex(ValueError, "cached RAVDESS"):
            create_public_fixture(destination, {**SETTINGS, "dataset": "invented"})
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

"""Offline MultiDialog cohort and restore-contract checks, without decoding audio."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "common"))
from multidialog_dataset import load_multidialog_records


class MultiDialogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="multidialog_fixture_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = []
        for clip, split, gold, qc in [("chosen", "test_freq", True, False),
                                     ("qc_rejected", "test_freq", True, True),
                                     ("other_actor", "test_rare", False, False)]:
            relative = "audio/" + split + "/dialogue/" + clip + ".wav"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()  # A metadata loader must not decode audio or import model libraries.
            self.rows.append({
                "split": split, "issues": "", "gold_actor": "True",
                "dialogue_gold_both": str(gold), "dialogue_complete": "True",
                "qc_dialogue_excluded": str(qc), "baseline_eligible": str(gold and not qc),
                "raw_emotion": "Curious to dive deeper", "emotion": "neutral",
                "audio_path": relative, "clip_id": clip, "dialogue_id": "dialogue",
                "conv_id": "dialogue", "turn_index": "0", "speaker_id": "a",
                "role": "human", "text": "Tell me more.", "duration_s": "1.5",
                "qc_clip_flags": "speed" if qc else "", "context_eligible": "False",
                "past_3_audio_paths": "[]",
            })
        self.write_index()

    def write_index(self):
        with (self.root / "utterances.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_default_qc_and_audit_views_are_distinct(self):
        default = load_multidialog_records(self.root)
        self.assertEqual([row["clip_id"] for row in default], ["chosen"])
        self.assertEqual(default[0]["emotion"], "neutral")
        self.assertTrue(default[0]["path"].is_absolute())
        self.assertEqual(len(load_multidialog_records(self.root, cohort="all")), 3)
        self.assertEqual(len(load_multidialog_records(self.root, cohort="gold_actor")), 2)
        self.assertEqual(load_multidialog_records(self.root, splits="test_rare"), [])

    def test_metadata_only_clone_does_not_silently_skip_missing_audio(self):
        (self.root / self.rows[0]["audio_path"]).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Restore MultiDialog"):
            load_multidialog_records(self.root)
        (self.root / "utterances.csv").unlink()
        self.assertEqual(load_multidialog_records(self.root), [])

    def test_index_cannot_redirect_audio_outside_dataset(self):
        self.rows[0]["audio_path"] = "audio/../../outside.wav"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "Unsafe MultiDialog"):
            load_multidialog_records(self.root)

    def test_canonical_labels_must_agree_with_original(self):
        self.rows[0]["emotion"] = "happy"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "canonical label"):
            load_multidialog_records(self.root)

    def test_unsupported_scope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only retained"):
            load_multidialog_records(self.root, splits="train")
        with self.assertRaisesRegex(ValueError, "Unknown MultiDialog cohort"):
            load_multidialog_records(self.root, cohort="gold")


if __name__ == "__main__":
    unittest.main()

"""Header-derived timing and exact cohort/identity tests; no real data or models."""
import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import consensus
import duration_inventory as inventory


class DurationInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)

    def audio(self, relative, frames, rate, channels=1):
        path = self.project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shape = (frames,) if channels == 1 else (frames, channels)
        sf.write(path, np.zeros(shape), rate, subtype="PCM_16")
        return relative

    def run_for(self, paths, *, dataset="crema_d", gold="sad", model="fixture"):
        return consensus.Run(model, dataset, Path("unused.json"), {}, ["sad", "neutral"], {},
                             {p: {"gold": gold, "prediction": gold} for p in paths}, [])

    def test_real_header_duration_uses_rate_frames_and_preserves_silent_clip(self):
        a = self.audio(inventory.KNOWN_SILENT, 12000, 48000, 2)
        b = self.audio("data/crema_d/AudioWAV/1001_DFA_SAD_XX.wav", 8000, 16000)
        empty = self.audio("data/crema_d/AudioWAV/1002_DFA_SAD_XX.wav", 0, 16000)
        run = self.run_for([a, b, empty])
        with mock.patch.object(sf, "read", side_effect=AssertionError("Must not decode waveform")):
            rows, manifest = inventory.build_inventory(self.project, {"fixture": {"crema_d": run}}, workers=2)
        indexed = {r["audio_path"]: r for r in rows}
        self.assertEqual(indexed[a]["duration_seconds"], .25)
        self.assertEqual(indexed[a]["sampling_rate"], 48000)
        self.assertEqual(indexed[a]["channels"], 2)
        self.assertTrue(indexed[a]["known_silent_recording"])
        self.assertFalse(indexed[a]["empty_audio"])
        self.assertEqual(indexed[b]["duration_seconds"], .5)
        self.assertTrue(indexed[empty]["empty_audio"])
        self.assertEqual(manifest["datasets"]["crema_d"]["groups"], 3)
        self.assertEqual({r["group_kind"] for r in rows}, {"speaker"})

    def test_shared_paths_are_counted_once_and_conflicting_gold_fails(self):
        path = self.audio("data/crema_d/AudioWAV/1001_DFA_SAD_XX.wav", 1600, 16000)
        first = self.run_for([path], model="one")
        second = self.run_for([path], model="two")
        selected = {"one": {"crema_d": first}, "two": {"crema_d": second}}
        rows, _ = inventory.build_inventory(self.project, selected, workers=1)
        self.assertEqual(len(rows), 1)
        second.rows[path]["gold"] = "neutral"
        with self.assertRaisesRegex(ValueError, "ground truth differs"):
            inventory.build_inventory(self.project, selected, workers=1)

    def test_dialogue_group_comes_from_exact_metadata_path_and_duplicate_fails(self):
        path = self.audio("data/emotiontalk/Audio/example.wav", 1600, 16000)
        source = self.project / "data/emotiontalk/utterances.csv"
        fields = ["audio_path", "dialogue_id", "speaker_id", "group_id"]
        record = {"audio_path": "Audio/example.wav", "dialogue_id": "dialogue42", "speaker_id": "01", "group_id": "G42"}
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader(); writer.writerow(record)
        run = self.run_for([path], dataset="emotiontalk")
        selected = {"fixture": {"emotiontalk": run}}
        rows, manifest = inventory.build_inventory(self.project, selected, workers=1)
        self.assertEqual(rows[0]["group"], "emotiontalk:dialogue:dialogue42")
        self.assertEqual(rows[0]["group_kind"], "dialogue")
        self.assertEqual(rows[0]["speaker_id"], "01")
        self.assertEqual(rows[0]["source_group_id"], "G42")
        self.assertEqual(len(manifest["grouping_sources"]), 1)
        with source.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(record)
        with self.assertRaisesRegex(ValueError, "duplicate selected audio"):
            inventory.build_inventory(self.project, selected, workers=1)

    def test_audio_mutation_during_header_read_is_rejected(self):
        path = self.audio("data/crema_d/AudioWAV/1001_DFA_SAD_XX.wav", 1600, 16000)
        run = self.run_for([path])
        original = sf.info
        def changed(filename):
            result = original(filename)
            stat = Path(filename).stat()
            os.utime(filename, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000))
            return result
        with mock.patch.object(sf, "info", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed while its header"):
                inventory.build_inventory(self.project, {"fixture": {"crema_d": run}}, workers=1)


if __name__ == "__main__":
    unittest.main()

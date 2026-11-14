"""Native intensity-scale preservation; no model inference or real datasets."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import intensity_inventory as intensity


class IntensityInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)

    def record(self, dataset, path, gold):
        return {"dataset": dataset, "audio_path": path, "gold": gold}

    def metadata(self, dataset, filename, fields, rows):
        source = self.project / "data" / dataset / filename
        source.parent.mkdir(parents=True, exist_ok=True)
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return source

    def test_acted_levels_keep_unknown_and_neutral_distinct(self):
        records = [self.record("crema_d", f"data/crema_d/AudioWAV/1001_IEO_ANG_{level}.wav", "angry")
                   for level in ["LO", "MD", "HI", "XX"]]
        records += [self.record("mead", "data/mead/audio/M001/neutral/level_1/001.wav", "neutral"),
                    self.record("tess", "data/tess/OAF_food_neutral.wav", "neutral")]
        rows, manifest = intensity.build_intensity_inventory(self.project, records)
        self.assertEqual([row["intensity_level"] for row in rows[:4]], [1, 2, 3, None])
        self.assertEqual(len({row["intensity_pair_key"] for row in rows[:3]}), 1)
        self.assertFalse(rows[3]["intensity_available"])
        self.assertEqual(rows[3]["intensity_missing_reason"], "unspecified_XX")
        self.assertTrue(rows[4]["intensity_available"])
        self.assertFalse(rows[4]["intensity_applicable"])
        self.assertEqual(rows[4]["intensity_applicability_reason"], "benchmark_neutral_excluded")
        self.assertFalse(rows[5]["intensity_available"])
        self.assertEqual(manifest["datasets"]["tess"]["available"], 0)

    def test_ravdess_normal_strong_and_native_neutral_validation(self):
        records = [self.record("ravdess", f"data/ravdess/audio/Actor_01/03-01-05-{code}-01-01-01.wav", "angry")
                   for code in ["01", "02"]]
        rows, _ = intensity.build_intensity_inventory(self.project, records)
        self.assertEqual([row["intensity_name"] for row in rows], ["normal", "strong"])
        self.assertEqual(rows[0]["intensity_pair_key"], rows[1]["intensity_pair_key"])
        invalid = self.record("ravdess", "data/ravdess/audio/Actor_01/03-01-01-02-01-01-01.wav", "neutral")
        with self.assertRaisesRegex(ValueError, "neutral cannot have strong"):
            intensity.build_intensity_inventory(self.project, [invalid])

    def test_mcae_uses_listener_intensity_not_acoustic_db(self):
        source = self.project / "data/mcae_spps/Corpus information.xlsx"
        source.parent.mkdir(parents=True)
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Sheet1"
        sheet.append(["Filename", "emotion", "Intensity", "meanIntensity"])
        sheet.append([13171, "Anger", 6.725, -21.5])
        workbook.save(source)
        rows, manifest = intensity.build_intensity_inventory(self.project,
            [self.record("mcae_spps", "data/mcae_spps/audio/Anger/13171.wav", "angry")])
        self.assertEqual(rows[0]["intensity_level"], 6.725)
        self.assertEqual(rows[0]["intensity_scale_kind"], "perceived_continuous_mean")
        self.assertEqual(rows[0]["intensity_pair_key"], "")
        self.assertIn("Intensity", rows[0]["intensity_source"])
        self.assertEqual(len(manifest["metadata_sources"]), 1)

    def test_thai_script_only_and_keys_preserve_exact_script(self):
        fields = ["audio_id", "assigned_emo", "script_intensity", "turn_type", "session_id", "actor_id", "script_sent", "script_trial"]
        common = {"assigned_emo": "Angry", "session_id": "studio001", "actor_id": "001", "script_sent": "same sentence", "script_trial": "1.0"}
        self.metadata("thai_ser", "labels.csv", fields,
            [common | {"audio_id": f"{code}.flac", "script_intensity": level, "turn_type": "script" if level else "impro"}
             for code, level in [("one", "Low"), ("two", "High"), ("three", "")]])
        rows, _ = intensity.build_intensity_inventory(self.project,
            [self.record("thai_ser", f"data/thai_ser/audio/{name}.flac", "angry") for name in ["one", "two", "three"]])
        self.assertEqual([r["intensity_level"] for r in rows], [1, 2, None])
        self.assertEqual(rows[0]["intensity_pair_key"], rows[1]["intensity_pair_key"])
        self.assertEqual(rows[2]["intensity_missing_reason"], "improvised_no_script_intensity")

    def test_dailytalk_keeps_public_ecss_levels_and_rejects_gold_mismatch(self):
        self.metadata("dailytalk", "utterances.csv", ["audio_path", "ecss_emotion", "original_emotion", "ecss_intensity"],
            [{"audio_path": "data/0/0_1_d0.wav", "ecss_emotion": "happiness", "original_emotion": "happiness", "ecss_intensity": "3"}])
        record = self.record("dailytalk", "data/dailytalk/data/0/0_1_d0.wav", "happy")
        rows, _ = intensity.build_intensity_inventory(self.project, [record])
        self.assertEqual(rows[0]["intensity_level"], 3)
        self.assertEqual(rows[0]["intensity_scale_kind"], "annotated_ordinal")
        with self.assertRaisesRegex(ValueError, "differs from inventory"):
            intensity.build_intensity_inventory(self.project, [record | {"gold": "sad"}])

    def test_duplicate_samples_and_inconsistent_filename_gold_fail(self):
        record = self.record("crema_d", "data/crema_d/AudioWAV/1001_IEO_ANG_LO.wav", "angry")
        with self.assertRaisesRegex(ValueError, "Duplicated inventory"):
            intensity.build_intensity_inventory(self.project, [record, record])
        with self.assertRaisesRegex(ValueError, "differs from inventory"):
            intensity.build_intensity_inventory(self.project, [record | {"gold": "sad"}])


if __name__ == "__main__":
    unittest.main()

"""Metadata-only dialogue loader/history regression checks; no audio decoding, weights or inference.

Run from emotion_recognition/: python -m unittest discover -s test_scripts -p test_dialogue_datasets.py -v
"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts" / "common"))
import dialogue_datasets as dialogue
import eval_datasets as evaluation


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class DialogueDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="dialogue_metadata_")
        self.addCleanup(self.tmp.cleanup)
        self.data = Path(self.tmp.name)
        self.rows = []
        for clip, speaker, start, end in [("past0", "01", 0, 1), ("past1", "02", 1.5, 3.5),
                                          ("overlap", "01", 3.9, 4.0005), ("target", "02", 4, 5),
                                          ("future", "01", 5, 6)]:
            self.emotion_row(clip, "train", "dialogueA", speaker, start, end)
        self.emotion_row("solo", "train", "soloDialogue", "03", 0, 1)
        self.emotion_row("test0", "test", "testDialogue", "01", 0, 1)
        self.emotion_row("test1", "test", "testDialogue", "02", 1, 2)
        self.emotion_row("valid0", "validation", "validDialogue", "01", 0, 1)
        self.emotion_row("valid1", "validation", "validDialogue", "02", 1, 2)
        write_csv(self.data / "emotiontalk" / "utterances.csv", self.rows)
        daily = []
        for clip, split, conv, turn, speaker, emotion in [
                ("0_0_d0", "train", "0", 0, "0", "none"),
                ("1_1_d0", "train", "0", 1, "1", "happiness"),
                ("0_0_d1451", "train", "1451", 0, "0", "none"),
                ("0_0_d1", "validation", "1", 0, "0", "fear"),
                ("1_1_d1", "validation", "1", 1, "1", "disgust"),
                ("0_0_d175", "", "175", 0, "0", "")]:
            path = Path("data") / conv / (clip + ".wav")
            audio = self.data / "dailytalk" / path
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.touch()  # Metadata-only loaders must not attempt to decode this empty fixture file.
            daily.append({"audio_path": path.as_posix(), "audio_id": clip, "dialogue_id": conv,
                          "turn_index": turn, "speaker_id": speaker, "text": f"Spoken words {turn}",
                          "original_emotion": emotion or "no emotion", "ecss_emotion": emotion,
                          "ecss_split": split, "duration_s": 1.25})
        write_csv(self.data / "dailytalk" / "utterances.csv", daily)
        self.patch = mock.patch.object(evaluation, "DATA", self.data)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def emotion_row(self, clip, split, conv, speaker, start, end):
        audio = Path("Audio/wav") / conv / (clip + ".wav")
        sidecar = Path("Audio/json") / conv / (clip + ".json")
        for relative in (audio, sidecar):
            (self.data / "emotiontalk" / relative).parent.mkdir(parents=True, exist_ok=True)
        (self.data / "emotiontalk" / audio).touch()
        # Released gold differs from both majority-vote and caption emotion in this fixture.
        (self.data / "emotiontalk" / sidecar).write_text(json.dumps({
            "emotion_result": "happy", "speaker_id": speaker, "content": "[angry] Actual spoken words",
            "paragraphs": {"startTime": start, "endTime": end},
            "data": {str(i): {"emotion": "sad"} for i in range(5)},
            "sourceAttr": {"caption_1": "A neutral voice"}}))
        self.rows.append({"clip_id": clip, "audio_path": audio.as_posix(), "json_path": sidecar.as_posix(),
                          "group_id": "group", "dialogue_id": conv, "speaker_id": speaker,
                          "speaker_group_key": f"group_{speaker}", "utterance_index": len(self.rows),
                          "start_s": start, "end_s": end, "duration_s": end - start,
                          "text": "[angry] Actual spoken words",
                          "text_without_markup": "【Sobbing】 Actual [laughing] spoken words",
                          "emotion": "happy", "split": split, "dialogue_speaker_count": 2,
                          "primary_dyadic_history_target": True})

    def records(self, key="emotiontalk", split="train", **kwargs):
        return dialogue.load_dialogue_records(key, split, data_root=self.data, **kwargs)

    def test_defaults_keep_heldout_splits_and_original_gold(self):
        emotion = evaluation.emotiontalk()
        self.assertEqual(len(emotion.clips), 2)
        self.assertEqual({c["audio"].stem for c in emotion.clips}, {"test0", "test1"})
        self.assertEqual({c["acted"] for c in emotion.clips}, {"happy"})
        self.assertTrue(all(c["listener"] == "" for c in emotion.clips))
        self.assertEqual(emotion.subset, evaluation.default_evaluation_subset("emotiontalk"))
        daily = evaluation.dailytalk()
        self.assertEqual({c["audio"].stem for c in daily.clips}, {"0_0_d1", "1_1_d1"})
        self.assertEqual({c["acted"] for c in daily.clips}, {"fearful", "disgusted"})
        self.assertEqual(daily.classes, dialogue.CLASSES)
        self.assertEqual(daily.subset, evaluation.default_evaluation_subset("dailytalk"))
        self.assertIn("independent", " ".join(daily.caveats))

    def test_all_splits_aliases_and_primary_dyads(self):
        self.assertEqual(len(self.records(split="all")), 10)
        self.assertEqual(len(self.records(split=["train", "test"])), 8)
        self.assertEqual(len(self.records(split="dev")), 2)
        self.assertEqual(len(self.records()), 6)
        self.assertEqual(len(self.records(primary_only=True)), 5)
        # The cached primary flag deliberately says True for the solo fixture; derive speaker count instead.
        self.assertNotIn("solo", {r["id"] for r in self.records(primary_only=True)})
        self.assertEqual(len(self.records("dailytalk", "all")), 6)
        self.assertEqual(len(self.records("dailytalk", "train", primary_only=True)), 2)
        self.assertEqual(self.records("dailytalk", "unassigned")[0]["gold"], "neutral")
        with self.assertRaisesRegex(ValueError, "no DailyTalk test"):
            self.records("dailytalk", "test")

    def test_completed_history_is_exact_past_sorted_and_annotation_free(self):
        records = self.records()
        target = next(r for r in records if r["id"] == "target")
        records += [records[0] | {"id": "otherDialogue", "dialogue": "elsewhere"},
                    records[0] | {"id": "otherSplit", "split": "validation"},
                    records[0] | {"id": "otherDataset", "dataset": "dailytalk"}]
        history = dialogue.completed_history(target, list(reversed(records)))
        self.assertEqual([r["id"] for r in history], ["past0", "past1"])
        self.assertNotIn("overlap", {r["id"] for r in history})  # end is only 0.5ms after target start.
        self.assertEqual([r["id"] for r in dialogue.completed_history(target, records, max_turns=1)], ["past1"])
        self.assertEqual(dialogue.completed_history(target, records, max_turns=0), [])
        for row in history:
            self.assertTrue(set(row) <= set(dialogue.CONTEXT_FIELDS))
            self.assertFalse(set(row) & {"gold", "emotion", "label_source", "votes", "caption", "json_path"})
            self.assertEqual(row["transcript"], "Actual spoken words")
        self.assertEqual(target["gold"], "happy")  # scoring records are not mutated or stripped.

    def test_history_includes_exact_boundary_but_never_target_or_future(self):
        target = next(r for r in self.records() if r["id"] == "target")
        past = target | {"id": "boundary", "start": 2.0, "end": 4.0}
        future = target | {"id": "futureCopy", "start": 4.1, "end": 4.2}
        self.assertEqual([r["id"] for r in dialogue.completed_history(target, [future, target, past])], ["boundary"])

    def test_transcript_cleaning_removes_both_annotation_styles_even_in_cached_plain_text(self):
        self.rows[0]["text_without_markup"] = "【laughing】 [Sobbing]"
        self.rows[1]["text_without_markup"] = "保留正常语义 【laughing】 hello[Sobbing]world。"
        write_csv(self.data / "emotiontalk" / "utterances.csv", self.rows)
        records = {row["id"]: row for row in self.records()}
        self.assertEqual(records["past0"]["transcript"], "")
        self.assertEqual(records["past1"]["transcript"], "保留正常语义 hello world。")
        target = records["target"]
        history = dialogue.completed_history(target, list(records.values()))
        self.assertEqual([row["transcript"] for row in history], ["", "保留正常语义 hello world。"])
        for row in history:
            self.assertNotIn("gold", row)
            self.assertNotIn("sourceAttr", row)
            self.assertNotIn("caption", row)
            self.assertNotIn("votes", row)

    def test_dailytalk_history_uses_turn_order_without_fabricating_timestamps(self):
        records = self.records("dailytalk", "validation")
        self.assertTrue(all(r["start"] is None and r["end"] is None for r in records))
        history = dialogue.completed_history(records[1], records)
        self.assertEqual([r["id"] for r in history], ["0_0_d1"])
        self.assertEqual(history[0]["time_basis"], "turn_order")
        self.assertNotIn("gold", history[0])

    def test_subset_selection_and_index_gold_mismatch_fail_clearly(self):
        selected = evaluation.emotiontalk(subset=["test1"])
        self.assertEqual([c["audio"].stem for c in selected.clips], ["test1"])
        with self.assertRaisesRegex(ValueError, "Subset IDs absent"):
            evaluation.emotiontalk(subset=["not-a-test-id"])
        selected_row = next(row for row in self.rows if row["clip_id"] == "test1")
        sidecar = self.data / "emotiontalk" / selected_row["json_path"]
        original = json.loads(sidecar.read_text())
        sidecar.write_text(json.dumps(original | {"emotion_result": "sad"}))
        with self.assertRaisesRegex(ValueError, "gold mismatch"):
            evaluation.emotiontalk()

    def test_multidialog_wrapper_is_lazy_and_keeps_provider_cohort(self):
        audio = self.data / "multidialog" / "clip.wav"
        audio.parent.mkdir(); audio.touch()
        provider = mock.Mock(return_value=[{"path": audio, "audio_path": "clip.wav", "emotion": "neutral"}])
        with mock.patch.dict(sys.modules, {"multidialog_dataset": SimpleNamespace(load_multidialog_records=provider)}):
            result = evaluation.multidialog()
        self.assertEqual(result.clips[0]["audio"], audio)
        self.assertEqual(result.subset, evaluation.default_evaluation_subset("multidialog"))
        provider.assert_called_once_with(self.data / "multidialog", splits=("test_freq", "test_rare"), cohort="gold_both")
        with mock.patch.dict(sys.modules, {"multidialog_dataset": SimpleNamespace(load_multidialog_records=provider)}):
            self.assertNotEqual(evaluation.multidialog(splits="test_freq").subset, result.subset)
            self.assertNotEqual(evaluation.multidialog(cohort="all").subset, result.subset)

    def test_registration_appends_new_datasets_and_separates_api_from_public_audio(self):
        previous = ["crema_d", "mcae_spps", "thai_ser", "nemo", "mess", "esd", "ravdess", "tess",
                    "jl_corpus", "mead", "meld", "emovo", "emozionalmente", "savee"]
        self.assertEqual(list(evaluation.LOADERS)[:len(previous)], previous)
        for key in ["emotiontalk", "dailytalk", "multidialog"]:
            self.assertTrue(callable(evaluation.LOADERS[key]))
            self.assertFalse(evaluation.SOURCES[key]["public_audio"])
            self.assertTrue(evaluation.SOURCES[key]["api_audio"])
        self.assertFalse(evaluation.SOURCES["esd"]["api_audio"])


if __name__ == "__main__":
    unittest.main()

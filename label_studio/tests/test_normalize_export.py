"""Keep Other, uncertainty and rejection distinct without rewriting old votes."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ser_normalize_export", PROJECT / "normalize_export.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def native_annotation(decision, identifier=11, annotator=3):
    return {"id": identifier, "completed_by": annotator, "lead_time": 4.5,
            "result": [{"from_name": "decision", "to_name": "audio", "type": "choices",
                        "value": {"choices": [decision]}}]}


def native_task(*annotations):
    return {"id": 42, "data": {"clip_id": "original-clip-id"}, "annotations": list(annotations)}


class NormalizeDecisionsTests(unittest.TestCase):
    def test_other_is_an_accepted_emotion_and_annotator_identity_is_preserved(self):
        rows, cancelled, empty = EXPORT.normalize([native_task(native_annotation("other", annotator={"id": 8}))])
        self.assertEqual(rows, [{"clip_id": "original-clip-id", "task_id": 42, "annotation_id": 11,
                                 "annotator_id": 8, "lead_time": 4.5, "decision": "other",
                                 "status": "accepted", "emotion": "other"}])
        self.assertEqual((cancelled, empty), (0, 0))

    def test_historical_unknown_is_preserved_without_mutating_native_export(self):
        tasks = [native_task(native_annotation("unknown"))]
        tasks[0]["data"]["historical_display"] = "Unknown / Other"
        before = deepcopy(tasks)
        rows, _, _ = EXPORT.normalize(tasks)
        self.assertEqual(tasks, before)
        self.assertEqual(rows[0]["decision"], "unknown")
        self.assertEqual(rows[0]["emotion"], "unknown")
        self.assertEqual(rows[0]["status"], "accepted")

    def test_distinct_annotator_votes_are_not_collapsed_or_reinterpreted(self):
        task = native_task(native_annotation("other", 11, 1), native_annotation("unknown", 12, 2),
                           native_annotation("bad_clip", 13, 3), native_annotation("neutral", 14, 4))
        rows, _, _ = EXPORT.normalize([task])
        self.assertEqual([row["decision"] for row in rows], ["other", "unknown", "bad_clip", "neutral"])
        self.assertEqual([row["emotion"] for row in rows], ["other", "unknown", None, "neutral"])
        self.assertEqual([row["status"] for row in rows], ["accepted", "accepted", "bad_clip", "accepted"])

    def test_display_names_are_not_automatically_converted_into_decision_ids(self):
        for display_name in ("Other", "Unknown / Other", "Unknown / Uncertain"):
            with self.subTest(display_name=display_name), self.assertRaisesRegex(ValueError, "unknown decision value"):
                EXPORT.normalize([native_task(native_annotation(display_name))])

    def test_cancelled_and_empty_task_handling_stays_unchanged(self):
        cancelled = native_annotation("other")
        cancelled["was_cancelled"] = True
        rows, cancelled_count, empty_count = EXPORT.normalize([native_task(cancelled), native_task()])
        self.assertEqual(rows, [])
        self.assertEqual((cancelled_count, empty_count), (1, 1))


class LabelingTemplateTests(unittest.TestCase):
    def test_both_protocols_preserve_existing_ids_and_hotkeys_with_distinct_other(self):
        expected = {"angry": "1", "disgusted": "2", "fearful": "3", "happy": "4", "neutral": "5",
                    "sad": "6", "surprised": "7", "unknown": "8", "other": "9", "bad_clip": "0"}
        for filename in ("config_audio_only.xml", "config_with_transcript.xml"):
            with self.subTest(filename=filename):
                root = ET.parse(PROJECT / filename).getroot()
                groups = root.findall(".//Choices")
                self.assertEqual(len(groups), 1)
                self.assertEqual(groups[0].attrib["name"], "decision")
                self.assertEqual(groups[0].attrib["toName"], "audio")
                self.assertEqual(groups[0].attrib["choice"], "single-radio")
                choices = {node.attrib["value"]: node.attrib for node in groups[0].iter("Choice")}
                self.assertEqual({value: attrs["hotkey"] for value, attrs in choices.items()}, expected)
                self.assertEqual(set(choices), EXPORT.DECISIONS)
                self.assertIn("Unknown / Uncertain", choices["unknown"]["html"])
                self.assertIn("cannot confidently", choices["unknown"]["hint"])
                self.assertNotIn("outside", choices["unknown"]["hint"])
                self.assertIn("clear emotion", choices["other"]["hint"])
                self.assertIn("seven named", choices["other"]["hint"])
                self.assertNotEqual(choices["other"]["style"], choices["unknown"]["style"])

    def test_copied_xml_in_both_guides_matches_the_current_template(self):
        expected = (PROJECT / "config_audio_only.xml").read_text(encoding="utf-8").strip()
        for filename in ("template_configuration.md", "template_configuration_en.md"):
            with self.subTest(filename=filename):
                text = (PROJECT / filename).read_text(encoding="utf-8")
                copied = re.search(r"```xml\n(.*?)\n```", text, flags=re.S)
                self.assertIsNotNone(copied)
                self.assertEqual(copied.group(1).strip(), expected)


if __name__ == "__main__":
    unittest.main()

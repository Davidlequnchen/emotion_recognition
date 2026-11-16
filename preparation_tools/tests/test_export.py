"""Export: calls and turn order, segment ids, the four group folders, context links across groups, and robustness
(a failing turn is skipped cleanly; the manifests keep every finished call). Models are replaced by fakes."""
import contextlib
import copy
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from fakes import LONG_EN, SHORT_EN, SPANISH, FakeAligner, constant_probs, write_turn  # noqa: E402
from segmenter import audio_io, pipeline  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "segment_turns.yaml").read_text(encoding="utf-8"))
EXTENSIONS = {".wav", ".mp3", ".flac"}


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def run_batch(tmp, turns_csv=None, write_clip=None):
    """Run the pipeline on tmp/in with fake VAD and aligner; returns (rows, failures)."""
    cfg = copy.deepcopy(CFG)
    cfg["paths"].update(input_dir="in", turns_csv="turns.csv" if turns_csv else None, output_dir="out",
                        weights_dir="w")
    if turns_csv:
        with open(tmp / "turns.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["turn_id", "call_id", "turn_index", "turn_start_in_call_s"])
            writer.writerows(turns_csv)
    patches = [mock.patch.object(pipeline.vad, "load_vad", return_value=None),
               mock.patch.object(pipeline.vad, "speech_probabilities", side_effect=constant_probs(0.9)),
               mock.patch.object(pipeline, "Aligner", FakeAligner)]
    if write_clip:
        patches.append(mock.patch.object(pipeline.audio_io, "write_clip", side_effect=write_clip))
    for patch in patches:
        patch.start()
    try:
        with contextlib.redirect_stdout(io.StringIO()):         # the run prints every step; keep test output short
            return pipeline.run(cfg, tmp)
    finally:
        for patch in patches:
            patch.stop()


class Subtypes(unittest.TestCase):
    def test_pcm_s8_flac_exports_as_pcm16(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "turn.flac"
            sf.write(str(src), np.zeros((800, 1)), 8000, subtype="PCM_S8", format="FLAC")
            data, rate, subtype = audio_io.read_audio(src)
            self.assertEqual(subtype, "PCM_S8")
            self.assertEqual(audio_io.write_clip(Path(tmp) / "clip.wav", data, rate, subtype), "PCM_16")
        self.assertEqual(audio_io.wav_subtype("PCM_24"), "PCM_24")
        self.assertEqual(audio_io.wav_subtype("MPEG_LAYER_III"), "PCM_16")


class Calls(unittest.TestCase):
    def test_call_ids_and_order_from_turns_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("z_first", "a_second", "m_third", "loose"):
                write_turn(tmp, name, 1.0, "hi")
            rows = {"z_first": {"call_id": "call/7 A", "turn_index": "0"},
                    "a_second": {"call_id": "call/7 A", "turn_index": "2"},
                    "m_third": {"call_id": "call/7 A", "turn_index": "10"}}
            calls, sources = pipeline.group_into_calls(pipeline.list_turns(tmp, EXTENSIONS), rows)
        self.assertEqual(sources, {"turns.csv": 3, "folder": 0, "own call": 1})
        call = calls["call/7 A"]
        self.assertEqual((call["folder"], call["order_by"]), ("call_7_A", "turn_index"))
        self.assertEqual([t["turn_id"] for t in call["turns"]], ["z_first", "a_second", "m_third"])  # 0, 2, 10
        self.assertEqual([t["turn_order"] for t in call["turns"]], [0, 1, 2])
        self.assertEqual(calls["loose"]["folder"], "loose")                  # not in turns.csv: its own call

    def test_order_falls_back_to_call_time_then_file_name(self):
        def order(metas):
            turns = [{"turn_id": n, "source": f"{n}.wav", "meta": m} for n, m in metas.items()]
            return pipeline.order_turns(turns), [t["turn_id"] for t in turns]
        self.assertEqual(order({"a": {"turn_start_in_call_s": "50"}, "b": {"turn_start_in_call_s": "7.5"}}),
                         ("turn_start_in_call_s", ["b", "a"]))
        self.assertEqual(order({"b": {"turn_index": "1"}, "a": {}}), ("file name", ["a", "b"]))

    def test_call_ids_from_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_turn(tmp / "callB", "y_turn", 1.0, "hi")
            write_turn(tmp / "callB", "x_turn", 1.0, "hi")
            write_turn(tmp / "callA", "z_turn", 1.0, "hi")
            calls, sources = pipeline.group_into_calls(pipeline.list_turns(tmp, EXTENSIONS), {})
            self.assertEqual(sources, {"turns.csv": 0, "folder": 3, "own call": 0})
            self.assertEqual(list(calls), ["callA", "callB"])
            self.assertEqual([t["turn_id"] for t in calls["callB"]["turns"]], ["x_turn", "y_turn"])
            self.assertEqual(calls["callB"]["turns"][0]["source"], "callB/x_turn.wav")
            write_turn(tmp / "callA", "x_turn", 1.0, "hi")                   # the same turn id in two calls
            with self.assertRaises(ValueError):
                pipeline.list_turns(tmp, EXTENSIONS)

    def test_call_folder_names(self):
        self.assertEqual(pipeline.safe_name(" ../call 1:a "), "call_1_a")
        self.assertEqual(pipeline.safe_name("???"), "call")
        turns = [{"turn_id": t, "folder": "", "source": f"{t}.wav"} for t in ("t1", "t2")]
        with self.assertRaises(ValueError):                                  # "a b" and "a_b" share a folder
            pipeline.group_into_calls(turns, {"t1": {"call_id": "a b"}, "t2": {"call_id": "a_b"}})

    def test_segment_ids_sort_in_time_order(self):
        in_time_order = [pipeline.segment_id("c1", t, s) for t in (0, 2, 9, 10, 11) for s in (0, 1, 9, 10)]
        self.assertEqual(sorted(in_time_order), in_time_order)
        self.assertEqual(pipeline.segment_id("c1", 3, 1), "c1_t003_s01")


class ContextLinks(unittest.TestCase):
    def test_links_cross_groups_in_call_order(self):
        """One call: a long English turn (cut), a Spanish turn in the middle, a short English turn."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_turn(tmp / "in", "t_long", 35.0, LONG_EN)
            write_turn(tmp / "in", "t_es", 4.0, SPANISH)
            write_turn(tmp / "in", "t_short", 5.0, SHORT_EN)
            rows, failures = run_batch(tmp, turns_csv=[["t_short", "call 1", 4, "50.0"],
                                                       ["t_long", "call 1", 0, "0.0"],
                                                       ["t_es", "call 1", 2, "40.0"]])
            out = tmp / "out"
            self.assertEqual(failures, [])
            self.assertEqual(set(rows[0]), set(pipeline.MANIFEST_FIELDS))      # every column filled in
            timeline = read_csv(out / "call_timeline.csv")
            final = read_csv(out / "final" / "manifest.csv")
            non_english = read_csv(out / "non_english" / "manifest.csv")
            for group in ("alignment_issues", "boundary_issues"):
                self.assertEqual(read_csv(out / group / "manifest.csv"), [])    # header only
            for row in timeline:                                             # files in <group>/<call>/
                self.assertTrue(row["audio_path"].startswith(f"{row['group']}/call_1/"), row["audio_path"])
                self.assertTrue((out / row["audio_path"]).is_file() and (out / row["transcript_path"]).is_file())
            self.assertTrue((out / "review" / "t_long.json").is_file())

        ids = [r["segment_id"] for r in timeline]
        self.assertEqual(ids, [f"call_1_t000_s0{k}" for k in range(5)] + ["call_1_t001_s00", "call_1_t002_s00"])
        self.assertEqual(sorted(ids), ids)
        self.assertEqual([r["group"] for r in timeline], ["final"] * 5 + ["non_english", "final"])
        self.assertEqual([r["order_in_call"] for r in timeline], [str(k) for k in range(7)])
        self.assertEqual([r["segment_id"] for r in final], ids[:5] + ids[6:])
        self.assertEqual([r["reason"] for r in non_english], ["language: es"])
        # prev/next cross the groups: last long piece -> Spanish turn -> short English turn
        last_long, spanish, short = timeline[4], timeline[5], timeline[6]
        self.assertEqual((last_long["next_segment_id"], last_long["next_group"]), (spanish["segment_id"], "non_english"))
        self.assertEqual((spanish["prev_segment_id"], spanish["prev_group"]), (last_long["segment_id"], "final"))
        self.assertEqual((spanish["next_segment_id"], spanish["next_group"]), (short["segment_id"], "final"))
        self.assertEqual((short["prev_segment_id"], short["prev_group"]), (spanish["segment_id"], "non_english"))
        self.assertEqual((timeline[0]["prev_segment_id"], short["next_segment_id"]), ("", ""))
        # call-time gaps: 0 inside a turn, the removed agent time between turns
        self.assertEqual([r["gap_to_prev_s"] for r in timeline], ["", "0.000", "0.000", "0.000", "0.000", "5.000", "6.000"])
        self.assertEqual(short["gap_to_next_s"], "")
        manifest_row = next(r for r in final if r["segment_id"] == short["segment_id"])
        self.assertEqual((manifest_row["prev_segment_id"], manifest_row["turn_index"], manifest_row["turn_order"]),
                         (spanish["segment_id"], "4", "2"))


class Failures(unittest.TestCase):
    @staticmethod
    def failing_writer():
        real_write = audio_io.write_clip

        def write_clip(path, *args):
            if "t001" in path.name or "b_bad" in path.name:
                raise ValueError("simulated export error")
            return real_write(path, *args)
        return write_clip

    def test_failed_export_skips_turn_and_keeps_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("a_turn", "b_bad", "c_turn"):                       # no turns.csv: each its own call
                write_turn(tmp / "in", name, 1.0, "hello there")
            rows, failures = run_batch(tmp, write_clip=self.failing_writer())
            self.assertEqual([r["turn_id"] for r in rows], ["a_turn", "c_turn"])
            self.assertEqual([f[0] for f in failures], ["b_bad"])
            self.assertEqual([r["turn_id"] for r in read_csv(tmp / "out" / "final" / "manifest.csv")],
                             ["a_turn", "c_turn"])
            leftovers = [p.name for p in (tmp / "out").rglob("*b_bad*")]  # no clip, text, review or call folder
            self.assertEqual(leftovers, [])
            self.assertEqual(read_csv(tmp / "out" / "failures.csv")[0]["turn_id"], "b_bad")

    def test_rest_of_the_call_is_linked_around_a_failed_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("a_turn", "b_bad", "c_turn"):                       # one call folder, file-name order
                write_turn(tmp / "in" / "call9", name, 1.0, "hello there")
            rows, failures = run_batch(tmp, write_clip=self.failing_writer())
            self.assertEqual([f[:2] for f in failures], [("b_bad", "call9")])
            timeline = read_csv(tmp / "out" / "call_timeline.csv")
            self.assertEqual([r["segment_id"] for r in timeline], ["call9_t000_s00", "call9_t002_s00"])
            self.assertEqual((timeline[0]["next_segment_id"], timeline[1]["prev_segment_id"]),
                             ("call9_t002_s00", "call9_t000_s00"))
            self.assertEqual([p.name for p in (tmp / "out" / "final" / "call9").iterdir() if "t001" in p.name], [])


if __name__ == "__main__":
    unittest.main()

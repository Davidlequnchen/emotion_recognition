"""Select stratified labelling rounds from synthetic curated batches, without Label Studio or real data."""

from collections import Counter
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, PROJECT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREP = load("ser_prepare_tasks", "prepare_tasks.py")
ATTACH = load("ser_attach_for_prepare_tests", "attach_task_data.py")

EMOTIONS = PREP.EMOTIONS
TIER_VOTES = {"agree3": lambda e: (e, e, e), "agree2": lambda e: (e, e, "neutral"),
              "single": lambda e: (e, "neutral", "neutral")}
CONFLICT = ("angry", "Sad ", "neutral")
NEUTRAL = ("neutral", "NEUTRAL", None)
UNASSIGNED = ("happy", None, "invalid")
EXTRA = list(PREP.EXTRA_COLUMNS)


def quotas(random=0, emotion=0, conflict=0, neutral=0):
    return {"random": random, "emotion": emotion, "conflict": conflict, "neutral": neutral}


class PrepareTasks(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name).resolve()
        self.curated = self.base / "curated"
        self.output = self.base / "selected"
        self.count = 0

    # ---- synthetic data ----

    def clip(self, votes, call=None, long=False, text="Hello.", audio=None):
        self.count += 1
        clip_id = f"clip-{self.count:05d}"
        return {"clip_id": clip_id, "audio_clip_file": audio or f"{clip_id}.wav", "text": text,
                "is_longer_than_10_seconds": long, "external_call_id": call or f"call-{clip_id}",
                "pred_e2v": votes[0], "pred_gpt": votes[1], "pred_gemini": votes[2], "speaker": "customer"}

    def clips(self, votes, count, **changes):
        return [self.clip(votes, **changes) for _ in range(count)]

    def batch(self, rows, name="batch_01"):
        """Write <curated>/<name>/manifest_with_predictions.parquet and fake audio for every bare file name."""
        folder = self.curated / name
        (folder / "audio").mkdir(parents=True, exist_ok=True)
        for row in rows:
            if "/" not in row["audio_clip_file"]:
                (folder / "audio" / row["audio_clip_file"]).write_bytes(f"RIFF fake {row['clip_id']}".encode())
        pq.write_table(pa.Table.from_pylist(rows), folder / "manifest_with_predictions.parquet")
        return folder

    def settings(self, batches=("batch_01",), **changes):
        return {"curated_root": str(self.curated), "batches": list(batches), "output_root": str(self.output),
                **changes}

    def run_prep(self, settings):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            PREP.run(settings)
        return printed.getvalue()

    def fails(self, settings, message):
        before = sorted(os.listdir(self.output)) if self.output.exists() else None
        with self.assertRaisesRegex(ValueError, message), contextlib.redirect_stdout(io.StringIO()):
            PREP.run(settings)
        # Nothing is created: no new round, no .partial folder, not even output_root itself.
        self.assertEqual(sorted(os.listdir(self.output)) if self.output.exists() else None, before)

    def round(self, number=1, output=None):
        folder = (output or self.output) / f"tasks_{number:03d}"
        tasks = json.loads((folder / f"tasks_{number:03d}.json").read_text(encoding="utf-8"))
        manifest = pq.read_table(folder / "manifest.parquet").to_pylist()
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        return tasks, manifest, summary

    # ---- labels and strata ----

    def test_votes_and_strata(self):
        for value in (None, float("nan"), "", "  ", "INVALID", " Other", "unknown", "None"):
            self.assertIsNone(PREP.vote(value), value)
        self.assertEqual(PREP.vote(" Angry "), "angry")
        self.assertEqual(PREP.vote("NEUTRAL"), "neutral")
        for value in ("anger", "calm", 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PREP.vote(value)
        cases = {
            ("angry", "angry", "angry"): "agree3:angry",
            ("angry", "angry", "neutral"): "agree2:angry",
            ("sad", "sad", "happy"): "agree2:sad",
            ("sad", "sad", None): "agree2:sad",          # only 2 valid votes
            ("happy", "neutral", "neutral"): "single:happy",
            ("happy", None, "neutral"): "single:happy",
            ("angry", "sad", "neutral"): "conflict",
            ("angry", "sad", "happy"): "conflict",
            (None, "angry", "sad"): "conflict",
            ("neutral", "neutral", "neutral"): "neutral",
            ("neutral", None, "neutral"): "neutral",
            ("angry", None, None): "unassigned",
            (None, None, None): "unassigned",
        }
        for votes, expected in cases.items():
            with self.subTest(votes=votes):
                self.assertEqual(PREP.stratum(list(votes)), expected)

    def test_unexpected_label_raises_and_lists_the_values(self):
        rows = self.clips(NEUTRAL, 3)
        rows[0]["pred_gpt"], rows[1]["pred_gpt"], rows[2]["pred_gemini"] = "Anger", "joy ", "calm"
        self.batch(rows)
        with self.assertRaises(ValueError) as caught, contextlib.redirect_stdout(io.StringIO()):
            PREP.run(self.settings(quotas=quotas(random=1)))
        for text in ("pred_gpt: 'anger', 'joy'", "pred_gemini: 'calm'", "merge bug"):
            self.assertIn(text, str(caught.exception))
        self.assertFalse(self.output.exists())

    # ---- quotas and the shortfall cascade ----

    def test_exact_quotas_when_pools_are_large(self):
        rows = [row for e in EMOTIONS for votes in TIER_VOTES.values() for row in self.clips(votes(e), 60)]
        rows += self.clips(CONFLICT, 300) + self.clips(NEUTRAL, 400) + self.clips(UNASSIGNED, 100)
        self.batch(rows)
        self.run_prep(self.settings())  # default quotas: 150 + 560 + 140 + 150
        tasks, manifest, summary = self.round()
        self.assertEqual(len(tasks), 1000)
        self.assertEqual(summary["shortfalls"], [])
        self.assertEqual(summary["drawn"]["emotion"], summary["targets"]["emotion"])
        self.assertEqual(Counter(row["layer"] for row in manifest),
                         {"random": 150, "emotion": 560, "conflict": 140, "neutral": 150})
        emotion = [row for row in manifest if row["layer"] == "emotion"]
        self.assertEqual(Counter(row["target_class"] for row in emotion),
                         {"angry": 94, "disgusted": 94, "fearful": 93, "happy": 93, "sad": 93, "surprised": 93})
        self.assertEqual(Counter(row["stratum"] for row in emotion if row["target_class"] == "angry"),
                         {"agree3:angry": 32, "agree2:angry": 31, "single:angry": 31})
        self.assertTrue(all(row["stratum"] == "conflict" for row in manifest if row["layer"] == "conflict"))
        self.assertEqual(summary["parts"], {f"part_{n:02d}": 100 for n in range(1, 11)})
        # The final shuffle mixes the layers through the task order.
        self.assertGreater(len({row["layer"] for row in manifest[:100]}), 2)

    def test_tier_shortfall_comes_from_the_same_emotion_in_refill_order(self):
        rows = self.clips(TIER_VOTES["agree2"]("angry"), 1) + self.clips(TIER_VOTES["single"]("angry"), 3)
        rows += self.clips(TIER_VOTES["agree2"]("disgusted"), 3) + self.clips(TIER_VOTES["single"]("disgusted"), 3)
        for e in EMOTIONS[2:]:
            rows += self.clips(TIER_VOTES["agree3"](e), 1) + self.clips(TIER_VOTES["agree2"](e), 1)
        self.batch(rows)
        self.run_prep(self.settings(quotas=quotas(emotion=12)))  # 2 per emotion: agree3 1, agree2 1, single 0
        _, manifest, summary = self.round()
        drawn = summary["drawn"]["emotion"]
        self.assertEqual(drawn["angry"], {"agree3": 0, "agree2": 1, "single": 1})      # agree2 empty -> single
        self.assertEqual(drawn["disgusted"], {"agree3": 0, "agree2": 2, "single": 0})  # agree2 comes first
        self.assertIn({"from": "emotion agree3:angry", "missing": 1, "to": "other tiers of angry"},
                      summary["shortfalls"])
        self.assertNotIn("conflict", [move["to"] for move in summary["shortfalls"]])
        self.assertEqual(Counter(row["layer"] for row in manifest), {"emotion": 12})

    def test_class_shortfall_moves_to_the_conflict_target(self):
        rows = [row for e in EMOTIONS if e != "fearful" for row in self.clips(TIER_VOTES["agree3"](e), 1)]
        rows += self.clips(CONFLICT, 4) + self.clips(NEUTRAL, 3)
        self.batch(rows)
        self.run_prep(self.settings(quotas=quotas(emotion=6, conflict=1)))
        _, manifest, summary = self.round()
        self.assertIn({"from": "emotion fearful", "missing": 1, "to": "conflict"}, summary["shortfalls"])
        self.assertEqual(summary["targets"]["conflict_with_moves"], 2)
        self.assertEqual(Counter(row["layer"] for row in manifest), {"emotion": 5, "conflict": 2})

    def test_remaining_shortfall_is_filled_from_the_rest_of_the_pool(self):
        rows = self.clips(CONFLICT, 1) + self.clips(UNASSIGNED, 6) + self.clips(TIER_VOTES["agree3"]("sad"), 2)
        self.batch(rows)
        self.run_prep(self.settings(quotas=quotas(conflict=3, neutral=2)))
        _, manifest, summary = self.round()
        self.assertEqual(Counter(row["layer"] for row in manifest), {"conflict": 1, "fill": 4})
        self.assertIn({"from": "conflict", "missing": 2, "to": "fill"}, summary["shortfalls"])
        self.assertIn({"from": "neutral", "missing": 2, "to": "fill"}, summary["shortfalls"])
        self.assertTrue(all(row["target_class"] is None for row in manifest if row["layer"] == "fill"))

    def test_pool_smaller_than_one_round_raises(self):
        self.batch(self.clips(NEUTRAL, 4))
        self.fails(self.settings(quotas=quotas(random=5)), "fewer than one round of 5")

    def test_per_call_cap(self):
        rows = self.clips(NEUTRAL, 5, call="A") + self.clips(NEUTRAL, 5, call="B") + self.clips(NEUTRAL, 1, call="C")
        rows += self.clips(UNASSIGNED, 3, call="D")
        self.batch(rows)
        self.run_prep(self.settings(quotas=quotas(neutral=6), max_clips_per_call=2))
        _, manifest, summary = self.round()
        calls = Counter(row["external_call_id"] for row in manifest)
        self.assertEqual(calls, {"A": 2, "B": 2, "C": 1, "D": 1})
        self.assertEqual((summary["drawn"]["neutral"], summary["drawn"]["fill"]), (5, 1))
        shutil.rmtree(self.output)
        # 4 calls × at most 2 clips cannot make a round of 8, although the pool has 14 clips.
        self.fails(self.settings(quotas=quotas(neutral=8), max_clips_per_call=2), "max_clips_per_call=2")

    # ---- eligibility ----

    def test_long_clips_are_excluded(self):
        short, names = [], []
        for number, (no, yes) in enumerate([(False, True), (0, 1), (" false ", "TRUE")], 1):
            names.append(f"batch_{number:02d}")
            rows = [self.clip(NEUTRAL, long=no), self.clip(TIER_VOTES["agree3"]("angry"), long=yes),
                    self.clip(TIER_VOTES["agree3"]("sad"), long=yes)]
            folder = self.batch(rows, name=names[-1])
            for row in rows[1:]:  # long clips need no audio
                (folder / "audio" / row["audio_clip_file"]).unlink()
            short.append(rows[0]["clip_id"])
        self.run_prep(self.settings(names, quotas=quotas(random=3)))
        _, manifest, summary = self.round()
        self.assertEqual({row["clip_id"] for row in manifest}, set(short))
        self.assertEqual(summary["batches"]["batch_03"],
                         {"rows": 3, "long_excluded": 2, "earlier_rounds": 0, "eligible": 1})
        # The flag column has a different type in each batch, so the round's manifest keeps it as text.
        self.assertEqual(summary["columns_kept_as_text"], ["is_longer_than_10_seconds"])
        self.assertEqual({row["is_longer_than_10_seconds"] for row in manifest}, {"false", "0", " false "})
        shutil.rmtree(self.output)
        self.batch([self.clip(NEUTRAL, long="maybe")], name="batch_04")
        self.fails(self.settings(names + ["batch_04"], quotas=quotas(random=3)), "invalid long-clip flags")

    def test_two_batches_and_audio_path_forms(self):
        self.batch(self.clips(NEUTRAL, 2))
        second = self.curated / "batch_02"
        rows = [self.clip(NEUTRAL, audio="audio/relative.wav"), self.clip(NEUTRAL, audio="extra/nested.wav"),
                self.clip(NEUTRAL, audio=str(second / "audio" / "absolute.wav"))]
        self.batch(rows, name="batch_02")
        (second / "extra").mkdir()
        for path in ("audio/relative.wav", "extra/nested.wav", "audio/absolute.wav"):
            (second / path).write_bytes(path.encode())
        self.run_prep(self.settings(["batch_01", "batch_02"], quotas=quotas(random=5)))
        tasks, manifest, _ = self.round()
        self.assertEqual(Counter(row["source_batch"] for row in manifest), {"batch_01": 2, "batch_02": 3})
        names = {task["data"]["audio"].rsplit("/", 1)[1] for task in tasks}
        self.assertTrue({"relative.wav", "nested.wav", "absolute.wav"} <= names)
        self.assertEqual((self.output / "tasks_001/audio/part_01/nested.wav").read_bytes(), b"extra/nested.wav")

    def test_duplicate_clip_id_across_batches_raises(self):
        first = self.clip(NEUTRAL)
        self.batch([first])
        self.batch([{**self.clip(NEUTRAL), "clip_id": first["clip_id"]}], name="batch_02")
        self.fails(self.settings(["batch_01", "batch_02"], quotas=quotas(random=1)),
                   f"clip_id must be unique(?s:.*){first['clip_id']} \\(batch_01, batch_02\\)")

    def test_audio_names_must_be_unique_and_short(self):
        self.batch([self.clip(NEUTRAL, audio="Clip One.wav"), self.clip(NEUTRAL, audio="clip_one.wav")])
        self.fails(self.settings(quotas=quotas(random=1)), "not unique(?s:.*)Clip One.wav = batch_01/clip_one.wav")
        shutil.rmtree(self.curated)
        self.batch([self.clip(NEUTRAL, audio="x" * 67 + ".wav")])
        self.fails(self.settings(quotas=quotas(random=1)), "longer than 70 characters")

    def test_audio_cannot_escape_curated_root(self):
        outside = self.base / "outside.wav"
        outside.write_bytes(b"outside")
        folder = self.batch([self.clip(NEUTRAL, audio="../../outside.wav")])
        self.fails(self.settings(quotas=quotas(random=1)), "outside curated_root")
        shutil.rmtree(self.curated)
        folder = self.batch([self.clip(NEUTRAL, audio="link.wav")])
        (folder / "audio" / "link.wav").unlink()
        (folder / "audio" / "link.wav").symlink_to(outside)
        self.fails(self.settings(quotas=quotas(random=1)), "resolve outside curated_root")

    def test_empty_call_id_raises(self):
        rows = self.clips(NEUTRAL, 2)
        rows[1]["external_call_id"] = "  "
        self.batch(rows)
        self.fails(self.settings(quotas=quotas(random=1)), "1 eligible clips have an empty call ID")

    # ---- rounds ----

    def test_second_run_makes_a_disjoint_second_round(self):
        self.batch(self.clips(NEUTRAL, 10))
        settings = self.settings(quotas=quotas(neutral=4))
        self.run_prep(settings)
        self.run_prep(settings)
        first, _, _ = self.round(1)
        second, _, summary = self.round(2)
        ids = [{task["data"]["clip_id"] for task in tasks} for tasks in (first, second)]
        self.assertEqual([len(group) for group in ids], [4, 4])
        self.assertFalse(ids[0] & ids[1])
        self.assertEqual((summary["round"], summary["random_seed"], summary["earlier_rounds"]),
                         (2, "42:2", ["tasks_001"]))
        self.assertEqual(summary["batches"]["batch_01"]["earlier_rounds"], 4)
        self.assertTrue(second[0]["data"]["audio"].startswith("/data/local-files/?d=tasks_002/audio/part_01/"))
        # Only 2 clips are left for a third round of 4.
        self.fails(settings, "fewer than one round")

    def test_same_inputs_give_the_same_round(self):
        rows = [row for e in EMOTIONS for row in self.clips(TIER_VOTES["agree2"](e), 4)]
        rows += self.clips(CONFLICT, 10) + self.clips(NEUTRAL, 10) + self.clips(UNASSIGNED, 10)
        self.batch(rows)
        settings = self.settings(quotas=quotas(random=5, emotion=12, conflict=4, neutral=4))
        self.run_prep(settings)
        # Same clips in reversed row order, into a fresh output_root.
        pq.write_table(pa.Table.from_pylist(rows[::-1]), self.curated / "batch_01/manifest_with_predictions.parquet")
        again = self.base / "again"
        self.run_prep({**settings, "output_root": str(again)})
        name = "tasks_001/tasks_001.json"
        self.assertEqual((again / name).read_bytes(), (self.output / name).read_bytes())
        first, second = self.round()[1], self.round(output=again)[1]
        self.assertEqual([(r["clip_id"], r["layer"]) for r in first], [(r["clip_id"], r["layer"]) for r in second])

    # ---- output ----

    def test_tasks_file_parts_copies_and_manifest(self):
        rows = self.clips(NEUTRAL, 5) + [self.clip(NEUTRAL, text=None), self.clip(NEUTRAL, text="  ")]
        folder = self.batch(rows)
        originals = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in (folder / "audio").iterdir()}
        printed = self.run_prep(self.settings(quotas=quotas(random=7), files_per_folder=3))
        tasks, manifest, summary = self.round()
        # Annotators see only audio, transcript and clip_id: no predictions, strata or votes.
        self.assertTrue(all(list(task) == ["data"] and list(task["data"]) == ["audio", "transcript", "clip_id"]
                            for task in tasks))
        self.assertEqual(sum(task["data"]["transcript"] == "" for task in tasks), 2)
        self.assertEqual(summary["transcript_blanks"], 2)
        self.assertEqual(summary["parts"], {"part_01": 3, "part_02": 3, "part_03": 1})
        audio = self.output / "tasks_001" / "audio"
        self.assertEqual(sorted(len(os.listdir(audio / part)) for part in os.listdir(audio)), [1, 3, 3])
        # Manifest: every original column plus the added ones, one row per task, in task order.
        self.assertEqual(list(manifest[0]), list(rows[0]) + EXTRA)
        self.assertEqual([row["clip_id"] for row in manifest], [task["data"]["clip_id"] for task in tasks])
        self.assertEqual([row["task_index"] for row in manifest], list(range(1, 8)))
        for task, row in zip(tasks, manifest):
            copy = self.output / row["copied_audio"]
            self.assertEqual(task["data"]["audio"], "/data/local-files/?d=" + row["copied_audio"])
            self.assertEqual(copy.parent.name, row["part"])
            self.assertEqual(copy.read_bytes(), originals[copy.name][0])
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in (folder / "audio").iterdir()}, originals)
        self.assertIn("attach_task_data.py", printed)

    def test_missing_selected_audio_creates_nothing(self):
        folder = self.batch(self.clips(NEUTRAL, 3))
        (folder / "audio" / "clip-00002.wav").unlink()
        self.fails(self.settings(quotas=quotas(random=3)), "1 selected audio files are missing(?s:.*)clip-00002.wav")
        self.assertFalse(self.output.exists())
        # With an earlier round present, a failing run leaves only that round.
        self.batch(self.clips(NEUTRAL, 3), name="batch_02")
        self.run_prep(self.settings(["batch_02"], quotas=quotas(random=3)))
        self.fails(self.settings(["batch_01", "batch_02"], quotas=quotas(random=3)), "selected audio files are missing")
        self.assertEqual(os.listdir(self.output), ["tasks_001"])

    def test_copy_error_removes_the_partial_folder(self):
        self.batch(self.clips(NEUTRAL, 3))
        copies = []

        def flaky_copy(source, target):
            copies.append(target)
            if len(copies) == 2:
                raise OSError("disk full")
            return shutil.copyfile(source, target)

        with mock.patch.object(PREP.shutil, "copy2", flaky_copy), self.assertRaisesRegex(OSError, "disk full"), \
                contextlib.redirect_stdout(io.StringIO()):
            PREP.run(self.settings(quotas=quotas(random=3)))
        self.assertEqual(os.listdir(self.output), [])

    def test_leftover_partial_folder_stops_the_run(self):
        self.batch(self.clips(NEUTRAL, 2))
        (self.output / "tasks_001.partial").mkdir(parents=True)
        self.fails(self.settings(quotas=quotas(random=1)), "interrupted")

    def test_output_root_and_curated_root_must_be_apart(self):
        self.batch(self.clips(NEUTRAL, 2))
        for output in (self.curated / "selected", self.base):
            with self.subTest(output=output):
                self.fails(self.settings(quotas=quotas(random=1), output_root=str(output)), "neither inside")
        self.assertFalse((self.curated / "selected").exists())

    def test_attach_task_data_matches_the_tasks_file(self):
        self.batch([self.clip(NEUTRAL, audio="my clip #1.wav"), self.clip(NEUTRAL)])
        self.run_prep(self.settings(quotas=quotas(random=2)))
        tasks, manifest, _ = self.round()
        clips, lookup = ATTACH.local_clips(self.output / "tasks_001" / "tasks_001.json")
        self.assertEqual(set(clips), {row["clip_id"] for row in manifest})
        uploaded = [{"id": index, "data": {"$undefined$": f"/data/upload/7/1a2b3c4d-{ATTACH.django_name(name)}"}}
                    for index, name in enumerate((Path(row["copied_audio"]).name for row in manifest), 1)]
        found = ATTACH.classify(uploaded, clips, lookup)
        self.assertEqual([clip_id for _, clip_id, _ in found["update"]], [row["clip_id"] for row in manifest])
        self.assertEqual((found["unmatched"], found["duplicates"], found["missing"]), ([], [], []))

    # ---- settings ----

    def test_settings_are_validated(self):
        shipped = PREP.load_settings(PREP.CONFIG)
        self.assertEqual(shipped["quotas"], PREP.DEFAULTS["quotas"])
        self.assertEqual(sum(shipped["quotas"].values()), 1000)
        path = self.base / "settings.yaml"
        base = "curated_root: /a\nbatches: [b1]\noutput_root: /o\n"
        for extra, message in [("", None), ("seeds: 1\n", "unknown keys"), ("quotas: {random: -1}\n", "quotas"),
                               ("quotas: {extra: 1}\n", "quotas"), ("columns: {predictions: [a, b]}\n", "3 different"),
                               ("max_clips_per_call: 0\n", "max_clips_per_call"), ("seed: true\n", "seed")]:
            with self.subTest(extra=extra):
                path.write_text(base + extra, encoding="utf-8")
                if message is None:
                    self.assertEqual(PREP.load_settings(path)["max_clips_per_call"], 3)
                else:
                    with self.assertRaisesRegex(ValueError, message):
                        PREP.load_settings(path)
        for text, message in [("batches: []\ncurated_root: /a\noutput_root: /o\n", "batches"),
                              ("batches: [../x]\ncurated_root: /a\noutput_root: /o\n", "batches"),
                              ("curated_root: /a\n", "missing keys")]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                path.write_text(text, encoding="utf-8")
                PREP.load_settings(path)


if __name__ == "__main__":
    unittest.main()

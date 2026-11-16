"""Exercise real Parquet-to-JSON imports without a running Label Studio service."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_tasks.py"
FLAG = "is_longer_than_10_seconds"


class PrepareTasks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / "dataset"
        self.batch = self.root / "batch"
        self.audio = self.batch / "audio"
        self.audio.mkdir(parents=True)
        self.manifest = self.batch / "manifest.parquet"
        self.output_dir = self.batch / "tasks"
        self.output = self.output_dir / "tasks_001.json"

    def audio_file(self, name):
        path = self.audio / name
        # Duration must come from the manifest flag, so no decoder should read these bytes.
        path.write_bytes(b"audio contents are not decoded")
        return path

    def row(self, **changes):
        return {"clip_id": "stable-id-001", "audio_clip_file": "clip.wav", "text": "Hello.",
                FLAG: False, **changes}

    def write_manifest(self, rows):
        pq.write_table(pa.Table.from_pylist(rows), self.manifest)

    def run_script(self, *extra):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--manifest", str(self.manifest),
             "--document-root", str(self.root), "--output-dir", str(self.output_dir), *extra],
            capture_output=True, text=True, check=False,
        )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [task for path in sorted(self.output_dir.glob("tasks_*.json"))
                for task in json.loads(path.read_text(encoding="utf-8"))]

    def assert_failure(self, result, message):
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.output.exists())

    def test_filters_manifest_rows_and_preserves_ids_text_and_audio(self):
        self.audio_file("clip #一.wav")
        self.audio_file("second.MP3")
        excluded = self.audio_file("long.wav")
        unlisted = self.audio_file("not-in-manifest.wav")
        self.write_manifest([
            self.row(audio_clip_file="clip #一.wav", text="  已有文字稿。  "),
            self.row(clip_id="long-id", audio_clip_file="long.wav", **{FLAG: True}),
            self.row(clip_id="second-id", audio_clip_file="second.MP3", text=None),
            self.row(clip_id="missing-long", audio_clip_file="missing.wav", **{FLAG: True}),
        ])

        result = self.run_script()
        self.assertCountEqual(self.assert_success(result), [
            {"data": {"audio": "/data/local-files/?d=batch/audio/clip%20%23%E4%B8%80.wav",
                      "transcript": "已有文字稿。", "clip_id": "stable-id-001"}},
            {"data": {"audio": "/data/local-files/?d=batch/audio/second.MP3",
                      "transcript": "", "clip_id": "second-id"}},
        ])
        self.assertIn("Manifest rows: 4; long clips excluded via " + FLAG + ": 2", result.stdout)
        self.assertIn("Missing transcripts: 1", result.stderr)
        self.assertEqual(excluded.read_bytes(), b"audio contents are not decoded")
        self.assertTrue(unlisted.is_file())

    def test_batch_relative_absolute_paths_and_custom_audio_directory(self):
        self.audio_file("relative.flac")
        absolute = self.audio_file("absolute.ogg")
        custom = self.batch / "clips"
        custom.mkdir()
        (custom / "bare.m4a").write_bytes(b"audio")
        self.write_manifest([
            self.row(audio_clip_file="audio/relative.flac"),
            self.row(clip_id="absolute", audio_clip_file=str(absolute)),
            self.row(clip_id="bare", audio_clip_file="bare.m4a"),
        ])
        tasks = self.assert_success(self.run_script("--audio-dir", str(custom)))
        self.assertCountEqual([task["data"]["audio"] for task in tasks], [
            "/data/local-files/?d=batch/audio/relative.flac",
            "/data/local-files/?d=batch/audio/absolute.ogg",
            "/data/local-files/?d=batch/clips/bare.m4a",
        ])

    def test_column_overrides_and_string_flags(self):
        self.audio_file("clip.wav")
        self.write_manifest([
            {"id": "0007", "file": "clip.wav", "transcript": "Hi", "over_15s": " FALSE "},
            {"id": "0008", "file": "missing.wav", "transcript": "Bye", "over_15s": "TrUe"},
        ])
        tasks = self.assert_success(self.run_script(
            "--clip-id-column", "id", "--audio-column", "file", "--text-column", "transcript",
            "--long-clip-column", "over_15s",
        ))
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["data"]["clip_id"], "0007")

    def test_numeric_and_string_zero_one_flags_and_integer_ids(self):
        self.audio_file("clip.wav")
        for false, true in [(0, 1), (0.0, 1.0), ("0", "1")]:
            with self.subTest(false=false, true=true):
                self.write_manifest([
                    self.row(clip_id=7, **{FLAG: false}),
                    self.row(clip_id=8, audio_clip_file="missing.wav", **{FLAG: true}),
                ])
                tasks = self.assert_success(self.run_script())
                self.assertEqual([task["data"]["clip_id"] for task in tasks], ["7"])
                self.output.unlink()

    def test_null_blank_and_nan_transcripts_become_empty_strings(self):
        self.audio_file("clip.wav")
        for text in [None, "  ", float("nan")]:
            with self.subTest(text=text):
                self.write_manifest([self.row(text=text)])
                result = self.run_script()
                self.assertEqual(self.assert_success(result)[0]["data"]["transcript"], "")
                self.assertIn("Missing transcripts: 1", result.stderr)
                self.output.unlink()

    def test_all_excluded_writes_no_batches_without_requiring_audio(self):
        self.write_manifest([self.row(**{FLAG: True})])
        self.audio.rmdir()
        self.assertEqual(self.assert_success(self.run_script()), [])
        self.assertFalse(self.output_dir.exists())

    def test_missing_or_invalid_flags_fail_without_writing(self):
        for flag in [None, "", "unknown", 2, float("nan")]:
            with self.subTest(flag=flag):
                self.write_manifest([self.row(**{FLAG: flag})])
                self.assert_failure(self.run_script(), "missing or invalid long-clip flags")

    def test_missing_columns_are_reported(self):
        self.write_manifest([{"clip_id": "one"}])
        result = self.run_script()
        self.assert_failure(result, "Missing manifest columns:")
        for column in ("audio_clip_file", "text", FLAG):
            self.assertIn(column, result.stderr)

    def test_duplicate_ids_and_duplicate_audio_fail(self):
        self.audio_file("clip.wav")
        self.audio_file("second.wav")
        for second, message in [
            (self.row(audio_clip_file="second.wav"), "duplicate clip ID"),
            (self.row(clip_id="different-id"), "duplicate audio file"),
        ]:
            with self.subTest(message=message):
                self.write_manifest([self.row(), second])
                self.assert_failure(self.run_script(), message)

    def test_missing_audio_and_invalid_transcript_fail(self):
        self.write_manifest([self.row()])
        self.assert_failure(self.run_script(), "clip.wav")
        self.audio_file("clip.wav")
        self.write_manifest([self.row(text=123)])
        self.assert_failure(self.run_script(), "transcript must be text or null")

    def test_relative_path_and_symlink_cannot_escape_document_root(self):
        outside = self.root.parent / "outside.wav"
        outside.write_bytes(b"outside")
        (self.audio / "link.wav").symlink_to(outside)
        for path in ["../../outside.wav", "link.wav"]:
            with self.subTest(path=path):
                self.write_manifest([self.row(audio_clip_file=path)])
                self.assert_failure(self.run_script(), "Path escapes document root")

    def test_existing_output_is_not_overwritten(self):
        self.write_manifest([self.row(**{FLAG: True})])
        self.output_dir.mkdir()
        self.output.write_text("original output", encoding="utf-8")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Batch output already exists", result.stderr)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "original output")

    def test_default_batches_are_disjoint_complete_and_reproducible(self):
        rows = []
        for index in range(2007):
            filename = f"clip_{index:04d}.wav"
            self.audio_file(filename)
            rows.append(self.row(clip_id=f"id_{index:04d}", audio_clip_file=filename,
                                 text=f"Transcript {index}"))
        excluded = [self.row(clip_id="excluded", audio_clip_file="missing.wav", **{FLAG: True})]
        self.write_manifest(rows + excluded)

        result = self.run_script()
        tasks = self.assert_success(result)
        files = sorted(self.output_dir.glob("*.json"))
        self.assertEqual([path.name for path in files], ["tasks_001.json", "tasks_002.json", "tasks_003.json"])
        batches = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        self.assertEqual([len(batch) for batch in batches], [1000, 1000, 7])
        ids = [task["data"]["clip_id"] for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {row["clip_id"] for row in rows})
        self.assertNotEqual(ids, [row["clip_id"] for row in rows])
        self.assertIn("batch size: 1000; seed: 42", result.stdout)
        expected = {row["clip_id"]: row for row in rows}
        for task in tasks:
            data = task["data"]
            original = expected[data["clip_id"]]
            self.assertEqual(data["transcript"], original["text"])
            self.assertTrue(data["audio"].endswith("/" + original["audio_clip_file"]))

        original_files = {path.name: path.read_bytes() for path in files}
        # Reordering the same eligible rows must not affect the seeded assignment or JSON bytes.
        self.write_manifest(excluded + list(reversed(rows)))
        repeat_dir = self.batch / "repeat"
        repeat = self.run_script("--output-dir", str(repeat_dir), "--seed", "42")
        self.assertEqual(repeat.returncode, 0, repeat.stderr)
        self.assertEqual({path.name: path.read_bytes() for path in repeat_dir.glob("*.json")}, original_files)

        other_dir = self.batch / "other_seed"
        other = self.run_script("--output-dir", str(other_dir), "--seed", "43")
        self.assertEqual(other.returncode, 0, other.stderr)
        other_ids = {task["data"]["clip_id"] for task in json.loads(
            (other_dir / "tasks_001.json").read_text(encoding="utf-8"))}
        self.assertNotEqual(other_ids, {task["data"]["clip_id"] for task in batches[0]})

    def test_exact_multiple_has_no_empty_trailing_batch(self):
        rows = []
        for index in range(6):
            filename = f"clip_{index}.wav"
            self.audio_file(filename)
            rows.append(self.row(clip_id=str(index), audio_clip_file=filename))
        self.write_manifest(rows)
        self.assertEqual(len(self.assert_success(self.run_script("--batch-size", "3"))), 6)
        self.assertEqual([len(json.loads(path.read_text(encoding="utf-8")))
                          for path in sorted(self.output_dir.glob("*.json"))], [3, 3])

    def test_nonpositive_batch_size_is_rejected(self):
        for value in ("0", "-1"):
            with self.subTest(value=value):
                self.assert_failure(self.run_script("--batch-size", value), "batch size must be greater than zero")

    def test_old_output_argument_is_not_treated_as_an_output_directory(self):
        old_output = self.batch / "tasks.json"
        self.assert_failure(self.run_script("--output", str(old_output)), "unrecognized arguments: --output")
        self.assertFalse(old_output.exists())

    def test_existing_later_or_stale_batch_prevents_partial_output(self):
        self.audio_file("clip.wav")
        self.write_manifest([self.row()])
        self.output_dir.mkdir()
        existing = self.output_dir / "tasks_002.json"
        existing.write_text("existing batch", encoding="utf-8")
        self.assert_failure(self.run_script(), "Batch output already exists")
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing batch")


if __name__ == "__main__":
    unittest.main()

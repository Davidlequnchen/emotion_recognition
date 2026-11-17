"""Hard-link one batch's audio into upload folders without copying, moving or changing the originals."""

import contextlib
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ser_link_upload_audio", PROJECT / "link_upload_audio.py")
LINK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LINK)


class LinkUploadAudio(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve() / "dataset"
        self.audio = self.root / "batch" / "audio"
        self.audio.mkdir(parents=True)
        self.tasks_file = self.root / "batch" / "tasks_001.json"
        self.upload = self.audio / "upload_001"

    def clip(self, name, clip_id=None, encoded=None):
        (self.audio / name).write_bytes(f"audio {name}".encode())
        return {"data": {"audio": "/data/local-files/?d=batch/audio/" + (encoded or name),
                         "transcript": "text", "clip_id": clip_id or name.split(".")[0]}}

    def run_link(self, tasks, size=2):
        self.tasks_file.write_text(json.dumps(tasks), encoding="utf-8")
        settings = {"tasks_file": str(self.tasks_file), "document_root": str(self.root),
                    "upload_dir": str(self.upload), "files_per_folder": size}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            LINK.run(settings)
        return output.getvalue()

    def test_links_share_inodes_split_into_parts_and_keep_sources(self):
        tasks = [self.clip(f"clip_{index}.wav") for index in range(5)]
        tasks.append(self.clip("my clip #1.wav", "spaced", encoded="my%20clip%20%231.wav"))
        before = {path.name: (path.read_bytes(), path.stat().st_mtime) for path in self.audio.glob("*.wav")}
        output = self.run_link(tasks)
        parts = sorted(self.upload.iterdir())
        self.assertEqual([part.name for part in parts], ["part_01", "part_02", "part_03"])
        self.assertEqual([sorted(path.name for path in part.iterdir()) for part in parts],
                         [["clip_0.wav", "clip_1.wav"], ["clip_2.wav", "clip_3.wav"], ["clip_4.wav", "my clip #1.wav"]])
        for part in parts:
            for link in part.iterdir():
                self.assertEqual(os.stat(link).st_ino, os.stat(self.audio / link.name).st_ino)
        self.assertEqual({path.name: (path.read_bytes(), path.stat().st_mtime)
                          for path in self.audio.glob("*.wav")}, before)
        self.assertIn("part_03: 2 files", output)

    def test_resolves_under_document_root_and_blocks_escapes(self):
        outside = self.root.parent / "outside.wav"
        outside.write_bytes(b"outside")
        (self.audio / "link.wav").symlink_to(outside)
        for audio in ["batch/audio/../../../outside.wav", "batch/audio/link.wav"]:
            with self.subTest(audio=audio), self.assertRaisesRegex(ValueError, "escapes document root"):
                self.run_link([{"data": {"audio": "/data/local-files/?d=" + audio, "clip_id": "x"}}])
        with self.assertRaisesRegex(ValueError, "audio file not found"):
            self.run_link([{"data": {"audio": "/data/local-files/?d=batch/audio/missing.wav", "clip_id": "x"}}])
        with self.assertRaisesRegex(ValueError, "local-files"):
            self.run_link([{"data": {"audio": "/data/upload/1/clip.wav", "clip_id": "x"}}])
        self.assertFalse(self.upload.exists())

    def test_duplicate_names_fail_before_linking(self):
        (self.audio / "other").mkdir()
        cases = [("Clip.wav", "other/clip.wav"), ("a b.wav", "a_b.wav"), ("x.wav", "other/x.wav"),
                 ("Übung.wav", "Ubung.wav")]
        for first, second in cases:
            with self.subTest(names=(first, second)):
                tasks = [self.clip(first, "one"), self.clip(second, "two")]
                with self.assertRaisesRegex(ValueError, "not unique"):
                    self.run_link(tasks)
                self.assertFalse(self.upload.exists())

    def test_upload_dir_must_be_new_or_empty(self):
        tasks = [self.clip("clip.wav")]
        self.upload.mkdir()
        self.run_link(tasks)  # an empty folder is fine
        with self.assertRaisesRegex(ValueError, "new or empty"):
            self.run_link(tasks)  # now it holds part_01

    def test_failing_link_leaves_no_partial_output(self):
        tasks = [self.clip(f"clip_{index}.wav") for index in range(3)]
        real_link, calls = os.link, []

        def flaky(source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError(errno.EXDEV, os.strerror(errno.EXDEV))
            real_link(source, target)

        with mock.patch.object(LINK.os, "link", flaky), self.assertRaisesRegex(ValueError, "same disk"):
            self.run_link(tasks)
        self.assertEqual(len(calls), 2)
        self.assertFalse(self.upload.exists())
        self.assertEqual(sorted(path.name for path in self.audio.iterdir()), ["clip_0.wav", "clip_1.wav", "clip_2.wav"])

    def test_invalid_tasks_and_folder_size(self):
        with self.assertRaisesRegex(ValueError, "clip_id"):
            self.run_link([self.clip("a.wav", "same"), self.clip("b.wav", "same")])
        with self.assertRaisesRegex(ValueError, "files_per_folder"):
            self.run_link([self.clip("a.wav")], size=0)
        with self.assertRaisesRegex(ValueError, "longer than"):
            self.run_link([self.clip("x" * 80 + ".wav")])


if __name__ == "__main__":
    unittest.main()

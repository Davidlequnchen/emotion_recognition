"""Match directly uploaded audio tasks to tasks_NNN.json and attach transcript and clip_id, without a network."""

import contextlib
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest import mock
from urllib.parse import urlparse


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ser_attach_task_data", PROJECT / "attach_task_data.py")
ATTACH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ATTACH)

ENV = {"LABEL_STUDIO_URL": "https://label-studio.example/", "LABEL_STUDIO_API_KEY": "legacy-secret-key"}
SECRET_TEXT = "private transcript words"


class Response:
    def __init__(self, status, body):
        self.status_code, self.body, self.text = status, body, json.dumps(body)

    def json(self):
        return deepcopy(self.body)


class FakeLabelStudio:
    """A requests.Session stand-in: one project, paginated task list, PATCH replaces task.data."""

    def __init__(self, tasks, project_id=7):
        self.tasks = {task["id"]: deepcopy(task) for task in tasks}
        self.project_id, self.calls, self.reject = project_id, [], False

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        path = urlparse(url).path
        self.calls.append((method, path, params, json, (headers or {}).get("Authorization")))
        if self.reject:
            return Response(401, {"detail": "Invalid token."})
        if path == "/api/token/refresh/":
            return Response(200, {"access": "short-lived"})
        if path == f"/api/projects/{self.project_id}/":
            return Response(200, {"title": "Demo", "task_number": len(self.tasks)})
        if path == "/api/tasks/":
            page, size = params["page"], params["page_size"]
            items = list(self.tasks.values())[(page - 1) * size:page * size]
            return Response(200 if items or page == 1 else 404, {"tasks": items, "total": len(self.tasks)})
        task_id = int(re.fullmatch(r"/api/tasks/(\d+)/", path).group(1))
        if method == "PATCH":
            self.tasks[task_id]["data"] = json["data"]
        return Response(200, self.tasks[task_id])

    def patches(self):
        return [call for call in self.calls if call[0] == "PATCH"]


def uploaded(task_id, name, key="$undefined$", project=7):
    return {"id": task_id, "data": {key: f"/data/upload/{project}/{task_id:08x}-{name}"}}


class AttachTaskData(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tasks_file = Path(temp.name) / "tasks_001.json"
        self.local = [{"data": {"audio": "/data/local-files/?d=batch/audio/" + encoded, "transcript": text,
                                "clip_id": clip}}
                      for encoded, text, clip in [("clip_0001.wav", SECRET_TEXT, "c1"),
                                                  ("clip_0002.wav", "second", "c2"),
                                                  ("my%20clip%20%231.wav", "", "c3")]]
        self.tasks_file.write_text(json.dumps(self.local), encoding="utf-8")

    def run_attach(self, server, write=False, limit=3, env=ENV):
        settings = {"project_id": 7, "tasks_file": str(self.tasks_file), "write": write, "limit": limit}
        with mock.patch.dict(os.environ, env, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            ATTACH.run(settings, server)
        return output.getvalue()

    def test_upload_names_lose_the_random_prefix_once(self):
        self.assertEqual(ATTACH.upload_name("/data/upload/7/1a2b3c4d-clip_0001.wav")[1], "clip_0001.wav")
        self.assertEqual(ATTACH.upload_name("https://host.example/data/upload/7/0f0f0f0f-deadbeef-x.wav?t=1")[1],
                         "deadbeef-x.wav")
        self.assertEqual(ATTACH.upload_name("upload/7/abcdef12-%E4%B8%80.wav")[1], "一.wav")
        self.assertEqual(ATTACH.django_name(" my clip #1.wav"), "my_clip_1.wav")

    def test_decomposed_accents_from_a_mac_still_match(self):
        local = [{"data": {"audio": "/data/local-files/?d=a/%C3%9Cbung%201.wav", "transcript": "t", "clip_id": "u1"}}]
        self.tasks_file.write_text(json.dumps(local), encoding="utf-8")
        server = FakeLabelStudio([uploaded(1, "Ubung_1.wav"), uploaded(2, "%C3%9Cbung_1.wav")])
        with self.assertRaisesRegex(ValueError, r"u1: tasks \[1, 2\]"):  # both forms match the same clip
            self.run_attach(server)

    def test_write_matches_both_keys_and_sanitized_names_and_replaces_data(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav"), uploaded(2, "clip_0002.wav", key="audio"),
                                  uploaded(3, "my_clip_1.wav")])
        server.tasks[1]["data"]["extra"] = "kept"
        output = self.run_attach(server, write=True)
        self.assertEqual(len(server.patches()), 3)
        self.assertEqual(server.tasks[1]["data"], {"extra": "kept", "audio": "/data/upload/7/00000001-clip_0001.wav",
                                                   "transcript": SECRET_TEXT, "clip_id": "c1"})
        self.assertEqual(server.tasks[3]["data"]["clip_id"], "c3")
        self.assertEqual(server.tasks[2]["data"]["audio"], "/data/upload/7/00000002-clip_0002.wav")
        self.assertIn("config_with_transcript.xml", output)
        self.assertNotIn(SECRET_TEXT, output)
        self.assertNotIn(ENV["LABEL_STUDIO_API_KEY"], output)
        self.assertTrue(all(call[4] == "Token legacy-secret-key" for call in server.calls))
        # A second run finds everything done and writes nothing.
        self.run_attach(server, write=True)
        self.assertEqual(len(server.patches()), 3)

    def test_dry_run_writes_nothing(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav")])
        output = self.run_attach(server)
        self.assertEqual(server.patches(), [])
        self.assertIn("Dry run, nothing written", output)
        self.assertIn("2 clips of the tasks file are not in Label Studio", output)

    def test_limit_caps_writes_per_run(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav"), uploaded(2, "clip_0002.wav"),
                                  uploaded(3, "my_clip_1.wav")])
        output = self.run_attach(server, write=True, limit=2)
        self.assertEqual(len(server.patches()), 2)
        self.assertIn("1 tasks still need", output)
        self.run_attach(server, write=True, limit=2)
        self.assertEqual(len(server.patches()), 3)

    def test_other_batches_are_counted_and_ignored(self):
        earlier = {"id": 9, "data": {"audio": "/data/upload/7/aaaaaaaa-old.wav", "transcript": "x", "clip_id": "b1"}}
        server = FakeLabelStudio([earlier, uploaded(1, "clip_0001.wav")])
        output = self.run_attach(server, write=True)
        self.assertIn("other batches (ignored): 1", output)
        self.assertEqual([call[1] for call in server.patches()], ["/api/tasks/1/"])
        self.assertEqual(server.tasks[9], earlier)

    def test_unmatched_upload_stops_before_any_write(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav"), uploaded(2, "stray.wav")])
        with self.assertRaisesRegex(ValueError, r"(?s)Nothing written.*1 tasks without clip_id.*stray\.wav"):
            self.run_attach(server, write=True)
        self.assertEqual(server.patches(), [])

    def test_duplicate_uploads_stop_before_any_write(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav"), uploaded(2, "clip_0001.wav")])
        with self.assertRaisesRegex(ValueError, r"(?s)delete the duplicate.*c1: tasks \[1, 2\]"):
            self.run_attach(server, write=True)
        self.assertEqual(server.patches(), [])

    def test_conflicting_attached_task_stops(self):
        done = {"id": 1, "data": {"audio": "/data/upload/7/00000001-clip_0001.wav", "transcript": "changed",
                                  "clip_id": "c1"}}
        server = FakeLabelStudio([done, uploaded(2, "clip_0002.wav")])
        with self.assertRaisesRegex(ValueError, "different file or transcript"):
            self.run_attach(server, write=True)
        self.assertEqual(server.patches(), [])

    def test_pages_stop_at_the_last_page(self):
        names = [f"clip_{index:04d}.wav" for index in range(250)]
        local = [{"data": {"audio": "/data/local-files/?d=a/" + name, "transcript": "t", "clip_id": name}}
                 for name in names]
        self.tasks_file.write_text(json.dumps(local), encoding="utf-8")
        server = FakeLabelStudio([uploaded(index + 1, name) for index, name in enumerate(names)])
        self.run_attach(server, write=True, limit=None)
        pages = [call[2]["page"] for call in server.calls if call[1] == "/api/tasks/"]
        self.assertEqual(pages, [1, 2, 3])
        self.assertEqual(len(server.patches()), 250)

    def test_personal_access_token_and_auth_errors(self):
        server = FakeLabelStudio([uploaded(1, "clip_0001.wav")])
        self.run_attach(server, env={**ENV, "LABEL_STUDIO_API_KEY": "eyJhbGciOi.eyJ0b2tlbiI6.c2lnbmF0dXJl"})
        self.assertEqual(server.calls[0][:2], ("POST", "/api/token/refresh/"))
        self.assertTrue(all(call[4] == "Bearer short-lived" for call in server.calls[1:]))
        server.reject = True
        with self.assertRaisesRegex(ValueError, "Personal Access Token"):
            self.run_attach(server)
        for missing in ("LABEL_STUDIO_URL", "LABEL_STUDIO_API_KEY"):
            with self.subTest(missing=missing), self.assertRaisesRegex(ValueError, f"Set {missing}"):
                self.run_attach(server, env={key: value for key, value in ENV.items() if key != missing})

    def test_settings_file_is_validated(self):
        path = Path(self.tasks_file.parent) / "settings.yaml"
        for text, message in [("project_id: 7\n", "missing keys"), ("project_id: 7\ntasks_file: x\nurl: y\n",
                                                                     "unknown keys"),
                              ("project_id: 0\ntasks_file: x\n", "project_id"),
                              ("project_id: 7\ntasks_file: x\nlimit: 0\n", "limit")]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                path.write_text(text, encoding="utf-8")
                ATTACH.load_settings(path)
        path.write_text("project_id: 7\ntasks_file: x\n", encoding="utf-8")
        self.assertEqual(ATTACH.load_settings(path)["write"], False)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Add transcript and clip_id from tasks_NNN.json to audio tasks uploaded directly into Label Studio.

Settings: label_studio/config/attach_task_data.yaml; server and key: LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY.
Run from the repository root: python3 label_studio/attach_task_data.py
"""

import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata
from urllib.parse import unquote, urlparse

import yaml


CONFIG = Path(__file__).resolve().parent / "config" / "attach_task_data.yaml"
REQUIRED = {"project_id", "tasks_file"}
DEFAULTS = {"write": False, "limit": 3}
UNDEFINED = "$undefined$"  # Label Studio's key for a directly uploaded file (settings.DATA_UNDEFINED_NAME)
PAGE_SIZE = 100
TIMEOUT = 60  # seconds per request


class Unauthorized(ValueError):
    def __str__(self):
        return ("HTTP 401: the server rejected LABEL_STUDIO_API_KEY. Check the token (Account & Settings). "
                "If the admin disabled legacy tokens, use a Personal Access Token instead.")


def step(number, title):
    print(f"\n== Step {number}: {title}")


def django_name(name):
    """The file name Label Studio (Django get_valid_filename) keeps for an uploaded file."""
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def upload_name(url):
    """'/data/upload/7/1a2b3c4d-clip_01.wav' -> ('1a2b3c4d-clip_01.wav', 'clip_01.wav')."""
    raw = unquote(PurePosixPath(urlparse(url).path).name)
    return raw, re.sub(r"^[0-9a-f]{8}-", "", raw)


def load_settings(path):
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected 'key: value' settings")
    missing, unknown = REQUIRED - settings.keys(), settings.keys() - REQUIRED - DEFAULTS.keys()
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
    settings = {**DEFAULTS, **settings}
    if not positive_int(settings["project_id"]):
        raise ValueError("project_id must be a positive integer")
    if not isinstance(settings["write"], bool):
        raise ValueError("write must be true or false")
    if settings["limit"] is not None and not positive_int(settings["limit"]):
        raise ValueError("limit must be a positive integer, or null for all tasks")
    return settings


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def local_clips(path):
    """Read tasks_NNN.json: clip_id -> (file name, transcript), and upload name -> clip_id."""
    tasks = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: expected a non-empty JSON list of tasks")
    clips, lookup, keys = {}, {}, {}
    for index, task in enumerate(tasks, 1):
        context = f"{path}, task {index}"
        data = task.get("data") if isinstance(task, dict) else None
        if not isinstance(data, dict):
            raise ValueError(f"{context}: missing data object")
        clip_id, transcript, audio = data.get("clip_id"), data.get("transcript"), data.get("audio")
        if not isinstance(clip_id, str) or not clip_id.strip() or clip_id in clips:
            raise ValueError(f"{context}: data.clip_id must be a unique, non-empty string")
        if not isinstance(transcript, str):
            raise ValueError(f"{context}: data.transcript must be text")
        if not isinstance(audio, str) or "?d=" not in audio:
            raise ValueError(f"{context}: data.audio must be a /data/local-files/?d=... URL")
        name = PurePosixPath(unquote(audio.split("?d=", 1)[1])).name
        # Forms Label Studio may store; a Mac may send accents decomposed (NFD), and Django then drops them.
        for form in {name, django_name(name), django_name(unicodedata.normalize("NFD", name))}:
            if keys.setdefault(form.casefold(), clip_id) != clip_id:
                raise ValueError(f"{context}: file name not unique (ignoring case and dropped characters): {name}")
            lookup[form] = clip_id
        clips[clip_id] = (name, transcript)
    return clips, lookup


def classify(tasks, clips, lookup):
    """Sort Label Studio tasks: to update, already done, other batch, unmatched, conflicting; find duplicates."""
    found = {"update": [], "done": 0, "other": 0, "unmatched": [], "conflict": []}
    owners = {}
    for task in tasks:
        data = task.get("data") or {}
        url = data.get("audio") or data.get(UNDEFINED)
        raw, name = upload_name(url) if isinstance(url, str) else ("", "(no audio URL)")
        match = lookup.get(name) or lookup.get(raw)
        clip_id = data.get("clip_id")
        if clip_id is None:
            if match is None:
                found["unmatched"].append(f"task {task['id']}: {name}")
                continue
            clip_id = match
        elif clip_id not in clips:
            found["other"] += 1  # an earlier batch in the same project
            continue
        elif match != clip_id or data.get("transcript") != clips[clip_id][1]:
            found["conflict"].append(f"task {task['id']}: clip_id {clip_id}")
            continue
        owners.setdefault(clip_id, []).append(task["id"])
        if "clip_id" in data and "audio" in data and UNDEFINED not in data:
            found["done"] += 1
        else:
            found["update"].append((task, clip_id, url))
    found["duplicates"] = [f"{clip}: tasks {ids}" for clip, ids in owners.items() if len(ids) > 1]
    found["missing"] = [clip for clip in clips if clip not in owners]
    return found


class Api:
    """Label Studio HTTP API with a Legacy Token or a Personal Access Token (a JWT refresh token)."""

    def __init__(self, session, url, key):
        self.session, self.url, self.key = session, url, key
        self.pat = key.startswith("eyJ") and key.count(".") == 2
        self.header = None if self.pat else f"Token {key}"

    def send(self, method, path, headers=None, **kwargs):
        response = self.session.request(method, self.url + path, headers=headers, timeout=TIMEOUT, **kwargs)
        if response.status_code == 401:
            raise Unauthorized()
        if not 200 <= response.status_code < 300:
            raise ValueError(f"{method} {path}: HTTP {response.status_code}: {response.text[:300]}")
        try:
            return response.json()
        except ValueError:
            raise ValueError(f"{method} {path}: response is not JSON (login page or proxy?): "
                             f"{response.text[:300]}") from None

    def call(self, method, path, retry=True, **kwargs):
        if self.header is None:  # PAT: trade it for an access token, which expires after about 5 minutes
            self.header = "Bearer " + self.send("POST", "/api/token/refresh/", json={"refresh": self.key})["access"]
        try:
            return self.send(method, path, headers={"Authorization": self.header}, **kwargs)
        except Unauthorized:
            if not (self.pat and retry):
                raise
            self.header = None  # expired access token: refresh once and retry
            return self.call(method, path, retry=False, **kwargs)


def fetch_tasks(api, project_id):
    """All tasks of the project, page by page, never asking past the last page."""
    tasks = []
    for page in range(1, 100000):
        body = api.call("GET", "/api/tasks/", params={"project": project_id, "page": page, "page_size": PAGE_SIZE})
        if not isinstance(body, dict) or not isinstance(body.get("tasks"), list) \
                or not isinstance(body.get("total"), int):
            raise ValueError(f"GET /api/tasks/: unexpected response shape: {str(body)[:300]}")
        tasks += body["tasks"]
        if not body["tasks"] or len(tasks) >= body["total"]:
            return tasks


def run(settings, session):
    # 1. Local tasks file and environment; the key is never printed.
    step(1, "Check settings, environment and the local tasks file")
    url = os.environ.get("LABEL_STUDIO_URL", "").strip().rstrip("/")
    key = os.environ.get("LABEL_STUDIO_API_KEY", "").strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError('Set LABEL_STUDIO_URL first, e.g. export LABEL_STUDIO_URL="https://<your-label-studio-host>"')
    if not key:
        raise ValueError("Set LABEL_STUDIO_API_KEY first: read -rs LABEL_STUDIO_API_KEY && export LABEL_STUDIO_API_KEY")
    clips, lookup = local_clips(settings["tasks_file"])
    api = Api(session, url, key)
    limit = settings["limit"]
    print(f"Server: {url} ({'Personal Access Token' if api.pat else 'Legacy Token'})")
    print(f"Tasks file: {Path(settings['tasks_file']).name} with {len(clips)} clips")
    print(f"Mode: {'WRITE' if settings['write'] else 'dry run'}; limit: {'all' if limit is None else limit}")

    # 2. The project.
    step(2, "Read the project")
    project = api.call("GET", f"/api/projects/{settings['project_id']}/")
    print(f"Project {settings['project_id']}: {project.get('title')!r}; tasks: {project.get('task_number')}")

    # 3. All tasks, in pages of 100.
    step(3, "Read all tasks")
    tasks = fetch_tasks(api, settings["project_id"])
    if isinstance(project.get("task_number"), int) and len(tasks) != project["task_number"]:
        raise ValueError(f"Read {len(tasks)} tasks, but the project has {project['task_number']}; nothing written")
    print(f"Read {len(tasks)} tasks")

    # 4. Match by file name; any problem stops the run before writing.
    step(4, "Match uploaded files to the tasks file")
    found = classify(tasks, clips, lookup)
    print(f"To update: {len(found['update'])}; already done: {found['done']}; "
          f"other batches (ignored): {found['other']}")
    for task, clip_id, audio in found["update"][:3]:
        print(f"  e.g. task {task['id']}: {upload_name(audio)[1]} -> {clip_id}")
    problems = [
        (found["unmatched"], "tasks without clip_id match no file in the tasks file (wrong tasks file?)"),
        (found["conflict"], "tasks already have this batch's clip_id, but a different file or transcript"),
        (found["duplicates"], "clips match more than one task (uploaded twice?); delete the duplicate tasks "
                              "in the Data Manager"),
    ]
    errors = [f"{len(items)} {text}:\n  " + "\n  ".join(items[:10]) for items, text in problems if items]
    if errors:
        raise ValueError("Nothing written.\n" + "\n".join(errors))
    if found["missing"]:
        print(f"Warning: {len(found['missing'])} clips of the tasks file are not in Label Studio yet "
              f"(part not uploaded?): {', '.join(found['missing'][:10])}")

    # 5. Write, or stop after the dry run.
    step(5, "Write transcript and clip_id")
    if not settings["write"]:
        print("Dry run, nothing written; set write: true in label_studio/config/attach_task_data.yaml")
        return
    pending = found["update"] if limit is None else found["update"][:limit]
    written = 0
    try:
        for task, clip_id, audio in pending:
            # PATCH replaces the whole task.data: keep other keys, drop $undefined$, name the audio key explicitly.
            data = {key: value for key, value in task["data"].items() if key != UNDEFINED}
            data.update(audio=audio, transcript=clips[clip_id][1], clip_id=clip_id)
            api.call("PATCH", f"/api/tasks/{task['id']}/", json={"data": data})
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{len(pending)}")
    except Exception as exc:
        raise ValueError(f"{exc}\n{written} tasks were updated before this error; run again to continue "
                         f"(finished tasks are skipped)") from exc
    print(f"Updated {written} tasks")
    if pending:
        task, clip_id, audio = pending[-1]
        data = api.call("GET", f"/api/tasks/{task['id']}/").get("data", {})
        if (data.get("audio"), data.get("transcript"), data.get("clip_id")) != (audio, clips[clip_id][1], clip_id):
            raise ValueError(f"Task {task['id']} does not show the written fields; check it in the Data Manager")
        print(f"Checked task {task['id']}: audio, transcript and clip_id are stored")

    # 6. Next step.
    step(6, "Next")
    remaining = len(found["update"]) - written
    if remaining:
        print(f"{remaining} tasks still need transcript and clip_id; run again (limit: null = all)")
    elif found["missing"]:
        print("Upload the missing files with the audio-only template, then run again")
    else:
        print("All tasks of this file are done. Switch the template to config_with_transcript.xml\n"
              "(Settings → Labeling Interface → Code), then let the annotators start")


def main():
    import requests  # only needed here, so the tests run without network code

    try:
        with requests.Session() as session:
            run(load_settings(CONFIG), session)
    except (OSError, ValueError, yaml.YAMLError, requests.RequestException) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()

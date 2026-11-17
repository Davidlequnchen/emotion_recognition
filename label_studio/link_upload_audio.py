#!/usr/bin/env python3
"""Hard-link the audio files of one tasks_NNN.json into small folders for Label Studio Import → Upload Files.

Hard links use no extra disk space and leave the originals where the manifest expects them.
Settings: label_studio/config/link_upload_audio.yaml. Run from the repository root:
    python3 label_studio/link_upload_audio.py
"""

import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import unicodedata
from urllib.parse import unquote

import yaml


CONFIG = Path(__file__).resolve().parent / "config" / "link_upload_audio.yaml"
REQUIRED = {"tasks_file", "document_root", "upload_dir"}
DEFAULTS = {"files_per_folder": 100}
# Label Studio stores an upload as upload/<project>/<8 hex>-<name> in a 100-character field and cuts longer
# names, which then no longer match tasks_NNN.json. Keep names at or below this length.
MAX_NAME_LENGTH = 70
NO_HARD_LINKS = {errno.EXDEV, errno.EPERM, errno.ENOTSUP, errno.EMLINK}


def step(number, title):
    print(f"\n== Step {number}: {title}")


def django_name(name):
    """The file name Label Studio (Django get_valid_filename) keeps for an uploaded file."""
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def name_keys(name):
    """Case-folded names Label Studio may store; a Mac may send accents decomposed (NFD), which Django drops."""
    forms = (name, django_name(name), django_name(unicodedata.normalize("NFD", name)))
    return {form.casefold() for form in forms}


def load_settings(path):
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError(f"{path}: expected 'key: value' settings")
    missing, unknown = REQUIRED - settings.keys(), settings.keys() - REQUIRED - DEFAULTS.keys()
    if missing or unknown:
        raise ValueError(f"{path}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
    return {**DEFAULTS, **settings}


def source_audio(audio, root, context):
    """Resolve /data/local-files/?d=<path> under the document root; return the file and its upload name."""
    if not isinstance(audio, str) or "?d=" not in audio:
        raise ValueError(f"{context}: data.audio must be a /data/local-files/?d=... URL")
    relative = unquote(audio.split("?d=", 1)[1])
    try:
        path = (root / relative).resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"{context}: audio file not found: {root / relative}") from None
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f"{context}: path escapes document root: {relative}") from None
    if not path.is_file():
        raise ValueError(f"{context}: audio path is not a file: {path}")
    return path, PurePosixPath(relative).name


def plan(tasks, root):
    """Validate every task and return [(source file, upload name)] in tasks-file order."""
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks_file must contain a non-empty JSON list of tasks")
    items, clip_ids, names = [], set(), {}
    for index, task in enumerate(tasks, 1):
        context = f"Task {index}"
        data = task.get("data") if isinstance(task, dict) else None
        if not isinstance(data, dict):
            raise ValueError(f"{context}: missing data object")
        clip_id = data.get("clip_id")
        if not isinstance(clip_id, str) or not clip_id.strip() or clip_id in clip_ids:
            raise ValueError(f"{context}: data.clip_id must be a unique, non-empty string")
        clip_ids.add(clip_id)
        source, name = source_audio(data.get("audio"), root, context)
        if len(django_name(name)) > MAX_NAME_LENGTH:
            raise ValueError(f"{context}: file name longer than {MAX_NAME_LENGTH} characters; "
                             f"Label Studio would shorten it on upload: {name}")
        # Matching in Label Studio is by file name, so names must differ even ignoring case and dropped characters.
        for key in name_keys(name):
            names.setdefault(key, {})[index] = name
        items.append((source, name))
    duplicates = sorted({" = ".join(group.values()) for group in names.values() if len(group) > 1})
    if duplicates:
        raise ValueError(f"{len(duplicates)} file names are not unique (ignoring case and the characters "
                         f"Label Studio drops); rename them first: " + "; ".join(duplicates[:10]))
    return items


def check_upload_dir(path):
    if path.exists() or path.is_symlink():
        if not path.is_dir():
            raise ValueError(f"upload_dir is not a directory: {path}")
        if any(path.iterdir()):
            raise ValueError(f"upload_dir must be new or empty: {path}")


def link_parts(parts, upload_dir):
    """Hard-link every part; on any failure remove what this run created and never fall back to copying."""
    created_root = not upload_dir.exists()
    folders, links = [], []
    try:
        for number, part in enumerate(parts, 1):
            folder = upload_dir / f"part_{number:02d}"
            folder.mkdir(parents=True)
            folders.append(folder)
            for source, name in part:
                os.link(source, folder / name)  # the first file doubles as the test that linking works here
                links.append(folder / name)
    except OSError as exc:
        for path in links:
            path.unlink()
        for folder in reversed(folders):
            folder.rmdir()
        if created_root and upload_dir.exists():
            upload_dir.rmdir()
        if exc.errno in NO_HARD_LINKS:
            raise ValueError(f"Cannot create hard links in {upload_dir} ({exc.strerror}). Put upload_dir on the "
                             "same disk as the audio, e.g. inside batch/audio/; exFAT and some network shares "
                             "do not support hard links. Nothing was created.") from None
        raise ValueError(f"Linking failed, nothing was created: {exc}") from None


def run(settings):
    # 1. Settings: real paths, a positive folder size.
    step(1, "Check settings")
    tasks_file, root = (Path(settings[key]).expanduser().resolve() for key in ("tasks_file", "document_root"))
    if not tasks_file.is_file() or not root.is_dir():
        raise ValueError(f"tasks_file must be an existing file and document_root an existing folder: "
                         f"{tasks_file}, {root}")
    upload_dir = Path(settings["upload_dir"]).expanduser().absolute()
    size = settings["files_per_folder"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("files_per_folder must be a positive integer")
    print(f"Tasks file: {tasks_file}\nDocument root: {root}\nUpload folder: {upload_dir}\nFiles per folder: {size}")

    # 2. Read the tasks and find every audio file; nothing is created yet.
    step(2, "Read tasks and find the audio files")
    items = plan(json.loads(tasks_file.read_text(encoding="utf-8")), root)
    total_mb = sum(source.stat().st_size for source, _ in items) / 1e6
    print(f"{len(items)} tasks; all audio files found; file names unique; {total_mb:.1f} MB")

    # 3. The upload folder must be new or empty, so it holds exactly this batch.
    step(3, "Check the upload folder")
    check_upload_dir(upload_dir)
    parts = [items[offset:offset + size] for offset in range(0, len(items), size)]
    print(f"OK; {len(parts)} part folders of at most {size} files")

    # 4. Hard links, in tasks-file order; the originals are not moved or changed.
    step(4, "Create hard links")
    link_parts(parts, upload_dir)
    for number, part in enumerate(parts, 1):
        print(f"  part_{number:02d}: {len(part)} files")
    print(f"Linked {len(items)} files ({total_mb:.1f} MB, no extra disk space) in {upload_dir}")

    # 5. What to do in Label Studio.
    step(5, "Next")
    print("In Label Studio, keep config_audio_only.xml as the labeling template. Open Import, select all files of\n"
          "one part folder and drag them in; one part per upload. Do not edit files in the upload folder (that\n"
          "changes the originals); delete the folder after uploading if you like.\n"
          "Then run: python3 label_studio/attach_task_data.py")


def main():
    try:
        run(load_settings(CONFIG))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()

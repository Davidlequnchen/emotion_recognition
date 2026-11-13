#!/usr/bin/env python3
"""Create clip-level Label Studio tasks from flat audio and optional UTF-8 transcripts."""

import argparse
import json
from pathlib import Path
import sys
from urllib.parse import quote


AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def directory(value, name):
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")
    return path


def within(path, root):
    try:
        return path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes document root: {path}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Flat directory of audio clips")
    parser.add_argument("--transcript-dir", help="Same-stem .txt files; defaults to audio directory")
    parser.add_argument("--document-root", required=True, help="Same root used by Label Studio local file serving")
    parser.add_argument("--output", required=True, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args()

    try:
        root = directory(args.document_root, "document root")
        audio_dir = directory(args.audio_dir, "audio directory")
        transcript_dir = directory(args.transcript_dir or args.audio_dir, "transcript directory")
        within(audio_dir, root)
        within(transcript_dir, root)
        output = Path(args.output).expanduser().absolute()
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists; choose a new filename: {output}")

        tasks = []
        stems = set()
        missing_transcripts = 0
        for audio in sorted(audio_dir.iterdir(), key=lambda p: p.name.casefold()):
            if audio.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            resolved_audio = audio.resolve(strict=True)
            relative_audio = within(resolved_audio, root)
            if not resolved_audio.is_file():
                continue
            stem_key = audio.stem.casefold()
            if stem_key in stems:
                raise ValueError(f"Duplicate audio stem (including case variants): {audio.stem}")
            stems.add(stem_key)

            transcript_path = transcript_dir / f"{audio.stem}.txt"
            if transcript_path.exists() or transcript_path.is_symlink():
                resolved_transcript = transcript_path.resolve(strict=True)
                within(resolved_transcript, root)
                transcript = resolved_transcript.read_text(encoding="utf-8").strip()
            else:
                transcript = ""
                missing_transcripts += 1
            tasks.append({"data": {
                "audio": "/data/local-files/?d=" + quote(relative_audio.as_posix(), safe="/"),
                "transcript": transcript,
                "clip_id": audio.stem,
            }})

        if not tasks:
            raise ValueError(f"No supported audio files found in: {audio_dir}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(tasks, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {len(tasks)} tasks to {output}")
    print(f"Missing transcripts: {missing_transcripts}; empty strings used.", file=sys.stderr)


if __name__ == "__main__":
    main()

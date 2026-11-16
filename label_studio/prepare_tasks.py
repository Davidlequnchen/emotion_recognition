#!/usr/bin/env python3
"""Create reproducible, non-overlapping Label Studio batches from eligible manifest clips."""

import argparse
import json
import math
from pathlib import Path
import random
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


def manifest_rows(path, columns):
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Reading manifest.parquet requires pyarrow; install label_studio/requirements.txt "
                         "with the Python environment used to run this script") from exc

    try:
        with pq.ParquetFile(path) as source:
            names = source.schema_arrow.names
            if len(names) != len(set(names)):
                raise ValueError("Manifest contains duplicate column names")
            missing = sorted(set(columns) - set(names))
            if missing:
                raise ValueError(f"Missing manifest columns: {', '.join(missing)}. "
                                 f"Available columns: {', '.join(names)}")
            for batch in source.iter_batches(columns=list(dict.fromkeys(columns))):
                yield from batch.to_pylist()
    except pa.ArrowException as exc:
        raise ValueError(f"Cannot read Parquet manifest {path}: {exc}") from exc


def long_clip_flag(value, context):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false", "1", "0"}:
        return value.strip().casefold() in {"true", "1"}
    raise ValueError(f"{context}: expected true/false or 1/0; missing or invalid long-clip flags are not allowed")


def resolve_audio(value, manifest_dir, audio_dir, root, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: audio file must be a non-empty filename or filesystem path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        # Bare filenames live in audio/; paths such as audio/clip.wav are batch-relative.
        path = (audio_dir if len(path.parts) == 1 else manifest_dir) / path
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"{context}: unsupported audio extension: {path.suffix}")
    path = path.resolve(strict=True)
    within(path, root)
    if not path.is_file():
        raise ValueError(f"{context}: audio path is not a file: {path}")
    return path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("batch size must be greater than zero")
    return number


def check_output_dir(output_dir):
    if (output_dir.exists() or output_dir.is_symlink()) and not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")
    existing = next(output_dir.glob("tasks_*.json"), None)
    if existing is not None:
        raise ValueError(f"Batch output already exists: {existing}; choose a new output directory")


def write_batches(tasks, output_dir, batch_size, seed):
    # Canonical ordering makes the split independent of the manifest's row order.
    tasks.sort(key=lambda task: task["data"]["clip_id"])
    random.Random(seed).shuffle(tasks)
    check_output_dir(output_dir)
    written = []
    if not tasks:
        return written
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, offset in enumerate(range(0, len(tasks), batch_size), 1):
            batch = tasks[offset:offset + batch_size]
            serialized = json.dumps(batch, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            output = output_dir / f"tasks_{index:03d}.json"
            with output.open("x", encoding="utf-8") as handle:
                written.append((output, len(batch)))
                handle.write(serialized)
    except (OSError, UnicodeError, ValueError):
        # Remove only files created by this run if a later batch fails to write.
        for output, _ in written:
            output.unlink()
        raise
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", required=True, help="Batch manifest.parquet")
    parser.add_argument("--audio-dir", help="Directory for bare audio filenames; defaults to audio/ beside the manifest")
    parser.add_argument("--clip-id-column", default="clip_id", help="Stable clip ID column (default: clip_id)")
    parser.add_argument("--audio-column", default="audio_clip_file", help="Audio filename/path column (default: audio_clip_file)")
    parser.add_argument("--text-column", default="text", help="Transcript column (default: text)")
    parser.add_argument("--long-clip-column", default="is_longer_than_10_seconds",
                        help="Boolean column: true excludes the row (default: is_longer_than_10_seconds)")
    parser.add_argument("--document-root", required=True, help="Same root used by Label Studio local file serving")
    parser.add_argument("--output-dir", required=True, help="Directory for tasks_001.json, tasks_002.json, etc.; existing batches are never overwritten")
    parser.add_argument("--batch-size", type=positive_int, default=1000, help="Maximum tasks per JSON file (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random shuffle seed (default: 42)")
    args = parser.parse_args()

    try:
        root = directory(args.document_root, "document root")
        manifest = Path(args.manifest).expanduser().resolve(strict=True)
        if not manifest.is_file():
            raise ValueError(f"Manifest is not a file: {manifest}")
        within(manifest, root)
        audio_dir = directory(args.audio_dir, "audio directory") if args.audio_dir else manifest.parent / "audio"
        within(audio_dir, root)
        output_dir = Path(args.output_dir).expanduser().absolute()
        check_output_dir(output_dir)

        tasks = []
        clip_ids = set()
        audio_files = set()
        row_count = 0
        excluded_count = 0
        missing_transcripts = 0
        columns = [args.clip_id_column, args.audio_column, args.text_column, args.long_clip_column]
        for row_count, row in enumerate(manifest_rows(manifest, columns), 1):
            context = f"Manifest row {row_count}"
            if long_clip_flag(row[args.long_clip_column], f"{context}, {args.long_clip_column}"):
                excluded_count += 1
                continue

            clip_id = row[args.clip_id_column]
            if isinstance(clip_id, int) and not isinstance(clip_id, bool):
                clip_id = str(clip_id)
            if not isinstance(clip_id, str) or not clip_id.strip():
                raise ValueError(f"{context}: clip ID must be a non-empty string or integer")
            if clip_id in clip_ids:
                raise ValueError(f"{context}: duplicate clip ID: {clip_id}")
            clip_ids.add(clip_id)

            audio = resolve_audio(row[args.audio_column], manifest.parent, audio_dir, root, context)
            if audio in audio_files:
                raise ValueError(f"{context}: duplicate audio file: {audio}")
            audio_files.add(audio)

            transcript = row[args.text_column]
            if transcript is None or (isinstance(transcript, float) and math.isnan(transcript)):
                transcript = ""
            if not isinstance(transcript, str):
                raise ValueError(f"{context}: transcript must be text or null")
            transcript = transcript.strip()
            if not transcript:
                missing_transcripts += 1
            tasks.append({"data": {
                "audio": "/data/local-files/?d=" + quote(within(audio, root).as_posix(), safe="/"),
                "transcript": transcript,
                "clip_id": clip_id,
            }})

        written = write_batches(tasks, output_dir, args.batch_size, args.seed)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {len(tasks)} tasks in {len(written)} batches to {output_dir} "
          f"(batch size: {args.batch_size}; seed: {args.seed})")
    for output, count in written:
        print(f"  {output.name}: {count} tasks")
    print(f"Manifest rows: {row_count}; long clips excluded via {args.long_clip_column}: {excluded_count}.")
    print(f"Missing transcripts: {missing_transcripts}; empty strings used.", file=sys.stderr)


if __name__ == "__main__":
    main()

# Data preparation: import and export interface

[中文](data_import.md) | English · [User guide](README_en.md)

Create one task for each eligible customer clip from the batch's `manifest.parquet`. Transcripts come directly
from its `text` column; no separate transcript directory is needed. Eligible clips are shuffled with a fixed seed
and split into non-overlapping JSON files containing at most 1,000 tasks each.
The project owner has already configured the local service's document root. The generic
paths below explain the data interface, without installation or startup steps.

## 1. Prepare a batch directory

Place each batch in a **strict subdirectory** of the document root, for example:

```text
/absolute/path/to/dataset/        # Configured service document root
└── batch/
    ├── audio/                   # clip_0001.wav, clip_0002.mp3 …
    ├── manifest.parquet         # Clip IDs, audio files, transcripts, long-clip flags, etc.
    └── tasks/                   # Generated task files; Local Files Source path
        ├── tasks_001.json       # At most 1,000 tasks
        ├── tasks_002.json       # Up to 1,000 different tasks
        └── …
```

The Local Files Source path must be a strict subdirectory of the document root; **it cannot equal the document root**.
The `batch/tasks/` directory above satisfies this requirement.

Run in a Python environment with `pyarrow` installed (already included in the project's `emotion_recognition_llm`
environment). For standalone use, install `python3 -m pip install -r label_studio/requirements.txt` from the repository
root in a Python 3.11 environment first. Then generate tasks from the repository root:

```bash
python3 label_studio/prepare_tasks.py \
  --manifest /absolute/path/to/dataset/batch/manifest.parquet \
  --document-root /absolute/path/to/dataset \
  --output-dir /absolute/path/to/dataset/batch/tasks \
  --batch-size 1000 \
  --seed 42
```

`--output-dir` replaces the old single-file `--output` argument. `--batch-size` defaults to `1000` and `--seed` to
`42`, so both are optional. After filtering and validation, the script sorts by `clip_id`, shuffles once with that
seed, and slices the shuffled list into batches. Every eligible clip appears exactly once, with no overlap between
files. The final file may contain fewer tasks: 2,007 clips produce files of 1,000, 1,000, and 7 tasks.
The same eligible clip set, seed, and batch size produce the same assignments, even if manifest rows are reordered.
Changing the eligible clip set or seed may change assignments; once labeling starts, keep using the original files.

The default columns are below. Other metadata stays in the manifest and can be joined to exports using `clip_id`.

| Manifest column | Use | Override column name |
|---|---|---|
| `clip_id` | Existing stable ID, written to `data.clip_id`; integer IDs become strings, string IDs are preserved | `--clip-id-column` |
| `audio_clip_file` | Audio filename or filesystem path, converted to `data.audio` | `--audio-column` |
| `text` | Existing transcript, written to `data.transcript` | `--text-column` |
| `is_longer_than_10_seconds` | `true` excludes the row; `false` keeps it | `--long-clip-column` |

**Filtering uses only the manifest flag; the script does not measure audio duration.** The current column denotes
10 seconds, so this uses the 10-second flag, rather than the previously discussed 15-second rule. If the manifest
later provides a 15-second flag, select it with `--long-clip-column is_longer_than_15_seconds`. Flags can be booleans,
numeric `1/0`, or strings `true/false` and `1/0`; strings ignore case and surrounding whitespace. Missing or invalid
flags cause an error rather than being guessed.

Audio paths in `audio_clip_file` resolve as follows:

- A bare filename, such as `clip_0001.wav`, is looked up in `audio/` beside the manifest. Override this directory with `--audio-dir`.
- A relative path containing directories, such as `audio/clip_0001.wav`, is relative to the manifest's batch directory.
- An absolute filesystem path is used directly, but must stay within the document root.

WAV, MP3, FLAC, OGG, and M4A are supported. The script reads only audio references in retained manifest rows;
it does not scan the audio directory or copy, delete, or modify audio. Unlisted or excluded files may remain in
`audio/`. Excluded rows do not require their audio files to exist. If every row is excluded, no batch JSON is written.
Blank, null, or NaN transcripts become empty strings and are counted; other non-text transcripts cause an error.
Retained clip IDs and resolved audio paths must each be unique. Missing columns, missing audio, and duplicate
records cause errors. The manifest, audio directory, and referenced audio must resolve within the document root;
symlinks pointing outside it are rejected. Output is written only after validation. If the output directory already
contains `tasks_*.json`, the script fails to prevent overwriting or mixing assignments from different runs. Use a new
output directory when regenerating batches. The summary reports the seed, task counts per file, and excluded rows.

The task structure within each JSON file is unchanged:

```json
[
  {
    "data": {
      "audio": "/data/local-files/?d=batch/audio/clip_0001.wav",
      "transcript": "Existing transcript for this clip.",
      "clip_id": "clip_0001"
    }
  }
]
```

The script URL-encodes special characters. The path after `?d=` is relative to the **document root**, not the Source
path, so it must retain `batch/` in this example. Do not use `file://` or a bare absolute local path.
Keep original call/turn/segment timestamps in the data preparation manifest and join them to annotation exports
using the stable `clip_id`. Company audio, transcripts, task JSON, and exports remain in authorized company
directories and are not written into this repository.

## 2. Import with Local Files / Tasks

Within the project, open **Settings → Cloud Storage → Add Source Storage → Local Files**:

1. Set Source path / Absolute local path to `/absolute/path/to/dataset/batch/tasks`, a strict subdirectory of the document root.
2. Click **Test Connection**, then **Next** after it succeeds.
3. Set **Import Method** to **Tasks** and the filename filter to `^tasks_001\.json$` to import only the first batch.
4. Click **Load Preview** and confirm that the preview reads tasks rather than treating the JSON as an audio file.
5. Click **Next → Save & Sync**, return to the data page, check the task count, and play one clip.

The public demo uses this Tasks synchronization method. In the screenshot, the filename is `import_tasks.json`, so
the actual filter is `import_tasks\.json$`. Use `^tasks_001\.json$` for the batch directory example above.

![Local Files wizard: Tasks, filename filter, and a successfully loaded preview](screenshots/04-import-settings.jpg)

Sync only the filtered task JSON. Do not also scan the audio directory using **Files** mode, which would bypass
the manifest filtering. Updating JSON does not automatically remove tasks that were imported previously.

## 3. Or upload JSON through Import

If the project owner has already configured local file access for the audio, the recommended incremental workflow
is **Import → Upload Files** on the project data page:

1. Upload only `tasks_001.json` and complete its labeling.
2. Upload `tasks_002.json` to the same project to continue; existing annotations are retained.
3. Continue as needed, stopping after a few thousand clips if sufficient. Remaining files do not need to be imported.

Uploading JSON imports task fields, not the referenced audio. Do not sync all batch files at once, which would add
all their tasks to the project. **Choose either Tasks Sync or Upload Files for each JSON, and import each file only
once.** Do not duplicate imports.

## 4. Read and normalize exports

First download and retain the complete native file using Label Studio **Export → JSON**. Join it to the manifest
using `data.clip_id`. In the native `result`, the entry with `from_name="decision"` and `type="choices"` stores the
single decision in `value.choices`: one of the 8 emotion values or the `bad_clip` rejection decision.
Bad clip is not an emotion category.

Use this project's normalization script to generate business data, retaining the native file:

```bash
python3 label_studio/normalize_export.py \
  --input /absolute/path/to/exports/label-studio.json \
  --output /absolute/path/to/exports/emotion-votes.json
```

The script outputs one record per annotation, **preserving every annotator's vote without aggregating or replacing
votes with a single final label**. It retains `clip_id`, `task_id`, `annotation_id`, `annotator_id`, `lead_time`,
`decision`, `status`, and `emotion`. When `completed_by` is a user object, only its ID is used; email addresses are
not included.

- For any of the 8 emotion decisions, including `unknown`: `status="accepted"`, and `emotion` is the corresponding emotion value.
- For a Bad clip decision: `status="bad_clip"`, and `emotion` is JSON `null`, for example:

```json
[
  {
    "clip_id": "clip_0001",
    "task_id": 1,
    "annotation_id": 1,
    "annotator_id": 1,
    "lead_time": 3.5,
    "decision": "bad_clip",
    "status": "bad_clip",
    "emotion": null
  }
]
```

`accepted` only means that the emotion branch was selected. It does not mean the vote has been adjudicated into a
gold label; `unknown` remains distinct. Select Unknown when the audio is usable but the emotion is uncertain.
Select Bad clip when the audio itself cannot be judged reliably; exclude these clips during downstream processing.

Cancelled annotations (`was_cancelled`) and tasks with no annotations are skipped, and their counts are reported.
Label Studio exports annotated tasks by default; unsubmitted tasks are not included by default.
Missing `clip_id` or `decision`, multiple decisions or selections, and unknown values cause errors. Output is written
only after all validation succeeds, and existing output files are never overwritten.

Official references: [Local storage](https://labelstud.io/guide/storage_local), [Choices](https://labelstud.io/tags/choices),
[read-only Text](https://labelstud.io/tags/text), and [export fields](https://labelstud.io/guide/export.html).

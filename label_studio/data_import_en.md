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

## 4. Server Label Studio: upload audio directly (no Local Files / GCS)

**When to use this**: only a company-hosted Label Studio Enterprise (LSE) is allowed, Local Files is not available,
and GCS cannot be set up in time. The project lead then uploads the audio files through **Import → Upload Files**.
`tasks_NNN.json` is **not imported**: it is the list of audio files to upload and the source of `transcript` and
`clip_id`.

**Why the template must be switched**: the with-transcript template has two data keys, `$audio` and `$transcript`,
and Label Studio refuses direct audio uploads for it:

> Your label config has more than one data key and direct file upload supports only one data key.
> To import data with multiple data keys, use a JSON or CSV file.

So upload with the audio-only template (only `$audio`), add transcript and `clip_id` with a script, then switch to
with-transcript.

| Step | Action | Check |
|---|---|---|
| 1 | Locally, run `link_upload_audio.py` to hard-link the audio of one `tasks_NNN.json` into `part_01/`, `part_02/`, … | Files per part |
| 2 | Create the LSE project with [config_audio_only.xml](config_audio_only.xml) | Preview shows only the audio and choices |
| 3 | Open **Import**; each time, select all files of one part folder and drag them in | Task count equals files uploaded |
| 4 | Run `attach_task_data.py` as a dry run (`write: false`) | No errors; `To update` equals the file count |
| 5 | Set `write: true` (`limit: 3`) to write 3 tasks | The Data Manager shows `transcript` and `clip_id` columns filled for those 3 |
| 6 | Set `limit: null` to write all | It prints `All tasks of this file are done` |
| 7 | In **Settings → Labeling Interface → Code**, switch to [config_with_transcript.xml](config_with_transcript.xml) | Open a task: the audio plays and the text matches |
| 8 | Only now let the annotators start | |

Run both scripts from the repository root with a Python that has [requirements.txt](requirements.txt) installed
(the `emotion_recognition_llm` environment already has it). They take no command-line arguments; their settings are
YAML files in `label_studio/config/`.

### 4.1 Build the upload folders (hard links)

`label_studio/config/link_upload_audio.yaml`:

```yaml
tasks_file: /absolute/path/to/dataset/batch/tasks/tasks_001.json
document_root: /absolute/path/to/dataset
upload_dir: /absolute/path/to/dataset/batch/audio/upload_001  # a new or empty folder
files_per_folder: 100
```

```bash
python3 label_studio/link_upload_audio.py
```

- The upload folder holds **hard links**, which use no extra disk space. Finder may still show the folder's full
  size, but disk usage does not grow, and `du -sh` on the whole batch folder counts each file once. The originals
  are not moved or changed, so the manifest and task JSON paths stay valid.
- After uploading, you may delete the upload folder; the originals are unaffected. **Do not edit files inside the
  upload folder**: edits would change the originals too.
- Hard links need the same disk as the audio. exFAT and some network shares do not support them; the script then
  stops, removes the links it created, and never falls back to copying.
- Each part holds at most `files_per_folder` files. Open-source Label Studio accepts 100 files per upload by default
  (`DATA_UPLOAD_MAX_NUMBER_FILES`); on LSE the admin's setting applies. Lower the value if an upload is refused.
- Label Studio replaces spaces with `_`, drops characters such as `#`, and adds an 8-character random prefix; the
  script handles this. File names must therefore be unique (ignoring case and dropped characters) and at most 70
  characters long, or the server shortens them.

### 4.2 Add transcript and clip_id

`label_studio/config/attach_task_data.yaml`:

```yaml
project_id: 1      # the number in the project address: https://<your-label-studio-host>/projects/<id>/data
tasks_file: /absolute/path/to/dataset/batch/tasks/tasks_001.json
write: false       # false = dry run (read and match only); true = write
limit: 3           # most tasks written per run; once checked, set null = all
```

The server address and token live only in environment variables, never in YAML (the repository is public):

```bash
export LABEL_STUDIO_URL="https://<your-label-studio-host>"
read -rs LABEL_STUDIO_API_KEY && export LABEL_STUDIO_API_KEY   # paste the token, press Enter: hidden, not in history
python3 label_studio/attach_task_data.py
```

- Where the token is: avatar (top right) → **Account & Settings** → **Legacy Token**. A Personal Access Token also
  works; the script exchanges it for a short-lived access token.
- Treat the token like a password: never put it in a file or send it to anyone. If it leaks, reset it on the same page.
- On HTTP 401, check the token; if the admin disabled legacy tokens, use a Personal Access Token.
- If the company network uses its own certificate authority, first set
  `export REQUESTS_CA_BUNDLE=/absolute/path/to/company-ca.pem`.
- The script prints counts, file names, and `clip_id` values only, never transcripts or the token.
- Problems such as an uploaded file with no match or a file uploaded twice stop the run before anything is written,
  and the tasks are listed. For a double upload, delete the extra task in the Data Manager and run again. Finished
  tasks are skipped, so re-running is safe.

### 4.3 Later batches in the same project

Only between batches, while nobody is labeling:

1. Switch the template back to `config_audio_only.xml`. Both templates have the same choices, so existing annotations stay.
2. Run `link_upload_audio.py` for `tasks_002.json` (with a new `upload_dir`, e.g. `upload_002`) and upload its parts.
3. Set `tasks_file` in `attach_task_data.yaml` to `tasks_002.json` and follow steps 4–6 of the table. Tasks of
   earlier batches appear as `other batches (ignored)` and are left unchanged.
4. Switch back to `config_with_transcript.xml`, then let the annotators continue.

## 5. Read and normalize exports

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

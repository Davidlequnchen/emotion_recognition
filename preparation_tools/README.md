# preparation_tools: turn segmenter for call-centre audio

This tool splits long customer turns into shorter pieces for emotion labelling. It reads one audio file and its
existing transcript per customer turn, grouped into calls. It finds when each word was spoken by forced alignment of
that transcript (torchaudio, English wav2vec 2.0), checks how quiet each pause between words is with Silero VAD, and
picks the cut points for the whole turn at once with dynamic programming: pieces close to 7 s, never outside 2–15 s,
cut at the best pauses. Audio and text are cut at the same place, so every word ends up in exactly one piece. Every
turn lands in one of four output groups, and every exported item knows its neighbours in the call. It runs fully
offline. The method and the reasons behind every number are in
[Data Preparation for Internal Call-Centre Audio](../emotion_recognition_benchmark/docs/data_preparation_call_centre_audio.md)
(§2–§4, §6).

Steps for each turn, call by call:

1. Decode the audio (WAV, MP3, FLAC) and make a mono 16 kHz copy for analysis. The source audio is kept for export.
2. Check the language of the transcript. Spanish or mixed turns go whole to `non_english/`, whatever their length.
3. English turns of 10 s or less go whole to `final/`. Edge trimming is off by default (`vad.trim_edges: false`);
   when it is on, long silence at the very start and end does not count towards those 10 s, and aligned long turns
   lose that edge silence (keeping 0.2 s, never into a word). Whole-turn exports are never trimmed.
4. Forced alignment: start, end and confidence for every word. If it fails, fails the quality check, or the
   transcript is empty, the turn goes whole to `alignment_issues/`.
5. Score every gap between words: `s = 0.5 p + 0.3 q + 0.2 v` (pause length relative to this speaker,
   punctuation, quietness).
6. Dynamic programming picks the cheapest set of cuts with every piece inside 2–15 s. If no set fits, the turn goes
   whole to `boundary_issues/`; otherwise its pieces go to `final/` (a 10–15 s turn may stay whole when no pause is
   worth a cut).
7. Cut in the middle of the quiet part of each chosen gap (frames within 0.1 of the gap's lowest VAD probability) and write the clips, transcripts and the review file.

After the last turn of a call, every exported item of that call (all four groups together, in call order) gets
its order in the call and links to the previous and next item; then the call's rows are appended to the group
manifests and to `call_timeline.csv`.

## Install

Prerequisite: the target Python 3.11 environment already has torch 2.7.1 and torchaudio 2.7.1. On the Linux server
these come from [`emotion_recognition_benchmark/dependencies/torch-cu118-cp311-linux_x86_64/`](../emotion_recognition_benchmark/dependencies/torch-cu118-cp311-linux_x86_64/README.md)
(`python install.py` there); on the Mac from [`requirements-llm-mac.txt`](../emotion_recognition_benchmark/requirements-llm-mac.txt).

Two steps, in the target environment (the first needs access to GitHub, the second no internet):

```bash
python preparation_tools/download.py   # 1. automatic: this platform's wheels + the checkpoint (about 391 MB)
python preparation_tools/install.py    # 2. offline install, no internet needed
```

`download.py` takes everything from the GitHub Release
[`preparation-tools-segmenter-v1`](https://github.com/Davidlequnchen/emotion_recognition/releases/tag/preparation-tools-segmenter-v1)
of this repository, which holds the files unchanged from PyPI and download.pytorch.org. It detects the platform
(Linux x86_64 or macOS arm64) and fetches `dependencies/any/`, that platform's wheel folder and the checkpoint into
`model_weights/checkpoints/`. Every file is checked against the size and SHA-256 in the manifests; verified files
are reused, an interrupted download resumes from its `.download` file on the next run, and a file gets its final
name only after it matches. It uses the operating system's certificates when
[truststore](https://pypi.org/project/truststore/) is installed, otherwise Python's; behind a company proxy with
its own root CA, set `CA_BUNDLE` at the top of the script (or `SSL_CERT_FILE`).

The repository is private, so the Release needs a GitHub token with read access (a fine-grained token limited to
this repository with *Contents: Read-only*, or a classic token with `repo`):

```bash
read -s GH_TOKEN && export GH_TOKEN    # paste the token; it is not echoed or kept in shell history
python preparation_tools/download.py   # or log in once with `gh auth login` instead of a token
```

Without `GH_TOKEN`/`GITHUB_TOKEN`, `download.py` asks the GitHub CLI (`gh auth token`) if it is installed, and
otherwise downloads anonymously (enough only while the repository is public). With a token it reads the asset list
through the GitHub REST API and downloads each asset from there. The token goes only to `api.github.com`, is never
printed, and is not passed on when GitHub redirects the download to its file storage host.

Other settings at the top of `download.py`: `PLATFORM` (`"cp311-linux_x86_64"`, `"cp311-macosx_arm64"` or
`"all"`, to download for another machine and copy the whole `preparation_tools/` folder there), `SOURCE =
"upstream"` (PyPI and download.pytorch.org instead of the Release) and `GET_WEIGHTS`. `install.py` checks Python,
platform and the torch prerequisites, verifies every file before installing anything, installs the wheels one by
one with `pip install --no-index --no-deps`, runs `pip check`, and finishes with a smoke test (Silero VAD, the
wav2vec2 checkpoint, WAV/FLAC/MP3 decoding). The scripts have their few settings as constants at the top; there
are no command-line options.

| Folder | Content | In Git |
|---|---|---|
| `dependencies/any/` | silero-vad 6.2.3, packaging 26.3, pycparser 3.12 (11.5 MB) | `manifest.json` only; wheels in the Release |
| `dependencies/cp311-linux_x86_64/` | cffi 2.1.1, soundfile 0.14.0, PyYAML 6.0.3 for the Linux server (2.3 MB) | no; Release |
| `dependencies/cp311-macosx_arm64/` | the same three for Apple Silicon Macs (1.5 MB) | no; Release |
| `model_weights/checkpoints/` | `wav2vec2_fairseq_base_ls960_asr_ls960.pth` (377.7 MB) | `manifest.json` only; checkpoint in the Release |
| `local/` | your input turns and outputs | never |

numpy and typing-extensions come with the torch installation. onnxruntime and torchcodec are not needed.

## Publishing (maintainers)

[`publish.py`](publish.py) uploads every wheel (all platforms), the checkpoint, `SHA256SUMS`, both manifests (as
`dependencies-manifest.json` and `model_weights-manifest.json`) and `LICENSES.md` to the Release with the GitHub
CLI. It needs `gh auth login` with write access and all files present locally (`download.py` with
`PLATFORM = "all"`; `SOURCE = "upstream"` before the Release exists).

```bash
python preparation_tools/publish.py    # set DRY_RUN = True at the top first to only print the plan
```

It verifies every local file before touching GitHub. If the Release does not exist, it prints the
`gh release create` command and creates it as a draft (tag on `main`, not marked latest). Assets are immutable:
an asset that already exists must have the same size and SHA-256, or the script stops before uploading
anything; a changed file needs a new tag (`RELEASE_TAG` in `download.py`, which `publish.py` also uses). Files
above 1.5 GiB would be uploaded as `<name>.part000`, `.part001`, … and joined again by `download.py`. The metadata
goes last and is the only thing that may be replaced. Finally every asset's size and SHA-256 are compared with
GitHub's digests and the draft is published. Rerunning reuses what is already verified.

## Input

Put the turns in `preparation_tools/local/input/turns/` (paths can be changed in the YAML):

```text
local/input/turns/
  <turn_id>.wav | .mp3 | .flac   one customer turn (agent turns already removed)
  <turn_id>.txt                  its existing transcript, UTF-8
  <call>/<turn_id>.wav + .txt    or: one subfolder per call
local/input/turns.csv            optional: turn_id, call_id, turn_index, turn_start_in_call_s[, customer_id]
```

The call of a turn comes from `turns.csv` (preferred); otherwise from its subfolder (call id = folder name);
otherwise the turn is its own call, and the run prints a warning that context links cannot be built for it. Turn
ids are the audio file names without extension and must be unique across all calls; only one level of subfolders is
read. Call ids are made safe for folder names (anything other than letters, digits, `.`, `_` and `-` becomes `_`).
Within a call, turns are ordered by `turn_index`, else by `turn_start_in_call_s`, else by file name. Without
`turn_start_in_call_s` the call-time fields stay empty and `call_time_status` is `unresolved`; call times are never
rebuilt by placing turns back to back.

## Run

```bash
python preparation_tools/segment_turns.py
```

All settings are in [`config/segment_turns.yaml`](config/segment_turns.yaml). The script prints the versions,
the model loading, how the turns were grouped into calls, one line per turn (group, status, segment lengths and
the reason when it is not `final`) and a run summary (turns and files per group, length distribution of the final
segments, `low_speech_ratio` count, failures). The output folder must be new or empty, so runs never mix.

## Checking one turn by eye: Segment Explorer

Before a large run, check the segmenter on single long files in the browser:

```bash
python preparation_tools/explorer/app.py   # local server on 127.0.0.1, opens the page
```

Choose an audio file and its transcript and click "Run segmenter". The page runs `segment_turn` with the YAML
settings and shows, on one time axis, the waveform with the segments and cuts, the VAD curve, the aligned words
(coloured by confidence), the gap scores (p, q, v, s on hover) and a segment table with play buttons. The settings
of the cut decision can be changed and re-run instantly (decoding, VAD and alignment are cached; the YAML is never
written, a "copy as YAML" snippet is offered), and the result can be exported as one self-contained offline HTML
file. **The exported HTML contains the audio and the transcript: share internal call data only through approved
internal channels.** Details: [`explorer/README.md`](explorer/README.md).

## Outputs

```text
local/output/batch/
  final/                                  English turns: <= 10 s whole, longer ones cut (2-15 s, near 7 s)
    manifest.csv                          one row per file in this group
    call_0001/
      call_0001_t000_s00.wav              <call_id>_t<turn order>_s<segment index>: name order = time order
      call_0001_t000_s00.txt              the original words of that segment, spelling and punctuation kept
      call_0001_t000_s01.wav / .txt
      call_0001_t002_s00.wav / .txt
  non_english/                            Spanish or mixed turns of any length, whole
    manifest.csv
    call_0001/call_0001_t001_s00.wav / .txt
  alignment_issues/                       long English turns: alignment failed or empty transcript, whole
    manifest.csv
  boundary_issues/                        long English turns: no partition within 2-15 s, whole
    manifest.csv
  review/<turn_id>.json                   every turn of every group: words with times and confidence, every gap
                                          with p, q, v, s, the chosen cuts and the dynamic-programming cost
  call_timeline.csv                       every exported item of every group, by call and order, with the links
  failures.csv                            only when a turn failed: turn_id, call_id and error
```

Clips are WAV at the source sample rate (PCM; MP3 sources as 16-bit). Every group folder has the same layout and a
`manifest.csv`, empty except for its header when nothing landed there. All paths in every CSV are relative to the
output folder. The manifests and the timeline are written call by call, so a crash keeps every finished call.

| Group | What goes there | Used for |
|---|---|---|
| `final/` | English turns ≤ 10 s exported whole, and longer English turns cut by the dynamic programme | Human labelling, benchmarking and fine-tuning |
| `non_english/` | Every turn detected as Spanish or mixed, any length, exported whole | Kept, not labelled |
| `alignment_issues/` | Long English turns whose alignment failed or failed the check, or with an empty transcript, exported whole | Kept, not labelled |
| `boundary_issues/` | Long English turns with no partition inside 2–15 s, exported whole | Kept, not labelled |

Only `final/` is used for labelling, benchmarking and fine-tuning. The other three groups are kept because
unlabelled audio — long, non-English or hard to align — can still serve later self-supervised continued
pre-training (e.g. of an emotion2vec-style encoder, or of a Whisper encoder using the transcripts); no such step is
implemented.

A turn that fails (unreadable audio, missing transcript, export error) is skipped and its partly written files
are removed, so the outputs never hold half a turn. The rest of its call is still exported and the links skip the
failed turn; `failures.csv` and the summary list the failures.

**Manifest columns** (`<group>/manifest.csv`, the same in every group):

| Columns | Meaning |
|---|---|
| `segment_id`, `group`, `reason` | Identity; the group; why the item is not in `final/` (empty in `final/`), e.g. `language: es`, `alignment: mean confidence 0.41 < 0.5`, `alignment: empty transcript`, `no partition within 2-15 s` |
| `call_id`, `customer_id`, `turn_id`, `turn_index`, `turn_order` | The call (as given, before making it folder-safe), the turn, `turn_index` from `turns.csv`, and the turn's position in the call (0, 1, 2, …) used in the name |
| `segment_index`, `segment_count` | Position of the segment in its turn and number of segments of the turn |
| `order_in_call`, `prev_segment_id`, `prev_group`, `next_segment_id`, `next_group` | Position among all exported items of the call (all groups) and its neighbours; empty at the call edges |
| `gap_to_prev_s`, `gap_to_next_s` | Call time to the neighbour: 0 inside a turn; between turns it usually holds the agent's turn, which is not in the data. Empty without call times |
| `audio_path`, `transcript_path`, `review_path` | Files of this item |
| `source_audio`, `source_transcript` (+ `_sha256`), `source_sample_rate`, `source_channels`, `clip_subtype` | Provenance |
| `turn_start_in_call_s`, `segment_start_in_turn_s`, `segment_end_in_turn_s`, `segment_start_in_call_s`, `segment_end_in_call_s`, `call_time_status` | Times in the turn and in the call |
| `duration_s`, `speech_duration_s`, `speech_ratio`, `low_speech_ratio` | Length, VAD speech time, and the flag (1 when speech time / duration is below 0.4; nothing was deleted) |
| `language`, `segmentation_status` | `en`, `es` or `mixed`; `whole_short`, `whole_dp`, `cut` (final), `whole_language`, `whole_alignment`, `whole_no_text`, `whole_boundary` |
| `alignment_model`, `alignment_status`, `alignment_mean_confidence`, `alignment_unaligned_share` | Alignment result of the turn |
| `start_boundary_score`, `end_boundary_score`, `word_count` | Boundary scores at both ends (empty at the turn start and end) and words in the segment |
| `segmentation_version` | The name in the YAML plus a hash of all settings that change the cuts |

**`call_timeline.csv`** is the single file for context experiments (like the dialogue context in DailyTalk or
MELD): one row per exported item of every group, sorted by call and `order_in_call`, with `call_id`,
`order_in_call`, `segment_id`, `group`, `reason`, `turn_id`, `turn_order`, `segment_index`, `segment_count`,
`audio_path`, `transcript_path`, `language`, `duration_s`, the times in the turn and in the call,
`call_time_status`, the two gaps and the four link columns. A model can be given, for example, the previous one or
two items of the same call as context, whatever group they are in.

## Settings

| YAML key | Start | What it does |
|---|---|---|
| `paths.groups` | `final`, `non_english`, `alignment_issues`, `boundary_issues` | Folder name of each output group |
| `segmentation.keep_whole_max_s` | 10 | English turns up to this length are never cut |
| `segmentation.target_s` | 7 | Length cost `((d − 7)/7)²` pulls every segment towards 7 s |
| `segmentation.min_s`, `max_s` | 2, 15 | Hard window: no segment outside it; if no partition fits, the turn goes whole to `boundary_issues/` |
| `segmentation.cut_weight_lambda` | 1.0 | Price of a cut: `λ (1 − s)` |
| `scoring.weight_pause` / `weight_text` / `weight_silence` | 0.5 / 0.3 / 0.2 | Boundary score; `0.7 p + 0.3 v` when the transcript has no punctuation |
| `scoring.typical_pause_*` | 0.1 s, 3, 0.3 s | Typical pause = median of gaps > 0.1 s, or 0.3 s with fewer than 3 |
| `alignment.word_min_confidence` | 0.3 | No cut next to a weaker word (or next to an unspelled word) |
| `alignment.turn_min_mean_confidence`, `turn_max_unaligned_share` | 0.5, 0.4 | Turn-level alignment check |
| `vad.trim_edges`, `trim_min_silence_s`, `trim_margin_s` | off, 1.0 s, 0.2 s | Optional edge trimming (step 3): only for the 10 s keep-whole test and the edges of aligned long turns; whole-turn exports are never trimmed |
| `vad.low_speech_ratio` | 0.4 | Threshold of the `low_speech_ratio` flag |
| `language.alignment_models` | `en: wav2vec2_asr_base_960h` | Which model aligns which language; turns in any other language go to `non_english/` |
| `device` | cpu | `cuda` or `mps` for the alignment model; VAD stays on the CPU |

Why 2–15 s: 2 s is the upstream minimum turn length. 15 s is Vox-Profile Whisper's fixed window, and below
emotion2vec+'s 20.48 s limit on the Mac GPU and the 30 s windows of the Whisper-encoder models, Qwen2-Audio and
Gemma 4. The length cost still pulls segments towards 7 s, so the wide hard window costs little quality and needs no
second pass: on the synthetic DailyTalk calls, the old 3–12 s window gives exactly the same cuts.

The alignment thresholds are **provisional**. On synthetic DailyTalk turns the mean word confidence per turn was
0.90–0.95 with the correct transcript, 0.88–0.94 for the same audio as 8 kHz μ-law, 0.63–0.74 with 30% of the
words replaced and 0.27–0.36 with a wrong transcript; the thresholds sit between the last two. Set them again
from the score distribution on real calls in the pilot.

### Text handling

The aligner only knows A–Z and the apostrophe, so it works on a copy of the text: upper case, accents removed,
numbers written as words (`$50` → FIFTY DOLLARS, `20%`, `21st`, `3:05`, years such as 1999; `24/7` stays spelled
as TWENTY FOUR SEVEN). Tokens that cannot be spelled reliably match any speech through a wildcard and are never
next to a cut: placeholders and annotations (`[NAME]`, `***`, `<unk>`, multi-word ones such as
`(inaudible speech)`), numbers with several readings (`-50`, `1/2`, `10-15`, `555-1234`), digit strings of 5+
digits or with a leading zero, codes like `A12B`, and 4-digit numbers outside 1100–2099. The output transcripts
are exact slices of the original text.

## Spanish turns

The calls are English with a minority in Spanish. A small built-in check counts Spanish and English function
words and Spanish accents in the transcript. Every Spanish or mixed turn, short or long, goes whole to
`non_english/` with `language = es` or `mixed`; the check runs before the 10 s rule. Long ones could not be cut
anyway, because the English model must not align Spanish text.

A Spanish model can be added later through configuration, without changing the cutting logic: an entry in
`model_weights/manifest.json`, one line under `language.alignment_models` (`es: ...`), and a Spanish text
normaliser. Spanish turns would then be handled like English ones and go to `final/`. The candidate is
`jonatasgrosman/wav2vec2-large-xlsr-53-spanish` (Apache-2.0, about 315M parameters, 1.3 GB); it needs a converter
to a torchaudio state dict, its own letter set (including á, é, í, ó, ú, ñ) and Spanish number words.
Mixed-language turns would still go whole to `non_english/`.

## Tests

```bash
cd preparation_tools
python -m unittest discover -s tests       # offline, about 10 s, no models
python tests/integration_dailytalk.py      # full pipeline on synthetic calls from the local DailyTalk copy
python tests/integration_explorer.py       # Segment Explorer server with the real models on those turns
```

`test_explorer.py` covers the Segment Explorer with the same fakes (settings, cache, page data, export, server);
`integration_explorer.py` is described in [`explorer/README.md`](explorer/README.md#files-and-tests).

The unit tests reproduce the worked example of the method document (cuts at 7.4, 15.3, 23.0 and 29.6 s;
totals 0.68 against 0.97) and compare the dynamic programme with all 116,674 partitions allowed by the 2–15 s
window. They cover the fast-speaker case, the routing of turns to each of the four groups (a short Spanish turn goes
to `non_english/`), calls from `turns.csv` and from subfolders, the turn order, segment ids that sort in time order,
context links and gaps across groups, the scoring, the text handling, the language check, the word assignment and
the export (a failing turn is skipped cleanly and the rest of its call stays linked). Models are replaced by small
fakes in `tests/fakes.py`. `test_release.py` runs `download.py` and `publish.py` against a local stand-in for
GitHub (download URLs, REST API and storage redirect) and a fake `gh`, with tiny files: fresh download, resume,
hash mismatch, reuse, the token path (the token never reaches the storage host or the output), split parts,
immutable assets and the dry run.

The integration script builds five synthetic calls, one per DailyTalk dialogue: a long English turn of 25–35 s
made from consecutive utterances of one speaker with known pauses, a Spanish-text turn and a short English turn,
with call times in `turns.csv`. It runs the pipeline and checks the groups, lengths, word coverage, exact clip
slices, how close the cuts land to the joins, and the links and gaps of every call. It writes only to
`local/integration/`. Last run (2026-11-13, CPU): 15 turns in 5 calls; `final/` 10 turns and 26 files (5 long
turns cut into 21 segments of 5.1–10.9 s, median 8.4 s; 5 short turns whole), `non_english/` 5 turns (3 of them
longer than 10 s), no alignment or boundary issues; 8 of the 16 cuts fall at an utterance join, 7 of the other 8 at
a sentence end with a real pause inside an utterance; the links and gaps of all 5 calls are correct.

## Licences

All components allow internal commercial use; see [LICENSES.md](LICENSES.md). torchaudio's `MMS_FA` and the
VoxPopuli models are not used (CC BY-NC).

## What it does not do

- It does not diarize, transcribe or separate speakers. Agent speech inside a customer turn (mono recording)
  stays in the audio; it only lowers the scores and confidences around it.
- It does not cut at emotion changes, and it does not delete pauses, sighs or laughter inside a turn.
- It does not align Spanish or mixed-language turns yet.
- It does not import anything into Label Studio; see [`label_studio/`](../label_studio/README.md).
- It does not recover missing call offsets or check how upstream extraction made the turn files. MP3 decoding
  was checked only with files written by the same library (no sample offset); check MP3s from the call system
  on a few examples.
- Its thresholds are not tuned on real calls yet.

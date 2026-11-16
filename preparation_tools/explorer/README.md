# Segment Explorer

A small local web page for checking the turn segmenter by eye and ear on **one** audio file and its transcript,
before running [`segment_turns.py`](../segment_turns.py) on many turns. It runs the real pipeline
(`segmenter.pipeline.segment_turn`, the same function the batch script calls) with the settings of
[`config/segment_turns.yaml`](../config/segment_turns.yaml) and shows every intermediate step on one time axis:
waveform, VAD, aligned words, gap scores, the chosen cuts and the segments, with playback.

It needs nothing beyond the segmenter's own environment (see the [main README](../README.md#install)): the server is
Python standard library only, the page is plain HTML, CSS and JavaScript, and nothing is loaded from the internet.

## Start

```bash
python preparation_tools/explorer/app.py
```

This starts a server on `127.0.0.1` (this computer only, port 8765 or the next free one) and opens the page in
the default browser. Stop it with Ctrl+C. Settings are constants at the top of [`app.py`](app.py): `PORT`,
`OPEN_BROWSER`, `DISPLAY_ALIGNMENT`, `KEEP_UPLOADS` (uploads kept in memory, default 3), `MAX_UPLOAD_MB` and
`MIN_FREE_MEMORY` (on macOS the models are not loaded when less than this share of memory is free, default 25%;
0 switches the check off). There are no command-line options.

## Steps

1. **Audio file**: choose or drop one WAV, MP3 or FLAC file (one customer turn, any length).
2. **Transcript**: choose or drop its `.txt` transcript (UTF-8), as it would sit next to the audio for the batch.
3. Click **Run segmenter**. The first run loads Silero VAD and the alignment model (a few seconds), then decodes
   the audio, runs VAD and forced alignment; the page shows a "Working…" line with the elapsed time.

The page then shows what the pipeline decided: the group (`final`, `non_english`, `alignment_issues`,
`boundary_issues`), the status and the reason, in one plain sentence at the top. Short English turns
(≤ `keep_whole_max_s`) and Spanish or mixed turns are kept whole by the pipeline without alignment; for these the
page still aligns the words and scores the gaps **for display only** (a yellow note says so, the tracks are marked
"display only", and no cut is shown). This extra run never changes the group or the segments. For non-English
text the English model is used, so those word times are rough. Set `DISPLAY_ALIGNMENT = False` to switch it off.

## What the page shows

**Summary**: file names and audio format, duration, language (with the English and Spanish cue-word counts),
group and reason, number and lengths of the segments, alignment mean confidence (and the share of weak or
unspelled words), the typical pause of this speaker, the dynamic programme's total cost split into the length part
and the cut part, and the settings version (the same `segmentation_version` the batch writes into its manifests).

**Tracks**, all on one shared time axis:

| Track | What it shows |
|---|---|
| Waveform | The whole file (peaks computed by the server). Shaded bands are the segments `s00`, `s01`, …; dashed lines are the cuts. |
| VAD speech probability | Silero VAD, one value per 32 ms frame, from 0 (quiet) to 1 (speech); the dashed line is the speech threshold (0.5) used for speech time and the low-speech flag. |
| Words | Every transcript word where forced alignment put it, coloured by confidence: green ≥ 0.7, amber between `word_min_confidence` (0.3) and 0.7, red with `!` below 0.3 (weak). Violet with `*` is a token the aligner cannot spell (e.g. `[NAME]`, long numbers), matched by a wildcard. No cut may touch a weak or unspelled word. |
| Gap scores s | One bar per pause between two words: width = pause length, height = boundary score s. Blue with a marker: chosen cut. Grey: allowed but not chosen. Red outline with ×: not allowed (next to a weak or unspelled word). Hover a bar to see the pause length and p, q, v, s. |

The boundary score of a pause combines three numbers between 0 and 1:

- **p, pause length**: how long the pause is compared with this speaker's typical pause:
  p = gap / (2 × typical pause), at most 1. A pause twice the typical one or longer gives 1. The typical pause is
  the median of the turn's pauses longer than 0.1 s (0.3 s when there are fewer than 3).
- **q, punctuation**: from the transcript: 1 after a sentence end (`.` `?` `!`), 0.5 after a comma, semicolon,
  colon or dash, 0 otherwise.
- **v, quietness**: 1 minus the mean VAD speech probability inside the pause: 1 is silent, 0 is speech.
- **s = 0.5 p + 0.3 q + 0.2 v** (0.7 p + 0.3 v when the transcript has no punctuation at all). A higher s is a
  better place to cut.

The dynamic programme then picks, for the whole turn at once, the set of cuts with the lowest total cost
Σ ((d − 7)/7)² + λ Σ (1 − s) with every piece d inside 2–15 s. Each cut goes to the middle of the quiet part of
its pause (VAD frames within 0.1 of the pause's lowest speech probability, the one nearest the pause centre).

**Segments table**: index, start–end, duration, text, boundary score at both ends, speech ratio (VAD speech time /
duration, with a "low speech" badge below 0.4) and word count. The play button of a row plays exactly that range;
clicking a row zooms to it.

**Playback and zoom**: the round button (or the space bar) plays and pauses; click a track to move the playhead;
drag across the tracks to zoom into a range; double-click, Esc or "Show all" to see the whole file again. When
zoomed in, the view follows the playhead.

## Trying other settings

Open **Settings (from segment_turns.yaml)**. It lists `keep_whole_max_s`, `target_s`, `min_s`, `max_s`,
`cut_weight_lambda`, the three score weights and the two no-punctuation weights with their YAML values. Edit them
and click **Re-run with these settings**. Changed values are marked with their YAML value next to them, and the
settings version in the summary changes. Decoding, VAD and alignment are cached per uploaded file, so a re-run
only redoes the language check, the gap scores and the dynamic programme and is instant (a changed
`keep_whole_max_s` that newly needs an alignment runs it once). "Reset to YAML values" restores the YAML values.

The YAML file is never written. **Copy as YAML** copies the changed values of the result shown, as lines to paste
into `config/segment_turns.yaml` by hand, e.g.

```yaml
segmentation:
  target_s: 6  # YAML: 7
```

## Export

**Export static HTML** downloads one self-contained file (`segment_explorer_<audio name>.html`) with the same
view of the result shown: inline CSS and JavaScript, the result as JSON (including the settings used, the
settings version and the changed values) and the original audio as a base64 data URI. It opens offline in any
current browser, with working playback, segment play buttons, zoom and tooltips, but without the upload, run and
re-run controls. Its Content-Security-Policy blocks every network request. The file is about 1.35 × the audio
size plus the result data.

> **Privacy:** the exported HTML contains the audio and the transcript. Share internal call data only through
> approved internal channels.

## Privacy and security

- The server listens on `127.0.0.1` only and answers only requests for its own host name and from its own page
  (Host, Origin and Sec-Fetch-Site checks), so other web pages cannot use it.
- Uploads are kept in memory (the last `KEEP_UPLOADS` files). For decoding, the audio is written to
  `preparation_tools/local/explorer_tmp/` (git-ignored) and deleted right after; the folder is emptied at start
  and removed when the server stops.
- The terminal shows the steps (upload size and format, settings version, group, segment lengths, timings) but
  never transcript text.

## Files and tests

| File | Content |
|---|---|
| [`app.py`](app.py) | The local server and its settings |
| [`core.py`](core.py) | Settings changes, cache, re-run, waveform peaks, page data and export (no server code) |
| [`page.html`](page.html) | The page; the export is this file with the result and the audio embedded |

`tests/test_explorer.py` (part of `python -m unittest discover -s tests`, fake models) covers the settings
changes and the YAML snippet, waveform peaks, the re-run from the cache (VAD and alignment run once), display-only
alignment for short and Spanish turns, the page data, the export (no live controls, no external URLs, no network,
escaped text) and the server end to end, including its guards. `tests/integration_explorer.py` runs the server
with the real models on the synthetic DailyTalk turns of `tests/integration_dailytalk.py` (run that first), checks
that the cuts match the batch run, re-runs from the cache and writes example exports to
`local/explorer_examples/`.

## Limits

- One turn at a time; it does not write clips, manifests or review files (use `segment_turns.py` for that).
- Only the ten settings above can be changed on the page; everything else comes from the YAML.
- Segment playback uses the browser's audio element and stops within one display frame (about 20 ms) of the
  segment end; seeking in MP3 files depends on the browser.
- Very long files make large pages: the waveform is capped at 200,000 bins and the VAD curve has one value per
  32 ms.

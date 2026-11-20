# Presentation: turn segmentation

`turn_segmentation.pptx`: 36 slides in the slide show (about 30–35 minutes) plus a hidden appendix of 6 slides
(settings, offline install and licences, references), speaker notes on every slide. It explains the turn segmenter in
`preparation_tools/`: why long customer turns are cut, the full procedure from the call to the clips, forced
alignment, Silero VAD, the boundary score and why it combines three cues, the whole-turn dynamic programme with a
worked example, routing to four output folders, per-call grouping for conversation context, and validation on
synthetic DailyTalk calls (two slides with playable audio, a comparison with four other ways of cutting, and a check of
how much the score weights matter).

Style follows `emotion_recognition_benchmark/presentation/` (pptxgenjs, same colours, title layout, notes and play buttons), with
stricter readability rules: body text at least 20 pt, tables 18–20 pt, diagram labels and figure text at least 16 pt
(citations and page numbers 14 pt), no grey text, figures drawn at their size on the slide. `build.js` stops if a
font is below these limits.

## Data and licences

No call data is used. Examples are the illustrative turn of the method document and synthetic calls built from the
public DailyTalk dataset (Lee et al., ICASSP 2023, CC BY-SA 4.0). Each synthetic turn joins consecutive utterances of
one DailyTalk speaker with inserted pauses, so it is not an original continuous recording.

The two listening slides embed 11 MP3 clips (two whole synthetic turns and their 9 exported segments, 16 kHz mono,
64 kbps, 1.15 MB). They and the figures made from DailyTalk audio and text (`vad_silero`, `compare_methods`,
`gap_zoom`, `listen_*`) are adaptations (joined and cut) of DailyTalk and are licensed under CC BY-SA 4.0. **Anyone
who shares the deck must keep the attribution on those slides** ("Audio: DailyTalk (Lee et al., ICASSP 2023),
CC BY-SA 4.0; clips adapted (joined and cut)") and share these parts under the same licence. No other audio is embedded.

## Rebuild

```bash
cd preparation_tools/presentation
npm install                               # pptxgenjs and jszip
python scripts/make_figures.py            # schematic figures + assets/worked_example.json (numpy, matplotlib)
python scripts/make_data_figures.py       # figures and numbers from the integration run (see below)
python scripts/sensitivity.py             # weight sensitivity: assets/sensitivity.json, fig/sensitivity.png
python scripts/make_audio.py              # assets/audio/*.mp3 + manifest.json (ffmpeg)
node build.js                             # writes turn_segmentation.pptx (hidden appendix slides included)
python fix_audio.py turn_segmentation.pptx   # audio plays on click; unique shape ids (standard library only)
./render.sh                               # optional: render/<deck>.pdf and render/p-NN.png, hidden slides included
```

The data scripts read the git-ignored outputs of `python tests/integration_dailytalk.py` in `../local/integration/`
and the local DailyTalk copy in `emotion_recognition_benchmark/data/dailytalk/`: they run the integration script's own
`make_inputs()` into a temporary folder to recover the utterance joins and check that every rebuilt transcript equals
the saved input. `make_data_figures.py` needs the segmenter's environment (torch, torchaudio, silero-vad, soundfile,
PyYAML); only Silero VAD runs, the alignment model is not loaded. `sensitivity.py` and the method comparison run the
dynamic programme in plain Python on the stored review JSONs. All outputs are kept in this folder, so `node build.js`
works without the data. A missing figure becomes a dashed placeholder and is reported by `build.js`.

`render.sh` exports hidden slides too (LibreOffice skips them in a PDF unless asked) and waits while free memory is
below 25% (macOS), because LibreOffice needs about 1 GB. `sheet.py` tiles rendered pages (`python sheet.py 1 6 2`).
LibreOffice shows the play buttons but not the audio; test playback in PowerPoint's slide show.

## Files

| File | Content |
|---|---|
| `build.js` | builds the deck, checks font sizes, marks appendix slides hidden, keeps one paragraph-properties element per paragraph (pptxgenjs repeats it for multi-run text) |
| `lib.js` | style and layout helpers (adapted from `emotion_recognition_benchmark/presentation/lib.js`): slide frame, stage strip, appendix numbering (A1…), bullets, boxes, arrows, tables, citations, notes, images, audio buttons |
| `refs.js` | the references (short label, full entry, link); all checked on 2026-11-13 |
| `s_intro.js` | title, problem statement, full procedure, rules, building blocks, pipeline |
| `s_align.js` | prepare, language check, forced alignment (concept, tool choice, CTC), text normalisation, quality checks |
| `s_vad.js` | Silero VAD: model, its own segmentation, its fragments, how the segmenter uses it |
| `s_dp.js` | boundary score, why three cues, relative pause, cost, DP algorithm, worked example, routing |
| `s_results.js` | outputs, call context, validation, listening slides, method comparison, weight sensitivity, limitations, hidden appendix |
| `fix_audio.py` | copied from `emotion_recognition_benchmark/presentation/`: audio plays on click |
| `img/` | play-button covers (from the SER deck) |
| `scripts/style.py` | figure style (deck colours, Calibri when available, 300 dpi) |
| `scripts/make_figures.py` | `ctc_trellis` (illustrative scores, true best path), `pause_score`, `length_cost`, `worked_example`, `confidence` |
| `scripts/make_data_figures.py` | `vad_silero`, `compare_methods`, `gap_zoom`, `listen_<turn>`, `assets/data_stats.json` |
| `scripts/sensitivity.py` | `sensitivity`, `assets/sensitivity.json` |
| `scripts/make_audio.py` | `assets/audio/` (MP3 clips and their manifest) |
| `assets/` | computed numbers, audio, and `integration_run.log` (console output of the integration run, paths replaced by `<repo>`) |
| `fig/` | the figures (PNG, 300 dpi, drawn at slide size) |
| `facts.md` | every number on the slides with its source, and the reference checks |
| `render.sh`, `sheet.py` | previews for checking |

# Presentation assets: scripts

The current 2026-11-17 refresh uses `build_dataset_refresh.mjs`, `build_model_refresh.mjs`,
`build_error_refresh.mjs`, `build_result_refresh.mjs`, `refresh_main_deck.py`,
`check_main_refresh.py` and `verify_main_imports.mjs`. The canonical output is
`../SER_evaluation.pptx` (92 slides), with only the two-page sentence excerpt under `updated/`.
See the parent README for the current build. The descriptions below document the older asset
generation and inspection pipeline; they do not define additional presentation deliverables.

The 2026-11-16 extension uses `prepare_inspections.py`, `build_semantic.mjs`, `build_inspections.mjs`,
`embed_inspection_audio.py`, `merge_inspections.py`, `check_inspections.py`, `verify_import_batches.mjs`
and `finalize_inspections.mjs`. It reads the complete
10-model × 16-dataset snapshot, verifies identical clip IDs and recomputed accuracy for all 160 runs,
and writes separate assets under `assets/inspections/`. See the parent README for the build sequence.
These extended assets are for the internal research copy, including datasets not cleared for public audio
redistribution. Their clips are not trimmed or loudness-normalized. The older scripts described below retain
their original six-dataset public selection and processing.

`prepare_lexical_example.py` and `build_lexical_example.mjs` supply the JL-Corpus continuation.
`insert_lexical_example.py` composes the 85-slide complete deck from private intermediates.
`consolidate_presentations.py` extracts both matched-sentence pages into one two-slide deck and publishes
the canonical `SER_evaluation.pptx` and `updated/Semantic_vs_Vocal_Emotion.pptx` only.
Temporary standalone pages, previews and validated staging packages remain in `.inspection-build/`.

These scripts rebuild everything in `../assets/` that the deck reads: the figures, `stats.json`, the audio
examples and `manifest.json`. They only read our evaluation outputs and the dataset metadata; they never run a model.

## Inputs

- Our evaluation outputs: `emotion_recognition/outputs/<model>/<dataset>_predictions.csv` (5 models × 6 datasets)
  and `emotion_recognition/reports/summary.json` (accuracy, and whether a dataset is in a model's training data).
- The datasets in `emotion_recognition/data/` (audio and metadata). They are not in the repository: download them
  first with `python scripts/common/download_data.py` (run from `emotion_recognition/`, `dataset: all` in
  `config/download_data.yaml`). Used files:
  - audio headers and WAVs of `crema_d`, `mcae_spps`, `mess`, `ravdess`, `tess`, `jl_corpus` (and the JL-Corpus `.txt` transcripts);
  - `data/crema_d/processedResults/tabulatedVotes.csv` (voice-only listener votes) and `data/crema_d/VideoDemographics.csv`;
  - `data/mcae_spps/Corpus information.xlsx` (votes of the 40 listeners, recognition rate, sentence text).

Only datasets whose licence allows public clips are used (never ESD or THAI-SER).

## Scripts

| Script | Writes | Notes |
|---|---|---|
| `common.py` | nothing | shared loading: predictions, listener votes, label orders, model and dataset names, training-data membership, audio trimming and levelling |
| `make_figures.py` | `assets/stats.json`, `assets/fig/*.png` (22 figures, 300 dpi) | confusion matrices of the two top models, CREMA-D actor vs listeners, MCAE-SPPS recognition rate, seen vs unseen accuracy, clips per emotion |
| `make_audio.py` | `assets/audio/*.mp3` (41 clips), `assets/manifest.json` | the clip list is the constant `CLIPS` at the top; change it on purpose only, the slides use these files. Needs `stats.json`, so run `make_figures.py` first |
| `make_candidates.py` | `assets/audio_candidates/*.mp3`, `assets/audio_candidates/candidates.json` | replacement candidates for the "ambiguous ground truth" example; not used by the deck |

Audio processing (same for every clip): leading and trailing silence trimmed (frames 35 dB below the loudest
20 ms frame, 0.12 s padding); the loudest clip of each quartet at about -18 dBFS RMS, quieter ones keep half of
their level difference (at most 8 dB); single clips at about -18 dBFS RMS; peaks at most -1 dBFS; 16 kHz mono MP3
at 64 kbps (ffmpeg).

## Run

Python with numpy, matplotlib, soundfile and openpyxl (the project environment has them; no pandas needed), and
`ffmpeg`/`ffprobe` on `PATH`. Paths are found from the script location, so any working folder works. From the
repository root:

```bash
python emotion_recognition/presentation/scripts/make_figures.py
python emotion_recognition/presentation/scripts/make_audio.py
python emotion_recognition/presentation/scripts/make_candidates.py   # optional
```

or from `emotion_recognition/presentation/`: `python scripts/make_figures.py` and so on. Then rebuild the deck
(see `../README.md`). With unchanged inputs, `stats.json`, `manifest.json` and (with the same ffmpeg) the MP3s come out byte-identical.

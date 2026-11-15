# Presentation: complete SER benchmark and adaptation proposal

`SER_evaluation.pptx` (67 slides) presents the completed **10-model × 16-dataset** benchmark, with 67,062 clips per model and
160 current-protocol runs, refreshed on 2026-11-15. It covers the task and protocol, the complete dataset and model
inventory, full-cohort results, selected listening cases and the earlier adaptation proposal. Speaker notes are
included. Six public datasets supply the retained playable audio examples; those six are examples within the
sixteen-dataset evaluation, not its complete scope.

Current benchmark values are read from `reports/summary.json` and `research_paper/output/cross_dataset_analysis.json` at build time.
`content.js` rejects an incomplete snapshot. The main results use the full current protocol for all models;
early 200-clip API samples and the old acoustic-focused prompt are not the current evaluation. UAR and macro-F1
appendix tables are native editable objects covering all sixteen datasets and ten models. Four Accuracy slides
show eight datasets per slide at 20 pt. Selected confusion
matrices and listener examples retain their explicit cohort/model scope. The adaptation section is a proposal;
its historical compute estimates are not new training results or measured internal-call performance.

- Click a blue play button in slide-show mode to play or pause the embedded public audio.
- The audio examples retain their original licences and provenance in `assets/manifest.json`.
- See [the benchmark summary](../reports/summary.md) and [research analyses](../../research_paper/README.md)
  for all current metrics, source hashes, consensus, duration, language and emotion-intensity analysis.

## Rebuild

First refresh the benchmark and cross-dataset analysis in the `emotion_recognition_llm` environment. Then:

```bash
cd emotion_recognition/presentation
node build.js SER_evaluation.pptx
python fix_audio.py SER_evaluation.pptx
./render.sh SER_evaluation.pptx
```

`package.json` declares the existing `pptxgenjs` dependency. Reuse installed dependencies; `render.sh` uses
LibreOffice and Poppler to produce PDF and PNG previews in `render/`.

| File | Content |
|---|---|
| `build.js` | builds the deck from the modules below |
| `lib.js`, `clips.js` | existing theme, layout helpers and playable audio cards |
| `s_intro.js`, `s_data.js`, `s_models.js` | current scope, sixteen-dataset overview, selected audio examples, ten-model inventory and protocol |
| `s_results.js` | complete benchmark scores and selected confusion-matrix/listener analyses |
| `fix_audio.py` | playable audio triggers and standards-compatible package/chart caches, preserving embedded workbooks |
| `s_cases.js`, `s_future.js` | scoped listening cases and the earlier adaptation/annotation proposal |
| `s_appendix.js` | complete UAR/macro-F1 tables, selected label-mapping examples and detailed SER architecture |
| `content.js` | current data loading, shared text, architecture and selected-case descriptions |
| `facts.md` | source notes for architecture and historical resource estimates; current benchmark metrics come from the analysis snapshot |
| `assets/` | preserved public audio, historical selected-case statistics and figures |

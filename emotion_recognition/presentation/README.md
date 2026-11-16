# Presentation: complete SER benchmark and adaptation proposal

The extended inspection copy is [`updated/SER_evaluation_extended.pptx`](updated/SER_evaluation_extended.pptx)
(84 slides, 2026-11-16). It preserves the complete 10-model × 16-dataset results and adds two playable
recordings per dataset with all ten predictions. Slide 39 is the dedicated same-sentence API illustration;
slides 40–55 cover all sixteen datasets in benchmark order. The priority slide is also available separately
as [`updated/Semantic_vs_Vocal_Emotion.pptx`](updated/Semantic_vs_Vocal_Emotion.pptx).

The new examples are deliberate contrasts, not a representative performance sample: A maximizes the
SER-minus-LLM correct-count gap, while B maximizes the reverse gap with a different reference emotion and speaker.
Listener votes remain separate from corpus reference labels. The semantic illustration shows four selected
CREMA-D takes of one actor saying “Don't forget a jacket.” Gemini predicts happy on all four, while GPT
Realtime predicts neutral. The slide also states that Gemini changes its prediction on the neutral and sad
takes. This is evidence of limited differentiation in those examples, not a causal demonstration of semantic reliance.

This is an internal research inspection copy. The sixteen corpora retain their source terms, including restrictions
on redistribution. New audio copies are full-length mono 16 kHz MP3s without trimming or loudness normalization.
Exact source paths, original-audio hashes, all 320 CSV/JSON source hashes and the selection rules are recorded
in `assets/inspections/manifest.json`. The source deck below is preserved.

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

To rebuild the extended copy, use the bundled presentation runtime (set `RUNTIME_NODE_MODULES` and
`RUNTIME_PYTHON` when needed). Run the following from this directory. `node` requires `@oai/artifact-tool`,
and Python requires numpy, Pillow, openpyxl and lxml; ffmpeg/ffprobe must be on `PATH`.

```bash
python scripts/prepare_inspections.py --semantic-only
node scripts/build_semantic.mjs
python scripts/embed_inspection_audio.py .inspection-build/semantic-base.pptx .inspection-build/semantic-audio.json .inspection-build/semantic-candidate.pptx
python scripts/prepare_inspections.py
node scripts/build_inspections.mjs
python scripts/embed_inspection_audio.py .inspection-build/inspections-base.pptx .inspection-build/inspections-audio.json .inspection-build/inspections-audio.pptx
python scripts/merge_inspections.py
python scripts/check_inspections.py
node scripts/verify_import_batches.mjs
node scripts/finalize_inspections.mjs .inspection-build/SER_evaluation_extended-candidate.pptx updated/SER_evaluation_extended.pptx 84 '[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55]'
```

The finalizer requires a new output filename when a previous output already exists. The semantic candidate can
also be finalized on its own with `node scripts/finalize_inspections.mjs .inspection-build/semantic-candidate.pptx updated/Semantic_vs_Vocal_Emotion.pptx`.
Rendering previews stay in `.inspection-build/`. New slides use editable text and native tables, and reuse
the source layout and notes master. The merge preserves original chart workbooks and audio triggers.
The source and extended decks exceed the native importer's XML construction budget, so the checker creates
seven exact-content 12-slide packages and verifies all 84 slides through those bounded imports. Full-package
structure, fonts, table geometry and native chart/workbook links are checked separately by the finalizer.
Mandarin and Thai transcripts specify Arial Unicode MS as the script fallback; the original Latin fonts are retained.

To rebuild only the preserved 67-slide base:

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

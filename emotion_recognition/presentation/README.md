# SER benchmark presentation

`updated/` contains exactly two deliverables:

- [`SER_evaluation_extended.pptx`](updated/SER_evaluation_extended.pptx): the complete **85-slide** presentation. It includes all **10 models × 16 datasets**, the original research/adaptation material, two playable ten-model inspections per dataset, and both same-sentence case studies.
- [`Semantic_vs_Vocal_Emotion.pptx`](updated/Semantic_vs_Vocal_Emotion.pptx): one **two-slide** presentation combining the CREMA-D and JL-Corpus examples, with all **nine** recordings embedded.

The complete deck places the two sentence examples on slides **39–40** and the sixteen dataset inspections on slides **41–56**. Both files use stable filenames; intermediate or version-numbered copies stay outside `updated/`.

## Examples and interpretation

The CREMA-D example uses four selected emotional takes of “Don't forget a jacket.” from actor 1002. Gemini predicts happy on all four and GPT Realtime predicts neutral. Voice-only listener votes are shown separately. Gemini changes its prediction on the same actor's neutral and sad takes, as disclosed on the slide.

The JL-Corpus example uses “Work hard or you lose.” Speaker `male2`, session b, repetition 1 provides all five evaluated emotions: angry, neutral, happy, sad and excited. Both APIs predict angry on all five. Across all 80 evaluated recordings of this sentence, both APIs predict angry together in 33 cases. These are acted labels without per-clip listener votes. The pattern is consistent with lexical bias but does not establish its cause.

The 32 dataset inspections deliberately show contrasts rather than estimate performance. Example A maximizes the SER-minus-LLM correct-count gap. Example B maximizes the reverse gap with a different reference emotion and speaker. Reference labels and listener votes remain separate. JL-Corpus's excited label accepts happy or surprised.

The quantitative results cover 67,062 clips per model and 160 current-protocol model–dataset runs. They use the completed cohort for every model, including Gemma 4 E4B, rather than the historical 200-clip API samples. Accuracy, UAR, macro-F1 and training-provenance caveats remain in the complete deck. The adaptation section is a proposal, not new training or internal-call results.

## Audio and provenance

Click a blue play button in PowerPoint slide-show mode to play or pause the embedded audio. This is an internal research inspection copy, and each corpus retains its source terms, including redistribution restrictions. New audio copies are full-length mono 16 kHz MP3s at 64 kbit/s, without trimming or loudness normalization.

- `assets/manifest.json` retains the earlier public listening examples and their provenance.
- `assets/inspections/manifest.json` records the 32 selected inspections, source audio hashes, 320 CSV/JSON source hashes and selection rules.
- `assets/inspections/semantic.json` records the first matched-sentence example.
- `assets/inspections/lexical/manifest.json` records the second example, source hashes, protocol and full-sentence context counts.

See [benchmark results](../reports/summary.md) and [research analyses](../../research_paper/README.md) for the full quantitative record.

## Rebuild the two deliverables

Run from this directory using the bundled presentation runtime. Set `RUNTIME_NODE_MODULES`, `RUNTIME_PYTHON` and `PRESENTATIONS_SKILL_DIR` when the bundled paths differ. JavaScript requires `@oai/artifact-tool`; Python requires numpy, Pillow, openpyxl and lxml. `ffmpeg` and `ffprobe` must be on `PATH`.

All drafts, previews, temporary one-slide packages and validation receipts stay in `.inspection-build/`. The original `SER_evaluation.pptx` and its existing source modules supply the base template. No additional deck is published from that intermediate base.

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
python scripts/prepare_lexical_example.py
node scripts/build_lexical_example.mjs
python scripts/embed_inspection_audio.py .inspection-build/lexical-base.pptx .inspection-build/lexical-audio.json .inspection-build/lexical-candidate.pptx
python scripts/insert_lexical_example.py
python scripts/consolidate_presentations.py
node scripts/finalize_inspections.mjs .inspection-build/SER_evaluation_complete-candidate.pptx .inspection-build/consolidated/SER_evaluation_extended.pptx 85 '[39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56]'
node scripts/finalize_inspections.mjs .inspection-build/Semantic_vs_Vocal_Emotion-candidate.pptx .inspection-build/consolidated/Semantic_vs_Vocal_Emotion.pptx 2 '[1,2]'
python scripts/consolidate_presentations.py --publish
```

Use a fresh private `consolidated/` staging directory when rebuilding because the finalizer does not overwrite an existing validated staging file. Publication then updates the two stable filenames and removes the obsolete standalone/version-numbered copies.

The checks verify all saved prediction cells, audio triggers and source hashes, preserve prior media/chart/workbook bytes, and validate package structure, fonts and native tables/charts. The source and full deck exceed the native importer's XML budget, so the intermediate 84-slide set is checked in seven exact-content 12-slide packages and the inserted page is validated separately. The two-slide example deck passes a complete native import. Mandarin and Thai transcripts retain Arial Unicode MS as the script fallback, while original Latin fonts are preserved.

## Source files

| File | Role |
|---|---|
| `build.js`, `lib.js`, `clips.js`, `content.js`, `s_*.js` | Existing base-deck content, theme and layout modules |
| `scripts/prepare_inspections.py`, `scripts/build_inspections.mjs` | Complete ten-model listening inspections |
| `scripts/build_semantic.mjs` | CREMA-D same-sentence page |
| `scripts/prepare_lexical_example.py`, `scripts/build_lexical_example.mjs` | JL-Corpus same-sentence page |
| `scripts/embed_inspection_audio.py` | Embedded audio and click-to-toggle triggers |
| `scripts/merge_inspections.py`, `scripts/insert_lexical_example.py` | Preserve source parts while composing the complete deck |
| `scripts/consolidate_presentations.py` | Extract the two-slide example deck and publish only the two canonical files |
| `scripts/check_inspections.py`, `scripts/verify_import_batches.mjs`, `scripts/finalize_inspections.mjs` | Prediction, package, native object and import validation |
| `facts.md` | Architecture sources and historical resource-estimate notes |

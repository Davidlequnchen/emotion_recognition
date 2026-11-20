# SER benchmark presentation

There are exactly two presentation deliverables:

- [`SER_evaluation.pptx`](SER_evaluation.pptx): the canonical **92-slide** presentation, covering **10 models × 16 datasets**. It contains the original research and adaptation material, a full dataset introduction with four playable clips per dataset, the two same-sentence case studies, sixteen later listening inspections, and five ten-model error analyses.
- [`updated/Semantic_vs_Vocal_Emotion.pptx`](updated/Semantic_vs_Vocal_Emotion.pptx): the same **two sentence-example slides**, with all **nine recordings** embedded, extracted from the canonical deck.

There is no separate extended version. Intermediate packages, previews and validation receipts remain in the ignored `.inspection-build/` directory.

## Content and interpretation

The single dataset overview preserves language, clip and speaker counts, shared-text design, annotation sources and dialogue versus standalone status. The label mapping covers all sixteen benchmark datasets. Each dataset introduction shows four recordings and predictions from all ten models. The actor-versus-listener stacked chart also includes all ten models on the same 3,699 disputed CREMA-D clips, retaining its original colours and axes. Published corpus totals and the actual benchmark cohorts are distinguished in slide notes.

The six-column model survey includes a brief architecture summary, release, parameter count, emotion training data and output classes for all ten models. Three native editable schematics explain the executed audio-to-text routes for Qwen2.5-Omni, Qwen2-Audio Instruct and Gemma 4 E4B. Model-card references and retained checkpoint evidence are recorded in `assets/model_refresh/facts.json` and speaker notes.

Accuracy, UAR and Macro-F1 each have one complete **16-dataset × 10-model** table. Coloured headers distinguish dedicated SER, open audio LLMs and API audio LLMs. Green and red mark the highest and lowest score within each dataset, including ties. These highlights compare models within a row rather than claim statistical significance.

The CREMA-D example uses four emotional takes of “Don't forget a jacket.” by actor 1002. Gemini predicts happy on all four and GPT Realtime predicts neutral. Voice-only listener votes remain separate from acted labels. Gemini changes its prediction on the same actor's neutral and sad takes, as disclosed on the slide.

The JL-Corpus example uses “Work hard or you lose.” Speaker `male2`, session b, repetition 1 provides angry, neutral, happy, sad and excited recordings. Both APIs predict angry on all five. Across all 80 evaluated recordings of this sentence, both APIs predict angry together in 33 cases. These are acted labels without per-clip listener votes. The pattern is consistent with lexical bias but does not establish its cause.

The 32 later listening inspections show contrasts rather than estimate performance. Example A maximizes the SER-minus-LLM correct-count gap. Example B maximizes the reverse gap using a different reference emotion and speaker. JL-Corpus excited accepts happy or surprised. The five error-analysis slides retain their original two recordings and add the five audio LLM predictions. Confidence values are shown only where the dedicated SER models supply meaningful class scores.

Quantitative results cover **67,062 clips per model** and all **160 current-protocol model–dataset runs**, including completed Gemma 4 E4B cohorts. The adaptation section remains a proposal. Sources and former footnotes are in speaker notes; every slide reserves at least the bottom 10% as blank space.

## Assets

- `assets/manifest.json`: earlier public listening examples and provenance.
- `assets/dataset_refresh/data.json`: 64 introduction clips, 640 prediction cells, metadata and 337 source hashes.
- `assets/error_refresh/data.json`: the ten retained error-analysis clips and 100 current prediction cells.
- `assets/model_refresh/facts.json`: open-model architecture, release and training-data evidence.
- `assets/result_refresh/data.json`: the full three-metric tables and source hashes.
- `assets/inspections/manifest.json`: 32 later inspections, source audio hashes, prediction-source hashes and selection rules.
- `assets/inspections/semantic.json` and `assets/inspections/lexical/manifest.json`: matched-sentence examples and full-sentence context counts.

The research deck includes internal listening material. Added audio is full-length mono 16 kHz MP3 at 64 kbit/s, without trimming or loudness normalization. Retained legacy examples preserve their existing media bytes.

## Rebuild the current deck

Run from this directory using the bundled runtime discovered by `load_workspace_dependencies`. Set `RUNTIME_NODE_MODULES` when the bundled package location differs from the script default. No model inference runs during the presentation build. Saved metadata and audio assets allow layout-only rebuilds; `prepare_dataset_refresh.py` and `prepare_error_refresh.py` refresh those inputs from local corpora and current predictions when needed. Rebuilding the actor-versus-listener chart also reads the local CREMA-D listener votes and all ten saved prediction CSVs.

```bash
node scripts/build_dataset_refresh.mjs
node scripts/build_model_refresh.mjs
node scripts/build_error_refresh.mjs
node scripts/build_result_refresh.mjs
node scripts/build_listener_plot_refresh.mjs
python scripts/embed_inspection_audio.py .inspection-build/dataset-refresh.pptx .inspection-build/dataset-refresh-audio.json .inspection-build/dataset-refresh-audio.pptx
python scripts/embed_inspection_audio.py .inspection-build/error-refresh.pptx .inspection-build/error-refresh-audio.json .inspection-build/error-refresh-audio.pptx
python scripts/refresh_main_deck.py --source SER_evaluation.pptx
python scripts/check_main_refresh.py
node scripts/verify_main_imports.mjs
python scripts/render_main_refresh.py
python scripts/consolidate_presentations.py
```

Finalize the main candidate and two-slide excerpt with `scripts/finalize_inspections.mjs`, passing the exact slide count and the native-table slide list from `.inspection-build/refresh/content-validation.json`. Its staging destination is `.inspection-build/refresh/validated-20261117-compact/`, with filenames matching the two deliverables. For the excerpt pass count `2` and table slides `[1,2]`. Use a fresh private staging directory because the finalizer does not overwrite a validated file. The finalizer validates every table owner in bounded passes over the identical candidate. After inspecting all contact sheets and changed pages, run `python scripts/render_main_refresh.py --record-review` to record the completed visual review and verify the blank bottom margin. Then publish with:

```bash
python scripts/consolidate_presentations.py --publish
```

Checks verify source hashes, every saved prediction and metric cell, unique shape/relationship IDs, audio timing targets, the 10% bottom margin, retained chart data and embedded workbooks, and native editable tables. The complete deck exceeds the Artifact Tool construction budget, so all slides are imported in bounded exact-content batches. The two-slide excerpt passes a complete import. Mandarin and Thai text use Arial Unicode MS script fallbacks while preserving the original Latin fonts.

For a survey-only revision, run `node scripts/build_model_refresh.mjs`, then `python scripts/refresh_main_deck.py --survey-only --source SER_evaluation.pptx` before validation. This replaces the survey page while preserving every other slide XML part.

The older `build.js`, `lib.js`, `clips.js`, `content.js` and `s_*.js` retain the historical base-deck source. Historical inspection/semantic builders remain available for source reconstruction; the refresh workflow above updates the canonical deck in place without publishing additional versions.

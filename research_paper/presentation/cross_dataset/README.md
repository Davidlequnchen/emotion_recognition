# Presentation: cross-dataset analysis

`cross_dataset_analysis.pptx`: 14 slides in the slide show (about 20–25 minutes) plus a hidden appendix of 7 slides,
speaker notes on every slide. It asks why the SER benchmark models do well on some datasets and poorly on others,
and tests three explanations on the saved predictions of 16 public corpora: language (English vs non-English),
setting (conversational vs acted) and clip duration. One section per question, each with its conclusion as the
slide title; then a summary, limitations and next steps. The appendix holds the secondary tables (all metrics,
splitting checks, duration sensitivity, direct slope comparisons, corpus-level correlations, the always-neutral
baseline, input windows).

Style follows `preparation_tools/presentation/` (pptxgenjs, same colours, title layout and notes) with the rules of
the research_paper decks: body text at least 20 pt, tables at least 18 pt, figure labels at least 16 pt as rendered
(figures are drawn at slide size and placed unscaled), page numbers and source lines 16 pt, no grey text.
`build.js` stops if a font is below these limits.

## Data and sources

Public datasets only, and only the predictions already saved by the benchmark: no model run, API call or audio is
used by the deck. Every number comes from `assets/facts.json`, written by `scripts/make_facts.py` from
`research_paper/output/cross_dataset_*.csv` and `cross_dataset_manifest.json`; `facts.md` lists each number with
its source file. Every figure is read directly from `research_paper/analysis_report/figures/`, where
`research_paper/scripts/cross_dataset_plot.py` draws it for both the reports and this deck (the single source of the
figures). The training-exposure statuses come from `emotion_recognition/docs/training_provenance.md` through the
analysis outputs.

## Rebuild

One command regenerates everything from the current saved results:

```bash
research_paper/presentation/cross_dataset/rebuild.sh
```

It runs, in order:

1. `research_paper/scripts/cross_dataset_analysis.py` (from the repository root): reads the saved predictions and
   writes the output CSVs, the figures and `analysis_report/cross_dataset_{en,zh}.md`. When new prediction files
   change the run selection, the script asks for a fresh audio-header inventory and `rebuild.sh` reruns it with
   `--refresh-inventory` (this only re-reads audio headers).
2. `scripts/make_facts.py`: `assets/facts.json` and `facts.md`.
3. `node build.js` (after `npm install` if `node_modules/` is missing): writes the deck, checks font sizes, marks the
   appendix hidden, keeps one paragraph-properties element per paragraph (pptxgenjs repeats it for multi-run text).
4. `./render.sh`: `render/<deck>.pdf` and `render/p-NN.png` previews, hidden slides included. It waits while free
   memory is below 25% (macOS), because LibreOffice needs about 1 GB. `python sheet.py 1 6 2` tiles the previews.

The Python steps use `/opt/homebrew/Caskroom/miniconda/base/envs/emotion_recognition_llm/bin/python` (override with
`PYTHON=...`); they need only numpy, scipy and matplotlib, and `make_facts.py` only the standard library.

## Complete Gemma 4 E4B results

The current build includes all ten models on all sixteen datasets (160 runs), including Gemma 4 E4B. It joins every
multi-dataset mean, duration estimate, per-corpus plot and table. Gemma has a positive main duration interval,
while its 1–15 second sensitivity interval includes zero. The direct slope comparison includes Gemma and does not
show a clear difference from either SER reference.

Slide text reads the facts dynamically. Fixed interpretation still requires review after a refresh; `build.js`
prints `CHECK WORDING` if a checked claim no longer holds.

## Files

| File | Content |
|---|---|
| `rebuild.sh` | the one-command rebuild (analysis, facts, deck, previews) |
| `build.js` | builds the deck from `assets/facts.json`; font-size guard, wording checks, paragraph-property fix |
| `lib.js` | style and layout helpers (adapted from `preparation_tools/presentation/lib.js`): slide frame, appendix numbering (A1…), bullets, tables, figures at native size, "preliminary" tag, source line, notes |
| `s_intro.js` | title, problem (accuracy spread), how each factor is tested, datasets, clip durations |
| `s_results.js` | setting, training-exposure check, language, duration (pooled vs within, per-dataset heat map, raw bins) |
| `s_end.js` | summary, limitations, next steps, hidden appendix A1–A7 |
| `scripts/make_facts.py` | `assets/facts.json` and `facts.md` from the saved analysis outputs |
| `facts.md` | every number on the slides with its source file |
| `render.sh`, `sheet.py` | previews for checking (LibreOffice, poppler, Pillow) |

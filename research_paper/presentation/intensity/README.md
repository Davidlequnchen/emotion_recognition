# Presentation: emotion intensity

`intensity_analysis.pptx`: 13 slides in the slide show (about 15–20 minutes) plus 5 hidden appendix slides, speaker
notes on every slide. It presents the intensity analysis of the benchmark (`../../analysis_report/intensity_en.md`):
is a clip with a stronger emotion label classified correctly more often? Six corpora carry intensity labels (four
acted levels, one listener rating, one annotator rating); the deck shows the weakest-vs-strongest accuracy change, the
accuracy at every level, how often emotional clips are answered "neutral", the class- and duration-adjusted change,
matched takes of the same sentence, the MEAD breakdown of Qwen2-Audio, caveats, summary and next steps.

Style follows `preparation_tools/presentation/` (pptxgenjs, same colours, title layout and notes; `lib.js`,
`build.js`, `render.sh` and `sheet.py` are adapted copies): body text at least 20 pt, tables at least 18 pt, figure
and diagram text at least 16 pt, citations and page numbers 16 pt, no grey text. `build.js` stops if a font is below
these limits and also counts a figure as too small when it would have to be scaled down on the slide.

## Data and sources

Only saved benchmark predictions of public corpora; no model was run. All numbers are read at build time from
`../../output/intensity_analysis.json` (`data.js`); `facts.md` lists every number with its source field. The figures
are not copied: the deck embeds `../../analysis_report/figures/emotion_intensity_*.png`, drawn at slide size (300 dpi)
by `../../scripts/intensity_plot.py`, the single source of these figures for the report, the paper and the slides.

## Rebuild (one command)

```bash
research_paper/presentation/intensity/rebuild.sh
```

It runs, from the repository root:

1. a check that the shared audio inventory still matches the selected prediction files (refreshed only if new runs
   changed the selection, with the same function the cross-dataset analysis uses);
2. `python research_paper/scripts/intensity_analysis.py`: statistics from the saved predictions, the figures
   (`intensity_plot.py`) and both reports (`intensity_en.md`, `intensity_zh.md`);
3. `node build.js`: writes `intensity_analysis.pptx` (`npm install` first if `node_modules/` is missing);
4. `./render.sh`: `render/<deck>.pdf` and `render/p-NN.png` (hidden slides included) for checking with LibreOffice;
   it waits while free memory is below 25%.

Python is the `emotion_recognition_llm` conda environment (set `PYTHON=...` to use another one). Partial steps:
`python research_paper/scripts/intensity_plot.py` redraws the figures from the saved JSON; `node build.js` rebuilds
only the deck; `python sheet.py 1 6 2` tiles rendered pages into `render/sheet_1.png`.

## Complete Gemma 4 E4B results

The benchmark now has all ten models on all sixteen datasets (160 runs). This deck includes all ten models on the
six corpora with eligible intensity labels. Gemma has a positive adjusted weakest-to-strongest interval on CREMA-D
and RAVDESS. Its MEAD and THAI-SER intervals include zero. MEAD's accuracy is slightly lower at strong than medium
for Gemma, so a stepwise increase no longer describes every model.

`write_facts.js` regenerates `facts.md` directly from `data.js` at each build. Re-read fixed slide wording after a
refresh, especially claims about all models, monotonic changes, and supported matched takes. MEAD has only three
published take keys from two actors and unverified text equivalence, so the matched estimate stays suppressed.

Dry run without touching the repository outputs: `INTENSITY_JSON=<analysis.json> INTENSITY_FIGS=<figure folder> node
build.js <out.pptx>` builds the deck from another analysis file and figure folder (used to test a simulated Gemma result).

## Files

| File | Content |
|---|---|
| `rebuild.sh` | the one command above |
| `build.js` | builds the deck, checks font sizes, keeps one paragraph-properties element per paragraph (pptxgenjs repeats it for multi-run text, which makes LibreOffice drop bullets) |
| `lib.js` | style and layout helpers: slide frame, appendix numbering (A1…), bullets, boxes, tables, citations, notes, figures at native size, preliminary tag |
| `data.js` | reads the saved analysis and computes every number used on the slides |
| `slides.js` | all slides with speaker notes |
| `write_facts.js`, `facts.md` | generated numbers with their source, refreshed by `build.js` |
| `render.sh`, `sheet.py` | LibreOffice rendering and contact sheets for checking |
| `package.json` | pptxgenjs and jszip (`npm install`); `node_modules/` and `render/` are git-ignored |

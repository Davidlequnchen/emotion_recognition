# Presentation: model consensus as a pseudo-label filter

`consensus_analysis.pptx`: 14 slides in the slide show plus a hidden appendix of 4 slides, speaker notes on every
slide. It presents the model-consensus analysis of the SER benchmark ([report](../../analysis_report/consensus_en.md),
[中文](../../analysis_report/consensus_zh.md)): if several of the ten models agree on a clip's emotion, how accurate is
the agreed label (accuracy of agreed labels = correct accepted / accepted) and how many clips get one (coverage =
accepted / common clips)?

| Slides | Content |
|---|---|
| 1–3 | title; problem (hand labels do not scale, single models are often wrong, agreement as a filter); the accuracy–coverage trade-off on CREMA-D |
| 4–6 | definitions: coverage and accuracy of agreed labels (diagram), vote rules (strict majority, k of m, unanimity; invalid answers abstain, ties rejected), the CREMA-D worked example |
| 7 | the four model groups (SER5, LLM3, AudioLLM5, All10) with their colours and markers |
| 8–11 | results: accuracy–coverage grids (2 slides), MELD on identical clips, unanimity heat map |
| 12–14 | summary, caveats, how to use agreement on target calls |
| A1–A4 (hidden) | appendix groups Selected3, SER+API7, AudioLLM4 (2 slides), share of agreed labels that are neutral, how to rebuild |

Style follows `preparation_tools/presentation/` (pptxgenjs, same colours, title layout and notes): figures and tables
over text, few short bullets, body text at least 20 pt, tables at least 18 pt, figure and diagram text at least 16 pt as
placed on the slide, page numbers 16 pt, no grey text. `build.js` stops if a font is smaller. Group colours are
colour-blind-safe (Okabe–Ito, plus Tol's indigo for AudioLLM4) and each group also has its own marker shape; they are the
same in the paper figures, the slide figures and the tables.

## Complete results and rebuild

All ten models now cover all sixteen datasets (160 current-protocol saved prediction runs), including the final
Gemma 4 E4B results. LLM3, AudioLLM5 and All10 use all 3, 5 and 10 members on every corpus. The legacy AudioLLM4
comparison remains in the appendix to preserve comparability with earlier analyses. Missing-member markers and
pending status no longer appear in the current figures or slides.

```bash
conda activate emotion_recognition_llm
cd research_paper/presentation/consensus
./rebuild.sh        # consensus analysis + paper figures + reports, slide figures + numbers + facts.md, the deck
./render.sh         # optional: render/<deck>.pdf and render/p-NN.png previews (LibreOffice), hidden slides included
```

`rebuild.sh` runs `research_paper/scripts/consensus.py` (CSVs, manifest, both reports and the four paper figures),
`scripts/make_assets.py` (slide figures in `fig/`, `assets/numbers.json`, `facts.md`) and `node build.js`. Every
number on the slides is read from `assets/numbers.json`, so titles, bullets and notes update with the data; the marks
for missing members disappear once every member is present. Check the rendered slides afterwards: wording such as
"much less reliable on dialogue" is chosen from the numbers, but new numbers can change line breaks.

## Files

| File | Content |
|---|---|
| `rebuild.sh` | the one-command rebuild above |
| `scripts/make_assets.py` | reads `research_paper/output/consensus_thresholds.csv`, `consensus_by_class.csv`, `consensus_baselines.csv` and `consensus_manifest.json`; draws the slide figures with `consensus_plot.plot_slides` (the same code as the paper figures, drawn at slide size); writes `assets/numbers.json` and `facts.md` |
| `build.js` | builds the deck, checks font sizes, marks appendix slides hidden, keeps one paragraph-properties element per paragraph (pptxgenjs repeats it for multi-run text) |
| `lib.js` | style and layout helpers adapted from `preparation_tools/presentation/lib.js`: slide frame, bullets, boxes, arrows, tables, stat tiles, group markers, images at native size, notes |
| `s_intro.js`, `s_results.js` | the slides and their speaker notes |
| `fig/` | slide figures (PNG, 300 dpi, drawn at their size on the slide) |
| `facts.md` | every number on the slides with its source row (generated) |
| `render.sh`, `sheet.py` | previews for checking (`render.sh` waits while free memory is below 25%) |

`npm install` (run by `rebuild.sh` when `node_modules/` is missing) installs pptxgenjs and jszip; `node_modules/` and
`render/` are git-ignored.

## Data

Public benchmark datasets only, through saved predictions; no audio, no call data and no company information is used.
The four example clips on slide 5 and the 1,000-clip funnel on slide 4 are illustrations and are labelled as such.

# Presentation: all research_paper analyses in one deck

`research_analyses.pptx`: 71 slides, 55 in the slide show and 16 hidden appendix slides, speaker notes on every slide.
It joins the three analysis decks and the venue slides, in this order, without changing them:

| Slides | Part | Source deck |
|---|---|---|
| 1 | cover with the contents and page ranges | (this folder) |
| 2 | divider: part 1 | (this folder) |
| 3–16, A1–A4 (hidden) | model consensus as a pseudo-label filter | [`../consensus/`](../consensus/) |
| 17 | divider: part 2 | (this folder) |
| 18–31, A5–A11 (hidden) | cross-dataset differences | [`../cross_dataset/`](../cross_dataset/) |
| 32 | divider: part 3 | (this folder) |
| 33–45, A12–A16 (hidden) | emotion intensity | [`../intensity/`](../intensity/) |
| 46 | divider: part 4 | (this folder) |
| 47–55 | target venues: shortlist, additional candidates, review windows | [`../venues/`](../venues/) |

Each part keeps its own title slide. The hidden appendix slides stay at the end of their own part; they are skipped
in the slide show and included in the PDF from `render.sh`. Page numbers run on across the deck: 1, 2, 3, ... for the
slides in the show and A1 to A16 for the appendix (the cover and the dividers count but show no number, like the
title slides). The ranges above are those of the current data; the cover and dividers are filled in by the build.

## How it is built

`build.js` rebuilds the deck from the same sources as the separate decks; nothing is copied. It loads each part's
data (`../consensus/assets/numbers.json`, `../cross_dataset/assets/facts.json`, `../intensity/data.js`) and runs its
slide modules (`s_*.js`, `slides.js`) with that part's own `lib.js`, so styles, figures (read from `fig/` and
`analysis_report/figures/`), notes and font-size tracking are exactly those of the separate decks. All four libs draw
on one pptxgenjs object; none uses a slide master, so there is nothing to merge. Each `lib.js` accepts the page
counters reached so far (`init(pptx, { slideNo, appendixNo })`); a deck built alone calls `init(pptx)` and is
unchanged. The cover and the dividers use `../venues/lib.js`.

The few lines of glue in each part's `build.js` (loading the data; the formatting helpers of the cross-dataset deck)
are repeated in `build.js` here. If you change them in a part's `build.js`, change them here too.

Checks run by `build.js`: the font-size guard of every part's `lib.js`, a second scan of the written slide XML
(smallest text 16 pt, smallest table text 18 pt), and the cross-dataset wording checks (`CHECK WORDING`, with slide
numbers of that deck). After the build, every slide of a part matches the slide of the separate deck except for its
page number.

## Rebuild

When saved data change, rebuild the parts first, then this deck. The current build uses all 160 runs from ten models on sixteen datasets, including final Gemma 4 E4B results:

```bash
research_paper/presentation/consensus/rebuild.sh       # analysis, figures, numbers, consensus deck
research_paper/presentation/cross_dataset/rebuild.sh   # analysis (refreshes the shared audio inventory if needed), facts, deck, previews
research_paper/presentation/intensity/rebuild.sh       # analysis, figures, reports, deck, previews
cd research_paper/presentation/combined
[ -d node_modules ] || npm install
node build.js        # research_analyses.pptx
./render.sh          # render/<deck>.pdf and render/p-NN.png, hidden slides included; waits while free memory < 25%
```

Run the parts in this order: the cross-dataset rebuild refreshes the shared audio inventory that the intensity rebuild
checks. Read each part's build output and its README ("after a rebuild") before sharing: titles that state
conclusions are typed text. When `docs/venues.md` changes, edit `../venues/slides.js`, run `node build.js` there to
check it, then run `node build.js` here. If only the slide code of a part changed, `node build.js` here is enough.
`python sheet.py 1 6 2` tiles rendered pages for a quick look.

## Files

| File | Content |
|---|---|
| `build.js` | builds the combined deck from the four decks' sources; cover, dividers, page counters, checks |
| `render.sh`, `sheet.py` | previews for checking (LibreOffice, poppler, Pillow) |
| `package.json` | pptxgenjs and jszip (`npm install`); `node_modules/` and `render/` are git-ignored |

Public datasets and saved predictions only, as in the separate decks; no company names or internal data.

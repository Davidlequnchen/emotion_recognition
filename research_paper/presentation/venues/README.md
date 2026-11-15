# Presentation: target venues

`target_venues.pptx`: a title slide and eight content slides, speaker notes on every slide. It presents the four
shortlisted venues of [`docs/venues.md`](../../docs/venues.md) (shortlist of 2026-11-09) with the document's own role
markers: ACL 2027 (main target), Interspeech 2027 and IEEE ASLI 2027 (focus candidates), EMNLP 2027 (fallback), then
the additional candidates checked on 2026-11-14 (section 2d) with the document's classification: NeurIPS 2027
Evaluations & Datasets (fallback), ICASSP 2028 (late fallback), ACM MM 2027 (conditional fallback), NeurIPS 2027 main
track and AAAI-28 (not recommended), EACL (no window). The markers and tags are not a ranking or an acceptance
forecast; the slides say so.

| Slide | Content |
|---|---|
| 1 | title: shortlist, the conditions for ACL 2027 (pilot result, qualified service contributor, company approval), source and "checked on" dates |
| 2 | merged venue table, part 1: Venue (role tag), Dates & location, Field / community, Standing & tier (CCF class, h5-index; for ASLI also its predecessors ASRU and SLT), Submission deadline |
| 3 | merged venue table, part 2: Venue (role tag), Acceptance rate & difficulty (for ASLI also ASRU 2025 and SLT 2026), Paper format & types, Link (hyperlink to the official call) |
| 4 | special requirements: reviewing duty (ARR designated service contributor), who qualifies, mandatory parts (Limitations, checklists), profiles and arXiv, paper caps and presentation; no parallel submission, also for the additional candidates |
| 5 | additional candidates, part 1, fallbacks (NeurIPS 2027 E&D, ICASSP 2028, ACM MM 2027): same columns as slide 2, one-line reasons |
| 6 | additional candidates, part 1, not recommended (NeurIPS 2027 main track, AAAI-28) and no window (EACL); MMM is not ACM MM |
| 7 | additional candidates, part 2, fallbacks: same columns as slide 3 |
| 8 | additional candidates, part 2, not recommended and no window: same columns as slide 3 |
| 9 | review windows 2027: submission, decision and the deadlines still reachable after a rejection, for the shortlist and the additional candidates |

The two tables merge the document's "venues at a glance" (section 2a) and "acceptance, fit, difficulty and rules"
(section 2b) tables; the "fit" column is left out. One table did not fit at readable size, so the same four rows are
split over two slides with the venue column repeated. The additional candidates use the same two column sets on
slides 5 to 8, split by classification (fallbacks; not recommended or no window); their fit goes into the difficulty
cell and the role tag is the classification of section 2d. Estimates stay marked "estimate", third-party figures
"unverified", and every slide carries the document's dates (first checked 2026-11-03, tables rechecked 2026-11-06,
shortlist 2026-11-09, ASRU/SLT data and additional candidates 2026-11-14). The speaker notes add the details that do
not fit (older acceptance rates, ASRU 2025 and SLT 2026 counts, the full service contributor rule, other ARR,
Interspeech, NeurIPS, ICASSP, ACM MM and AAAI rules), their sources and the document's recommendation.

Style follows the other research_paper decks (`lib.js` is adapted from `../consensus/lib.js`): tables at least 18 pt,
body text at least 20 pt, source lines and page numbers 16 pt, no grey text (role tags in dark blue, dark green,
dark orange and dark red). Long cells use line breaks, never smaller text. `build.js` stops if a font is below these limits.

## Rebuild

```bash
cd research_paper/presentation/venues
npm install          # once: pptxgenjs and jszip
node build.js        # target_venues.pptx
./render.sh          # optional: render/<deck>.pdf and render/p-N.png (LibreOffice; waits while free memory < 25%)
```

There is no data pipeline: every fact is typed in `slides.js`, with the section of `venues.md` it comes from. When
`venues.md` changes (new dates, acceptance rates or rules), edit `slides.js`, rebuild, check the rendered slides (a
longer cell can push a table onto the source line), then rebuild the combined deck (`../combined/`).

## Files

| File | Content |
|---|---|
| `slides.js` | the nine slides and their speaker notes |
| `lib.js` | style helpers: slide frame, page numbers, tables with line breaks, role tags and hyperlinks, source line, font-size check; also used by `../combined/build.js` for the cover and the section dividers |
| `build.js` | builds the deck; font-size guard, one paragraph-properties element per paragraph, drops the unused slide-master entries pptxgenjs lists in `[Content_Types].xml` |
| `render.sh`, `sheet.py` | previews for checking (LibreOffice, poppler, Pillow) |

`node_modules/` and `render/` are git-ignored. No company names or internal data; the links go to the official public
calls, conference sites and the ARR dates page.

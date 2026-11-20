# Presentation: adapting emotion2vec+ large

`emotion2vec_adaptation.pptx`: 28 slides in the slide show plus a hidden appendix of 9 slides (default settings,
data-scale numbers, code map, references), speaker notes on every slide. It explains the two adaptation paths of
`emotion2vec_adaptation/`: why adaptation is needed, where the clips come from, Path A/B (frozen encoder, cached
embeddings, small head: option A = last block + linear layer, option B = learned mix of all 12 blocks + MLP), Path E (masked student, EMA teacher, frame-level feature regression onto the teacher's layer-normalised last block)
with its encoder, masking, equations, EMA, training phases and the two gates (learnability probe, downstream check), what differs from the original emotion2vec, how an encoder and a head
are combined safely, leakage control, the data scale and million-clip handling (audit cache, duration buckets, checkpoints, exact resume),
running on the B200, monitoring, the validation status (end-to-end mock tests of 17 November 2026: Path A/B ready;
the gates stop harmful SSL set-ups; the new target does no harm but shows no reliable gain; padding and clip length
fixed) and the plan.

Style follows `preparation_tools/presentation/` and `emotion_recognition_benchmark/presentation/` (pptxgenjs, same colours,
title layout, tables and notes): body text at least 20 pt, tables at least 18 pt, diagram labels and figure text at
least 16 pt (citations and page numbers 14 pt), no grey text, figures drawn at their size on the slide. `build.js`
stops if a font is below these limits. Diagrams use one colour code: blue = frozen, orange = trainable, purple = EMA
teacher, red = loss, green = output file, a dark "new" pill = added by this project.

## Data and licences

No call audio, transcripts or customer data are used. The figures and equations are generated from code, the
configuration and public sources (`refs.js`). The validation slides quote results of runs on public corpora (RAVDESS,
CC BY-NC-SA 4.0; CREMA-D; DailyTalk, CC BY-SA 4.0); no audio is embedded.

## Rebuild

```bash
cd emotion2vec_adaptation/presentation
npm install                              # once: pptxgenjs and jszip
python scripts/render_equations.py       # img/eq_*.png, img/sym_*.png (matplotlib mathtext, STIX)
python scripts/make_figures.py           # img/data_scale.png, ema_weights.png, block_weights.png (matplotlib)
node build.js                            # writes emotion2vec_adaptation.pptx (hidden appendix slides included)
./render.sh                              # optional: render/<deck>.pdf and render/p-NN.png, hidden slides included
python sheet.py 1 37 4                   # optional: contact sheet render/sheet_1.png
```

The Python scripts need only matplotlib and numpy (the `emotion_recognition_llm` environment has both); all images are
kept in `img/`, so `node build.js` works without Python. A missing image becomes a dashed placeholder and is reported
by `build.js`. `render.sh` needs LibreOffice and poppler and waits while free memory is below 25% (macOS).

## Files

| File | Content |
|---|---|
| `build.js` | builds the deck, checks font sizes, marks appendix slides hidden, keeps one paragraph-properties element per paragraph |
| `lib.js` | style and layout helpers (adapted from `preparation_tools/presentation/lib.js`): slide frame, colour roles, boxes, arrows, pills, tables, symbol tables, code blocks, legends, citations, notes, images |
| `refs.js` | the references (short label, full entry, link); checked on 2026-11-17 |
| `s_intro.js` | title, problem (domain gap; few labels, much unlabelled audio), data sources, overview of both paths |
| `s_head.js` | Path A/B: architecture (options A and B) and training recipe |
| `s_ssl.js` | Path E: encoder anatomy, teacher–student figure, masking, equations, EMA, training phases, the two gates, original vs ours |
| `s_ops.js` | encoder + head bundle, leakage control, data scale, million-clip scale, B200, monitoring, validation status, option A vs B, mock tests, plan |
| `s_appendix.js` | hidden appendix: default settings, data-scale numbers, code map, references |
| `scripts/style.py` | figure style (deck colours, Calibri when available, 300 dpi) |
| `scripts/render_equations.py` | equations and table symbols as PNG |
| `scripts/make_figures.py` | `data_scale` (log-scale audio hours), `ema_weights` (EMA weight of past student steps), `block_weights` (option B's learned block weights, mock test) |
| `img/` | every image the deck uses |
| `render.sh`, `sheet.py` | previews for checking (`render/` is git-ignored) |
| `archive/` | the previous 11-slide deck (`emotion2vec_adaptation_framework.pptx`) and its builder (`build.mjs`, `render_formulas.py`), kept for reference only |

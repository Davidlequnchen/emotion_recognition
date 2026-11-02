# Presentation: SER evaluation and next steps

`SER_evaluation.pptx` (59 slides, about 35–40 minutes, speaker notes on every slide): the task, the six datasets with
playable audio examples and listener votes, the seven models and their architectures, the results, ten error cases to
listen to, and the plan for fine-tuning emotion2vec+ large and MERaLiON-SER-v1 and labelling our own calls.

- Audio: click a blue play button in the slide show (click again to pause).
- The slide "Accuracy Across Models and Datasets" is a placeholder: replace it with the figure that includes GPT Realtime and Gemini.
- The audio clips come from CREMA-D (ODbL), MCAE-SPPS (CC BY-NC 4.0), MESS (CC BY 4.0), RAVDESS (CC BY-NC-SA 4.0),
  TESS (CC BY-NC 4.0) and JL-Corpus (CC0), the datasets whose licences allow public clips (`public_audio` in
  `scripts/common/eval_datasets.py`); credits on the viewer page.

## Rebuild

```bash
cd emotion_recognition/presentation
npm install                                   # pptxgenjs
node build.js SER_evaluation.pptx             # writes the deck
python fix_audio.py SER_evaluation.pptx       # unique shape ids + audio that plays on click (standard library only)
./render.sh SER_evaluation.pptx               # optional: PDF and PNG previews in render/ (LibreOffice, poppler)
```

| File | Content |
|---|---|
| `build.js` | builds the deck from the parts below |
| `lib.js`, `clips.js` | style, layout helpers, audio cards |
| `s_intro.js`, `s_data.js`, `s_models.js`, `s_results.js`, `s_cases.js`, `s_future.js`, `s_appendix.js` | the slides of each part |
| `content.js` | text that depends on the assets and on `facts.md` (architectures, confusion matrices, LISTEN) |
| `scripts/` | Python scripts that regenerate `assets/` (figures, `stats.json`, audio clips, `manifest.json`) from the evaluation outputs; see `scripts/README.md` |
| `facts.md` | the numbers on the model and fine-tuning slides, each with its source |
| `assets/` | audio clips (16 kHz MP3), figures, `stats.json` and `manifest.json` (every clip with its labels, votes and predictions) |
| `img/` | the emotion2vec+ architecture figure, the labelling-tool screenshot, the play button, the placeholder figure |

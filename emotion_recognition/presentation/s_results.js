// Evaluation results
const fs = require("fs");
const path = require("path");

const ACC = { // accuracy vs acted label (%), reports/summary.md; LLMs: 200 clips per corpus
  crema_d: { name: "CREMA-D", lang: "English", chance: 16.7, e2v: 86.5, meralion: 84.7, melati: 51.4, voxw: 33.7, voxl: 30.7, gpt: 16.1, gemini: 28.2, seen: ["e2v", "meralion"] },
  mcae_spps: { name: "MCAE-SPPS", lang: "Mandarin", chance: 14.3, e2v: 61.3, meralion: 64.9, melati: 48.8, voxw: 46.0, voxl: 39.6, gpt: 9.5, gemini: 33.7, seen: [] },
  mess: { name: "MESS", lang: "English", chance: 25.0, e2v: 40.8, meralion: 74.5, melati: 40.4, voxw: 36.4, voxl: 36.3, gpt: 18.3, gemini: 28.3, seen: [] },
  ravdess: { name: "RAVDESS", lang: "English", chance: 14.3, e2v: 92.1, meralion: 59.3, melati: 48.4, voxw: 50.8, voxl: 44.7, gpt: 24.7, gemini: 44.8, seen: ["e2v"] },
  tess: { name: "TESS", lang: "English", chance: 14.3, e2v: 85.5, meralion: 49.2, melati: 50.4, voxw: 30.1, voxl: 40.1, gpt: 1.6, gemini: 27.2, seen: ["e2v"] },
  jl_corpus: { name: "JL-Corpus", lang: "English (NZ)", chance: 24.0, e2v: 92.5, meralion: 75.9, melati: 99.5, voxw: 45.1, voxl: 48.3, gpt: 28.5, gemini: 35.5, seen: ["e2v", "melati"] },
};
const ORDER = ["crema_d", "mcae_spps", "mess", "ravdess", "tess", "jl_corpus"];
const MODELS = ["e2v", "meralion", "melati", "voxw", "voxl", "gpt", "gemini"];

module.exports = function (L, A) {
  const { C, W, H, MX, FONT, MODEL } = L;
  const p = L.pptx;
  const fig = (f) => path.join(L.DIR, "assets/fig", f);
  const has = (f) => fs.existsSync(fig(f));

  // ---- Protocol
  {
    const s = L.slide("4 · Results", "How we score: same clips, same labels, three metrics");
    L.bullets(s, [
      { t: "Prediction = the most probable of the dataset's own emotions", sub: ["A model's extra classes (other, contempt, unknown) are ignored; chance = 1 / number of emotions in the dataset (17–25%)"] },
      { t: "Accuracy: share of clips where the prediction equals the acted emotion", sub: ["Unweighted accuracy (mean recall per emotion) and macro-F1 also reported; the ranking is the same"] },
      { t: "† = the dataset is in the model's training data", sub: ["emotion2vec+: CREMA-D, RAVDESS, TESS, JL-Corpus (EmoBox) · MERaLiON: CREMA-D · Melati: JL-Corpus · Vox-Profile: none", "A † score measures memory more than generalisation"] },
      { t: "Audio LLMs: forced choice", sub: ["System prompt: \"the words are neutral; judge the voice, not the words; answer with exactly one word from: angry, disgusted, …\"", "200 clips per corpus, sampled at even intervals over its emotions; an invalid answer is asked again once"] },
    ], { y: 1.5, h: 5.4, fontSize: 18 });
    L.notes(s, `Scoring details. Each model may have more classes than a dataset; we only let it choose among the dataset's emotions, so chance level is one over the number of emotions.
    The dagger is the most important annotation: it marks datasets the model has very probably seen during training. Those numbers are optimistic.
    For the LLMs we used a strict forced-choice prompt that tells them the words are neutral.`);
  }

  // ---- Headline figure (placeholder, to be replaced by the figure with GPT Realtime and Gemini)
  {
    const s = L.slide("4 · Results", "Dedicated SER models beat both audio LLMs on every dataset");
    L.image(s, "img/accuracy_by_dataset_PLACEHOLDER.png", MX, 1.4, W - 2 * MX, 5.55);
    L.text(s, "PLACEHOLDER: replace with the figure that includes GPT Realtime and Gemini", W - MX - 5.2, 1.3, 5.2, 0.3, { fontSize: 11, bold: true, color: C.bad, align: "right" });
    L.source(s, "Accuracy vs acted emotion (%). Hatched = dataset in the model's training data; dashed = chance. Open models: all clips; audio LLMs: 200 clips per corpus.");
    L.notes(s, `The headline figure. Blue is emotion2vec+ large, orange MERaLiON. Hatched bars mean the model trained on that dataset.
    Three things to see: (1) the two best open models are well above chance everywhere; (2) Gemini is above chance but far behind; GPT Realtime is near or below chance on four of six corpora; (3) the hatched bars are the tallest: training data matters a lot.`);
  }

  // ---- Table with all numbers
  {
    const s = L.slide("4 · Results", "All numbers: accuracy vs the acted emotion (%)");
    const hdr = ["Dataset", "Chance", ...MODELS.map((m) => MODEL[m].name + (["gpt", "gemini"].includes(m) ? " *" : ""))];
    const rows = ORDER.map((k) => {
      const d = ACC[k];
      const open = ["e2v", "meralion", "melati", "voxw", "voxl"], best = Math.max(...open.map((m) => d[m]));
      return [{ text: `${d.name}\n${d.lang}`, bold: true }, { text: d.chance.toFixed(1), options: { color: C.muted } },
        ...MODELS.map((m) => ({ text: d[m].toFixed(1) + (d.seen.includes(m) ? " †" : ""), bold: d[m] === best,
          options: { color: d[m] < d.chance + 3 ? C.bad : (d.seen.includes(m) ? C.muted : C.ink), align: "center" } }))];
    });
    const avg = (m) => (ORDER.reduce((a, k) => a + ACC[k][m], 0) / ORDER.length).toFixed(1);
    rows.push([{ text: "Mean", bold: true }, { text: (ORDER.reduce((a, k) => a + ACC[k].chance, 0) / 6).toFixed(1), options: { color: C.muted } },
      ...MODELS.map((m) => ({ text: avg(m), bold: true, options: { align: "center" } }))]);
    L.table(s, hdr, rows, { y: 1.45, colW: [1.7, 0.8, 1.35, 1.35, 1.25, 1.35, 1.35, 1.4, 1.38], fontSize: 13, rowH: 0.55 });
    L.text(s, "Bold = best open model · † = in the model's training data (grey) · red = within 3 points of chance or below · * 200 clips per corpus", MX, 5.95, W - 2 * MX, 0.3, { fontSize: 12, color: C.muted });
    L.bullets(s, [
      { runs: [{ text: "Unseen data only: ", options: { bold: true } }, { text: "MERaLiON is the most robust (65% mean on its 5 unseen corpora), emotion2vec+ drops to 51% (MCAE-SPPS, MESS)" }] },
    ], { y: 6.3, h: 0.6, fontSize: 16 });
    L.notes(s, `The full table. Read it column by column. emotion2vec+ is best on the four corpora it has seen, but on the two it has not seen, MCAE-SPPS and MESS, it falls to 61 and 41 percent.
    MERaLiON has seen only CREMA-D, and it is the best model on both unseen corpora. Melati is perfect on JL-Corpus, which it trained on, and around 50 percent elsewhere.
    Vox-Profile, trained only on natural podcast speech, is weak on acted data. GPT Realtime averages below the average chance level; Gemini is about 15 points above chance.`);
  }

  // ---- Seen vs unseen
  {
    const s = L.slide("4 · Results", "Training data explains most of the gap: seen corpora score 25–50 points higher");
    if (has("seen_vs_unseen.png")) L.image(s, fig("seen_vs_unseen.png"), MX, 1.45, 7.4, 5.3);
    else L.box(s, MX, 1.45, 7.4, 5.3, "[seen_vs_unseen.png]");
    L.bullets(s, [
      { t: "emotion2vec+ large", b: true, sub: ["seen (4 corpora): 89% · unseen (2): 51%"] },
      { t: "MERaLiON-SER-v1", b: true, sub: ["seen (CREMA-D): 85% · unseen (5): 65%"] },
      { t: "Melati", b: true, sub: ["seen (JL-Corpus): 99.5% · unseen (5): 48%"] },
      { t: "Consequence", b: true, color: C.accent2, sub: ["Published scores on popular corpora do not transfer: expect unseen-level accuracy on our calls until we fine-tune on them"] },
    ], { x: 8.3, y: 1.6, w: 4.45, h: 5.2, fontSize: 17 });
    L.notes(s, `Same data, regrouped: for each model, the mean accuracy on corpora it was trained on versus corpora it was not.
    The drop is 20 to 50 points. This is the single most useful number for planning: on our own calls, which no model has seen, we should expect the unseen-level numbers or lower, not the published ones. This is the main argument for fine-tuning on our own data.`);
  }

  // ---- Fair 4-emotion comparison
  {
    const s = L.slide("4 · Results", "On the four emotions every corpus shares, the top two models are close");
    const four = { // accuracy on angry/happy/neutral/sad only (reports/summary.md)
      crema_d: [93.2, 93.2, 66.5, 49.8, 44.0], mcae_spps: [87.8, 88.4, 66.2, 58.7, 57.1], mess: [40.8, 74.5, 40.4, 36.4, 36.3],
      ravdess: [92.5, 65.4, 64.2, 59.0, 59.3], tess: [100.0, 73.8, 75.4, 49.9, 56.4], jl_corpus: [94.7, 74.6, 99.7, 42.6, 48.2],
    };
    const ms = ["e2v", "meralion", "melati", "voxw", "voxl"];
    const rows = ORDER.map((k) => {
      const v = four[k], best = Math.max(...v);
      return [{ text: ACC[k].name, bold: true }, ...v.map((x, i) => ({ text: x.toFixed(1) + (ACC[k].seen.includes(ms[i]) ? " †" : ""), bold: x === best, options: { align: "center", color: ACC[k].seen.includes(ms[i]) ? C.muted : C.ink } }))];
    });
    L.table(s, ["Dataset (chance 25%)", ...ms.map((m) => MODEL[m].name)], rows, { y: 1.5, w: 8.2, colW: [2.0, 1.3, 1.3, 1.1, 1.25, 1.25], fontSize: 13, rowH: 0.5 });
    L.bullets(s, [
      "Scored only on angry, happy, neutral, sad; the model chooses among these 4",
      "Mandarin (MCAE-SPPS, unseen by all): emotion2vec+ 88%, MERaLiON 88%",
      "MESS stays hard for emotion2vec+ (41%): it calls almost every MESS clip neutral (next slides)",
      "Most errors on 7 classes come from fearful, disgusted and surprised",
    ], { x: 9.1, y: 1.5, w: 3.65, h: 5.3, fontSize: 15 });
    L.notes(s, `To remove the effect of different label sets, we scored every corpus on the four emotions they all share. On this easier task the top two models are nearly tied on Mandarin, 88 percent, which neither has seen.
    MESS remains the outlier for emotion2vec+. The remaining difficulty on the full task comes mostly from the rarer, subtler classes: fear, disgust, surprise.`);
  }

  // ---- Audio LLMs and LISTEN
  {
    const s = L.slide("4 · Results", "Audio LLMs listen to words more than to voices: consistent with LISTEN (2025)");
    const rows = ORDER.map((k) => [ACC[k].name, ACC[k].chance.toFixed(1), { text: ACC[k].gpt.toFixed(1), options: { align: "center", color: ACC[k].gpt < ACC[k].chance ? C.bad : C.ink } }, { text: ACC[k].gemini.toFixed(1), options: { align: "center" } }, { text: Math.max(...["e2v", "meralion", "melati", "voxw", "voxl"].map((m) => ACC[k][m])).toFixed(1), options: { align: "center", color: C.muted } }]);
    L.table(s, ["Dataset", "Chance", "GPT Realtime", "Gemini 3.5 Flash", "Best open"], rows, { y: 1.5, w: 6.2, colW: [1.6, 1.0, 1.2, 1.3, 1.1], fontSize: 13, rowH: 0.48 });
    L.text(s, "red = below chance · 200 clips per corpus", MX, 5.05, 6, 0.3, { fontSize: 11, color: C.muted });
    L.bullets(s, A.listen || [
      { t: "Our result", b: true, sub: ["GPT Realtime: below chance on 4 of 6 corpora (mean 16% vs 18% chance)", "Gemini: above chance everywhere, but 15–50 points behind the best open model"] },
      { t: "LISTEN benchmark (arXiv 2510.10444)", b: true, sub: ["[findings to be filled in]"] },
    ], { x: 7.2, y: 1.5, w: 5.55, h: 5.3, fontSize: 16 });
    L.notes(s, A.listenNotes || `Audio LLMs are general assistants. When the words carry no emotion, as in these corpora by design, they do poorly.`);
  }

  // ---- Confusion matrices
  const cm = (key, title, bullets, notes) => {
    const s = L.slide("4 · Results", title);
    if (has(`cm_pair_${key}.png`)) L.image(s, fig(`cm_pair_${key}.png`), MX, 1.4, 8.6, 5.4);
    else L.box(s, MX, 1.4, 8.6, 5.4, `[cm_pair_${key}.png]`);
    L.bullets(s, bullets, { x: 9.45, y: 1.6, w: 3.3, h: 5.2, fontSize: 15 });
    L.source(s, "Rows = acted emotion, columns = prediction; each row sums to 100%. Left: emotion2vec+ large, right: MERaLiON-SER-v1.");
    L.notes(s, notes);
  };
  A.cms.forEach((c) => cm(c.key, c.title, c.bullets, c.notes));
  (A.afterCms || []).forEach((f) => f(L));
};
module.exports.ACC = ACC;

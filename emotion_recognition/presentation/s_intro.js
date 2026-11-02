// Title, outline, research objective (task definition, challenges, methodology)
module.exports = function (L) {
  const { C, W, H, MX, FONT } = L;
  const p = L.pptx;
  const SEC = "1 · Research Objective";

  // ---- Title
  {
    const s = p.addSlide();
    s.background = { color: C.white };
    s.addShape(p.ShapeType.rect, { x: MX, y: 2.2, w: 0.08, h: 2.2, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Speech Emotion Recognition", MX + 0.35, 2.1, 11, 0.8, { fontSize: 40, bold: true, color: C.ink });
    L.text(s, "Benchmarking open-source and closed audio models on public emotional speech corpora,\nand a roadmap for adaptation to call-centre audio", MX + 0.35, 2.95, 11.5, 1.0, { fontSize: 22, color: C.text, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  Project update  ·  November 2026", MX + 0.35, 4.15, 11, 0.4, { fontSize: 16, color: C.muted });
    L.text(s, "Audio clips from public research corpora, used under their licences (CREMA-D ODbL, RAVDESS CC BY-NC-SA, TESS and MCAE-SPPS CC BY-NC, MESS CC BY, JL-Corpus CC0).", MX, H - 0.7, W - 2 * MX, 0.4, { fontSize: 10, color: C.faint });
    L.notes(s, `Scope of this talk (about 35 minutes): results of benchmarking speech emotion recognition (SER) models on public emotional speech corpora, the main sources of difficulty in the task, and a plan to adapt the two strongest models to our own call-centre audio.
    Audio clips are embedded: click a blue play button to start or pause a clip.`);
  }

  // ---- Outline
  {
    const s = L.slide("", "Outline");
    const steps = [
      ["1", "Research Objective", "The speech emotion recognition (SER) task: definition, challenges and methodology"],
      ["2", "Public Datasets", "Six emotional speech corpora: design, audio examples and inter-rater agreement"],
      ["3", "Benchmarking Models", "Five open-source SER models and two audio LLMs: architectures and scale"],
      ["4", "Evaluation Results", "Accuracy across corpora, effect of training-data overlap and confusion analysis"],
      ["5", "Error Analysis", "Case studies of disagreement between models, acted labels and listeners"],
      ["6", "Future Work", "Adaptation of emotion2vec+ and MERaLiON-SER-v1; annotation and active learning"],
    ];
    steps.forEach(([n, t, d], i) => {
      const y = 1.6 + i * 0.86;
      s.addText(n, { x: MX, y, w: 0.6, h: 0.6, shape: p.ShapeType.ellipse, fill: { color: i === 5 ? C.accent2 : C.accent }, line: { color: C.white, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 0.85, y + 0.02, 3.4, 0.6, { fontSize: 20, bold: true, color: C.ink, valign: "middle" });
      L.text(s, d, MX + 4.3, y + 0.02, 7.8, 0.6, { fontSize: 17, color: C.text, valign: "middle" });
      if (i < 5) s.addShape(p.ShapeType.line, { x: MX + 0.3, y: y + 0.62, w: 0, h: 0.24, line: { color: C.line, width: 1.5 } });
    });
    L.notes(s, `The talk has six parts. Parts two to five present the benchmark: the corpora, the models, the quantitative results and a qualitative error analysis with audio. Part six presents the proposed adaptation and annotation plan.`);
  }

  // ---- Task definition
  {
    const s = L.slide(SEC, "Task Definition");
    const y0 = 1.65, bh = 1.75;
    L.box(s, MX, y0, 3.0, bh, "Input: speech clip\n1–5 s, 16 kHz mono\n\"Don't forget a jacket.\"", { fill: "EEF4FA", line: "9CC3E6", fontSize: 15 });
    L.arrow(s, MX + 3.05, y0 + bh / 2, MX + 3.75, y0 + bh / 2);
    L.box(s, MX + 3.8, y0, 3.0, bh, "SER model\n(or audio LLM)", { fill: C.ink, line: C.ink, color: C.white, bold: true, fontSize: 17 });
    L.arrow(s, MX + 6.85, y0 + bh / 2, MX + 7.55, y0 + bh / 2);
    // output panel of the same height
    const ox = MX + 7.6, ow = W - MX - ox;
    s.addShape(p.ShapeType.roundRect, { x: ox, y: y0, w: ow, h: bh, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    L.text(s, "Output: one probability per class", ox + 0.15, y0 + 0.06, ow - 0.3, 0.25, { fontSize: 12, bold: true, color: C.ink });
    const emos = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"];
    const probs = [0.02, 0.01, 0.03, 0.05, 0.12, 0.74, 0.03];
    emos.forEach((e, i) => {
      const y = y0 + 0.36 + i * 0.185;
      L.text(s, e, ox + 0.1, y, 1.15, 0.18, { fontSize: 10, color: C.text, align: "right", valign: "middle" });
      s.addShape(p.ShapeType.rect, { x: ox + 1.35, y: y + 0.03, w: 2.6 * probs[i] + 0.02, h: 0.12, fill: { color: L.EMO[e] }, line: { color: L.EMO[e], width: 0 } });
      L.text(s, (probs[i] * 100).toFixed(0) + "%", ox + 1.42 + 2.6 * probs[i], y, 0.55, 0.18, { fontSize: 9, color: C.muted, valign: "middle" });
    });
    L.bullets(s, [
      { t: "Utterance-level categorical classification into seven emotion classes", sub: ["angry · disgusted · fearful · happy · neutral · sad · surprised (the emotion2vec+ label set); the predicted label is the class of highest probability"] },
      { t: "Lexically neutral stimuli", sub: ["Each corpus uses the same sentences across all emotions, so the lexical content carries no emotional information; recognition must rely on paralinguistic cues (pitch, intensity, tempo, voice quality)"] },
      { t: "Reference labels", sub: ["Intended emotion: the emotion portrayed by the actor (all corpora)", "Perceived emotion: listener ratings per clip (CREMA-D and MCAE-SPPS), analysed in Part 2"] },
    ], { y: 3.75, h: 3.2, fontSize: 17 });
    L.notes(s, `The task is utterance-level classification: one clip in, one of seven labels out. Every model returns a probability per class and the class with the highest probability is the prediction.
    All corpora are lexically neutral by design: the same sentence is spoken in every emotion. A system that relies on the words therefore cannot exceed chance level; this is particularly relevant for audio LLMs, which are strong at text understanding.
    The reference label is the intended emotion of the actor. For two corpora we also have the perceived emotion from listener ratings.`);
  }

  // ---- Challenges
  {
    const s = L.slide(SEC, "Challenges in Speech Emotion Recognition");
    const cards = [
      ["Label subjectivity", "41.6%", "of CREMA-D clips: listeners' majority vote equals the acted emotion", ["No single ground truth", "Requires multiple raters and agreement statistics"], C.accent2],
      ["Acted vs spontaneous speech", "6 of 6", "evaluation corpora are acted, short and exaggerated", ["Call-centre speech is spontaneous, subtle and mostly neutral", "Benchmark accuracy is an upper bound"], C.accent],
      ["Domain shift", "8 kHz", "telephone bandwidth, vs 16–48 kHz studio recordings in the corpora", ["Also language, accent, noise and turn length", "Accuracy falls 20–52 points on unseen corpora"], C.accent],
      ["Training-data overlap", "4 of 6", "evaluation corpora are in the training data of emotion2vec+", ["Published scores are optimistic", "Every overlap is marked (†) in our results"], C.accent],
    ];
    const cw = 2.88, gap = 0.2;
    cards.forEach(([t, num, cap, pts, col], i) => {
      const x = MX + i * (cw + gap), y = 1.55, h = 5.3;
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      s.addText(String(i + 1), { x: x + 0.25, y: y + 0.32, w: 0.5, h: 0.5, shape: p.ShapeType.ellipse, fill: { color: col }, line: { color: col, width: 0 }, fontFace: FONT, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, x + 0.25, y + 0.95, cw - 0.45, 0.7, { fontSize: 18, bold: true, color: C.ink });
      L.text(s, num, x + 0.25, y + 1.7, cw - 0.45, 0.75, { fontSize: 38, bold: true, color: col });
      L.text(s, cap, x + 0.25, y + 2.45, cw - 0.45, 0.85, { fontSize: 13, color: C.text });
      s.addShape(p.ShapeType.line, { x: x + 0.25, y: y + 3.4, w: cw - 0.5, h: 0, line: { color: C.line, width: 1 } });
      L.bullets(s, pts, { x: x + 0.2, y: y + 3.55, w: cw - 0.35, h: 1.6, fontSize: 13, gap: 6 });
    });
    L.notes(s, `Four sources of difficulty, each with one number.
    First, label subjectivity: on CREMA-D the listeners' majority agrees with the acted emotion on only 42 percent of clips. There is no single ground truth, which we will hear in Part 2.
    Second, all six corpora are acted; call-centre speech is spontaneous and mostly neutral, so benchmark accuracy is an upper bound.
    Third, domain shift: telephone audio is band-limited to 8 kHz, and language, accent, noise and turn length also differ. On corpora outside their training data, the models lose 20 to 52 points of accuracy.
    Fourth, training-data overlap: four of our six corpora are in the training data of emotion2vec+, so its scores on them are optimistic. We mark every such case with a dagger.`);
  }

  // ---- Evaluation methodology
  {
    const s = L.slide(SEC, "Evaluation Methodology");
    const st = [
      ["Dataset survey", "37 public emotional speech corpora reviewed", "6 corpora selected: parallel text, validated labels, clear licence"],
      ["Model survey", "42 SER models and audio LLMs reviewed", "5 open models + 2 closed audio LLMs"],
      ["Benchmark", "Every model run as released (zero-shot) on identical clips", "Labels mapped to 7 shared classes"],
      ["Analysis", "Accuracy, unweighted accuracy, macro-F1, confusion matrices", "Listening-based error analysis"],
      ["Adaptation plan", "Fine-tuning strategy, annotation protocol, active learning", "Roadmap with decision gates"],
    ];
    const cw = 2.3, gap = 0.16, y = 1.6;
    st.forEach(([t, d, o], i) => {
      const x = MX + i * (cw + gap);
      L.box(s, x, y, cw, 0.7, `${i + 1}  ${t}`, { shape: i === 0 ? p.ShapeType.rect : p.ShapeType.chevron, fill: i === 4 ? C.accent2 : C.ink, line: i === 4 ? C.accent2 : C.ink, color: C.white, bold: true, fontSize: 14, r: 0 });
      s.addShape(p.ShapeType.rect, { x, y: y + 0.85, w: cw, h: 2.3, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.text(s, d, x + 0.12, y + 0.97, cw - 0.24, 1.15, { fontSize: 13, color: C.ink });
      L.text(s, o, x + 0.12, y + 2.15, cw - 0.24, 0.95, { fontSize: 12, color: C.accent, italic: true });
    });
    const tiles = [["6", "corpora"], ["21,357", "clips per model"], ["7", "models"], ["3", "metrics"]];
    tiles.forEach(([n, c], i) => {
      const x = MX + i * 3.08;
      L.text(s, n, x, 4.95, 2.9, 0.75, { fontSize: 38, bold: true, color: C.accent, align: "center" });
      L.text(s, c, x, 5.7, 2.9, 0.35, { fontSize: 14, color: C.text, align: "center" });
    });
    L.text(s, "Open models: all clips, run locally · audio LLMs: 200 clips per corpus through their APIs · predictions and probabilities stored per clip; an HTML viewer supports listening to any prediction", MX, 6.3, W - 2 * MX, 0.5, { fontSize: 13, color: C.muted, align: "center" });
    L.notes(s, `The methodology in five stages. We surveyed public corpora and models, selected six corpora and seven models, and ran every model as released on exactly the same clips with the same label mapping. The analysis combines standard metrics with listening to individual clips, and the final stage is the adaptation plan.
    The open models were run locally on all 21,357 clips. GPT Realtime and Gemini were run through their APIs on 200 clips per corpus, sampled at even intervals over the emotions.`);
  }
};

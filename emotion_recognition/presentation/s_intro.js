// Title, agenda, motivation, challenges, approach
module.exports = function (L) {
  const { C, W, H, MX, FONT } = L;
  const p = L.pptx;

  // ---- Title
  {
    const s = p.addSlide();
    s.background = { color: C.white };
    s.addShape(p.ShapeType.rect, { x: MX, y: 2.2, w: 0.08, h: 2.2, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Speech Emotion Recognition", MX + 0.35, 2.1, 11, 0.8, { fontSize: 40, bold: true, color: C.ink });
    L.text(s, "Benchmarking open-source and closed audio models on public emotional speech,\nand a path to our own call-centre data", MX + 0.35, 2.95, 11.5, 1.0, { fontSize: 22, color: C.text, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  Project update  ·  November 2026", MX + 0.35, 4.15, 11, 0.4, { fontSize: 16, color: C.muted });
    L.text(s, "Audio clips from public research corpora, used under their licences (CREMA-D ODbL, RAVDESS CC BY-NC-SA, TESS and MCAE-SPPS CC BY-NC, MESS CC BY, JL-Corpus CC0).", MX, H - 0.7, W - 2 * MX, 0.4, { fontSize: 10, color: C.faint });
    L.notes(s, `Goal of this talk (about 35 minutes): what we learned from benchmarking speech emotion recognition (SER) models on public data, why the task is harder than it looks, and a concrete plan to adapt the two best models to our own calls.
    Audio clips are embedded: click a blue play button to listen.`);
  }

  // ---- Agenda / storyline
  {
    const s = L.slide("", "Today's story");
    const steps = [
      ["1", "The task", "What we ask the models to do, and why it is hard"],
      ["2", "Data", "Six public emotional speech corpora: what they sound like, and how much humans disagree"],
      ["3", "Models", "Five open SER models and two audio LLMs: architecture and scale"],
      ["4", "Results", "Accuracy on every dataset, the effect of training data, confusion patterns"],
      ["5", "Listening to errors", "Clips where models and labels disagree: who is right?"],
      ["6", "Next steps", "Fine-tuning emotion2vec+ and MERaLiON on our calls; labelling with active learning"],
    ];
    steps.forEach(([n, t, d], i) => {
      const y = 1.6 + i * 0.86;
      s.addText(n, { x: MX, y, w: 0.6, h: 0.6, shape: p.ShapeType.ellipse, fill: { color: i === 5 ? C.accent2 : C.accent }, line: { color: C.white, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 0.85, y + 0.02, 3.2, 0.6, { fontSize: 20, bold: true, color: C.ink, valign: "middle" });
      L.text(s, d, MX + 4.1, y + 0.02, 8.0, 0.6, { fontSize: 17, color: C.text, valign: "middle" });
      if (i < 5) s.addShape(p.ShapeType.line, { x: MX + 0.3, y: y + 0.62, w: 0.001, h: 0.24, line: { color: C.line, width: 1.5 } });
    });
    L.notes(s, `Six parts. The middle of the talk is about listening: datasets first (to calibrate our ears), then the models, the numbers, and finally clips where the models fail.
    The last part is the proposal: which two models to adapt, how, with how much data and compute, and how to label our own data.`);
  }

  // ---- The task
  {
    const s = L.slide("1 · The task", "The task: hear one clip, name one of seven emotions");
    // pipeline: clip -> model -> 7 probabilities
    L.box(s, MX, 1.9, 2.6, 1.3, "Speech clip\n1–5 s, 16 kHz mono\n\"Don't forget a jacket.\"", { fill: "EEF4FA", line: "9CC3E6", fontSize: 14 });
    L.arrow(s, MX + 2.65, 2.55, MX + 3.35, 2.55);
    L.box(s, MX + 3.4, 1.9, 2.6, 1.3, "SER model\n(or audio LLM)", { fill: C.ink, line: C.ink, color: C.white, bold: true, fontSize: 16 });
    L.arrow(s, MX + 6.05, 2.55, MX + 6.75, 2.55);
    const emos = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"];
    const probs = [0.02, 0.01, 0.03, 0.05, 0.12, 0.74, 0.03];
    emos.forEach((e, i) => {
      const y = 1.72 + i * 0.24;
      L.text(s, e, MX + 6.85, y, 1.2, 0.22, { fontSize: 11, color: C.text, align: "right" });
      s.addShape(p.ShapeType.rect, { x: MX + 8.15, y: y + 0.03, w: 2.6 * probs[i] + 0.02, h: 0.16, fill: { color: L.EMO[e] }, line: { color: L.EMO[e], width: 0 } });
      L.text(s, (probs[i] * 100).toFixed(0) + "%", MX + 8.25 + 2.6 * probs[i], y, 0.6, 0.22, { fontSize: 10, color: C.muted });
    });
    L.text(s, "one probability per class → predicted label = the largest", MX + 6.85, 3.45, 5, 0.3, { fontSize: 12, color: C.muted, italic: true });
    L.bullets(s, [
      { t: "Categorical SER: seven classes shared by most corpora and models", sub: ["angry · disgusted · fearful · happy · neutral · sad · surprised (the emotion2vec+ label set)"] },
      { t: "We test the voice, not the words", sub: ["All six corpora use the same neutral sentences in every emotion, so the text gives no clue; models must use tone, pitch, loudness and tempo"] },
      { t: "Ground truth = the emotion the actor was asked to portray", sub: ["Two corpora also have listener votes, a second, perceptual ground truth (part 2)"] },
    ], { y: 4.0, h: 3.0, fontSize: 18 });
    L.notes(s, `Speech emotion recognition here means classification: one clip in, one of seven labels out. Every model gives a probability per class; we take the highest.
    Key design choice of the benchmark: parallel sentences. The same sentence is spoken in every emotion, so a model that only reads the words cannot do better than chance. This matters especially for audio LLMs, which are strong at text.
    The ground truth is the acted (intended) emotion. For CREMA-D and MCAE-SPPS we also have what listeners perceived.`);
  }

  // ---- Why it's hard
  {
    const s = L.slide("1 · The task", "Why speech emotion recognition is harder than it looks");
    const cards = [
      ["Subjective labels", "There is no single truth. Listeners disagree with each other and with the actor: on CREMA-D the listeners' majority matches the actor on only about 4 in 10 clips (part 2).", C.accent2],
      ["Acted ≠ real", "Public corpora are acted, short and exaggerated. Real customers are spontaneous, mostly neutral, with subtle and mixed emotion.", C.accent],
      ["Domain shift", "New languages and accents, 8 kHz telephone channel, background noise, long turns. Models degrade outside the data they were trained on.", C.accent],
      ["Leaky benchmarks", "Popular corpora are inside the training data of popular models, so their published scores are optimistic. We mark every such case (†).", C.accent],
    ];
    cards.forEach(([t, d, col], i) => {
      const x = MX + (i % 2) * 6.15, y = 1.6 + Math.floor(i / 2) * 2.65, w = 5.95, h = 2.4;
      s.addShape(p.ShapeType.rect, { x, y, w, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: 0.07, h, fill: { color: col }, line: { color: col, width: 0 } });
      L.text(s, t, x + 0.3, y + 0.22, w - 0.5, 0.45, { fontSize: 20, bold: true, color: C.ink });
      L.text(s, d, x + 0.3, y + 0.8, w - 0.55, h - 1.0, { fontSize: 16, color: C.text, ls: 1.1 });
    });
    L.notes(s, `Four reasons. The first is the most important and we will hear it in the clips: emotion is subjective. Even the "ground truth" is uncertain.
    Second, the public data is acted; our calls are not. Third, domain shift: language, accent, telephone audio. Fourth, benchmark leakage: many models were trained on the very corpora people evaluate them on, which inflates scores. We tracked this for every model and dataset.`);
  }

  // ---- What we did (approach)
  {
    const s = L.slide("1 · The task", "What we did: one benchmark, every model on the same clips");
    const steps = [
      ["Survey datasets", "37 public emotional\nspeech corpora"],
      ["Select 6", "parallel text, good labels,\nclear licence"],
      ["Survey models", "42 SER models\nand audio LLMs"],
      ["Evaluate 7 models", "5 open (local)\n2 closed (API)"],
      ["Analyse", "metrics, confusion,\nlistening viewer"],
      ["Plan", "fine-tuning\n+ labelling"],
    ];
    const bw = 1.85, gap = 0.17;
    steps.forEach(([t, d], i) => {
      const x = MX + i * (bw + gap);
      L.box(s, x, 1.9, bw, 0.6, t, { fill: i === 5 ? C.accent2 : C.ink, line: i === 5 ? C.accent2 : C.ink, color: C.white, bold: true, fontSize: 14 });
      L.text(s, d, x, 2.6, bw, 0.7, { fontSize: 13, color: C.text, align: "center" });
      if (i < 5) L.arrow(s, x + bw + 0.01, 2.2, x + bw + gap - 0.01, 2.2, { width: 1.5 });
    });
    L.stat(s, "21,357", "clips per model across the 6 corpora\n(all clips; audio LLMs: 200 per corpus)", MX, 3.8, 3.8);
    L.stat(s, "7", "models: emotion2vec+ large, MERaLiON-SER-v1, Melati, Vox-Profile ×2, GPT Realtime, Gemini", MX + 4.15, 3.8, 4.0);
    L.stat(s, "3", "metrics: accuracy (vs acted label), unweighted accuracy (mean recall), macro-F1", MX + 8.45, 3.8, 3.7);
    L.bullets(s, [
      "Every model runs as released (zero-shot, no tuning on these corpora); labels mapped to the same 7 classes",
      "Reproducible: one YAML per script, predictions and probabilities saved per clip, an HTML viewer to listen to examples",
    ], { y: 5.45, h: 1.4, fontSize: 16 });
    L.notes(s, `The workflow. We surveyed public datasets and models, selected six corpora and seven models, and ran every model on exactly the same clips with the same label mapping.
    All open models run locally; GPT Realtime and Gemini are API models and were run on 200 clips per corpus, sampled evenly per emotion, to keep cost down.
    Everything is saved per clip, so we can go back and listen to any prediction.`);
  }
};

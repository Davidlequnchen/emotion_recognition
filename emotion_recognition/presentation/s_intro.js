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
    L.text(s, "Chen Lequn  ·  Completed benchmark  ·  15 November 2026", MX + 0.35, 4.15, 11, 0.4, { fontSize: 16, color: C.muted });
    L.text(s, "Audio clips from public research corpora, used under their licences (CREMA-D ODbL, RAVDESS CC BY-NC-SA, TESS and MCAE-SPPS CC BY-NC, MESS CC BY, JL-Corpus CC0).", MX, H - 0.7, W - 2 * MX, 0.4, { fontSize: 10, color: C.faint });
    L.notes(s, `Scope of this talk (about 35 minutes): results of benchmarking speech emotion recognition (SER) models on public emotional speech corpora, the main sources of difficulty in the task, and a proposed local adaptation pilot with emotion2vec+ and MERaLiON on independently labelled call-centre audio.
    Audio clips are embedded: click a blue play button to start or pause a clip.`);
  }

  // ---- Outline
  {
    const s = L.slide("", "Outline");
    const steps = [
      ["1", "Research Objective", "The speech emotion recognition (SER) task: definition, challenges and methodology"],
      ["2", "Public Datasets", "Sixteen datasets; six selected listening examples and inter-rater agreement"],
      ["3", "Benchmarking Models", "Five dedicated SER models and five audio LLMs: protocol and architectures"],
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
    L.box(s, MX, y0, 3.0, bh, "Input: one utterance\nresampled per model\n\"Don't forget a jacket.\"", { fill: "EEF4FA", line: "9CC3E6", fontSize: 15 });
    L.arrow(s, MX + 3.05, y0 + bh / 2, MX + 3.75, y0 + bh / 2);
    L.box(s, MX + 3.8, y0, 3.0, bh, "SER model\n(or audio LLM)", { fill: C.ink, line: C.ink, color: C.white, bold: true, fontSize: 17 });
    L.arrow(s, MX + 6.85, y0 + bh / 2, MX + 7.55, y0 + bh / 2);
    // output panel of the same height
    const ox = MX + 7.6, ow = W - MX - ox;
    s.addShape(p.ShapeType.roundRect, { x: ox, y: y0, w: ow, h: bh, rectRadius: 0.08, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    L.text(s, "SER example: class probabilities", ox + 0.15, y0 + 0.06, ow - 0.3, 0.25, { fontSize: 12, bold: true, color: C.ink });
    const emos = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"];
    const probs = [0.02, 0.01, 0.03, 0.05, 0.12, 0.74, 0.03];
    emos.forEach((e, i) => {
      const y = y0 + 0.36 + i * 0.185;
      L.text(s, e, ox + 0.1, y, 1.15, 0.18, { fontSize: 10, color: C.text, align: "right", valign: "middle" });
      s.addShape(p.ShapeType.rect, { x: ox + 1.35, y: y + 0.03, w: 2.6 * probs[i] + 0.02, h: 0.12, fill: { color: L.EMO[e] }, line: { color: L.EMO[e], width: 0 } });
      L.text(s, (probs[i] * 100).toFixed(0) + "%", ox + 1.42 + 2.6 * probs[i], y, 0.55, 0.18, { fontSize: 9, color: C.muted, valign: "middle" });
    });
    L.bullets(s, [
      { t: "Utterance-level emotion classification with dataset-specific candidate labels", sub: ["Seven canonical labels; only the dataset's candidates are eligible. SER uses candidate argmax; audio LLMs generate a label that is parsed and scored"] },
      { t: "Mixed recording and label sources", sub: ["Parallel-sentence corpora probe voice cues; four dialogue-source datasets also contain semantic cues. Each utterance is evaluated without dialogue history"] },
      { t: "Reference labels", sub: ["Primary reference: each dataset's documented category labels, with explicit mappings", "Listener labels are additional references where supplied; neither source is silently substituted"] },
    ], { y: 3.75, h: 3.2, fontSize: 17 });
    L.notes(s, `One saved prediction is evaluated for each utterance, without surrounding dialogue or video. The candidate set follows the dataset, drawn from seven canonical emotions. Dedicated SER selects the largest candidate-class score. Audio LLMs answer the current neutral baseline prompt; their parsed answer is the primary prediction and invalid outputs stay errors. The probability plot is a SER illustration, not an assertion that every API exposes calibrated class probabilities. The benchmark includes both parallel-sentence and dialogue-source recordings, and documented reference labels differ.`);
  }

  // ---- Challenges
  {
    const s = L.slide(SEC, "Challenges in Speech Emotion Recognition");
    const cards = [
      ["Label subjectivity", "41.6%", "of CREMA-D clips: listeners' majority vote equals the acted emotion", ["No single ground truth", "Requires multiple raters and agreement statistics"], C.accent2],
      ["Recording sources", "4 of 16", "datasets retain dialogue structure; the rest are standalone utterances", ["Dialogue sources also include performed speech", "Public scores do not establish internal-call accuracy"], C.accent],
      ["Domain shift", "8 kHz", "telephone bandwidth, vs 16–48 kHz studio recordings in the corpora", ["Also language, accent, noise and turn length", "Measure transfer on independently labelled target audio"], C.accent],
      ["Training provenance", "†", "marks documented involvement in the training process", ["Unmarked means exposure not established", "Corpus overlap does not prove test-clip leakage"], C.accent],
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
    L.notes(s, `The benchmark covers sixteen datasets with different recording sources and label constructs. Four retain dialogue structure, but each model receives only the current utterance; performed dialogue is not necessarily spontaneous. Class imbalance, recording conditions and training exposure all affect interpretation. Daggers denote documented training-process involvement, including upstream lineage. Unmarked cases are not certified unseen; overlap does not prove that a specific test clip leaked. Public scores are not an upper bound or a forecast for internal call audio.`);
  }

  // ---- Evaluation methodology
  {
    const s = L.slide(SEC, "Evaluation Methodology");
    const st = [
      ["Dataset survey", "37 public emotional speech corpora reviewed", "16 adopted datasets; documented labels, cohort and licences"],
      ["Model survey", "42 SER models and audio LLMs reviewed", "5 dedicated SER + 3 open audio LLMs + 2 API audio LLMs"],
      ["Benchmark", "Every model run as released (zero-shot) on identical clips", "Dataset-specific candidates and composite-label scoring"],
      ["Analysis", "Accuracy, UAR, macro-F1, weighted-F1 and confusion matrices", "Listening-based error analysis"],
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
    const tiles = [["16", "datasets"], ["67,062", "clips per model"], ["10", "models"], ["160", "complete runs"]];
    tiles.forEach(([n, c], i) => {
      const x = MX + i * 3.08;
      L.text(s, n, x, 4.95, 2.9, 0.75, { fontSize: 38, bold: true, color: C.accent, align: "center" });
      L.text(s, c, x, 5.7, 2.9, 0.35, { fontSize: 14, color: C.text, align: "center" });
    });
    L.text(s, "All models: the same full evaluation cohort · current baseline prompt · invalid labels remain errors · per-clip results and source hashes retained", MX, 6.3, W - 2 * MX, 0.5, { fontSize: 13, color: C.muted, align: "center" });
    L.notes(s, `The completed benchmark contains ten models on sixteen public datasets, 67,062 clips per model and 160 current-protocol runs. Every model uses the same dataset-specific cohort and accepted-label rules, including JL-Corpus excited. API runs use the complete cohort, not an early 200-clip sample. This refresh imports final Gemma outputs and regenerates metrics, reports and figures without new inference. The adaptation section is a proposal, not evidence that target-domain performance has been achieved.`);
  }
};

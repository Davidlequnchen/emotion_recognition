// Models: landscape, two families, emotion2vec+ (architecture, training), MERaLiON, Melati / Vox-Profile, audio LLMs
module.exports = function (L, A) {
  const { C, W, H, MX, FONT, MODEL } = L;
  const p = L.pptx;
  const SEC = "3 · Models";

  // ---- Landscape
  {
    const s = L.slide(SEC, A.surveyTitle || "From a survey of SER models to seven we evaluated");
    const rows = [
      ["e2v", "SJTU, Alibaba et al.", "2024-05", "data2vec 2.0-style encoder", "162.1M", "42.5k h pseudo-labelled (+ EmoBox)", "9", "commercial OK"],
      ["meralion", "A*STAR I²R, Singapore", "2025-10", "Whisper-medium enc. + ECAPA-TDNN", "313.4M", "CREMA-D, ESD, MELD, M3ED, SG speech", "7 + VAD", "commercial OK"],
      ["melati", "Scicom, Malaysia", "2026-10", "Whisper large-v3 enc. (frozen) + MLP", "637.3M", "9 acted corpora (28.9k clips)", "7", "non-commercial"],
      ["voxw", "USC SAIL et al.", "2025-05", "Whisper large-v3 enc. (LoRA inactive)", "639.9M", "MSP-Podcast (natural)", "9", "non-commercial"],
      ["voxl", "USC SAIL et al.", "2025-05", "WavLM-large (fully fine-tuned)", "316.2M", "MSP-Podcast (natural)", "9", "non-commercial"],
      ["gpt", "OpenAI", "2025-08", "speech-to-speech LLM", "undisclosed", "undisclosed", "prompted", "API"],
      ["gemini", "Google", A.geminiDate || "2026", "multimodal LLM", "undisclosed", "undisclosed", "prompted", "API"],
    ];
    const tbl = rows.map((r) => [{ text: MODEL[r[0]].name, bold: true, options: { color: C.ink } }, ...r.slice(1).map((c, j) => ({ text: c, options: { align: j === 3 || j === 5 ? "center" : "left", color: c === "non-commercial" ? C.bad : C.ink } }))]);
    L.table(s, ["Model", "Organisation", "Released", "Architecture", "Params", "Emotion training data", "Classes", "Licence"], tbl, { y: 1.45, colW: [1.85, 1.6, 0.9, 2.45, 1.0, 2.35, 0.95, 1.03], fontSize: 12, rowH: 0.5 });
    // colour chips
    rows.forEach((r, i) => s.addShape(p.ShapeType.rect, { x: MX - 0.14, y: 1.45 + 0.5 * (i + 1) + 0.1, w: 0.08, h: 0.3, fill: { color: MODEL[r[0]].color }, line: { color: MODEL[r[0]].color, width: 0 } }));
    L.bullets(s, A.surveyBullets || [
      "Selection: downloadable weights + runnable code, English supported, output classes close to our 7; the whole emotion2vec family considered",
      "Dedicated SER models: 0.16–0.64B parameters; open audio LLMs: 5.5–35B; closed LLMs: size not published",
    ], { y: 5.6, h: 1.3, fontSize: 15 });
    L.notes(s, `The seven models. Five open models that we run locally, from 162 million to 640 million parameters, and two closed audio LLMs through their APIs.
    Note the licence column: only emotion2vec+ and MERaLiON allow commercial use. And note the training data: that is where the daggers in the results come from.`);
  }

  // ---- Two families
  {
    const s = L.slide(SEC, "Two families: a speech encoder with a classifier, or a general audio LLM");
    // family 1
    L.text(s, "Dedicated SER model (5 open models)", MX, 1.5, 6, 0.4, { fontSize: 18, bold: true, color: C.accent });
    const y1 = 2.05, bh = 0.95;
    const f1 = [["Waveform", 1.2, C.panel, C.line], ["Pre-trained speech encoder\ndata2vec · Whisper · WavLM", 2.9, "EEF4FA", C.accent], ["Pooling\nover time", 1.3, "EEF4FA", C.accent], ["Classifier\nhead", 1.3, "E2F2EA", C.good], ["P(7 emotions)", 1.6, C.panel, C.line]];
    let x = MX;
    f1.forEach(([t, w, f, l], i) => { L.box(s, x, y1, w, bh, t, { fill: f, line: l, fontSize: 14, lw: 1.25 }); if (i < f1.length - 1) L.arrow(s, x + w + 0.02, y1 + bh / 2, x + w + 0.28, y1 + bh / 2); x += w + 0.3; });
    L.text(s, "Encoder learns speech from 1,000s of hours (self-supervised or ASR); head trained on labelled emotional speech. Small (0.16–0.64B), fast, runs on a laptop, can be fine-tuned.", MX, y1 + 1.1, 9.6, 0.6, { fontSize: 14, color: C.text });
    // family 2
    L.text(s, "Audio large language model (GPT Realtime, Gemini)", MX, 4.0, 8, 0.4, { fontSize: 18, bold: true, color: C.accent2 });
    const y2 = 4.55;
    L.box(s, MX, y2, 1.2, 0.95, "Waveform", { fontSize: 14 });
    L.box(s, MX, y2 + 1.05, 1.2, 0.6, "Text prompt", { fontSize: 13, fill: "FFF6E0", line: "E6C619" });
    L.arrow(s, MX + 1.22, y2 + 0.47, MX + 1.48, y2 + 0.47);
    L.box(s, MX + 1.5, y2, 2.0, 0.95, "Audio encoder", { fill: "FCE9D9", line: C.accent2, fontSize: 14, lw: 1.25 });
    L.arrow(s, MX + 3.52, y2 + 0.47, MX + 3.78, y2 + 0.47);
    L.arrow(s, MX + 1.22, y2 + 1.35, MX + 3.8, y2 + 0.9);
    L.box(s, MX + 3.8, y2, 2.9, 0.95, "LLM decoder\n(billions of parameters)", { fill: "FCE9D9", line: C.accent2, fontSize: 14, lw: 1.25 });
    L.arrow(s, MX + 6.72, y2 + 0.47, MX + 6.98, y2 + 0.47);
    L.box(s, MX + 7.0, y2, 1.2, 0.95, "\"sad\"\n(text)", { fontSize: 14 });
    L.arrow(s, MX + 8.22, y2 + 0.47, MX + 8.48, y2 + 0.47);
    L.box(s, MX + 8.5, y2, 1.25, 0.95, "Parser →\n1 label", { fontSize: 13 });
    L.text(s, "Generalist: trained mostly to understand and answer. Emotion is a side skill; no probabilities, no fine-tuning, data sent to a cloud API.", MX, y2 + 1.75, 9.6, 0.6, { fontSize: 14, color: C.text });
    // right column
    L.box(s, 10.55, 1.5, 2.2, 5.3, "", { fill: C.panel, line: C.panel });
    L.text(s, "What it means for us", 10.7, 1.65, 2.0, 0.4, { fontSize: 14, bold: true, color: C.ink });
    L.text(s, "Dedicated models give calibrated probabilities per clip, run on-premises, and can be adapted to our calls.\n\nLLMs are convenient, but we cannot inspect or adapt them, and audio leaves the company.", 10.7, 2.1, 1.95, 4.5, { fontSize: 13, color: C.text, ls: 1.1 });
    L.notes(s, `Two very different kinds of model. The five open models share one recipe: a large pre-trained speech encoder, pooling over time, and a small classification head trained on emotion labels. They output a probability per class.
    The audio LLMs take the audio plus a text instruction and generate text, which we parse into a label. They are generalists, they give no probabilities, and they cannot be fine-tuned by us.`);
  }

  // ---- emotion2vec+ architecture (measured)
  {
    const s = L.slide(SEC, "emotion2vec+ large: a CNN front end and 12 Transformer blocks, 162.1M params");
    L.image(s, "img/e2v_arch.png", MX, 1.35, W - 2 * MX, 5.65);
    L.notes(s, `emotion2vec+ large, measured from the released weights. A seven-layer CNN turns the waveform into one 512-dimensional vector every 20 milliseconds; a projection to 1024 dimensions; a convolutional positional encoding; then twelve Transformer blocks with 16 heads, which hold 151 of the 162 million parameters.
    The frame features are averaged over time and a single linear layer maps them to nine classes. Note that the model card says 300 million; the file has 162 million, because the depth is halved compared with data2vec 2.0 Large. The 1.95 GB file is mostly optimizer state.`);
  }

  // ---- emotion2vec+ training lineage
  {
    const s = L.slide(SEC, "How emotion2vec+ learned emotion: self-supervision, then pseudo-labels");
    const st = [
      ["emotion2vec", "self-supervised pre-training\n(data2vec 2.0 style)", "262 h unlabelled\nemotional speech", "no labels"],
      ["+ seed", "fine-tuned on academic\nlabelled data (EmoBox)", "201 h", "human labels"],
      ["+ base", "trained on filtered\npseudo-labels", "4,788 h", "model labels"],
      ["+ large", "trained on filtered\npseudo-labels (≈40k of 160k h)", "42,526 h", "model labels"],
    ];
    const bw = 2.75, gap = 0.37;
    st.forEach(([n, d, h, lab], i) => {
      const x = MX + i * (bw + gap);
      L.box(s, x, 1.6, bw, 0.6, n, { fill: i === 3 ? MODEL.e2v.color : C.ink, line: i === 3 ? MODEL.e2v.color : C.ink, color: C.white, bold: true, fontSize: 16 });
      L.text(s, d, x, 2.3, bw, 0.7, { fontSize: 13, color: C.text, align: "center" });
      const hh = [0.45, 0.4, 1.0, 2.2][i];
      s.addShape(p.ShapeType.rect, { x: x + bw / 2 - 0.5, y: 5.6 - hh, w: 1.0, h: hh, fill: { color: i === 3 ? MODEL.e2v.color : "9CC3E6" }, line: { color: C.white, width: 0 } });
      L.text(s, h, x, 5.65, bw, 0.55, { fontSize: 14, bold: true, color: C.ink, align: "center" });
      L.text(s, lab, x, 6.2, bw, 0.3, { fontSize: 12, color: C.muted, align: "center" });
      if (i < 3) L.arrow(s, x + bw + 0.03, 1.9, x + bw + gap - 0.03, 1.9);
    });
    L.text(s, "hours of training audio (bar height not to scale)", MX, 3.15, 6, 0.3, { fontSize: 12, color: C.muted, italic: true });
    L.bullets(s, [
      "EmoBox includes CREMA-D, RAVDESS, TESS and JL-Corpus → the † on these corpora",
      "Sources of the 42.5k h pseudo-labelled data and the training code are not published",
    ], { x: MX, y: 6.5, h: 0.6, fontSize: 13, noBullet: false });
    L.notes(s, `How emotion2vec+ was trained, and why it matters for fine-tuning. First, self-supervised pre-training on 262 hours of unlabelled emotional speech. Then three generations: seed was fine-tuned on about 200 hours of academic labelled data from EmoBox; base and large were trained on pseudo-labelled data, 4.8 thousand and 42.5 thousand hours.
    Two consequences: the model has already seen four of our six corpora, and its features are already emotion features, which is why freezing it and training a small head is a strong first step.`);
  }

  // ---- MERaLiON / Melati / Vox-Profile diagrams (from facts.md)
  (A.archSlides || []).forEach((f) => f(L));

  // ---- Audio LLM protocol
  {
    const s = L.slide(SEC, "How we asked GPT Realtime and Gemini");
    L.box(s, MX, 1.55, 2.3, 1.0, "Audio clip\n(Realtime: PCM16, 24 kHz)", { fontSize: 13 });
    L.arrow(s, MX + 2.32, 2.05, MX + 2.78, 2.05);
    L.box(s, MX + 2.8, 1.55, 2.6, 1.0, "System + user prompt\n(below)", { fill: "FFF6E0", line: "E6C619", fontSize: 14 });
    L.arrow(s, MX + 5.42, 2.05, MX + 5.88, 2.05);
    L.box(s, MX + 5.9, 1.55, 2.2, 1.0, "GPT Realtime /\nGemini 3.5 Flash", { fill: "FCE9D9", line: C.accent2, fontSize: 14, bold: true });
    L.arrow(s, MX + 8.12, 2.05, MX + 8.58, 2.05);
    L.box(s, MX + 8.6, 1.55, 1.6, 1.0, "one word", { fontSize: 14 });
    L.arrow(s, MX + 10.22, 2.05, MX + 10.68, 2.05);
    L.box(s, MX + 10.7, 1.55, 1.43, 1.0, "label\n(retry once\nif invalid)", { fontSize: 12 });
    s.addText([
      { text: "System prompt (abridged)\n", options: { bold: true, fontSize: 13, color: C.muted, breakLine: true } },
      { text: "\"You label the emotion in speech recordings for a research study. Each clip is acted speech from an emotion research dataset: an actor says a sentence whose words are neutral while portraying one emotion, so the emotion is in the voice (tone, pitch, loudness, tempo), not in the words. … Judge the voice, not the words … Answer with exactly one word from this list, in lowercase, and nothing else: angry, disgusted, fearful, happy, neutral, sad, surprised.\"", options: { fontSize: 15, color: C.ink, italic: true } },
    ], { x: MX, y: 2.9, w: 7.6, h: 3.2, fontFace: FONT, fill: { color: C.panel }, margin: 12, valign: "top" });
    L.bullets(s, [
      "200 clips per corpus, sampled at even intervals over its emotions",
      "The prompt tells the model that the words carry no emotion: the most favourable setting we could give it",
      "Answers parsed leniently (\"Anger.\" → angry); invalid answers asked again once",
      "Data went to cloud APIs: only public corpora were sent",
    ], { x: 8.5, y: 2.9, w: 4.25, h: 3.9, fontSize: 15 });
    L.notes(s, `For the LLMs we built a forced-choice protocol. The audio goes in with a system prompt that explains the task and explicitly says the words are neutral, so judge the voice. The model must answer with exactly one of the seven words. We parse leniently and retry once if the answer is not a label.
    This is a favourable setup for the LLMs. Only public corpora were sent to the APIs; no company audio.`);
  }
};

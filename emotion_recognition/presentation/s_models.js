// Models: landscape, two families, emotion2vec+ (architecture, training), MERaLiON, Melati / Vox-Profile, audio LLMs
module.exports = function (L, A) {
  const { C, W, H, MX, FONT, MODEL } = L;
  const p = L.pptx;
  const SEC = "3 · Benchmarking Models";

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
    const s = L.slide(SEC, "Model Families: Dedicated SER Models vs Audio LLMs");
    L.text(s, "Dedicated SER model (5 open models)", MX, 1.45, 6, 0.4, { fontSize: 18, bold: true, color: C.accent });
    const y1 = 1.95, bh = 0.95;
    const f1 = [["Waveform", 1.2, C.panel, C.line], ["Pre-trained speech encoder\ndata2vec · Whisper · WavLM", 2.9, "EEF4FA", C.accent], ["Pooling\nover time", 1.3, "EEF4FA", C.accent], ["Classifier\nhead", 1.3, "E2F2EA", C.good], ["P(7 emotions)", 1.6, C.panel, C.line]];
    let x = MX;
    f1.forEach(([t, w, f, l], i) => { L.box(s, x, y1, w, bh, t, { fill: f, line: l, fontSize: 14, lw: 1.25 }); if (i < f1.length - 1) L.arrow(s, x + w + 0.02, y1 + bh / 2, x + w + 0.28, y1 + bh / 2); x += w + 0.3; });
    L.text(s, "Encoder pre-trained on thousands of hours of speech (self-supervised or speech recognition); pooling and head trained on labelled emotional speech. 0.16–0.64B parameters; runs on-premises; can be fine-tuned.", MX, y1 + 1.05, 9.6, 0.6, { fontSize: 14, color: C.text });
    L.text(s, "Audio large language model (GPT Realtime, Gemini)", MX, 3.85, 8, 0.4, { fontSize: 18, bold: true, color: C.accent2 });
    const y2 = 4.35;
    L.box(s, MX, y2, 1.2, 0.95, "Waveform", { fontSize: 14 });
    L.arrow(s, MX + 1.22, y2 + 0.47, MX + 1.48, y2 + 0.47);
    L.box(s, MX + 1.5, y2, 2.0, 0.95, "Audio encoder", { fill: "FCE9D9", line: C.accent2, fontSize: 14, lw: 1.25 });
    L.arrow(s, MX + 3.52, y2 + 0.47, MX + 3.78, y2 + 0.47);
    L.box(s, MX + 3.8, y2, 2.9, 0.95, "LLM decoder\n(billions of parameters)", { fill: "FCE9D9", line: C.accent2, fontSize: 14, lw: 1.25 });
    L.box(s, MX + 4.35, y2 + 1.25, 1.8, 0.5, "Text prompt", { fontSize: 13, fill: "FFF6E0", line: "E6C619" });
    L.arrow(s, MX + 5.25, y2 + 1.23, MX + 5.25, y2 + 0.97);
    L.arrow(s, MX + 6.72, y2 + 0.47, MX + 6.98, y2 + 0.47);
    L.box(s, MX + 7.0, y2, 1.2, 0.95, "\"sad\"\n(text)", { fontSize: 14 });
    L.arrow(s, MX + 8.22, y2 + 0.47, MX + 8.48, y2 + 0.47);
    L.box(s, MX + 8.5, y2, 1.25, 0.95, "Parser →\n1 label", { fontSize: 13 });
    L.text(s, "General-purpose model trained to understand and respond; emotion recognition is not its training objective. No class probabilities, no fine-tuning by us, audio processed by a cloud API.", MX, y2 + 1.9, 9.6, 0.6, { fontSize: 14, color: C.text });
    L.box(s, 10.55, 1.45, 2.2, 5.35, "", { fill: C.panel, line: C.panel });
    L.text(s, "Implications", 10.7, 1.6, 2.0, 0.4, { fontSize: 14, bold: true, color: C.ink });
    L.text(s, "Dedicated models return class probabilities, run on-premises and can be adapted to call-centre audio.\n\nAudio LLMs are convenient, but cannot be inspected or adapted, and require sending audio to an external service.", 10.7, 2.05, 1.95, 4.6, { fontSize: 13, color: C.text, ls: 1.1 });
    L.notes(s, `Two families of models. All five open models follow the same three-stage structure: a pre-trained speech encoder turns the waveform into a sequence of feature vectors, one every 20 milliseconds; a pooling stage summarises that sequence into one vector for the whole clip; and a small classification head maps the clip vector to class probabilities. The models differ in which encoder they use (data2vec-style for emotion2vec+, Whisper for MERaLiON, Melati and Vox-Profile Whisper, WavLM for Vox-Profile WavLM), in how they pool, and in how large the head is. The next slides map each model onto this structure.
    The audio LLMs take the audio together with a text instruction and generate text, which we parse into a label. They are general-purpose models, give no probabilities and cannot be fine-tuned by us.`);
  }

  // ---- emotion2vec+ architecture (simplified, with rationale)
  {
    const s = L.slide(SEC, "emotion2vec+ large: Architecture");
    const y0 = 1.95, bh = 1.05;
    const blocks = [
      ["Waveform\n16 kHz", 1.05, C.panel, C.line, ""],
      ["CNN feature\nextractor\n7 conv layers", 1.45, "EEF4FA", C.accent, "4.2M"],
      ["Projection\n512 → 1024", 1.2, "EEF4FA", C.accent, "0.5M"],
      ["Conv positional\nencoding", 1.35, "EEF4FA", C.accent, "6.2M"],
      ["+ 10 extra\ntokens", 1.05, "EEF4FA", C.accent, "0.01M"],
      ["Transformer\n12 blocks\n1024-d, 16 heads", 1.6, "EEF4FA", C.accent, "151.2M"],
      ["Mean over\ntime", 1.05, "E8F1EC", C.good, "0"],
      ["Linear\n1024 → 9\n+ softmax", 1.2, "E2F2EA", C.good, "9k"],
    ];
    let x = MX; const pos = [];
    blocks.forEach(([t, w, f, l, prm], i) => {
      L.box(s, x, y0, w, bh, t, { fill: f, line: l, fontSize: 12, lw: 1.25 });
      L.text(s, prm, x, y0 + bh + 0.04, w, 0.25, { fontSize: 11, color: C.muted, align: "center" });
      pos.push([x, w]);
      if (i < blocks.length - 1) L.arrow(s, x + w + 0.01, y0 + bh / 2, x + w + 0.13, y0 + bh / 2, { width: 1.25 });
      x += w + 0.14;
    });
    const brace = (i0, i1, label, col) => {
      const xa = pos[i0][0], xb = pos[i1][0] + pos[i1][1];
      s.addShape(p.ShapeType.line, { x: xa, y: y0 - 0.12, w: xb - xa, h: 0, line: { color: col, width: 2 } });
      L.text(s, label, xa, y0 - 0.45, xb - xa, 0.3, { fontSize: 12, bold: true, color: col, align: "center" });
    };
    brace(1, 5, "Pre-trained speech encoder · 162.1M parameters in total", C.accent);
    brace(6, 6, "Pooling", C.good);
    brace(7, 7, "Head", C.good);
    L.bullets(s, [
      { runs: [{ text: "CNN feature extractor: ", options: { bold: true } }, { text: "learns its own filter bank from the raw waveform; 16,000 samples per second become one 512-d feature vector every 20 ms (a learned spectrogram)" }] },
      { runs: [{ text: "Projection and convolutional positional encoding: ", options: { bold: true } }, { text: "widen each frame to 1,024 dimensions and mix it with its neighbours (±0.9 s), so the Transformer knows the order of frames without fixed position numbers" }] },
      { runs: [{ text: "Extra tokens: ", options: { bold: true } }, { text: "10 learned vectors placed before the frames; in pre-training they summarise the whole utterance; removed before pooling at inference" }] },
      { runs: [{ text: "Transformer encoder: ", options: { bold: true } }, { text: "self-attention lets every frame use information from all other frames, so each frame representation reflects the context of the whole clip (93% of the parameters)" }] },
      { runs: [{ text: "Mean over time and linear head: ", options: { bold: true } }, { text: "the frames are averaged into one 1,024-d clip embedding; one linear layer maps it to 9 class probabilities" }] },
      { runs: [{ text: "Input length: ", options: { bold: true } }, { text: "variable, no fixed window and no padding; memory grows with the square of the length; the authors recommend VAD-segmented utterances (training crops ≤ 20 s)" }] },
    ], { y: 3.5, h: 3.45, fontSize: 14, gap: 4 });
    L.source(s, "Measured from the released model.pt and FunASR 1.4.16. Detailed diagram with tensor shapes: Appendix A3.");
    L.notes(s, `emotion2vec+ large maps onto the three-stage structure as follows: the pre-trained encoder is everything from the CNN to the Transformer; pooling is a plain mean over time; the head is a single linear layer.
    The CNN feature extractor is seven one-dimensional convolution layers applied to the raw waveform. Think of it as a learned spectrogram: it slides over the audio and produces one 512-number summary every 20 milliseconds, about 50 per second.
    The projection widens each of these vectors to 1,024 numbers. Transformers have no built-in notion of order, so the convolutional positional encoding mixes every frame with its neighbours within about 0.9 seconds on either side; the model thus knows which frames are adjacent. Because this is relative rather than a fixed table of positions, there is no maximum input length.
    The ten extra tokens are learned vectors put in front of the sequence. During self-supervised pre-training they are trained to summarise the whole utterance; at inference they are simply removed before pooling, so they do not affect the prediction directly.
    The Transformer encoder is twelve blocks of self-attention: each frame looks at all other frames and updates its representation with the context of the whole clip. This is where the model learns what emotional speech sounds like over time, and where 151 of the 162 million parameters are.
    Finally the frame vectors are averaged into one clip vector, and one linear layer maps it to nine probabilities.
    Input length: emotion2vec+ processes the real length of the clip, with no padding and no fixed window. The practical limit is memory, which grows with the square of the length because of self-attention; on this Mac, clips longer than about 20 seconds must run on the CPU. The authors recommend cutting long audio into utterances with voice activity detection.`);
  }

  // ---- emotion2vec+ training lineage
  {
    const s = L.slide(SEC, "emotion2vec+ large: Training Pipeline");
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

  // ---- Common structure and input handling of the five open models
  {
    const s = L.slide(SEC, "Dedicated SER Models: Structure and Input Handling");
    const rows = [
      ["emotion2vec+ large", "data2vec 2.0-style: CNN + 12-block Transformer", "mean over frames", "Linear → 9", "real length", "no padding", "no hard limit; memory ∝ length²; VAD segments recommended"],
      ["MERaLiON-SER-v1", "Whisper-medium (speech recognition) + LoRA", "ECAPA-TDNN + attentive pooling", "MLP → 7 (+ V/A/D)", "fixed 30 s", "zero-padded to 30 s; padding included in pooling", "first 30 s only"],
      ["Melati", "Whisper large-v3, frozen", "mean per layer + learned layer weights", "MLP → 7", "fixed 30 s", "zero-padded to 30 s; pooled over real frames only", "cut at 30 s"],
      ["Vox-Profile Whisper", "Whisper large-v3 (LoRA inactive)", "1×1 conv + mean", "MLP → 9", "fixed 15 s", "zero-padded to 15 s; trained on 3–15 s, < 3 s unreliable", "first 15 s only"],
      ["Vox-Profile WavLM", "WavLM-large, fully fine-tuned", "1×1 conv + mean", "MLP → 9", "real length", "no padding; trained on 3–15 s", "card: at most 15 s"],
    ];
    L.table(s, ["Model", "Pre-trained encoder", "Pooling", "Head", "Input window", "Short clips", "Long clips"], rows.map((r) => r.map((c, j) => ({ text: c, bold: j === 0, options: { color: j === 4 && c.startsWith("fixed") ? C.accent2 : C.ink } }))),
      { y: 1.4, colW: [1.6, 2.25, 1.75, 1.15, 1.0, 2.45, 1.93], fontSize: 10, rowH: 0.56 });
    L.bullets(s, [
      { runs: [{ text: "Padding = digital silence: ", options: { bold: true } }, { text: "Whisper-based models append zeros (not noise) up to their window; all clips in our corpora are 0.4–5.3 s, so they are always padded and never cut" }] },
      { runs: [{ text: "Cost: ", options: { bold: true } }, { text: "a 2 s clip costs a full 30 s (or 15 s) window for Whisper-based models; emotion2vec+ and WavLM compute only the real length" }] },
      { runs: [{ text: "Call-centre audio: ", options: { bold: true } }, { text: "long turns must be segmented (VAD) into pieces shorter than the window, and the segment predictions combined" }] },
    ], { y: 5.55, h: 1.4, fontSize: 13, gap: 3 });
    L.notes(s, `This table maps all five open models onto the same three stages, encoder, pooling and head, and summarises how each handles input length.
    Two models, emotion2vec+ and Vox-Profile WavLM, process the real length of the clip. The three Whisper-based models, MERaLiON, Melati and Vox-Profile Whisper, always process a fixed window, 30 or 15 seconds. Shorter clips are padded with zeros, which is digital silence rather than white noise; longer audio is cut, and only the first window is used.
    MERaLiON includes the padded silence in its pooling, so for a 4-second clip, 26 of the 30 seconds the head sees are silence. Melati pools only over the real frames. Since all clips in our corpora are at most about 5 seconds long, no clip was ever cut, but every clip was padded for the Whisper-based models.
    For call-centre audio, long turns will have to be segmented with voice activity detection into pieces shorter than the window.`);
  }

  // ---- Audio LLM protocol
  {
    const s = L.slide(SEC, "Audio LLM Evaluation Methodology");
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

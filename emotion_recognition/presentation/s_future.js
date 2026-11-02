// Next steps: shortlist, fine-tuning, data and compute, labelling, active learning, roadmap, summary
module.exports = function (L, A) {
  const { C, W, H, MX, FONT, MODEL } = L;
  const p = L.pptx;
  const SEC = "6 · Next steps";

  // ---- Shortlist
  {
    const s = L.slide(SEC, "Shortlist for adaptation: emotion2vec+ large and MERaLiON-SER-v1");
    const rows = [
      ["Accuracy, corpora not in training", "51% (2 corpora)", "65% (5 corpora)"],
      ["Accuracy, all 6 corpora (mean)", "76% (4 seen)", "68% (1 seen)"],
      ["Parameters", "162.1M", A.mer?.params || "313.4M"],
      ["Backbone", "data2vec 2.0-style encoder (CNN + 12 Transformer blocks)", "Whisper-medium encoder + ECAPA-TDNN"],
      ["Classes", "9 (includes our 7)", "exactly our 7 + arousal, valence, dominance"],
      ["Languages in training", "not published (multilingual)", "English (incl. Singapore), Mandarin, Malay, Tamil"],
      ["Input length", "any length (memory grows with length²)", "fixed 30 s window (padded / cut)"],
      ["Licence", "FunASR model licence: commercial use allowed", "MERaLiON Public Licence: commercial use allowed"],
      ["Fine-tuning code", "none official; small PyTorch loop", A.mer?.code || "none official; small PyTorch loop"],
    ];
    L.table(s, ["", MODEL.e2v.name, MODEL.meralion.name], rows.map((r) => [{ text: r[0], bold: true }, r[1], r[2]]), { y: 1.4, colW: [3.1, 4.5, 4.53], fontSize: 12, rowH: 0.4 });
    L.text(s, "Melati and Vox-Profile are excluded: non-commercial licences, and weaker on unseen data. The audio LLMs are excluded: far less accurate, and no control over the model.", MX, 6.35, W - 2 * MX, 0.6, { fontSize: 14, color: C.text });
    L.notes(s, `Why these two. They are the two best models, they both allow commercial use, and they are complementary: emotion2vec+ is smaller and handles any clip length, and is strongest where it has seen similar data; MERaLiON generalises better to unseen corpora, matches our seven classes exactly and is multilingual.
    Melati and Vox-Profile are research-only licences. The LLMs are not competitive and are black boxes.`);
  }

  // ---- Domain gap
  {
    const s = L.slide(SEC, "Our calls are a new domain for every model, so we need our own labelled data");
    const rows = [
      ["Speech", "acted, exaggerated, one emotion per clip", "spontaneous, subtle, mixed and changing within a call"],
      ["Channel", "studio microphone, 16–48 kHz", "telephone, 8 kHz G.711, compression, noise"],
      ["Length", "1–5 s sentences", "long turns: need segmentation (VAD) into 3–10 s pieces"],
      ["Class balance", "balanced by design", "mostly neutral; angry or upset callers are rare but matter most"],
      ["Labels", "the actor's intended emotion", "none: must be created by listeners"],
    ];
    L.table(s, ["", "Public corpora (benchmark)", "Our customer calls"], rows.map((r) => [{ text: r[0], bold: true }, r[1], r[2]]), { y: 1.45, colW: [2.0, 4.9, 5.23], fontSize: 15, rowH: 0.58 });
    L.bullets(s, [
      "Unseen-corpus accuracy (51–65%) is the realistic starting point, before any channel or spontaneity effects",
      "Quick check before labelling: re-run the benchmark through a simulated telephone channel (G.711, already built for the deepfake work)",
    ], { y: 5.5, h: 1.45, fontSize: 17 });
    L.notes(s, `Everything that made the benchmark clean is different on our calls: spontaneous speech, the telephone channel, long turns, and a heavy neutral majority. And there are no labels.
    So the realistic expectation is the unseen-corpus accuracy or lower. A cheap first experiment is to pass the benchmark through a simulated G.711 telephone channel, which we already implemented for the anti-deepfake evaluation, to measure the channel effect alone.`);
  }

  // ---- Options ladder
  {
    const s = L.slide(SEC, "Retrain everything, or freeze most of it? Five options, from cheap to expensive");
    const rows = [
      ["A", "Train only a new linear head", "encoder frozen", "≈9k", "a few hundred – 5k", "minutes on CPU", "baseline; first step"],
      ["B", "Weighted sum of all layers + small MLP head", "encoder frozen", "≈0.26M", "1k – 20k", "minutes", "recommended start"],
      ["C", "LoRA on attention, or unfreeze top 2–4 blocks", "most of encoder frozen", "0.6M – 50M", "5k – 50k", "≈0.35–0.5 × D", "medium data, domain shift"],
      ["D", "All layers (keep CNN frozen)", "nothing but the CNN", "158M", "≥10k (20k–100k better)", "10k clips: 0.2–0.4 A100-h", "large data, A–C plateau"],
      ["E", "Continue self-supervised pre-training on unlabelled calls, then A–D", "—", "162M", "no labels, 10–1,000 h audio", "5–600 A100-h", "lots of audio, few labels"],
    ];
    L.table(s, ["", "What is trained", "Frozen", "Trainable (e2v+)", "Labelled clips", "Compute", "When"], rows.map((r, i) => r.map((c, j) => ({ text: c, bold: j === 0 || (i === 1 && j === 6), options: { color: i === 1 && j === 6 ? C.accent2 : C.ink } }))),
      { y: 1.4, colW: [0.45, 3.4, 1.7, 1.45, 1.85, 1.75, 1.53], fontSize: 12, rowH: 0.55 });
    L.bullets(s, [
      { runs: [{ text: "Why freeze first: ", options: { bold: true } }, { text: "emotion2vec+ already learned these classes from ≈40k hours, so its frozen features are emotion features; full fine-tuning mainly adapts to a new domain and needs ≥10k clips from many speakers" }] },
      { runs: [{ text: "Training from scratch is not realistic: ", options: { bold: true } }, { text: "self-supervised pre-training on 60k h (≈4,900 A100-h), then 42k h of pseudo-labelled data" }] },
    ], { y: 5.45, h: 1.5, fontSize: 15 });
    L.source(s, "Parameter counts measured on the released weights; data ranges inferred from published fine-tuning studies (Chen & Rudnicky 2021, Wang et al. 2021, PEFT-SER 2023, Wagner et al. 2023, SAILER 2025). Details: docs/finetuning_emotion2vec_plus_large_en.md");
    L.notes(s, `The core question: retrain everything or only the top? Our analysis says: start frozen. Option A trains a linear layer on embeddings extracted once; option B learns a weighted mix of all 12 layers plus a small MLP, which is what the strongest SER systems such as SAILER do.
    Only when we have around ten thousand labelled clips from many speakers, and the frozen options stop improving on a speaker-independent validation set, is it worth unfreezing: first with LoRA or the top blocks, then everything except the CNN.
    Compute is not the bottleneck at any of these sizes: one GPU is enough.`);
  }

  // ---- emotion2vec+ freeze diagram
  {
    const s = L.slide(SEC, "emotion2vec+ large: which layers to freeze and which to train");
    const y0 = 2.0, bh = 1.25;
    const blocks = [
      ["Waveform\n16 kHz", 1.1, "F4F6F9", C.line, C.ink, "—"],
      ["CNN feature\nextractor\n7 conv layers", 1.55, "E3E8EF", "8C99AA", C.ink, "4.2M"],
      ["Projection +\nconv positional\nencoding", 1.55, "E3E8EF", "8C99AA", C.ink, "6.8M"],
      ["Transformer\nblocks 1–8\n(1024-d, 16 heads)", 1.9, "E3E8EF", "8C99AA", C.ink, "100.8M"],
      ["Transformer\nblocks 9–12", 1.6, "FCE9D9", C.accent2, C.ink, "50.4M"],
      ["Pooling\nmean, or weighted\nsum of 12 layers", 1.55, "E2F2EA", C.good, C.ink, "0–12"],
      ["New head\nLinear / MLP\n→ K classes", 1.45, "E2F2EA", C.good, C.ink, "9k–0.26M"],
    ];
    let x = MX;
    blocks.forEach(([t, w, fill, line, col, prm], i) => {
      L.box(s, x, y0, w, bh, t, { fill, line, color: col, fontSize: 13, lw: 1.5 });
      L.text(s, prm, x, y0 + bh + 0.08, w, 0.3, { fontSize: 12, color: C.muted, align: "center" });
      if (i < blocks.length - 1) L.arrow(s, x + w + 0.01, y0 + bh / 2, x + w + 0.17, y0 + bh / 2, { width: 1.25 });
      x += w + 0.18;
    });
    // legend
    const lg = [["E3E8EF", "8C99AA", "frozen (all options)"], ["FCE9D9", C.accent2, "LoRA or unfrozen in C; all trained in D"], ["E2F2EA", C.good, "always trained (A, B, C, D)"]];
    lg.forEach(([f, l, t], i) => { const lx = MX + i * 4.1; s.addShape(p.ShapeType.roundRect, { x: lx, y: 3.95, w: 0.35, h: 0.25, rectRadius: 0.04, fill: { color: f }, line: { color: l, width: 1.5 } }); L.text(s, t, lx + 0.45, 3.93, 3.6, 0.3, { fontSize: 13, color: C.text }); });
    L.bullets(s, [
      { runs: [{ text: "Phase 1 (≈1–5k labels): ", options: { bold: true, color: C.good } }, { text: "B — freeze the whole encoder, cache the 12 layer outputs once, train a weighted layer sum + MLP (≈0.26M params). Minutes per run, so many runs for cross-validation" }] },
      { runs: [{ text: "Phase 2 (≈5–20k): ", options: { bold: true, color: C.accent2 } }, { text: "C — LoRA rank 8 on all 12 blocks (0.6–1.6M) or unfreeze blocks 9–12 (50M); encoder LR 1e-5–3e-5, head 1e-4–1e-3" }] },
      { runs: [{ text: "Phase 3 (≥20k, if C plateaus): ", options: { bold: true, color: C.ink } }, { text: "D — all but the CNN (158M); layer-wise LR decay; mix in public data to avoid forgetting" }] },
    ], { y: 4.45, h: 2.5, fontSize: 15 });
    L.notes(s, `Concretely for emotion2vec+. Grey parts stay frozen in every option: the CNN, the projection and positional encoding, and the lower eight Transformer blocks. Green parts are always trained: pooling and a new classification head. The orange top four blocks are the dial: LoRA or unfrozen in option C, fully trained in D.
    Phase one is option B: because the encoder is frozen we extract the features once and every training run takes minutes, which lets us do proper speaker-independent cross-validation.`);
  }

  // ---- MERaLiON freeze diagram (from facts.md)
  if (A.merFT) A.merFT(L);

  // ---- Data & compute
  {
    const s = L.slide(SEC, "How much data and compute: labels are the bottleneck, GPUs are not");
    const rows = [
      ["< 1k", "A", "seconds", "seconds"],
      ["1k", "A, B (C to try)", "1–3 min", A.mer?.c1k || "minutes"],
      ["10k", "B, C (D to try)", "0.2–0.4 A100-h", A.mer?.c10k || "≈1 A100-h"],
      ["100k", "C or D", "2–4 A100-h", A.mer?.c100k || "≈10 A100-h"],
    ];
    L.table(s, ["Labelled clips (5 s)", "Suitable options", "emotion2vec+ (D, 10 epochs)", "MERaLiON-SER-v1 (D, 10 epochs)"], rows.map((r) => r.map((c, j) => ({ text: c, bold: j === 0 }))), { y: 1.45, w: 8.0, colW: [1.9, 1.9, 2.1, 2.1], fontSize: 14, rowH: 0.55 });
    L.text(s, A.mer?.computeNote || "MERaLiON always computes a 30 s window, so a 5 s clip costs as much as a 30 s one.", MX, 4.35, 8.0, 0.6, { fontSize: 12, color: C.muted });
    L.stat(s, "1 GPU", "is enough for every option up to 100k clips (24 GB+). Repetition costs more: 5 folds × 3 LRs × 3 seeds = 45 runs", 9.0, 1.45, 3.75, { size: 34, capH: 1.2 });
    L.stat(s, "10–20k", "verified clips: the target for a reliable in-domain model (full fine-tuning beats frozen at ≈20k+)", 9.0, 3.6, 3.75, { size: 34, capH: 1.2, color: C.accent2 });
    L.bullets(s, [
      "Below ≈5k clips, full fine-tuning rarely beats a frozen encoder and overfits to the training speakers",
      "Learning curves flatten after a few thousand clips: decide with our own curve (train on 10 / 25 / 50 / 100%)",
      "Speaker-independent splits are essential: no caller in both training and test",
    ], { y: 5.0, h: 1.9, w: 8.2, fontSize: 15 });
    L.source(s, "emotion2vec+: measured FLOPs (21.9 GFLOP per s of audio) and M4 Max timings; A100 assumed 50–100 TFLOPS sustained, ×2 overhead. MERaLiON: estimated from the Whisper-medium encoder on a 30 s window.");
    L.notes(s, `Compute is cheap: a full fine-tune of emotion2vec+ on ten thousand clips is well under an A100-hour. MERaLiON is more expensive per clip because the Whisper encoder always processes a 30-second window, but still one GPU.
    The real cost is labelled data. Literature and our analysis point to ten to twenty thousand verified clips for a reliable in-domain model; below five thousand, keep the encoder frozen.`);
  }

  // ---- Labelling protocol
  {
    const s = L.slide(SEC, "Labelling our calls: four listeners per clip, majority vote");
    L.image(s, "img/labeler.png", 7.55, 1.4, 5.2, 4.3);
    L.text(s, "Our offline labelling tool (single HTML file, no install, nothing leaves the PC): random order per listener, keys 1–8, bad-clip option, optional model hints, CSV per listener", 7.55, 5.75, 5.2, 0.9, { fontSize: 11, color: C.muted });
    L.bullets(s, [
      { t: "Why four: one listener is not ground truth", sub: ["CREMA-D listeners agree with the actor on only ≈4 in 10 clips; with 4 votes we can measure agreement and keep ambiguity"] },
      { t: "Rule", sub: ["≥ 3 of 4 agree → label", "2–2 or all different → \"ambiguous\": keep the vote distribution as a soft label, or adjudicate", "any \"bad clip\" vote (noise, overlap, silence) → review and drop"] },
      { t: "Quality", sub: ["Inter-rater agreement (Fleiss' κ / Krippendorff's α) per round", "Gold clips repeated to check consistency; short calibration session with examples first"] },
      { t: "Bias control", sub: ["Seed set and test set: model hints off (predictions_mode \"off\")", "Verification rounds: hints on, to speed up checking"] },
    ], { x: MX, y: 1.45, w: 6.7, h: 5.4, fontSize: 15 });
    L.notes(s, `Labels are the bottleneck, so the protocol matters. Four listeners per clip, majority vote; ties are not forced into a class but kept as ambiguous with the vote distribution, which we can use as a soft training target.
    We built a small offline labelling tool for this: one HTML file, no installation, no network; each listener gets their own shuffled order and writes their own CSV. It can show model predictions as hints, which we switch off for the seed and test sets to avoid anchoring the listeners.`);
  }

  // ---- Active learning loop
  {
    const s = L.slide(SEC, "Scaling up with active learning: humans verify what the model is unsure about");
    const bw = 2.85, bh = 1.25, y1 = 1.55, y2 = 3.75;
    const xs = [MX, MX + 3.55, MX + 7.1];
    L.box(s, xs[0], y1, bw, bh, "① Seed labels\n1k clips × 4 listeners\nmajority vote", { fill: "EEF4FA", line: C.accent, lw: 1.5, fontSize: 14 });
    L.box(s, xs[1], y1, bw, bh, "② Fine-tune\nemotion2vec+ and MERaLiON\n(B first; C/D as data grows)", { fill: C.ink, line: C.ink, color: C.white, fontSize: 14 });
    L.box(s, xs[2], y1, bw, bh, "③ Predict\nunlabelled pool\n≈100k clips", { fill: "EEF4FA", line: C.accent, lw: 1.5, fontSize: 14 });
    L.box(s, xs[2], y2, bw, bh, "④ Select for review\nlow confidence · two models disagree\nrare emotions · random audit", { fill: "FCE9D9", line: C.accent2, lw: 1.5, fontSize: 13 });
    L.box(s, xs[1], y2, bw, bh, "⑤ Human verification\n2 listeners confirm or correct\n(3rd if they disagree)", { fill: "FCE9D9", line: C.accent2, lw: 1.5, fontSize: 13 });
    L.arrow(s, xs[0] + bw + 0.02, y1 + bh / 2, xs[1] - 0.02, y1 + bh / 2, { width: 2 });
    L.arrow(s, xs[1] + bw + 0.02, y1 + bh / 2, xs[2] - 0.02, y1 + bh / 2, { width: 2 });
    L.arrow(s, xs[2] + bw / 2, y1 + bh + 0.02, xs[2] + bw / 2, y2 - 0.02, { width: 2 });
    L.arrow(s, xs[2] - 0.02, y2 + bh / 2, xs[1] + bw + 0.02, y2 + bh / 2, { width: 2 });
    L.arrow(s, xs[1] + bw / 2, y2 - 0.02, xs[1] + bw / 2, y1 + bh + 0.02, { width: 2, color: C.accent2 });
    L.text(s, "add to training set,\nrepeat", xs[1] + bw / 2 + 0.1, y1 + bh + 0.2, 1.6, 0.6, { fontSize: 12, color: C.accent2, italic: true });
    // test set
    L.box(s, xs[0], y2, bw, bh, "Fixed test set\n≈500 clips, 4 listeners, separate callers\nnever trained on: score every round", { fill: C.white, line: C.ink, lw: 1.5, dash: "dash", fontSize: 13 });
    L.arrow(s, xs[0] + bw + 0.02, y2 + bh / 2 + 0.35, xs[1] - 0.02, y1 + bh - 0.05, { dash: "dash", width: 1.25 });
    // rounds table
    const rows = [["Round 0", "1k (4 listeners)", "1k", "B"], ["Round 1", "≈3k selected from 100k", "≈4k", "B, C"], ["Round 2", "≈5k", "≈9k", "C"], ["Round 3+", "≈5–10k", "10–20k", "C or D"]];
    L.table(s, ["", "New labels", "Total", "Option"], rows, { x: MX, y: 5.25, w: 6.6, colW: [1.2, 2.6, 1.4, 1.4], fontSize: 12, rowH: 0.32 });
    L.bullets(s, [
      "Verifying a suggested label is faster than labelling from scratch, and uncertain clips teach the model most",
      "Stop when the test-set score stops improving between rounds",
      "Two models give a free disagreement signal",
    ], { x: 7.5, y: 5.2, w: 5.25, h: 1.7, fontSize: 14 });
    L.notes(s, `This is the active learning loop. Round zero: one thousand clips, each labelled by four listeners, majority vote. Fine-tune both models. Run them on a large unlabelled pool, around one hundred thousand clips.
    Then, instead of labelling at random, select the clips the models are unsure about, the clips where the two models disagree, and clips predicted as rare emotions such as anger, plus a random sample to check calibration. Two listeners verify or correct those, and we retrain.
    A fixed test set, labelled by four listeners from callers never used for training, is scored after every round; we stop when it stops improving. Within three or four rounds we should reach the ten to twenty thousand verified clips the analysis asks for.`);
  }

  // ---- Roadmap
  {
    const s = L.slide(SEC, "Roadmap with decision points");
    const ph = [
      ["Step 1", "Prepare", ["Sample calls, VAD into 3–10 s customer turns", "Agree on label set (7 + unknown, or call-centre classes)", "Telephone-channel re-run of the benchmark"], "Go: label set and data access agreed"],
      ["Step 2", "Seed labels", ["1k clips × 4 listeners + 500-clip test set", "Measure listener agreement", "Zero-shot baseline of both models on the test set"], "Go: agreement acceptable (κ), baseline known"],
      ["Step 3", "First fine-tune", ["Option B for both models", "Speaker-independent cross-validation", "Compare with zero-shot"], "Go: clear gain over zero-shot"],
      ["Step 4", "Active learning", ["Rounds of predict → select → verify → retrain", "Move to C / D when ≥ 10k clips", "Track per-class recall, esp. angry"], "Stop: test score plateaus"],
    ];
    const cw = 2.9, gap = 0.18;
    ph.forEach(([n, t, items, gate], i) => {
      const x = MX + i * (cw + gap);
      L.box(s, x, 1.55, cw, 0.75, `${n}\n${t}`, { fill: i === 3 ? C.accent2 : C.ink, line: i === 3 ? C.accent2 : C.ink, color: C.white, bold: true, fontSize: 15, shape: p.ShapeType.homePlate });
      L.bullets(s, items, { x: x + 0.05, y: 2.5, w: cw - 0.1, h: 2.6, fontSize: 14 });
      L.box(s, x, 5.3, cw, 0.9, gate, { fill: C.panel, line: C.line, fontSize: 13, color: C.text });
    });
    L.text(s, "Decision gates", MX, 4.95, 3, 0.3, { fontSize: 12, bold: true, color: C.muted });
    L.notes(s, `Four steps, each with a gate. Step one is preparation, including the decision on the label set: we may want call-centre categories such as frustrated instead of disgusted or surprised; the labelling tool makes the label set a configuration change.
    Step two produces the seed labels and the test set, and tells us how much our own listeners agree. Step three is the first fine-tune. Step four is the active learning loop.`);
  }

  // ---- Summary
  {
    const s = L.slide("Summary", "Take-aways");
    const items = [
      ["Emotion labels are subjective", "Listeners disagree with actors and with each other: plan for multiple listeners and soft labels."],
      ["Open SER models clearly beat audio LLMs", "Best open model 65–99.5% per corpus vs Gemini 27–45% and GPT Realtime 2–29%; LLMs lean on words, not voice (as in LISTEN)."],
      ["Benchmark scores are inflated by training data", "Seen corpora score 25–50 points higher. On our calls expect unseen-level accuracy (≈50–65%) before adaptation."],
      ["emotion2vec+ large and MERaLiON-SER-v1 are the candidates", "Best accuracy, commercial licences, complementary strengths."],
      ["Adapt, don't retrain", "Freeze the encoder first (minutes per run); unfreeze only with ≥10k clips. One GPU suffices; labels are the cost."],
      ["Label smartly", "1k seed clips × 4 listeners, then active learning on ≈100k unlabelled clips to reach 10–20k verified labels."],
    ];
    items.forEach(([t, d], i) => {
      const y = 1.5 + i * 0.88;
      s.addText(String(i + 1), { x: MX, y: y + 0.05, w: 0.5, h: 0.5, shape: p.ShapeType.ellipse, fill: { color: i >= 4 ? C.accent2 : C.accent }, line: { color: C.white, width: 0 }, fontFace: FONT, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 0.75, y, 11.5, 0.38, { fontSize: 18, bold: true, color: C.ink });
      L.text(s, d, MX + 0.75, y + 0.38, 11.5, 0.42, { fontSize: 15, color: C.text });
    });
    L.notes(s, `Six take-aways to close.`);
  }

  // ---- Q&A
  {
    const s = p.addSlide();
    s.background = { color: C.white };
    L.text(s, "Thank you — questions?", MX, 2.8, W - 2 * MX, 1.0, { fontSize: 40, bold: true, color: C.ink, align: "center" });
    L.text(s, "Code, reports and listening viewer: emotion_recognition/ (README, reports/summary.md, outputs/viewer.html)", MX, 3.9, W - 2 * MX, 0.5, { fontSize: 16, color: C.muted, align: "center" });
  }
};

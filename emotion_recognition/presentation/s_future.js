// Next steps: shortlist, fine-tuning, data and compute, labelling, active learning, roadmap, summary
module.exports = function (L, A) {
  const { C, W, H, MX, FONT, MODEL } = L;
  const p = L.pptx;
  const SEC = "6 · Future Work";

  // ---- Shortlist
  {
    const s = L.slide(SEC, "Model Shortlist for Adaptation");
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
    const s = L.slide(SEC, "Domain Gap: Public Corpora vs Call-Centre Audio");
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

  // ---- Problem formulation
  {
    const s = L.slide(SEC, "Problem Formulation");
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.4, w: W - 2 * MX, h: 0.6, fill: { color: C.ink }, line: { color: C.ink, width: 0 } });
    L.text(s, "Semi-supervised domain adaptation with active learning", MX, 1.4, W - 2 * MX, 0.6, { fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle" });
    // source -> target
    L.box(s, MX, 2.25, 4.2, 1.45, "Source domain: public corpora\nacted speech · studio 16–48 kHz\nintended (acted) labels · balanced classes", { fill: "EEF4FA", line: C.accent, lw: 1.5, fontSize: 13 });
    L.arrow(s, MX + 4.25, 2.97, MX + 5.55, 2.97, { width: 2.25 });
    L.text(s, "adaptation", MX + 4.25, 2.6, 1.3, 0.3, { fontSize: 12, italic: true, color: C.muted, align: "center" });
    L.box(s, MX + 5.6, 2.25, 4.2, 1.45, "Target domain: call-centre audio\nspontaneous speech · telephone 8 kHz\nperceived labels (few) · mostly neutral", { fill: "FCE9D9", line: C.accent2, lw: 1.5, fontSize: 13 });
    L.box(s, MX + 10.0, 2.25, W - 2 * MX - 10.0, 1.45, "+ abundant unlabelled target audio", { fill: "EEE6F7", line: "8E4EC6", lw: 1.5, fontSize: 12, color: "5B2C83" });
    // three shifts
    L.text(s, "Three kinds of shift between the domains", MX, 3.9, 8, 0.3, { fontSize: 14, bold: true, color: C.ink });
    const sh = [
      ["Covariate shift", "input distribution: channel, accent, speaking style, noise"],
      ["Label shift", "class proportions: balanced in the corpora, mostly neutral in calls"],
      ["Annotation shift", "label meaning: intended emotion of an actor vs emotion perceived by listeners"],
    ];
    sh.forEach(([t, d], i) => {
      const x = MX + i * 4.1;
      s.addShape(p.ShapeType.rect, { x, y: 4.25, w: 3.9, h: 1.0, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.text(s, t, x + 0.15, 4.3, 3.6, 0.32, { fontSize: 14, bold: true, color: C.accent });
      L.text(s, d, x + 0.15, 4.62, 3.6, 0.6, { fontSize: 12, color: C.text });
    });
    // components
    const rows = [
      ["Supervised fine-tuning (A–D)", "few human-labelled target clips", "adapt the head, then more of the encoder"],
      ["Domain-adaptive pre-training (E)", "unlabelled target audio", "adapt the encoder to telephone speech without labels"],
      ["Self-training (E′)", "pseudo-labelled target audio", "add confident, agreeing model predictions as training data"],
      ["Active learning", "selection of clips for annotation", "spend listener time on the most informative clips"],
    ];
    L.table(s, ["Component", "Uses", "Purpose"], rows.map((r) => [{ text: r[0], bold: true }, r[1], r[2]]), { y: 5.4, colW: [3.4, 3.4, 5.33], fontSize: 11, rowH: 0.29 });
    L.notes(s, `In machine-learning terms, our plan is semi-supervised domain adaptation with active learning. The source domain is the public acted corpora on which the models were trained; the target domain is our call-centre audio, for which we will have only a limited number of labels, but plenty of unlabelled recordings.
    The two domains differ in three ways. Covariate shift: the input itself differs, telephone channel, accents, spontaneous speaking style and noise. Label shift: the class proportions differ, balanced in the corpora and dominated by neutral in calls. Annotation shift: the meaning of the label differs, the emotion an actor intended versus the emotion a listener perceives.
    The plan combines four standard components: supervised fine-tuning with the few labelled clips, options A to D; domain-adaptive pre-training on unlabelled call audio, option E; self-training with pseudo-labels, E prime; and active learning, which decides which clips our listeners should label.`);
  }

  // ---- Fine-tuning strategies: overview diagram
  {
    const s = L.slide(SEC, "Retraining / Fine-tuning Strategy: Overview");
    const FR = ["E3E8EF", "8C99AA"], TR = ["E2F2EA", C.good], LO = ["FCE9D9", C.accent2], SS = ["EEE6F7", "8E4EC6"];
    const layers = ["Pooling + head", "Encoder: upper blocks", "Encoder: lower blocks", "CNN / front end"];
    const opts = [
      ["A", "New linear head", [TR, FR, FR, FR], ["linear layer", "frozen", "frozen", "frozen"], "0.5k–5k", true],
      ["B", "Layer mix + MLP", [TR, FR, FR, FR], ["layer weights + MLP", "frozen", "frozen", "frozen"], "1k–5k", true],
      ["C", "LoRA / top blocks", [TR, LO, FR, FR], ["trained", "LoRA or unfrozen", "frozen", "frozen"], "≥ 5k", true],
      ["D", "Full fine-tuning", [TR, TR, TR, FR], ["trained", "trained", "trained", "frozen"], "≥ 10k", true],
      ["E", "Continued pre-training", [["F4F6F9", C.line], SS, SS, SS], ["then A, B or C", "self-supervised", "self-supervised", "self-supervised"], "none (unlabelled audio)", true],
    ];
    const TERM = { A: "linear probing", B: "layer-weighted probing", C: "parameter-efficient fine-tuning (PEFT)", D: "full fine-tuning", E: "domain-adaptive pre-training (DAPT)" };
    layers.forEach((t, j) => L.text(s, t, MX, 2.05 + j * 0.62, 1.75, 0.55, { fontSize: 12, color: C.muted, valign: "middle", align: "right" }));
    const cw = 1.95, gap = 0.18, x0 = MX + 1.9;
    opts.forEach(([k, name, cols, labs, need, ok], i) => {
      const x = x0 + i * (cw + gap);
      L.text(s, `${k}  ${name}`, x, 1.36, cw, 0.32, { fontSize: 14, bold: true, color: C.ink, align: "center", valign: "middle" });
      L.text(s, TERM[k], x, 1.67, cw, 0.34, { fontSize: 10, italic: true, color: C.accent, align: "center", valign: "top" });
      cols.forEach((c, j) => L.box(s, x, 2.05 + j * 0.62, cw, 0.55, labs[j], { fill: c[0], line: c[1], lw: 1.25, fontSize: 11, color: ok ? C.ink : C.faint }));
      L.text(s, "Human-labelled clips", x, 4.6, cw, 0.25, { fontSize: 10, color: C.muted, align: "center" });
      L.text(s, need, x, 4.83, cw, 0.32, { fontSize: 13, bold: true, color: C.ink, align: "center" });
      L.box(s, x + 0.25, 5.22, cw - 0.5, 0.34, ok ? "feasible" : "not feasible", { fill: ok ? (i === 1 ? C.accent2 : C.good) : C.white, line: ok ? (i === 1 ? C.accent2 : C.good) : C.bad, color: ok ? C.white : C.bad, bold: true, fontSize: 11, r: 0.17 });
    });
    L.text(s, "recommended start", x0 + (cw + gap) + 0.1, 5.58, cw - 0.2, 0.25, { fontSize: 10, color: C.accent2, align: "center", italic: true });
    const lg = [[FR, "frozen"], [TR, "trained with labels"], [LO, "adapted with labels (LoRA)"], [SS, "trained without labels"]];
    lg.forEach(([c, t], i) => { const lx = MX + i * 3.0; s.addShape(L.pptx.ShapeType.roundRect, { x: lx, y: 6.0, w: 0.35, h: 0.25, rectRadius: 0.04, fill: { color: c[0] }, line: { color: c[1], width: 1.25 } }); L.text(s, t, lx + 0.45, 5.98, 2.5, 0.3, { fontSize: 12, color: C.text }); });
    L.text(s, "A → B → C → D is a ladder: each step is a separate experiment on the same speaker-independent split, adopted only if it improves on the previous one; D once ≥ 10k labelled clips exist. E is complementary: it adapts the encoder with unlabelled call audio before A–D.", MX, 6.4, W - 2 * MX, 0.55, { fontSize: 13, color: C.ink });
    L.notes(s, `Five ways to adapt a pre-trained model, ordered from the cheapest to the most expensive. Each column is the same network drawn as four parts: the CNN front end at the bottom, the lower and upper encoder blocks, and pooling with the classification head at the top. Grey means frozen, green means trained with our labels, orange means adapted with small LoRA matrices, purple means trained without labels.
    A trains only a new final layer. B keeps the encoder frozen as well, but combines the outputs of all encoder layers and trains a small neural network on top. C additionally adapts the upper encoder blocks, either with LoRA or by unfreezing them. D trains the whole network.
    "Human-labelled clips" is the number of clips annotated by our listeners that each option needs to work reliably, including training, validation and test data.
    A, B, C and D form a ladder: we try A, then B, then C, each as a separate experiment on the same data split, and we move up only if the step improves the result. D trains the whole network; with our four H100 GPUs compute is not an obstacle, but it needs at least ten thousand labelled clips to beat the frozen options, so it is the last step. E is different: it continues the model's own label-free pre-training on our unlabelled call audio, of which we have plenty, and is then followed by A to D.`);
  }

  // ---- Fine-tuning strategies: details and feasibility
  {
    const s = L.slide(SEC, "Retraining / Fine-tuning Strategy: Details and Feasibility");
    const rows = [
      ["A", "Linear probing — keep the whole model fixed; compute one 1,024-d clip embedding per labelled clip once; train only a new linear layer (logistic regression) on these embeddings", "≈ 9k", "0.5k–5k", "minutes on a CPU", "feasible · baseline"],
      ["B", "Layer-weighted probing — encoder fixed; keep the outputs of all 12 Transformer layers (lower layers carry acoustic and prosodic detail, upper layers more abstract information), learn 12 mixing weights, then a small 2-layer neural network (MLP 1024 → 256 → K) classifies", "≈ 0.26M", "1k–5k", "minutes (features cached once)", "feasible · recommended start"],
      ["C", "Parameter-efficient fine-tuning (PEFT) — add small trainable low-rank matrices (LoRA) to the attention layers, or unfreeze only the top 2–4 blocks; the encoder itself adapts to call audio", "0.6–1.6M (LoRA); 25–50M (top blocks)", "≥ 5k", "0.1–0.2 A100-h ≈ 0.03–0.1 H100-h per 10k clips × 10 epochs", "feasible once ≈ 5k clips exist"],
      ["D", "Full fine-tuning — train all layers except the CNN, with a low learning rate and layer-wise decay", "158M", "≥ 10k, better 20k–100k", "0.2–0.4 A100-h ≈ 0.07–0.2 H100-h per 10k clips × 10 epochs", "feasible on 4 × H100 once ≥ 10k clips exist"],
      ["E", "Domain-adaptive pre-training (DAPT) — continue the label-free pre-training objective (data2vec: predict the model's own representations of masked audio) on unlabelled call audio; then apply A, B or C", "162M (no labels)", "none; 10–1,000 h of unlabelled audio", "5–10 A100-h ≈ 2–5 H100-h (10–20 h of audio); 600 A100-h ≈ 200–300 H100-h (1,000 h)", "feasible: abundant unlabelled audio"],
    ];
    L.table(s, ["", "What is done", "Trainable parameters (emotion2vec+)", "Human-labelled clips", "Compute (one GPU; H100 ≈ 2–3 × A100)", "Feasibility"], rows.map((r, i) => r.map((c, j) => ({ text: c, bold: j === 0 || j === 5, options: { color: j === 5 ? (i === 1 ? C.accent2 : C.good) : C.ink, valign: "middle" } }))),
      { y: 1.35, colW: [0.4, 5.0, 1.65, 1.5, 1.95, 1.63], fontSize: 10, rowH: [0.4, 0.6, 0.8, 0.6, 0.6, 0.8] });
    L.bullets(s, [
      { runs: [{ text: "Human-labelled clips: ", options: { bold: true } }, { text: "clips annotated with the 4-listener protocol, split by speaker into training, validation and test sets" }] },
      { runs: [{ text: "Why start frozen: ", options: { bold: true } }, { text: "emotion2vec+ was already trained on ≈40k hours of emotion data, so its frozen features are emotion features; full fine-tuning rarely beats a frozen encoder below ≈5k clips and overfits to the training speakers" }] },
      { runs: [{ text: "E caveat: ", options: { bold: true } }, { text: "no official emotion2vec pre-training code; the objective must be rebuilt from fairseq data2vec 2.0. For MERaLiON (Whisper, trained on transcripts) unlabelled audio is used through pseudo-labelling instead" }] },
    ], { y: 5.75, h: 1.3, fontSize: 12, gap: 2 });
    L.source(s, "Parameter counts measured on the released weights; data ranges inferred from published studies (Chen & Rudnicky 2021, Wang et al. 2021, PEFT-SER 2023, Wagner et al. 2023, SAILER 2025); compute estimated (docs/finetuning_emotion2vec_plus_large_en.md).");
    L.notes(s, `The same five options in detail.
    Option A, a linear probe: the pre-trained model is not changed at all. We pass every labelled clip through it once, keep the 1,024-number clip embedding, and train a logistic regression on these numbers. It needs only a few hundred clips per class and trains in minutes on a laptop CPU.
    Option B is our recommended starting point. The encoder stays frozen, but instead of using only the output of the last Transformer layer we keep the outputs of all twelve layers. Lower layers carry more acoustic and prosodic detail, upper layers more abstract information, and for emotion the best information is often in the middle. The model learns twelve weights to mix the layers, and a small two-layer neural network, an MLP, classifies the mixture. Because the encoder is frozen, the layer outputs are computed once and every training run takes minutes, which allows proper cross-validation.
    Option C lets the encoder itself adapt to call audio, either by adding small LoRA matrices to the attention layers, about one million parameters, or by unfreezing the top two to four blocks. It needs a pass through the encoder at every training step and more data, about five thousand clips.
    Option D trains everything except the CNN. Compute is not a constraint with four H100 GPUs: ten thousand clips for ten epochs take minutes. Published studies show it pays off only with about ten thousand labelled clips or more, so it comes after C, once active learning has produced enough labels.
    Option E uses no labels: we continue the self-supervised pre-training objective on our own unlabelled call audio, which we have in abundance, so the encoder learns what telephone speech sounds like. Afterwards we apply A, B or C. The caveat is that the original pre-training code is not public and must be rebuilt from fairseq. For MERaLiON, whose Whisper encoder was trained on transcripts rather than self-supervised, unlabelled audio is used through pseudo-labelling instead.`);
  }

  // ---- emotion2vec+ freeze diagram
  {
    const s = L.slide(SEC, "Fine-tuning emotion2vec+ large: Frozen and Trainable Layers");
    const y0 = 1.95, bh = 1.25;
    const blocks = [
      ["Waveform\n16 kHz", 1.1, "F4F6F9", C.line, C.ink, "—"],
      ["CNN feature\nextractor\n7 conv layers", 1.55, "E3E8EF", "8C99AA", C.ink, "4.2M"],
      ["Projection +\nconv positional\nencoding", 1.55, "E3E8EF", "8C99AA", C.ink, "6.8M"],
      ["Transformer\nblocks 1–8\n(1024-d, 16 heads)", 1.9, "E3E8EF", "8C99AA", C.ink, "100.8M"],
      ["Transformer\nblocks 9–12", 1.6, "FCE9D9", C.accent2, C.ink, "50.4M"],
      ["Pooling\nmean, or weighted\nsum of 12 layers", 1.55, "E2F2EA", C.good, C.ink, "0–12"],
      ["New head\nLinear / MLP\n→ K classes", 1.45, "E2F2EA", C.good, C.ink, "9k–0.26M"],
    ];
    let x = MX; const pos = [];
    blocks.forEach(([t, w, fill, line, col, prm], i) => {
      L.box(s, x, y0, w, bh, t, { fill, line, color: col, fontSize: 13, lw: 1.5 });
      L.text(s, prm, x, y0 + bh + 0.08, w, 0.3, { fontSize: 12, color: C.muted, align: "center" });
      pos.push([x, w]);
      if (i < blocks.length - 1) L.arrow(s, x + w + 0.01, y0 + bh / 2, x + w + 0.17, y0 + bh / 2, { width: 1.25 });
      x += w + 0.18;
    });
    // E: self-supervised continued pre-training spans the encoder
    const xa = pos[1][0], xb = pos[4][0] + pos[4][1];
    s.addShape(p.ShapeType.line, { x: xa, y: y0 - 0.15, w: xb - xa, h: 0, line: { color: "8E4EC6", width: 2.25 } });
    L.text(s, "E: continued self-supervised pre-training on unlabelled call audio updates the whole encoder (no labels)", xa, y0 - 0.48, xb - xa, 0.3, { fontSize: 12, bold: true, color: "8E4EC6", align: "center" });
    const lg = [["E3E8EF", "8C99AA", "frozen in A, B, C (D: all but the CNN trained)"], ["FCE9D9", C.accent2, "LoRA or unfrozen in C; trained in D"], ["E2F2EA", C.good, "trained in A, B, C, D"]];
    lg.forEach(([f, l, t], i) => { const lx = MX + i * 4.1; s.addShape(p.ShapeType.roundRect, { x: lx, y: 3.95, w: 0.35, h: 0.25, rectRadius: 0.04, fill: { color: f }, line: { color: l, width: 1.5 } }); L.text(s, t, lx + 0.45, 3.93, 3.6, 0.3, { fontSize: 13, color: C.text }); });
    L.bullets(s, [
      { runs: [{ text: "Step 1 (≈1–3k labelled clips): ", options: { bold: true, color: C.good } }, { text: "B — freeze the encoder, cache the 12 layer outputs once, train the layer weights + MLP (≈0.26M parameters); minutes per run, so full speaker cross-validation is affordable" }] },
      { runs: [{ text: "In parallel (unlabelled audio): ", options: { bold: true, color: "8E4EC6" } }, { text: "E — continue self-supervised pre-training on call audio (10–100 h of audio ≈ 2–30 H100-h), then repeat B on the adapted encoder and compare" }] },
      { runs: [{ text: "Step 2 (≈5k labelled clips): ", options: { bold: true, color: C.accent2 } }, { text: "C — LoRA rank 8 on all 12 blocks (0.6–1.6M) or unfreeze blocks 9–12 (50M); encoder learning rate 1e-5–3e-5, head 1e-4–1e-3" }] },
      { runs: [{ text: "Step 3 (≥ 10k labelled clips): ", options: { bold: true, color: C.ink } }, { text: "D — all layers except the CNN (158M); low learning rate with layer-wise decay; mix in public data to avoid forgetting; ≈ 0.07–0.2 H100-h per 10k clips × 10 epochs" }] },
    ], { y: 4.45, h: 2.5, fontSize: 14, gap: 3 });
    L.notes(s, `Concretely for emotion2vec+. Grey parts stay frozen whenever we train with labels: the CNN, the projection and positional encoding, and the lower eight Transformer blocks. Green parts are always trained: pooling and a new classification head. The orange top four blocks are adapted only in option C.
    The purple line marks option E: continued self-supervised pre-training on our unlabelled call audio updates the whole encoder without any labels. After E, we repeat option B on the adapted encoder and compare with B on the original encoder.
    Full fine-tuning, option D, is the third step: computationally easy on our four H100 GPUs, but only worthwhile once active learning has produced at least ten thousand labelled clips.`);
  }

  // ---- MERaLiON freeze diagram (from facts.md)
  if (A.merFT) A.merFT(L);

  // ---- Data & compute
  {
    const s = L.slide(SEC, "Data and Compute Requirements");
    const rows = [
      ["A", "0.5k–5k labelled", "minutes on a CPU", "one encoder pass per clip (≈ 1 min per 1k clips)"],
      ["B", "1k–5k labelled", "minutes (12 layer outputs cached)", "minutes (encoder output cached, 3 MB per clip)"],
      ["C", "≥ 5k labelled", "0.1–0.2 A100-h · 0.03–0.1 H100-h per 10k clips × 10 epochs", "0.5–0.65 A100-h · 0.2–0.3 H100-h per 10k clips × 10 epochs"],
      ["D", "≥ 10k labelled", "0.2–0.4 A100-h · 0.07–0.2 H100-h per 10k clips × 10 epochs", "0.8–1 A100-h · 0.3–0.5 H100-h per 10k clips × 10 epochs"],
      ["E", "10–1,000 h unlabelled", "5–10 A100-h · 2–5 H100-h (10–20 h of audio); 600 A100-h · 200–300 H100-h (1,000 h)", "pseudo-labelling: one pass over 100k clips, 0.3–0.7 A100-h · 0.1–0.35 H100-h"],
    ];
    L.table(s, ["Option", "Data", "emotion2vec+ large", "MERaLiON-SER-v1"], rows.map((r) => r.map((c, j) => ({ text: c, bold: j === 0 }))), { y: 1.4, w: 8.3, colW: [0.75, 1.45, 3.05, 3.05], fontSize: 11, rowH: 0.5 });
    L.text(s, "Estimates for one GPU: A100 at 50–100 TFLOPS sustained with ×2 overhead; H100 counted 2–3 × faster. With 4 × H100, folds and seeds run in parallel (wall-clock ≈ ¼). MERaLiON computes a full 30 s window per clip (1.15 TFLOP), 13× emotion2vec+ on a 4 s clip; caching the frozen encoder output avoids this cost in A and B.", MX, 4.65, 8.3, 0.75, { fontSize: 11, color: C.muted });
    L.stat(s, "4 × H100", "available; each option fits on one GPU. 45 runs (5 folds × 3 learning rates × 3 seeds) of D on 10k clips ≈ 3–9 H100-h, about 1–2 h on 4 GPUs", 9.3, 1.45, 3.45, { size: 34, capH: 1.3 });
    L.stat(s, "3–5k → 10k+", "human-labelled clips: 3–5k for B and C, ≥ 10k for D; complemented by unlabelled call audio for E", 9.3, 3.65, 3.45, { size: 30, capH: 1.2, color: C.accent2 });
    L.bullets(s, [
      "Labels are the bottleneck, not GPUs: with 4 × H100 even full fine-tuning takes minutes to hours",
      "Learning curves flatten after a few thousand clips; the exact point is determined on our own data (train on 10 / 25 / 50 / 100%)",
      "Speaker-independent splits are essential: no caller appears in both training and test data",
    ], { y: 5.5, h: 1.45, w: 8.4, fontSize: 14, gap: 4 });
    L.source(s, "emotion2vec+: measured FLOPs (21.9 GFLOP per second of audio) and M4 Max timings; E scaled from emotion2vec's published pre-training compute. MERaLiON: Whisper-medium encoder on a 30 s window.");
    L.notes(s, `Compute is not the limiting factor: every feasible option runs on a single GPU, and options A and B take minutes. The cost that matters is the number of runs needed for proper cross-validation, and above all the human labels.
    An H100 is about two to three times faster than an A100, so all A100 estimates divide by two to three; with four H100s, cross-validation folds run in parallel. Even full fine-tuning of emotion2vec+ on ten thousand clips takes minutes per run. The plan is therefore sized by labels: three to five thousand human-labelled clips for options B and C, at least ten thousand for D, and our large pool of unlabelled call audio through option E: continued pre-training for emotion2vec+, and pseudo-labelling for MERaLiON.`);
  }

  // ---- Full fine-tuning (D): feasibility and decision criteria
  {
    const s = L.slide(SEC, "Full Fine-tuning (D): Feasibility and Decision Criteria");
    // three stat tiles
    const tiles = [
      ["< 1 H100-h", "per run on 10k clips × 10 epochs (emotion2vec+ 0.07–0.2, MERaLiON 0.3–0.5): compute is not the constraint", C.good],
      ["≈ 22,000", "annotation decisions for 10k clips: 1k seed × 4 listeners + 9k verified × 2 listeners", C.accent],
      ["≈ 90–120 h", "of listener time at 15–20 s per decision, ≈ 23–30 h per listener for 4 listeners (to be re-measured in the seed round)", C.accent2],
    ];
    tiles.forEach(([n, c, col], i) => {
      const x = MX + i * 4.1, w = 3.9;
      s.addShape(p.ShapeType.rect, { x, y: 1.45, w, h: 1.55, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: 1.45, w: 0.07, h: 1.55, fill: { color: col }, line: { color: col, width: 0 } });
      L.text(s, n, x + 0.25, 1.52, w - 0.35, 0.6, { fontSize: 28, bold: true, color: col });
      L.text(s, c, x + 0.25, 2.12, w - 0.4, 0.85, { fontSize: 12, color: C.text });
    });
    // risks and mitigations
    L.text(s, "Risks and mitigations", MX, 3.2, 7.5, 0.3, { fontSize: 14, bold: true, color: C.ink });
    const rows = [
      ["Few speakers → overfitting to their voices", "Sample across calls and customers; split training, validation and test by customer"],
      ["Class imbalance (mostly neutral; few angry clips)", "Active learning targets clips predicted as rare emotions"],
      ["Label noise limits the gain over C", "Measure inter-rater agreement (κ); use vote distributions as soft labels"],
      ["Forgetting of other classes and languages", "Mix in public labelled data; re-run the public benchmark before and after"],
    ];
    L.table(s, ["Risk", "Mitigation"], rows, { x: MX, y: 3.55, w: 7.6, colW: [3.2, 4.4], fontSize: 11, rowH: 0.47 });
    // pseudo-label route
    const px = 8.45, pw = W - MX - px;
    s.addShape(p.ShapeType.rect, { x: px, y: 3.2, w: pw, h: 2.68, fill: { color: "EEE6F7" }, line: { color: "8E4EC6", width: 1 } });
    L.text(s, "Complementary route: pseudo-labels", px + 0.15, 3.28, pw - 0.3, 0.3, { fontSize: 13, bold: true, color: "8E4EC6" });
    L.bullets(s, [
      "Train D on human labels plus pseudo-labels: clips of the 100k pool on which both models agree with high confidence",
      "Precedent: emotion2vec+ and MERaLiON were themselves trained on pseudo-labelled data",
      "Caveat: pseudo-labels reinforce existing model biases; the test set stays purely human-labelled",
    ], { x: px + 0.1, y: 3.62, w: pw - 0.2, h: 2.2, fontSize: 12, gap: 4 });
    // decision gate
    s.addShape(p.ShapeType.rect, { x: MX, y: 6.05, w: W - 2 * MX, h: 0.85, fill: { color: C.ink }, line: { color: C.ink, width: 0 } });
    s.addText([
      { text: "Decision gate at ≈ 5k labelled clips:  ", options: { bold: true, color: "FFD8B8" } },
      { text: "train C and D on 10 / 25 / 50 / 100% of the data and compare the learning curves. Adopt D only if C has plateaued and D still improves with more data; otherwise continue with C and collect more labels." , options: { color: C.white } },
    ], { x: MX + 0.2, y: 6.05, w: W - 2 * MX - 0.4, h: 0.85, fontFace: FONT, fontSize: 13, valign: "middle", margin: 0 });
    L.notes(s, `Is full fine-tuning feasible for us? Computationally, clearly yes: one run on ten thousand clips for ten epochs takes well under one H100-hour, and a complete cross-validation of forty-five runs fits in one to two hours on our four GPUs.
    The real requirement is about ten thousand human-labelled clips. With our protocol, a thousand seed clips labelled by four listeners and nine thousand clips verified by two listeners, that is about twenty-two thousand annotation decisions. At fifteen to twenty seconds per decision this is roughly ninety to one hundred and twenty listener-hours, about a week of part-time work per listener for four listeners. The labelling tool records the time spent per clip, so we will re-measure this in the seed round.
    The main risks concern data quality rather than quantity: too few distinct callers, too few angry clips, label noise from listener disagreement, and forgetting of other classes or languages. Each has a mitigation listed in the table.
    A complementary route is to add pseudo-labels: clips from the unlabelled pool on which both models agree with high confidence. emotion2vec+ and MERaLiON were themselves trained this way. Pseudo-labels reinforce model biases, so the test set must remain purely human-labelled.
    Decision: at about five thousand labelled clips we compare learning curves of C and D. D is adopted only if C has plateaued while D keeps improving with more data.`);
  }

  // ---- Main challenges of adaptation
  {
    const s = L.slide(SEC, "Main Challenges in Adapting to Call-Centre Audio");
    const cards = [
      ["Low inter-rater agreement", "Natural speech is more ambiguous than acted speech; agreement between listeners sets the ceiling for any model.",
        ["Measure κ in the seed round before scaling up", "Keep vote distributions as soft labels", "Merge rare or confusable classes if κ stays low"], C.accent2],
      ["Class imbalance", "Most call segments are neutral; angry segments, the most important ones, may be only a few percent.",
        ["Active learning targets clips predicted as rare emotions", "Class-weighted loss or balanced sampling", "Report unweighted accuracy and macro-F1, not accuracy"], C.accent],
      ["Lexical content carries emotion", "The benchmark removed lexical cues on purpose; in real calls the words (\"this is ridiculous\") are informative, and acoustic models ignore them.",
        ["Fuse the acoustic model with a text-emotion model on ASR transcripts", "Start with late fusion of the two probability vectors"], C.good],
      ["Telephone channel (8 kHz)", "Both models were trained on 16 kHz wideband speech; telephone audio has no content above 4 kHz.",
        ["Re-run the benchmark through a simulated G.711 channel to measure the drop", "Option E and fine-tuning on telephone audio", "Same resampling in training and inference"], C.accent],
    ];
    const cw = 2.88, gap = 0.2;
    cards.forEach(([t, why, mit, col], i) => {
      const x = MX + i * (cw + gap), y = 1.45, h = 5.45;
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      L.text(s, t, x + 0.2, y + 0.25, cw - 0.4, 0.7, { fontSize: 17, bold: true, color: C.ink });
      L.text(s, why, x + 0.2, y + 1.0, cw - 0.4, 1.6, { fontSize: 12.5, color: C.text });
      s.addShape(p.ShapeType.line, { x: x + 0.2, y: y + 2.7, w: cw - 0.4, h: 0, line: { color: C.line, width: 1 } });
      L.text(s, i === 2 ? "Opportunity" : "Mitigation", x + 0.2, y + 2.8, cw - 0.4, 0.3, { fontSize: 12, bold: true, color: col });
      L.bullets(s, mit, { x: x + 0.15, y: y + 3.12, w: cw - 0.3, h: 2.25, fontSize: 12, gap: 5 });
    });
    L.notes(s, `Four challenges stand out.
    First, inter-rater agreement. Natural speech is far more ambiguous than acted speech, and agreement between listeners sets the ceiling for any model. We measure it in the seed round before scaling up, keep vote distributions as soft labels, and merge classes if agreement stays low.
    Second, class imbalance. Most segments are neutral, and angry segments, which matter most, may be only a few percent. Active learning deliberately looks for them, the loss is class-weighted, and we report unweighted accuracy and macro-F1 rather than plain accuracy.
    Third, words matter in real calls. Our benchmark removed lexical cues on purpose, but customers often say what they feel, and purely acoustic models ignore it. Fusing the acoustic model with a text-emotion model on ASR transcripts is a clear opportunity; the LISTEN finding that LLMs rely on words becomes an asset here.
    Fourth, the telephone channel. Both models were trained on 16 kHz speech; telephone audio has nothing above 4 kHz. We first measure the drop by re-running the benchmark through a simulated G.711 channel, then address it with continued pre-training and fine-tuning on telephone audio.`);
  }

  // ---- Labelling protocol
  {
    const s = L.slide(SEC, "Annotation Protocol");
    L.image(s, "img/labeler.png", 7.55, 1.4, 5.2, 4.3);
    L.text(s, "Our offline labelling tool (single HTML file, no install, nothing leaves the PC): random order per listener, keys 1–8, bad-clip option, optional model hints, CSV per listener", 7.55, 5.75, 5.2, 0.9, { fontSize: 11, color: C.muted });
    L.bullets(s, [
      { t: "Four listeners per clip", sub: ["A single rating is not a reliable reference: CREMA-D listeners agree with the actor on ≈4 in 10 clips; four votes allow agreement to be measured and ambiguity to be recorded"] },
      { t: "Decision rule", sub: ["≥ 3 of 4 agree → label", "2–2 or all different → \"ambiguous\": keep the vote distribution as a soft label, or adjudicate", "any \"bad clip\" vote (noise, overlap, silence) → review and drop"] },
      { t: "Quality control", sub: ["Inter-rater agreement (Fleiss' κ / Krippendorff's α) per round", "Gold clips repeated to check consistency; short calibration session with examples first"] },
      { t: "Bias control", sub: ["Seed set and test set: model hints off (predictions_mode \"off\")", "Verification rounds: hints on, to speed up checking"] },
    ], { x: MX, y: 1.45, w: 6.7, h: 5.4, fontSize: 15 });
    L.notes(s, `Labels are the bottleneck, so the protocol matters. Four listeners per clip, majority vote; ties are not forced into a class but kept as ambiguous with the vote distribution, which we can use as a soft training target.
    We built a small offline labelling tool for this: one HTML file, no installation, no network; each listener gets their own shuffled order and writes their own CSV. It can show model predictions as hints, which we switch off for the seed and test sets to avoid anchoring the listeners.`);
  }

  // ---- Active learning loop
  {
    const s = L.slide(SEC, "Active Learning Pipeline");
    const bw = 2.85, bh = 1.25, y1 = 1.55, y2 = 3.75;
    const xs = [MX, MX + 3.55, MX + 7.1];
    L.box(s, xs[0], y1, bw, bh, "① Seed labels\n1k clips × 4 listeners\nmajority vote", { fill: "EEF4FA", line: C.accent, lw: 1.5, fontSize: 14 });
    L.box(s, xs[1], y1, bw, bh, "② Fine-tune\nemotion2vec+ and MERaLiON\n(B first; C from ≈5k, D from ≥10k)", { fill: C.ink, line: C.ink, color: C.white, fontSize: 14 });
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
    // test set feeds the evaluation of every round: right-angled dashed connector through the column gap
    const gx = xs[0] + bw + 0.35;
    L.arrow(s, xs[0] + bw + 0.02, y2 + 0.35, gx, y2 + 0.35, { dash: "dash", width: 1.25, noHead: true });
    L.arrow(s, gx, y2 + 0.35, gx, y1 + bh - 0.25, { dash: "dash", width: 1.25, noHead: true });
    L.arrow(s, gx, y1 + bh - 0.25, xs[1] - 0.02, y1 + bh - 0.25, { dash: "dash", width: 1.25 });
    // rounds table
    const rows = [["Round 0", "1k (4 listeners)", "1k", "B (+ E in parallel)"], ["Round 1", "≈1k selected from 100k", "≈2k", "B"], ["Round 2", "≈1.5k", "≈3.5k", "B, C"], ["Round 3", "≈1.5k", "≈5k", "C"], ["Round 4+", "≈2–3k per round", "≥10k", "D"]];
    L.table(s, ["", "New labels", "Total", "Option"], rows, { x: MX, y: 5.2, w: 6.6, colW: [1.2, 2.6, 1.4, 1.4], fontSize: 11, rowH: 0.28 });
    L.bullets(s, [
      "Verifying a proposed label is faster than labelling from scratch, and uncertain clips are the most informative for training",
      "Stopping criterion: no further improvement of the test-set score between rounds",
      "Disagreement between the two models is an additional selection signal at no cost",
    ], { x: 7.5, y: 5.2, w: 5.25, h: 1.7, fontSize: 14 });
    L.notes(s, `This is the active learning loop. Round zero: one thousand clips, each labelled by four listeners, majority vote. Fine-tune both models. Run them on a large unlabelled pool, around one hundred thousand clips.
    Then, instead of labelling at random, select the clips the models are unsure about, the clips where the two models disagree, and clips predicted as rare emotions such as anger, plus a random sample to check calibration. Two listeners verify or correct those, and we retrain.
    A fixed test set, labelled by four listeners from callers never used for training, is scored after every round; we stop when it stops improving. Within three rounds we reach about five thousand verified clips, enough for option C; further rounds take us beyond ten thousand, where full fine-tuning, option D, becomes worthwhile. In parallel, the unlabelled pool is used for continued pre-training of emotion2vec+ and for pseudo-labels.`);
  }

  // ---- Roadmap
  {
    const s = L.slide(SEC, "Roadmap");
    const ph = [
      ["Step 1", "Preparation", ["Sample calls; segment into 3–10 s customer turns (VAD)", "Agree on the label set (7 + unknown, or call-centre classes)", "Re-run the benchmark through a telephone channel"], "Go: label set and data access agreed"],
      ["Step 2", "Seed annotation", ["1k clips × 4 listeners + 500-clip test set", "Measure inter-rater agreement", "Zero-shot baseline of both models on the test set"], "Go: agreement acceptable (κ), baseline known"],
      ["Step 3", "First adaptation", ["Option B for both models", "E on unlabelled call audio (emotion2vec+), pseudo-labels (MERaLiON)", "Speaker-independent cross-validation vs zero-shot"], "Go: clear gain over zero-shot"],
      ["Step 4", "Active learning", ["Rounds of predict → select → verify → retrain", "C at ≈ 5k; D at ≥ 10k labelled clips if the learning curves justify it", "Track per-class recall, especially angry"], "Gate at ≈ 5k: C vs D learning curves · Stop: test score plateaus"],
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
    Step two produces the seed labels and the test set, and tells us how much our own listeners agree. Step three is the first adaptation: option B for both models, plus the use of unlabelled call audio through continued pre-training for emotion2vec+ and pseudo-labelling for MERaLiON. Step four is the active learning loop, moving to option C at about five thousand labelled clips.`);
  }

  // ---- Summary
  {
    const s = L.slide("Conclusion", "Summary");
    const items = [
      ["Emotion labels are subjective", "Listeners disagree with actors and with each other; annotation requires multiple listeners and soft labels."],
      ["Dedicated SER models outperform audio LLMs", "Best open model 65–99.5% per corpus vs Gemini 27–45% and GPT Realtime 2–29%; audio LLMs rely on lexical content (consistent with LISTEN)."],
      ["Training-data overlap inflates benchmark scores", "Corpora seen in training score 20–52 points higher; on call-centre audio, unseen-level accuracy (≈50–65%) is expected before adaptation."],
      ["Shortlist: emotion2vec+ large and MERaLiON-SER-v1", "Highest accuracy, licences permitting commercial use, complementary strengths."],
      ["Adaptation strategy", "Frozen encoder with a new head first (B), LoRA (C) from ≈5k and full fine-tuning (D) from ≥10k labelled clips on 4 × H100; unlabelled call audio through E / pseudo-labelling."],
      ["Annotation strategy", "1k seed clips × 4 listeners, then active learning on ≈100k unlabelled clips: 3–5k verified labels for B and C, ≥10k for D."],
    ];
    items.forEach(([t, d], i) => {
      const y = 1.5 + i * 0.88;
      s.addText(String(i + 1), { x: MX, y: y + 0.05, w: 0.5, h: 0.5, shape: p.ShapeType.ellipse, fill: { color: i >= 4 ? C.accent2 : C.accent }, line: { color: C.white, width: 0 }, fontFace: FONT, fontSize: 16, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 0.75, y, 11.5, 0.38, { fontSize: 18, bold: true, color: C.ink });
      L.text(s, d, MX + 0.75, y + 0.38, 11.5, 0.42, { fontSize: 15, color: C.text });
    });
    L.notes(s, `Six conclusions: the subjectivity of the labels, the gap between dedicated models and audio LLMs, the effect of training-data overlap, the shortlist, the adaptation strategy and the annotation strategy.`);
  }

  // ---- Q&A
  {
    const s = p.addSlide();
    s.background = { color: C.white };
    L.text(s, "Thank you — questions?", MX, 2.8, W - 2 * MX, 1.0, { fontSize: 40, bold: true, color: C.ink, align: "center" });
    L.text(s, "Code, reports and listening viewer: emotion_recognition/ (README, reports/summary.md, outputs/viewer.html)", MX, 3.9, W - 2 * MX, 0.5, { fontSize: 16, color: C.muted, align: "center" });
  }
};

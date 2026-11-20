// Content that depends on the assets (manifest, stats) and on facts.md
module.exports = function (L, { manifest, stats }) {
  const fs = require("fs"), path = require("path");
  const benchmark = JSON.parse(fs.readFileSync(path.resolve(__dirname, "../../research_paper/output/cross_dataset_analysis.json"), "utf8"));
  if (benchmark.manifest.selected_run_count !== 160 || benchmark.manifest.full_models.length !== 10)
    throw new Error("Rebuild the complete 10-model / 16-dataset analysis before this deck");
  const A = { manifest, stats, benchmark };
  A.datasetOrder = ["meld", "emotiontalk", "dailytalk", "multidialog", "crema_d", "ravdess", "savee", "jl_corpus", "tess", "mead", "mess", "mcae_spps", "thai_ser", "nemo", "emovo", "emozionalmente"];
  A.metric = (model, dataset, metric = "accuracy") => {
    const row = benchmark.model_rows.find(r => r.model === model && r.dataset === dataset);
    if (!row || row[metric] == null) throw new Error(`Missing benchmark metric ${model}/${dataset}/${metric}`);
    return 100 * row[metric];
  };
  A.mean = (model, metric = "accuracy", filter = () => true) => {
    const rows = benchmark.model_rows.filter(r => r.model === model && filter(r));
    return rows.length ? 100 * rows.reduce((sum, r) => sum + r[metric], 0) / rows.length : null;
  };
  A.exposureMean = (model, established) => {
    const rows = benchmark.model_rows.filter(r => r.model === model && ((r.training_status !== "not_established") === established));
    return rows.length ? `${(100 * rows.reduce((sum, r) => sum + r.accuracy, 0) / rows.length).toFixed(1)}% (${rows.length} datasets)` : "not available";
  };
  const { C, W, MX, FONT, MODEL } = L;

  // ---- Survey numbers (facts.md §4)
  A.surveyTitle = "Model Survey and Selection";
  A.surveyBullets = [
    "Ten evaluated models: five dedicated SER, three open audio LLMs and two API audio LLMs; current registry and inference protocol",
    "All ten models completed the same 67,062 clips across 16 datasets; no new training or inference in this refresh",
  ];
  A.geminiDate = "2026";

  // ---- MERaLiON numbers used on the shortlist and compute slides
  A.mer = {
    params: "313.4M (98% in the Whisper encoder)",
    code: "none official (inference code only); model class has a LoRA path",
    c1k: "5–6 min", c10k: "0.8–1 A100-h", c100k: "8–10 A100-h",
    computeNote: "MERaLiON computes a full 30 s window for every clip: 1.15 TFLOP forward, 13× emotion2vec+ on a 4 s clip. Options A/B avoid this by caching encoder outputs once (3 MB per clip in fp16).",
  };

  // ---- Architecture slides: MERaLiON, then Melati and Vox-Profile
  const row = (s, items, y, h, x0 = MX, gap = 0.2) => {
    let x = x0;
    items.forEach(([t, w, f, l, prm], i) => {
      L.box(s, x, y, w, h, t, { fill: f, line: l, fontSize: 12, lw: 1.25 });
      if (prm) L.text(s, prm, x, y + h + 0.05, w, 0.28, { fontSize: 11, color: C.muted, align: "center" });
      if (i < items.length - 1) L.arrow(s, x + w + 0.01, y + h / 2, x + w + gap - 0.01, y + h / 2, { width: 1.25 });
      x += w + gap;
    });
    return x - gap;
  };
  const ENC = ["EEF4FA", "0072B2"], DOWN = ["E2F2EA", "2E8B57"], IO = [C.panel, C.line];
  A.archSlides = [
    (L) => {
      const s = L.slide("3 · Benchmarking Models", "MERaLiON-SER-v1: Architecture");
      L.text(s, "Pre-trained speech encoder: Whisper-medium encoder (speech recognition pre-training) · 307.2M", MX, 1.4, 11, 0.3, { fontSize: 13, bold: true, color: C.accent });
      row(s, [
        ["Waveform\npadded / cut\nto 30 s", 1.45, ...IO, ""],
        ["Log-mel\nspectrogram\n80 × 3000", 1.3, ...IO, ""],
        ["Conv stem\n2 × Conv1d\n(stride 2)", 1.5, ...ENC, "3.4M"],
        ["+ sinusoidal\npositions\n(fixed)", 1.45, ...ENC, "1.5M"],
        ["24 Transformer blocks\nd 1024 · 16 heads · FFN 4096\nLoRA r16 on q/k/v (merged)", 3.15, ...ENC, "302.3M"],
        ["Frames\n1500 × 1024", 1.45, ...IO, ""],
      ], 1.75, 1.05);
      L.text(s, "Pooling and head: ECAPA-TDNN network (from speaker verification) · 6.0M, two output heads · model total 313.4M", MX, 3.3, 11.5, 0.3, { fontSize: 13, bold: true, color: C.good });
      row(s, [
        ["proj_in\n1024 → 512", 1.35, ...DOWN, "0.53M"],
        ["3 ECAPA blocks\nRes2 convs, dilation\n1 / 2 / 3, SE", 1.9, ...DOWN, "2.23M"],
        ["Multi-layer\naggregation\n2048 → 512", 1.6, ...DOWN, "1.05M"],
        ["Attentive\npooling: 4 scales +\n3 resolutions", 1.95, ...DOWN, "1.81M"],
        ["Embedding\nMLP → 256", 1.4, ...DOWN, "0.40M"],
      ], 3.65, 1.05);
      const yc = 3.65 + 0.525;
      L.arrow(s, 9.6, yc, 9.85, yc, { width: 1.25, noHead: true });
      L.arrow(s, 9.85, yc - 0.37, 9.85, yc + 0.37, { width: 1.25, noHead: true });
      L.arrow(s, 9.85, yc - 0.37, 10.08, yc - 0.37, { width: 1.25 }); L.arrow(s, 9.85, yc + 0.37, 10.08, yc + 0.37, { width: 1.25 });
      L.box(s, 10.1, yc - 0.62, 2.63, 0.5, "7 emotions (softmax)", { fill: "FCE9D9", line: C.accent2, fontSize: 12 });
      L.box(s, 10.1, yc + 0.12, 2.63, 0.5, "valence · arousal · dominance", { fill: "FCE9D9", line: C.accent2, fontSize: 12 });
      L.bullets(s, [
        { runs: [{ text: "ECAPA blocks: ", options: { bold: true } }, { text: "1-D convolutions over time (TDNN) look at neighbouring frames; dilation 1 / 2 / 3 widens the context block by block; Res2 processes channel groups at growing scales; squeeze-excitation (SE) re-weights channels using a summary of the whole clip" }] },
        { runs: [{ text: "Aggregation and attentive pooling: ", options: { bold: true } }, { text: "the outputs of all three blocks are concatenated (several time scales); learned attention weights each frame before averaging, so salient frames count more than pauses" }] },
        { runs: [{ text: "Training and input: ", options: { bold: true } }, { text: "encoder frozen with LoRA (2.4M), head trained (15 epochs, 8 × H100); fixed 30 s window, 1.15 TFLOP per clip; padding is included in pooling (a 4 s clip = 200 of 1,500 frames)" }] },
      ], { y: 5.15, h: 1.85, fontSize: 13, gap: 3 });
      L.source(s, "", [["MERaLiON-SER (arXiv 2511.04914)", "https://arxiv.org/abs/2511.04914"], ["MERaLiON-SER-v1 model card", "https://huggingface.co/MERaLiON/MERaLiON-SER-v1"], ["Whisper (Radford et al., ICML 2023)", "https://arxiv.org/abs/2212.04356"], ["ECAPA-TDNN (Desplanques et al., Interspeech 2020)", "https://arxiv.org/abs/2005.07143"], ["LoRA (Hu et al., ICLR 2022)", "https://arxiv.org/abs/2106.09685"]]);
      L.notes(s, `MERaLiON-SER-v1 from A*STAR in Singapore, 2025, follows the same three-stage structure. The pre-trained encoder is the encoder of Whisper-medium, a speech recognition model: the waveform is padded or cut to 30 seconds, turned into a log-mel spectrogram, passed through a small convolutional stem and 24 Transformer blocks. This produces 1,500 feature vectors of 1,024 numbers, one every 20 milliseconds, and holds 98 percent of the parameters. During MERaLiON's training the encoder was frozen and adapted with LoRA, small low-rank updates to the attention projections, which were then merged into the weights.
      The second row replaces the simple mean of emotion2vec+ with a much richer pooling network called ECAPA-TDNN, borrowed from speaker verification, where it is the standard way to turn a sequence of frames into one vector that describes a voice. In plain terms: TDNN means time-delay neural network, a one-dimensional convolution that looks at a few neighbouring frames at a time. Each of the three ECAPA blocks contains such convolutions with a different dilation, 1, 2 and 3, so later blocks look at wider stretches of time. Inside a block, Res2 splits the channels into groups and processes them one after another, so different groups see different time spans, a multi-scale view. Squeeze-excitation first summarises the whole clip into one vector, then uses it to turn feature channels up or down, a form of channel attention. Multi-layer aggregation concatenates the outputs of all three blocks, so short- and long-range patterns are both kept.
      Attentive pooling then computes a weight for every frame and takes a weighted average, at four temporal scales and three resolutions, so emotionally informative frames count more than pauses. A small MLP produces a 256-number clip embedding, followed by two heads: seven emotion probabilities, and three continuous dimensions, valence, arousal and dominance.
      Practical points: every clip costs a full 30-second window, and the pooling does not exclude the padded silence, so for a 4-second clip only 200 of the 1,500 frames contain speech. Both matter for call audio.`);
    },
    (L) => {
      const s = L.slide("3 · Benchmarking Models", "Melati and Vox-Profile: Architectures");
      const blocks = [
        ["Melati · 637.3M · Scicom 2026 · non-commercial", MODEL.melati.color, [
          ["Waveform\n≤ 30 s window", 1.5, ...IO, ""], ["Whisper large-v3 encoder\n32 blocks · d 1280 · frozen", 3.3, ...ENC, "637.0M"],
          ["Mean-pool each\nof 33 layers →\nlearned weights", 2.1, ...DOWN, ""], ["LayerNorm →\n1280 → 256 → 7", 1.9, ...DOWN, "0.33M (the only trained part)"]]],
        ["Vox-Profile Whisper · 639.9M · USC 2025 · non-commercial", MODEL.voxw.color, [
          ["Waveform\n≤ 15 s window", 1.5, ...IO, ""], ["Whisper large-v3 encoder\n+ LoRA r16 on FFN, blocks 17–31", 3.3, ...ENC, "636.0M + 3.1M LoRA"],
          ["1×1 convs +\nmean pooling", 2.1, ...DOWN, ""], ["MLP heads\n9 classes (+ others)", 1.9, ...DOWN, "0.8M"]]],
        ["Vox-Profile WavLM · 316.2M · USC 2025 · non-commercial", MODEL.voxl.color, [
          ["Waveform\nreal frames ≤ 15 s", 1.5, ...IO, ""], ["WavLM-large\nfully fine-tuned", 3.3, ...ENC, "315.5M"],
          ["1×1 convs +\nmean pooling", 2.1, ...DOWN, ""], ["MLP heads\n9 classes (+ others)", 1.9, ...DOWN, "0.7M"]]],
      ];
      blocks.forEach(([t, col, items], i) => {
        const y = 1.45 + i * 1.75;
        s.addShape(L.pptx.ShapeType.rect, { x: MX, y: y + 0.05, w: 0.08, h: 0.25, fill: { color: col }, line: { color: col, width: 0 } });
        L.text(s, t, MX + 0.18, y, 9, 0.35, { fontSize: 14, bold: true, color: C.ink });
        row(s, items, y + 0.42, 0.85);
      });
      L.box(s, 10.45, 1.45, 2.28, 5.2, "", { fill: C.panel, line: C.panel });
      L.text(s, "Forward compute per clip", 10.6, 1.55, 2.0, 0.5, { fontSize: 13, bold: true, color: C.ink });
      const fl = [["emotion2vec+ (4 s)", "0.09"], ["Vox WavLM (4 s)", "0.15"], ["Vox Whisper", "1.05"], ["MERaLiON", "1.15"], ["Melati", "2.27"]];
      fl.forEach(([m, v], i) => { L.text(s, m, 10.6, 2.1 + i * 0.42, 1.4, 0.38, { fontSize: 12, color: C.text, valign: "middle" }); L.text(s, v, 11.95, 2.1 + i * 0.42, 0.65, 0.38, { fontSize: 13, bold: true, color: C.ink, align: "right", valign: "middle" }); });
      L.text(s, "TFLOP. Whisper models pay for the full window on every clip.", 10.6, 4.25, 2.0, 0.8, { fontSize: 11, color: C.muted });
      L.text(s, "Melati and Vox-Profile Whisper keep the Whisper encoder as released (Vox-Profile's LoRA weights are all zero, so it has no effect).", 10.6, 5.05, 2.0, 1.5, { fontSize: 11, color: C.muted });
      L.source(s, "", [["Melati model card", "https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3"], ["Vox-Profile (Feng et al., arXiv 2505.14648)", "https://arxiv.org/abs/2505.14648"], ["Whisper (Radford et al., ICML 2023)", "https://arxiv.org/abs/2212.04356"], ["WavLM (Chen et al., IEEE JSTSP 2022)", "https://arxiv.org/abs/2110.13900"]]);
      L.notes(s, `Input handling: Melati and Vox-Profile Whisper use a fixed window, 30 and 15 seconds; shorter clips are padded with zeros, which is digital silence, not white noise, and longer audio is cut. Melati pools only over the real frames. Vox-Profile WavLM processes real frames after the local fifteen-second cap; its card warns that clips under three seconds are unreliable. The next slide compares all five models.
      The other three open models follow the same recipe with bigger encoders. Melati freezes the Whisper large-v3 encoder, 637 million parameters, and trains only a third of a million: a learned mix of the 33 layer outputs and a small MLP. Vox-Profile has a Whisper version and a WavLM version, trained on natural podcast speech; the WavLM one is fully fine-tuned.
      All three are research-only licences. And note the compute column: the Whisper-based models cost about ten to twenty-five times more per clip than emotion2vec+.`);
    },
  ];

  // ---- MERaLiON fine-tuning diagram
  A.merFT = (L) => {
    const s = L.slide("6 · Future Work", "Fine-tuning MERaLiON-SER-v1: Frozen and Trainable Layers");
    const y0 = 2.0, bh = 1.25;
    const FR = ["E3E8EF", "8C99AA"], OPT = ["FCE9D9", C.accent2], TR = ["E2F2EA", C.good];
    const items = [
      ["Waveform\n(30 s window)", 1.35, C.panel, C.line, "—"],
      ["Conv stem +\npositions", 1.4, ...FR, "4.9M"],
      ["Whisper encoder\n24 blocks\n(1024-d, 16 heads)", 2.3, ...FR, "302.3M"],
      ["LoRA r8 on q/k/v\n(all 24 blocks)", 1.7, ...OPT, "1.18M"],
      ["ECAPA-TDNN +\nhybrid pooling +\nembedding MLP", 2.0, ...TR, "6.0M"],
      ["Heads\n7 emotions + valence,\narousal, dominance", 1.45, ...TR, "68k–135k"],
    ];
    let x = MX;
    items.forEach(([t, w, f, l, prm], i) => {
      L.box(s, x, y0, w, bh, t, { fill: f, line: l, fontSize: 13, lw: 1.5, dash: i === 3 ? "dash" : "solid" });
      L.text(s, prm, x, y0 + bh + 0.08, w, 0.3, { fontSize: 12, color: C.muted, align: "center" });
      if (i < items.length - 1) L.arrow(s, x + w + 0.01, y0 + bh / 2, x + w + 0.17, y0 + bh / 2, { width: 1.25 });
      x += w + 0.18;
    });
    const lg = [["E3E8EF", "8C99AA", "frozen in A, B, C (trained in D)"], ["FCE9D9", C.accent2, "added in C (LoRA adapters)"], ["E2F2EA", C.good, "trained (A: heads only; B, C, D: all green)"]];
    lg.forEach(([f, l, t], i) => { const lx = MX + i * 4.1; s.addShape(L.pptx.ShapeType.roundRect, { x: lx, y: 3.95, w: 0.35, h: 0.25, rectRadius: 0.04, fill: { color: f }, line: { color: l, width: 1.5 } }); L.text(s, t, lx + 0.45, 3.93, 3.6, 0.3, { fontSize: 13, color: C.text }); });
    L.table(s, ["Option", "Trains", "Trainable", "Cost per training clip"], [
      ["A", "heads only (on cached 256-d embeddings)", "68k", "none after one forward pass"],
      [{ text: "B", bold: true }, { text: "ECAPA + pooling + MLP + heads (as in the original recipe)", bold: true }, { text: "6.1M", bold: true }, "cache encoder output once (3 MB per clip): minutes per run"],
      ["C", "B + LoRA r8 on encoder q/k/v", "7.3M", "≈ 2.3 TFLOP per clip · 0.2–0.3 H100-h per 10k clips × 10 epochs"],
      ["D", "everything (except the fixed positions)", "311.8M", "≈ 3.4 TFLOP per clip · 0.3–0.5 H100-h per 10k clips × 10 epochs"],
      [{ text: "E′", bold: true, options: { color: "8E4EC6" } }, "Self-training — B or C on pseudo-labelled call audio (confident, agreeing predictions of both models)", "as B / C", "one inference pass over the unlabelled pool"],
    ], { y: 4.4, w: 8.3, colW: [0.8, 3.4, 1.0, 3.1], fontSize: 11, rowH: 0.4 });
    L.bullets(s, [
      "B mirrors how MERaLiON itself was trained: encoder frozen, head trained",
      "Unlabelled audio: Whisper was trained on transcripts, not self-supervised, so E is replaced by pseudo-labelling (E′), as MERaLiON's own training did with emotion2vec+ labels",
      "D (all 311.8M parameters) runs on 4 × H100 once ≥ 10k labelled clips exist; while fine-tuning, mask padding in pooling and consider a shorter window",
    ], { x: 9.15, y: 4.4, w: 3.6, h: 2.55, fontSize: 12, gap: 4 });
    L.notes(s, `The same logic for MERaLiON. Ninety-eight percent of its parameters are the Whisper encoder, which was already frozen in the original training. So option B, retraining the ECAPA head, the pooling and the output heads, six million parameters, is exactly the original recipe applied to our data. Because the encoder is frozen we can cache its output once and each training run takes minutes.
    If B plateaus, option C adds LoRA adapters to the encoder attention, about 1.2 million extra parameters.
    Unlabelled audio cannot be used through continued self-supervised pre-training here, because the Whisper encoder was trained on transcripts rather than with a self-supervised objective. Instead we use pseudo-labelling: both models label the unlabelled call audio, and clips on which they agree with high confidence are added as training data. MERaLiON's own training used exactly this idea, with labels from emotion2vec+. Full fine-tuning of all 311.8 million parameters, option D, is computationally easy on four H100 GPUs, about half an H100-hour per ten thousand clips and ten epochs, and becomes worthwhile once we have at least ten thousand labelled clips. Fine-tuning is also the moment to fix the unmasked padding and possibly shorten the 30-second window, which would make inference about four times cheaper.`);
  };

  // ---- Current results and the limits of semantic/acoustic explanations
  A.listen = [
    { t: "Complete benchmark", b: true, sub: ["10 models × 16 datasets; 67,062 clips per model", "Audio LLM performance depends on the corpus and metric; there is no universal SER-versus-LLM ranking"] },
    { t: "Qwen2.5-Omni on dialogue sources", b: true, sub: ["Its macro-F1 exceeds the best dedicated SER on all four dialogue datasets", "Accuracy exceeds every SER on only two; neutral-only beats every model on DailyTalk and MultiDialog"] },
    { t: "LISTEN: a related evaluation", b: true, sub: ["Lexically neutral and non-verbal stimuli can expose weak use of emotional voice cues", "Different prompts, labels and datasets prevent direct score comparisons"] },
    { t: "Interpretation", b: true, sub: ["Semantic versus acoustic reliance, training exposure and label conventions are hypotheses to test with controls", "Observed score differences do not identify training leakage or prove that a model ignores prosody"] },
  ];
  A.listenNotes = `Sources: research_paper/output/cross_dataset_analysis.json and the bilingual cross-dataset report, refreshed from all 160 compatible runs. The four dialogue datasets are evaluated as isolated utterances, without dialogue history. Neutral priors and reference-label sources differ. LISTEN (https://arxiv.org/abs/2510.10444) supplies related evidence under another protocol, not a causal explanation of these results. Use transcript-only and controlled acoustic/text comparisons before attributing a failure to semantic reliance. No result in this deck establishes training leakage.`;

  // ---- Which ground truth do models follow? (after the confusion matrices)
  A.afterCms = [(L) => {
    const s = L.slide("4 · Evaluation Results", "Selected SER Models: Actor vs Listener Labels (CREMA-D)");
    const w = stats.crema_d_listeners.when_listeners_disagree_with_actor;
    const keys = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm"];
    const names = keys.map((k) => MODEL[L.MODEL_KEY[k]].name.replace("Vox-Profile ", "Vox "));
    const act = keys.map((k) => +(100 * w.models[k].predicts_acted).toFixed(1));
    const lis = keys.map((k) => +(100 * w.models[k].predicts_listener_majority).toFixed(1));
    L.text(s, "Disputed clips: the 3,699 CREMA-D clips whose listeners' majority differs from the acted emotion. For each model, the share of these clips on which it predicts:", MX, 1.4, 7.6, 0.6, { fontSize: 13, color: C.text });
    s.addChart(L.pptx.ChartType.bar, [
      { name: "the acted emotion", labels: names, values: act },
      { name: "the listeners' majority", labels: names, values: lis },
      { name: "another emotion", labels: names, values: act.map((v, i) => +(100 - v - lis[i]).toFixed(1)) },
    ], { x: MX, y: 2.0, w: 7.6, h: 4.8, barDir: "col", barGrouping: "stacked", chartColors: ["1B2A41", "D55E00", "C9CED6"],
      showValue: true, dataLabelFormatCode: "0", dataLabelFontSize: 11, dataLabelFontFace: FONT, dataLabelColor: "FFFFFF", catAxisLabelFontSize: 12, catAxisLabelFontFace: FONT,
      valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFontSize: 11, valAxisTitle: "% of the 3,699 disputed clips", showValAxisTitle: true, valAxisTitleFontSize: 12,
      showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: FONT, valGridLine: { color: "E5E8EC", size: 0.5 } });
    L.bullets(s, [
      { t: "Models trained on CREMA-D's acted labels reproduce the actor", sub: ["emotion2vec+ 82%, MERaLiON 77% (CREMA-D is in their training data)"] },
      { t: "Models trained on perceived labels follow the listeners", sub: ["Vox-Profile (MSP-Podcast, listener-annotated natural speech): 62–65%; its low accuracy partly reflects the choice of reference label"] },
      { t: "Implication", b: true, color: C.accent2, sub: ["Our annotation will record perceived emotion; the models must therefore be adapted to perceived labels"] },
    ], { x: 8.5, y: 1.45, w: 4.25, h: 5.4, fontSize: 15 });
    L.notes(s, `This analysis uses CREMA-D only, because it is the corpus with both acted labels and listener votes for every clip. We take the 3,699 clips on which the listeners' majority differs from the acted emotion. For each model, the dark part of the bar is the share of those clips where the model predicts the acted emotion, the orange part where it predicts what the listeners heard, and the grey part where it predicts something else.
    emotion2vec+ and MERaLiON, with documented CREMA-D training involvement, side with the actor in about 80 percent of these disputed clips. This association does not isolate the effect of training labels. The Vox-Profile models, trained on listener annotations of natural podcast speech, side with the listeners about two thirds of the time. Part of Vox-Profile's low accuracy therefore reflects which reference label we score against.`);
  }];

  // ---- Confusion matrix slides
  A.cms = [
    { key: "mcae_spps", title: "Confusion Matrices: MCAE-SPPS",
      bullets: [
        { t: "Solid: angry, neutral, sad", sub: ["≥ 87% recall for both models"] },
        { t: "Shared failures", sub: ["disgusted → neutral (67–68%)", "fearful → sad (60–62%)", "surprised → angry (46–54%)"] },
        { t: "Listeners find these hard too", sub: ["MCAE-SPPS listener recognition: disgusted 63%, fearful 65% (lowest)"] },
      ],
      notes: `MCAE-SPPS is Mandarin. Exposure is not established for these checkpoints; that is not certification that every training component was unseen. The two models behave almost identically: very good on angry, neutral and sad; disgust collapses to neutral, fear to sad, and surprise to anger.
      Shared model mistakes and lower listener recognition motivate inspection of cue ambiguity, but do not establish its cause.` },
    { key: "mess", title: "Confusion Matrices: MESS",
      bullets: [
        { t: "emotion2vec+ (41%)", sub: ["85% of angry and 92% of sad clips → neutral", "MESS equalises loudness, which may remove a cue it relies on (not yet tested)"] },
        { t: "MERaLiON (75%)", sub: ["angry, happy, sad ≥ 79%", "MESS \"calm\" → sad in 52%: calm is not quite neutral (label mapping)"] },
        { t: "Implication", sub: ["Retraining the classification head on in-domain clips (option A) is the low-cost remedy for this type of shift"] },
      ],
      notes: `MESS shows two different failure modes. emotion2vec+ predicts neutral for nearly everything, including 85 percent of angry clips. MESS normalises every clip to the same loudness, which may remove a cue the model relies on; that is a hypothesis we have not tested. Either way it is a domain problem, and exactly what a new head trained on a few hundred local clips would address.
      MERaLiON is good on three classes; its main error is calm heard as sad. Remember we mapped MESS calm to neutral, so part of this is our label mapping.` },
    { key: "tess", title: "Confusion Matrices: TESS",
      bullets: [
        { t: "emotion2vec+ (training involvement, 86%)", sub: ["perfect on six emotions", "\"pleasant surprise\" → fearful in 70%, never surprised"] },
        { t: "MERaLiON (upstream involvement, 49%)", sub: ["neutral → sad in 100%", "disgusted → happy 60%, fearful → sad 66%"] },
        { t: "Only 2 speakers", sub: ["a systematic misjudgement of one voice shifts an entire class; speaker diversity matters as much as the number of clips"] },
      ],
      notes: `TESS has only two speakers. Both systems have documented training-process involvement, including upstream lineage. emotion2vec+ is nearly perfect except for pleasant surprise, which it calls fearful; label conventions are a possible contributor, not an isolated cause.
      MERaLiON has probable upstream training involvement in TESS and hears every neutral clip as sad in this saved analysis. This is not a seen-versus-unseen experiment. With only two voices, one systematic misjudgement of a voice moves an entire class. For our own data: many different callers, not many clips from a few callers.` },
  ];
  return A;
};

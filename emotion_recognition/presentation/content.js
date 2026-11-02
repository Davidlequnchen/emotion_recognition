// Content that depends on the assets (manifest, stats) and on facts.md
module.exports = function (L, { manifest, stats }) {
  const A = { manifest, stats };
  const { C, W, MX, FONT, MODEL } = L;

  // ---- Survey numbers (facts.md §4)
  A.surveyTitle = "From 42 surveyed SER models to seven we evaluated";
  A.surveyBullets = [
    "Survey: 42 entries (dedicated SER, dimensional, community checkpoints, open audio LLMs); selection: downloadable weights + runnable code, English, classes close to our 7",
    "Dedicated SER models: 0.16–0.64B parameters; open audio LLMs: 5.5–35B; closed LLMs: size not published",
  ];
  A.geminiDate = "2026-05";

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
      const s = L.slide("3 · Models", "MERaLiON-SER-v1: a Whisper encoder with a speaker-verification style head");
      L.text(s, "Whisper-medium encoder (speech recognition pre-training) · 307.2M", MX, 1.4, 8, 0.3, { fontSize: 13, bold: true, color: C.accent });
      row(s, [
        ["Waveform\npadded / cut\nto 30 s", 1.45, ...IO, ""],
        ["Log-mel\n80 × 3000", 1.3, ...IO, ""],
        ["Conv stem\n2 × Conv1d\n(stride 2)", 1.5, ...ENC, "3.4M"],
        ["+ sinusoidal\npositions\n(fixed)", 1.45, ...ENC, "1.5M"],
        ["24 Transformer blocks\nd 1024 · 16 heads · FFN 4096\nLoRA r16 on q/k/v (merged)", 3.15, ...ENC, "302.3M"],
        ["Frames\n1500 × 1024", 1.45, ...IO, ""],
      ], 1.8, 1.15);
      L.text(s, "Downstream head (ECAPA-TDNN, from speaker verification) · 6.0M, and two output heads · total 313.4M", MX, 3.55, 11, 0.3, { fontSize: 13, bold: true, color: C.good });
      row(s, [
        ["proj_in\n1024 → 512", 1.35, ...DOWN, "0.53M"],
        ["3 ECAPA blocks\nRes2 convs, dilation\n1 / 2 / 3, SE", 1.9, ...DOWN, "2.23M"],
        ["Multi-layer\naggregation\n2048 → 512", 1.6, ...DOWN, "1.05M"],
        ["Hybrid attention\npooling: 4 scales +\n3 resolutions", 1.95, ...DOWN, "1.81M"],
        ["Embedding\nMLP → 256", 1.4, ...DOWN, "0.40M"],
      ], 3.95, 1.15);
      L.box(s, 10.1, 3.85, 2.63, 0.6, "7 emotions (softmax)", { fill: "FCE9D9", line: C.accent2, fontSize: 12 });
      L.box(s, 10.1, 4.6, 2.63, 0.6, "valence · arousal · dominance", { fill: "FCE9D9", line: C.accent2, fontSize: 12 });
      L.arrow(s, 9.95, 4.52, 10.08, 4.15, { width: 1.25 }); L.arrow(s, 9.95, 4.52, 10.08, 4.9, { width: 1.25 });
      L.text(s, "68k + 67k params", 10.1, 5.25, 2.63, 0.28, { fontSize: 11, color: C.muted, align: "center" });
      L.bullets(s, [
        { runs: [{ text: "Original training: ", options: { bold: true } }, { text: "encoder frozen + LoRA (2.4M) on attention; downstream and heads trained · 15 epochs on 8 × H100 · weighted CE + CCC loss · MixUp, noise, speed perturbation" }] },
        { runs: [{ text: "Fixed 30 s window: ", options: { bold: true } }, { text: "every clip costs 1.15 TFLOP (13× emotion2vec+ on 4 s); padding is not masked in pooling, so a 4 s clip is 200 real frames of 1,500" }] },
      ], { y: 5.75, h: 1.25, fontSize: 14 });
      L.notes(s, `MERaLiON-SER-v1 from A*STAR in Singapore, 2025. The backbone is the encoder of Whisper-medium, a speech recognition model: a convolutional stem and 24 Transformer blocks, 307 million parameters, 98 percent of the model. During training it was frozen and adapted with LoRA on the attention projections, then merged.
      On top sits a head borrowed from speaker verification, ECAPA-TDNN, with multi-scale attention pooling, about six million parameters, and two output heads: seven emotions and three dimensions.
      Two practical points. Whisper always processes 30 seconds, so a two-second clip costs as much as a thirty-second one. And the pooling does not mask the padding, so short clips are mostly silence to the head. Both matter for call audio and are things we can change when we fine-tune.`);
    },
    (L) => {
      const s = L.slide("3 · Models", "Melati and Vox-Profile: large Whisper or WavLM encoders with small heads");
      const blocks = [
        ["Melati · 637.3M · Scicom 2026 · non-commercial", MODEL.melati.color, [
          ["Waveform\n≤ 30 s window", 1.5, ...IO, ""], ["Whisper large-v3 encoder\n32 blocks · d 1280 · frozen", 3.3, ...ENC, "637.0M"],
          ["Mean-pool each\nof 33 layers →\nlearned weights", 2.1, ...DOWN, ""], ["LayerNorm →\n1280 → 256 → 7", 1.9, ...DOWN, "0.33M (the only trained part)"]]],
        ["Vox-Profile Whisper · 639.9M · USC 2025 · non-commercial", MODEL.voxw.color, [
          ["Waveform\n≤ 15 s window", 1.5, ...IO, ""], ["Whisper large-v3 encoder\n+ LoRA r16 on FFN, blocks 17–31", 3.3, ...ENC, "636.0M + 3.1M LoRA"],
          ["1×1 convs +\nmean pooling", 2.1, ...DOWN, ""], ["MLP heads\n9 classes (+ others)", 1.9, ...DOWN, "0.8M"]]],
        ["Vox-Profile WavLM · 316.2M · USC 2025 · non-commercial", MODEL.voxl.color, [
          ["Waveform\nreal length", 1.5, ...IO, ""], ["WavLM-large\nfully fine-tuned", 3.3, ...ENC, "315.5M"],
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
      L.notes(s, `The other three open models follow the same recipe with bigger encoders. Melati freezes the Whisper large-v3 encoder, 637 million parameters, and trains only a third of a million: a learned mix of the 33 layer outputs and a small MLP. Vox-Profile has a Whisper version and a WavLM version, trained on natural podcast speech; the WavLM one is fully fine-tuned.
      All three are research-only licences. And note the compute column: the Whisper-based models cost about ten to twenty-five times more per clip than emotion2vec+.`);
    },
  ];

  // ---- MERaLiON fine-tuning diagram
  A.merFT = (L) => {
    const s = L.slide("6 · Next steps", "MERaLiON-SER-v1: keep the Whisper encoder frozen, retrain the head");
    const y0 = 2.0, bh = 1.25;
    const FR = ["E3E8EF", "8C99AA"], OPT = ["FCE9D9", C.accent2], TR = ["E2F2EA", C.good];
    const items = [
      ["Waveform\n(30 s window)", 1.35, C.panel, C.line, "—"],
      ["Conv stem +\npositions", 1.4, ...FR, "4.9M"],
      ["Whisper encoder\n24 blocks\n(1024-d, 16 heads)", 2.3, ...FR, "302.3M"],
      ["LoRA r8 on q/k/v\n(all 24 blocks)", 1.7, ...OPT, "1.18M"],
      ["ECAPA-TDNN +\nhybrid pooling +\nembedding MLP", 2.0, ...TR, "6.0M"],
      ["Heads\n7 emotions\n(+ V/A/D)", 1.45, ...TR, "68k–135k"],
    ];
    let x = MX;
    items.forEach(([t, w, f, l, prm], i) => {
      L.box(s, x, y0, w, bh, t, { fill: f, line: l, fontSize: 13, lw: 1.5, dash: i === 3 ? "dash" : "solid" });
      L.text(s, prm, x, y0 + bh + 0.08, w, 0.3, { fontSize: 12, color: C.muted, align: "center" });
      if (i < items.length - 1) L.arrow(s, x + w + 0.01, y0 + bh / 2, x + w + 0.17, y0 + bh / 2, { width: 1.25 });
      x += w + 0.18;
    });
    const lg = [["E3E8EF", "8C99AA", "frozen (A, B, C)"], ["FCE9D9", C.accent2, "added in C (LoRA adapters)"], ["E2F2EA", C.good, "trained (A: heads only; B, C: all green)"]];
    lg.forEach(([f, l, t], i) => { const lx = MX + i * 4.1; s.addShape(L.pptx.ShapeType.roundRect, { x: lx, y: 3.95, w: 0.35, h: 0.25, rectRadius: 0.04, fill: { color: f }, line: { color: l, width: 1.5 } }); L.text(s, t, lx + 0.45, 3.93, 3.6, 0.3, { fontSize: 13, color: C.text }); });
    L.table(s, ["Option", "Trains", "Trainable", "Cost per training clip"], [
      ["A", "heads only (on cached 256-d embeddings)", "68k", "none after one forward pass"],
      [{ text: "B", bold: true }, { text: "ECAPA + pooling + MLP + heads (as in the original recipe)", bold: true }, { text: "6.1M", bold: true }, "cache encoder output once (3 MB per clip): minutes per run"],
      ["C", "B + LoRA r8 on encoder q/k/v", "7.3M", "≈ 2.3 TFLOP (full encoder pass each step)"],
      ["D", "everything", "311.8M", "≈ 3.4 TFLOP · 10k clips × 10 epochs ≈ 0.8–1 A100-h"],
    ], { y: 4.45, w: 8.3, colW: [0.8, 3.6, 1.0, 2.9], fontSize: 12, rowH: 0.42 });
    L.bullets(s, [
      "B mirrors how MERaLiON itself was trained: encoder frozen, head trained",
      "While fine-tuning, fix two things: mask padding in pooling; try a shorter window (8 s ≈ 4.4× cheaper, needs retraining)",
    ], { x: 9.15, y: 4.4, w: 3.6, h: 2.5, fontSize: 14 });
    L.notes(s, `The same logic for MERaLiON. Ninety-eight percent of its parameters are the Whisper encoder, which was already frozen in the original training. So option B, retraining the ECAPA head, the pooling and the output heads, six million parameters, is exactly the original recipe applied to our data. Because the encoder is frozen we can cache its output once and each training run takes minutes.
    If B plateaus, option C adds LoRA adapters to the encoder attention, about 1.2 million extra parameters. Fine-tuning is also the moment to fix the unmasked padding and possibly shorten the 30-second window, which would make inference about four times cheaper.`);
  };

  // ---- LISTEN bullets
  A.listen = [
    { t: "Our result", b: true, sub: ["GPT Realtime: below chance on 4 of 6 corpora (mean 16% vs 18% chance)", "Gemini 3.5 Flash: above chance everywhere, but 31–64 points behind the best open model"] },
    { t: "LISTEN (Ohio State et al., arXiv 2510.10444)", b: true, sub: [
      "\"Do audio LLMs really listen, or just transcribe?\" 6 models incl. Gemini 2.5 Flash / Pro, Qwen-Omni",
      "Neutral words, audio only: 16.5–34.9% accuracy; models default to \"neutral\"",
      "Adding the transcript often hurts (Qwen2.5-Omni 34.0 → 19.8%)",
      "Non-verbal sounds (laughs, sighs): 15.7–22.7%, near chance"] },
    { t: "Take-away: today's audio LLMs read emotion from words; our corpora remove that cue on purpose", b: true, color: C.accent2 },
  ];
  A.listenNotes = `Our LLM results agree with a 2025 benchmark called LISTEN. Its authors tested six audio LLMs, including Gemini 2.5, and found that when the words are emotionally neutral, the models mostly answer neutral: 16 to 35 percent accuracy. Adding the transcript often made things worse, and on purely non-verbal sounds they were near chance.
  Our corpora are designed exactly this way: neutral words, emotional voice. GPT Realtime is at or below chance on four of six corpora; Gemini does better but is far behind the dedicated models. Today's audio LLMs mostly read emotion from what is said, not how it is said.`;

  // ---- Which ground truth do models follow? (after the confusion matrices)
  A.afterCms = [(L) => {
    const s = L.slide("4 · Results", "Which ground truth do models learn? Actor-trained models follow the actor");
    const w = stats.crema_d_listeners.when_listeners_disagree_with_actor;
    const keys = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm"];
    const names = keys.map((k) => MODEL[L.MODEL_KEY[k]].name.replace("Vox-Profile ", "Vox "));
    s.addChart(L.pptx.ChartType.bar, [
      { name: "predicts the actor's emotion", labels: names, values: keys.map((k) => +(100 * w.models[k].predicts_acted).toFixed(1)) },
      { name: "predicts the listeners' majority", labels: names, values: keys.map((k) => +(100 * w.models[k].predicts_listener_majority).toFixed(1)) },
    ], { x: MX, y: 1.45, w: 7.6, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: ["1B2A41", "D55E00"],
      showValue: true, dataLabelFormatCode: "0", dataLabelFontSize: 12, dataLabelFontFace: FONT, catAxisLabelFontSize: 12, catAxisLabelFontFace: FONT,
      valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFontSize: 11, valAxisTitle: "% of the 3,699 disputed clips", showValAxisTitle: true, valAxisTitleFontSize: 12,
      showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: FONT, valGridLine: { color: "E5E8EC", size: 0.5 } });
    L.bullets(s, [
      { t: "3,699 CREMA-D clips where the listeners' majority differs from the actor", b: true },
      { t: "emotion2vec+ and MERaLiON (trained on CREMA-D's acted labels) reproduce the actor in 77–82% of them" },
      { t: "Vox-Profile (trained on perceived labels of natural podcast speech) agrees with the listeners in 62–65%: its low accuracy is partly the choice of ground truth" },
      { t: "For us: we will label what listeners perceive, so the models must be adapted to perceived labels", b: true, color: C.accent2 },
    ], { x: 8.5, y: 1.55, w: 4.25, h: 5.3, fontSize: 15 });
    L.notes(s, `A revealing analysis. Take the 3,699 CREMA-D clips where listeners disagree with the actor. emotion2vec+ and MERaLiON, which were trained on CREMA-D's acted labels, predict the actor's emotion in about 80 percent of them: they have learned the actor labels, not what people hear. The Vox-Profile models, trained on perceived emotion in natural podcasts, side with the listeners about two thirds of the time.
    So part of Vox-Profile's low accuracy is that it answers a different question. Since our own labels will come from listeners, this is another reason to adapt the models on our perceived labels.`);
  }];

  // ---- Confusion matrix slides
  A.cms = [
    { key: "mcae_spps", title: "Mandarin (unseen by both): the two models fail on the same three emotions",
      bullets: [
        { t: "Solid: angry, neutral, sad", sub: ["≥ 87% recall for both models"] },
        { t: "Shared failures", sub: ["disgusted → neutral (67–68%)", "fearful → sad (60–62%)", "surprised → angry (46–54%)"] },
        { t: "Listeners find these hard too", sub: ["MCAE-SPPS listener recognition: disgusted 63%, fearful 65% (lowest)"] },
      ],
      notes: `MCAE-SPPS is Mandarin and neither model has seen it, so this is the cleanest comparison. The two models behave almost identically: very good on angry, neutral and sad; disgust collapses to neutral, fear to sad, and surprise to anger.
      When two independently trained models make the same mistakes, and listeners also find disgust and fear the hardest, the cue is probably ambiguous in the audio itself.` },
    { key: "mess", title: "MESS: emotion2vec+ says neutral to almost everything",
      bullets: [
        { t: "emotion2vec+ (41%)", sub: ["85% of angry and 92% of sad clips → neutral", "MESS equalises loudness, which may remove a cue it relies on (not yet tested)"] },
        { t: "MERaLiON (75%)", sub: ["angry, happy, sad ≥ 79%", "MESS \"calm\" → sad in 52%: calm is not quite neutral (label mapping)"] },
        { t: "Lesson", sub: ["A new head on a few hundred in-domain clips (option A) is the cheap fix for this kind of shift"] },
      ],
      notes: `MESS shows two different failure modes. emotion2vec+ predicts neutral for nearly everything, including 85 percent of angry clips. MESS normalises every clip to the same loudness, which may remove a cue the model relies on; that is a hypothesis we have not tested. Either way it is a domain problem, and exactly what a new head trained on a few hundred local clips would address.
      MERaLiON is good on three classes; its main error is calm heard as sad. Remember we mapped MESS calm to neutral, so part of this is our label mapping.` },
    { key: "tess", title: "TESS: even a model trained on the corpus can miss a whole class",
      bullets: [
        { t: "emotion2vec+ (seen, 86%)", sub: ["perfect on six emotions", "\"pleasant surprise\" → fearful in 70%, never surprised"] },
        { t: "MERaLiON (unseen, 49%)", sub: ["neutral → sad in 100%", "disgusted → happy 60%, fearful → sad 66%"] },
        { t: "Only 2 speakers", sub: ["two voices can shift a whole class at once: collect many callers, not many clips per caller"] },
      ],
      notes: `TESS has only two speakers. emotion2vec+ has seen it and is perfect except for pleasant surprise, which it calls fearful: a label-convention issue.
      MERaLiON, which has not seen TESS, hears every neutral clip as sad. With only two voices, one systematic misjudgement of a voice moves an entire class. For our own data: many different callers, not many clips from a few callers.` },
  ];
  return A;
};

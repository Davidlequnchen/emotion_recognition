// Path E: the encoder, the teacher-student objective, masking, equations, one training step, original vs ours
const R = require("./refs");

module.exports = function (L) {
  const { C, W, MX, CW, TOP, FONT } = L;
  const p = L.pptx;
  const SEC = "Path E · continued self-supervised pretraining";

  // ---- The encoder: where masking happens and where targets come from
  {
    const s = L.slide(SEC, "Inside emotion2vec+ large: masking enters after the CNN");
    const x0 = MX, bw = 4.55, bh = 0.48, step = 0.65, y0 = 2.08;
    const px = x0 + bw + 0.1, pw = 0.95, ax = px + pw + 0.15, tx = ax + 0.5;
    L.text(s, "waveform · 16 kHz · normalised per clip", x0, 1.64, bw, 0.32, { fontSize: 18, align: "center", kind: "diagram" });
    L.arrow(s, x0 + bw / 2, 1.96, x0 + bw / 2, y0);
    const parts = [
      ["CNN feature extractor · 7 conv layers", "4.2M", "frozen"],
      ["LayerNorm + Linear 512 → 1024", "0.5M", "train"],
      ["convolutional positional encoder", "6.2M", "train"],
      ["+ 10 extra tokens (prepended)", "10K", "train"],
      ["4 prenet Transformer blocks", "50.4M", "train"],
      ["8 shared Transformer blocks · 16 heads", "100.8M", "train"],
    ];
    parts.forEach(([t, n, role], i) => {
      const y = y0 + i * step;
      L.box(s, x0, y, bw, bh, t, { role, fontSize: 18, bold: i === 5 });
      L.text(s, n, px, y, pw, bh, { fontSize: 18, bold: true, align: "right", valign: "middle", kind: "diagram" });
      if (i < 5) L.arrow(s, x0 + bw / 2, y + bh, x0 + bw / 2, y + step);
    });
    const yOut = y0 + 5 * step + bh;
    L.arrow(s, x0 + bw / 2, yOut, x0 + bw / 2, yOut + 0.17);
    L.text(s, "frames: T × 1024, 50 per second", x0, yOut + 0.17, bw, 0.34, { fontSize: 18, align: "center", kind: "diagram" });
    // the student mask sits between the projection and the positional encoder
    const ym = y0 + 1 * step + bh + (step - bh) / 2;
    s.addShape(p.ShapeType.line, { x: x0 - 0.2, y: ym, w: bw + 0.2 + pw + 0.2, h: 0, line: { color: C.bad, width: 2.5, dashType: "dash" } });
    // annotations with leader lines
    const note = (i, t, o = {}) => {
      const yc = o.y ?? (y0 + i * step + bh / 2);
      s.addShape(p.ShapeType.line, { x: ax, y: yc, w: 0.38, h: 0, line: { color: o.color || C.ink, width: 1.5 } });
      L.text(s, t, tx, yc - (o.h || 0.5) / 2, W - MX - tx, o.h || 0.5, { fontSize: 18, valign: "middle", color: o.text || C.ink, bold: !!o.bold, kind: "diagram" });
    };
    note(0, "stride 320 samples: one frame per 20 ms · frozen in Path E");
    note(0, "student mask: about 50% of frames set to zero here (teacher: no mask)", { y: ym, color: C.bad, text: C.darkRed, bold: true });
    note(3, "added after masking, so they are never masked or scored");
    note(4, "option B reads all 12 block outputs (prenet + shared)");
    note(5, [{ text: "SSL target (final): last block, layer norm", options: { breakLine: true } },
      { text: "option A: mean of the last block’s output" }], { h: 0.62 });
    L.text(s, "Total 162.1M (9-class output layer removed)", tx, 5.98, W - MX - tx, 0.36, { fontSize: 18, bold: true, kind: "diagram" });
    L.key(s, tx, 6.42, "frozen", "frozen in Path E");
    L.key(s, tx + 2.75, 6.42, "train", "trained in Path E");
    L.notes(s, `To understand Path E we need the inside of the encoder. This is the published emotion2vec+ large encoder as FunASR builds it, from top to bottom; parameter counts are computed from the local weights.
    The waveform first goes through a CNN feature extractor: seven convolution layers with 512 channels and a total stride of 320 samples, so at 16 kHz we get one frame every 20 milliseconds, 50 frames per second. A layer norm and a linear layer project each frame from 512 to 1024 dimensions.
    The red dashed line is where Path E's student mask is applied: about half of these projected frames are set to zero. The teacher sees the same clip without any mask.
    Then a convolutional positional encoder adds relative position information. Ten learned extra tokens are prepended; because this happens after masking, they are never masked, and the frame loss never scores them. Four prenet Transformer blocks belong to the audio-specific encoder, followed by eight shared Transformer blocks with 1024 dimensions and 16 attention heads.
    The default SSL target, called final, is the teacher's last block output, layer-normalised per frame. The earlier target, ffn_mean, averaged the feed-forward outputs of all eight shared blocks, each normalised over time; it is kept as a control. In Path A/B, option A's clip embedding is the mean of the last block's output frames, while option B averages the output of each of the 12 blocks, prenet and shared, and learns how to mix them.
    Colours refer to Path E: by default only the CNN, 4.2 million parameters, is frozen; everything else, 157.9 million parameters, is trained. In Path A/B the whole encoder is frozen. The released model also has a 9-class output layer, which we remove; the total encoder size is 162.1 million parameters.`);
    L.cites(s, R, ["e2v", "d2v2", "funasr"], "Source: model config.yaml.");
  }

  // ---- The key figure: teacher-student objective
  {
    const s = L.slide(SEC, "Path E: a masked student predicts an EMA teacher’s targets");
    const yT = 1.85, yS = 4.75, h = 1.0, cT = yT + h / 2, cS = yS + h / 2;
    const xc = 11.38, wc = W - MX - xc;                       // right column: target, loss, prediction
    // input and split
    L.box(s, MX, 3.32, 1.5, 1.06, "Clip\n2–15 s", { role: "data", fontSize: 18, bold: true });
    const yi = 3.85, xs = 2.35;
    L.path(s, [[MX + 1.5, yi], [xs, yi], [xs, cT], [2.6, cT]]);
    L.path(s, [[xs, yi], [xs, cS], [2.6, cS]]);
    // teacher row
    const T = [[2.6, 2.9], [5.8, 2.3], [8.4, 2.55], [xc, wc]];
    L.box(s, T[0][0], yT, T[0][1], h, "Teacher encoder\ncopy of the encoder, full clip\nno gradients · eval mode", { role: "teacher", fontSize: 16, bold: false });
    L.box(s, T[1][0], yT, T[1][1], h, "last block output\n(target: final)", { role: "data", fontSize: 16 });
    L.box(s, T[2][0], yT, T[2][1], h, "layer norm\nper frame", { role: "data", fontSize: 16 });
    L.box(s, T[3][0], yT, T[3][1], h, "target y\nT × 1024", { role: "data", fontSize: 16, bold: true });
    for (let i = 0; i < 3; i++) L.arrow(s, T[i][0] + T[i][1], cT, T[i + 1][0], cT);
    // student row
    const S = [[2.6, 1.6], [4.5, 1.9], [6.7, 2.2], [9.2, 1.75], [xc, wc]];
    L.box(s, S[0][0], yS, S[0][1], h, "CNN\nfront end", { role: "frozen", fontSize: 16 });
    L.box(s, S[1][0], yS, S[1][1], h, "span mask:\n~50% of valid\nframes → 0", { fill: C.white, line: C.bad, dash: "dash", lw: 2, fontSize: 16 });
    L.box(s, S[2][0], yS, S[2][1], h, "Student encoder\nprojection +\n12 blocks", { role: "train", fontSize: 16, bold: true });
    L.box(s, S[3][0], yS, S[3][1], h, "MLP predictor gφ\n1024 → 1024\n→ 1024", { role: "train", fontSize: 16 });
    L.box(s, S[4][0], yS, S[4][1], h, "prediction p\nT × 1024", { role: "data", fontSize: 16, bold: true });
    for (let i = 0; i < 4; i++) L.arrow(s, S[i][0] + S[i][1], cS, S[i + 1][0], cS);
    L.pill(s, S[3][0] + S[3][1] - 0.6, yS - 0.17, "new");
    // loss
    const yL = 3.32, hL = 1.06, cx = xc + wc / 2;
    L.box(s, xc, yL, wc, hL, "MSE on\nmasked\nframes", { role: "loss", fontSize: 16, bold: true });
    L.arrow(s, cx, yT + h, cx, yL, { color: C.bad, width: 2 });
    L.text(s, "stop-gradient", cx - 1.72, yT + h + 0.06, 1.62, 0.32, { fontSize: 16, bold: true, color: C.darkRed, align: "right", kind: "diagram" });
    L.arrow(s, cx, yS, cx, yL + hL, { width: 2 });
    // gradient back into the predictor and the student
    const gy = 4.2, gx3 = S[2][0] + S[2][1] - 0.45, gx4 = S[3][0] + S[3][1] / 2;
    L.path(s, [[xc, gy], [gx3, gy], [gx3, yS]], { color: C.accent2, dash: "dash", width: 2.25 });
    L.arrow(s, gx4, gy, gx4, yS, { color: C.accent2, dash: "dash", width: 2.25 });
    L.text(s, "gradient → AdamW", gx3 + 0.12, gy - 0.4, 2.6, 0.34, { fontSize: 16, bold: true, color: C.darkOrange, kind: "diagram" });
    // EMA from the student back to the teacher
    const ex = S[2][0] + 0.45, ey = 3.55, tcx = T[0][0] + T[0][1] / 2;
    L.path(s, [[ex, yS], [ex, ey], [tcx, ey], [tcx, yT + h]], { color: C.teacher, dash: "dash", width: 2.25 });
    L.text(s, [{ text: "EMA after each optimizer step:", options: { breakLine: true } }, { text: "θT ← 0.999 θT + 0.001 θS" }],
      tcx + 0.15, ey + 0.08, 3.3, 0.64, { fontSize: 16, bold: true, color: C.darkPurple, kind: "diagram" });
    // legend
    const l1 = 6.0, l2 = 6.42;
    L.key(s, MX, l1, "frozen");
    L.key(s, MX + 3.0, l1, "train");
    L.key(s, MX + 5.0, l1, "teacher");
    L.key(s, MX + 8.4, l1, "loss");
    L.key(s, MX, l2, C.ink, "forward pass", { arrow: true });
    L.key(s, MX + 3.0, l2, C.accent2, "gradient", { arrow: true, dash: "dash" });
    L.key(s, MX + 5.0, l2, C.teacher, "EMA weight update", { arrow: true, dash: "dash" });
    L.pill(s, MX + 8.4, l2 + 0.01, "new");
    L.text(s, "added by this project", MX + 9.22, l2, 2.8, 0.34, { fontSize: 16, valign: "middle", kind: "diagram" });
    L.notes(s, `This is the key figure of Path E. It is masked feature regression with an exponential-moving-average teacher, the recipe of data2vec and emotion2vec, rebuilt as a new adaptation objective on top of the released encoder.
    One clip enters on the left and is used twice. The purple top row is the teacher: a copy of the encoder that receives no gradients and always runs in evaluation mode, so dropout is off. It sees the full clip. By default, target final, we take the output of its last Transformer block and layer-normalise each frame across its 1024 channels, without subtracting the mean over time. That is the target y: one 1024-dimensional vector per frame. The earlier target, ffn_mean, the mean of the top 8 feed-forward outputs normalised over time as in data2vec 2.0, is kept as a control; the mock test showed it cannot be learned from this fine-tuned model.
    The bottom row is the student. The CNN front end is frozen by default. On the CNN's frames we sample a span mask that covers about half of the valid frames and set those frames to zero. The student encoder, which is the projection, the positional encoder, the extra tokens and the twelve Transformer blocks, is trained. A new two-layer MLP predictor, 1024 to 1024 to 1024 with a GELU, maps each student frame to a prediction p. The predictor is ours; it has 2.1 million parameters and is not exported with the encoder.
    The loss, in red, is the mean squared error between p and y, but only on frames that were masked and are real audio. The target is a constant for the optimiser: stop-gradient. So the student must infer the hidden frames from their context, in the teacher's feature space.
    The dashed orange arrows are the gradient: AdamW updates the student encoder and the predictor. After each optimizer step, the dashed purple arrow updates the teacher: its weights move 0.1 percent of the way towards the student's. The teacher is therefore a slowly moving average of recent students, which gives stable targets.
    A decreasing loss only means the student predicts the teacher better. It does not show better emotion recognition; that needs a head trained on top and evaluated on labelled clips.`);
    L.cites(s, R, ["data2vec", "byol", "e2v"], "Source: adaptation/ssl.py.");
  }

  // ---- Masking explained visually
  {
    const s = L.slide(SEC, "Masking hides half of the real frames in spans of five");
    const lx = MX, lw = 2.1, x0 = MX + 2.25, n = 60, cw = (W - MX - x0) / n, gap = 0.022, rh = 0.48;
    const extra = 10, valid = 40;
    const masked = new Set([2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31].map((i) => i + extra));
    const kind = (i) => (i < extra ? "extra" : (i < extra + valid ? "valid" : "pad"));
    const cell = (i, y, st) => {
      const x = x0 + i * cw + gap / 2;
      s.addShape(p.ShapeType.rect, { x, y, w: cw - gap, h: rh, fill: { color: st.fill }, line: { color: st.line, width: st.lw || 0.75, dashType: st.dash || "solid" } });
    };
    const ST = {
      extra: { fill: C.purpleFill, line: C.teacher }, valid: { fill: C.blueFill, line: C.accent },
      masked: { fill: C.accent2, line: C.darkOrange }, pad: { fill: C.white, line: C.ink, dash: "dash" },
      loss: { fill: C.bad, line: C.darkRed }, none: { fill: C.white, line: C.line },
    };
    // group labels above the strip
    const ly = 1.6, lh = 0.32;
    L.text(s, "10 extra tokens", x0, ly, extra * cw, lh, { fontSize: 16, align: "center", bold: true, kind: "diagram" });
    L.text(s, "valid frames, 20 ms each (schematic: 40 shown of 99–749)", x0 + extra * cw, ly, valid * cw, lh, { fontSize: 16, align: "center", bold: true, kind: "diagram" });
    L.text(s, "padding", x0 + (extra + valid) * cw, ly, 10 * cw, lh, { fontSize: 16, align: "center", bold: true, kind: "diagram" });
    const rows = [
      ["Teacher input", (i) => ST[kind(i)]],
      ["Student input", (i) => (masked.has(i) ? ST.masked : ST[kind(i)])],
      ["Frame loss on", (i) => (masked.has(i) ? ST.loss : ST.none)],
    ];
    rows.forEach(([label, style], r) => {
      const y = 2.0 + r * 0.72;
      L.text(s, label, lx, y, lw, rh, { fontSize: 20, bold: true, valign: "middle" });
      for (let i = 0; i < n; i++) cell(i, y, style(i));
    });
    // legend
    const ky = 4.2;
    L.key(s, x0, ky, C.purpleFill, "extra token", { edge: C.teacher, sw: 0.3 });
    L.key(s, x0 + 1.75, ky, C.blueFill, "valid frame", { edge: C.accent, sw: 0.3 });
    L.key(s, x0 + 3.5, ky, C.accent2, "masked: set to 0", { edge: C.darkOrange, sw: 0.3 });
    L.key(s, x0 + 5.75, ky, C.white, "padding (unbucketed only)", { edge: C.ink, dash: "dash", sw: 0.3, w: 2.7 });
    L.key(s, x0 + 8.65, ky, C.bad, "scored", { edge: C.darkRed, sw: 0.3, w: 1.0 });
    L.text(s, [{ text: "Real clips: ", options: { bold: true } }, { text: "2 s → 99 frames → 50 masked  ·  15 s → 749 frames → 374 masked" }],
      MX, 4.68, CW, 0.4, { fontSize: 20, valign: "middle" });
    // four rules
    const cards = [
      ["50% of valid frames", "round(0.5 × valid); at least one frame stays visible"],
      ["Spans of 5 frames", "100 ms each, placed at random until the count is reached"],
      ["Zeroed after the CNN", "masked frames are set to 0 after the projection"],
      ["Padding never scored", "and extra tokens are added after masking"],
    ];
    const cw4 = (CW - 3 * 0.25) / 4, cy = 5.22, ch = 1.55;
    cards.forEach(([t, b], i) => {
      const x = MX + i * (cw4 + 0.25);
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw4, h: ch, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw4, h: 0.07, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
      L.text(s, t, x + 0.15, cy + 0.15, cw4 - 0.3, 0.38, { fontSize: 20, bold: true });
      L.text(s, b, x + 0.15, cy + 0.58, cw4 - 0.3, 0.9, { fontSize: 18, kind: "diagram" });
    });
    L.notes(s, `This slide shows the mask on one clip of a batch, as a strip of encoder positions. It is a schematic: 40 valid frames are drawn, while a real clip of 2 to 15 seconds has 99 to 749 frames.
    The teacher input, top row, has ten extra tokens, then the valid frames of real audio, each 20 milliseconds, then padding. Padding only appears when bucket_by_duration is switched off: by default, batches hold clips of near-equal duration and every clip is cropped to the batch's shortest clip, so no padding reaches the encoder at all.
    The student input, middle row, is the same, except that the orange frames are masked: they are set to zero after the CNN and its projection, before the positional encoder. The mask is sampled per clip. The number of masked frames is 50 percent of the valid frames, rounded, and at least one frame always stays visible. Spans of five frames, 100 milliseconds, start at valid positions in random order until that number is reached; neighbouring spans can touch and form a longer block, as the right-most block here shows, and a span at the end of the clip can be shorter. For real clips, a 2-second clip has 99 frames, of which 50 are masked; a 15-second clip has 749 frames, of which 374 are masked.
    The bottom row shows where the frame loss is computed: only on masked valid frames. Padding is never masked and never scored. Visible frames get no direct loss, but they still matter, because attention uses them as context to fill in the hidden frames, and they receive gradient through that path.
    Our setting differs from the original description: the paper describes a probability that a span starts at a frame, while our mask_probability is the fraction of valid frames covered. The numbers are therefore not interchangeable. Span masking on latent speech frames comes from wav2vec 2.0 and the data2vec family.`);
    L.cites(s, R, ["wav2vec2", "d2v2"], "Source: adaptation/ssl.py sample_span_mask(); config/pretrain.yaml.");
  }

  // ---- Equations 1: targets and predictions
  {
    const s = L.slide(SEC, "Default target: the teacher’s last block, layer-normalised", { titleSize: 28 });
    const lw = 6.2;
    const lab = (t, y, color) => L.text(s, t, MX, y, lw, 0.36, { fontSize: 20, bold: true, color: color || C.link });
    lab("Default target: final (no gradient)", 1.68, C.darkPurple);
    L.eq(s, "eq_final", MX + 0.2, 2.06, lw, 0.6);
    lab("Layer norm over the D = 1024 channels of a frame", 2.78, C.darkPurple);
    L.eq(s, "eq_ln", MX + 0.2, 3.16, lw, 1.3);
    lab("Control: ffn_mean (data2vec 2.0 style)", 4.5, C.ink);
    L.eq(s, "eq_target", MX + 0.2, 4.88, lw, 1.05);
    lab("Student prediction from the masked clip", 5.98, C.darkOrange);
    L.eq(s, "eq_pred", MX + 0.2, 6.36, lw, 0.46);
    L.symtable(s, [
      ["sym_z", "teacher’s last-block output"],
      ["sym_LN", "layer norm per frame, no learned scale"],
      ["sym_h", "FFN output of block k (ffn_mean)"],
      ["sym_IN", "normalise over the clip’s valid frames"],
      ["sym_K", "8 shared blocks (ffn_mean only)"],
      ["sym_xtilde", "clip b with masked frames set to 0"],
      ["sym_f", "student encoder (trainable)"],
      ["sym_g", "new MLP: 1024 → 1024 → 1024"],
    ], { x: 6.95, y: 1.72, w: W - MX - 6.95, symW: 1.3, rowH: 0.52 });
    L.notes(s, `The target is what the masked frames regress onto; this is the part of Path E that changed after the mock test.
    The default target, final, is the teacher's last block output z for clip b and frame t, layer-normalised per frame: for each frame, subtract the mean over its 1024 channels and divide by their standard deviation, with no learned scale or shift and without subtracting any mean over time. Because the last layer of this fine-tuned model is almost constant over time, this largely means recovering the whole clip's representation from partial audio, which matches the model's utterance-level nature.
    The control target, ffn_mean, is the earlier default and the data2vec 2.0 and emotion2vec recipe: the mean over the top K equals 8 shared blocks of their feed-forward outputs h, each normalised by IN over the clip's valid frames, per channel. The mock test showed it cannot be learned from emotion2vec+ large: the eight normalised outputs are nearly uncorrelated, and even with the encoder frozen and no mask the predictor stays at the predict-zero level. It is kept so that the comparison can be repeated.
    The student prediction p is unchanged: the new MLP g phi applied to the student encoder's output at frame t, where the student sees x tilde b, the clip with the masked latent frames set to zero.
    The symbol table defines every symbol. The next slide gives the loss and the update rules.`);
    L.cites(s, R, ["data2vec", "d2v2"], "Source: adaptation/ssl.py forward().");
  }

  // ---- Equations 2: loss, gradient step, EMA
  {
    const s = L.slide(SEC, "Loss on masked frames only; the teacher follows by EMA");
    L.text(s, "Frame loss: mean squared error over masked valid frames and all D = 1024 channels", MX, 1.66, CW, 0.36, { fontSize: 20, bold: true, color: C.darkRed });
    L.eq(s, "eq_loss", MX + 0.2, 2.06, CW - 0.2, 1.1);
    const lw = 5.9, ys = [3.3, 4.36, 5.42];
    const lab = (t, y, color, w = lw) => L.text(s, t, MX, y, w, 0.36, { fontSize: 20, bold: true, color });
    lab("Total loss: utterance and anchor terms optional", ys[0], C.ink, CW);
    L.eq(s, "eq_total", MX + 0.2, ys[0] + 0.38, CW - 0.2, 0.5);
    lab("Gradient step: student 10⁻⁵, predictor 10⁻³", ys[1], C.darkOrange);
    L.eq(s, "eq_update", MX + 0.2, ys[1] + 0.4, lw, 0.6);
    lab("EMA teacher, τ = 0.999, after each step", ys[2], C.darkPurple);
    L.eq(s, "eq_ema", MX + 0.2, ys[2] + 0.4, lw, 0.5);
    const tx = 6.85;
    L.symtable(s, [
      ["sym_M", "masked valid positions (b, t)"],
      ["sym_sg", "stop-gradient: target is a constant"],
      ["sym_lambda", "pooled extra tokens; λ = 0 (off)"],
      ["sym_mu", "L2-SP anchor to original; μ = 0"],
    ], { x: tx, y: 4.36, w: W - MX - tx, symW: 1.6, rowH: 0.5, headH: 0.46 });
    L.notes(s, `The frame loss is a plain mean squared error. M is the set of masked, valid positions in the batch, pairs of clip and frame, and D is 1024 channels. We sum the squared differences between prediction and target over those positions and channels and divide by the count, so the loss is an average per masked frame and channel. sg is the stop-gradient: the target is treated as a constant, so no gradient flows into the teacher. Padding and visible frames are excluded before the reduction, so the batch composition does not change the scale.
    The total loss weights the frame loss by w, frame_weight, 1 by default, and can add two optional terms. The utterance loss, lambda, now follows the paper: the student's ten extra tokens are first averaged, passed through the same predictor and matched to the layer-normalised mean of the teacher's last-block frames; it is off by default and gave no extra benefit in the mock test. The anchor term, mu, is L2-SP: it pulls the student's weights towards the original encoder weights theta zero; it is also off by default. The gradient step updates the student encoder theta S and the predictor phi with AdamW: the new, randomly initialised predictor has its own learning rate of 10 to the minus 3; the encoder's rate is zero during the learnability probe and then warms up linearly to 10 to the minus 5 over 1,000 steps. Weight decay is 0.01 and the gradient norm is clipped at 1.0. The CNN is frozen by default and receives no update.
    Then, once per optimizer step, the teacher is updated by an exponential moving average: the new teacher keeps 99.9 percent of its old weights and takes 0.1 percent from the updated student. This covers all encoder weights; buffers are copied; the predictor has no teacher copy. With tau equal to 0.999 the teacher averages roughly the last thousand student steps, since one over one minus tau is 1,000. Unlike the original, which raises the decay during training, our decay is fixed.`);
    L.cites(s, R, ["byol", "meanteacher", "data2vec"], "Source: ssl.py, pretrain.py.");
  }

  // ---- EMA explained
  {
    const s = L.slide(SEC, "The EMA teacher averages about the last 1,000 student steps", { titleSize: 28 });
    L.image(s, "img/ema_weights.png", MX, TOP + 0.15, 5.9, 3.55, { align: "left", valign: "top", alt: "Weight of past student steps in the EMA teacher" });
    L.text(s, [{ text: "Weight of the student from j steps ago: (1 − τ) τ" }, { text: "j", options: { superscript: true } }, { text: ", τ = 0.999" }], MX, 5.5, 5.9, 0.4, { fontSize: 18, kind: "diagram" });
    L.bullets(s, [
      { runs: [{ text: "Each optimizer step: ", options: { bold: true } }, { text: "the teacher keeps 99.9% of its weights and takes 0.1% from the student" }] },
      { runs: [{ text: "Time constant 1 / (1 − τ) = 1,000 steps: ", options: { bold: true } }, { text: "targets move slowly and stay stable" }] },
      { runs: [{ text: "No gradients, always eval mode: ", options: { bold: true } }, { text: "dropout off; the predictor has no teacher copy" }] },
      { runs: [{ text: "Original emotion2vec ", options: { bold: true } }, { text: "raises τ during training; we keep it fixed" }] },
    ], { x: 6.9, y: TOP + 0.15, w: W - MX - 6.9, h: 4.9, fontSize: 20, gap: 12 });
    L.notes(s, `This slide explains what the EMA update means in practice. After every optimizer step the teacher keeps 99.9 percent of its weights and takes 0.1 percent from the updated student. Unrolled over time, the teacher is a weighted average of all past students: the student from j steps ago has weight one minus tau times tau to the power j, the curve on the left, apart from a small remaining share of the initial copy.
    With tau equal to 0.999 the time constant is one over one minus tau, which is 1,000 steps: the most recent 1,000 student steps carry about 63 percent of the total weight. So the targets follow the student, but slowly; a single noisy step barely moves them. This is why EMA teachers give stable targets in BYOL, Mean Teacher and data2vec.
    The teacher receives no gradients and always runs in evaluation mode, so dropout is off and its targets are deterministic for a given clip. The MLP predictor has no teacher copy; it exists only on the student side.
    The original emotion2vec raises the decay during training, so the teacher freezes more and more towards the end; we keep the decay fixed at 0.999 for this first baseline. The curve is computed by scripts/make_figures.py.`);
    L.cites(s, R, ["meanteacher", "byol", "data2vec"], "Source: ssl.py update_teacher().");
  }

  // ---- One training step, inside three phases
  {
    const s = L.slide(SEC, "Training runs in three phases; every step has four stages", { titleSize: 28 });
    // phase timeline
    const py = 1.72, ph = 0.62, pw = [3.6, 3.6, CW - 7.2 - 0.3];
    const phases = [["1 · Probe: predictor only", C.white, C.ink], ["2 · Warm-up: encoder lr 0 → 10⁻⁵", C.white, C.accent2], ["3 · Training: downstream check every 500", C.white, C.accent2]];
    let px = MX;
    phases.forEach(([t, f, col], i) => {
      L.box(s, px, py, pw[i], ph, t, { fill: i ? C.orangeFill : C.blueFill, line: i ? C.accent2 : C.accent, fontSize: 18, bold: true, r: 0.04 });
      if (i < 2) L.arrow(s, px + pw[i], py + ph / 2, px + pw[i] + 0.15, py + ph / 2, { width: 2 });
      px += pw[i] + 0.15;
    });
    const cards = [
      ["Teacher targets", C.teacher, ["full clip → teacher", "no gradients, eval mode", "last block, layer norm", "per frame → y"], "ssl.py\nforward()"],
      ["Masked prediction", C.accent2, ["spans over ~50% of", "the valid frames", "zeroed after the CNN", "student + MLP → p"], "ssl.py\nforward()"],
      ["Loss and AdamW", C.bad, ["MSE on masked frames", "backward, clip at 1.0", "predictor lr 10⁻³", "encoder lr 10⁻⁵"], "pretrain.py\n_fit()"],
      ["EMA teacher", C.teacher, ["θT ← 0.999 θT", "        + 0.001 θS", "once per optimizer step", "predictor: student only"], "ssl.py\nupdate_teacher()"],
    ];
    const gap = 0.36, cw = (CW - 3 * gap) / 4, cy = 2.62, ch = 3.42;
    cards.forEach(([t, col, lines, ref], i) => {
      const x = MX + i * (cw + gap);
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: ch, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      s.addText(String(i + 1), { x: x + 0.2, y: cy + 0.24, w: 0.52, h: 0.52, shape: p.ShapeType.ellipse, fill: { color: col }, line: { color: col, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, x + 0.82, cy + 0.2, cw - 0.9, 0.6, { fontSize: 20, bold: true, valign: "middle" });
      L.text(s, lines.join("\n"), x + 0.2, cy + 0.92, cw - 0.32, 1.6, { fontSize: 18, ls: 1.08, kind: "diagram" });
      L.box(s, x + 0.2, cy + ch - 0.8, cw - 0.4, 0.62, ref, { fill: C.white, line: C.line, mono: true, fontSize: 16, r: 0.04 });
      if (i < 3) L.arrow(s, x + cw + 0.03, cy + ch / 2, x + cw + gap - 0.03, cy + ch / 2, { width: 2 });
    });
    const yl = cy + ch + 0.24, c1 = MX + cw / 2, c4 = MX + 3 * (cw + gap) + cw / 2;
    L.path(s, [[c4, cy + ch], [c4, yl], [c1, yl], [c1, cy + ch]], { width: 2 });
    L.text(s, "next batch", (c1 + c4) / 2 - 0.8, yl - 0.18, 1.6, 0.36, { fontSize: 18, bold: true, align: "center", valign: "middle", kind: "diagram", extra: { fill: { color: C.white } } });
    L.text(s, "During the probe only the predictor learns; the student still feeds the EMA, which then leaves the teacher equal to the original.", MX, 6.5, CW, 0.34, { fontSize: 16, kind: "diagram" });
    L.notes(s, `Training now runs in three phases, and every step inside them has the same four stages.
    Phase 1, the learnability probe: for 500 steps by default, only the new predictor is trained; the encoder's learning rate is zero, so its weights do not change. At the end, the probe checks on the SSL validation clips whether the predictor has learned anything beyond a constant prediction; if not, the run stops with the encoder untouched. The next slide shows this gate.
    Phase 2, warm-up: the encoder's learning rate ramps linearly from zero to 10 to the minus 5 over 1,000 steps, so the first encoder updates, driven by a still-young predictor, are small.
    Phase 3, training at the full rate, with the downstream check every 500 steps.
    Inside every step, stage 1: under no_grad, the teacher, always in evaluation mode, encodes the full clip; the default target is its last block output, layer-normalised per frame. Stage 2: the span mask covers about half of the valid frames, which are set to zero after the CNN, and the student plus the MLP predicts them. Stage 3: the mean squared error over the masked valid frames is back-propagated, the gradient norm is clipped at 1.0, and AdamW updates the predictor with learning rate 10 to the minus 3 and the encoder with its current rate. Stage 4: exactly once after each optimizer step, the teacher moves 0.1 percent towards the student. During the probe the student does not change, so the teacher stays equal to the original encoder.
    Then the next batch. Every 5,000 steps by default, and at every epoch end, the run saves last_training.pt for an exact resume, encoder.pt and an encoder-only checkpoint; the best encoder by the downstream check is kept separately as encoder_best.pt.`);
    L.cite(s, "Sources: adaptation/ssl.py, adaptation/pretrain.py, config/pretrain.yaml");
  }

  // ---- The two gates
  {
    const s = L.slide(SEC, "Two gates stop a run before it can damage the encoder");
    const by = 2.2, bh = 1.0, cyy = by + bh / 2, dh = 1.24, dy = cyy - dh / 2;
    const dia = (x, w, t) => {
      s.addShape(p.ShapeType.diamond, { x, y: dy, w, h: dh, fill: { color: C.amberFill }, line: { color: C.warn, width: 1.5 } });
      L.text(s, t, x + 0.25, dy + 0.22, w - 0.5, dh - 0.44, { fontSize: 16, bold: true, align: "center", valign: "middle", kind: "diagram" });
    };
    const lab = (t, x, y, w = 0.6) => L.text(s, t, x, y, w, 0.3, { fontSize: 16, bold: true, kind: "diagram" });
    L.box(s, MX, by, 2.2, bh, "original encoder:\ndownstream\nbaseline UAR₀", { role: "frozen", fontSize: 16 });
    L.arrow(s, MX + 2.2, cyy, 3.1, cyy);
    L.box(s, 3.1, by, 2.2, bh, "probe: 500 steps,\npredictor only,\nencoder frozen", { fill: C.blueFill, line: C.accent, fontSize: 16 });
    L.arrow(s, 5.3, cyy, 5.6, cyy);
    dia(5.6, 2.05, "gain ≥ 10%?");
    L.arrow(s, 7.65, cyy, 8.15, cyy);
    lab("yes", 7.68, cyy - 0.36);
    L.box(s, 8.15, by, 2.05, bh, "warm-up, then\ntraining", { role: "train", fontSize: 18, bold: true });
    L.arrow(s, 10.2, cyy, 10.6, cyy);
    dia(10.6, W - MX - 10.6, "UAR ≥\nUAR₀ − 0.02?");
    // loop back from gate 2 to training
    const dcx = 10.6 + (W - MX - 10.6) / 2, tcx = 8.15 + 2.05 / 2, ly = 1.78;
    L.path(s, [[dcx, dy], [dcx, ly], [tcx, ly], [tcx, by]], { width: 1.75 });
    lab("yes: next 500 steps", dcx - 2.9, ly - 0.34, 2.2);
    // stops
    const sy = 3.75, sh = 0.86;
    L.arrow(s, 5.6 + 2.05 / 2, dy + dh, 5.6 + 2.05 / 2, sy);
    lab("no", 5.6 + 2.05 / 2 + 0.1, dy + dh + 0.02);
    L.box(s, 4.95, sy, 3.3, sh, "stop: encoder untouched\n(probe.json)", { fill: C.redFill, line: C.bad, fontSize: 16, bold: true });
    L.arrow(s, dcx, dy + dh, dcx, sy);
    lab("no", dcx + 0.1, dy + dh + 0.02);
    L.box(s, 10.45, sy, W - MX - 10.45, sh, "stop the run", { fill: C.redFill, line: C.bad, fontSize: 16, bold: true });
    L.arrow(s, tcx, by + bh, tcx, sy);
    L.box(s, 8.4, sy, 1.85, sh, "encoder_best.pt\n(if above UAR₀)", { role: "out", fontSize: 16, bold: true });
    L.text(s, "best UAR", tcx + 0.1, by + bh + 0.1, 1.2, 0.32, { fontSize: 16, bold: true, kind: "diagram" });
    L.table(s, ["Gate", "Measured on", "Pass rule"], [
      [{ text: "1 · Learnability", bold: true }, "SSL validation clips, fixed masks", "loss ≥ 10% below constant prediction"],
      [{ text: "2 · Downstream", bold: true }, "labelled train / val clips (never test)", "UAR of a fresh linear head ≥ UAR₀ − 0.02"],
    ], { y: 5.0, colW: [2.45, 4.6, 5.08], fontSize: 18, rowH: 0.52 });
    L.notes(s, `Path E now has two gates, because the mock test showed that a run can look healthy while it damages the encoder.
    Before any training, the downstream check measures the original encoder: it embeds a fixed set of labelled training and validation clips, one clip at a time, fits a fresh class-balanced linear head and records the validation UAR, called UAR zero here.
    Gate 1, learnability. For probe_steps, 500 by default, only the predictor is trained and the encoder is frozen. Then, on the SSL validation clips with fixed masks, its loss is compared with the loss of a constant prediction, the per-channel mean of the batch. The gain is one minus the probe loss divided by the constant-prediction loss. If the gain is below probe_min_gain, 10 percent by default, the run stops with an error and the encoder has not changed a single weight; the numbers are in probe.json.
    If the gate passes, the encoder warms up and trains. Gate 2, downstream: every downstream_every_steps, 500 by default, the current student re-embeds the same labelled clips and a fresh linear head is fitted. The check logs the validation UAR together with the cosine to the original embeddings, the spread across clips and the effective rank, in downstream_history.jsonl. If the UAR falls more than downstream_max_drop, 0.02, below UAR zero, the run stops. The encoder with the highest UAR so far is saved as encoder_best.pt, but only if it beats the original.
    The labelled test split is never used by either gate. And the gates protect rather than prove: in the mock test the quick linear-probe UAR rose while the full head's test accuracy did not. The final decision is still a full head trained on each encoder, compared on validation and tested once.`);
    L.cite(s, "Sources: adaptation/pretrain.py (_probe, _downstream), adaptation/downstream.py, config/pretrain.yaml");
  }

  // ---- What we keep and what we change
  {
    const s = L.slide(SEC, "What we keep from emotion2vec, and what we change");
    const keep = { fill: C.greenFill }, chg = { fill: C.amberFill };
    L.table(s, ["Aspect", "Original emotion2vec", "Our adaptation baseline"], [
      [{ text: "Core idea", bold: true }, "masked student, EMA teacher, feature targets", { text: "kept", bold: true, ...keep }],
      [{ text: "Target", bold: true }, "top-K FFN outputs, time-normalised", { text: "last block, layer norm per frame (final)", ...chg }],
      [{ text: "Loss", bold: true }, "frame loss + pooled utterance loss", { text: "frame loss; pooled utterance loss optional", ...chg }],
      [{ text: "Student side", bold: true }, "linear, or data2vec 2.0 style (CNN decoder)", { text: "all frames kept, masked zeroed; 2-layer MLP", ...chg }],
      [{ text: "Teacher update", bold: true }, "EMA with a rising decay; CNN copied", { text: "fixed decay 0.999; CNN frozen", ...chg }],
      [{ text: "Mask setting", bold: true }, "probability a span starts at a frame", { text: "fraction of valid frames covered (0.5)", ...chg }],
      [{ text: "Start, data", bold: true }, "pre-trained base model; 262 h public speech", { text: "released emotion2vec+ large; call clips", ...chg }],
      [{ text: "Safeguards", bold: true }, "not part of the method", { text: "learnability probe + downstream check", ...chg }],
    ], { y: TOP + 0.02, colW: [2.15, 5.0, 4.98], fontSize: 18, rowH: 0.47 });
    L.box(s, MX, 6.25, CW, 0.52, [
      { text: "Goal: a simple, auditable baseline, ", options: { bold: true } },
      { text: "not a reproduction of the original training." },
    ], { fill: C.blueFill, line: C.accent, fontSize: 20, align: "left", margin: [3, 14, 3, 14] });
    L.notes(s, `How does this compare with the original emotion2vec method? The comparison is against the ACL 2024 paper, sections 3 and 4.2 and appendices A and C, and the official code we checked.
    The core idea is kept: a masked student, an EMA teacher, regression onto teacher features and stop-gradient.
    The target and the loss: the original regresses onto time-normalised top-K feed-forward outputs and adds a pooled utterance loss. Our default target is the teacher's last block, layer-normalised per frame, because the original target cannot be learned from the already fine-tuned large model; ffn_mean remains available as a control. The utterance loss now pools the extra tokens first, as in the paper, but is off by default.
    The student side: depending on the start model, the original uses a linear prediction or the data2vec 2.0 design, which encodes only the visible frames and reconstructs the masked ones with a CNN decoder. We keep all positions in the student, set masked frames to zero and use a new two-layer pointwise MLP. That is simpler to audit, but loses data2vec 2.0's efficiency of not encoding masked frames.
    The teacher: the original copies the CNN and updates the Transformer by EMA with a decay that rises during training. We use one fixed decay of 0.999 for all encoder weights; since the CNN is frozen, its teacher copy stays identical anyway.
    The mask setting: the paper describes a span-start probability, while our 0.5 is the fraction of valid frames covered, so the numbers are not comparable.
    The start point and data: the original starts from a pre-trained base model and uses 262 hours of public emotional speech; we start from the released emotion2vec+ large encoder and use our own unlabelled call clips.
    Why the changes? The public release contains no SSL teacher, decoder or optimiser state, and the checked FunASR and official code return features but no full SSL loss, so the original run cannot simply be resumed. We chose the simplest objective that is easy to audit as a first baseline. These choices are not required by our short clips, and there is no evidence that they beat the original two-part objective; the paper's own frame-only ablations do not transfer to our domain. Finally, the two gates, the learnability probe and the downstream check, are our own safeguards; they are not part of the original method.`);
    L.cites(s, R, ["e2v", "repo", "fairseq"], "Source: docs/guide.md (comparison with the paper).");
  }
};

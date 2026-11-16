// Silero VAD: the model, its own segmentation, why we do not use it, how we use the probabilities
const R = require("./refs");

module.exports = function (L, A) {
  const { C, W, H, MX, CW, TOP, FONT } = L;
  const p = L.pptx;
  const D = A.data, S = D.summary;

  // ---- 13 Silero VAD model
  {
    const s = L.slide("", "Silero VAD: a small network that scores every 32 ms", { stage: "VAD" });
    const n = 6, gap = 0.36, bw = (CW - gap * (n - 1)) / n, bh = 1.05, y = TOP + 0.75;
    const blocks = [
      ["512 samples\n(32 ms)", C.panel, C.ink],
      ["STFT\n129 bins", C.blueFill, C.accent],
      ["4 × Conv1d\nk = 3, ReLU", C.blueFill, C.accent],
      ["LSTM cell\n128 units", C.blueFill, C.accent],
      ["Conv1d\n+ sigmoid", C.blueFill, C.accent],
      ["p(speech)\n0 … 1", C.amberFill, C.warn],
    ];
    blocks.forEach(([t, f, l], i) => {
      const x = MX + i * (bw + gap);
      L.box(s, x, y, bw, bh, t, { fill: f, line: l, fontSize: 18, bold: i === 5 });
      if (i < n - 1) L.arrow(s, x + bw, y + bh / 2, x + bw + gap, y + bh / 2, { width: 1.75 });
    });
    // recurrent state loop on the LSTM block
    const lx = MX + 3 * (bw + gap);
    L.path(s, [[lx + bw * 0.7, y], [lx + bw * 0.7, y - 0.3], [lx + bw * 0.3, y - 0.3], [lx + bw * 0.3, y]], { width: 1.5 });
    L.text(s, "state carried to the next 32 ms", lx - 1.2, y - 0.68, bw + 2.4, 0.34, { fontSize: 16, align: "center" });
    const tiles = [[`${(D.silero_16k_parameters / 1e6).toFixed(2)}M`, "parameters\n(16 kHz model)"], ["2.3 MB", "model file,\ninside the wheel"], ["< 1 ms", "per 32 ms chunk,\none CPU thread"], ["MIT", "licence;\n8 and 16 kHz input"]];
    const tw = (CW - 3 * 0.29) / 4, ty = 3.85, th = 2.2;
    tiles.forEach(([n_, c], i) => {
      const x = MX + i * (tw + 0.29);
      s.addShape(p.ShapeType.rect, { x, y: ty, w: tw, h: th, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: ty, w: 0.09, h: th, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
      L.text(s, n_, x + 0.3, ty + 0.15, tw - 0.4, 0.8, { fontSize: 40, bold: true, color: C.link, valign: "middle" });
      L.text(s, c, x + 0.3, ty + 1.05, tw - 0.4, 1.0, { fontSize: 20 });
    });
    L.notes(s, `Silero VAD is a voice activity detector: it says, for every short chunk of audio, how likely it is that someone is speaking.
    The diagram shows the 16 kHz model as it is stored in the silero-vad 6.2.3 wheel; we read the structure from the TorchScript file. Each step takes 512 samples, which is 32 milliseconds at 16 kHz. A short-time Fourier transform turns them into 129 frequency bins. Four small convolution blocks, each a one-dimensional convolution with kernel size 3 followed by a ReLU, extract features. An LSTM cell with 128 units adds memory: its state is carried from one chunk to the next, so the decision for this chunk also uses what came before. A final convolution and a sigmoid give one number between 0 and 1, the speech probability.
    The model is tiny, about 0.24 million parameters for the 16 kHz version, and the TorchScript file is 2.3 megabytes, shipped inside the wheel, so nothing is downloaded at run time. The authors report under 1 millisecond per chunk on a single CPU thread. It also accepts 8 kHz audio directly. The licence is MIT.
    What we take from it is simply the sequence of probabilities, one every 32 milliseconds.`);
    L.cite(s, "Structure read from silero_vad.jit (silero-vad 6.2.3)", [[R.silero.short, R.silero.url]]);
  }

  // ---- 14 Silero's own segmentation
  {
    const s = L.slide("", "How Silero turns probabilities into speech segments", { stage: "VAD" });
    L.text(s, "get_speech_timestamps(): speech starts above a threshold, ends after enough quiet frames", MX, TOP, CW, 0.5, { fontSize: 22 });
    L.table(s, ["Parameter", "Default", "What it does"], [
      [{ text: "threshold", mono: true }, "0.5", "a frame at or above it starts (or continues) speech"],
      [{ text: "neg_threshold", mono: true }, "0.35", "below it, speech may be ending (hysteresis)"],
      [{ text: "min_silence_duration_ms", mono: true }, "100", "quiet this long closes the chunk"],
      [{ text: "min_speech_duration_ms", mono: true }, "250", "shorter chunks are dropped"],
      [{ text: "speech_pad_ms", mono: true }, "30", "padding added on both sides of each chunk"],
      [{ text: "max_speech_duration_s", mono: true }, "∞", "a longer chunk is split at its longest silence"],
    ], { y: TOP + 0.75, colW: [4.3, 1.3, 6.53], fontSize: 20, rowH: 0.62, align: { 1: "center" } });
    L.notes(s, `Silero also comes with its own segmentation function, get_speech_timestamps. It is worth understanding, because it is the obvious baseline and the pilot will compare against it.
    It walks through the probabilities frame by frame with two thresholds, which is called hysteresis. When a frame reaches 0.5, speech starts. While in speech, a frame below the lower threshold of 0.35 marks a possible end; using a lower exit threshold stops the state from flickering when the probability hovers around 0.5. The possible end only becomes a real end if the quiet stretch lasts at least min_silence_duration_ms, 100 milliseconds by default. Chunks shorter than 250 milliseconds are thrown away, and every chunk is padded by 30 milliseconds on both sides.
    There is also max_speech_duration_s, infinite by default. If it is set and a chunk grows longer than that, the chunk is split at the longest silence found inside it so far.
    These defaults come from the silero-vad 6.2.3 source code. The next slide shows what they do on a real turn.`);
    L.cite(s, "Defaults from silero_vad/utils_vad.py (6.2.3)", [[R.silero.short, R.silero.url]]);
  }

  // ---- 15 Silero's own segments over-split
  {
    const s = L.slide("", "Silero’s own segments split long turns into fragments", { stage: "VAD" });
    const M = D.methods, vt = D.vad_turn;
    L.table(s, ["5 long English turns (synthetic DailyTalk)", "Pieces", "Median", "Under 3 s"], [
      ["Silero, default 100 ms silence rule", String(M.silero_100.pieces), M.silero_100.median_s.toFixed(1) + " s", String(M.silero_100.under_3s)],
      ["Silero, 300 ms silence rule", String(M.silero_300.pieces), M.silero_300.median_s.toFixed(1) + " s", String(M.silero_300.under_3s)],
    ], { y: TOP - 0.05, colW: [6.3, 1.9, 1.9, 2.03], fontSize: 18, rowH: 0.42, align: { 1: "center", 2: "center", 3: "center" } });
    L.text(s, [{ text: "Our fix, step 6: ", options: { bold: true, color: C.link } }, { text: "choose all cut points of a turn at once, with the words" }], MX, 2.95, CW, 0.42, { fontSize: 20, valign: "middle" });
    L.image(s, "fig/vad_silero.png", MX, 3.4, CW, 3.45, { alt: "Silero's own segments on one synthetic turn" });
    L.notes(s, `Here is the problem with using Silero's segments directly. The figure shows one synthetic ${vt.duration_s.toFixed(1)}-second turn made of DailyTalk utterances. The top panel is the speech probability, with the start threshold of 0.5 and the end threshold of 0.35. Below are the speech chunks that Silero's own function produces: ${vt.silero_100} pieces with the default 100 millisecond silence rule, and still ${vt.silero_300} with a 300 millisecond rule.
    Over the five long English turns of the test set, Silero's default settings give ${M.silero_100.pieces} pieces with a median of ${M.silero_100.median_s.toFixed(1)} seconds, and ${M.silero_100.under_3s} of them are shorter than 3 seconds, the length below which predictions become unreliable. A 300 millisecond rule still gives ${M.silero_300.pieces} pieces with a median of ${M.silero_300.median_s.toFixed(1)} seconds, ${M.silero_300.under_3s} of them under 3 seconds. And these DailyTalk speakers are not even slow; a slow speaker who pauses every few words would come out as dozens of one- to two-second pieces, which is what the method document estimates for its 40-second example: about 25 pieces.
    Silero also knows nothing about words or sentences, so it cannot keep the transcript in step, and its max_speech_duration_s option only splits chunks that are too long, front to back, which can leave a very short tail.
    So we keep Silero for what it does well, judging whether a moment is speech, and leave the choice of cut points to step 6, which looks at the whole turn at once. A direct comparison of all methods comes in the validation section, after the method has been explained.`);
    L.cite(s, "Silero 6.2.3 on the pipeline's own probabilities", [[R.dailytalk.short, R.dailytalk.url]]);
  }

  // ---- 16 How we use VAD
  {
    const s = L.slide("", "We use VAD only to judge each word gap, never to delete audio", { stage: "VAD", titleSize: 28 });
    const z = D.zoom;
    const fig = L.image(s, "fig/gap_zoom.png", MX, TOP - 0.05, 6.6, 4.75, { alt: "A chosen word gap with VAD probabilities and the cut in the middle of its quiet part" });
    const bx = fig.x + fig.w + 0.3, bw = W - MX - bx;
    L.bullets(s, [
      { runs: [{ text: "v = 1 − mean p(speech)", options: { bold: true } }, { text: " in the gap" }] },
      "Quiet → v ≈ 1; sigh or “mm-hmm” → low",
      { runs: [{ text: "Cut in the middle of the quiet part", options: { bold: true } }] },
      "Nothing is deleted",
    ], { x: bx, y: TOP + 0.1, w: bw, h: 3.6, fontSize: 22, gap: 14 });
    L.box(s, bx, 5.55, bw, 0.95, `This gap: ${z.gap_s.toFixed(2)} s, v = ${z.v.toFixed(2)}, s = ${z.s.toFixed(2)}`, { fill: C.greenFill, line: C.good, fontSize: 20 });
    L.notes(s, `So how do we use VAD? Only its frame probabilities, and only to answer one question for each word gap that the aligner found: is this gap really quiet?
    The figure zooms into one gap of a synthetic DailyTalk turn, between "cameras." and "Getting". The top shows the waveform with the aligned words; the green band is the word gap, ${z.gap_s.toFixed(2)} seconds long. The bottom shows the speech probability for each 32 millisecond frame. Inside the gap, the green bars, the probability drops close to zero.
    The quietness of a gap is v equals one minus the mean speech probability inside it. Here v is ${z.v.toFixed(2)}. A gap whose probability stays high is not a clean pause: it may hold a breath, a sigh, a laugh, crying, background voices or the agent saying "mm-hmm". Cutting there would split a sound that may carry the emotion, so such a gap gets a low v and therefore a lower boundary score.
    If this gap is chosen, and in this turn it was, the cut is placed in the middle of the gap's quiet part. The quiet part, the dark green bars, is every 32 millisecond frame whose speech probability is within 0.1 of the lowest probability in the gap; among those frames the cut takes the one nearest the centre of the gap, marked by the dashed line and the orange triangle. An earlier version took the single lowest frame, which in near-digital silence could sit right at the start or end of a pause; on the test run the cut position within the gap moved from 0.43 to 0.92 of the gap length to 0.48 to 0.60. Cutting in the middle splits the pause evenly, so neither neighbouring word loses its first or last sound and the next word's onset is not clipped.
    VAD never deletes anything. It can mark quiet speech, crying, laughter or sighs as non-speech, so pauses and non-verbal sounds inside a segment stay in the audio; hesitation and sighing are part of the emotion.`);
    L.cite(s, "Synthetic turn d418_long_en (DailyTalk utterances)", [[R.silero.short, R.silero.url], [R.dailytalk.short, R.dailytalk.url]]);
  }
};

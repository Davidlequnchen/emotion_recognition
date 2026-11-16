// Title, the problem, inputs and rules, the three building blocks, the overall pipeline
const R = require("./refs");

module.exports = function (L) {
  const { C, W, H, MX, CW, TOP, FONT } = L;
  const p = L.pptx;
  const SEC = "Turn segmentation";

  // ---- 1 Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 2.0, w: 0.09, h: 2.45, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Turn Segmentation for Emotion Labelling", MX + 0.4, 1.9, 11.8, 0.85, { fontSize: 40, bold: true });
    L.text(s, "Cutting long customer turns at the best pauses with forced alignment,\nSilero VAD and dynamic programming", MX + 0.4, 2.85, 11.8, 1.0, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  preparation_tools  ·  November 2026", MX + 0.4, 4.05, 11.8, 0.42, { fontSize: 20 });
    L.text(s, "Examples: the illustrative turn of the method document and synthetic turns built from DailyTalk (CC BY-SA 4.0). No call data.", MX, H - 0.95, CW, 0.4, { fontSize: 16, kind: "diagram" });
    L.notes(s, `This talk explains the turn segmenter in preparation_tools. It takes one customer turn at a time, which is an audio clip plus the transcript we already have, and cuts long turns into shorter pieces that listeners can label with one emotion.
    The method has three parts. Forced alignment tells us when each word was spoken. Silero VAD tells us how quiet each pause between words really is. Dynamic programming then chooses all cut points of the turn at once, so that the pieces are close to 7 seconds and the cuts sit at the best pauses.
    All examples in the talk use either the illustrative 40-second turn from the method document or synthetic turns built from the public DailyTalk dataset. No company audio or transcripts are shown.`);
  }

  // ---- 2 Model input limits
  {
    const s = L.slide("Problem", "A 40–50 s turn does not fit the models’ input");
    const cut = { fill: C.redFill };
    L.table(s, ["Model", "Input length", "A 40–50 s turn"], [
      [{ text: "emotion2vec+ large", bold: true }, "real length, no window", { text: "fails on the Mac GPU above 20.48 s", ...cut }],
      [{ text: "MERaLiON-SER-v1, Melati", bold: true }, "fixed 30 s window", { text: "everything after 30 s is cut", ...cut }],
      [{ text: "Vox-Profile Whisper", bold: true }, "fixed 15 s window", { text: "only the first 15 s", ...cut }],
      [{ text: "Vox-Profile WavLM", bold: true }, "real length, cut at 15 s", { text: "only the first 15 s", ...cut }],
      [{ text: "Qwen2-Audio, Gemma 4", bold: true }, "at most 30 s", { text: "everything after 30 s is cut", ...cut }],
      [{ text: "Qwen2.5-Omni 7B", bold: true }, "variable, capped at 300 s", "fits, but one answer for the whole turn"],
    ], { y: TOP + 0.05, colW: [3.75, 3.6, 4.78], fontSize: 20, rowH: 0.6 });
    L.text(s, [{ text: "Too short is also bad: ", options: { bold: true } }, { text: "Vox-Profile was trained on 3–15 s clips and calls predictions under 3 s unreliable." }], MX, 6.2, CW, 0.55, { fontSize: 20, valign: "middle" });
    L.notes(s, `We already receive one clip per customer turn, so why cut further? The first reason is simply that long turns do not fit the models. Most turns are short, but some run 40 to 50 seconds or longer.
    This table lists how each model we benchmarked handles input length, from our model surveys, the model documentation in the repository and our own loaders.
    emotion2vec+ large has no fixed window and reads the real length, but on the Mac GPU its first convolution fails for clips longer than 20.48 seconds, which we measured: 20.4 seconds runs, 20.5 fails, and our scripts then fall back to the CPU. Its memory also grows roughly with the square of the clip length: about 2.3 gigabytes extra for a 60-second clip on the CPU.
    MERaLiON-SER-v1 and Melati are built on a Whisper encoder with a fixed 30-second window: shorter clips are padded with silence, and everything after 30 seconds is cut, so the last 10 to 20 seconds of a long turn are never heard. Vox-Profile Whisper has a fixed 15-second window, so it only hears the first 15 seconds. Vox-Profile WavLM reads the real length but its card says at most 15 seconds, and our loader cuts there.
    The audio LLMs Qwen2-Audio and Gemma 4 take at most 30 seconds and cut the rest. Qwen2.5-Omni accepts variable length up to a processor cap of 300 seconds, so a long turn fits, but it still gives only one answer for the whole turn.
    The last line is the other side of the coin: pieces must not be too short either. Vox-Profile was trained on 3 to 15 second clips and its card calls predictions for clips under 3 seconds unreliable. Input limits of the API models, GPT Realtime and Gemini, were not checked for this slide.`);
    L.cite(s, "Sources: models_survey_en.md, emotion2vec_plus_large_en.md, research_paper/docs/experimental_design.md §4.2, scripts/models/*.py");
  }

  // ---- 3 Why segment
  {
    const s = L.slide("Problem", "Why cut a long turn into sentence-length pieces");
    const x0 = MX + 2.0, x1 = W - MX, X = (t) => x0 + (x1 - x0) * t / 40, y1 = TOP + 0.1, y2 = TOP + 0.85, bh = 0.55;
    L.text(s, "whole turn", MX, y1, 1.9, bh, { fontSize: 18, bold: true, valign: "middle" });
    L.box(s, X(0), y1, X(40) - X(0), bh, "one label for 40 s:  neutral?  angry?", { fill: C.panel, line: C.ink, fontSize: 18, r: 0.04 });
    L.text(s, "segments", MX, y2, 1.9, bh, { fontSize: 18, bold: true, valign: "middle" });
    const segs = [[0, 7.4, "neutral", "E8E9EC"], [7.4, 15.3, "neutral", "E8E9EC"], [15.3, 23.0, "frustrated", "FDEBC8"], [23.0, 29.6, "angry", "F9D4D5"], [29.6, 40, "angry", "F9D4D5"]];
    segs.forEach(([a, b, lab, f]) => L.box(s, X(a) + 0.03, y2, X(b) - X(a) - 0.06, bh, lab, { fill: f, line: C.ink, fontSize: 18, r: 0.04 }));
    L.text(s, "Illustration, not real data", x1 - 3.5, y2 + bh + 0.04, 3.5, 0.32, { fontSize: 16, align: "right", italic: true });
    L.table(s, ["Reason", "Why it matters"], [
      [{ text: "Model input limits", bold: true }, "15–30 s windows cut long input (previous slide)"],
      [{ text: "One label per clip", bold: true }, "a 40 s turn can mix emotions; 31,249 of 56,264 benchmark clips are under 3 s"],
      [{ text: "Emotion over time", bold: true }, "see how emotion changes within a turn and across the call"],
      [{ text: "Reliable listeners", bold: true }, "a 7 s piece is one or two sentences: one label fits"],
      [{ text: "Same cut everywhere", bold: true }, "training and production clips are cut by the same segmenter"],
    ], { y: 3.5, colW: [3.2, 8.93], fontSize: 20, rowH: 0.54 });
    L.notes(s, `The figure is an illustration, not real data. The top row is what we have today: one label for a whole 40-second turn. If the customer starts calm, gets frustrated and ends angry, any single label is wrong for most of the turn. The bottom row is the same turn cut into five pieces of about 7 seconds, each of which can get its own label, so the change becomes visible as different labels on neighbouring pieces.
    The table lists the reasons. First, the model input limits from the previous slide: 15 and 30-second windows simply cut long input. Second, one label per clip: a long turn can mix emotions, and the models were trained and benchmarked on sentence-length clips; in the 13 acted and TV datasets of our benchmark, 31,249 of 56,264 clips are shorter than 3 seconds, and Vox-Profile was trained on 3 to 15 second clips. Third, emotion over time: we want to see how emotion develops within a long turn and across the call, not just one average. Fourth, listeners: a 7-second piece is one or two sentences, which a listener can give one label to consistently. Fifth, the same segmenter must be used for the labelled training clips and in production, otherwise we create a new mismatch between training and use.
    We do not try to cut at emotion changes, because before labelling we do not know where they are, and using a model to guess them would build that model's bias into the labels. Natural 7-second pieces are enough to show a change.`);
    L.cite(s, "Method document §1, §4.2, §4.9; benchmark clip counts from §4.2");
  }

  // ---- 4 Full procedure: upstream steps, then this tool
  {
    const s = L.slide("Procedure", "The full procedure: upstream steps, then this tool");
    const W5 = 2.2, g5 = (CW - 5 * W5) / 4, cx5 = (i) => MX + i * (W5 + g5);
    const yA = 1.92, hA = 0.8;
    s.addShape(p.ShapeType.rect, { x: MX - 0.15, y: 1.5, w: CW + 0.3, h: 1.37, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    s.addShape(p.ShapeType.rect, { x: MX - 0.15, y: 3.05, w: CW + 0.3, h: 3.85, fill: { color: C.blueFill }, line: { color: C.blueFill, width: 0 } });
    L.text(s, "A · Already done upstream (not part of this tool)", MX, 1.53, 8, 0.36, { fontSize: 18, bold: true });
    L.text(s, "B · This tool: preparation_tools", MX + 1.35, 3.08, 8, 0.36, { fontSize: 18, bold: true, color: C.link });
    ["1 · Whole call\nmono MP3", "2 · VibeVoice-ASR\nwho · when · what", "3 · Keep customer,\ndrop agent (CCP)", "4 · Cut turn audio\n+ its transcript", "5 · Drop turns\nunder 2 s"].forEach((t, i) => {
      L.box(s, cx5(i), yA, W5, hA, t, { fill: C.white, line: C.ink, fontSize: 18 });
      if (i < 4) L.arrow(s, cx5(i) + W5, yA + hA / 2, cx5(i + 1), yA + hA / 2, { width: 1.75 });
    });
    const laneX = W - 0.33;
    // row B1: language, 10 s rule, alignment
    const y1 = 3.45, h1 = 0.9, w1 = [2.0, 1.7, 1.7, 3.0, 1.8], g1 = (CW - w1.reduce((a_, b_) => a_ + b_)) / 4;
    const x1 = w1.map((_, i) => MX + w1.slice(0, i).reduce((a_, b_) => a_ + b_, 0) + i * g1);
    L.path(s, [[cx5(4) + W5, yA + hA / 2], [laneX, yA + hA / 2], [laneX, 2.97], [x1[0] + w1[0] / 2, 2.97], [x1[0] + w1[0] / 2, y1]], { width: 1.75 });
    const dia = (x, y, w, h, t) => {
      s.addShape(p.ShapeType.diamond, { x, y, w, h, fill: { color: C.amberFill }, line: { color: C.warn, width: 1.5 } });
      L.text(s, t, x + 0.22, y + 0.2, w - 0.44, h - 0.4, { fontSize: 16, bold: true, align: "center", valign: "middle", kind: "diagram" });
    };
    const folder = (x, y, w, t, keep) => L.box(s, x, y, w, 0.44, t, { fill: keep ? C.amberFill : C.greenFill, line: keep ? C.warn : C.good, fontSize: 16, bold: true, r: 0.3, mono: false });
    const lab = (t, x, y, h = 0.3) => L.text(s, t, x, y, 0.5, h, { fontSize: 16, kind: "diagram" });
    L.box(s, x1[0], y1, w1[0], h1, "6 · Language check\nen · es · mixed", { fill: C.white, line: C.accent, fontSize: 16 });
    dia(x1[1], y1, w1[1], h1, "English?");
    dia(x1[2], y1, w1[2], h1, "≤ 10 s?");
    L.box(s, x1[3], y1, w1[3], h1, "7 · Forced alignment\n(match each transcript word\nto its time in the audio)", { fill: C.white, line: C.accent, fontSize: 16 });
    dia(x1[4], y1, w1[4], h1, "aligned\nwell?");
    for (let i = 0; i < 4; i++) L.arrow(s, x1[i] + w1[i], y1 + h1 / 2, x1[i + 1], y1 + h1 / 2, { width: 1.75 });
    lab("yes", x1[1] + w1[1] + 0.02, y1 + h1 / 2 - 0.36);
    lab("no", x1[2] + w1[2] + 0.06, y1 + h1 / 2 - 0.36);
    const py = y1 + h1 + 0.23;
    [[1, "non_english/", true, "no"], [2, "final/", false, "yes"], [4, "alignment_issues/", true, "no"]].forEach(([i, t, keep, l]) => {
      const cxm = x1[i] + w1[i] / 2;
      L.arrow(s, cxm, y1 + h1, cxm, py, { width: 1.75 });
      lab(l, cxm + 0.1, y1 + h1 - 0.03, 0.24);
      folder(cxm - 1.0, py, 2.0, t, keep);
    });
    // carriage to row B2
    const y2 = 5.25, h2 = 0.95, lane2 = 5.13;
    const b8 = [MX, 3.45], b9 = [MX + 3.45 + 0.3, 4.1], dx = MX + 3.45 + 0.3 + 4.1 + 0.3, dw = 1.75;
    L.path(s, [[x1[4] + w1[4], y1 + h1 / 2], [laneX, y1 + h1 / 2], [laneX, lane2], [b8[0] + b8[1] / 2, lane2], [b8[0] + b8[1] / 2, y2]], { width: 1.75 });
    lab("yes", x1[4] + w1[4] + 0.02, y1 + h1 / 2 - 0.36);
    L.box(s, b8[0], y2, b8[1], h2, "8 · Score word gaps\n(rate each pause as a cut point:\npause length · punctuation · silence)", { fill: C.white, line: C.accent, fontSize: 16 });
    L.box(s, b9[0], y2, b9[1], h2, "9 · Choose cut points for the whole turn\n(dynamic programming: pieces near\n7 s, cuts at the best pauses)", { fill: C.white, line: C.accent, fontSize: 16 });
    dia(dx, y2, dw, h2, "fits\n2–15 s?");
    L.arrow(s, b8[0] + b8[1], y2 + h2 / 2, b9[0], y2 + h2 / 2, { width: 1.75 });
    L.arrow(s, b9[0] + b9[1], y2 + h2 / 2, dx, y2 + h2 / 2, { width: 1.75 });
    const fx = dx + dw + 0.3;
    L.arrow(s, dx + dw, y2 + h2 / 2, fx, y2 + h2 / 2, { width: 1.75 });
    lab("yes", dx + dw + 0.02, y2 + h2 / 2 - 0.36);
    folder(fx, y2 + h2 / 2 - 0.22, W - MX - fx, "final/", false);
    const py2 = y2 + h2 + 0.22;
    L.arrow(s, dx + dw / 2, y2 + h2, dx + dw / 2, py2, { width: 1.75 });
    lab("no", dx + dw / 2 + 0.1, y2 + h2 - 0.03, 0.24);
    folder(dx + dw / 2 - 1.0, py2, 2.0, "boundary_issues/", true);
    // legend
    L.key(s, MX, py2 + 0.04, C.greenFill, "final/ → Label Studio", { edge: C.good, w: 2.6 });
    L.key(s, MX + 3.3, py2 + 0.04, C.amberFill, "other folders → kept (future pre-training)", { edge: C.warn, w: 4.2 });
    L.notes(s, `Before going into the tool, here is the whole procedure, from a recorded call to labelling-ready clips. The grey band is what already happens upstream; the blue band is this tool.
    Step 1: the raw input is one entire call, a mono MP3 with the customer and the agent on the same channel. Step 2: VibeVoice-ASR, an open speech recognition model that also does speaker diarization and timestamps, transcribes the call. For every speaker turn it gives who spoke, when, meaning the start and end time in the call, and what was said. Step 3: only the customer turns are kept; all agent turns, called CCP turns, are discarded. Step 4: each customer turn's audio is cut from the call using those timestamps, together with its own transcript. Step 5: turns shorter than 2 seconds are dropped. That is the input of this tool: one audio clip and one transcript per customer turn, without word times.
    Step 6, the first thing the tool does, is a language check on the transcript. Every turn that is not English, meaning Spanish or mixed, of any length, goes whole into its own folder, non_english, without any cutting. English turns of 10 seconds or less are already sentence-length, so they go whole into final.
    Longer English turns go through the chain. Step 7, forced alignment, matches each word of the existing transcript to its time in the audio. If that does not work well, for example because the transcript and the audio disagree, the whole turn goes into alignment_issues. Step 8 rates every pause between two words as a possible cut point, from the pause length, the punctuation and how silent the pause really is. Step 9 chooses all cut points of the turn at once with dynamic programming, so that the pieces are near 7 seconds and the cuts sit at the best pauses. If no way of cutting keeps every piece between 2 and 15 seconds, the whole turn goes into boundary_issues; otherwise the pieces go into final.
    So every turn ends in one of four folders. final holds the clean results, which go to Label Studio for labelling and later to benchmarking and fine-tuning. The other three folders are not labelled but kept: unlabelled audio can still be used later for self-supervised continued pre-training, which is not implemented. The turn start times from step 2 go into turns.csv, so every segment keeps its exact time in the call, which is what the emotion timeline needs.`);
    L.cite(s, "", [[R.vibevoice.short, R.vibevoice.url]]);
  }

  // ---- 5 Design rules
  {
    const s = L.slide("Procedure", "Rules the tool keeps");
    L.table(s, ["Rule", "Why"], [
      [{ text: "No new diarization or speech recognition", bold: true }, "the upstream transcript stays the text"],
      [{ text: "Offline, on a CPU", bold: true }, "runs on the offline server, no network at run time"],
      [{ text: "Licences that allow commercial use", bold: true }, "internal business use is commercial use"],
      [{ text: "Deterministic", bold: true }, "same input → same cuts; every label stays traceable"],
      [{ text: "Never delete audio inside a turn", bold: true }, "pauses, sighs and laughter can carry the emotion"],
      [{ text: "Cut audio and text at the same place", bold: true }, "each clip comes with exactly its own words"],
    ], { y: TOP + 0.05, colW: [5.4, 6.73], fontSize: 20, rowH: 0.72 });
    L.notes(s, `Six rules shaped every design decision in the tool.
    We do not repeat diarization or speech recognition: the transcript from upstream is the text of record, and we only add timing to it.
    The tool runs fully offline on a CPU, because it has to work on the offline server; nothing is downloaded at run time.
    Every model and library must have a licence that allows commercial use, because internal business use counts as commercial use. This rules out some convenient models, as we will see.
    It must be deterministic: the same turn always gives the same cuts, and the settings are hashed into every output row, so any label can be traced back.
    It never deletes audio inside a turn. Pauses, hesitations, sighs and laughter can carry the emotion, so they stay in the clips.
    And the audio and the text are always cut at the same time points, so each clip comes with exactly its own words. Because the recording is mono, agent backchannels inside a customer turn stay in the audio; they cannot be separated, but they lower the scores around them, as we will see.`);
  }

  // ---- 4 Three building blocks
  {
    const s = L.slide("Method", "Three building blocks");
    const cards = [
      ["1", "Forced alignment", "When was each word said?", "torchaudio + English wav2vec 2.0", "word times", C.accent],
      ["2", "Silero VAD", "Is this pause really quiet?", "p(speech) every 32 ms", "quietness of each gap", C.accent],
      ["3", "Dynamic programming", "Which gaps do we cut?", "one cost per turn, minimised exactly", "the cut points", C.accent2],
    ];
    const cw = 3.85, gap = 0.29, y = TOP + 0.05, h = 3.95;
    cards.forEach(([n, t, q, tool, out, col], i) => {
      const x = MX + i * (cw + gap);
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      s.addText(n, { x: x + 0.25, y: y + 0.3, w: 0.6, h: 0.6, shape: p.ShapeType.ellipse, fill: { color: col }, line: { color: col, width: 0 }, fontFace: FONT, fontSize: 22, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, x + 1.0, y + 0.3, cw - 1.15, 0.6, { fontSize: 24, bold: true, valign: "middle" });
      L.text(s, q, x + 0.25, y + 1.1, cw - 0.5, 0.8, { fontSize: 22, italic: true });
      L.text(s, tool, x + 0.25, y + 1.95, cw - 0.5, 0.95, { fontSize: 20 });
      L.box(s, x + 0.25, y + 3.0, cw - 0.5, 0.7, "→ " + out, { fill: C.white, line: col, fontSize: 18, bold: true });
      if (i < 2) L.arrow(s, x + cw + 0.02, y + h / 2, x + cw + gap - 0.02, y + h / 2, { width: 2 });
    });
    L.text(s, [
      { text: "Not “is this pause long enough?” but ", options: {} },
      { text: "“how many pieces, and where?”", options: { bold: true } },
    ], MX, 5.95, CW, 0.6, { fontSize: 22, align: "center", valign: "middle" });
    L.notes(s, `The segmenter combines three building blocks, each answering one question.
    First, forced alignment answers "when was each word said?". It takes the audio and the transcript and returns a start time, an end time and a confidence for every word. The gaps between words are the only places where we may cut, so audio and text can always be cut together.
    Second, Silero VAD answers "is this pause really quiet?". It gives a speech probability every 32 milliseconds. We use it to score each word gap: a gap that holds a sigh, a laugh or the agent's backchannel is not a clean pause.
    Third, dynamic programming answers "which gaps should we cut?". It writes down one cost for the whole turn, made of how far each piece is from 7 seconds plus a price for every cut, and finds the cheapest set of cuts exactly.
    The key shift is in the question. A usual pause-based splitter asks whether a pause is long enough, which needs a fixed threshold that is wrong for slow or fast speakers. We ask how many pieces the turn should have and at which gaps, for the whole turn at once.`);
  }

  // ---- 7 Overall pipeline (detailed)
  {
    const s = L.slide("Method", "The pipeline for one turn");
    const laneX = W - 0.24;
    const dia = (x, y, w, h, t) => {
      s.addShape(p.ShapeType.diamond, { x, y, w, h, fill: { color: C.amberFill }, line: { color: C.warn, width: 1.5 } });
      L.text(s, t, x + 0.2, y + 0.2, w - 0.4, h - 0.4, { fontSize: 16, bold: true, align: "center", valign: "middle", kind: "diagram" });
    };
    const proc = (x, y, w, h, t, o = {}) => L.box(s, x, y, w, h, t, { fill: o.fill || C.blueFill, line: o.line || C.accent, fontSize: 16, dash: o.dash, bold: o.bold });
    const folder = (cxm, y, w, t, keep) => L.box(s, cxm - w / 2, y, w, 0.44, t, { fill: keep ? C.amberFill : C.greenFill, line: keep ? C.warn : C.good, fontSize: 16, bold: true, r: 0.3 });
    const lab = (t, x, y) => L.text(s, t, x, y, 0.5, 0.3, { fontSize: 16, kind: "diagram" });
    const row = (ws, y, h) => {
      const g = (CW - ws.reduce((a_, b_) => a_ + b_)) / (ws.length - 1);
      return ws.map((w, i) => ({ x: MX + ws.slice(0, i).reduce((a_, b_) => a_ + b_, 0) + i * g, y, w, h, cx: MX + ws.slice(0, i).reduce((a_, b_) => a_ + b_, 0) + i * g + w / 2, cy: y + h / 2 }));
    };
    const link = (a_, b_, l) => {
      L.arrow(s, a_.x + a_.w, a_.cy, b_.x, b_.cy, { width: 1.5 });
      if (l) lab(l, a_.x + a_.w + 0.03, a_.cy - 0.34);
    };
    const down = (n, t, keep, l, w, shift = 0) => {
      L.arrow(s, n.cx, n.y + n.h, n.cx, n.y + n.h + 0.24, { width: 1.5 });
      L.text(s, l, n.cx + 0.1, n.y + n.h - 0.03, 0.5, 0.24, { fontSize: 16, kind: "diagram" });
      folder(n.cx + shift, n.y + n.h + 0.24, w, t, keep);
    };
    // row 1: preparation, language, 10 s rule
    const r1 = row([2.05, 2.05, 2.05, 2.05, 2.05], 1.62, 0.85);
    proc(r1[0].x, r1[0].y, r1[0].w, r1[0].h, "Decode audio\nWAV · MP3 · FLAC");
    proc(r1[1].x, r1[1].y, r1[1].w, r1[1].h, "Mono 16 kHz copy\nfor analysis");
    proc(r1[2].x, r1[2].y, r1[2].w, r1[2].h, "Silero VAD\np(speech) per 32 ms");
    dia(r1[3].x, r1[3].y, r1[3].w, r1[3].h, "English\ntext?");
    dia(r1[4].x, r1[4].y, r1[4].w, r1[4].h, "≤ 10 s?");
    link(r1[0], r1[1]); link(r1[1], r1[2]); link(r1[2], r1[3]); link(r1[3], r1[4], "yes");
    down(r1[3], "non_english/ (whole)", true, "no", 2.3);
    down(r1[4], "final/ (whole)", false, "yes", 2.0);
    // row 2: alignment, scores, DP
    const r2 = row([1.75, 1.75, 2.95, 2.65, 1.7], 3.4, 1.25);
    const lane1 = r2[0].y - 0.13;
    L.path(s, [[r1[4].x + r1[4].w, r1[4].cy], [laneX, r1[4].cy], [laneX, lane1], [r2[0].cx, lane1], [r2[0].cx, r2[0].y]], { width: 1.5 });
    lab("no", r1[4].x + r1[4].w + 0.03, r1[4].cy - 0.34);
    proc(r2[0].x, r2[0].y, r2[0].w, r2[0].h, "Forced alignment\n(wav2vec 2.0):\nword times");
    dia(r2[1].x, r2[1].y, r2[1].w, r2[1].h, "aligned\nwell?");
    proc(r2[2].x, r2[2].y, r2[2].w, r2[2].h, "Score every word gap:\np pause vs usual pause,\nq punctuation, v VAD silence\n→ s from 0 (bad) to 1 (ideal)");
    proc(r2[3].x, r2[3].y, r2[3].w, r2[3].h, "Whole-turn DP:\ncheapest set of cuts,\npieces near 7 s,\ncuts at high-s gaps", { bold: true });
    dia(r2[4].x, r2[4].y, r2[4].w, r2[4].h, "fits\n2–15 s?");
    link(r2[0], r2[1]); link(r2[1], r2[2], "yes"); link(r2[2], r2[3]); link(r2[3], r2[4]);
    down(r2[1], "alignment_issues/ (whole)", true, "no", 2.75);
    down(r2[4], "boundary_issues/ (whole)", true, "no", 2.9, -0.55);
    // row 3: cut, check, export
    const r3 = row([2.66, 2.66, 2.66, 2.66], 5.58, 0.85);
    const lane2 = r3[0].y - 0.13;
    L.path(s, [[r2[4].x + r2[4].w, r2[4].cy], [laneX, r2[4].cy], [laneX, lane2], [r3[0].cx, lane2], [r3[0].cx, r3[0].y]], { width: 1.5 });
    lab("yes", r2[4].x + r2[4].w + 0.02, r2[4].cy - 0.34);
    proc(r3[0].x, r3[0].y, r3[0].w, r3[0].h, "Cut in the middle of\nthe quiet part of each gap");
    proc(r3[1].x, r3[1].y, r3[1].w, r3[1].h, "Check: every word kept\nonce, no cut inside a word");
    proc(r3[2].x, r3[2].y, r3[2].w, r3[2].h, "Export to final/\nclips · text · manifest", { fill: C.greenFill, line: C.good, bold: true });
    proc(r3[3].x, r3[3].y, r3[3].w, r3[3].h, "Label Studio labelling\n(separate step)", { fill: C.white, line: C.ink, dash: "dash" });
    link(r3[0], r3[1]); link(r3[1], r3[2]); link(r3[2], r3[3]);
    L.box(s, r3[1].x + 0.2, r3[1].y + r3[1].h + 0.14, 2 * r3[1].w, 0.42, "any error: the turn is skipped and listed in failures.csv", { fill: C.white, line: C.bad, dash: "dash", fontSize: 16, r: 0.3 });
    L.notes(s, `This is the whole pipeline for one turn; the next slides go through it step by step, and a strip at the top of each slide shows where we are. Green pills are the clean output folder, amber pills the three folders that are kept but not labelled.
    Row one. The audio is decoded once, from WAV, MP3 or FLAC; the source stays untouched for the export. A mono 16 kHz copy is made for analysis only, and Silero VAD gives a speech probability every 32 milliseconds. Then the language check on the transcript comes first: a Spanish or mixed turn of any length goes whole into non_english, because we have no Spanish aligner yet. An English turn of 10 seconds or less goes whole into final.
    Row two handles long English turns. Forced alignment with wav2vec 2.0 gives a start and end time for every word. If the alignment looks poor, or the transcript is empty, the whole turn goes into alignment_issues; we never invent timings. Otherwise every gap between two words gets a score s from 0 to 1, from three plain cues: p, how long the pause is compared with this speaker's usual pause; q, the punctuation before it; and v, how silent the gap really is according to VAD. The dynamic programme then looks at the whole turn at once and picks the cheapest set of cuts: pieces as close to 7 seconds as possible, cuts at gaps with high scores. Every piece must be between 2 and 15 seconds. If no way of cutting achieves that, the whole turn goes into boundary_issues.
    Row three. Each cut is placed in the middle of the quiet part of its gap: among the 32 millisecond frames whose speech probability is within 0.1 of the lowest one in the gap, the one nearest the gap centre. That splits the pause evenly and keeps the cut away from the next word's onset, so no word loses a sound. A check confirms that every word is kept exactly once and that no cut falls inside a word. The clips, their transcripts and the manifest rows go into final. If anything throws an error for a turn, its files are removed and it is listed in failures.csv, so a large batch never stops. Labelling in Label Studio is a separate step, done by the label_studio subproject.`);
  }
};

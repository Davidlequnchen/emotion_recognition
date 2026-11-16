// Outputs, validation on public data, settings, offline deployment and licences, limitations, references
const R = require("./refs");

module.exports = function (L, A) {
  const { C, W, H, MX, CW, TOP, FONT, MONO } = L;
  const p = L.pptx;
  const D = A.data, M = D.methods, SEN = A.sens.summary, AU = A.audio;

  // ---- Outputs
  {
    const s = L.slide("", "Step 7 · Four folders, grouped by call, plus a timeline", { stage: "Export" });
    const tree = [
      ["batch/", true], ["  final/", true], ["    <call_id>/<call_id>_t000_s01.wav", false], ["    <call_id>/<call_id>_t000_s01.txt", false],
      ["    manifest.csv", false], ["  non_english/       same layout", false], ["  alignment_issues/  same layout", false],
      ["  boundary_issues/   same layout", false], ["  review/<turn_id>.json", false], ["  call_timeline.csv", false], ["  failures.csv       if any", false],
    ];
    L.box(s, MX, TOP, 5.75, 3.95, tree.map(([t, b_], i) => ({ text: t, options: { bold: b_, breakLine: i < tree.length - 1 } })),
      { fill: C.panel, line: C.line, fontSize: 18, mono: true, align: "left", valign: "middle", margin: [6, 12, 6, 12], ls: 1.0 });
    const tx = MX + 6.0, tw = CW - 6.0;
    L.table(s, ["Item", "Used for"], [
      [{ text: "final/", bold: true, fill: C.greenFill }, "labelling, benchmarks, fine-tuning"],
      [{ text: "3 issue folders", bold: true, fill: C.amberFill }, "kept: possible future pre-training"],
      [{ text: "review/", bold: true }, "check one turn by hand"],
      [{ text: "call_timeline.csv", bold: true }, "every item of a call, in order"],
      [{ text: "failures.csv", bold: true }, "turns that raised an error"],
    ], { x: tx, y: TOP, w: tw, colW: [2.2, tw - 2.2], fontSize: 18, rowH: 0.55 });
    L.bullets(s, [
      "Segment id <call_id>_t000_s01: turn 0, segment 1; sorting = time order",
      "Written turn by turn: a crash keeps the finished turns",
    ], { y: 5.75, h: 1.1, fontSize: 22, gap: 6, x: MX, w: CW });
    L.notes(s, `The output is organised for both labelling and later research. There are four groups, final, non_english, alignment_issues and boundary_issues, and all four have the same layout: one subfolder per call, and inside it one WAV clip and one text file per segment, plus one manifest.csv per group.
    Segment ids are built as call id, then t and the turn's position in the call with three digits, then s and the segment number with two digits, both counted from 0; so sorting the ids puts the clips of a call in time order. The clips are exact sample slices of the source audio, at the source's own sample rate; the source subtype is kept when WAV can store it, otherwise 16-bit PCM is used, for example for MP3 sources. The text files hold the original words of that segment, with the original spelling and punctuation.
    final is what goes to Label Studio for labelling, and later into benchmarking and fine-tuning. The three issue folders are kept but not labelled; unlabelled audio could later be used for self-supervised continued pre-training, which is not implemented.
    Next to the groups there is a review folder with one JSON file per turn for checking by hand: all words with their times and confidences, every gap with its p, q, v and s values, the chosen cuts and the cost. call_timeline.csv lists every item of every call in order, across all four groups. failures.csv lists turns that raised an error.
    Every manifest row also carries the group and the reason, the call times, the speech ratio and the low_speech_ratio flag, the alignment confidence, the boundary scores at both ends, and the settings version with a hash of the settings, so every label stays traceable. Paths in the manifest are relative to the output folder. Files are written turn by turn, so a crash keeps every finished turn. The next slide shows how the clips of one call stay connected.`);
  }

  // ---- Conversation context
  {
    const s = L.slide("", "Clips stay grouped by call, in time order", { stage: "Export" });
    const call = D.context_call;   // one real synthetic call from call_timeline.csv
    const items = [];
    call.forEach((r, i) => {
      if (i > 0 && parseFloat(r.gap_to_prev_s) > 0) items.push(["agent", "agent"]);
      items.push([r.segment_id.replace(/^.*_(t\d+_s\d+)$/, "$1"), r.group]);
    });
    const x0 = MX + 0.1, wTot = CW - 0.2, gap = 0.1;
    const widths = items.map(([, g]) => (g === "agent" ? 0.95 : 1.3));
    const scale = (wTot - gap * (items.length - 1)) / widths.reduce((a_, b_) => a_ + b_);
    let x = x0;
    const yb = 2.3, hb = 0.6, pos = [];
    items.forEach(([t, g], i) => {
      const w = widths[i] * scale;
      const style = g === "final" ? { fill: C.greenFill, line: C.good } : g === "agent" ? { fill: C.white, line: C.ink, dash: "dash" } : { fill: C.amberFill, line: C.warn };
      L.box(s, x, yb, w, hb, t, { ...style, fontSize: 16, bold: g !== "agent", r: 0.05 });
      pos.push({ x, w, g });
      x += w + gap;
    });
    const cust = pos.filter((q) => q.g !== "agent");
    for (let k = 0; k < cust.length - 1; k++) {
      L.arrow(s, cust[k].x + cust[k].w / 2 + 0.08, yb - 0.2, cust[k + 1].x + cust[k + 1].w / 2 - 0.08, yb - 0.2, { both: true, color: C.accent, width: 1.75 });
    }
    L.text(s, "prev / next links, also across folders", MX, yb - 0.62, 6.0, 0.34, { fontSize: 16, color: C.link, bold: true });
    L.arrow(s, x0, yb + hb + 0.2, x0 + wTot, yb + hb + 0.2, { width: 1.5 });
    L.text(s, "time in call dailytalk_d1562  (ids: dailytalk_d1562_t000_s00, …)", x0, yb + hb + 0.27, wTot, 0.34, { fontSize: 16, align: "right" });
    const fy = 3.62, fh = 0.75;
    const fin = call.filter((r) => r.group === "final").map((r) => r.segment_id.replace(/^.*_(t\d+_s\d+)$/, "$1"));
    const non = call.filter((r) => r.group === "non_english").map((r) => r.segment_id.replace(/^.*_(t\d+_s\d+)$/, "$1"));
    L.box(s, MX, fy, 8.4, fh, "final/dailytalk_d1562/\n" + fin.join(" · "), { fill: C.greenFill, line: C.good, fontSize: 16 });
    L.box(s, MX + 8.65, fy, CW - 8.65, fh, "non_english/dailytalk_d1562/\n" + non.join(" · "), { fill: C.amberFill, line: C.warn, fontSize: 16 });
    L.table(s, ["Column (manifests and call_timeline.csv)", "What it gives"], [
      [{ text: "order_in_call", bold: true }, "position of the item in its call, from 0"],
      [{ text: "prev_segment_id, prev_group, next_…", bold: true }, "the neighbours, even in another folder"],
      [{ text: "gap_to_prev_s, gap_to_next_s", bold: true }, "call time to the neighbours (agent speech, silence)"],
      [{ text: "call_timeline.csv", bold: true }, "all items of every call in order, for in-context use"],
    ], { y: 4.55, colW: [5.1, CW - 5.1], fontSize: 18, rowH: 0.46 });
    L.notes(s, `This slide is about conversation context, which matters for emotion: how a customer sounds often depends on what happened just before. So the clips are never a flat bag of files; they stay grouped by the call they came from, the way conversational emotion datasets such as DailyTalk and MELD group utterances by dialogue.
    The row of boxes is a real call from the test run, the synthetic call dailytalk_d1562, read from call_timeline.csv. Its first customer turn, turn 0, was 38.8 seconds long and was cut into five segments. Turn 1 was Spanish, so it went whole into non_english. Turn 2 was a short English turn and went whole into final. The dashed boxes are the agent turns that were removed upstream; in this test they leave 4-second gaps in the call time.
    In the file system, each group has one subfolder per call, so the clips of this call sit together in final/dailytalk_d1562, with the Spanish turn in non_english/dailytalk_d1562. Segment ids start with the call id, then the turn and segment numbers counted from 0, so sorting them gives time order.
    The links are in the data, not only in the folder names. Every manifest row and every row of call_timeline.csv has order_in_call, the ids and groups of the previous and next items, even when a neighbour lives in another folder, and the gaps in call time to both neighbours. call_timeline.csv lists every item of every call in order, across all folders; in the test run it has 31 rows, exactly the 31 manifest rows, and the links were correct in all 5 calls, each of which crosses folders twice. Together these allow in-context experiments later, for example giving a model the previous segments as context, or studying how emotion develops through a call.`);
    L.cite(s, "Same idea as dialogue-level grouping in", [[R.dailytalk.short, R.dailytalk.url], [R.meld.short, R.meld.url]]);
  }

  // ---- Validation: numbers
  {
    const s = L.slide("Validation on public data", "Validation: 5 synthetic calls built from DailyTalk");
    L.text(s, "Each call: a long English turn (32–39 s) + a Spanish-text turn + a short English turn = 15 turns", MX, TOP - 0.02, CW, 0.5, { fontSize: 21 });
    const tiles = [
      ["21", "segments from 5 long turns,\n4.9–11.9 s (median 8.2 s)"],
      ["16 / 16", "cuts at an utterance join\nor a sentence end"],
      ["−41 dB", "median level at the cut\nvs speech (worst −37 dB)"],
      ["10 · 5 · 0 · 0", "turns in final · non_english ·\nalignment · boundary issues"],
      ["100%", "of words in exactly one\nsegment; clips exact slices"],
      ["5 / 5", "calls with correct prev/next\nlinks and call times"],
    ];
    const tw = 3.85, th = 2.05, gx = 0.29, gy = 0.25;
    tiles.forEach(([n, c], i) => {
      const x = MX + (i % 3) * (tw + gx), y = TOP + 0.65 + Math.floor(i / 3) * (th + gy);
      s.addShape(p.ShapeType.rect, { x, y, w: tw, h: th, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: 0.09, h: th, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
      L.text(s, n, x + 0.3, y + 0.12, tw - 0.45, 0.8, { fontSize: 40, bold: true, color: C.link, valign: "middle" });
      L.text(s, c, x + 0.3, y + 0.98, tw - 0.45, 0.95, { fontSize: 20 });
    });
    L.notes(s, `We cannot show real calls, so the validation uses public data. The integration test builds five synthetic calls, one per DailyTalk dialogue; DailyTalk is CC BY-SA 4.0. In each call, the speaker with the most audio plays the customer and has three turns: a long English turn made by joining consecutive utterances with inserted pauses of known position, 32 to 39 seconds long; a turn with Spanish text over the next utterances, longer than 10 seconds in three calls; and a short English turn. One long turn is stored as MP3. That gives 15 turns, with turn start times in turns.csv and a 4-second gap for each removed agent turn.
    Routing worked as designed: 10 English turns went to final, the 5 long ones cut into 21 segments and the 5 short ones whole, 26 clips in total; the 5 Spanish-text turns went whole to non_english; the two issue folders stayed empty.
    The 21 segments are 4.9 to 11.9 seconds long, median 8.2 seconds, all inside the 2 to 15 second window. Of the 16 cuts, 10 fell inside the inserted pause at an utterance join, and the other 6 at a real pause after a sentence end inside an utterance; so all 16 cuts are at a join or a sentence end. These turns are made of a few long, multi-sentence utterances, only 3 or 4 joins per turn, so sentence ends inside utterances are natural cut places too. Every cut now sits in the middle of the quiet part of its gap, between 48 and 60 percent of the way through the gap. The audio level in 20 milliseconds around each cut was a median of 41 dB below speech, at worst 37 dB below, so no cut fell into speech. Every word is in exactly one segment, and every WAV clip is an exact sample slice of its source.
    The context links were correct in all 5 calls: call_timeline.csv has 31 rows, the same 31 as the four manifests together, and each call crosses folders twice. Mean word confidence per long turn was 0.87 to 0.97. Speed: 3.2 to 4.8 seconds per long turn on the Mac CPU with 4 threads; an earlier run measured a peak memory footprint of 1.2 gigabytes. The 67 unit tests pass.`);
    L.cite(s, "tests/integration_dailytalk.py, run 2026-11-13 (assets/integration_run.log)", [[R.dailytalk.short, R.dailytalk.url]]);
  }

  // ---- Validation: listen to two turns
  const listen = (turnId, title, lead, notesText) => {
    const s = L.slide("Validation on public data", title);
    const info = D.listen[turnId], au = AU[turnId];
    L.text(s, "Synthetic: consecutive utterances of one DailyTalk speaker, joined with inserted pauses (green)", MX, TOP - 0.05, CW, 0.42, { fontSize: 18 });
    const fig = L.image(s, `fig/listen_${turnId}.png`, MX, 2.05, CW, 3.8, { alt: `Synthetic DailyTalk turn ${turnId} with its cuts` });
    const ax0 = fig.x + info.axes_left * fig.w, axw = info.axes_width * fig.w, dur = info.duration_s;
    const yb = 5.95, d = 0.42;
    L.audio(s, au.whole.file, MX + 0.4, yb, d);
    L.text(s, "whole turn", MX, yb + d + 0.02, 1.25, 0.3, { fontSize: 16, align: "center", bold: true });
    au.segments.forEach((g) => {
      const cx = ax0 + ((g.start_s + g.end_s) / 2) / dur * axw;
      L.audio(s, g.file, cx - d / 2, yb, d);
      L.text(s, `${g.start_s.toFixed(1)}–${g.end_s.toFixed(1)} s`, cx - 0.8, yb + d + 0.02, 1.6, 0.3, { fontSize: 16, align: "center" });
    });
    const texts = au.segments.map((g, i) => `Segment ${i + 1}, ${g.start_s.toFixed(1)} to ${g.end_s.toFixed(1)} seconds: "${g.text}"`).join("\n    ");
    L.notes(s, notesText + `\n    Click a play button to hear a clip in the slide show; click again to pause. The left button plays the whole input turn, the others play the exported segments under their part of the timeline. The transcripts of the segments, exactly as exported:\n    ${texts}`);
    L.cite(s, "Audio: DailyTalk (Lee et al., ICASSP 2023), CC BY-SA 4.0; clips adapted (joined and cut)", [["DailyTalk", R.dailytalk.url]]);
  };
  listen("d418_long_en", "Listen: a turn cut at the utterance joins", "",
    `This is one of the synthetic turns from the test, d418_long_en, 32.2 seconds, so you can hear what the segmenter does. Be clear that it is synthetic: four consecutive utterances of one DailyTalk speaker joined with inserted pauses, the green bands, not one continuous recording; the pauses contain quiet room noise.
    The rows show the waveform, the Silero speech probability, the aligned words as dark bars and the four exported segments with their lengths. All three cuts fell in the middle of the inserted pauses at the utterance joins, each with a boundary score of about 0.97 to 0.98. The segments are 5.6, 7.8, 10.2 and 8.6 seconds long.`);
  listen("d1562_long_en", "Listen: most cuts at sentence ends inside long utterances", "",
    `A contrasting example, d1562_long_en, 38.8 seconds, again synthetic: five consecutive utterances of one speaker with inserted pauses. Here the utterances are long and contain several sentences each. The programme cut at three sentence ends inside the utterances and at one join, at about 28.7 seconds. The first join, at about 4 seconds, would leave a piece that is too short, and the second, at about 10 seconds, would make the pieces around it uneven, so real pauses after sentence ends were the better places there. The segments are 7.7, 8.5, 5.1, 7.3 and 10.1 seconds long. Listen to whether each segment starts and ends cleanly.`);

  // ---- Validation: method comparison
  {
    const s = L.slide("Validation on public data", "Only whole-turn DP gives sentence-length pieces with clean cuts", { titleSize: 28 });
    const pct = (x) => (x == null ? "–" : Math.round(x * 100) + "%");
    const row = (k, bold) => {
      const m = M[k], b = (t) => ({ text: t, bold });
      return [b(m.label), b(String(m.pieces)), b(m.median_s.toFixed(1) + " s"), b(String(m.under_3s)), b(String(m.cuts_inside_word)), b(pct(m.join_or_sentence_share)), b(m.mean_s_at_cuts.toFixed(2))];
    };
    L.table(s, ["5 long English turns", "Pieces", "Median", "< 3 s", "In a word", "Join or sentence end", "Mean s at cut"], [
      row("silero_100", false), row("silero_300", false), row("fixed", false), row("greedy", false),
      row("ours", true).map((c) => ({ ...c, fill: C.blueFill })),
    ], { y: TOP - 0.05, colW: [3.7, 1.0, 1.15, 0.9, 1.35, 2.4, 1.63], fontSize: 18, rowH: 0.42, align: { 1: "center", 2: "center", 3: "center", 4: "center", 5: "center", 6: "center" } });
    L.image(s, "fig/compare_methods.png", MX, 4.15, CW, 2.75, { alt: "Five ways to cut one synthetic turn" });
    L.notes(s, `This is the method comparison, now that the dynamic programme has been explained. All five methods run on the same five long English turns of the test set; Silero's own segments come from its probabilities, and the other baselines are recomputed from the gaps and words stored in the review files, without any model.
    Rows: Silero's own segments with the default 100 millisecond silence rule and with a 300 millisecond rule; a fixed chop every 7 seconds; a greedy rule that cuts at the first allowed pause at least 7 seconds after the previous cut; and our method, the whole-turn dynamic programme of this tool. For Silero a piece is a speech chunk, and its cut is the middle of the silence between two chunks. Every cut is checked against the aligned words: inside a word, in the word gap at an utterance join, after a sentence end, or elsewhere, and gets the boundary score of its gap, zero inside a word.
    Silero's default gives ${M.silero_100.pieces} pieces, ${M.silero_100.under_3s} of them under 3 seconds. Its cuts are good, ${pct(M.silero_100.join_or_sentence_share)} at a join or a sentence end, because it cuts at real silences, but the pieces are fragments. With a 300 millisecond rule Silero's cut quality is as good as ours, ${pct(M.silero_300.join_or_sentence_share)} with a mean score of ${M.silero_300.mean_s_at_cuts.toFixed(2)}, but it still leaves ${M.silero_300.under_3s} pieces under 3 seconds, a median of only ${M.silero_300.median_s.toFixed(1)} seconds, and it ignores the transcript and drops the silence between chunks.
    The fixed 7-second chop gives perfect lengths but cuts inside a word ${M.fixed.cuts_inside_word} times out of ${M.fixed.cuts}. The greedy rule never cuts inside a word, but on these turns its first pause after 7 seconds is almost always an ordinary short pause: only ${M.greedy.cuts_at_join + M.greedy.cuts_at_sentence_end} of ${M.greedy.cuts} cuts are at a join or a sentence end, the mean score is ${M.greedy.mean_s_at_cuts.toFixed(2)}, and it leaves a short tail under 3 seconds (${M.greedy.under_3s} in total).
    Our method gives ${M.ours.pieces} pieces with a median of ${M.ours.median_s.toFixed(1)} seconds, none under 3 seconds, no cut inside a word, ${M.ours.cuts_at_join + M.ours.cuts_at_sentence_end} of ${M.ours.cuts} cuts at a join or a sentence end, and a mean score of ${M.ours.mean_s_at_cuts.toFixed(2)}. So the honest summary is: Silero with a long silence rule matches us on cut quality but not on length control; greedy and the fixed chop control length but cut in bad places; only the whole-turn programme gets both. The figure shows the five methods on one turn, d1562.`);
    L.cite(s, "Baselines recomputed from the review files (scripts/make_data_figures.py)", [[R.dailytalk.short, R.dailytalk.url]]);
  }

  // ---- Validation: are the weights critical?
  {
    const s = L.slide("Validation on public data", "Are the weights critical? Cut quality stays high; exact cuts move");
    const pct = (x) => Math.round(x * 100) + "%";
    const sg = SEN.single, df = SEN.default;
    L.table(s, ["Weights p · q · v", "Join or sentence end", "Pieces", "< 3 s", "Same cuts as default"], [
      [{ text: "0.5 · 0.3 · 0.2 (default)", bold: true, fill: C.blueFill }, { text: pct(df.good_share), fill: C.blueFill }, { text: String(df.n_segments), fill: C.blueFill }, { text: "0", fill: C.blueFill }, { text: "100%", fill: C.blueFill }],
      [`${SEN.settings} settings around it`, `${pct(SEN.good_share_min_max[0])}–${pct(SEN.good_share_min_max[1])}`, `${SEN.segments_min_max[0]}–${SEN.segments_min_max[1]}`, String(SEN.under_3s_max), `median ${pct(SEN.agree_default_min_median[1])}`],
      ["pause only", pct(sg["pause only"].good_share), String(sg["pause only"].n_segments), String(sg["pause only"].under_3s), pct(sg["pause only"].agree_default)],
      ["punctuation only", pct(sg["punctuation only"].good_share), String(sg["punctuation only"].n_segments), String(sg["punctuation only"].under_3s), pct(sg["punctuation only"].agree_default)],
      ["silence only", pct(sg["silence only"].good_share), String(sg["silence only"].n_segments), String(sg["silence only"].under_3s), pct(sg["silence only"].agree_default)],
    ], { y: TOP - 0.05, colW: [3.9, 2.8, 1.3, 1.0, 3.13], fontSize: 18, rowH: 0.42, align: { 1: "center", 2: "center", 3: "center", 4: "center" } });
    L.image(s, "fig/sensitivity.png", MX, 4.15, CW, 2.7, { alt: "Share of cuts identical to the default for 90 weight settings" });
    L.notes(s, `Because the weights are our own choice, we checked how much they matter. For the five long English turns of the test set, the review files store every gap with its p, q and v. We recomputed the score for ${SEN.settings} weight settings, pause weight 0.3 to 0.7, punctuation weight 0 to 0.5 and silence weight 0.1 to 0.3, each normalised to sum to one, and re-ran the dynamic programme in plain Python, with no models.
    The default weights put ${pct(df.good_share)} of the cuts at a join or a sentence end. Across all ${SEN.settings} settings that share stays between ${pct(SEN.good_share_min_max[0])} and ${pct(SEN.good_share_min_max[1])}, ${SEN.settings_as_good_as_default} settings are as good as the default, no setting produces a piece under 3 seconds, and the number of pieces only moves between ${SEN.segments_min_max[0]} and ${SEN.segments_min_max[1]}. The heat maps show the share of cuts that are identical to the default cuts, with the default marked in orange: the median is ${pct(SEN.agree_default_min_median[1])}, and the cuts move most when the punctuation weight goes to zero or the silence weight is low. When cuts move, they mostly move to another good place: the join-or-sentence-end share never falls below ${pct(SEN.good_share_min_max[0])}.
    The single cues are interesting and need an honest reading. Pause alone is clearly worse: ${pct(sg["pause only"].good_share)} good cuts and more pieces. Punctuation alone and silence alone reach 100% here, but these synthetic turns are unusually clean: the transcripts have perfect punctuation, and the inserted pauses are near-digital silence. On real calls, recogniser punctuation is imperfect and pauses contain noise, breathing and backchannels, which is exactly when a single cue fails. So this check says the exact weights are not critical; it does not prove that the combination is better, which only real calls can show. The pilot should repeat this check on real data.`);
    L.cite(s, "Heat maps: % of cuts identical to the default (orange box); scripts/sensitivity.py on the 2026-11-13 run");
  }

  // ---- Limitations and next steps
  {
    const s = L.slide("Next steps", "Limitations and next steps");
    L.table(s, ["Limitation today", "Next step"], [
      ["Tested on public and synthetic audio only", "pilot on real turns: slow, fast, noisy, quiet"],
      ["Thresholds and weights are provisional", "recalibrate on real scores; compare with Silero"],
      ["MP3 offsets checked only on our own files", "check MP3s from the call system on a few examples"],
      ["Mono: agent backchannels stay in the audio", "they lower scores; listeners can mark bad clips"],
      ["No Spanish aligner: non-English turns stay whole", "add a Spanish model through the settings"],
      ["Base model trained on read audiobooks", "listen to word edges; try the LV60K model"],
      ["Issue folders are not used yet", "possible self-supervised pre-training later"],
    ], { y: TOP, colW: [5.9, 6.23], fontSize: 20, rowH: 0.6 });
    L.notes(s, `What is not done yet. Most importantly, the segmenter has only been tested on public and synthetic audio, not on real calls. The next step is a pilot on a manageable sample of real customer turns that covers slow and fast speech, frequent and absent pauses, low volume, noise and expressive sounds, compared against Silero's own segments on the same turns.
    The thresholds and weights are starting values. In the pilot we set the alignment thresholds from the real distribution of confidence scores, and tune the target length, lambda and the score weights by listening to the cuts and by listener agreement.
    MP3 decoding was checked only on MP3 files written by the same library, which showed no sample offset. MP3s from the call system may have encoder delays, so we must check a few examples to make sure segment times line up with the original call.
    The recordings are mono, so the agent's backchannels and overlaps stay in the customer audio. Neither VAD nor the aligner can separate speakers; agent speech only lowers the scores and confidences around it, and listeners can mark agent-dominated clips as bad.
    Spanish and mixed turns go whole into non_english for now. A Spanish wav2vec 2.0 model can be added through the settings; the current candidate is jonatasgrosman/wav2vec2-large-xlsr-53-spanish under Apache-2.0, whose licence must be checked again before use.
    The base English model was fine-tuned on read audiobooks, so telephone conversation is out of domain. Forced alignment is far more robust than recognition, but in the pilot we will listen to a sample of word boundaries and try the larger LV60K model if the base model misplaces many words.
    Finally, the three issue folders are kept but not used yet. Their audio could later feed self-supervised continued pre-training of an emotion2vec-style encoder, or of a Whisper encoder together with the transcripts; that is an option, not a plan.`);
  }

  // ---- Appendix divider (hidden)
  {
    const s = L.slide("Appendix", "Appendix: settings, installation, licences, references", { appendix: true });
    L.text(s, "These slides are hidden in the slide show; they are here for reading and for questions.", MX, TOP + 0.2, CW, 0.6, { fontSize: 22 });
    L.notes(s, `Appendix. The following slides are hidden in the slide show: the settings to tune in the pilot, the offline installation and the licences of every component, and the references. They are kept for reading after the talk and for answering questions.`);
  }

  // ---- Settings
  {
    const s = L.slide("Appendix · Settings", "Settings to tune in the pilot (config/segment_turns.yaml)", { titleSize: 28, appendix: true });
    L.table(s, ["Setting", "Start", "Effect"], [
      [{ text: "keep_whole_max_s", mono: true }, "10 s", "higher: fewer turns are cut"],
      [{ text: "target_s", mono: true }, "7 s", "lower (5 s) if pieces often hold two emotions"],
      [{ text: "min_s, max_s", mono: true }, "2 s, 15 s", "one window; the length cost keeps pieces near 7 s"],
      [{ text: "cut_weight_lambda", mono: true }, "1.0", "higher: cuts follow the best pauses, lengths less even"],
      [{ text: "weights p · q · v", mono: true }, "0.5 · 0.3 · 0.2", "0.7 · 0 · 0.3 when there is no punctuation"],
      [{ text: "typical pause", mono: true }, "median > 0.1 s", "0.3 s if fewer than 3 such gaps"],
      [{ text: "alignment thresholds", mono: true }, "0.3 · 0.5 · 0.4", "word · turn mean · weak share; provisional"],
      [{ text: "trim_edges", mono: true }, "off", "on: edge silence ignored for the 10 s test"],
    ], { y: TOP, colW: [3.6, 2.6, 5.93], fontSize: 18, rowH: 0.56 });
    L.notes(s, `All numbers are starting values for a pilot, kept in one YAML file. The ones most likely to change:
    keep_whole_max_s, 10 seconds: turns up to this length are never cut. The target length, 7 seconds: if listeners often hear two emotions in one piece, try 5 seconds. The single length window of 2 to 15 seconds comes from the upstream minimum turn length and the 15-second window of Vox-Profile Whisper, and should only change with evidence; the length cost keeps pieces near 7 seconds anyway.
    Lambda, 1.0, is the price of a cut: higher means cuts follow the best pauses even if lengths become uneven; lower means more even lengths with less attention to pause quality. The score weights are 0.5, 0.3 and 0.2, or 0.7 and 0.3 without punctuation. The typical pause is the median of gaps longer than 0.1 seconds, or 0.3 seconds if there are fewer than three.
    The alignment thresholds, 0.3 for a word, 0.5 for the turn mean and 0.4 for the share of weak words, are provisional; they come from synthetic turns and must be set again from real calls. Edge trimming is off by default.
    There are also technical settings not shown: the device for the alignment model, 60-second chunks with 2 seconds of context for long audio, the VAD speech threshold of 0.5, and the low-speech-ratio flag at 0.4. Changing any setting that affects the cuts changes the settings hash written to every manifest row, so labels stay traceable to the configuration.`);
  }

  // ---- Offline deployment and licences
  {
    const s = L.slide("Appendix · Deployment", "Offline install and licences", { appendix: true });
    const steps = [
      ["download.py", "with internet: ≈ 391 MB per platform, SHA-256 checked"],
      ["copy", "the preparation_tools/ folder to the offline server"],
      ["install.py", "offline: verify, install wheels, smoke test"],
      ["segment_turns.py", "runs fully offline on a CPU"],
    ];
    const bw = 2.85, gap = (CW - 4 * bw) / 3, y = TOP, bh = 1.65;
    steps.forEach(([t, d], i) => {
      const x = MX + i * (bw + gap);
      s.addShape(p.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: i === 3 ? C.greenFill : C.blueFill }, line: { color: i === 3 ? C.good : C.accent, width: 1.25 } });
      L.text(s, t, x + 0.12, y + 0.08, bw - 0.24, 0.42, { fontSize: 18, bold: true, align: "center", mono: true });
      L.text(s, d, x + 0.12, y + 0.52, bw - 0.24, bh - 0.6, { fontSize: 16, align: "center", kind: "diagram" });
      if (i < 3) L.arrow(s, x + bw, y + bh / 2, x + bw + gap, y + bh / 2, { width: 1.75 });
    });
    L.table(s, ["Component", "Size", "Licence"], [
      ["wav2vec 2.0 base 960 h checkpoint", "377.7 MB", "MIT"],
      ["Silero VAD 6.2.3 (model inside the wheel)", "11.3 MB", "MIT"],
      ["soundfile 0.14.0 (+ bundled libsndfile)", "1.1–1.3 MB", "BSD-3 (libsndfile LGPL-2.1, unmodified)"],
      ["PyYAML, cffi, pycparser, packaging", "0.6–1.2 MB", "MIT · MIT-0 · BSD-3 · Apache-2.0/BSD-2"],
      [{ text: "Not used: MMS_FA, VoxPopuli wav2vec 2.0", bold: true }, "–", { text: "CC BY-NC 4.0 (non-commercial)", bold: true, fill: C.redFill }],
    ], { y: 3.55, colW: [5.0, 1.9, 5.23], fontSize: 18, rowH: 0.56 });
    L.notes(s, `The tool must run on an offline company machine, so everything it needs is pinned and verified. On a machine with internet, download.py fetches the wheels and the alignment checkpoint. By default it takes them from this repository's GitHub Release, which holds the files unchanged from PyPI and download.pytorch.org; the upstream hosts can be selected instead. Every file is checked against the size and SHA-256 recorded in the manifests, and interrupted downloads resume. For one platform this is about 391 megabytes: 13 to 14 megabytes of wheels plus the 377.7 megabyte checkpoint. Then the whole preparation_tools folder is copied to the offline machine, where install.py checks Python and the torch prerequisites, verifies every file before installing anything, installs the wheels one by one without any index, runs pip check and finishes with a smoke test of VAD, the checkpoint and audio decoding. After that, segment_turns.py runs fully offline. A maintainer script, publish.py, uploads the pinned files to the Release.
    The licence table shows why every component is allowed for internal commercial use. The wav2vec 2.0 checkpoint is MIT, originally published by the wav2vec 2.0 authors and redistributed by torchaudio under the same licence. Silero VAD is MIT, and its model file ships inside the 11.3 megabyte wheel. soundfile is BSD-3; it bundles libsndfile, which is LGPL 2.1, loaded as a separate, unmodified shared library. The small helper packages are MIT, MIT-0, BSD-3 and Apache or BSD. torch and torchaudio themselves come from the existing environment, under BSD licences.
    Two convenient alternatives are deliberately not used: torchaudio's multilingual MMS_FA model and the VoxPopuli wav2vec 2.0 models, because they are CC BY-NC 4.0, which forbids commercial use.`);
    L.cite(s, "preparation_tools/LICENSES.md, dependencies/manifest.json, model_weights/manifest.json, download.py, install.py");
  }

  // ---- References
  const refs = ["vibevoice", "silero", "wav2vec2", "ctc", "librispeech", "torchaudio", "tutorial", "viterbi", "bellman", "jackson", "pelt", "shriberg", "wightman", "swerts", "goldman", "liu", "ctcseg", "whisperx", "mfa", "dailytalk", "meld"];
  const pages = [refs.slice(0, 7), refs.slice(7, 14), refs.slice(14)];
  pages.forEach((keys, k) => {
    const s = L.slide("Appendix · References", `References (${k + 1}/${pages.length})`, { appendix: true });
    const runs = [];
    keys.forEach((key, i) => {
      const r = R[key];
      if (r.url) {
        runs.push({ text: r.full + "  ", options: { breakLine: false, paraSpaceAfter: 10 } });
        runs.push({ text: r.url, options: { color: C.link, underline: { style: "sng", color: C.link }, hyperlink: { url: r.url }, breakLine: true, paraSpaceAfter: 10 } });
      } else runs.push({ text: r.full, options: { breakLine: true, paraSpaceAfter: 10 } });
    });
    s.addText(runs, { x: MX, y: TOP - 0.05, w: CW, h: 5.3, fontFace: FONT, fontSize: 16, color: C.ink, valign: "top", margin: 0, lineSpacingMultiple: 1.05 });
    L.notes(s, [
      `References, part one: VibeVoice-ASR, the upstream recogniser with diarization and timestamps; Silero VAD for speech probabilities; wav2vec 2.0, the acoustic model behind the aligner, and LibriSpeech, the data it was fine-tuned on; CTC, the training criterion that gives the blank label and makes forced alignment possible; the TorchAudio 2.1 paper and the official forced-alignment tutorial; and Viterbi's 1967 paper for the best-path search. Every entry was checked against the publisher, Crossref or arXiv on 13 November 2026.`,
      `References, part two: Bellman's book on dynamic programming; optimal partitioning by Jackson and colleagues and PELT by Killick and colleagues, the penalised-cost segmentation methods our cost follows; and the literature behind the three cues of the boundary score: Shriberg and colleagues on combining prosodic and word cues, Wightman and colleagues on timing at prosodic boundaries, Swerts on pauses at discourse boundaries, and Goldman-Eisler's book on pauses in spontaneous speech, which has no stable link.`,
      `References, part three: Liu and colleagues on sentence-boundary detection with word and prosodic cues; CTC-Segmentation, WhisperX and the Montreal Forced Aligner as related or alternative alignment tools; DailyTalk, the public CC BY-SA 4.0 dialogue dataset used to build the synthetic validation calls and the audio examples; and MELD, a conversational emotion dataset that, like DailyTalk, groups utterances by dialogue.`,
    ][k]);
  });
};

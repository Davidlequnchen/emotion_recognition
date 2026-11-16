// Boundary score, relative pause, DP formulation and algorithm, worked example, over-fragmentation, routing
const R = require("./refs");

module.exports = function (L, A) {
  const { C, W, H, MX, CW, TOP, FONT } = L;
  const p = L.pptx;
  const WE = A.worked;

  // ---- 17 Boundary score
  {
    const s = L.slide("", "Step 5 · Every word gap gets a boundary score s from 0 to 1", { stage: "Score" });
    L.text(s, [
      { text: "s = 0.5 ", options: {} }, { text: "p", options: { bold: true, color: C.link } },
      { text: "  +  0.3 ", options: {} }, { text: "q", options: { bold: true, color: C.darkOrange } },
      { text: "  +  0.2 ", options: {} }, { text: "v", options: { bold: true, color: C.darkGreen } },
    ], MX, TOP - 0.05, CW, 0.75, { fontSize: 36, bold: true, align: "center", valign: "middle" });
    const cards = [
      ["p", "relative pause", "gap ÷ (2 × typical pause),\nat most 1", C.accent, C.blueFill],
      ["q", "punctuation", "1 after . ? !\n0.5 after , ; :   0 otherwise", C.accent2, C.orangeFill],
      ["v", "real silence", "1 − mean VAD probability\nin the gap", C.good, C.greenFill],
    ];
    const cw = 3.85, gap = 0.29, y = TOP + 0.85, h = 1.95;
    cards.forEach(([sym, t, d, col, fill], i) => {
      const x = MX + i * (cw + gap);
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h, fill: { color: fill }, line: { color: col, width: 1.5 } });
      L.text(s, sym, x + 0.25, y + 0.1, 0.7, 0.7, { fontSize: 40, bold: true, valign: "middle" });
      L.text(s, t, x + 1.0, y + 0.1, cw - 1.15, 0.7, { fontSize: 22, bold: true, valign: "middle" });
      L.text(s, d, x + 0.25, y + 0.9, cw - 0.45, 0.95, { fontSize: 20 });
    });
    L.bullets(s, [
      "Why a score: a ranked list of acceptable cut places; step 6 chooses among them",
      "Only word gaps, never inside a word; every gap counts, so fast speakers can still be cut",
      "No punctuation in the transcript: s = 0.7 p + 0.3 v",
      { t: "The weighted sum and its weights are our design: starting values, tuned in the pilot", b: true },
    ], { y: 4.6, h: 2.25, fontSize: 21, gap: 6 });
    L.notes(s, `Every gap between two neighbouring words is a candidate cut, and each gets a score between 0, a bad place to cut, and 1, an ideal place. The score combines three cues with weights 0.5, 0.3 and 0.2.
    p is the pause length, but relative to this speaker, which the slide after next explains. A pause twice as long as the speaker's usual pause gets full marks. q is the text cue from the transcript's punctuation: 1 if the word before the gap ends a sentence with a full stop, question mark or exclamation mark; 0.5 after a comma, semicolon or colon; 0 otherwise. v is real silence from the VAD: one minus the mean speech probability inside the gap. A quiet gap scores near 1; a gap that holds a sigh, a laugh or the agent's backchannel scores low.
    Why score gaps at all? The dynamic programme in step 6 needs to know, for every possible cut place, how acceptable it is. The score gives exactly that: a ranked list of places where cutting is acceptable, and the programme then chooses among them while also keeping the pieces near 7 seconds.
    Only gaps between words are candidates, so a cut can never fall inside a word. But every gap is a candidate, even a very short one, so a fast speaker who barely pauses still has somewhere to cut, just at a lower score.
    If a transcript has no punctuation at all, q carries no information, so its weight goes to zero and the score becomes 0.7 p plus 0.3 v.
    To be clear about where this comes from: the weighted sum and the weights 0.5, 0.3 and 0.2 are our own design choice, not taken from a paper. They are starting values to be tuned in the pilot. What does come from the literature is the choice of cues, which the next slide shows, and later a sensitivity check shows how much the exact weights matter.`);
  }

  // ---- 17b Why combine three cues
  {
    const s = L.slide("", "Why three cues, and not just one?", { stage: "Score" });
    L.table(s, ["Cue alone", "Fails when", "Evidence for the cue"], [
      [{ text: "pause length", bold: true }, "slow speakers over-cut, fast ones under-cut; a hesitation looks like a sentence end", "pauses and timing mark boundaries (Wightman 1992; Swerts 1997); pause length varies with speaker and task (Goldman-Eisler 1968)"],
      [{ text: "punctuation", bold: true }, "punctuation missing or wrong; ignores the audio", "word cues help find sentence ends (Liu et al. 2006)"],
      [{ text: "VAD silence", bold: true }, "a thinking pause looks like a sentence end", "—"],
      [{ text: "all three", bold: true, fill: C.greenFill }, { text: "each covers the others’ failure", fill: C.greenFill }, { text: "prosody + words beat words alone (Shriberg et al. 2000)", fill: C.greenFill }],
    ], { y: TOP, colW: [2.2, 4.4, 5.53], fontSize: 18, rowH: [0.5, 1.05, 0.72, 0.72, 0.72] });
    L.text(s, [{ text: "Weights follow the roles: ", options: { bold: true } }, { text: "pause = main acoustic evidence (0.5) · punctuation = linguistic support (0.3) · silence = safety check (0.2)" }],
      MX, 5.75, CW, 0.85, { fontSize: 20, valign: "middle" });
    L.notes(s, `The audience will rightly ask where the score comes from, and why we do not simply use one cue. This table is the answer.
    Pause length alone is the classic approach, and pauses do mark boundaries: Wightman and colleagues showed that timing around a boundary distinguishes several boundary strengths once speaking rate is normalised, and Swerts showed that pauses, together with pitch range, cue discourse boundaries in spontaneous speech. But pause behaviour also depends on the speaker and the task; Goldman-Eisler found longer pauses when speakers had to interpret rather than describe. That is why our p is relative to the same speaker. Even then, pause length alone cuts slow speakers too often and fast speakers too rarely, and it cannot tell a hesitation in the middle of a sentence from a sentence end.
    Punctuation alone depends entirely on the transcript. If the upstream recogniser puts punctuation in the wrong place or leaves it out, it fails, and it ignores the audio completely. Word-based cues are useful for finding sentence boundaries, as in the sentence-boundary detection work of Liu and colleagues, but they are not enough on their own.
    VAD silence alone cannot tell a thinking pause inside a sentence from the end of a sentence; both are silent.
    Combining the cues lets each cover the others' failure cases. The strongest evidence for that is Shriberg and colleagues: on broadcast news and telephone conversations, combining prosodic cues such as pauses with word-based cues gave a significant improvement over words alone, and pause features were among the most informative.
    The weights reflect the roles we give the cues: the relative pause is the main acoustic evidence, punctuation is linguistic support, and silence is a safety check that stops us cutting through a sigh, a laugh or the agent's backchannel. The exact numbers are ours and provisional.`);
    L.cite(s, "", [[R.shriberg.short, R.shriberg.url], [R.wightman.short, R.wightman.url], [R.swerts.short, R.swerts.url], [R.goldman.short, null], [R.liu.short, R.liu.url]]);
  }

  // ---- 18 Relative pause
  {
    const s = L.slide("", "Pause length is judged against the same speaker", { stage: "Score" });
    const fig = L.image(s, "fig/pause_score.png", W - MX - 6.0, TOP - 0.05, 6.0, 4.5, { alt: "Pause score p against gap length for a slow and a fast speaker" });
    L.bullets(s, [
      "Typical pause = median of the turn’s gaps over 0.1 s",
      "Slow (0.6 s): ordinary pauses get p = 0.5",
      "Fast (0.2 s): full marks from 0.4 s",
      { t: "No fixed pause threshold", b: true },
    ], { x: MX, y: TOP + 0.1, w: fig.x - MX - 0.3, h: 4.8, fontSize: 22, gap: 16 });
    L.notes(s, `A fixed pause threshold is always wrong for someone. If we say "cut at pauses longer than 0.5 seconds", a slow speaker who pauses half a second every few words gets cut into tiny pieces, and a fast speaker whose pauses are 0.2 seconds never gets cut at all.
    So p measures pause length relative to the same speaker in the same turn. First we find the speaker's typical pause: the median of all gaps longer than 0.1 seconds in this turn. If there are fewer than three such gaps, we use 0.3 seconds. Then p is the gap divided by twice the typical pause, capped at 1.
    The figure shows two speakers. For the slow speaker in blue, the typical pause is 0.6 seconds, so an ordinary pause scores only 0.5, and full marks start at 1.2 seconds. Their many ordinary pauses therefore do not attract cuts. For the fast speaker in orange, the typical pause is 0.2 seconds, and a 0.4-second pause already gets full marks, so their few clear pauses stand out.
    This is what replaces a hard-coded threshold: what counts as a strong pause adapts to each speaker automatically.`);
  }

  // ---- 19 DP cost
  {
    const s = L.slide("", "Step 6 · Every choice has a price; the cheapest plan wins", { stage: "DP" });
    L.text(s, "A plan = a set of cuts. Its price = the length price of every piece + the price of every cut.", MX, TOP - 0.05, CW, 0.5, { fontSize: 21 });
    const fig = L.image(s, "fig/length_cost.png", MX, 2.15, 6.3, 4.6, { alt: "Length price L(d) and the price of one poor cut" });
    const rx = fig.x + fig.w + 0.3, rw = W - MX - rx;
    const card = (y, h, title, lines, col, fill) => {
      s.addShape(p.ShapeType.rect, { x: rx, y, w: rw, h, fill: { color: fill }, line: { color: col, width: 1.5 } });
      L.text(s, title, rx + 0.2, y + 0.1, rw - 0.4, 0.42, { fontSize: 20, bold: true });
      L.text(s, lines, rx + 0.2, y + 0.55, rw - 0.4, h - 0.62, { fontSize: 20 });
    };
    card(2.2, 1.55, "Length of each piece", "((d − 7) ÷ 7)², only 2–15 s\n0 at 7 s · 0.18 at 4 or 10 s", C.accent, C.blueFill);
    card(3.9, 1.3, "Each cut: λ (1 − s), λ = 1", "perfect pause 0 · poor pause 0.8", C.accent2, C.orangeFill);
    card(5.35, 1.45, "So, in practice", `1 poor cut costs as much as a ${WE.poor_cut_equals_length_s.toFixed(1)} s piece;\na good pause (0.1) is worth it`, C.ink, C.panel);
    L.notes(s, `This slide makes the cost intuitive. Think of the programme as shopping for a plan, where a plan is a set of cuts. Every choice has a price, and the programme picks the plan with the lowest total price.
    The first price is for the length of each piece: ((d minus 7) divided by 7) squared. It is zero at 7 seconds, 0.18 at 4 or at 10 seconds, 0.51 at 2 seconds and 1.31 at 15 seconds, and pieces shorter than 2 or longer than 15 seconds are not allowed at all. The price is squared on purpose: being twice as far from 7 seconds costs four times as much, so small deviations are cheap and large ones expensive. It looks lopsided in seconds only because 7 is not the middle of the 2 to 15 window. Why 7 seconds? It is the middle of the preferred range of 4 to 10 seconds, about one or two sentences. The window itself comes from the shortest upstream turn, 2 seconds, and the 15-second window of Vox-Profile Whisper, which is also under emotion2vec+'s 20.48-second Mac GPU limit and the 30-second windows of the other models.
    The second price is for each cut: lambda times one minus the boundary score. A perfect pause with s equal to 1 is free; a poor pause with s equal to 0.2 costs 0.8. Lambda is 1.
    The dashed line in the figure makes the trade-off visible. One poor cut costs 0.8, the same as one piece of about ${WE.poor_cut_equals_length_s.toFixed(1)} seconds. So the programme would rather keep a 13-second piece than cut it at a bad pause. On the other hand, a good pause with s equal to 0.9 costs only 0.1, so it is worth using even just to bring a 10-second piece back towards 7 seconds, which saves 0.18 in length price.
    This form is standard. Minimising the sum of segment costs plus a price per cut is the penalised-cost formulation of optimal partitioning and change-point detection, as in Jackson and colleagues and in PELT. The difference is that our price per cut is not a constant: it depends on how good the boundary is.
    On the five long synthetic turns, re-running the programme with the earlier 3 to 12 second window gives exactly the same cuts, so the wider window changed nothing there; it only matters for unusual turns.`);
    L.cite(s, "Penalised-cost partitioning:", [[R.jackson.short, R.jackson.url], [R.pelt.short, R.pelt.url]]);
  }

  // ---- 20 DP algorithm
  {
    const s = L.slide("", "Dynamic programming: one forward pass, then walk back", { stage: "DP" });
    L.box(s, MX, TOP, CW, 1.05, [
      { text: "best[j] = min over i  { best[i] + L(t_j − t_i) + λ (1 − s_j) }", options: { breakLine: true, bold: true } },
      { text: "over earlier candidates i with t_j − t_i ≤ 15 s;   prev[j] = that i;   best[0] = 0", options: { fontSize: 18 } },
    ], { fill: C.panel, line: C.ink, fontSize: 22 });
    const nodes = [0, 7.4, 12.1, 15.3, 21.8, 23.0, 29.6, 34.2, 40];
    const chosen = [0, 7.4, 15.3, 23.0, 29.6, 40];
    const x0 = MX + 0.5, x1 = W - MX - 0.5, X = (t) => x0 + (x1 - x0) * t / 40, ya = 4.55, r = 0.15;
    s.addShape(p.ShapeType.line, { x: x0, y: ya, w: x1 - x0, h: 0, line: { color: C.ink, width: 1.5 } });
    // candidate last segments into j = 29.6 s
    const j = 29.6;
    const brackets = [[23.0, 3.35, C.accent, 3, "solid", 0], [15.3, 3.75, C.ink, 1.5, "dash", 1], [12.1, 3.0, C.bad, 1.5, "sysDot", 2]];
    brackets.forEach(([i, yb, col, wdt, dash, k]) => {
      L.path(s, [[X(i), ya - r], [X(i), yb], [X(j) - 0.05 * k, yb], [X(j) - 0.05 * k, ya - r]], { color: col, width: wdt, dash });
    });
    L.text(s, "best: from 23.0 s", X(23.0) + 0.1, 3.36, 2.6, 0.34, { fontSize: 16, bold: true, color: C.link });
    L.text(s, "from 15.3 s (14.3 s): costly", X(12.1) - 3.9, 3.58, 3.8, 0.34, { fontSize: 16, align: "right" });
    L.text(s, "from 12.1 s: 17.5 s > 15 s, not allowed", (X(12.1) + X(j)) / 2 - 2.6, 2.66, 5.2, 0.34, { fontSize: 16, align: "center" });
    nodes.forEach((t) => {
      const on = chosen.includes(t), isJ = t === j;
      s.addShape(p.ShapeType.ellipse, { x: X(t) - r, y: ya - r, w: 2 * r, h: 2 * r, fill: { color: isJ ? C.accent2 : on ? C.accent : C.white }, line: { color: isJ ? C.accent2 : on ? C.accent : C.ink, width: 1.5 } });
    });
    const lab = { 0: "0", 7.4: "7.4", 12.1: "12.1", 15.3: "15.3", 21.8: "21.8", 23.0: "23.0", 29.6: "j = 29.6", 34.2: "34.2", 40: "T = 40" };
    nodes.forEach((t) => {
      const low = t === 23.0;
      L.text(s, lab[t], X(t) - 0.6, ya + 0.2 + (low ? 0.32 : 0), 1.2, 0.32, { fontSize: 16, align: "center", bold: t === j });
    });
    L.text(s, "candidate cut times (s)", x1 - 3.0, ya - 0.55, 3.0, 0.32, { fontSize: 16, align: "right" });
    // walk back: a chain of left-pointing arrows under the axis
    const yw = 5.55;
    for (let k = chosen.length - 1; k > 0; k--) {
      L.arrow(s, X(chosen[k]) - 0.08, yw, X(chosen[k - 1]) + 0.08, yw, { color: C.accent, width: 2.5 });
    }
    L.text(s, "walk back with prev[ ]: cuts at 7.4, 15.3, 23.0 and 29.6 s; total cost " + WE.dp.total.toFixed(2), MX, 5.85, CW, 0.5, { fontSize: 20, align: "center", valign: "middle" });
    L.notes(s, `How do we find the cheapest partition without trying every combination? With dynamic programming.
    Put the candidate gaps in time order, and add the turn start, time zero, and the turn end, T, as fixed ends. For every candidate j we store best of j: the lowest possible cost of cutting the audio from the start up to t_j, with a cut at t_j. best of 0 is zero.
    Then we go through the candidates from left to right. For candidate j, we try every earlier candidate i as the start of the last segment. The cost of that choice is best of i, the cheapest way to get to i, plus the length cost of the segment from t_i to t_j, plus the price of cutting at j. We keep the smallest value as best of j and remember which i gave it in prev of j. Only starts within 15 seconds need checking, because longer segments are not allowed. The turn end has a score of 1, so it adds no cut cost.
    The diagram uses the strong candidates of the worked example. For j at 29.6 seconds, a last segment starting at 23.0 is the cheapest, shown in bold blue. Starting at 15.3 would make a 14.3-second segment: allowed by the window, but its length cost is 1.09, so it loses. Starting at 12.1 would make a 17.5-second segment, which is not allowed at all.
    When we reach the turn end, best of T is the cost of the best partition of the whole turn, here ${WE.dp.total.toFixed(2)}. Following prev back from the end, T to 29.6 to 23.0 to 15.3 to 7.4 to zero, reads off the chosen cuts.`);
    L.cite(s, "", [[R.bellman.short, R.bellman.url], [R.jackson.short, R.jackson.url]]);
  }

  // ---- 21 Exact, fast, repeatable
  {
    const s = L.slide("", "Why the result is exact, fast and repeatable", { stage: "DP" });
    L.table(s, ["Property", "Why"], [
      [{ text: "Exact", bold: true }, "the best cut-up to t_j extends the best cut-up to some t_i (Bellman); equals brute force (unit test)"],
      [{ text: "Fast", bold: true }, "at most M²/2 comparisons: 40 s ≈ 100 words → ≈ 5,000, milliseconds"],
      [{ text: "Repeatable", bold: true }, "no randomness; the settings hash is in every manifest row"],
      [{ text: "Related", bold: true }, "optimal partitioning and PELT: same recursion, statistical costs"],
    ], { y: TOP + 0.1, colW: [2.4, 9.73], fontSize: 20, rowH: 0.95 });
    L.notes(s, `Three properties make this suitable for a data pipeline.
    It is exact. The best partition that ends with a cut at t_j must consist of the best partition up to some earlier cut t_i, plus one last segment. If a cheaper way to reach t_i existed, we could swap it in and get a cheaper partition overall. This is Bellman's principle of optimality, and it means that remembering only best of i for every i loses nothing. The result is identical to comparing every possible partition; the unit tests check this against a brute-force search on the worked example.
    It is fast. With M candidates there are at most M squared over two comparisons, and fewer because only starts within 15 seconds are checked. A 40-second turn has roughly 100 words, so about 5,000 comparisons, which takes milliseconds. Forced alignment takes far longer than the dynamic programme.
    It is repeatable. There is nothing random in it: the same audio and transcript always give the same cuts. Every manifest row also stores a hash of all settings that change the cuts, so a label can always be traced back to the exact configuration.
    The same kind of recursion is known from optimal partitioning of data on an interval, by Jackson and colleagues, and from PELT, a fast change-point method by Killick and colleagues. They use statistical costs to find change points; our cost is about segment length and pause quality, and we deliberately do not try to detect emotion changes.`);
    L.cite(s, "", [[R.bellman.short, R.bellman.url], [R.jackson.short, R.jackson.url], [R.pelt.short, R.pelt.url]]);
  }

  // ---- 22 Worked example: figure
  {
    const s = L.slide("", "Worked example: a slow speaker, 40 s, 25 pauses", { stage: "DP" });
    L.text(s, "Typical pause 0.6 s → full marks from 1.2 s. 7 strong gaps labelled; 18 ordinary ones score 0.2–0.45.", MX, TOP - 0.02, CW, 0.5, { fontSize: 20 });
    L.image(s, "fig/worked_example.png", MX, 2.35, CW, 4.5, { alt: "Worked example: candidate scores and the DP and greedy partitions" });
    L.notes(s, `Let us run the method on the worked example from the method document, which the unit tests reproduce exactly, with the 2 to 15 second window. A slow speaker talks for 40 seconds and pauses 25 times. Their typical pause is 0.6 seconds, so a pause needs 1.2 seconds for full marks.
    Each bar is one candidate gap, at its time in the turn, with height equal to its score. Seven are strong: 7.4 seconds after "charged twice." with 0.99; 12.1 seconds, a plain 0.6-second pause, 0.43; 15.3 seconds after a comma, 0.75; 21.8 seconds, a long pause after a full stop but with a sigh in it, 0.84; 23.0 seconds after a full stop, quiet, 0.87; 29.6 seconds after a question mark, 0.99; and 34.2 seconds after a comma, 0.62. The other 18 light bars are ordinary pauses of 0.4 to 0.6 seconds.
    The programme cuts at 7.4, 15.3, 23.0 and 29.6 seconds, giving five pieces of 7.4, 7.9, 7.7, 6.6 and 10.4 seconds. The annotations say why. The question-mark pause at 29.6 is nearly perfect, so it is used. The gap at 21.8 is long and ends a sentence, but it holds a sigh, so its score is lower than that of the quiet gap 1.2 seconds later, and the cut goes to 23.0, leaving the sigh intact. The gap at 34.2 would even out the last piece, but at 0.62 it costs more than it saves, as the next slide shows.
    Many pauses did not mean many cuts: 25 candidates, four cuts. The DP's answer matches an exhaustive search over all 116,674 allowed partitions in the unit tests.`);
    L.cite(s, "Method document §4.6; data from tests/test_dp.py (worked_example)");
  }

  // ---- 23 Worked example: costs
  {
    const s = L.slide("", "Why the programme returns partition A", { stage: "DP" });
    const f2 = (x) => x.toFixed(2);
    L.table(s, ["Partition", "Segment lengths (s)", "Length price", "Cut price", "Total"], [
      [{ text: "A: cuts at 7.4, 15.3, 23.0, 29.6", bold: true }, "7.4, 7.9, 7.7, 6.6, 10.4", f2(WE.A.length), f2(WE.A.cut), { text: f2(WE.A.total), bold: true, fill: C.greenFill }],
      ["B: A + a cut at 34.2", "7.4, 7.9, 7.7, 6.6, 4.6, 5.8", f2(WE.B.length), f2(WE.B.cut), f2(WE.B.total)],
    ], { y: TOP, colW: [4.0, 3.6, 1.6, 1.4, 1.53], fontSize: 20, rowH: 0.62, align: { 2: "center", 3: "center", 4: "center" } });
    L.bullets(s, [
      `B evens out the end: length price ${f2(WE.A.length)} → ${f2(WE.B.length)}`,
      "but its extra cut at a 0.62 gap costs 0.38: total 0.68 → 0.97",
      "The sigh gap (0.84) loses to the quiet gap 1.2 s later (0.87); the sigh stays whole",
    ], { y: 3.75, h: 3.0, fontSize: 22, gap: 12 });
    L.notes(s, `The table compares the programme's answer with the obvious alternative, using the prices from three slides ago and the 2 to 15 second window.
    Partition A has segment lengths 7.4, 7.9, 7.7, 6.6 and 10.4 seconds. Its length price is ${f2(WE.A.length)}, mostly from the 10.4-second last piece, and its cut price is ${f2(WE.A.cut)}, because all four cuts have high scores. Total ${f2(WE.A.total)}.
    Partition B adds a cut at 34.2 seconds to even out the end. The lengths improve and the length price drops to ${f2(WE.B.length)}, but the 34.2 gap only scores 0.62, so cutting there costs 0.38, and the total rises to ${f2(WE.B.total)}. The extra cut is not worth it. The runner-up overall is A with the cut at 21.8 instead of 23.0, at ${WE.runner_up.total.toFixed(2)}.
    All these segments lie inside both the old 3 to 12 and the new 2 to 15 second window, so these totals did not change with the window.
    A question that came up: what would a simple greedy rule do, "cut at the first pause after 7 seconds"? On this turn it cuts at 7.4, 15.3, 23.0 and then 31.2 seconds. It skips the 0.99 gap at 29.6 because that comes only 6.6 seconds after the previous cut, and then takes the next pause after 30 seconds, an ordinary one with a score of ${WE.greedy.scores[3].toFixed(2)}, without looking at scores. So greedy gets the same number of pieces with similar lengths here; the difference is cut quality. The validation section compares the methods on real audio.`);
  }

  // ---- 24 Over-fragmentation and fast speakers
  {
    const s = L.slide("", "Many pauses do not mean many cuts, but long turns are always cut", { stage: "DP", titleSize: 28 });
    const cw = 5.9, y = TOP + 0.05, h = 5.05;
    const panel = (x, title, col, fill) => {
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h, fill: { color: fill }, line: { color: fill, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      L.text(s, title, x + 0.3, y + 0.22, cw - 0.6, 0.5, { fontSize: 24, bold: true });
    };
    panel(MX, "Slow speaker: no over-splitting", C.accent, C.blueFill);
    panel(MX + cw + 0.33, "Fast speaker: still cut", C.accent2, C.orangeFill);
    const mech = [["1", "Minimum 2 s per piece"], ["2", "Length cost prefers about 7 s: 40 s → 5–6 pieces"], ["3", "Every cut has a price λ(1 − s)"]];
    mech.forEach(([n, t], i) => {
      const yy = y + 0.95 + i * 1.33;
      s.addText(n, { x: MX + 0.3, y: yy + 0.05, w: 0.55, h: 0.55, shape: p.ShapeType.ellipse, fill: { color: C.accent }, line: { color: C.accent, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 1.05, yy, cw - 1.35, 1.2, { fontSize: 21 });
    });
    const rx = MX + cw + 0.33 + 0.3, rw = cw - 0.6;
    L.text(s, "20 s, every gap s = 0.2:", rx, y + 0.9, rw, 0.45, { fontSize: 20 });
    const bar = (yy, parts, label, win) => {
      let x = rx;
      parts.forEach((d) => { L.box(s, x, yy, rw * d / 20 - 0.04, 0.55, d.toFixed(d % 1 ? 1 : 0) + " s", { fill: win ? C.blueFill : C.white, line: win ? C.accent : C.ink, fontSize: 18, r: 0.03 }); x += rw * d / 20; });
      L.text(s, label, rx, yy + 0.62, rw, 0.42, { fontSize: 20, bold: win });
    };
    bar(y + 1.45, [10, 10], `length 0.37 + 1 cut 0.80 = 1.17  ✓`, true);
    bar(y + 2.65, [6.67, 6.67, 6.66], `length 0.01 + 2 cuts 1.60 = 1.61`, false);
    L.text(s, "Over 15 s a cut is required; below that, “no cut” is also an option", rx, y + 3.85, rw, 1.0, { fontSize: 20 });
    L.notes(s, `Two worries about any automatic segmenter are that it chops slow speakers into fragments, and that it fails to cut fast speakers at all. This method avoids both.
    On the left, why it cannot over-fragment a slow speaker. Three things work together. First, the hard minimum of 2 seconds allows at most one piece per 2 seconds of audio, and in practice the length cost keeps pieces far above that. Second, the length cost prefers pieces of about 7 seconds, so a 40-second turn naturally wants five or six pieces, not twenty. Third, every extra cut has a price, so a cut is made only when it improves the lengths by more than it costs. Many pauses do not mean many cuts; they only mean more choice of where the five or six cuts go.
    On the right, why it still cuts a fast speaker. If a turn is longer than 15 seconds, no partition without a cut is allowed, so the programme has to cut somewhere, and it takes the best available gap, even a short one. Because poor cuts are expensive, it makes as few as the length cost justifies. The unit test case: a 20-second turn with almost no pauses, every gap scoring 0.2. In the idealised picture on the slide, two segments of 10 seconds cost 0.37 in length plus one poor cut of 0.8, 1.17 in total; three segments of 6.7 seconds cost almost nothing in length, 0.01, but need two poor cuts, 1.61 in total. One cut at 13 seconds would also be allowed now, but a 13-second piece costs 0.73 on its own, so that loses too. So the answer is two segments, cut near the middle, at about 10 seconds, and at a sentence end if there is one: with a sentence end at 9.1 seconds the cut moves there.
    Finally, an English turn between 10 and 15 seconds goes through the same programme, but "no cut" is one of its options: the whole turn then goes into final in one piece. It is cut only when a pause is worth its price; for example a 12-second turn with only poor pauses stays whole, while a 14-second turn gets cut even at a mediocre pause, because a 14-second piece is expensive.`);
    L.cite(s, "Method document §4.4 and §4.7; costs recomputed with the 2–15 s window (facts.md)");
  }

  // ---- 25 Routing to four folders
  {
    const s = L.slide("", "Routing: every turn ends in one of four folders", { stage: "Route" });
    const fin = (t) => ({ text: t, bold: true, fill: C.greenFill });
    const keep = (t) => ({ text: t, bold: true, fill: C.amberFill });
    L.table(s, ["Case", "Folder", "Then"], [
      ["English, ≤ 10 s", fin("final/"), "labelling"],
      ["English, > 10 s, after the dynamic programme", fin("final/"), "labelling"],
      ["Spanish or mixed, any length", keep("non_english/"), "kept"],
      ["Alignment weak or failed, or empty transcript", keep("alignment_issues/"), "kept"],
      ["No partition with every piece in 2–15 s", keep("boundary_issues/"), "kept"],
      ["Error in one turn", { text: "failures.csv", bold: true }, "fix and re-run"],
    ], { y: TOP, colW: [6.2, 3.2, 2.73], fontSize: 20, rowH: 0.6 });
    L.text(s, [{ text: "Not a folder: ", options: { bold: true } }, { text: "a segment with under 40% speech gets the flag low_speech_ratio; nothing is deleted" }], MX, 6.2, CW, 0.5, { fontSize: 20, valign: "middle" });
    L.notes(s, `Every turn ends in exactly one of four folders, and the manifest of each folder records a group and a reason for every row. This replaces the old review flags.
    final holds the clean results: English turns of 10 seconds or less, exported whole, and longer English turns cut by the dynamic programme, or kept in one piece when "no cut" was the cheapest choice. These go to Label Studio for labelling, and later to benchmarking and fine-tuning.
    non_english holds every Spanish or mixed turn, of any length, uncut, because we have no Spanish aligner yet.
    alignment_issues holds long English turns whose forced alignment failed or looked unreliable, or whose transcript was empty. We could cut the audio at VAD silences alone, but the text could not follow the cuts reliably, so we keep the whole turn.
    boundary_issues holds long English turns where no way of cutting keeps every piece between 2 and 15 seconds, for example because there is a long stretch with no usable word gap.
    The three issue folders are not labelled, but they are kept. Unlabelled audio is still useful later: it could be used for self-supervised continued pre-training, for example of an emotion2vec-style encoder, or of a Whisper encoder together with the transcripts. That is not implemented; it is only the reason the audio is kept.
    If anything throws an error for a turn, for example an unreadable file, the turn is skipped, any files it already wrote are removed, and it is listed in failures.csv with the error, so it can be fixed and re-run.
    Separately from the folders, a segment whose VAD speech time is under 40 percent of its duration, very slow speech with long silences, gets the flag low_speech_ratio for review. Nothing is joined or deleted.`);
  }
};

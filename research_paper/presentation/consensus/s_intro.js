// Title, problem statement, trade-off, definitions, vote rules, worked example, model groups
module.exports = function (L, N) {
  const { C, W, H, MX, CW } = L;
  const p = L.pptx;
  const S = N.stats, G = N.groups;
  const f1 = (x) => (x == null ? "—" : x.toFixed(1));
  const n0 = (x) => x.toLocaleString("en-US");
  const pend = N.pending.names.length > 0;
  const pendWho = N.pending.names.join(", ");
  const [r3, r4, r5] = N.crema;                       // CREMA-D, SER5: >=3/5, >=4/5, 5/5
  const meldCommon = N.meld.length ? N.meld[0].common : null;

  // ---- 1 Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.85, w: 0.09, h: 2.75, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Model Consensus as a Pseudo-Label Filter", MX + 0.4, 1.75, 11.8, 0.85, { fontSize: 40, bold: true });
    L.text(s, "When several emotion models agree on a clip, how often is the agreed label right,\nand how many clips get one?", MX + 0.4, 2.7, 11.8, 1.0, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  research_paper  ·  November 2026", MX + 0.4, 4.0, 11.8, 0.42, { fontSize: 20 });
    L.text(s, `${N.stats.datasets} public emotion datasets  ·  10 models  ·  ${N.runs} saved prediction runs, no model rerun`, MX, 5.55, CW, 0.4, { fontSize: 20 });
    if (pend) L.text(s, `Preliminary: ${pendWho} results pending on ${N.pending.datasets} of ${S.datasets} datasets`, MX, 6.05, CW, 0.4, { fontSize: 20, bold: true });
    L.notes(s, `This deck summarises the model-consensus analysis of our speech emotion recognition benchmark. The question is simple: if several of the ten benchmarked models give the same emotion label for a clip, how often is that agreed label correct, and for how many clips do they agree at all? The answer tells us whether agreement can select reliable pseudo-labels, so that people only label the clips where the models disagree.
    Everything here is computed from predictions that were already saved for ${S.datasets} public datasets; no model or API was run again for this analysis. ${pend ? `The results are preliminary: ${pendWho} has current results on MELD only, and its other ${N.pending.datasets} datasets are being re-run on another machine. Three of the four groups include it, so their numbers will change; the whole deck is rebuilt with one command when the results arrive.` : "All model results are complete."}`);
  }

  // ---- 2 Problem
  {
    const s = L.slide("Problem", "Hand labels do not scale; one model alone is often wrong");
    const y = 1.85, h = 1.0;
    L.box(s, MX, y, 2.45, h, "Unlabelled\nclips", { fill: C.white, line: C.ink, fontSize: 20 });
    L.arrow(s, MX + 2.45, y + h / 2, 3.55, y + h / 2);
    L.box(s, 3.55, y, 2.75, h, "10 emotion models\npredict a label", { fill: C.white, line: C.ink, fontSize: 20 });
    L.arrow(s, 6.3, y + h / 2, 6.95, y + h / 2);
    s.addShape(p.ShapeType.diamond, { x: 6.95, y: y - 0.2, w: 2.6, h: h + 0.4, fill: { color: C.amberFill }, line: { color: C.warn, width: 1.5 } });
    L.text(s, "enough models\nagree?", 7.25, y + 0.12, 2.0, 0.76, { fontSize: 20, bold: true, align: "center", valign: "middle" });
    L.arrow(s, 9.55, y + h / 2, 10.1, y + h / 2);
    L.text(s, "yes", 9.55, y + h / 2 - 0.42, 0.55, 0.34, { fontSize: 18, align: "center", kind: "diagram" });
    L.box(s, 10.1, y, W - MX - 10.1, h, "agreed label =\npseudo-label", { fill: C.greenFill, line: C.good, fontSize: 20, bold: true });
    L.arrow(s, 8.25, y + h + 0.2, 8.25, 3.6);
    L.text(s, "no", 8.35, 3.08, 0.5, 0.34, { fontSize: 18, kind: "diagram" });
    L.box(s, 6.95, 3.6, 2.6, 0.8, "human listeners", { fill: C.panel, line: C.ink, fontSize: 20, bold: true });
    const items = [
      S.meld_single ? { runs: [{ text: "One model alone: ", options: { bold: true } }, { text: `${f1(S.meld_single[0])}–${f1(S.meld_single[1])}% accuracy on MELD (10 models, same ${n0(meldCommon)} clips)` }] } : "One model alone is often wrong",
      { runs: [{ text: "Idea: ", options: { bold: true } }, { text: "accept a label only when several models agree; people label the rest" }] },
      { runs: [{ text: "Question: ", options: { bold: true } }, { text: "how accurate are agreed labels, and how many clips get one?" }] },
    ];
    L.bullets(s, items, { y: 4.75, h: 2.1, fontSize: 22 });
    L.notes(s, `Emotion labels from human listeners are slow and expensive, and we need many of them. Models can label everything, but a single model is often wrong: on MELD, the ten models score between ${f1(S.meld_single?.[0])} and ${f1(S.meld_single?.[1])} percent accuracy on exactly the same ${meldCommon ? n0(meldCommon) : ""} clips (lowest ${S.meld_single_names?.[0]}, highest ${S.meld_single_names?.[1]}).
    The idea of this analysis is a filter. All models predict a label for every clip. If enough of them agree, we accept their agreed label as a pseudo-label. If they do not agree, the clip goes to human listeners. How many models must agree is the vote rule.
    So two questions decide whether this is useful: how accurate are the accepted labels, and what share of the clips gets one. The rest of the deck measures both on the public benchmark.`);
  }

  // ---- 3 Trade-off
  {
    const s = L.slide("Problem", "Stricter agreement: more accurate labels, but fewer of them");
    L.image(s, "fig/concept_crema_d.png", 0.35, 1.6, 6.7, 5.0, { align: "left", valign: "top", alt: "Accuracy versus coverage of three vote rules for the five SER models on CREMA-D" });
    L.table(s, ["Five SER models", "Clips labelled", "Correct"], N.crema.map((r) => [
      { text: r.rule, bold: true, align: "center" }, { text: `${f1(r.coverage)}%`, align: "right" }, { text: `${f1(r.accuracy)}%`, align: "right" },
    ]), { x: 7.35, y: 1.8, w: 5.38, colW: [2.1, 1.75, 1.53], rowH: 0.5, fontSize: 20 });
    L.bullets(s, [
      `CREMA-D: the same ${n0(r3.common)} clips under every rule`,
      "Pseudo-labels need both numbers: coverage and accuracy",
      "Goal: label the most clips at a target accuracy",
    ], { x: 7.35, y: 4.05, w: 5.38, h: 2.8, fontSize: 22 });
    L.notes(s, `Here is the trade-off on one real example: the five SER models on CREMA-D. Each point is a vote rule. At least three of five agreeing labels ${f1(r3.coverage)} percent of the clips, and ${f1(r3.accuracy)} percent of those labels are correct. Requiring at least four labels ${f1(r4.coverage)} percent at ${f1(r4.accuracy)} percent. Requiring all five labels only ${f1(r5.coverage)} percent, but ${f1(r5.accuracy)} percent of them are correct.
    So a stricter rule moves left and up: fewer clips, more accurate labels. Neither number alone is enough. A rule that labels almost nothing can be very accurate and useless; a rule that labels everything is just a weak classifier.
    For pseudo-labels the practical goal is the rule that labels the most clips while staying above the accuracy we can tolerate. In this deck the x-axis is always coverage, the share of clips that get an agreed label, and the y-axis is always the accuracy of those agreed labels.`);
  }

  // ---- 4 Definitions: two numbers
  {
    const s = L.slide("Definitions", "Two numbers per vote rule: coverage and accuracy of agreed labels", { titleSize: 28 });
    const y = 2.1;
    L.box(s, MX, y, 3.2, 1.2, "Common cohort\n1,000 clips predicted\nby every member", { fill: C.white, line: C.ink, fontSize: 20 });
    L.arrow(s, MX + 3.2, y + 0.6, 4.4, y + 0.6);
    L.box(s, 4.4, y + 0.1, 2.0, 1.0, "vote rule\n(e.g. ≥4/5)", { fill: C.amberFill, line: C.warn, fontSize: 20, bold: true });
    L.arrow(s, 6.4, y + 0.6, 7.0, 2.2);
    L.arrow(s, 6.4, y + 0.6, 7.0, 3.7);
    L.box(s, 7.0, 1.75, 3.0, 0.9, "accepted: 200 clips\nget an agreed label", { fill: C.greenFill, line: C.good, fontSize: 20 });
    L.box(s, 7.0, 3.25, 3.0, 0.9, "rejected: 800 clips\n→ human listeners", { fill: C.panel, line: C.ink, fontSize: 20 });
    L.arrow(s, 10.0, 2.2, 10.45, 1.9);
    L.arrow(s, 10.0, 2.2, 10.45, 2.55);
    L.box(s, 10.45, 1.6, 2.28, 0.55, "194 correct", { fill: C.white, line: C.good, fontSize: 20, bold: true });
    L.box(s, 10.45, 2.3, 2.28, 0.55, "6 wrong", { fill: C.white, line: C.bad, fontSize: 20, bold: true });
    const fy = 4.6, fw = (CW - 0.3) / 2;
    L.box(s, MX, fy, fw, 1.35, [
      { text: "Coverage", options: { bold: true, breakLine: true } },
      { text: "accepted / common clips = 200 / 1,000 = 20%" }], { fill: C.blueFill, line: C.accent, fontSize: 22 });
    L.box(s, MX + fw + 0.3, fy, fw, 1.35, [
      { text: "Accuracy of agreed labels", options: { bold: true, breakLine: true } },
      { text: "correct / accepted = 194 / 200 = 97%" }], { fill: C.blueFill, line: C.accent, fontSize: 22 });
    L.text(s, "Illustration with round numbers; every result below uses the real counts.", MX, 6.25, CW, 0.4, { fontSize: 20 });
    L.notes(s, `Every result in this deck is a pair of numbers for one model group and one vote rule. The illustration uses round numbers from the reading guide of the report.
    We start from the common cohort: the clips that every member of the group has a prediction for, here 1,000 clips. Within one comparison the cohort never changes, so different vote rules share the same denominator.
    The vote rule accepts some clips, here 200, which get the agreed label, and rejects the other 800, which would go to human listeners. Of the 200 agreed labels, 194 match the dataset's reference label and 6 do not.
    Coverage is accepted divided by common clips, 20 percent. Accuracy of agreed labels is correct divided by accepted, 97 percent. Note that this accuracy is only about the accepted clips: it is not the accuracy on the whole dataset, and it is not a model's own confidence. When a rule accepts nothing, coverage is zero and accuracy is undefined; we never show that as zero percent accuracy.`);
  }

  // ---- 5 Vote rules
  {
    const s = L.slide("Definitions", "Vote rules: how many of the m models must agree");
    L.table(s, ["Rule", "Accept the top label if", "m = 5", "m = 10"], [
      [{ text: "Strict majority", bold: true }, "more than half of the m models vote for it", { text: "≥3/5", align: "center" }, { text: "≥6/10", align: "center" }],
      [{ text: "k of m", bold: true }, "at least k models vote for it", { text: "≥4/5", align: "center" }, { text: "≥8/10", align: "center" }],
      [{ text: "Unanimity", bold: true }, "all m models vote for it", { text: "5/5", align: "center" }, { text: "10/10", align: "center" }],
    ], { y: 1.7, colW: [2.5, 6.03, 1.7, 1.9], rowH: 0.5, fontSize: 20 });
    const ok = (t) => ({ text: t, align: "center", bold: t === "yes", fill: t === "yes" ? C.greenFill : C.white });
    L.table(s, ["Clip", "Votes of the five SER models", "Top label", "≥3/5", "≥4/5", "5/5"], [
      [{ text: "A", bold: true, align: "center" }, "angry × 5", "angry (5 votes)", ok("yes"), ok("yes"), ok("yes")],
      [{ text: "B", bold: true, align: "center" }, "sad × 4, neutral", "sad (4)", ok("yes"), ok("yes"), ok("no")],
      [{ text: "C", bold: true, align: "center" }, "happy × 3, invalid answer × 2", "happy (3 of 5)", ok("yes"), ok("no"), ok("no")],
      [{ text: "D", bold: true, align: "center" }, "angry × 2, sad × 2, neutral", "tie", ok("no"), ok("no"), ok("no")],
    ], { y: 3.85, colW: [0.9, 4.73, 2.6, 1.3, 1.3, 1.3], rowH: 0.46, fontSize: 20, zebra: false });
    L.text(s, [{ text: "Invalid answers abstain but still count in m (C)  ·  ties are rejected (D)  ·  ", options: {} },
      { text: "clips A–D: illustration", options: { italic: true } }], MX, 6.3, CW - 1.0, 0.45, { fontSize: 20 });
    L.notes(s, `A vote rule says how many of the m models in a group must vote for the most common label. We use three kinds. Strict majority means more than half: at least three of five, or at least six of ten. k of m means at least k votes, for every k from the strict majority up to m. Unanimity means all m models.
    The four clips are an illustration with five SER models. Clip A has five votes for angry, so every rule accepts it. Clip B has four votes for sad, so at least three and at least four accept it, but unanimity does not. In clip C, two models gave an invalid answer, for example a word outside the label set. Invalid answers abstain: they do not vote for any label, but they still count in m, so happy has three of five votes and only the majority rule accepts it. Clip D is a tie between angry and sad, which every rule rejects.
    Because the denominator m is fixed, a stricter rule always accepts a subset of the clips of a looser rule: unanimity is inside at least four, which is inside at least three. Coverage can only go down when the rule gets stricter; accuracy usually goes up, but that is not guaranteed.`);
  }

  // ---- 6 Worked example
  {
    const s = L.slide("Definitions", "Worked example: five SER models on CREMA-D");
    L.table(s, ["Rule", "Clips accepted", "Correct", "Wrong", "Coverage", "Accuracy"], N.crema.map((r) => [
      { text: r.rule, bold: true, align: "center" }, { text: n0(r.accepted), align: "right" }, { text: n0(r.correct), align: "right" },
      { text: n0(r.errors), align: "right" }, { text: `${r.coverage.toFixed(2)}%`, align: "right" }, { text: `${r.accuracy.toFixed(2)}%`, align: "right" },
    ]), { y: 1.75, colW: [1.6, 2.4, 1.9, 1.7, 2.2, 2.33], rowH: 0.58, fontSize: 22 });
    L.bullets(s, [
      `Always the same ${n0(r3.common)} clips: only the vote threshold changes`,
      `From ≥3/5 to 5/5: wrong labels fall from ${n0(r3.errors)} to ${n0(r5.errors)}, but ${n0(r3.correct - r5.correct)} correct labels are lost too`,
      "Each stricter rule keeps a subset of the looser rule's clips",
    ], { y: 4.35, h: 2.4, fontSize: 22 });
    L.notes(s, `This is the worked example from the report, with exact counts. The five SER models predict all ${n0(r3.common)} CREMA-D clips. At least three of five agree on ${n0(r3.accepted)} clips: ${n0(r3.correct)} correct and ${n0(r3.errors)} wrong, so coverage is ${r3.coverage.toFixed(2)} percent and accuracy ${r3.accuracy.toFixed(2)} percent. At least four of five accept ${n0(r4.accepted)} clips with ${n0(r4.errors)} errors. All five agree on ${n0(r5.accepted)} clips with only ${n0(r5.errors)} errors: ${r5.accuracy.toFixed(2)} percent accuracy at ${r5.coverage.toFixed(2)} percent coverage.
    Going from the loosest to the strictest rule removes most errors, but it also throws away ${n0(r3.correct - r5.correct)} correct labels, which then need human listeners. That is the price of accuracy.
    These three rows are the three blue points in the CREMA-D panel of the results figure. The same counts are in research_paper/output/consensus_thresholds.csv.`);
  }

  // ---- 7 Model groups
  {
    const s = L.slide("Model groups", "Four fixed model groups, one colour each");
    const status = (g) => (G[g].partial ? `${G[g].missing.join(", ")}: MELD only*` : `all ${S.datasets} datasets`);
    const members = {
      SER5: "emotion2vec+ large, MERaLiON-SER-v1, Melati,\nVox-Profile Whisper, Vox-Profile WavLM",
      LLM3: "Qwen2.5-Omni 7B, Qwen2-Audio 7B Instruct,\nGemma 4 E4B (open audio LLMs)",
      AudioLLM5: "LLM3 + GPT Realtime + Gemini 3.8 Flash",
      All10: "SER5 + AudioLLM5 (all ten models)",
    };
    const rowH = [0.52, 0.86, 0.86, 0.6, 0.6], y0 = 1.75;
    L.table(s, ["Group", "Members", "m", "Results"], N.main.map((g) => [
      { text: G[g].label + (G[g].partial ? "*" : ""), bold: true, options: { margin: [3, 7, 3, 40] } }, members[g],
      { text: String(G[g].size), align: "center" }, status(g),
    ]), { y: y0, colW: [2.5, 5.95, 0.75, 2.93], rowH, fontSize: 20, zebra: false });
    let y = y0 + rowH[0];
    N.main.forEach((g, i) => { L.marker(s, G[g], MX + 0.14, y + rowH[i + 1] / 2 - 0.15, 0.3); y += rowH[i + 1]; });
    L.text(s, pend
      ? `* ${pendWho} has current results on MELD only. On the other ${N.pending.datasets} datasets these groups vote with the available members (2/2, 4/4, 9/9): preliminary.`
      : "All members have results on every dataset.", MX, 5.45, CW, 0.75, { fontSize: 20 });
    L.text(s, "Groups are fixed by model family before looking at accuracy. Appendix: Selected3, SER+API7, AudioLLM4.", MX, 6.25, CW, 0.45, { fontSize: 20 });
    L.notes(s, `We compare four groups, fixed by model family before looking at any accuracy, so we are not searching the test labels for the best combination. Each group has one colour and one marker shape in every figure and table: blue circles, orange triangles, green diamonds and purple squares. The colours are the Okabe and Ito palette, which stays distinguishable with colour-vision deficiencies.
    SER5 is the five dedicated speech emotion models. LLM3 is the three open audio language models we run locally. AudioLLM5 adds the two commercial APIs, GPT Realtime and Gemini 3.8 Flash. All10 is everything.
    ${pend ? `The asterisk matters: ${pendWho} has results under the current prompt protocol only on MELD; its other ${N.pending.datasets} datasets are being re-run. Until then, LLM3, AudioLLM5 and All10 vote with the members that are available, as the analysis code has always done: LLM3 becomes two of two, AudioLLM5 four of four and All10 nine of nine. Figures mark these cases with hollow markers and dashed lines, and tables with an asterisk. On MELD all four groups are complete and use exactly the same clips, which is why MELD gets its own slide.` : "All four groups are complete on every dataset."}
    Three older groups, Selected3, SER plus APIs, and AudioLLM4, are kept in the appendix so earlier results stay comparable.`);
  }
};

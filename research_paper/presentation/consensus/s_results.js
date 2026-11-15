// Results, summary, caveats, next steps and the hidden appendix
module.exports = function (L, N) {
  const { C, MX, CW } = L;
  const S = N.stats, G = N.groups;
  const f0 = (x) => (x == null ? "—" : String(Math.round(x)));
  const f1 = (x) => (x == null ? "—" : x.toFixed(1));
  const n0 = (x) => x.toLocaleString("en-US");
  const pend = N.pending.names.length > 0;
  const pendWho = N.pending.names.join(", ");
  const star = (g) => (G[g].partial ? "*" : "");
  const range = (r, d = 0) => (r[0] === r[1] ? `${r[0].toFixed(d)}` : `${r[0].toFixed(d)}–${r[1].toFixed(d)}`);
  const grid = (s, file, alt) => L.image(s, file, MX, 1.5, CW, 4.65, { valign: "top", alt });
  const twoBullets = (s, a, b) => {
    L.bullets(s, [a], { x: MX, y: 6.2, w: CW / 2 - 0.15, h: 0.75, fontSize: 20, gap: 0 });
    L.bullets(s, [b], { x: MX + CW / 2 + 0.15, y: 6.2, w: CW / 2 - 0.15, h: 0.75, fontSize: 20, gap: 0 });
  };
  const pendNote = pend ? ` Hollow markers and dashed lines mark the groups that vote without ${pendWho}; those lines will move when its results arrive.` : "";

  // ---- 8 Results: dialogue and acted English
  {
    const harder = S.ser5_dialogue_acc[1] < S.ser5_english_acted_acc[0];
    const s = L.slide("Results", harder ? "Agreement is much less reliable on dialogue than on acted English"
      : "Accuracy and coverage: dialogue and acted English corpora", { titleSize: 28 });
    grid(s, "fig/tradeoff_main_1.png", "Accuracy versus coverage of the four main groups on eight datasets");
    twoBullets(s,
      { runs: [{ text: "Dialogue: ", options: { bold: true } }, { text: `SER5 5/5 only ${range(S.ser5_dialogue_acc)}% accurate` }] },
      { runs: [{ text: "Acted English: ", options: { bold: true } }, { text: `SER5 5/5 ${range(S.ser5_english_acted_acc)}% at ${range(S.ser5_english_acted_cov)}% coverage` }] });
    L.notes(s, `Each panel is one dataset, in the benchmark's order: the four dialogue corpora in the top row and four acted English corpora below. Each line is one model group and each point on it one vote rule. The leftmost point of a line is unanimity, the rightmost strict majority. Up is better accuracy, right is more clips labelled. All panels share the same axes from 0 to 100 percent.${pendNote}
    In dialogue speech, agreement is much less reliable: even when all five SER models agree, only ${range(S.ser5_dialogue_acc)} percent of the agreed labels are correct, and on MELD the best group reaches ${f1(S.meld_best)} percent. On the acted English corpora, CREMA-D, RAVDESS, SAVEE and JL-Corpus, SER5 unanimity is ${range(S.ser5_english_acted_acc)} percent accurate while labelling ${range(S.ser5_english_acted_cov)} percent of the clips.
    Part of the acted-speech result may come from training exposure: several SER models were trained on some of these corpora, which the benchmark marks with hatching. Dialogue corpora also differ in language, label sets and recording, so the gap is not caused by context alone.`);
  }

  // ---- 9 Results: other corpora and languages
  {
    const s = L.slide("Results", "Other corpora and languages: the groups diverge");
    grid(s, "fig/tradeoff_main_2.png", "Accuracy versus coverage of the four main groups on the other eight datasets");
    twoBullets(s,
      { runs: [{ text: "SER5: ", options: { bold: true } }, { text: `unanimity ≥90% accurate on ${S.ser5_second_ge90} of 8` }] },
      { runs: [{ text: `AudioLLM5${star("AudioLLM5")}: `, options: { bold: true } }, { text: `unanimity <50% accurate on ${S.audio_second_below50} of 8` }] });
    L.notes(s, `The second half of the grid: TESS, MEAD and MESS in English, then MCAE-SPPS in Mandarin, THAI-SER, nEMO in Polish and the two Italian corpora.
    The SER5 line stays high: its unanimity is at least 90 percent accurate on ${S.ser5_second_ge90} of these 8 datasets. The audio-LLM groups behave very differently. AudioLLM5 unanimity is below 50 percent accurate on ${S.audio_second_below50} of the 8, and LLM3 on ${S.llm3_second_below50}. MEAD is the clearest case: ${f0(S.neutral.mead_AudioLLM5)} percent of AudioLLM5's unanimous labels are neutral, and only ${f0(S.mead_audio_acc)} percent of them are correct.${pendNote}
    All10's unanimity point is the most accurate of the four groups on ${S.all10_best} of ${S.datasets} datasets, ties included, but it sits close to zero coverage: requiring ${pend ? "all available" : "all ten"} models to agree leaves very few clips. The next slides look at that trade-off on matched clips.`);
  }

  // ---- 10 MELD, identical cohort
  {
    const best = N.meld.reduce((a, b) => (b.accuracy > a.accuracy ? b : a), N.meld[0]);
    const all10 = N.meld.find((m) => m.group === "All10");
    const covs = N.meld.map((m) => m.coverage);
    const title = N.meld_matched && all10 && best.group !== "All10"
      ? `MELD, same ${n0(N.meld[0].common)} clips: All10 labels ${f0(all10.coverage)}%, yet is not the most accurate`
      : `MELD: unanimity of the four groups on ${N.meld_matched ? "identical" : "each group's"} clips`;
    const s = L.slide("Results", title, { titleSize: 28 });
    L.image(s, "fig/meld_tradeoff.png", 0.35, 1.55, 6.8, 5.25, { align: "left", valign: "top", alt: "MELD accuracy versus coverage with every vote threshold labelled" });
    const mrow = [0.66, 0.46, 0.46, 0.46, 0.46];
    L.table(s, ["Group", "Rule", "Clips (wrong)", "Cov.", "Acc."], N.meld.map((m) => [
      { text: m.label, bold: true, options: { margin: [2, 4, 2, 29] } }, { text: m.rule, align: "center", options: { margin: [2, 3, 2, 3] } },
      { text: `${n0(m.accepted)} (${n0(m.errors)})`, align: "right" }, { text: `${f1(m.coverage)}%`, align: "right", options: { margin: [2, 4, 2, 2] } },
      { text: `${f1(m.accuracy)}%`, align: "right", options: { margin: [2, 4, 2, 2] } }]),
    { x: 7.3, y: 1.65, w: 5.43, colW: [1.68, 0.78, 1.29, 0.84, 0.84], rowH: mrow, fontSize: 18, zebra: false });
    N.meld.forEach((m, i) => L.marker(s, G[m.group], 7.3 + 0.07, 1.65 + mrow[0] + 0.46 * i + 0.11, 0.24));
    L.bullets(s, [
      `Unanimity labels ${f0(Math.min(...covs))}–${f0(Math.max(...covs))}% of the clips`,
      `Best agreed accuracy ${f1(best.accuracy)}% (${best.label} ${best.rule}); single models ${f1(S.meld_single[0])}–${f1(S.meld_single[1])}%`,
      all10 ? `All10 ${all10.rule}: only ${n0(all10.accepted)} clips, ${n0(all10.errors)} still wrong` : "All10: pending",
    ], { x: 7.3, y: 4.35, w: 5.43, h: 2.5, fontSize: 20, gap: 8 });
    L.notes(s, `MELD is ${pend ? "the one dataset" : "a worked example"} where all four groups are complete${N.meld_matched ? " and use exactly the same clips" : ""}, so here the comparison is fair. ${pend ? "" : "All sixteen datasets now have all ten model results. "}Every point is labelled with its vote rule; the box colour shows the group.
    The table lists the unanimity endpoints. SER5 five of five accepts ${n0(N.meld[0].accepted)} clips at ${f1(N.meld[0].accuracy)} percent accuracy. The best unanimous accuracy is ${best.label} ${best.rule} with ${f1(best.accuracy)} percent at ${f1(best.coverage)} percent coverage. ${all10 ? `All ten models agree on only ${n0(all10.accepted)} clips, ${f1(all10.coverage)} percent, and ${n0(all10.errors)} of them are still wrong.` : ""}
    For comparison, single models reach ${f1(S.meld_single[0])} to ${f1(S.meld_single[1])} percent on these clips, so agreement does raise accuracy substantially, but on dialogue speech it does not get anywhere near the accuracy we would want from labels used for training.
    Adding voters always shrinks coverage on the same clips, because ten-of-ten is a subset of five-of-five, but accuracy only rises if the new models reject wrong labels more often than correct ones. Shared training data and shared biases can prevent that. Points in the middle of a line, like six to nine of ten, are different rules from another group's unanimity and should not be compared as if they were the same.`);
  }

  // ---- 11 Unanimity heat map
  {
    const s = L.slide("Results", `Unanimity: SER5 ≥95% accurate on ${S.ser5_u95}/${S.datasets} datasets, but labels ≤${f0(S.ser5_cov[1])}% of clips`, { titleSize: 28 });
    L.image(s, "fig/unanimous_heatmap.png", MX, 1.5, CW, 5.05, { valign: "top", alt: "Heat map of unanimous accuracy and coverage per dataset and group" });
    L.text(s, (pend ? `* ${pendWho} pending: voted with available members (2/2, 4/4, 9/9)  ·  ` : "") + "— no clip accepted",
      MX, 6.62, CW - 1.0, 0.4, { fontSize: 20 });
    L.notes(s, `This heat map gives the exact unanimity numbers behind the figures, rounded to whole percent. The left block is the accuracy of agreed labels, the right block coverage. Darker cells are higher values; each block has its own scale.
    SER5 unanimity is at least 95 percent accurate on ${S.ser5_u95} of ${S.datasets} datasets: ${S.ser5_u95_names.join(", ")}. But it labels only ${f1(S.ser5_cov[0])} to ${f1(S.ser5_cov[1])} percent of the clips. All10 unanimity is at least 90 percent accurate on ${S.all10_acc_ge90} of the ${S.all10_acc_defined} datasets where it accepts anything, but its coverage is at most ${f1(S.all10_cov_max)} percent, median ${f1(S.all10_cov_median)} percent${S.all10_zero.length ? `, and on ${S.all10_zero.join(", ")} it accepts nothing, shown as a dash` : ""}.
    ${pend ? `Values with an asterisk vote without ${pendWho}. ` : ""}Exact values with two decimals are in the report table "Main groups: unanimous coverage and accuracy" and in output/consensus_thresholds.csv.`);
  }

  // ---- 12 Summary
  {
    const s = L.slide("Summary", "What agreement can and cannot do for pseudo-labels");
    const w = (CW - 3 * 0.3) / 4;
    const tiles = [
      [`${S.ser5_u95}/${S.datasets}`, "datasets where SER5 unanimity is ≥95% accurate"],
      [`${range(S.ser5_cov)}%`, "of clips get a SER5 unanimous label"],
      [`${range(S.ser5_dialogue_acc)}%`, "SER5 unanimous accuracy on dialogue corpora"],
      [`≤${f0(S.all10_cov_max)}%`, `All10${star("All10")} unanimous coverage (median ${f0(S.all10_cov_median)}%)`],
    ];
    tiles.forEach(([num, cap], i) => {
      const x = MX + i * (w + 0.3);
      s.addShape(L.pptx.ShapeType.rect, { x, y: 1.7, w, h: 2.25, fill: { color: C.panel }, line: { color: C.line, width: 0.75 } });
      L.stat(s, num, cap, x + 0.2, 1.75, w - 0.4, { size: 40, capH: 1.1 });
    });
    L.bullets(s, [
      "Agreement is a useful filter on acted speech, much weaker on conversations",
      "More voters cut coverage sharply; higher accuracy is not guaranteed",
      `Agreed labels skew to neutral (CREMA-D: ${f0(S.neutral.crema_d_SER5)}% of SER5, ${f0(S.neutral.crema_d_All10)}% of All10${star("All10")} labels)`,
    ], { y: 4.3, h: 2.5, fontSize: 22 });
    L.notes(s, `Four numbers summarise the analysis. SER5 unanimity is at least 95 percent accurate on ${S.ser5_u95} of ${S.datasets} datasets, so agreement can select very reliable labels on some corpora. But it labels only ${range(S.ser5_cov)} percent of the clips, so most clips would still need people. On dialogue corpora even SER5 unanimity reaches only ${range(S.ser5_dialogue_acc)} percent. And All10 unanimity labels at most ${f1(S.all10_cov_max)} percent of any dataset.
    So agreement is a useful filter for acted speech and a weak one for conversations, which is the kind of speech our target calls contain. Adding voters cuts coverage sharply, while the accuracy gain depends on whether the new models make different mistakes.
    The agreed labels are also not balanced: on CREMA-D ${f0(S.neutral.crema_d_SER5)} percent of the SER5 unanimous labels and ${f0(S.neutral.crema_d_All10)} percent of the All10 ones are neutral, because models agree most easily on neutral speech. A pseudo-label set built this way would under-represent the emotions we care about most, so class balance has to be checked; the appendix shows the neutral share for every dataset.`);
  }

  // ---- 13 Caveats
  {
    const s = L.slide("Caveats", "Limits of public-data agreement as evidence");
    const rows = [];
    if (pend) rows.push([{ text: `${pendWho} pending`, bold: true }, `${N.pending.datasets}/${S.datasets} datasets: LLM3, AudioLLM5 and All10 vote without it (*). Rebuild with one command when it arrives.`]);
    rows.push(
      [{ text: "Public data ≠ target calls", bold: true }, "Acted, TV and podcast speech, other languages and label sets. Calibrate the group and rule on a small labelled call sample."],
      [{ text: "Rules not tuned", bold: true }, "Every threshold is reported; none was chosen on these test labels (that would overfit)."],
      [{ text: "Common clips", bold: true }, pend ? "Missing members change m (k/m). Compare groups only on identical clips, as on MELD." : "All ten models cover all sixteen datasets. Compare groups on the same common clips and keep their vote rules explicit."],
      [{ text: "Class skew", bold: true }, `Agreed labels are mostly neutral on some corpora (MultiDialog SER5: ${f0(S.neutral.multidialog_SER5)}%).`],
      [{ text: "No uncertainty", bold: true }, "Point estimates only; votes share training data and errors, so they are not independent evidence."],
    );
    L.table(s, ["Caveat", "Consequence and action"], rows, { y: 1.7, colW: [3.3, 8.83], rowH: 0.72, fontSize: 20 });
    L.notes(s, `Please read these results with six caveats.
    ${pend ? `First, ${pendWho} is pending on ${N.pending.datasets} of ${S.datasets} datasets, so three of the four groups are computed without it there. The numbers for LLM3, AudioLLM5 and All10 are preliminary. When the re-run finishes, one command regenerates the analysis, the paper figures, the reports and this deck.` : ""}
    Second, these are public research corpora: mostly acted speech, TV series and podcasts, in several languages and with their own label sets. They are not our target calls. Before using agreement for pseudo-labels, the group and the vote rule have to be calibrated on a small sample of calls labelled by our listeners.
    Third, we report every vote threshold and did not pick the best one on these test labels; picking it here would overfit to these corpora.
    Fourth, ${pend ? "when members are missing, m changes, so groups are directly comparable only on identical clips, which is the case on MELD" : "all ten models now cover every dataset, but comparisons still require identical clips and clearly stated vote rules"}.
    Fifth, the agreed labels are dominated by neutral on several corpora, so a pseudo-label set would be unbalanced.
    Sixth, these are point estimates without confidence intervals, and the models share training data and errors, so more votes are not independent evidence.`);
  }

  // ---- 14 Next steps
  {
    const s = L.slide("Next steps", "How to use agreement on target calls");
    const steps = ["1 · Our listeners label a small call sample", "2 · Set the error rate we can tolerate", "3 · Run the fixed groups and all vote rules",
      "4 · Pick the rule with most coverage at that accuracy", "5 · Test on the other half by class; audit accepted labels"];
    const w = 2.2, gap = (CW - 5 * w) / 4, y = 1.95, h = 1.75;
    steps.forEach((t, i) => {
      const x = MX + i * (w + gap);
      L.box(s, x, y, w, h, t, { fill: i === 3 ? C.greenFill : C.white, line: i === 3 ? C.good : C.ink, fontSize: 20 });
      if (i < 4) L.arrow(s, x + w, y + h / 2, x + w + gap, y + h / 2);
    });
    L.bullets(s, [
      "Clips without enough agreement go to human listeners",
      "Steps 3–4 use a calibration half of the sample, step 5 the other half",
      "Repeat when a model, prompt or the call mix changes",
    ], { y: 4.15, h: 2.6, fontSize: 22 });
    L.notes(s, `If we want to use agreement for pseudo-labels on our calls, this is the procedure. First, our listeners label a small sample of calls; this is the only way to know the accuracy on the target audio. Second, we decide how many wrong pseudo-labels we can tolerate. Third, we run the fixed groups from this analysis with every vote rule on the sample. Fourth, on one half of the sample, we pick the rule that labels the most clips while staying at or above the required accuracy. Fifth, we check that rule on the other half, separately for each emotion, because the agreed labels can be dominated by neutral, and we keep auditing a share of the accepted labels.
    Clips without enough agreement go to the listeners as before. The procedure has to be repeated whenever a model, a prompt or the mix of calls changes. The public-data results in this deck tell us which groups and rules are worth testing, not which accuracy we will get on calls.`);
  }

  // ---- A1, A2 Appendix groups
  [1, 2].forEach((half) => {
    const s = L.slide("Appendix", `Appendix groups: Selected3, SER+API7, AudioLLM4 (${half}/2)`, { appendix: true });
    grid(s, `fig/tradeoff_extra_${half}.png`, `Accuracy versus coverage of the appendix groups, part ${half}`);
    const done = N.extra.every((g) => !G[g].partial && G[g].complete === S.datasets);
    L.text(s, half === 1
      ? "Selected3: emotion2vec+, Gemini, Qwen2.5-Omni  ·  SER+API7: SER5 + GPT Realtime + Gemini  ·  AudioLLM4: GPT Realtime, Gemini, Qwen2.5-Omni, Qwen2-Audio"
      : `${done ? `All three are complete on ${S.datasets}/${S.datasets} datasets` : "Groups wait for every member"}; SER+API7 and AudioLLM4 wait for missing members instead of voting without them.`,
    MX, 6.2, CW, 0.75, { fontSize: 20 });
    L.notes(s, `These are the three older groups, kept so that earlier results stay comparable. Selected3 is a legacy name for emotion2vec+, Gemini and Qwen2.5-Omni; the name does not mean it was selected as the best. SER plus API7 adds GPT Realtime and Gemini to the five SER models. AudioLLM4 is the four audio language models without Gemma 4.
    ${half === 1 ? "The panels and axes are the same as in the main grid." : "SER plus API7 and AudioLLM4 never vote with missing members: they wait until every member has results, and they are complete on all datasets."} ${Object.entries(S.extra_zero).filter(([, v]) => v.length).map(([g, v]) => `${G[g].label} unanimity accepts no clip on ${v.join(', ')}, so its line has no unanimity point there.`).join(' ')} The same numbers are in output/consensus_thresholds.csv.`);
  });

  // ---- A3 Neutral share
  {
    const s = L.slide("Appendix", "Agreed labels are often neutral", { appendix: true });
    L.image(s, "fig/neutral_share_heatmap.png", 0.35, 1.5, 7.5, 5.1, { align: "left", valign: "top", alt: "Share of unanimous labels that are neutral, per dataset and group" });
    L.bullets(s, [
      `CREMA-D: ${f0(S.neutral.crema_d_SER5)}% of SER5 and ${f0(S.neutral.crema_d_All10)}% of All10${star("All10")} unanimous labels are neutral`,
      `MultiDialog: ${f0(S.neutral.multidialog_SER5)}% (SER5)`,
      "Check class balance before training on pseudo-labels",
    ], { x: 8.1, y: 1.8, w: 4.63, h: 4.9, fontSize: 22 });
    L.notes(s, `This appendix slide shows, for each dataset and group, which share of the unanimous labels is neutral. Models agree most easily on neutral speech, so the agreed labels over-represent it: on CREMA-D, ${f0(S.neutral.crema_d_SER5)} percent of SER5 unanimous labels and ${f0(S.neutral.crema_d_All10)} percent of All10 ones are neutral; on MultiDialog ${f0(S.neutral.multidialog_SER5)} percent of SER5's. MELD is an exception for SER5 because its agreed labels are mostly happy.
    Dashes mean no unanimous label at all, n/a means the dataset has no neutral class or the group has no result. The per-class counts and per-class precision for every threshold are in output/consensus_by_class.csv.`);
  }

  // ---- A4 Rebuild
  {
    const s = L.slide("Appendix", "Rebuild everything with one command", { appendix: true });
    L.box(s, MX, 1.75, CW, 0.75, "cd research_paper/presentation/consensus && ./rebuild.sh", { mono: true, fill: C.panel, line: C.ink, fontSize: 22, align: "left", margin: [3, 14, 3, 14] });
    L.table(s, ["Step", "Writes"], [
      [{ text: "1 · scripts/consensus.py", bold: true }, "output/consensus*.csv, manifest, both reports, the four paper figures"],
      [{ text: "2 · scripts/make_assets.py", bold: true }, "slide figures (fig/), assets/numbers.json, facts.md"],
      [{ text: "3 · node build.js", bold: true }, "consensus_analysis.pptx (with the font-size check)"],
      [{ text: "./render.sh (optional)", bold: true }, "render/: PDF and PNG previews (LibreOffice)"],
    ], { y: 2.85, colW: [3.6, 8.53], rowH: 0.55, fontSize: 20 });
    L.text(s, "Reads saved predictions only: no model, audio or API is run. New results (e.g. Gemma 4) are picked up automatically.", MX, 5.75, CW, 0.75, { fontSize: 20 });
    L.notes(s, `When saved predictions change, run rebuild.sh from this folder in the emotion_recognition_llm environment. It runs the consensus analysis, which rereads all saved current-protocol predictions and rewrites the CSVs, the manifest, both language versions of the report and the four paper figures. Then it redraws the slide figures, recomputes every number on the slides and facts.md, and builds this deck. ${pend ? "Groups with missing members fill in automatically once every member is present." : "The current build includes every member of each group on all sixteen datasets."}
    Nothing in this chain runs a model, an API or reads audio. render.sh is only for checking the slides as images with LibreOffice.`);
  }
};

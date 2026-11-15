// Title, problem statement, how each factor is tested, the datasets, and the clip durations
module.exports = function (L, H) {
  const { C, W, H: SH, MX, CW, TOP } = L;
  const { D, F, V, list, claim, gemmaPending } = H;
  const p = L.pptx;
  const gemmaNote = gemmaPending ? `Gemma 4 E4B has current-protocol results on ${F("gemma_datasets")} only, so every Gemma result is preliminary.` : "All ten models cover all sixteen datasets.";

  // ---- 1 Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.75, w: 0.09, h: 2.75, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Why Do SER Models Win on Some Datasets\nand Lose on Others?", MX + 0.4, 1.65, 11.8, 1.45, { fontSize: 40, bold: true, ls: 0.95 });
    L.text(s, `Language, conversational versus acted speech, and clip duration\nacross ${F("n_datasets")} public corpora and ${F("n_models")} models`, MX + 0.4, 3.2, 11.8, 0.95, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  research_paper  ·  November 2026", MX + 0.4, 4.3, 11.8, 0.42, { fontSize: 20 });
    L.text(s, `Saved benchmark predictions only (${F("runs")} runs, ${F("clips")} clips); no new model runs.${gemmaPending ? ` Gemma 4 E4B: ${F("gemma_datasets")} only, preliminary.` : ""}`,
      MX, SH - 1.0, CW, 0.45, { fontSize: 18, kind: "diagram" });
    L.notes(s, `This talk asks why the same emotion recognition model can look excellent on one benchmark dataset and poor on another. We look at three properties of the datasets that could explain it: the language, whether the speech comes from conversations or from acted single sentences, and how long the audio clips are.
    Everything here comes from the predictions we already saved for the benchmark: ${F("runs")} model–dataset runs over ${F("clips")} clips from ${F("n_datasets")} public corpora. No model was rerun and no API was called for this analysis.
    ${gemmaNote}${gemmaPending ? " Its numbers are marked as preliminary, and the whole deck regenerates with one command when its other datasets arrive." : " All multi-dataset comparisons include Gemma 4 E4B."}`);
  }

  // ---- 2 Problem: accuracy spread per model
  {
    const s = L.slide("Problem", `Each model's accuracy swings by ${F("spread_min")}–${F("spread_max")} percentage points across the ${F("n_datasets")} datasets`);
    const rows = D.spread.map((r) => r.n === V("n_datasets")
      ? [{ text: r.name, bold: true }, `${r.min.toFixed(1)}%  (${r.min_ds})`, `${r.max.toFixed(1)}%  (${r.max_ds})`, { text: `${r.range.toFixed(1)} pp`, bold: true, align: "right" }]
      : [{ text: r.name, bold: true }, `${r.min.toFixed(1)}%  (${r.min_ds})`, `${r.n} of ${V("n_datasets")} datasets so far`, { text: "preliminary", bold: true, color: C.darkOrange, align: "right" }]);
    L.table(s, ["Model", "Lowest accuracy (dataset)", "Highest accuracy (dataset)", "Range"], rows,
      { y: TOP - 0.02, colW: [3.5, 3.55, 3.55, 1.53], fontSize: 18, rowH: 0.385 });
    L.text(s, `${F("mead_lowest")} of ${F("n_full")} complete models score lowest on MEAD; the best dataset differs from model to model.`, MX, 6.35, CW, 0.45, { fontSize: 20, valign: "middle" });
    claim(V("mead_lowest") >= V("n_full") / 2, "slide 2: MEAD is no longer the lowest dataset for most models");
    L.notes(s, `Start with the problem. The table shows, for every model, its worst and its best dataset by accuracy. Every complete model moves by ${F("spread_min")} to ${F("spread_max")} percentage points between its worst and its best corpus. So the choice of benchmark dataset changes the picture of a model as much as the choice of model does.
    ${F("mead_lowest")} of the ${F("n_full")} complete models are worst on MEAD, while their best datasets differ: ${D.spread.filter((r) => r.n === V("n_datasets")).map((r) => `${r.short} on ${r.max_ds}`).join(", ")}.
    ${gemmaPending ? `Gemma 4 E4B is in the table with its ${F("gemma_datasets")} result only (${F("gemma_first_accuracy")}% accuracy); its spread cannot be judged yet.` : ""}
    Accuracy here is the share of clips whose predicted emotion matches the label. The next slides ask which dataset properties go together with these swings.`);
    L.source(s, "Source: research_paper/output/cross_dataset_model_rows.csv (see facts.md)");
  }

  // ---- 3 How we test the three factors
  {
    const s = L.slide("Problem", "Three candidate explanations, and how we test each one");
    L.table(s, ["Factor", "Comparison", "How we test it", "What can mislead"], [
      [{ text: "Language", bold: true }, `English (${F("n_english")}) vs non-English (${F("n_non_english")}) corpora`, "mean score per group; repeat within acted corpora only", "corpora differ in more than language"],
      [{ text: "Setting", bold: true }, `conversational (${F("n_conv")}) vs acted (${F("n_acted")}) corpora`, "mean score per group; repeat without corpora seen in training", "“neutral” dominates dialogue labels"],
      [{ text: "Clip duration", bold: true }, "length of every clip, from the audio file", "accuracy change per doubling of length, within each dataset and emotion", "correlation, not cause; inputs capped at 15–30\u00a0s"],
    ], { y: TOP, colW: [2.0, 3.05, 3.75, 3.33], fontSize: 18, rowH: [0.5, 0.95, 0.95, 0.95] });
    L.bullets(s, [
      L.lead("Equal weight: ", "every dataset counts once in each mean, so large corpora do not dominate."),
      L.lead("Two scores: ", "Accuracy (share of clips right) and UAR (recall averaged over emotions, robust to a majority class)."),
    ], { y: 5.25, h: 1.5, fontSize: 20, gap: 6 });
    L.notes(s, `We test three explanations. For language, we compare the mean score on the ${F("n_english")} English corpora with the ${F("n_non_english")} non-English ones. For the setting, we compare the ${F("n_conv")} conversational corpora with the ${F("n_acted")} acted ones. Each dataset counts once in each mean, so a big corpus like THAI-SER with almost twelve thousand clips does not outweigh a small one.
    These group comparisons are only descriptive, because the groups differ in many ways at once. So each one gets a check: the language comparison is repeated within the acted corpora only, which removes the conversational corpora from the mix, and the setting comparison is repeated without the corpora that a model is documented to have seen in training.
    For clip duration we can do better than comparing datasets, because duration varies within each dataset. We ask whether longer clips are more often classified correctly than shorter clips of the same dataset and the same true emotion, and express it as the change in accuracy per doubling of the length.
    We report Accuracy, the share of clips right, and UAR, the unweighted average recall, which gives every emotion the same weight. UAR matters for the conversational corpora, where most clips are neutral.`);
  }

  // ---- 4 The datasets
  {
    const s = L.slide("Data", `${F("n_datasets")} public corpora: ${F("n_conv")} conversational and ${F("n_acted")} acted, ${F("n_english")} English and ${F("n_non_english")} in other languages`);
    const row = (d) => [{ text: d.name, bold: true }, d.language,
      { text: d.setting === "Conversational" ? "Conv." : "Acted", fill: d.setting === "Conversational" ? C.orangeFill : C.blueFill },
      { text: d.n.toLocaleString("en-US"), align: "right" }, { text: d.median.toFixed(1), align: "right" }];
    const header = ["Dataset", "Language", "Setting", "Clips", "Median s"];
    const colW = [1.92, 1.18, 0.9, 0.86, 1.1];   // fits "Emozionalmente" in bold 18 pt on one line
    L.table(s, header, D.datasets.slice(0, 8).map(row), { x: MX, y: TOP, colW, fontSize: 18, rowH: 0.4, zebra: false, margin: [3, 5, 3, 5] });
    L.table(s, header, D.datasets.slice(8).map(row), { x: MX + 6.17, y: TOP, colW, fontSize: 18, rowH: 0.4, zebra: false, margin: [3, 5, 3, 5] });
    L.bullets(s, [
      L.lead("Conv. = conversational: ", "turns from TV or scripted dialogues; each model hears one turn without context."),
      L.lead("Acted: ", "single sentences spoken with a target emotion (Emozionalmente by non-professional speakers)."),
    ], { y: 5.35, h: 1.45, fontSize: 20, gap: 6 });
    claim(V("n_conv") === 4 && V("n_acted") === 12, "slide 4: dataset groups changed");
    L.notes(s, `These are the ${F("n_datasets")} corpora, in the same order as in the benchmark figures. Orange marks the ${F("n_conv")} conversational corpora: MELD is TV dialogue, DailyTalk and MultiDialog are scripted conversations, and EmotionTalk is Mandarin dialogue with scenario-based acting. So conversational does not mean spontaneous; it means that the clips are turns taken from a dialogue. Every model still hears one turn alone, without the turns around it.
    Blue marks the ${F("n_acted")} acted corpora: single sentences recorded with a target emotion. All of them are acted; Emozionalmente was recorded by non-professional speakers through crowd-sourcing.
    ${F("n_english")} corpora are English, the other ${F("n_non_english")} are Mandarin, Thai, Polish and Italian. Note that language and setting overlap: three of the four conversational corpora are English. Clips is the number of evaluated clips, and Median s the median clip length in seconds.`);
    L.source(s, "Source: research_paper/output/cross_dataset_dataset_rows.csv");
  }

  // ---- 5 Clip durations
  {
    const s = L.slide("Data", `Most clips are short: dataset medians range from ${F("median_min")} to ${F("median_max")} s, and ${F("pct_under3")}% of all clips last under 3 s`);
    L.figure(s, "duration_distributions.png", MX, TOP - 0.02, 8.4, 5.3, { align: "left", valign: "top", alt: "Share of clips per duration bin for each of the 16 datasets" });
    L.bullets(s, [
      `Only ${F("pct_over10")}% of clips are longer than 10 s.`,
      `Dialogue corpora vary most: the middle half of MultiDialog spans ${F("multidialog_q25")}–${F("multidialog_q75")} s.`,
      `RAVDESS is almost constant: ${F("ravdess_q25")}–${F("ravdess_q75")} s.`,
      "Length = whole file, including pauses and silence.",
    ], { x: 9.2, y: TOP + 0.1, w: 3.55, h: 5.1, fontSize: 20, gap: 12 });
    L.notes(s, `Each panel is one dataset. The bars show the share of its clips in each length bin, orange for conversational and blue for acted corpora, and the black line is the median. The bins are 0 to 1, 1 to 2, 2 to 3, 3 to 5, 5 to 10 seconds and 10 seconds or more.
    Most clips are short. The dataset medians go from ${F("median_min")} seconds on ${F("median_min_name")} to ${F("median_max")} seconds on ${F("median_max_name")}, ${F("pct_under3")} percent of all ${F("clips")} clips are shorter than 3 seconds, and only ${F("pct_over10")} percent are longer than 10 seconds.
    The conversational corpora have the widest spread: the middle half of the MultiDialog clips lies between ${F("multidialog_q25")} and ${F("multidialog_q75")} seconds, and ${F("multidialog_over10")} percent are longer than 10 seconds. THAI-SER is the acted corpus with the widest range, ${F("thai_ser_q5")} to ${F("thai_ser_q95")} seconds between its 5th and 95th percentiles. RAVDESS is the opposite: the middle half of its clips lies between ${F("ravdess_q25")} and ${F("ravdess_q75")} seconds, which matters later because a length effect cannot be measured on an almost constant length.
    The length is the length of the audio file, measured from its header, so it includes pauses, silence and background noise; it is not the speaking time. The exact quantiles for every dataset are in the report table.`);
  }
};

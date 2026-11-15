// Summary, limitations, next steps, and the hidden appendix (secondary tables)
module.exports = function (L, H) {
  const { C, MX, CW, TOP } = L;
  const { D, F, V, list, signed, claim, gemmaPending } = H;
  const G = Object.fromEntries(D.groups.map((g) => [g.short, g]));
  const pp = (v) => `${signed(v)} pp`;
  const ci = (e) => (e.lo === null ? "—" : `[${signed(e.lo)}, ${signed(e.hi)}]`);
  const est = (e) => (e.slope === null ? "—" : `${signed(e.slope)} ${ci(e)}`);
  const convUp = D.groups.filter((g) => g.setting_gap_uar >= 5);
  const actedSer = D.groups.filter((g) => g.family === "SER" && g.setting_gap_uar <= -5);
  const positive = D.duration.filter((d) => d.evidence === "positive").sort((a, b) => b.within.slope - a.within.slope);
  const range = (vals) => { const a = Math.round(Math.min(...vals)), b = Math.round(Math.max(...vals)); return a === b ? `${a}` : `${a}–${b}`; };
  const exposedRuns = Object.values(D.exposed).reduce((n, a) => n + a.length, 0);

  // ---- 12 Summary
  {
    const s = L.slide("Summary", "What explains the dataset differences?");
    const qa = G["Qwen2-Audio"];
    L.table(s, ["Factor", "What we found", "Evidence"], [
      [{ text: "Setting", bold: true }, `${list(convUp.map((g) => g.short))}: ${range(convUp.map((g) => g.setting_gap_uar))} pp more UAR on conversational speech; ${list(actedSer.map((g) => g.short))}: ${range(actedSer.map((g) => -g.setting_gap_uar))} pp less, partly linked to training exposure`, "moderate: 4 vs 12 corpora; labels and exposure confounded"],
      [{ text: "Language", bold: true }, `small for ${F("lang_small_n")} of ${F("n_full")} models; Qwen2-Audio ${pp(qa.language_gap_uar)} UAR on English, also within acted corpora`, "clear for Qwen2-Audio; otherwise weak (6 corpora)"],
      [{ text: "Clip duration", bold: true }, `within datasets ${positive.map((d) => `${d.short} ${signed(d.within.slope)}`).join(", ")} pp per doubling; unclear for ${F("dur_unclear_n")} models; differs by dataset`, "weak to moderate: observational, 16 dataset units"],
      [{ text: "Dialogue Accuracy", bold: true }, `always “neutral” (${F("neutral_conv_mean")}%) beats every model's mean accuracy: judge dialogue corpora by UAR`, "strong: simple arithmetic"],
    ], { y: TOP, colW: [2.25, 6.65, 3.23], fontSize: 18, rowH: [0.48, 1.05, 0.8, 0.8, 0.8] });
    if (gemmaPending) L.tag(s, `Preliminary: Gemma 4 E4B has ${F("gemma_datasets")} only; ./rebuild.sh updates every number when its datasets arrive`, MX, 6.2, CW, { h: 0.5, fontSize: 18 });
    L.notes(s, `To sum up. The setting matters most, but differently for different models: ${list(convUp.map((g) => g.short))} gain ${range(convUp.map((g) => g.setting_gap_uar))} points of UAR on conversational speech, while ${list(actedSer.map((g) => g.short))} lose ${range(actedSer.map((g) => -g.setting_gap_uar))} points there. Part of the SER advantage on acted speech comes from training exposure to those corpora. The evidence is moderate, because there are only four conversational corpora and they differ from the acted ones in labels and recording too.
    Language matters little for most models, but Qwen2-Audio is much better on English, ${pp(qa.language_gap_uar)} in UAR, and that holds within the acted corpora as well.
    Clip duration: within a dataset, longer clips go with higher accuracy for ${list(positive.map((d) => d.short))}, by ${positive.map((d) => signed(d.within.slope)).join(", ")} points per doubling; for the others the direction is unclear, and the effect varies a lot between datasets. This is observational.
    And one practical lesson: on the conversational corpora, Accuracy rewards answering neutral. A constant neutral answer scores ${F("neutral_conv_mean")} percent on average, more than any model, so those corpora should be compared by UAR.
    ${gemmaPending ? `Gemma 4 E4B is preliminary: only ${F("gemma_datasets")} has current-protocol results. The rebuild script regenerates the analysis, figures, report and this deck when the rest arrives; it also prints any slide wording that no longer matches the numbers.` : ""}`);
  }

  // ---- 13 Limitations
  {
    const s = L.slide("Summary", "Limitations: what this analysis cannot show");
    const items = [
      L.lead("Observational: ", "the corpora differ in language, setting, labels, speakers and recording at once; no factor is isolated."),
      L.lead("Few units: ", `${F("n_conv")} conversational and ${F("n_non_english")} non-English corpora; ${F("n_datasets")} datasets behind every interval; no correction for multiple tests.`),
      L.lead("Training exposure: ", `documented for ${Object.keys(D.exposed).length} models (${exposedRuns} runs); “not established” does not mean unseen.`),
      L.lead("Length ≠ heard length: ", `file length includes silence; several local models crop to 15 or 30 s. Others have longer or no fixed caps (appendix).`),
      L.lead("No context: ", "one utterance per prediction, also for the dialogue corpora."),
    ];
    if (gemmaPending) items.push(L.lead("Gemma 4 E4B: ", `${F("gemma_datasets")} only; all its numbers are preliminary.`));
    L.bullets(s, items, { y: TOP + 0.05, h: 5.2, fontSize: 22, gap: 14 });
    L.notes(s, `The main limitation is that this is a comparison of existing corpora, not an experiment. A conversational corpus differs from an acted one not only in the setting but also in its labels, its speakers, its recording conditions and its clip lengths, so none of the three factors is isolated.
    The number of independent units is small: ${F("n_conv")} conversational corpora, ${F("n_non_english")} non-English corpora, and ${F("n_datasets")} datasets behind every confidence interval. With ten models and many comparisons, some intervals will exclude zero by chance; we did not correct for that.
    Training exposure is documented for ${Object.keys(D.exposed).length} models and ${exposedRuns} model–dataset runs. For the other runs the status is "not established", which means we found no evidence, not that the model never saw the data.
    Clip length is the length of the file, including silence. Several local models use a 15- or 30-second crop, while Qwen2.5-Omni has a 300-second configuration and emotion2vec+ has no fixed crop. Few clips exceed the shorter caps, as shown in the input-window appendix, but the distinction limits what the length analysis can say about long recordings.
    Finally, every model gets one utterance at a time, without the dialogue around it, so the conversational corpora are not a test of contextual emotion recognition.`);
  }

  // ---- 14 Next steps
  {
    const s = L.slide("Summary", "Next steps: small controlled tests that separate the explanations");
    const items = [
      L.lead("Controlled length test: ", "crop the same clips to 1, 2, 3, 5 and 10 s, or pad them with silence, with everything else fixed."),
      L.lead("Words versus voice: ", "compare audio with transcript-only input on listener-labelled clips."),
      L.lead("Answer format: ", "free answers versus constrained labels for the audio LLMs (invalid outputs, label priors)."),
      L.lead("Target domain: ", "validate any rule on held-out, listener-labelled data before trusting benchmark ranks."),
    ];
    if (gemmaPending) items.push(L.lead("Gemma 4 E4B: ", "when its other datasets arrive, ./rebuild.sh regenerates the analysis, figures, reports and this deck."));
    L.bullets(s, items, { y: TOP + 0.05, h: 5.2, fontSize: 22, gap: 14 });
    L.notes(s, `These are proposed experiments, not results. A controlled length test changes only the length of the same clips, by cropping or by padding with silence, which separates length from corpus and speaker. Comparing audio with transcript-only input tells us how much the models rely on the words. For the audio language models, comparing free answers with constrained label choices separates format problems from emotion errors. And before any rule is used on our own data, it should be checked on held-out data with listener labels from that domain.
    ${gemmaPending ? "When Gemma 4 E4B finishes its remaining datasets, one command, ./rebuild.sh in this folder, reruns the cross-dataset analysis from the saved predictions, redraws the figures, rewrites the English and Chinese reports, recomputes the facts and rebuilds this deck." : ""}`);
  }

  // ---- Appendix A1: all group means
  {
    const s = L.slide("Appendix", "Group means for all three metrics (equal-dataset means, %)", { appendix: true });
    const cell = (g, p) => ({ text: ["accuracy", "uar", "macro_f1"].map((k) => g[`${p}_${k}`].toFixed(1)).join(" / "), align: "center" });
    L.table(s, ["Model", "English", "Non-English", "Conversational", "Acted"],
      D.groups.map((g) => [{ text: g.short, bold: true }, cell(g, "eng"), cell(g, "non"), cell(g, "conv"), cell(g, "acted")]),
      { y: TOP, colW: [2.65, 2.37, 2.37, 2.37, 2.37], fontSize: 18, rowH: 0.42 });
    L.text(s, "Each cell: Accuracy / UAR / macro-F1. Only models with all 16 datasets.", MX, 6.35, CW, 0.4, { fontSize: 18, kind: "diagram" });
    L.notes(s, `The full table behind the two heat maps, with macro-F1 added. Each cell gives Accuracy, UAR and macro-F1 in that order, as equal-dataset means over the group. Macro-F1 follows UAR closely, so the conclusions do not depend on the choice between the two. ${gemmaPending ? "Gemma 4 E4B is excluded until it covers all datasets." : ""}`);
    L.source(s, "Source: research_paper/output/cross_dataset_group_rows.csv");
  }

  // ---- Appendix A2: splitting checks for all models
  {
    const s = L.slide("Appendix", "Splitting checks for all models (differences in pp, Accuracy / UAR)", { appendix: true });
    const two = (g, k) => ({ text: `${signed(g[`${k}_accuracy`])} / ${signed(g[`${k}_uar`])}`, align: "center" });
    L.table(s, ["Model", "English − non-English, acted only", "Conv. − acted, all runs", "Conv. − acted, no exposure", "Kept"],
      D.groups.map((g) => [{ text: g.short, bold: true }, two(g, "language_acted_gap"), two(g, "setting_gap"), two(g, "setting_clean_gap"),
        { text: `${g.clean_conv_n} + ${g.clean_acted_n}`, align: "center" }]),
      { y: TOP, colW: [2.65, 2.95, 2.45, 2.65, 1.43], fontSize: 18, rowH: 0.42 });
    L.text(s, "Kept = conversational + acted corpora without documented training exposure.", MX, 6.64, CW, 0.35, { fontSize: 18, kind: "diagram" });
    L.notes(s, `Both checks for every model. The first column repeats the language comparison within the twelve acted corpora only, seven English and five non-English. The second column is the setting difference from the main heat map, and the third repeats it without the runs that have documented training exposure. For the ${D.groups.filter((g) => !g.exposed.length).length} models without documented exposure the second and third columns are identical. Differences subtract one-decimal means.`);
    L.source(s, "Source: research_paper/output/cross_dataset_model_rows.csv");
  }

  // ---- Appendix A3: duration slope sensitivity
  {
    const s = L.slide("Appendix", "Length slopes under four specifications (pp per doubling, 95% CI)", { appendix: true });
    L.table(s, ["Model", "All clips pooled", "Within datasets (main)", "Speaker/dialogue CI", "Clips 1–15 s only"],
      D.duration.map((d) => [{ text: d.short + (d.n_datasets < V("n_datasets") ? "*" : ""), bold: true }, { text: est(d.pooled), align: "center" },
        { text: est(d.within), align: "center", bold: d.evidence === "positive" }, { text: est(d.speaker), align: "center" }, { text: est(d.within_1to15), align: "center" }]),
      { y: TOP, colW: [2.5, 2.42, 2.42, 2.42, 2.37], fontSize: 18, rowH: 0.4 });
    L.text(s, `Bold: main interval above 0.${gemmaPending ? ` * Gemma 4 E4B: ${F("gemma_datasets")} only, no interval across datasets.` : ""}`, MX, 6.42, CW, 0.4, { fontSize: 18, kind: "diagram" });
    L.notes(s, `The duration slopes under four specifications. Pooled fits one slope to all clips without any adjustment. The main analysis compares clips within the same dataset and true emotion, with equal dataset and emotion weights, and clusters the interval by dataset. The speaker or dialogue column keeps the same estimate but clusters by speaker or dialogue, which gives narrower intervals; it checks dependence within speakers but does not test new speakers. The last column refits the main analysis on clips between 1 and 15 seconds, inside every local model's input window. ${(() => { const k = D.duration.filter((d) => d.within_1to15.lo !== null && d.within_1to15.lo > 0).map((d) => d.short); const lost = positive.filter((d) => !k.includes(d.short)).map((d) => d.short); return `With clips of 1 to 15 seconds only, ${k.length ? list(k) : "no model"} keep${k.length === 1 ? "s" : ""} an interval above zero${lost.length ? `, while ${list(lost)} ${lost.length === 1 ? "loses" : "lose"} it` : ""}.`; })()} Most other models move around zero in every version.`);
    L.source(s, "Source: research_paper/output/cross_dataset_effects.csv");
  }

  // ---- Appendix A4: between-model slope differences
  {
    const s = L.slide("Appendix", "Is the length slope really larger for some models? Direct differences", { appendix: true });
    const rows = D.gap_effects.map((g) => [{ text: g.model, bold: true }, g.reference, { text: signed(g.slope), align: "center" },
      { text: g.lo === null ? "—" : `[${signed(g.lo)}, ${signed(g.hi)}]`, align: "center", bold: g.lo !== null && (g.lo > 0 || g.hi < 0) }]);
    L.table(s, ["Model", "Reference", "Difference (pp per doubling)", "95% CI"], rows, { y: TOP, colW: [3.0, 3.0, 3.3, 2.83], fontSize: 18, rowH: 0.45 });
    const clear = D.gap_effects.filter((g) => g.lo !== null && g.lo > 0);
    const clearModels = [...new Set(clear.map((g) => g.model))].filter((m) => clear.filter((g) => g.model === m).length === 2);
    L.text(s, `Bold: interval excludes 0. ${clearModels.length ? `Only ${list(clearModels)}'s slope is larger than both SER references.` : "No slope is clearly larger than both SER references."}`, MX, 6.15, CW, 0.5, { fontSize: 20 });
    claim(clearModels.length === 1 && clearModels[0] === "Gemini 3.8 Flash", "appendix A4 / slide 9 notes: the direct slope comparison changed");
    L.notes(s, `One model having a significant slope and another not is not itself evidence that their slopes differ. This table tests the difference directly, on the same clips: each row is the slope of the per-clip correctness difference between a model and a reference SER model. ${clearModels.length ? `Only ${list(clearModels)} gains clearly more with length than both emotion2vec+ and MERaLiON.` : ""} GPT Realtime points the same way but its intervals include zero. Melati's and Gemma 4's slopes are not distinguishable from the references.`);
    L.source(s, "Source: research_paper/output/cross_dataset_gap_effects.csv");
  }

  // ---- Appendix A5: corpus-level correlations
  {
    const s = L.slide("Appendix", `Corpora with longer clips are not systematically easier (Accuracy ρ from ${F("rho_min")} to ${F("rho_max")})`, { appendix: true });
    const cell = (v) => ({ text: `${signed(v.rho, 2)} (${v.p.toFixed(3)})`, align: "center", bold: v.p < 0.05 });
    L.table(s, ["Model", "Accuracy ρ (p)", "UAR ρ (p)", "Macro-F1 ρ (p)"],
      Object.entries(D.correlations).map(([m, v]) => [{ text: m, bold: true }, cell(v.accuracy), cell(v.uar), cell(v.macro_f1)]),
      { y: TOP, colW: [3.1, 3.01, 3.01, 3.01], fontSize: 18, rowH: 0.42 });
    L.text(s, "One point per dataset (16 points); bold: uncorrected p < 0.05, exploratory only.", MX, 6.35, CW, 0.4, { fontSize: 18, kind: "diagram" });
    L.notes(s, `A different question from the main length analysis: are corpora with a longer median clip easier overall? Each correlation uses one point per dataset, so only sixteen points. The correlations go from ${F("rho_min")} to ${F("rho_max")} for Accuracy, and the few with p below 0.05 (${F("rho_sig")}) are uncorrected and exploratory. Corpus difficulty, label sets and language differ far more between corpora than length does.`);
    L.source(s, "Source: research_paper/output/cross_dataset_correlations.csv");
  }

  // ---- Appendix A6: conversational corpora vs always-neutral
  {
    const s = L.slide("Appendix", "On the conversational corpora, always answering “neutral” is a strong baseline", { appendix: true });
    L.table(s, ["Dataset", "Always “neutral”", "Best Accuracy (model)", "Best UAR (model)"],
      Object.values(D.conv_neutral).map((v) => [{ text: v.name, bold: true }, { text: `${v.neutral.toFixed(1)}%`, align: "center", bold: v.neutral > v.best },
        { text: `${v.best.toFixed(1)}% (${v.best_model})`, align: "center" }, { text: `${v.best_uar.toFixed(1)}% (${v.best_uar_model})`, align: "center" }]),
      { y: TOP, colW: [2.6, 2.6, 3.5, 3.43], fontSize: 20, rowH: 0.55 });
    L.text(s, `Bold: the constant answer beats every model (${F("neutral_beats_all")}). The UAR of a constant answer is 1 / number of emotions.`, MX, 4.6, CW, 0.9, { fontSize: 20 });
    L.notes(s, `The conversational corpora are dominated by neutral clips. A model that always answered neutral would get the accuracy in the second column. On ${F("neutral_beats_all")} that constant answer beats every model's accuracy. UAR does not have this problem: a constant answer only gets one divided by the number of emotions, so the best UAR values in the last column are real discrimination.`);
    L.source(s, "Source: cross_dataset_dataset_rows.csv (neutral_accuracy), cross_dataset_model_rows.csv");
  }

  // ---- Appendix A7: input windows
  {
    const s = L.slide("Appendix", "How much of each clip the local models hear", { appendix: true });
    const rows = Object.entries(D.caps).map(([m, v]) => [{ text: m, bold: true }, { text: `first ${v.cap} s`, align: "center" }, { text: v.beyond.toLocaleString("en-US"), align: "center" }]);
    rows.push([{ text: "emotion2vec+", bold: true }, { text: "no fixed crop", align: "center" }, { text: "—", align: "center" }]);
    rows.push([{ text: "GPT Realtime, Gemini 3.8 Flash", bold: true }, { text: "server side, not verified", align: "center" }, { text: "—", align: "center" }]);
    L.table(s, ["Model", "Input window", "Clips longer than the window"], rows, { y: TOP, colW: [4.4, 3.9, 3.83], fontSize: 18, rowH: 0.43 });
    L.notes(s, `The loaders crop long clips: the Vox-Profile models use the first 15 seconds, Qwen2-Audio, Gemma 4, MERaLiON and Melati the first 30 seconds, and Qwen2.5-Omni's processor allows 300 seconds. emotion2vec+ has no fixed crop. For the API models the server-side handling is not documented in our outputs. The last column counts clips over all completed datasets that are longer than the window; they are few, mostly THAI-SER and MultiDialog clips beyond 15 seconds, so cropping cannot explain the main results.`);
    L.source(s, "Source: cross_dataset_model_rows.csv (known_cap_seconds, beyond_cap_count)");
  }
};

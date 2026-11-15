// One section per question: setting (and its training-exposure check), language, clip duration (three slides)
module.exports = function (L, H) {
  const { C, MX, CW, TOP } = L;
  const { D, F, V, list, signed, claim, gemmaPending } = H;
  const G = Object.fromEntries(D.groups.map((g) => [g.short, g]));
  const full = D.groups.map((g) => g.model);
  const pp = (v) => `${signed(v)} pp`;
  const t = (name) => (name === "Gemini 3.8 Flash" ? "Gemini" : name);   // shorter name in titles and bullets

  // ---- 6 Setting
  {
    const convUp = D.groups.filter((g) => g.setting_gap_uar >= 5);
    const actedSer = D.groups.filter((g) => g.family === "SER" && g.setting_gap_uar <= -5).sort((a, b) => a.setting_gap_uar - b.setting_gap_uar);
    const qa = G["Qwen2-Audio"];
    const s = L.slide("Question 1 · Setting", `${list(convUp.map((g) => t(g.short)))} do better on conversational speech; ${list(actedSer.map((g) => g.short))} on acted speech`);
    const fig = L.figure(s, "performance_by_setting.png", MX, TOP + 0.05, 8.2, 5.2, { align: "left", valign: "top", alt: "Equal-dataset Accuracy and UAR on conversational and acted corpora, with their difference" });
    const x = MX + fig.w + 0.3, w = 12.73 - x;
    L.bullets(s, [
      lead("UAR gain on dialogue: ", convUp.map((g) => `${t(g.short)} ${signed(g.setting_gap_uar)}`).join(", ") + " pp."),
      lead("UAR loss: ", actedSer.map((g) => `${g.short} ${signed(g.setting_gap_uar)}`).join(", ") + " pp."),
      `Qwen2-Audio: the audio-LLM exception (${pp(qa.setting_gap_uar)}).`,
      lead("Dialogue Accuracy flatters: ", `always “neutral” gets ${F("neutral_conv_mean")}%, more than any model (best ${F("best_conv_accuracy")}%).`),
    ], { x, y: TOP + 0.05, w, h: 5.2, fontSize: 20, gap: 10 });
    L.text(s, "Δ = conversational − acted (pp) · orange: higher on conversational · blue: higher on acted", MX, TOP + 0.1 + fig.h, fig.w + 0.2, 0.34, { fontSize: 16, kind: "diagram" });
    claim(convUp.every((g) => g.family !== "SER"), "slide 6: an SER model now does better on conversational speech");
    claim(qa.setting_gap_uar < 0, "slide 6: Qwen2-Audio is no longer better on acted speech");
    claim(V("best_conv_accuracy") < 100 * V("neutral_conv_mean"), "slide 6: a model now beats the always-neutral mean on conversational accuracy");
    L.notes(s, `The first question is the setting. The heat map shows, for each model, the mean Accuracy and UAR over the ${F("n_conv")} conversational and the ${F("n_acted")} acted corpora, and the difference Δ in percentage points. Orange means better on conversational speech, blue better on acted speech. The ${F("n_full")} models are those with results on all sixteen datasets${gemmaPending ? "; Gemma 4 E4B joins when its datasets arrive" : ", including Gemma 4 E4B"}.
    The pattern splits the models. ${list(convUp.map((g) => g.short))} gain ${convUp.map((g) => signed(g.setting_gap_uar)).join(", ")} points of UAR on conversational speech. ${list(actedSer.map((g) => g.short))} lose ${actedSer.map((g) => signed(g.setting_gap_uar)).join(", ")} points: they are much stronger on the acted corpora. The two Vox-Profile models are in between. Qwen2-Audio is the exception among the audio language models: it is ${pp(qa.setting_gap_uar)} in UAR and ${pp(qa.setting_gap_accuracy)} in Accuracy, because it scores very high on some acted English corpora such as TESS and CREMA-D.
    A warning about Accuracy on the conversational corpora: most of their clips are neutral. Always answering neutral would score ${F("neutral_conv_mean")} percent on average over the four, which is higher than the mean Accuracy of every model; the best is ${F("best_conv_model")} with ${F("best_conv_accuracy")} percent. On ${F("neutral_beats_all")} the constant answer beats every model. That is why we read the setting effect mainly from UAR, which weights every emotion equally.
    This is a comparison of corpora, not an experiment: the conversational corpora also differ in labels, recording and length, and the SER models have seen some acted corpora in training. The next slide checks that last point.`);
  }

  // ---- 7 Training-exposure check
  {
    const exposed = D.groups.filter((g) => g.exposed.length);
    const s = L.slide("Question 1 · Setting", "Without corpora seen in training, the SER models' advantage on acted speech shrinks");
    const arrow = (a, b) => `${signed(a)} → ${signed(b)}`;
    L.table(s, ["Model", "Corpora with documented training exposure", "Δ Acc.: all → kept", "Δ UAR: all → kept", "Kept"],
      exposed.map((g) => [{ text: g.short, bold: true }, `${g.exposed.length}: ${g.exposed.join(", ")}`,
        { text: arrow(g.setting_gap_accuracy, g.setting_clean_gap_accuracy), align: "center" },
        { text: arrow(g.setting_gap_uar, g.setting_clean_gap_uar), align: "center" },
        { text: `${g.clean_conv_n} + ${g.clean_acted_n}`, align: "center" }]),
      { y: TOP, colW: [1.9, 5.5, 1.75, 1.75, 1.23], fontSize: 18, rowH: [0.72, 0.72, 0.72, 0.45] });
    const e2v = G["emotion2vec+"];
    const keptUar = exposed.map((g) => Math.abs(g.setting_clean_gap_uar));
    const fewKept = exposed.filter((g) => g.clean_acted_n <= 6);
    const keptN = fewKept.map((g) => g.clean_acted_n);
    const keptText = Math.min(...keptN) === Math.max(...keptN) ? String(keptN[0]) : `${Math.min(...keptN)}–${Math.max(...keptN)}`;
    L.bullets(s, [
      lead("emotion2vec+: ", `the Accuracy gap shrinks from ${pp(e2v.setting_gap_accuracy)} to ${pp(e2v.setting_clean_gap_accuracy)} once its ${e2v.exposed.length} exposed corpora are dropped.`),
      lead("A UAR gap remains: ", `${Math.round(Math.min(...keptUar))}–${Math.round(Math.max(...keptUar))} pp for all three models.`),
      lead("Weak check: ", `only ${keptText} acted corpora are left for ${list(fewKept.map((g) => g.short))}; “no documented exposure” does not prove a corpus was unseen.`),
    ], { y: 4.55, h: 2.3, fontSize: 20, gap: 8 });
    claim(exposed.every((g) => Math.abs(g.setting_clean_gap_accuracy) < Math.abs(g.setting_gap_accuracy)), "slide 7: dropping exposed corpora no longer shrinks every accuracy gap");
    claim(exposed.every((g) => g.setting_clean_gap_uar < 0), "slide 7: a UAR gap no longer remains for every exposed model");
    L.notes(s, `${exposed.length} SER models have documented training links to some of our corpora, from the training provenance review of the project: ${exposed.map((g) => `${g.short} to ${g.exposed.length} (${g.exposed.join(", ")})`).join("; ")}. A model that has seen a corpus in training is expected to score high on it, which would inflate its acted-corpus mean.
    So we repeat the comparison keeping only the runs whose provenance review found no documented exposure. For emotion2vec+ the Accuracy gap between conversational and acted speech shrinks from ${pp(e2v.setting_gap_accuracy)} to ${pp(e2v.setting_clean_gap_accuracy)}, and the UAR gap from ${pp(e2v.setting_gap_uar)} to ${pp(e2v.setting_clean_gap_uar)}. MERaLiON and Melati shrink less. A UAR gap of ${Math.round(Math.min(...keptUar))} to ${Math.round(Math.max(...keptUar))} points remains for all three, so exposure explains part of the advantage, not all of it.
    This check is weak: ${list(fewKept.map((g) => g.short))} keep only ${fewKept[0].clean_conv_n} conversational and ${keptText} acted corpora, so a single corpus moves the mean a lot. And "no documented exposure" only means that we found no evidence of training use; it does not prove that the corpus was unseen. The other ${D.groups.length - exposed.length} models have no documented exposure, so their numbers do not change.`);
    L.source(s, "Source: cross_dataset_model_rows.csv (training_status from emotion_recognition/docs/training_provenance.md)");
  }

  // ---- 8 Language
  {
    const qa = G["Qwen2-Audio"], me = G["Melati"];
    const s = L.slide("Question 2 · Language", `Qwen2-Audio is far better on English (${pp(qa.language_gap_uar)} UAR); for ${F("lang_small_n")} of ${F("n_full")} models the language gap stays within ${F("lang_small_max")} pp`);
    const fig = L.figure(s, "performance_by_language.png", MX, TOP + 0.05, 8.2, 5.25, { align: "left", valign: "top", alt: "Equal-dataset Accuracy and UAR on English and non-English corpora, with their difference" });
    const x = MX + fig.w + 0.3, w = 12.73 - x;
    const melatiNonEnglish = (D.exposed["Melati"] || []).filter((n) => D.datasets.find((d) => d.name === n && d.language !== "English"));
    L.bullets(s, [
      lead("Qwen2-Audio, acted corpora only: ", `${pp(qa.language_acted_gap_accuracy)} Accuracy, ${pp(qa.language_acted_gap_uar)} UAR, so the setting mix does not explain it.`),
      lead("Melati: ", `better outside English (${pp(me.language_gap_uar)} UAR); trained on ${list(melatiNonEnglish)}.`),
      `Non-English = ${F("n_non_english")} corpora in 4 languages.`,
    ], { x, y: TOP + 0.05, w, h: 5.0, fontSize: 20, gap: 10 });
    L.text(s, "Δ = English − non-English (pp) · orange: higher on English · blue: higher on non-English", MX, TOP + 0.08 + fig.h, fig.w + 0.2, 0.34, { fontSize: 16, kind: "diagram" });
    claim(D.facts.lang_big.value.includes("Qwen2-Audio"), "slide 8: Qwen2-Audio no longer has the largest language gap");
    claim(me.language_gap_uar < 0 && melatiNonEnglish.length > 0, "slide 8: the Melati statement no longer holds");
    L.notes(s, `The second question is language. Same layout as before: mean Accuracy and UAR on the ${F("n_english")} English and the ${F("n_non_english")} non-English corpora, and the difference. Orange means better on English.
    For ${F("lang_small_n")} of the ${F("n_full")} models the difference stays within ${F("lang_small_max")} points, which is small next to the spreads we saw at the start. The clear exception is Qwen2-Audio: ${pp(qa.language_gap_accuracy)} in Accuracy and ${pp(qa.language_gap_uar)} in UAR in favour of English.
    Because three of the four conversational corpora are English, the language groups also differ in setting. So we repeat the comparison within the acted corpora only, seven English against five non-English. Qwen2-Audio's English advantage stays, at ${pp(qa.language_acted_gap_accuracy)} Accuracy and ${pp(qa.language_acted_gap_uar)} UAR, so it is not a side effect of the setting. It could still come from the particular corpora, their labels or training exposure that is not documented; Italian, for example, is among the languages the Qwen2-Audio authors list, yet it scores low on both Italian corpora.
    Melati goes the other way, ${pp(me.language_gap_uar)} in UAR: it is better outside English, and its training data include ${list(melatiNonEnglish)}, two of our non-English corpora. The non-English group is small, ${F("n_non_english")} corpora in Mandarin, Thai, Polish and Italian, so one corpus can move its mean.`);
  }

  // ---- 9 Duration: pooled versus within each dataset
  {
    const positive = D.duration.filter((d) => d.evidence === "positive").sort((a, b) => b.within.slope - a.within.slope);
    const flips = D.duration.filter((d) => D.facts.dur_flips.value.includes(d.short));
    const s = L.slide("Question 3 · Clip duration", `Within a dataset, ${list(positive.map((d) => t(d.short)))} are more accurate on longer clips; for the other ${F("dur_unclear_n")} models it is unclear`);
    const fig = L.figure(s, "duration_effects.png", MX, TOP + 0.05, 7.8, 5.0, { align: "left", valign: "top", alt: "Accuracy change per doubling of clip duration: all clips pooled and within each dataset" });
    const x = MX + fig.w + 0.3, w = 12.73 - x;
    L.bullets(s, [
      lead("Right = fair test: ", "longer vs shorter clips of the same dataset and emotion."),
      lead("Per doubling of length: ", positive.map((d) => `${t(d.short)} ${signed(d.within.slope)}`).join(", ") + " pp."),
      lead("Left mixes datasets: ", `${list(flips.map((d) => d.short))} look worse on long clips (${flips.map((d) => signed(d.pooled.slope)).join(", ")} pp): their long-clip corpora are harder for them; within datasets ${flips.map((d) => signed(d.within.slope)).join(", ")}.`),
      ...(!gemmaPending ? [lead("Gemma sensitivity: ", "its 1–15 s interval includes zero (appendix).") ] : []),
    ], { x, y: TOP + 0.05, w, h: 4.5, fontSize: 20, gap: 10 });
    L.text(s, "Dots: estimate · bars: 95% CI · right of 0: more accurate on longer clips", MX, TOP + fig.h + 0.12, fig.w + 0.2, 0.34, { fontSize: 16, kind: "diagram" });
    if (gemmaPending) L.tag(s, `Gemma 4 E4B (${F("gemma_datasets")} only): not shown yet`, x, 6.15, w, { h: 0.45 });
    claim(V("dur_negative_n") === 0, "slide 9: a model now has a negative within-dataset association");
    claim(positive.length > 0 && flips.length > 0, "slide 9: the positive or the pooled-flip list is empty");
    L.notes(s, `The third question is clip duration. Both panels show, for each model, how much accuracy changes when a clip is twice as long, in percentage points, with a 95% confidence interval. Right of the zero line means longer clips are classified correctly more often.
    The left panel is the naive analysis: all clips of all datasets pooled into one fit. It is misleading because the datasets differ in both length and difficulty. ${list(flips.map((d) => d.short))} look clearly worse on long clips there, ${flips.map((d) => signed(d.pooled.slope)).join(" and ")} points per doubling, but that only says that the corpora with long clips are harder for these models.
    The right panel is the fair test and our main analysis. It only compares longer and shorter clips within the same dataset and the same true emotion, and gives every dataset and every emotion the same weight. There, ${list(positive.map((d) => `${d.short} ${signed(d.within.slope)}`))} points per doubling have intervals entirely above zero. For the other ${F("dur_unclear_n")} models the interval includes zero, so the direction is unresolved; no model is clearly worse on longer clips.
    The intervals treat the ${F("n_datasets")} datasets as the independent units, so they are wide, and they are not corrected for testing ten models. A difference between two models also needs a direct test: in the appendix only Gemini's slope is clearly larger than both SER references. And this is an association in existing data, not an experiment: longer clips may also carry more words or stronger emotion.${gemmaPending ? ` Gemma 4 E4B is not shown because one dataset (${F("gemma_datasets")}) gives no interval across datasets; its within-dataset slope there is ${F("gemma_first_slope")}.` : " Gemma's positive main interval is sensitive to the duration range: restricting clips to 1–15 seconds gives an interval that includes zero, as shown in the appendix. Its duration slope is not clearly larger than either SER reference."}`);
  }

  // ---- 10 Duration: heat map per dataset
  {
    const thai = V("heat_thai_ser_pos"), jl = V("heat_jl_corpus_neg");
    const s = L.slide("Question 3 · Clip duration", `The length association differs by dataset: positive on THAI-SER for ${thai} of ${F("n_full")} models, negative on JL-Corpus for ${jl}`);
    L.figure(s, "duration_within_dataset_slopes.png", MX, TOP - 0.08, CW, 4.85, { valign: "top", alt: "Heat map of the within-dataset accuracy change per duration doubling, models by datasets" });
    L.text(s, [{ text: "Bold: ", options: { bold: true } }, { text: `95% CI excludes 0 (only the ${F("heat_ci_n")} datasets with ≥ 20 speakers or dialogues)` }], MX, 6.38, 6.6, 0.55, { fontSize: 18, kind: "diagram", valign: "middle" });
    L.text(s, [{ text: "RAVDESS: ", options: { bold: true } }, { text: `clips last ${F("ravdess_q25")}–${F("ravdess_q75")} s, so its huge slopes mean little` }], MX + 6.85, 6.38, 5.25, 0.55, { fontSize: 18, kind: "diagram", valign: "middle" });
    claim(thai >= V("n_full") - 1 && jl >= V("n_full") - 2, "slide 10: THAI-SER or JL-Corpus counts changed");
    L.notes(s, `This heat map is the per-dataset version of the right panel. Each cell is one model on one dataset: the change in accuracy per doubling of clip length, within that dataset and adjusted for the true emotion. Orange means more accurate on longer clips, blue less accurate, and the colour saturates at 20 points. Thick lines separate the conversational corpora, the acted English corpora and the other languages.
    The effect is not one number per model. THAI-SER, the acted corpus with the widest length range and 200 speakers, is positive for ${thai} of ${F("n_full")} complete models, ${F("heat_thai_ser_boldpos")} of them with an interval above zero. MEAD is positive for ${F("heat_mead_pos")}. JL-Corpus is negative for ${jl}, although none of these is significant there. CREMA-D splits: ${F("heat_crema_d_boldpos")} models clearly positive and ${F("heat_crema_d_boldneg")} clearly negative.
    Bold numbers have a 95% interval that excludes zero. Intervals need enough independent speakers or dialogues, so they exist only for the ${F("heat_ci_n")} datasets with at least 20: ${F("heat_ci_datasets")}. They are not corrected for the many cells, so one bold cell alone is weak evidence.
    Ignore RAVDESS: its clips last ${F("ravdess_q25")} to ${F("ravdess_q75")} seconds, so a doubling of length lies far outside the data and the slope, up to ${F("heat_ravdess_max")} points, is an extrapolation. Cell labels are capped at 50: any slope beyond ±50 points is shown as "≥50" or "≤−50" so it fits its cell, and the exact values are in facts.md and the CSV. ${gemmaPending ? `Gemma 4 E4B has only its ${F("gemma_datasets")} cell so far; the dashes are pending results.` : ""}`);
  }

  // ---- 11 Duration: raw accuracy by bin
  {
    const md = full.map((m) => D.heat.multidialog[m].slope);
    const lo = Math.round(Math.min(...md)), hi = Math.round(Math.max(...md));
    const s = L.slide("Question 3 · Clip duration", `Raw accuracy by length misleads: on MultiDialog it falls for all ${F("raw_multidialog_fall_n")} models, yet within each emotion the change is only ${signed(lo, 0)} to ${signed(hi, 0)} pp per doubling`);
    L.figure(s, "accuracy_by_duration.png", MX, TOP - 0.06, CW, 5.3, { valign: "top", alt: "Observed accuracy in each clip-duration bin, per dataset and model" });
    claim(V("raw_multidialog_fall_n") === V("n_full") && Math.max(Math.abs(lo), Math.abs(hi)) <= 8, "slide 11: the MultiDialog contrast changed");
    L.notes(s, `For completeness, the raw view: accuracy in each length bin, per dataset, one line per model, only for bins with at least 50 clips. This is what most people would plot first, and it can mislead.
    On MultiDialog every one of the ${F("raw_multidialog_fall_n")} complete models is less accurate on its longest than on its shortest clips; Qwen2-Audio drops from ${F("raw_multidialog_qwen2_audio_7b")}. But within each true emotion, the change per doubling is only ${signed(lo, 0)} to ${signed(hi, 0)} points, as the heat map showed. The raw fall comes from the mix of emotions: on MultiDialog the share of each emotion changes with clip length, and some emotions are harder than others.
    On THAI-SER the raw lines rise for the API models, GPT Realtime ${F("raw_thai_gpt_realtime")} and Gemini ${F("raw_thai_gemini_3_8_flash")}, and here the adjusted slopes agree (${F("heat_thai_gpt_realtime")} and ${F("heat_thai_gemini_3_8_flash")} points per doubling).
    So read the raw bins as a description of where the errors are, and use the adjusted slopes to judge whether length matters.`);
  }

  function lead(b, t) { return L.lead(b, t); }
};

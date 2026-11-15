// Current full benchmark results. Historical listening diagnostics remain at the end of this section.
const fs = require("fs");
const path = require("path");
const SUMMARY = path.resolve(__dirname, "../reports/summary.json");
const ORDER = ["meld", "emotiontalk", "dailytalk", "multidialog", "crema_d", "ravdess", "savee", "jl_corpus",
  "tess", "mead", "mess", "mcae_spps", "thai_ser", "nemo", "emovo", "emozionalmente"];
const MODEL_IDS = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm",
  "qwen2_5_omni_7b", "qwen2_audio_7b", "gemma4_e4b", "gpt_realtime", "gemini-3.8-flash"];
const SHORT = ["emotion2vec+", "MERaLiON", "Melati", "Vox Whisper", "Vox WavLM", "Qwen2.5-Omni", "Qwen2-Audio", "Gemma 4 E4B", "GPT Realtime", "Gemini 3.8"];
const HEADER = ["emotion2vec+", "MERaLiON", "Melati", "Vox\nWhisper", "Vox\nWavLM", "Qwen2.5-\nOmni", "Qwen2-\nAudio", "Gemma 4\nE4B", "GPT\nRealtime", "Gemini\n3.8 Flash"];
const DISPLAY_DATASETS = { meld: "MELD", emotiontalk: "EmotionTalk", dailytalk: "DailyTalk", multidialog: "MultiDialog",
  crema_d: "CREMA-D", ravdess: "RAVDESS", savee: "SAVEE", jl_corpus: "JL-Corpus", tess: "TESS", mead: "MEAD",
  mess: "MESS", mcae_spps: "MCAE-SPPS", thai_ser: "THAI-SER", nemo: "nEMO", emovo: "EMOVO", emozionalmente: "Emozionalmente" };

module.exports = function (L, A) {
  const { C, W, MX, FONT } = L;
  const D = JSON.parse(fs.readFileSync(SUMMARY, "utf8"));
  const datasets = Object.fromEntries(D.datasets.map((d) => [d.key, d]));
  const names = Object.fromEntries(D.models.map((m) => [m.model, m.name]));
  const score = (m, d, metric = "accuracy") => D.results[m][d][metric];
  const mean = (m, metric = "accuracy", ds = ORDER) => 100 * ds.reduce((a, d) => a + score(m, d, metric), 0) / ds.length;
  const f1 = (v) => v.toFixed(1);
  const fmt = (v) => Number(v).toLocaleString("en-US");
  const allRows = MODEL_IDS.flatMap((m) => ORDER.map((d) => D.results[m]?.[d]));
  if (allRows.some((r) => !r || r.sample || r.clips !== r.total_clips)) throw new Error("Full benchmark slides require all 160 complete model–dataset results.");
  const clips = ORDER.reduce((a, d) => a + D.results[MODEL_IDS[0]][d].clips, 0);
  const foot = (s, text) => L.text(s, text, MX, 6.89, W - 2 * MX - 0.25, 0.3,
    { fontSize: 16, color: "000000" });
  const notes = (s, text) => L.notes(s, `${text}\nSource: emotion_recognition/reports/summary.json, generated ${D.date}. All 160 results cover the full saved benchmark cohorts. Model names: ${MODEL_IDS.map((m) => names[m]).join(", ")}.`);

  // ---- Protocol: current full cohorts, not the earlier 200-clip API pilot
  {
    const s = L.slide("4 · Evaluation Results", "Complete Benchmark: 10 Models on 16 Datasets");
    L.bullets(s, [
      { t: `${fmt(clips)} distinct clips, 160 model–dataset results`, b: true,
        sub: ["Five dedicated SER models, three local audio LLMs and two API models. Every model covers the full cohort of each dataset."] },
      { t: "Accuracy compares one predicted label with the corpus reference", b: true,
        sub: ["UAR averages recall over emotions. Macro-F1 weights each class equally. Weighted-F1 follows the corpus class distribution."] },
      { t: "The candidate labels are the dataset's mapped emotions", b: true,
        sub: ["SER uses argmax over allowed scores. Audio LLMs generate an answer that is parsed into a label. Invalid answers count as errors."] },
      { t: "Training provenance affects interpretation", b: true,
        sub: ["Documented corpus exposure includes upstream teachers and development use. It does not establish exact test-clip overlap."] },
    ], { y: 1.5, h: 5.05, fontSize: 21 });
    foot(s, "Full saved predictions, including final Gemma 4 E4B results. One utterance per prediction.");
    notes(s, "These slides replace the earlier six-dataset, seven-model pilot. All 10 models now have results on all 16 datasets. The corpus reference can describe intended or perceived emotion depending on the dataset. Different scoring metrics can change model rankings, especially on imbalanced conversational corpora. SER's constrained score selection and an LLM's generated label are different answer protocols, so format failures are part of the benchmark outcome.");
  }

  // ---- Overall means, one equal weight per dataset, editable native chart
  {
    const s = L.slide("4 · Evaluation Results", "Accuracy and UAR Across All 16 Datasets");
    s.addChart(L.pptx.ChartType.bar, [
      { name: "Accuracy", labels: SHORT, values: MODEL_IDS.map((m) => +f1(mean(m))) },
      { name: "UAR (mean recall)", labels: SHORT, values: MODEL_IDS.map((m) => +f1(mean(m, "unweighted_accuracy"))) },
    ], { x: MX, y: 1.4, w: W - 2 * MX, h: 5.27, barDir: "bar", barGrouping: "clustered",
      chartColors: ["0072B2", "D55E00"], showLegend: true, legendPos: "t", legendFontFace: FONT,
      legendFontSize: 18, legendColor: "000000", showValue: true, dataLabelFormatCode: "0.0",
      dataLabelPosition: "outEnd", dataLabelFontSize: 16, dataLabelFontFace: FONT, dataLabelColor: "000000",
      catAxisLabelFontSize: 18, catAxisLabelFontFace: FONT, catAxisLabelColor: "000000",
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 20, valAxisLabelFontSize: 18,
      valAxisLabelFontFace: FONT, valAxisLabelColor: "000000", showValAxisTitle: true,
      valAxisTitle: "Equal-dataset mean (%)", valAxisTitleFontSize: 18, valAxisTitleColor: "000000",
      showCatName: false, showTitle: false, showBorder: false, showCatName: false,
      valGridLine: { color: "D5DAE1", width: 0.6 }, showCatName: false });
    foot(s, "Each dataset counts once. These means describe this benchmark, without a significance claim.");
    notes(s, `The native chart averages every metric separately over the same 16 datasets, giving a small corpus the same weight as a large one. It does not pool clips. ${MODEL_IDS.map((m, i) => `${SHORT[i]} Accuracy ${f1(mean(m))}%, UAR ${f1(mean(m, "unweighted_accuracy"))}%`).join(". ")}. The next two tables retain the individual-dataset accuracy values for every model.`);
  }

  // ---- All 16 datasets, split by model family to retain readable editable tables
  for (const [title, start] of [["Dataset Accuracy: Five Dedicated SER Models", 0], ["Dataset Accuracy: Five Audio LLMs, Including Gemma", 5]]) {
    const selected = MODEL_IDS.slice(start, start + 5);
    for (let half = 0; half < 2; half++) {
    const part = ORDER.slice(half * 8, half * 8 + 8);
    const s = L.slide("4 · Evaluation Results", `${title} (${half + 1}/2)`);
    const cell = (text, extra = {}) => ({ text: String(text), options: { margin: [1, 4, 1, 4], color: "000000", ...extra } });
    const header = [cell("Dataset", { color: "FFFFFF" }), ...HEADER.slice(start, start + 5).map((t) => cell(t, { color: "FFFFFF", align: "center" }))];
    const body = part.map((d) => {
      const best = Math.max(...MODEL_IDS.map((m) => score(m, d)));
      return [cell(DISPLAY_DATASETS[d], { bold: true }), ...selected.map((m) => cell(f1(100 * score(m, d)), { bold: score(m, d) === best, align: "center" }))];
    });
    L.table(s, header, body, { y: 1.5, colW: [2.35, 1.95, 1.95, 1.95, 1.95, 1.98], fontSize: 20,
      rowH: [0.68, ...part.map(() => 0.42)] });
    foot(s, "Accuracy (%). Bold = best of all ten models on that dataset. Full cohorts for every model.");
    notes(s, `Part ${half + 1} of two editable tables for ${selected.map((m) => names[m]).join(", ")}, preserving all 16 datasets across both slides. Each value is the fraction of reference labels classified correctly. Bold compares all ten models, not only the family shown on this slide. UAR and macro-F1 appear in the appendix because Accuracy alone can reward the majority neutral class. ${part.map((d) => `${datasets[d].dataset}: ${selected.map((m) => `${names[m]} ${f1(100 * score(m, d))}%`).join(", ")}`).join(". ")}`);
    }
  }

  // ---- Updated training-exposure comparison
  {
    const s = L.slide("4 · Evaluation Results", "Training Exposure and Benchmark Interpretation");
    const exposed = (r) => ["confirmed_exposure", "probable_exposure"].includes(r.training_status);
    const body = MODEL_IDS.slice(0, 5).filter((m) => ORDER.some((d) => exposed(D.results[m][d]))).map((m) => {
      const yes = ORDER.filter((d) => exposed(D.results[m][d])), no = ORDER.filter((d) => !exposed(D.results[m][d]));
      return [{ text: names[m], bold: true }, `${yes.length} datasets`, `${f1(mean(m, "accuracy", yes))}%`, `${no.length} datasets`, `${f1(mean(m, "accuracy", no))}%`];
    });
    L.table(s, ["Model", "Exposure documented", "Mean Acc.", "No evidence found", "Mean Acc."], body,
      { y: 1.6, colW: [3.1, 2.65, 1.7, 2.98, 1.7], fontSize: 20, rowH: 0.65 });
    L.bullets(s, [
      "Provenance counts direct training, upstream teacher lineage and documented development use.",
      "The two groups also differ in language, setting and labels. Their score gap cannot isolate the effect of training exposure.",
      "No documented exposure does not prove unseen data. Exact clip leakage needs a separate overlap audit.",
    ], { y: 4.65, h: 1.95, fontSize: 21, gap: 8 });
    foot(s, "Source: current reports/summary.json and docs/training_provenance.md.");
    notes(s, "This refresh replaces the earlier seen-versus-unseen figures, which used six datasets and a narrower account of model lineage. The current provenance review counts broader training involvement. The table reports equal-dataset accuracy separately for documented exposure and runs where exposure is not established. It is descriptive and cannot predict performance on internal calls.");
  }

  // ---- Updated audio-LLM result, including final Gemma
  {
    const s = L.slide("4 · Evaluation Results", "Audio LLM Performance Depends on the Dataset Setting");
    const conv = ORDER.slice(0, 4), acted = ORDER.slice(4);
    L.table(s, ["Audio LLM", "Dialogue Acc.", "Dialogue UAR", "Acted Acc.", "Acted UAR"],
      MODEL_IDS.slice(5).map((m, i) => [{ text: SHORT[i + 5], bold: true },
        f1(mean(m, "accuracy", conv)), f1(mean(m, "unweighted_accuracy", conv)),
        f1(mean(m, "accuracy", acted)), f1(mean(m, "unweighted_accuracy", acted))]),
      { y: 1.5, colW: [3.3, 2.2, 2.2, 2.2, 2.23], fontSize: 20, rowH: 0.58 });
    const gemma = MODEL_IDS[7];
    L.bullets(s, [
      `Gemma 4 E4B now covers all 16 datasets: dialogue UAR ${f1(mean(gemma, "unweighted_accuracy", conv))}%, acted UAR ${f1(mean(gemma, "unweighted_accuracy", acted))}%.`,
      "Dialogue corpora have many neutral clips. UAR exposes errors that Accuracy can hide.",
      "Language, clip duration, intensity and answer format are candidate explanations. Saved predictions establish associations rather than mechanisms.",
    ], { y: 5.15, h: 1.5, fontSize: 20, gap: 6 });
    foot(s, "Equal-dataset means (%): 4 conversational corpora versus 12 acted corpora.");
    notes(s, "All five audio language models are present, including the final Gemma 4 E4B runs. This comparison replaces the earlier 200-clip API pilot and its broad claim that audio LLMs always underperform dedicated SER. The models show different patterns by corpus, and no single model family wins universally. The accompanying cross-dataset and intensity decks examine the proposed explanations with appropriate class balancing and uncertainty.");
  }

  // ---- Selected historical diagnostics retain their original audio and SER prediction cohorts
  const fig = (f) => path.join(L.DIR, "assets/fig", f);
  for (const c of A.cms) {
    const title = c.title.replace(/^Historical[^:]*:\s*/i, "");
    const s = L.slide("4 · Selected SER Diagnostics", `Historical SER diagnostic: ${title.replace(/^Confusion Matrices:\s*/, "")}`);
    if (fs.existsSync(fig(`cm_pair_${c.key}.png`))) L.image(s, fig(`cm_pair_${c.key}.png`), MX, 1.4, 8.6, 5.15);
    L.bullets(s, c.bullets, { x: 9.45, y: 1.6, w: 3.3, h: 4.9, fontSize: 18 });
    foot(s, "Historical selected SER diagnostic. Left: emotion2vec+ large. Right: MERaLiON-SER-v1.");
    notes(s, `Historical diagnostic retained for the listening discussion, using its original selected SER results and label reference. It is not a comparison of all ten models. ${c.notes}`);
  }
  (A.afterCms || []).forEach((f) => f(L));
};

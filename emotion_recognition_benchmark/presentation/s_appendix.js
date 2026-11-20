// Appendix: other metrics, label mapping
module.exports = function (L, A) {
  const { C, MX, MODEL } = L;
  const order = A.benchmark.manifest.model_order;
  const names = { emotion2vec_plus_large: "e2v+", meralion_ser_v1: "MER", melati_emotion: "Melati", vox_profile_whisper: "Vox W", vox_profile_wavlm: "Vox L", qwen2_5_omni_7b: "Omni", qwen2_audio_7b: "Qwen-A", gemma4_e4b: "Gemma", gpt_realtime: "GPT", "gemini-3.8-flash": "Gemini" };
  let appendix = 0;
  for (const [metric, label] of [["uar", "UAR"], ["macro_f1", "Macro-F1"]]) {
    for (let half = 0; half < 2; half++) {
      appendix++;
      const s = L.slide("Appendix", `A${appendix} · ${label}: All Ten Models (${half * 8 + 1}–${half * 8 + 8} of 16 datasets)`);
      const rows = A.datasetOrder.slice(half * 8, half * 8 + 8).map(d => {
        const dataset = A.benchmark.dataset_rows.find(r => r.dataset === d);
        const values = order.map(m => A.metric(m, d, metric));
        const best = Math.max(...values);
        return [{ text: dataset.name, bold: true }, ...values.map(v => ({ text: v.toFixed(1), bold: v === best, align: "center" }))];
      });
      L.table(s, ["Dataset", ...order.map(m => names[m])], rows,
        { y: 1.65, colW: [2.13, ...order.map(() => 1.0)], fontSize: 16, rowH: 0.48 });
      L.text(s, "Percentages; bold = highest within this dataset. Invalid answers remain errors. Candidate sets and original composite-label scoring are preserved.", MX, 6.25, 12.1, 0.6, { fontSize: 14, color: C.ink });
      L.source(s, "Source: research_paper/output/cross_dataset_model_rows.csv; complete benchmark refreshed 2026-11-15");
      L.notes(s, `Native editable ${label} table over the same full cohorts for all ten models. UAR averages original gold-class recall. Macro-F1 follows the benchmark composite-label convention, including JL-Corpus excited, and invalid predictions remain false negatives. These are point estimates, not cross-model significance claims. Sources: reports/summary.md and research_paper/output/cross_dataset_analysis.json.`);
    }
  }
  {
    const s = L.slide("Appendix", "A5 · Label Mapping: Six Listening-Example Corpora");
    const rows = [
      ["angry", "ANG", "Anger", "A", "05 angry", "angry", "angry"],
      ["disgusted", "DIS", "Disgust", "–", "07 disgust", "disgust", "–"],
      ["fearful", "FEA", "Fear", "–", "06 fearful", "fear", "–"],
      ["happy", "HAP", "Happiness", "H", "03 happy", "happy", "happy; excited (happy counts)"],
      ["neutral", "NEU", "Neutral", "C (calm)", "01 neutral; 02 calm", "neutral", "neutral"],
      ["sad", "SAD", "Sadness", "S", "04 sad", "sad", "sad"],
      ["surprised", "–", "Surprise", "–", "08 surprised", "ps (pleasant surprise)", "excited (surprised also counts)"],
      ["not evaluated", "", "", "", "", "", "anxious, apologetic, assertive, concerned, encouraging (1,200 clips)"],
    ];
    L.table(s, ["Project class", "CREMA-D", "MCAE-SPPS", "MESS", "RAVDESS", "TESS", "JL-Corpus"], rows.map((r) => r.map((c, j) => ({ text: c, bold: j === 0 }))), { y: 1.45, colW: [1.5, 1.0, 1.25, 1.1, 1.9, 1.9, 3.48], fontSize: 12, rowH: 0.46 });
    L.text(s, "Model side: emotion2vec+ \"other\" and \"<unk>\", Vox-Profile \"Contempt\" and \"Other\" are never chosen (the prediction is the most probable of the dataset's emotions).", MX, 5.5, 12, 0.6, { fontSize: 13, color: C.text });
    L.source(s, "Source: emotion_recognition_benchmark/docs/label_mapping.md");
  }
  {
    const s = L.slide("Appendix", "A6 · emotion2vec+ large: Detailed Architecture");
    L.image(s, "img/e2v_arch.png", MX, 1.35, L.W - 2 * MX, 5.65);
    L.notes(s, `Detailed architecture of emotion2vec+ large with tensor shapes for a 4-second clip at 16 kHz, measured with forward hooks on the released model.`);
  }
};

// Appendix: other metrics, label mapping
module.exports = function (L, A) {
  const { C, MX, MODEL } = L;
  const ms = ["e2v", "meralion", "melati", "voxw", "voxl"];
  const ds = ["CREMA-D", "MCAE-SPPS", "MESS", "RAVDESS", "TESS", "JL-Corpus"];
  const seen = { "CREMA-D": ["e2v", "meralion"], RAVDESS: ["e2v"], TESS: ["e2v"], "JL-Corpus": ["e2v", "melati"] };
  const UA = [[86.5, 84.9, 52.4, 35.2, 32.1], [54.2, 57.8, 48.1, 40.9, 36.7], [40.8, 74.5, 40.4, 36.4, 36.3], [92.3, 61.3, 49.3, 49.3, 42.4], [85.5, 49.2, 50.4, 30.1, 40.1], [92.5, 75.9, 99.5, 45.1, 48.3]];
  const F1 = [[86.6, 84.7, 46.7, 29.9, 27.4], [50.1, 54.4, 43.3, 35.2, 33.6], [35.2, 74.7, 37.5, 26.8, 28.7], [92.2, 59.7, 46.6, 44.5, 38.4], [79.8, 37.9, 43.4, 17.4, 31.6], [92.3, 77.3, 99.5, 47.3, 50.5]];
  const tbl = (vals) => ds.map((d, i) => { const best = Math.max(...vals[i]); return [{ text: d, bold: true }, ...vals[i].map((v, j) => ({ text: v.toFixed(1) + ((seen[d] || []).includes(ms[j]) ? " †" : ""), bold: v === best, options: { align: "center", color: (seen[d] || []).includes(ms[j]) ? C.muted : C.ink } }))]; });
  const hdr = ["Dataset", ...ms.map((m) => MODEL[m].name)];
  {
    const s = L.slide("Appendix", "A1 · Unweighted accuracy and macro-F1 (open models, all clips)");
    L.text(s, "Unweighted accuracy (mean recall per emotion, %)", MX, 1.4, 6, 0.3, { fontSize: 14, bold: true, color: C.ink });
    L.table(s, hdr, tbl(UA), { x: MX, y: 1.75, w: 5.95, colW: [1.25, 1.0, 1.0, 0.85, 0.95, 0.9], fontSize: 11, rowH: 0.4 });
    L.text(s, "Macro-F1 (%)", 6.85, 1.4, 6, 0.3, { fontSize: 14, bold: true, color: C.ink });
    L.table(s, hdr, tbl(F1), { x: 6.85, y: 1.75, w: 5.95, colW: [1.25, 1.0, 1.0, 0.85, 0.95, 0.9], fontSize: 11, rowH: 0.4 });
    L.text(s, "† in the model's training data · bold = best · same ranking as accuracy", MX, 4.95, 10, 0.3, { fontSize: 12, color: C.muted });
    L.source(s, "Source: emotion_recognition/reports/summary.md (evaluate.py, 2026-10-29)");
  }
  {
    const s = L.slide("Appendix", "A2 · Label mapping to the seven project classes");
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
    L.source(s, "Source: emotion_recognition/docs/label_mapping.md");
  }
};

// Audio clip widgets shared by the data and case-study slides
module.exports = function (L) {
  const { C, FONT } = L;
  const p = L.pptx;
  const SHORT = { "emotion2vec+ large": "emotion2vec+", "MERaLiON-SER-v1": "MERaLiON", Melati: "Melati", "Vox-Profile Whisper": "Vox Whisper", "Vox-Profile WavLM": "Vox WavLM" };

  // Listener votes as a sorted list of [label, count]
  const votes = (c) => Object.entries(c.listener_votes || {}).sort((a, b) => b[1] - a[1]);

  // Small horizontal stacked bar of the listener votes
  function voteBar(s, c, x, y, w, h = 0.22) {
    const v = votes(c), n = v.reduce((a, b) => a + b[1], 0);
    let cx = x;
    v.forEach(([lab, k]) => {
      const ww = (w * k) / n;
      s.addShape(p.ShapeType.rect, { x: cx, y, w: ww, h, fill: { color: L.EMO[lab] || C.muted }, line: { color: C.white, width: 0.75 } });
      if (ww > 0.55) s.addText(`${lab} ${k}`, { x: cx, y, w: ww, h, fontFace: FONT, fontSize: 10, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      cx += ww;
    });
    return n;
  }

  // Prediction runs: "emotion2vec+ angry 1.00 · MERaLiON ..." coloured green/red against a reference label
  function predRuns(c, models, ref, fontSize = 12) {
    const runs = [];
    models.forEach((m, i) => {
      const [lab, pr] = c.predictions[m];
      runs.push({ text: (i ? "   " : "") + SHORT[m] + " ", options: { color: C.muted, fontSize } });
      runs.push({ text: `${lab} ${pr.toFixed(2)}`, options: { bold: true, color: lab === ref ? L.C.good : L.C.bad, fontSize } });
    });
    return runs;
  }

  // One row: play button, acted tag, listener votes, two top-model predictions
  function exampleRow(s, c, x, y, w) {
    s.addShape(p.ShapeType.rect, { x, y, w, h: 1.0, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    L.audio(s, c.path.startsWith("assets") ? c.path : "assets/" + c.path, x + 0.15, y + 0.2, 0.6);
    L.text(s, "Actor", x + 0.95, y + 0.12, 1.0, 0.25, { fontSize: 11, color: C.muted });
    L.tag(s, c.acted_label, x + 0.95, y + 0.38, { w: 1.15 });
    const vx = x + 2.3, vw = w - 2.45;
    if (c.listener_votes) {
      const n = votes(c).reduce((a, b) => a + b[1], 0);
      L.text(s, `Listeners (n = ${n})`, vx, y + 0.1, vw, 0.25, { fontSize: 11, color: C.muted });
      voteBar(s, c, vx, y + 0.36, vw, 0.24);
    } else {
      L.text(s, "Listeners: no per-clip ratings published", vx, y + 0.1, vw, 0.25, { fontSize: 11, color: C.muted });
    }
    s.addText(predRuns(c, ["emotion2vec+ large", "MERaLiON-SER-v1"], c.acted_label, 12), { x: vx, y: y + 0.66, w: vw, h: 0.28, fontFace: FONT, margin: 0, valign: "middle", fit: "shrink" });
  }

  // Large case card for the failure analysis
  function caseRow(s, c, y, h, label) {
    const x = L.MX, W = L.W - 2 * L.MX;
    s.addShape(p.ShapeType.rect, { x, y, w: W, h, fill: { color: C.white }, line: { color: C.line, width: 1 } });
    L.audio(s, "assets/" + c.path, x + 0.2, y + 0.25, 0.75, true);
    L.text(s, `${c.dataset} · speaker ${c.speaker}`, x + 1.15, y + 0.15, 4.5, 0.3, { fontSize: 12, color: C.muted });
    L.text(s, c.sentence ? `"${c.sentence.replace(/^"|"$/g, "")}"` : "(sentence text not published)", x + 1.15, y + 0.42, 4.6, 0.3, { fontSize: 13, italic: true, color: C.text });
    L.text(s, "Actor", x + 1.15, y + 0.85, 0.8, 0.3, { fontSize: 12, color: C.muted, valign: "middle" });
    L.tag(s, c.acted_label, x + 1.85, y + 0.85, { w: 1.2 });
    if (c.listener_votes) {
      const n = votes(c).reduce((a, b) => a + b[1], 0);
      L.text(s, `Listeners (n = ${n})`, x + 1.15, y + 1.3, 3, 0.25, { fontSize: 12, color: C.muted });
      voteBar(s, c, x + 1.15, y + 1.58, 4.4, 0.28);
    } else L.text(s, "Listeners: no per-clip ratings published", x + 1.15, y + 1.3, 4.4, 0.3, { fontSize: 12, color: C.muted });
    // predictions of the 5 models
    const px = x + 5.95;
    L.text(s, "Model predictions (probability)", px, y + 0.15, 3.4, 0.3, { fontSize: 12, color: C.muted });
    Object.keys(SHORT).forEach((m, i) => {
      const [lab, pr] = c.predictions[m], yy = y + 0.45 + i * 0.3;
      s.addText([{ text: SHORT[m] + ((c.models_in_training_data || []).includes(m) ? " †" : ""), options: { color: C.text } }], { x: px, y: yy, w: 1.45, h: 0.28, fontFace: FONT, fontSize: 12, margin: 0, valign: "middle" });
      s.addShape(p.ShapeType.rect, { x: px + 1.5, y: yy + 0.05, w: 1.05 * pr, h: 0.18, fill: { color: L.EMO[lab] || C.muted }, line: { color: C.white, width: 0 } });
      s.addText(`${lab} ${pr.toFixed(2)}`, { x: px + 1.55 + 1.05 * pr, y: yy, w: 1.4, h: 0.28, fontFace: FONT, fontSize: 12, bold: true, color: lab === c.acted_label ? C.good : C.bad, margin: 0, valign: "middle" });
    });
    // discussion
    const dx = x + 9.75;
    s.addShape(p.ShapeType.rect, { x: dx, y: y + 0.12, w: W - 9.87, h: h - 0.24, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    L.text(s, label, dx + 0.12, y + 0.2, W - 10.1, 0.3, { fontSize: 12, bold: true, color: C.accent2 });
    L.text(s, c.why_interesting, dx + 0.12, y + 0.5, W - 10.1, h - 0.7, { fontSize: 12, color: C.text, ls: 1.05 });
  }

  return { exampleRow, caseRow, voteBar, predRuns, votes, SHORT };
};

// All slides of the intensity deck. Numbers are computed in data.js from the saved analysis; see facts.md.
module.exports = function (L, X) {
  const { C, W, H, MX, CW, TOP } = L;
  const p = L.pptx;
  const { S, DS, SHORT, rows, get, pp, pct, int, rangeText, range } = X;
  const FIG_TOP = 1.5;
  const prelim = () => {};
  const gemmaNote = S.gemma
    ? "The final Gemma 4 E4B predictions are included on all six intensity corpora."
    : "Gemma 4 E4B has no eligible intensity results in this build.";

  // Full-width figure with short bullets below it
  function figSlide(section, title, fig, items, notes, o = {}) {
    const s = L.slide(section, title, { titleSize: o.titleSize });
    prelim(s);
    const img = L.image(s, fig, MX, FIG_TOP, CW, o.figH || 4.6, { valign: "top", alt: o.alt });
    const y = img.y + img.h + 0.1;
    L.bullets(s, items, { y, h: 6.98 - y, fontSize: 20, gap: 3, ls: 1.0 });
    L.notes(s, notes);
    return s;
  }
  const cr = rows("crema_d"), rv = rows("ravdess"), md = rows("mead"), th = rows("thai_ser"), mc = rows("mcae_spps"), dt = rows("dailytalk");
  const B = S.byDataset;
  const names = (list) => list.map((r) => `${SHORT[r.model]} ${pp(r.observed_change_pp)}`).join(", ");

  // ---- 1 Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.55, w: 0.09, h: 3.35, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Is a Stronger Emotional Expression\nEasier to Classify?", MX + 0.4, 1.45, 11.9, 1.5, { fontSize: 40, bold: true, ls: 0.95 });
    L.text(s, `Emotion intensity in our speech emotion recognition benchmark:\n${Object.keys(DS).length} corpora with intensity labels, ${S.models} models`, MX + 0.4, 3.15, 11.8, 1.0, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  research_paper  ·  November 2026", MX + 0.4, 4.4, 11.8, 0.42, { fontSize: 20 });
    L.text(s, "Saved predictions of the benchmark only; no model was re-run. Public corpora only.", MX, H - 0.95, CW, 0.4, { fontSize: 18, kind: "diagram" });
    prelim(s);
    L.notes(s, `This talk asks one question about our speech emotion recognition benchmark: when an emotion is expressed more strongly, do the models classify it correctly more often?
    Six of the sixteen benchmark corpora say how strongly each clip expresses its emotion. We use those labels, and the predictions the models already made in the benchmark. Nothing was re-run for this analysis.
    ${S.models} models are included: five speech emotion classifiers and ${S.models - 5} audio language models. ${gemmaNote}`);
  }

  // ---- 2 Question
  {
    const s = L.slide("Question", "Are strong emotions easier for the models than mild ones?");
    const y0 = TOP + 0.02, bw = 3.86, gap = 0.27, bh = 1.9;
    const kinds = [
      ["Acted level", "actors perform the same sentence weakly or strongly", "CREMA-D · RAVDESS ·\nMEAD · THAI-SER", C.blueFill],
      ["Listener rating", "40 listeners rate each clip from 1 to 9; we use the mean", "MCAE-SPPS", C.greenFill],
      ["Annotator rating", "levels 1–3 added later to dialogue turns", "DailyTalk", C.amberFill],
    ];
    L.text(s, "What “intensity” means here: a label the corpus itself provides", MX, y0, CW, 0.4, { fontSize: 22, bold: true });
    kinds.forEach(([head, body, sets], i) => {
      const x = MX + i * (bw + gap), y = y0 + 0.5;
      s.addShape(p.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: kinds[i][3] }, line: { color: C.line, width: 1 } });
      L.text(s, head, x + 0.15, y + 0.1, bw - 0.3, 0.4, { fontSize: 22, bold: true });
      L.text(s, body, x + 0.15, y + 0.5, bw - 0.3, 0.62, { fontSize: 20, ls: 0.95 });
      L.text(s, sets, x + 0.15, y + 1.14, bw - 0.3, 0.7, { fontSize: 20, bold: true, color: C.link, ls: 0.95 });
    });
    L.bullets(s, [
      { runs: [{ text: "Question: ", options: { bold: true } }, { text: "is a clip with a higher intensity label classified correctly more often?" }] },
      { runs: [{ text: "Why it matters: ", options: { bold: true } }, { text: "everyday speech is often milder than acted emotion; if models only work on strong emotion, benchmark scores overstate their accuracy on mild speech." }] },
      { runs: [{ text: "Not tested: ", options: { bold: true } }, { text: "whether a model can estimate intensity itself; the models only output an emotion category." }] },
    ], { y: y0 + 2.6, h: 2.7, fontSize: 22, gap: 10 });
    L.notes(s, `Intensity in this talk always means a label that the corpus itself provides. We never derive intensity from loudness, from clip length or from how confident a model is.
    There are three kinds of labels. In four corpora, actors were asked to perform the same sentence at different strengths: CREMA-D has low, medium and high, RAVDESS has normal and strong, MEAD has three levels, and the scripted part of THAI-SER has low and high. In MCAE-SPPS, a Mandarin corpus, 40 listeners rated how strong each clip sounds on a scale from 1 to 9, and we use their mean. DailyTalk, an English dialogue corpus, has levels 1 to 3 that another research group (ECSS) added later.
    The question is simple: is a clip with a higher intensity label classified correctly more often? It matters because everyday speech is usually milder than the acted emotions in most benchmark corpora. If the models only work when emotion is strong, the benchmark scores look better than what we would get on mild, natural speech.
    What we do not test: whether a model can tell how strong an emotion is. All benchmark models output an emotion category, so a higher accuracy at strong levels says nothing about intensity estimation.`);
  }

  // ---- 3 Data
  {
    const s = L.slide("Data", "Six corpora carry intensity labels; we use their emotional clips");
    const sc = X.scope, cl = X.clusters;
    const level3 = X.smallStrata.find((r) => r.dataset === "dailytalk");
    const r_ = (ds, lang, kind, levels, speakers) => [{ text: DS[ds], bold: true }, lang, kind, levels, { text: int(sc[ds].n_non_neutral_known), align: "right" }, speakers];
    L.table(s, ["Corpus", "Language", "Intensity label", "Levels", "Clips", "Speakers"], [
      r_("crema_d", "English", "acted (one sentence)", "low · medium · high", `${cl.crema_d} actors`),
      r_("ravdess", "English", "acted", "normal · strong", `${cl.ravdess} actors`),
      r_("mead", "English", "acted", "weak · medium · strong", `${cl.mead} actors`),
      r_("thai_ser", "Thai", "acted (scripted part)", "low · high", `${cl.thai_ser} actors`),
      r_("mcae_spps", "Mandarin", "mean of 40 listeners", "1–9 (shown as quartiles)", `${cl.mcae_spps} speakers`),
      r_("dailytalk", "English", "annotator levels", level3 ? `1 · 2 (3: only ${level3.n} clips)` : "1 · 2 · 3", "2 actors"),
    ], { y: TOP, colW: [1.85, 1.45, 2.85, 3.05, 1.08, 1.85], fontSize: 20, rowH: 0.5, align: { 4: "right" } });
    L.bullets(s, [
      "Clips: emotional clips with a known level; neutral clips and clips without a level are left out.",
      "The other 10 benchmark corpora have no intensity labels and are not part of this analysis.",
    ], { y: 5.35, h: 1.5, fontSize: 20, gap: 6 });
    L.notes(s, `The table lists the six corpora. Clips is the number of emotional clips that have a known intensity level; together ${int(Object.values(sc).reduce((a, r) => a + r.n_non_neutral_known, 0))} clips. Neutral clips are left out, because "a weak neutral" is not a weak emotion, and clips without a level are never filled in as low.
    Some details matter. In CREMA-D only one of the twelve sentences was recorded at three levels; all other sentences have an unspecified level and are not used, which is why only ${int(sc.crema_d.n_non_neutral_known)} of ${int(sc.crema_d.n_total)} clips enter. In THAI-SER only the scripted part has a level. MCAE-SPPS gives a continuous mean rating; we keep the raw value for the statistics and only show quartiles in the figures. DailyTalk has only ${level3 ? level3.n : "few"} clips at its top level, too few to show, so its figures compare level 1 with level 2.
    The speaker column matters for the confidence intervals: they are computed over speakers, or over dialogues for DailyTalk. MCAE-SPPS has only ${cl.mcae_spps} speakers and DailyTalk only two actors, so their results are much less certain than those of the acted corpora with 24 to 200 actors.`);
    L.cite(s, "Sources: output/intensity_scope_rows.csv, intensity_inventory_manifest.json; report intensity_en.md");
  }

  // ---- 4 Method
  {
    const s = L.slide("Method", "Three comparisons, always within one corpus");
    const bw = 3.86, gap = 0.27, y = TOP + 0.05, bh = 2.75;
    const steps = [
      ["1 · Weakest vs strongest", "accuracy at the strongest level minus the weakest level, in percentage points (pp); no adjustment"],
      ["2 · Adjusted", "the same change after controlling for the emotion mix and clip length; 95% interval over speaker/dialogue source clusters"],
      ["3 · Matched takes", "same actor, same sentence, same emotion: strong take minus weak take"],
    ];
    steps.forEach(([head, body], i) => {
      const x = MX + i * (bw + gap);
      s.addShape(p.ShapeType.roundRect, { x, y, w: bw, h: bh, rectRadius: 0.08, fill: { color: i === 0 ? C.blueFill : C.panel }, line: { color: C.line, width: 1 } });
      L.text(s, head, x + 0.18, y + 0.15, bw - 0.36, 0.45, { fontSize: 22, bold: true });
      L.text(s, body, x + 0.18, y + 0.7, bw - 0.36, 1.95, { fontSize: 20, ls: 1.0 });
    });
    L.bullets(s, [
      "One result per model and corpus; corpora are never pooled, because their scales differ.",
      "Positive = more clips classified correctly at the stronger level.",
    ], { y: y + bh + 0.3, h: 1.6, fontSize: 20, gap: 6 });
    L.notes(s, `All three comparisons are made separately for each model and each corpus, because a step on the RAVDESS scale and a step on the CREMA-D scale are not the same thing. We never average across corpora.
    First, the simplest number: accuracy at the strongest level minus accuracy at the weakest level, in percentage points. A level must have at least 20 clips to be shown.
    Second, an adjusted version. The levels can differ in which emotions they contain and in how long the clips are, and both affect accuracy. So we fit, for each model and corpus, a weighted linear model of correct or wrong on the intensity level, with a separate baseline for each emotion, equal total weight for each emotion, and the logarithm of the clip length as a second predictor. The slope is converted back to a weakest-to-strongest change. The 95% interval is clustered by speaker, or by dialogue for DailyTalk, and is not shown when there are fewer than 20 clusters. These intervals are exploratory and not corrected for the many comparisons.
    Third, matched takes: in CREMA-D, RAVDESS and THAI-SER the same actor recorded the same sentence with the same emotion at different strengths, so we compare the strong take with the weak take directly. This removes differences in speaker and wording.`);
  }

  // ---- 5 Main result: observed change
  {
    const thaiR = range(th, "observed_change_pp");
    figSlide("Results", "Stronger expressions are classified better, except on THAI-SER", "emotion_intensity_change.png", [
      `CREMA-D ${B.crema_d.low}→${B.crema_d.high}: all ${B.crema_d.n} models gain (${rangeText(cr, "observed_change_pp")}); RAVDESS: ${B.ravdess.observedUp} of ${B.ravdess.n} (${names(rv.filter((r) => r.observed_change_pp < 0))}).`,
      `THAI-SER ${B.thai_ser.low}→${B.thai_ser.high}: ${pp(thaiR[0])} to ${pp(thaiR[1])} pp, no gain. MEAD: small gains (${rangeText(md, "observed_change_pp")}).`,
      `MCAE-SPPS gains most (${rangeText(mc, "observed_change_pp")}), partly because its emotion mix changes with the rating.`,
    ], `This is the main result. Each cell is one model on one corpus: accuracy at the strongest level minus accuracy at the weakest level, in percentage points. Blue means more correct at the stronger level, red means fewer. There is no adjustment here.
    Across all ${S.pairs} model and corpus pairs, ${S.observedUp} are more accurate at the stronger level.
    On CREMA-D every model gains, from ${rangeText(cr, "observed_change_pp")}; on RAVDESS ${B.ravdess.observedUp} of ${B.ravdess.n} gain, and only Melati is lower. MEAD gains are consistent but small.
    THAI-SER is the exception: low and high scripted takes are classified about equally well, between ${pp(thaiR[0])} and ${pp(thaiR[1])} points. We do not know why; possible reasons are that its low and high takes differ less, or that the models handle Thai differently. This corpus alone shows that the effect is not universal.
    The two right columns use rated intensity. MCAE-SPPS compares the lowest and highest listener-rating quartiles and shows the largest raw gains, but the emotions are not evenly spread over the quartiles, so part of that gain is a change in the emotion mix. The adjusted slide later corrects for this. DailyTalk compares level 1 with level 2.
    ${gemmaNote}`, { alt: "Heat map: accuracy change from the weakest to the strongest level, model by corpus" });
  }

  // ---- 6 Accuracy per level
  {
    const ceiling = X.C.filter((r) => ["crema_d", "ravdess"].includes(r.dataset) && r.low_accuracy >= 0.75);
    const meadOther = md.filter((r) => r.model !== "emotion2vec_plus_large");
    const meadMonotone = X.models.filter((m) => X.levels("mead", m).every((r, i, a) => !i || r.accuracy >= a[i - 1].accuracy));
    const meadMax = Math.max(...meadOther.map((r) => r.high_accuracy));
    figSlide("Results", "Accuracy usually rises with intensity; models near the ceiling gain little", "emotion_intensity_accuracy.png", [
      `Starting at ≥ 75%, models gain ≤ ${pp(Math.max(...ceiling.map((r) => r.observed_change_pp)))} pp: ${[...new Set(ceiling.map((r) => SHORT[r.model]))].join(", ")} (CREMA-D, RAVDESS).`,
      `MEAD: ${meadMonotone.length}/${S.models} rise at every step. All except emotion2vec+ stay ≤ ${pct(meadMax)} even at “strong”.`,
    ], `Here are the accuracies at every level, one line per model, in the benchmark colours. For three-level corpora the rise is mostly step by step, not only at the top.
    The models that gain least are the ones that are already accurate at the weakest level: ${ceiling.map((r) => `${SHORT[r.model]} on ${DS[r.dataset]} starts at ${pct(r.low_accuracy)} and gains ${pp(r.observed_change_pp)} points`).join("; ")}. They have little room left. emotion2vec+ was trained on these acted corpora, which is marked in the benchmark figures.
    MEAD is a special case: accuracy rises at every step for ${meadMonotone.length} of ${S.models} models. ${md.filter((r) => !meadMonotone.includes(r.model)).map((r) => SHORT[r.model]).join(", ")} ${S.models - meadMonotone.length === 1 ? "has" : "have"} a small middle-to-high decline. Apart from emotion2vec+ all models stay low, at most ${pct(meadMax)} even at the strong level. So a stronger expression does not fix MEAD.
    THAI-SER lines are flat. In MCAE-SPPS the quartiles are display bins of the listener rating, not native levels. DailyTalk level 3 has too few clips to show.`, { alt: "Accuracy at each native intensity level, one panel per corpus" });
  }

  // ---- 7 Neutral answers
  {
    const g = (ds, m) => get(ds, m);
    const gpt = g("crema_d", "gpt_realtime"), gem = g("crema_d", "gemini-3.8-flash"), omni = g("ravdess", "qwen2_5_omni_7b");
    const rare = cr.filter((r) => r.low_neutral_rate < 0.05);
    const rareAcc = range(rare, "low_accuracy");
    figSlide("Results", "Weak emotion is often answered “neutral”", "emotion_intensity_neutral.png", [
      `Weakest CREMA-D level: ${B.crema_d.neutralMajority} of ${B.crema_d.n} models call most emotional clips “neutral” (GPT Realtime ${pct(gpt.low_neutral_rate)}).`,
      `Stronger takes, fewer “neutral” answers: Gemini ${pct(gem.low_neutral_rate)} → ${pct(gem.high_neutral_rate)} on CREMA-D.`,
      `Models that rarely say “neutral” already score ${pct(rareAcc[0])}–${pct(rareAcc[1])} at the weakest level.`,
    ], `Why does accuracy rise? This figure shows how often a model answers "neutral" for a clip whose label is an emotion. The open circle is the weakest level, the filled circle the strongest.
    On the weakest CREMA-D level, ${B.crema_d.neutralMajority} of ${B.crema_d.n} models call most emotional clips neutral. GPT Realtime answers neutral for ${pct(gpt.low_neutral_rate)} of them, and still ${pct(gpt.high_neutral_rate)} at the high level, which is why its accuracy stays so low. With stronger takes the neutral answers drop sharply for Gemini, from ${pct(gem.low_neutral_rate)} to ${pct(gem.high_neutral_rate)}, and for Qwen2.5-Omni on RAVDESS, from ${pct(omni.low_neutral_rate)} to ${pct(omni.high_neutral_rate)}. So much of the accuracy gain is simply that the clip is no longer heard as neutral.
    The models that rarely answer neutral, ${rare.map((r) => SHORT[r.model]).join(", ")}, are already ${pct(rareAcc[0])} to ${pct(rareAcc[1])} accurate at the weakest CREMA-D level.
    This is not necessarily a model error. A weakly acted emotion may also sound neutral to people; we did not check human perception of these clips here. THAI-SER shows almost no change, matching its flat accuracy.`, { alt: "Share of emotional clips answered neutral at the weakest and strongest level" });
  }

  // ---- 8 Adjusted
  {
    const downs = X.C.filter((r) => X.adj(r) === "down");
    const adjMc = range(mc, "adjusted_change_pp"), obsMc = range(mc, "observed_change_pp");
    figSlide("Results", "The gain survives adjustment for emotion mix and clip length", "emotion_intensity_associations.png", [
      `Clear gain: CREMA-D ${B.crema_d.adjUp}/${B.crema_d.n}, RAVDESS ${B.ravdess.adjUp}/${B.ravdess.n}, MEAD ${B.mead.adjUp}/${B.mead.n}, THAI-SER ${B.thai_ser.adjUp}/${B.thai_ser.n}; ${downs.length} clear drop${downs.length === 1 ? "" : "s"}${downs.length ? ` (${[...new Set(downs.map((r) => DS[r.dataset]))].join(", ")})` : ""}.`,
      `MCAE-SPPS shrinks from ${pp(obsMc[0])}…${pp(obsMc[1])} to ${pp(adjMc[0])}…${pp(adjMc[1])} pp; CI withheld with ${X.clusters.mcae_spps} speakers.`,
      `DailyTalk: ${int(X.scope.dailytalk.n_non_neutral_known)} clips from 2 actors; tiny emotion classes make its slopes unstable.`,
    ], `This is the same layout after adjustment. For each model and corpus we control for the emotion mix, giving every emotion equal weight, and for clip length, and convert the slope back into a weakest-to-strongest change. Coloured cells have a 95% interval that excludes zero. White cells with brackets have an interval that includes zero or no interval at all.
    On the four acted corpora, ${S.actedUp} of ${S.actedFits} results show a clear gain: CREMA-D ${B.crema_d.adjUp} of ${B.crema_d.n}, RAVDESS ${B.ravdess.adjUp} of ${B.ravdess.n}, MEAD ${B.mead.adjUp} of ${B.mead.n}, and THAI-SER ${B.thai_ser.adjUp} of ${B.thai_ser.n}. The only clear drop is ${downs.map((r) => `${SHORT[r.model]} on ${DS[r.dataset]}`).join(" and ")}. Gemma has a positive interval on CREMA-D and RAVDESS, while its MEAD and THAI-SER intervals include zero. Adjustment does not remove most gains on CREMA-D, RAVDESS and MEAD.
    The rated corpora need care. In MCAE-SPPS the raw gains of ${pp(obsMc[0])} to ${pp(obsMc[1])} points shrink to ${pp(adjMc[0])} to ${pp(adjMc[1])} once the emotion is controlled: emotions that listeners rate as strong are also easier. With only ${X.clusters.mcae_spps} speakers we cannot give an interval. DailyTalk has ${int(X.scope.dailytalk.n_non_neutral_known)} emotional clips from two actors, with only one fearful and three disgusted clips. Equal emotion weights give those few clips a large influence, which explains extreme values such as ${pp(get("dailytalk", "meralion_ser_v1").adjusted_change_pp)} for MERaLiON. Its intervals are computed over dialogues and do not show that the result holds for other speakers.`, { alt: "Heat map: class- and duration-adjusted change; white cells have intervals including zero or no interval" });
  }

  // ---- 9 Matched takes
  {
    const s = L.slide("Results", "Same actor, same sentence: the stronger take is still easier");
    prelim(s);
    const img = L.image(s, "emotion_intensity_matched.png", MX, FIG_TOP, 8.2, 4.6, { valign: "top", align: "left", alt: "Heat map: strong minus weak take for the same actor, sentence and emotion" });
    const up = (list) => list.filter((r) => X.mat(r) === "up");
    const crUp = range(up(cr), "matched_difference_pp");
    const thDown = th.filter((r) => X.mat(r) === "down");
    const x = img.x + img.w + 0.3;
    L.bullets(s, [
      `CREMA-D: ${B.crema_d.matUp} of ${B.crema_d.n} models clearly better on the strong take (${pp(crUp[0])} to ${pp(crUp[1])} pp).`,
      `RAVDESS: ${B.ravdess.matUp} of ${B.ravdess.n}.`,
      `THAI-SER: none better; ${thDown.map((r) => `${SHORT[r.model]} ${pp(r.matched_difference_pp)}`).join(", ")}.`,
      `MEAD: only ${B.mead.pairs} take keys, 2 actors. Text equivalence unverified; not shown.`,
    ], { x, y: FIG_TOP + 0.6, w: W - MX - x, h: 4.6, fontSize: 20, gap: 10 });
    L.notes(s, `The strongest check removes speaker and wording completely. In CREMA-D, RAVDESS and THAI-SER the same actor recorded the same sentence with the same emotion at different strengths. For each such set we take the strong take minus the weak take, average over sets with equal weight per emotion, and compute the interval over actors.
    The picture matches the adjusted slide. On CREMA-D, ${B.crema_d.matUp} of ${B.crema_d.n} models are clearly better on the strong take, by ${pp(crUp[0])} to ${pp(crUp[1])} points. On RAVDESS ${B.ravdess.matUp} of ${B.ravdess.n} are clearly better; the others, including emotion2vec+ and Qwen2-Audio, are already close to the ceiling. On THAI-SER no model is better on the high take, and the two Qwen models are slightly worse.
    MEAD would be the fourth corpus, but our benchmark uses a stratified sample of MEAD, and only ${B.mead.pairs} published take keys from two actors kept all three levels. Text equivalence across those takes has not been verified, so we do not show a matched estimate. Matching rules differ by corpus. Even the supported matched comparisons are observational: actors may change more than strength between takes.`);
  }

  // ---- 10 MEAD diagnostic
  {
    const s = L.slide("Results", "On MEAD, Qwen2-Audio’s gain is mostly “angry”; accuracy stays low", { titleSize: 28 });
    prelim(s);
    const m = X.meadClasses();
    const strong = m.ids.length - 1;
    const angryRow = m.classes.indexOf("angry");
    const body = m.classes.map((g) => [{ text: g, bold: true }, ...m.ids.map((i) => {
      const c = m.cell(g, i);
      return { text: pct(c.accuracy), align: "center", fill: g === "angry" ? C.blueFill : undefined };
    })]);
    body.push([{ text: "accuracy (all)", bold: true }, ...m.accuracy.map((a) => ({ text: pct(a), align: "center", bold: true }))]);
    body.push([{ text: "answers “angry”", bold: true }, ...m.angryShare.map((a) => ({ text: pct(a), align: "center", fill: C.orangeFill }))]);
    body.push([{ text: "clips", bold: true }, ...m.total.map((n) => ({ text: int(n), align: "center" }))]);
    L.table(s, ["Recall by emotion", ...m.labels.map((l) => ({ text: l, align: "center" }))], body,
      { x: MX, y: TOP, w: 7.6, colW: [3.25, 1.45, 1.45, 1.45], fontSize: 20, rowH: 0.44 });
    const others = m.classes.filter((g) => g !== "angry");
    const otherMax = Math.max(...others.flatMap((g) => m.ids.map((i) => m.cell(g, i).accuracy)));
    const dis = m.cell("disgusted", 0).accuracy, disS = m.cell("disgusted", strong).accuracy;
    L.bullets(s, [
      `Recall of angry: ${pct(m.cell("angry", 0).accuracy)} → ${pct(m.cell("angry", strong).accuracy)}.`,
      `The other five emotions stay at or below ${pct(otherMax)}; disgusted even falls (${pct(dis)} → ${pct(disS)}).`,
      `At “strong”, ${pct(m.angryShare[strong])} of all answers are “angry”, but only ${pct(m.angryClips[strong] / m.total[strong])} of the clips are angry.`,
    ], { x: MX + 7.9, y: TOP + 0.1, w: CW - 7.9, h: 4.9, fontSize: 20, gap: 12 });
    L.notes(s, `A higher accuracy does not mean that every emotion is recognised better. This table breaks Qwen2-Audio's MEAD result down by the reference emotion; each cell is the share of clips of that emotion classified correctly, its recall.
    Overall accuracy rises only from ${pct(m.accuracy[0])} to ${pct(m.accuracy[strong])}. Almost all of that comes from angry, whose recall rises from ${pct(m.cell("angry", 0).accuracy)} to ${pct(m.cell("angry", strong).accuracy)}. The other five emotions stay low, and disgusted even falls.
    The last row explains it: the model answers angry more often as the expression gets stronger, ${pct(m.angryShare[0])} of all answers at the weak level and ${pct(m.angryShare[strong])} at the strong level, although only about one sixth of the clips are angry. So the stronger expression mainly pushes the model further towards one answer, and angry clips happen to profit. Invalid answers are rare and do not explain the errors.
    We can only describe this pattern; the predictions cannot tell us why the model does it.`);
    L.cite(s, "Source: research_paper/output/intensity_class_rows.csv and intensity_level_rows.csv (MEAD, Qwen2-Audio)");
  }

  // ---- 11 Caveats: rated corpora and design
  {
    const s = L.slide("Caveats", "What these results can and cannot show");
    const level3 = X.smallStrata.find((r) => r.dataset === "dailytalk");
    L.table(s, ["Caveat", "Consequence"], [
      [{ text: "Acted level ≠ perceived strength", bold: true }, "a “high” take may not sound stronger to listeners; we use the corpus label as given"],
      [{ text: "Observational, not an experiment", bold: true }, "levels can differ in more than strength; adjustment and matched takes reduce, not remove, this"],
      [{ text: `MCAE-SPPS: ${X.clusters.mcae_spps} speakers`, bold: true }, "no interval; listener ratings may also track how recognisable a clip is"],
      [{ text: "DailyTalk: 2 actors", bold: true }, `${int(X.scope.dailytalk.n_non_neutral_known)} clips, ${level3 ? level3.n : "few"} at level 3; tiny classes make slopes unstable`],
      [{ text: "Many comparisons", bold: true }, "intervals are exploratory and not corrected for multiple testing"],
      [{ text: "Category accuracy only", bold: true }, "no model was asked to rate intensity; that remains untested"],
    ], { y: TOP, colW: [4.3, 7.83], fontSize: 20, rowH: 0.62 });
    L.notes(s, `Before the summary, the limits of what we can conclude.
    First, an acted level is an instruction to the actor, not a measurement of how strong the clip sounds. Some low takes may sound strong and the other way round. We use the labels as the corpora give them.
    Second, this is observational. Even within one corpus, the levels may differ in other ways, for example pitch range or how carefully the actor speaks. Controlling for emotion and length, and comparing matched takes, reduces this problem but cannot remove it.
    Third, the two rated corpora have very few speakers: ${X.clusters.mcae_spps} in MCAE-SPPS and two actors in DailyTalk. In MCAE-SPPS a high listener rating may partly mean "easy to recognise", which would make the link with accuracy circular. DailyTalk has ${int(X.scope.dailytalk.n_non_neutral_known)} emotional clips, only ${level3 ? level3.n : "a few"} at its top level, and one fearful and three disgusted clips, which can swing the adjusted slopes.
    Fourth, we compare many models and corpora, and the intervals are not corrected for that.
    Finally, we measured category accuracy only. Whether a model can rate intensity is a different question that this analysis does not answer.`);
  }

  // ---- 12 Summary
  {
    const s = L.slide("Summary", "Summary");
    const items = [
      ["Most acted comparisons improve", `${S.actedUp}/${S.actedFits} adjusted results are clearly positive. CREMA-D and RAVDESS also support matched takes. MEAD: ${B.mead.adjUp}/${B.mead.n} adjusted gains, insufficient matched support.`],
      ["Not universal", `THAI-SER shows no gain (${rangeText(th, "observed_change_pp")})${B.thai_ser.adjDown ? `; ${B.thai_ser.adjDown === 1 ? "one model is" : `${B.thai_ser.adjDown} models are`} even slightly worse` : ""}.`],
      ["Observed patterns", "weak emotion is often answered “neutral”; models near the ceiling gain little; Qwen2-Audio’s MEAD gain is mostly “angry”."],
      ["Practical implication", "report accuracy by native intensity level. Strong acting may overstate accuracy on mild speech in a target domain."],
    ];
    items.forEach(([head, body], i) => {
      const y = TOP + i * 1.12;
      s.addShape(p.ShapeType.ellipse, { x: MX, y: y + 0.08, w: 0.62, h: 0.62, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
      L.text(s, String(i + 1), MX, y + 0.08, 0.62, 0.62, { fontSize: 24, bold: true, color: C.white, align: "center", valign: "middle" });
      L.text(s, [{ text: head + ": ", options: { bold: true } }, { text: body }], MX + 0.85, y, CW - 0.85, 1.0, { fontSize: 22, valign: "middle", ls: 1.0 });
    });
    L.text(s, `${S.models} models across all six eligible corpora, including Gemma 4 E4B.`,
      MX, TOP + 4.6, CW, 0.45, { fontSize: 20, bold: true, color: C.ink });
    L.notes(s, `Four points to take away.
    One: most acted comparisons are positive after controlling for emotion mix and clip length. ${S.actedUp} of ${S.actedFits} acted results show a clear gain. CREMA-D and RAVDESS also support the matched-take comparison. MEAD has ${B.mead.adjUp} positive intervals out of ${B.mead.n}, but only three take keys from two actors and unverified text equivalence, so it provides no reliable matched estimate.
    Two: it is not universal. THAI-SER shows no gain at all, and Qwen2-Audio is even slightly worse on its high takes.
    Three: the gain has simple mechanisms. Many models answer neutral when the emotion is weak; models that are already near the ceiling have little room to gain; and on MEAD the gain of Qwen2-Audio comes almost entirely from angry.
    Four, the practical point: benchmark accuracy depends on the corpus intensity level. Strong acting may overstate what to expect on mild speech in a different target domain. This analysis does not directly measure that transfer, so report accuracy per intensity level where the corpus allows it.
    ${gemmaNote}`);
  }

  // ---- 13 Limitations and next steps
  {
    const s = L.slide("Next steps", "Limitations and next steps");
    L.table(s, ["Limitation today", "Next step"], [
      ["Acted levels, not perceived strength", "compare with listener ratings where a corpus has them"],
      ["Only 6 of 16 corpora, 3 languages", "add corpora with intensity labels and more speakers"],
      ["Category accuracy only", "save MERaLiON’s arousal output and test it against intensity"],
      ["Exploratory comparisons", "confirm the patterns on a separate listener-labelled sample"],
      ["Observational design", "a small controlled listening set to test cause and effect"],
    ], { y: TOP, colW: [5.6, 6.53], fontSize: 20, rowH: 0.62 });
    L.notes(s, `What we would do next.
    The acted levels are instructions. Where a corpus also has listener ratings of the same clips, we can check whether perceived strength gives the same picture as the acted level.
    Only six corpora have intensity labels, in English, Thai and Mandarin, and the two rated corpora have very few speakers. More corpora with intensity labels and more speakers would make the result more general.
    We only measured category accuracy. MERaLiON-SER-v1 also has a valence, arousal and dominance output, but our adapter saves only the emotion probabilities, so those outputs were never stored. Saving them would let us test, within one emotion, whether its arousal follows the intensity label. That is a proposal, not something done here.
    All ten benchmark models now have predictions on all sixteen corpora, including Gemma 4 E4B. This analysis uses the six corpora with eligible intensity labels. The rebuild command recomputes every figure and number from the saved predictions.
    Finally, everything here is observational. A small set of clips presented in a controlled way would be needed to say that stronger expression causes better recognition.`);
  }

  // ---- Appendix (hidden)
  {
    const s = L.slide("Appendix", "Adjusted slopes per native step with 95% intervals", { appendix: true });
    const img = L.image(s, "emotion_intensity_forest.png", MX, FIG_TOP, CW, 4.6, { valign: "top", alt: "Adjusted slopes with 95% intervals, one panel per corpus" });
    L.bullets(s, [
      "Per native step (MCAE-SPPS: per listener-rating point); scales differ, so compare within a panel only.",
      `No interval below 20 source clusters (MCAE-SPPS: ${X.clusters.mcae_spps}). DailyTalk uses dialogues from 2 actors.`,
    ], { y: img.y + img.h + 0.1, h: 6.98 - (img.y + img.h + 0.1), fontSize: 20, gap: 3 });
    L.notes(s, `For readers who want the intervals: each point is the class-balanced, duration-adjusted slope in percentage points per native step, with its 95% interval clustered by speaker or dialogue. The heat map on the adjusted slide multiplies these slopes by the distance between the weakest and strongest level. MCAE-SPPS is per listener-rating point and has no interval because it has only ${X.clusters.mcae_spps} speakers. The x axes differ between panels because the scales differ.`);
  }
  {
    const s = L.slide("Appendix", "Accuracy at the weakest and strongest level (%)", { appendix: true });
    const order = Object.keys(X.NAMES).filter((m) => X.models.includes(m) || m === "gemma4_e4b");
    const body = order.map((m) => [{ text: X.NAMES[m], bold: true },
      ...Object.keys(DS).map((ds) => { const r = get(ds, m); return { text: r ? `${Math.round(100 * r.low_accuracy)} → ${Math.round(100 * r.high_accuracy)}` : "—", align: "center" }; })]);
    L.table(s, ["Model", ...Object.keys(DS).map((ds) => ({ text: DS[ds], align: "center" }))], body,
      { y: TOP - 0.05, colW: [3.13, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], fontSize: 18, rowH: 0.42 });
    L.notes(s, `The numbers behind the main heat map: accuracy in percent at the weakest and at the strongest shown level, for each model and corpus. The levels compared are those of the main heat map: ${Object.keys(DS).map((ds) => `${DS[ds]} ${S.byDataset[ds].low} to ${S.byDataset[ds].high}`).join("; ")}. Full tables with every level, the adjusted effects and their intervals are in the analysis report, intensity_en.md.`);
  }
  {
    const s = L.slide("Appendix", "How each corpus defines intensity", { appendix: true });
    L.table(s, ["Corpus", "Definition and restrictions"], [
      [{ text: "CREMA-D", bold: true }, "LO / MD / HI in the file name, only for one sentence (IEO); all other clips are XX (unspecified)"],
      [{ text: "RAVDESS", bold: true }, "normal / strong in the file name; neutral exists only at normal; calm is merged into neutral"],
      [{ text: "MEAD", bold: true }, "level_1 to level_3 folders; neutral has only level_1; our 8,000-clip sample keeps few matched sentences"],
      [{ text: "THAI-SER", bold: true }, "Low / High for scripted turns only; improvised turns have no level"],
      [{ text: "MCAE-SPPS", bold: true }, "Intensity: mean of 40 listeners’ ratings, 1–9 (not the meanIntensity field, which is loudness in dB)"],
      [{ text: "DailyTalk", bold: true }, "ECSS levels 1–3 added by another group; an observed rating, not an acting instruction"],
    ], { y: TOP - 0.05, colW: [2.2, 9.93], fontSize: 20, rowH: 0.66 });
    L.notes(s, `The exact definitions, from the corpus documentation and file formats, as recorded in the intensity inventory manifest. Neutral clips and clips without a level are never used as weak emotion. The MCAE-SPPS metadata also has a field called meanIntensity, which is loudness in decibels, not emotional strength; we use the listener rating. DailyTalk's levels come from the ECSS project and are described there as weak, medium and strong.`);
    L.cite(s, "Sources: research_paper/output/intensity_scope_rows.csv; ECSS annotations", [["arXiv 2312.11947", "https://arxiv.org/abs/2312.11947"]]);
  }
  {
    const s = L.slide("Appendix", "Can a model estimate intensity itself? Not tested yet", { appendix: true });
    L.bullets(s, [
      "All benchmark models output an emotion category; none outputs a generic intensity.",
      "MERaLiON-SER-v1 also has valence, arousal and dominance outputs (0–1), but our adapter saves only the category probabilities.",
      "Arousal is not intensity: strong sadness can be calm. A test would compare arousal with the intensity level within one emotion.",
      "This needs a new run that saves those outputs; it is a proposal, not done here.",
    ], { y: TOP, h: 4.8, fontSize: 22, gap: 12 });
    L.notes(s, `A natural follow-up question is whether a model can tell how strong an emotion is. None of the benchmark models has a general intensity output. MERaLiON-SER-v1 has, besides the seven emotion scores, three dimensional outputs between 0 and 1 for valence, arousal and dominance. Our project adapter keeps only the emotion probabilities, so the existing prediction files contain no arousal values, and they cannot be recovered from the probabilities.
    Arousal is also not the same as intensity: arousal runs from calm to excited, and a strongly sad voice can be calm. A fair test would save the arousal output in a new run and check, within one emotion at a time, whether it rises with the intensity level. That is a proposal for later.`);
  }
  {
    const s = L.slide("Appendix", "Gemma 4 E4B status and how to rebuild", { appendix: true });
    L.bullets(s, [
      `Complete benchmark: 10 models on 16 datasets. This deck uses ${S.models} models on the six intensity corpora, including Gemma 4 E4B.`,
      { runs: [{ text: "One command: ", options: { bold: true } }, { text: "research_paper/presentation/intensity/rebuild.sh", options: { fontFace: L.MONO } }] },
      { t: "It re-reads the saved predictions, recomputes the analysis, redraws the figures, rewrites intensity_en.md / _zh.md, rebuilds and renders this deck.", sub: [] },
      "After a rebuild, check that the slide titles still match the numbers (README).",
    ], { y: TOP, h: 4.8, fontSize: 22, gap: 12 });
    L.notes(s, `Gemma 4 E4B is the tenth benchmark model. The final predictions now cover all sixteen datasets, including all six corpora with eligible intensity labels. Gemma joins the comparisons on the same saved benchmark clips as the other models. One command updates the analysis, figures, reports and presentation: rebuild.sh in this folder. The README explains it. After any refresh, review fixed wording against the recomputed numbers.`);
  }
};

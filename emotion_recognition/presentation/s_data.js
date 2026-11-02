// Part 2: datasets, one slide each with audio, then subjectivity
module.exports = function (L, A) {
  const { C, W, H, MX, FONT } = L;
  const p = L.pptx;
  const K = require("./clips")(L);
  const SEC = "2 · Public Datasets";
  const clips = (A.manifest && A.manifest.audio) || [];
  const ex = (sub) => clips.filter((c) => c.group === "dataset_example" && c.subgroup === sub);

  // ---- Survey and selection
  {
    const s = L.slide(SEC, "Dataset Survey and Selection");
    // funnel
    const f = [["37", "corpora surveyed", "papers, project pages, HF, Zenodo, OSF"], ["9", "downloaded", "licence and access checked"], ["6", "evaluated", "ESD left out (licence), THAI-SER and nEMO (language)"]];
    f.forEach(([n, t, d], i) => {
      const w = 3.6 - i * 0.55, x = MX + (3.6 - w) / 2, y = 1.55 + i * 1.15;
      s.addText(`${n}  ${t}`, { x, y, w, h: 0.9, shape: p.ShapeType.rect, fill: { color: [C.ink, "34507A", C.accent][i] }, line: { color: C.white, width: 0 }, fontFace: FONT, fontSize: 18, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, d, MX, y + 0.92, 3.6, 0.22, { fontSize: 10, color: C.muted, align: "center" });
    });
    L.text(s, "Selection criteria", MX, 5.15, 3.6, 0.3, { fontSize: 14, bold: true, color: C.ink });
    L.bullets(s, ["Same sentences in every emotion (words give no clue)", "Good labels, ideally listener-validated", "Size, and a clear licence and download"], { x: MX, y: 5.45, w: 3.7, h: 1.5, fontSize: 13, gap: 3 });
    const rows = [
      ["CREMA-D", "US English", "91", "7,442", "5.3", "6", "12 sentences", "actor + ≈9 listeners", "ODbL"],
      ["MCAE-SPPS", "Mandarin", "6", "6,675", "2.3", "7", "200 sentences", "actor + 40 listeners", "CC BY-NC"],
      ["MESS", "US English", "6", "1,800", "1.3", "4", "90 sentences", "actor", "CC BY"],
      ["RAVDESS", "N. Am. English", "24", "1,440", "1.5", "7", "2 sentences", "actor", "CC BY-NC-SA"],
      ["TESS", "Canadian Eng.", "2", "2,800", "1.6", "7", "200 words", "actor", "CC BY-NC"],
      ["JL-Corpus", "NZ English", "4", "1,200", "0.7", "5", "15 sentences", "actor", "CC0"],
    ];
    L.table(s, ["Dataset", "Language", "Speakers", "Clips", "Hours", "Emotions", "Parallel text", "Labels", "Licence"], rows.map((r) => r.map((c, j) => ({ text: c, bold: j === 0, options: { align: j >= 2 && j <= 5 ? "center" : "left" } }))),
      { x: 4.6, y: 1.55, w: 8.13, colW: [1.05, 1.15, 0.75, 0.7, 0.6, 0.75, 1.05, 1.18, 0.9], fontSize: 11, rowH: 0.5 });
    L.bullets(s, [
      "21,357 clips, 12.7 hours, 133 speakers, 2 languages",
      "None is spontaneous speech; MCAE-SPPS and MESS are in no model's training data",
    ], { x: 4.6, y: 5.4, w: 8.1, h: 1.4, fontSize: 15 });
    L.notes(s, `We surveyed 37 public corpora, downloaded nine and evaluate six. The deciding criterion is parallel text: the same sentence in every emotion, so the words cannot give the answer away.
    Two corpora, CREMA-D and MCAE-SPPS, also have listener votes per clip, which we use to show how subjective the task is. MCAE-SPPS and MESS are the two corpora no model has trained on; they are the most honest test.`);
  }

  // ---- Label space
  {
    const s = L.slide(SEC, "Unified Label Space");
    const emos = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised", "unknown / other"];
    const D_ = "●";
    const cov = {
      "CREMA-D": { angry: D_, disgusted: D_, fearful: D_, happy: D_, neutral: D_, sad: D_ },
      "MCAE-SPPS": { angry: D_, disgusted: D_, fearful: D_, happy: D_, neutral: D_, sad: D_, surprised: D_ },
      MESS: { angry: D_, happy: D_, neutral: "calm", sad: D_ },
      RAVDESS: { angry: D_, disgusted: D_, fearful: D_, happy: D_, neutral: "neutral + calm", sad: D_, surprised: D_ },
      TESS: { angry: D_, disgusted: D_, fearful: D_, happy: D_, neutral: D_, sad: D_, surprised: "pleasant surprise" },
      "JL-Corpus": { angry: D_, happy: "happy + excited*", neutral: D_, sad: D_, surprised: "excited*" },
    };
    const hdr1 = [{ text: "", options: { fill: { color: C.white } } }, { text: "Project label set: the 7 emotion classes of emotion2vec+, plus unknown / other", options: { colspan: 8, align: "center", fill: { color: C.ink }, color: C.white, bold: true, fontSize: 11 } }];
    const hdr2 = [{ text: "Dataset", options: { bold: true, color: C.white, fill: { color: "34507A" } } }, ...emos.map((e) => ({ text: e, options: { bold: true, color: C.white, fill: { color: e === "unknown / other" ? "6B7785" : "34507A" }, align: "center", fontSize: 10 } }))];
    const rows = Object.entries(cov).map(([d, m]) => [{ text: d, options: { bold: true } }, ...emos.map((e) => {
      const v = m[e] || (e === "unknown / other" ? "–" : "");
      return { text: v, options: { align: "center", color: v === D_ ? C.accent : (v === "–" ? C.faint : C.accent2), fontSize: v === D_ ? 13 : 9, bold: v !== D_ && v !== "–" } };
    })]);
    const cell = (c) => ({ text: c.text, options: { fontFace: FONT, fontSize: 11, color: C.ink, valign: "middle", margin: [2, 3, 2, 3], ...c.options } });
    s.addTable([hdr1, hdr2, ...rows].map((r) => r.map(cell)), { x: MX, y: 1.45, w: 6.75, colW: [1.1, 0.64, 0.76, 0.66, 0.72, 0.72, 0.55, 0.72, 0.88], rowH: [0.32, 0.36, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42], border: { type: "solid", color: C.line, pt: 0.75 }, autoPage: false });
    const lg = [
      [{ text: "●  ", options: { color: C.accent, bold: true } }, { text: "the dataset's own label is identical to the project class" }],
      [{ text: "orange  ", options: { color: C.accent2, bold: true } }, { text: "the dataset's own label name, mapped to the project class of that column (e.g. MESS \"calm\" is scored as neutral)" }],
      [{ text: "*  ", options: { color: C.accent2, bold: true } }, { text: "JL-Corpus \"excited\" is scored as correct if the prediction is happy or surprised" }],
      [{ text: "unknown / other  ", options: { color: C.muted, bold: true } }, { text: "no corpus uses it; the models' extra classes (emotion2vec+ other, <unk>; Vox-Profile contempt, other) are excluded, since the prediction is the most probable of the dataset's own emotions" }],
      [{ text: "Shared by all six corpora:  ", options: { bold: true } }, { text: "angry, happy, neutral, sad (basis of the 4-class comparison in Part 4)" }],
    ];
    lg.forEach((runs, i) => s.addText(runs.map((r) => ({ text: r.text, options: { fontFace: FONT, fontSize: 11, color: C.text, ...r.options } })), { x: MX, y: 4.8 + i * 0.42, w: 6.75, h: 0.42, margin: 0, valign: "top", fit: "shrink" }));
    L.image(s, "assets/fig/emotion_counts.png", 7.55, 1.4, W - MX - 7.55, 4.9);
    L.text(s, "Clips per emotion in each corpus: balanced by design, unlike call-centre audio", 7.55, 6.35, W - MX - 7.55, 0.4, { fontSize: 12, color: C.muted, align: "center" });
    L.notes(s, `All dataset labels are mapped to the project label set: the seven emotion classes of emotion2vec+, plus unknown or other, which no corpus uses. A dot means the dataset uses the same label. Orange text shows the dataset's own label name where it had to be mapped: calm is scored as neutral in MESS and RAVDESS, pleasant surprise as surprised in TESS, and JL-Corpus excited is counted as correct if a model predicts happy or surprised.
    Models with additional classes, such as emotion2vec+ "other" or Vox-Profile "contempt", never have those classes chosen, because the prediction is restricted to the dataset's own emotions. On the right, the corpora are balanced by design, which is very different from real calls, where neutral dominates.`);
  }

  // ---- One slide per dataset
  const D = [
    { sub: "crema_d", title: "Dataset: CREMA-D",
      facts: [["Language", "US English"], ["Speakers", "91 actors (48 m, 43 f, aged 20–74)"], ["Size", "7,442 clips · 5.3 h · 16 kHz"], ["Emotions", "6: no surprised"], ["Design", "12 sentences × 6 emotions"], ["Labels", "actor's intent + crowd votes (≈9 listeners per clip, voice only; 2,443 raters)"], ["Licence", "ODbL"], ["In training of", "emotion2vec+, MERaLiON"]],
      note: "The most used benchmark. Listener votes show what people actually hear (next slides).",
      notes: `CREMA-D, 2014: 91 actors of different ages and ethnicities, each saying 12 sentences in six emotions. It is the only large corpus with crowd-sourced votes for every clip, which makes it valuable for studying subjectivity. Here one actress says "I'm on my way to the meeting" in four emotions; listeners agree with her on all four. Let's listen.` },
    { sub: "mcae_spps", title: "Dataset: MCAE-SPPS",
      facts: [["Language", "Mandarin Chinese"], ["Speakers", "6 professional actors (3 m, 3 f)"], ["Size", "6,675 clips · 2.3 h · 48 kHz"], ["Emotions", "all 7"], ["Design", "200 sentences × 7 emotions × 6 actors"], ["Labels", "actor's intent; kept only if listeners' top choice (40 native listeners, 720 in total)"], ["Licence", "CC BY-NC 4.0 (Sci Data 2026)"], ["In training of", "none"]],
      note: "Newest corpus, unseen by every model: the cleanest test of generalisation.",
      notes: `MCAE-SPPS was published in Scientific Data in 2026: six professional Mandarin actors, 200 sentences, all seven emotions, and 40 native listeners per clip. Clips were kept only if listeners recognised the intended emotion more than any other. No model has seen it. The sentence here means "They attended a lecture".` },
    { sub: "mess", title: "Dataset: MESS",
      facts: [["Language", "US English"], ["Speakers", "6 (3 m, 3 f)"], ["Size", "1,800 clips · 1.3 h · 24.4 kHz"], ["Emotions", "4: angry, calm (→ neutral), happy, sad"], ["Design", "90 sentences × 4 emotions"], ["Labels", "actor's intent (listener ratings in the paper, not public)"], ["Licence", "CC BY 4.0 (2019)"], ["In training of", "none"]],
      note: "Every clip is normalised to the same loudness (≈ −27 dBFS RMS): loudness gives no clue, only pitch, timbre and timing.",
      notes: `MESS, 2019: six speakers, 90 sentences, four emotions, with calm instead of neutral. One detail matters: all clips were normalised to the same loudness, so an angry clip is not louder than a sad one. Models that rely on loudness will struggle. Listen to the same speaker and sentence in four emotions.` },
    { sub: "ravdess", title: "Dataset: RAVDESS",
      facts: [["Language", "North American English"], ["Speakers", "24 actors (12 m, 12 f)"], ["Size", "1,440 clips · 1.5 h · 48 kHz"], ["Emotions", "8 → 7 (calm → neutral)"], ["Design", "2 sentences × 8 emotions × 2 intensities × 2 takes"], ["Labels", "actor's intent (validated: 10 ratings per file, 247 raters)"], ["Licence", "CC BY-NC-SA 4.0 (2018)"], ["In training of", "emotion2vec+"]],
      note: "Many speakers, few words: tests voice variation more than lexical variety.",
      notes: `RAVDESS, 2018: 24 actors, but only two sentences, "Kids are talking by the door" and "Dogs are sitting by the door", at normal and strong intensity. The strong versions are very expressive. emotion2vec+ has seen it.` },
    { sub: "tess", title: "Dataset: TESS",
      facts: [["Language", "Canadian English"], ["Speakers", "2 actresses, aged 26 and 64"], ["Size", "2,800 clips · 1.6 h · 24.4 kHz"], ["Emotions", "7 (pleasant surprise → surprised)"], ["Design", "\"Say the word ___\" × 200 words × 7 emotions"], ["Labels", "actor's intent"], ["Licence", "CC BY-NC 4.0"], ["In training of", "emotion2vec+"]],
      note: "Very clean and very narrow: two voices only, so one voice misjudged shifts a whole class.",
      notes: `TESS, from Toronto: two actresses, 26 and 64 years old, saying "Say the word" followed by one of 200 words. Very clean recordings, very exaggerated emotions, but only two voices. Remember this when we look at MERaLiON's confusion matrix.` },
    { sub: "jl_corpus", title: "Dataset: JL-Corpus",
      facts: [["Language", "New Zealand English"], ["Speakers", "4 (2 m, 2 f)"], ["Size", "1,200 clips evaluated (of 2,400) · 0.7 h · 44.1 kHz"], ["Emotions", "5 primary: angry, excited, happy, neutral, sad"], ["Design", "15 sentences × 5 emotions"], ["Labels", "actor's intent (≈70% listener recognition for primary emotions)"], ["Licence", "CC0"], ["In training of", "emotion2vec+, Melati"]],
      note: "Secondary emotions (anxious, apologetic, …) are not in our label set and are not scored.",
      notes: `JL-Corpus, New Zealand English: four speakers and fifteen sentences. It also has secondary emotions such as apologetic or concerned, interesting for call centres, but outside our seven classes, so we score only the five primary emotions. Excited counts as happy or surprised.` },
  ];
  D.forEach((d) => {
    const s = L.slide(SEC, d.title);
    // facts
    d.facts.forEach(([k, v], i) => {
      const y = 1.5 + i * 0.52;
      L.text(s, k, MX, y, 1.45, 0.48, { fontSize: 13, bold: true, color: C.muted, valign: "middle" });
      L.text(s, v, MX + 1.5, y, 3.95, 0.48, { fontSize: 14, color: C.ink, valign: "middle" });
      s.addShape(p.ShapeType.line, { x: MX, y: y + 0.5, w: 5.45, h: 0, line: { color: C.line, width: 0.75 } });
    });
    L.text(s, d.note, MX, 5.75, 5.45, 0.9, { fontSize: 13, italic: true, color: C.accent });
    // clips
    const cs = ex(d.sub);
    const x0 = 6.35, w0 = W - MX - x0;
    if (cs.length) {
      const c0 = cs[0];
      const sent = c0.sentence ? `"${c0.sentence.replace(/^"|"$/g, "")}"` : `sentence ${c0.sentence_id || ""} (text not published)`;
      L.text(s, `Same speaker: ${c0.speaker} · same sentence: ${sent} · click ▶ to listen`, x0, 1.45, w0, 0.45, { fontSize: 12, color: C.muted });
      cs.slice(0, 4).forEach((c, i) => K.exampleRow(s, c, x0, 1.95 + i * 1.12, w0));
      L.text(s, "Predictions: green = matches the actor, red = differs", x0, 6.45, w0, 0.3, { fontSize: 11, color: C.faint });
    }
    L.notes(s, d.notes);
  });

  // ---- Subjectivity 1: CREMA-D numbers
  const cl = A.stats && A.stats.crema_d_listeners;
  {
    const s = L.slide(SEC, "Inter-Rater Agreement: Actor vs Listeners (CREMA-D)");
    L.image(s, "assets/fig/crema_acted_vs_listener.png", MX, 1.4, 5.6, 4.6);
    L.text(s, "Rows: what the actor portrayed · columns: what the listeners' majority heard (voice only)", MX, 6.05, 5.6, 0.5, { fontSize: 11, color: C.muted });
    L.stat(s, "41.6%", "of clips: listener majority = actor's emotion (8.7% ties)", 6.6, 1.45, 3.0, { size: 36 });
    L.stat(s, "14.8%", "of clips: not a single listener chose the actor's emotion", 9.75, 1.45, 3.0, { size: 36, color: C.accent2 });
    L.bullets(s, [
      { t: "Neutral is easy (96%), angry moderate (61%)" },
      { t: "Sad (16%), happy (26%), disgusted (27%), fearful (32%) are mostly heard as neutral" },
      { t: "With the face visible, agreement rises to 76.5% (CREMA-D paper): voice alone carries less emotion than actors intend" },
      { t: "Implication: intended and perceived emotion are distinct reference labels and must be chosen explicitly", b: true, color: C.accent2 },
    ], { x: 6.6, y: 3.25, w: 6.15, h: 3.6, fontSize: 16 });
    L.notes(s, `This is the key slide about subjectivity. In CREMA-D every clip was rated by about nine people who only heard the voice. The majority agreed with the actor on only 42 percent of clips, and on 15 percent of clips not one listener chose the intended emotion.
    The pattern is systematic: subtle emotions such as sadness, happiness, disgust and fear are mostly heard as neutral. With the video, agreement rises to 77 percent. So the voice carries less emotion than the actors think they put in. This raises the question for our own labels: do we want the speaker's intent, or what a listener perceives? For a customer-experience use case, perception is arguably what matters.`);
  }

  // ---- Subjectivity 2: listen
  {
    const s = L.slide(SEC, "Intended vs Perceived Emotion: Audio Examples (CREMA-D)");
    const cs = clips.filter((c) => c.group === "subjectivity" && c.subgroup === "crema_d");
    const OBS = {
      "1014_WSI_HAP_XX": "All 10 listeners heard angry; MERaLiON agrees with them, emotion2vec+ with the actor",
      "1089_TAI_FEA_XX": "All 10 listeners heard sad; both top models still predict the actor's fearful",
      "1029_IEO_DIS_HI": "9 of 10 listeners heard angry; both top models predict the actor's disgusted",
      "1082_TAI_SAD_XX": "All 10 listeners heard neutral; emotion2vec+ predicts sad, MERaLiON happy",
    };
    // columns: play | clip | actor | listeners | model predictions | observation
    const cols = [[MX, 0.85, ""], [MX + 0.85, 2.45, "Clip"], [MX + 3.3, 1.35, "Actor (intended)"], [MX + 4.65, 2.6, "Listeners (perceived, n = 10)"], [MX + 7.4, 2.45, "Model predictions"], [MX + 9.95, W - MX - (MX + 9.95), "Observation"]];
    cols.forEach(([x, w, t]) => L.text(s, t, x + 0.08, 1.4, w - 0.1, 0.3, { fontSize: 12, bold: true, color: C.muted }));
    s.addShape(p.ShapeType.line, { x: MX, y: 1.72, w: W - 2 * MX, h: 0, line: { color: C.line, width: 1 } });
    const SH = { "emotion2vec+ large": "emotion2vec+", "MERaLiON-SER-v1": "MERaLiON", "Vox-Profile WavLM": "Vox WavLM" };
    cs.forEach((c, i) => {
      const y = 1.82 + i * 1.27, h = 1.15, stem = c.path.split("__")[1].replace(".mp3", "");
      s.addShape(p.ShapeType.rect, { x: MX, y, w: W - 2 * MX, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.audio(s, "assets/" + c.path, MX + 0.12, y + 0.24, 0.65);
      L.text(s, `"${c.sentence}"`, cols[1][0] + 0.08, y + 0.18, cols[1][1] - 0.1, 0.45, { fontSize: 13, italic: true, color: C.ink });
      L.text(s, `speaker ${c.speaker}`, cols[1][0] + 0.08, y + 0.65, cols[1][1] - 0.1, 0.3, { fontSize: 11, color: C.muted });
      L.tag(s, c.acted_label, cols[2][0] + 0.08, y + 0.42, { w: 1.15 });
      K.voteBar(s, c, cols[3][0] + 0.08, y + 0.42, cols[3][1] - 0.2, 0.32);
      ["emotion2vec+ large", "MERaLiON-SER-v1", "Vox-Profile WavLM"].forEach((m, j) => {
        const [lab, pr] = c.predictions[m];
        const col = lab === c.acted_label ? C.good : (lab === c.listener_majority ? C.accent2 : C.muted);
        const yy = y + 0.12 + j * 0.3;
        L.text(s, SH[m] + ((c.models_in_training_data || []).includes(m) ? " †" : ""), cols[4][0] + 0.08, yy, 1.2, 0.3, { fontSize: 12, color: C.text, valign: "middle" });
        L.text(s, `${lab} ${pr.toFixed(2)}`, cols[4][0] + 1.3, yy, cols[4][1] - 1.3, 0.3, { fontSize: 12, bold: true, color: col, valign: "middle" });
      });
      L.text(s, OBS[stem] || c.why_interesting, cols[5][0] + 0.08, y + 0.1, cols[5][1] - 0.15, h - 0.2, { fontSize: 12, color: C.text, valign: "middle" });
    });
    L.source(s, "Prediction colours: green = the actor's emotion · orange = the listeners' majority · grey = neither. † = CREMA-D is in the model's training data.");
    L.notes(s, `Four clips where the actor and the listeners disagree completely. Play each and decide for yourself before revealing the labels.
    First: acted happy, all ten listeners heard angry. Second: acted fearful, all heard sad. Third: acted disgusted, nine heard angry. Fourth: acted sad, all heard neutral.
    Notice that emotion2vec+ and MERaLiON, both trained on CREMA-D's acted labels, often side with the actor, even when every human disagrees.`);
  }

  // ---- Subjectivity 3: MCAE-SPPS
  {
    const s = L.slide(SEC, "Listener Recognition Rates (MCAE-SPPS)");
    L.image(s, "assets/fig/mcae_recognition_hist.png", MX, 1.45, 5.9, 3.55);
    const ml = A.stats && A.stats.mcae_spps_listeners;
    L.bullets(s, [
      { runs: [{ text: "How to read: ", options: { bold: true } }, { text: "each clip was rated by 40 listeners; its recognition rate is the share of them who chose the intended emotion. Each bar counts the clips with that rate" }] },
      "Mean 76%: a typical clip is recognised by about 31 of 40 listeners (range 47.5–100%); only 36 of 6,675 clips by all 40",
      "Lowest: disgusted 63%, fearful 65%, happy 69% · highest: neutral 89%, sad 82%",
    ], { x: MX, y: 5.05, w: 5.9, h: 1.85, fontSize: 13, gap: 4 });
    const cs = clips.filter((c) => c.group === "subjectivity" && c.subgroup === "mcae_spps");
    cs.forEach((c, i) => {
      const y = 1.5 + i * 1.75, x = 6.85, w = W - MX - x;
      s.addShape(p.ShapeType.rect, { x, y, w, h: 1.6, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.audio(s, "assets/" + c.path, x + 0.15, y + 0.2, 0.65);
      L.text(s, c.sentence, x + 1.0, y + 0.1, w - 1.1, 0.3, { fontSize: 12, color: C.muted });
      L.text(s, "Actor", x + 1.0, y + 0.45, 0.6, 0.3, { fontSize: 12, color: C.muted, valign: "middle" });
      L.tag(s, c.acted_label, x + 1.6, y + 0.45, { w: 1.15 });
      const n = K.votes(c).reduce((a2, b2) => a2 + b2[1], 0);
      L.text(s, `Listeners (n = ${n})`, x + 1.0, y + 0.86, 1.45, 0.3, { fontSize: 11, color: C.muted, valign: "middle" });
      K.voteBar(s, c, x + 2.45, y + 0.87, w - 2.6, 0.28);
      s.addText([{ text: "Models   ", options: { color: C.muted, fontSize: 11 } }, ...K.predRuns(c, ["emotion2vec+ large", "MERaLiON-SER-v1"], c.acted_label, 12)], { x: x + 1.0, y: y + 1.24, w: w - 1.1, h: 0.28, fontFace: FONT, margin: 0, fit: "shrink" });
    });
    L.notes(s, `How to read the histogram: every clip was rated by 40 native listeners. For each clip we compute the share of these 40 who chose the emotion the actor intended; this is the clip's recognition rate. The histogram counts clips per rate. For example, the bar at 50 percent counts 177 clips that were recognised by only 20 of the 40 listeners. The dashed line is the mean over all clips, 76 percent, so a typical clip is recognised by about 31 of 40 listeners. It is not the share of listeners who agree with a group of clips.
    MCAE-SPPS is a curated corpus: only clips where listeners chose the intended emotion more than any other were kept. Even so, the average clip convinces 76 percent of 40 listeners, and only 36 clips convince all of them.
    On the right, two borderline clips, where under half of the listeners heard the intended emotion, and a clear one, where all 40 agree and all five models are right. Ambiguity for humans and errors for models go together.`);
  }
};

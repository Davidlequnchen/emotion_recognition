// Part 5: listening to errors
module.exports = function (L, A) {
  const { C, W, MX } = L;
  const K = require("./clips")(L);
  const SEC = "5 · Error Analysis";
  const clips = ((A.manifest && A.manifest.audio) || []).filter((c) => c.group === "failure");
  const by = (sub) => clips.filter((c) => c.subgroup === sub);

  {
    const s = L.slide(SEC, "Error Analysis: Overview");
    L.bullets(s, [
      { t: "Four kinds of disagreement", b: true, sub: [
        "Models side with the listeners against the actor (CREMA-D)",
        "Models confidently wrong on corpora they never saw (MESS, MCAE-SPPS, TESS, RAVDESS)",
        "The two top models disagree with each other",
        "The ground truth itself is doubtful (listeners split)"] },
      { t: "Procedure: each clip is played before its labels are shown, then compared with", b: true, sub: ["the actor's intended emotion", "the listener votes (where published)", "all five models' predictions with their probabilities († = model trained on this corpus)"] },
      { t: "Guiding question: does the error lie with the model or with the reference label?", b: true, color: C.accent2 },
    ], { y: 1.55, h: 5.2, fontSize: 18 });
    L.notes(s, `This part is a qualitative error analysis based on listening. Ten clips were selected with the two top models in mind, in four groups. Each clip is played before its labels and predictions are shown, so the audience can form its own judgement. The guiding question is whether an error lies with the model or with the reference label.`);
  }

  // ---- Limitations of audio LLMs (literature)
  {
    const s = L.slide(SEC, "Error Analysis: Why Audio LLMs Fall Short");
    const p = L.pptx;
    L.text(s, "Our benchmark: GPT Realtime is below chance on 4 of 6 corpora; Gemini 3.5 Flash is 31–64 points behind the dedicated models. Published studies explain why:", MX, 1.38, W - 2 * MX, 0.55, { fontSize: 15, color: C.ink });
    const cards = [
      ["1. Audio LLMs rely on words rather than voice", "16–35% accuracy", "when the words carry no emotion", "LISTEN, EACL 2026"],
      ["2. Strong bias towards \"neutral\"", "70% of answers", "from gpt-realtime-2 are \"neutral\"", "VocalAffectBench, 2026"],
      ["3. Results depend on the prompt", "> 20 points", "macro-F1 difference from prompt wording alone", "VoxEmo, 2026"],
      ["4. Few-shot examples do not improve accuracy", "no gain", "1–8 audio examples fix the output format only", "When Demonstrations Fail, SLT 2026"],
      ["5. Voice cues are encoded but not used", "lost before the answer", "speaking style is present inside the audio encoder", "Heard but Not Heeded, 2026"],
      ["6. No confidence scores", "label only", "no probabilities or thresholds; cloud-only", "ASOM, ICASSP 2027"],
    ];
    const cw = 3.95, ch = 2.25, gx = 0.14, gy = 0.14;
    cards.forEach(([t, num, d, src], i) => {
      const x = MX + (i % 3) * (cw + gx), y = 2.05 + Math.floor(i / 3) * (ch + gy);
      s.addShape(p.ShapeType.rect, { x, y, w: cw, h: ch, fill: { color: C.panel }, line: { color: C.line, width: 0.75 } });
      L.text(s, t, x + 0.22, y + 0.15, cw - 0.4, 0.62, { fontSize: 15, bold: true, color: C.ink, valign: "middle" });
      L.text(s, num, x + 0.22, y + 0.85, cw - 0.4, 0.45, { fontSize: 20, bold: true, color: C.ink });
      L.text(s, d, x + 0.22, y + 1.3, cw - 0.4, 0.5, { fontSize: 13, color: C.text });
      L.text(s, src, x + 0.22, y + ch - 0.36, cw - 0.4, 0.28, { fontSize: 10.5, italic: true, color: C.muted });
    });
    L.source(s, "Papers:", [
      ["LISTEN", "https://arxiv.org/abs/2510.10444"], ["VoxParadox", "https://arxiv.org/abs/2605.27772"], ["VocalAffectBench", "https://arxiv.org/abs/2608.28932"],
      ["VoxEmo", "https://arxiv.org/abs/2603.08936"], ["Hearing the Order", "https://arxiv.org/abs/2510.00628"], ["When Demonstrations Fail", "https://arxiv.org/abs/2603.20433"],
      ["Heard but Not Heeded", "https://arxiv.org/abs/2609.00727"], ["C²SER", "https://arxiv.org/abs/2502.18186"], ["ASOM", "https://arxiv.org/abs/2610.22401"],
    ]);
    L.notes(s, `Before listening to individual errors: why do audio LLMs perform so poorly? Our numbers are consistent with a growing body of work.
    First, audio LLMs rely on words rather than voice. LISTEN, at EACL 2026, shows that with emotionally neutral words and audio only, six audio LLMs reach 16 to 35 percent and default to neutral; VoxParadox, at ICML 2026, shows Audio Flamingo 3 collapsing to 17 percent when the transcript is misleading.
    Second, a strong bias towards neutral: VocalAffectBench, a 2026 study on acted speech, finds that 45 percent of all model outputs are neutral, and gpt-realtime-2 answers neutral 70 percent of the time, with the lowest accuracy, 28 percent, against 44 percent for gemini-3.5-flash. This matches what we see.
    Third, inconsistency and prompt sensitivity. Generative outputs are stochastic; VoxEmo finds more than 20 points of macro-F1 difference between prompt wordings, and "Hearing the Order", at Interspeech 2026, finds that just shuffling the answer options changes accuracy by up to 24 percent and reorders the models. VoxEmo also observes that the spread of outputs across prompts resembles the disagreement among human listeners. How often the label flips when exactly the same request is repeated has not been measured systematically yet; this is one of our candidate research questions.
    Fourth, few-shot examples do not help: at SLT 2026, adding one to eight audio examples improved answer formatting but not accuracy, and shuffling the example labels changed nothing, so the models do not learn the audio-to-emotion mapping from examples.
    Fifth, the information is encoded but not used: speaking style is present inside the audio encoder but lost before the answer, and hallucinated answers are common.
    Sixth, practical: only a label comes back, no calibrated probabilities, and the closed models change version and run in the cloud.
    References:
    1. LISTEN: Chen et al., "Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance", EACL 2026. https://arxiv.org/abs/2510.10444
    1. VoxParadox: Pang, Chaubey and Soleymani, ICML 2026. https://arxiv.org/abs/2605.27772
    2. VocalAffectBench: Debaupte et al., arXiv 2026. https://arxiv.org/abs/2608.28932
    3. VoxEmo: Zhang, Chou, Narayanan and Hain, "VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs", submitted to Interspeech 2026. https://arxiv.org/abs/2603.08936
    3. Hearing the Order: Lin et al., Interspeech 2026. https://arxiv.org/abs/2510.00628
    4. When Demonstrations Fail: Piao et al., IEEE SLT 2026. https://arxiv.org/abs/2603.20433
    5. Heard but Not Heeded: Koduru et al., arXiv 2026. https://arxiv.org/abs/2609.00727
    5. C2SER: Zhao et al., IEEE/ACM TASLP. https://arxiv.org/abs/2502.18186
    6. ASOM: Bellver-Soler et al., ICASSP 2027. https://arxiv.org/abs/2610.22401`);
  }

  const groups = [
    ["i_models_side_with_listeners", "Error Analysis: Agreement with Listeners", "Consistent with listeners, not with the acted label",
      `Two CREMA-D clips. In the first, the actor intended happy, nine of ten listeners heard angry, and both top models say angry. In the second, acted angry, eight listeners heard disgusted, and both models say disgusted. Scored against the actor these are errors; against the listeners they are correct. For our own data, where labels will come from listeners, this is what we want the model to do.`],
    ["ii_confident_on_unseen_data", "Error Analysis: Domain Shift (1/2)", "Domain shift",
      `MESS: acted angry, and all five models say neutral, emotion2vec+ with probability 1.00. Recall that emotion2vec+ labels 81 percent of MESS neutral, and MESS removes loudness differences. MCAE-SPPS: 38 of 40 listeners heard surprise, yet three models say angry. Mandarin surprise is systematically heard as anger by the models. High probability does not mean correct outside the training domain.`],
    ["ii_confident_on_unseen_data", "Error Analysis: Domain Shift (2/2)", "Domain shift",
      `TESS: a neutral clip that MERaLiON calls sad with 0.91, as it does for all 400 neutral TESS clips; emotion2vec+, trained on TESS, is right. RAVDESS: a strong surprise; emotion2vec+, trained on RAVDESS, is right, MERaLiON says fearful at 0.91. The pattern is the same in both directions: the model that has seen the corpus wins.`],
    ["iii_models_disagree", "Error Analysis: Model Disagreement", "Model disagreement indicates uncertainty",
      `MESS: four models say sad, emotion2vec+ alone says neutral, at probability 1.00. MCAE-SPPS: 22 listeners heard happy and 17 neutral, and the two models split the same way. Model disagreement often marks clips that humans also find ambiguous; this is exactly the signal we will use to pick clips for human review in active learning.`],
    ["iv_questionable_ground_truth", "Error Analysis: Ambiguous Ground Truth", "Questionable reference label",
      `CREMA-D, first clip: the intended emotion is angry, but the nine listeners split three ways, angry, happy and neutral, three votes each; there is no majority at all, and both top models say happy. Second clip: acted fearful, but no listener chose fear; votes are spread over five emotions. Scoring these as model errors is unfair. With our own data we should keep such clips as ambiguous, with their vote distribution, rather than force a single label.`],
  ];
  const seen = {};
  groups.forEach(([sub, title, label, notes]) => {
    const cs = by(sub), k = seen[sub] || 0;
    seen[sub] = k + 2;
    const s = L.slide(SEC, title);
    cs.slice(k, k + 2).forEach((c, i) => K.caseRow(s, c, 1.5 + i * 2.65, 2.45, label));
    L.source(s, "Bars: listener votes and each model's probability for its predicted label (green = actor's emotion). † = corpus in the model's training data.");
    L.notes(s, notes);
  });

  {
    const s = L.slide(SEC, "Error Analysis: Summary");
    const items = [
      ["Reference labels are uncertain", "Many \"errors\" are defensible: listeners hear the same emotion as the model. Accuracy against a single label understates model quality and obscures genuine failures."],
      ["Probabilities are over-confident out of domain", "Probability 1.00 on incorrect predictions (MESS, TESS); raw confidence cannot serve as a quality gate on call-centre audio before calibration on in-domain data."],
      ["Training data determines the outcome", "On every disputed clip, the model trained on that corpus is correct. Call-centre audio is unseen by all models, so in-domain data is decisive."],
      ["Disagreement is an informative signal", "Clips on which the models disagree, or the listeners split, are the most informative for human review: the basis of the annotation plan."],
    ];
    items.forEach(([t, d], i) => {
      const y = 1.55 + i * 1.3;
      L.box(s, MX, y, 0.12, 1.05, "", { fill: i === 3 ? C.accent2 : C.accent, line: i === 3 ? C.accent2 : C.accent, r: 0 });
      L.text(s, t, MX + 0.35, y, 11.5, 0.42, { fontSize: 19, bold: true, color: C.ink });
      L.text(s, d, MX + 0.35, y + 0.45, 11.6, 0.6, { fontSize: 15, color: C.text });
    });
    L.notes(s, `Four lessons from listening. Labels are uncertain, so some errors are not errors. Confidence is unreliable outside the training domain. The model trained on a corpus wins the disputed clips. And disagreement between models or listeners indicates where human review is most valuable, which leads directly to the annotation plan.`);
  }
};

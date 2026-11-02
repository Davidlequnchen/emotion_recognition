// Part 5: listening to errors
module.exports = function (L, A) {
  const { C, W, MX } = L;
  const K = require("./clips")(L);
  const SEC = "5 · Listening to errors";
  const clips = ((A.manifest && A.manifest.audio) || []).filter((c) => c.group === "failure");
  const by = (sub) => clips.filter((c) => c.subgroup === sub);

  {
    const s = L.slide(SEC, "Ten clips where the models and the labels disagree");
    L.bullets(s, [
      { t: "Four kinds of disagreement", b: true, sub: [
        "Models side with the listeners against the actor (CREMA-D)",
        "Models confidently wrong on corpora they never saw (MESS, MCAE-SPPS, TESS, RAVDESS)",
        "The two top models disagree with each other",
        "The ground truth itself is doubtful (listeners split)"] },
      { t: "For each clip: listen first, guess, then compare with", b: true, sub: ["the actor's intended emotion", "the listener votes (where published)", "all five models' predictions with their probabilities († = model trained on this corpus)"] },
      { t: "Question to keep in mind: is the model wrong, or the label?", b: true, color: C.accent2 },
    ], { y: 1.55, h: 5.2, fontSize: 18 });
    L.notes(s, `Now the most important part: listening. I picked ten clips with the two top models in mind, in four groups. For each, I suggest we listen first and guess, then look at the labels and predictions. The recurring question: is the model wrong, or is the label?`);
  }

  const groups = [
    ["i_models_side_with_listeners", "Models side with the listeners, not the actor", "Wrong by the actor, right by the listeners",
      `Two CREMA-D clips. In the first, the actor intended happy, nine of ten listeners heard angry, and both top models say angry. In the second, acted angry, eight listeners heard disgusted, and both models say disgusted. Scored against the actor these are errors; against the listeners they are correct. For our own data, where labels will come from listeners, this is what we want the model to do.`],
    ["ii_confident_on_unseen_data", "Confidently wrong on corpora the model never saw (1/2)", "Domain shift",
      `MESS: acted angry, and all five models say neutral, emotion2vec+ with probability 1.00. Recall that emotion2vec+ labels 81 percent of MESS neutral, and MESS removes loudness differences. MCAE-SPPS: 38 of 40 listeners heard surprise, yet three models say angry. Mandarin surprise is systematically heard as anger by the models. High probability does not mean correct outside the training domain.`],
    ["ii_confident_on_unseen_data", "Confidently wrong on corpora the model never saw (2/2)", "Domain shift",
      `TESS: a neutral clip that MERaLiON calls sad with 0.91, as it does for all 400 neutral TESS clips; emotion2vec+, trained on TESS, is right. RAVDESS: a strong surprise; emotion2vec+, trained on RAVDESS, is right, MERaLiON says fearful at 0.91. The pattern is the same in both directions: the model that has seen the corpus wins.`],
    ["iii_models_disagree", "When the two top models disagree", "Disagreement = uncertainty",
      `MESS: four models say sad, emotion2vec+ alone says neutral, at probability 1.00. MCAE-SPPS: 22 listeners heard happy and 17 neutral, and the two models split the same way. Model disagreement often marks clips that humans also find ambiguous; this is exactly the signal we will use to pick clips for human review in active learning.`],
    ["iv_questionable_ground_truth", "When the label itself is doubtful", "Is the label right?",
      `MCAE-SPPS: the intended emotion is happy, but listeners were split 19 happy against 14 angry; both models say angry. CREMA-D: acted fearful, but no listener chose fear; votes are spread over five emotions. Scoring these as model errors is unfair. With our own data we should keep such clips as ambiguous, with their vote distribution, rather than force a single label.`],
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
    const s = L.slide(SEC, "What the clips teach us");
    const items = [
      ["Labels are uncertain", "Many \"errors\" are defensible: humans hear the same thing as the model. Accuracy against one label understates how good models are and hides real failures."],
      ["Probabilities are over-confident out of domain", "1.00 on a wrong answer (MESS, TESS). We cannot use raw confidence as a quality gate on our calls until we calibrate on our own data."],
      ["Training data decides", "On every disputed clip, the model that has seen the corpus wins. Our calls are unseen by all models, so in-domain data is the lever."],
      ["Disagreement is a useful signal", "Clips where the models disagree, or listeners split, are the ones worth human time: the basis of the labelling plan."],
    ];
    items.forEach(([t, d], i) => {
      const y = 1.55 + i * 1.3;
      L.box(s, MX, y, 0.12, 1.05, "", { fill: i === 3 ? C.accent2 : C.accent, line: i === 3 ? C.accent2 : C.accent, r: 0 });
      L.text(s, t, MX + 0.35, y, 11.5, 0.42, { fontSize: 19, bold: true, color: C.ink });
      L.text(s, d, MX + 0.35, y + 0.45, 11.6, 0.6, { fontSize: 15, color: C.text });
    });
    L.notes(s, `Four lessons from listening. Labels are uncertain, so some errors are not errors. Confidence is unreliable outside the training domain. Training data decides who wins. And disagreement between models or listeners tells us where human effort pays off, which leads directly to the plan.`);
  }
};

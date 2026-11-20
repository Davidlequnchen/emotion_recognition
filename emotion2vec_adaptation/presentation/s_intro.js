// Title, the problem (domain gap; few labels, much unlabelled audio), data sources, overview of both paths
const R = require("./refs");

module.exports = function (L) {
  const { C, W, H, MX, CW, TOP } = L;
  const p = L.pptx;

  // ---- Title
  {
    const s = L.blank();
    s.addShape(p.ShapeType.rect, { x: MX, y: 1.95, w: 0.09, h: 2.55, fill: { color: C.accent }, line: { color: C.accent, width: 0 } });
    L.text(s, "Adapting emotion2vec+ large to Customer Calls", MX + 0.4, 1.85, 11.8, 0.85, { fontSize: 40, bold: true });
    L.text(s, "Two independent paths: train a head on the frozen encoder (options A, B), or\ncontinue self-supervised pretraining on unlabelled call audio, then train a head", MX + 0.4, 2.8, 11.8, 1.0, { fontSize: 24, ls: 1.1 });
    L.text(s, "Chen Lequn  ·  emotion2vec_adaptation  ·  November 2026", MX + 0.4, 4.05, 11.8, 0.42, { fontSize: 20 });
    L.text(s, "Internal workstream. No call audio, transcripts or customer data in this deck; all citations are public.", MX, H - 0.95, CW, 0.4, { fontSize: 16, kind: "diagram" });
    L.notes(s, `This talk explains the emotion2vec_adaptation subproject: how we adapt the emotion2vec+ large model, which we already use in the benchmark, to our own customer-call audio.
    There are two independent paths. Path A/B keeps the whole encoder frozen and trains only a small classification head on listener labels, in one of two options: option A puts a linear layer on the last block's averaged output, option B learns a weighted mix of all 12 block outputs followed by a small MLP. Path E first continues self-supervised pretraining of the encoder on unlabelled call clips, and then trains a new head on the adapted encoder in exactly the same way as Path A/B. The names come from the original fine-tuning plan: options A and B are the two head variants, now both implemented in one head-training path, and E was continued pretraining.
    The deck covers why adaptation is needed, where the data comes from, both architectures, the self-supervised objective with its equations, how an encoder and a head are combined safely, how leakage between training and test customers is prevented, the data scale, how to run it on the B200 and what is verified so far.
    Everything shown is either code, configuration or public literature. No call audio, transcripts or customer data appear in the deck.`);
  }

  // ---- Problem 1: domain gap
  {
    const s = L.slide("Problem", "Public emotion corpora do not sound like customer calls");
    L.table(s, ["", "Public corpora (emotion2vec)", "Customer calls (our target)"], [
      [{ text: "Corpora", bold: true }, "IEMOCAP, MELD, CMU-MOSEI, MEAD, MSP-Podcast: 262 h", "30+ batches of 2–10 s call clips"],
      [{ text: "Who speaks", bold: true }, "actors, TV characters, YouTube and podcast speakers", "customers talking to a service agent"],
      [{ text: "Speaking style", bold: true }, "acted or performed; often one clear emotion per clip", "spontaneous; subtle or mixed, can change in a call"],
      [{ text: "Channel", bold: true }, "studio, broadcast or online-video audio", "telephone: narrow band, codec, line noise"],
      [{ text: "Labels", bold: true }, "acted intent or listener ratings, per corpus", "none until our listeners label the clips"],
    ], { y: TOP + 0.05, colW: [2.2, 5.3, 4.63], fontSize: 20, rowH: [0.55, 0.72, 0.72, 0.72, 0.72, 0.72] });
    L.box(s, MX, 6.2, CW, 0.52, [
      { text: "Public benchmark accuracy does not predict accuracy on calls: ", options: { bold: true } },
      { text: "adapt, and test on labelled call clips." },
    ], { fill: C.blueFill, line: C.accent, fontSize: 20, align: "left", margin: [3, 14, 3, 14] });
    L.notes(s, `Why adapt at all? emotion2vec+ large was trained on public data, and public emotion corpora sound very different from our calls.
    The emotion2vec paper lists its self-supervised pre-training data in its Table 1: 262 hours, 169,053 utterances, from IEMOCAP, MELD, CMU-MOSEI, MEAD and MSP-Podcast. These are acted dyadic sessions, a TV series, YouTube monologues, studio-recorded acted sentences and podcasts. Our unlabelled data are more than 30 batches of call clips between 2 and 10 seconds.
    Our clips are customers talking to an agent on the telephone. The speech is spontaneous, emotions are usually subtle or mixed and can change within one call, and the channel is a narrow-band telephone line with codec compression and line noise.
    The label situation is also different: public corpora come with acted intent or listener ratings, while our calls have no labels until our own listeners label them in Label Studio.
    So a high benchmark score on public corpora does not tell us how well the model works on calls. The practical consequence, in the blue box, is that we adapt the model to the call domain, and that every change is judged on labelled call clips from customers the model has never been trained on.`);
    L.cites(s, R, ["e2v"]);
  }

  // ---- Problem 2: few labels, much unlabelled audio -> two paths
  {
    const s = L.slide("Problem", "Few labels, much unlabelled audio: two ways to adapt");
    const ph = 1.3, py = TOP + 0.05, pw = (CW - 0.33) / 2;
    [[MX, "1K / 5K", "labelled clips per head\nexperiment; 5 listeners each"],
     [MX + pw + 0.33, "30+", "batches of 100K–200K\nunlabelled 2–10 s clips"]].forEach(([x, num, cap]) => {
      s.addShape(p.ShapeType.rect, { x, y: py, w: pw, h: ph, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.text(s, num, x + 0.25, py + 0.1, 2.2, ph - 0.2, { fontSize: 40, bold: true, color: C.link, valign: "middle" });
      L.text(s, cap, x + 2.45, py + 0.1, pw - 2.55, ph - 0.2, { fontSize: 20, valign: "middle" });
    });
    L.table(s, ["", "Path A/B · head fine-tuning", "Path E · continued SSL, then a new head"], [
      [{ text: "Uses", bold: true }, "labelled clips", "unlabelled clips; then labelled clips for the head"],
      [{ text: "Trains", bold: true }, "A: linear head (8.2K); B: 12 block weights + MLP (0.26M)", "encoder (157.9M) + new predictor (2.1M); then a head"],
      [{ text: "Keeps frozen", bold: true }, "the whole encoder (162.1M)", "CNN front end (4.2M); teacher gets no gradients"],
      [{ text: "Cost", bold: true }, "one encoder pass per clip; tiny head", "teacher + student + backward pass per step, on GPU"],
      [{ text: "Output", bold: true }, "head.pt + model_bundle.json", "encoder.pt → retrain the head → bundle"],
    ], { y: py + ph + 0.25, colW: [1.9, 4.2, 6.03], fontSize: 18, rowH: 0.52 });
    L.notes(s, `The second half of the problem is the label budget. Labels are expensive: our Label Studio setup asks five different listeners to label every clip, and the head experiments are planned with 1,000 and later 5,000 labelled training clips. Unlabelled audio, on the other hand, is plentiful: the latest description is more than 30 batches with about 100,000 to 200,000 pre-processed clips each, every clip between 2 and 10 seconds.
    This gives two ways to adapt, which the table compares.
    Path A/B only needs labelled clips. It keeps the whole encoder, 162.1 million parameters, frozen. Option A trains a linear layer with 8,200 parameters on the last block's averaged output. Option B trains 12 block weights and an MLP with one hidden layer of 256 units, about 0.26 million parameters for 8 classes. Its cost is one forward pass of the encoder per clip, after which the head trains in a very small model. It writes head.pt and a model_bundle.json that prediction uses.
    Path E uses the unlabelled clips. It trains the encoder itself, 157.9 million parameters once the CNN front end of 4.2 million parameters is frozen, plus a new predictor of 2.1 million parameters, with a teacher copy that never receives gradients. Every step runs the teacher and the student and a backward pass, so it needs the GPU. Its output is a new encoder.pt, and a head must then be trained on top of it, exactly as in Path A/B.
    The parameter counts are computed from the local model weights and the code.
    Continued pretraining on unlabelled target-domain audio has helped in other settings, for example speech recognition under domain shift and domain-adaptive pretraining of language models. That is motivation, not evidence for emotion recognition on our calls; our own comparison has to show it.`);
    L.cites(s, R, ["robust", "dapt"], "Target-domain pretraining elsewhere:");
  }

  // ---- Data sources
  {
    const s = L.slide("Data", "Labelled clips come from Label Studio; the rest feeds SSL");
    const y1 = TOP + 0.25, h1 = 1.0, ws = [2.15, 2.0, 2.5, 2.3, 2.45], g = (CW - ws.reduce((a, b) => a + b)) / 4;
    const xs = ws.map((_, i) => MX + ws.slice(0, i).reduce((a, b) => a + b, 0) + i * g);
    const t1 = ["preparation_tools\nfinal/ clips", "Label Studio\n5 listeners / clip", "normalize_export.py\none vote / listener", "prepare_labels.py\nmerge by clip_id", "labeled/\nmanifest.csv"];
    t1.forEach((t, i) => {
      L.box(s, xs[i], y1, ws[i], h1, t, { role: i === 4 ? "out" : (i === 0 ? "data" : undefined), fill: i > 0 && i < 4 ? C.panel : undefined, line: i > 0 && i < 4 ? C.ink : undefined, fontSize: 18 });
      if (i < 4) L.arrow(s, xs[i] + ws[i], y1 + h1 / 2, xs[i + 1], y1 + h1 / 2);
    });
    const nx = xs[2] + 0.2, nw = xs[3] + ws[3] - nx;
    L.box(s, nx, y1 + h1 + 0.3, nw, 0.9, "conflicting votes → adjudicate first\nunknown, bad_clip → excluded", { fill: C.white, line: C.bad, dash: "dash", fontSize: 18 });
    L.arrow(s, xs[3] + ws[3] / 2, y1 + h1, xs[3] + ws[3] / 2, y1 + h1 + 0.3, { color: C.bad, width: 1.5 });
    L.arrow(s, xs[4] + ws[4] / 2, y1 + h1, xs[4] + ws[4] / 2, y1 + h1 + 0.3);
    L.box(s, xs[4], y1 + h1 + 0.3, ws[4], 0.9, "Path A/B, and the\nhead of Path E", { fill: C.white, line: C.link, fontSize: 18, bold: true, color: C.link, r: 0.3 });
    // unlabelled branch
    const y2 = 4.3, h2 = 0.9;
    L.arrow(s, xs[0] + ws[0] / 2, y1 + h1, xs[0] + ws[0] / 2, y2);
    L.text(s, "not labelled,\nor unknown", xs[0] + ws[0] / 2 + 0.12, y1 + h1 + 0.3, 1.8, 0.6, { fontSize: 18, kind: "diagram" });
    L.box(s, MX, y2, 3.3, h2, "unlabelled clips\n(bad_clip removed)", { role: "data", fontSize: 18 });
    L.arrow(s, MX + 3.3, y2 + h2 / 2, MX + 3.75, y2 + h2 / 2);
    L.box(s, MX + 3.75, y2, 3.0, h2, "unlabeled/ folder or\nCSV: path, group", { role: "out", fontSize: 18 });
    L.arrow(s, MX + 6.75, y2 + h2 / 2, MX + 7.2, y2 + h2 / 2);
    L.box(s, MX + 7.2, y2 + 0.06, 2.3, 0.78, "Path E · SSL", { fill: C.white, line: C.link, fontSize: 18, bold: true, color: C.link, r: 0.3 });
    // rules
    const y3 = 5.5, h3 = 1.2, w3 = (CW - 0.5) / 3;
    L.box(s, MX, y3, w3, h3, [
      { text: "group = customer ID", options: { bold: true, breakLine: true } },
      { text: "(else call ID); never the quality column group of preparation_tools" },
    ], { fill: C.amberFill, line: C.warn, fontSize: 18, align: "left", margin: [3, 10, 3, 10] });
    L.box(s, MX + w3 + 0.25, y3, w3, h3, [
      { text: "2–15 s", options: { bold: true, breakLine: true } },
      { text: "same window as preparation_tools; checked before training" },
    ], { fill: C.amberFill, line: C.warn, fontSize: 18, align: "left", margin: [3, 10, 3, 10] });
    L.box(s, MX + 2 * (w3 + 0.25), y3, w3, h3, [
      { text: "8 classes", options: { bold: true, breakLine: true } },
      { text: "angry, disgusted, fearful, happy, neutral, other, sad, surprised" },
    ], { fill: C.panel, line: C.ink, fontSize: 18, align: "left", margin: [3, 10, 3, 10] });
    L.notes(s, `This slide shows where the two kinds of input come from. No new tooling is involved; it connects subprojects we already have.
    The top row is the labelled path. preparation_tools cuts customer turns into clips; its final folder holds whole turns up to 10 seconds and cut pieces of 2 to 15 seconds. This project now accepts the same window, 2 to 15 seconds; the current internal batches are 2 to 10 seconds and fit. Every clip's header is checked before any weights are loaded, in training, self-supervised training and prediction, and an out-of-range or unreadable clip stops the run at once with a list, instead of being cropped silently. The clips are labelled in Label Studio, where five different listeners label each clip. label_studio's normalize_export.py writes one vote per listener, not a final label. prepare_labels.py in this project merges the votes by clip_id: votes for a clip must agree, a conflict stops the import and has to be adjudicated first, and clips decided as unknown or bad_clip are excluded. The result is a manifest with the columns path, label, group, split and clip_id, which Path A/B uses and which the head of Path E uses as well.
    The bottom row is the unlabelled path. Clips that are not labelled, or whose emotion was undecidable but whose audio is fine, can go into the unlabelled folder or a CSV with path and group. Bad clips are removed from both paths. The self-supervised training reads only this audio.
    Three rules matter most. First, the group column must identify the customer, or the call when the customer is unknown. It must never be the group column of preparation_tools, which is a quality category, not a speaker identity. All leakage checks depend on this. Second, the length window from the middle box. Third, the label set: we use the eight named classes of the released model, including its formal other class; unknown is kept separate and excluded from supervised training.`);
    L.cite(s, "Sources: config/common.yaml, config/labels.yaml, adaptation/data.py, label_studio/README_en.md");
  }

  // ---- Overview of both paths
  {
    const s = L.slide("Overview", "Both paths end in the same head trainer and the same test");
    const bandX = MX - 0.15, bandW = 10.4;
    s.addShape(p.ShapeType.rect, { x: bandX, y: 1.68, w: bandW, h: 1.36, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    s.addShape(p.ShapeType.rect, { x: bandX, y: 3.22, w: bandW, h: 2.92, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    L.text(s, "Path A/B · head fine-tuning (option A or B)", MX, 1.71, 7, 0.36, { fontSize: 18, bold: true, color: C.link });
    L.text(s, "Path E · continued SSL, then a new head", MX, 3.25, 7, 0.36, { fontSize: 18, bold: true, color: C.link });
    const col = [[MX, 1.9], [MX + 2.3, 2.4], [MX + 5.1, 2.4], [MX + 7.9, 2.15]], bh = 0.8;
    const row = (y, enc, encRole) => {
      L.box(s, col[0][0], y, col[0][1], bh, "Labelled clips\n(train split)", { role: "data", fontSize: 18 });
      L.box(s, col[1][0], y, col[1][1], bh, enc, { role: encRole, fontSize: 18 });
      L.box(s, col[2][0], y, col[2][1], bh, "Head trainer, A or B\nembed once, fit head", { role: "train", fontSize: 18 });
      L.box(s, col[3][0], y, col[3][1], bh, "head.pt +\nmodel_bundle.json", { role: "out", fontSize: 18 });
      for (let i = 0; i < 3; i++) L.arrow(s, col[i][0] + col[i][1], y + bh / 2, col[i + 1][0], y + bh / 2);
    };
    const yA = 2.12, yE = 3.66, yS = 5.12;
    row(yA, "Original encoder\nfrozen", "frozen");
    row(yE, "Adapted encoder\nfrozen", "frozen");
    L.box(s, col[0][0], yS, col[0][1], bh, "Unlabelled clips\nno labels", { role: "data", fontSize: 18 });
    L.box(s, col[1][0], yS, col[1][1], bh, "Continued SSL, 2 gates\nstudent + EMA teacher", { role: "train", fontSize: 16 });
    L.arrow(s, col[0][0] + col[0][1], yS + bh / 2, col[1][0], yS + bh / 2);
    L.arrow(s, col[1][0] + col[1][1] / 2, yS, col[1][0] + col[1][1] / 2, yE + bh, { width: 2 });
    L.text(s, "exports encoder.pt", col[1][0] + col[1][1] / 2 + 0.12, yE + bh + 0.08, 2.4, 0.34, { fontSize: 16, bold: true, kind: "diagram" });
    L.text(s, "the SSL loss uses no emotion labels", col[2][0], yS + 0.2, 4.6, 0.4, { fontSize: 18, italic: true, kind: "diagram" });
    // comparison
    const cx = 11.05, cw = W - MX - cx;
    L.box(s, cx, yA, cw, yE + bh - yA, "Same\ncustomer-\ndisjoint\ntest set:\nUAR,\nmacro-F1", { fill: C.white, line: C.ink, lw: 2, fontSize: 18, bold: true });
    L.arrow(s, col[3][0] + col[3][1], yA + bh / 2, cx, yA + bh / 2);
    L.arrow(s, col[3][0] + col[3][1], yE + bh / 2, cx, yE + bh / 2);
    // legend
    const ly = 6.32;
    L.key(s, MX, ly, "frozen");
    L.key(s, MX + 3.0, ly, "train");
    L.key(s, MX + 5.4, ly, "data");
    L.key(s, MX + 7.2, ly, "out");
    L.notes(s, `Here are both paths side by side. Blue boxes are frozen, orange boxes are trained, green boxes are files written to disk.
    Path A/B, on the grey band at the top: the labelled training clips go through the original encoder, which stays frozen. The head trainer embeds every clip once, caches the embeddings and fits only the head, either option A or option B. It writes head.pt and model_bundle.json, which together describe a complete classifier.
    Path E has two stages. First, continued self-supervised learning runs on unlabelled clips only, with a student and an EMA teacher; its loss uses no emotion labels at all. It exports a new encoder, encoder.pt. Second, that adapted encoder is frozen and goes through exactly the same head trainer, with the same labelled clips and the same splits, which gives a second head and a second bundle.
    Because everything after the encoder is identical, the two bundles differ only in the encoder. That is what makes the comparison on the right fair: both are evaluated on the same customer-disjoint test set, with unweighted average recall and macro-F1. Only if the adapted encoder gives a better head on validation is continued pretraining worth scaling up.
    The two paths are independent programs. Path A/B can run on the first labels while self-supervised training is still running.`);
    L.cite(s, "Sources: train_head.py, pretrain_encoder.py, adaptation/head.py, adaptation/pretrain.py");
  }
};

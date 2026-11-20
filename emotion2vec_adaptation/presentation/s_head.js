// Path A/B: frozen encoder, cached utterance embeddings, a small trained head
const R = require("./refs");

module.exports = function (L) {
  const { C, W, MX, CW, TOP } = L;
  const p = L.pptx;
  const SEC = "Path A/B · head fine-tuning";

  // ---- Path A/B architecture
  {
    const s = L.slide(SEC, "Path A/B: embed each clip once, train only a small head");
    const band = (y, h, label) => {
      s.addShape(p.ShapeType.rect, { x: MX - 0.15, y, w: CW + 0.3, h, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      L.text(s, label, W - MX - 9, y + 0.06, 9, 0.36, { fontSize: 18, bold: true, color: C.link, align: "right" });
    };
    band(1.68, 1.88, "Once per clip · one clip at a time · no gradients");
    band(3.78, 2.08, "Train · up to 100 epochs · select by validation UAR");
    // row 1: clip -> frozen encoder -> option A or B pooling -> cache
    const y1 = 2.3, h1 = 0.95, c1 = y1 + h1 / 2, oh = 0.64, oA = 2.13, oB = 2.9, ox = 5.5, ow = 3.55;
    L.box(s, MX, y1, 1.8, h1, "Clip, 2–15 s\n16 kHz mono", { role: "data", fontSize: 16 });
    L.arrow(s, MX + 1.8, c1, 2.75, c1);
    L.box(s, 2.75, y1, 2.35, h1, "emotion2vec+ large\nencoder · 162.1M", { role: "frozen", fontSize: 18, bold: true });
    L.box(s, ox, oA, ow, oh, "A · last block:\nmean over valid frames → 1024", { role: "data", fontSize: 16 });
    L.box(s, ox, oB, ow, oh, "B · all 12 blocks:\nmean of each → 12 × 1024", { role: "data", fontSize: 16 });
    const sx = 5.3, mx = 9.28, cx = 9.5;
    L.path(s, [[5.1, c1], [sx, c1], [sx, oA + oh / 2], [ox, oA + oh / 2]]);
    L.path(s, [[sx, c1], [sx, oB + oh / 2], [ox, oB + oh / 2]]);
    L.box(s, cx, y1, W - MX - cx, h1, "embeddings\ncached once\n(features.pt)", { role: "out", fontSize: 16 });
    L.path(s, [[ox + ow, oA + oh / 2], [mx, oA + oh / 2], [mx, c1], [cx, c1]]);
    L.path(s, [[ox + ow, oB + oh / 2], [mx, oB + oh / 2], [mx, c1]], { noHead: true });
    // carriage from the cache to the head row
    const y2 = 4.3, h2 = 0.95, c2 = y2 + h2 / 2, lane = 3.67, ccx = cx + (W - MX - cx) / 2, c0x = MX + 0.9;
    L.path(s, [[ccx, y1 + h1], [ccx, lane], [c0x, lane], [c0x, y2]]);
    // row 2: cached embeddings -> head A or B -> softmax -> loss <- label
    const hA = 4.06, hB = 4.82, hh = 0.68, hx = 2.75, hw = 4.2;
    L.box(s, MX, y2, 1.8, h2, "cached\nembeddings", { role: "out", fontSize: 16 });
    L.box(s, hx, hA, hw, hh, "A · Linear 1024 → 8  (8.2K)", { role: "train", fontSize: 18, bold: true });
    L.box(s, hx, hB, hw, hh, "B · layer norm, 12 learned weights,\nMLP 1024 → 256 → 8  (0.26M)", { role: "train", fontSize: 16, bold: true });
    L.pill(s, hx + hw - 0.62, hA - 0.17, "new");
    const s2 = 2.55, m2 = 7.15;
    L.path(s, [[MX + 1.8, c2], [s2, c2], [s2, hA + hh / 2], [hx, hA + hh / 2]]);
    L.path(s, [[s2, c2], [s2, hB + hh / 2], [hx, hB + hh / 2]]);
    const smx = 7.4, smw = 1.5, lx = 9.25, lw = 1.85, tx = 11.45;
    L.box(s, smx, y2, smw, h2, "softmax\n8 classes", { role: "data", fontSize: 16 });
    L.path(s, [[hx + hw, hA + hh / 2], [m2, hA + hh / 2], [m2, c2], [smx, c2]]);
    L.path(s, [[hx + hw, hB + hh / 2], [m2, hB + hh / 2], [m2, c2]], { noHead: true });
    L.box(s, lx, y2, lw, h2, "class-balanced\ncross-entropy", { role: "loss", fontSize: 16 });
    L.arrow(s, smx + smw, c2, lx, c2);
    L.box(s, tx, y2, W - MX - tx, h2, "listener\nlabel", { role: "data", fontSize: 16 });
    L.arrow(s, tx, c2, lx + lw, c2);
    // gradient back to the heads
    const gy = 5.68, gl = lx + lw / 2, gh = hx + hw / 2;
    L.path(s, [[gl, y2 + h2], [gl, gy], [gh, gy], [gh, hB + hh]], { color: C.accent2, dash: "dash", width: 2 });
    L.text(s, "gradient → AdamW", gl + 0.12, y2 + h2 + 0.05, W - MX - gl - 0.12, 0.32, { fontSize: 16, color: C.darkOrange, bold: true, kind: "diagram" });
    L.text(s, [{ text: "A (default): ", options: { bold: true } }, { text: "layer_fusion: last, hidden_dim: 0    " },
      { text: "B: ", options: { bold: true } }, { text: "layer_fusion: weighted, hidden_dim: 256" }], MX, 5.96, CW, 0.36, { fontSize: 18, valign: "middle", kind: "diagram" });
    const ly = 6.42;
    L.key(s, MX, ly, "frozen");
    L.key(s, MX + 3.0, ly, "train");
    L.key(s, MX + 5.0, ly, "loss");
    L.key(s, MX + 6.6, ly, "out");
    L.key(s, MX + 8.6, ly, C.accent2, "gradient", { arrow: true, dash: "dash" });
    L.notes(s, `Path A/B is the simple and cheap path, with two options for the head. The top band runs once per clip and computes no gradients.
    Each clip is read as 16 kHz mono and normalised per clip, the same way the benchmark's FunASR predictor does it, and must be between 2 and 15 seconds, the same window as preparation_tools; every clip is checked before any weights load, and nothing is cropped. The emotion2vec+ large encoder, 162.1 million parameters, stays frozen and in evaluation mode, and produces 1024-dimensional frames at 50 frames per second. It always embeds one clip at a time, never a padded batch: FunASR's convolutional positional encoder reads padded frames, so padding changed a clip's embedding depending on its batch neighbours. With one clip at a time, batched and single-clip predictions now agree exactly, a difference of 0.000, and match the benchmark's single-clip inference.
    Option A, the default, averages the frames of the last block over the valid frames, ignoring padding, and gets one 1024-dimensional embedding per clip. This is the embedding the released model uses for its own classifier, and the parity test checks that the probabilities match the benchmark predictor.
    Option B averages the output of every one of the 12 Transformer blocks, the 4 prenet and the 8 shared blocks, which gives 12 times 1024 numbers per clip. FunASR's layer_results are feed-forward outputs, not block outputs, so the block outputs are captured with forward hooks; the last of the 12 equals option A's embedding.
    All embeddings are computed once and stored in features.pt with the encoder fingerprint and a digest of the audio list, so a cache can never be reused silently with another encoder or other clips. The large model is then released from memory.
    The bottom band trains for up to 100 epochs on the cached embeddings. Option A's head is a single linear layer from 1024 to 8 classes, 8,200 parameters. Option B first layer-normalises each of the 12 block embeddings, mixes them with 12 learned softmax weights that start equal, at one twelfth each, and then applies an MLP from 1024 to 256 to 8 classes; that is 264,456 parameters for the MLP plus the 12 weights, about 0.26 million. The block weights have their own learning rate of 0.01 and no weight decay, and after training they are written to layer_weights.json.
    Both options use cross-entropy with balanced class weights and train with large batches, which is fine because the inputs are cached features; the gradient only reaches the head and, in option B, the 12 weights. The saved epoch is chosen by validation UAR by default. The encoder stays frozen either way. The option is chosen in config/head.yaml with layer_fusion and hidden_dim, and it is stored in head.pt and model_bundle.json, so prediction uses the same path.`);
    L.cites(s, R, ["e2v", "card"], "Source: adaptation/head.py, backbone.py.");
  }

  // ---- Path A/B recipe
  {
    const s = L.slide(SEC, "Path A/B recipe: unpadded features, selection by validation UAR", { titleSize: 28 });
    L.table(s, ["Setting", "Default (A) · option B", "Why"], [
      [{ text: "Encoder pass", bold: true }, "one clip at a time, never padded", "padding changes FunASR’s output"],
      [{ text: "Embedding", bold: true }, "A: last block mean · B: 12 block means", "A matches the benchmark predictor"],
      [{ text: "Head", bold: true }, "A: Linear 1024 → 8 · B: MLP 1024 → 256 → 8", "simple baselines"],
      [{ text: "Block weights (B)", bold: true }, "12 softmax weights from 1/12; lr 0.01, no decay", "B learns which blocks to use"],
      [{ text: "Loss", bold: true }, "cross-entropy; class weight N / (C × class count)", "rare classes weigh like common ones"],
      [{ text: "Optimiser", bold: true }, "AdamW, lr 0.001, weight decay 0.01, batch 64", "a standard choice for a small head"],
      [{ text: "Selection", bold: true }, "validation UAR (or macro-F1, loss); patience 15", "per-class recall is what matters"],
      [{ text: "Data", bold: true }, "1K, then 5K training clips; 70 / 20 / 10 % of groups", "no customer in two splits"],
      [{ text: "Test", bold: true }, "once: accuracy, UAR, macro-F1, confusion", "the test set never steers training"],
    ], { y: TOP + 0.02, colW: [2.35, 5.65, 4.13], fontSize: 18, rowH: 0.5 });
    L.notes(s, `This is the default recipe from config/head.yaml and config/common.yaml.
    Option A, the default, uses the mean of the last block's valid frames and a linear layer. Option B uses the means of all 12 blocks, mixes them with 12 learned softmax weights that start at one twelfth each and have their own learning rate of 0.01 without weight decay, and feeds the mix to an MLP with 256 hidden units.
    The loss is cross-entropy with balanced class weights computed on the selected training clips only: the weight of class c is N divided by C times n c, where N is the number of training clips, C the number of classes and n c the number of clips of class c. A class with half as many clips gets twice the weight. The curves plot unweighted cross-entropy so that training and validation losses are comparable; the weighted objective is logged separately.
    The optimiser is AdamW with learning rate 0.001, weight decay 0.01 and batch size 64. Training runs at most 100 epochs. The saved epoch and early stopping follow selection_metric: validation UAR by default, macro-F1 or loss on request; ties go to the lower validation loss, and training stops after 15 epochs without improvement. We select by UAR because we care about recall on every emotion; with a weighted loss and imbalanced classes, the unweighted validation loss can pick a different epoch. Each run writes split_counts.json; a class missing from validation is an error when selecting by UAR or macro-F1, and a class with fewer than 10 validation clips gives a warning, because one clip moves its recall by more than 10 points. The encoder embeds one clip at a time, never padded.
    The label budget is applied after the split, to the training set only: 1,000 clips first, then 5,000. The budget sampler keeps at least one clip of every class and samples the rest at random with a fixed seed. If the training split lacks a class, the run stops with an error.
    Splits are made by customer group, 70, 20 and 10 percent of the groups, unless the manifest gives explicit splits. The test set is evaluated once, with the best epoch, and reports accuracy, unweighted average recall, macro-F1, per-class support and the confusion matrix. UAR and macro-F1 average over all configured classes; a missing class counts as zero.`);
    L.cite(s, "Sources: config/head.yaml, config/common.yaml, adaptation/head.py, adaptation/data.py");
  }
};

// Hidden appendix: default settings, data-scale numbers, code map, references
const R = require("./refs");

module.exports = function (L) {
  const { C, W, MX, CW, TOP, FONT } = L;
  const SEC = "Appendix";
  const k = (t) => ({ text: t, mono: true });

  // ---- A1 Shared settings
  {
    const s = L.slide(SEC, "Default settings: shared (common.yaml)", { appendix: true });
    L.table(s, ["common.yaml key", "Default", "Meaning"], [
      [k("runtime.device"), "cuda", "stop if no GPU; cpu / mps only when set"],
      [k("runtime.precision"), "auto", "BF16 autocast on CUDA, FP32 on CPU / MPS"],
      [k("runtime.num_workers"), "4", "DataLoader processes; measure on the B200"],
      [k("audio.min/max_seconds"), "2.0 / 15.0", "checked before training; never cropped"],
      [k("labels"), "8 classes", "angry … surprised, incl. other"],
      [k("data.max_train_clips"), "1000", "head budget, train split only"],
      [k("data.validation_fraction"), "0.2", "of the customer groups"],
      [k("data.test_fraction"), "0.1", "of the customer groups"],
      [k("data.audit_cache"), "outputs/…jsonl", "SHA-256 + header, one read per file"],
    ], { y: TOP + 0.02, colW: [4.4, 1.9, 5.83], fontSize: 18, rowH: 0.5 });
    L.notes(s, `Appendix slide: the defaults of the shared configuration, config/common.yaml, as of 17 November 2026. The audio window is 2 to 15 seconds, the same as preparation_tools; the current internal batches are 2 to 10 seconds. Every path in the YAML files is relative to the repository root; paths inside a CSV are relative to that CSV. Each experiment needs a new output directory.`);
    L.cite(s, "Source: config/common.yaml");
  }

  // ---- A2 Path A/B settings
  {
    const s = L.slide(SEC, "Default settings: Path A/B head (head.yaml)", { appendix: true });
    L.table(s, ["head.yaml key", "Default", "Meaning"], [
      [k("layer_fusion"), "last", "last = option A; weighted = option B"],
      [k("hidden_dim"), "0", "0 = linear (A); 256 = MLP (B)"],
      [k("layer_weights_learning_rate"), "0.01", "B: the 12 block weights, no decay"],
      [k("selection_metric"), "uar", "saved epoch, early stop (macro_f1 | loss)"],
      [k("learning_rate, batch_size"), "0.001, 64", "AdamW, weight decay 0.01"],
      [k("epochs, patience"), "100, 15", "early stop on the selection metric"],
      [k("class_weights"), "balanced", "N / (C × class count), training split"],
    ], { y: TOP + 0.02, colW: [4.4, 1.9, 5.83], fontSize: 18, rowH: 0.55 });
    L.notes(s, `Appendix slide: the defaults of Path A/B in config/head.yaml. Option A is layer_fusion last with hidden_dim 0. Option B is layer_fusion weighted with hidden_dim 256; its 12 block weights use their own learning rate of 0.01 without weight decay, because at the head's learning rate they stay close to equal. selection_metric, uar by default, picks the saved epoch and drives early stopping. extraction_batch_size was removed: the encoder now always embeds one clip at a time.`);
    L.cite(s, "Source: config/head.yaml");
  }

  // ---- A3, A4 Path E settings
  {
    const s = L.slide(SEC, "Default settings: Path E objective (pretrain.yaml)", { appendix: true, titleSize: 28 });
    L.table(s, ["pretrain.yaml key", "Default", "Meaning"], [
      [k("batch_size, epochs"), "2, 10", "teacher + gradients + Adam state"],
      [k("learning_rate"), "1e-5", "encoder, after probe and warm-up"],
      [k("predictor_learning_rate"), "1e-3", "the new predictor"],
      [k("warmup_steps"), "1000", "encoder rate 0 → 1e-5 after the probe"],
      [k("target"), "final", "last block, layer norm; ffn_mean = control"],
      [k("target_top_layers"), "8", "ffn_mean only"],
      [k("frame / utterance_weight"), "1.0, 0.0", "pooled utterance loss off"],
      [k("anchor_weight"), "0.0", "L2-SP to original weights off"],
      [k("mask_probability, _length"), "0.5, 5", "valid-frame fraction; span length"],
      [k("ema_decay, freeze_frontend"), "0.999, true", "fixed decay; CNN not trained"],
    ], { y: TOP - 0.02, colW: [4.4, 1.9, 5.83], fontSize: 18, rowH: 0.46 });
    L.notes(s, `Appendix slide: the objective defaults of Path E in config/pretrain.yaml, as of 17 November 2026. Not shown: gradient_clip 1.0, weight_decay 0.01 and predictor_hidden 1024. The encoder learning rate is zero during the learnability probe, then warms up linearly over warmup_steps.`);
    L.cite(s, "Source: config/pretrain.yaml");
  }
  {
    const s = L.slide(SEC, "Default settings: Path E gates, data and checkpoints", { appendix: true, titleSize: 28 });
    L.table(s, ["pretrain.yaml key", "Default", "Meaning"], [
      [k("probe_steps"), "500", "predictor-only steps, encoder frozen"],
      [k("probe_min_gain"), "0.1", "must beat constant prediction by 10%"],
      [k("downstream_every_steps"), "500", "linear-head check; 0 = off"],
      [k("downstream_max_drop"), "0.02", "stop below original UAR − 0.02"],
      [k("bucket_by_duration"), "true", "no padding; crop to shortest clip"],
      [k("save_every_steps"), "5000", "mid-epoch saves; 0 = epoch ends only"],
      [k("keep_checkpoints"), "5", "newest encoder-only copies kept"],
      [k("validation_fraction"), "0.05", "unlabelled groups held out"],
      [k("max_unlabeled_clips"), "13000", "pilot budget; nested across budgets"],
    ], { y: TOP - 0.02, colW: [4.4, 1.9, 5.83], fontSize: 18, rowH: 0.48 });
    L.notes(s, `Appendix slide: the gate, data and checkpoint defaults of Path E. The downstream check needs data.labeled_manifest; with downstream_every_steps 0 the run can go without labels, but then nothing protects the emotion information. The learnability probe needs SSL validation clips, from validation_fraction or validation_manifest. save_every_steps is best a multiple of downstream_every_steps, so that every kept encoder has a downstream UAR. resume_checkpoint continues exactly, also mid-epoch, into a new output folder.`);
    L.cite(s, "Source: config/pretrain.yaml");
  }

  // ---- A3 Data-scale numbers
  {
    const s = L.slide(SEC, "Data scale in numbers: a 30-batch reference", { appendix: true });
    L.table(s, ["30-batch reference", "100K clips per batch", "200K clips per batch"], [
      [{ text: "Clips", bold: true }, "3,000,000", "6,000,000"],
      [{ text: "Hours at 2–10 s (limits)", bold: true }, "1,667–8,333", "3,333–16,667"],
      [{ text: "Hours at an assumed 5 s mean", bold: true }, "4,167", "8,333"],
      [{ text: "5 s case vs 262 h (emotion2vec SSL)", bold: true }, "15.9×", "31.8×"],
      [{ text: "5 s case vs 42,526 h (plus-large fine-tuning)", bold: true }, "9.8%", "19.6%"],
    ], { y: TOP + 0.05, colW: [5.7, 3.2, 3.23], fontSize: 20, rowH: 0.62, align: ["left", "right", "right"] });
    L.bullets(s, [
      "5 s is an assumption, not a measured mean; the real batch count is above 30 (scale by B / 30)",
      "Exact hours = sum of decoded clip durations / 3,600",
    ], { y: 5.6, h: 1.2, fontSize: 20, gap: 4 });
    L.notes(s, `Appendix slide: the numbers behind the data-scale chart, computed by scripts/make_figures.py. Clips are 30 batches times 100,000 or 200,000. Hours at the limits multiply the clips by 2 and 10 seconds and divide by 3,600. The 5-second case is an illustration only. The ratios compare audio volume across different training stages and do not equate self-supervised and pseudo-label supervised training.`);
    L.cites(s, R, ["e2v", "card"], "Sources:");
  }

  // ---- A4 Code map
  {
    const s = L.slide(SEC, "Code map: what FunASR provides and what we add", { appendix: true });
    const ours = { fill: C.orangeFill }, theirs = { fill: C.blueFill };
    L.table(s, ["File (adaptation/)", "Role", "Origin"], [
      [k("FunASR Emotion2vec"), "CNN, Transformer blocks, native mask application, weights", { text: "reused", ...theirs }],
      [k("backbone.py"), "loading, one-clip embeddings, 12-block embeddings (B), fingerprint", { text: "new wrapper", ...ours }],
      [k("ssl.py"), "span mask, targets final / ffn_mean, predictor, loss, EMA", { text: "new", ...ours }],
      [k("pretrain.py"), "SSL loop, probe gate, warm-up, checkpoints, exact resume", { text: "new", ...ours }],
      [k("downstream.py"), "downstream gate: linear head, UAR, cosine, rank", { text: "new", ...ours }],
      [k("head.py"), "embedding cache, heads A and B, UAR selection, bundle", { text: "new", ...ours }],
      [k("inference.py"), "matched prediction, mismatch refusal", { text: "new", ...ours }],
      [k("data.py, audio.py"), "group splits, leakage checks, buckets, clip checks", { text: "new", ...ours }],
      [k("audit.py"), "audio audit cache: SHA-256 + header, reuse", { text: "new", ...ours }],
    ], { y: TOP - 0.02, colW: [3.0, 7.23, 1.9], fontSize: 18, rowH: 0.5 });
    L.notes(s, `Appendix slide: the division of work. FunASR provides the released encoder, the convolution and Transformer modules, the native application of a precomputed latent mask, and the weights. This project adds one-clip embedding, the 12-block embeddings of option B captured with forward hooks, the two SSL targets, the learnability and downstream gates, the audio audit cache, duration buckets, step checkpoints with exact resume, the mask sampling, the padding-aware target normalisation, the MLP predictor, the loss, the EMA teacher, the training loop with validation, resume and export, the head training, safe prediction, data handling and monitoring. It is neither a rewrite of emotion2vec from scratch nor a call to an official full SSL trainer; the official version we checked returns features but no complete SSL loss.`);
    L.cite(s, "Sources: adaptation/, utilities/, docs/guide.md");
  }

  // ---- A5, A6 References
  const keys = ["e2v", "card", "repo", "data2vec", "d2v2", "fairseq", "wav2vec2", "byol", "meanteacher", "robust", "dapt", "funasr", "ravdess", "cremad", "dailytalk", "torch27", "cudagpus"];
  const pages = [keys.slice(0, 6), keys.slice(6, 12), keys.slice(12)];
  const said = [
    `References, part one. All entries were checked online on 17 November 2026: title, authors, venue, pages where available, and that the link resolves. The emotion2vec paper and the emotion2vec+ large model card give the original method and the data volumes; the official repository is the code version we checked; data2vec, data2vec 2.0 and the fairseq implementation describe the teacher targets, masking and decoder.`,
    `References, part two. wav2vec 2.0 introduced span masking of latent speech frames; BYOL and Mean Teacher are the classic EMA-teacher methods; Robust wav2vec 2.0 and Don't Stop Pretraining motivate continued pretraining on target-domain data; FunASR provides the encoder implementation we load.`,
    `References, part three. RAVDESS, under CC BY-NC-SA 4.0, is the public speech of the functional validation; CREMA-D and DailyTalk are the public corpora of the end-to-end mock test; the PyTorch 2.7 release and NVIDIA's compute-capability list support the B200 environment.`,
  ];
  pages.forEach((page, i) => {
    const s = L.slide(SEC, `References (${i + 1}/${pages.length})`, { appendix: true });
    const runs = [];
    page.forEach((key) => {
      const r = R[key];
      runs.push({ text: r.full + "  ", options: { breakLine: false, paraSpaceAfter: 10 } });
      runs.push({ text: r.url.replace(/^https:\/\//, ""), options: { color: C.link, underline: { style: "sng", color: C.link }, hyperlink: { url: r.url }, breakLine: true, paraSpaceAfter: 10 } });
    });
    L.text(s, runs, MX, TOP - 0.05, CW, 5.25, { fontSize: 16, ls: 1.05, kind: "diagram" });
    L.notes(s, said[i]);
  });
};

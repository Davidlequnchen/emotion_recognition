// Combining encoder and head, leakage control, data scale, running on the B200, monitoring, validation, plan
const R = require("./refs");

module.exports = function (L) {
  const { C, W, MX, CW, TOP, FONT } = L;
  const p = L.pptx;

  // ---- Encoder + head: why retrain, fingerprint, bundle
  {
    const s = L.slide("Using the result", "A new encoder needs a newly trained head");
    const rx = MX, rw = 2.0, cx = [2.85, 4.85], cw = 1.85, ry = [2.5, 3.9], ch = 1.2;
    L.text(s, "head trained on the … encoder", cx[0], 1.68, 2 * cw + 0.15, 0.34, { fontSize: 18, bold: true, align: "center", kind: "diagram" });
    ["original", "adapted"].forEach((t, j) => L.text(s, t, cx[j], 2.06, cw, 0.36, { fontSize: 18, align: "center", valign: "middle", kind: "diagram" }));
    ["original\nencoder", "adapted\nencoder.pt"].forEach((t, i) => L.box(s, rx, ry[i], rw, ch, t, { role: "frozen", fontSize: 18, bold: true }));
    const cell = (i, j, ok, t) => L.box(s, cx[j], ry[i], cw, ch, [
      { text: ok ? "✓" : "✗", options: { fontSize: 30, bold: true, color: ok ? C.darkGreen : C.darkRed, breakLine: true } },
      { text: t, options: { fontSize: 18, bold: true, color: ok ? C.darkGreen : C.darkRed } },
    ], { fill: ok ? C.greenFill : C.redFill, line: ok ? C.good : C.bad, fontSize: 18 });
    cell(0, 0, true, "Path A/B bundle");
    cell(0, 1, false, "refused");
    cell(1, 0, false, "refused");
    cell(1, 1, true, "Path E bundle");
    L.text(s, "rows: the encoder loaded at prediction time", rx, 5.22, 6.0, 0.34, { fontSize: 16, kind: "diagram" });
    // bundle contents and checks
    const tx = 7.0, tw = W - MX - tx;
    L.text(s, "model_bundle.json (simplified)", tx, 1.68, tw, 0.36, { fontSize: 18, bold: true, color: C.link });
    L.code(s, [
      "{",
      '  "model_directory": "<original model>",',
      '  "encoder_checkpoint": "encoder.pt",',
      '  "head_checkpoint": "head.pt",',
      '  "encoder_fingerprint": "<SHA-256>",',
      '  "layer_fusion": "last",',
      '  "labels": [ ... 8 classes ... ],',
      '  "audio": { ... }',
      "}",
    ], tx, 2.08, tw, 2.9, { fontSize: 16 });
    L.bullets(s, [
      "fingerprint: loaded encoder = bundle = head",
      "same architecture (config hash)",
      "same labels, audio, layer_fusion",
    ], { x: tx, y: 5.06, w: tw, h: 1.1, fontSize: 20, gap: 2 });
    L.box(s, MX, 6.2, CW, 0.5, "Head/encoder mismatch: retrain the classification head with the selected encoder",
      { fill: C.white, line: C.bad, mono: true, fontSize: 16, color: C.darkRed, r: 0.04 });
    L.notes(s, `Path E produces only an encoder. A head is a set of decision boundaries in the embedding space of one particular encoder, and continued pretraining moves that space, so an old head on a new encoder has no reliable meaning. That is why the head must be retrained on the adapted encoder.
    The grid shows the four combinations. The original encoder with a head trained on it is the Path A/B bundle. The adapted encoder with a head trained on it is the Path E bundle. The two mixed combinations are refused.
    The refusal is automatic. Every encoder has a fingerprint, a SHA-256 hash over the names, shapes, data types and bytes of all its weights. head.pt stores the fingerprint of the encoder it was trained on. model_bundle.json, on the right, names the original model folder, the replacement encoder.pt or null for the original encoder, the head checkpoint, the encoder fingerprint, the head option in layer_fusion, last for option A or weighted for option B, and the labels and audio settings. When prediction loads a bundle, it recomputes the fingerprint of the encoder it actually loaded and requires it to equal both the bundle's and the head's; it also checks the architecture configuration hash and that labels, audio settings and the head option agree, so a bundle that says weighted can never load an option-A head. Otherwise it stops with the error at the bottom. The smoke test and the public validation both check that a deliberately mismatched bundle is refused.
    The bundle is a local manifest: it does not copy the original weights. To move a model to another machine, copy the original model folder, the head and the replacement encoder, and update the paths. The original model files are never modified.`);
    L.cite(s, "Sources: adaptation/inference.py, adaptation/head.py, adaptation/backbone.py state_fingerprint()");
  }

  // ---- Leakage control and splits
  {
    const s = L.slide("Data", "No validation or test customer ever reaches training");
    const bx = 3.25, bw = 2.2, sx = 5.85, sw = 2.35, ux = 8.55, uw = W - MX - ux, h = 0.52;
    // labelled
    L.box(s, MX, 1.95, 2.3, 1.3, "Labelled\nmanifest\npath, label, group", { role: "data", fontSize: 18 });
    L.arrow(s, MX + 2.3, 2.6, bx, 2.6);
    L.box(s, bx, 1.75, bw, 1.7, "split by\ngroup:\ncustomer\n(or call)", { fill: C.panel, line: C.ink, fontSize: 18, bold: true });
    const lab = [["train  70%", "head training (1K / 5K budget)"], ["validation  20%", "early stopping, model choice"], ["test  10%", "opened once, at the end"]];
    lab.forEach(([t, u], i) => {
      const y = 1.75 + i * 0.59;
      L.box(s, sx, y, sw, h, t, { role: "data", fontSize: 18, bold: true });
      L.arrow(s, bx + bw, y + h / 2, sx, y + h / 2);
      L.arrow(s, sx + sw, y + h / 2, ux, y + h / 2);
      L.text(s, u, ux + 0.1, y, uw - 0.1, h, { fontSize: 18, valign: "middle", kind: "diagram" });
    });
    // unlabelled
    const y2 = 4.25;
    L.box(s, MX, y2, 2.3, 1.2, "Unlabelled\nclips\npath, group", { role: "data", fontSize: 18 });
    L.arrow(s, MX + 2.3, y2 + 0.6, bx, y2 + 0.6);
    L.box(s, bx, y2, bw, 1.2, "remove: holdout\ngroups, same paths,\nidentical bytes", { fill: C.white, line: C.bad, dash: "dash", lw: 2, fontSize: 16 });
    [["SSL train", "teacher–student updates"], ["SSL validation", "5% of groups: loss only"]].forEach(([t, u], i) => {
      const y = y2 + i * 0.68;
      L.box(s, sx, y, sw, h, t, { role: "data", fontSize: 18, bold: true });
      L.arrow(s, bx + bw, y + h / 2, sx, y + h / 2);
      L.arrow(s, sx + sw, y + h / 2, ux, y + h / 2);
      L.text(s, u, ux + 0.1, y, uw - 0.1, h, { fontSize: 18, valign: "middle", kind: "diagram" });
    });
    // labelled validation + test are excluded from SSL
    const tcx = sx + sw / 2, tb = 1.75 + 2 * 0.59 + h, yLane = 3.86;
    L.path(s, [[tcx, tb], [tcx, yLane], [bx + bw / 2, yLane], [bx + bw / 2, y2]], { color: C.bad, width: 2, dash: "dash" });
    L.text(s, "validation + test groups: excluded from SSL", tcx + 0.15, tb + 0.06, 5.0, 0.34, { fontSize: 16, bold: true, color: C.darkRed, kind: "diagram" });
    L.box(s, MX, 5.9, CW, 0.8, [
      { text: "Any shared group, path or identical file stops the run with an error.", options: { bold: true, breakLine: true } },
      { text: "Re-encoded or trimmed copies are not detected: correct customer IDs are essential." },
    ], { fill: C.amberFill, line: C.warn, fontSize: 18, align: "left", margin: [3, 12, 3, 12] });
    L.notes(s, `Leakage is the easiest way to fool ourselves here, because one customer can produce many clips with the same voice and the same phone line. The rule is simple: a customer in validation or test must never reach any training step, in either path.
    Top row: the labelled manifest is split by group, which is the customer ID, or the call ID if the customer is unknown. By default 70 percent of the groups go to training, 20 to validation and 10 to test, with a fixed seed; explicit splits in the manifest are also accepted. The training split is used for the head, within the label budget; validation drives early stopping; the test set is opened once at the end.
    Bottom row: the unlabelled clips for self-supervised learning. Before training, the code removes every clip whose group is in the labelled validation or test set, every clip with the same path, and every file whose bytes are identical to a held-out file, compared by SHA-256. This is the red dashed connection. Then 5 percent of the remaining groups are held out as an SSL validation set, which only measures the loss and never updates the weights. A fixed SSL validation manifest can replace the automatic split; that is recommended when comparing data scales.
    Any overlap of groups, paths or file contents between partitions stops the run with an error rather than a warning.
    The limitation is in the amber box: a byte comparison cannot recognise a copy that was re-encoded or trimmed. Correct customer IDs are therefore essential, and the held-out customers must be fixed before either experiment starts, so that both paths and all future label batches use the same holdouts.`);
    L.cite(s, "Sources: adaptation/data.py split_records(), assert_disjoint_splits(), read_unlabeled(); adaptation/pretrain.py");
  }

  // ---- Data scale
  {
    const s = L.slide("Data", "Even a 30-batch reference holds 6–64× the original SSL audio", { titleSize: 28 });
    L.image(s, "img/data_scale.png", MX, TOP + 0.05, CW, 3.95, { alt: "Audio hours on a log scale: 262 h, 42,526 h and the internal 30-batch reference ranges" });
    L.text(s, [
      { text: "Orange bar: ", options: { bold: true } }, { text: "the 2–10 s clip-length limits.  " },
      { text: "Dot: ", options: { bold: true } }, { text: "an assumed 5 s mean, an illustration only." },
    ], MX, 5.72, CW, 0.38, { fontSize: 20 });
    L.text(s, [
      { text: "Both assume 30 batches; the real number is above 30, so totals scale by B / 30.  ", options: {} },
      { text: "262 h and 42,526 h are different training stages: never add them up.", options: { bold: true } },
    ], MX, 6.14, CW, 0.66, { fontSize: 20 });
    L.notes(s, `How much audio is this compared with the original training? The chart uses audio hours on a log scale, because clip counts are not comparable across datasets with different segmentation.
    The two blue bars are published numbers. The emotion2vec paper used 262 hours, 169,053 utterances, of public emotional speech for self-supervised pre-training. The emotion2vec+ large model card lists 42,526 hours of filtered pseudo-labelled data, selected from a pool of about 160,000 hours, as fine-tuning data. These are different stages, self-supervised pre-training versus pseudo-label fine-tuning, so they must not be added up or compared as the same thing; the full training history of the large model is not documented in these sources.
    The orange bars are our reference, not a measurement. We know there are more than 30 batches of about 100,000 to 200,000 clips each, every clip strictly between 2 and 10 seconds. For exactly 30 batches that is 3 to 6 million clips. The orange bars show the range allowed by the length limits: 1,667 to 8,333 hours for 100,000 clips per batch, and 3,333 to 16,667 hours for 200,000. The dots assume a mean of 5 seconds, which is an illustration, not a measured value: 4,167 and 8,333 hours, that is about 16 and 32 times the 262 hours, or about 10 and 20 percent of the 42,526 hours.
    So even the most conservative reading, 30 batches of 100,000 two-second clips, is more than six times the original self-supervised audio. Because the real number of batches is above 30, all these numbers scale up by B over 30. The exact total is the sum of the decoded clip durations divided by 3,600; it should be computed from the manifests.
    More audio does not automatically mean better emotion representations, and it does not mean the first run should use all of it. The plan at the end starts small.`);
    L.cites(s, R, ["e2v", "card"], "Sources:");
  }

  // ---- Million-clip scale
  {
    const s = L.slide("Data", "Million-clip scale: one read per file, no padding, exact resume", { titleSize: 28 });
    const cards = [
      ["Audio audit", C.accent, ["SHA-256 + header, one read per file", "reused while size and mtime match", "only the clips in use are hashed"], "data.audit_cache"],
      ["No padding", C.accent2, ["batches of near-equal duration", "cropped to the shortest clip", "0.4–0.7% audio dropped (simulated) vs 25–72% padding"], "bucket_by_duration: true"],
      ["Checkpoints", C.good, ["every 5,000 steps; newest 5 kept", "exact resume, also mid-epoch", "bit-identical encoder (CPU test)"], "save_every_steps: 5000"],
      ["Loading", C.ink, ["4 workers decode, resample, crop", "data_wait_seconds logged", "right value: measure on the B200"], "num_workers: 4"],
    ];
    const gap = 0.3, cw = (CW - 3 * gap) / 4, cy = TOP + 0.05, ch = 4.2;
    cards.forEach(([t, col, lines, key], i) => {
      const x = MX + i * (cw + gap);
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: ch, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      L.text(s, t, x + 0.2, cy + 0.22, cw - 0.4, 0.45, { fontSize: 22, bold: true });
      L.bullets(s, lines, { x: x + 0.2, y: cy + 0.82, w: cw - 0.3, h: 2.5, fontSize: 18, gap: 6, kind: "diagram" });
      L.box(s, x + 0.2, cy + ch - 0.84, cw - 0.4, 0.66, key, { fill: C.white, line: C.line, fontSize: 16, bold: true, color: C.link, r: 0.04 });
    });
    L.text(s, [{ text: "Budgets are nested: ", options: { bold: true } }, { text: "the 13K sample is a subset of the 65K sample (seeded path hash), so later stages reuse the audit." }],
      MX, 6.08, CW, 0.66, { fontSize: 20 });
    L.notes(s, `Four changes make the code ready for millions of clips.
    The audio audit: every file is read completely once, and its size, modification time, SHA-256, frame count, sample rate and duration are stored in the audit cache, by default outputs/audio_audit.jsonl. As long as size and modification time are unchanged, later runs reuse the entry instead of reading the file again. Leakage checks, resume digests and duration checks all read this cache; leakage is still checked on byte content. Group and path exclusions run on all candidates, but content hashes are computed only for the clips actually used, so a 13K pilot reads about 13K files, not the whole source.
    No padding: FunASR's convolutional positional encoder reads padded frames, so padding would change a clip's representation depending on its batch. With bucket_by_duration, each epoch shuffles the clips, sorts pools of 100 batches by duration and cuts them into batches; every clip in a batch is cropped to the batch's shortest clip at a seeded offset and normalised again. In a simulation with durations spread evenly over 2 to 15 seconds, this drops about 0.4 to 0.7 percent of the audio for batch sizes 2 to 32, against 25 to 72 percent padding without bucketing. The real share is logged as cropped_fraction.
    Checkpoints: by default every 5,000 steps and at every epoch end the run writes last_training.pt and encoder.pt, plus an encoder-only checkpoint, of which the newest 5 are kept. Resume is exact, also in the middle of an epoch: the batch order depends only on seed and epoch, consumed batches are skipped without decoding audio, and a CPU unit test shows a bit-identical encoder against an uninterrupted run. GPU kernels do not guarantee bit-identical results.
    Loading: num_workers 4 decodes, resamples, crops and normalises in background processes. Four is only a starting point; the right value must be measured on the B200 from data_wait_seconds.
    Budgets are nested: the sample is chosen by a seeded hash of the path, so the 13K sample is contained in the 65K sample and later stages reuse the audit.`);
    L.cite(s, "Sources: docs/guide.md (million-clip section), adaptation/audit.py, adaptation/audio.py, config/pretrain.yaml");
  }

  // ---- Running on the B200
  {
    const s = L.slide("Running", "On the B200: the benchmark's environment, one check, three entry points", { titleSize: 28 });
    const cm = (t) => ({ text: t, options: { color: C.link, bold: true } });
    L.code(s, [
      [{ text: "pip install -r emotion_recognition_benchmark/requirements-llm-cuda128.txt " }, cm("# once")],
      [{ text: "pip install -r emotion_recognition_benchmark/requirements.txt               " }, cm("# once")],
      [{ text: "python utilities/check_environment.py             " }, cm("# GPU check")],
      [{ text: "python train_head.py                              " }, cm("# Path A/B")],
      [{ text: "python pretrain_encoder.py                        " }, cm("# Path E")],
      [{ text: "python predict.py                                 " }, cm("# predictions")],
    ], MX, TOP + 0.08, CW, 2.25, { fontSize: 18 });
    L.table(s, ["Setting", "Value"], [
      [{ text: "PyTorch, torchaudio", bold: true }, "2.7.1, CUDA 12.8 build (Mac: same version)"],
      [{ text: "GPU", bold: true }, "B200 (Blackwell), compute capability 10.0"],
      [{ text: "Precision", bold: true }, "BF16 autocast; weights, Adam state, loss in FP32"],
      [{ text: "Device", bold: true }, "cuda: stops if no GPU, no silent CPU fallback"],
      [{ text: "Scale", bold: true }, "one GPU; DataLoader with 4 workers"],
    ], { x: MX, y: 4.2, w: 7.75, colW: [2.45, 5.3], fontSize: 18, rowH: 0.43 });
    L.box(s, 8.65, 4.2, W - MX - 8.65, 2.58, [
      { text: "Not run on a B200 yet", options: { bold: true, fontSize: 22, color: C.darkRed, breakLine: true } },
      { text: "Mac only so far. To measure on the server: workers (data_wait_seconds), memory, throughput.", options: { fontSize: 20 } },
    ], { fill: C.white, line: C.bad, lw: 2, align: "left", valign: "top", margin: [10, 14, 8, 14], fontSize: 20 });
    L.notes(s, `Running it on the B200 takes the benchmark's environment, one check and three entry points; every setting comes from the YAML files in config, without command-line arguments.
    This project shares the emotion_recognition_llm environment with the benchmark, because its packages are the same versions. On the server, the hardware file is the benchmark's requirements-llm-cuda128.txt, then its requirements.txt, both installed once in a new Python 3.11 environment. The CUDA 12.8 build is needed because the B200 is a Blackwell GPU with compute capability 10.0: PyTorch added Blackwell kernels with its CUDA 12.8 builds in version 2.7, and the CUDA 11.8 build used on the older server cannot run on it. The version, 2.7.1, is the same as on the Mac; only the CUDA build differs.
    check_environment.py records Python, PyTorch, CUDA, cuDNN, the driver and the GPU, and runs a real matrix multiply, a convolution, an FP32 loss and a backward pass. It refuses an old CUDA build on Blackwell.
    Then the three entry points: train_head.py for Path A/B, pretrain_encoder.py for Path E, predict.py for predictions from a bundle. After SSL, point model.encoder_checkpoint in common.yaml at the new encoder.pt, give the head a new output folder, and run train_head.py again.
    On CUDA, precision auto uses BF16 autocast while weights, Adam state, target normalisation and loss reduction stay in FP32. The default device is cuda, and the run stops if no GPU is visible instead of silently falling back to the CPU. Training uses one GPU, with four DataLoader workers by default; multi-GPU data parallel training is not implemented.
    The red box is the honest status: nothing has run on a B200 yet. The large-model tests ran on the Mac, on the CPU and, in the end-to-end mock test, on the Apple GPU through MPS in FP32. There, self-supervised training ran at about one step per second with batch size 4, roughly five clips per second, with a peak memory of about 43 GB. These Mac numbers do not predict the B200; the number of workers, memory, throughput and the right batch size have to be measured on the server, using the new step metrics data_wait_seconds, clips per second with and without waiting, and the CUDA memory counters.`);
    L.cites(s, R, ["torch27", "cudagpus"], "Sources: emotion_recognition_benchmark/requirements-llm-cuda128.txt, utilities/hardware.py;");
  }

  // ---- Monitoring
  {
    const s = L.slide("Running", "Watch every run live: gates, loss baseline and data speed");
    const lw = 5.3;
    L.code(s, [
      "outputs/<experiment>/",
      "  logs/metrics.jsonl",
      "  monitoring/dashboard.html",
      "  probe.json",
      "  downstream_history.jsonl",
      "  encoder.pt, encoder_best.pt",
      "  checkpoints/  last_training.pt",
      "  data_audit.json",
    ], MX, TOP + 0.05, lw, 2.85, { fontSize: 18 });
    L.code(s, [
      "python utilities/monitor_training.py",
      "ssh -L 8766:127.0.0.1:8766 <server>",
    ], MX, 4.7, lw, 0.88, { fontSize: 16 });
    L.text(s, "then open 127.0.0.1:8766 locally", MX, 5.66, lw, 0.36, { fontSize: 18, kind: "diagram" });
    const tx = 6.2, tw = W - MX - tx;
    L.table(s, ["Signal (plot or log)", "Stop and check if …"], [
      [{ text: "SSL loss vs constant baseline", bold: true }, "the loss stays on the baseline"],
      [{ text: "downstream UAR", bold: true }, "drops > 0.02: auto stop"],
      [{ text: "cosine, spread, eff. rank", bold: true }, "spread or rank collapses"],
      [{ text: "gradient norm", bold: true }, "spikes again and again"],
      [{ text: "data_wait_seconds", bold: true }, "high: add workers"],
      [{ text: "CUDA / MPS / RSS memory", bold: true }, "close to the limit"],
    ], { x: tx, y: TOP + 0.05, w: tw, colW: [3.15, tw - 3.15], fontSize: 18, rowH: 0.54 });
    L.box(s, tx, 5.72, tw, 0.95, [
      { text: "target_std is gone: ", options: { bold: true } },
      { text: "normalisation keeps it constant even if all clips collapse." },
    ], { fill: C.blueFill, line: C.accent, fontSize: 20, align: "left", margin: [3, 12, 3, 12] });
    L.notes(s, `Every run writes everything into its own new folder. metrics.jsonl keeps every recorded step; the dashboard refreshes every 5 seconds; probe.json holds the learnability result; downstream_history.jsonl holds every downstream check; encoder.pt is the latest encoder and encoder_best.pt the best by the downstream check; checkpoints and last_training.pt allow comparison and exact resume; data_audit.json records clip counts, hours, holdout exclusions and how many audit entries were read or reused. On a remote server, monitor_training.py serves the outputs folder read-only on the loopback address, port 8766; forward the port with SSH or the IDE.
    The SSL plot now draws the constant-prediction baseline next to the loss: a loss that stays on this line means the target is not being learned, exactly what happened with the old objective. The panel that used to show the target standard deviation is replaced by the downstream UAR, because target_std cannot reveal collapse: the normalisation holds it near a constant even if every clip's representation becomes the same. Collapse is visible instead in the downstream check's cosine to the original, spread across clips and effective rank, and in the UAR itself, which stops the run automatically when it falls 0.02 below the original.
    The new step metrics answer the B200 questions: data_wait_seconds is the time spent waiting for the next batch; clips per second is measured over wall time including waiting, logging and checks, and compute clips per second only over the forward and backward pass and the update; cropped_fraction is the audio dropped by bucketing; and the memory counters cover CUDA allocated, reserved and peak, MPS allocated and driver memory, and the process's peak resident memory. If reserved or driver memory is far above allocated, the difference is cache, not model tensors.`);
    L.cite(s, "Sources: docs/guide.md (monitoring), utilities/monitoring.py, config/common.yaml");
  }

  // ---- Validation 1: what works
  {
    const s = L.slide("Status", "What works: Path A/B is ready, and the software checks pass", { titleSize: 28 });
    const ok = { fill: C.greenFill };
    L.table(s, ["Check", "Result"], [
      [{ text: "99 unit tests", bold: true }, { text: "all pass (4 CUDA-only tests skip on the Mac)", ...ok }],
      [{ text: "Smoke test, real weights", bold: true }, { text: "both paths run; benchmark-predictor parity error 0.0", ...ok }],
      [{ text: "Batched vs single-clip prediction", bold: true }, { text: "difference 0.000 (was up to 0.29): no padding", ...ok }],
      [{ text: "Clip check before loading weights", bold: true }, { text: "a 16.4 s clip stops the run after ≈ 5 s, before step 1", ...ok }],
      [{ text: "Mock test, Path A/B", bold: true }, { text: "options A and B: head, bundle, predict.py (MP3 too) fine", ...ok }],
      [{ text: "Mock test, Path E", bold: true }, { text: "both gates and the exact mid-epoch resume work", ...ok }],
      [{ text: "Public speech: RAVDESS, 24 clips", bold: true }, { text: "train, resume, export, retrain, predict: pass", ...ok }],
    ], { y: TOP + 0.02, colW: [4.75, 7.38], fontSize: 18, rowH: 0.54 });
    L.box(s, MX, 6.12, CW, 0.6, [
      { text: "Mock baseline (option A): ", options: { bold: true } },
      { text: "test accuracy 0.708, validation UAR 0.750." },
    ], { fill: C.greenFill, line: C.good, fontSize: 20, align: "left", margin: [3, 14, 3, 14] });
    L.notes(s, `What works so far.
    99 unit tests pass, covering data splits and leakage checks, the two head options and model selection, masking, both SSL targets, the loss and its derivative on masked valid frames only, the EMA update, the gates, bucketing, step checkpoints and exact resume, encoder export and fingerprints, hardware checks, monitoring and plots; four CUDA-only tests are skipped on the Mac. The smoke test with the real emotion2vec+ large weights runs both paths, and its parity check against the benchmark predictor now shows an error of 0.0.
    The padding problem is fixed: the encoder embeds one clip at a time for head features and prediction, and batched and single-clip predictions now differ by 0.000, where they used to differ by up to 0.29. The clip check runs before any weights load: a 16.4-second clip stops the run after about 5 seconds, before the first step.
    The end-to-end mock test uses real weights on the Mac with CREMA-D resampled to 8 kHz, 6 classes, 40 speakers as customer groups and a group split of 504 training, 144 validation and 72 test clips. Path A/B works in both options, including the bundle and predict.py with MP3 input. Path E's two gates and the exact mid-epoch resume work as designed on mock data. The RAVDESS public-speech validation still passes.
    The baseline number has changed: with one-clip features and selection by validation UAR, the original encoder with an option-A head reaches 0.708 test accuracy and 0.750 validation UAR. The earlier 0.750 test accuracy used padded batch features and selection by validation loss. Both changes are within the plus or minus 5 points of noise that 72 test clips allow.`);
    L.cites(s, R, ["ravdess", "cremad"], "Sources: tests/, utilities/, mock test of 17 Nov 2026;");
  }

  // ---- Validation 1b: option A vs option B on mock data
  {
    const s = L.slide("Status", "Option B works; 72 test clips cannot decide between A and B", { titleSize: 28 });
    const tw = 6.25;
    const r = (name, v, fill) => [{ text: name, bold: true, ...fill }, ...v.map((x) => ({ text: x, align: "right", ...fill }))];
    L.table(s, ["Head", "Val. acc.", "Test acc.", "UAR", "Macro-F1"], [
      r("A · last block", ["0.729", "0.750", "0.750", "0.751"], {}),
      r("B · 12 blocks", ["0.771", "0.750", "0.750", "0.739"], {}),
    ], { x: MX, y: TOP + 0.05, w: tw, colW: [1.85, 1.1, 1.1, 1.0, 1.2], fontSize: 18, rowH: 0.52 });
    L.text(s, "CREMA-D 8 kHz, 6 classes, test 72 clips; run before the one-clip fix", MX, 3.33, tw, 0.36, { fontSize: 16, kind: "diagram" });
    L.bullets(s, [
      { runs: [{ text: "Same test accuracy; ", options: { bold: true } }, { text: "72 clips give about ±5 points" }] },
      { runs: [{ text: "B leans on blocks 7–10 ", options: { bold: true } }, { text: "(0.10–0.13), least on 1–4 and 12" }] },
      { runs: [{ text: "Single-block probe: ", options: { bold: true } }, { text: "block 9 best (val. 0.779; block 12: 0.729)" }] },
    ], { x: MX, y: 3.9, w: tw, h: 2.2, fontSize: 20, gap: 8 });
    L.image(s, "img/block_weights.png", MX + tw + 0.25, TOP + 0.05, W - MX - (MX + tw + 0.25), 3.9, { alt: "Learned weights of the 12 encoder blocks in option B" });
    L.text(s, "learned block weights of option B (start: 1/12 each)", MX + tw + 0.25, 5.65, W - MX - (MX + tw + 0.25), 0.36, { fontSize: 16, align: "center", kind: "diagram" });
    L.box(s, MX, 6.2, CW, 0.54, "Compare A and B on real labelled call clips before choosing one.",
      { fill: C.blueFill, line: C.accent, fontSize: 20, bold: true, align: "left", margin: [3, 14, 3, 14] });
    L.notes(s, `Option B, the learned mix of all 12 blocks, was run on the same mock split as option A, before the one-clip fix and the UAR-based selection, so both rows used padded batch features: CREMA-D resampled to 8 kHz, 6 classes, a test set of 72 clips from 4 unseen speakers.
    On validation, option B is better, 0.771 against 0.729 accuracy. On the test set the two are equal in accuracy and UAR, 0.750 each, and B's macro-F1 is slightly lower, 0.739 against 0.751. With only 72 test clips, one clip is 1.4 points and the uncertainty is about plus or minus 5 points, so these numbers do not decide between A and B.
    The chart shows what option B learned. All weights start at one twelfth, the dashed line. After training, the weights peak at blocks 7 to 10, between 0.10 and 0.13, and are lowest, about 0.06, for the four prenet blocks and for the last block, block 12, which is the only block option A uses. A separate check agrees: a linear probe on a single block, averaged over 5 seeds, is best at block 9 with 0.779 validation accuracy, while block 12 reaches 0.729. So on this mock data the middle-upper blocks carry more usable emotion information for the telephone-band clips than the last block.
    The conclusion is modest: option B works end to end, prediction uses the same path, a bundle and head with different options are refused, and the learned weights are interpretable. Whether A or B is better on our calls must be decided on real labelled call clips, on the same customer-disjoint validation set.`);
    L.cite(s, "Source: end-to-end mock test, 17 Nov 2026; layer_weights.json of the option B run");
  }

  // ---- Validation 2: mock-test results of Path E
  {
    const s = L.slide("Status", "Mock test: the gates stop harmful runs; no reliable gain yet", { titleSize: 28 });
    const good = { fill: C.greenFill }, bad = { fill: C.redFill }, stop = { fill: C.amberFill };
    const r = (name, probe, test, f) => [{ text: name, bold: true }, { text: probe, ...f }, { text: test, ...f }];
    L.table(s, ["Configuration", "Learnability probe", "Head test (72 clips)"], [
      r("Original encoder", "—", "0.708", good),
      r("Old default SSL (previous objective)", "— (no gate then)", "0.542–0.583", bad),
      r("ffn_mean target", "gain −1.7%: stopped", "encoder untouched", stop),
      r("final, run 1 (no bucketing)", "24.7%: pass", "best 0.764 · last 0.736", {}),
      r("final, run 2 (bucketed; resumed at step 150)", "10.1%: pass (just)", "best 0.694", {}),
      r("final + utterance loss", "23.3%: pass", "downstream UAR ≈ final", {}),
      r("final, unlabelled = DailyTalk", "6%: stopped", "encoder untouched", stop),
    ], { y: TOP + 0.02, colW: [5.35, 3.4, 3.38], fontSize: 18, rowH: 0.5 });
    L.text(s, "CREMA-D 8 kHz, 6 classes, test 72 clips (±5 points); probe 100 steps, then 300 encoder steps", MX, 5.68, CW, 0.34, { fontSize: 16, kind: "diagram" });
    L.bullets(s, [
      { runs: [{ text: "Gates: ", options: { bold: true } }, { text: "both unlearnable set-ups stopped before any encoder change" }] },
      { runs: [{ text: "final: ", options: { bold: true } }, { text: "no harm, but +5.6 points (run 1) did not replicate (−1.4 in run 2)" }] },
    ], { y: 6.06, h: 0.82, fontSize: 20, gap: 2 });
    L.notes(s, `These are the mock results of the redesigned Path E. Labelled data: CREMA-D at 8 kHz, 6 classes, 504, 144 and 72 clips split by speaker. Unlabelled data: 384 CREMA-D clips of 30 other speakers. Batch size 4, a learnability probe of 100 steps, then 300 encoder steps, with a downstream check every 50 steps. The last column is a full head trained on each encoder and evaluated once on the 72 test clips; one clip is 1.4 points and the noise is about plus or minus 5 points.
    The original encoder reaches 0.708. The old default objective, from the first round and without gates, dropped the head to 0.542 to 0.583.
    The ffn_mean target, the old objective, now fails the learnability probe: its gain over a constant prediction is minus 1.7 percent, so the run stops and the encoder is untouched. The same happens when the unlabelled data are DailyTalk: only two voice actors, so a constant prediction is already good, the gain is 6 percent, below the 10 percent threshold, and the run stops.
    The final target passes the probe. In run 1, without bucketing, the gain was 24.7 percent; the best encoder by the downstream check, at step 300, gave a head test accuracy of 0.764, and the last encoder 0.736. Run 2 used bucketing, was interrupted at step 150 and resumed to step 400, which also exercised the exact resume; its probe gain was 10.1 percent, just above the threshold, and its best encoder, at step 400, gave 0.694. Adding the pooled utterance loss passed the probe with 23.3 percent and gave about the same downstream UAR as final alone.
    The conclusion, stated plainly: the gates stop harmful configurations, and the final target does not harm the encoder on mock data, but it shows no reliable gain. The plus 5.6 points of run 1 did not replicate; run 2 was minus 1.4 points, one clip, well within the noise. The probe gain also depends on batch composition, 24.7 against 10.1 percent.`);
    L.cite(s, "Source: end-to-end mock tests, 17 Nov 2026 (Mac, real emotion2vec+ large weights)");
  }

  // ---- Validation 3: why the old objective failed and what replaced it
  {
    const s = L.slide("Status", "Why the old target failed, and what replaced it");
    const lw = 6.55, rx = MX + lw + 0.3, rw = W - MX - rx, py = TOP + 0.05, ph = 3.95;
    s.addShape(p.ShapeType.rect, { x: MX, y: py, w: lw, h: ph, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    s.addShape(p.ShapeType.rect, { x: MX, y: py, w: lw, h: 0.08, fill: { color: C.bad }, line: { color: C.bad, width: 0 } });
    s.addShape(p.ShapeType.rect, { x: rx, y: py, w: rw, h: ph, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
    s.addShape(p.ShapeType.rect, { x: rx, y: py, w: rw, h: 0.08, fill: { color: C.good }, line: { color: C.good, width: 0 } });
    L.text(s, "Diagnosis (old target, ffn_mean)", MX + 0.2, py + 0.2, lw - 0.4, 0.42, { fontSize: 22, bold: true, color: C.darkRed });
    const steps = [
      "The last layer barely changes over time: within-clip variance is 0.2% of between-clip",
      "The 8 normalised FFN outputs are nearly uncorrelated (mean r = 0.04): not learnable",
      "So the gradients eroded the emotion features instead",
    ];
    steps.forEach((t, i) => {
      const y = py + 0.82 + i * 1.02;
      s.addText(String(i + 1), { x: MX + 0.2, y, w: 0.48, h: 0.48, shape: p.ShapeType.ellipse, fill: { color: C.bad }, line: { color: C.bad, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, MX + 0.85, y - 0.05, lw - 1.05, 0.95, { fontSize: 20 });
    });
    L.text(s, "What replaced it", rx + 0.2, py + 0.2, rw - 0.4, 0.42, { fontSize: 22, bold: true, color: C.darkGreen });
    L.bullets(s, [
      "target final: last block, layer norm",
      "gate 1: learnability probe",
      "gate 2: downstream check",
      "predictor lr 10⁻³; encoder warm-up",
      "optional: pooled utterance loss, L2-SP anchor",
    ], { x: rx + 0.2, y: py + 0.78, w: rw - 0.35, h: ph - 0.9, fontSize: 20, gap: 6 });
    L.box(s, MX, 5.82, CW, 0.9, [
      { text: "The gates protect; they do not prove a gain. ", options: { bold: true } },
      { text: "Run 2’s quick linear-probe UAR rose from 0.653 to 0.750, but the full head’s test accuracy did not." },
    ], { fill: C.blueFill, line: C.accent, fontSize: 20, align: "left", margin: [3, 14, 3, 14] });
    L.notes(s, `Why did the old objective fail, and what replaced it?
    The diagnosis on the left. First, emotion2vec+ large was fine-tuned on utterance-level labels, and its last layer barely changes over time: the within-clip variance is only 0.2 percent of the between-clip variance, and each frame has a cosine of about 0.99998 with its clip's mean embedding. Second, the old target averaged the eight time-normalised feed-forward outputs; these are nearly uncorrelated with each other, a mean correlation of 0.04, and even with the encoder frozen and no mask the predictor could not learn them: the loss stayed at the predict-zero level, about 0.15, the target variance. Third, the gradients therefore taught the student nothing and only pushed the encoder away from its emotion features.
    What replaced it, on the right. The default target is now final, the teacher's last block, layer-normalised per frame, which the student can learn. Two gates guard the encoder: the learnability probe stops a run whose target cannot be learned, and the downstream check stops a run that lowers the emotion UAR and keeps the best encoder. The predictor has its own learning rate of 10 to the minus 3 and the encoder warms up over 1,000 steps after the probe. The pooled utterance loss and the L2-SP anchor are available but off by default.
    The blue box is the important caution. In run 2, the downstream check's quick linear head rose from 0.653 to 0.750 validation UAR, but the full head trained on that encoder did not improve on the test set. The downstream gate is good at blocking harm; it is not evidence of a gain. Real call data and the final head comparison decide.`);
    L.cite(s, "Sources: docs/guide.md (SSL target, two gates); end-to-end mock tests, 17 Nov 2026");
  }

  // ---- Validation 4: padding and clip length fixed
  {
    const s = L.slide("Status", "Batch padding and clip length are both fixed");
    const gap = 0.35, cw = (CW - gap) / 2, cy = TOP + 0.05, ch = 4.95;
    const card = (x, title, lines, fix) => {
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: ch, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: 0.08, fill: { color: C.good }, line: { color: C.good, width: 0 } });
      L.text(s, title, x + 0.22, cy + 0.2, cw - 0.44, 0.45, { fontSize: 22, bold: true });
      L.bullets(s, lines, { x: x + 0.22, y: cy + 0.8, w: cw - 0.4, h: 2.75, fontSize: 20, gap: 8 });
      L.box(s, x + 0.22, cy + ch - 1.35, cw - 0.44, 1.15, fix, { fill: C.white, line: C.good, lw: 1.5, fontSize: 20, align: "left", margin: [4, 10, 4, 10] });
    };
    card(MX, "Batch padding: fixed", [
      "head features and prediction: one clip at a time",
      "SSL: duration buckets, cropped to the shortest clip",
      "no padded frame reaches the encoder",
    ], [{ text: "Done: ", options: { bold: true } }, { text: "batched vs single-clip prediction differ by 0.000 (was up to 0.29)" }]);
    card(MX + cw + gap, "Clip length: fixed", [
      "window now 2–15 s, the same as preparation_tools",
      "every clip header checked before weights load",
      "a 16.4 s clip stops the run after ≈ 5 s, before step 1",
    ], [{ text: "Done: ", options: { bold: true } }, { text: "10–12 s segmenter clips now train; bad clips are listed at once" }]);
    L.notes(s, `The two further problems the mock test found are both fixed.
    Batch padding. FunASR's convolutional positional encoder reads neighbouring frames, including padded ones, so a padded clip's embedding depended on its batch neighbours: cosine down to 0.985, and class probabilities shifted by up to 0.29 for the released head and 0.168 for our head. Now the encoder embeds one clip at a time, never padded, for head feature extraction, predict.py and predict_batch; the head itself still trains with large batches on the cached features. Batched and single-clip predictions now differ by 0.000. In self-supervised training, batches are built from clips of near-equal duration and every clip is cropped to the batch's shortest clip, so no padding reaches the encoder there either; in simulation this drops 0.4 to 0.7 percent of the audio. The settings extraction_batch_size and predict.batch_size were removed.
    Clip length. Adaptation now accepts 2 to 15 seconds, the same window as preparation_tools; the current internal batches are 2 to 10 seconds and fit. Every clip's header is checked before any weights load, in training, self-supervised training and prediction, and out-of-range or unreadable clips stop the run at once with a list. Verified both ways: an unlabelled set with 10 to 12-second segmenter clips now trains, and a 16.4-second clip stops the run after about 5 seconds, before step 1.`);
    L.cite(s, "Sources: docs/guide.md (head fine-tuning, million-clip section); mock re-tests, 17 Nov 2026");
  }

  // ---- Experiment plan
  {
    const s = L.slide("Next steps", "Plan: real call labels first; SSL only through both gates", { titleSize: 28 });
    const st = [
      ["1", "Path A/B", "A and B on real call labels: 1K, then 5K clips", C.accent],
      ["2", "B200 check", "measure workers, memory and throughput", C.ink],
      ["3", "SSL pilot", "final target on real calls; set probe_min_gain from probe.json", C.accent2],
      ["4", "Compare", "heads on encoder_best and checkpoints vs the A/B baseline", C.accent2],
      ["5", "Scale up?", "scale up only if validation improves; else consider pseudo-labels", C.good],
    ];
    const gap = 0.3, cw = (CW - 4 * gap) / 5, cy = 1.78, ch = 3.2;
    st.forEach(([n, t, b, col], i) => {
      const x = MX + i * (cw + gap);
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: ch, fill: { color: C.panel }, line: { color: C.panel, width: 0 } });
      s.addShape(p.ShapeType.rect, { x, y: cy, w: cw, h: 0.09, fill: { color: col }, line: { color: col, width: 0 } });
      s.addText(n, { x: x + 0.18, y: cy + 0.27, w: 0.52, h: 0.52, shape: p.ShapeType.ellipse, fill: { color: col }, line: { color: col, width: 0 }, fontFace: FONT, fontSize: 20, bold: true, color: C.white, align: "center", valign: "middle", margin: 0 });
      L.text(s, t, x + 0.8, cy + 0.27, cw - 0.85, 0.52, { fontSize: 20, bold: true, valign: "middle" });
      L.text(s, b, x + 0.18, cy + 1.0, cw - 0.3, 2.0, { fontSize: 18, ls: 1.08, kind: "diagram" });
      if (i < 4) L.arrow(s, x + cw + 0.02, cy + ch / 2, x + cw + gap - 0.02, cy + ch / 2, { width: 2 });
    });
    L.bullets(s, [
      { runs: [{ text: "Fixed in every stage: ", options: { bold: true } }, { text: "the customer-disjoint splits and one SSL validation manifest (about 1K–2K clips)" }] },
      { runs: [{ text: "Decision: ", options: { bold: true } }, { text: "the full head’s validation UAR and macro-F1 against the Path A/B baseline" }] },
      { runs: [{ text: "Fallback, not built: ", options: { bold: true } }, { text: "pseudo-labels from the released 9-class head" }] },
    ], { y: 5.25, h: 1.55, fontSize: 20, gap: 4 });
    L.notes(s, `The remaining open items are all about real data and the real machine; the code issues found by the mock tests are fixed.
    Step 1: Path A/B is ready. Train both head options on real call labels as they arrive, about 1,000 training clips first, then 5,000, with customer-disjoint validation and test sets, and choose between A and B on validation UAR. This is the baseline every later step must beat.
    Step 2: on the B200, run the environment check and measure what the Mac cannot tell us: the right number of DataLoader workers from data_wait_seconds, memory from the CUDA counters, and throughput with and without data waiting.
    Step 3: an SSL pilot on real call audio with the final target. The probe threshold, probe_min_gain at 10 percent, was set on mock data, where the gain varied between 10 and 25 percent with batch composition; on real data, look at probe.json before trusting a pass or a stop near the threshold, and tune it if needed.
    Step 4: train full heads on encoder_best.pt and the kept checkpoints and compare their validation UAR and macro-F1 with the Path A/B baseline. The downstream gate only protects; this comparison decides.
    Step 5: scale up step by step, 100,000 to 200,000 clips across all batches, then a million or all, only if the validation result improves. If continued pretraining does not help on real calls, a fallback is pseudo-labelling unlabelled clips with the released 9-class head; that route is not built.
    Throughout, the splits and one fixed SSL validation manifest stay the same, and the test set is opened once, after the final choice.`);
    L.cite(s, "Sources: docs/guide.md; end-to-end mock tests, 17 Nov 2026");
  }
};

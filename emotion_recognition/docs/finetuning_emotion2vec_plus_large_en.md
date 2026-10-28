# Fine-tuning emotion2vec+ large: an Estimate

English translation of [finetuning_emotion2vec_plus_large.md](finetuning_emotion2vec_plus_large.md) (Chinese); the two have the same content. The model's architecture, size and training:
[emotion2vec_plus_large_en.md](emotion2vec_plus_large_en.md).

Date: 2026-10-28. Questions: how much data and how many GPUs fine-tuning needs; whether to train only the last fully connected layer or all layers; how many samples training all layers needs.

Method: trainable parameters are computed from the measured architecture; compute comes from PyTorch's FLOP counter and timings on the M4 Max (section 3); the advice on data size comes from
published studies, each cited. Numbers marked "estimate" are derived from the stated assumptions, not measured. No public fine-tuning learning curve for emotion2vec or emotion2vec+ was found, so
the data ranges in section 2 are inferred.

## 0. Conclusions

1. **Compute is not the bottleneck.** With up to 100k labelled clips, each of the options A–D in section 1 needs only one GPU. Fine-tuning all layers on 10k clips of 5 s for 10 epochs takes about
   0.2–0.4 A100-hours; 100k clips about 2–4 A100-hours (estimates, including data loading and validation). This project's M4 Max can do it too (about 1.4 h for 10k clips). What really costs compute
   is repetition: 5 speaker folds × 3 learning rates × 3 random seeds is already 45 runs.
2. **Start by training only the last layer** (or "a weighted sum of all layers + a small head") with the encoder frozen. emotion2vec+ large was already trained on about 40k hours of emotion data,
   so its frozen features were trained for emotion classification; the embeddings are extracted once and the head trains in minutes on a CPU; nothing the model already does is lost. A few hundred
   to a few thousand labelled clips are enough.
3. **Fine-tuning all layers** is worth it only when three conditions hold: about 10k or more labelled clips from many speakers; a target domain clearly different from the training data (language,
   recording channel, spontaneous conversation); and frozen approaches have stopped improving on a speaker-independent validation set. In between lie LoRA and unfreezing only the top blocks.
4. **How many samples to train all layers**: at least about 10k clips (4–6 s each, about 11–17 h); 20k–100k clips (about 22–170 h) to beat frozen approaches reliably; below about 5k clips,
   fine-tuning all layers rarely beats a frozen encoder and easily overfits to the training speakers. This is inferred from the studies in section 2.
5. **Training from scratch is not realistic**: it needs self-supervised pre-training on tens of thousands of hours (data2vec 2.0 Large: 64 A100s × 76.7 h), then training on about 40k hours of
   pseudo-labelled data (about 25–38 million clips), and the pseudo-labelling pipeline is not published.

## 1. Five options compared

| Option | What is trained | Trainable parameters | Labelled data needed | GPU memory | GPU time (A100, estimate) | Risks | When to use |
|---|---|---|---|---|---|---|---|
| **A. Only the last linear layer** | Encoder frozen; extract the 1024-dimensional utterance embeddings once, then train Linear(1024→K) or logistic regression | 1,025 × K (9,225 for K = 9) | a few hundred clips (≥ 50 per class) up to about 5k; 1k clips ≈ 1.1–1.7 h | inference only: ≤ 1.2 GB up to 16 s (measured, including 0.66 GB of weights) | 10k clips: extraction < 1 minute; the head trains in minutes on a CPU | very low; low-level features cannot adapt to a new language or channel; the last layer was trained for 9 classes, so new classes may not separate | first step and baseline; < 5k clips; a new label set |
| **B. Weighted sum of all layers + small head** | Encoder frozen; learned weighted sum of the 12 block outputs → small MLP (e.g. 1024→256→K) | about 0.26 M | 1k–20k clips (1.1–33 h) | as A. Caching each layer's utterance mean needs only 12 × 1024 numbers per clip; attention pooling over frames needs cached frame features (about 6 MB per 5 s clip in fp16) | as A; without caching, one forward pass per epoch: about 0.1 h for 10k clips × 10 epochs | low | when A is not enough. PEFT-SER and SAILER use this kind of head |
| **C. LoRA/adapters, or unfreezing only the top blocks** | LoRA (rank 8) on the qkv and output projections of the 12 blocks; or train only the top 2–4 blocks + head | LoRA: 0.59 M (qkv + projection), 1.57 M (plus the FFN); top 2 blocks: 25.2 M, top 4: 50.4 M | 5k–50k clips (5.6–83 h) | LoRA: gradients pass through all 12 blocks, so activations are close to full fine-tuning, but frozen weights have no gradients or Adam state; top blocks only: only their activations are stored | LoRA about 0.5 × D, top 2 blocks about 0.35 × D (ratios measured in section 3.2) | medium; less forgetting than D | medium data size, some domain difference |
| **D. All layers** | all parameters; freeze at least the CNN | 162.1 M (CNN frozen: 157.9 M; whole front end frozen: 151.2 M) | at least about 10k clips (11–17 h); 20k–100k (22–170 h) for steadier gains | fixed about 2.6 GB (fp32 weights + gradients + Adam state); activations in fp32 about 0.3–0.6 GB per 4 s clip, about 2.4–3 GB per 16 s clip, roughly half in bf16 (estimate); a 24 GB card fits 16–32 clips × 5 s | 10k clips × 10 epochs: 0.2–0.4 h; 100k: 2–4 h | overfitting to the training speakers; forgetting (other languages, classes missing from the data such as other or surprised); shifted probability calibration | much data, large domain difference, and A–C have plateaued |
| **E. Continued self-supervised pre-training (not fine-tuning)** | continue data2vec 2.0-style pre-training on unlabelled audio of the target domain; then A–D are still needed for classification | the whole encoder (162.1 M) + a newly built decoder (1.9 M), plus an EMA teacher copy (memory only) | no labels; from the task audio itself (TAPT, a few hours) to hundreds of hours (emotion2vec used 262 h) | emotion2vec used 24 GB A10 GPUs with at most 62.5 s of audio per GPU per step | base model, 262 h × 100 epochs: 4 A10s × about 62 h ≈ 250 A10-hours (published figures); large on 1,000 h × 100 epochs about 600 A100-hours, on 10–20 h × 100 epochs about 5–10 A100-hours (estimates) | no public emotion2vec pre-training code; `model.pt` has no teacher and no decoder; the representation changes, so the existing head no longer fits and must be retrained | much unlabelled audio, few labels, large domain difference: a step before A–D |

Assumptions in the table:
- Clips of 5 s; 4–6 s per clip when converting to hours.
- Compute (measured, section 3): forward 21.9 GFLOP per second of audio; a full fine-tuning step costs 3.42 forward passes, i.e. about 75 GFLOP per second of audio; 2.98 with the CNN frozen.
- A100: assumed to sustain 50–100 TFLOPS in bf16 (16–32% of its 312 TFLOPS dense peak; an assumption, not measured), then × 2 for data loading, padding and validation. The H100's dense bf16 peak is
  989 TFLOPS; counted as 2–3 times faster.
- E is anchored on emotion2vec's published compute: multiplied by the ratio of large to base forward compute, 1.5 (87.6 vs 58.3 GFLOP for a 4 s clip), divided by the ratio of A100 to A10 fp16 peaks,
  2.5 (312 vs 125 TFLOPS), and scaled linearly with hours.
- **Training from scratch** is left out because it is not realistic: data2vec 2.0 Large's self-supervised pre-training on 60k hours of audio used 64 A100s × 76.7 h (about 4,900 A100-hours);
  emotion2vec+ large then used 42,526 h of pseudo-labelled data (about 25–38 million clips at 4–6 s each), produced by a pipeline that is not published; and all public labelled emotional speech
  together is only a few hundred hours (the 32 datasets of EmoBox total 294.4 h).

## 2. Which option for which data size

| Labelled clips (hours at 4–6 s) | Suitable options | Evidence |
|---|---|---|
| < 1k (< 1.7 h) | A; do not fine-tune all layers | "Frozen encoder + small downstream model" is the standard evaluation of the emotion2vec paper and of EmoBox, and it trains even on datasets of a few hundred clips (SAVEE 480, RAVDESS 1,440). On Chen & Rudnicky's learning curve, fine-tuning wav2vec 2.0 directly (V-FT) reaches only 56.8% UA with about 0.5 h |
| 1k–5k (1.1–8 h) | A or B; C can be tried | Chen & Rudnicky: V-FT gains 9.6 points of UA from 0.5 h to 3 h (56.8 → 66.4), then only 0.9 more up to about 7 h. Wang et al.: on IEMOCAP (12 h), freezing the CNN and fine-tuning only the Transformer matches fine-tuning all layers (and beats it in every speaker-dependent setting); the authors attribute this to overfitting when many parameters are trained on little data |
| 5k–20k (5.6–33 h) | B or C; D can be tried, but compare it with B and C | PEFT-SER: on 4 datasets (4.8k–37.9k clips), LoRA beats "frozen + downstream model" by 1–2 points of average UAR (WavLM Base+ 65.28 → 67.28; wav2vec 2.0 base 62.27 → 63.30), with no gain for Whisper. Wang et al.: for general SSL models, fine-tuning all layers beats freezing by 4.7–7.3 points of WA (IEMOCAP, speaker-independent); but the self-supervised-only emotion2vec reaches 71.8% WA frozen, on par with fully fine-tuned wav2vec 2.0 base (70.8%; the two papers use different splits) |
| 20k–100k (22–170 h) | D (CNN frozen, low learning rate) or C | Wagner et al.: on MSP-Podcast's 62 h training set, fine-tuning the Transformer layers beats training only the output layer by 0.03–0.24 CCC (most for valence); arousal and dominance saturate with 25% of the training data, valence still improves at 75%. AFTER: with TAPT before fine-tuning, going from 10% (about 1.8k clips) to 80% of the labels (about 14k clips, computed from its splits) raises UA from 70.2% to 78.9%. SAILER: on 84k clips of natural speech, a fully fine-tuned WavLM Large (validation macro-F1 0.376) does not beat a frozen Whisper Large-v3 encoder (0.383) |
| > 100k, or thousands of hours of pseudo-labels | D | emotion2vec+ itself was trained with all layers ([issue #64](https://github.com/ddlBoJack/emotion2vec/issues/64)); the last stage of large saw about 176k clips per epoch for 30 epochs, at a learning rate of at most 4.1e-5 (model document, section 4.3) |

**Why the bar is higher for emotion2vec+.** Most of the gains from fine-tuning all layers in the table come from general encoders that never saw emotion labels (wav2vec 2.0, HuBERT, WavLM): for
them, fine-tuning teaches the task. emotion2vec+ large has already learned these 9 classes from about 40k hours, so its frozen features are emotion features (even the self-supervised-only emotion2vec,
frozen, matches fine-tuned wav2vec 2.0 base). For the same classes, then, fine-tuning all layers mainly adapts the model to a new domain (language, channel, speaking style), and that needs enough data
from the domain. For a new label set, the author recommends emotion2vec (base) rather than emotion2vec+ for feature extraction ([issue #39](https://github.com/ddlBoJack/emotion2vec/issues/39)), which is
worth comparing in A and B.

### How many samples to train all layers

- **At least about 10k clips** (4–6 s each, about 11–17 h), from as many speakers as possible (hundreds rather than tens), with at least a few hundred clips per class.
- **20k–100k clips** (about 22–170 h): fine-tuning all layers can be expected to beat frozen approaches reliably.
- **Below about 5k clips**: the gain is uncertain and overfitting likely; use A, B or C.
- This range is inferred from the table above: learning curves flatten after a few hours or a few thousand clips; LoRA adds only 1–2 points in PEFT-SER; Wagner et al. see clear gains with 62 h;
  in SAILER, with 84k clips a frozen large model matches a fully fine-tuned one; and the scale of the emotion2vec+ authors' own last stage.
- The most reliable answer comes from a learning curve on your own data: train A and D on 10%, 25%, 50% and 100% of the training set and see where the two curves cross and where they flatten.

## 3. Local measurements and conversion to data-centre GPUs

Setup: M4 Max (12 PyTorch CPU threads, 40-core GPU, 48 GB), FunASR 1.4.16, PyTorch 2.5.1, fp32, random waveforms; times are the median of 5 runs after warm-up.
CPU memory = peak physical footprint of the process (sampled every 2 ms) minus the value after loading the model; MPS memory = PyTorch's driver-allocated memory after the first run (includes the
allocator's cache, so it errs high).

### 3.1 Inference (forward pass, batch 1)

| Clip length | 2 s | 4 s | 8 s | 16 s |
|---|---|---|---|---|
| Frames | 99 | 199 | 399 | 799 |
| Forward compute | 44.6 GFLOP | 87.6 GFLOP | 176.4 GFLOP | 365.8 GFLOP |
| CPU time | 51 ms | 88 ms | 183 ms | 379 ms |
| CPU peak extra memory | 0.03 GB | 0.04 GB | 0.13 GB | 0.29 GB |
| MPS time (same length repeated) | 8 ms | 14 ms | 25 ms | 50 ms |
| MPS driver memory (including 0.66 GB of weights) | 1.18 GB | 1.19 GB | 1.17 GB | 1.17 GB |

- About 22 GFLOP per second of audio, almost proportional to length up to 16 s; beyond that the N² term of attention shows: 24.5 at 30 s and 28.1 GFLOP per second of audio at 60 s.
- Longer clips (CPU): 30 s takes 0.86 s and 0.56 GB extra; 60 s 1.9 s and 2.3 GB; 120 s 6.7 s and 9.3 GB. Each block's attention scores are 16 × N × N fp32 values: 2.8 MB at 4 s, 42 MB at 16 s,
  0.58 GB at 60 s, 2.3 GB at 120 s. At inference a few such tensors exist at once; in training all 12 blocks keep theirs until the backward pass, so memory for long clips grows fast.
- 8 clips of 4 s in one batch: 86 ms on MPS (11 ms per clip), 630 ms on the CPU. MPS cannot process clips longer than 20.48 s (model document, section 5).

### 3.2 One training step (fp32, AdamW)

| Configuration | MPS per step | MPS driver memory | CPU per step |
|---|---|---|---|
| D: 4 s × 1 | 74 ms | 3.5 GB | 0.43 s |
| D: 16 s × 1 | 224 ms | 5.7 GB | 1.64 s |
| D: 4 s × 8 | 328 ms | 5.9 GB | 2.39 s |
| D: 8 s × 8 | 681 ms | 12.8 GB | — |
| D: 4 s × 16 | 631 ms | 11.7 GB | — |
| D with the CNN frozen: 4 s × 8 | 247 ms | 4.7 GB | 1.86 s |
| C, LoRA approximation: 4 s × 8 | 161 ms | 4.4 GB | — |
| C, top 2 blocks only: 4 s × 8 | 114 ms | 2.3 GB | — |

- In full fine-tuning about 2.6 GB is fixed: weights 0.65 + gradients 0.65 + Adam state 1.30 GB. From this, activations in fp32 are about 0.3–0.6 GB per 4 s clip, about 1.2 GB per 8 s clip and
  about 2.4–3 GB per 16 s clip.
- The "LoRA approximation" trains only the qkv bias of the lowest block and the head, so gradients pass through all 12 blocks: it mimics LoRA's memory and compute (without the small compute of the
  LoRA matrices themselves).
- A training step costs 3.42 forward passes in FLOPs (2.98 with the CNN frozen), so "training ≈ 3 × forward" roughly holds for compute; the measured time ratio (forward + backward over an inference
  forward pass) is 3.5–4.2 on this machine.
- MPS training throughput: about 100 s of audio per second at 4 s × 16 (about 7.6 TFLOPS). At that speed, fine-tuning all layers on this machine takes about 1.4 h for 10k clips of 5 s × 10 epochs and
  about 14 h for 100k clips (compute only).
- PyTorch 2.5.1 on MPS fails in the backward pass of this model's Conv1d layers (`view size is not compatible with input tensor's size and stride`). For the measurements, a small autograd function
  that makes the gradient entering each Conv1d contiguous works around it; freezing the whole front end (CNN + projection + positional convolution) also avoids it. The CPU is not affected; CUDA was
  not tested.

### 3.3 Conversion to an A100 (estimate)

Assumptions as in section 1 (50–100 TFLOPS sustained, then × 2 for overheads):

| Labelled clips (5 s each) | A/B: extract embeddings once | D: all layers, 10 epochs |
|---|---|---|
| 1k (1.4 h of audio) | seconds | 1–3 minutes |
| 10k (14 h) | < 1 minute | 0.2–0.4 h |
| 100k (139 h) | 4–7 minutes | 2–4 h |
| 1M (1,389 h), D for only 3 epochs | 0.6–1.2 h | 6–13 h (about 1–2 h on 8 GPUs) |

- One GPU is enough throughout; an H100 is about 2–3 times faster. C costs about 0.35–0.5 × D.
- The real cost is repetition: for example 45 runs (5 folds × 3 learning rates × 3 seeds) at 100k clips is about 100–200 A100-hours.
- Against published figures: SAILER, with a frozen Whisper Large-v3, trained on 84k clips for 15 epochs with about 15 GPU-hours and 12 GB of GPU memory; AFTER's TAPT plus active-learning fine-tuning
  on about 20k clips took 1,508 minutes with all labels and 316 minutes with 20% of them. Neither states the GPU type, but the order of magnitude agrees with the estimates above.

## 4. How to fine-tune (all layers as the example; C is similar)

1. **Baselines first**: zero-shot (map your labels to the 9 classes, as this project's evaluation does) → A → B → C → D. Compare each step on the same speaker-independent validation set and stop when
   the gains stop.
2. **Splits**: training, validation and test split by speaker, each speaker in one set only; the test set is used once at the end, ideally from different recording sessions or a different corpus.
   Report UA (mean per-class recall) and macro-F1 with confusion matrices; use 3 random seeds or speaker k-fold and report mean and spread.
3. **Pre-processing**: 16 kHz mono; `layer_norm` over each whole clip first (as at inference, before padding); crop clips to at most 20 s (as the authors did; attention memory grows with the
   square of the length) and drop clips shorter than 1 s; batch clips of similar length and pass a padding mask; mean-pool over valid frames only. Padding makes the output differ slightly from
   single-clip inference (the convolutional positional encoding sees padded frames), so keep it small.
4. **Learning rates**: AdamW; encoder 1e-5–3e-5 (Wang et al. used 1e-5; the authors' last stage, with large batches, peaked at 4.1e-5); a new head 1e-4–1e-3; warm-up over the first 5–10% of steps,
   then cosine or linear decay; weight decay 0–0.01; gradient clipping at 1.0 (a common setting); bf16 mixed precision.
5. **Freezing and layer-wise learning rates**: freeze at least the CNN feature extractor (Wang et al. found that this matches fine-tuning all layers, and Wagner et al. freeze the CNN too; it saves about
   25% of the step time in the measurements above). Layer-wise learning-rate decay can be added (multiply by 0.8–0.9 per block going down; a common setting) so that the lower layers change less.
   Training only the head for 1–2 epochs before unfreezing is another option (fairseq's `freeze_finetune_updates` does this).
6. **Class imbalance**: weight the cross-entropy by inverse class frequency, or use balanced sampling. With several annotators, the label distribution can serve as a soft target (SAILER uses a KL loss).
7. **Regularisation and early stopping**: keep the configuration's dropout of 0.1; time masking and mixup can be added (the authors' last stage used both); few epochs (5–20); stop early on validation UA
   or macro-F1 (after 3–5 epochs without improvement) and keep the best checkpoint.
8. **Forgetting**: classes missing from your data (such as other or surprised) and other languages may get worse. Mixing in some public labelled data (for example EmoBox datasets) helps; run this
   project's evaluation before and after fine-tuning, especially on MCAE-SPPS, nEMO and MESS, which are not in EmoBox, to check that nothing regressed.
9. **Hardware**: one GPU with 24 GB or more is enough; small experiments run on the M4 Max (MPS needs the workaround in section 3.2).

## 5. Official fine-tuning code

There is **no** official code for fine-tuning the emotion2vec+ encoder (checked on 2026-10-28):
- The emotion2vec GitHub repository has only feature extraction and one "frozen features + downstream model" example (`iemocap_downstream`): Linear 768→256 + ReLU → mean over valid frames →
  Linear 256→4; RMSprop (learning rate 5e-4, momentum 0.9) + CyclicLR (up to 1e-3), batch 128, 100 epochs, model chosen by validation WA, IEMOCAP leave-one-session-out 5-fold. That is options A
  and B above; the encoder is not trained.
- FunASR has only an inference example (`examples/industrial_data_pretraining/emotion2vec/demo.py`). `Emotion2vec.forward()` has no loss: with `features_only=True` it returns features, otherwise
  nothing, so FunASR's trainer cannot fine-tune it as it is (true of FunASR 1.4.16 and of the main branch on 2026-10-28).
- The author's statements disagree: in January 2024, no training code is provided and the inference code can be modified for fine-tuning ([#10](https://github.com/ddlBoJack/emotion2vec/issues/10));
  in June 2024, the model can be fine-tuned with FunASR's AutoModel pipeline ([#28](https://github.com/ddlBoJack/emotion2vec/issues/28)), which the code above does not support; in August 2024,
  fine-tuning support is planned with no date ([#39](https://github.com/ddlBoJack/emotion2vec/issues/39)); an April 2025 request for the upstream training code
  ([#70](https://github.com/ddlBoJack/emotion2vec/issues/70)) was closed without a reply.
- The released checkpoint was trained with an unpublished fairseq fork (`user_dir: examples/emo2vec` in its training configuration). The closest public code is fairseq's
  [`examples/data2vec/models/audio_classification.py`](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/audio_classification.py): mean pooling + linear layer, mixup,
  `freeze_finetune_updates`.

So a small PyTorch training loop is needed, around the module FunASR builds (`net = AutoModel(...).model`, then `net.train()`). The core lines:

```python
# wav: (B, L), each clip layer-normed, then zero-padded; pad: (B, L) bool, True where padded
out = net.extract_features(wav, padding_mask=pad)       # no masking, extra tokens already removed
x, frame_pad = out["x"], out["padding_mask"]             # (B, T, 1024), (B, T); padding_mask is None without padding
keep = (~frame_pad).unsqueeze(-1).float()
emb = (x * keep).sum(1) / keep.sum(1)                     # mean over valid frames only
loss = torch.nn.functional.cross_entropy(net.proj(emb), labels)  # for a new label set, use a new Linear(1024, K)
```

The padding mask of `extract_features` was checked on this machine: a 3 s clip zero-padded to 4 s gives an utterance embedding with cosine similarity 0.9996 to the one from the clip alone.

## 6. Main sources

- Architecture, parameters and local measurements: [emotion2vec_plus_large_en.md](emotion2vec_plus_large_en.md); FunASR code: https://github.com/modelscope/FunASR/tree/main/funasr/models/emotion2vec
- emotion2vec paper (frozen encoder + linear layers evaluation, pre-training compute): https://arxiv.org/abs/2312.15185 ; official GitHub and downstream example: https://github.com/ddlBoJack/emotion2vec , https://github.com/ddlBoJack/emotion2vec/tree/main/iemocap_downstream ; issues #10, #28, #39, #64, #70
- EmoBox (protocol: frozen last-layer features + linear layer, ReLU, pooling and classification head; 32 datasets, 294.4 h): https://arxiv.org/abs/2406.07162
- Chen & Rudnicky, Exploring wav2vec 2.0 fine-tuning for improved speech emotion recognition (V-FT, TAPT, P-TAPT, learning curve): https://arxiv.org/abs/2110.06309
- Wang et al., A Fine-tuned Wav2vec 2.0/HuBERT Benchmark (frozen, Transformer-only and full fine-tuning): https://arxiv.org/abs/2111.02735
- Feng & Narayanan, PEFT-SER (LoRA, adapters): https://arxiv.org/abs/2306.05350
- Wagner et al., Dawn of the Transformer Era in Speech Emotion Recognition (IEEE TPAMI 2023; fine-tuned vs frozen, less training data): https://arxiv.org/abs/2203.07378
- Li et al., AFTER: Active Learning Based Fine-Tuning Framework for Speech Emotion Recognition: https://arxiv.org/abs/2310.00283
- Feng et al., the SAILER system (Interspeech 2025, MSP-Podcast 1.12): https://www.isca-archive.org/interspeech_2025/feng25_interspeech.pdf
- data2vec 2.0 (pre-training compute of the Large model): https://arxiv.org/abs/2212.07525
- fairseq's data2vec classification model: https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/audio_classification.py
- GPU peaks: A100 https://www.nvidia.com/en-us/data-center/a100/ ; H100 https://www.nvidia.com/en-us/data-center/h100/ (the listed 1,979 TFLOPS is with sparsity; dense is half); A10 https://www.nvidia.com/en-us/data-center/products/a10-gpu/

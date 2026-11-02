# Facts for the SER evaluation presentation

Collected 2026-11-02. Read-only: nothing in the public repository was changed. Paths: `ER` =
`emotion_recognition/emotion_recognition/` in the public repo. Parameter counts come from the safetensors headers (tensor
shapes, no weights loaded), via a scratch script. FLOPs: `torch.utils.flop_counter.FlopCounterMode` on meta-device
modules built from the local configs (counts only matmul and conv, 2 FLOPs per MAC; elementwise ops left out).

---

## 0. Headline numbers (slide-ready)

| Model | Params in use | Backbone (frozen?) | Trained in the original release | Input window | Forward cost per clip ≤ window |
|---|---|---|---|---|---|
| emotion2vec+ large | 162.1M | data2vec-style CNN + 12 blocks (all trained) | everything | real length | 87.6 GFLOP for 4 s (`ER/docs/finetuning_emotion2vec_plus_large_en.md` §3.1) |
| MERaLiON-SER-v1 | **313.4M** (307.2M encoder + 6.0M downstream + 0.13M heads) | Whisper-medium encoder, frozen + LoRA r=16 on q/k/v (merged) | LoRA + downstream + heads | **fixed 30 s** | **1.15 TFLOP** (13× emotion2vec+ at 4 s) |
| Melati | **637.3M** (637.0M encoder + 0.33M head) | Whisper-large-v3 encoder, frozen | head only (332,328 params) | **fixed 30 s** | **2.27 TFLOP** (26×) |
| Vox-Profile Whisper | **639.9M** (636.0M encoder + 3.07M LoRA + 0.80M heads) | Whisper-large-v3 encoder (750 positions), frozen + LoRA r=16 on fc1/fc2 of layers 17–31, **LoRA B = 0, so no effect** | heads (+ the ineffective LoRA) | **fixed 15 s** | **1.05 TFLOP** (12×) |
| Vox-Profile WavLM | **316.2M** (315.5M backbone + 0.73M heads) | WavLM-large, **fully fine-tuned** (config `finetune_method: finetune`) | everything | real length (≤ 15 s) | 0.147 TFLOP for 4 s (1.7×) |

Survey counts: **37 datasets surveyed → 9 downloaded → 6 evaluated**; **42 model entries surveyed → 5 open models + 2
audio LLMs (GPT Realtime, Gemini) chosen** (section 4).

LISTEN (arXiv 2510.10444): on sentences with neutral words, six audio LLMs reach only **16.5–34.9 % accuracy from audio
alone** (uniform guess 12.5 %), while text alone gives "neutral" correctly 81–97 %; they default to *neutral* (section 3a).

---

## 1. Module-by-module breakdowns (for block diagrams)

Shapes are for one **4 s, 16 kHz clip** (64,000 samples), batch 1. Whisper front ends pad every clip to the window
(30 s → 3,000 mel frames → 1,500 encoder frames; Vox-Profile Whisper 15 s → 1,500 → 750). Only 200 of the 1,500 frames
(13 %) carry the clip.

### 1.1 MERaLiON-SER-v1 (A*STAR I²R)

Sources: `ER/models/meralion_ser_v1/modeling_ser_whisper_ecapa.py`, `config.json`, `exporter_lora_whisper.py`,
`whisper-medium/config.json`, header of `model.safetensors`; paper https://arxiv.org/abs/2511.04914 (§2, §3.2).

| # | Module | Configuration | Output (4 s clip) | Params |
|---|---|---|---|---|
| 0 | Whisper feature extractor | 80 log-mel bins, n_fft 400, hop 160, pad/cut to 30 s | (1, 80, 3000) | 0 |
| 1 | conv1 | Conv1d 80→1024, k3, s1, GELU | (1, 1024, 3000) | 246,784 |
| 2 | conv2 | Conv1d 1024→1024, k3, **s2**, GELU | (1, 1024, 1500) | 3,146,752 |
| 3 | positional embedding | sinusoidal 1500×1024, fixed (`requires_grad_(False)` in transformers) | (1, 1500, 1024) | 1,536,000 |
| 4 | 24 Transformer blocks | pre-LN, d 1024, 16 heads, FFN 4096, GELU; per block 12,595,200 (attn 4,197,376 + FFN 8,393,728 + LN 4,096); LoRA merged into q/k/v | (1, 1500, 1024) | 302,284,800 |
| 5 | final LayerNorm | | (1, 1500, 1024) | 2,048 |
| | **Encoder total** | | | **307,216,384 (307.2M)** |
| 6 | proj_in | Linear 1024→512 + LayerNorm + GELU + Dropout 0.2 | (1, 1500, 512) → transposed (1, 512, 1500) | 525,824 |
| 7 | 3 ECAPA blocks | each: Conv1x1 → GroupNorm → ReLU → Res2 (scale 4, three 128-ch k3 convs, dilation 1 / 2 / 3) → GroupNorm → SE (512→64→512) → Conv1x1 → GroupNorm → residual; BatchNorm replaced by GroupNorm (paper); 741,760 each | (1, 512, 1500) each | 2,225,280 |
| 8 | MFA (multi-layer feature aggregation) | concat input + 3 block outputs (2048 ch) → Conv1x1 2048→512 + GroupNorm + GELU | (1, 512, 1500) → (1, 1500, 512) | 1,050,112 |
| 9a | Hybrid pooling: multiscale branch | 4 grouped convs (groups 64) with kernels 1, 2, 6, 16 → attention pooling per scale → input-dependent softmax weights over the 4 scales (MLP 512→256→4) → Linear 512→512 | (1, 512) | 501,512 |
| 9b | Hybrid pooling: hierarchical branch | attention pooling at 3 resolutions (avg-pool ×2: 1500 → 750 → 375 frames) → concat 1536 → Linear 1536→512 | (1, 512) | 788,483 |
| 9c | pooling combiner | concat 1024 → Linear 1024→512 | (1, 512) | 524,800 |
| 10 | embedding MLP | Linear 512→512, LN, GELU, Dropout 0.3, Linear 512→256, LN, GELU, Dropout 0.2 | (1, 256) shared embedding | 395,520 |
| | **Downstream total (6–10)** | | | **6,011,531 (6.0M)** |
| 11 | categorical head | Linear 256→256 → RMSNorm → GELU → Linear 256→7 (softmax) | (1, 7) | 67,847 |
| 12 | dimensional head | Linear 256→256 → RMSNorm → GELU → Linear 256→3 → Sigmoid (valence, arousal, dominance in 0–1) | (1, 3) | 66,819 |
| | **Total in use** | encoder + downstream + heads | | **313,362,581 (313.4M)** ✔ survey |
| – | Whisper decoder (stored, never run) | 24 layers | – | 456,641,536 |
| – | File total | `model.safetensors`, 3.08 GB fp32 | | 770,004,117 |

Notes
- The attention mask is passed in but never used: both pooling branches pool over all 1,500 frames, padding included
  (official behaviour; `ER/scripts/models/meralion_ser.py` docstring).
- Card and paper say 309M; measured 313.4M (311.8M without the fixed positional table). The 309M figure cannot be
  reproduced from the file.
- Original training (paper §2, §3.2; card): Whisper encoder **frozen**, **LoRA on q/k/v** of the attention layers
  (exporter: r = 16, lora_alpha = 16, rsLoRA, targets `q_proj`, `k_proj`, `v_proj`; the LoRA count would be
  24 × 3 × 16 × 2048 = **2,359,296**), downstream + heads **trained**, decoder frozen and unused. Two LR groups:
  backbone+LoRA 5e-5 (wd 4e-5), downstream 6e-4 (wd 8e-5); cosine with 0.08 warm-up; label smoothing 0.1; 15 epochs,
  batch 32, 1 node × 8 H100; early stop on dev categorical loss. Loss: weighted CE + CCC. Augmentation: MixUp (p 0.5,
  α 0.3), MUSAN noise, speed 0.9 / 1.1.
- Trained in PyTorch Lightning (the exporters convert a Lightning `.ckpt`), then LoRA merged into the encoder weights.

### 1.2 Melati emotion classifier (Scicom, Malaysia)

Sources: `ER/models/melati_emotion/modeling_melati_head.py`, `config.json`, `README.md`, header of `model.safetensors`.

| # | Module | Configuration | Output (4 s clip) | Params |
|---|---|---|---|---|
| 0 | Whisper feature extractor | **128** mel bins, pad/cut to 30 s | (1, 128, 3000) | 0 |
| 1 | conv1 | Conv1d 128→1280, k3, s1, GELU | (1, 1280, 3000) | 492,800 |
| 2 | conv2 | Conv1d 1280→1280, k3, s2, GELU | (1, 1280, 1500) | 4,916,480 |
| 3 | positional embedding | sinusoidal 1500×1280 | (1, 1500, 1280) | 1,920,000 |
| 4 | 32 Transformer blocks | d 1280, 20 heads, FFN 5120; 19,676,160 per block; fp16 in the file | 33 hidden states (embeddings + 32 layers) of (1, 1500, 1280) | 629,637,120 |
| 5 | final LayerNorm | | | 2,560 |
| | **Encoder total (frozen, = openai/whisper-large-v3)** | | | **636,968,960** |
| 6 | per-layer mean pooling | mean over the clip's own frames only (`num_frames` = 50 / s → 200), rounded to fp16 | (1, 33, 1280) | 0 |
| 7 | layer mix | softmax over 33 learned weights, weighted sum | (1, 1280) | 33 |
| 8 | LayerNorm | 1280 | (1, 1280) | 2,560 |
| 9 | hidden layer | Linear 1280→256 + GELU | (1, 256) | 327,936 |
| 10 | output | Linear 256→7 | (1, 7) | 1,799 |
| | **Head total (the only trained part)** | | | **332,328** |
| | **Total** | | | **637,301,288 (637.3M)** ✔ survey |

Original training (card): encoder frozen, "a small head trained on its frozen features", 9 acted corpora (28,895 clips),
corpora weighted equally, classes equally within each corpus.

### 1.3 Vox-Profile Whisper categorical (USC SAIL et al.)

Sources: `ER/models/vox_profile_whisper/config.json`, `ER/scripts/models/vox_profile.py` (re-implements the official
`whisper_emotion.py`), header of `model.safetensors`; https://arxiv.org/abs/2505.14648.

| # | Module | Configuration | Output (4 s clip) | Params |
|---|---|---|---|---|
| 0 | Whisper feature extractor | 128 mel bins, pad/cut to **15 s** | (1, 128, 1500) | 0 |
| 1–2 | conv1, conv2 | as Melati | (1, 1280, 750) | 5,409,280 |
| 3 | positional embedding | sinusoidal, **750** positions | (1, 750, 1280) | 960,000 |
| 4 | 32 Transformer blocks | as Melati; base weights = openai/whisper-large-v3 | (1, 750, 1280) | 629,637,120 |
| 4b | LoRA | rank 16 on fc1 and fc2 of layers **17–31** (15 layers, 30 adapters); loralib scale 1/r | – | 3,072,000 |
| 5 | final LayerNorm | | (1, 750, 1280) | 2,560 |
| 6 | 1×1 conv stack (`model_seq`) | Conv1d 1280→256, ReLU, Dropout 0.1, Conv1d 256→256, ReLU, Dropout, Conv1d 256→256 | (1, 750, 256) | 459,520 |
| 7 | mean pooling | over the clip's own frames ((64000//160 − 1)//2 + 1 = 200) | (1, 256) | 0 |
| 8 | categorical MLP | Linear 256→256, ReLU, Linear 256→9 | (1, 9) | 68,105 |
| – | other heads (in file, not used by us) | 17 detailed emotions; arousal, valence, dominance (256→256→1 each) | – | 268,308 |
| – | layer weights (33, unused on the Whisper path) | | – | 33 |
| | **Total without the decoder** | | | **639,876,926 (639.9M)** ✔ survey |
| – | Whisper decoder in the release (unused) | | | 906,521,600 (file total 1,546,398,526) |

Notes: all 30 `lora_B` tensors are exactly 0 (checked: max |B| = 0.0), so the adapters change nothing and the encoder
is exactly openai/whisper-large-v3. Categorical path after merging: 636.5M. Trained in the release: 1×1 convs + heads
(and LoRA, which ended at B = 0). Paper (App. C): emotion systems trained 15 epochs at LR 5e-4; batch 16 with LoRA, 32
without.

### 1.4 Vox-Profile WavLM categorical

Sources: `ER/models/vox_profile_wavlm/config.json`, `wavlm-large/config.json`, header of `model.safetensors`.

| # | Module | Configuration | Output (4 s clip) | Params |
|---|---|---|---|---|
| 0 | input | waveform normalised to zero mean, unit variance, not padded | (1, 64000) | 0 |
| 1 | CNN feature extractor | 7 conv layers, 512 ch, kernels 10,3,3,3,3,2,2, strides 5,2,2,2,2,2,2 (×320 → 50 frames / s), LayerNorm | (1, 512, 199) | 4,206,592 |
| 2 | feature projection | LayerNorm + Linear 512→1024 | (1, 199, 1024) | 526,336 |
| 3 | convolutional positional embedding | Conv1d k128, 16 groups, weight-norm | (1, 199, 1024) | 8,389,760 |
| 4 | 24 Transformer blocks | d 1024, 16 heads, FFN 4096, gated relative position bias (320 buckets, in block 0: +5,120); 12,596,760 per block | 25 hidden states (1, 199, 1024) | 302,327,360 |
| 5 | encoder LayerNorm + masked_spec_embed | | | 3,072 |
| | **Backbone total (fine-tuned)** | | | **315,453,120** |
| 6 | layer mix | softmax over 25 weights | (1, 199, 1024) | 25 |
| 7 | 1×1 conv stack | Conv1d 1024→256, 256→256, 256→256 | (1, 199, 256) | 393,984 |
| 8 | mean pooling | over all frames | (1, 256) | 0 |
| 9 | categorical MLP | Linear 256→256, ReLU, Linear 256→9 | (1, 9) | 68,105 |
| – | other heads (unused) | detailed 17 + A/V/D | – | 268,308 |
| | **Total** | | | **316,183,542 (316.2M)** ✔ survey |

Training: config `finetune_method: "finetune"` (not LoRA; there are no LoRA tensors in the file); paper App. C: "unfreezing
pre-trained weights for WavLM leads to improved performance in emotion prediction tasks"; `ER/models/README.md`: "a fully
fine-tuned WavLM-large". **Inconsistency to fix:** the model survey (table 1b #5) and the slide
(`s_models.js`: "WavLM-large + LoRA") say LoRA.

### 1.5 Forward compute per clip (FlopCounterMode, matmul + conv only)

| Model | What runs | GFLOP per clip |
|---|---|---|
| emotion2vec+ large, 4 s | real length | 87.6 (measured in `ER/docs/finetuning_emotion2vec_plus_large_en.md`) |
| MERaLiON-SER-v1, any clip ≤ 30 s | Whisper-medium encoder 1,138 + downstream 11 | **1,149** |
| Melati, any clip ≤ 30 s | Whisper-large-v3 encoder | **2,274** |
| Vox-Profile Whisper, any clip ≤ 15 s | encoder with 750 positions | **1,045** (+0.7 head) |
| Vox-Profile WavLM, 4 s | real length | 147 |

Cross-check by hand for Whisper-medium: 2 × 307M params × 1500 frames ≈ 0.92 TFLOP, plus attention scores
(2 × 1500² × 1024 × 24 × 2 ≈ 0.22 TFLOP) and conv1 over 3,000 frames ≈ 1.14 TFLOP. ✔

---

## 2. Fine-tuning MERaLiON-SER-v1 (options)

| Option | What is trained | Trainable params | Encoder pass per step | Notes |
|---|---|---|---|---|
| **A. Heads only** | categorical head (+ VAD head) on the frozen 256-d embedding | 67,847 (categorical only); 134,666 with the VAD head | none: cache the 256-d embedding once (1 forward per clip) | new label set possible; cheapest |
| **B. Downstream + heads** | ECAPA-TDNN, hybrid pooling, embedding MLP, heads | **6,146,197 (6.1M)** | none if the encoder output (1500×1024) is cached: 3.07 MB per clip in fp16 (about 31 GB per 10k clips); downstream ≈ 11 GFLOP fwd | caching rules out on-the-fly audio augmentation |
| **C. LoRA r=8 on encoder q/k/v (all 24 layers)** | LoRA + usually B too | LoRA **1,179,648 (1.18M)** = 24 × 3 × 8 × (1024 + 1024); with B: 7,325,845 | full fwd + bwd through the encoder every step (≈ 2× forward, ~2.3 TFLOP per clip) | variants: q,v only 786,432; q,k,v,o 1,572,864; top 12 layers q,k,v 589,824; the release used r=16 on q,k,v = 2,359,296 |
| **D. Full** | encoder + downstream + heads | **313,362,581**; 311,826,581 really trainable (positional table fixed) | fwd + bwd ≈ 3× forward ≈ 3.4 TFLOP per clip | fp32 weights + grads + AdamW ≈ 16 B/param ≈ 5.0 GB before activations (1,500 tokens × 24 layers per clip) |

Rough compute (estimate): forward 1.15 TFLOP per clip whatever its length (≤ 30 s), i.e. a 2 s clip costs as much as a
30 s one. On one A100 (312 TFLOPS bf16 peak, assume 30–40 % utilisation ≈ 100–125 TFLOPS): caching features for 10k
clips ≈ 2 min; 10k clips × 10 epochs ≈ 0.5–0.65 h for C and ≈ 0.8–1 h for D. Fine-tuning is also a chance to shorten the
window: an 8 s window (400 frames) would cost about 0.26 TFLOP (≈ 4.4× less; estimate, needs retraining because the
model was trained on 30 s padded inputs).

Official code: **no training or fine-tuning code is published.** The HF repo has only inference remote code
(`modeling_ser_whisper_ecapa.py`) and two exporters (`exporter_lora_whisper.py`, `exporter_stub.py`) that convert a
Lightning checkpoint and merge LoRA; the 3 HF discussions ask about metrics, the whisper-medium config and checkpoint
mismatch, none about training; no GitHub repository found (https://huggingface.co/MERaLiON/MERaLiON-SER-v1/discussions,
web search 2026-11-02). The paper §3 gives the recipe (above). The modeling code already contains the LoRA path
(`finetune_strategy: "peft"` + `peft_config`) and `use_weighted_sum` (layer weights), so a training script can reuse the
model class.

Licence for fine-tuning: MERaLiON Public Licence (MIT plus attribution terms), commercial use allowed (survey).

---

## 3. Audio LLMs

### 3a. LISTEN (arXiv 2510.10444)

- Title: **"Do Audio LLMs Really LISTEN, or Just Transcribe? Measuring Lexical vs. Acoustic Emotion Cues Reliance"**.
  Jingyi Chen, Zhimeng Guo, Jiyun Chun, Pichao Wang, Andrew Perrault, Micha Elsner: **The Ohio State University**
  (Linguistics; CSE), Penn State (Guo), Amazon (Wang). v1 2025-10-12, v2 2025-10-17.
- Models (6): Qwen2.5-Omni-7B, Qwen3-Omni-30B, Baichuan-Omni-1.5, Qwen3-Instruct (text only), Gemini 2.5 Flash,
  Gemini 2.5 Pro. Conditions: Neutral-Text, Emotion-Matched, Emotion-Mismatched, Paralinguistic (laughs, sighs); each as
  text-only / audio-only / text+audio. Sources: CREMA-D, ESD, TESS, SAVEE, RAVDESS, IEMOCAP, CMU-MOSEI, OMG-Emotion,
  MSP-Podcast, MELD, MUSTARD++. Metric: accuracy.

Key findings (Table 2, https://arxiv.org/html/2510.10444v2):

1. **Neutral words → neutral answers.** Neutral-Text, audio only: Qwen2.5-Omni 34.0 %, Qwen3-Omni 29.3 %, Baichuan
   16.5 %, Gemini 2.5 Flash 25.6 %, Gemini 2.5 Pro 34.9 % (uniform 12.5 %, majority 16.9 %). With text only (all labels
   neutral) they reach 81.2–96.6 %. Confusion matrices show "a single dominant bar in the neutral column".
2. **Adding the transcript makes it worse, not better.** Neutral-Text audio → text+audio: Qwen2.5-Omni 34.0 → 19.8 %,
   Baichuan 16.5 → 15.2 %, Gemini 2.5 Flash 25.6 → 24.6 %; only Gemini 2.5 Pro gains (34.9 → 41.7 %).
3. **Without words, near chance.** Paralinguistic (audio only): 15.7–22.7 % (uniform 12.5 %); models "revert to neutral
   as the safest default". Under cue conflict they detect a mismatch but collapse it into sarcasm-like classes. Authors'
   conclusion: LALMs largely "transcribe" rather than "listen".

Correction: `ER/docs/datasets_survey_en.md` (Key points) says text-only accuracy was "85–97%"; Table 2 gives 81.2–96.6 %
for the five audio LLMs (66.6 % for text-only Qwen3-Instruct).

Related: the MERaLiON-SER paper cites LISTEN and also compares with GPT-4o-Audio and Gemini 2.5 Flash, which "lag behind
specialized SER models" (arXiv 2511.04914 §5–6).

### 3b. GPT Realtime (OpenAI)

- `gpt-realtime` announced **2025-08-28** with the GA Realtime API; snapshot `gpt-realtime-2025-08-28`. One
  speech-to-speech model (no STT → LLM → TTS chain). Input text, audio, image; output text, audio. Context 32,000 tokens,
  max output 4,096, knowledge cutoff 2023-10-01. Prices per 1M tokens: audio in $32, audio out $64, text in $4.
  Parameters **not disclosed**. (https://developers.openai.com/api/docs/models/gpt-realtime; launch post
  https://openai.com/index/introducing-gpt-realtime/ returned 403 to the fetcher, date from search results.)
- Audio input: `audio/pcm` = PCM16, **24 kHz**, mono, little-endian; also `audio/pcmu` / `audio/pcma` (G.711 µ-law /
  A-law, the telephone codecs); base64 in `input_audio_buffer.append` or an `input_audio` content part
  (https://developers.openai.com/api/docs/guides/realtime-conversations). So our README's "the only input" is true only
  for linear PCM.
- **Status now: marked deprecated** ("superseded by newer versions"); the models page lists GPT-Realtime-1.5, -2, -2.1,
  -2.1 Mini and GPT-Live 1. Our registry still calls `gpt-realtime` (`ER/config/models.yaml`). Decide before the paid run.

### 3c. Gemini Flash

- **Gemini 3.5 Flash exists**: `gemini-3.5-flash`, launched 2026-05-19 at Google I/O 2026 (secondary sources), input
  text, image, video, audio, PDF; output text; 1,048,576 input / 65,536 output tokens. Parameters **not disclosed**. Its
  Gemini API page now says it is **deprecated and routed to `gemini-3.6-flash`**
  (https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash). Current stable Flash per the models page:
  `gemini-3.8-flash` (3.6 Flash 2026-07-21, 3.8 Flash 2026-09-02 per secondary sources, unverified).
- Audio in the Gemini API: WAV, MP3, AIFF, AAC, OGG, FLAC and others; **32 tokens per second**; up to 9.5 h per prompt;
  downsampled to 16 kbps and mixed to mono; inline requests ≤ 20 MB (https://ai.google.dev/gemini-api/docs/audio).
- Our state: `ER/config/models.yaml` has the placeholder `gemini-2.5-flash`, which the models page now lists as limited to
  existing users; `ER/scripts/models/gemini.py` is a placeholder (not implemented). No GPT Realtime results exist yet in
  `ER/outputs/` either.

### 3d. Our audio-LLM protocol (`ER/scripts/llm_eval/README.md`, `prompts.yaml`, `ER/config/llm.yaml`, `ER/scripts/models/gpt_realtime.py`)

- **Clips**: 200 per dataset × 6 datasets = **1,200 clips, about 47.5 min of audio** per model; the same clips as the
  open models with `limit: 200` (`sample()` in `ER/scripts/common/eval_datasets.py`: clips sorted by ground-truth label,
  then every k-th; so every label is covered, in proportion to its share; equal only when the dataset is balanced, e.g.
  JL-Corpus 40 per emotion).
- **Audio**: GPT Realtime gets PCM16 24 kHz mono base64 (resampled with torchaudio, clipped, rounded to int16); Gemini
  will get a 16 kHz mono 16-bit WAV. ESD is never sent (licence).
- **Prompt**: forced choice of exactly one lowercase word from the 7 labels; system prompt says the words are neutral and
  the emotion is in the voice ("judge the voice, not the words"), and not to reply, translate or transcribe.
- **Session**: one WebSocket session per clip, text output only, no turn detection; the Realtime API does not allow
  temperature 0 (answers can vary); Gemini planned at temperature 0.
- **Parsing / retries**: `parse_label` accepts one label or a common variant in ≤ 15 words, no negation; invalid answers
  are asked again once (`retries_invalid: 1`), then the class `invalid` (counted wrong); transient errors retried 3 times
  with 2/4/8 s backoff; timeout 60 s; every answer cached in `responses.jsonl` (nothing paid twice).
- Label rules shared with the open models: calm → neutral (RAVDESS, MESS), TESS pleasant surprise → surprised, JL-Corpus
  excited → happy or surprised counts as correct.

---

## 4. Survey counts and selection criteria

| Survey | Rows | Selection |
|---|---|---|
| Datasets (`ER/docs/datasets_survey_en.md`, 2026-10-23) | **37 datasets**: 8 high-quality not in EmoBox (§1), 14 high-quality in EmoBox (§2), 15 small / not parallel / flawed (§3) | criteria: the same sentences in different emotions (words give no clue), high-quality labels, large size, clear licence and access. **9 downloaded** (✅) → **6 evaluated**: CREMA-D, MCAE-SPPS, MESS, RAVDESS, TESS, JL-Corpus. Excluded: ESD (research-only licence, no listener test, in EmoBox), THAI-SER (Thai) and nEMO (Polish) as less relevant languages (`ER/README.md` §6; `ER/config/evaluation.yaml` `exclude`) |
| Models (`ER/docs/models_survey_en.md`, 2026-10-28) | **42 entries**: 17 categorical SER (§1a; the SUPERB row is 4 checkpoints), 4 community checkpoints (§1c), 7 dimensional (§2a), 2 with emotion tags (§3), 12 open-weight audio LLMs (§4) | criteria: downloadable weights with code or docs to run them, English supported, any size, whole emotion2vec family; non-commercial included but flagged. ⭐ = good reported performance, classes close to the project's 7, runnable, then licence. **Evaluated: 5 open** (emotion2vec+ large, MERaLiON-SER-v1, Melati, Vox-Profile Whisper, Vox-Profile WavLM) **+ 2 closed** API LLMs (GPT Realtime, Gemini; not part of the open-weight survey) |

Slide wording: "Surveyed 37 datasets → selected 6" and "Surveyed 42 open models → evaluated 5 open + 2 closed".

---

## 5. Licences (one line each)

Evaluation datasets (`ER/docs/datasets_survey_en.md`):

| Dataset | Licence |
|---|---|
| CREMA-D | ODbL (contents under DbCL) |
| MCAE-SPPS | CC BY-NC 4.0 (paper CC BY-NC-ND 4.0) |
| MESS | CC BY 4.0 |
| RAVDESS | CC BY-NC-SA 4.0 |
| TESS | CC BY-NC 4.0 |
| JL-Corpus | CC0 |

Open models (`ER/docs/models_survey_en.md`, `ER/models/README.md`):

| Model | Licence |
|---|---|
| emotion2vec+ large | FunASR Model License v1.1 on HF (attribution, no commercial restriction); Apache-2.0 on ModelScope; code MIT |
| MERaLiON-SER-v1 | MERaLiON Public Licence (MIT plus attribution terms), commercial use allowed |
| Melati | non-commercial research only |
| Vox-Profile Whisper | OpenRAIL; card rules out commercial use, surveillance, clinical use (inference code: Responsible AI Source Code License v1.1) |
| Vox-Profile WavLM | same as Vox-Profile Whisper |

---

## 6. Things to fix or watch

1. Vox-Profile WavLM is fully fine-tuned, not LoRA: model survey table 1b #5 and `s_models.js` say "LoRA".
2. LISTEN text-only range is 81–97 %, not "85–97 %" (dataset survey Key points).
3. `gpt-realtime` is now marked deprecated; `gemini-2.5-flash` is limited to existing users and Gemini 3.5 Flash is
   deprecated (routed to 3.6 Flash). Choose the API model names before the paid runs; neither LLM has results yet.
4. MERaLiON card says 309M; the file gives 313.4M in use (311.8M without the fixed positional table).

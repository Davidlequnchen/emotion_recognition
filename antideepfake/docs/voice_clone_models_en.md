# Voice-Cloning Models

English translation of [voice_clone_models.md](voice_clone_models.md) (Chinese); the two have the same content.

Date: 2026-10-30. Purpose: generate the 200 lines of `data/voice_clone/texts.csv` as synthetic speech in the voice of
a consenting speaker, then score it with AntiDeepfake. Only models that allow commercial use were chosen (Apache-2.0,
MIT); F5-TTS (weights CC-BY-NC) and XTTS-v2 (Coqui Public Model License) forbid it and were left out.

Sources:
- "measured": measured on this M4 Max with `scripts/voice_clone/generate.py`, or read from the downloaded configs and code;
- "paper", "model card": from the links in section 2.

## 1. Overview

| | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox (English) |
|---|---|---|---|
| Key in this project | `cosyvoice2` | `cosyvoice3` | `chatterbox` |
| Developer | Alibaba Tongyi Lab (FunAudioLLM) | Alibaba Tongyi Lab (FunAudioLLM) | Resemble AI |
| Released | 2024-12 | 2025-12 | 2025 |
| Parameters (measured, PyTorch parts + ONNX) | 639.1M + 130.6M ≈ 0.77B | 859.2M + 248.9M ≈ 1.11B | 797.8M (plus the 9.3M Perth watermark network) |
| Download size (measured, only the files needed to run) | 3.09 GB | 4.44 GB | 3.19 GB |
| Licence | Apache-2.0 | Apache-2.0 | MIT |
| Languages | Chinese, English, Japanese, Korean (training data in the paper) | 9: Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian, plus 18+ Chinese dialects (model card) | English only |
| Cloning | zero-shot: reference recording + transcript | zero-shot: reference recording + transcript | zero-shot: reference recording only |
| Output | 24 kHz mono | 24 kHz mono | 24 kHz mono, with an inaudible Perth watermark |
| Device on the Mac (measured) | CPU (the code knows only CUDA or CPU) | CPU | MPS (CPU also works) |
| Speed, RTF (measured, M4 Max) | about 0.95 | about 1.1–1.3 | MPS about 1.3–1.6; CPU about 2.3–2.5 |
| Peak memory (measured) | 6.7 GB | 9.3 GB | 6.1 GB (14.7 GB with the GPU) |

RTF = generation time ÷ audio duration; below 1 is faster than real time.

## 2. Sources, papers, and how the company computer gets them

The company computer reaches GitHub, Zenodo and PyPI, not Hugging Face or ModelScope.

| | Weights | Code | Paper | Company computer |
|---|---|---|---|---|
| CosyVoice2-0.5B | [HF FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B), revision `eec1ae6c`; also on [ModelScope iic/CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) | [GitHub FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), commit `074ca6dc` (Matcha-TTS submodule `dd9105b3`) | [CosyVoice 2 (arXiv:2412.10117)](https://arxiv.org/abs/2412.10117) | weights: 90 MiB parts in the private repository [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models); code: cloned from GitHub by the download script |
| Fun-CosyVoice3-0.5B-2512 | [HF FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512), revision `29e01c4e`; also on [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | as above | [CosyVoice 3 (arXiv:2505.17589)](https://arxiv.org/abs/2505.17589) | as above |
| Chatterbox | [HF ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox), revision `5bb1f6ee` | [GitHub resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox); pip package `chatterbox-tts==0.1.7` | no paper (model card and GitHub only) | weights: parts in the private repository [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models); code: PyPI |
| wetext text-normalisation FSTs (used by CosyVoice) | [ModelScope pengzhendong/wetext](https://www.modelscope.cn/models/pengzhendong/wetext), revision `67928a15`, Apache-2.0 | pip package `wetext==0.0.4` | – | the private repository [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models) (4 files, 11 MB in all) |

The full revisions and commits are in `config/voice_clone.yaml`. On the company computer: clone the private repository [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models) into `models/voice_clone/`, run its `join_parts.py` to join the parts (each SHA-256 is checked), then run `download_models.py`: it clones the CosyVoice code from GitHub and skips the model files that are already there, without contacting Hugging Face.

## 3. Input and output

| | CosyVoice2 / CosyVoice3 | Chatterbox |
|---|---|---|
| Reference recording | any sample rate, resampled inside to 16 kHz (speech tokenizer, CAM++) and 24 kHz (prompt mel); at most 30 s (an assert in the code) | resampled inside to 16 and 24 kHz; T3 uses only the first 6 s (150 tokens), S3Gen the first 10 s |
| Reference transcript | required, word for word; CosyVoice3 needs `You are a helpful assistant.<|endofprompt|>` in front of it (set in the config) | not needed |
| Text to speak | any length: normalised by wetext first (numbers, dates, amounts to words), split into sentences, made one by one and joined | at most 2,048 text tokens; at most 1,000 speech tokens are generated (about 40 s) |
| Output | float32 waveform, 24 kHz mono; saved here as 16-bit PCM wav | as left |
| Files in this project | `outputs/voice_clone/<speaker>/<model>/<id>.wav`, plus `manifest.csv` | as left |

## 4. Architecture

| Component | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox |
|---|---|---|---|
| Text tokenizer | Qwen2 BPE (vocab 151,936) | Qwen2 BPE | BPE, 704 tokens, English |
| Speech tokenizer (reference → tokens) | ONNX, 123.7M; SenseVoice-Large encoder + FSQ; input 16 kHz 128-dim log-mel; 25 tokens/s, codebook 6561 | ONNX, 242.0M; MinMo speech encoder + FSQ, multi-task training (ASR, emotion, language, audio events, speaker); 25 tokens/s, codebook 6561 | S3TokenizerV2, 123.7M (the same structure as CosyVoice2's); 25 tokens/s, codebook 6561 |
| Speaker encoder | CAM++ (ONNX, 6.9M), 16 kHz 80-dim fbank → 192-dim | CAM++ (as left) | voice encoder 1.4M (256-dim, for T3) + CAM++ 6.8M (for S3Gen) |
| Language model (text → speech tokens) | 505.8M, Qwen2.5-0.5B backbone: 24 layers, hidden 896, 14 heads (2 KV heads); autoregressive, continues after the reference tokens | 506.1M, the same Qwen2.5-0.5B backbone; post-trained with DiffRO (differentiable reward) | T3, 532.4M: Llama structure, 30 layers, hidden 1024, 16 heads; conditioned on the speaker vector, the emotion intensity and the reference tokens |
| Flow matching (tokens → mel) | 112.5M: 6-layer UpsampleConformerEncoder + causal convolutional U-Net CFM; chunk-aware, streams; 10 Euler steps, CFG 0.7 | 332.3M: DiT, 22 layers, dim 1024; 10 Euler steps, CFG 0.7 | S3Gen's flow, 112.5M (the same structure as CosyVoice2's), 10 steps, CFG 0.7 |
| Vocoder (mel → waveform) | HiFT (HiFi-GAN + iSTFT, NSF source), 20.8M, 24 kHz | CausalHiFT, 20.8M, 24 kHz | HiFT, 20.8M, 24 kHz |
| Other | – | – | Perth implicit watermark, 9.3M (ships with the pip package), added to every output; upstream has no switch |
| Sampling | RAS: top_p 0.8, top_k 25, window 10, tau_r 0.1 | as left | temperature 0.8, min_p 0.05, top_p 1.0, repetition_penalty 1.2, CFG 0.5 |

## 5. Training data and reported metrics

| | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox |
|---|---|---|---|
| Training data | about 167k h: Chinese 130k, English 30k, Japanese 4,600, Korean 2,200 (paper) | about 1M h, 9 languages and 18 dialects; speech tokenizer trained on 530k h of multi-task data (paper) | 0.5M h of cleaned data (model card) |
| SEED test-zh CER / SS | 1.45% / 75.7% | 1.21% / 78.0% | – |
| SEED test-en WER / SS | 2.57% / 65.9% | 2.24% / 71.8% | – |
| SEED test-hard CER / SS | 6.83% / 72.4% | 6.71% / 75.8% | – |
| Source of the metrics | CosyVoice3 model card | CosyVoice3 model card (the RL version, `llm.rl.pt`, has lower WER but was not downloaded) | the model card gives only a listening comparison with ElevenLabs |

Lower CER/WER is better (content); SS is speaker similarity, higher means closer to the reference voice.

## 6. Generation parameters (`config/voice_clone.yaml`; the defaults are upstream's)

| Parameter | Model | Default | Effect |
|---|---|---|---|
| `speed` | CosyVoice2/3 | 1.0 | speaking rate |
| `text_frontend` | CosyVoice2/3 | true | wetext text normalisation (numbers, amounts to words) |
| `prompt_prefix` | CosyVoice3 | `You are a helpful assistant.<\|endofprompt\|>` | put before the reference transcript |
| `exaggeration` | Chatterbox | 0.5 | emotion intensity, 0.25–2.0, 0.5 is neutral |
| `cfg_weight` | Chatterbox | 0.5 | CFG weight, 0–1; lower is slower |
| `temperature`, `repetition_penalty`, `min_p`, `top_p` | Chatterbox | 0.8, 1.2, 0.05, 1.0 | sampling of the speech tokens |
| `seed` | all | 1234 | row i (from 0) uses seed + i; on the same machine and device every run gives byte-identical files (measured: CosyVoice on the CPU, Chatterbox on MPS and on the CPU; MPS and CPU give different clips) |
| `mac_device` | each model | CosyVoice: cpu; Chatterbox: mps | the device on the Mac |

## 7. Downloaded files

| Model | Files (size) | Not downloaded, and why |
|---|---|---|
| `models/voice_clone/cosyvoice2_0.5b/` | `llm.pt` 2023 MB, `speech_tokenizer_v2.onnx` 496 MB, `flow.pt` 451 MB, `hift.pt` 83 MB, `campplus.onnx` 28 MB, `cosyvoice2.yaml`, the small tokenizer and config files of `CosyVoice-BlankEN/`, `README.md` | `CosyVoice-BlankEN/model.safetensors` (988 MB, the plain Qwen2.5-0.5B weights, all overwritten by `llm.pt`; this project builds the backbone from config.json alone, with output bit-identical to the official path), `*.batch.onnx` (training), `flow.decoder.estimator.fp32.onnx` (TensorRT) |
| `models/voice_clone/cosyvoice3_0.5b/` | `llm.pt` 2025 MB, `flow.pt` 1329 MB, `speech_tokenizer_v3.onnx` 970 MB, `hift.pt` 83 MB, `campplus.onnx` 28 MB, `cosyvoice3.yaml`, the small files of `CosyVoice-BlankEN/`, `README.md` | as above, plus `llm.rl.pt` (2.0 GB, the RL-tuned language model; the official code loads only `llm.pt`) |
| `models/voice_clone/chatterbox/` | `t3_cfg.safetensors` 2130 MB, `s3gen.safetensors` 1057 MB, `ve.safetensors` 5.7 MB, `conds.pt`, `tokenizer.json`, `README.md` | the `.pt` copies of the same weights, `s3gen_v3`, all multilingual files |
| `models/voice_clone/wetext/` | the tagger and verbalizer FSTs of `en/tn/` and `zh/tn/` (11 MB in all), `README.md` | newer FSTs (made for wetext 0.1.x; with 0.0.4 they misread numbers or crash) |

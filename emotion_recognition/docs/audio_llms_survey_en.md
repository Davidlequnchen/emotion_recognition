# Audio LLM Survey: What to Add Next

English version of [audio_llms_survey.md](audio_llms_survey.md) (Chinese); the two have the same content. The survey of dedicated SER models is in [models_survey_en.md](models_survey_en.md), the dataset survey in [datasets_survey_en.md](datasets_survey_en.md).

**Updated 2026-11-06 for plan B**: 13 datasets (SAVEE added; ESD out of scope), with THAI-SER limited to the clips with agreement ≥ 0.71 and MEAD to a stratified 8,000 clips, for every model; API models now run every clip of every dataset (no 200-clip sample). The audio volume, costs and run times below are for this plan-B set: 56,264 clips, 52.2 h, all measured.

Survey date: 2026-11-05. Goal: decide which audio LLMs to add to this project's benchmark after the five SER models, GPT Realtime, Gemini and the three local audio LLMs being added now (Qwen2.5-Omni-7B, Qwen2-Audio-7B-Instruct, Gemma 4 E4B-it): what each costs through an API, what hardware it needs to run here, and how hard it is to integrate.
Setting: academic research only, so non-commercial licences are acceptable (they are flagged, not excluded). Hardware: a Mac M4 Max with 48 GB; an NVIDIA L40 (48 GB) for certain; an NVIDIA L4 (24 GB) and 4×H100 80 GB possibly.

Method: read the price and billing pages of Alibaba Cloud Model Studio, OpenAI, Google and StepFun, the OpenRouter model list, the model cards, papers and GitHub code; counted parameters from the safetensors headers with HTTP range requests (no weights were downloaded); measured the audio length of all 13 datasets from the file headers only (`soundfile.info`, clip lists from `LOADERS[key]().clips` in `scripts/common/eval_datasets.py`, which already apply the plan-B subsets of `config/evaluation.yaml`); counted the prompt tokens of `scripts/llm_eval/prompts.yaml` with the Qwen2.5-Omni tokenizer.
Units: M = million, B = billion parameters; sizes in decimal GB; prices in USD per million tokens ($/M). "Estimate" marks a number worked out here rather than read from a source; "unverified" marks what could not be confirmed from a primary source. Prices are those listed on 2026-11-05.

## 0. In short

- **Audio volume (plan B)**: 13 datasets, THAI-SER only agreement ≥ 0.71 (11,848 clips), MEAD a stratified 8,000 clips: **56,264 clips, 52.2 h**, all measured from the file headers. API models run every one of these clips; ESD is never sent.
- **API cost is small** (section 3). The whole plan-B set costs **$31.25** with Qwen3.5-Omni Plus, **$8.69** with Qwen3.5-Omni Flash, **$14.35** with qwen3-omni-flash and $1.84 with the new qwen3.8-omni-flash, against **$106.15** with GPT Realtime and $9.87 with Gemini 2.5 Flash (international prices; mainland China is 20–40% cheaper). A 200-clip-per-dataset pilot (2,600 clips) would cost $0.08–1.38 with the Qwen models.
  The fixed text prompt (about 175 tokens) is 80–90% of the input tokens of every Qwen model (a 2.5 s clip is only 18–31 audio tokens) and 30–82% of the cost: shortening it is the main way to save.
- **Run time**: GPT Realtime, one request at a time, measured 5.5 s per clip: the plan-B set takes about **86 h** (3.6 days) per API model run this way; requests in parallel shorten it, within each API's rate limit.
- **Qwen3-Omni-30B-A3B-Instruct** (open, Apache-2.0): the API model `qwen3-omni-flash` is a closed, further-trained Flash version, not the open weights. Self-hosted, the Thinker alone (talker off) is **31.72B parameters = 63.4 GB in bf16**: it fits **1×H100 80 GB** (tight) or 2×H100 (comfortable); on the **L40 only with a community FP8 or 4-bit checkpoint** (no official quantised one). transformers 5.18.0 (pinned here) loads it natively; vLLM supports the Thinker.
- **Qwen3.5-Omni** (2026-03-30, arXiv 2604.15804): **API only**, `qwen3.5-omni-plus` and `qwen3.5-omni-flash`; no Qwen3.5-Omni repository exists on Hugging Face (checked 2026-11-05; the "open Light version" claimed by some blogs was not found). Audio is billed at 7 tokens per second. Its technical report has no emotion recognition results.
- **Kimi-Audio-7B-Instruct** (MIT, 9.77B): fits L40 and H100, not the Mac (CUDA, flash-attn, torch 2.6). Its own MELD 59.13 is the best published MELD figure among these candidates, but **its SFT data includes RAVDESS, MELD and SAVEE**, three of our datasets.
- **Step-Audio 2 mini** (Apache-2.0, 8.32B): fits L4/L40/H100; needs its own environment (transformers==4.49.0, `.cuda()` hard-coded). Its emotion score (82) is on its own 50-clip Chinese podcast test, judged by an LLM; no MELD or IEMOCAP result. The larger Step-Audio 2 is API only (`step-audio-2`, a realtime API).
- **Audio Flamingo 3** (8.27B, NVIDIA non-commercial): the easiest to add (transformers native, fits every GPU, probably the Mac too), but **its training data contain all of JL-Corpus and TESS and MELD's training data** (paper, table 11). Its results on JL-Corpus and TESS must be marked "seen", and on MELD "possibly seen".
- **Recommended order** (section 7): (1) Qwen3-Omni open weights on 1×H100 (or L40 with FP8); (2) Qwen3.5-Omni Plus and Flash through the API on every plan-B clip (about $40 together: $31.25 + $8.69); (3) Audio Flamingo 3 with "seen" flags; (4) Step-Audio 2 mini; (5) Kimi-Audio. MiDashengLM and Voxtral: not worth it for SER.

## 1. Candidates at a glance

The figures come from different protocols and cannot be compared across rows; "overlap" is the overlap of the model's published training data with this project's 13 datasets (section 6).

| Model | Organisation, release | Size | Licence | Weights / API | Published SER result | Overlap with our 13 datasets | Verdict |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | Alibaba Qwen, 2025-09 | 35.26B in all; Thinker 31.72B (MoE, about 3B active per token) | Apache-2.0 | open ([HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)); API `qwen3-omni-flash` is a closed Flash version | none in its [technical report](https://arxiv.org/abs/2509.17765); third-party MELD 56.8 (MiniCPM-o 4.5 card, see [models_survey_en.md](models_survey_en.md)) | training data not published | **add first** (open, strongest open Qwen) |
| Qwen3.5-Omni Plus / Flash | Alibaba Qwen, 2026-03-30 | not disclosed ("hundreds of billions", MoE; [report](https://arxiv.org/abs/2604.15804)) | proprietary | API only | none in the report | not published | **add via API** |
| qwen3.8-omni-flash | Alibaba Qwen, 2026-09 (on OpenRouter since 2026-09-21) | not disclosed | proprietary | API only | none found | not published | optional, cheapest API |
| Kimi-Audio-7B-Instruct | Moonshot AI, 2025-04 | 9.77B (safetensors) | MIT (parts from Qwen2.5-7B: Apache-2.0) | open ([HF](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)) | MELD 59.13 (self-reported, [GitHub](https://github.com/MoonshotAI/Kimi-Audio)) | **RAVDESS, MELD, SAVEE** in SFT ([paper](https://arxiv.org/abs/2504.18425)) | add later; flag RAVDESS, MELD, SAVEE |
| Step-Audio 2 mini | StepFun, 2025-08 | 8.32B | Apache-2.0 | open ([HF](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)); larger Step-Audio 2: API only | emotion 82 on its own StepEval-Audio-Paralinguistic (50 Chinese podcast clips, open-set, LLM-judged; [paper](https://arxiv.org/abs/2507.16632) table 8) | not published | add later |
| Audio Flamingo 3 | NVIDIA, UMD, 2025-07 | 8.27B | NVIDIA OneWay Noncommercial | open ([HF](https://huggingface.co/nvidia/audio-flamingo-3-hf)) | IEMOCAP 63.8% ([paper](https://arxiv.org/abs/2507.08128)) | **JL-Corpus, TESS (all clips), MELD** | add, with "seen" flags |
| Audio Flamingo Next | NVIDIA, UMD, 2026-04 | 8.27B | NVIDIA OneWay Noncommercial | open ([HF](https://huggingface.co/nvidia/audio-flamingo-next-hf)) | none found | builds on AF3's data (AF-Skills); overlap likely, unverified | only if AF3 is useful |
| MiDashengLM-7B (1021) | Xiaomi, 2025-10 | 8.28B | Apache-2.0 | open ([HF](https://huggingface.co/mispeech/midashenglm-7b-1021-bf16)) | none (card reports speaker, language, sound classification only) | not published | skip for now |
| Voxtral Mini / Small | Mistral, 2025-07 | 4.68B / 24.26B | Apache-2.0 | open ([Mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507), [Small](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)); Small also on the Mistral API and OpenRouter | none found (built for transcription and spoken QA) | not published | skip |

## 2. API models: names, prices and audio token rates

Prices in $/M for audio input to text output (no audio output). "Text out" is the price for text output when the input contains audio.

| Model name (API) | Region | Audio in | Text in | Text out | Audio tokens per second | Source |
|---|---|---|---|---|---|---|
| `qwen3-omni-flash` | International (Singapore) | 3.81 | 0.43 | 3.06 (1.66 for text-only input) | 12.5 | [Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing); [token rules](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) |
| `qwen3-omni-flash` | Chinese mainland (Beijing) | 2.265 | 0.258 | 1.821 | 12.5 | as above |
| `qwen3.5-omni-plus` (= `qwen3.5-omni-plus-2026-03-15`) | International | 11 | 1.4 | 8.3 | 7 | as above |
| `qwen3.5-omni-plus` | Chinese mainland | 7.29 | 0.96 | 5.5 | 7 | as above |
| `qwen3.5-omni-flash` (= `qwen3.5-omni-flash-2026-03-15`) | International | 3 | 0.4 | 2.2 | 7 | as above |
| `qwen3.5-omni-flash` | Chinese mainland | 2.48 | 0.3 | 1.83 | 7 | as above |
| `qwen3.8-omni-flash` | International | 0.15 (one price for all inputs) | 0.15 | 0.47 | 7 | as above; also [OpenRouter](https://openrouter.ai/qwen/qwen3.8-omni-flash) `qwen/qwen3.8-omni-flash` at the same price |
| `qwen3.8-omni-flash` | Chinese mainland, Hong Kong, Tokyo, Frankfurt, US (Virginia) | 0.113 | 0.113 | 0.382 | 7 | as above |
| `step-audio-2` (StepFun, realtime API) | international platform | 1.43 (input, cache miss) | 1.43 | 10.00 (output) | not published | [StepFun pricing](https://platform.stepfun.ai/docs/en/guides/pricing/details) |
| `gpt-realtime` (reference) | — | 32 | 4 | 16 | 10 (1 token per 100 ms) | [OpenAI pricing](https://developers.openai.com/api/docs/pricing); [Realtime costs](https://developers.openai.com/api/docs/guides/realtime-costs) |
| `gpt-realtime-mini` (reference) | — | 10 | 0.6 | 2.4 | 10 | as above |
| `gemini-2.5-flash` (reference; the placeholder in `config/models.yaml`) | — | 1.00 | 0.30 | 2.50 | 32 | [Gemini pricing](https://aistudio.google.com/docs/pricing); [audio docs](https://aistudio.google.com/docs/audio) |
| `gemini-3-flash-preview` (reference) | — | 1.00 | 0.50 | 3.00 | 32 | as above |
| `gemini-3.5-flash` (reference) | — | 0.75 | 0.75 | 3.75 | 32 | as above (price valid until 2026-12-31) |

Notes:
- **Qwen token rules** (Model Studio, "Billing and rate limits"): Qwen3.8-Omni-Flash and the Qwen3.5-Omni series count input audio as duration (s) × 7; Qwen3-Omni-Flash as duration (s) × 12.5; audio shorter than 1 s counts as 1 s (2,508 plan-B clips are under 1 s: 1,797 in MCAE-SPPS, 468 in THAI-SER, 225 in MELD, 18 in other datasets; the costs below include this). Longer clips are billed by their exact duration, not per started second. These match the encoders: Qwen3-Omni's AuT runs at 12.5 Hz ([paper](https://arxiv.org/abs/2509.17765)), Qwen3.5-Omni's encoder at 6.25 Hz ([paper](https://arxiv.org/abs/2604.15804)).
- **Endpoints**: international (Singapore) and Chinese mainland (Beijing) workspaces need separate API keys; the OpenAI-compatible base URL is per workspace, e.g. `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` for Singapore. `qwen3.8-omni-flash` is also in Hong Kong, Tokyo, Frankfurt and Virginia; the others only in Singapore and Beijing.
- **Streaming only**: Qwen3.5-Omni and Qwen3-Omni-Flash accept only `stream=True` (a non-streaming request fails); qwen3.8-omni-flash accepts both ([docs](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)). A loader must collect the streamed text.
- **Audio input limits**: base64 data under 10 MB; Qwen3-Omni-Flash up to 20 minutes of audio, Qwen3.5-Omni and Qwen3.8 up to 3 hours. Our longest clip is 91 s (THAI-SER).
- **Free quota**: in Singapore, `qwen3.5-omni-plus` comes with 1M free tokens, valid for 90 days from Model Studio activation, model release or application approval, whichever is later; the dated snapshot and `qwen3.5-omni-flash` have none, nor does `qwen3-omni-flash` (pricing page). The plan-B run with Plus is about 11.7M tokens, so the free quota covers under a tenth of it (about $2.7); a 2,600-clip pilot (about 0.53M tokens) would fit in it.
- **Rate limit**: `qwen3-omni-flash` in Beijing: 60 requests per minute, 100,000 tokens per minute ([model page](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-omni-flash)); at 60 per minute the 56,264 plan-B clips take at least 15.6 h (about 220 tokens per clip, so the token limit is not reached). International limits unverified.
- **qwen3-omni-flash is not the open model**: Qwen3-Omni-Flash-2025-12-01 is "a comprehensively upgraded iteration built upon Qwen3-Omni" ([Alibaba Cloud blog](https://www.alibabacloud.com/blog/602735)). Results through the API do not stand for the open weights, and the API version may change without notice; the open weights are the reproducible choice.
- **OpenRouter** (model list read 2026-11-05): has `qwen/qwen3.8-omni-flash` but neither Qwen3-Omni nor Qwen3.5-Omni.
- **Gemini thinking tokens** are billed as output: keep the thinking budget at its minimum, or the output cost grows far above the 5 tokens assumed below.

## 3. Audio volume and API cost

### 3a Audio volume of the 13 datasets (plan B)

Measured from the file headers (`soundfile.info`) of the clips the loaders return (`LOADERS[key]().clips`), which already apply the plan-B subsets of `config/evaluation.yaml`. Every model, API models included, runs all of these clips; ESD is never evaluated or sent.

| Dataset | Clips | Hours | Mean s per clip | Plan B |
|---|---|---|---|---|
| CREMA-D | 7,442 | 5.26 | 2.54 | all clips |
| MCAE-SPPS | 6,675 | 2.30 | 1.24 | all clips (1,797 are under 1 s) |
| MESS | 1,800 | 1.33 | 2.66 | all clips |
| RAVDESS | 1,440 | 1.48 | 3.70 | all clips |
| TESS | 2,800 | 1.60 | 2.06 | all clips |
| JL-Corpus | 1,200 | 0.70 | 2.09 | the 5 primary emotions (1,200 of 2,400 clips) |
| THAI-SER | 11,848 | 17.36 | 5.28 | only clips with agreement ≥ 0.71, the authors' recommended subset (11,848 of 22,584; the full set is 33.19 h) |
| nEMO | 4,481 | 3.07 | 2.47 | all clips |
| MEAD (without contempt) | 8,000 | 9.51 | 4.28 | stratified 8,000 of 27,514 clips (even intervals over actor, emotion, intensity and sentence) |
| MELD (test) | 2,608 | 2.31 | 3.18 | the test split (2,610 utterances) less 2 clips with wrong time stamps |
| EMOVO | 588 | 0.51 | 3.12 | all clips |
| Emozionalmente | 6,902 | 6.30 | 3.29 | all clips |
| SAVEE | 480 | 0.51 | 3.84 | all clips (added in plan B) |
| **Total** | **56,264** | **52.24** | 3.34 | |

THAI-SER and MEAD are 51% of the audio. A 200-clip-per-dataset pilot (`sample(ds, 200)`, 2,600 clips) would be 2.20 h.

### 3b Cost formula

For one model and one set of clips:

cost = 1.02 × (A × r × P_audio + N × 175 × P_text_in + N × 5 × P_text_out) / 1,000,000

- A = seconds of audio (for Qwen, clips under 1 s counted as 1 s); r = audio tokens per second (table in section 2); N = number of clips.
- 175 text input tokens per clip: the system prompt is 119 tokens and the user prompt 42 (Qwen2.5-Omni tokenizer, labels filled in), plus about 14 for the chat template. Other tokenizers differ by a few tokens.
- 5 output tokens per clip: a label is 1–3 tokens, plus the end token.
- 1.02: about 2% of clips asked again (`retries_invalid` and failed requests).

Worked example, Qwen3.5-Omni Plus (international), plan B: audio 52.36 h (with the under-1 s clips counted as 1 s) × 3,600 × 7 = 1.32M tokens × $11 = $14.51; text input 56,264 × 175 = 9.85M tokens × $1.4 = $13.78; output 56,264 × 5 = 0.28M tokens × $8.3 = $2.33; total $30.63 × 1.02 = **$31.25**.

### 3c Estimated cost

| Model | Plan B, every clip (56,264 clips, 52.2 h) | Tokens (audio / all) |
|---|---|---|
| qwen3-omni-flash, international | $14.35 | 2.36M / 12.48M |
| qwen3-omni-flash, mainland | $8.56 | as above |
| qwen3.5-omni-plus, international | $31.25 | 1.32M / 11.45M |
| qwen3.5-omni-plus, mainland | $21.03 | as above |
| qwen3.5-omni-flash, international | $8.69 | 1.32M / 11.45M |
| qwen3.5-omni-flash, mainland | $6.88 | as above |
| qwen3.8-omni-flash, international | $1.84 | 1.32M / 11.45M |
| qwen3.8-omni-flash, mainland and other regions | $1.40 | as above |
| gpt-realtime (reference) | $106.15 | 1.88M / 12.01M |
| gpt-realtime-mini (reference) | $25.90 | as above |
| gemini-2.5-flash (reference, thinking off) | $9.87 | 6.02M / 16.15M |
| gemini-3-flash-preview (reference, thinking off) | $12.02 | as above |
| gemini-3.5-flash (reference, thinking off) | $13.21 | as above |

Token counts are before the 2% of retries. All estimates; the real bill also depends on prompt caching (Qwen3.8 caches automatically; repeated text input then costs $0.016/M instead of $0.15/M) and on how many answers need a follow-up. For every Qwen model the text prompt is 80–90% of the input tokens and 30% (qwen3-omni-flash) to 82% (qwen3.8-omni-flash) of the cost; about 45% for Qwen3.5-Omni.
A 200-clip-per-dataset pilot (2,600 clips, 2.20 h) would cost $1.38 with Qwen3.5-Omni Plus, $0.38 with Flash, $0.63 with qwen3-omni-flash, $0.08 with qwen3.8-omni-flash and $4.65 with GPT Realtime (international prices).
Run time: GPT Realtime, one request at a time, measured 5.5 s per clip, so the plan-B set takes about 86 h (3.6 days). The Qwen API is not measured; at the Beijing limit of 60 requests per minute a model needs at least 15.6 h, and one request at a time at GPT Realtime's speed would also take about 86 h, so the loader should send a few requests in parallel.
Step-Audio 2 is not costed: StepFun does not publish its audio token rate; if its audio rate is like Qwen's (7–12.5 tokens per second), a plan-B run is 11.5–12.5M input tokens × $1.43 plus 0.28M output tokens × $10, about $20 (estimate).

## 4. Self-hosting: hardware and integration

### 4a Hardware table (open models, including the three being added)

bf16 memory = parameters × 2 bytes (weights only); short clips (1–5 s, under 300 tokens with the prompt) add about 1–3 GB of activations and almost no KV cache. "Fits" means weights plus that margin; "s/clip" is the order of magnitude for one clip at a time with transformers, ≤ 5 output tokens: all **estimates, not measured**.

| Model | Parameters loaded | bf16 memory | L4 24 GB | L40 48 GB | H100 80 GB | Mac M4 Max 48 GB | s/clip (estimate) | Integration |
|---|---|---|---|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct (being added) | 8.40B | 16.8 GB | yes (tight) | yes | yes | yes (MPS) | L40 0.2–0.5; H100 0.1–0.3; Mac 1–3 | transformers native (`qwen2_audio` loader) |
| Qwen2.5-Omni-7B (being added) | Thinker about 8.9B (repo 10.73B with talker) | about 17.9 GB | yes (tight) | yes | yes | yes (MPS) | as above | transformers native (`qwen2_5_omni` loader, talker not loaded) |
| Gemma 4 E4B-it (being added) | 8.00B with embeddings (4.5B effective) | 16.0 GB | yes | yes | yes | yes (MPS) | slightly faster than the 7B models | transformers ≥ 5.5 native (`gemma4_audio` loader) |
| Qwen3-Omni-30B-A3B-Instruct | Thinker 31.72B (talker and code2wav, 3.54B, not loaded) | 63.4 GB | no (a 4-bit checkpoint might, unverified) | **FP8 or 4-bit only** (community checkpoints, see 4b) | **yes, 1 card** (about 15 GB left) | no in bf16 | transformers: H100 0.5–2 (MoE overhead); vLLM batched: about 0.02–0.1 per clip throughput | transformers native (`Qwen3OmniMoeForConditionalGeneration`, in 5.18.0) or vLLM (Thinker); close to the `qwen2_5_omni` loader |
| Kimi-Audio-7B-Instruct | 9.77B (main safetensors) + Whisper-large-v3 folder (3.1 GB); detokenizer 19.0 GB and vocoder 1.0 GB not needed for text output (`load_detokenizer=False`) | about 20–24 GB (estimate) | borderline, unverified | yes | yes | no (CUDA only) | L40 0.3–0.8; H100 0.2–0.5 | its own repo (`KimiAudio` class): torch 2.6.0, flash_attn 2.7.4.post1, deepspeed; separate environment or Docker |
| Step-Audio 2 mini | 8.32B (+ token2wav 1.2 GB, not needed) | 16.6 GB | yes (tight) | yes | yes | no without patching `.cuda()` | L40 0.2–0.5 | its own repo (`StepAudio2` in `examples.py`): transformers==4.49.0, `.cuda()` hard-coded; separate environment; a vLLM Docker image exists |
| Audio Flamingo 3 | 8.27B | 16.5 GB | yes | yes | yes | probably (MPS; unverified) | L40 0.2–0.5; Mac 1–3 | transformers native (`AudioFlamingo3ForConditionalGeneration`, transformers ≥ 5.0; works with 5.18.0) |
| Audio Flamingo Next | 8.27B | 16.5 GB | yes | yes | yes | probably (unverified) | as AF3 | transformers (`AutoModel`) |
| MiDashengLM-7B | 8.28B | 16.6 GB | yes | yes | yes | unverified | as above | transformers `trust_remote_code`; official FP8 and GPTQ versions exist |
| Voxtral Mini 3B / Small 24B | 4.68B / 24.26B | 9.4 / 48.5 GB | Mini yes / Small no | Mini yes / Small no (just over 48 GB) | both | Mini yes / Small no | Mini 0.1–0.3 | transformers native |

Time for the whole benchmark: at 0.3 s per clip, the 56,264 plan-B clips take about 4.7 h per 7–9B model on the L40, and about 16–47 h on the Mac at 1–3 s per clip (estimates). The 4×H100 are not needed for any single model; they help only to run four models (or four Qwen3-Omni copies) in parallel.

### 4b Qwen3-Omni in detail

- **Parameters** (counted from the safetensors headers of the official repository): Thinker 31.72B = language model 30.22B + audio encoder (AuT) 0.65B + vision encoder 0.54B + output layer 0.31B; talker 3.18B (with its projections), its code predictor 0.14B and code2wav 0.22B (3.54B together); total 35.26B. In transformers the vision encoder is loaded even for audio-only use.
- **Memory**: the card gives a minimum of 78.85 GB in bf16 for a 15 s video with the whole model, and says `model.disable_talker()` saves about 10 GB ([card](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)). For short audio clips with the talker off: 63.4 GB of weights plus a few GB (estimate), so one H100 80 GB is enough but has little room; 2×H100 with tensor parallelism is comfortable.
- **KV cache**: 48 layers × 4 KV heads × 128 dimensions × 2 (K and V) × 2 bytes = 96 KiB per token (config.json). A clip with prompt is about 200–300 tokens (12.5 audio tokens per second), i.e. about 20–30 MB; even THAI-SER's longest clip (91 s, about 1,140 audio tokens) is about 0.13 GB. KV cache is not the limit.
- **Quantised checkpoints** (no official FP8, AWQ or GPTQ version from Qwen; the official account has only the Instruct, Thinking and Captioner repositories, checked 2026-11-05). Community ones, with file sizes that include the talker:
  [cyankiwi AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit) 27.6 GB (142,949 downloads, made with llm-compressor), [cyankiwi AWQ-8bit](https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit) 42.5 GB, [sammysun0711 FP8-Dynamic](https://huggingface.co/sammysun0711/Qwen3-Omni-30B-A3B-Instruct-FP8-Dynamic) 37.3 GB, [Intel int4-AutoRound](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) 26.3 GB, [thomasip GPTQ-4bit](https://huggingface.co/thomasip/Qwen3-Omni-30B-A3B-Instruct-GPTQ-4bit).
  On the L40 (Ada, which has FP8 support) the FP8 or a 4-bit checkpoint fits with room for batching. A quantised model is not the reference model: report it as "Qwen3-Omni (FP8, community)" and, if the H100 becomes available, check agreement on a small dataset in bf16.
- **vLLM**: mainline vLLM has the Thinker (`Qwen3OmniMoeThinkerForConditionalGeneration`, text output; [vLLM docs](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_omni_moe_thinker/)); [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/stable/models/supported_models/) serves the whole model including speech output. The card suggests `max_num_seqs=8` and `tensor_parallel_size` = number of GPUs. With vLLM and batching, the Thinker (about 3B active parameters) should process many clips per second on an H100 (estimate, not measured).

## 5. Notes per candidate

- **Qwen3.5-Omni**: Plus and Flash, both instruct models with 256k context; Thinker and talker are hybrid-attention MoE ([report](https://arxiv.org/abs/2604.15804), v2 2026-04-21). No weights: no Qwen3.5-Omni repository on Hugging Face (search of 2026-11-05 found only unrelated community models with "Qwen3.5-Omni" in the name). There are also `qwen3.5-omni-flash-realtime` ($0.55 text in, $4.5 audio in, $3.3 text out, international) and Plus realtime versions; the non-realtime API is the simpler one here.
- **qwen3.8-omni-flash** (not in the original list): the newest Qwen omni API, released in September 2026 (launch 2026-09-14 according to press reports, unverified; on OpenRouter since 2026-09-21); one input price of $0.15/M for every modality, non-streaming allowed, 1M context, automatic caching. API only. With $1.84 for the whole plan-B set it costs almost nothing to add as a third Qwen API point; no SER figures found.
- **Kimi-Audio**: Whisper-large-v3 features plus discrete semantic tokens at 12.5 Hz, LLM initialised from Qwen2.5-7B. MELD 59.13 against Qwen2.5-Omni 49.83 in its own evaluation, while the Qwen2.5-Omni card reports 57.0: the two protocols differ. Its SFT data list IEMOCAP 10 h, MELD 9 h, RAVDESS 3 h, SAVEE 0.1 h and ESD 29 h ([paper](https://arxiv.org/abs/2504.18425)). The 42.6 GB repository holds the LLM (19.5 GB), the detokenizer (19.0 GB), Whisper (3.1 GB) and the vocoder (1.0 GB). Integration: a separate CUDA environment, the `KimiAudio` class with `output_type="text"`.
- **Step-Audio 2 mini**: 8.32B, Apache-2.0. StepEval-Audio-Paralinguistic has 550 clips over 11 tasks, 50 per task; the emotion clips are Chinese podcast recordings with open-set answers judged by an LLM ([paper](https://arxiv.org/abs/2507.16632), section 4.2): emotion Step-Audio 2 86, mini 82, GPT-4o Audio 82, Qwen-Omni 76, Kimi-Audio 66. Not comparable with a 7-class test. Requirements: Python ≥ 3.10, PyTorch ≥ 2.3 with CUDA, transformers==4.49.0 ([GitHub](https://github.com/stepfun-ai/Step-Audio2)). The larger Step-Audio 2 is reachable only through StepFun's realtime API (`step-audio-2`: $1.43/M input, $10/M output, [pricing](https://platform.stepfun.ai/docs/en/guides/pricing/details)), which would need a WebSocket loader like GPT Realtime's.
- **Audio Flamingo 3**: the card's training list includes MSP-Podcast, JL-Corpus, MELD, TESS, OMGEmotion and EmoV-DB ([card](https://huggingface.co/nvidia/audio-flamingo-3-hf)); the paper's table 11 gives JL-Corpus 1.4 h / 2.4K pairs (the whole corpus is 2,400 clips) trained for 6 epochs, TESS 1.6 h / 2.8K (the whole corpus is 2,800 clips), MELD 8.7 h / 32.9K and MSP-Podcast 73.9 h; table 9 adds long concatenated MELD and IEMOCAP recordings ([paper](https://arxiv.org/abs/2507.08128)). The licence (NVIDIA OneWay Noncommercial) allows non-commercial research only. The repository keeps the weights twice (sharded and as one file, 16.5 GB each, both bf16) plus a 1.4 GB "think" adapter: download only one copy. [VoxEmo](https://arxiv.org/abs/2603.08936) has already evaluated AF3 and Qwen2-Audio zero-shot on 35 corpora, including THAI-SER, nEMO, MEAD and CREMA-D: a published reference to compare against.
- **Audio Flamingo Next**: same size and licence as AF3; adds longer audio (up to 30 minutes) and timestamps; its paper does not name emotion datasets, but its data extend AF3's (AF-Skills), so treat the same datasets as possibly seen.
- **MiDashengLM-7B**: Dasheng audio encoder (not Whisper), trained on general audio captions; its card reports speaker (VoxCeleb1 92.36), language, gender and sound classification, but no emotion results.
- **Voxtral**: built for transcription, translation and spoken QA; no emotion results found. Voxtral Small (48.5 GB in bf16) does not fit the L40.

## 6. Training-data overlap with this project's datasets

A model that saw a dataset in training is not tested zero-shot on it. The table lists what the papers and cards say; "not published" does not mean no overlap.

| Model | Seen in training (our datasets in bold) | Source |
|---|---|---|
| Audio Flamingo 3 | **JL-Corpus** (all 2,400 clips), **TESS** (all 2,800), **MELD** (training data; whether the test set was used is unverified), MSP-Podcast, EmoV-DB, OMGEmotion, IEMOCAP (long-audio data) | card; paper tables 9 and 11 |
| Audio Flamingo Next | probably as AF3 (unverified) | paper (data extend AF3's) |
| Kimi-Audio | **RAVDESS** (3 h, more than the 1.5 h of speech clips: probably the song clips too, or repeats), **MELD** (9 h), **SAVEE** (0.1 h), ESD (excluded here), IEMOCAP | paper, SFT data table |
| Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Qwen3.5-Omni | not published; they report MELD results, so MELD's training split was probably used (unverified) | papers |
| Step-Audio 2, Gemma 4, MiDashengLM, Voxtral | not published | — |
| For reference, models already in the benchmark | emotion2vec: **MEAD** and **MELD** in its (unlabelled) pre-training; MERaLiON-SER: **CREMA-D**, **MELD**; Melati: **Emozionalmente**, **JL-Corpus** | [models_survey_en.md](models_survey_en.md), [datasets_survey_en.md](datasets_survey_en.md) |

So the reports should mark AF3's JL-Corpus and TESS results "seen" and its MELD results "possibly seen", and Kimi-Audio's RAVDESS, MELD and SAVEE results "seen". CREMA-D, RAVDESS (for AF3), MCAE-SPPS, MESS, THAI-SER, nEMO, MEAD, EMOVO, Emozionalmente and SAVEE remain unseen for AF3.

## 7. Recommendation

1. **Qwen3-Omni-30B-A3B-Instruct, open weights, next.** The strongest open Qwen and the natural step after Qwen2.5-Omni; Apache-2.0; loads with the transformers already pinned (5.18.0), so the loader is close to `qwen2_5_omni.py` (talker off, Thinker only).
   Hardware: 1×H100 80 GB in bf16 if available (2×H100 for comfort and speed). Otherwise the L40 with the community FP8 checkpoint (or AWQ-4bit), reported as quantised. Not on the Mac or the L4 in bf16.
2. **Qwen3.5-Omni Plus and Flash through the API, on every plan-B clip** (56,264 clips, 52.2 h): $31.25 + $8.69 = about $40 (international); the 1M-token free quota of Plus covers about $2.7 of it. They are the newest Qwen omni models and are API only, so this is the only way to include them. Needs a new streaming loader for Model Studio's OpenAI-compatible API (`stream=True`) and an international (Singapore) key. Adding `qwen3.8-omni-flash` in the same loader costs $1.84 more. A 2,600-clip pilot first (Plus $1.38, within the free quota; Flash $0.38) checks the loader before the full run. Time: at 60 requests per minute, at least 15.6 h per model; send a few requests in parallel (one at a time at GPT Realtime's 5.5 s per clip would be about 86 h). `qwen3-omni-flash` ($14.35) is optional: it is a closed variant, and the open weights of step 1 are the better reference.
3. **Audio Flamingo 3**: the least work (transformers native, runs on the L40 and probably the Mac), non-commercial licence acceptable here. Report JL-Corpus and TESS as "seen" and MELD as "possibly seen", or leave them out of its averages.
4. **Step-Audio 2 mini on the L40**: a separate environment (transformers 4.49.0) and patches for `.cuda()`; brings a non-Qwen, non-NVIDIA model with strong paralinguistic claims.
5. **Kimi-Audio on the L40**: the most integration work (separate CUDA environment with flash-attn and torch 2.6); mark RAVDESS, MELD and SAVEE as "seen".
- Not now: Audio Flamingo Next (only if AF3 is informative), MiDashengLM and Voxtral (no emotion focus or results), the Step-Audio 2 realtime API (WebSocket loader for one model).
- Hardware summary: the L40 covers every 7–9B model in bf16 and Qwen3-Omni quantised; an H100 is needed only for Qwen3-Omni in bf16; the L4 can run the 7–9B models one at a time (Kimi-Audio borderline); the 4×H100 are only useful to run models in parallel.

## 8. Main sources

- Alibaba Cloud Model Studio: pricing https://www.alibabacloud.com/help/en/model-studio/model-pricing ; Qwen-Omni (token rules, streaming, limits) https://www.alibabacloud.com/help/en/model-studio/qwen-omni ; qwen3-omni-flash page https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-omni-flash ; Qwen3-Omni-Flash-2025-12-01 https://www.alibabacloud.com/blog/602735
- Qwen3-Omni: https://arxiv.org/abs/2509.17765 , https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct , https://github.com/QwenLM/Qwen3-Omni ; vLLM: https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_omni_moe_thinker/ , https://docs.vllm.ai/projects/vllm-omni/en/stable/models/supported_models/
- Qwen3.5-Omni: https://arxiv.org/abs/2604.15804
- OpenRouter model list: https://openrouter.ai/api/v1/models
- OpenAI: https://developers.openai.com/api/docs/pricing , https://developers.openai.com/api/docs/guides/realtime-costs ; Gemini: https://aistudio.google.com/docs/pricing , https://aistudio.google.com/docs/audio ; StepFun: https://platform.stepfun.ai/docs/en/guides/pricing/details
- Kimi-Audio: https://arxiv.org/abs/2504.18425 , https://github.com/MoonshotAI/Kimi-Audio , https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
- Step-Audio 2: https://arxiv.org/abs/2507.16632 , https://github.com/stepfun-ai/Step-Audio2 , https://huggingface.co/stepfun-ai/Step-Audio-2-mini
- Audio Flamingo 3 and Next: https://arxiv.org/abs/2507.08128 , https://huggingface.co/nvidia/audio-flamingo-3-hf , https://arxiv.org/abs/2604.10905 , https://huggingface.co/nvidia/audio-flamingo-next-hf
- MiDashengLM: https://arxiv.org/abs/2508.03983 , https://huggingface.co/mispeech/midashenglm-7b-1021-bf16 ; Voxtral: https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 , https://huggingface.co/mistralai/Voxtral-Small-24B-2507
- Dataset figures: EmoBox https://arxiv.org/abs/2406.07162 ; MELD https://github.com/declare-lab/MELD ; VoxEmo https://arxiv.org/abs/2603.08936
- Query endpoints: Hugging Face `https://huggingface.co/api/models/<id>?blobs=true` and safetensors headers via HTTP range requests

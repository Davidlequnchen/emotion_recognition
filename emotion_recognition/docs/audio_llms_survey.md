# 音频大模型调研：下一步加哪些（Audio LLM Survey）

英文版：[audio_llms_survey_en.md](audio_llms_survey_en.md)（内容相同）。专用 SER 模型的调研见 [models_survey.md](models_survey.md)，数据集调研见 [datasets_survey.md](datasets_survey.md)。

**2026-11-06 按方案 B 更新**：13 个数据集（新加 SAVEE；ESD 不在范围内），THAI-SER 只用一致率 ≥ 0.71 的片段，MEAD 分层取 8,000 条，所有模型都一样；API 模型现在跑每个数据集的全部片段（不再抽 200 条）。下面的音频量、费用和运行时间都按方案 B 算：56,264 条、52.2 小时，全部实测。

**2026-11-13 更新**：基准现在有 16 个数据集。三个对话数据集 EmotionTalk（官方 test，1,929 条）、DailyTalk（validation，1,197 条）和 MultiDialog（`gold_both`，7,672 条）加入后共 67,062 条。GPT Realtime（`gpt-realtime`）、Gemini 3.8 Flash（`gemini-3.8-flash`）和 Qwen2.5-Omni 7B 已有全部 16 个数据集的结果；Qwen2-Audio 7B Instruct 正在跑，Gemma 4 E4B 会重跑（进度和数字见 [reports/summary.md](../reports/summary.md)）。GPT Realtime 和 Gemini 已经跑完，所以删去了它们的价格行和费用估算。下面的音频量和费用仍只算方案 B 的 13 个数据集；候选模型如果加跑这 10,798 条对话片段，费用会增加：API 按音频时长计费，单 MultiDialog 就有约 13.1 小时（方案 B 共 52.2 小时），所以增幅会超过 19% 的条数比例，具体没有重新估算。

调研日期：2026-11-05。目标：在 5 个 SER 模型、GPT Realtime、Gemini 和正在接入的 3 个本地音频大模型（Qwen2.5-Omni-7B、Qwen2-Audio-7B-Instruct、Gemma 4 E4B-it）之后，决定本项目的基准测试下一步加哪些音频大模型：走 API 要花多少钱，在这里跑需要什么硬件，接入有多难。
前提：只做学术研究，所以非商用许可可以接受（标出来，不排除）。硬件：Mac M4 Max 48 GB；确定有 NVIDIA L40（48 GB）；可能有 NVIDIA L4（24 GB）和 4×H100 80 GB。

方法：读了阿里云百炼（Model Studio）、OpenAI、Google、阶跃星辰（StepFun）的价格和计费页面，OpenRouter 的模型列表，模型卡片、论文和 GitHub 代码；参数量用 HTTP range 请求读 safetensors 文件头来数（没有下载权重）；13 个数据集都只读文件头量时长（`soundfile.info`，片段列表来自 `scripts/common/eval_datasets.py` 的 `LOADERS[key]().clips`，已经套用了 `config/evaluation.yaml` 里方案 B 的子集）；`scripts/llm_eval/prompts.yaml` 的提示词 token 数用 Qwen2.5-Omni 的分词器数。
单位：M = 百万、B = billion 参数；大小以十进制 GB 计；价格是美元每百万 token（$/M）。标“估算”的是这里算出来的数，不是从来源读到的；标“未核实”的没能从一手资料确认。价格都是 2026-11-05 页面上列的价格。

## 0. 简要结论

- **音频总量（方案 B）**：13 个数据集，THAI-SER 只用一致率 ≥ 0.71 的片段（11,848 条），MEAD 分层取 8,000 条：共 **56,264 条、52.2 小时**，全部读文件头实测。API 模型跑这里的每一条；ESD 从不发送。
- **API 费用很低**（第 3 节）。方案 B 全部片段：Qwen3.5-Omni Plus **$31.25**，Qwen3.5-Omni Flash **$8.69**，qwen3-omni-flash **$14.35**，新出的 qwen3.8-omni-flash $1.84（国际价；中国内地便宜 20–40%）。每个数据集 200 条的试跑（2,600 条）用 Qwen 模型只要 $0.08–1.38。
  固定的文本提示词（约 175 个 token）在每个 Qwen 模型的输入 token 里占 80–90%（2.5 秒的片段只有 18–31 个音频 token），在费用里占 30–82%：缩短提示词是最主要的省钱办法。
- **运行时间**：一次一个请求很慢（GPT Realtime 每条约 5.5 秒），所以新的 API 加载器应同时发几个请求，但不能超过各家 API 的限流。
- **Qwen3-Omni-30B-A3B-Instruct**（开源，Apache-2.0）：API 上的 `qwen3-omni-flash` 是闭源、再训练过的 Flash 版本，不是开源权重。自己部署时只用 Thinker（关掉 talker）是 **31.72B 参数，bf16 下 63.4 GB**：**1×H100 80 GB** 放得下（偏紧），2×H100 宽裕；**L40 只能用社区的 FP8 或 4-bit 版本**（官方没有量化版）。本项目固定的 transformers 5.18.0 原生支持；vLLM 支持 Thinker。
- **Qwen3.5-Omni**（2026-03-30，arXiv 2604.15804）：**只有 API**，`qwen3.5-omni-plus` 和 `qwen3.5-omni-flash`；Hugging Face 上没有 Qwen3.5-Omni 的仓库（2026-11-05 查过；有些博客说的“开源 Light 版”没找到）。音频按每秒 7 个 token 计费。技术报告里没有情绪识别结果。
- **Kimi-Audio-7B-Instruct**（MIT，9.77B）：L40 和 H100 能跑，Mac 不行（CUDA、flash-attn、torch 2.6）。它自报的 MELD 59.13 是这些候选里最高的 MELD 数字，但**它的 SFT 数据里有 RAVDESS、MELD 和 SAVEE**，正是我们的三个数据集。
- **Step-Audio 2 mini**（Apache-2.0，8.32B）：L4/L40/H100 都能跑；要单独的环境（transformers==4.49.0，写死了 `.cuda()`）。它的情绪分数（82）来自自家 50 条中文播客的测试、由大模型判分；没有 MELD 或 IEMOCAP 结果。大一些的 Step-Audio 2 只有 API（`step-audio-2`，实时 API）。
- **Audio Flamingo 3**（8.27B，NVIDIA 非商用）：最容易接入（transformers 原生，所有 GPU 都放得下，Mac 多半也行），但**训练数据包含全部 JL-Corpus、全部 TESS 和 MELD 的训练数据**（论文表 11）。它在 JL-Corpus 和 TESS 上的结果必须标“训练中见过”，MELD 标“可能见过”。
- **建议顺序**（第 7 节）：(1) Qwen3-Omni 开源权重，1×H100（或 L40 + FP8）；(2) Qwen3.5-Omni Plus 和 Flash 走 API 跑方案 B 的全部片段（两个合计约 $40：$31.25 + $8.69）；(3) Audio Flamingo 3，带“见过”标记；(4) Step-Audio 2 mini；(5) Kimi-Audio。MiDashengLM 和 Voxtral 对 SER 不值得加。

## 1. 候选一览

各行数字来自不同的评测方法，不能跨行比较；“重叠”指模型公开的训练数据和本项目 2026-11-05 时 13 个数据集的重叠（第 6 节；三个对话数据集没有针对这些候选模型核查）。

| 模型 | 机构、发布 | 大小 | 许可 | 权重 / API | 公开的 SER 结果 | 和我们 13 个数据集的重叠（2026-11-05） | 结论 |
|---|---|---|---|---|---|---|---|
| Qwen3-Omni-30B-A3B-Instruct | 阿里 Qwen，2025-09 | 共 35.26B；Thinker 31.72B（MoE，每个 token 约激活 3B） | Apache-2.0 | 开源（[HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)）；API `qwen3-omni-flash` 是闭源 Flash 版本 | [技术报告](https://arxiv.org/abs/2509.17765)里没有；第三方 MELD 56.8（MiniCPM-o 4.5 卡片，见 [models_survey.md](models_survey.md)） | 训练数据未公开 | **最先加**（开源里最强的 Qwen） |
| Qwen3.5-Omni Plus / Flash | 阿里 Qwen，2026-03-30 | 未公开（报告只说 “hundreds of billions”，即数百 B 参数，MoE；[报告](https://arxiv.org/abs/2604.15804)） | 闭源 | 只有 API | 报告里没有 | 未公开 | **走 API 加** |
| qwen3.8-omni-flash | 阿里 Qwen，2026-09（2026-09-21 起上架 OpenRouter） | 未公开 | 闭源 | 只有 API | 没找到 | 未公开 | 可选，最便宜的 API |
| Kimi-Audio-7B-Instruct | 月之暗面，2025-04 | 9.77B（safetensors） | MIT（来自 Qwen2.5-7B 的部分：Apache-2.0） | 开源（[HF](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct)） | MELD 59.13（自报，[GitHub](https://github.com/MoonshotAI/Kimi-Audio)） | SFT 里有 **RAVDESS、MELD、SAVEE**（[论文](https://arxiv.org/abs/2504.18425)） | 以后加；标出 RAVDESS、MELD、SAVEE |
| Step-Audio 2 mini | 阶跃星辰，2025-08 | 8.32B | Apache-2.0 | 开源（[HF](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)）；大的 Step-Audio 2 只有 API | 自家 StepEval-Audio-Paralinguistic 的情绪项 82（50 条中文播客，开放式回答，大模型判分；[论文](https://arxiv.org/abs/2507.16632)表 8） | 未公开 | 以后加 |
| Audio Flamingo 3 | NVIDIA、马里兰大学，2025-07 | 8.27B | NVIDIA OneWay Noncommercial | 开源（[HF](https://huggingface.co/nvidia/audio-flamingo-3-hf)） | IEMOCAP 63.8%（[论文](https://arxiv.org/abs/2507.08128)） | **JL-Corpus、TESS（全部片段）、MELD** | 加，带“见过”标记 |
| Audio Flamingo Next | NVIDIA、马里兰大学，2026-04 | 8.27B | NVIDIA OneWay Noncommercial | 开源（[HF](https://huggingface.co/nvidia/audio-flamingo-next-hf)） | 没找到 | 在 AF3 的数据（AF-Skills）上扩充；多半有重叠，未核实 | AF3 有用再加 |
| MiDashengLM-7B（1021） | 小米，2025-10 | 8.28B | Apache-2.0 | 开源（[HF](https://huggingface.co/mispeech/midashenglm-7b-1021-bf16)） | 没有（卡片只报说话人、语种、声音分类） | 未公开 | 暂不加 |
| Voxtral Mini / Small | Mistral，2025-07 | 4.68B / 24.26B | Apache-2.0 | 开源（[Mini](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)、[Small](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)）；Small 也在 Mistral API 和 OpenRouter 上 | 没找到（为转写和语音问答设计） | 未公开 | 不加 |

## 2. API 模型：名称、价格和音频 token 换算

价格是“音频输入、文本输出”（不要音频输出）时的 $/M。“文本输出”指输入含音频时的文本输出价。

| 模型名（API） | 地域 | 音频输入 | 文本输入 | 文本输出 | 每秒音频的 token 数 | 来源 |
|---|---|---|---|---|---|---|
| `qwen3-omni-flash` | 国际（新加坡） | 3.81 | 0.43 | 3.06（纯文本输入时 1.66） | 12.5 | [百炼价格](https://www.alibabacloud.com/help/en/model-studio/model-pricing)；[token 规则](https://www.alibabacloud.com/help/en/model-studio/qwen-omni) |
| `qwen3-omni-flash` | 中国内地（北京） | 2.265 | 0.258 | 1.821 | 12.5 | 同上 |
| `qwen3.5-omni-plus`（= `qwen3.5-omni-plus-2026-03-15`） | 国际 | 11 | 1.4 | 8.3 | 7 | 同上 |
| `qwen3.5-omni-plus` | 中国内地 | 7.29 | 0.96 | 5.5 | 7 | 同上 |
| `qwen3.5-omni-flash`（= `qwen3.5-omni-flash-2026-03-15`） | 国际 | 3 | 0.4 | 2.2 | 7 | 同上 |
| `qwen3.5-omni-flash` | 中国内地 | 2.48 | 0.3 | 1.83 | 7 | 同上 |
| `qwen3.8-omni-flash` | 国际 | 0.15（所有输入同价） | 0.15 | 0.47 | 7 | 同上；[OpenRouter](https://openrouter.ai/qwen/qwen3.8-omni-flash) 上 `qwen/qwen3.8-omni-flash` 同价 |
| `qwen3.8-omni-flash` | 中国内地、香港、东京、法兰克福、美国（弗吉尼亚） | 0.113 | 0.113 | 0.382 | 7 | 同上 |
| `step-audio-2`（阶跃星辰，实时 API） | 国际平台 | 1.43（输入，未命中缓存） | 1.43 | 10.00（输出） | 未公开 | [阶跃星辰价格](https://platform.stepfun.ai/docs/en/guides/pricing/details) |

GPT Realtime 和 Gemini 的参考行已于 2026-11-13 删去：两个模型都已评测。

说明：
- **Qwen 的 token 规则**（百炼“Billing and rate limits”）：Qwen3.8-Omni-Flash 和 Qwen3.5-Omni 系列的输入音频按时长（秒）× 7 计；Qwen3-Omni-Flash 按时长（秒）× 12.5 计；不到 1 秒按 1 秒算（方案 B 有 2,508 条不到 1 秒：MCAE-SPPS 1,797 条、THAI-SER 468 条、MELD 225 条、其他数据集 18 条；下面的费用已算进去）。更长的片段按实际时长计，不是每开始一秒就算一整秒。这和编码器一致：Qwen3-Omni 的 AuT 是 12.5 Hz（[论文](https://arxiv.org/abs/2509.17765)），Qwen3.5-Omni 的编码器是 6.25 Hz（[论文](https://arxiv.org/abs/2604.15804)）。
- **接入点**：国际（新加坡）和中国内地（北京）的工作空间要分别申请 API key；OpenAI 兼容的 base URL 按工作空间区分，新加坡形如 `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1`。`qwen3.8-omni-flash` 还在香港、东京、法兰克福、弗吉尼亚有；其余只在新加坡和北京。
- **只支持流式输出**：Qwen3.5-Omni 和 Qwen3-Omni-Flash 只接受 `stream=True`（非流式请求会报错）；qwen3.8-omni-flash 两种都行（[文档](https://www.alibabacloud.com/help/en/model-studio/qwen-omni)）。加载器要把流式返回的文本拼起来。
- **音频输入限制**：base64 数据小于 10 MB；Qwen3-Omni-Flash 最长 20 分钟音频，Qwen3.5-Omni 和 Qwen3.8 最长 3 小时。我们最长的片段是 91 秒（THAI-SER）。
- **免费额度**：新加坡的 `qwen3.5-omni-plus` 送 100 万 token，有效期 90 天（从开通百炼、模型发布或申请通过中最晚的一天算起）；带日期的快照、`qwen3.5-omni-flash` 和 `qwen3-omni-flash` 都没有（价格页）。Plus 跑方案 B 约 1,170 万 token，免费额度只够不到十分之一（约 $2.7）；2,600 条的试跑（约 53 万 token）能放进免费额度。
- **限流**：北京的 `qwen3-omni-flash` 是每分钟 60 次请求、10 万 token（[模型页](https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-omni-flash)）；按每分钟 60 次，方案 B 的 56,264 条至少要 15.6 小时（每条约 220 个 token，碰不到 token 限额）。国际版的限额未核实。
- **qwen3-omni-flash 不是开源模型**：Qwen3-Omni-Flash-2025-12-01 是“在 Qwen3-Omni 基础上全面升级的版本”（[阿里云博客](https://www.alibabacloud.com/blog/602735)）。API 的结果不能代表开源权重，API 版本还可能不打招呼就更新；能复现的是开源权重。
- **OpenRouter**（2026-11-05 读的模型列表）：有 `qwen/qwen3.8-omni-flash`，没有 Qwen3-Omni 和 Qwen3.5-Omni。

## 3. 音频总量和 API 费用

### 3a 13 个数据集的音频量（方案 B）

按加载器返回的片段（`LOADERS[key]().clips`，已经套用了 `config/evaluation.yaml` 里方案 B 的子集）读文件头（`soundfile.info`）实测。所有模型，包括 API 模型，都跑这里的全部片段；ESD 从不评测，也从不发送。

| 数据集 | 片段数 | 小时 | 平均每条秒数 | 方案 B |
|---|---|---|---|---|
| CREMA-D | 7,442 | 5.26 | 2.54 | 全部片段 |
| MCAE-SPPS | 6,675 | 2.30 | 1.24 | 全部片段（其中 1,797 条不到 1 秒） |
| MESS | 1,800 | 1.33 | 2.66 | 全部片段 |
| RAVDESS | 1,440 | 1.48 | 3.70 | 全部片段 |
| TESS | 2,800 | 1.60 | 2.06 | 全部片段 |
| JL-Corpus | 1,200 | 0.70 | 2.09 | 5 种主要情绪（2,400 条里的 1,200 条） |
| THAI-SER | 11,848 | 17.36 | 5.28 | 只用一致率 ≥ 0.71 的片段，即作者推荐的子集（22,584 条里的 11,848 条；全集 33.19 小时） |
| nEMO | 4,481 | 3.07 | 2.47 | 全部片段 |
| MEAD（去掉 contempt） | 8,000 | 9.51 | 4.28 | 27,514 条里分层取 8,000 条（按演员、情绪、强度、句子等间隔取） |
| MELD（测试集） | 2,608 | 2.31 | 3.18 | 测试集（2,610 句）去掉 2 条时间戳错误的片段 |
| EMOVO | 588 | 0.51 | 3.12 | 全部片段 |
| Emozionalmente | 6,902 | 6.30 | 3.29 | 全部片段 |
| SAVEE | 480 | 0.51 | 3.84 | 全部片段（方案 B 新加） |
| **合计** | **56,264** | **52.24** | 3.34 | |

THAI-SER 和 MEAD 占全部音频的 51%。每个数据集 200 条的试跑（`sample(ds, 200)`，共 2,600 条）是 2.20 小时。

后来加入的三个对话数据集不在这张表里：EmotionTalk 官方 test（1,929 条）、DailyTalk validation（1,197 条）和 MultiDialog `gold_both`（7,672 条，13.10 小时）。加上它们，基准共 67,062 条；EmotionTalk 和 DailyTalk 这两个划分的时长没有为本调研实测。

### 3b 费用公式

对一个模型、一组片段：

费用 = 1.02 × (A × r × P_audio + N × 175 × P_text_in + N × 5 × P_text_out) / 1,000,000

- A = 音频秒数（Qwen 不到 1 秒按 1 秒算）；r = 每秒音频的 token 数（第 2 节表格）；N = 片段数。
- 每条 175 个文本输入 token：系统提示词 119 个、用户提示词 42 个（Qwen2.5-Omni 分词器，标签已填入），加上对话模板约 14 个。别的分词器差几个 token。
- 每条 5 个输出 token：一个标签 1–3 个 token，加结束符。
- 1.02：约 2% 的片段要重问（`retries_invalid` 和失败重试）。

算例，Qwen3.5-Omni Plus（国际），方案 B：音频 52.36 小时（不到 1 秒的按 1 秒算）× 3,600 × 7 = 132 万 token × $11 = $14.51；文本输入 56,264 × 175 = 985 万 token × $1.4 = $13.78；输出 56,264 × 5 = 28 万 token × $8.3 = $2.33；合计 $30.63 × 1.02 = **$31.25**。

### 3c 费用估算

| 模型 | 方案 B，全部片段（56,264 条，52.2 小时） | token 数（音频 / 全部） |
|---|---|---|
| qwen3-omni-flash，国际 | $14.35 | 2.36M / 12.48M |
| qwen3-omni-flash，内地 | $8.56 | 同上 |
| qwen3.5-omni-plus，国际 | $31.25 | 1.32M / 11.45M |
| qwen3.5-omni-plus，内地 | $21.03 | 同上 |
| qwen3.5-omni-flash，国际 | $8.69 | 1.32M / 11.45M |
| qwen3.5-omni-flash，内地 | $6.88 | 同上 |
| qwen3.8-omni-flash，国际 | $1.84 | 1.32M / 11.45M |
| qwen3.8-omni-flash，内地及其他地域 | $1.40 | 同上 |

token 数不含 2% 的重问。都是估算；实际账单还取决于提示词缓存（Qwen3.8 自动缓存，重复的文本输入按 $0.016/M 而不是 $0.15/M 计）和多少回答需要追问。每个 Qwen 模型的文本提示词都占输入 token 的 80–90%，占费用从 30%（qwen3-omni-flash）到 82%（qwen3.8-omni-flash），Qwen3.5-Omni 约 45%。
每个数据集 200 条的试跑（2,600 条，2.20 小时）：Qwen3.5-Omni Plus $1.38、Flash $0.38、qwen3-omni-flash $0.63、qwen3.8-omni-flash $0.08（国际价）。
运行时间：Qwen API 没有实测；按北京每分钟 60 次的限流，一个模型跑方案 B 至少 15.6 小时。一次一个请求要慢得多（GPT Realtime 每条约 5.5 秒），所以加载器应该同时发几个请求。
Step-Audio 2 没有算：阶跃星辰没有公开音频 token 换算；如果和 Qwen 差不多（每秒 7–12.5 个 token），方案 B 是 1,150–1,250 万输入 token × $1.43 加 28 万输出 token × $10，约 $20（估算）。

## 4. 自己部署：硬件和接入

### 4a 硬件表（开源模型，含基准里已有的 3 个本地音频大模型）

bf16 显存 = 参数量 × 2 字节（只算权重）；短片段（1–5 秒，加提示词不到 300 个 token）另需约 1–3 GB 激活，KV cache 几乎可以忽略。“放得下”指权重加这点余量；“秒/条”是每次一条、用 transformers、输出 ≤ 5 个 token 时的数量级：全是**估算，没有实测**。

| 模型 | 加载的参数 | bf16 显存 | L4 24 GB | L40 48 GB | H100 80 GB | Mac M4 Max 48 GB | 秒/条（估算） | 接入方式 |
|---|---|---|---|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct（已在基准中） | 8.40B | 16.8 GB | 能（偏紧） | 能 | 能 | 能（MPS） | L40 0.2–0.5；H100 0.1–0.3；Mac 1–3 | transformers 原生（`qwen2_audio` 加载器） |
| Qwen2.5-Omni-7B（已在基准中） | Thinker 约 8.9B（仓库含 talker 共 10.73B） | 约 17.9 GB | 能（偏紧） | 能 | 能 | 能（MPS） | 同上 | transformers 原生（`qwen2_5_omni` 加载器，不加载 talker） |
| Gemma 4 E4B-it（已在基准中） | 含嵌入 8.00B（有效 4.5B） | 16.0 GB | 能 | 能 | 能 | 能（MPS） | 比 7B 模型稍快 | transformers ≥ 5.5 原生（`gemma4_audio` 加载器） |
| Qwen3-Omni-30B-A3B-Instruct | Thinker 31.72B（talker 和 code2wav 共 3.54B 不加载） | 63.4 GB | 不能（4-bit 版本也许能，未核实） | **只能 FP8 或 4-bit**（社区版本，见 4b） | **能，1 张卡**（约剩 15 GB） | bf16 不能 | transformers：H100 0.5–2（MoE 开销）；vLLM 批处理：吞吐约 0.02–0.1 秒/条 | transformers 原生（`Qwen3OmniMoeForConditionalGeneration`，5.18.0 里有）或 vLLM（Thinker）；和 `qwen2_5_omni` 加载器很像 |
| Kimi-Audio-7B-Instruct | 9.77B（主 safetensors）+ Whisper-large-v3 文件夹（3.1 GB）；文本输出用不到 19.0 GB 的 detokenizer 和 1.0 GB 的声码器（`load_detokenizer=False`） | 约 20–24 GB（估算） | 临界，未核实 | 能 | 能 | 不能（只支持 CUDA） | L40 0.3–0.8；H100 0.2–0.5 | 它自己的仓库（`KimiAudio` 类）：torch 2.6.0、flash_attn 2.7.4.post1、deepspeed；单独环境或 Docker |
| Step-Audio 2 mini | 8.32B（+ token2wav 1.2 GB，用不到） | 16.6 GB | 能（偏紧） | 能 | 能 | 不改 `.cuda()` 就不能 | L40 0.2–0.5 | 它自己的仓库（`examples.py` 里的 `StepAudio2`）：transformers==4.49.0，写死 `.cuda()`；单独环境；有 vLLM 的 Docker 镜像 |
| Audio Flamingo 3 | 8.27B | 16.5 GB | 能 | 能 | 能 | 多半能（MPS；未核实） | L40 0.2–0.5；Mac 1–3 | transformers 原生（`AudioFlamingo3ForConditionalGeneration`，transformers ≥ 5.0；5.18.0 可用） |
| Audio Flamingo Next | 8.27B | 16.5 GB | 能 | 能 | 能 | 多半能（未核实） | 同 AF3 | transformers（`AutoModel`） |
| MiDashengLM-7B | 8.28B | 16.6 GB | 能 | 能 | 能 | 未核实 | 同上 | transformers `trust_remote_code`；官方有 FP8 和 GPTQ 版本 |
| Voxtral Mini 3B / Small 24B | 4.68B / 24.26B | 9.4 / 48.5 GB | Mini 能 / Small 不能 | Mini 能 / Small 不能（刚超过 48 GB） | 都能 | Mini 能 / Small 不能 | Mini 0.1–0.3 | transformers 原生 |

方案 B 的 13 个数据集要多久：按每条 0.3 秒，一个 7–9B 模型在 L40 上跑这 56,264 条约 4.7 小时；在 Mac 上按每条 1–3 秒约 16–47 小时（估算）。任何单个模型都用不着 4×H100；它们只在想同时跑 4 个模型（或 4 份 Qwen3-Omni）时有用。

### 4b Qwen3-Omni 细节

- **参数**（从官方仓库的 safetensors 文件头数出来）：Thinker 31.72B = 语言模型 30.22B + 音频编码器（AuT）0.65B + 视觉编码器 0.54B + 输出层 0.31B；talker 3.18B（含投影层）、它的 code predictor 0.14B 和 code2wav 0.22B（共 3.54B）；总计 35.26B。在 transformers 里，只用音频时视觉编码器也会加载。
- **显存**：卡片给的最低显存是 bf16、整个模型、15 秒视频 78.85 GB，并说 `model.disable_talker()` 能省约 10 GB（[卡片](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)）。短音频、关掉 talker 时：权重 63.4 GB 再加几 GB（估算），所以一张 H100 80 GB 够用但余量小；2×H100 张量并行就宽裕了。
- **KV cache**：48 层 × 4 个 KV 头 × 128 维 × 2（K 和 V）× 2 字节 = 每个 token 96 KiB（config.json）。一条片段加提示词约 200–300 个 token（每秒 12.5 个音频 token），即约 20–30 MB；THAI-SER 最长的片段（91 秒，约 1,140 个音频 token）也只有约 0.13 GB。KV cache 不是瓶颈。
- **量化版本**（Qwen 官方没有 FP8、AWQ 或 GPTQ 版；官方账号下只有 Instruct、Thinking、Captioner 三个仓库，2026-11-05 查过）。社区版本，文件大小含 talker：
  [cyankiwi AWQ-4bit](https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit) 27.6 GB（下载 142,949 次，用 llm-compressor 做的）、[cyankiwi AWQ-8bit](https://huggingface.co/cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit) 42.5 GB、[sammysun0711 FP8-Dynamic](https://huggingface.co/sammysun0711/Qwen3-Omni-30B-A3B-Instruct-FP8-Dynamic) 37.3 GB、[Intel int4-AutoRound](https://huggingface.co/Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound) 26.3 GB、[thomasip GPTQ-4bit](https://huggingface.co/thomasip/Qwen3-Omni-30B-A3B-Instruct-GPTQ-4bit)。
  L40（Ada 架构，支持 FP8）上 FP8 或 4-bit 版本放得下，还有批处理的余量。量化模型不是参考模型：报告里写成“Qwen3-Omni（FP8，社区版）”；如果有了 H100，在一个小数据集上用 bf16 核对一致性。
- **vLLM**：vLLM 主线有 Thinker（`Qwen3OmniMoeThinkerForConditionalGeneration`，文本输出；[vLLM 文档](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_omni_moe_thinker/)）；[vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/stable/models/supported_models/) 支持整个模型，包括语音输出。卡片建议 `max_num_seqs=8`、`tensor_parallel_size` = GPU 数。用 vLLM 批处理时，Thinker（约 3B 激活参数）在 H100 上每秒应能处理很多条（估算，没有实测）。

## 5. 各候选的补充说明

- **Qwen3.5-Omni**：Plus 和 Flash 两种，都是 instruct 模型，256k 上下文；Thinker 和 talker 都是混合注意力 MoE（[报告](https://arxiv.org/abs/2604.15804)，v2 2026-04-21）。没有权重：Hugging Face 上没有 Qwen3.5-Omni 仓库（2026-11-05 搜索只找到名字里带“Qwen3.5-Omni”的无关社区模型）。另有 `qwen3.5-omni-flash-realtime`（国际：文本输入 $0.55、音频输入 $4.5、文本输出 $3.3）和 Plus 的实时版；这里用非实时 API 更简单。
- **qwen3.8-omni-flash**（不在原来的列表里）：最新的 Qwen omni API，2026 年 9 月发布（媒体报道 2026-09-14 发布，未核实；2026-09-21 起上架 OpenRouter）；所有模态输入同价 $0.15/M，支持非流式，100 万上下文，自动缓存。只有 API。方案 B 全部片段只要 $1.84，作为第三个 Qwen API 数据点几乎不花钱；没找到 SER 数字。
- **Kimi-Audio**：Whisper-large-v3 特征加 12.5 Hz 的离散语义 token，语言模型从 Qwen2.5-7B 初始化。在它自己的评测里 MELD 59.13 对 Qwen2.5-Omni 49.83，而 Qwen2.5-Omni 卡片报的是 57.0：两边的评测方法不同。SFT 数据列了 IEMOCAP 10 小时、MELD 9 小时、RAVDESS 3 小时、SAVEE 0.1 小时、ESD 29 小时（[论文](https://arxiv.org/abs/2504.18425)）。42.6 GB 的仓库里是语言模型（19.5 GB）、detokenizer（19.0 GB）、Whisper（3.1 GB）和声码器（1.0 GB）。接入：单独的 CUDA 环境，用 `KimiAudio` 类加 `output_type="text"`。
- **Step-Audio 2 mini**：8.32B，Apache-2.0。StepEval-Audio-Paralinguistic 有 11 个任务共 550 条，每个任务 50 条；情绪那部分是中文播客录音，开放式回答由大模型判分（[论文](https://arxiv.org/abs/2507.16632) 4.2 节）：情绪项 Step-Audio 2 86、mini 82、GPT-4o Audio 82、Qwen-Omni 76、Kimi-Audio 66。和 7 分类测试不可比。依赖：Python ≥ 3.10、带 CUDA 的 PyTorch ≥ 2.3、transformers==4.49.0（[GitHub](https://github.com/stepfun-ai/Step-Audio2)）。大的 Step-Audio 2 只能通过阶跃星辰的实时 API 用（`step-audio-2`：输入 $1.43/M、输出 $10/M，[价格](https://platform.stepfun.ai/docs/en/guides/pricing/details)），要像 GPT Realtime 那样写一个 WebSocket 加载器。
- **Audio Flamingo 3**：卡片的训练数据列表里有 MSP-Podcast、JL-Corpus、MELD、TESS、OMGEmotion、EmoV-DB（[卡片](https://huggingface.co/nvidia/audio-flamingo-3-hf)）；论文表 11 写 JL-Corpus 1.4 小时 / 2.4K 对（整个语料就是 2,400 条），训练 6 个 epoch，TESS 1.6 小时 / 2.8K（整个语料就是 2,800 条），MELD 8.7 小时 / 32.9K，MSP-Podcast 73.9 小时；表 9 还有拼接起来的长段 MELD 和 IEMOCAP 录音（[论文](https://arxiv.org/abs/2507.08128)）。许可（NVIDIA OneWay Noncommercial）只允许非商用研究。仓库里权重存了两份（分片和单文件，各 16.5 GB，都是 bf16），外加 1.4 GB 的“think”适配器：只下载一份就行。[VoxEmo](https://arxiv.org/abs/2603.08936) 已经在 35 个语料上零样本评测过 AF3 和 Qwen2-Audio，其中有 THAI-SER、nEMO、MEAD、CREMA-D：可以拿来对照。
- **Audio Flamingo Next**：大小和许可同 AF3；增加了更长的音频（最长 30 分钟）和时间戳；论文没写情绪数据集的名字，但它的数据是在 AF3 的数据（AF-Skills）上扩充的，所以同样这些数据集按“可能见过”处理。
- **MiDashengLM-7B**：用 Dasheng 音频编码器（不是 Whisper），用通用音频描述训练；卡片报了说话人（VoxCeleb1 92.36）、语种、性别和声音分类，没有情绪结果。
- **Voxtral**：为转写、翻译和语音问答设计；没找到情绪结果。Voxtral Small（bf16 下 48.5 GB）放不进 L40。

## 6. 训练数据和本项目数据集的重叠

在训练中见过某个数据集的模型，在这个数据集上就不算零样本测试。表里只列论文和卡片写明的；“未公开”不等于没有重叠。

| 模型 | 训练中见过（我们的数据集加粗） | 来源 |
|---|---|---|
| Audio Flamingo 3 | **JL-Corpus**（全部 2,400 条）、**TESS**（全部 2,800 条）、**MELD**（训练数据；是否用了测试集未核实）、MSP-Podcast、EmoV-DB、OMGEmotion、IEMOCAP（长音频数据） | 卡片；论文表 9 和表 11 |
| Audio Flamingo Next | 多半同 AF3（未核实） | 论文（数据在 AF3 的基础上扩充） |
| Kimi-Audio | **RAVDESS**（3 小时，比 1.5 小时的语音片段多：多半还包括歌唱片段或重复）、**MELD**（9 小时）、**SAVEE**（0.1 小时）、ESD（本项目已排除）、IEMOCAP | 论文 SFT 数据表 |
| Qwen2-Audio、Qwen2.5-Omni、Qwen3-Omni、Qwen3.5-Omni | 未公开；它们报 MELD 结果，所以多半用了 MELD 的训练集（未核实） | 论文 |
| Step-Audio 2、Gemma 4、MiDashengLM、Voxtral | 未公开 | — |
| 参考：基准里已有的模型 | 这里不再重复：本项目的核查把直接和上游使用都计入（例如 emotion2vec+ large 和 MERaLiON-SER-v1 各标了我们的 9 个语料，Melati 3 个）；GPT Realtime、Gemini 3.8 Flash、Gemma 4 和两个 Qwen 模型都没有对应语料的证据 | [training_provenance.md](training_provenance.md)、[reports/summary.md](../reports/summary.md) 的 Training data 一节 |

所以报告里要把 AF3 在 JL-Corpus、TESS 上的结果标“见过”，MELD 标“可能见过”；Kimi-Audio 的 RAVDESS、MELD 和 SAVEE 结果标“见过”。对 AF3 来说，CREMA-D、RAVDESS、MCAE-SPPS、MESS、THAI-SER、nEMO、MEAD、EMOVO、Emozionalmente、SAVEE 没有公开的重叠（同本项目的核查规则，这不等于确认没见过）。

## 7. 建议

1. **下一个加 Qwen3-Omni-30B-A3B-Instruct 开源权重。** 开源里最强的 Qwen，是 Qwen2.5-Omni 之后顺理成章的下一步；Apache-2.0；已固定的 transformers（5.18.0）就能加载，加载器和 `qwen2_5_omni.py` 很像（关 talker，只用 Thinker）。
   硬件：有 H100 就用 1×H100 80 GB 跑 bf16（2×H100 更宽裕更快）。否则用 L40 加社区 FP8 版本（或 AWQ-4bit），并注明是量化版。Mac 和 L4 都跑不了 bf16。
2. **Qwen3.5-Omni Plus 和 Flash 走 API，跑方案 B 的全部片段**（56,264 条，52.2 小时）：$31.25 + $8.69 = 约 $40（国际）；Plus 的 100 万 token 免费额度能抵约 $2.7。它们是最新的 Qwen omni 模型，只有 API，这是唯一的加法。要为百炼的 OpenAI 兼容 API 写一个新的流式加载器（`stream=True`），并申请国际（新加坡）的 key。同一个加载器顺便加 `qwen3.8-omni-flash` 只多 $1.84。正式跑之前先试跑 2,600 条（Plus $1.38，在免费额度内；Flash $0.38）检查加载器。时间：按每分钟 60 次，每个模型至少 15.6 小时；要同时发几个请求。`qwen3-omni-flash`（$14.35）可选：它是闭源变体，第 1 步的开源权重更适合作参考。
3. **Audio Flamingo 3**：工作量最小（transformers 原生，L40 能跑，Mac 多半也行），非商用许可这里可以接受。JL-Corpus 和 TESS 标“见过”，MELD 标“可能见过”，或者不计入它的平均值。
4. **Step-Audio 2 mini，用 L40**：单独环境（transformers 4.49.0），还要改掉 `.cuda()`；带来一个非 Qwen、非 NVIDIA、副语言能力宣传很强的模型。
5. **Kimi-Audio，用 L40**：接入工作量最大（单独的 CUDA 环境，flash-attn、torch 2.6）；RAVDESS、MELD 和 SAVEE 标“见过”。
- 暂不加：Audio Flamingo Next（AF3 有信息量再说）、MiDashengLM 和 Voxtral（不针对情绪、没有情绪结果）、Step-Audio 2 实时 API（为一个模型写 WebSocket 加载器）。
- 硬件小结：L40 能跑所有 7–9B 模型的 bf16，以及量化的 Qwen3-Omni；只有 bf16 的 Qwen3-Omni 需要 H100；L4 能一个一个地跑 7–9B 模型（Kimi-Audio 临界）；4×H100 只在并行跑多个模型时有用。

## 8. 主要来源

- 阿里云百炼：价格 https://www.alibabacloud.com/help/en/model-studio/model-pricing ；Qwen-Omni（token 规则、流式、限制）https://www.alibabacloud.com/help/en/model-studio/qwen-omni ；qwen3-omni-flash 页面 https://docs.modelstudio.console.alibabacloud.com/en/model-studio/qwen3-omni-flash ；Qwen3-Omni-Flash-2025-12-01 https://www.alibabacloud.com/blog/602735
- Qwen3-Omni：https://arxiv.org/abs/2509.17765 , https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct , https://github.com/QwenLM/Qwen3-Omni ；vLLM：https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/qwen3_omni_moe_thinker/ , https://docs.vllm.ai/projects/vllm-omni/en/stable/models/supported_models/
- Qwen3.5-Omni：https://arxiv.org/abs/2604.15804
- OpenRouter 模型列表：https://openrouter.ai/api/v1/models
- 阶跃星辰：https://platform.stepfun.ai/docs/en/guides/pricing/details
- Kimi-Audio：https://arxiv.org/abs/2504.18425 , https://github.com/MoonshotAI/Kimi-Audio , https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct
- Step-Audio 2：https://arxiv.org/abs/2507.16632 , https://github.com/stepfun-ai/Step-Audio2 , https://huggingface.co/stepfun-ai/Step-Audio-2-mini
- Audio Flamingo 3 和 Next：https://arxiv.org/abs/2507.08128 , https://huggingface.co/nvidia/audio-flamingo-3-hf , https://arxiv.org/abs/2604.10905 , https://huggingface.co/nvidia/audio-flamingo-next-hf
- MiDashengLM：https://arxiv.org/abs/2508.03983 , https://huggingface.co/mispeech/midashenglm-7b-1021-bf16 ；Voxtral：https://huggingface.co/mistralai/Voxtral-Mini-3B-2507 , https://huggingface.co/mistralai/Voxtral-Small-24B-2507
- 数据集数字：EmoBox https://arxiv.org/abs/2406.07162 ；MELD https://github.com/declare-lab/MELD ；VoxEmo https://arxiv.org/abs/2603.08936
- 查询接口：Hugging Face `https://huggingface.co/api/models/<id>?blobs=true`，以及用 HTTP range 请求读 safetensors 文件头

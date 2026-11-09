# 音频输入时，API 能否给出答案的 token logprobs？

调研日期：2026-11-06。只读官方文档、SDK 源码、官方论坛和公开 PR，**没有调用任何 API，也没有用任何 key**。
本文件已被 `.gitignore` 忽略（`research_paper/`），不要提交。

## 一句话结论

- **GPT Realtime（gpt-realtime / gpt-realtime-2 / gpt-realtime-2.1）：拿不到答案的 logprobs。** Realtime API 唯一的 logprobs 选项是 `include: ["item.input_audio_transcription.logprobs"]`，它给的是**输入音频转写文字**的概率，不是模型**答案**的概率。
- **Gemini gemini-3.8-flash：拿不到。** 参数 `responseLogprobs` / `logprobs` 还在文档里，但官方写明“对 Gemini 3.x 已弃用”；Google 论坛员工回复“logprobs are no longer returned for 3.X models（WAI）”；2026-10-31 有人对 `gemini-3.8-flash` 实测得到 400 `Logprobs is not enabled for this model`。
- 想要 7 个情绪的概率，现实做法是 **重复采样 N 次，用答案频率当概率**（两家都能用）；可再加一个“请输出 7 类概率 JSON”的口头置信度作对照。

## 总表（模型 × API × 音频输入时能否拿到“答案”的 logprobs）

| 模型 | API | 答案 logprobs（音频输入） | 依据 |
|---|---|---|---|
| gpt-realtime、gpt-realtime-2、gpt-realtime-2.1 | Realtime API（WebSocket） | **否** | [client events](https://developers.openai.com/api/reference/resources/realtime/client-events)、[server events](https://developers.openai.com/api/reference/resources/realtime/server-events)；openai-python SDK（2026-11-05 版）`realtime_response_create_params_param.py` 无 logprobs/include 字段 |
| 同上 | Realtime 的 `include: ["item.input_audio_transcription.logprobs"]` | **否（这是输入转写的概率，不是答案）** | 同上；SDK `realtime_session_create_request_param.py` 第 28 行 |
| gpt-realtime 系列 | Chat Completions / Responses | **不能用**（这些模型只支持 `v1/realtime`） | [gpt-realtime-2](https://developers.openai.com/api/docs/models/gpt-realtime-2)、[gpt-realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1) 模型页 |
| gpt-audio-1.5（以及已弃用的 gpt-audio、gpt-4o-audio-preview） | Chat Completions，`logprobs: true` + `top_logprobs`（0–20） | **不清楚**：参数存在，文档没写音频模型或音频输入的限制，也没说支持；网上找不到成功实例 | [Chat Completions create](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[gpt-audio-1.5 模型页](https://developers.openai.com/api/docs/models/gpt-audio-1.5) |
| 任何 OpenAI 模型 | Responses API，`include: ["message.output_text.logprobs"]` + `top_logprobs` | **否**：Responses 的输入内容类型没有音频（只有 `input_text` / `input_image` / `input_file`），音频模型也只开放 Chat Completions | [Responses create](https://developers.openai.com/api/reference/resources/responses/methods/create)、SDK `response_includable.py` |
| gemini-3.8-flash | Gemini Developer API（AI Studio）`generateContent` | **否**（实测 400 `Logprobs is not enabled for this model`） | [litellm PR #49514](https://github.com/BerriAI/litellm/pull/49514)（2026-10-31 合并，含实测日志）、[论坛 Google 员工回复](https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557) |
| gemini-3.8-flash | Vertex AI（Gemini Enterprise Agent Platform）`generateContent` | **否**（文档：对 3.x 已弃用；同代模型实测 400 `Logprobs is not supported for this model`） | [Vertex 推理参考](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference)、[litellm PR #31642](https://github.com/BerriAI/litellm/pull/31642)、[论坛](https://discuss.ai.google.dev/t/were-logprobs-disabled-for-gemini-3-3-1-in-vertex-api/132426) |
| gemini-2.5-flash（参考） | Vertex AI | 曾经**是**，但该模型 2026-10-20 已到退役日 | [Vertex 模型版本页](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) |

---

## 1. OpenAI

### 1.1 Realtime API（我们现在用的路径）

我们的做法（`scripts/models/gpt_realtime.py`）：`session.update`（`output_modalities: ["text"]`，`turn_detection: None`）→ `conversation.item.create`（`input_audio` + `input_text`）→ `response.create` → 读 `response.output_text.delta/done` 和 `response.done`。

查到的事实：

- **`response.create`** 的参数里没有 `logprobs`、`top_logprobs`，也没有 `include`。可用的有 `output_modalities`、`instructions`、`tools`、`reasoning`（只对 gpt-realtime-2 这类推理模型）、`max_output_tokens` 等。
- **`session.update`** 的 `include` 只有一个合法值：`"item.input_audio_transcription.logprobs"`，官方说明是“Include logprobs for input audio transcription”。
- **服务端事件**：`response.output_text.delta` 只有 `event_id, response_id, output_index, content_index, text, type`；`response.output_text.done` 同理；`response.done` 里也没有 logprobs。
  只有 `conversation.item.input_audio_transcription.delta/completed` 带 `logprobs`（SDK 类型 `LogProbProperties`）。

**重点区分**：

| | 是什么 | 对我们有没有用 |
|---|---|---|
| `item.input_audio_transcription.logprobs` | 另一个转写模型（如 gpt-4o-transcribe）把**用户的音频转成文字**时，每个词的概率 | 没用。它只说明“说了什么词”，和模型回答的情绪无关；而且是另一个模型算的 |
| 答案的 logprobs | 模型输出 `happy` 这个词时，7 个情绪各自的概率 | 我们要的，但 Realtime **不提供** |

**温度（temperature）**：旧的 beta 接口温度范围是 0.6–1.2（默认 0.8）。beta 接口已于 **2026-05-12 下线**（[deprecations](https://developers.openai.com/api/docs/deprecations)），现在的 GA 接口**根本没有温度参数**（SDK 里 realtime 类型中搜不到 `temperature`）。所以 Realtime 的答案永远是按模型默认方式采样的，同一段音频多次请求可能得到不同答案。这正好可以拿来做“重复采样估概率”。

**推理（reasoning）**：gpt-realtime-2 / 2.1 是推理模型，`reasoning.effort` 可选 `minimal / low / medium / high / xhigh`。一个词的答案建议用 `minimal`，省钱也省时间（推理 token 按输出计费）。

**模型状态**：`gpt-realtime` 和 `gpt-realtime-mini` 将在 **2027-01-20 关停**，官方推荐换成 `gpt-realtime-2.1`（[deprecations](https://developers.openai.com/api/docs/deprecations)）。

### 1.2 Chat Completions + 音频模型（gpt-audio-1.5）

- 参数名：`logprobs: true`，`top_logprobs: 0–20`（官方原文：“An integer between 0 and 20 …”）。温度 0–2，可以设 0。
- 音频输入：消息内容里放 `{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav" | "mp3"}}`。注意只接受 **wav / mp3**，不是 Realtime 的裸 PCM16。
- **能否和音频输入一起用：不清楚。** Chat Completions 文档没有写“音频模型不支持 logprobs”，gpt-audio-1.5 模型页也没把 logprobs 列为功能（它只列 streaming、function calling）。网上没找到成功返回 logprobs 的公开例子；第三方文档里的示例响应 `logprobs` 为 `null`，但那很可能只是没开这个参数，不能当证据。
- 只有 1 次付费调用就能确认：放一个小脚本到 `emotion_recognition/test_scripts/`，发一段短音频，`logprobs: true, top_logprobs: 20`，看返回里的 `choices[0].logprobs.content` 是否为空。
- **注意**：gpt-audio-1.5 和 gpt-realtime 是**不同的模型**。就算它能给 logprobs，也只能说明 gpt-audio-1.5 的概率，不能代表我们论文里 GPT Realtime 的结果；要用就得把它当第三个 API 模型单独评测。
- 价格（模型页）：音频输入 $32/1M tokens，文字输入 $2.50/1M，文字输出 $10/1M。gpt-audio、gpt-4o-audio 等旧模型 2027-01-20 关停，gpt-4o-audio-preview 已于 2026-05-07 关停。

### 1.3 Responses API

- 参数名：`include: ["message.output_text.logprobs"]` 加 `top_logprobs`（0–20）。这是 Responses API 拿 logprobs 的正式方法。
- **但 Responses API 不能用于我们的场景**：`responses.create` 的输入内容类型只有 `input_text`、`input_image`、`input_file`。SDK 里虽然有一个 `ResponseInputAudioParam` 类型，但它没有被放进 `responses.create` 的输入联合类型里；而且现有的音频模型（gpt-audio-1.5）、Realtime 模型都不支持 `v1/responses`。社区也有“从 completions 迁到 responses 后音频输入不能用”的报告。结论：**否**。

---

## 2. Google Gemini（gemini-3.8-flash）

### 2.1 这个模型存在吗

存在。官方文档有 `gemini-3.8-flash` 的模型页（[AI Studio 文档](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[Vertex 文档](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)、[model card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)）：2026-09-02 发布，GA（Stable），输入支持文字、图片、视频、**音频**、PDF，输出只有文字；Vertex 上退役日期“2027-09-02 或更晚”。模型页的功能列表**完全没提 logprobs**。

### 2.2 logprobs 参数

- 参数名（`generationConfig` 里）：`responseLogprobs: true`（开关），`logprobs: 1–20`（每一步返回前几名候选；必须先开 `responseLogprobs`）。返回在 `candidates[].logprobsResult`（`chosenCandidates`、`topCandidates`），以及 `avgLogprobs`。
- **Vertex 文档原文**：“The `responseLogprobs` parameter is deprecated for Gemini 3.x models and will soon be completely deprecated.” `logprobs` 参数同样写了这句话。
- **Google 员工在官方论坛的回复**（2026-08-05，问题针对 3.1 Pro 和 3.6 Flash，Vertex 和 AI Studio 都问了）：“This is currently WAI, logprobs are no longer returned for 3.X models”（WAI = 设计如此）。
- **对 gemini-3.8-flash 的实测**（litellm PR #49514，2026-10-31，Gemini Developer API）：`{"code": 400, "message": "Logprobs is not enabled for this model", "status": "INVALID_ARGUMENT"}`。作者还说测试 key 上所有 AI Studio 模型都返回这个错误。
- **Vertex 上的 3.x**（litellm PR #31642，2026-06-29）：gemini-3-pro-preview、gemini-3.1-pro-preview、gemini-3.5-flash 返回 400 `Logprobs is not supported for this model`。我没找到 3.8 Flash 在 Vertex 上的实测记录，但文档的“3.x 已弃用”和员工回复都覆盖它，所以判断为**否**。
- 和音频输入无关：问题出在模型本身，文字输入也拿不到，所以 inline 音频（`inlineData`，`mimeType: audio/wav`）当然也拿不到。
- 能拿 logprobs 的旧模型 gemini-2.5-flash 已经到了退役日（Vertex 2026-10-20，Developer API 2026-10-16）；litellm 在 10-31 还在 Vertex 上测通过，说明下线是逐步进行的，**不能依赖**，而且它是另一个模型。

### 2.3 thinking（思考 token）设置

- gemini-3.8-flash 的 thinking **关不掉**。`thinkingConfig.thinkingLevel` 只能是 `LOW`、`MEDIUM`（默认）、`HIGH`；`MINIMAL` 对 3.8 Flash 不可用，会报错。
- 一个词的答案建议设 `thinkingConfig: {thinkingLevel: "LOW"}`。思考 token 按输出价格计费。
- Gemini 3 系列不再推荐用数字的 `thinkingBudget`；同一个请求里同时设 `thinkingLevel` 和 `thinkingBudget` 会报错。
- 温度：Vertex 模型页写的默认值是 temperature 1.0、topP 0.95、topK 64。Gemini 3 的开发指南曾建议 3.x 保持温度 1.0（第三方转载的说法，原页面现在已改成 Gemini 4 的内容，我没能在官方页面上再次核对）。我们计划的 temperature 0 可以设，但要注意 Google 不推荐；如果做重复采样，反而应该用默认的 1.0。

---

## 3. 如果有 logprobs：怎样算 7 个情绪的概率

（目前两家都不行；这一节给以后可能开放的 gpt-audio-1.5 Chat Completions，或本地模型参考。本地模型我们已经在 `audio_llm.py` 里这样做了：`save_label_probs`。）

1. 只看**答案的第一个 token**，请求 `top_logprobs: 20`（两家上限都是 20）。
2. 每个标签取它的第一个 token。用 o200k_base 分词器（GPT-4o 系列使用的分词器；gpt-realtime / gpt-audio 是否也用它，官方没写）在本地查到：答案开头**没有前导空格**时，
   `angry → "ang"`、`disgusted → "dis"`、`fearful → "fear"`、`happy`、`neutral`、`sad`、`surprised → "sur"`。7 个首 token 互不相同，可以区分。
   带前导空格时（` angry`、` fearful`、` surprised`）多数是单个 token，大写开头又是另一组 token（`Ang`、`Sur` …）。所以要把同一标签的**大小写、有无空格**几种变体的概率加起来。
3. 7 个标签的概率加起来再归一化（renormalise），同时记下它们占全部概率的比例（我们本地叫 `label_mass`），比例太低说明模型想说别的东西（比如 `I`、`The`、`**`）。
4. 不在 top-20 里的标签：概率记为 0，或用“top-20 里最小的那个概率”作上限。
5. 注意事项：
   - logprobs 是模型在**采样之前**的分布，所以即使答案是随机采出来的，概率本身还是有意义的。OpenAI 文档没有明确说温度会不会改变返回的 logprobs，不要假设，最好固定温度（Chat Completions 可以设 0）。
   - 推理模型的思考过程先于答案，答案第一个 token 的概率是“想完以后”的概率，要说明清楚。
   - Realtime 现在没有温度参数，所以就算它以后开放 logprobs，也只能用默认采样。

---

## 4. 拿不到 logprobs 时的替代方法

按推荐程度排列：

| 方法 | 做法 | 优点 | 缺点 / 成本 |
|---|---|---|---|
| **A. 重复采样，用频率当概率**（推荐） | 每段音频请求 N 次（如 N = 10 或 20），7 个标签的出现次数 / N 就是概率估计 | 两家都能用；不依赖模型说实话；文献里叫 sampling consistency / self-consistency（Wang et al. 2022，[arXiv:2203.11171](https://arxiv.org/abs/2203.11171)） | 成本 ×N；N = 10 时概率精度只有 0.1；必须真的有随机性：Realtime 没有温度、天然随机；Gemini 要用默认温度 1.0（温度 0 时 N 次几乎一样，没意义） |
| **B. 让模型直接输出 7 类概率（JSON）** | 提示词要求输出 `{"angry": 0.1, ...}`。Gemini 可用 structured outputs（`responseMimeType: "application/json"` + `responseSchema`）保证格式；Realtime 没有 structured outputs，只能靠提示词，再自己解析 | 1 次请求就有完整分布，最便宜 | 这是“口头置信度”（verbalised confidence），通常过度自信、数值扎堆在 0.8/0.9（Xiong et al. 2024，[arXiv:2306.13063](https://arxiv.org/abs/2306.13063)）；也有研究发现 RLHF 模型的口头置信度比 token 概率更准（Tian et al. 2023，[arXiv:2305.14975](https://arxiv.org/abs/2305.14975)）。而且改了输出格式，主实验的准确率可能会变 |
| **C. 单个置信度** | 答案 + 0–100 的信心分 | 简单 | 只有一个数，没有完整分布；同样过度自信 |
| **D. 用 structured outputs 把答案限死在 7 个标签** | Gemini：`responseMimeType: "text/x.enum"` + `responseSchema` 的 `enum` | 减少 `invalid` 答案 | 本身不给概率，只是让 A 的统计更干净 |

建议：**主结果仍用现在的单次答案**；另外抽一个子集（例如每个数据集 50–100 段）做 A（N = 10），得到频率分布，用于校准（calibration）/ 熵分析；B 可在同一子集上做 1 次作为对照，单独报告，不混进主结果。

### 成本粗算（只算量级，价格以官方页为准）

- **gpt-realtime-2**：用户音频 1 token / 100 ms（[成本指南](https://developers.openai.com/api/docs/guides/realtime-costs)），3 秒片段约 30 个音频 token。价格：音频输入 $32/1M、文字输入 $4/1M、文字输出 $24/1M。每次请求大约：音频 $0.001 + 提示词（约 150 token）$0.0006 + 输出和推理 token（用 `minimal` 时很少），合计 **约 $0.002**。
  例：200 段 × 13 个数据集 × N = 10 ≈ 26,000 次 ≈ **$50**；只做子集（13 × 100 × 10 = 13,000 次）≈ $25。重复发送同一段音频可能命中 prompt caching（缓存输入更便宜），但不影响采样结果。
- **gemini-3.8-flash**：价格页写音频按每秒 25 个 token 计（较早的音频文档写的是每秒 32 个），输入 $0.75/1M（2027-01-01 起 $1.50），输出（含思考）$3.75/1M（2027 年起 $7.50）；Batch API 半价。每次请求：输入约 300 token ≈ $0.0002，输出主要是 LOW 档的思考 token（几十到几百个）≈ $0.0004–0.002。N = 10 的全量也只有 **约 $10–50**，用 Batch API 可减半。`candidateCount > 1` 能否在 3.x 上一次请求返回多个答案，文档没写清楚，要试过才知道。

## 参考来源

- OpenAI：[Realtime client events](https://developers.openai.com/api/reference/resources/realtime/client-events)、[Realtime server events](https://developers.openai.com/api/reference/resources/realtime/server-events)、[Realtime conversations 指南](https://developers.openai.com/api/docs/guides/realtime-conversations)、[Realtime 成本](https://developers.openai.com/api/docs/guides/realtime-costs)、[Chat Completions create](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)、[Responses create](https://developers.openai.com/api/reference/resources/responses/methods/create)、[gpt-realtime-2](https://developers.openai.com/api/docs/models/gpt-realtime-2)、[gpt-realtime-2.1](https://developers.openai.com/api/docs/models/gpt-realtime-2.1)、[gpt-audio-1.5](https://developers.openai.com/api/docs/models/gpt-audio-1.5)、[Deprecations](https://developers.openai.com/api/docs/deprecations)、[Changelog](https://developers.openai.com/api/docs/changelog)、[openai-python SDK](https://github.com/openai/openai-python)（2026-11-05 提交 c6fc264 的 `types/realtime`、`types/responses`、`types/chat`）、[社区：Realtime 转写 logprobs 缺失](https://community.openai.com/t/logprobs-missing-from-realtime-transcription-api-responses/1278181)、[社区：Realtime 温度](https://community.openai.com/t/temperature-in-the-new-gpt-realtime-model/1356936)、[社区：Responses 音频输入不可用](https://community.openai.com/t/audio-input-not-working-when-migrating-from-completions-to-responses/1364108)。
- Google：[gemini-3.8-flash（AI Studio）](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)、[gemini-3.8-flash（Vertex）](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)、[Model card](https://deepmind.google/models/model-cards/gemini-3-8-flash/)、[generateContent 参考（Developer API）](https://ai.google.dev/api/generate-content)、[Vertex 推理参考（3.x 弃用说明）](https://docs.cloud.google.com/gemini-enterprise-agent-platform/reference/models/inference)、[Thinking 指南](https://ai.google.dev/gemini-api/docs/thinking)、[价格](https://ai.google.dev/gemini-api/docs/pricing)、[Vertex 模型版本与退役日期](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions)、[论坛：3.X 不再返回 logprobs（员工回复）](https://discuss.ai.google.dev/t/missing-logprobs-support-in-the-newest-gemini-models-3-1-pro-3-6-flash-on-vertex-ai-and-ai-studio/176557)、[论坛：Vertex 上 3/3.1 的 logprobs 被关](https://discuss.ai.google.dev/t/were-logprobs-disabled-for-gemini-3-3-1-in-vertex-api/132426)、[论坛：2.5 Flash 在 Developer API 被关](https://discuss.ai.google.dev/t/logprobs-is-not-enabled-for-gemini-models/107989)、[Vertex logprobs 博客（2.5 Flash 示例）](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/)。
- 第三方实测：[litellm PR #49514](https://github.com/BerriAI/litellm/pull/49514)（gemini-3.8-flash 实测 400）、[litellm PR #31642](https://github.com/BerriAI/litellm/pull/31642)（Vertex 3.x 实测 400）。

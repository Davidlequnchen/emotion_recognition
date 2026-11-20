# 训练来源与图表标记核查

更新：2026-11-10。覆盖八个已评测 checkpoint ×十六个采用数据集。两个 API 模型（GPT Realtime、Gemini 3.8 Flash）不公布训练语料，见下文。

**图表只用一种斜线 `///`，含义为 “Part of the training process”。** 直接或间接、上游预训练、seed 微调、教师伪标签传播、最终微调，均计为用过／见过；作者明确使用过的开发校准也计入并注明阶段。图中不再区分 direct/upstream/possible，也不放解释脚注。

论文、技术报告、模型卡和作者说明都可以作为依据，不要求拿到训练日志。资料没有列出某语料时，不自动认证为未见。语料参与训练流程，也不等于本项目测试文件发生泄漏。这些解释放在论文和本页，不挤进图内。

以下 ✓ 均使用同一种训练标记；— 表示未找到对应使用依据。配置的 `training_data`、逐结果的 `in_training_data` 与图表保持一致。具体来源和阶段保留在 [机器可读审核](training_provenance.json)，不作额外的视觉等级。

| 数据集 | e2v+ large | MERaLiON | Melati | Vox Whisper | Vox WavLM | Qwen Omni | Qwen Audio | Gemma4 |
|---|---|---|---|---|---|---|---|---|
| CREMA-D | ✓ | ✓ | — | — | — | — | — | — |
| MCAE-SPPS | — | — | — | — | — | — | — | — |
| MESS | — | — | — | — | — | — | — | — |
| RAVDESS | ✓ | ✓ | — | — | — | — | — | — |
| TESS | ✓ | ✓ | — | — | — | — | — | — |
| JL-Corpus | ✓ | ✓ | ✓ | — | — | — | — | — |
| THAI-SER | — | — | — | — | — | — | — | — |
| nEMO | — | — | ✓ | — | — | — | — | — |
| MEAD | ✓ | ✓ | — | — | — | — | — | — |
| MELD | ✓ | ✓ | — | — | — | — | — | — |
| EMOVO | ✓ | ✓ | — | — | — | — | — | — |
| Emozionalmente | ✓ | ✓ | ✓ | — | — | — | — | — |
| SAVEE | ✓ | ✓ | — | — | — | — | — | — |
| DailyTalk | — | — | — | — | — | — | — | — |
| EmotionTalk | — | — | — | — | — | — | — | — |
| MultiDialog | — | — | — | — | — | — | — | — |

## 逐模型证据和限制

### emotion2vec+ large

42,526 h 伪标签／内部工业数据，具体语料未公开。EmoBox、原始 emotion2vec 与教师关系保留为潜在来源，计入已见；具体的上游传播路径保留在来源记录中，不在图上分级。

一手来源：[来源1](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/6c303ba987b86b93193de93e34bb2b077a6bedc4/README.md)；[来源2](https://github.com/ddlBoJack/emotion2vec/issues/73)；[来源3](https://github.com/ddlBoJack/emotion2vec/issues/46)。

### MERaLiON-SER-v1

SER 训练明确包括 CREMA-D、ESD、MELD、M3ED、SG-ECMT、SGTV。采用的16个中确认 CREMA-D/MELD；ESD 仍有依据但已归档。Whisper-medium 保留的网络预训练来源没有完整逐语料清单。

一手来源：[来源1](https://arxiv.org/html/2511.04914v1#S4.SS1)；[来源2](https://huggingface.co/MERaLiON/MERaLiON-SER-v1/blob/7e3ee6fa4534dea8316e8ca43e377e2fbb58496b/README.md)；[来源3](https://huggingface.co/openai/whisper-medium#training-data)。

### Melati

当前公开 head 的九语料中确认 JL-Corpus、Emozionalmente；另一个内部 head 的数据不能移给当前 checkpoint。JL 只确认语料／子集关系，未证明本项目1,200条都曾训练。作者用 nEMO/NonverbalTTS 调过 podcast 阈值，本项目 raw argmax 不采用该阈值；仍披露这项开发关联，不称 nEMO 整个开发过程中从未被接触。

一手来源：[来源1](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3/blob/145f4eb05d07dc4d06352132787a8b4f836e7710/README.md)；[来源2](https://huggingface.co/openai/whisper-large-v3#training-data)。

### Vox-Profile Whisper

当前 SER checkpoint 用 MSP-Podcast v1.12；保留 Whisper-large-v3。Vox-Profile 其他任务的 benchmark 不是此 emotion head 的训练清单。Whisper 的大规模弱标／伪标来源不完整。

一手来源：[来源1](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion/blob/b92dab65151206a603810ec8b72eb528b9dd983c/README.md)；[来源2](https://huggingface.co/openai/whisper-large-v3#training-data)。

### Vox-Profile WavLM

当前 SER checkpoint 用 MSP-Podcast v1.12；保留 WavLM-large，公开骨干训练源为 Libri-Light、GigaSpeech、VoxPopuli。没有逐录音重叠审计，不自动认证其余语料干净。

一手来源：[来源1](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion/blob/621a2ceaaa2408e390e0fc9454e8c29490f689e2/README.md)；[来源2](https://huggingface.co/microsoft/wavlm-large/blob/c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c/README.md)。

### Qwen2.5-Omni 7B

Qwen2.5＋Whisper-large-v3，随后多模态预训练和指令微调；未公布覆盖本项目16语料的完整清单。论文的 MELD 结果是评测证据。

一手来源：[来源1](https://arxiv.org/html/2503.20215v1#S3)；[来源2](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)。

### Qwen2-Audio 7B Instruct

Qwen-7B＋Whisper-large-v3，随后音频预训练、SFT、DPO。论文声明排除评测数据，并列 MELD test；这是带范围的作者声明，不能独立证明全部继承组件从未见过该语料。

一手来源：[来源1](https://arxiv.org/html/2407.10759v1#S3.SS1)；[来源2](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)。

### Gemma 4 E4B

E4B 含 USM 架构的音频编码器，预训练时冻结；随后多模态与指令训练。完整语料清单未公开。January 2025 截止日期、数据集发表时间或通用去污染声明，不足以认证这16个逐项从未使用。

一手来源：[来源1](https://ai.google.dev/gemma/docs/core/model_card_4)；[来源2](https://huggingface.co/google/gemma-4-E4B-it)。

### GPT Realtime 与 Gemini 3.8 Flash

两者都没有公布训练语料来源（OpenAI、Google），所以对 16 个数据集都不画训练斜线；这不表示确认未见。与 [reports/summary.md](../reports/summary.md) 的 Training data 一节一致。

## DailyTalk、EmotionTalk 与 MultiDialog

三套新增数据对八个已评测模型目前都没找到对应的直接或间接训练使用证据；不画训练斜线，但不因此声称确认未见。MultiDialog 逐模型复核见 [数据侧记录](../data/multidialog/training_review.json)。DailyTalk 与 DailyDialog 文本的继承关系另有语义重叠可能；不能把未发现数据集名称等同于排除音频或文本重叠。今后发现新证据，应更新 registry、这份来源记录，并重生成图表与报告，无需重跑预测。

预测 JSON 的 `in_training_data: true` 统一表示参与过训练流程；`training_status`、`training_scope` 与 `training_reason` 仅保留来源判断和具体路径，不用于图表分级。旧预测 JSON 的布尔字段不是永久事实来源。图表、汇总和 viewer 读取当前 registry 来统一标记；推理 protocol、预测值和指标不随来源审核改变。

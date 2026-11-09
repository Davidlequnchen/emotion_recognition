# 训练来源与图表标记核查

核查日期：2026-11-09；证据分级修订：2026-11-10。覆盖当前八个已评测 checkpoint ×十五个采用数据集，预训练、保留骨干与微调均计入。**按证据强弱分级，不要求拿到训练日志。** 技术报告、论文、模型卡和可归属的作者说明，都能提供有效证据；判断还需看它说的是哪个版本、哪个阶段。

**本项目“用过／见过”的最终口径：直接和间接都算。** 只要公开资料支持某语料参与训练流程，包含上游预训练、seed 微调、教师伪标签传播、后续微调，都计入已见；无需 large 再读取一次原始音频。作者明确使用过的开发校准数据也计入已见，并单列阶段。只有评测表出现而没有训练/开发用途的证据，不据此认定训练使用。

配置中的 `training_data`、预测 JSON 的 `in_training_data` 和图表均按同一口径：**C/I 均为“用过”**。不同纹理解释使用方式，不把间接使用排除出去。

| 等级 | 浅显解释 | 图表 / 表格 |
|---|---|---|
| **C：已见，直接有据** | 论文、报告、模型卡或作者明确说：这个 checkpoint 的训练/开发流程或保留骨干直接用过该语料（含明确的开发校准）。足以按来源确认，不代表我们独立审过全部训练文件。 | 斜线 `///` / † |
| **I：已见，上游间接使用** | 单一资料没有直接说最终模型用过，但多份资料支持具体的上游语料→教师→模型传播链。它可指间接影响，不确认最终模型直接训练过原始音频；需写出推理与缺口。 | 点状 `...` / ‡ |
| **P：可能有关联** | 有教师、家族或开发过程的具体关联，但中间缺了语料子集、版本继承或实际适用环节。 | 交叉纹 `xx` / ? |
| **—：暂无语料对应证据** | 知道一些总体训练来源，但没找到本语料与该模型的对应依据。 | 无纹理 / 无标记 |

这四级是**证据分类，不是概率估计**。“直接有据”已经接受公开论文/报告的明确陈述；不另设“绝对确认”，以免暗示审计过作者的训练日志。没有纹理不表示确认没用过；语料级训练使用也不等于本项目 test 音频泄漏。开发校准和教师传播不是直接训练音频，具体范围单独说明。

目前四级框架覆盖 120 项：21 项计为已见：5 项直接有据（含 1 项开发校准）、16 项上游间接使用；99 项暂无对应证据。16 项 I 虽然记录的是上游影响推断，仍全部计入已见；不要求最终模型直接重读原音频。机器可读记录见 [training_provenance.json](training_provenance.json)，实际绘图从 [models.yaml](../config/models.yaml) 读取。

| 数据集 | e2v+ large | MERaLiON | Melati | Vox Whisper | Vox WavLM | Qwen Omni | Qwen Audio | Gemma4 |
|---|---|---|---|---|---|---|---|---|
| CREMA-D | I | C | — | — | — | — | — | — |
| MCAE-SPPS | — | — | — | — | — | — | — | — |
| MESS | — | — | — | — | — | — | — | — |
| RAVDESS | I | I | — | — | — | — | — | — |
| TESS | I | I | — | — | — | — | — | — |
| JL-Corpus | I | I | C | — | — | — | — | — |
| THAI-SER | — | — | — | — | — | — | — | — |
| nEMO | — | — | C | — | — | — | — | — |
| MEAD | I | I | — | — | — | — | — | — |
| MELD | I | C | — | — | — | — | — | — |
| EMOVO | I | I | — | — | — | — | — | — |
| Emozionalmente | I | I | C | — | — | — | — | — |
| SAVEE | I | I | — | — | — | — | — | — |
| DailyTalk | — | — | — | — | — | — | — | — |
| EmotionTalk | — | — | — | — | — | — | — | — |

## emotion2vec+ 为什么属于“间接证据支持”

作者明确说明 seed 用 EmoBox 学术数据、base 用 seed 提供的伪标签；large 模型卡也说明了数据迭代，并把 large 的约 42,526 h 训练数据描述为内部工业伪标注数据。这些是有效证据，九个 EmoBox 相关语料应该可见地保留为 I（上游影响推断），而不是与“没有资料”混在一起。

尚缺的环节是：seed 的 201 h 到底选了 EmoBox 哪些语料/划分，以及每个原始语料如何进入当前 large。官方没有给出逐语料清单，也没有明确 seed→large 的权重继承链。因此这九项计为“已见，经上游间接使用”，不进一步断言 large 直接读过这些原始文件。原始 emotion2vec 论文中的 MELD/MEAD 也单独保留为来源线索，不能自动变成 plus-large 的直接训练事实。

[large 官方模型卡](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/6c303ba987b86b93193de93e34bb2b077a6bedc4/README.md)、[作者关于 large 数据的说明](https://github.com/ddlBoJack/emotion2vec/issues/73)、[作者说明 seed/base 教师关系](https://github.com/ddlBoJack/emotion2vec/issues/46)。

MERaLiON 的报告同样说明使用 seed/base 为另一批音频生成标签，所以额外七项记为 I：这是可能的知识/标签传播，不是该原始语料训练音频被 MERaLiON 直接使用的证据。Melati–nEMO 是作者明确进行的阈值校准，按宽口径计为已见、标 C，并注明本项目 raw argmax 不使用那套阈值。

### 不要混淆两种 teacher 和两种训练阶段

**原始 emotion2vec 的自监督预训练**：student 学习 teacher 的声音表征；teacher 随 student 的权重移动平均（EMA）更新，不是拿情绪类别给其他音频自动标注的 seed 分类器。原论文表 1 / 附录 B.1 列的五个预训练源是 IEMOCAP、MELD、CMU-MOSEI、MEAD、MSP-Podcast v1.8，不能写成整个 EmoBox，更不能据此确认 CREMA-D 用于当前 large 的 SSL。[原论文](https://aclanthology.org/2024.findings-acl.931.pdf)

**emotion2vec+ 的有标签/伪标签迭代**：seed 用带标签的 EmoBox 学术数据微调，再给其他录音产生情绪类别伪标签供后续训练；这是有标签/半监督训练链。后续模型直接接收的是这些新录音和教师给出的答案，不自动等于重新听过教师最初的 CREMA-D 音频。即便这些训练也更新编码器权重，训练目标也不能因此叫“自监督”。[官方模型卡](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)、[作者解释](https://github.com/ddlBoJack/emotion2vec/issues/46)

因此分开记：CREMA-D 属于 EmoBox（直接有据）；EmoBox→seed→伪标注迭代家族关系（直接有据）；具体 CREMA-D→当前 large 的上游影响（间接支持）；CREMA-D 原音频是否直接进入当前 large 的某个训练阶段（未公开）。

## 逐模型证据和限制

### emotion2vec+ large

42,526 h 伪标签／内部工业数据，具体语料未公开。EmoBox、原始 emotion2vec 与教师关系保留为潜在来源，以“间接支持”单独显示，并保留尚未确认最终原始音频使用的限制。

一手来源：[来源1](https://huggingface.co/emotion2vec/emotion2vec_plus_large/blob/6c303ba987b86b93193de93e34bb2b077a6bedc4/README.md)；[来源2](https://github.com/ddlBoJack/emotion2vec/issues/73)；[来源3](https://github.com/ddlBoJack/emotion2vec/issues/46)。

### MERaLiON-SER-v1

SER 训练明确包括 CREMA-D、ESD、MELD、M3ED、SG-ECMT、SGTV。采用的15个中确认 CREMA-D/MELD；ESD 仍有依据但已归档。Whisper-medium 保留的网络预训练来源没有完整逐语料清单。

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

Qwen2.5＋Whisper-large-v3，随后多模态预训练和指令微调；未公布覆盖本项目15语料的完整清单。论文的 MELD 结果是评测证据。

一手来源：[来源1](https://arxiv.org/html/2503.20215v1#S3)；[来源2](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)。

### Qwen2-Audio 7B Instruct

Qwen-7B＋Whisper-large-v3，随后音频预训练、SFT、DPO。论文声明排除评测数据，并列 MELD test；这是带范围的作者声明，不能独立证明全部继承组件从未见过该语料。

一手来源：[来源1](https://arxiv.org/html/2407.10759v1#S3.SS1)；[来源2](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)。

### Gemma 4 E4B

E4B 含 USM 架构的音频编码器，预训练时冻结；随后多模态与指令训练。完整语料清单未公开。January 2025 截止日期、数据集发表时间或通用去污染声明，不足以认证这15个逐项从未使用。

一手来源：[来源1](https://ai.google.dev/gemma/docs/core/model_card_4)；[来源2](https://huggingface.co/google/gemma-4-E4B-it)。

## DailyTalk 与 EmotionTalk

两套新增数据对八个已评测模型目前都没找到对应的直接或间接训练使用证据；均记“暂无语料对应证据”；不因此称为未见数据。DailyTalk 与 DailyDialog 文本的继承关系另有语义重叠可能；不能把未发现数据集名称等同于排除音频或文本重叠。今后发现新证据，应更新 registry、这份来源记录，并重生成图表与报告，无需重跑预测。

为兼容旧消费者，预测 JSON 的 `in_training_data` 在 C/I 为 true、P/— 为 false；它已不能单独表达证据强弱，必须同时读取 `training_status`、`training_scope` 与 `training_reason`。旧预测 JSON 的布尔字段不是永久事实来源。图表、汇总和 viewer 读取当前 registry 来统一标记；推理 protocol、预测值和指标不随来源审核改变。

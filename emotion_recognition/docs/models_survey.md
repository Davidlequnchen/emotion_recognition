# 情绪识别开源模型调研（Emotion Recognition Open-Source Model Survey）

英文版：[models_survey_en.md](models_survey_en.md)（内容相同）。数据集调研见 [datasets_survey.md](datasets_survey.md)。

调研日期：2026-10-28。目标：调研和 emotion2vec+ large 类似、可以下载权重在本地运行的语音情绪识别（SER）模型，选出本项目下一步值得评估的模型。
筛选标准：权重可以下载，有代码或文档说明怎么运行（非商用许可也收录，但会标出）；支持英语（多语言也可以）；大小不限；emotion2vec 全家都收录。

方法：查 Hugging Face 和 ModelScope 的 API（文件列表和大小、许可、创建日期）、模型卡、论文（arXiv）和 GitHub 代码。
参数量：有 safetensors 的取 Hugging Face API 给的数；只有 .pt、.bin、.ckpt 的，用 HTTP range 请求只读出 checkpoint 里每个张量的形状来数，没有下载权重。参数量只算网络本身，文件里多出来的内容（优化器状态、用不到的解码器）写在括号里。
大小以十进制 MB、GB 计；M 是百万参数，B 是十亿参数。年份取权重首次公开或论文发表的年月。标"约"的是推算值，标"未核实"的没能从一手资料确认。
"重叠"指模型的训练数据含本项目评估用的数据集（CREMA-D、MCAE-SPPS、nEMO、MESS、RAVDESS、TESS、JL-Corpus），这些数据集上的结果会偏高。

## 0. 要点

- **emotion2vec+ large 实际是 162M 参数**，不是卡片写的约 300M；1.95 GB 的文件里约三分之二是优化器状态。seed 和 base 都是 93M。
- **emotion2vec+ 之外还有什么：**
  - 维度输出（arousal、valence、dominance）：audEERING、Odyssey 2024 基线、Vox-Profile、MERaLiON-SER、LAION 都有，emotion2vec+ 没有。
  - 其他骨干：最常见的是 WavLM-large（约 316M）；Whisper 编码器（small、medium、large-v3）越来越多；最小的是 17K 参数的 CNN（Wav2Small）。EmoBox 用冻结特征比较骨干，Whisper large-v3 编码器在 23/32 个数据集上第一，其次是 WavLM-large。
  - 许可：可商用的有 Odyssey 基线（MIT）、SpeechBrain 和 SUPERB（Apache-2.0）、MERaLiON-SER（MIT 加署名条款）、Aniemore（MIT）、LAION（CC BY 4.0），emotion2vec+ 本身也不限商用；audEERING、Vox-Profile、Melati、Audio Flamingo 3 只能非商用。
  - 大小：专用 SER 模型大多在 90M 到 0.64B 之间；开源音频大模型 7–35B。
- **不少新模型的标签来自 emotion2vec+**：SenseVoice 的情绪标签是 emotion2vec+ large 打的；MERaLiON-SER 和 C²SER 的部分训练数据也用 emotion2vec+ 打标签。它们不能算独立的对照。
- **和 emotion2vec+ large 最可比的**：MERaLiON-SER-v1（类别正是本项目的 7 类，另有维度输出，可商用）、Melati（7 类，Whisper large-v3，非商用），以及只用 MSP-Podcast 训练的 Vox-Profile（9 类）和 Odyssey 2024 基线（8 类）。
  Melati 的卡片在 7,791 条留出数据（nEMO 和 NonverbalTTS 的 VoxCeleb 部分）上比较过，UAR：emotion2vec+ large 0.522，MERaLiON-SER 0.464，Melati 0.538。
- **建议下一步评估**（接入难度见第 5 节）：
  1. emotion2vec+ seed 和 base：只改 `config/config.yaml`，不用写代码。seed 用 EmoBox 训练，CREMA-D、RAVDESS、TESS、JL-Corpus 上会偏高。
  2. SenseVoiceSmall：同样走 FunASR，只需把输出的情绪 tag 转成类别。可以看从 emotion2vec+ large 蒸馏出来的"学生"差多少。
  3. MERaLiON-SER-v1：类别和本项目一致，用 transformers 加载，要写几十行代码。训练数据含 CREMA-D。
  4. Odyssey 2024 分类基线和 Vox-Profile WavLM 分类模型：和本项目所有评估集都不重叠，是最干净的对照。但它们学的是播客里的自然情绪，Vox-Profile 还说短于 3 秒的片段预测不可靠。

## 1. 分类输出的专用 SER 模型

表 1a 是基本信息，表 1b 是架构、数据和输出，两张表的 # 一一对应。

### 1a 基本信息

| # | 模型 | 机构 | 年份 | 论文 | 下载 | 大小 | 许可 | 运行方式 |
|---|---|---|---|---|---|---|---|---|
| 1 | **emotion2vec base** | 上海交大、复旦、港中文、阿里巴巴 | 2023-12 | [arXiv 2312.15185](https://arxiv.org/abs/2312.15185)（ACL 2024 Findings） | [HF](https://huggingface.co/emotion2vec/emotion2vec_base)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_base) | 93.8M；1.13 GB（含优化器状态） | ① | FunASR；fairseq |
| 2 | **emotion2vec base finetuned** | 同上 | 2024-01 | 同上 | [ModelScope](https://modelscope.cn/models/iic/emotion2vec_base_finetuned)（HF 上没有官方版） | 93.2M；1.12 GB（同上） | ① | FunASR |
| 3 | **emotion2vec+ seed** | 同上 | 2024-05 | 无单独论文，见 [GitHub](https://github.com/ddlBoJack/emotion2vec) | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_seed) | 93.2M；1.12 GB（同上） | ① | FunASR |
| 4 | **emotion2vec+ base** | 同上 | 2024-05 | 同上 | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_base)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_base) | 93.2M（卡片写约 90M）；1.12 GB（同上） | ① | FunASR |
| 5 | **emotion2vec+ large**（本项目在用） | 同上 | 2024-05 | 同上 | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) | **162.1M**（卡片写约 300M）；1.95 GB（同上） | ① | FunASR |
| 6 | **Emotion2Vec-S** | 西北工业大学 ASLP | 2025-02 | [arXiv 2502.18186](https://arxiv.org/abs/2502.18186)（IEEE TASLP） | [HF](https://huggingface.co/ASLP-lab/Emotion2Vec-S) | 93.8M；1.13 GB（含优化器状态） | Apache-2.0 | fairseq（[C²SER 仓库](https://github.com/ASLP-lab/C2SER)） |
| 7 | **MERaLiON-SER-v1** | A*STAR I²R（新加坡） | 2025-10 | [arXiv 2511.04914](https://arxiv.org/abs/2511.04914) | [HF](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) | 309M（卡片）；文件 770.0M、3.08 GB（含用不到的 Whisper 解码器） | MERaLiON Public Licence（MIT 加署名等条款，可商用） | transformers（`trust_remote_code`） |
| 8 | **Vox-Profile Whisper 分类** | USC SAIL 等 | 2025-05 | [arXiv 2505.14648](https://arxiv.org/abs/2505.14648)（DMLR） | [HF](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion) | 约 0.64B（编码器）；文件 1.55B、6.19 GB（含用不到的解码器） | OpenRAIL；**卡片写明不可商用** | [vox-profile-release](https://github.com/tiantiaf0627/vox-profile-release) + loralib |
| 9 | **Vox-Profile WavLM 分类** | 同上 | 2025-05 | 同上 | [HF](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion) | 316.2M；1.26 GB | 同上 | 同上 |
| 10 | **Odyssey 2024 基线（分类）** | UT Dallas MSP Lab、JHU 等 | 2024-03 | [Odyssey 2024](https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf) | [HF](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical) | 318.6M；1.27 GB | MIT | transformers（`trust_remote_code`） |
| 11 | **autrainer MSP-Podcast 4 类** | TUM CHI | 2024-10 | [arXiv 2412.11943](https://arxiv.org/abs/2412.11943)（工具包） | [HF](https://huggingface.co/autrainer/msp-podcast-emo-class-big4-w2v2-l-emo) | 164.8M；0.66 GB | **CC BY-NC-SA 4.0** | autrainer 命令行 |
| 12 | **SpeechBrain wav2vec2 IEMOCAP** | SpeechBrain | 2021-10 | [arXiv 2106.04624](https://arxiv.org/abs/2106.04624)（工具包） | [HF](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | 94.4M；0.38 GB | Apache-2.0 | SpeechBrain（`foreign_class`） |
| 13 | **SpeechBrain 情绪分段**（emotion diarization） | SpeechBrain、Zaion | 2023-07 | [arXiv 2306.12991](https://arxiv.org/abs/2306.12991)（ASRU 2023） | [HF](https://huggingface.co/speechbrain/emotion-diarization-wavlm-large) | 315.5M；1.26 GB | Apache-2.0 | SpeechBrain |
| 14 | **SUPERB ER**（4 个） | SUPERB 团队（S3PRL） | 2021 | [arXiv 2105.01051](https://arxiv.org/abs/2105.01051)（Interspeech 2021） | HF：[wav2vec2-base](https://huggingface.co/superb/wav2vec2-base-superb-er)、[hubert-base](https://huggingface.co/superb/hubert-base-superb-er)、[wav2vec2-large](https://huggingface.co/superb/wav2vec2-large-superb-er)、[hubert-large](https://huggingface.co/superb/hubert-large-superb-er) | base 94.6M、0.38 GB；large 315.7M、1.26 GB | Apache-2.0 | transformers `pipeline` |
| 15 | **ParaCLAP** | TUM CHI、奥格斯堡大学 | 2024-06 | [arXiv 2406.07203](https://arxiv.org/abs/2406.07203)（Interspeech 2024） | [HF](https://huggingface.co/KeiKinn/paraclap)；[GitHub](https://github.com/KeiKinn/ParaCLAP) | 276.3M；1.11 GB | 只写了 "cc"，没说哪一种；音频分支来自 CC BY-NC-SA 模型 | GitHub 代码 |
| 16 | **Aniemore WavLM 跨语言 v1** | Aniemore（俄罗斯开源团队） | 2026-08 | 无（只有模型卡） | [HF](https://huggingface.co/Aniemore/wavlm-emotion-v1-crosslingual) | 316.5M；1.27 GB（另有 int8 0.37 GB、int4 0.22 GB） | MIT | transformers（标准 `AutoModelForAudioClassification`） |
| 17 | **Melati 情绪分类** | Scicom（马来西亚） | 2026-10 | 无（只有模型卡） | [HF](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) | 637.3M；1.28 GB（编码器 fp16） | **仅限非商用研究** | transformers（`trust_remote_code`） |

① emotion2vec 全家：Hugging Face 卡片写 FunASR Model License v1.1（可以使用、修改、分享，须注明出处并保留模型名，不限商用）；ModelScope 页面写 Apache-2.0；GitHub 代码是 MIT。

### 1b 架构、训练数据和输出

| # | 模型 | 架构 | 训练数据 | 语言 | 输出 | 重叠 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | emotion2vec base | data2vec 2.0 式编码器：CNN + 12 层 Transformer（768 维），自监督预训练 | 262 h 无标注：IEMOCAP、MELD、CMU-MOSEI、MEAD、MSP-Podcast v1.8 | 英语 | **只输出特征**（768 维），没有分类头 | 无 | 论文：在 IEMOCAP 上只训线性层，就超过了当时其他预训练模型 |
| 2 | emotion2vec base finetuned | #1 + 线性分类头 | 先用学术 SER 数据微调，再给 15 万 h 中英语音打标签，挑出文本情绪和语音情绪一致、置信度高的 1 万多小时再微调 | 中、英 | 9 类，同 #3 | 学术数据未列出 | 官方建议改用 emotion2vec+ |
| 3 | emotion2vec+ seed | data2vec 2.0 编码器（12 层，768 维）+ 线性分类头 | EmoBox 学术数据 201 h | 多语言（EmoBox 共 14 种） | 9 类：angry、disgusted、fearful、happy、neutral、other、sad、surprised、unknown（包含本项目的 7 类） | **CREMA-D、RAVDESS、TESS、JL-Corpus**（都在 EmoBox 中） | 官方只放了 EmoBox 4 类结果的图，没有表 |
| 4 | emotion2vec+ base | 同 #3 | 过滤后的伪标签数据 4,788 h，来源未公布 | 未公布 | 同 #3 | 同 #3（从 seed 迭代而来）；伪标签数据来源不明 | — |
| 5 | emotion2vec+ large | 同 #3，但 1024 维 | 伪标签数据 42,526 h（从 16 万 h 里选出约 4 万 h），来源未公布 | 未公布 | 同 #3 | 同 #4 | 卡片雷达图（EmoBox 4 类准确率）：多数数据集上高于 ehcalabres（1c 节）、SpeechBrain IEMOCAP（#12）和一个 wav2vec2-base 社区模型 |
| 6 | Emotion2Vec-S | emotion2vec 加类别级半监督对比损失 | C²SER 训练集约 1,216 h（IEMOCAP、ESD、MER2024、BIIC-Podcast、MSP-Podcast v1.11 和内部中文 788 h；内部数据的标签是 emotion2vec+ large 和 GLM-4 结果的交集）+ 约 2,400 h 内部无标注数据 | 中、英 | **只输出特征**，分类头要自己训 | 无 | C²SER-LLM 的声学编码器（第 4 节） |
| 7 | MERaLiON-SER-v1 | Whisper-medium 编码器（冻结，LoRA 已合并）+ 注意力池化 + 改进的 ECAPA-TDNN；分类和维度两个头 | CREMA-D、ESD、MELD、M3ED；SG-ECMT 66,209 条（英、中、马来、泰米尔；标签是 emotion2vec+ seed 和 base 一致的预测，再用本模型二次标注）；SGTV 约 117,000 条、约 120 h（影视对白，人工标注，英、中） | 英（含新加坡英语）、中、马来、泰米尔；泰、印尼、越南有限支持 | 7 类，正是本项目的 7 类；另有 valence、arousal、dominance（0–1） | **CREMA-D** | 论文：7 类 UAR 在 MSP-Podcast、IEMOCAP、MELD 上为 64–70%，比最好的 emotion2vec（seed）高 4–6 个点；MSP-Podcast 和 IEMOCAP 没参与训练 |
| 8 | Vox-Profile Whisper 分类 | Whisper large-v3 编码器（后半层 LoRA）+ 1×1 卷积 + 平均池化 + MLP 头 | MSP-Podcast v1.12（IS2025 挑战赛数据，只用 3–15 秒的片段） | 英语 | 9 类：anger、contempt、disgust、fear、happiness、neutral、sadness、surprise、other；另有次级情绪和 arousal、valence、dominance 头 | 无 | 卡片称训练流程来自 IS2025 SER 挑战赛的最佳系统 SAILER（这里去掉了文本）；短于 3 秒的片段预测不可靠；官方代码在 forward 里写死了 `.cuda()` |
| 9 | Vox-Profile WavLM 分类 | WavLM-large（LoRA）+ 同样的下游头 | 同 #8 | 英语 | 9 类，同 #8 | 无 | 片段长度限制同 #8 |
| 10 | Odyssey 2024 基线（分类） | WavLM-large（微调）+ attentive statistics pooling + MLP | MSP-Podcast release 1.11 的子集（训练集 68,360 段、1,405 人） | 英语 | 8 类：angry、sad、happy、surprise、fear、disgust、contempt、neutral | 无 | 卡片：Test3 macro-F1 0.311 |
| 11 | autrainer 4 类 | audEERING 维度模型（第 2 节 #1）再微调成分类 | MSP-Podcast v1.7（原模型）+ v1.11（微调） | 英语 | 4 类：anger、happiness、neutral、sadness | 无 | 卡片：Test1 UAR 0.650 |
| 12 | SpeechBrain IEMOCAP | wav2vec2-base（微调）+ 平均池化 + 线性层 | IEMOCAP（4 类） | 英语 | 4 类：neu、ang、hap、sad | 无 | 卡片：IEMOCAP 测试集准确率 78.7% |
| 13 | SpeechBrain 情绪分段 | WavLM-large + 逐帧线性分类 | IEMOCAP、RAVDESS、JL-Corpus、ESD、EmoV-DB 拼接成的音频（每段只有一种非中性情绪） | 英语 | 4 类（neutral、happy、sad、angry），带起止时间 | **RAVDESS、JL-Corpus** | ZED 测试集 EDER 29.7% |
| 14 | SUPERB ER | 冻结的 SSL 骨干（各层加权求和）+ 投影 + 平均池化 + 线性层 | IEMOCAP 4 类（第 1 折，Session 1 作测试） | 英语 | 4 类：neu、hap、ang、sad | 无 | 卡片：Session 1 准确率 63.4–67.6% |
| 15 | ParaCLAP | CLAP 双塔：音频是 audEERING 的 wav2vec2，文本是 BERT-base | MSP-Podcast（55,283 条、110 多小时），文本查询由专家声学特征和标签生成 | 英语（卡片还列了德语） | 零样本：和任意文本标签算相似度，可以直接用本项目的 7 个类名 | 无 | — |
| 16 | Aniemore WavLM 跨语言 v1 | WavLM-large + 分类头 | 27,939 条：RESD、Dusha（俄语）、CAMEO（12 个语料，含 CREMA-D、nEMO、JL-Corpus、RAVDESS 等）、IEMOCAP、ASVP-ESD | 俄、英、德、波兰、意、法、西、孟加拉 | 7 类：anger、disgust、enthusiasm、fear、happiness、neutral、sadness（没有 surprised） | **CREMA-D、nEMO、JL-Corpus、RAVDESS** | 卡片自己说明 CAMEO 上的随机切分不是说话人独立的；另有 wav2vec2 版（Apache-2.0） |
| 17 | Melati 情绪分类 | Whisper large-v3 编码器（冻结）+ 各层平均池化后加权混合 + LayerNorm + 256 维隐层 | 28,895 条：EMNS、Emozionalmente、JL-Corpus、MESD、Oreau、RESD、SUBESCO、EmoV-DB、Expresso | 训练语料有 6 种语言（孟加拉、英、法、意、俄、西） | 7 类，正是本项目的 7 类 | **JL-Corpus** | 卡片：留出数据上 UAR 0.538（emotion2vec+ large 0.522，见要点） |

### 1c 社区 checkpoint（只列几个有代表性的）

下载量高，但都是个人上传：训练数据小，测试集多是随机切分（同一说话人或同一句子同时出现在训练和测试里），卡片上的准确率不可信。

| 模型 | 年份 | 大小 | 许可 | 架构 | 训练数据 | 输出 | 重叠 | 问题 |
|---|---|---|---|---|---|---|---|---|
| [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) | 约 2021 | 316.5M；1.27 GB | Apache-2.0 | wav2vec2-large-xlsr-53（英语 ASR 微调版）+ 平均池化 + 分类头 | 只有 RAVDESS（1,440 条） | 8 类（含 calm） | RAVDESS | 卡片准确率 82.2%，没说怎么切分；RAVDESS 是 CC BY-NC-SA，模型却标 Apache-2.0；emotion2vec+ 雷达图的对照之一 |
| [r-f/wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) | 2022-09 | 316.5M；1.27 GB | Apache-2.0 | 同上骨干，但用 `Wav2Vec2ForCTC`：把每帧的 7 维输出平均后取最大 | SAVEE、RAVDESS、TESS（共 4,720 条） | 7 类 | RAVDESS、TESS | 卡片准确率 97.5%，很可能是随机切分造成的泄漏（TESS 只有 2 位说话人） |
| [firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) | 2024-09 | 637.3M；2.55 GB | Apache-2.0 | Whisper large-v3 编码器 + 池化 + 分类头（`WhisperForAudioClassification`） | RAVDESS、SAVEE、TESS、URDU（乌尔都语） | 7 类 | RAVDESS、TESS | 卡片准确率 92.0%，没说切分方式 |
| [Dpngtm/wav2vec2-emotion-recognition](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition) | 2024-10 | 94.6M；0.38 GB | MIT | wav2vec2-base-960h + 分类头 | TESS、CREMA-D、SAVEE、RAVDESS（Kaggle 副本，约 12,000 条） | 7 类（calm 并入 neutral，但 config 里的类名仍是 calm） | CREMA-D、RAVDESS、TESS | 验证集准确率 79.9% |

## 2. 维度输出（arousal、valence、dominance）

MERaLiON-SER-v1（1a #7）和 Vox-Profile 分类模型（#8、#9）也有维度头，这里不重复。这些模型大多用 MSP-Podcast 的维度标签训练，输出约 0–1；本项目的数据集没有维度标签。

### 2a 基本信息

| # | 模型 | 机构 | 年份 | 论文 | 下载 | 大小 | 许可 | 运行方式 |
|---|---|---|---|---|---|---|---|---|
| 1 | **audEERING wav2vec2 MSP-dim** | audEERING | 2022-04 | [arXiv 2203.07378](https://arxiv.org/abs/2203.07378)（IEEE TPAMI 2023） | [HF](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)；ONNX 在 [Zenodo](https://zenodo.org/record/6221127) | 165.3M；0.66 GB | **CC BY-NC-SA 4.0**（商用版要向 audEERING 购买） | transformers + 卡片里约 30 行的模型类 |
| 2 | **Wav2Small 2.0** | audEERING | 2025-08（论文 2024-08） | [arXiv 2408.13920](https://arxiv.org/abs/2408.13920) | [HF](https://huggingface.co/audeering/wav2small) | 16,071（约 17K）；69 KB | **CC BY-NC-SA 4.0** | transformers + 卡片里的模型类 |
| 3 | **Odyssey 2024 基线（维度）** | UT Dallas MSP Lab、JHU 等 | 2024-03 | [Odyssey 2024](https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf) | [HF](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes)（另有单维度的 -Arousal、-Valence、-Dominance） | 318.6M；1.27 GB | MIT | transformers（`trust_remote_code`） |
| 4 | **Vox-Profile WavLM 维度** | USC SAIL 等 | 2025-06 | [arXiv 2505.14648](https://arxiv.org/abs/2505.14648) | [HF](https://huggingface.co/tiantiaf/wavlm-large-msp-podcast-emotion-dim) | 316.0M；1.26 GB | OpenRAIL；**卡片写明不可商用** | vox-profile-release |
| 5 | **Vox-Profile Whisper 维度** | 同上 | 2025-06 | 同上 | [HF](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion-dim) | 约 0.64B（编码器）；文件 1.54B、6.17 GB | 同上 | 同上 |
| 6 | **Empathic Insight Voice**（Small、Large） | LAION | 2025-05 | [arXiv 2506.09827](https://arxiv.org/abs/2506.09827)（EmoNet-Voice） | [HF Small](https://huggingface.co/laion/Empathic-Insight-Voice-Small)；[HF Large](https://huggingface.co/laion/Empathic-Insight-Voice-Large) | 每个维度一个头，共 55 个：Small 每个 73.7M（共 16.2 GB），Large 每个 147.5M（共 32.4 GB）；另需 BUD-E-Whisper（Whisper small，241.7M，只用编码器） | CC BY 4.0 | 卡片的 Colab 代码 |
| 7 | **VoiceCLAP-Commercial + 属性头** | LAION | 2026-06（头 2026-07） | 无单独论文 | [HF 编码器](https://huggingface.co/laion/voiceclap-commercial)；[HF 属性头](https://huggingface.co/laion/voiceclap-commercial-attribute-heads) | 编码器 113.2M、0.45 GB；61 个头共 17 MB | 编码器 CC BY 4.0，头 Apache-2.0；训练数据都允许商用 | PyTorch（仓库里的代码） |

### 2b 架构、训练数据和输出

| # | 模型 | 架构 | 训练数据 | 语言 | 输出 | 重叠 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | audEERING wav2vec2 MSP-dim | wav2vec2-large-robust 剪到 12 层 + 平均池化 + 回归头 | MSP-Podcast v1.7 | 英语 | arousal、dominance、valence（约 0–1）；也可取池化后的特征 | 无 | autrainer（1a #11）和 ParaCLAP（1a #15）都基于它 |
| 2 | Wav2Small 2.0 | CNN：用卷积算 log-mel，再接 VGG 式卷积层 | 蒸馏：老师是 audEERING 模型和 Odyssey 2024 维度赛道冠军 WavLM 的平均输出，在 MSP-Podcast 音频和无标注、非语音音频上训练，不用人工标签 | 英语 | arousal、dominance、valence（0–1） | 无 | 论文版 72K 参数，量化 ONNX 约 120 KB |
| 3 | Odyssey 2024 基线（维度） | 同 1a #10 | 同 1a #10 | 英语 | arousal、dominance、valence（约 0–1） | 无 | 卡片：Test3 CCC 0.405–0.577 |
| 4 | Vox-Profile WavLM 维度 | WavLM-large（LoRA）+ 卷积 + 池化 + 回归头 | MSP-Podcast v1.12（3–15 秒片段） | 英语 | arousal、valence、dominance（0–1） | 无 | 卡片称训练流程来自 IS2025 挑战赛维度任务第一名（[arXiv 2506.10930](https://arxiv.org/abs/2506.10930)） |
| 5 | Vox-Profile Whisper 维度 | Whisper large-v3 编码器（LoRA）+ 同上 | 同上 | 英语 | 同上 | 无 | 同上 |
| 6 | Empathic Insight Voice | Whisper small 编码器的完整输出（1500×768）→ 投影 → 每个维度一个 MLP | 约 5,000 h 合成表演语音（LAION's Got Talent）+ 约 5,000 h 网络 vlog；标签是 Gemini Flash 2.0 打的分 | 多语言（具体语言未核实） | 40 种情绪的强度 + 15 个属性（valence、arousal、年龄、性别等） | 未见重叠 | 卡片提醒：主要用合成语音训练，对真实语音的泛化还要验证；编码器卡片写 BUD-E-Whisper，示例代码用的是 mkrausio/EmoWhisper-AnS-Small-v0.1（没有许可声明） |
| 7 | VoiceCLAP-Commercial + 属性头 | CLAP 式双塔（110M），冻结的 768 维语音向量 + 每个属性一个小 MLP | 编码器：Emilia-YODAS（CC BY 子集）、LAION's Got Talent、Majestrino；头：296,422 条（TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets） | 英、德、法、西、中 | 55 个情绪和感知维度（Anger、Valence、Arousal 等）+ 音质、年龄、性别、语速 | 未见重叠 | 卡片：留出集上 61 个维度的 Pearson r 中位数 0.683；头的标签来源未核实 |

## 3. 自带情绪标签的语音模型

| 模型 | 机构、年份 | 论文；下载 | 大小 | 许可 | 架构 | 训练数据 | 情绪输出 | 运行方式 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| **SenseVoiceSmall** | 阿里巴巴通义实验室（FunAudioLLM），2024-07 | [arXiv 2407.04051](https://arxiv.org/abs/2407.04051)；[HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)、[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) | 234.0M；0.94 GB | 同 ①（HF 写 FunASR 协议，ModelScope 写 Apache-2.0）；代码 MIT | 非自回归 SAN-M 编码器 + CTC；输入前拼上语种、情绪、事件、ITN 四个任务向量 | 约 30 万 h（中、粤、英、日、韩）；情绪标签是 emotion2vec+ large 打的伪标签（3,000 万条） | 每段一个情绪 tag，同时输出转写、语种和声音事件。后处理代码列了 7 种情绪 tag，模型代码的 `emo_dict` 只有 happy、sad、angry、neutral，实际会输出哪些类未核实 | FunASR（本项目现用）；sherpa-onnx、GGUF | 只给 tag，不给各类概率；情绪是从 emotion2vec+ large 蒸馏来的；重叠未知 |
| **BUD-E-Whisper** | LAION，2025-05（V1.1 2025-11，V1.2 2026-04） | 无单独论文（EmoNet-Voice 项目）；[HF](https://huggingface.co/laion/BUD-E-Whisper) | 241.7M；0.97 GB | CC BY 4.0 | Whisper small 微调成直接生成情绪描述 | LAION's Got Talent + 约 5,000 h vlog；描述由 Gemini Flash 2.0 的打分转成模板句再改写 | 一段自由文本的情绪描述 | transformers（Whisper `generate`） | 编码器也给 Empathic Insight 用（第 2 节）；另有输出声音属性标签的 [voice-tagging-whisper](https://huggingface.co/laion/voice-tagging-whisper)（Apache-2.0，2025-12） |

## 4. 能识别情绪的开源音频大模型（简表）

和 emotion2vec+ 不同：它们按提示词生成文字答案，不给各类概率；参数量大，要在 CUDA 服务器上跑。表中数字来自不同来源和评测协议，不能直接相互比较。Kimi-Audio、OSUM、Audio Flamingo、SALMONN 和 Qwen 系列用过哪些情绪数据，见 [datasets_survey.md](datasets_survey.md) 第 4 节。

| 模型 | 机构 | 年份 | 论文 | 下载 | 大小 | 许可 | 情绪方面的依据 | 运行方式 |
|---|---|---|---|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct | 阿里巴巴 Qwen | 2024-07 | [arXiv 2407.10759](https://arxiv.org/abs/2407.10759) | [HF](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | 8.4B | Apache-2.0 | Qwen2.5-Omni 卡片表：MELD 0.553 | transformers |
| Qwen2.5-Omni-7B（3B） | 阿里巴巴 Qwen | 2025-03 | [arXiv 2503.20215](https://arxiv.org/abs/2503.20215) | [HF 7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)；[HF 3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 10.7B（含语音生成部分）；3B 版 5.5B | 7B：Apache-2.0；**3B：Qwen Research License（非商用）** | 卡片：MELD 0.570（3B 0.558） | transformers |
| Qwen3-Omni-30B-A3B-Instruct | 阿里巴巴 Qwen | 2025-09 | [arXiv 2509.17765](https://arxiv.org/abs/2509.17765) | [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)；另有 [Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) | 35.3B（MoE，每个 token 约 3B 激活） | Apache-2.0 | 技术报告没有语音情绪基准；MiniCPM-o 4.5 卡片的第三方结果 MELD 56.8；Captioner 版会描述说话人情绪 | transformers、vLLM |
| Kimi-Audio-7B-Instruct | 月之暗面 Moonshot AI | 2025-04 | [arXiv 2504.18425](https://arxiv.org/abs/2504.18425) | [HF](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | 9.8B | MIT（源自 Qwen2.5-7B 的部分为 Apache-2.0） | [GitHub](https://github.com/MoonshotAI/Kimi-Audio)：MELD 59.13；SFT 用了 RAVDESS（**重叠**） | Kimi-Audio 代码 |
| Step-Audio 2 mini | 阶跃星辰 StepFun | 2025-08 | [arXiv 2507.16632](https://arxiv.org/abs/2507.16632) | [HF](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | 8.3B | Apache-2.0 | 自建 StepEval-Audio-Paralinguistic 的情绪项 82 | 官方代码 |
| OSUM（OSUM-EChat） | 西北工业大学 ASLP | 2025-01（EChat 2025-08） | [arXiv 2501.13306](https://arxiv.org/abs/2501.13306)；[EChat](https://arxiv.org/abs/2508.09600) | [HF](https://huggingface.co/ASLP-lab/OSUM)；[HF EChat](https://huggingface.co/ASLP-lab/OSUM-EChat) | 约 8.0B（16.0 GB bf16；Whisper 编码器 + Qwen2-7B） | Apache-2.0 | SER 是训练任务之一，论文在 IEMOCAP、MSP-IMPROV 等上测试；EChat 加了共情对话 | 官方代码 |
| Audio Flamingo 3（AF-Next） | NVIDIA、马里兰大学 | 2025-07（AF-Next 2026-04） | [arXiv 2507.08128](https://arxiv.org/abs/2507.08128)；[AF-Next](https://arxiv.org/abs/2604.10905) | [HF](https://huggingface.co/nvidia/audio-flamingo-3-hf)；[HF AF-Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) | 8.3B | **NVIDIA OneWay Noncommercial License（非商用）** | 训练含 MSP-Podcast、MELD、JL-Corpus、TESS 等情绪数据（**重叠：JL-Corpus、TESS**），在 IEMOCAP 上测试 | transformers |
| SALMONN | 清华大学、字节跳动 | 2023-10 | [arXiv 2310.13289](https://arxiv.org/abs/2310.13289) | [HF](https://huggingface.co/tsinghua-ee/SALMONN) | 适配权重 0.40 GB；另需 Whisper large-v2、BEATs、Vicuna-13B v1.1 | 适配权重 Apache-2.0；Vicuna v1.1 有 LLaMA 许可限制（细节未核实） | 用 IEMOCAP Session 1–4 训练、Session 5 测试 | 官方代码 |
| MiniCPM-o 2.6（4.5） | OpenBMB、面壁智能 | 2025-01（4.5：2026-02） | [arXiv 2408.01800](https://arxiv.org/abs/2408.01800)（MiniCPM-V）；[4.5](https://arxiv.org/abs/2604.27393) | [HF 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)；[HF 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 8.7B（4.5：9.4B） | Apache-2.0 | 卡片：MELD 52.4（4.5：60.2） | transformers（`trust_remote_code`） |
| MERaLiON-2-10B（3-10B） | A*STAR | 2025-05（3-10B：2026-03） | [arXiv 2412.09818](https://arxiv.org/abs/2412.09818) | [HF](https://huggingface.co/MERaLiON/MERaLiON-2-10B)；[HF 3-10B](https://huggingface.co/MERaLiON/MERaLiON-3-10B) | 10.1B（Whisper large-v3 编码器 + Gemma-2-9B） | MERaLiON 许可 v3（MIT、Whisper 许可和 Gemma 条款） | 卡片：自家基准的 "Emotion Recognition" 63.7；MERaLiON-SER 论文里不如 MERaLiON-SER | transformers |
| C²SER-LLM | 西北工业大学 ASLP | 2025-02 | [arXiv 2502.18186](https://arxiv.org/abs/2502.18186)（IEEE TASLP） | [HF](https://huggingface.co/ASLP-lab/C2SER-LLM) | 约 8.0B（16.0 GB bf16；另需下载 Qwen2-7B） | Apache-2.0 | 专为 SER 训练：先描述内容和说话风格，再判断情绪；7 类；论文另发布 Emo-Emilia 测试集 | 官方代码（Emotion2Vec-S 特征 + Whisper） |
| MOSS-Audio-8B-Instruct | 模思智能（MOSI.AI）、OpenMOSS、上海创智学院 | 2026-04 | 无（"即将发布"） | [HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | 9.1B | Apache-2.0 | 卡片列出说话人和情绪分析能力 | 官方代码 |

## 5. 接入本项目的难度

`scripts/emotion2vec/evaluate.py` 用 FunASR 的 `AutoModel` 加载模型，从 `tokens.txt` 读 9 个类别，再取 `generate()` 返回的各类概率（`scores`），对应到每个数据集的类别。其他框架的模型要自己写加载代码。

| 难度 | 模型 | 要做的事 |
|---|---|---|
| 只改配置 | emotion2vec+ seed、base；emotion2vec base finetuned | 按 `models/README.md` 在 `config/config.yaml` 的 `models:` 下注册、下载，再改 `model:`。三者都输出同样的 9 类 |
| 小改 | SenseVoiceSmall | 也用 FunASR 加载，但 `generate()` 返回带 tag 的文本：要把 tag 转成类别，拿不到各类概率（除非读情绪 token 的 CTC logits） |
| 写一个 transformers 加载器（几十行） | MERaLiON-SER-v1、Odyssey 2024 基线、Melati（要 `trust_remote_code`）；Aniemore、SUPERB 和 1c 节的社区模型（标准 `AutoModelForAudioClassification`） | 写一个返回各类概率的 `Predictor`，再把模型的类名对到本项目的 7 类（如 `ang` → angry，`Happiness` → happy）。项目锁定 transformers 5.19.0，为 4.x 写的远程代码可能要改（未测试） |
| 要装别的包或改代码 | Vox-Profile（vox-profile-release、loralib；Whisper 版要去掉写死的 `.cuda()` 才能在 Mac 上跑）、SpeechBrain 的两个模型、autrainer、ParaCLAP；Emotion2Vec-S 和 emotion2vec base 只出特征，要先训分类头 | 新依赖和 FunASR 1.4.16、transformers 5.19.0 能否共存未测试 |
| 要另写评估 | 第 2 节的维度模型；第 4 节的音频大模型 | 本项目的数据集只有类别标签，维度输出要换分析方法（例如看各情绪在 arousal、valence 上的分布）；音频大模型可以沿用 `scripts/llm_eval/` 的提示词和解析，但要在本地 GPU 上推理 |

## 6. 注意事项

- **非商用或限制性许可**：audEERING 的两个模型和 autrainer（CC BY-NC-SA 4.0）、Vox-Profile（OpenRAIL，但卡片写明不可商用）、Melati（仅限研究）、Audio Flamingo 3 和 AF-Next（NVIDIA 非商用）、Qwen2.5-Omni-3B（Qwen Research）。ParaCLAP 的许可没写清（只写 "cc"），音频分支又来自 NC 模型。
  社区模型常把用 NC 数据（如 RAVDESS）训练的模型标成 Apache-2.0 或 MIT，不可靠。SALMONN 依赖的 Vicuna-13B v1.1 受 LLaMA 许可限制（细节未核实）。
- **与本项目评估集重叠**（这些数据集上的结果偏高）：
  - CREMA-D：emotion2vec+ seed、base、large（EmoBox）、MERaLiON-SER、Aniemore、Dpngtm。
  - RAVDESS：emotion2vec+ 三个版本、SpeechBrain 情绪分段、Aniemore、1c 节全部 4 个社区模型、Kimi-Audio。
  - TESS：emotion2vec+ 三个版本、r-f、firdhokk、Dpngtm、Audio Flamingo 3。
  - JL-Corpus：emotion2vec+ 三个版本、SpeechBrain 情绪分段、Aniemore、Melati、Audio Flamingo 3。
  - nEMO：Aniemore（通过 CAMEO）。MCAE-SPPS 和 MESS：没有找到用过它们的模型。
  - 只用 MSP-Podcast 或 IEMOCAP 训练的模型（Odyssey、Vox-Profile、audEERING、Wav2Small、autrainer、ParaCLAP、SUPERB、SpeechBrain IEMOCAP）和所有评估集都不重叠。
- **卡片写的参数量和文件不符**：emotion2vec+ large（卡片约 300M，实际 162M）；MERaLiON-SER（卡片 309M，文件 770M，多出的是用不到的 Whisper 解码器）；Vox-Profile 的 Whisper 版（文件 1.5B，同样含解码器）。emotion2vec 全家的 .pt 文件里约三分之二是优化器状态。
- **伪标签链**：SenseVoice（emotion2vec+ large）、MERaLiON-SER（emotion2vec+ seed 和 base）、C²SER 的内部数据（emotion2vec+ large）都用 emotion2vec+ 打过标签，emotion2vec+ base 和 large 自己的伪标签数据来源也没公布。这些模型的错误可能和 emotion2vec+ 相关。
- **领域和片段长度**：MSP-Podcast 系模型学的是播客里的自然情绪，本项目是表演语音。Vox-Profile 训练时去掉了短于 3 秒的片段，卡片说这样的片段预测不可靠；本项目的片段大多 1–5 秒。
- **公开结果不能直接比较**：数据集、类别数、指标（准确率、UAR、F1、CCC、EDER）和切分都不同；表里的数字只说明作者报告过什么。
- **Hugging Face 的创建日期**：2022-03-02 是旧仓库迁移时的统一日期，不是发布日期；SUPERB、SpeechBrain IEMOCAP、ehcalabres 的年份按卡片和依赖版本推算。

## 7. 主要来源

- emotion2vec：https://github.com/ddlBoJack/emotion2vec 、https://arxiv.org/abs/2312.15185 、https://huggingface.co/emotion2vec 、https://modelscope.cn/models/iic/emotion2vec_base_finetuned （以及 iic/emotion2vec_base、iic/emotion2vec_plus_seed、iic/emotion2vec_plus_base、iic/emotion2vec_plus_large）；FunASR 模型协议：https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE ；EmoBox：https://arxiv.org/abs/2406.07162
- Emotion2Vec-S、C²SER：https://arxiv.org/abs/2502.18186 、https://github.com/ASLP-lab/C2SER 、https://huggingface.co/ASLP-lab/Emotion2Vec-S 、https://huggingface.co/ASLP-lab/C2SER-LLM
- MERaLiON-SER：https://arxiv.org/abs/2511.04914 、https://huggingface.co/MERaLiON/MERaLiON-SER-v1 、许可：https://huggingface.co/MERaLiON/MERaLiON-SER-v1/resolve/main/MERaLiON-Public-Licence-SER-V1.pdf
- Vox-Profile：https://arxiv.org/abs/2505.14648 、https://github.com/tiantiaf0627/vox-profile-release 、https://arxiv.org/abs/2506.10930
- Odyssey 2024：https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf 、https://github.com/MSP-UTD/MSP-Podcast_Challenge 、https://huggingface.co/3loi
- autrainer：https://arxiv.org/abs/2412.11943 、https://github.com/autrainer/autrainer ；ParaCLAP：https://arxiv.org/abs/2406.07203 、https://github.com/KeiKinn/ParaCLAP
- SpeechBrain：https://arxiv.org/abs/2106.04624 、https://arxiv.org/abs/2306.12991 ；SUPERB：https://arxiv.org/abs/2105.01051
- Aniemore：https://huggingface.co/Aniemore/wavlm-emotion-v1-crosslingual ；CAMEO：https://arxiv.org/abs/2505.11051 、https://huggingface.co/datasets/amu-cai/CAMEO ；Melati：https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3
- audEERING：https://arxiv.org/abs/2203.07378 、https://arxiv.org/abs/2408.13920 、https://zenodo.org/record/6221127
- LAION：https://arxiv.org/abs/2506.09827 、https://huggingface.co/laion/Empathic-Insight-Voice-Small 、https://huggingface.co/laion/BUD-E-Whisper 、https://huggingface.co/laion/voiceclap-commercial
- SenseVoice：https://arxiv.org/abs/2407.04051 、https://huggingface.co/FunAudioLLM/SenseVoiceSmall 、https://github.com/FunAudioLLM/SenseVoice （现跳转到 QwenAudio/SenseVoice）
- 音频大模型：Qwen2-Audio https://arxiv.org/abs/2407.10759 ；Qwen2.5-Omni https://arxiv.org/abs/2503.20215 ；Qwen3-Omni https://arxiv.org/abs/2509.17765 ；Kimi-Audio https://arxiv.org/abs/2504.18425 、https://github.com/MoonshotAI/Kimi-Audio ；Step-Audio 2 https://arxiv.org/abs/2507.16632 ；OSUM https://arxiv.org/abs/2501.13306 ；OSUM-EChat https://arxiv.org/abs/2508.09600 ；Audio Flamingo 3 https://arxiv.org/abs/2507.08128 ；AF-Next https://arxiv.org/abs/2604.10905 ；SALMONN https://arxiv.org/abs/2310.13289 ；MiniCPM-V https://arxiv.org/abs/2408.01800 ；MiniCPM-o 4.5 https://arxiv.org/abs/2604.27393 ；MERaLiON-AudioLLM https://arxiv.org/abs/2412.09818 ；MOSS-Audio https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct
- 查询接口：Hugging Face `https://huggingface.co/api/models/<id>?blobs=true`；ModelScope `https://modelscope.cn/api/v1/models/<id>`

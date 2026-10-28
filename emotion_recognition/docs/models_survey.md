# 情绪识别开源模型调研（Emotion Recognition Open-Source Model Survey）

英文版：[models_survey_en.md](models_survey_en.md)（内容相同）。数据集调研见 [datasets_survey.md](datasets_survey.md)。

调研日期：2026-10-28。目标：调研和 emotion2vec+ large 类似、可以下载权重在本地运行的语音情绪识别（SER）模型，选出本项目下一步值得评估的模型。
筛选标准：权重可以下载，有代码或文档说明怎么运行（非商用许可也收录，但会标出）；支持英语（多语言也可以）；大小不限；emotion2vec 全家都收录。

方法：查 Hugging Face 和 ModelScope 的 API（文件列表和大小、许可、创建日期）、模型卡、论文（arXiv）和 GitHub 代码。
参数量：有 safetensors 的取 Hugging Face API 给的数，或从文件头读出每个张量的形状来数；只有 .pt、.bin、.ckpt 的，用 HTTP range 请求只读出 checkpoint 里每个张量的形状来数。都没有下载权重。参数量只算网络本身，文件里多出来的内容（优化器状态、用不到的解码器）写在括号里。
大小以十进制 MB、GB 计。参数量：M = million（百万）参数，B = billion 参数；不到 1B 的保留 1 位小数，1B 以上保留 2 位小数，来源只给约数的标"约"。年份取权重首次公开或论文发表的年月。标"未核实"的没能从一手资料确认。

## 0. 要点

- **emotion2vec+ large 实际是 162.1M 参数**，不是卡片写的约 300M；1.95 GB 的文件里约三分之二是优化器状态。seed 和 base 都是 93.2M。
- **emotion2vec+ 之外还有什么：**
  - 维度输出（arousal、valence、dominance）：audEERING、Odyssey 2024 基线、Vox-Profile、MERaLiON-SER、LAION 都有，emotion2vec+ 没有。
  - 其他骨干：最常见的是 WavLM-large（约 316M）；Whisper 编码器（small、medium、large-v3）越来越多；最小的是只有 0.016M 参数的 CNN（Wav2Small）。EmoBox 用冻结特征比较骨干，Whisper large-v3 编码器在 23/32 个数据集上第一，其次是 WavLM-large。
  - 许可：可商用的有 Odyssey 基线（MIT）、SpeechBrain 和 SUPERB（Apache-2.0）、MERaLiON-SER（MIT 加署名条款）、Aniemore（MIT）、LAION（CC BY 4.0），emotion2vec+ 本身也不限商用；audEERING、Vox-Profile、Melati、Audio Flamingo 3 只能非商用。
  - 大小：专用 SER 模型大多在 93.2M 到 639.9M 之间；开源音频大模型 5.54B 到 35.26B。
- **不少新模型的标签来自 emotion2vec+**：SenseVoice 的情绪标签是 emotion2vec+ large 打的；MERaLiON-SER 和 C²SER 的部分训练数据也用 emotion2vec+ 打标签。它们不能算独立的对照。
- **输入长度**：基于 Whisper 编码器的模型（MERaLiON-SER-v1、Melati、Vox-Profile Whisper 等）按固定窗口计算（30 s，Vox-Profile Whisper 为 15 s），短片段补静音，长音频被截断；emotion2vec 和 wav2vec2、WavLM、HuBERT 类模型按实际长度计算。长通话要先切段再合并预测（见表 1a 前的提示和各表的“输入音频”列）。
- **和 emotion2vec+ large 最可比的**：MERaLiON-SER-v1（类别正是本项目的 7 类，另有维度输出，可商用）、Melati（7 类，Whisper large-v3，非商用），以及只用 MSP-Podcast 训练的 Vox-Profile（9 类）和 Odyssey 2024 基线（8 类）。
  Melati 的卡片在 7,791 条留出数据（nEMO 和 NonverbalTTS 的 VoxCeleb 部分）上比较过，UAR：emotion2vec+ large 0.522，MERaLiON-SER 0.464，Melati 0.538。
- **建议下一步评估**（表中标 ⭐ 的模型；接入难度见第 5 节）：
  1. MERaLiON-SER-v1：类别正是本项目的 7 类，另有维度输出，可商用；用 transformers 加载，要写几十行代码。
  2. Melati：7 类，卡片上的留出结果略高于 emotion2vec+ large；非商用，也用 transformers 加载。
  3. Vox-Profile 分类模型（Whisper、WavLM）和 Odyssey 2024 分类基线：只用 MSP-Podcast（播客里的自然情绪）训练。Vox-Profile 是 IS2025 挑战赛最佳方案的纯语音版，但非商用、要装它的包，卡片还说短于 3 秒的片段预测不可靠；Odyssey 基线是 MIT 许可，最容易加载。
  4. 维度模型（audEERING、Vox-Profile WavLM 维度、Odyssey 2024 维度）：输出 arousal、valence、dominance，要另写分析。
  - emotion2vec+ seed、base 和 SenseVoiceSmall 接入最省事（只改配置或小改），但都和 emotion2vec+ large 同源，作对照意义不大，所以没标 ⭐。

## 1. 分类输出的专用 SER 模型

表 1a 是基本信息和公开报告的效果，表 1b 是架构、数据、输出和输入音频，两张表的 # 一一对应。

⭐ = 推荐优先评估：公开报告的效果好，输出类别接近本项目的 7 类，能下载并运行，再参考许可；只有推荐的模型名字加粗。

指标：WA 是准确率，UA（UAR）是各类召回率的平均（类别不均衡时更公平），macro-F1 是各类 F1 的平均，CCC 是维度预测和人工评分的一致程度（1 最好）。各行数字来自不同的数据集和评测方法，不能横向比较；只有 emotion2vec+ large 在本项目的数据集上实测过。

### 1a 基本信息和公开报告的效果

#### ⚠️ 输入窗口：基于 Whisper 编码器的模型（MERaLiON-SER-v1、Melati、Vox-Profile Whisper 等）每条都按固定窗口计算（30 s，Vox-Profile Whisper 为 15 s）：短片段补静音，长音频被截断；1 s 和 30 s 的片段算得一样慢。emotion2vec 和 wav2vec2 / WavLM / HuBERT 类模型按实际长度计算。

所以呼叫中心里较长的通话轮次要先切成不超过窗口长度的片段，再把各片段的预测合并；每个模型的情况见表 1b、1c、2b 和第 3、4 节表格的“输入音频”列。

| # | 模型 | 机构 | 年份 | 公开报告的效果 | 论文 | 下载 | 大小 | 许可 | 运行方式 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ⭐ **emotion2vec+ large**（本项目在用） | 上海交大、复旦、港中文、阿里巴巴 | 2024-05 | 本项目实测 6 个数据集，准确率 40.8–92.5%（[报告](../reports/emotion2vec_plus_large/summary.md)）；Melati 卡片的留出数据，7 类 UAR 0.522 | 无单独论文，见 [GitHub](https://github.com/ddlBoJack/emotion2vec) | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) | 162.1M（卡片写约 300M）；1.95 GB（含优化器状态） | ① | FunASR |
| 2 | ⭐ **MERaLiON-SER-v1** | A*STAR I²R（新加坡） | 2025-10 | MSP-Podcast、IEMOCAP、MELD 上 7 类 UAR 64–70%（论文；前两个没参与训练）；Melati 卡片的留出数据，UAR 0.464 | [arXiv 2511.04914](https://arxiv.org/abs/2511.04914) | [HF](https://huggingface.co/MERaLiON/MERaLiON-SER-v1) | 313.4M（不含解码器；卡片写 309M）；文件 770.0M、3.08 GB（含用不到的 Whisper 解码器） | MERaLiON Public Licence（MIT 加署名等条款，可商用） | transformers（`trust_remote_code`） |
| 3 | ⭐ **Melati 情绪分类** | Scicom（马来西亚） | 2026-10 | nEMO 和 NonverbalTTS 的 VoxCeleb 部分（7,791 条，没参与训练）上 7 类 UAR 0.538，其中 nEMO 0.559（卡片） | 无（只有模型卡） | [HF](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3) | 637.3M；1.28 GB（编码器 fp16） | 仅限非商用研究 | transformers（`trust_remote_code`） |
| 4 | ⭐ **Vox-Profile Whisper 分类** | USC SAIL 等 | 2025-05 | MSP-Podcast v1.12 开发集 macro-F1 0.416（论文表 4） | [arXiv 2505.14648](https://arxiv.org/abs/2505.14648)（DMLR） | [HF](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion) | 639.9M（不含用不到的解码器）；文件 1.55B、6.19 GB | OpenRAIL，但卡片写明不可商用 | [vox-profile-release](https://github.com/tiantiaf0627/vox-profile-release) + loralib |
| 5 | ⭐ **Vox-Profile WavLM 分类** | 同 #4 | 2025-05 | MSP-Podcast v1.12 开发集 macro-F1 0.406（论文表 4） | 同 #4 | [HF](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion) | 316.2M；1.26 GB | 同 #4 | 同 #4 |
| 6 | ⭐ **Odyssey 2024 基线（分类）** | UT Dallas MSP Lab、JHU 等 | 2024-03 | MSP-Podcast 1.11 测试集 8 类 macro-F1 0.311（论文、卡片） | [Odyssey 2024](https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf) | [HF](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Categorical) | 318.6M；1.27 GB | MIT | transformers（`trust_remote_code`） |
| 7 | emotion2vec+ base | 同 #1 | 2024-05 | EmoBox 测试切分 4 类：32 个语料平均 UA 80.0%，其中 MSP-Podcast UA 40.2%（[GitHub 表](https://github.com/ddlBoJack/emotion2vec/blob/main/src/emotion2vec%2Bperformance.png)） | 同 #1 | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_base)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_base) | 93.2M（卡片写约 90M）；1.12 GB（含优化器状态） | ① | FunASR |
| 8 | emotion2vec+ seed | 同 #1 | 2024-05 | EmoBox 测试切分 4 类：平均 UA 78.0%，其中 MSP-Podcast UA 47.4%（GitHub 表） | 同 #1 | [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_seed)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_seed) | 93.2M；1.12 GB（含优化器状态） | ① | FunASR |
| 9 | emotion2vec base finetuned | 同 #1 | 2024-01 | 未公布 | 同 #16 | [ModelScope](https://modelscope.cn/models/iic/emotion2vec_base_finetuned)（HF 上没有官方版） | 93.2M；1.12 GB（含优化器状态） | ① | FunASR |
| 10 | Aniemore WavLM 跨语言 v1 | Aniemore（俄罗斯开源团队） | 2026-08 | Dusha 播客（自然俄语）4 类 UA 0.647；CAMEO 测试 7 类 UA 0.517（卡片；卡片说 CAMEO 的切分不是说话人独立的） | 无（只有模型卡） | [HF](https://huggingface.co/Aniemore/wavlm-emotion-v1-crosslingual) | 316.5M；1.27 GB（另有 int8 0.37 GB、int4 0.22 GB） | MIT | transformers（标准 `AutoModelForAudioClassification`） |
| 11 | autrainer MSP-Podcast 4 类 | TUM CHI | 2024-10 | MSP-Podcast Test1 4 类 UAR 0.650（卡片） | [arXiv 2412.11943](https://arxiv.org/abs/2412.11943)（工具包） | [HF](https://huggingface.co/autrainer/msp-podcast-emo-class-big4-w2v2-l-emo) | 164.8M；0.66 GB | CC BY-NC-SA 4.0 | autrainer 命令行 |
| 12 | SpeechBrain wav2vec2 IEMOCAP | SpeechBrain | 2021-10 | IEMOCAP 4 类准确率 78.7%（卡片；多次运行平均 75.3%）；emotion2vec+ 卡片的雷达图里，多数数据集上低于 emotion2vec+ large | [arXiv 2106.04624](https://arxiv.org/abs/2106.04624)（工具包） | [HF](https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP) | 94.4M；0.38 GB | Apache-2.0 | SpeechBrain（`foreign_class`） |
| 13 | SUPERB ER（4 个） | SUPERB 团队（S3PRL） | 2021 | IEMOCAP 4 类（Session 1）准确率 63.4–67.6%（卡片） | [arXiv 2105.01051](https://arxiv.org/abs/2105.01051)（Interspeech 2021） | HF：[wav2vec2-base](https://huggingface.co/superb/wav2vec2-base-superb-er)、[hubert-base](https://huggingface.co/superb/hubert-base-superb-er)、[wav2vec2-large](https://huggingface.co/superb/wav2vec2-large-superb-er)、[hubert-large](https://huggingface.co/superb/hubert-large-superb-er) | base 94.6M、0.38 GB；large 315.7M、1.26 GB | Apache-2.0 | transformers `pipeline` |
| 14 | SpeechBrain 情绪分段（emotion diarization） | SpeechBrain、Zaion | 2023-07 | ZED（真实录音）上 EDER 29.7%，越低越好（卡片） | [arXiv 2306.12991](https://arxiv.org/abs/2306.12991)（ASRU 2023） | [HF](https://huggingface.co/speechbrain/emotion-diarization-wavlm-large) | 315.5M；1.26 GB | Apache-2.0 | SpeechBrain |
| 15 | ParaCLAP | TUM CHI、奥格斯堡大学 | 2024-06 | IEMOCAP 4 类零样本 UAR 0.560–0.567（论文；随训练时用的文本查询而变，HF 上是哪一版未核实） | [arXiv 2406.07203](https://arxiv.org/abs/2406.07203)（Interspeech 2024） | [HF](https://huggingface.co/KeiKinn/paraclap)；[GitHub](https://github.com/KeiKinn/ParaCLAP) | 276.3M；1.11 GB | 只写了 "cc"，没说哪一种；音频分支来自 CC BY-NC-SA 模型 | GitHub 代码 |
| 16 | emotion2vec base | 同 #1 | 2023-12 | IEMOCAP 4 类 WA 71.8%（论文，冻结编码器 + 线性层） | [arXiv 2312.15185](https://arxiv.org/abs/2312.15185)（ACL 2024 Findings） | [HF](https://huggingface.co/emotion2vec/emotion2vec_base)；[ModelScope](https://modelscope.cn/models/iic/emotion2vec_base) | 93.8M；1.13 GB（含优化器状态） | ① | FunASR；fairseq |
| 17 | Emotion2Vec-S | 西北工业大学 ASLP | 2025-02 | Emo-Emilia 7 类 UA 80.7%，MELD UA 21.3%（论文，冻结特征 + 下游分类器） | [arXiv 2502.18186](https://arxiv.org/abs/2502.18186)（IEEE TASLP） | [HF](https://huggingface.co/ASLP-lab/Emotion2Vec-S) | 93.8M；1.13 GB（含优化器状态） | Apache-2.0 | fairseq（[C²SER 仓库](https://github.com/ASLP-lab/C2SER)） |

① emotion2vec 全家：Hugging Face 卡片写 FunASR Model License v1.1（可以使用、修改、分享，须注明出处并保留模型名，不限商用）；ModelScope 页面写 Apache-2.0；GitHub 代码是 MIT。

### 1b 架构、训练数据和输出

| # | 模型 | 架构 | 训练数据 | 语言 | 输出 | 输入音频 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | ⭐ **emotion2vec+ large** | data2vec 2.0 式编码器：CNN + 12 层 Transformer（1024 维）+ 线性分类头，162.1M 参数 | 伪标签数据 42,526 h（从 160,000 h 里选出约 40,000 h），来源未公布 | 未公布 | 9 类：angry、disgusted、fearful、happy、neutral、other、sad、surprised、unknown（包含本项目的 7 类） | 16 kHz 单声道（FunASR 会重采样），可变长度，卡片没给上限；本项目 Apple MPS 跑不了长于约 20 s 的片段（第一层卷积输出超过 65,536 步），脚本改用 CPU | 每一版都在前一版的数据上迭代：seed（#8）→ base（#7）→ large |
| 2 | ⭐ **MERaLiON-SER-v1** | Whisper-medium 编码器（冻结，LoRA 已合并）+ 注意力池化 + 改进的 ECAPA-TDNN；分类和维度两个头；313.4M 参数 | CREMA-D、ESD、MELD、M3ED；SG-ECMT 66,209 条（英、中、马来、泰米尔；标签是 emotion2vec+ seed 和 base 一致的预测，再用本模型二次标注）；SGTV 约 117,000 条、约 120 h（影视对白，人工标注，英、中） | 英（含新加坡英语）、中、马来、泰米尔；泰、印尼、越南有限支持 | 7 类：neutral、happy、sad、angry、fearful、disgusted、surprised（正是本项目的 7 类）；另有 valence、arousal、dominance（0–1） | ⚠️ **固定 30 s 窗口**：16 kHz；短片段补静音到 30 s，长于 30 s 只用前 30 s；池化包含补的静音 | — |
| 3 | ⭐ **Melati 情绪分类** | Whisper large-v3 编码器（冻结）+ 各层平均池化后加权混合 + LayerNorm + 256 维隐层，637.3M 参数 | 28,895 条：EMNS、Emozionalmente、JL-Corpus、MESD、Oreau、RESD、SUBESCO、EmoV-DB、Expresso | 训练语料有 6 种语言（孟加拉、英、法、意、俄、西） | 7 类：neutral、happy、sad、angry、surprised、fearful、disgusted（正是本项目的 7 类） | ⚠️ **固定 30 s 窗口**：16 kHz；补静音到 30 s，长于 30 s 截断（短于 0.1 s 先补齐）；只在片段自己的帧上池化 | — |
| 4 | ⭐ **Vox-Profile Whisper 分类** | Whisper large-v3 编码器（后半层 LoRA）+ 1×1 卷积 + 平均池化 + MLP 头，639.9M 参数 | MSP-Podcast v1.12（IS2025 挑战赛数据，只用 3–15 秒的片段） | 英语 | 9 类：Anger、Contempt、Disgust、Fear、Happiness、Neutral、Sadness、Surprise、Other；另有次级情绪和 arousal、valence、dominance 头 | ⚠️ **固定 15 s 窗口**：16 kHz 单声道；补静音到 15 s，长于 15 s 只用前 15 s；训练用 3–15 s 的片段，短于 3 s 不可靠 | 卡片称训练流程来自 IS2025 SER 挑战赛的最佳系统 SAILER（这里去掉了文本）；短于 3 秒的片段预测不可靠；官方代码在 forward 里写死了 `.cuda()` |
| 5 | ⭐ **Vox-Profile WavLM 分类** | WavLM-large（LoRA）+ 同 #4 的下游头，316.2M 参数 | 同 #4 | 英语 | 9 类：Anger、Contempt、Disgust、Fear、Happiness、Neutral、Sadness、Surprise、Other（同 #4） | 16 kHz 单声道，可变长度；卡片：最长 15 s（要自己截），训练用 3–15 s 的片段，短于 3 s 不可靠 | 片段长度限制同 #4 |
| 6 | ⭐ **Odyssey 2024 基线（分类）** | WavLM-large（微调）+ attentive statistics pooling + MLP，318.6M 参数 | MSP-Podcast release 1.11 的子集（训练集 68,360 段、1,405 人） | 英语 | 8 类：Angry、Sad、Happy、Surprise、Fear、Disgust、Contempt、Neutral | 16 kHz，可变长度，卡片没给上限；波形先按 config 里的 mean、std 归一化 | — |
| 7 | emotion2vec+ base | 同 #1，但 768 维，93.2M 参数 | 过滤后的伪标签数据 4,788 h，来源未公布 | 未公布 | 同 #1 | 同 #1（16 kHz 单声道，可变长度） | — |
| 8 | emotion2vec+ seed | 同 #7，93.2M 参数 | EmoBox 学术数据 201 h | 多语言（EmoBox 共 14 种） | 同 #1 | 同 #1 | — |
| 9 | emotion2vec base finetuned | #16 + 线性分类头，93.2M 参数 | 先用学术 SER 数据微调，再给 150,000 h 中英语音打标签，挑出文本情绪和语音情绪一致、置信度高的部分（超过 10,000 h）再微调 | 中、英 | 同 #1 | 同 #1 | 官方建议改用 emotion2vec+ |
| 10 | Aniemore WavLM 跨语言 v1 | WavLM-large + 分类头，316.5M 参数 | 27,939 条：RESD、Dusha（俄语）、CAMEO（12 个非俄语语料，如 CREMA-D、CaFE、EMNS、nEMO、RAVDESS、SUBESCO）、IEMOCAP、ASVP-ESD | 俄、英、德、波兰、意、法、西、孟加拉 | 7 类：anger、disgust、enthusiasm、fear、happiness、neutral、sadness（没有 surprised） | 16 kHz 单声道（卡片强调必须重采样），可变长度，卡片没给上限 | 另有 wav2vec2 版（Apache-2.0） |
| 11 | autrainer MSP-Podcast 4 类 | audEERING 维度模型（第 2 节 #1）再微调成分类，164.8M 参数 | MSP-Podcast v1.7（原模型）+ v1.11（微调） | 英语 | 4 类：A、H、N、S（卡片：Anger、Happiness、Neutral、Sadness） | 16 kHz，可变长度；短于 3 s 补到 3 s（`inference_transform.yaml`），没给上限 | — |
| 12 | SpeechBrain wav2vec2 IEMOCAP | wav2vec2-base（微调）+ 平均池化 + 线性层，94.4M 参数 | IEMOCAP（4 类） | 英语 | 4 类：neu、ang、hap、sad | 16 kHz 单声道（代码自动重采样），可变长度，没给上限 | — |
| 13 | SUPERB ER | 冻结的 SSL 骨干（各层加权求和）+ 投影 + 平均池化 + 线性层；base 94.6M、large 315.7M 参数 | IEMOCAP 4 类（第 1 折，Session 1 作测试） | 英语 | 4 类：neu、hap、ang、sad | 16 kHz，可变长度，卡片没给上限 | — |
| 14 | SpeechBrain 情绪分段 | WavLM-large + 逐帧线性分类，315.5M 参数 | IEMOCAP、RAVDESS、JL-Corpus、ESD、EmoV-DB 拼接成的音频（每段只有一种非中性情绪） | 英语 | 4 类：a、n、h、s（angry、neutral、happy、sad），带起止时间 | 16 kHz 单声道（代码自动重采样），可变长度，每 20 ms 一个标签，没给上限 | — |
| 15 | ParaCLAP | CLAP 双塔：音频是 audEERING 的 wav2vec2，文本是 BERT-base；共 276.3M 参数 | MSP-Podcast（55,283 条、110 多小时），文本查询由专家声学特征和标签生成 | 英语（卡片还列了德语） | 零样本：和任意文本标签算相似度，可以直接用本项目的 7 个类名 | 16 kHz 单声道，可变长度（音频分支同第 2 节 #1），没给上限 | — |
| 16 | emotion2vec base | data2vec 2.0 式编码器：CNN + 12 层 Transformer（768 维），自监督预训练，93.8M 参数 | 262 h 无标注：IEMOCAP、MELD、CMU-MOSEI、MEAD、MSP-Podcast v1.8 | 英语 | 只输出特征（768 维），没有分类头 | 同 #1（16 kHz，可变长度） | — |
| 17 | Emotion2Vec-S | emotion2vec base（#16）加类别级半监督对比损失，93.8M 参数 | C²SER 训练集约 1,216 h（IEMOCAP、ESD、MER2024、BIIC-Podcast、MSP-Podcast v1.11 和内部中文 788 h；内部数据的标签是 emotion2vec+ large 和 GLM-4 结果的交集）+ 约 2,400 h 内部无标注数据 | 中、英 | 只输出特征，分类头要自己训 | 16 kHz（示例代码重采样），可变长度，没给上限 | C²SER-LLM 的声学编码器（第 4 节） |

### 1c 社区 checkpoint（只列几个有代表性的）

下载量高，但都是个人上传：训练数据小，测试集多是随机切分（同一说话人或同一句子同时出现在训练和测试里），卡片上的准确率不可信。

| 模型 | 年份 | 公开报告的效果 | 大小 | 许可 | 架构 | 训练数据 | 输出 | 输入音频 | 问题 |
|---|---|---|---|---|---|---|---|---|---|
| [ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition) | 约 2021 | 它自己的评估集上准确率 82.2%（卡片）；emotion2vec+ 卡片的雷达图里，多数数据集上低于 emotion2vec+ large | 316.5M；1.27 GB | Apache-2.0 | wav2vec2-large-xlsr-53（英语 ASR 微调版）+ 平均池化 + 分类头，316.5M 参数 | 只有 RAVDESS（1,440 条） | 8 类：angry、calm、disgust、fearful、happy、neutral、sad、surprised | 16 kHz，可变长度，卡片没给上限 | 没说怎么切分；RAVDESS 是 CC BY-NC-SA，模型却标 Apache-2.0 |
| [r-f/wav2vec-english-speech-emotion-recognition](https://huggingface.co/r-f/wav2vec-english-speech-emotion-recognition) | 2022-09 | 它自己的评估集上准确率 97.5%（卡片） | 316.5M；1.27 GB | Apache-2.0 | 同上骨干，但用 `Wav2Vec2ForCTC`：把每帧的 7 维输出平均后取最大；316.5M 参数 | SAVEE、RAVDESS、TESS（共 4,720 条） | 7 类：angry、disgust、fear、happy、neutral、sad、surprise | 16 kHz，可变长度，卡片没给上限 | 卡片的准确率很可能偏高：随机切分会让同一说话人同时出现在训练和测试里（TESS 只有 2 位说话人） |
| [firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3](https://huggingface.co/firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3) | 2024-09 | 测试集准确率 92.0%（卡片） | 637.3M；2.55 GB | Apache-2.0 | Whisper large-v3 编码器 + 池化 + 分类头（`WhisperForAudioClassification`），637.3M 参数 | RAVDESS、SAVEE、TESS、URDU（乌尔都语） | 7 类：angry、disgust、fearful、happy、neutral、sad、surprised | ⚠️ **固定 30 s 窗口**：16 kHz；卡片代码补静音到 30 s，长于 30 s 截断 | 没说切分方式 |
| [Dpngtm/wav2vec2-emotion-recognition](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition) | 2024-10 | 验证集准确率 79.9%（卡片） | 94.6M；0.38 GB | MIT | wav2vec2-base-960h + 分类头，94.6M 参数 | TESS、CREMA-D、SAVEE、RAVDESS（Kaggle 副本，约 12,000 条） | 7 类：angry、calm、disgust、fearful、happy、sad、surprised（calm 其实并入了 neutral，config 里仍叫 calm） | 16 kHz 单声道，可变长度；卡片建议短于 10 s，更长的要切段 | 没说切分方式 |

## 2. 维度输出（arousal、valence、dominance）

注：三个维度描述情绪的程度而不是类别。arousal（唤醒度）是激动还是平静，valence（效价）是正面还是负面，dominance（支配度）是有掌控感还是无力感；例如愤怒是高唤醒、负效价、高支配，悲伤是低唤醒、负效价、低支配。

MERaLiON-SER-v1（1a #2）和 Vox-Profile 分类模型（#4、#5）也有维度头，这里不重复。这些模型大多用 MSP-Podcast 的维度标签训练，输出约 0–1；本项目的数据集没有维度标签。

### 2a 基本信息和公开报告的效果

| # | 模型 | 机构 | 年份 | 公开报告的效果 | 论文 | 下载 | 大小 | 许可 | 运行方式 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ⭐ **audEERING wav2vec2 MSP-dim** | audEERING | 2022-04 | MSP-Podcast 测试集 CCC：arousal 0.745、dominance 0.634、valence 0.638（论文；数字是 24 层原版，论文说剪到 12 层后没有下降） | [arXiv 2203.07378](https://arxiv.org/abs/2203.07378)（IEEE TPAMI 2023） | [HF](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)；ONNX 在 [Zenodo](https://zenodo.org/record/6221127) | 165.3M；0.66 GB | CC BY-NC-SA 4.0（商用版要向 audEERING 购买） | transformers + 卡片里约 30 行的模型类 |
| 2 | ⭐ **Vox-Profile WavLM 维度** | USC SAIL 等 | 2025-06 | MSP-Podcast v1.12 开发集 CCC：arousal 0.585、valence 0.665（论文表 4） | [arXiv 2505.14648](https://arxiv.org/abs/2505.14648) | [HF](https://huggingface.co/tiantiaf/wavlm-large-msp-podcast-emotion-dim) | 316.0M；1.26 GB | OpenRAIL，但卡片写明不可商用 | vox-profile-release |
| 3 | ⭐ **Odyssey 2024 基线（维度）** | UT Dallas MSP Lab、JHU 等 | 2024-03 | MSP-Podcast 1.11 测试集 CCC：arousal 0.567、valence 0.607、dominance 0.424，挑战赛维度赛道平均分第一（论文表 2；卡片上的数不同） | [Odyssey 2024](https://www.isca-archive.org/odyssey_2024/goncalves24_odyssey.pdf) | [HF](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes)（另有单维度的 -Arousal、-Valence、-Dominance） | 318.6M；1.27 GB | MIT | transformers（`trust_remote_code`） |
| 4 | Vox-Profile Whisper 维度 | 同 #2 | 2025-06 | MSP-Podcast v1.12 开发集 CCC：arousal 0.588、valence 0.645（论文表 4） | 同 #2 | [HF](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion-dim) | 636.8M（不含用不到的解码器）；文件 1.54B、6.17 GB | 同 #2 | 同 #2 |
| 5 | Wav2Small 2.0 | audEERING | 2025-08（论文 2024-08） | 论文的 0.072M 版：MSP-Podcast test-1 上 arousal CCC 0.66，IEMOCAP（没参与训练）上 0.56；HF 上的 2.0 版未公布 | [arXiv 2408.13920](https://arxiv.org/abs/2408.13920) | [HF](https://huggingface.co/audeering/wav2small) | 0.016M（16,071 个）；69 KB | CC BY-NC-SA 4.0 | transformers + 卡片里的模型类 |
| 6 | VoiceCLAP-Commercial + 属性头 | LAION | 2026-06（头 2026-07） | 留出的 10% 数据上，61 个维度的 Pearson r 中位数 0.683（卡片） | 无单独论文 | [HF 编码器](https://huggingface.co/laion/voiceclap-commercial)；[HF 属性头](https://huggingface.co/laion/voiceclap-commercial-attribute-heads) | 编码器 113.2M、0.45 GB；61 个头共 4.3M、17 MB | 编码器 CC BY 4.0，头 Apache-2.0；训练数据都允许商用 | PyTorch（仓库里的代码） |
| 7 | Empathic Insight Voice（Small、Large） | LAION | 2025-05 | EmoNet-Voice Bench 上和专家评分的 Pearson r：Small 0.414，Large 0.421；Large 零样本 EmoDB 准确率 70.6%、RAVDESS 74.2%（论文） | [arXiv 2506.09827](https://arxiv.org/abs/2506.09827)（EmoNet-Voice） | [HF Small](https://huggingface.co/laion/Empathic-Insight-Voice-Small)；[HF Large](https://huggingface.co/laion/Empathic-Insight-Voice-Large) | 每个维度一个头，共 55 个：Small 每个 73.7M（共 16.2 GB），Large 每个 147.5M（共 32.4 GB）；另需 BUD-E-Whisper 的编码器（88.2M；整个模型 241.7M） | CC BY 4.0 | 卡片的 Colab 代码 |

### 2b 架构、训练数据和输出

| # | 模型 | 架构 | 训练数据 | 语言 | 输出 | 输入音频 | 备注 |
|---|---|---|---|---|---|---|---|
| 1 | ⭐ **audEERING wav2vec2 MSP-dim** | wav2vec2-large-robust 剪到 12 层 + 平均池化 + 回归头，165.3M 参数 | MSP-Podcast v1.7 | 英语 | arousal、dominance、valence（约 0–1）；也可取池化后的特征 | 16 kHz，可变长度，卡片没给上限 | autrainer（1a #11）和 ParaCLAP（1a #15）都基于它 |
| 2 | ⭐ **Vox-Profile WavLM 维度** | WavLM-large（LoRA）+ 卷积 + 池化 + 回归头，316.0M 参数 | MSP-Podcast v1.12（3–15 秒片段） | 英语 | arousal、valence、dominance（0–1） | 同 1b #5（16 kHz 单声道，可变长度；卡片：最长 15 s，训练用 3–15 s，短于 3 s 不可靠） | 卡片称训练流程来自 IS2025 挑战赛维度任务第一名（[arXiv 2506.10930](https://arxiv.org/abs/2506.10930)） |
| 3 | ⭐ **Odyssey 2024 基线（维度）** | 同 1a #6，318.6M 参数 | 同 1a #6 | 英语 | arousal、dominance、valence（约 0–1） | 同 1b #6（16 kHz，可变长度，先按 mean、std 归一化） | — |
| 4 | Vox-Profile Whisper 维度 | Whisper large-v3 编码器（LoRA）+ 同 #2 的下游头，636.8M 参数 | 同 #2 | 英语 | 同 #2 | ⚠️ **固定 15 s 窗口**：同 1b #4（补静音到 15 s，长于 15 s 只用前 15 s；短于 3 s 不可靠） | 同 #2 |
| 5 | Wav2Small 2.0 | CNN：用卷积算 log-mel，再接 VGG 式卷积层，0.016M 参数 | 蒸馏：老师是 audEERING 模型和 Odyssey 2024 维度赛道冠军 WavLM 的平均输出，在 MSP-Podcast 音频和无标注、非语音音频上训练，不用人工标签 | 英语 | arousal、dominance、valence（0–1） | 16 kHz，可变长度，没给上限 | 论文版 0.072M 参数，量化 ONNX 约 120 KB |
| 6 | VoiceCLAP-Commercial + 属性头 | CLAP 式双塔，冻结的 768 维语音向量 + 每个属性一个小 MLP；编码器 113.2M 参数，61 个头共 4.3M | 编码器：Emilia-YODAS（CC BY 子集）、LAION's Got Talent、Majestrino；头：296,422 条（TTS-AGI/Emotion-Voice-Attribute-Reference-Snippets） | 英、德、法、西、中 | 55 个情绪和感知维度（Anger、Valence、Arousal 等）+ 音质、年龄、性别、语速 | ⚠️ **固定 30 s 窗口**：16 kHz 单声道；属性头要求先截到 30 s，编码器再补到 30 s；单用编码器时，长音频按 30 s 分块后取平均 | 头的标签来源未核实 |
| 7 | Empathic Insight Voice | Whisper small 编码器（88.2M）的完整输出（1500×768）→ 投影 → 每个维度一个 MLP；每个头 73.7M（Small）或 147.5M（Large）参数 | 约 5,000 h 合成表演语音（LAION's Got Talent）+ 约 5,000 h 网络 vlog；标签是 Gemini Flash 2.0 打的分 | 合成部分有英、德、西、法（论文表 2）；vlog 部分未核实 | 40 种情绪的强度 + 15 个属性（valence、arousal、年龄、性别等） | ⚠️ **固定 30 s 窗口**：16 kHz 单声道；卡片代码截到 30 s，Whisper 补到 30 s，头读全部 1500 帧（含补的静音） | 卡片提醒：主要用合成语音训练，对真实语音的泛化还要验证；编码器卡片写 BUD-E-Whisper，示例代码用的是 mkrausio/EmoWhisper-AnS-Small-v0.1（没有许可声明） |

## 3. 自带情绪标签的语音模型

| 模型 | 机构、年份 | 论文；下载 | 大小 | 许可 | 架构 | 训练数据 | 情绪输出 | 输入音频 | 运行方式 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| SenseVoiceSmall | 阿里巴巴通义实验室（FunAudioLLM），2024-07 | [arXiv 2407.04051](https://arxiv.org/abs/2407.04051)；[HF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)、[ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) | 234.0M；0.94 GB | 同 ①（HF 写 FunASR 协议，ModelScope 写 Apache-2.0）；代码 MIT | 非自回归 SAN-M 编码器 + CTC；输入前拼上语种、情绪、事件、ITN 四个任务向量；234.0M 参数 | 约 300,000 h（中、粤、英、日、韩）；情绪标签是 emotion2vec+ large 打的伪标签（30,000,000 条） | 每段一个情绪 tag，同时输出转写、语种和声音事件。后处理代码列了 7 种情绪 tag：HAPPY、SAD、ANGRY、NEUTRAL、FEARFUL、DISGUSTED、SURPRISED；模型代码的 `emo_dict` 只有 happy、sad、angry、neutral（另有 unk），实际会输出哪些类未核实 | 16 kHz（FunASR 重采样），可变长度；卡片：直接推理限 30 s 以内，更长的用 fsmn-vad 切段（每段最长 30 s），每段一个 tag | FunASR（本项目现用）；sherpa-onnx、GGUF | 只给 tag，不给各类概率；情绪是从 emotion2vec+ large 蒸馏来的 |
| BUD-E-Whisper | LAION，2025-05（V1.1 2025-11，V1.2 2026-04） | 无单独论文（EmoNet-Voice 项目）；[HF](https://huggingface.co/laion/BUD-E-Whisper) | 241.7M；0.97 GB | CC BY 4.0 | Whisper small 微调成直接生成情绪描述，241.7M 参数 | LAION's Got Talent + 约 5,000 h vlog；描述由 Gemini Flash 2.0 的打分转成模板句再改写 | 一段自由文本的情绪描述 | ⚠️ **固定 30 s 窗口**：16 kHz；短片段补静音到 30 s，长于 30 s 只用前 30 s（Whisper 特征提取的默认做法） | transformers（Whisper `generate`） | 编码器也给 Empathic Insight 用（第 2 节）；另有输出声音属性标签的 [voice-tagging-whisper](https://huggingface.co/laion/voice-tagging-whisper)（Apache-2.0，2025-12） |

## 4. 能识别情绪的开源音频大模型（简表）

和 emotion2vec+ 不同：它们按提示词生成文字答案，不给各类概率；参数量大，要在 CUDA 服务器上跑。表中数字来自不同来源和评测协议，不能直接相互比较。

| 模型 | 机构 | 年份 | 论文 | 下载 | 大小 | 许可 | 公开报告的效果 | 运行方式 | 输入音频 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2-Audio-7B-Instruct | 阿里巴巴 Qwen | 2024-07 | [arXiv 2407.10759](https://arxiv.org/abs/2407.10759) | [HF](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | 8.40B | Apache-2.0 | MELD 准确率 0.553（Qwen2.5-Omni 卡片表） | transformers | ⚠️ **固定 30 s 窗口**：16 kHz；处理器补到 30 s，长于 30 s 截断 |
| Qwen2.5-Omni-7B（3B） | 阿里巴巴 Qwen | 2025-03 | [arXiv 2503.20215](https://arxiv.org/abs/2503.20215) | [HF 7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)；[HF 3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 10.73B（含语音生成部分）；3B 版 5.54B | 7B：Apache-2.0；3B：Qwen Research License（非商用） | MELD 准确率 0.570（3B 0.558；卡片） | transformers | 16 kHz，可变长度（去掉补齐部分再编码）；处理器上限 300 s，更长截断；3B 相同 |
| Qwen3-Omni-30B-A3B-Instruct | 阿里巴巴 Qwen | 2025-09 | [arXiv 2509.17765](https://arxiv.org/abs/2509.17765) | [HF](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)；另有 [Captioner](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner) | 35.26B（MoE，每个 token 约 3B 激活） | Apache-2.0 | 技术报告没有语音情绪基准；MiniCPM-o 4.5 卡片的第三方结果 MELD 56.8；Captioner 版会描述说话人情绪 | transformers、vLLM | 16 kHz，可变长度（处理器不截断）；论文：每条最长约 40 分钟；Captioner 卡片建议不超过 30 s |
| Kimi-Audio-7B-Instruct | 月之暗面 Moonshot AI | 2025-04 | [arXiv 2504.18425](https://arxiv.org/abs/2504.18425) | [HF](https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct) | 9.77B | MIT（源自 Qwen2.5-7B 的部分为 Apache-2.0） | MELD 59.13（[GitHub](https://github.com/MoonshotAI/Kimi-Audio)） | Kimi-Audio 代码 | 16 kHz；Whisper 编码器按 30 s 分块（每块补到 30 s 再算，输出截回实际长度）；总长上限未核实 |
| Step-Audio 2 mini | 阶跃星辰 StepFun | 2025-08 | [arXiv 2507.16632](https://arxiv.org/abs/2507.16632) | [HF](https://huggingface.co/stepfun-ai/Step-Audio-2-mini) | 8.32B | Apache-2.0 | 自建 StepEval-Audio-Paralinguistic 的情绪项 82（卡片） | 官方代码 | 16 kHz，可变长度；官方代码按 25 s 分块提特征；总长上限未核实 |
| OSUM（OSUM-EChat） | 西北工业大学 ASLP | 2025-01（EChat 2025-08） | [arXiv 2501.13306](https://arxiv.org/abs/2501.13306)；[EChat](https://arxiv.org/abs/2508.09600) | [HF](https://huggingface.co/ASLP-lab/OSUM)；[HF EChat](https://huggingface.co/ASLP-lab/OSUM-EChat) | 8.02B（16.05 GB，bf16；Whisper 编码器 + Qwen2-7B） | Apache-2.0 | MELD 测试集准确率 53.4%、MER2023 86.4%（论文，只测 OSUM 支持的类别） | 官方代码 | 16 kHz（代码重采样），可变长度，最长约 30 s（Whisper 位置编码 1500 帧，训练滤掉长于 29 s 的）；更长要自己切 |
| Audio Flamingo 3（AF-Next） | NVIDIA、马里兰大学 | 2025-07（AF-Next 2026-04） | [arXiv 2507.08128](https://arxiv.org/abs/2507.08128)；[AF-Next](https://arxiv.org/abs/2604.10905) | [HF](https://huggingface.co/nvidia/audio-flamingo-3-hf)；[HF AF-Next](https://huggingface.co/nvidia/audio-flamingo-next-hf) | 8.27B | NVIDIA OneWay Noncommercial License（非商用） | IEMOCAP 准确率 63.8%（论文） | transformers | 16 kHz；按 30 s 窗口分块；AF3 每条最长 10 分钟，更长截断；AF-Next 最长 30 分钟 |
| SALMONN | 清华大学、字节跳动 | 2023-10 | [arXiv 2310.13289](https://arxiv.org/abs/2310.13289) | [HF](https://huggingface.co/tsinghua-ee/SALMONN) | 适配权重 0.40 GB；另需 Whisper large-v2、BEATs、Vicuna-13B v1.1 | 适配权重 Apache-2.0；Vicuna v1.1 有 LLaMA 许可限制（细节未核实） | IEMOCAP Session 5 准确率 0.69（论文；用 Session 1–4 训练） | 官方代码 | ⚠️ **固定 30 s 窗口**：16 kHz；代码把音频补到至少 1 s、截到 30 s，Whisper 再补到 30 s |
| MiniCPM-o 2.6（4.5） | OpenBMB、面壁智能 | 2025-01（4.5：2026-02） | [arXiv 2408.01800](https://arxiv.org/abs/2408.01800)（MiniCPM-V）；[4.5](https://arxiv.org/abs/2604.27393) | [HF 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)；[HF 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) | 8.67B（4.5：9.37B） | Apache-2.0 | MELD 52.4（4.5：60.2；卡片） | transformers（`trust_remote_code`） | 16 kHz 单声道；长于 30 s 按 30 s 分块；总长上限和 4.5 的做法未核实 |
| MERaLiON-2-10B（3-10B） | A*STAR | 2025-05（3-10B：2026-03） | [arXiv 2412.09818](https://arxiv.org/abs/2412.09818) | [HF](https://huggingface.co/MERaLiON/MERaLiON-2-10B)；[HF 3-10B](https://huggingface.co/MERaLiON/MERaLiON-3-10B) | 10.11B（Whisper large-v3 编码器 + Gemma-2-9B） | MERaLiON 许可 v3（MIT、Whisper 许可和 Gemma 条款） | 自家基准的 "Emotion Recognition" 63.7（卡片）；MERaLiON-SER 论文里不如 MERaLiON-SER | transformers | 16 kHz 单声道；按 30 s 分块（每块补到 30 s），最多 10 块即 300 s，更长的部分丢弃；ASR 建议 30 s 以内 |
| C²SER-LLM（专为 SER 训练：先描述内容和说话风格，再判断情绪） | 西北工业大学 ASLP | 2025-02 | [arXiv 2502.18186](https://arxiv.org/abs/2502.18186)（IEEE TASLP） | [HF](https://huggingface.co/ASLP-lab/C2SER-LLM) | 8.02B（16.05 GB，bf16；另需下载 Qwen2-7B） | Apache-2.0 | Emo-Emilia 7 类 UA 69.0%（论文，零样本） | 官方代码（Emotion2Vec-S 特征 + Whisper） | 16 kHz（代码重采样），可变长度，最长约 30 s（同 OSUM 的 Whisper 编码器，训练滤掉长于 29 s 的） |
| MOSS-Audio-8B-Instruct | 模思智能（MOSI.AI）、OpenMOSS、上海创智学院 | 2026-04 | 无（"即将发布"） | [HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct) | 9.05B | Apache-2.0 | 语音描述任务里情绪一项的 LLM 评分 3.31（卡片；Qwen3-Omni 3.33） | 官方代码 | 16 kHz，可变长度（不补到 30 s，编码器用窗口注意力）；总长上限未核实 |

## 5. 接入本项目的难度

每个模型由一个加载器 `scripts/models/<loader>.py` 运行：它只从 `models/` 下模型自己的文件夹构建模型（不联网），对一个片段返回模型各类别的概率，类名统一用本项目的名字；`scripts/evaluation/evaluate.py` 再把它们对应到每个数据集的情绪（见 `models/README.md` 的 "Adding a model"）。emotion2vec+ large 用 FunASR 运行。MERaLiON-SER-v1、Melati 和两个 Vox-Profile 分类模型已于 2026-10-28 接入（表的前三行）。项目锁定 transformers 5.18.0，这是能配 PyTorch 2.5.1 的最新版本：5.19.0 在 PyTorch 2.5.1 上无法导入任何模型类，因为它调用 PyTorch 2.6 才有的 `torch.accelerator`。

| 难度 | 模型 | 要做的事 |
|---|---|---|
| 已接入（2026-10-28） | MERaLiON-SER-v1 | `meralion_ser.py` 先用模型类构建模型、再加载权重，因为 transformers 5 在 meta device 上构建模型，模型自带的远程代码在那里会出错；这份代码导入了 peft 和 omegaconf（已加进 `requirements.txt`，实际不用）；`prepare()` 保存 openai/whisper-medium 的 config.json，加载时不用联网 |
| 已接入（2026-10-28） | Melati | `melati.py` 用 transformers 加载它的远程代码（`trust_remote_code`），不用改；预测取原始概率最大的类（argmax），不用官方 `melati_emotion.py` 的阈值 |
| 已接入（2026-10-28） | Vox-Profile Whisper、WavLM（分类） | `vox_profile.py` 重写了官方的推理代码，不需要 loralib、speechbrain，也不调用 `.cuda()`；概率和官方代码最多差 2.4e-6；`prepare()` 去掉用不到的文本解码器，把 Whisper 版的文件从 6.19 GB 减到 1.29 GB |
| 只改配置 | emotion2vec+ seed、base；emotion2vec base finetuned | 按 `models/README.md` 在 `config/models.yaml` 里注册（`loader: emotion2vec`）并下载。三者都输出同样的 9 类 |
| 小改 | SenseVoiceSmall | 也用 FunASR 加载，但 `generate()` 返回带 tag 的文本：要把 tag 转成类别，拿不到各类概率（除非读情绪 token 的 CTC logits） |
| 写一个 transformers 加载器（几十行） | Odyssey 2024 基线（要 `trust_remote_code`）；Aniemore、SUPERB 和 1c 节的社区模型（标准 `AutoModelForAudioClassification`） | 写一个返回各类概率的加载器（可参考 `melati.py`），再把模型的类名对到本项目的 7 类（如 `ang` → angry，`Happiness` → happy）。为 transformers 4.x 写的远程代码可能要绕开问题，像 MERaLiON-SER-v1 那样 |
| 要装别的包或改代码 | SpeechBrain 的两个模型、autrainer、ParaCLAP；Emotion2Vec-S 和 emotion2vec base 只出特征，要先训分类头 | 新依赖和 FunASR 1.4.16、transformers 5.18.0 能否共存未测试；像 `vox_profile.py` 那样重写推理代码，就不用装这些包 |
| 要另写评估 | 第 2 节的维度模型；第 4 节的音频大模型 | 本项目的数据集只有类别标签，维度输出要换分析方法（例如看各情绪在 arousal、valence 上的分布）；音频大模型可以沿用 `scripts/llm_eval/` 的提示词和解析，但要在本地 GPU 上推理 |

## 6. 注意事项

- **非商用或限制性许可**：audEERING 的两个模型和 autrainer（CC BY-NC-SA 4.0）、Vox-Profile（OpenRAIL，但卡片写明不可商用）、Melati（仅限研究）、Audio Flamingo 3 和 AF-Next（NVIDIA 非商用）、Qwen2.5-Omni-3B（Qwen Research）。ParaCLAP 的许可没写清（只写 "cc"），音频分支又来自 NC 模型。
  社区模型常把用 NC 数据（如 RAVDESS）训练的模型标成 Apache-2.0 或 MIT，不可靠。SALMONN 依赖的 Vicuna-13B v1.1 受 LLaMA 许可限制（细节未核实）。
- **卡片写的参数量和文件不符**：emotion2vec+ large（卡片约 300M，实际 162.1M）；MERaLiON-SER（卡片 309M，实测不含解码器 313.4M，文件 770.0M，多出的是用不到的 Whisper 解码器）；Vox-Profile 的 Whisper 版（文件 1.55B，同样含解码器）。emotion2vec 全家的 .pt 文件里约三分之二是优化器状态。
- **伪标签链**：SenseVoice（emotion2vec+ large）、MERaLiON-SER（emotion2vec+ seed 和 base）、C²SER 的内部数据（emotion2vec+ large）都用 emotion2vec+ 打过标签，emotion2vec+ base 和 large 自己的伪标签数据来源也没公布。这些模型的错误可能和 emotion2vec+ 相关。
- **领域和片段长度**：MSP-Podcast 系模型学的是播客里的自然情绪。Vox-Profile 训练时去掉了短于 3 秒的片段，卡片说这样的片段预测不可靠。
- **Hugging Face 的创建日期**：2022-03-02 是旧仓库迁移时的统一日期，不是发布日期；SUPERB、SpeechBrain IEMOCAP、ehcalabres 的年份按卡片和依赖版本推算。

## 7. 主要来源

- emotion2vec：https://github.com/ddlBoJack/emotion2vec 、https://arxiv.org/abs/2312.15185 、https://huggingface.co/emotion2vec 、https://modelscope.cn/models/iic/emotion2vec_base_finetuned （以及 iic/emotion2vec_base、iic/emotion2vec_plus_seed、iic/emotion2vec_plus_base、iic/emotion2vec_plus_large）；EmoBox 4 类结果表：https://github.com/ddlBoJack/emotion2vec/blob/main/src/emotion2vec%2Bperformance.png ；本项目的 emotion2vec+ large 结果：[../reports/emotion2vec_plus_large/summary.md](../reports/emotion2vec_plus_large/summary.md)；FunASR 模型协议：https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE ；EmoBox：https://arxiv.org/abs/2406.07162
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

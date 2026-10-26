# 与 CREMA-D 同类的情感语音数据集调研

调研日期：2026-10-23。目标：找到像 CREMA-D 那样"同一句话用不同情绪表达"（排除语义线索）、标签质量高、而且规模更大的数据集，
用于评估情绪识别模型和音频大模型（audio LLM）。

方法：检索论文、官方项目页、GitHub、HuggingFace、Zenodo、OSF 等，汇总数据集信息。
大小以十进制 GB 计，来自各平台 API 的文件列表或 HTTP 请求；标"约"的是推算值，标"未公开"的是平台没有给出大小。
标"未核实"的数字没能从一手资料确认。✅ 表示已下载到本地（见第 6 节）；这些数据集的条数、时长和采样率是我下载后实测的。

## 0. 要点

- **CREMA-D 本身就在 EmoBox 里**，所以 emotion2vec+ 在 CREMA-D 上的结果不是 held-out 结果。
  候选清单里的 ESD、MEAD、RAVDESS、TESS、SAVEE、EmoDB、CASIA、Emozionalmente、SUBESCO、JL-Corpus、CaFE、EMOVO、EmoV-DB 也都在 EmoBox 中。
- **emotion2vec+ 训练数据的公开程度：**
  - seed 用 EmoBox 的学术数据微调（201 h）。EmoBox 全部约 294 h，具体用了哪些子集没有列出，保守起见把 EmoBox 全部视为模型已见过的数据。
  - base 和 large 用"过滤后的大规模伪标签数据"微调（4,788 h 和 42,526 h）。官方说数据工程细节以后公布，伪标签数据的来源不明，所以任何公开数据都无法 100% 排除。
  - emotion2vec 的预训练用了 IEMOCAP、MELD、CMU-MOSEI、MEAD 和 MSP-Podcast。
- **最好的"干净"候选（都不在 EmoBox 中）**：MCAE-SPPS（普通话，2025/26）、THAI-SER（泰语，规模最大）、MSP-IMPROV 的 target sentences（英语）、nEMO（波兰语，2024）。
- **英语**：没有找到同时满足"规模大、平行文本、有感知验证、不在 EmoBox"的数据集，只能把 MSP-IMPROV、MESS、Expresso 组合起来用。
- **支持这个思路的证据**：[LISTEN](https://arxiv.org/abs/2510.10444)（2025）用 CREMA-D、ESD、TESS、SAVEE、RAVDESS 里文本中性的句子测试音频大模型。
  只给音频时，准确率只有 16.5–34.9%（[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) 34.0%，[Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261) 34.9%）；只给文本时则有 85–97%。
  说明这些模型主要靠"转写成文字"，而不是真正在听声音。

## 1. 高质量数据集（不在 EmoBox 中）

| # | 数据集（年份，语言） | 说话人 / 条数 / 时长 / **大小** | 情绪类别；强度 | 平行文本设计 | 标签质量 | 音频；模态 | 许可；获取 | 链接 |
|---|---|---|---|---|---|---|---|---|
| 1 | **MCAE-SPPS** ✅（DataPort 2025-01，Sci Data 2026-03；普通话） | 6 名专业演员（3 男 3 女）/ 6,675 条（共录 8,400 条，评估了 8,379 条）/ 2.3 h / **0.65 GB**（7 个按情绪打包的 zip） | 中性、开心、悲伤、愤怒、恐惧、厌恶、惊讶，与 emotion2vec+ 的类别一一对应；有感知强度评分（1–9） | 200 句 × 7 种情绪 × 6 人 | 每条由 40 名母语听者做 7 选 1 并评强度（共 720 人）；保留条件：目标情绪得票 ≥43%，且任何其他情绪 <43%；各情绪识别率 60–82%（DataPort 版本的数字） | 专业隔音棚，48 kHz 单声道；只有音频 | CC BY-NC-ND 4.0；OSF 公开。注意：OSF 上按情绪打包的 zip 是完整的 6,675 条，按情绪分的散文件夹只有 6,460 条；IEEE DataPort 副本需要订阅 | [OSF 9jyzc](https://osf.io/9jyzc/)；[IEEE DataPort（需订阅）](https://ieee-dataport.org/documents/mandarin-chinese-auditory-emotions-stimulus-database-validated-set-subject-personal) |
| 2 | **THAI-SER** ✅（数据 2021，论文 2025；泰语） | 200 名专业演员 / 27,854 条（脚本 ≈17,711 + 即兴 ≈10,143）/ 41.6 h / **12.71 GB**（HF parquet；GitHub zip 版 12.61 GB） | 中性、愤怒、开心、悲伤、沮丧（frustrated）；脚本部分有 2 级表演强度 | 脚本部分：3 句 × 5 种情绪 × 2 种强度 × 2 遍；即兴部分不平行 | 众包，每条 3–8 人；同时给出指定标签和多数票标签，以及一致率；过滤后 Krippendorff α = 0.692，人类准确率 0.772；高一致子集（一致率 ≥0.71）14,182 条 | 80 个录音棚 session（其中一类房间不隔音）+ 20 个 Zoom session；44.1 / 48 kHz；每条最多 4 路麦克风；只有音频 | **CC BY-SA 4.0**（可商用）；直接下载 | [HF airesearch/thai-ser](https://huggingface.co/datasets/airesearch/thai-ser)；[GitHub release v1](https://github.com/vistec-AI/dataset-releases/releases/tag/v1) |
| 3 | **MSP-IMPROV**（2017；美式英语） | 12 名演员（6 对）/ 共 8,438 轮，受控文本部分是 652 条 target 句 + 620 条朗读 / 时长未核实 / **未公开**（签协议后提供，含视频） | 愤怒、开心、悲伤、中性；无表演强度，但有效价（valence）和唤醒度（arousal）的维度评分 | 20 个 target 句 × 4 种情绪情境 × 12 人；target 句嵌在即兴对话里说出，另有朗读版 | MTurk 众包，每轮 ≥5 人评分 | 音视频；采样率未核实 | 学术许可，需要签协议；商业许可 8,000 美元 | [官方页（已迁到 lab-msp.com）](https://www.lab-msp.com/MSP/MSP-Improv.html)；[学术许可 PDF](https://www.lab-msp.com/MSP/publications/AcademicLicense-MSP-IMPROV.pdf) |
| 4 | **nEMO** ✅（2024；波兰语） | 9 人（4 女 5 男，其中 3 人是职业配音演员）/ 4,481 条 / 3.07 h / **0.53 GB**（HF parquet） | 愤怒、恐惧、开心、悲伤、惊讶、中性；无强度 | 90 句（按音素覆盖设计）× 6 种情绪 × 9 人 | 经过人工评估，只保留表达准确的录音（按 4,860 条推算约保留 92%）；没有报告一致性数值 | 在家用电容麦克风录制，24 kHz / 16-bit；只有音频 | CC BY-NC-SA 4.0；直接下载 | [HF amu-cai/nEMO](https://huggingface.co/datasets/amu-cai/nEMO)；[GitHub](https://github.com/amu-cai/nEMO) |
| 5 | **HCUDB 1+2**（发布年份未核实，经费为 2014–2016 财年；日语） | 14 + 20 名专业演员 / 4,620 + 6,600 条 / 时长未核实 / **未公开**（1 张 DVD 分发，≤4.7 GB） | 11 类（基于 Russell 情绪环形模型，包括惊讶、愤怒、轻蔑、困倦/疲惫等）；无表演强度 | HCUDB1：10 句 × 11 种情绪 × 3 遍；HCUDB2：其中 5 句 × 11 种情绪 × 3 遍 × 2 种表演方法 | HCUDB1 每条由 16 人评分（效价/唤醒度、各类别是否贴切、自然度）；HCUDB2 没有评分 | 48 kHz / 16-bit；只有音频 | 仅限研究，免费；需申请，文档为日文 | [NII-SRC HCUDB](https://research.nii.ac.jp/src/en/HCUDB.html)；[NII-SRC 申请说明](https://research.nii.ac.jp/src/en/register.html) |
| 6 | **MCAE 伪句子**（2023；普通话伪句） | 6 名专业演员 / 验证后 4,361 条（OSF 的 Audio 文件夹有 4,561 个文件）/ 时长未核实 / **2.42 GB**（散文件，没有打包的 zip） | 开心、悲伤、愤怒、恐惧、厌恶、惊喜、中性；2 级强度（普通 / 强烈，中性除外） | 没有词义的伪句子 × 陈述/疑问两种句型 × 7 种情绪 × 2 种强度 | 每条由 40 名母语听者做强迫选择 | 采样率未核实；只有音频 | 许可未核实；OSF 只读链接 | [OSF 只读链接（项目名 MCAE-NS）](https://osf.io/sfbm6/?view_only=e22a521e2a7d44c6b3343e11b88f39e3) |
| 7 | **MESS** ✅（2019；美式英语） | 6 人（3 男 3 女）/ 1,800 条（Zenodo 公开版；设计为 90 句 × 4 种情绪 × 6 人 = 2,160 条）/ 1.3 h / **0.19 GB**（zip） | 愤怒、平静（calm）、开心、悲伤；有感知的激活度和愉悦度评分 | 90 句 × 4 种情绪 × 6 人 | 论文中每条由 10 名听者评分（类别 + 两个维度），但**评分数据需向作者索取**；Zenodo 包只有表演标签（写在文件名里） | 24,414 Hz 单声道；只有音频 | CC BY 4.0；直接下载（注意：不是 EmoBox 里的 MESD） | [Zenodo 3813437](https://zenodo.org/records/3813437) |
| 8 | **MASC**（LDC2007S09，2007；普通话） | 68 人（23 女 45 男）/ 25,636 条 / 时长未公布 / **未公开**（约 0.5 GB，按 8 kHz / 16-bit 推算，另含 WMA 原始文件） | 中性、愤怒、欣喜、恐慌、悲伤；无强度 | 每种情绪都读同一套 5 个短语和 20 个句子（12 句内容中性，另 8 句四种情绪各 2 句），各读 3 遍；另有 2 段中性段落 | 只有表演标签，说明文档没有提到听辨验证 | 在安静办公室用录音笔录制（22.05 kHz）；发布版是 8 kHz / 16-bit，另附 44 kHz 的 WMA 原始文件 | LDC 许可，非会员需要付费 | [LDC2007S09](https://catalog.ldc.upenn.edu/LDC2007S09) |

**排名理由：**

1. **MCAE-SPPS**：最接近"CREMA-D 加强版"。平行设计，每条有 40 人的感知标签和强度评分；200 个句子的词汇多样性远高于 CREMA-D 的 12 句；
   7 个类别正好对应 emotion2vec+ 的类别。2025 年以后才发布，没有找到它被任何音频大模型用于训练或评测的记录。
   缺点：只有 6 位说话人，许可是 NC-ND。
   引用：Li M, Zhou A, Yan H, Li Q, Ma C, Wu C. *Scientific Data*, 2026, doi:10.1038/s41597-026-06976-z；数据 doi:10.17605/OSF.IO/9JYZC。
2. **THAI-SER**：说话人最多，有众包感知标签和一致率，有两级强度，许可最宽松。
   缺点：脚本部分只有 3 个句子；"沮丧"在 emotion2vec+ 里没有对应类别，建议只用中性、愤怒、开心、悲伤四类。
   [VoxEmo](https://arxiv.org/abs/2603.08936)（2026）已经用它对 [Qwen2-Audio](https://arxiv.org/abs/2407.10759) 和 [AF3](https://arxiv.org/abs/2507.08128) 做过零样本评测（只评测，没有训练）。引用：arXiv:2507.09618。
3. **MSP-IMPROV**：英语中最成熟的"受控文本 + 众包感知标签 + 不在 EmoBox"的数据。
   缺点：受控文本部分只有约 1.3k 条，需要签协议，而且 [OSUM](https://arxiv.org/abs/2501.13306) 把它当作测试集用过。引用：Busso et al., IEEE TAC 8(1):119–130, 2017。
4. **nEMO**：2024 年的新数据，完全平行，可以直接下载；情绪类别与 CREMA-D 接近（少了厌恶，多了惊讶）。
   缺点：规模小，在家录制；VoxEmo 已用它评测过。引用：Christop, LREC-COLING 2024, pp. 12111–12116。
5. **HCUDB**：34 名专业演员，11,220 条，同一句话录了多种情绪；HCUDB1 有维度评分，可以做强度分析。
   缺点：11 个类别映射到 emotion2vec+ 时需要取舍；申请流程是日文。
6. **MCAE 伪句子**：完全去掉了词义，模型只能靠声音判断，最适合测试"模型是否真的在听"；有两级强度。缺点：条数比 CREMA-D 少。
   引用：Gong et al., *Behavior Research Methods* 55(3):1441–1459, 2023。
7. **MESS**：英语的小型补充，平行，CC BY 许可。缺点：用"平静"代替"中性"，只有 4 类；公开版只有表演标签。
   引用：Morgan, JSLHR 62(11), 2019。
8. **MASC**：普通话中唯一"多说话人 + 大规模 + 平行"的数据，可以测试对说话人的泛化。缺点：8 kHz，没有听辨验证，需要付费。

## 2. 高质量数据集（在 EmoBox 中）

这些数据集在 EmoBox 中，不能作为 emotion2vec+ 的干净测试集，但可以用于测试音频大模型，或作参照。

| 数据集（年份，语言） | 规模 / **大小** | 情绪；强度 | 平行设计 | 标签 | 音频；许可 | 已知被哪些音频大模型用过 | 链接 |
|---|---|---|---|---|---|---|---|
| ESD ✅（2021，英语 + 中文） | 20 人 / 35,000 条 / 29.1 h / **2.45 GB**（zip） | 5 类；无强度 | 每人 350 句 × 5 种情绪 | 只有表演标签；论文只用自动情绪识别模型检验（中文 92%，英文 89%），没有人工听辨 | 室内录制，信噪比 > 20 dB，16 kHz；仅限研究，需填表并发邮件申请 | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT（29 h）；[OSUM](https://arxiv.org/abs/2501.13306) 训练和测试；[LISTEN](https://arxiv.org/abs/2510.10444)；[VoxEmo](https://arxiv.org/abs/2603.08936) | [GitHub（许可表 + Google Drive 链接）](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| MEAD（2020，英语） | 60 人（公开的 Part0 只有 48 人，已确认）/ 31,729 条 / 37.3 h / **音频 3.07 GB**（48 人的 audio.tar）；视频约 0.4 TB | 8 类；3 级强度 | 每种情绪 30 句（3 句通用 + 7 句与该情绪相关 + 20 句一般句），不同情绪用不同的句表；同一情绪的不同强度用相同句子 | 100 人参与的视频评估 | 48 kHz，音视频；按使用条款 | [emotion2vec](https://arxiv.org/abs/2312.15185) 预训练；[VoxEmo](https://arxiv.org/abs/2603.08936) | [Google Drive Part0](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)；[百度网盘（提取码 th8e）](https://pan.baidu.com/s/1zYKz19kmQH6vnO2nogPk4Q)；[GitHub](https://github.com/uniBruce/Mead)（官方项目页现已 404） |
| RAVDESS（2018，英语） | 24 人 / 1,440 条 / 1.5 h / **语音音频 0.208 GB**；全部（含歌唱和视频）25.6 GB | 8 类；2 级强度 | 2 个句子 | 每个文件评 10 次（共 247 人） | 48 kHz，音视频；CC BY-NC-SA | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT；[AF1](https://arxiv.org/abs/2402.01831)/[AF2](https://arxiv.org/abs/2503.03983) 零样本评测 | [Zenodo 1188976](https://zenodo.org/records/1188976) |
| Emozionalmente（意大利语） | 431 人 / 6,902 条 / 6.3 h / **0.559 GB**（v1.1） | 7 类 | 18 个句子 | 829 名听者，每条 5 次评价，人类 UAR 66% | 16 kHz，非专业设备；CC BY 4.0 | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 12616095（v1.1）](https://zenodo.org/records/12616095)；[OpenSLR 161](https://openslr.org/161/) |
| SUBESCO（2021，孟加拉语） | 20 人 / 7,000 条 / 7.8 h / **1.665 GB** | 7 类 | 10 句 × 5 遍 | 50 名评分者，准确率 71% | CC BY 4.0 | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 4526477](https://zenodo.org/records/4526477) |
| CaFE（2018，加拿大法语） | 12 人 / 936 条 / **0.187 GB**（48 kHz 版；192 kHz 版 2.03 GB） | 7 类；2 级强度 | 6 个句子 | 未核实 | 192 kHz 和 48 kHz；CC BY-NC-SA | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 1478765](https://zenodo.org/records/1478765) |
| EmoDB / **EmoDB 2.0**（2025，德语） | 10 人 / 535 条，2.0 版新增 281 条"模糊"样本（共 817 条）/ **0.041 GB**（原版）；2.0 版约 0.11 GB | 7 类 | 10 个句子 | 20 名评审的一致率和自然度评分 | 消声室录制；audb 2.0 版为 CC0-1.0 | 新增的 281 条不在 EmoBox 中，可以用来测试模糊样本 | [原版 emodb.bilderbar.info](http://emodb.bilderbar.info/)；[audb emodb 2.0.0](https://github.com/audeering/emodb)（[说明页](https://audeering.github.io/datasets/datasets/emodb.html)） |
| EMOVO（意大利语） | 6 人 / 588 条 / **0.253 GB**（Kaggle 非官方镜像） | 7 类 | 14 个句子 | 两组各 24 名听者，总体 80% | — | [VoxEmo](https://arxiv.org/abs/2603.08936) | 官方站 voice.fub.it 当天无法访问；[Kaggle 镜像](https://www.kaggle.com/datasets/sourabhy/emovo-italian-ser-dataset)；[DagsHub 镜像](https://dagshub.com/DagsHub/audio-datasets/src/main/EMOVO) |
| TESS（英语） | 2 人 / 2,800 条 / **0.281 GB** | 7 类 | 200 个词 × "Say the word ___" 句式 | 未核实 | CC BY-NC 4.0 | [AF1](https://arxiv.org/abs/2402.01831) 训练 | [Borealis Dataverse](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF) |
| SAVEE（英式英语） | 4 人 / 480 条 / 官方版**未公开**；Kaggle 纯音频镜像 0.113 GB（非官方） | 7 类 | 每种情绪 15 句，只有 3 句跨情绪共用 | 10 名评估者（官网证书报错，当天没能复核） | 音视频 | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT | [下载页（需注册后登录）](http://kahlan.eps.surrey.ac.uk/savee/Download.html)；[注册](http://www.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Register.html)；[Kaggle 镜像](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) |
| CASIA（中文） | 4 人 / 1,200 条 / 官方版**未公开**；Kaggle 镜像 0.059 GB（非官方，许可不明） | 6 类 | 平行（未核实） | 只有表演标签 | 许可不清楚 | [VoxEmo](https://arxiv.org/abs/2603.08936) | 官方渠道未核实；[Kaggle 镜像](https://www.kaggle.com/datasets/rocklagoon/casia-emotion-recognition) |
| JL-Corpus（新西兰英语） | 4 人 / 2,400 条 / **0.948 GB** | 5 种主要情绪 + 5 种次要情绪 | 未核实 | 120 人测试：主要情绪约 70%，次要情绪约 40% | CC0 | [AF1](https://arxiv.org/abs/2402.01831) 训练 | [Kaggle（作者本人上传）](https://www.kaggle.com/datasets/tli725/jl-corpus)；[GitHub](https://github.com/tli725/JL-Corpus) |
| EmoV-DB（英语） | 4 人 / 6,887 条 / 9.5 h / **约 1.41 GB**（18 个 tar.gz） | 中性、愉快、愤怒、厌恶、困倦 | 用 CMU Arctic 的句子，跨情绪平行 | 只有表演标签；句子前后常有笑声或哈欠 | 非商用 | [AF1](https://arxiv.org/abs/2402.01831) 训练；[Qwen-Audio](https://arxiv.org/abs/2311.07919) 论文的 SFT 示例里出现了 EmoV-DB 的音频文件 | [OpenSLR 115](https://openslr.org/115/)；[GitHub](https://github.com/numediart/EmoV-DB) |

注：AF1、AF2 是 NVIDIA 的 Audio Flamingo 1 和 2；emotion2vec 是 emotion2vec+ 的预训练模型，不是音频大模型；LISTEN 和 VoxEmo 不是模型，而是用这些数据集评测音频大模型的论文（评测了哪些模型见第 4 节）。

## 3. 不在 EmoBox 中，但规模小、不平行，或者标签、音频有缺陷

| 数据集（年份，语言） | 规模 / **大小** | 关键设计 | 标签 | 许可 / 获取 | 主要问题 | 链接 |
|---|---|---|---|---|---|---|
| **Expresso**（2023，英语） | 4 人；朗读 11,615 条 / 11.5 h，即兴 34.4 h / **38.44 GB**（expresso.tar，含即兴对话）；HF 社区镜像只有朗读部分，5.76 GB | 7 种朗读风格（default、happy、sad、confused、enunciated、laughing、whisper）读同一套文本 | 没有感知验证 | CC BY-NC 4.0，开放下载 | 情绪类别少；音质最好（专业录音棚，48 kHz / 24-bit）；Meta 的 [SpiRit-LM](https://arxiv.org/abs/2402.05755) 用它训练过风格 tokenizer | [官方下载说明](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset)；[expresso.tar 直链](https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar)；[HF 社区镜像](https://huggingface.co/datasets/ylacombe/expresso) |
| KESDy18（2018，韩语） | 30 名配音演员 / 2,880 条 / **未公开** | 20 句 × 4 种情绪 | 6 名标注者，7 类 + 唤醒度/效价（1–5） | ETRI 许可协议（站点当天无法访问） | — | [ETRI nanum](https://nanum.etri.re.kr/share/kjnoh2/KESDy18?lang=En_us) |
| IITKGP-SEHSC（印地语） | 10 人 / 12,000 条 / 约 9 h / **未公开**（约 1 GB，按 16 kHz / 16-bit 推算） | 15 句 × 8 种情绪 × 10 个 session | 未核实 | 向 IIT Kharagpur 申请 | 16 kHz | [作者资源页](https://cse.iitkgp.ac.in/~ksrao/res.html)；[申请声明 PDF](https://cse.iitkgp.ac.in/~ksrao/pdf/declaration.pdf) |
| CAVES（2024，粤语） | 10 人 / 3,475 条 / **未公开**（含 1080p / 50 fps 视频） | 50 个语义中性的句子 × 7 种情绪 | 15 名听者 | 仅限研究，需填表申请 | 规模小（48 kHz 隔音间录制，音视频） | [申请表](https://forms.office.com/r/3VfeWQnAVa) |
| OGVC 表演部分（日语） | 4 人 / 2,656 条 / **未公开**（整套含自然对话，2 张 DVD 分发） | 同一组台词跨情绪录制，4 级强度 | 未核实 | NII-SRC，免费 | 规模小 | [NII-SRC OGVC](https://research.nii.ac.jp/src/en/OGVC.html)；[NII-SRC 申请说明](https://research.nii.ac.jp/src/en/register.html) |
| MCAE-Monosyllable（2025，中文） | 6 人 / 18,089 条 / **1.18 GB**（散文件） | 422 个单音节 × 7 种情绪 | 720 名评分者 + 强度评分 | OSF | 只有单音节，不是句子 | [OSF h3uem](https://osf.io/h3uem/) |
| BERSt（2025，多种口音英语） | 98 人 / 4,526 条 / 3.75 h / **1.06 GB**（parquet；原始 wav 1.27 GB；整个仓库 3.18 GB） | 14 个无意义短语 × 7 种情绪 × 说话/喊叫/尖叫 | 只有意图标签 | HuggingFace 卡片写 CC BY，论文写 CC BY-SA | 演员在家用手机录制，有远场录音，不是干净的测试集 | [HF Rosie-Lab/BERSt](https://huggingface.co/datasets/Rosie-Lab/BERSt) |
| EmoTale（2025，丹麦语 + 英语） | 18 人 / 800 条 / **0.43 GB** | 5 句 × 5 种情绪 | κ 0.71–0.85 | 有版权声明但没有正式许可（GitHub） | 规模很小 | [GitHub](https://github.com/snehadas/EmoTale) |
| Thorsten Emotional（2021，德语） | 1 人 / 2,400 条 / **0.399 GB** | 300 句 × 8 种风格 | 无 | CC0 | 只有一位说话人 | [OpenSLR 110](https://openslr.org/110/)；[Zenodo 5525023](https://zenodo.org/records/5525023) |
| GEMEP（2012，伪语言） | 10 人；核心集 145 条 / **未公开** | 伪语言句子，18 类 | 1,260 条经过验证 | 需签使用条件；完整集受限 | 能拿到的部分太小 | [Core set 页面](https://www.unige.ch/cisa/gemep/coreset)；[使用条件表 PDF](https://www.unige.ch/cisa/download_file/view/143/304) |
| EARS（2024，英语） | 107 人，共约 100 h / **69.17 GB**（107 个按说话人打包的 zip；情绪部分不能单独下载） | 22 种情绪 + 中性 | 无 | CC BY-NC 4.0 | **不平行**：每种情绪用内容相符的不同句子，词汇本身会泄露情绪 | [GitHub](https://github.com/facebookresearch/ears_dataset)；[release "dataset"](https://github.com/facebookresearch/ears_dataset/releases/tag/dataset) |
| CSEMOTIONS（2025，中文） | 10 人 / 4,160 条 / 10.2 h / **3.08 GB**（parquet） | 7 类 | 未报告 | Apache-2.0 | **不平行**，文本内容与情绪一致；可以作"词汇线索"的对照组 | [HF ATH-MaaS/CSEMOTIONS](https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS)（原 AIDC-AI/CSEMOTIONS，会自动跳转） |
| KazEmoTTS（2024，哈萨克语） | 3 人 / 54,760 条 / 74.9 h / **10.46 GB**（EmoKaz.zip） | 每个句子只配一种情绪 | 听辨测试中愤怒只有 22% | CC BY 4.0 | 不平行，标签质量差 | [HF issai/KazEmoTTS](https://huggingface.co/datasets/issai/KazEmoTTS)；[GitHub](https://github.com/IS2AI/KazEmoTTS) |
| EmoNet-Voice Bench（2025） | 12,600 条 / **1.08 GB**（3 个 parquet） | 40 类 | 专家标注，Cronbach α 0.14 | CC BY 4.0 | 用 GPT-4o 合成的语音，不平行；数据仓库在匿名账号下，由 LAION 官方合集链接 | [HF emonet-voice-bench](https://huggingface.co/datasets/t1a5anu-anon/emonet-voice-bench)；[LAION EmoNet 合集](https://huggingface.co/collections/laion/emonet-6855238709fb1b2a7295d4ed) |
| mExpresso（2023） | — / **0.0033 GB**（只有文本） | 把 Expresso 的 7 种风格翻译成包括中文在内的 5 种语言 | — | 只公开了文本，音频没有发布 | 目前不可用 | [官方说明](https://github.com/facebookresearch/seamless_communication/blob/main/docs/expressive/README.md)；[文本 tar](https://dl.fbaipublicfiles.com/seamless/datasets/mexpresso_text/mexpresso_text.tar) |

## 4. 注意事项

- **非商用或限制性许可**：MCAE-SPPS（NC-ND：可以用来评测，但不能发布修改后的音频）、nEMO（NC-SA）、Expresso 和 EARS（NC）、RAVDESS 和 CaFE（NC-SA）、
  TESS（NC）、ESD（仅限研究）、EmoV-DB（非商用）、HCUDB 和 OGVC（仅限研究）、MSP-IMPROV（学术许可）、MASC（LDC 付费）。
  许可宽松的有：THAI-SER（BY-SA）、MESS、Emozionalmente 和 SUBESCO（BY）、CSEMOTIONS（Apache）、Thorsten（CC0）。
- **需要注册或申请**：MSP-IMPROV（签协议）、ESD（填表 + 发邮件）、HCUDB 和 OGVC（NII-SRC，日文）、KESDy18（ETRI）、MASC（LDC）、IITKGP、GEMEP、
  CAVES（填表）、MCAE 伪句子（只读链接）、IEEE DataPort（需订阅）。
- **Kaggle 等非官方镜像**（EMOVO、SAVEE、CASIA）：内容和许可没有官方保证，使用前尽量与原始来源核对。
- **已知的标签和音频问题**：
  - THAI-SER：指定标签和众包多数票标签经常不一致，建议只用一致率 ≥0.71 的子集；部分录音来自 Zoom 或不隔音的房间；4 路麦克风要固定用同一路。
  - MCAE-SPPS：保留阈值 43% 偏宽松；过滤后类别不均衡（愤怒 1,129 条、恐惧 671 条，中性和悲伤约 1,190 条）。
    要做严格的平行比较，可以只用"同一说话人、同一句子，7 种情绪全部保留"的子集。OSF 上按情绪分的散文件夹不完整，要用按情绪打包的 zip。
  - MESS：Zenodo 公开版只有 1,800 条和表演标签，听者评分要向作者索取。
  - nEMO：9 人里只有 3 人是职业配音演员；没有预设的数据划分。
  - MSP-IMPROV：target 句是在即兴对话里说出的，感知标签可能与预期情绪不一致。
  - ESD：没有人工听辨验证；16 kHz。MEAD：只有部分句子跨情绪平行。MASC：8 kHz，原始文件是有损的 WMA。HCUDB2：没有评分。
- **音频大模型用过哪些数据（只列有出处的）**：
  - [Kimi-Audio](https://arxiv.org/abs/2504.18425) 的 SFT 用了 IEMOCAP、MELD、RAVDESS、SAVEE、ESD。
  - [OSUM](https://arxiv.org/abs/2501.13306) 训练用了 ESD、IEMOCAP、MSP-Podcast、MER2023、MELD、BIIC-Podcast；测试用了 IEMOCAP、MER2023、M3ED、**MSP-IMPROV**、ESD。
  - [Audio Flamingo](https://arxiv.org/abs/2402.01831) 训练用了 MSP-Podcast、EmoV-DB、JL-Corpus、TESS、MELD、OMGEmotion；CREMA-D 和 RAVDESS 明确从训练中排除，只用于零样本评测。
    [AF2](https://arxiv.org/abs/2503.03983) 也在 CREMA-D 和 RAVDESS 上做零样本评测，[AF3](https://arxiv.org/abs/2507.08128) 在 IEMOCAP 上评测。
  - [SALMONN](https://arxiv.org/abs/2310.13289)：用 IEMOCAP 第 1–4 session 训练，第 5 session 测试。
  - [Qwen-Audio](https://arxiv.org/abs/2311.07919)：情绪识别训练数据不到 1k 小时且没有列出名称；SFT 示例里出现了 EmoV-DB 的文件；在 MELD 上评测。
  - [Qwen2-Audio](https://arxiv.org/abs/2407.10759) 和 [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)：只报告了 MELD 上的评测，训练数据未公开。
  - Gemini：没有公开训练数据；第三方的 [LISTEN](https://arxiv.org/abs/2510.10444) 用 CREMA-D、ESD、TESS、SAVEE、RAVDESS 评测过 [Gemini 2.5](https://arxiv.org/abs/2507.06261)。
  - [VoxEmo](https://arxiv.org/abs/2603.08936)（2026）对 Qwen2-Audio-7B-Instruct 和 AF3 做了零样本评测，用到 THAI-SER、nEMO、ESD、MEAD、CREMA-D 等 35 个语料。
  - **MCAE（SPPS 和伪句子）、HCUDB、MESS、MASC 没有找到任何音频大模型用于训练或评测的记录。**

## 5. 主要来源

- EmoBox：https://github.com/emo-box/EmoBox ；emotion2vec+：https://github.com/ddlBoJack/emotion2vec 、https://huggingface.co/emotion2vec/emotion2vec_plus_large ；emotion2vec 论文：https://arxiv.org/abs/2312.15185
- MCAE-SPPS：https://pmc.ncbi.nlm.nih.gov/articles/PMC13079751/ 、https://doi.org/10.17605/OSF.IO/9JYZC 、https://ieee-dataport.org/documents/mandarin-chinese-auditory-emotions-stimulus-database-validated-set-subject-personal
- THAI-SER：https://arxiv.org/abs/2507.09618 、https://huggingface.co/datasets/airesearch/thai-ser
- MSP-IMPROV：https://www.lab-msp.com/MSP/MSP-Improv.html 、https://www.lab-msp.com/MSP/publications/AcademicLicense-MSP-IMPROV.pdf
- nEMO：https://aclanthology.org/2024.lrec-main.1059/ 、https://huggingface.co/datasets/amu-cai/nEMO
- HCUDB：https://research.nii.ac.jp/src/en/HCUDB.html ；OGVC：https://research.nii.ac.jp/src/en/OGVC.html
- MCAE 伪句子：https://doi.org/10.3758/s13428-022-01868-7 ；MESS：https://zenodo.org/records/3813437 ；MASC：https://catalog.ldc.upenn.edu/LDC2007S09 （说明文档：https://catalog.ldc.upenn.edu/docs/LDC2007S09/readme.pdf ）
- ESD：https://github.com/HLTSingapore/Emotional-Speech-Data 、https://arxiv.org/abs/2105.14762 ；MEAD：https://github.com/uniBruce/Mead 、https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660698.pdf ；RAVDESS：https://zenodo.org/records/1188976
- Emozionalmente：https://zenodo.org/records/12616095 ；SUBESCO：https://zenodo.org/records/4526477 ；CaFE：https://zenodo.org/records/1478765 ；EmoDB 2.0：https://www.isca-archive.org/interspeech_2025/burkhardt25_interspeech.pdf ；TESS：https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF ；EmoV-DB：https://github.com/numediart/EmoV-DB
- Expresso：https://speechbot.github.io/expresso/ 、https://arxiv.org/abs/2308.05725 ；EARS：https://github.com/facebookresearch/ears_dataset 、https://arxiv.org/abs/2406.06185 ；CSEMOTIONS：https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS ；BERSt：https://arxiv.org/abs/2505.00059 ；EmoTale：https://arxiv.org/abs/2508.14548 ；CAVES：https://pmc.ncbi.nlm.nih.gov/articles/PMC11289252/ ；KESDy18：https://nanum.etri.re.kr/share/kjnoh2/KESDy18?lang=En_us ；KazEmoTTS：https://arxiv.org/abs/2404.01033 ；EmoNet-Voice：https://arxiv.org/abs/2506.09827 ；mExpresso：https://github.com/facebookresearch/seamless_communication/blob/main/docs/expressive/README.md ；GEMEP：https://www.unige.ch/cisa/gemep ；Thorsten：https://openslr.org/110/
- 音频大模型相关：LISTEN https://arxiv.org/abs/2510.10444 ；VoxEmo https://arxiv.org/abs/2603.08936 ；Kimi-Audio https://arxiv.org/abs/2504.18425 ；OSUM https://arxiv.org/abs/2501.13306 ；Qwen-Audio https://arxiv.org/abs/2311.07919 ；Qwen2-Audio https://arxiv.org/abs/2407.10759 ；Qwen2.5-Omni https://arxiv.org/abs/2503.20215 ；SALMONN https://arxiv.org/abs/2310.13289 ；Audio Flamingo https://arxiv.org/abs/2402.01831 ；AF2 https://arxiv.org/abs/2503.03983 ；AF3 https://arxiv.org/abs/2507.08128 ；SpiRit-LM https://arxiv.org/abs/2402.05755 ；Gemini 2.5 https://arxiv.org/abs/2507.06261

## 6. 已下载到本地（2026-10-23；ESD 为 2026-10-26）

在 `config/config.yaml` 里把 `download.dataset` 设为数据集名称（`all` 表示全部），然后运行 `python scripts/download_data.py`。每个文件都按托管平台公布的校验值核对过（ESD 例外，见表格最后一列），
中断后重新运行同一条命令即可续传（ESD 是 Google Drive 上的单个 zip，中断后会从头下载）。数据都在 `data/` 下，不进 git（多个许可禁止分发）。

| 数据集 | `download.dataset` | 本地位置 | 内容（实测） | 校验 |
|---|---|---|---|---|
| MCAE-SPPS | `mcae_spps` | `data/mcae_spps/` | 6,675 个 WAV（48 kHz 单声道，0.4–4.0 秒，共 2.3 h），解压在 `audio/<情绪>/`；文件名为"演员 + 情绪 + 句子编号"，如 `13171.wav` = 1 号演员（男）、愤怒、第 171 句；标签、40 人的识别率和强度在 `Corpus information.xlsx`，评分者信息在 `Rater information.xlsx`，原始评分和分析代码在 `Codes and Raw Data/` | 7 个 zip 和 16 个元数据文件按 OSF 的 SHA-256 核对 |
| THAI-SER | `thai_ser` | `data/thai_ser/` | 29 个 parquet（12.71 GB），共 27,854 行；音频和标签在同一个文件里：最多 4 路麦克风的音频（`mic_clip`、`mic_con`、`mic_middle`、`mic_zoom`），以及指定情绪、多数票情绪、一致率、演员信息等列 | 按 HuggingFace 上的 SHA-256 核对 |
| nEMO | `nemo` | `data/nemo/` | 2 个 parquet（0.53 GB），共 4,481 行：音频、情绪、原始文本和规范化文本、说话人、性别、年龄 | 按 HuggingFace 上的 SHA-256 核对 |
| MESS | `mess` | `data/mess/` | 1,800 个 WAV（24,414 Hz 单声道，1.7–3.7 秒，共 1.3 h），解压在 `audio/`。官方没有附命名规则，以下是根据文件名统计推断的：首字母是情绪（`A` 愤怒、`C` 平静、`H` 开心、`S` 悲伤，各 450 条），第 2 个字母是性别（`M` / `F`），随后的数字是说话人编号（每种性别 1–3），再后面是句子编号（`V`、`T`、`M` 加两位数字）；例如 `HM1V10_SCR.wav` | 按 Zenodo 的 MD5 核对 |
| ESD | `esd` | `data/esd/` | 35,000 个 WAV（16 kHz 单声道，1.1–8.1 秒，共 29.1 h：中文 15.7 h，英语 13.4 h），解压在 `audio/Emotion Speech Dataset/`；20 位说话人，`0001`–`0010` 说中文，`0011`–`0020` 说英语；每人把同一套 350 句话用 5 种情绪各说一遍，按 `<说话人>/<情绪>/<说话人>_<编号>.wav` 存放，情绪文件夹为 `Neutral`、`Angry`、`Happy`、`Sad`、`Surprise`；每位说话人的文本和情绪标签在 `<说话人>/<说话人>.txt`（制表符分隔：编号、文本、情绪；中文说话人的标签是中文，如"中立""生气"） | Google Drive 不公布校验值：先用 zip 自带的 CRC 自检，再核对脚本里固定的 SHA-256 |

这五个数据集在本地共占用约 26.1 GB（THAI-SER 17.3 GB，ESD 5.8 GB）。zip 解压后连同原 zip 一起保留，所以 MCAE-SPPS、MESS 和 ESD 约占下载大小的两倍；
THAI-SER 和 nEMO 是 parquet 格式（读取需要 `pyarrow`），`scripts/evaluate.py` 第一次运行时会把评估要用的音频解压到各自的 `audio/` 文件夹（THAI-SER 取录音棚电容麦那一路，线上 session 取 Zoom 录音）。
ESD 仅限研究使用：使用前要按它的 [GitHub 页面](https://github.com/HLTSingapore/Emotional-Speech-Data)的说明填写许可表，并发邮件给作者。

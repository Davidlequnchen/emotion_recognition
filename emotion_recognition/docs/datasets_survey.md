# 情绪识别公开数据集调研（Emotion Recognition Public Dataset Survey）

英文版：[datasets_survey_en.md](datasets_survey_en.md)（内容相同）。

调研日期：2026-10-23。目标：调研能用于评估情绪识别模型（emotion2vec+）和音频大模型（audio LLM）的公开情感语音数据集。
筛选标准：同一句话用不同情绪表达，排除语义线索；标签质量高；规模大；许可和获取方式清楚。

补充核对：2026-11-09 新增 IEMOCAP 和 MSP-Conversation 的对话上下文调研（第 3a 节）。两者尚未下载或加入评测，现有 13 个数据集不变。

方法：检索论文、官方项目页、GitHub、HuggingFace、Zenodo、OSF 等，汇总数据集信息。
大小以十进制 GB 计，来自各平台 API 的文件列表或 HTTP 请求；标"约"的是推算值，标"未公开"的是平台没有给出大小。
标"未核实"的数字没能从一手资料确认。已下载到本地的数据集（见第 6 节），条数、时长和采样率是我下载后实测的。

**图例（全文通用）：** ✅ = 本项目评估使用的数据集（13 个，见第 0 节末尾的表）；🗄️ = 之前评估过、现已移出评估范围（只有 ESD）；没有标记 = 本项目没有使用。

## 0. 要点

- **CREMA-D 在 EmoBox 里**，所以很可能是 emotion2vec+ 训练数据的一部分，emotion2vec+ 在 CREMA-D 上的结果偏高（不是严格的 held-out 结果）。
  调研到的数据集里，ESD、MEAD、RAVDESS、TESS、SAVEE、EmoDB、CASIA、Emozionalmente、SUBESCO、JL-Corpus、CaFE、EMOVO、EmoV-DB 也都在 EmoBox 中。
- **emotion2vec+ 训练数据的公开程度：**
  - seed 用 EmoBox 的学术数据微调（201 h）。EmoBox 全部约 294 h，具体用了哪些子集没有列出，保守起见把 EmoBox 全部视为模型已见过的数据。
  - base 和 large 用"过滤后的大规模伪标签数据"微调（4,788 h 和 42,526 h）。官方说数据工程细节以后公布，伪标签数据的来源不明，所以任何公开数据都无法 100% 排除。
  - emotion2vec 的预训练用了 IEMOCAP、MELD、CMU-MOSEI、MEAD 和 MSP-Podcast。
- **最好的"干净"候选（都不在 EmoBox 中）**：MCAE-SPPS（普通话，2025/26）、THAI-SER（泰语，规模最大）、MSP-IMPROV 的 target sentences（英语）、nEMO（波兰语，2024）。
- **英语**：没有找到同时满足"规模大、平行文本、有感知验证、不在 EmoBox"的数据集，只能把 MSP-IMPROV、MESS、Expresso 组合起来用。
- **对话上下文候选**：IEMOCAP 有逐句类别及维度标注，较容易衔接当前分类任务；MSP-Conversation 有长对话的连续 V/A/D 标注，需要另设计维度评测。两者均需申请，不能当作匿名下载、自由再分发的开放许可数据；模型训练关系见第 3a 节。
- **支持这个思路的证据**：[LISTEN](https://arxiv.org/abs/2510.10444)（2025）用 CREMA-D、ESD、TESS、SAVEE、RAVDESS 里文本中性的句子测试音频大模型。
  只给音频时，准确率只有 16.5–34.9%（[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) 34.0%，[Gemini 2.5 Pro](https://arxiv.org/abs/2507.06261) 34.9%）；只给文本时则有 85–97%。
  说明这些模型主要靠"转写成文字"，而不是真正在听声音。

### 本项目使用的数据集（13 个，2026-11-06 核对）

| 数据集 | 语言 | 类型 | 评估条数 | 时长 | 本地占用（现在） | 许可 | 我们的模型中训练时见过它的 |
|---|---|---|---|---|---|---|---|
| CREMA-D ✅ | 美式英语 | 表演 | 7,442 | 5.26 h | 0.6 GB | ODbL | emotion2vec+、MERaLiON-SER-v1 |
| MCAE-SPPS ✅ | 普通话 | 表演 | 6,675 | 2.30 h | 0.8 GB | CC BY-NC 4.0 | 无 |
| MESS ✅ | 美式英语 | 表演 | 1,800 | 1.33 h | 0.2 GB | CC BY 4.0 | 无 |
| RAVDESS ✅ | 英语（北美） | 表演 | 1,440 | 1.48 h | 0.6 GB | CC BY-NC-SA 4.0 | emotion2vec+ |
| TESS ✅ | 英语（加拿大） | 表演 | 2,800 | 1.60 h | 0.3 GB | CC BY-NC 4.0 | emotion2vec+ |
| JL-Corpus ✅ | 新西兰英语 | 表演 | 1,200（只用 5 种主要情绪） | 0.70 h | 0.9 GB | CC0 | emotion2vec+、Melati |
| THAI-SER ✅ | 泰语 | 表演（脚本 + 即兴） | 11,848：4 类 22,584 条中的高一致子集（一致率 ≥ 0.71；不含 frustrated；全部 22,584 条为 33.19 h） | 17.36 h | 16 GB（12 GB 原始 parquet + 4.4 GB 解压出的音频） | CC BY-SA 4.0 | 无 |
| nEMO ✅ | 波兰语 | 表演（9 人中 3 人是职业配音演员） | 4,481 | 3.07 h | 1.5 GB | CC BY-NC-SA 4.0 | 无 |
| MEAD ✅ | 英语 | 表演 | 8,000：27,514 条中的分层子集（不含 contempt；原始 tar 里 5 个空文件也不算；全部 27,514 条为 32.73 h） | 9.51 h | 4.1 GB（由 3.07 GB 的 audio.tar 转成的 16 kHz WAV） | 数据许可未公布 | emotion2vec+（EmoBox） |
| MELD ✅（只用测试集） | 英语 | 电视剧对白（*Friends*） | 2,608（2 条时间戳有误，不算） | 2.31 h | 0.3 GB | 仓库 GPL-3.0；片段来自有版权的电视剧 | emotion2vec+、MERaLiON-SER-v1 |
| EMOVO ✅ | 意大利语 | 表演 | 588 | 0.51 h | 0.3 GB | 许可未公布；官方站已失效，用的是非官方副本 | emotion2vec+ |
| Emozionalmente ✅ | 意大利语 | 众包，非专业说话人 | 6,902 | 6.30 h | 0.71 GB（解压后的 WAV；下载的 zip 是 0.559 GB） | CC BY 4.0 | emotion2vec+、Melati |
| SAVEE ✅ | 英式英语 | 表演（4 位男性说话人） | 480 | 0.51 h | 0.16 GB | 注册后可免费用于研究，禁止再分发；用的是 Kaggle 上的非官方副本 | emotion2vec+（EmoBox） |
| **合计** | | | **56,264**（不用子集为 86,514） | **52.2 h**（不用子集为 91.3 h） | | | |

- **子集**（2026-11-06 决定，为了节省算力；`config/evaluation.yaml` 的 `subsets`）：两个数据集对所有模型都用同一个固定子集。THAI-SER：只用一致率 ≥ 0.71 的片段（一致率 = 选了多数票标签的众包评分者所占比例），即数据集作者推荐的子集，共 11,848 条（happy 3,665、neutral 2,999、angry 2,883、sad 2,301）。MEAD：把片段按演员、情绪、强度、句子编号排序后等间隔取 8,000 条，912 个演员/情绪/强度格子各自按比例保留（3–12 条）。时长按音频文件头测得。
- **所有公布了训练数据的模型都没见过的**：MCAE-SPPS、MESS、THAI-SER、nEMO。emotion2vec+ 按 EmoBox 和预训练数据保守计算（见上面的要点）。
- "本地占用"是现在 `data/` 里的大小。zip 解压后就删除，所以 zip 类数据集算的是解压后文件的大小；THAI-SER 和 nEMO 仍包含它们的 parquet 文件。
- **ESD 🗄️ 已移出评估范围**：许可仅限研究，禁止向第三方转让任何部分，并把"测试商业系统"列为商业用途。数据还在本地，emotion2vec+ 的旧结果在 `reports/archive/`。
- **SAVEE ✅ 用的是 Kaggle 上的非官方副本**：官方版注册后可免费用于研究，许可禁止再分发（第 2.2 条），所以 Kaggle 副本不是授权镜像。它的片段不放公开网页（`public_audio: False`）；和除 ESD 外的所有数据集一样，研究评估时会发给 API 模型。
- MELD 不是平行文本（电视剧对白），所以不在第 1–3 节的调研表里。

## 1. 高质量数据集（不在 EmoBox 中）

| # | 数据集（年份，语言） | 说话人 / 条数 / 时长 / **大小** | 情绪类别（英文）；强度 | 平行文本设计 | 标签质量 | 音频；模态 | 许可；获取 | 链接 |
|---|---|---|---|---|---|---|---|---|
| 1 | **MCAE-SPPS** ✅（DataPort 2025-01，Sci Data 2026-03；普通话） | 6 名专业演员（3 男 3 女）/ 6,675 条（共录 8,400 条，评估了 8,379 条）/ 2.3 h / **0.65 GB**（7 个按情绪打包的 zip） | neutral, happy, sad, angry, fearful, disgusted, surprised，与 emotion2vec+ 的类别一一对应；有感知强度评分（1–9） | 200 句 × 7 种情绪 × 6 人 | 每条由 40 名母语听者做 7 选 1 并评强度（共 720 人）；保留条件：目标情绪得票 ≥43%，且任何其他情绪 <43%；各情绪识别率 60–82%（DataPort 版本的数字） | 专业隔音棚，48 kHz 单声道；只有音频 | CC BY-NC 4.0（OSF 上数据集的许可；论文本身是 CC BY-NC-ND 4.0）；OSF 公开。注意：OSF 上按情绪打包的 zip 是完整的 6,675 条，按情绪分的散文件夹只有 6,460 条；IEEE DataPort 副本需要订阅 | [OSF 9jyzc](https://osf.io/9jyzc/)；[IEEE DataPort（需订阅）](https://ieee-dataport.org/documents/mandarin-chinese-auditory-emotions-stimulus-database-validated-set-subject-personal) |
| 2 | **THAI-SER** ✅（数据 2021，论文 2025；泰语） | 200 名专业演员 / 27,854 条（脚本 ≈17,711 + 即兴 ≈10,143）/ 41.6 h / **12.71 GB**（HF parquet 的下载大小；GitHub zip 版 12.61 GB；本地另有解压出的评估音频 4.4 GB，共约 16 GB） | neutral, angry, happy, sad, frustrated；脚本部分有 2 级表演强度 | 脚本部分：3 句 × 5 种情绪 × 2 种强度 × 2 遍；即兴部分不平行 | 众包，每条 3–8 人；同时给出指定标签和多数票标签，以及一致率；过滤后 Krippendorff α = 0.692，人类准确率 0.772；高一致子集（一致率 ≥0.71）14,182 条 | 80 个录音棚 session（其中一类房间不隔音）+ 20 个 Zoom session；44.1 / 48 kHz；每条最多 4 路麦克风；只有音频 | **CC BY-SA 4.0**（可商用）；直接下载 | [HF airesearch/thai-ser](https://huggingface.co/datasets/airesearch/thai-ser)；[GitHub release v1](https://github.com/vistec-AI/dataset-releases/releases/tag/v1) |
| 3 | **MSP-IMPROV**（2017；美式英语） | 12 名演员（6 对）/ 共 8,438 轮，受控文本部分是 652 条 target 句 + 620 条朗读 / 时长未核实 / **未公开**（签协议后提供，含视频） | angry, happy, sad, neutral；无表演强度，但有效价（valence）和唤醒度（arousal）的维度评分 | 20 个 target 句 × 4 种情绪情境 × 12 人；target 句嵌在即兴对话里说出，另有朗读版 | MTurk 众包，每轮 ≥5 人评分 | 音视频；采样率未核实 | 学术许可，需要签协议；商业许可 8,000 美元 | [官方页（已迁到 lab-msp.com）](https://www.lab-msp.com/MSP/MSP-Improv.html)；[学术许可 PDF](https://www.lab-msp.com/MSP/publications/AcademicLicense-MSP-IMPROV.pdf) |
| 4 | **nEMO** ✅（2024；波兰语） | 9 人（4 女 5 男，其中 3 人是职业配音演员）/ 4,481 条 / 3.07 h / **0.53 GB**（HF parquet） | angry, fearful, happy, sad, surprised, neutral；无强度 | 90 句（按音素覆盖设计）× 6 种情绪 × 9 人 | 经过人工评估，只保留表达准确的录音（按 4,860 条推算约保留 92%）；没有报告一致性数值 | 在家用电容麦克风录制，24 kHz / 16-bit；只有音频 | CC BY-NC-SA 4.0；直接下载 | [HF amu-cai/nEMO](https://huggingface.co/datasets/amu-cai/nEMO)；[GitHub](https://github.com/amu-cai/nEMO) |
| 5 | **HCUDB 1+2**（发布年份未核实，经费为 2014–2016 财年；日语） | 14 + 20 名专业演员 / 4,620 + 6,600 条 / 时长未核实 / **未公开**（1 张 DVD 分发，≤4.7 GB） | 11 类（类别未核实：基于 Russell 情绪环形模型，官方页只举出 surprise, anger, contempt, sleepy/tired 等 4 类）；无表演强度 | HCUDB1：10 句 × 11 种情绪 × 3 遍；HCUDB2：其中 5 句 × 11 种情绪 × 3 遍 × 2 种表演方法 | HCUDB1 每条由 16 人评分（效价/唤醒度、各类别是否贴切、自然度）；HCUDB2 没有评分 | 48 kHz / 16-bit；只有音频 | 仅限研究，免费；需申请，文档为日文 | [NII-SRC HCUDB](https://research.nii.ac.jp/src/en/HCUDB.html)；[NII-SRC 申请说明](https://research.nii.ac.jp/src/en/register.html) |
| 6 | **MCAE 伪句子**（2023；普通话伪句） | 6 名专业演员 / 验证后 4,361 条（OSF 的 Audio 文件夹有 4,561 个文件）/ 时长未核实 / **2.42 GB**（散文件，没有打包的 zip） | happy, sad, angry, fearful, disgusted, pleasant surprise, neutral；2 级强度（普通 / 强烈，中性除外） | 没有词义的伪句子 × 陈述/疑问两种句型 × 7 种情绪 × 2 种强度 | 每条由 40 名母语听者做强迫选择 | 采样率未核实；只有音频 | 许可未核实；OSF 只读链接 | [OSF 只读链接（项目名 MCAE-NS）](https://osf.io/sfbm6/?view_only=e22a521e2a7d44c6b3343e11b88f39e3) |
| 7 | **MESS** ✅（2019；美式英语） | 6 人（3 男 3 女）/ 1,800 条（Zenodo 公开版；设计为 90 句 × 4 种情绪 × 6 人 = 2,160 条）/ 1.3 h / **0.19 GB**（zip） | angry, calm, happy, sad；有感知的激活度和愉悦度评分 | 90 句 × 4 种情绪 × 6 人 | 论文中每条由 10 名听者评分（类别 + 两个维度），但**评分数据需向作者索取**；Zenodo 包只有表演标签（写在文件名里） | 24,414 Hz 单声道；只有音频 | CC BY 4.0；直接下载（注意：不是 EmoBox 里的 MESD） | [Zenodo 3813437](https://zenodo.org/records/3813437) |
| 8 | **MASC**（LDC2007S09，2007；普通话） | 68 人（23 女 45 男）/ 25,636 条 / 时长未公布 / **未公开**（约 0.5 GB，按 8 kHz / 16-bit 推算，另含 WMA 原始文件） | neutral, angry, elated, panic, sad；无强度 | 每种情绪都读同一套 5 个短语和 20 个句子（12 句内容中性，另 8 句四种情绪各 2 句），各读 3 遍；另有 2 段中性段落 | 只有表演标签，说明文档没有提到听辨验证 | 在安静办公室用录音笔录制（22.05 kHz）；发布版是 8 kHz / 16-bit，另附 44 kHz 的 WMA 原始文件 | LDC 许可，非会员需要付费 | [LDC2007S09](https://catalog.ldc.upenn.edu/LDC2007S09) |

**排名理由：**

1. **MCAE-SPPS**：平行设计彻底，每条都有 40 人的感知标签和强度评分；200 个句子的词汇多样性远高于 CREMA-D 的 12 句；
   7 个类别正好对应 emotion2vec+ 的类别。2025 年以后才发布，没有找到它被任何音频大模型用于训练或评测的记录。
   缺点：只有 6 位说话人，许可是 NC（不能商用）。
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

| 数据集（年份，语言） | 规模 / **大小** | 情绪类别（英文）；强度 | 平行设计 | 标签 | 音频；许可 | 已知被哪些音频大模型用过 | 链接 |
|---|---|---|---|---|---|---|---|
| CREMA-D ✅（2014，美式英语） | 91 人（48 男 43 女，20–74 岁）/ 7,442 条 / 5.3 h / **0.61 GB**；在 EmoBox 中，很可能是 emotion2vec+ 的训练数据，结果偏高 | 6 类（angry, disgusted, fearful, happy, neutral, sad）；只有 IEO 一句按低/中/高 3 级强度各录一遍，其余句子和全部中性片段只标未指定（XX） | 12 个句子 × 6 种情绪 | 2,443 名众包评分者，每人评 90 条（30 条只听声音、30 条只看脸、30 条音视频）；只听声音时多数票与表演标签一致率 45.6%，音视频时 76.5%（表演标签当 ground truth） | 16 kHz 单声道（原始含音视频，我们只用音频）；ODbL（单条内容为 DbCL） | [AF1](https://arxiv.org/abs/2402.01831)/[AF2](https://arxiv.org/abs/2503.03983) 零样本评测（训练时明确排除）；[LISTEN](https://arxiv.org/abs/2510.10444)；[VoxEmo](https://arxiv.org/abs/2603.08936) | [GitHub（Git LFS）](https://github.com/CheyneyComputerScience/CREMA-D) |
| ESD 🗄️（2021，英语 + 中文；已移出评估范围：许可禁止向第三方转让，并把测试商业系统列为商业用途） | 20 人 / 35,000 条 / 29.1 h / **2.45 GB**（zip） | 5 类（neutral, angry, happy, sad, surprise）；无强度 | 每人 350 句 × 5 种情绪 | 只有表演标签；论文只用自动情绪识别模型检验（中文 92%，英文 89%），没有人工听辨 | 室内录制，信噪比 > 20 dB，16 kHz；仅限研究，需填表并发邮件申请 | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT（29 h）；[OSUM](https://arxiv.org/abs/2501.13306) 训练和测试；[LISTEN](https://arxiv.org/abs/2510.10444)；[VoxEmo](https://arxiv.org/abs/2603.08936) | [GitHub（许可表 + Google Drive 链接）](https://github.com/HLTSingapore/Emotional-Speech-Data) |
| MEAD ✅（2020，英语） | 60 人（公开的 Part0 只有 48 人，已确认）/ 31,729 条 / 37.3 h / **音频 3.07 GB**（48 人的 audio.tar，下载大小；转成 16 kHz WAV 后本地 4.1 GB）；视频约 0.4 TB | 8 类（angry, disgust, contempt, fear, happy, sad, surprised, neutral）；3 级强度 | 每种情绪 30 句（3 句通用 + 7 句与该情绪相关 + 20 句一般句），不同情绪用不同的句表；同一情绪的不同强度用相同句子 | 100 人参与的视频评估 | 48 kHz，音视频；按使用条款 | [emotion2vec](https://arxiv.org/abs/2312.15185) 预训练；[VoxEmo](https://arxiv.org/abs/2603.08936) | [Google Drive Part0](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)；[百度网盘（提取码 th8e）](https://pan.baidu.com/s/1zYKz19kmQH6vnO2nogPk4Q)；[GitHub](https://github.com/uniBruce/Mead)（官方项目页现已 404） |
| RAVDESS ✅（2018，英语） | 24 人 / 1,440 条 / 1.5 h / **语音音频 0.208 GB**；全部（含歌唱和视频）25.6 GB | 8 类（neutral, calm, happy, sad, angry, fearful, disgust, surprised）；2 级强度 | 2 个句子 | 每个文件评 10 次（共 247 人） | 48 kHz，音视频；CC BY-NC-SA | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT；[AF1](https://arxiv.org/abs/2402.01831)/[AF2](https://arxiv.org/abs/2503.03983) 零样本评测 | [Zenodo 1188976](https://zenodo.org/records/1188976) |
| Emozionalmente ✅（意大利语） | 431 人 / 6,902 条 / 6.3 h / **0.559 GB**（v1.1 的 zip，下载大小；解压后的 WAV 0.71 GB） | 7 类（anger, disgust, fear, joy, sadness, surprise, neutrality） | 18 个句子 | 829 名听者，每条 5 次评价，人类 UAR 66% | 16 kHz，非专业设备；CC BY 4.0 | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 12616095（v1.1）](https://zenodo.org/records/12616095)；[OpenSLR 161](https://openslr.org/161/) |
| SUBESCO（2021，孟加拉语） | 20 人 / 7,000 条 / 7.8 h / **1.665 GB** | 7 类（anger, disgust, fear, happiness, neutral, sadness, surprise） | 10 句 × 5 遍 | 50 名评分者，准确率 71% | CC BY 4.0 | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 4526477](https://zenodo.org/records/4526477) |
| CaFE（2018，加拿大法语） | 12 人 / 936 条 / **0.187 GB**（48 kHz 版；192 kHz 版 2.03 GB） | 7 类（anger, disgust, joy, neutral, fear, surprise, sadness；官方文件夹名为法语 Colère, Dégoût, Joie, Neutre, Peur, Surprise, Tristesse）；2 级强度 | 6 个句子 | 未核实 | 192 kHz 和 48 kHz；CC BY-NC-SA | [VoxEmo](https://arxiv.org/abs/2603.08936) | [Zenodo 1478765](https://zenodo.org/records/1478765) |
| EmoDB / **EmoDB 2.0**（2025，德语） | 10 人 / 535 条，2.0 版新增 281 条"模糊"样本（共 817 条）/ **0.041 GB**（原版）；2.0 版约 0.11 GB | 7 类（anger, boredom, disgust, fear, happiness, neutral, sadness） | 10 个句子 | 20 名评审的一致率和自然度评分 | 消声室录制；audb 2.0 版为 CC0-1.0 | 新增的 281 条不在 EmoBox 中，可以用来测试模糊样本 | [原版 emodb.bilderbar.info](http://emodb.bilderbar.info/)；[audb emodb 2.0.0](https://github.com/audeering/emodb)（[说明页](https://audeering.github.io/datasets/datasets/emodb.html)） |
| EMOVO ✅（意大利语） | 6 人 / 588 条 / **0.253 GB**（Kaggle 非官方镜像；本项目用 DagsHub 副本，本地约 0.3 GB） | 7 类（disgust, fear, anger, joy, surprise, sadness, neutral） | 14 个句子 | 两组各 24 名听者，总体 80% | — | [VoxEmo](https://arxiv.org/abs/2603.08936) | 官方站 voice.fub.it 当天无法访问；[Kaggle 镜像](https://www.kaggle.com/datasets/sourabhy/emovo-italian-ser-dataset)；[DagsHub 镜像](https://dagshub.com/DagsHub/audio-datasets/src/main/EMOVO) |
| TESS ✅（英语） | 2 人 / 2,800 条 / 1.6 h / **0.281 GB** | 7 类（anger, disgust, fear, happiness, pleasant surprise, sadness, neutral） | 200 个词 × "Say the word ___" 句式 | 未核实 | CC BY-NC 4.0 | [AF1](https://arxiv.org/abs/2402.01831) 训练 | [Borealis Dataverse](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF) |
| SAVEE ✅（英式英语；使用 Kaggle 非官方副本，许可需要注册） | 4 人 / 480 条 / 官方版**未公开**；Kaggle 纯音频镜像 0.113 GB（非官方，解压后 0.16 GB） | 7 类（anger, disgust, fear, happiness, sadness, surprise, neutral） | 每种情绪 15 句，只有 3 句跨情绪共用 | 10 名评估者（官网证书报错，当天没能复核） | 音视频 | [Kimi-Audio](https://arxiv.org/abs/2504.18425) SFT | [下载页（需注册后登录）](http://kahlan.eps.surrey.ac.uk/savee/Download.html)；[注册](http://www.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/Register.html)；[Kaggle 镜像](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee) |
| CASIA（中文） | 4 人 / 1,200 条 / 官方版**未公开**；Kaggle 镜像 0.059 GB（非官方，许可不明） | 6 类（angry, fear, happy, neutral, sad, surprise；据引用该库的论文，官方文档未核实） | 平行（未核实） | 只有表演标签 | 许可不清楚 | [VoxEmo](https://arxiv.org/abs/2603.08936) | 官方渠道未核实；[Kaggle 镜像](https://www.kaggle.com/datasets/rocklagoon/casia-emotion-recognition) |
| JL-Corpus ✅（新西兰英语） | 4 人 / 2,400 条 / 1.4 h / **1.28 GB**（Kaggle 整包，含感知测试材料和图片；录音本身 0.45 GB） | 10 类：5 种主要情绪（happy, angry, neutral, sad, excited）+ 5 种次要情绪（enthusiastic, apologetic, pensive, worried, anxious；公开文件名里的次要情绪是 anxious, apologetic, assertive, concerned, encouraging） | 主要情绪：15 句 × 5 种情绪，句子完全相同；次要情绪另用一套句子，其中 2 句带情绪词 | 120 人测试：主要情绪约 70%，次要情绪约 40% | CC0 | [AF1](https://arxiv.org/abs/2402.01831) 训练 | [Kaggle（作者本人上传）](https://www.kaggle.com/datasets/tli725/jl-corpus)；[GitHub](https://github.com/tli725/JL-Corpus) |
| EmoV-DB（英语） | 4 人 / 6,887 条 / 9.5 h / **约 1.41 GB**（18 个 tar.gz） | neutral, amused, angry, disgusted, sleepy | 用 CMU Arctic 的句子，跨情绪平行 | 只有表演标签；句子前后常有笑声或哈欠 | 非商用 | [AF1](https://arxiv.org/abs/2402.01831) 训练；[Qwen-Audio](https://arxiv.org/abs/2311.07919) 论文的 SFT 示例里出现了 EmoV-DB 的音频文件 | [OpenSLR 115](https://openslr.org/115/)；[GitHub](https://github.com/numediart/EmoV-DB) |

注：AF1、AF2 是 NVIDIA 的 Audio Flamingo 1 和 2；emotion2vec 是 emotion2vec+ 的预训练模型，不是音频大模型；LISTEN 和 VoxEmo 不是模型，而是用这些数据集评测音频大模型的论文（评测了哪些模型见第 4 节）。

## 3. 不在 EmoBox 中，但规模小、不平行，或者标签、音频有缺陷

| 数据集（年份，语言） | 规模 / **大小** | 关键设计 | 标签 | 许可 / 获取 | 主要问题 | 链接 |
|---|---|---|---|---|---|---|
| **Expresso**（2023，英语） | 4 人；朗读 11,615 条 / 11.5 h，即兴 34.4 h / **38.44 GB**（expresso.tar，含即兴对话）；HF 社区镜像只有朗读部分，5.76 GB | 7 种朗读风格（default、happy、sad、confused、enunciated、laughing、whisper）读同一套文本 | 没有感知验证 | CC BY-NC 4.0，开放下载 | 情绪类别少；音质最好（专业录音棚，48 kHz / 24-bit）；Meta 的 [SpiRit-LM](https://arxiv.org/abs/2402.05755) 用它训练过风格 tokenizer | [官方下载说明](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset)；[expresso.tar 直链](https://dl.fbaipublicfiles.com/textless_nlp/expresso/data/expresso.tar)；[HF 社区镜像](https://huggingface.co/datasets/ylacombe/expresso) |
| KESDy18（2018，韩语） | 30 名配音演员 / 2,880 条 / **未公开** | 20 句 × 4 种情绪（angry, happy, neutral, sad） | 6 名标注者，7 类 + 唤醒度/效价（1–5） | ETRI 许可协议（站点当天无法访问） | — | [ETRI nanum](https://nanum.etri.re.kr/share/kjnoh2/KESDy18?lang=En_us) |
| IITKGP-SEHSC（印地语） | 10 人 / 12,000 条 / 约 9 h / **未公开**（约 1 GB，按 16 kHz / 16-bit 推算） | 15 句 × 8 种情绪（anger, disgust, fear, happy, neutral, sad, sarcastic, surprise）× 10 个 session | 未核实 | 向 IIT Kharagpur 申请 | 16 kHz | [作者资源页](https://cse.iitkgp.ac.in/~ksrao/res.html)；[申请声明 PDF](https://cse.iitkgp.ac.in/~ksrao/pdf/declaration.pdf) |
| CAVES（2024，粤语） | 10 人 / 3,475 条 / **未公开**（含 1080p / 50 fps 视频） | 50 个语义中性的句子 × 7 种情绪（anger, disgust, fear, happy, sad, surprise, neutral） | 15 名听者 | 仅限研究，需填表申请 | 规模小（48 kHz 隔音间录制，音视频） | [申请表](https://forms.office.com/r/3VfeWQnAVa) |
| OGVC 表演部分（日语） | 4 人 / 2,656 条 / **未公开**（整套含自然对话，2 张 DVD 分发） | 同一组台词跨 8 种情绪（joy, acceptance, fear, surprise, sadness, disgust, anger, anticipation，即 Plutchik 的 8 种基本情绪；类名来自引用它的论文，官方文档未核实）录制，4 级强度 | 未核实 | NII-SRC，免费 | 规模小 | [NII-SRC OGVC](https://research.nii.ac.jp/src/en/OGVC.html)；[NII-SRC 申请说明](https://research.nii.ac.jp/src/en/register.html) |
| MCAE-Monosyllable（2025，中文） | 6 人 / 18,089 条 / **1.18 GB**（散文件） | 422 个单音节 × 7 种情绪（neutral, happy, angry, sad, fearful, disgusted, surprised） | 720 名评分者 + 强度评分 | OSF | 只有单音节，不是句子 | [OSF h3uem](https://osf.io/h3uem/) |
| BERSt（2025，多种口音英语） | 98 人 / 4,526 条 / 3.75 h / **1.06 GB**（parquet；原始 wav 1.27 GB；整个仓库 3.18 GB） | 14 个无意义短语 × 7 种情绪（anger, disgust, fear, joy, neutral, sadness, surprise）× 说话/喊叫/尖叫 | 只有意图标签 | HuggingFace 卡片写 CC BY，论文写 CC BY-SA | 演员在家用手机录制，有远场录音，不是干净的测试集 | [HF Rosie-Lab/BERSt](https://huggingface.co/datasets/Rosie-Lab/BERSt) |
| EmoTale（2025，丹麦语 + 英语） | 18 人 / 800 条 / **0.43 GB** | 5 句 × 5 种情绪（anger, happiness, sadness, boredom, neutral） | κ 0.71–0.85 | 有版权声明但没有正式许可（GitHub） | 规模很小 | [GitHub](https://github.com/snehadas/EmoTale) |
| Thorsten Emotional（2021，德语） | 1 人 / 2,400 条 / **0.399 GB** | 300 句 × 8 种风格（neutral, amused, angry, disgusted, sleepy, surprised, drunk, whispering） | 无 | CC0 | 只有一位说话人 | [OpenSLR 110](https://openslr.org/110/)；[Zenodo 5525023](https://zenodo.org/records/5525023) |
| GEMEP（2012，伪语言） | 10 人；核心集 145 条 / **未公开** | 伪语言句子，18 类（见表下注） | 1,260 条经过验证 | 需签使用条件；完整集受限 | 能拿到的部分太小 | [Core set 页面](https://www.unige.ch/cisa/gemep/coreset)；[使用条件表 PDF](https://www.unige.ch/cisa/download_file/view/143/304) |
| EARS（2024，英语） | 107 人，共约 100 h / **69.17 GB**（107 个按说话人打包的 zip；情绪部分不能单独下载） | 22 种情绪 + neutral（见表下注） | 无 | CC BY-NC 4.0 | **不平行**：每种情绪用内容相符的不同句子，词汇本身会泄露情绪 | [GitHub](https://github.com/facebookresearch/ears_dataset)；[release "dataset"](https://github.com/facebookresearch/ears_dataset/releases/tag/dataset) |
| CSEMOTIONS（2025，中文） | 10 人 / 4,160 条 / 10.2 h / **3.08 GB**（parquet） | 7 类（neutral, happy, angry, sad, surprise, playfulness, fearful） | 未报告 | Apache-2.0 | **不平行**，文本内容与情绪一致；可以作"词汇线索"的对照组 | [HF ATH-MaaS/CSEMOTIONS](https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS)（原 AIDC-AI/CSEMOTIONS，会自动跳转） |
| KazEmoTTS（2024，哈萨克语） | 3 人 / 54,760 条 / 74.9 h / **10.46 GB**（EmoKaz.zip） | 每个句子只配一种情绪 | 听辨测试中愤怒只有 22% | CC BY 4.0 | 不平行，标签质量差 | [HF issai/KazEmoTTS](https://huggingface.co/datasets/issai/KazEmoTTS)；[GitHub](https://github.com/IS2AI/KazEmoTTS) |
| EmoNet-Voice Bench（2025） | 12,600 条 / **1.08 GB**（3 个 parquet） | 40 类（见表下注） | 专家标注，Cronbach α 0.14 | CC BY 4.0 | 用 GPT-4o 合成的语音，不平行；数据仓库在匿名账号下，由 LAION 官方合集链接 | [HF emonet-voice-bench](https://huggingface.co/datasets/t1a5anu-anon/emonet-voice-bench)；[LAION EmoNet 合集](https://huggingface.co/collections/laion/emonet-6855238709fb1b2a7295d4ed) |
| mExpresso（2023） | — / **0.0033 GB**（只有文本） | 把 Expresso 的 7 种风格翻译成包括中文在内的 5 种语言 | — | 只公开了文本，音频没有发布 | 目前不可用 | [官方说明](https://github.com/facebookresearch/seamless_communication/blob/main/docs/expressive/README.md)；[文本 tar](https://dl.fbaipublicfiles.com/seamless/datasets/mexpresso_text/mexpresso_text.tar) |

注：类别较多的数据集——

- **GEMEP** 的 18 类：admiration, amusement, anxiety, cold anger, contempt, despair, disgust, elation, hot anger, interest, panic fear, pleasure, pride, relief, sadness, shame, surprise, tenderness（据 [INTERSPEECH 2013 ComParE](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2013/Schuller13-TI2.pdf) 表 3；其中 admiration, contempt, disgust, shame, surprise, tenderness 样本很少）。
- **EARS** 的 22 种情绪：adoration, amazement, amusement, anger, confusion, contentment, cuteness, desire, disappointment, disgust, distress, embarrassment, ecstasy, fear, guilt, interest, pain, pride, realization, relief, sadness, serenity，另加 neutral（据 GitHub 仓库的 `transcripts.json`，其中拼作 `embarassment`、`extasy`）。
- **EmoNet-Voice Bench** 的 40 类：完整列表见[论文](https://arxiv.org/abs/2506.09827)附录 A.1（EmoNet-Voice Taxonomy），例如 amusement, elation, contentment, pride, awe, relief, longing, doubt, fear, distress, shame, sadness, contempt, anger, pain, helplessness, emotional numbness。

## 3a. 对话上下文候选（2026-11-09 新增，尚未加入评测）

这两项补充服务于“给同一句话增加过去对话后，模型怎样判断情绪”的研究，不要求平行文本。这里单独按用途列出，不据此把数据集标为所有模型都没见过；它们不计入第 0 节的 13 个正式评测数据集。

| # | 数据集（年份，语言） | 说话人 / 条数 / 时长 / **大小** | 情绪类别（英文）；维度 | 平行文本 / 会话结构 | 标签质量 | 音频；模态 | 许可；获取 | 链接 |
|---|---|---|---|---|---|---|---|---|
| 1 | **IEMOCAP**（2008，*Language Resources and Evaluation*；英语） | 10 名演员（5 男 5 女），5 个双人 sessions / **10,039 个说话轮次**（5,255 脚本、4,784 即兴）/ 约 **12 h**；常用四分类是 **5,531 条、约 7 h**的子集 / **官方下载包 GB 未公开，本项目未下载** | anger, sadness, happiness, disgust, fear, surprise, frustration, excited, neutral，另可选 other；常用四类为 angry/happy/sad/neutral，将 excited 并入 happy；另有 valence、activation/arousal、dominance（1–5） | 非平行；脚本及情境即兴双人对话；保留完整会话、逐句分段、转写及词/音素/音节对齐 | 原论文每条类别由 3 人评价，维度由 2 人评价；评价者按顺序看视频，可接触前文；部分条目无一致类别。不是直接把演员指令当真值 | 音频、视频、动作捕捉；原论文录制为 48 kHz，实际分发波形未本地核实 | 需电子 release 申请；限内部非商用研究、不可未经许可转交，不是开放许可或匿名直接下载 | [官方 release](https://sail.usc.edu/iemocap/iemocap_release.htm)；[原论文](https://sail.usc.edu/iemocap/Busso_2008_iemocap.pdf)；[许可](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)；[四类子集口径](https://aclanthology.org/2024.findings-acl.931.pdf) |
| 2 | **MSP-Conversation**（2020 初版，Interspeech；2026 arXiv 扩展介绍，当前 Release 2.0；英语） | 论文概览称 >450 名说话人（跨对话身份信息不完整）/ **310 段完整对话**，每段约 10–20 min；分成 **908 个 3–7 min 的 conversation parts** 用于标注 / **77 h 26 min** / **官方下载包 GB 未公开，本项目未下载** | **连续 valence、arousal、dominance，−100～100**；原生没有 angry/happy 等离散分类真值 | 非平行、自然播客会话；提供人工完成或人工修订的 speaker diarization、轮次时间边界及重叠发言信息 | 人听长对话并用 joystick 连续评分；每个 part、每个属性至少 6 名有效标注者；论文处理后的轨迹为 10 Hz（不是音频采样率） | 自然会话音频；原始采样率、是否附完整逐字转写尚未核实，不能默认与 MELD 一样已有转写 | 免费 Academic License，但需机构授权签署人签协议后申请；不是自由开放下载/再分发；Test2 真值不公开 | [官方 Release 2.0](https://lab-msp.com/MSP/MSP-Conversation.html)；[2020 论文](https://www.isca-archive.org/interspeech_2020/martinezlucas20_interspeech.html)；[2026 论文](https://arxiv.org/html/2603.22536v1)；[协议](https://lab-msp.com/MSP/publications/Busso-FDPDTUA-MSP-Conversation-v2.pdf) |

**怎样理解 ground truth。** 两者都有人工情绪标注，但标注的是观察者的情绪感知，不是能直接验证的说话人内心状态。IEMOCAP 可按预先固定的类别和一致性规则接入分类；不能将全量 10,039 条与四类 5,531 条混写，原论文当时的维度标注也未全部完成，获取包后应核对实际完整性。MSP-Conversation 是随时间变化的三条评分曲线：valence 表示负面到正面，arousal 表示平静到激动，dominance 表示弱势到强势。它适合维度回归及动态分析，不能随意切阈值后称为官方 angry/happy 标签。

**MSP-Conversation 的“条数”与公开真值。** 12,555 是 Release 2.0 与 MSP-Podcast 重叠的 speaking turns，**不是整个 Conversation 的逐句 clip 总数**，也不是已核实的本项目模型训练重叠条数。官方按长对话组织；最终切成多少片段取决于轮次/窗口方案。划分为 Train 169 对话（41 h 22 min）、Dev 52（12 h 36 min）、Test1 58（14 h 17 min）、Test2 31（9 h 9 min）；Test2 标签保留作外部评测，不随 release 公开。Conversation 与 Podcast 的标注还同时存在连续/逐句、量表等差异，不能把差异全部归因于上下文。[2026 论文 §V](https://arxiv.org/html/2603.22536v1)

**存储预算与下载大小分开。** 官方公开页没有可核对的原始包字节数，本项目尚未取得数据，所以 GB 栏保留“未公开”。仅供规划：若按公布时长保存一份 **16 kHz、单声道、PCM16** 音频，12 h 约 **1.38 GB**，77 h 26 min 约 **8.92 GB**（时长 × 16,000 × 2 字节）。这些是转换格式下的计算值，**不是原始下载包大小**，不含视频、动作数据、重复的整段/切片和中间文件。

IEMOCAP 的公开许可第 6 条还要求在计划评测、比较及公开报告前，与许可方沟通；申请时应说明本项目的研究用途。MSP-Conversation 需要机构授权签署，不是个人点击接受即可下载。两者若日后加入，API 传输与数据共享范围也应按各自获准的条款配置，不能直接沿用现有数据集的设置。[IEMOCAP 许可](https://sail.usc.edu/iemocap/Data_Release_Form_IEMOCAP.pdf)、[MSP-Conversation 获取说明](https://lab-msp.com/MSP/MSP-Conversation.html)

### 与本项目五个 SER 模型的训练关系

下表只描述公开的情绪任务训练来源。没有列出某语料，不等于能证明 Whisper/WavLM 等骨干预训练从未见过相关录音；同语料暴露也不等于已经证实测试文件泄漏。

| 模型 | IEMOCAP | MSP-Conversation |
|---|---|---|
| **emotion2vec+ large** | emotion2vec 家族公开预训练使用过 IEMOCAP；large 的伪标签来源未完全公开。存在训练来源暴露，不能作为明确未见语料；不据此断言该 large checkpoint 确定用过全部类别标签 | 未找到直接使用 Conversation 的声明；公开 emotion2vec 预训练使用 MSP-Podcast v1.8，另有 large 未公开来源。存在相关录音重叠风险 |
| **MERaLiON-SER-v1** | 官方明确声明未用于该模型训练，仅评测 | 公开 SER 训练清单未列 Conversation / Podcast；MSP-Podcast 用于评测 |
| **Melati** | 公开九个 SER 训练语料中未列出 | 公开九个 SER 训练语料中未列 Conversation / Podcast |
| **Vox-Profile Whisper** | 公开 SER 训练来源是 MSP-Podcast，未列 IEMOCAP | 用 MSP-Podcast 训练，可能与 Conversation 共享录音；未声明直接训练过 Conversation |
| **Vox-Profile WavLM** | 同 Whisper 版，未列 IEMOCAP | 同 Whisper 版，有 MSP-Podcast 来源重叠风险 |

依据：[emotion2vec 论文](https://aclanthology.org/2024.findings-acl.931.pdf)、[MERaLiON 模型卡](https://huggingface.co/MERaLiON/MERaLiON-SER-v1/raw/main/README.md)及[论文](https://arxiv.org/html/2511.04914v1)、[Melati 官方卡](https://huggingface.co/Scicom-intl/Melati-Emotion-Whisper-Large-V3)（本轮核对已下载的官方卡；在线入口访问未成功）、[Vox-Profile Whisper 卡](https://huggingface.co/tiantiaf/whisper-large-v3-msp-podcast-emotion/raw/main/README.md)、[WavLM 卡](https://huggingface.co/tiantiaf/wavlm-large-categorical-emotion/raw/main/README.md)。Podcast 版本和 train/test split 必须另行按音频 ID 核对；12,555 的交集数字不能直接套到当前两个 Vox 模型的训练集上。

**这两者的定位。** IEMOCAP 更容易衔接现有分类任务，但需先申请并固定标签规则。MSP-Conversation 更适合自然会话与连续情绪研究，不直接塞入七类分类表。若优先避免机构签约，先看第 3b 节的低门槛替代项；这两者保留为后续候选，不默认申请或接入。

## 3b. 获取更方便的对话音频候选（2026-11-09 核对，尚未加入评测）

本轮优先考虑无需机构签署或邮件审批的资源，MIT 是偏好，不将“有 GitHub”“仓库代码是 MIT”和“原音频也是 MIT”混为一谈。按用户选择，DailyTalk 和 EmotionTalk 均已下载、完成全量核验并保留；RESD 只核实元数据，暂不下载。它们均未加入当前 13 个正式评测数据集。

| 数据集（年份，语言） | 规模 / **公开包大小** | 对话与标注 | 数据许可 / 获取 | 适合本项目的程度 | 一手来源 |
|---|---|---|---|---|---|
| **DailyTalk**（2022 arXiv；ICASSP 2023，英语）＋可选 **ECSS 补标**（AAAI 2024） | 实测 **2,541 段对话、23,773 条录音、21.674 h、2 名演员**；官方 ZIP **5.341 GB**，解包 WAV **6.900 GB** | 轮次连续，每段 5–20 轮，有 speaker、转写；原情绪属性继承 DailyDialog 文本。ECSS 论文描述专业听音与语义补标，但公开类别逐条与原标签一致，仅 intensity 是新增信息，不能当作独立听感类别金标准 | **音频 CC BY-SA 4.0**，官方 Drive 直接下载，无机构签约；ECSS 仓库 MIT 不能覆盖上游音频许可 | **已保留，适合英语历史输入 pilot**。但只有两演员、表演对话，公开 ECSS 的 none 实测 **79.77%**；不适合单独证明自然人群泛化 | [DailyTalk 官方](https://github.com/keonlee9420/DailyTalk)、[论文](https://arxiv.org/html/2207.01063v3)、[官方 Drive](https://drive.google.com/drive/folders/1WRt-EprWs-2rmYxoWYT9_13omlhDHcaL)、[ECSS 官方](https://github.com/walker-hyf/ECSS)、[本地审计](../data/dailytalk/audit.json) |
| **EmotionTalk**（2025 arXiv；Findings ACL 2026，普通话） | 论文 **19 名演员、744 段对话**；固定版本实测 **742 段（734 段双人）、19,250 句、23.6025 h**；原包 **14.812 GB**，WAV **14.750 GB**；整个 HF 仓库 **36.108 GB** | 以同会话 start/end 时间合并双 speaker；七类音频 gold 与五名标注者判断/置信度。音频标签不是文本标签；标注时可回听过去，不能称孤立句投票 | **CC BY-NC-SA 4.0**；HF 登录并接受免费自动访问条件，无机构合同流程；不是 MIT | **已保留，适合中文历史 pilot**。双人对话有 **18,306 个历史目标（test 1,830）**；音频 neutral **48.72%**；val/test 有 speaker ID 重叠 | [官方卡](https://huggingface.co/datasets/BAAI/Emotiontalk)、[ACL 论文及附录 A/C](https://aclanthology.org/2026.findings-acl.440.pdf)、[本地审计](../data/emotiontalk/audit.json) |
| **RESD**（2022，俄语） | 数据卡称 20 名演员；当前 HF **1,396 条**（train 1,116、test 280），两个 parquet 共 **0.485 GB**；项目总述称约 4 h，本轮未测音频总时长 | 演员无固定台词、按指定情绪即兴双人对话，七类；标签代表演员被要求表达的情绪，不是独立听感投票。当前字段为 name/path/emotion/speech，历史顺序与说话人身份能否可靠恢复仍待核实 | **MIT**，官方数据卡明确适用于数据；HF `gated=false`，可直接下载 | **最符合 MIT 偏好，但暂非历史实验首选**。俄语、enthusiasm 等类别需另定映射；Melati 的公开训练清单已含 RESD，不能视为所有 SER 的未见语料 | [官方数据卡](https://huggingface.co/datasets/Aniemore/resd)、[版本统计](https://huggingface.co/datasets/Aniemore/resd/blob/main/dataset_infos.json)、[作者项目](https://github.com/aniemore/Aniemore) |
| **Dialogs**（Interspeech 2026，俄语） | 官方报告 **3 名演员、11,796 句、20.6 h**；公开 HF 仓库 **5.571 GB** | 人声面对面对话，脚本由 GPT-3.5 辅助、允许即兴；每句三人评价，12 个风格/情绪混合标签。公开 CSV 有 speaker/text/emotion，但未显式给 dialogue_id/turn_id；文件名线索是否足以恢复顺序未核实 | **OpenRAIL**，非 MIT；HF `gated=false`，官方明确可免费商用、无需作者另行批准，仍有使用限制 | 可进一步核对声音参照用途；在确认轮次组织前，不作为现成完整历史数据。不能把 poem/whisper 等风格直接当作七类情绪 | [官方数据卡与许可](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations)、[论文](https://arxiv.org/html/2607.14310v1) |

**DailyTalk 的实际核验结果。** 全部 WAV 的头、PCM 内容和 metadata 对应关系通过检查。ECSS 的 `train_frame.txt`＋`val_frame.txt` 有 **23,759 个唯一 ID、2,540 段对话**；归一化 `no emotion`/`none` 后，**全部类别都与 DailyTalk 原标签相同**。因此不能把两套文件解释成两组独立人工类别判断。公开文件的 none 为 18,952 条（79.77%），fear 为 18 条，与论文表中的计数不一致，实验以所用文件的实际分布为准。对话 d175 的 14 句没有 ECSS 标签，d1451 声明 19 轮但实际只有连续 10 轮；严格 pilot 排除两段后有 **2,539 段完整对话**。原文件保留，排除原因写入审计。

ID 如 `5_1_d2136` 表示第 2136 段对话的第 5 轮、speaker 1，数字轮次可恢复过去历史。全量数据有 **1,545 段多情绪对话、4,495 条有过去历史的非中性目标**，其中 1,161 条是短回应（≤3 词或≤2 秒），适合先做历史输入 pilot。实例和原文见 [pilot_examples.json](../data/dailytalk/pilot_examples.json)。它没有同一目标在 H0/H1 条件下的两套人工投票；若研究人类判断随上下文变化，仍需另收标注。[ECSS 标签与处理代码](https://github.com/walker-hyf/ECSS)

**EmotionTalk 的实际核验结果。** 全部 **19,250 对 WAV/JSON** 通过音频和配对检查。只有已完整结束的过去轮次（`endTime <= target startTime + 1ms`）才纳入历史；4,033 个目标与更早开始的轮次有重叠，不能简单按文件名取前几句。八段只发布单 speaker 音频的对话保留原文件，但从主要双人分析排除。官方 train/validation/test 分别为 **15,413 / 1,908 / 1,929 句**；train 与其余划分的发布 speaker IDs 不重叠，validation/test 共享 `02`、`13`。采用 `Audio/json` 的 `emotion_result`，neutral 为 **9,378 条**，不能套用 HF README 的 multimodal 总表。发布 gold 与简单多数票不总一致，保留原 gold 与逐人投票，不擅自覆盖。它同样没有 H0/H1 两套独立人类判断。[详细检查及具体对话例子](../data/emotiontalk/README.md)

**其他论文用什么。** [ECSS（AAAI 2024）](https://arxiv.org/pdf/2312.11947)直接用 DailyTalk 做情绪会话语音研究并补标；[EmotionTalk（Findings ACL 2026）](https://aclanthology.org/2026.findings-acl.440/)提供更丰富的音频感知标注。[情绪 M3ED（ACL 2022）](https://github.com/AIM3-RUC/RUCM3ED)也公开中文影视对话音频入口，但本轮未找到明确的数据 MIT/CC 条款，所以不因下载链接存在就认定许可宽松。不要与 `m3ed.io` 的同名机器人视觉数据集混淆。

**本轮决定。** **DailyTalk、EmotionTalk 均保留**，分别用于英语与中文历史输入 pilot；完整来源、许可、索引和审计随数据 repo 保存。**RESD 暂不下载**：官方字段只有 name/path/emotion/speech，未找到可验证的 dialogue ID、时间戳或轮次定义，不能凭文件名中的数字猜顺序。这不证明顺序绝对无法恢复，但目前不足以支持可靠的历史实验。

核查日期：2026-11-09。DailyTalk 已下载并逐文件检查；EmotionTalk 只下载 Audio.tar 与必要小文件，不下载 Multimodal.tar；RESD、Dialogs 仍仅检查公开元数据。仓库总大小不等于需要下载的音频大小；数据包与代码许可分别记录。

## 4. 注意事项

- **非商用或限制性许可**：MCAE-SPPS（NC；论文本身是 NC-ND）、nEMO（NC-SA）、Expresso 和 EARS（NC）、RAVDESS 和 CaFE（NC-SA）、
  TESS（NC）、ESD（仅限研究）、EmoV-DB（非商用）、HCUDB 和 OGVC（仅限研究）、IEMOCAP（内部非商用研究）、MSP-Conversation 和 MSP-IMPROV（学术许可）、MASC（LDC 付费）。
  许可宽松的有：THAI-SER（BY-SA）、MESS、Emozionalmente 和 SUBESCO（BY）、CSEMOTIONS（Apache）、Thorsten（CC0）。
- **需要注册或申请**：IEMOCAP（电子 release）、MSP-Conversation（机构签协议）、MSP-IMPROV（签协议）、ESD（填表 + 发邮件）、HCUDB 和 OGVC（NII-SRC，日文）、KESDy18（ETRI）、MASC（LDC）、IITKGP、GEMEP、
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

## 6. 已下载到本地（2026-10-23；ESD 为 2026-10-26；RAVDESS、TESS、JL-Corpus 为 2026-10-27；MEAD、MELD、EMOVO、Emozionalmente 为 2026-11-05；SAVEE 为 2026-11-06）

在 `config/download_data.yaml` 里把 `dataset` 设为数据集名称（`all` 表示全部），然后运行 `python scripts/common/download_data.py`。每个文件都按托管平台公布的校验值核对过（ESD、JL-Corpus、SAVEE、MEAD 和 EMOVO 例外，见表格最后一列），
中断后重新运行同一条命令即可续传（ESD、JL-Corpus 和 SAVEE 各是一个 zip，中断后会从头下载）。`data/` 现为固定版本的 `emotion_recognition_datasets` 子模块：Git 只保存来源说明和小样例，MCAE-SPPS、nEMO、TESS、JL-Corpus、EMOVO、THAI-SER、MEAD、MELD 和 SAVEE 的指定副本由 `python3 scripts/common/setup_datasets.py` 从 Release 归档恢复，归档不存入 Git 或 Git LFS。新的 nEMO 恢复直接使用原始 Parquet，无须重组旧传输分片。本段原来源下载器需加 `--source --dataset NAME`；`--source --dataset all` 不含 ESD。[data/README.md](../data/README.md) 记录准确范围、校验清单和来源条款，其中 SAVEE 的额外上传授权由仓库所有者于 2026-11-09 确认。CREMA-D、MESS、RAVDESS 和 Emozionalmente 可从 Zenodo 单独下载。

| 数据集 | `dataset` | 本地位置 | 内容（实测） | 校验 |
|---|---|---|---|---|
| MCAE-SPPS ✅ | `mcae_spps` | `data/mcae_spps/` | 6,675 个 WAV（48 kHz 单声道，0.4–4.0 秒，共 2.3 h），解压在 `audio/<情绪>/`；文件名为"演员 + 情绪 + 句子编号"，如 `13171.wav` = 1 号演员（男）、愤怒、第 171 句；标签、40 人的识别率和强度在 `Corpus information.xlsx`，评分者信息在 `Rater information.xlsx`，原始评分和分析代码在 `Codes and Raw Data/` | 7 个 zip 和 16 个元数据文件按 OSF 的 SHA-256 核对 |
| THAI-SER ✅ | `thai_ser` | `data/thai_ser/` | 29 个 parquet（12.71 GB），共 27,854 行；音频和标签在同一个文件里：最多 4 路麦克风的音频（`mic_clip`、`mic_con`、`mic_middle`、`mic_zoom`），以及指定情绪、多数票情绪、一致率、演员信息等列；评估用的音频在第一次评估时解压到 `audio/`（4.4 GB），本地共约 16 GB | 按 HuggingFace 上的 SHA-256 核对 |
| nEMO ✅ | `nemo` | `data/nemo/` | 2 个 parquet（0.53 GB），共 4,481 行：音频、情绪、原始文本和规范化文本、说话人、性别、年龄 | 按 HuggingFace 上的 SHA-256 核对 |
| MESS ✅ | `mess` | `data/mess/` | 1,800 个 WAV（24,414 Hz 单声道，1.7–3.7 秒，共 1.3 h），解压在 `audio/`。官方没有附命名规则，以下是根据文件名统计推断的：首字母是情绪（`A` angry、`C` calm、`H` happy、`S` sad，各 450 条），第 2 个字母是性别（`M` / `F`），随后的数字是说话人编号（每种性别 1–3），再后面是句子编号（`V`、`T`、`M` 加两位数字）；例如 `HM1V10_SCR.wav` | 按 Zenodo 的 MD5 核对 |
| CREMA-D ✅ | `crema_d`（`crema_d: full`） | `data/crema_d/` | 7,442 个 WAV（实测 16 kHz 单声道，1.3–5.0 秒，共 5.3 h），在 `AudioWAV/`；标签（句子、情绪、强度代码）编码在文件名里；听众的众包多数票在 `processedResults/summaryTable.csv` | 按 Git LFS 指针给出的 SHA-256 逐个核对（锚定 commit `1658cd3`，2025-03-21） |
| ESD 🗄️ | `esd` | `data/esd/` | 35,000 个 WAV（16 kHz 单声道，1.1–8.1 秒，共 29.1 h：中文 15.7 h，英语 13.4 h），解压在 `audio/Emotion Speech Dataset/`；20 位说话人，`0001`–`0010` 说中文，`0011`–`0020` 说英语；每人把同一套 350 句话用 5 种情绪各说一遍，按 `<说话人>/<情绪>/<说话人>_<编号>.wav` 存放，情绪文件夹为 `Neutral`、`Angry`、`Happy`、`Sad`、`Surprise`；每位说话人的文本和情绪标签在 `<说话人>/<说话人>.txt`（制表符分隔：编号、文本、情绪；中文说话人的标签是中文，如"中立""生气"） | Google Drive 不公布校验值：先用 zip 自带的 CRC 自检，再核对脚本里固定的 SHA-256 |
| RAVDESS ✅ | `ravdess` | `data/ravdess/` | 1,440 个 WAV（48 kHz，其中 5 个是双声道；2.9–5.3 秒，共 1.5 h），解压在 `audio/Actor_<编号>/`；文件名是 7 组数字，如 `03-01-05-01-01-01-22.wav`：模态（03 = 只有音频）、声道（01 = 说话）、情绪（01 neutral、02 calm、03 happy、04 sad、05 angry、06 fearful、07 disgust、08 surprised）、强度（01 普通、02 强烈；neutral 只有普通）、句子（01 "Kids are talking by the door"，02 "Dogs are sitting by the door"）、第几遍（01、02）、演员（01–24，单数为男，双数为女） | 按 Zenodo 的 MD5 核对；只下载语音音频的 zip（不含歌唱和视频） |
| TESS ✅ | `tess` | `data/tess/` | 2,800 个 WAV（24,414 Hz 单声道，只有 `OAF_food_fear.wav` 是 96 kHz；1.3–3.0 秒，共 1.6 h），直接放在文件夹里；文件名为"说话人_单词_情绪"，如 `OAF_back_angry.wav`：说话人 `OAF`（64 岁；`OA_bite_neutral.wav` 是写错前缀的同一人）或 `YAF`（26 岁），情绪为 angry、disgust、fear、happy、neutral、ps（pleasant surprise，惊喜）、sad，各 400 条 | 按 Borealis 的 MD5 逐个核对 |
| JL-Corpus ✅ | `jl_corpus` | `data/jl_corpus/` | Kaggle 整包 zip（17,143 个文件），解压在 `audio/`；评估用的是 `audio/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/`：2,400 个 WAV（44.1 kHz 单声道，1.2–3.8 秒，共 1.4 h），每个配一个转写文本 `.txt`；文件名为"说话人_情绪_句子编号+session_第几遍"，如 `female1_angry_10a_1.wav`：说话人 female1、female2、male1、male2，情绪为主要的 angry、excited、happy、neutral、sad 和次要的 anxious、apologetic、assertive、concerned、encouraging（与论文中次要情绪的名称不完全一样），每人每种 60 条。另有感知测试材料（录音的副本、问卷设计，没有逐条结果，还有一张含参与者个人信息的表）和图片素材（含第三方图库照片），这两部分不要转发 | Kaggle 不公布校验值：先用 zip 自带的 CRC 自检，再核对脚本里固定的 SHA-256 |
| MEAD ✅ | `mead` | `data/mead/` | 官方 Google Drive 文件夹 *Part0* 里 48 位演员各自的 `audio.tar`（不下载视频），解压到 `m4a/<演员>/<情绪>/level_<n>/<编号>.m4a`（31,734 个 AAC 文件，48 kHz 双声道；每个 tar 解压后删除），再用 ffmpeg 转成 `audio/`（同样的目录结构，16 kHz 单声道 WAV；全部转换完成后删除 `m4a/`，本地只剩 `audio/`，4.1 GB）：31,729 个 WAV（有 5 个原始文件没有音频），0.9–17.2 秒，共 37.3 h；不含 contempt 的 27,514 条共 32.7 h，评估用其中 8,000 条的分层子集（9.5 h）。情绪 angry、contempt、disgusted、fear、happy、sad、surprised 各有三个强度（1 弱、2 中、3 强；每个强度约 30 句），neutral 只有一个强度（40 句）；演员编号以 `M` 开头为男性，`W` 开头为女性 | Google Drive 不公布校验值：每个 tar 通过解压来检查 |
| MELD ✅ | `meld` | `data/meld/` | 只保留测试集：`test_sent_emo.csv`（280 段对话共 2,610 句；列：台词、说话人、情绪、情感倾向、对话和句子编号、季、集、起止时间）和 `audio/dia<对话>_utt<句子>.wav`（2,610 个 WAV，16 kHz 单声道，由测试集视频转换；0.1–16.7 秒，共 2.3 h；评估用其中 2,608 条，不含 `dia38_utt4` 和 `dia220_utt0`：这两条时间戳有误，长达 305 秒和 235 秒）。转换完成后删除 10.88 GB 的全集压缩包和视频 | 压缩包按 Hugging Face 的 SHA-256 核对，CSV（GitHub，锚定 commit `e8cedf2`）按脚本里写死的 SHA-256 核对 |
| EMOVO ✅ | `emovo` | `data/emovo/` | 588 个 WAV（48 kHz，除一个外都是双声道；1.3–14.0 秒，共 0.5 h），在 `audio/<说话人>/`；文件名为"情绪-说话人-句子"，如 `dis-f1-b1.wav`：情绪 `dis`gusto、`gio`ia、`neu`trale、`pau`ra、`rab`bia、`sor`presa、`tri`stezza（各 84 条），说话人 `f1`–`f3`、`m1`–`m3`，句子 `b1`–`b3`、`d1`–`d2`、`l1`–`l4`、`n1`–`n5`。官方网站（voice.fub.it）已失效，文件来自 DagsHub 上的非官方副本（`kingabzpro/EMOVO`） | 没有公布校验值：对全部 588 个文件计算一个 SHA-256，写死在脚本里 |
| Emozionalmente ✅ | `emozionalmente` | `data/emozionalmente/` | Zenodo 上的 zip（v1.1，下载大小 0.559 GB，解压后删除），解压到 `audio/emozionalmente/`（解压后的 WAV 0.71 GB）：`audio/` 里 6,902 个 WAV（16 kHz 单声道，0.75–13.9 秒，共 6.3 h），文件名是一串数字；`metadata/samples.csv`（文件、句子、说话人、想表达的情绪；每种情绪 986 条）、`metadata/evaluations.csv`（每条 5 位听众：听出的情绪和音质评价）、`metadata/users.csv`（说话人和听众的性别、年龄、母语，已匿名化），以及作者划分的说话人独立数据集切分 `metadata/split/` | 按 Zenodo 的 MD5 核对 |
| SAVEE ✅ | `savee` | `data/savee/` | Kaggle 上的非官方 zip（0.113 GB，解压后删除），解压在 `audio/ALL/`：480 个 WAV（44.1 kHz 单声道，1.6–7.1 秒，共 0.51 h，0.16 GB）；文件名为"说话人_情绪+句子编号"，如 `DC_a01.wav`：说话人 `DC`、`JE`、`JK`、`KL`（都是男性，各 120 条），情绪 `a` anger、`d` disgust、`f` fear、`h` happiness、`sa` sadness、`su` surprise（各 60 条，即 15 句）和 `n` neutral（120 条，30 句）。官方版需要注册，许可禁止再分发：这些文件不要转发 | Kaggle 不公布校验值：先用 zip 自带的 CRC 自检，再核对脚本里固定的 SHA-256 |

这十四个数据集在本地共占用约 30 GB（13 个评估用的约 27 GB，加上 ESD 3.2 GB；最大的是 THAI-SER 16 GB、MEAD 4.1 GB、ESD 3.2 GB、nEMO 1.5 GB，2026-11-06 实测）。zip 完整解压后就删除，所以 MCAE-SPPS、MESS、ESD、RAVDESS、JL-Corpus、Emozionalmente 和 SAVEE 只占解压后的大小；每个数据集的 `unpacked.json` 列出已解压 zip 里的文件，之后再运行下载，只有其中有文件缺失时才会重新下载 zip；
THAI-SER 和 nEMO 是 parquet 格式（读取需要 `pyarrow`），`scripts/evaluation/evaluate.py` 第一次运行时会把评估要用的音频解压到各自的 `audio/` 文件夹（THAI-SER 取录音棚电容麦那一路，线上 session 取 Zoom 录音）。
MEAD 和 MELD 转换成 WAV 需要 ffmpeg。MELD 转换完成后再运行下载，只有缺少 WAV 时才会重新下载 10.88 GB 的压缩包。
ESD 仅限研究使用：使用前要按它的 [GitHub 页面](https://github.com/HLTSingapore/Emotional-Speech-Data)的说明填写许可表，并发邮件给作者。
ESD 🗄️ 还在本地，但已移出评估范围，被 `config/evaluation.yaml` 的 `exclude` 排除（许可仅限研究，禁止向第三方转让任何部分，并把"测试商业系统"列为商业用途）；它之前的评估报告在 `reports/archive/`。THAI-SER 和 nEMO 曾被排除，2026-11-05 起重新参与评估。

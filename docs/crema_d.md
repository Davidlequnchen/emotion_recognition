# CREMA-D 数据集说明

本文介绍本项目使用的 CREMA-D 数据集：数据集概况、文件命名、两种标签（表演标签与听众感知标签）、
`summaryTable.csv` 等标签文件各列的含义，以及我们评估时生成的文件。
下载方法见 [README 第 3 节](../README.md#3-download-the-data-crema-d)。

## 1. 概况

CREMA-D（Crowd-sourced Emotional Multimodal Actors Dataset）于 2014 年发布，是一个由演员表演的英文情绪数据集。

| 项目 | 内容 |
|---|---|
| 片段数 | 7,442 个（每个片段一句话，1.3–5.0 秒，共约 5.3 小时） |
| 演员 | 91 位：48 男、43 女，20–74 岁，包括非裔、亚裔、白人、西班牙裔等 |
| 句子 | 12 个固定的英文句子 |
| 情绪 | 6 种：愤怒、厌恶、恐惧、开心、中性、悲伤 |
| 强度 | 低、中、高、未指定（见第 2 节） |
| 模态 | 每个片段都有视频和音频；我们只用音频 `AudioWAV`（16 kHz 单声道 WAV） |
| 许可 | Open Database License（ODbL），单个内容采用 Database Contents License（DbCL） |

**为什么适合评估情绪识别模型：** 同一个演员把同一句话用不同情绪各说一遍，句子内容（语义）完全相同，
只有说话方式不同。模型因此只能靠声音本身（语调、音量、语速、音色等）判断情绪，不能靠文字内容。
这对普通情绪识别模型和音频大模型（audio LLM）都是一个干净的测试。

**需要注意：** CREMA-D 属于 [EmoBox](https://github.com/emo-box/EmoBox) 基准，而 emotion2vec+ 的训练最早就用了
EmoBox 的数据，所以在 CREMA-D 上的结果不算严格的独立测试，可能偏高。

引用：H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova and R. Verma, "CREMA-D: Crowd-sourced Emotional
Multimodal Actors Dataset", *IEEE Transactions on Affective Computing*, 5(4), 377–390, 2014.
官方仓库：<https://github.com/CheyneyComputerScience/CREMA-D>

## 2. 文件命名

以 `1001_DFA_ANG_XX.wav` 为例，文件名由下划线分成四部分：`演员编号_句子_情绪_强度`。

**演员编号**：`1001` 到 `1091` 的四位数字。每位演员的年龄、性别、种族见 `VideoDemographics.csv`。

**句子**（12 个，三个字母的缩写）：

| 缩写 | 句子 | 中文意思 |
|---|---|---|
| IEO | It's eleven o'clock. | 现在十一点了。 |
| TIE | That is exactly what happened. | 事情就是这样发生的。 |
| IOM | I'm on my way to the meeting. | 我正在去开会的路上。 |
| IWW | I wonder what this is about. | 我想知道这是怎么回事。 |
| TAI | The airplane is almost full. | 飞机快坐满了。 |
| MTI | Maybe tomorrow it will be cold. | 明天可能会冷。 |
| IWL | I would like a new alarm clock. | 我想要一个新闹钟。 |
| ITH | I think I have a doctor's appointment. | 我好像约了医生。 |
| DFA | Don't forget a jacket. | 别忘了带外套。 |
| ITS | I think I've seen this before. | 我好像以前见过这个。 |
| TSI | The surface is slick. | 表面很滑。 |
| WSI | We'll stop in a couple of minutes. | 我们几分钟后停下来。 |

**情绪**（三个字母的代码；括号里是我们文件中使用的英文名）：

| 代码 | 情绪 | 片段数 |
|---|---|---|
| ANG | 愤怒（angry） | 1,271 |
| DIS | 厌恶（disgusted） | 1,271 |
| FEA | 恐惧（fearful） | 1,271 |
| HAP | 开心（happy） | 1,271 |
| NEU | 中性（neutral） | 1,087 |
| SAD | 悲伤（sad） | 1,271 |

**强度**（两个字母的代码）：`LO` 低、`MD` 中、`HI` 高、`XX` 未指定。只有句子 IEO 按低、中、高三种强度各录了一遍
（各 455 个片段）；其余 11 个句子和所有中性片段都只有一种，标为 `XX`（6,077 个片段）。

## 3. 两种标签：表演标签与感知标签

CREMA-D 的每个片段都有两类标签：

1. **表演标签（acted emotion）**：演员被要求表演的情绪，写在文件名里。这是演员的意图，每个片段都有。
   我们的评估把它当作 ground truth。
2. **感知标签（perceived emotion）**：其他人看到或听到后判断的情绪，通过众包评分得到。

**评分方式：** 共 2,443 名评分者，每人评 90 个片段：30 个只听声音、30 个只看视频画面（无声）、
30 个音视频同时呈现。95% 的片段有 7 个以上的评分。每次评分时，评分者先从 6 种情绪中选一种，
再给出这种情绪的强度（1–100 的分值）。

**表演标签和感知标签经常不一致。** 下表是三种呈现方式下评分者多数票与表演标签一致的比例（不计平票）：

| 呈现方式 | 多数票与表演标签一致 | 多数票为中性的片段 | 平票片段 |
|---|---|---|---|
| 只听声音（voice） | 45.6% | 52% | 644 |
| 只看脸（face） | 70.7% | 28% | 470 |
| 音视频（audio-visual） | 76.5% | 26% | 420 |

可以看出，只听声音判断情绪对人也很难，而且人倾向于选"中性"。音视频同时呈现时人判断得最准。
我们评估的模型只用声音，所以报告里拿它与"只听声音"的听众做对比。

## 4. `processedResults/summaryTable.csv` 各列含义

每行对应一个片段（共 7,442 行），汇总三种呈现方式下评分者的**多数票**和**平均强度**。
情绪用一个字母表示：`A` 愤怒、`D` 厌恶、`F` 恐惧、`H` 开心、`N` 中性、`S` 悲伤。

| 列 | 含义 |
|---|---|
| `FileName` | 片段名（不含扩展名），如 `1001_DFA_SAD_XX` |
| `VoiceVote` | **只听声音**的评分者的多数票，即得票最多的情绪。几种情绪票数相同时用冒号连接，表示平票，如 `N:S` |
| `VoiceLevel` | 与 `VoiceVote` 对应的平均强度：选了该情绪的评分者给出的强度（1–100）的平均值。平票时用冒号分隔，顺序与 `VoiceVote` 相同 |
| `FaceVote` | **只看画面**（无声视频）的评分者的多数票 |
| `FaceLevel` | 与 `FaceVote` 对应的平均强度 |
| `MultiModalVote` | **音视频同时呈现**时评分者的多数票 |
| `MultiModalLevel` | 与 `MultiModalVote` 对应的平均强度 |

**例 1：** `1001_DFA_SAD_XX`（演员 1001 用悲伤的情绪说 "Don't forget a jacket"）

| FileName | VoiceVote | VoiceLevel | FaceVote | FaceLevel | MultiModalVote | MultiModalLevel |
|---|---|---|---|---|---|---|
| 1001_DFA_SAD_XX | N | 46.8 | N | 66.71 | N | 63.6 |

演员想表演悲伤，但只听声音的 9 个有效评分里，5 人选了中性、2 人厌恶、1 人悲伤、1 人愤怒，所以 `VoiceVote` 是 `N`；
这 5 位选中性的人给出的平均强度是 46.8。只看脸和音视频时，多数人也选了中性。

**例 2（平票）：** `1001_IEO_ANG_MD` 的 `VoiceVote` 是 `A:D`、`VoiceLevel` 是 `37.2:58.6`：只听声音时，选愤怒和选厌恶的人数相同；
选愤怒的人平均强度为 37.2，选厌恶的人平均强度为 58.6。

**核对说明：** 我们用官方的原始评分 `finishedResponses.csv` 抽查了 400 个片段，重算多数票与表中一致的比例为：
声音 91%、脸 99%、音视频 99%。差异来自作者整理时过滤掉了少量评分（例 1 的原始数据里有 10 个声音评分，表中只计 9 个），
官方没有说明过滤规则。所以请以表中的值为准。

## 5. 其他标签文件

`python scripts/download_data.py --subset full` 会把以下文件和音频一起下载到 `data/crema_d/`：

**`processedResults/tabulatedVotes.csv`**：逐片段、逐呈现方式的投票明细，共 22,326 行（7,442 × 3）。
第一列（无列名）是编号，首位数字表示呈现方式：`1` 只听声音、`2` 只看脸、`3` 音视频；后面几位是片段序号。

| 列 | 含义 |
|---|---|
| `A`、`D`、`F`、`H`、`N`、`S` | 选择各情绪的票数 |
| `fileName` | 片段名 |
| `numResponses` | 有效评分数 |
| `agreement` | 一致率：多数票的票数占比，如 5/9 = 0.56 |
| `emoVote` | 多数票情绪 |
| `meanEmoResp`、`medianEmoResp` | 所有评分强度的平均值、中位数 |
| `meanAngerResp` … `meanSadResp` | 选择各情绪的评分者给出的平均强度 |
| 以 `Norm` 结尾的列 | 上面各值的归一化版本 |

**`VideoDemographics.csv`**：演员信息，列为 `ActorID`（演员编号）、`Age`（录制时年龄）、`Sex`（性别）、
`Race`（种族）、`Ethnicity`（是否西班牙裔）。可以用来按演员划分训练集和测试集（说话人无关实验），或按性别、年龄分组分析。

**`SentenceFilenames.csv`**：全部 7,442 个片段的列表（`Stimulus_Number` 序号、`Filename` 片段名）。

**`README.md`、`LICENSE.txt`**：官方说明和许可证。

官方仓库还有原始评分（`finishedResponses.csv` 等，每条记录是一个评分者对一个片段的一次评分）、MP3 音频和视频。
我们的下载脚本没有下载它们，需要时见 README 第 3.3 节的 git-lfs 方法。

## 6. 已知问题

- `1076_MTI_SAD_XX.wav` **没有声音**（全为 0）。这是官方列出的问题；我们的评估保留了它，以便每个片段都计入。
- 官方还列出了几个有问题的 MP3 和视频文件，但对应的 WAV 是正常的，不影响我们。

## 7. 我们生成的文件

运行 `python scripts/evaluate_crema_d.py` 后会生成：

| 文件 | 内容 |
|---|---|
| `reports/crema_d_emotion2vec_plus_large.md` | 评估报告：两种准确率、各情绪结果、两个混淆矩阵、与听众的对比 |
| `reports/figures/crema_d_emotion2vec_plus_large_confusion.png` | 混淆矩阵图（严格标准：以表演标签为 ground truth） |
| `reports/figures/crema_d_emotion2vec_plus_large_confusion_acted_or_listener.png` | 混淆矩阵图（宽松标准：表演标签或听众投票，命中其一即算正确） |
| `outputs/crema_d_emotion2vec_plus_large_predictions.csv` | 每个片段一行的预测结果（不进 git） |
| `outputs/crema_d_emotion2vec_plus_large_predictions.json` | 上面 CSV 的说明（数据集、模型、哪一列是 ground truth、各列含义），供 `scripts/make_viewer.py` 读取 |

**预测 CSV 各列含义：**

| 列 | 含义 |
|---|---|
| `audio_path` | 音频文件路径（相对于仓库根目录） |
| `acted_emotion` | 表演标签，即 ground truth |
| `listener_vote` | 只听声音的听众多数票，即 `summaryTable.csv` 的 `VoiceVote`，转成了完整英文名；平票写成 `neutral or sad` 的形式 |
| `prediction` | 模型预测：在 CREMA-D 的 6 种情绪中取概率最大的一个。报告里的准确率按这一列计算 |
| `prediction_9_classes` | 模型预测：在模型全部 9 个类别中取概率最大的一个，可能是 CREMA-D 没有的 `other`、`surprised`、`<unk>` |
| `correct` | 严格标准：`prediction` 是否等于 `acted_emotion`（True / False） |
| `correct_acted_or_listener` | 宽松标准：`prediction` 等于 `acted_emotion`，或等于 `listener_vote`（平票时等于其中任一种情绪），就算正确 |
| `p_angry` … `p_unk` | 模型对 9 个类别的概率（softmax 输出，加起来等于 1）；`p_unk` 对应模型的 `<unk>` 类 |

运行 `python scripts/make_viewer.py` 会生成 `outputs/viewer.html`：随机挑选预测正确和错误的片段，
可以在浏览器里播放，并同时看到表演标签、听众投票和模型预测。

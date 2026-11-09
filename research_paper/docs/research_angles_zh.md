# 研究方向（Research Angles）

基于我们的语音情感识别（SER）基准写一篇论文的思路：5 个专用 SER 模型、5 个音频大模型（audio LLM），13 个公开情感语音
数据集。目前是头脑风暴阶段；2026-11-06 用通俗语言重写。英文版：[research_angles.md](research_angles.md)。另见
[related_work.md](related_work.md)、[venues.md](venues.md)、[experimental_design.md](experimental_design.md)。

**基本规则。** 只用公开数据，不用公司数据，不做可能变成产品或专利的东西：我们只测试已有的模型和方法。"未核实"（unverified）
= 没有从一手来源确认；"估计"（estimate）= 我们自己算的；"初步"（preliminary）= 来自还没跑完的实验。

**图例。** 🥇 主要贡献 1 · 🥈 主要贡献 2（分量一样，但风险更大）· 🧱 地基（必须有，但单独不算新）· 🥉 有自己结果的支撑章节 ·
🔹 小发现（一段话或一张表）· 🔍 在别处分析中 · 💤 暂停或搁置 · 📎 只提一两句 · 🆕 本版新增

## 优先级总览

"Mac 时间" = 我们的 Mac M4 Max（48 GB）跑三个开源音频大模型（Qwen2-Audio、Qwen2.5-Omni、Gemma 4）所需的时间。API 费用
不是问题（用户的决定）。

| # | 方向 | 角色 | 风险 | Mac 时间 | 状态 |
|---|---|---|---|---|---|
| A | 在 prompt 里放音频示例（上下文学习，in-context learning，ICL） | 🥇 | 中 | 核心 ≈ 14 h（留余量 17 h）+ 扩展 ≈ 15 h | 设计完成 |
| B | 用 GEPA 自动改写 prompt | 🥈 | 高 | 合计 ≈ 30–50 h | 设计完成 |
| 0 | 零样本基准，10 个模型 × 13 个数据集 | 🧱 | 低 | 第 2 阶段 ≈ 36 h | 进行中 |
| C | 训练数据重叠和模型"家谱" | 🥉 | 低 | 不需要 | 有初步数字 |
| D | 同一段音频、同一个 prompt，答案会变吗？ | 🥉 | 低–中 | 每个模型 0–6 h | 设计完成 |
| E | 只看文字的对照（语音转文字 → 文本大模型） | 🔹 | 低 | ≈ 1 h | 以后 |
| F | 🆕 标签概率、去掉 neutral 偏好、三种读法对比 | 🔹 | 低 | features 完整时无需额外大模型推理；另需 CPU 分析 | probe 已实现；先验校正待做 |
| G | 演员想演的情绪 vs 听众听到的情绪 | 🔹 | 低 | 不需要 | 部分完成 |
| H | 模型给的把握程度靠谱吗？（校准） | 🔹 | 低 | 不需要 | 以后 |
| I | JEV 式结构化输出 | 💤 暂时搁置（2026-11-06） | — | — | — |
| J | 电话音质 | 💤 | 高 | 每种编码跑 1 次 SER | 暂停 |
| K | 音频长度和补零（padding） | 📎 | 低 | 不需要 | 只提一句 |

---

## 论文概要

**一句话结论：** 只改 prompt，能不能让音频大模型更好地听出情绪？

**为什么重要。** "音频大模型不如 SER 模型，而且太爱答 neutral"，这已经有人说过了：LISTEN
（[EACL 2026](https://aclanthology.org/2026.eacl-long.274/)）、VoxEmo（[arXiv 2603.08936](https://arxiv.org/abs/2603.08936)）、
EmoBox（[Interspeech 2024](https://arxiv.org/abs/2406.07162)）、VocalAffectBench
（[arXiv 2608.28932](https://arxiv.org/abs/2608.28932)）。所以光做基准不够。真正没回答的问题是：差距有多少来自 prompt 下的
读出方式和标签偏好，有多少来自当前表示的限制？F 用监督 probe 检查保存的表示是否能读出标签信息，但不能单靠它把原因分清。
相关线索包括：开源音频大模型的音频编码器里保留了说话风格，但到回答前就丢了（"Heard but Not Heeded"，
[arXiv 2609.00727](https://arxiv.org/abs/2609.00727)），在 prompt 里描述声学线索也有帮助
（[arXiv 2606.07309](https://arxiv.org/abs/2606.07309)）；但随机示例对 SER 没用（Piao et al.，
[SLT 2026](https://arxiv.org/abs/2603.20433)），GEPA 用在 omni 模型的音视频问答上也只有 −0.3 到 +4.1 分
（[arXiv 2610.20085](https://arxiv.org/abs/2610.20085)）。

**怎么做。** 两种只动 prompt 的方法：🥇 **给示例**（A）和 🥈 **改写指令**（B）。两者用同一批测试音频，都和 SER 模型比，
指标也一样：accuracy、UA（unweighted accuracy，各类准确率的平均，大类不会占便宜）、macro-F1、**neutral 率**（答
"neutral" 的比例）、**去掉 neutral 后的准确率**（提升是不是只因为少说了 neutral？），以及计划增加的 **先验校正基线**（prior
correction，见 F）。C、D、E 用来确认这些数字靠得住。

| RQ | 问题 | 章节 |
|---|---|---|
| RQ1 | 每个语料上零样本差距有多大？其中多少来自"爱说 neutral"？ | 🧱 0 |
| RQ2 | 哪种音频示例有用？模型到底有没有把示例的声音和它的标签对应起来？ | 🥇 A |
| RQ3 | 优化过的指令有用吗？能"听"错例的教练比只能"读"错例的教练更好吗？ | 🥈 B |
| RQ4 | 提升能不能迁移到别的语料、语言和模型？A 和 B 能叠加吗？ | A + B |
| RQ5 | 我们到底在测什么：训练数据重叠、文字线索、每次运行的随机波动？ | C、D、E |

**贡献。** (1) 第一个有对照的 SER 音频 ICL 研究：5 个平行语料（文字不泄露答案；其中 4 个没被任何公开训练数据的模型见过），
3 个开源和 2 个 API 音频大模型。(2) 第一次把 GEPA 用在音频大模型的副语言任务上（关注"怎么说"，不是"说了什么"），包括一个
能听音频的教练。(3) 一个 10 × 13 的基准，区分见过/没见过的语料，并给出噪声底线和只看文字的对照。

**标题候选**："Can You Prompt an Audio LLM to Hear Emotion? Demonstrations vs Optimised Instructions on 13 Corpora"；
"Steering Audio LLMs Through the Prompt for Speech Emotion Recognition"。**大纲**（8 页，ACL long）：实验设置；零样本差距
（RQ1）；示例（RQ2）；指令（RQ3）；迁移（RQ4）；我们在测什么（RQ5，加上演员 vs 听众标签）；Limitations。**以后也许**：
C、G、H、J 可以另写一篇投语音会议（Interspeech 或 TASLP；见 [venues.md](venues.md)）。

**最接近的工作，以及它们留给我们的空白**

| 工作 | 做了什么 | 留给我们的空白 |
|---|---|---|
| LISTEN（[EACL 2026](https://aclanthology.org/2026.eacl-long.274/)） | 6 个音频大模型；模型"在转写而不是在听" | 没有 SER 模型、示例或 prompt 优化；用的是人工转写稿 |
| VoxEmo（[arXiv 2603.08936](https://arxiv.org/abs/2603.08936)） | 35 个语料；换 prompt 时 macro-F1 波动 > 20 分；5 个 prompt 投票 | 没有闭源模型，没有优化，没有重复调用 |
| VocalAffectBench（[arXiv 2608.28932](https://arxiv.org/abs/2608.28932)） | gpt-realtime-2 准确率 27.9%，45.3% 的回答是 "neutral" | 只有一个内部语料（280 条） |
| EmoBox（[Interspeech 2024](https://arxiv.org/abs/2406.07162)） | 32 个语料，冻结特征 | 没有大模型 |
| Piao et al.（[SLT 2026](https://arxiv.org/abs/2603.20433)） | 在 CREMA-D 上给 6 个音频大模型 1–8 个随机示例；示例只修好了格式，没提升任务本身 | 没有同说话人、类别平衡或检索来的示例 |
| Wong et al.，A*STAR（[ASRU 2025](https://doi.org/10.1109/ASRU65441.2025.11434787)） | 副语言 ICL "有一定程度"的效果 | 没有对照，没有闭源模型 |
| AudioICL-Bench（[SLT 2026](https://arxiv.org/abs/2609.11252)） | 听起来容易时，音频大模型能学会新的"声音→标签"对应 | 没有情绪任务 |
| 音视频问答上的 GEPA（[arXiv 2610.20085](https://arxiv.org/abs/2610.20085)） | −0.32 / +0.95 / +4.12 分 | 不是情绪，没有能听的教练 |
| GEPA（[ICLR 2026 oral](https://arxiv.org/abs/2507.19457)） | 文本任务；平均比 GRPO 高 6%，模型调用次数最多少 35 倍 | 从没用在音频上 |

---

## A 和 B 的共同设置

**模型。** Mac 上：Qwen2-Audio-7B-Instruct（8.40B）、Qwen2.5-Omni-7B（Thinker，加载 8.93B）、Gemma 4 E4B-it（8.00B，
有效 4.5B）。API（公司电脑）：GPT Realtime 和 Gemini（gemini-3.8-flash）。5 个 SER 模型是参照，不给它们 prompt。

**语料。** 5 个平行语料：每句话都用每种情绪说一遍，所以文字本身不会泄露答案。另外 8 个语料用来测迁移。

| 语料 | 语言 | 说话人 | 类别 | 为什么选它 |
|---|---|---|---|---|
| CREMA-D | 英语 | 91 | 6 | Piao et al. 用的语料；每条 4–12 个听众投票 |
| MCAE-SPPS | 普通话 | 6 | 7 | 没被见过，最新（2025/26）；每条 40 个听众 |
| nEMO | 波兰语 | 9 | 6 | 没被见过；大多是非专业说话人 |
| THAI-SER（脚本部分、高一致性子集，5,012 条） | 泰语 | 200 | 4 | 没被见过；声调语言 |
| MESS | 英语 | 6 | 4（calm = neutral） | 没被见过；考 calm 和 neutral 的区别 |

**数据划分**（🆕，每个语料一个文件；同一个说话人不会出现在两部分）：训练（示例池和 GEPA 反馈）/ 验证（GEPA 打分）/
测试说话人 = CREMA-D 61 / 15 / 15；THAI-SER 140 / 30 / 30；nEMO 5 / 2 / 2；MCAE-SPPS 和 MESS 4 / 1 / 1（2 折，验证和测试
说话人互换）。

**测试音频。** 每个语料 500 条，按情绪和说话人平衡，**所有条件下用同一批音频**。这样每次比较都是"配对"的：同一条音频在
两种 prompt 下对比（McNemar 检验、配对 bootstrap）。500 条上一个准确率的标准误约 2 分；2,500 条上配对检验能看出约 2–3 分
的差别。

**Mac 速度**（2026-11-06 实测，零样本，一次一条）：Gemma 4 **0.47 s**、Qwen2.5-Omni **0.66 s**、Qwen2-Audio **1.00 s**
每条。1 秒音频 ≈ 25 个 token，所以多一条 2.5 s 的示例，大约多花 0.13 s 到一次调用时间的 30%（两种估计，彼此相差约 30%；
[experimental_design.md](experimental_design.md) §5.1）。Qwen2-Audio 会把每条音频补到 30 s（`padding="max_length"`），
每个示例约多 0.15 s（未核实）。

| 模型 | k = 0（实测） | k = 1 | k = 3 | k = 5 | k = 7 |
|---|---|---|---|---|---|
| Gemma 4 E4B | 0.47 s | ≈ 0.6 s | ≈ 0.9 s | ≈ 1.2 s | ≈ 1.5 s |
| Qwen2.5-Omni | 0.66 s | ≈ 0.8 s | ≈ 1.1–1.3 s | ≈ 1.3–1.7 s | ≈ 1.6–2.1 s |
| Qwen2-Audio | 1.00 s | ≈ 1.3 s | ≈ 1.8–1.9 s | ≈ 2.3–2.5 s | ≈ 2.8–3.1 s |

k = 示例音频的条数（每条约 2.5 s，最长 4 s）。MPS 每遇到新的输入长度都要重新编译，所以时间按 ×1.2 预留；把示例统一补到
3.0 s、复用共享的示例前缀可以缓解。

---

## 🥇 A. 在 prompt 里放音频示例（有对照的上下文学习）

**一句话结论：** 让模型在测试音频之前先听几条带标签的示例，再用对照实验看它是真的从示例里学到了东西，还是只抄了标签列表。

**为什么重要。** Piao et al. 发现随机示例没用。但没人试过人会觉得有用的那种示例：**同一个说话人**说各种情绪。情绪是相对的：
同样偏高的声音，对一个人是正常，对另一个人就是激动。如果模型能用上"*这个人*平静时是这样说话的"，这就是新发现。如果用不上，
我们的对照能说明**为什么**。

**"k-shot" 是什么意思。** k = 测试音频之前放了几条带标签的示例；k = 0 就是普通基准。什么都不训练，示例只是放在 prompt 里：

```text
[system]    ... Answer with exactly one word: angry, disgusted, fearful, happy, neutral, sad, surprised.
[user]      Example 1: <audio: example clip 1>  Which emotion does the voice express?
[assistant] angry
[user]      Example 2: <audio: example clip 2>  Which emotion does the voice express?
[assistant] sad
[user]      <audio: TEST clip>  Which emotion does the voice express?
[assistant] → 打分
```

我们的零样本 prompt 已经能保证格式正确（2026-11-05 的 18 条试跑没有一个无效回答），所以和 Piao et al. 不同，提升不可能来自
"修好了格式"。

### 怎么做

| 变量 | 改了什么 | 能告诉我们什么 |
|---|---|---|
| **k** | 0、1、3、5、每类一条 | 如果示例有信息，越多应该越好 |
| **平衡 vs 随机** | 每种情绪一条 vs 随机 k 条 | 随机示例会让模型偏向刚看到的标签（[Zhao et al. 2021](https://arxiv.org/abs/2102.09690)） |
| **同一说话人 vs 其他说话人** | 测试说话人的其他录音 vs 同性别的另一个人 | 模型能不能把说话人的平常声音当参照？经过专门训练后有用（[Ihori et al., ASRU 2025](https://arxiv.org/abs/2509.08344)：7-shot 提升 8.2 分），但没人在现成大模型上测过。**A 的主要新意** |
| **检索 vs 随机** | 选和测试音频听起来最像的几条 | 在一个语料上让 Voxtral 的 macro-F1 从 0.30 升到 0.49（[Huang et al. 2026](https://arxiv.org/abs/2606.14784)）；在 Whisper ASR 上也有效（[Wang et al., ICASSP 2024](https://arxiv.org/abs/2309.07081)） |
| **打乱标签**（对照） | 同样的音频，标签互相调换 | 准确率不降 = 模型根本没管哪个声音配哪个标签 |
| **只有标签没有音频**（对照） | "Example 1: [no audio] → angry" | 区分"看到标签"和"听到示例" |
| **顺序**（对照） | 同样的示例换个顺序 | 答案有多少次只是照抄最后一个示例的标签 |

**两个陷阱。** (1) **"落单"泄露**：如果示例是测试说话人用其他情绪说*测试那句话*，那正确情绪那条只能换一句话，结果它一落单
就把答案暴露了。所以我们用**同说话人参照集**（🆕）：测试说话人用每种情绪说**另外同一句话**；其他说话人的参照集也这样构造，
两者**只差在谁说**。(2) **检索可能泄露标签**：如果用情绪模型（emotion2vec+）来找相似音频，检索出来的示例标签本身就能猜中
答案。所以我们报告一个**抄答案基线**（copy baseline，🆕）："取检索示例里最多的那个标签"，不用大模型。大模型必须比它好。
我们也用不懂情绪的嵌入（说话人特征或 eGeMAPS 特征）来检索。

**分阶段设计**（每条测试音频单独随机抽示例，固定随机种子；调用次数按每个模型算）：

| 阶段 | 条件 | 音频数 | 调用次数 | Mac 小时（Gemma / Omni / Qwen2-Audio） |
|---|---|---|---|---|
| 0. 多音频检查 | 20 个 prompt，每个 3 条音频，问一个能核对的问题（"转写第 2 条"） | — | ≈ 50 | 几分钟 |
| **1. 核心**（RQ2） | C0 零样本（复用）；C1 随机 k = 3、其他说话人（复现 Piao）；C2 平衡、其他说话人；C3 平衡、**同一说话人**；C4 = C3 但打乱标签 | 5 × 500 | 10,000 | 3.2 / 4.0 / 6.5 h；**13.7 h**（×1.2 后 ≈ 17 h） |
| 2. 扩展 | 随机 k = 1、5；检索 k = 3（两种嵌入 + 抄答案基线）；只有标签；换顺序；**neutral 锚点**（🆕：一条同说话人的 neutral 音频）；每类 2 条 | 3 × 500 | 10,500 | 3.3 / 4.3 / 6.9 h；14.5 h（只跑最好的两个 8–11 h） |
| 3. API | 阶段 1 + 选定的扩展 | 同上 | ≈ 20,500 | 公司电脑；GPT Realtime 每条 ≈ 5.5 s，要并行发 |
| 可选 | 3 组示例多数投票（Wong et al.） | 2 × 500 | 3,000 | 每个模型 1–2.5 h |

阶段 1 就能回答核心问题：示例有没有用（C1、C2 vs C0）？说话人重不重要（C3 vs C2）？模型有没有用到"声音↔标签"的对应
（C4 vs C3）？阶段 2 用 CREMA-D、MCAE-SPPS 和 THAI-SER。

**一个 prompt 能放多条音频吗？** Qwen2-Audio **能**（[card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)；每条补到
30 s）。Qwen2.5-Omni **能**（[card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)）。Gemma 4 **大概能**
（[card](https://huggingface.co/google/gemma-4-E4B-it) 没说；我们装的 processor 支持；它的局部注意力层只往回看 512 个
token，所以前面的示例只有全局层能看到）。Gemini **大概能**（最多 9.5 小时音频、内联 20 MB；[docs](https://aistudio.google.com/docs/audio)）。
GPT Realtime **待验证**，先发一次测试调用。音频大模型常常处理不好多条音频（"Beyond Single-Audio"，
[EMNLP 2024 Findings](https://arxiv.org/abs/2409.18680)），所以要有阶段 0。

**最接近的两篇论文。** Piao et al.，"When Demonstrations Fail"（[SLT 2026](https://arxiv.org/abs/2603.20433)）：6 个音频
大模型，包括 Qwen2-Audio、Qwen2.5-Omni 和 Gemini 2.5 Flash；示例 "improve format compliance but fail to improve the core
task performance"；打乱标签后 "SER performance remains comparable"；遮住示例标签 SER 几乎归零，遮住示例音频却没影响。我们在
CREMA-D 上的 C1 + C4 复现他们的关键结果，其余都是新的。Wong、Huzaifah、Chen、Aw（A*STAR）
（[ASRU 2025](https://doi.org/10.1109/ASRU65441.2025.11434787)，[record](https://oar.a-star.edu.sg/communities-collections/articles/22939)）：
开源模型在口吃、发音和 SER 上 "already exhibit some degree of speech in-context learning"；细节不公开，所以**动笔前先从
IEEE Xplore 拿到原文**。两个结论可以同时成立；我们的对照来判断。A*STAR 在新加坡：作者很可能是审稿人。

**可能的结果（编的例子）。** 说话人参照有用：C3 在没见过的语料上胜过 C2，C4 掉回 C0。若 neutral 率下降，而未来的先验
校正（F）给出相近提升，标签偏好可能贡献了部分收益，仍不能断言示例只改变习惯。若所有条件都没有提升，只能说明这些受测 prompt
没有奏效，不能据此证明编码器没有情绪信息。

**需要多少算力和时间（我们的 Mac）：** 阶段 1 ≈ 14 h（两个晚上）；阶段 2 ≈ 15 h。

**风险：** "没提升"的可能性不小，但有这些对照，它依然是一个清楚的答案；Gemma 4 可能不支持多音频（阶段 0 会发现）。

---

## 🥈 B. 用 GEPA 自动改写 prompt

**一句话结论：** 让另一个大模型读音频大模型的错题，一轮一轮地改写指令；看这样能不能帮它听出情绪。

**为什么重要。** 光是 prompt 措辞就能让 SER 的 macro-F1 变化超过 20 分（VoxEmo），而且音频大模型太爱说 "neutral"（我们的试跑
里 Gemma 4 基本都答 "neutral"，尽管它能正确转写语音内容）。这正是 prompt 能改的地方。而且**还没人试过**在音频大模型上用
prompt 优化器做 SER 或任何副语言任务（2026-11-06 检索）。

**GEPA 是什么，通俗说**（Agrawal et al.，[ICLR 2026 oral](https://arxiv.org/abs/2507.19457)）。它改的是 prompt 的文字，
模型权重完全不动。一轮是这样：(1) 用当前 prompt 让音频大模型跑几条训练音频；(2) 一个**反思模型**（reflection LM，就是另一个
当"教练"的大模型）读 prompt、答案和反馈（"正确答案：angry；你答了 neutral"），写出更好的 prompt；(3) 如果更好，就在验证集上
打分；(4) 保留 **Pareto 前沿**（Pareto front）：只要在至少一条验证音频上是最好的 prompt，都留下。可以想成一支专家队伍：一个
prompt 擅长 angry，另一个擅长 sad；GEPA 把它们都留着继续改，而不是只打磨一个。(5) **rollout 预算**用完就停（一次 rollout =
音频大模型对一条音频的一次调用）。

论文里的数字（文本任务，Qwen3-8B 和 GPT-4.1 mini）：每次运行 **1,839–7,051 次 rollout**，大部分花在验证上（训练 rollout 只有
**79–737** 次）；**17–90 次教练调用**；平均比 GRPO（24,000 次 rollout）高 6%（最多 20%），比 MIPROv2 高 10% 以上；在一个模型
上优化的 prompt 换到另一个模型上仍有 +9.0%（原生优化是 +12.2%）。DSPy 建议用比任务模型更强的教练
（[GEPA in depth](https://dspy.ai/current/diving-deeper/gepa-in-depth/)）。

**已知对音频有用吗？没有。** 最接近的研究（[arXiv 2610.20085](https://arxiv.org/abs/2610.20085)，音视频问答，2,500 次试验）：
Qwen3-Omni Instruct **−0.32**、Qwen3-Omni Thinking **+0.95**、Gemini 2.5 Flash **+4.12** 分。所以"没提升"是真实可能的；
有对照的负面结果和 A 也很配（"示例和指令都修不好'听'"）。

**它能改的**：指令、标签定义、声学线索描述（"anger: loud, tense, fast"；见
[Juslin & Laukka 2003](https://doi.org/10.1037/0033-2909.129.5.770)）、判断规则（"calm 算 neutral"）、语言提示（"泰语是声调
语言"）。**它改不了的**：音频编码器传下来的信息。

### 怎么做

| 教练 | 每个例子能看到什么 | 风险 |
|---|---|---|
| **R-text**：强文本大模型（API） | prompt、答案、表演标签、**听众投票**（🆕，"angry 5/9，neutral 4/9"）、转写、相对说话人平均值的音高/响度/语速（eGeMAPS） | 听不见；线索描述可能是套话 |
| **R-audio**：音频大模型（Gemini Pro 或 GPT 音频模型） | 同上，**再加上错题音频**和一条答对的同情绪音频 | 它自己听情绪也不行（gpt-realtime-2 27.9%）；音频要发给 API（13 个语料都允许，ESD 绝不发） |
| R-self（可选）：音频大模型自己 | 同 R-text | 7–8B 的教练太弱 |

- **数据**：反馈集 ≈ 200 条平衡音频（训练说话人）；打分集 105–150 条平衡音频（验证说话人），这样"永远答 neutral"赢不了。反馈
  里绝不出现验证或测试标签。
- **打分**每条 1 / 0；minibatch 用 7 条（每种情绪一条），保证教练总能看到类别之间的混淆。
- **在哪儿优化**：CREMA-D 和 MCAE-SPPS（有时间再加 THAI-SER）。**预算** 1,000 / **2,500（主实验）** / 7,000 次 rollout。在一个
  格子上（CREMA-D、Qwen2-Audio、R-text）跑 3 个随机种子看波动。
- **基线**：我们自己的 prompt（`prompts.yaml`）；手工修改（+ 标签定义、+ 线索描述、+ "不要默认 neutral"、简短思维链）；VoxEmo
  的 5 个 prompt 投票；同预算的随机改写搜索（教练到底有没有用？）；A 里最好的 ICL 条件，以及 GEPA + 最好的示例（RQ4）；先验
  校正（F，待实现）；SER 模型。
- **迁移（RQ4）**：到全部 13 个语料（4 个没见过的是关键）；英语和普通话来源，在泰语、波兰语、意大利语上测；一个模型的 prompt
  用到其他模型和 API 上；7 类 prompt 用到 4 类语料上。

**需要多少算力和时间（我们的 Mac，最担心的部分）。** prompt 变长加上重新编译，每次 rollout 约 0.8 / 1.05 / 1.5 s
（Gemma / Omni / Qwen2-Audio）：2,500 次 = 0.6 / 0.7 / 1.0 h；7,000 次 = 1.6 / 2.0 / 2.9 h。一个 prompt 在 13 × 500 条上测试 =
1.4 / 1.9 / 2.7 h。教练调用只要几美元。

| 模块 | 运行 | Mac 小时 |
|---|---|---|
| 试跑 | 1 次小预算（CREMA-D、Gemma 4、R-text） | 0.5 h |
| 主网格 | 3 个模型 × 2 个语料 × 2 个教练 = 12 次（+ 2 个种子） | 12–15 h |
| 12 个 prompt 在 13 个语料上测试 | 12 × 6,500 次调用 | ≈ 24 h（每个语料 300 条则 ≈ 15 h） |
| 跨模型迁移 | 每个模型最好的 prompt 用到其他模型上 | 8–13 h |
| **合计** | | **≈ 30–50 h**，大约一周的晚上和周末 |

工程：给 [gepa](https://github.com/gepa-ai/gepa) 库写一个自定义适配器；GEPA 等第 2 阶段跑完再开始。GPT Realtime 每 2,500 次
rollout 要 ≈ 4 h，除非并行。Mac 上做不了：大网格、30B 模型（Qwen3-Omni）、GRPO。

**风险：** 可能对 7–8B 模型完全没用；线索描述流于套话；R-audio 教练自己就听不好情绪。

---

## 需要和 GRPO 比吗？

**一句话结论：** 不需要；引用已发表的结果，有空闲 GPU 才加一次小规模 LoRA 微调，GRPO 写进未来工作。

**为什么审稿人可能会问。** GEPA 的卖点是"胜过 GRPO"。GRPO 是强化学习：模型对每条音频回答好几次，答对得奖励（1 分），然后
更新权重。但我们的问题是 prompt 和*训练过的 SER 模型*比，SER 模型本身已经是"训练权重"的参照。

**需要什么。** R1-AQA（[arXiv 2503.11197](https://arxiv.org/abs/2503.11197)）：在 Qwen2-Audio 上做 GRPO，**8 × H800**，MMAU
test-mini 64.5%。Omni-R1（[arXiv 2505.09439](https://arxiv.org/abs/2505.09439)）：大部分提升来自文本推理。情绪方向的 RL：
EMO-RL（[EMNLP 2025 Findings](https://arxiv.org/abs/2509.15654)）、EmotionThinker（[ICLR 2026 oral](https://arxiv.org/abs/2601.15668)）、
R1-Omni（[arXiv 2503.05379](https://arxiv.org/abs/2503.05379)）、Yu et al.（[arXiv 2603.08230](https://arxiv.org/abs/2603.08230)）、
FSA-GRPO（[arXiv 2606.02615](https://arxiv.org/abs/2606.02615)）。在我们的 Mac 上不现实：光模型就 ≈ 17 GB，Apple GPU 跑不了
vLLM，24,000 次 rollout 光采样就要 7–10 h，还没开始训练。

**决定。** 一定做：引用 RL 数字（标明设置不同），把 GRPO 列为未来工作。只有 12 月初前有 CUDA GPU 才做：在 Qwen2-Audio 上用
**和 GEPA 看到的同一批带标签音频**跑一次 **LoRA-SFT**（就是普通的按标签微调），≈ 1–2 GPU 小时。它回答"同样的数据，改 prompt
还是改权重？"。对于"一个标签、对或错"的任务，GRPO 基本等于训练一个分类器，所以 SFT 是更简单的参照。

---

## 🧱 0. 零样本基准

**一句话结论：** 10 个模型 × 13 个语料，用一个固定 prompt；其他所有东西都建立在它之上。

**为什么重要。** 单独看不新，但它量出了 A 和 B 要缩小的差距（RQ1），每条音频的输出还会喂给 C、D、F、G、H。**状态**：3 个 SER
模型 13 个语料全跑完；Vox-Profile 在跑；接下来是 3 个本地大模型（第 2 阶段，≈ 36 h）；GPT Realtime 和 Gemini 在公司电脑上做
第 3 阶段。

**🆕 第 2 阶段除了答案还会保存什么**，都来自同一次模型调用（不用额外跑；见 `config/llm.yaml`）：

- **各数据集候选标签的首 token 分数**（`token_p_*`）及其词表总质量（`token_mass`）：是生成端概率代理，不是完整标签概率，详见 F；
- **平均池化的隐藏状态**（mean-pooled hidden states）：语言模型 audio-token 位置的内部数值，每层平均成一个向量，取 4 层
  （float16，每条约 20–33 kB，不进 git）。它们受 prompt 和上下文影响。

若这些配套输出完整保存，F 的分析不需要重跑大模型；仍需单独在 CPU 上训练 probe，并完成读法对齐和比较。先验校正尚未实现。

---

## 🥉 C. SER 基准到底在测什么？训练数据重叠和模型家谱

**一句话结论：** emotion2vec+ 在训练过的语料上比在新语料上高 28 分，所以领域里很多"零样本"数字其实有一部分是记忆。

**为什么重要。** 数据污染（contamination）= 测试数据出现在训练数据里，就像学生考前看过考卷。不区分见过和没见过的语料，就没法
判断 A 和 B 的提升是不是真的。还没人这样审查过已发布的 SER 模型（最接近的：ASR 测试集，[arXiv 2505.22251](https://arxiv.org/abs/2505.22251)；
说话人泄露，[arXiv 2610.33952](https://arxiv.org/abs/2610.33952)）。

**怎么做。**

- **见过 vs 没见过。** 只有 **MCAE-SPPS、MESS、THAI-SER 和 nEMO** 没被任何公开训练数据的模型见过
  （[experimental_design.md](experimental_design.md) §2.4）。emotion2vec+ 见过另外 9 个；MERaLiON 见过 CREMA-D 和 MELD；
  Melati 见过 JL-Corpus 和 Emozionalmente；Vox-Profile 一个都没见过。音频大模型不公开训练数据，而"没公开"不等于"没见过"。
- **家谱。** 好几个模型是用 emotion2vec 打的标签学出来的，emotion2vec 相当于老师：emotion2vec+
  （[card](https://huggingface.co/emotion2vec/emotion2vec_plus_large)）、SenseVoice（[arXiv 2407.04051](https://arxiv.org/abs/2407.04051)）、
  MERaLiON-SER（[arXiv 2511.04914](https://arxiv.org/abs/2511.04914)）、C²SER（[arXiv 2502.18186](https://arxiv.org/abs/2502.18186)）。
  一个从没见过 CREMA-D 的模型，也可能继承了见过它的老师的习惯。
- 🆕 **发布日期检验**（针对大模型）：在模型发布之后才公开的语料，不可能在它的训练数据里（MCAE-SPPS 是 2025/26 年的，比
  Qwen2-Audio 的 2024-07 新）。

**初步数字**（5 个 SER 模型中的 3 个，`reports/*/summary.json`，2026-11-06，平均准确率）：

| 模型 | emotion2vec+ 见过的 9 个语料 | 4 个没见过的语料 | 差 |
|---|---|---|---|
| emotion2vec+ large（见过这 9 个） | 84.5% | 56.2% | **+28.3** |
| MERaLiON-SER-v1（见过其中 2 个） | 54.0% | 60.4% | −6.4 |
| Melati（见过其中 2 个） | 56.8% | 56.4% | +0.4 |

对其他模型来说，这 9 个语料并不更容易，所以差距来自"见过"，不是难度（扣除随机水平后：0.82 vs 0.45）。单个语料也一样：CREMA-D
86.5% 和 84.7%（见过）vs 51.4%（Melati，没见过）；JL-Corpus 99.5%（Melati，见过）vs 75.9%；Emozionalmente 86.9% 和 85.0%
（见过）vs 47.4%。但家谱不决定一切：在 MESS 上，MERaLiON（74.5%）远超它的老师 emotion2vec+（40.8%）。最终设计："谁见过什么"
表、家谱图，并推荐 4 个没见过的语料作为干净测试集。

**需要多少算力和时间：** 不需要。**风险：** 没见过的那组有两个 4 类语料（随机水平 20.2% vs 15.6%）；见过的那组有 MELD（电视剧
语音）；还没做显著性检验。

---

## 🥉 D. 同一段音频、同一个 prompt，答案会变吗？

**一句话结论：** 每条音频发 10 次，数答案变了几次，再看模型摇摆的音频是不是正好是听众意见不一的那些。

**为什么重要。** 两个 API 天生就带随机性：GPT Realtime 根本没有 temperature 设置；gemini-3.8-flash 默认 temperature 1.0，
而且思考（thinking）关不掉，最低只能设到 LOW。所以 API 只跑一次是有噪声的；又因为两个 API 都不给 logprobs，重复调用也是我们拿到
它们标签概率的唯一办法（见 F）。
A 或 B 的提升必须大于这个**噪声底线**。还没人把音频大模型的答案翻转和每条音频的听众投票对比过（相近的："Scaling Ambiguity"，
[arXiv 2601.14620](https://arxiv.org/abs/2601.14620)；错误随听众分歧增加，[arXiv 2602.03873](https://arxiv.org/abs/2602.03873)；
GPT-4o 在音乐元数据上，[arXiv 2508.12626](https://arxiv.org/abs/2508.12626)）。

**怎么做。** API：每条音频同样调用 10 次。开源模型：直接用 F 的标签概率，再在一个模型上真的采样 10 次核对。人：CREMA-D（每条
平均 9.2 票）、MCAE-SPPS（40 票）、THAI-SER（3–11 票）、Emozionalmente（5 票）的投票。指标：

- **翻转率**（flip rate）= 1 − 最常见答案的比例（10 次都是 angry = 0；6 次 angry + 4 次 neutral = 0.4）；
- **摇摆 vs 分歧**：Spearman 相关，以及用摇摆程度找出"有争议"音频的 AUROC（0.5 = 瞎猜，1 = 完美）；
- 模型分布和投票分布之间的 **Jensen–Shannon 散度**（0 = 完全一样）；
- 10 次运行之间的 **Fleiss' κ**（kappa，扣除碰巧之后的一致程度：0 = 和碰巧一样，1 = 总是一致），和听众的 κ 并排展示；
- 一次调用 vs 10 次多数投票；翻转是不是集中在 "neutral" 上？

**需要多少算力和时间。** 2,000 条音频（4 个语料 × 500）。API：每个模型 20,000 次调用（GPT Realtime ≈ 38 美元）。开源模型：
概率从第 2 阶段直接拿，不花时间；真采样 10 次 ≈ 2.6 / 3.7 / 5.6 h。

**风险：** 和人类分歧的关系可能很弱；那样 D 就只当噪声底线用。

---

## 🔹 E. 只看文字的对照：语音转文字，再交给文本大模型

**一句话结论：** 把每条音频转成文字，让文本大模型猜情绪：光靠文字能泄露多少答案？

**为什么重要。** 靠文字"做对"的模型是在读，不是在听。LISTEN 已经用人工转写稿证明过这一点（只给文字时，82.5–96.6% 的回答是
"neutral"）。我们新做的：在全部 13 个语料上跑真实的 Whisper → 文本大模型流程，包括普通话、泰语、波兰语、意大利语；给不完全
平行的语料（MEAD、MELD、SAVEE、THAI-SER 即兴部分）算一个**文字线索分**；以及给 A 和 B 做检查。AHELM 里这种流程在音频模型中
排第 6（[arXiv 2508.21376](https://arxiv.org/abs/2508.21376)）。

**需要多少算力和时间：** Whisper 跑 ≈ 6,500 条，大约一小时。**风险：** 低；泰语和波兰语的转写质量。

---

## 🔹 F. 🆕 标签概率、去掉 neutral 偏好、三种读法对比

**一句话结论：** 对同一条音频，分别看模型说了什么、生成端更倾向哪个标签，以及一个受过监督的小分类器能从保存的表示中读出什么。
这三种读法帮助我们定位差距，但不能单靠一个分数判定模型是否“听懂了”。

**三种读法分别测什么。**

| 读法 | 怎么得到 | 它回答的问题 |
|---|---|---|
| (a) **说出口的答案**（one-word answer） | 解析大模型实际生成的回答 | 在当前 prompt 下，模型直接做这个任务表现如何？这是原来的零样本读法。 |
| (b) **标签分数**（`token_p_*`，计划比较先验校正前后） | 看生成第一个 token 时，各标签对应 token 的分数，再在标签之间归一化 | 正确标签是否已有较高分数，却没有成为最后的回答？改变标签间的相对偏好能改善多少？ |
| (c) **线性探针**（linear probe） | 冻结大模型，用带标签的训练音频教一个小分类器读取隐藏向量 | 在当前表示、训练数据和线性读出限制下，有多少情绪信息可以被监督学习利用？ |

**linear probe 到底训练了什么？** 大模型处理音频时，会在每层留下由许多数字组成的隐藏向量。可以把它看成已经写好的笔记。
probe 不改这些笔记，也不更新大模型参数；它只学一套“怎样读这些数字来分情绪”的权重。对每个类别，把向量各维乘以对应权重、相加，
再加一个偏置，就得到该类的分数：`score_k = w_k · z + b_k`。多分类时用 softmax 将分数变成分类概率。
这里的“线性”指打分只有这一层加权求和，没有额外的隐藏层。

本项目用的是 L2 正则化的 `LogisticRegression`：输入是标准化后的保存向量 `z`，监督信号是训练音频的真实情绪标签。
因此，**大模型被冻结，但小分类器确实接受了有标签训练**。这是有监督的表示诊断，不是大模型的零样本性能，也不是给原来的零样本模型
免费换一种输出格式。`probe_p_*` 是新分类器的预测概率。

**这些向量来自哪里？** 当前保存的是语言模型在 prefill 阶段、若干层的 audio-token 位置隐藏状态，并对这些位置取平均。
它们已经经过语言模型处理，会受 prompt 和上下文影响，不能直接称为“纯 audio encoder 输出”。默认要求 `pooling=audio`；定位音频
token 失败、改为对整个 prompt 取平均的向量会被拒绝。即使只平均 audio-token 位置，也不等于把 prompt 的影响去掉了。

**当前实现怎样避免把测试答案教给 probe？**

1. 先按说话人划分 train、validation、test，同一个人的录音不能跨分区；同一数据集的不同模型复用同一份 manifest。
   划分可以用标签平衡类别，但不用隐藏向量或模型预测。缺失 features 不能成为某个模型另抽一份测试集的理由。
2. 只用 train 拟合 `StandardScaler` 和分类器；只用 validation 的 UAR 选择正则化参数 `C`，再选择层。
   选好后不再用 train+validation 重训，也不能看 test 哪层最高再决定报告哪层。
3. 最后在 test 上报告结果。与 (a)、(b) 或 SER 比较时，必须按音频 ID 对齐到**同一批 eligible test clips、同一目标标签和计分规则**。
   不能把 probe 的测试子集分数直接减去原来全数据集的 LLM/SER 分数。比如排除了复合标签，其他读法也要按同样的测试条目与目标重算；
   对齐样本和真值也不等于任务完全相同：JL-Corpus 排除 excited 后，probe 训练四类，但原 LLM 推理仍有五个候选标签；这一候选集差异
   不能靠筛选测试条目消除，须保留并单独说明。probe 得到过有标签监督这一差别也仍要明确写出。

TESS 说话人不足以划成三个互斥分区，当前默认跳过；MELD 目前只有官方 test 的 features，也默认跳过，不能用它训练后再称为正式测试表现。
完整输入要求、例外协议、运行命令和导出文件见 [linear probe 操作说明](../../emotion_recognition/scripts/analysis/README.md)。

**标签分数和先验校正要分开说。** 原始 LLM CSV 的 `p_*` 只是回答的 one-hot 编码，不是置信度；应查看 `token_p_*` 和
`token_mass`。前者来自生成第一个 token 时、标签相关 token 的概率质量，并在标签之间归一化；后者记录这些 token 占整个词表概率的
多少。比如归一化后 angry 占 60%，但所有标签 token 总共只占词表的 1%，不能解释成“模型有 60% 把握这条音频是 angry”。
首 token 打分也不是完整标签序列的精确概率；共享首 token、后续 token、回答格式和重试都可能影响它与最终答案的对应关系。

**先验校正目前尚未实现**，是计划增加的对照，不是 linear probe 脚本自动执行的步骤。一个候选方案是在 train 上估计各标签的平均
`token_p_*`，用它调整每条音频的相对分数，再归一化取最高项；规则及任何参数须在 train/validation 上确定，不能用 test 来调。
平均分数高既可能来自模型的输出偏好，也可能来自真实类别比例或样本难度，不能先假定每类“理应平均分”。校正若有效，说明改变这种
读出规则有帮助；即便它与 A/B 提升相近，也不能据此断定 A/B 只是在消除 neutral 习惯。

**结果可以支持什么？**

| 观察 | 较稳妥的解释 |
|---|---|
| probe 强，直接回答弱 | 当前导出的表示包含可被监督线性分类器利用的标签信息；值得进一步研究原生回答为何没有充分利用它。不能仅凭此断言大模型已经“理解情绪”，或差距全在输出端。 |
| 首 token 分数或未来的先验校正优于直接回答 | 当前生成答案和这套标签读出之间存在差距；需继续检查标签 token、格式及类别偏好。 |
| probe 也弱 | 当前层、池化、训练样本、划分和线性分类器的组合没有读出强信号。不能证明 encoder 没有情绪信息，也不能证明换 prompt 无效。 |

**API 模型**不提供隐藏状态，所以没有 (c)。它们也不给答案的概率（2026-11-06 已查清，见 [api_logprobs.md](api_logprobs.md)）。
GPT Realtime 的答案没有 logprobs：它唯一的 `include` 选项给的是输入音频转写文字的概率，而且转写是另一个模型做的。gemini-3.8-flash
完全不返回 logprobs，Google 对 3.x 模型是故意去掉的。所以 API 模型改用 D 的重复采样（比如在一个子集上每条调 10 次）：10 次里
6 次 "angry" ≈ 60%。

**需要多少算力和时间：** 若配套 features 已完整保存，probe 分析不需要再跑大模型、读音频或用 GPU；在 CPU 上读取向量并训练即可。
实际时间和 RAM 取决于样本数、维度、层数及 `C` 搜索，不能统一承诺“几分钟”。目前已实现的是线性 probe 训练与导出，以上三种读法的
配对比较和先验校正仍需另做。说话人少的语料容易过拟合；最终应报告多个 seed，并按说话人或对话分组估计不确定性。

---

## 🔹 G. 演员想演的情绪 vs 听众听到的情绪

**一句话结论：** 模型复现的是演员想演的情绪，还是听众听到的情绪？模型排名会不会因此改变？

**为什么重要。** 表演标签不是全部真相；模型在某条音频上"答错"，可能恰恰听到了人听到的东西。CREMA-D：听众多数意见只在 6,798
条中的 45.6% 上和演员一致；在 3,699 条有争议的音频上，用表演数据训练的模型 77–82% 跟演员走，用播客训练的 Vox-Profile 62–65%
跟听众走。MCAE-SPPS：多数意见和表演标签 **100%** 一致，所以要用**投票比例**。THAI-SER：即使在高一致性子集里，也有 **2,698** /
11,848 条多数意见和标签不同（1,676 条多数是 "frustrated"，904 条是 "neutral"）。Emozionalmente：人类 UAR 66%。最接近的工作：
Label Authority in SER（[arXiv 2610.26446](https://arxiv.org/abs/2610.26446)）。

**需要多少算力和时间：** 不需要。**风险：** 低；一段话加一张表。

---

## 🔹 H. 模型给的把握程度靠谱吗？（校准）

**一句话结论：** 模型说"80% 是 angry"时，它真的大约 80% 的时候是对的吗？它的概率和听众投票比例像不像？

**为什么重要。** 一个很自信地答错的模型，比一个知道自己没把握的模型更糟。**怎么做**：在有投票的 4 个语料上，把 SER 输出、F 和
D 和投票对比，用 **ECE**（expected calibration error，"有多大把握"和"有多常答对"之间的平均差距；0 = 完全诚实）和 Jensen–Shannon
散度。部分已有人做：[Chou et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html)、
[Kuan et al. 2026](https://arxiv.org/abs/2604.25591)。

**需要多少算力和时间：** 不需要（用 D 和 F 的数据）。**风险：** 低；一张表。

---

## 💤 I. JEV 式结构化输出：暂时搁置（2026-11-06）

JEV 是一个闭源的文本和图像产品，没有音频；而 JEV 式的语音模型已经有了（ASOM，[arXiv 2610.22401](https://arxiv.org/abs/2610.22401)）。
我们保留的是分析 F（三种读法对比）。背景见 [jev_analysis.md](jev_analysis.md)。

## 💤 J. 电话音质

模型在电话音质（G.711、AMR-NB、8 kHz）上损失多少？基本还没人做（EMO-Codec，[APSIPA 2024](https://arxiv.org/abs/2407.15458)）；
我们的 G.711 模拟已经有了。暂停：用户怀疑结果不明显，而且和 prompt 这条主线不搭。

## 📎 K. 音频长度和补零

基于 Whisper 的 SER 模型会把每条音频补到 30 s（或 15 s），MERaLiON 还会把补零部分也一起平均；我们 56,264 条音频里有 31,249 条
短于 3 s。在 Limitations 里提一句。

**放弃的方向**：单独研究 prompt 敏感性（VoxEmo；"Hearing the Order"，[arXiv 2510.00628](https://arxiv.org/abs/2510.00628)）；
单独研究跨语言；思维链（现在是 B 的一个基线）；LLM-as-judge（[EMNLP 2025 Findings](https://aclanthology.org/2025.findings-emnlp.25/)）；
把 SER 模型的预测作为提示给大模型（太像产品）。

---

## 还需要做的东西

| 事项 | 用于 | 状态 |
|---|---|---|
| 跑完基准（Vox-Profile、第 2 阶段 ≈ 36 h、第 3 阶段 API） | 全部 | 进行中 |
| 3 个本地大模型每条音频保存候选标签首 token 分数、`token_mass` 和平均池化隐藏状态 | F、D、H | 🆕 已设置，随第 2 阶段运行 |
| GPT Realtime 和 Gemini 的 logprobs 核查（[api_logprobs.md](api_logprobs.md)） | F、D | 🆕 已完成：都不给，改用重复采样 |
| 确定 GPT Realtime 用哪个模型：`gpt-realtime` 将于 2027-01-20 下线，官方继任者是 `gpt-realtime-2.1` | 全部 | 🆕 等用户决定 |
| 在 `evaluate.py` 里加 neutral 率、去掉 neutral 的准确率、bootstrap 和配对检验 | 全部 | 待做 |
| 划分文件和 500 条测试列表；多音频 loader；示例抽取器 | A、B | 新增 |
| 线性 probe 的分组划分、CPU 训练和 CSV/JSON 导出 | F | 已实现；正式结果需完整 features 后运行 |
| 先验校正及三种读法的配对比较 | F | 待实现 |
| 围绕 `Predictor` 的 GEPA 适配器；教练 API；转写、eGeMAPS、投票表 | B、E、G | 新增 |
| 在 CUDA GPU 上做 LoRA-SFT 参照 | GRPO 问题 | 可选 |
| 发布代码、划分、prompt、每条预测；不发布受限音频 | 论文 | 投稿时 |

## 待定的决定

1. 投 ACL 2027（ARR 截止 2027-01-04），还是 EMNLP 2027（ARR 2027 年 5 月）多留点时间。
2. 用哪个文本大模型和音频大模型当教练；公司对每个 API 的政策（13 个语料都允许；ESD 绝不）。
3. A 阶段 2 和 B 跑全部三个开源模型，还是只跑最好的两个（Gemma 4 几乎总答 "neutral"，是最有意思的例子）。
4. 最终表格用每个语料 500 条，还是完整语料。
5. 只有 12 月初前有 CUDA GPU 才做 LoRA-SFT。
6. SAVEE 放不放（[experimental_design.md](experimental_design.md) §7.1）。
7. 在 GPT Realtime 上跑 GEPA，还是只把 prompt 迁移过去。
8. 公司审批流程和时间。

## 到 ACL 2027 的时间表（估计）

| 周 | 工作 |
|---|---|
| 11 月 6–13 日 | Mac 上跑第 2 阶段（含概率和隐藏状态）；划分；多音频 loader；阶段 0；第 3 阶段 |
| 11 月 13–20 日 | A 阶段 1；GEPA 适配器和试跑；F 的初步分析 |
| 11 月 20 日 – 12 月 4 日 | B 网格和迁移；A 阶段 2；API 上的 D |
| 12 月 4–18 日 | C、E、G 的表；图；统计 |
| 12 月 18 日 – 1 月 4 日 | 写作、Limitations、checklist；指定的 service contributor（[venues.md](venues.md) §2.3） |

很紧，但如果 Mac 从 11 月 10 日左右空出来就可行；如果 A 或 B 延误，EMNLP 2027 沿用同一个计划。

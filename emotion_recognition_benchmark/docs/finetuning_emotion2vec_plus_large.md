# emotion2vec+ large 微调评估（Fine-tuning emotion2vec+ large: an Estimate）

英文版：[finetuning_emotion2vec_plus_large_en.md](finetuning_emotion2vec_plus_large_en.md)（内容相同）。模型的结构、大小和训练过程见 [emotion2vec_plus_large.md](emotion2vec_plus_large.md)。

日期：2026-10-28。要回答的问题：微调需要多少数据、多少 GPU；只训练最后的全连接层，还是训练全部层；训练全部层需要多少样本。

方法：可训练参数按实测的模型结构计算；计算量用 PyTorch 的 FLOP counter 和 M4 Max 上的实测耗时（第 3 节）；数据量的建议来自已发表的研究，逐条给出处。
标"估算"的数字是按所列假设推算的，不是测量值。没有找到 emotion2vec 或 emotion2vec+ 的公开微调学习曲线，所以第 2 节的数据量范围是推断。

## 0. 结论

1. **算力不是瓶颈。** 有标签片段在 10 万条以内时，第 1 节的 A–D 每种做法用一块 GPU 就够。全部层微调 1 万条 5 s 片段、10 个 epoch，约 0.2–0.4 A100 小时；10 万条约 2–4 A100 小时（估算，含读数据和验证的开销）。本项目的 M4 Max 也能跑（1 万条约 1.4 h）。真正花算力的是重复训练：按说话人 5 折 × 3 个学习率 × 3 个随机种子就是 45 次。
2. **先只训练最后一层**（或"所有层的加权和 + 小分类头"），编码器冻结。emotion2vec+ large 已经在约 4 万小时情感数据上训练过，冻结的特征本身就是为情绪分类训练的；embedding 只需提取一次，分类头在 CPU 上几分钟就训练完；原有能力不会丢。几百到几千条有标签数据就够用。
3. **全部层微调**只在三个条件都满足时才值得：有约 1 万条以上的有标签片段，说话人多；目标领域和训练数据明显不同（语言、录音信道、自然对话）；冻结的做法在按说话人划分的验证集上已经不再进步。介于两者之间的是 LoRA，或只解冻最上面几个 block。
4. **训练全部层需要多少样本**：下限约 1 万条（4–6 s/条，约 11–17 h）；想要稳定地超过冻结方案，2–10 万条（约 22–170 h）更稳妥；少于约 5 千条时，全部层微调很难超过冻结编码器，而且容易过拟合到训练集的说话人。这是根据第 2 节的研究推断的。
5. **从头训练不现实**：先要在数万小时音频上做自监督预训练（data2vec 2.0 Large 用了 64 块 A100 × 76.7 h），再用约 4 万小时伪标签数据训练（约 2,500–3,800 万条），而伪标签流程没有公开。

## 1. 五种做法对比

| 做法 | 训练什么 | 可训练参数 | 需要的有标签数据 | GPU 内存 | GPU 时间（A100，估算） | 风险 | 什么时候用 |
|---|---|---|---|---|---|---|---|
| **A. 只训练最后的线性层** | 编码器冻结；先一次性提取 1024 维句级 embedding，再训练 Linear(1024→K) 或逻辑回归 | 1,025 × K（K = 9 时 9,225） | 几百条（每类 ≥ 50）到约 5 千条；1 千条 ≈ 1.1–1.7 h | 只做推理：16 s 以内 ≤ 1.2 GB（实测，含 0.66 GB 权重） | 1 万条：提取 < 1 分钟；分类头在 CPU 上几分钟 | 很低；底层特征不能适应新语言或新信道；最后一层是为 9 类训练的，新类别可能分不开 | 第一步和基线；< 5 千条；新的标签体系 |
| **B. 所有层加权和 + 小分类头** | 编码器冻结；12 个 block 输出的可学习加权和 → 小 MLP（如 1024→256→K） | 约 0.26 M | 1 千–2 万条（1.1–33 h） | 同 A。缓存每层的句级平均只要 12 × 1024 个数/条；要在帧上做注意力池化，就要缓存帧特征（5 s 一条约 6 MB，fp16） | 同 A；不缓存、每个 epoch 都前向一次时，1 万条 × 10 epoch 约 0.1 h | 低 | A 不够时。PEFT-SER、SAILER 用的就是这类下游模型 |
| **C. LoRA/adapter，或只解冻最上面几个 block** | LoRA（秩 8）加在 12 个 block 的 qkv 和输出投影上；或只训练最上面 2–4 个 block + 分类头 | LoRA：0.59 M（qkv + 投影），1.57 M（再加 FFN）；最上面 2 个 block：25.2 M，4 个：50.4 M | 5 千–5 万条（5.6–83 h） | LoRA：梯度要穿过全部 12 个 block，激活和全部层微调接近，但冻结的权重没有梯度和 Adam 状态；只训最上面几层时，只存这几层的激活 | LoRA 约为 D 的 0.5 倍，最上面 2 层约 0.35 倍（第 3.2 节的实测比例） | 中；比 D 遗忘少 | 数据量中等、领域有差异 |
| **D. 全部层微调** | 全部参数；建议至少冻结 CNN | 162.1 M（冻结 CNN：157.9 M；冻结整个前端：151.2 M） | 至少约 1 万条（11–17 h）；2–10 万条（22–170 h）收益更稳 | 固定约 2.6 GB（fp32 的权重 + 梯度 + Adam 状态）；激活 fp32 约 0.3–0.6 GB/条（4 s）、约 2.4–3 GB/条（16 s），bf16 约减半（估算）；24 GB 的卡可以跑 16–32 条 × 5 s | 1 万条 × 10 epoch：0.2–0.4 h；10 万条：2–4 h | 过拟合到训练集的说话人；遗忘（其他语言、数据里没有的类别如 other、surprised）；概率校准变化 | 数据多、领域差异大，且 A–C 已到顶 |
| **E. 继续自监督预训练（不是微调）** | 用目标领域的无标注音频继续做 data2vec 2.0 式的预训练；之后仍要用 A–D 训练分类 | 编码器全部（162.1 M）+ 重新建的 decoder（1.9 M），另需一份 EMA teacher（只占内存） | 不需要标签；从任务音频本身（TAPT，几小时）到几百小时（emotion2vec 用了 262 h） | emotion2vec 用的是 24 GB 的 A10，每块每步最多 62.5 s 音频 | base 模型 262 h × 100 epoch：4 块 A10 × 约 62 h ≈ 250 A10 小时（发表的数字）；large 1,000 h × 100 epoch 约 600 A100 小时，10–20 h × 100 epoch 约 5–10 A100 小时（估算） | 没有公开的 emotion2vec 预训练代码；`model.pt` 里没有 teacher 和 decoder；表征会改变，原来的分类头失效，必须重新训练 | 无标注音频多、标签少、领域差异大时，作为 A–D 之前的一步 |

表中的假设：
- 片段按 5 s 算；换算成小时时用 4–6 s/条。
- 计算量（实测，第 3 节）：前向 21.9 GFLOP/秒音频；全部层微调一步是前向的 3.42 倍，即约 75 GFLOP/秒音频；冻结 CNN 时 2.98 倍。
- A100：假设 bf16 下实际达到 50–100 TFLOPS（是 312 TFLOPS 稠密峰值的 16–32%；这是假设，没有测）；再乘 2，计入读数据、padding 和验证。H100 的 bf16 稠密峰值是 989 TFLOPS，按快 2–3 倍算。
- E 以 emotion2vec 发表的算力为基准：乘以 large 与 base 前向计算量之比 1.5（4 s 片段 87.6 对 58.3 GFLOP），除以 A100 与 A10 的 fp16 峰值之比 2.5（312 对 125 TFLOPS），再按小时数线性缩放。
- **从头训练**不在表中，因为不现实：data2vec 2.0 Large 的自监督预训练在 6 万小时音频上用了 64 块 A100 × 76.7 h（约 4,900 A100 小时）；emotion2vec+ large 之后还用了 42,526 h 伪标签数据（按 4–6 s/条约 2,500–3,800 万条），生成伪标签的流程没有公开；所有公开的有标签情感语料加起来也只有几百小时（EmoBox 的 32 个数据集共 294.4 h）。

## 2. 多少数据用哪种做法

| 有标签片段（按 4–6 s 折合） | 合适的做法 | 依据 |
|---|---|---|
| < 1 千条（< 1.7 h） | A；不要做全部层微调 | "冻结编码器 + 小下游模型"是 emotion2vec 论文和 EmoBox 的标准评测方式，几百条的数据集（SAVEE 480 条、RAVDESS 1,440 条）也能训练。Chen & Rudnicky 的学习曲线上，直接微调 wav2vec 2.0（V-FT）在约 0.5 h 时只有 56.8% UA |
| 1 千–5 千条（1.1–8 h） | A 或 B，可以试 C | Chen & Rudnicky：V-FT 的 UA 从 0.5 h 到 3 h 升了 9.6 个点（56.8 → 66.4），之后到约 7 h 只再升 0.9 个点。Wang et al.：在 IEMOCAP（12 h）上，冻结 CNN、只微调 Transformer 与全部层微调相当（说话人相关的设置下都更好）；作者认为数据少时训练的参数太多容易过拟合 |
| 5 千–2 万条（5.6–33 h） | B 或 C；可以试 D，但要和 B、C 比 | PEFT-SER：在 4 个数据集（4.8 千–3.8 万条）上，LoRA 比"冻结 + 下游模型"的平均 UAR 高 1–2 个点（WavLM Base+ 65.28 → 67.28；wav2vec 2.0 base 62.27 → 63.30），对 Whisper 没有提升。Wang et al.：通用 SSL 模型全部层微调比冻结高 4.7–7.3 个点 WA（IEMOCAP，说话人无关）；但只做了自监督的 emotion2vec 冻结时就有 71.8% WA，与全部微调的 wav2vec 2.0 base（70.8%）相当（两篇论文的划分不同） |
| 2 万–10 万条（22–170 h） | D（冻结 CNN、低学习率）或 C | Wagner et al.：在 MSP-Podcast 62 h 的训练集上，微调 Transformer 层比只训练输出层的 CCC 高 0.03–0.24（valence 提升最多）；arousal 和 dominance 用 25% 的训练数据就基本饱和，valence 到 75% 仍在提升。AFTER：先做 TAPT 再微调，有标签数据从 10%（约 1.8 千条）增加到 80%（约 1.4 万条，按其划分推算），UA 从 70.2% 升到 78.9%。SAILER：在 8.4 万条自然语音上，全部层微调的 WavLM Large（验证集 macro-F1 0.376）没有超过冻结编码器的 Whisper Large-v3（0.383） |
| > 10 万条，或数千小时伪标签 | D | emotion2vec+ 本身就是全部层训练的（[issue #64](https://github.com/ddlBoJack/emotion2vec/issues/64)）；large 的最后阶段每个 epoch 约 17.6 万条，训练了 30 个 epoch，学习率最高 4.1e-5（模型文档 4.3 节） |

**为什么 emotion2vec+ 的门槛更高。** 上表里全部层微调的大部分收益，来自没见过情绪标签的通用编码器（wav2vec 2.0、HuBERT、WavLM）：对它们来说，微调是在学这个任务。emotion2vec+ large 已经用约 4 万小时数据学过这 9 类，冻结的特征就是情绪特征（连只做了自监督的 emotion2vec，冻结时都与微调后的 wav2vec 2.0 base 相当）。所以对同样的类别，全部层微调主要用于适应新领域（语言、信道、说话方式），需要足够多来自这个领域的数据。如果要的是新的标签体系，作者建议用 emotion2vec（base）而不是 emotion2vec+ 来提取特征（[issue #39](https://github.com/ddlBoJack/emotion2vec/issues/39)），值得作为 A、B 的对照。

### 训练全部层需要多少样本

- **下限约 1 万条**（4–6 s/条，约 11–17 h）。说话人越多越好（几百人比几十人好），每类至少几百条。
- **2–10 万条**（约 22–170 h）时，可以期待比冻结方案稳定地好。
- **少于约 5 千条**：收益不确定，容易过拟合，用 A、B 或 C。
- 这个范围是推断，依据是上表：学习曲线在几小时、几千条后变平；PEFT-SER 中 LoRA 只多 1–2 个点；Wagner 在 62 h 上有明显收益；SAILER 在 8.4 万条时，冻结的大模型与全部微调的模型相当；以及 emotion2vec+ 作者自己最后阶段的规模。
- 最可靠的办法是在自己的数据上画一条学习曲线：用 10%、25%、50%、100% 的训练集分别训练 A 和 D，看两条曲线在哪里交叉、在哪里变平。

## 3. 本地实测和 GPU 换算

环境：M4 Max（CPU 12 个 PyTorch 线程，40 核 GPU，48 GB 内存），FunASR 1.4.16，PyTorch 2.5.1，fp32，随机波形；时间取预热后 5 次的中位数。
CPU 内存 = 进程物理内存（每 2 ms 采样一次）的峰值减去加载模型后的值；MPS 显存 = 第一次运行后 PyTorch 的 driver 分配量（含分配器缓存，偏大）。

### 3.1 推理（前向，batch 1）

| 片段长度 | 2 s | 4 s | 8 s | 16 s |
|---|---|---|---|---|
| 帧数 | 99 | 199 | 399 | 799 |
| 前向计算量 | 44.6 GFLOP | 87.6 GFLOP | 176.4 GFLOP | 365.8 GFLOP |
| CPU 耗时 | 51 ms | 88 ms | 183 ms | 379 ms |
| CPU 峰值额外内存 | 0.03 GB | 0.04 GB | 0.13 GB | 0.29 GB |
| MPS 耗时（同一长度重复运行） | 8 ms | 14 ms | 25 ms | 50 ms |
| MPS driver 显存（含 0.66 GB 权重） | 1.18 GB | 1.19 GB | 1.17 GB | 1.17 GB |

- 计算量约 22 GFLOP/秒音频，16 s 以内几乎与长度成正比；再长时注意力的 N² 项变得明显：30 s 时 24.5、60 s 时 28.1 GFLOP/秒音频。
- 更长的片段（CPU）：30 s 耗时 0.86 s、额外内存 0.56 GB；60 s 1.9 s、2.3 GB；120 s 6.7 s、9.3 GB。每个 block 的注意力分数有 16 × N × N 个 fp32：4 s 2.8 MB，16 s 42 MB，60 s 0.58 GB，120 s 2.3 GB。推理时同时有几个这样的张量；训练时 12 个 block 的都要保存到反向传播，所以长片段的显存涨得很快。
- 8 条 4 s 片段一起推理：MPS 86 ms（每条 11 ms），CPU 630 ms。MPS 不能处理长于 20.48 s 的片段（模型文档第 5 节）。

### 3.2 训练一步（fp32，AdamW）

| 配置 | MPS 每步 | MPS driver 显存 | CPU 每步 |
|---|---|---|---|
| D：4 s × 1 | 74 ms | 3.5 GB | 0.43 s |
| D：16 s × 1 | 224 ms | 5.7 GB | 1.64 s |
| D：4 s × 8 | 328 ms | 5.9 GB | 2.39 s |
| D：8 s × 8 | 681 ms | 12.8 GB | — |
| D：4 s × 16 | 631 ms | 11.7 GB | — |
| D 冻结 CNN：4 s × 8 | 247 ms | 4.7 GB | 1.86 s |
| C 近似 LoRA：4 s × 8 | 161 ms | 4.4 GB | — |
| C 只训最上面 2 个 block：4 s × 8 | 114 ms | 2.3 GB | — |

- 全部层微调时，显存里固定有约 2.6 GB：权重 0.65 + 梯度 0.65 + Adam 状态 1.30 GB。由此推算 fp32 下每条的激活约 0.3–0.6 GB（4 s）、约 1.2 GB（8 s）、约 2.4–3 GB（16 s）。
- "近似 LoRA"只训练最底层 block 的 qkv bias 和分类头，让梯度穿过全部 12 个 block，模拟 LoRA 的内存和计算（不含 LoRA 矩阵本身那一点计算）。
- 训练一步的计算量是前向的 3.42 倍（冻结 CNN 时 2.98 倍），所以"训练约为前向的 3 倍"按计算量基本成立；实测耗时（前向 + 反向，除以推理前向）在这台机器上是 3.5–4.2 倍。
- MPS 训练吞吐：4 s × 16 时每秒处理约 100 s 音频（约 7.6 TFLOPS）。照这个速度，在本机做全部层微调，1 万条 5 s 片段 × 10 epoch 约 1.4 h，10 万条约 14 h（纯计算）。
- PyTorch 2.5.1 的 MPS 在这个模型的 Conv1d 反向传播中报错（`view size is not compatible with input tensor's size and stride`）。测量时用一个小的 autograd 函数把传入 Conv1d 的梯度变成连续内存来绕过；冻结整个前端（CNN + 投影 + 位置卷积）也能避开。CPU 上没有这个问题；CUDA 没有测。

### 3.3 换算到 A100（估算）

假设同第 1 节（50–100 TFLOPS 实际吞吐，再乘 2 的额外开销）：

| 有标签片段（5 s/条） | A/B：提取一次 embedding | D：全部层微调 10 个 epoch |
|---|---|---|
| 1 千条（1.4 h 音频） | 几秒 | 1–3 分钟 |
| 1 万条（14 h） | < 1 分钟 | 0.2–0.4 h |
| 10 万条（139 h） | 4–7 分钟 | 2–4 h |
| 100 万条（1,389 h），D 只训 3 个 epoch | 0.6–1.2 h | 6–13 h（8 块 GPU 约 1–2 h） |

- 都只需要一块 GPU；H100 约快 2–3 倍。C 约为 D 的 0.35–0.5 倍。
- 真正的成本来自重复训练：例如 45 次训练（5 折 × 3 个学习率 × 3 个种子），10 万条时约 100–200 A100 小时。
- 与已发表的数字对照：SAILER 冻结 Whisper Large-v3，在 8.4 万条上训练 15 个 epoch，用了约 15 GPU 小时、12 GB 显存；AFTER 在约 2 万条上做 TAPT + 主动学习微调，用全部标签时 1,508 分钟，用 20% 标签时 316 分钟。两篇都没写 GPU 型号，但量级与上面的估算一致。

## 4. 微调做法（以全部层微调为例，C 类似）

1. **先做基线**：零样本（把你的标签映射到 9 类，像本项目的评估那样）→ A → B → C → D。每一步都在同一个按说话人划分的验证集上比较，没有收益就停。
2. **数据划分**：训练、验证、测试按说话人分开，每个人只出现在一个集合里；测试集只在最后用一次，最好来自不同的录音场次或语料。指标用 UA（各类召回率的平均）和 macro-F1，附混淆矩阵；用 3 个随机种子，或按说话人做 k 折，报告均值和波动。
3. **预处理**：16 kHz 单声道；每条先做整段 `layer_norm`（与推理一致，在补零之前）；片段裁到 20 s 以内（作者训练时也是这样，注意力显存与长度的平方成正比），去掉短于 1 s 的片段；按长度分桶组 batch，传入 padding mask，平均池化只算有效帧。补零会让结果与单条推理略有不同（卷积位置编码会看到补零的帧），所以补得越少越好。
4. **学习率**：AdamW；编码器 1e-5–3e-5（Wang et al. 用 1e-5；作者最后阶段用大 batch，学习率最高 4.1e-5）；新的分类头 1e-4–1e-3；前 5–10% 的步数 warm-up，之后 cosine 或线性衰减；weight decay 0–0.01；梯度裁剪 1.0（常用设置）；bf16 混合精度。
5. **冻结和逐层学习率**：至少冻结 CNN 特征提取器（Wang et al. 发现冻结 CNN 与全部层微调相当，Wagner et al. 也冻结 CNN；本机实测每步省约 25% 时间）。可以再加逐层学习率衰减（每往下一个 block 乘 0.8–0.9，常用设置），让底层变化小。也可以先只训练分类头 1–2 个 epoch 再解冻（fairseq 的 `freeze_finetune_updates` 就是做这个的）。
6. **类别不平衡**：交叉熵按类别频率的倒数加权，或均衡采样。有多名标注者时，可以用标签分布做软标签（SAILER 用 KL 损失）。
7. **正则化和早停**：保留 config 里的 dropout 0.1；可以加时间 mask 和 mixup（作者最后阶段用了这两种）；epoch 不要多（5–20 个）；按验证集的 UA 或 macro-F1 早停（3–5 个 epoch 不提升就停），保存最好的 checkpoint。
8. **防止遗忘**：数据里没有的类别（如 other、surprised）和语言，微调后可能变差。可以混入一部分公开的有标签数据（例如 EmoBox 里的数据集）；微调前后都在本项目的评估集上跑一遍，特别是不在 EmoBox 里的 MCAE-SPPS 和 MESS，确认没有退步。
9. **硬件**：一块 24 GB 以上的 GPU 足够；小实验可以直接在 M4 Max 上做（MPS 需要第 3.2 节的绕过方法）。

## 5. 官方微调代码

**没有**可以微调 emotion2vec+ 编码器的官方代码（查到 2026-10-28）：
- emotion2vec 的 GitHub 只有特征提取，和一个"冻结特征 + 下游模型"的例子（`iemocap_downstream`）：Linear 768→256 + ReLU → 对有效帧求平均 → Linear 256→4；RMSprop（学习率 5e-4，momentum 0.9）+ CyclicLR（最高 1e-3），batch 128，100 个 epoch，按验证集 WA 选模型，IEMOCAP leave-one-session-out 5 折。这相当于上面的 A、B，不训练编码器。
- FunASR 只有推理示例（`examples/industrial_data_pretraining/emotion2vec/demo.py`）。`Emotion2vec.forward()` 没有损失函数：`features_only=True` 时返回特征，否则什么也不返回，所以 FunASR 的训练器不能直接微调它（FunASR 1.4.16 和 2026-10-28 的 main 分支都是这样）。
- 作者的说法前后不一：2024-01 说不提供训练代码，可以修改推理代码来微调（[#10](https://github.com/ddlBoJack/emotion2vec/issues/10)）；2024-06 说可以用 FunASR 的 AutoModel 流程微调（[#28](https://github.com/ddlBoJack/emotion2vec/issues/28)），与上面的代码不符；2024-08 说计划支持微调，但没有时间表（[#39](https://github.com/ddlBoJack/emotion2vec/issues/39)）；2025-04 有人索要上游训练代码（[#70](https://github.com/ddlBoJack/emotion2vec/issues/70)），没有回复就被关闭。
- 发布的 checkpoint 是用一个没有公开的 fairseq 分支训练的（训练配置里 `user_dir: examples/emo2vec`）。公开代码里最接近的是 fairseq 的 [`examples/data2vec/models/audio_classification.py`](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/audio_classification.py)：平均池化 + 线性层、mixup、`freeze_finetune_updates`。

所以要自己写一个小的 PyTorch 训练循环，直接用 FunASR 构建好的模块（`net = AutoModel(...).model`，再 `net.train()`）。核心几行：

```python
# wav: (B, L)，每条已做过 layer_norm 再补零；pad: (B, L) bool，True 表示补零的位置
out = net.extract_features(wav, padding_mask=pad)       # 不做 mask，已去掉 extra token
x, frame_pad = out["x"], out["padding_mask"]             # (B, T, 1024), (B, T)；不补零时 padding_mask 为 None
keep = (~frame_pad).unsqueeze(-1).float()
emb = (x * keep).sum(1) / keep.sum(1)                     # 只对有效帧求平均
loss = torch.nn.functional.cross_entropy(net.proj(emb), labels)  # 新标签体系就换成新的 Linear(1024, K)
```

本机验证过 `extract_features` 的 padding mask：一条 3 s 片段补零到 4 s 后，句级 embedding 与单独推理的余弦相似度为 0.9996。

## 6. 主要来源

- 模型结构、参数和本地测量：[emotion2vec_plus_large.md](emotion2vec_plus_large.md)；FunASR 代码：https://github.com/modelscope/FunASR/tree/main/funasr/models/emotion2vec
- emotion2vec 论文（冻结编码器 + 线性层的评测、预训练算力）：https://arxiv.org/abs/2312.15185 ；官方 GitHub 和下游示例：https://github.com/ddlBoJack/emotion2vec 、https://github.com/ddlBoJack/emotion2vec/tree/main/iemocap_downstream ；issues #10、#28、#39、#64、#70
- EmoBox（评测协议：冻结最后一层特征 + 线性层、ReLU、池化、分类头；32 个数据集 294.4 h）：https://arxiv.org/abs/2406.07162
- Chen & Rudnicky，Exploring wav2vec 2.0 fine-tuning for improved speech emotion recognition（V-FT、TAPT、P-TAPT、学习曲线）：https://arxiv.org/abs/2110.06309
- Wang et al.，A Fine-tuned Wav2vec 2.0/HuBERT Benchmark（冻结、只微调 Transformer、全部层微调）：https://arxiv.org/abs/2111.02735
- Feng & Narayanan，PEFT-SER（LoRA、adapter）：https://arxiv.org/abs/2306.05350
- Wagner et al.，Dawn of the Transformer Era in Speech Emotion Recognition（IEEE TPAMI 2023；微调与冻结、减少训练数据）：https://arxiv.org/abs/2203.07378
- Li et al.，AFTER: Active Learning Based Fine-Tuning Framework for Speech Emotion Recognition：https://arxiv.org/abs/2310.00283
- Feng et al.，SAILER 系统（Interspeech 2025，MSP-Podcast 1.12）：https://www.isca-archive.org/interspeech_2025/feng25_interspeech.pdf
- data2vec 2.0（Large 模型的预训练算力）：https://arxiv.org/abs/2212.07525
- fairseq 的 data2vec 分类模型：https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/audio_classification.py
- GPU 峰值：A100 https://www.nvidia.com/en-us/data-center/a100/ ；H100 https://www.nvidia.com/en-us/data-center/h100/ （表中 1,979 TFLOPS 含稀疏，稠密为一半）；A10 https://www.nvidia.com/en-us/data-center/products/a10-gpu/

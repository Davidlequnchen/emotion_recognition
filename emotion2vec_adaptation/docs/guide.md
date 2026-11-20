# Emotion2Vec+ large：内部数据适配

这个独立子项目复用仓库已有的 `emotion_recognition_benchmark/models/emotion2vec_plus_large/`，不下载模型、不修改原权重。配置全部放在 YAML，入口脚本没有自定义命令行参数。

提供两条可独立运行的流程：

| 流程 | 输入 | 训练内容 | 输出 |
|---|---|---|---|
| 分类头微调（方案 A/B） | 人工标注的短音频 | 冻结 encoder；A 缓存最后一层的 1024 维 utterance embedding，B 缓存全部 12 个 block 的 embedding；只训练分类头 | `head.pt`、指标、可预测的 `model_bundle.json` |
| 持续自监督预训练（E） | 无标签短音频 | masked student + EMA teacher 特征回归 | 独立的 `encoder.pt`，供后续重训分类头 |

两种分类头与最初方案（`emotion_recognition_benchmark/docs/finetuning_emotion2vec_plus_large.md` 第 1 节）一致，encoder 都冻结：

| 方案 | `head.yaml` | 输入 | 分类头 |
|---|---|---|---|
| A：只训练最后的线性层 | `layer_fusion: last`，`hidden_dim: 0`（默认） | 最后一个 block 输出的有效帧均值，1024 维 | `Linear(1024→K)` |
| B：所有层加权和 + 小分类头 | `layer_fusion: weighted`，`hidden_dim: 256` | 全部 12 个 block（4 个 prenet + 8 个 shared）各自的有效帧均值，12 × 1024 | 每层先做 layer norm，再按 12 个可学习权重（softmax）加权求和，接 `MLP(1024→256→K)` |

B 的 12 个层权重从均等（1/12）开始，使用单独的学习率 `layer_weights_learning_rate`（默认 0.01，不做 weight decay）；用分类头的 0.001 时，权重几乎停在均等。训练结束后写出 `layer_weights.json`，可看出哪些层最有用。FunASR 的 `layer_results` 是 FFN 中间输出（SSL 的目标），不是 block 输出，所以 B 用 forward hook 取每个 block 的输出；最后一行与 A 的 embedding 相同。

## 运行

完成下方环境设置后，从仓库根目录开始：

```bash
conda activate emotion_recognition_llm
cd emotion2vec_adaptation
python train_head.py
```

首次运行前，把数据放到下面约定的位置，或在 `config/common.yaml` 修改路径。模型路径已经指向现有模型，无需再次下载。已有合适的 Python 3.11 Conda 环境可复用；B200 应使用本目录的 CUDA 12.8 组合。Mac/CPU 本地验证仍可用原 `emotion_recognition_llm` 环境，并在 YAML 显式选择对应设备。

入口脚本固定读取同名用途的 YAML：

| 命令 | 配置 |
|---|---|
| `python train_head.py` | `config/common.yaml` + `config/head.yaml` |
| `python pretrain_encoder.py` | `config/common.yaml` + `config/pretrain.yaml` |
| `python predict.py` | `config/common.yaml` + `config/predict.yaml` |
| `python utilities/prepare_labels.py` | `config/common.yaml` + `config/labels.yaml` |
| `python utilities/generate_demo.py` | 生成固定的本地合成示例，无需参数 |
| `python utilities/smoke_test.py` | `config/smoke.yaml`；用真实缓存 large 权重验证两个流程 |
| `python utilities/check_environment.py` | `config/common.yaml` + `config/environment_check.yaml` |
| `python utilities/validate_public_audio.py` | `config/public_validation.yaml`；可在 CPU 或 CUDA 验证真实语音链路 |
| `python utilities/monitor_training.py` | `config/monitor_viewer.yaml`；远程训练的只读曲线入口 |

所有 YAML 路径都相对**仓库根目录**，不受执行位置影响。CSV 内的音频路径相对**该 CSV 所在目录**。绝对路径也支持。输出目录必须尚不存在；每个实验使用新的名字。不会覆盖以前的实验。

## B200 / NVIDIA Conda 环境

训练默认 `runtime.device: cuda`，CUDA 不可用时明确报错，不会静默切到 CPU。Mac/CPU 本地工作可显式改为 `cpu` 或 `mps`；测试脚本有独立设备设置。

本项目与 benchmark 共用 `emotion_recognition_llm` 环境：通用包就是 `emotion_recognition_benchmark/requirements.txt`（FunASR 1.4.16、transformers 5.18.0、numpy 2.4.6 等，与本项目所需版本完全相同），PyTorch 按硬件从 benchmark 的三个文件中选一个，因此本项目不再单独维护 requirements。

| 机器 | 硬件文件 | 安装后 `torch.__version__` |
|---|---|---|
| B200 服务器（Blackwell） | `emotion_recognition_benchmark/requirements-llm-cuda128.txt` | `2.7.1+cu128` |
| 旧 NVIDIA 服务器 | `emotion_recognition_benchmark/requirements-llm-cuda118.txt` | `2.7.1+cu118` |
| Apple Silicon Mac | `emotion_recognition_benchmark/requirements-llm-mac.txt` | `2.7.1` |

B200 是 Blackwell、compute capability 10.0，只有 CUDA 12.8 及以上的构建带有它的 kernel；cu118/cu121 构建在 B200 上无法运行。torch 2.7.1 正好有 cu128 构建，所以三台机器的版本完全相同，只是 CUDA 构建不同。[PyTorch 官方 Blackwell 支持](https://pytorch.org/blog/pytorch-2-7/) · [官方安装组合](https://pytorch.org/get-started/previous-versions/) · [NVIDIA GPU 列表](https://developer.nvidia.com/cuda/gpus)

从仓库根目录在 Linux GPU 服务器上运行：

```bash
nvidia-smi    # 需显示 "CUDA Version: 12.8" 或更高（驱动 >= 570）
uname -m      # 需为 x86_64：torchaudio/torchvision 的 cu128 wheel 只有 x86_64 版本
conda create -n emotion_recognition_llm python=3.11 -y
conda activate emotion_recognition_llm
python -m pip install -r emotion_recognition_benchmark/requirements-llm-cuda128.txt
python -m pip install -r emotion_recognition_benchmark/requirements.txt
python -m pip check
cd emotion2vec_adaptation
python utilities/check_environment.py
```

已有该 Conda 环境时可直接激活，无需重复创建。环境检查记录实际 Python/Conda、PyTorch CUDA、cuDNN、驱动、选中 GPU、架构和显存；执行真实 matmul、Conv1d、FP32 loss 和 backward。它拒绝 B200 上的旧 CUDA 构建。驱动仍须支持 B200/CUDA 12.8；无需为了 wheel 运行另装本地 nvcc。

`precision: auto` 在支持 BF16 的 CUDA GPU 上启用 autocast；CPU/MPS 使用 FP32。Encoder/teacher 参数和 Adam 状态保持 FP32，回归目标归一化与 loss reduction 使用 FP32。没有启用 FP16 或未经验证的 FP8。分类头训练也使用所选设备，特征缓存保存在 CPU 以释放 encoder 显存。

本机没有 B200，因此 CPU 验证不能代替服务器验收。`public_validation.yaml` 的 device/precision 默认 auto，在 B200 服务器上可设为 cuda/auto，再运行 `python utilities/validate_public_audio.py` 完成实际 large encoder 的训练、恢复及推理验证。单设备 CUDA 支持已接入；本项目尚未实现多 GPU DDP。服务器有 4 块 B200 时，每个训练进程只用一块卡：用环境变量把不同实验放到不同的卡上并行，例如 `CUDA_VISIBLE_DEVICES=0 python train_head.py` 与 `CUDA_VISIBLE_DEVICES=1 python pretrain_encoder.py`（各自使用不同的 YAML 输出目录）；进程内 `runtime.device: cuda` 即可见的那一块卡。

`runtime.num_workers` 默认 4：DataLoader worker 在后台进程中整批读取、重采样、裁剪和归一化音频，每个 worker 预取 4 个 batch；`pin_memory` 只在 CUDA 上开启，SSL 训练 loader 的 worker 跨 epoch 保留。4 只是估计的起点，**B200 上的合适值必须在服务器实测**：看每步的 `data_wait_seconds`，若长期超过单步时间的几个百分点就加大；数据在网络存储上，或 batch 变大时，通常需要更多 worker。worker 数不影响 batch 内容和随机数，改它不会改变训练结果。0 表示在主进程读取；Mac 内存紧张时可设为 0–2。

## 数据

默认位置：

```text
curated_dataset/                 # 内容忽略于 Git，仅在本地使用
  labeled/
    manifest.csv
    audio/...
  unlabeled/
    customer_001/clip_001.wav
    customer_002/clip_002.wav
```

标注清单例子：

```csv
path,label,group,split,clip_id
audio/c001_01.wav,neutral,customer_001,train,c001_01
audio/c002_01.wav,happy,customer_002,validation,c002_01
audio/c003_01.wav,sad,customer_003,test,c003_01
```

必填列为 `path,label,group`。`split` 和 `clip_id` 可选。所有行都提供 split，或所有行都不提供；未提供时按 group 固定随机划分。示例只说明格式，真实训练集必须覆盖 `common.yaml` 中所有类别。`max_train_clips: 1000` 或 `5000` 只限制划分后的训练集；不足预算时使用全部可用训练样本。类别名单和顺序由 YAML 显式决定。

`group` 应优先用 customer ID；不可获得时使用 call ID。如果知道同一客户跨多个通话，应使用同一 customer group。**不要直接使用 preparation_tools 的质量分组字段 `group`**，那不是客户／说话人身份。自动划分按 group 数量计算比例，不保证各集合类别均衡；类别缺失会报错，应提供人工确认的分组划分或调整固定 seed，保留独立测试集。最终实际清单保存在每次运行目录。

支持 WAV、FLAC、MP3、OGG、M4A；实际解码能力以本机 libsndfile 为准，建议 curated 数据使用 WAV/FLAC。读取时混合声道、重采样到 16 kHz，并在 padding 前对每段音频做模型所需的 layer normalization。超出时长范围、空文件或非有限值都会报错，不会静默裁剪；SSL 按时长分桶时把 batch 裁到最短片段，比例有日志记录（见[百万级数据](#百万级数据一次读取无-padding可恢复)）。默认接受 2–15 秒，与 preparation_tools 的切分窗口一致；现有内部批次均在 2–10 秒之间，也在范围内。训练和预测开始前会先检查每段时长（训练读取音频审计缓存，预测读取文件头），有任何超范围或无法解码的片段就立刻报错并列出，不会等训练进行到一半才失败。

无标签输入可以是递归音频目录，也可以是含 `path,group` 的 CSV。目录模式将相对父目录作为 group，根目录文件则用文件名；需要精确关联客户时请使用 CSV。若 CSV 有显式 split，SSL 只读 train，仍检查原清单的跨集合泄漏。

若标注清单存在，SSL 会按同一设置计算标注数据的 validation/test，并排除相同 group、路径和字节相同的音频副本（SHA256，来自音频审计缓存，见[百万级数据](#百万级数据一次读取无-padding可恢复)）。若尚无标注清单，SSL 可以单独运行；需要自行预留客户／通话，未来新增标签时继续沿用留出安排。字节比较不能识别重新编码或被裁剪的副本，正确的 group 信息仍是必要条件。应在两个实验开始前固定留出集合。

## 接入 Label Studio

现有 `label_studio/normalize_export.py` 输出的是**每位标注员的一票**，不是每个 clip 的最终标签。将其 JSON 路径和包含 `clip_id,path,group` 的音频清单填入 `config/labels.yaml`，运行：

```bash
python utilities/prepare_labels.py
```

导入器按 clip_id 合并：已有票必须一致；冲突会报错并要求先裁决；没有标签、`bad_clip` 和 `unknown` 排除。类别必须在 YAML 中显式配置。单票也可导入，一致票不代表已满足你们的标注人数要求；需要多人复核的批次请先完成复核。输出清单不会覆盖已有文件。

### `other` / `others` 与 `unknown`

正式定义的“其他情绪”类别可以训练，不应仅因名称叫 other 就丢弃。将同一个准确的字符串加入 `labels` 和人工裁决后的 manifest，分类头输出维度会自动增加。`other` 和 `others` 都可以作为名字，但 CSV、YAML 和裁决结果必须一致；代码不会自动把二者合并。原发布模型的 `other` 与 `<unk>` 也是两个不同 token，不能据此认定内部标签具有相同语义。

按本次确认，默认标签已采用原模型的正式拼写 `other`，共 8 个具名情绪类别：angry、disgusted、fearful、happy、neutral、other、sad、surprised。分类头会输出这 8 类；`unknown`/`<unk>` 不并入 other。

历史 Label Studio 模板有一个特殊情况：界面为 **Unknown / Other**，实际 `decision` 值为 **unknown**，说明同时涵盖“情绪不确定”和“超出已有类别”。因此历史 unknown 不能整批重命名为 other，应先区分或复核这些样本，保留原始投票记录，再输出明确的人工标签。新的模板将 other 和 unknown 分开；已有正式项目及历史导出不会自动迁移。本项目默认导入器仍将 unknown 排除于监督清单，正式 other 会保留；未配置的拼写（例如默认未配置的 others）会报错，不会静默丢弃。

情绪无法判定但音频有效的片段，可进入合格的无标签 SSL 输入清单，继续遵守客户/通话留出规则。bad_clip 应从监督与 SSL 输入中都移除；SSL 的目录模式只负责读取音频，不会自行读取 Label Studio 的坏片段决定，所以不要将坏片段混入其 curated 输入目录。已剔出监督训练不等于应删除原音频。

## 只做分类头微调

1. 在 `common.yaml` 设置 labels、labeled_manifest、分组划分和 `max_train_clips`。保持 `model.encoder_checkpoint: null` 即使用原 encoder。
2. 在 `head.yaml` 选择方案 A 或 B（见上表），调整学习率、epoch 和类别权重。建议先跑 A 作为基线，再用同一划分跑 B 比较 validation UAR/macro-F1。
3. 运行 `python train_head.py`。

encoder 全程 eval 且无梯度；每段音频仅提取一次特征，再释放 large 模型，在所选设备训练小分类头。`features.pt` 是本次实验的内存缓存落盘，包含 encoder 指纹及音频内容摘要；不会从其他实验静默重用缓存。

**encoder 每次只处理一段、不补零。**FunASR 的卷积位置编码会读到 padding 区域，补零后同一段音频的 embedding 会随同 batch 的其他片段而变：mock 测试中 cosine 最低 0.985，原模型 9 类 head 的概率最多偏移 0.29。因此提特征、`predict.py` 和 `predict_batch` 都逐段运行 encoder（`EmotionBackbone.embed_each`），批量与单条预测结果完全一致（实测概率差 0.000），也与 benchmark 的单条推理一致。分类头训练仍用 `batch_size` 的大 batch，因为那时输入已是缓存的特征。

使用 training 子集计算类别权重；按 `selection_metric`（默认 `uar`，可选 `macro_f1`、`loss`）选最优 epoch 并决定提前停止，同分时取 validation loss 更低者；训练结束后仅评估一次 test。训练用类别加权 CE，而我们关心各类情绪的召回，所以默认按 validation UAR 选模型，而不是按未加权的 validation CE（类别不平衡时两者可能选出不同的 epoch）。每次运行写出 `split_counts.json`（各集合、各类别片段数）；validation 中缺少某个类别时按 UAR/macro-F1 选模型会直接报错，少于 10 段的类别会给出警告，因为一段就会让该类召回变化 10 个点以上。输出 `history.jsonl`、`test_metrics.json`（accuracy、UAR、macro-F1、支持数、混淆矩阵）及实际 train/validation/test CSV。UAR 和 macro-F1 对全部配置类别平均；缺席类别记零，同时提供各类支持数。

`model_bundle.json` 引用原模型目录、可选的替换 encoder 和本次训练 head。它是本地组合清单，没有复制原 1.8 GB 权重；移机时需同步原模型、head、替换 encoder 并更新路径。

## 持续预训练，再训练分类头

```bash
python pretrain_encoder.py
```

SSL 不使用情感标签构造损失。teacher 从当前 encoder 复制并冻结，每步 optimizer 更新后才进行 EMA。student 在 CNN 输出后的 latent 时间轴使用原生 span mask；只对 masked 且非 padding 的帧计算预测误差。`freeze_frontend: true` 冻结 CNN，其余 encoder 参数仍可更新。

**这是一套新建的 domain-adaptation 目标，不是恢复原作者训练。**公开 large 发布权重没有原 SSL teacher 和 decoder；FunASR 也没有完整的对应预训练 loss。这里新建 MLP predictor 和 EMA teacher。下降的 SSL loss 不证明情感识别改善；最终仍须与原 encoder 的 frozen-head baseline 比较相同留出集结果。

### SSL 目标的选择（`target`）

| `target` | masked 帧回归什么 | 现状 |
|---|---|---|
| `final`（默认） | teacher 最后一个 block 的输出，逐帧 layer norm（跨 channel 归一化，不减时间均值） | 能学，mock 中没有损害下游 |
| `ffn_mean` | 最后 K 个 shared block 的 FFN 输出，各自按片段时间归一化后求平均（data2vec 2.0 / emotion2vec 的做法） | 从这个模型学不到，保留作对照 |

原默认 `ffn_mean` 用在已微调的 emotion2vec+ large 上不成立，mock 测试的证据：

- 最后一层几乎不随时间变化：段内方差只有段间方差的 0.2%，每帧与本段平均 embedding 的 cosine 约 0.99998。这个模型在 utterance 级标签上微调过，最后一层已基本是 utterance 表示。
- 8 层归一化 FFN 输出彼此几乎不相关（平均 r = 0.04）；连 encoder 冻结、不加 mask 时，predictor 也学不到它们，loss 始终停在“预测为 0”的平凡水平（约 0.15，即 target 方差）。
- 结果是梯度没有教会 student 任何东西，只把 encoder 推离原表示：约 200 步后下游 test 准确率从 0.75 降到 0.54–0.58（同域无标签数据也一样）。

`final` 目标让 student 从上下文恢复被遮挡位置的表示；由于最后一层近似常量，这在很大程度上等价于“从部分音频恢复整段的表示”，与模型的 utterance 性质一致。`utterance_weight > 0` 另加 emotion2vec 的 utterance 损失：student 的 10 个 extra token **先平均池化**，经同一 MLP 后匹配 teacher 最后一层有效帧均值的 layer norm。mock 中它没有带来额外收益，默认关闭。

### 两道门槛：宁可停下，也不损坏 encoder

| 门槛 | 做法 | 不通过时 |
|---|---|---|
| 可学性检查（`probe_steps`、`probe_min_gain`） | 先冻结 encoder、只训 predictor `probe_steps` 步，然后在固定 mask 的 SSL 验证集上比较 loss 与两个基线：预测为 0（`zero_loss`）、预测该 batch 各 channel 的均值（`constant_loss`，二者中更强的基线） | loss 比常数预测基线好不到 `probe_min_gain`（默认 10%）就报错停止，encoder 一个权重都没改；结果写在 `probe.json` |
| 下游检查（`downstream_every_steps`、`downstream_max_drop`） | 训练前先用原 encoder 建立基线；之后每 N 步用当前 student 逐段提取固定标注 train/validation 片段的特征，重新拟合一个类别加权的线性头，记录 validation UAR，同时记录相对原 encoder 的 cosine、跨片段分散度和有效秩（`downstream_history.jsonl`）。不读 test | UAR 比原 encoder 低超过 `downstream_max_drop`（默认 0.02）就停止训练；下游 UAR 最高的 encoder 另存为 `encoder_best.pt` |

下游检查需要 `data.labeled_manifest`；设 `downstream_every_steps: 0` 可在没有标注时运行，但那时没有任何东西保护情感信息。可学性检查需要 SSL 验证集（`validation_fraction > 0` 或 `validation_manifest`）。

另外三个稳定训练的设置：predictor 是随机初始化的新模块，用单独的 `predictor_learning_rate`（默认 1e-3）；encoder 学习率在可学性检查之后从 0 线性升到 `learning_rate`，用时 `warmup_steps`；`anchor_weight > 0` 启用 L2-SP，把 encoder 拉向原权重（默认关闭，需要时再按下游检查调）。这些能稳定训练，但不能单独解决目标失配，所以门槛仍然必要。

**mock 对照实验**（CREMA-D 8 kHz 6 类；标注 504/144/72 段，按说话人划分；无标签为另外 30 位说话人的 CREMA-D 384 段；batch 4，可学性检查 100 步，之后 encoder 训 300 步，每 50 步下游检查）：

| 配置 | 可学性检查 | 用完整 head 在 test（72 段）上 |
|---|---|---|
| 原 encoder | — | 0.708（validation UAR 0.750） |
| `ffn_mean` | 增益 −1.7%，被拦下 | encoder 未改动 |
| `final`，第 1 次（未分桶） | 增益 24.7%，通过 | `encoder_best.pt`（第 300 步）0.764；最终 0.736 |
| `final`，第 2 次（按时长分桶裁剪；第 150 步中断后续训到 400 步） | 增益 10.1%，刚好通过 | `encoder_best.pt`（第 400 步）0.694 |
| `final` + utterance 损失 | 增益 23.3%，通过 | 下游 UAR 与 `final` 相近 |
| `final`，无标签换成 DailyTalk | 增益 6%，被拦下（只有 2 位配音演员，常数预测已很准） | encoder 未改动 |

test 只有 72 段，误差约 ±5 个点；mock 是同句子的表演语料。第 1 次的 +5.6 个点在第 2 次没有复现（−1.4 个点，即 1 段），所以结论是：门槛能拦住有害的配置，`final` 目标在 mock 上**不损害下游，但也看不出提升**。下游检查的快速线性头 UAR 在第 2 次从 0.653 升到 0.750，而完整 head 的 test 没有跟着提高：这道门槛适合挡住伤害，不能当作提升的证据。可学性检查的增益随 batch 组成变化（24.7% 与 10.1%），默认 `probe_steps: 500` 比实验里的 100 步给 predictor 更多时间；真实数据上若增益在阈值附近，先看 `probe.json` 再决定。真实通话上是否有收益，仍以两道门槛和最终的 head 对比为准。

### 当前目标的精确定义

一条完整音频为 `x`，有效帧集合为 `V`，其中随机选出的遮挡帧为 `M`。teacher 不遮挡音频。默认 `target: final` 取 teacher 最后一个 block 的输出 `z_teacher[t]`，逐帧做不带参数的 layer norm；`target: ffn_mean` 则从最后 K 个 shared Transformer 的 FFN 输出 `h_teacher[k,t]` 出发，在**每个 clip、每个 channel 的有效时间轴**上减均值、除标准差，再对层平均：

```text
final:    target[t] = layer_norm(z_teacher[t])
ffn_mean: target[t] = mean_k(temporal_instance_normalize(h_teacher[k,t], valid_frames))
prediction[t] = MLP(student(mask(latent_features(x)))[t])
L_frame = sum_(t in M, d)(prediction[t,d] - stop_gradient(target[t,d]))^2 / (|M| * D)
L = frame_weight * L_frame + utterance_weight * L_utterance + anchor_weight * sum ||theta_S - theta_original||^2
```

默认 `D=1024`、`K=8`（仅 `ffn_mean`）。MSE 只在 `M` 上计算，未遮挡帧和 padding 对这个输出端损失的直接梯度为零；未遮挡帧仍通过 attention 为遮挡位置提供上下文，也会因此获得间接梯度。这里没有波形重建、情感类别或文本预测。

student 与 MLP 通过 AdamW 反向传播更新；随后每次 optimizer step 执行一次：

```text
teacher_next = 0.999 * teacher_previous + 0.001 * student_next
```

teacher 无梯度且始终 eval。默认 CNN 冻结，projection、位置编码和 Transformer 等其余 encoder 参数可训练。teacher 也保留发布架构中的 extra tokens；frame loss 不直接监督这些 token。

utterance 分支的具体含义：student 的 10 个 extra token 先求平均，经同一个 MLP 后匹配 `LayerNorm(mean_time(final_teacher_frames))`，得到 `L_utterance`。这与原论文“先池化 token 再计算误差”的做法一致（此前版本是逐 token 匹配，已改）。默认关闭。不能用已按时间减均值的 `ffn_mean` target 再做时间平均作为全局目标，因为该平均接近零。

### 与原论文的边界及选择理由

对照[ACL 最终论文](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.931.pdf) §3、§4.2 和 Appendix A/C：

| 项目 | 原 emotion2vec 方法 | 本项目当前适配基线 |
|---|---|---|
| 核心思想 | 完整 teacher、masked student、表示匹配、stop-gradient + EMA | 保留 |
| 目标组成 | frame loss（top-K FFN 输出）+ utterance loss | 默认 frame loss，target 为最后一层（`final`）；原论文式的 `ffn_mean` 和池化 utterance loss 可选 |
| student 的预测部分 | 依初始模型，linear 或 data2vec2 的可见 token 编码 + CNN decoder | 保留全部位置，latent 零遮挡 + 两层 pointwise MLP |
| teacher 更新 | CNN 直接复制；backbone EMA，decay 递增 | 固定 decay，对全部 encoder 参数 EMA；默认冻结 CNN 时 CNN 数值保持一致 |
| mask 参数 | 论文以 span 起点概率描述 | 约定为有效帧目标覆盖率，不能直接当作同一定义 |
| 起点 | 原论文的预训练 base 模型 | 仓库现有 emotion2vec+ large 发布 encoder |

选择 MLP、固定 EMA 是为了先得到结构清晰、容易审计的小规模适配基线；target 改为最后一层，是因为原论文的 FFN target 从已微调的 large 模型学不到（见上文“SSL 目标的选择”），不是因为内部短音频。2–15 秒的完整片段可以直接构造 teacher/student 同时间轴目标，无需另行切长通话；保留全部 latent 位置也使 padding 和对齐更直接，但不具有 data2vec2 删除 masked token 的编码效率优势。

已有 large encoder 提供初始化；默认冻结 CNN、小学习率，目的是先限制低层声学变化。它仍可能遗忘情感特征或学习到通道特性；EMA/归一化也不能保证不塌缩。原论文附录比较过 frame-only 与双目标，但这些结果不能直接转移到内部域。是否有效，需要在固定的客户独立留出集上比较原 encoder 与适配 encoder 重训 head 后的 UAR/macro-F1；再决定是否增加更接近原论文的 pooled utterance objective。**不应因为片段较短就推断全局情感目标不必要。**

代码来源分工：FunASR 提供已发布的 encoder、卷积、Transformer、原生 latent mask 应用及权重；本项目新增 mask 采样、padding-aware target normalization、MLP、损失、EMA、训练、恢复和导出。不是从零重写 Emotion2Vec，也不是直接调用官方完整 SSL trainer。当前检查的[官方版本](https://github.com/ddlBoJack/emotion2vec/blob/a0e606c668227ab0c01527e196a4acc42b1efad9/upstream/models/emotion2vec.py)提供特征提取／下游代码，`features_only=False` 没有完整 SSL loss 返回；[fairseq data2vec2](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py)公开了相关通用训练实现，可供参考。这个结论限于本次核查的官方版本，不能泛化为网上不存在其他实现。

`encoder.pt` 是最近一次保存的 student encoder，**不会按训练 loss 宣称最佳情感模型**；`encoder_best.pt` 是下游检查 UAR 最高的 encoder（只有超过原 encoder 时才会生成），`checkpoints/` 保留最近几次保存的 encoder 供比较。`target_std` 不能用来判断塌缩：归一化会把它固定在接近常数的水平，即使所有片段的表示变得一样也照样正常；应看下游检查记录的跨片段分散度、有效秩、相对原 encoder 的 cosine，以及最终的下游 UAR。当前采用单设备训练，CUDA 支持 BF16 autocast，CPU/MPS 使用 FP32；large teacher、student、梯度和 Adam 状态需要数 GB 内存，先以小 batch 开始。本次实际 large 模型验证环境为 CPU，CUDA 专用测试在没有 GPU 时明确 skip。

然后在 `common.yaml` 设置：

```yaml
model:
  directory: emotion_recognition_benchmark/models/emotion2vec_plus_large
  encoder_checkpoint: emotion2vec_adaptation/outputs/continued_encoder/encoder.pt
```

给 `head.yaml` 一个新的 output_dir，再运行 `python train_head.py`，即可得到替换 encoder 后的完整分类模型。可以先独立进行 SSL，同时使用原 encoder 做第一轮 head baseline；待 SSL 完成后再用新 encoder 重训 head。两条流程使用不同输出目录。

直接把旧 head 套到新 encoder 通常没有可靠的分类含义。预测入口会校验 encoder 权重指纹、架构配置、labels 和音频设置，拒绝不匹配的组合。原模型文件始终保留。

**保存与恢复。**每次保存都重写 `last_training.pt`（student、teacher、predictor、optimizer、随机数状态，以及当前 epoch 已消耗的 batch 数）和 `encoder.pt`。百万级数据的一个 epoch 有上百万步，所以 `save_every_steps`（默认 5000）在 epoch 中途也定期保存；0 表示只在 epoch 结束和运行停止时保存。每次保存另写一份仅含 encoder 的 `checkpoints/encoder_step_XXXXXXXX.pt`，`keep_checkpoints`（默认 5）只保留最新几份，旧的自动删除，用于之后分别重训 head 比较；`save_every_steps` 取 `downstream_every_steps` 的整数倍时，每份保留的 encoder 都能在 `downstream_history.jsonl` 查到 UAR。

需要续训时，在 `pretrain.yaml` 把 resume_checkpoint 指向 `last_training.pt`，设置新的 output_dir，必要时增加 epochs 或去掉 max_steps。恢复是精确的，epoch 中途中断（max_steps、断电或被终止）也一样：每个 epoch 的 batch 顺序只由 (seed, epoch) 决定，恢复时重建同一顺序、跳过已消耗的 batch（不解码音频），再还原随机数状态。单元测试确认，中断后恢复与不中断训练得到逐位相同的 encoder 权重（CPU；GPU 上部分 kernel 本身不保证逐位可复现）。数据内容、目标配置、batch 设置和初始 encoder 必须一致，否则拒绝恢复。被下游检查叫停的运行不能续训，应从其 `encoder_best.pt` 开始新实验；本次修改之前的旧格式 `last_training.pt` 也不能恢复，可把其 `encoder.pt` 作为新的初始化。训练状态文件比仅 encoder 的文件大，可通过 save_training_state 关闭。

## 训练监测与自动更新曲线

每次运行都有独立目录：

```text
outputs/<experiment>/
  logs/train.log                 # 生命周期、指标摘要及异常 traceback
  logs/metrics.jsonl             # 每个 batch/epoch 的完整 scalar 事件
  monitoring/dashboard.html      # 每 5 秒自动刷新
  monitoring/curves.png           # 训练中原子更新的曲线
  history.jsonl                  # 兼容的训练历史
  epoch_history.jsonl            # SSL 每 epoch 的 train/validation loss
  data_audit.json                # SSL 片段数、小时数、留出排除、音频审计的读取/复用数
  checkpoints/                   # SSL 最近 keep_checkpoints 份 encoder
```

终端有 epoch 和 batch 进度条、吞吐/ETA，并按 `log_every_steps` 输出详细指标。Head 每 epoch 更新训练/验证 loss、accuracy 和 macro-F1。分类头图里的两条 loss 都是未加权 CE，便于直接比较；实际使用的类别加权优化目标另存 `train_objective_loss`。

SSL 图包含原始 loss、显示用 EMA loss、验证 loss 和常数预测基线（loss 贴着这条线说明 target 没有被学到）、下游检查的 validation UAR、gradient norm 和 encoder learning rate（可学性检查期间为 0，之后 warm-up）。显示用的 loss smoothing 不改变优化目标。默认每 20 step 重绘，设 `monitoring.plot_every_steps: 1` 即每步重绘；每个 epoch 和结束时都会强制更新。绘图有界采样最多 `max_plot_points`，避免大任务持有全部历史或反复绘制百万点；JSONL 仍保存全部 batch。

SSL 每步还记录（`history.jsonl` 和 `logs/metrics.jsonl`）：

| 指标 | 含义 |
|---|---|
| `data_wait_seconds` | 等待下一个 batch 的时间；占单步时间比例高说明读数据是瓶颈 |
| `clips_per_second` | 两步之间墙钟时间内的片段数，包含等数据、日志以及中间的检查和保存 |
| `compute_clips_per_second` | 只算前向、反向和参数更新 |
| `cropped_fraction` | 分桶裁剪丢弃的音频比例 |
| `cuda_allocated_gib` / `cuda_reserved_gib` / `cuda_peak_allocated_gib` | CUDA：张量占用、缓存分配器保留、张量峰值 |
| `mps_allocated_gib` / `mps_driver_gib` | MPS：张量占用、Metal 驱动分配总量 |
| `process_peak_rss_gib` | 所有设备：本进程峰值常驻内存（不含 DataLoader worker；Apple 统一内存下包含部分 GPU 占用） |

reserved 或 driver 远大于张量占用时，差额来自分配器缓存或驱动，不是模型张量本身。

所有开关在 `common.yaml` 的 `monitoring`：enabled、progress_bar、console_log、log_every_steps、plot_every_steps、max_plot_points、refresh_seconds、loss_smoothing。Head 的 epoch 图不会随每个 batch 重复重绘。异常会写入专用日志，dashboard 标记 failed，并继续向调用者抛出异常。

出版级绘图的全部样式在 `config/plots.yaml`：沿用 benchmark 的 Okabe–Ito 配色和 Arial/DejaVu Sans 字体；默认刻度 16 pt、坐标标签 19 pt、标题 21 pt。每个指标分别导出 300 dpi PNG 和矢量 SVG，PDF 可按配置启用。独立图与实时总览共用样式，accuracy/F1 的显示单位一致，epoch/step 使用整数刻度。`figures/plot_manifest.json` 记录实际字体、尺寸与配置来源；这些只是显示设置，不会更改日志数值或训练目标。

直接打开运行目录中的 `monitoring/dashboard.html` 即可实时查看同一文件系统上的结果。在远程 B200 上运行时，可以另开一个终端执行 `python utilities/monitor_training.py`；它只读服务配置的 outputs 根目录，默认仅绑定 `127.0.0.1:8766`。通过 IDE 的端口转发，或本地执行 `ssh -L 8766:127.0.0.1:8766 <GPU-server>` 后，在浏览器打开 `http://127.0.0.1:8766` 选择实验。无需启动图形桌面，训练本身不依赖 viewer 是否运行。

## 数据规模与原模型对比

最新输入说明：30 多个 batch，每批约 100,000–200,000 个已预处理音频片段，每段严格大于 2 秒、小于 10 秒。此前 13K × 30 = 390K 的估计已被这次更正取代；13K 仅是现有小规模技术试跑预算。

必须区分公开报告中的训练阶段：

| 对象 | 报告的数据规模 | 含义 |
|---|---|---|
| 原始 emotion2vec（ACL 2024） | 262 小时，约 16.9 万段 | 无标签情绪语音自监督预训练 |
| emotion2vec+ large | 42,526 小时 | 筛选后的大规模伪标签情绪识别微调数据 |
| 本项目 | 目标域内部音频 | 在已有 large encoder 上进行新的无标签持续适配 |

原论文 §4.2/Table 1 报告 262 小时及总计 169,053 utterances，来自 IEMOCAP、MELD、CMU-MOSEI、MEAD、MSP-Podcast；这里只按论文报告概述为约 16.9 万段。[论文](https://aclanthology.org/anthology-files/pdf/findings/2024.findings-acl.931.pdf)

官方 large 模型卡将 42,526 小时列为 Fine-tuning Data，并明确描述为 filtered pseudo-labeled data。其约 160K 小时指筛选候选池，不能全部算成使用过的训练数据。不能将 42,526 小时称为 large 已公开证明的原始 SSL 总时长，也不应将各阶段、epoch 或候选池简单相加；large 的完整初始化与 SSL 训练历史并未由这些资料完整列出。[官方模型卡](https://huggingface.co/emotion2vec/emotion2vec_plus_large) · [官方仓库](https://github.com/ddlBoJack/emotion2vec#model-card)

实际 batch 数 B 尚未给出精确值，因此以下以 **30 批作为参考**，而非当前数据总量的精确上下界：

| 30 批参考场景 | 每批 100K 段 | 每批 200K 段 |
|---|---:|---:|
| 片段总数 | 3,000,000 | 6,000,000 |
| 每段 2–10 秒，仅给出的时长范围 | 约 1,667–8,333 小时 | 约 3,333–16,667 小时 |
| **假设**实际平均 5 秒的示例 | 约 4,167 小时 | 约 8,333 小时 |
| 5 秒示例相对原始 262 小时 SSL | 约 15.9 倍 | 约 31.8 倍 |
| 5 秒示例相对 large 的 42,526 小时微调 | 约 9.8% | 约 19.6% |

只根据 2–10 秒范围，30 批参考的合并宽范围约为 1,667–16,667 小时。**5 秒是示例假设，不是已测平均值。**实际 B 超过 30 时，上述参考数量和示例时长按 B/30 比例上调，因此 6M 不能当成真实数据量的上限。准确总时长为逐片段实际时长之和除以 3,600。片段数也受分段策略影响，跨数据集更应以音频小时数为主比较。

当前数据规模已进入数百万片段级别；这不自动证明更好效果，也不要求从第一轮就使用全量。可以先保留 13K 的技术检查，再跨所有批次准备有代表性的 100K–200K 阶段实验，之后依据同一客户独立验证集的 UAR/macro-F1 决定扩大到 1M 或全量。这些是工程阶段建议，不是已证明的最低数据量或最优学习曲线。

`pretrain.max_unlabeled_clips` 仍默认 13000，作为显式的小规模试跑预算，不会因这次数据规模更正就自动启动百万级训练。可按计划改为 100000、200000、1000000 或 null；预算限制发生在剔除 labeled holdouts 和划分 SSL validation 之后，实际清单会落盘。自动预算是固定 seed 的伪随机 clip 抽样（同一 seed 下小预算是大预算的子集）；需要按渠道、语言、客户或时间段分层时，应先准备代表性 CSV。`data.max_train_clips` 独立控制有标签 head 的 1K/5K 预算。

SSL 默认按 group 留出 5% 无标签数据，完全不参与 optimizer 或 EMA 更新，并使用固定遮挡随机种子计算每 epoch validation loss；验证后恢复 RNG，避免改变训练随机序列。推荐为不同规模实验固定 `pretrain.validation_manifest`；该清单中的客户/通话会从训练候选中预留。没有该清单时，优先使用无标签 CSV 的 validation 行，再按 `validation_fraction` 自动划分。设 fraction 为 0 可关闭自动划分；显式 validation 行仍保留为验证用途。标注集的 validation/test 继续排除于全部 SSL 数据之外。

自动划分针对预算抽样前的候选数据；若源目录含数百万片段，5% 留出也会很大。小规模 pilot 可提供约 1K–2K 条、有代表性且客户独立的固定 validation manifest，以限制每次验证开销。不同阶段使用同一份 manifest，避免把验证集变化误当成数据规模带来的改善。

先用 1K 左右标注训练 head 基线，另留客户独立 validation/test；若 1K/5K 指全部标注总量，实际训练数还需扣除留出集。SSL pilot 可先设 `max_steps: 2000` 检查稳定性和吞吐，随后在 `checkpoints/` 保留的若干 encoder 上重训同样的 head、比较验证 UAR/macro-F1。训练 loss 下降而验证 loss 上升、表示方差接近零，或下游验证指标变差，都值得停下来检查。固定 EMA teacher 的目标本身仍在变化，SSL validation loss 也不是最终情感质量指标；最终 test 仅在方案确定后评估。

### 百万级数据：一次读取、无 padding、可恢复

**音频审计缓存**（`data.audit_cache`，默认 `emotion2vec_adaptation/outputs/audio_audit.jsonl`，位于 Git 忽略的 outputs 下）：每个文件只完整读取一次，按路径记录 size、mtime_ns、SHA256、帧数、采样率和时长；之后 size 与 mtime_ns 不变就直接复用，变化则重读。标注留出副本的排除、SSL train/validation/留出的不相交检查、恢复签名里的数据摘要和时长检查都读这份缓存，泄漏检查仍按字节内容。首次读取用 16 个线程，每 1 万条追加一次，中断后已读的部分不丢；日志和 `data_audit.json` 记录本次实际读取与复用的文件数。按 size 与 mtime_ns 判断意味着：大小相同且恢复了旧 mtime 的改写不会被发现，这种情况下删除缓存即可重建。设为 null 则只在本次运行的内存中保存。

**先抽样，再读内容**：group 和路径的排除对全部候选进行，只需目录或 CSV；字节内容只对实际使用的训练、验证和留出片段计算。13K 试跑因此只读约 13K 个文件，而不是整个数据源。抽到的留出副本会被剔除，数量写入 `holdout_copies_dropped`，所以训练片段可能略少于预算。预算按带 seed 的路径哈希排序取最小的 N 个：同一 seed 下 13K 样本是 65K 样本的子集，后一阶段直接复用前一阶段的审计；新增批次后原样本大部分保留。

**按时长分桶、无 padding**（`pretrain.bucket_by_duration: true`，默认）：FunASR 的卷积位置编码会读到 padding 帧，padding 后片段的表示会随同一 batch 的其他片段变化。每个 epoch 先打乱，把每 100 个 batch 的片段池按时长排序、切成 batch，再打乱 batch 顺序；每个 batch 的片段裁到其中最短片段的长度（偏移由 seed、epoch 和 batch 序号决定），裁剪后重新做逐段 layer normalization。SSL 验证 batch 按时长排序、裁剪固定。每个 epoch 开始时日志给出裁剪丢弃的音频比例，每步记录 `cropped_fraction`。在 2–15 秒均匀分布的模拟时长上，batch size 2–32 丢弃约 0.4–0.7%（不分桶时为 25–72%）；实际比例以日志为准。设为 false 则使用随机 batch 加右侧 padding。

读取并行度见 [B200 环境](#b200--nvidia-conda-环境)中的 `runtime.num_workers`，保存与恢复见上文“持续预训练”。

## 方法演示稿

[Emotion2Vec+ large 适配框架 PPTX](../presentation/emotion2vec_adaptation.pptx) 共 28 页，另有 9 页隐藏附录，沿用原 SER 白底蓝色风格，包含四步流程、逐符号公式解释、原方法对照及 rationale、encoder/head 组合、数据规模建议和验证现状。参见 [重建说明](../presentation/README.md)。

## 预测与验证

在 `predict.yaml` 指向本次 `model_bundle.json` 和待预测目录／CSV，再运行 `python predict.py`。输出每段音频的标签及每类概率；已有输出文件会报错。

程序接口也可直接使用：

```python
from pathlib import Path
import torch
from adaptation.inference import AdaptedEmotionModel

model = AdaptedEmotionModel(Path("outputs/head_baseline/model_bundle.json"), torch.device("cpu"))
result = model.predict_file(Path("/absolute/path/to/clip.wav"))
```

快速验证无需真实内部数据：

```bash
python utilities/smoke_test.py
python utilities/validate_public_audio.py
python -m unittest discover -s tests -v
```

smoke 使用现有 large 权重，以及本地生成的约 3 秒合成音频，验证 head baseline、SSL 更新、encoder 导出重载、适配后重训 head、预测和错配拒绝。合成标签仅用于接口测试，**没有情感语义，不能作为准确率或训练效果证据**。smoke 默认生成独立运行目录，不改生产 YAML。重复运行可保留新的报告，示例数据位于 `curated_dataset/demo/`。

`utilities/validate_public_audio.py` 读取 `config/public_validation.yaml`，用已有 RAVDESS **真实语音**模拟内部数据；保留原始标签、speaker、完整时长与源文件 SHA256，原文件只读，不下载数据。默认 24 条标注样本按 speaker 划分，另有独立的无标签样本、8 kHz stereo 处理探针和故意放入的 heldout 音频副本。它检查完整训练／恢复／导出／重训 head／预测链路，报告与模型保存在独立的 `outputs/public_validation/`。该小集合不是新的性能 benchmark，不能用来证明内部数据提升或原模型从未见过这些公开语音。

目录职责参见 [项目入口](../README.md) 的目录树。模型代码在 `adaptation/`，配置、运行、监控与验证辅助代码在 `utilities/`，全部 YAML 与环境依赖定义在 `config/`。

参考：[原 emotion2vec 项目](https://github.com/ddlBoJack/emotion2vec)、[论文](https://aclanthology.org/2024.findings-acl.931/)、[FunASR 实现](https://github.com/modelscope/FunASR/tree/main/funasr/models/emotion2vec)、[fairseq data2vec2](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py)。原权重及派生模型遵循本地模型目录中的 `LICENSE`，保留 emotion2vec+ large 名称和上游来源。

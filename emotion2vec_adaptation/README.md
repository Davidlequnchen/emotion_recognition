# Emotion2Vec+ Large Adaptation

两个独立流程：**冻结 encoder、训练分类头**；或**无标签 SSL 适配 encoder，再重训分类头**。复用仓库已有 large 权重，不下载模型，不修改原权重。全部参数由 `config/` 控制，入口没有自定义命令行参数。

## 先看这三个配置

| 文件 | 用途 |
|---|---|
| `config/common.yaml` | 数据路径、标签、head 的 1K/5K 预算、音频审计缓存、CUDA/precision、DataLoader worker、进度条和日志 |
| `config/head.yaml` | 方案 A/B（`layer_fusion`）、分类头的学习率、epochs、结构与输出目录 |
| `config/pretrain.yaml` | SSL、13K/65K/全量预算、验证集、时长分桶、checkpoint 与精确恢复 |

绘图样式独立放在 `config/plots.yaml`。预测、示例及辅助工具的配置也都在 `config/`，需要时再看。

YAML 中的路径相对**仓库根目录**；CSV 内的音频路径相对**该 CSV 所在目录**，不受运行位置影响。

## 在 B200 上开始

本项目与 benchmark 共用同一个 `emotion_recognition_llm` 环境，依赖版本完全相同，不另设 requirements。在 B200 服务器上从仓库根目录运行：

```bash
conda create -n emotion_recognition_llm python=3.11 -y
conda activate emotion_recognition_llm
python -m pip install -r emotion_recognition_benchmark/requirements-llm-cuda128.txt   # torch 2.7.1+cu128（Mac 用 requirements-llm-mac.txt）
python -m pip install -r emotion_recognition_benchmark/requirements.txt              # FunASR、transformers 等通用包
cd emotion2vec_adaptation
python utilities/check_environment.py
```

B200 必须用 CUDA 12.8 构建（版本与 Mac 相同，都是 torch 2.7.1）；旧服务器用的 cu118 构建没有 Blackwell kernel，不能在 B200 上运行。默认 `device: cuda`，不可用会报错；CPU/Mac 工作需显式改为 `cpu`/`mps`。`precision: auto` 在 CUDA 上选择可用的 BF16，权重及 loss reduction 保持 FP32。`runtime.num_workers: 4` 是估计的起点，须在 B200 上按 `data_wait_seconds` 实测调整。本机未实测 B200；[部署和检查说明](docs/guide.md#b200--nvidia-conda-环境)解释了验收步骤。

把真实数据路径和 labels 填入 `common.yaml` 后：

```bash
python train_head.py
python pretrain_encoder.py
python predict.py
```

三条命令分别运行，按需要选择。SSL 完成后，将 `common.yaml` 的 `model.encoder_checkpoint` 指向其 `encoder.pt`，给 head 配置新的输出目录，再运行 `train_head.py`。每个实验使用新目录。

## 实时查看

终端显示 epoch/batch、ETA 和指标。训练自动更新图，不需要 GUI；在另一个终端运行：

```bash
python utilities/monitor_training.py
```

默认仅提供 `127.0.0.1:8766`。远程服务器通过 IDE 端口转发或 SSH 转发访问。也可直接打开共享文件系统中的 dashboard。

```text
outputs/<experiment>/
  logs/                       # train.log、完整 metrics.jsonl、错误 traceback
  figures/                    # 高分辨率 PNG 和矢量 SVG
  monitoring/dashboard.html   # 自动刷新
  monitoring/curves.png        # 同一出版级样式的实时总览
  settings.yaml               # 本次完整配置
  environment.json            # 实际 Conda/Python/CUDA/GPU 信息
  checkpoints/                # SSL：最近几份 encoder；另有 encoder.pt、encoder_best.pt、last_training.pt
```

Head 每 epoch 更新 train/validation loss、accuracy、macro-F1。SSL 按配置更新 raw/EMA loss、validation loss、表示方差、梯度和学习率；完整 batch 指标始终保留，包括等数据时间、含等待的 clips/s 和 CUDA/MPS/进程内存（[指标说明](docs/guide.md#训练监测与自动更新曲线)）。`plot_every_steps: 1` 可逐步更新，默认 20 减少绘图开销。

## 清晰的目录分工

```text
emotion2vec_adaptation/
  train_head.py               # 主要入口：有标签分类头
  pretrain_encoder.py         # 主要入口：无标签 encoder
  predict.py                  # 主要入口：匹配模型预测
  config/                     # 所有 YAML、Conda 环境和 requirements
  adaptation/                 # encoder、SSL、head、音频与数据等核心逻辑
  utilities/                  # 配置读取、运行、监控、环境检查、验证和示例工具
  tests/                      # 自动化测试
  presentation/               # English slides、公式源码和构建工具
  docs/                       # 完整方法与操作说明
  outputs/                    # 每次实验的全部产物，忽略于 Git
```

辅助操作集中在 `utilities/`：`prepare_labels.py`、`generate_demo.py`、`smoke_test.py`、`validate_public_audio.py`。例如：

```bash
python utilities/validate_public_audio.py
python -m unittest discover -s tests -v
```

真实公开语音验证检查加载、训练、恢复、encoder 替换、预测及监测；不构成内部情感准确率提升证据。

## 方法与数据规模

当前 SSL 是 **masked EMA 适配基线**，不是原论文完整预训练的精确恢复。默认 target 为 teacher 最后一层（`target: final`）；原论文式的 FFN target 从已微调的 large 模型学不到，mock 中会把下游准确率从 0.75 拉到 0.54–0.58，保留为对照。训练受两道门槛保护：先只训 predictor 检查目标是否可学，再每 N 步重训线性头检查下游 UAR，不达标就停止，并保存下游最好的 `encoder_best.pt`。[目标选择、门槛和 mock 对照实验](docs/guide.md#ssl-目标的选择target)。

已采用原模型的正式 **`other`** 类：默认共 8 个具名情绪类别，`unknown`/`<unk>` 单独保留为不确定状态。历史 Label Studio **Unknown / Other** 实际导出为 `unknown`，需要先区分/复核，不能整批改成 other。[标签说明](docs/guide.md#other--others-与-unknown)。

最新数据规模为 **30 多批，每批约 100K–200K 段，全部大于 2 秒、小于 10 秒**。以 30 批参考，约 3M–6M 段；若仅作平均 5 秒的示例，约 4,167–8,333 小时。实际批次数超过 30，准确数量及时长须统计清单。原论文 emotion2vec 的情绪语音 SSL 为 262 小时；emotion2vec+ large 模型卡报告的 42,526 小时属于伪标签微调，不能混为同一预训练阶段。[完整对比与来源](docs/guide.md#数据规模与原模型对比)。

在这个规模上，SSL 的每个音频文件只完整读取一次，SHA256 与时长缓存在 `data.audit_cache` 并跨运行复用；预算先抽样再读内容；按时长分桶让 SSL batch 不含 padding；`save_every_steps` 在 epoch 中途保存，中断后可精确恢复。[说明](docs/guide.md#百万级数据一次读取无-padding可恢复)

[English presentation](presentation/emotion2vec_adaptation.pptx) · [Presentation source](presentation/README.md)

所有 slides、图表、公式解释和 speaker notes 使用英文；公式采用规范数学排版。训练图使用 benchmark 配色、大字号及高分辨率/矢量导出。

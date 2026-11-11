# EmotionTalk 上下文 pilot：先准备，之后手动运行

2026-11-10。**当前只准备了代码和输入，没有加载 Omni、没有运行预测。** 本机有 Docker 等进程，按用户要求暂缓推理；没有后台任务会自动开始。

## 这次具体比较什么

我们固定同一批目标音频，只改变它前面的输入：

| 条件 | 模型得到什么 | 用来观察什么 |
|---|---|---|
| H0 | 目标音频 | 没有历史时的判断 |
| H1 | 最近一轮真实过去音频＋同一目标音频 | 真实对话历史的总作用 |
| H2 | 同一历史的文字转写＋同一目标音频 | 只有历史文字是否也有帮助 |
| H3 | 另一段对话中匹配的音频＋同一目标音频 | 帮助是否依赖历史与目标相关 |

H3 从 **train 的另一段对话**选音频，匹配真实历史的公开 speaker ID 和时长（±25%），不看它的情绪标签，不使用本轮选中的目标。每条 H3 控制音频只分配一次。H1/H3 的文字提示完全相同，仅历史音频不同。

真实历史满足 `history.end <= target.start`，没有 1ms 容差，也不使用未来或尚未结束的交叠轮次。只用一轮历史，不将几段交叠录音简单拼接。说话人统一写成 Speaker A/B，不暴露人物姓名；目标始终是 Speaker A。

**目标音频始终完整、相同，并作为最后一个独立 audio 输入。** 不拼成长 WAV，不截去目标末尾；目标与历史各不超过 12 秒，总音频预算 30 秒。不符合预算的候选被排除，不靠裁切凑长度。

H2 只给转写，去掉 `[]`、`【】` 中的人工行为/声学标记，例如 laughing、Sobbing；不提供 caption、投票、置信度或任何历史情绪答案。目标转写不发送给模型。所有条件都使用同一套七候选类别和解码设置，不提示少选 neutral。

## 240 条怎么选

配置在 [context_pilot.yaml](../config/context_pilot.yaml)。只从官方 **train** 的完整双人对话选择，不用 validation/test 调 prompt。

- 固定随机种子 `20261110`，七类近似均衡，合计 240 个目标。
- 每段对话最多三个目标；目标至少 0.4 秒，需有非空历史转写。
- 选择不依赖模型答对/答错；gold 仅用于目标分层抽样和之后计分。
- 这是为了观察不同类别的**探索样本**，并非原始类别分布上的正式测试成绩。不能直接与整个 MELD/test 的总分比较。

准备命令会导出实际类别数量、对话数、目标/历史/H3路径与时长，以及全部 960 份请求。运行前可以直接检查这些小文件。

## 准备输入：不会加载模型

在 `emotion_recognition/` 内层目录、已安装项目依赖的 Python 环境中运行：

```bash
python scripts/evaluation/context_pilot.py prepare
```

默认不写子命令时也只执行 prepare。准备只读 CSV/JSON、WAV 文件头并流式计算所选文件的 SHA256；不导入 torch/transformers，不提取 features，也不调用 API。

输出在 `outputs/context_pilot/emotiontalk_train_240_<hash>/`：

| 文件 | 用途 |
|---|---|
| `manifest.json` | 固定目标、真实历史、H3控制、gold、音频校验、模型来源与代码身份；gold只供计分 |
| `targets.csv` | 一行一个目标，包含说话人、划分、配对路径、时长与历史转写 |
| `config.json` | 本轮设置 |
| `requests.jsonl` | 960 份实际消息；没有情绪答案、投票或caption |
| `prompt_preview.json` | 第一条目标的四种完整请求，便于人工检查 |

当前已准备的固定清单：[manifest.json](../outputs/context_pilot/emotiontalk_train_240_067050e3ca58/manifest.json)、[targets.csv](../outputs/context_pilot/emotiontalk_train_240_067050e3ca58/targets.csv)。240 个目标来自 197 段 train 对话，每类 34–35 条；准备文件合计约 1.46 MB，尚无任何模型回答。

## 内存空闲后才运行

下面命令中的路径用 prepare 输出的真实 manifest 路径替换。**本轮没有执行它。**

```bash
python scripts/evaluation/context_pilot.py run --manifest outputs/context_pilot/emotiontalk_train_240_067050e3ca58/manifest.json
```

默认优先 CUDA，其次 MPS；没有 GPU 就停止，不自动悄悄切到 CPU。可显式加 `--device mps`、`--device cuda:0` 或 `--device cpu`。模型仅从本地文件读取，使用现有 Qwen2.5-Omni Thinker，不加载语音输出 Talker，不下载模型、不调用付费接口。多 audio 的处理方式已对照本地 transformers 实现核查；**尚未做真实模型推理验证**。

960 次是首轮请求数；无效格式回答最多追问一次。默认保存首 token 标签分数，不保存 hidden states 或 `.f16`，减少内存和磁盘开销。这些分数不是校准后的情绪概率。

同一目标的条件顺序循环轮换，每条都重新构建独立请求。每个回答立即写入 `runs/<execution-hash>/responses.jsonl`；中断后重跑同一命令可续跑。硬件/模型错误会停止，不伪装成一次答错；已完成请求保留。音频、转写、历史、候选、prompt、解码设置、相关代码或运行环境变化，都不会错误命中旧实验缓存。

## 跑完会得到什么，怎样解释

脚本自动生成 `predictions.csv`、`summary.json`、`report.md`，含逐音频预测、原始回答、首 token 标签分数、配置身份，以及 Accuracy/UAR/macro-F1/weighted-F1、neutral预测率、每类支持数与混淆矩阵。weighted-F1 按真实类别的样本数加权，invalid 仍算错误。H1–H3 对 H0 的变化会报告改判数、由错变对、由对变错及分数差。

所有条件的比较都使用**四条件都完成的相同目标交集**，不会拿较多的 H0 与较少的 H1 比分。未完成时报告明确标为 partial。JSON/CSV与manifest即可用于另一台电脑上的论文分析，无需 `.f16` 或搬回原始大音频。

只重做分析、不加载模型：

```bash
python scripts/evaluation/context_pilot.py analyze --manifest <manifest.json> --run-dir <runs/实际execution-hash>
```

H1 对 H0 是音频历史的总作用；H2 对 H0 是文字历史的作用；H1/H2 并非纯声学/纯语义分解，因为人工转写和音频处理方式不同。H3 匹配 speaker 和时长也无法保证所有录音因素一致。该 pilot 不预设 ICL 一定有效，也不自动给出统计显著性结论。

EmotionTalk 是演员主题即兴对话，发布音频 gold 的标注者可以回听历史。四个条件仍用同一个发布 gold 计分，**不能用它代替无历史/有历史两套独立人类判断**。

## 其他新数据集也能跑普通基线

原 13 个 loader 已有；新增 EmotionTalk、DailyTalk、MultiDialog 均接入统一 `LOADERS`，普通 `evaluate.py` 会自动使用同一模型推理接口。默认范围为 EmotionTalk 官方 test、DailyTalk ECSS validation、MultiDialog 双演员 gold 名单＋整对话 QC 的两套 test 子集。DailyTalk 不伪造一个不存在的官方 test；MultiDialog 的 gold 是演员筛选队列，不是每句的听众多数票。

另备 [evaluation_dialogue_baselines.yaml](../config/evaluation_dialogue_baselines.yaml)，不会替换原来的 MELD 配置。等算力可用后才执行：

```bash
python scripts/evaluation/evaluate.py --config config/evaluation_dialogue_baselines.yaml
```

它计划三个开源音频 LLM 的普通单句基线，API关闭；这与本页 Omni 的 240 条 H0–H3 pilot 是两项独立实验，缓存与输出也分开。

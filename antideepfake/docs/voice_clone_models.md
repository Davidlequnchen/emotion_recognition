# 声音克隆模型一览

English version: [voice_clone_models_en.md](voice_clone_models_en.md)（内容相同）。

日期：2026-10-30。用途：用征得同意的说话人的声音，生成 `data/voice_clone/texts.csv` 里的 200 句合成语音，再用 AntiDeepfake 打分。只选了允许商用的模型（Apache-2.0、MIT）；F5-TTS（权重 CC-BY-NC）和 XTTS-v2（Coqui Public Model License）禁止商用，没有选。

数据来源：
- 「实测」：在本机 M4 Max 上运行 `scripts/voice_clone/generate.py` 测得，或读取下载的配置和代码得到；
- 「论文」「model card」：来自第 2 节的链接。

## 1. 总览

| | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox（英文版） |
|---|---|---|---|
| 本项目里的 key | `cosyvoice2` | `cosyvoice3` | `chatterbox` |
| 开发者 | 阿里通义实验室（FunAudioLLM） | 阿里通义实验室（FunAudioLLM） | Resemble AI |
| 发布 | 2024-12 | 2025-12 | 2025 |
| 参数量（实测，PyTorch 部分 + ONNX） | 639.1M + 130.6M ≈ 0.77B | 859.2M + 248.9M ≈ 1.11B | 797.8M（另有 Perth 水印网络 9.3M） |
| 下载大小（实测，只含运行需要的文件） | 3.09 GB | 4.44 GB | 3.19 GB |
| 许可 | Apache-2.0 | Apache-2.0 | MIT |
| 语言 | 中、英、日、韩（论文训练数据） | 9 种：中、英、日、韩、德、西、法、意、俄，另有 18 种以上中文方言（model card） | 只有英文 |
| 克隆方式 | zero-shot：参考录音 + 文字稿 | zero-shot：参考录音 + 文字稿 | zero-shot：只要参考录音 |
| 输出 | 24 kHz 单声道 | 24 kHz 单声道 | 24 kHz 单声道，带不可听的 Perth 水印 |
| Mac 上的设备（实测） | CPU（代码只认 CUDA 或 CPU） | CPU | MPS（也可用 CPU） |
| 速度 RTF（实测，M4 Max） | 约 0.95 | 约 1.1–1.3 | MPS 约 1.3–1.6；CPU 约 2.3–2.5 |
| 内存峰值（实测） | 6.7 GB | 9.3 GB | 6.1 GB（含 GPU 共 14.7 GB） |

RTF = 生成时间 ÷ 音频时长，小于 1 表示比实时快。

## 2. 来源、论文和公司电脑的获取方式

公司电脑能访问 GitHub、Zenodo 和 PyPI，不能访问 Hugging Face 和 ModelScope。

| | 权重 | 代码 | 论文 | 公司电脑怎么拿到 |
|---|---|---|---|---|
| CosyVoice2-0.5B | [HF FunAudioLLM/CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B)，revision `eec1ae6c`；也在 [ModelScope iic/CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) | [GitHub FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)，commit `074ca6dc`（子模块 Matcha-TTS `dd9105b3`） | [CosyVoice 2（arXiv:2412.10117）](https://arxiv.org/abs/2412.10117) | 权重：公开仓库 [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models) 里的 90 MiB 分片；代码：下载脚本从 GitHub clone |
| Fun-CosyVoice3-0.5B-2512 | [HF FunAudioLLM/Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)，revision `29e01c4e`；也在 [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) | 同上 | [CosyVoice 3（arXiv:2505.17589）](https://arxiv.org/abs/2505.17589) | 同上 |
| Chatterbox | [HF ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)，revision `5bb1f6ee` | [GitHub resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox)；pip 包 `chatterbox-tts==0.1.7` | 无论文（只有 model card 和 GitHub） | 权重：公开仓库 [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models) 里的分片；代码：PyPI |
| wetext 文本归一化 FST（CosyVoice 用） | [ModelScope pengzhendong/wetext](https://www.modelscope.cn/models/pengzhendong/wetext)，revision `67928a15`，Apache-2.0 | pip 包 `wetext==0.0.4` | – | 公开仓库 [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models)（4 个文件，共 11 MB） |

完整的 revision 和 commit 写在 `config/voice_clone.yaml` 里。公司电脑的做法：把公开仓库 [voice_clone_models](https://github.com/Davidlequnchen/voice_clone_models) clone 到 `models/voice_clone/`，运行其中的 `join_parts.py` 合并分片（逐个校验 SHA-256），再运行 `download_models.py`：它从 GitHub clone CosyVoice 代码，模型文件已存在就跳过，不会连 Hugging Face。

## 3. 输入和输出

| | CosyVoice2 / CosyVoice3 | Chatterbox |
|---|---|---|
| 参考录音 | 任意采样率，内部重采样为 16 kHz（speech tokenizer、CAM++）和 24 kHz（prompt mel）；最长 30 s（代码里有 assert） | 内部重采样为 16 kHz 和 24 kHz；T3 只用前 6 s（150 个 token），S3Gen 只用前 10 s |
| 参考文字稿 | 必需，要和录音一字不差；CosyVoice3 要在前面加 `You are a helpful assistant.<|endofprompt|>`（config 里已设置） | 不需要 |
| 要合成的文字 | 任意长度：先用 wetext 做文本归一化（数字、日期、金额转成单词），再按句切分，逐句生成后拼接 | 最多 2048 个文本 token；生成上限 1000 个语音 token（约 40 s） |
| 输出 | float32 波形，24 kHz 单声道；本项目存为 16-bit PCM wav | 同左 |
| 本项目的文件 | `outputs/voice_clone/<speaker>/<model>/<id>.wav`，加 `manifest.csv` | 同左 |

## 4. 网络结构

| 组件 | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox |
|---|---|---|---|
| 文本 tokenizer | Qwen2 BPE（vocab 151,936） | Qwen2 BPE | BPE，704 词，英文 |
| 语音 tokenizer（参考录音 → token） | ONNX，123.7M；基于 SenseVoice-Large 编码器 + FSQ；输入 16 kHz 128 维 log-mel；25 token/s，码本 6561 | ONNX，242.0M；基于 MinMo 语音编码器 + FSQ，多任务训练（ASR、情绪、语种、声音事件、说话人）；25 token/s，码本 6561 | S3TokenizerV2，123.7M（和 CosyVoice2 同结构）；25 token/s，码本 6561 |
| 说话人编码器 | CAM++（ONNX，6.9M），16 kHz 80 维 fbank → 192 维 | CAM++（同左） | Voice encoder 1.4M（256 维，给 T3）+ CAM++ 6.8M（给 S3Gen） |
| 语言模型（文字 → 语音 token） | 505.8M，Qwen2.5-0.5B 骨干：24 层，hidden 896，14 头（2 个 KV 头）；自回归，接在参考 token 后面续写 | 506.1M，同样的 Qwen2.5-0.5B 骨干；用 DiffRO（可微奖励）做后训练 | T3，532.4M：Llama 结构，30 层，hidden 1024，16 头；条件为说话人向量、情绪强度和参考 token |
| Flow matching（token → mel） | 112.5M：6 层 UpsampleConformerEncoder + 因果卷积 U-Net CFM；chunk-aware，支持流式；10 步 Euler，CFG 0.7 | 332.3M：DiT，22 层，dim 1024；10 步 Euler，CFG 0.7 | S3Gen 的 flow，112.5M（和 CosyVoice2 同结构），10 步，CFG 0.7 |
| 声码器（mel → 波形） | HiFT（HiFi-GAN + iSTFT，NSF 音源），20.8M，24 kHz | CausalHiFT，20.8M，24 kHz | HiFT，20.8M，24 kHz |
| 其他 | – | – | Perth 隐式水印，9.3M（随 pip 包分发），每条输出都会加，上游没有开关 |
| 采样方式 | RAS：top_p 0.8，top_k 25，窗口 10，tau_r 0.1 | 同左 | temperature 0.8，min_p 0.05，top_p 1.0，repetition_penalty 1.2，CFG 0.5 |

## 5. 训练数据和报告的指标

| | CosyVoice2-0.5B | Fun-CosyVoice3-0.5B-2512 | Chatterbox |
|---|---|---|---|
| 训练数据 | 约 16.7 万小时：中文 13 万、英文 3 万、日语 4,600、韩语 2,200（论文） | 约 100 万小时，9 种语言加 18 种方言；语音 tokenizer 用 53 万小时多任务数据训练（论文） | 50 万小时清洗过的数据（model card） |
| SEED test-zh CER / SS | 1.45% / 75.7% | 1.21% / 78.0% | – |
| SEED test-en WER / SS | 2.57% / 65.9% | 2.24% / 71.8% | – |
| SEED test-hard CER / SS | 6.83% / 72.4% | 6.71% / 75.8% | – |
| 指标来源 | CosyVoice3 model card | CosyVoice3 model card（RL 版 `llm.rl.pt` 的 WER 更低，但没有下载） | model card 只给了和 ElevenLabs 的主观对比 |

CER/WER 越低越好（内容准确）；SS 是说话人相似度，越高越像参考声音。

## 6. 生成参数（`config/voice_clone.yaml`，默认值即上游默认值）

| 参数 | 模型 | 默认 | 作用 |
|---|---|---|---|
| `speed` | CosyVoice2/3 | 1.0 | 语速 |
| `text_frontend` | CosyVoice2/3 | true | wetext 文本归一化（数字、金额转成单词） |
| `prompt_prefix` | CosyVoice3 | `You are a helpful assistant.<\|endofprompt\|>` | 加在参考文字稿前面 |
| `exaggeration` | Chatterbox | 0.5 | 情绪强度，0.25–2.0，0.5 为中性 |
| `cfg_weight` | Chatterbox | 0.5 | CFG 权重，0–1，越低语速越慢 |
| `temperature`、`repetition_penalty`、`min_p`、`top_p` | Chatterbox | 0.8、1.2、0.05、1.0 | 语音 token 的采样参数 |
| `seed` | 全部 | 1234 | 第 i 行（从 0 数）用 seed + i，同一台机器、同一设备上每次生成的文件逐字节相同（实测：CosyVoice 在 CPU 上、Chatterbox 在 MPS 和 CPU 上；MPS 和 CPU 的结果彼此不同） |
| `mac_device` | 每个模型 | CosyVoice: cpu；Chatterbox: mps | Mac 上用的设备 |

## 7. 下载的文件

| 模型 | 文件（大小） | 没下载的文件及原因 |
|---|---|---|
| `models/voice_clone/cosyvoice2_0.5b/` | `llm.pt` 2023 MB、`speech_tokenizer_v2.onnx` 496 MB、`flow.pt` 451 MB、`hift.pt` 83 MB、`campplus.onnx` 28 MB、`cosyvoice2.yaml`、`CosyVoice-BlankEN/` 的 tokenizer 和 config 小文件、`README.md` | `CosyVoice-BlankEN/model.safetensors`（988 MB，是原版 Qwen2.5-0.5B 权重，会被 `llm.pt` 全部覆盖；本项目只用 config.json 建骨干，生成结果和官方路径逐位相同）、`*.batch.onnx`（训练用）、`flow.decoder.estimator.fp32.onnx`（TensorRT 用） |
| `models/voice_clone/cosyvoice3_0.5b/` | `llm.pt` 2025 MB、`flow.pt` 1329 MB、`speech_tokenizer_v3.onnx` 970 MB、`hift.pt` 83 MB、`campplus.onnx` 28 MB、`cosyvoice3.yaml`、`CosyVoice-BlankEN/` 小文件、`README.md` | 同上，另有 `llm.rl.pt`（2.0 GB，RL 版语言模型，官方代码只加载 `llm.pt`） |
| `models/voice_clone/chatterbox/` | `t3_cfg.safetensors` 2130 MB、`s3gen.safetensors` 1057 MB、`ve.safetensors` 5.7 MB、`conds.pt`、`tokenizer.json`、`README.md` | 同一权重的 `.pt` 副本、`s3gen_v3`、全部多语种文件 |
| `models/voice_clone/wetext/` | `en/tn/` 和 `zh/tn/` 的 tagger 和 verbalizer FST（共 11 MB）、`README.md` | 新版 FST（给 wetext 0.1.x 用，配 0.0.4 会读错数字或崩溃） |

# 语音深度伪造检测开源模型调研（Speech Deepfake Detection Open-Source Model Survey）

英文版：[deepfake_models_survey_en.md](deepfake_models_survey_en.md)（内容相同）。本项目现在用的模型见 [xlsr_1b_antideepfake.md](xlsr_1b_antideepfake.md)。

调研日期：2026-10-30。目标：调研可以下载权重的语音深度伪造检测（anti-spoofing）模型，找出和 XLS-R-1B-AntiDeepfake 同类、或可能更好的模型，选出下一步值得评估的。使用场景：新加坡银行呼叫中心，英语，8 kHz G.711 电话音频，检测来电者被克隆或合成的声音。
筛选标准：权重可以下载，有代码或文档说明怎么运行（非商用许可也收录，但会标出）；针对语音，不含歌声、环境声；大小不限；NII 的 AntiDeepfake 全家都收录。没有可下载权重的模型只在第 5b 节列一下。

方法：查 Hugging Face API（文件列表和大小、许可、创建日期、近 30 天下载量）、Zenodo API 和 GitHub API；读模型卡、论文（arXiv、ISCA、ACL、PMLR）和代码；读两个在同一批测试集上重新打分很多开源模型的公开排行榜：**Speech DF Arena**（Idiap、TalTech 等，2025；下文简称 DF-Arena）和 **Speech Anti-Spoofing Arena**（SpeechAntiSpoofingBenchmarks，2026；下文简称 ASAS-Arena），后者的完整分数表读自其 Space 的 `cache.json`（2026-10-30 的状态）。
参数量：safetensors 文件从文件头里每个张量的形状来数（HTTP range 请求）；.pt、.pth、.ckpt、.bin（PyTorch zip 格式）用 HTTP range 请求只读出压缩包里的 `data.pkl`，按张量形状来数。这些标"实测"。都没有下载权重。参数量算文件里的全部内容（含 BatchNorm 统计量）；文件里还有只在预训练用的部分时，另给推理用的参数量。
大小以十进制 MB、GB 计。参数量：M = million（百万），B = billion；不到 1B 保留 1 位小数，1B 以上保留 2 位。EER = 等错误率，越低越好，数字都是 %。标"未核实"的没能从一手资料确认。
公司电脑能否下载（能访问 GitHub、Zenodo、PyPI，不能访问 Hugging Face 和 ModelScope）：✅ = 权重在 GitHub 或 Zenodo；❌ = 只在 Hugging Face 或 ModelScope；❓ = Google Drive、OneDrive 或其他网站，未测试。

## 0. 要点

- **训练数据公开的开源模型里，AntiDeepfake 全家仍是最好的选择。** 凡是收录了它的排行榜和论文，它都在开源模型的第一梯队；只用 ASVspoof 2019 训练的模型，一离开 ASVspoof 就差得远。
- **最省事的升级是 XLS-R-2B-AntiDeepfake**（同一作者、同一套代码、在 Zenodo 上）：6 个零样本测试集里有 5 个 EER 比 XLS-R-1B 低（In-the-Wild 1.23 对 1.35，ADD 2023 4.67 对 5.39，FakeOrReal-norm 1.64 对 12.14），但 Deepfake-Eval-2024 上没有更好（27.76 对 26.76）。代价：2.16B 参数（2.2 倍），文件 8.65 GB，计算量也大约 2.2 倍（估计，未实测）。`inference.py` 只需改宽度（1920）和 FFN（7680）。
- **没有哪个开源模型能搞定真实场景的音频。** 在 Deepfake-Eval-2024（2024 年网上流传的深度伪造）上，AntiDeepfake 所有模型零样本 EER 都在 25.8–34.1%；AASIST、RawNet2 和 NII 更早的 P3 在 43–55%。只有用目标领域的数据微调才有用（AntiDeepfake 微调后 8.3–12.1%，权重未公开）。对银行来说，用自己的电话录音和克隆语音微调，比换一个模型更重要。
- **只用 ASVspoof 2019 LA 训练的模型纸面上很好，泛化很差**：AASIST、wav2vec2-AASIST、XLSR-SLS、TCM、XLSR-Mamba、Nes2Net、XLSR-MamBo、SONAR 在 ASVspoof 2019/2021 上 0.1–3%，在 In-the-Wild 上 5–11%，在较新的数据集上 20–45%。在 ASAS-Arena 里 AntiDeepfake 从没见过的 12 个数据集上，它们的平均错误率 13–42%，XLS-R-1B/2B-AntiDeepfake 是 2.0–2.4%。它们不算升级。
- **另有两类强模型，都有附加条件：**
  - DF_Arena_1B_V_1（Speech DF Arena 团队，XLS-R 1B + Conformer，实测 1.15B）：DF-Arena 上最好的开源模型（In-the-Wild 0.91%）；在 ASAS-Arena 那 12 个数据集上和 XLS-R-1B-AntiDeepfake 大致持平（平均 2.83 对 2.37）。训练数据不同，适合做第二意见；但许可非商用，只在 Hugging Face 上，而且只看前 4.04 秒。
  - Spectra-AASIST3 / Spectra-AASIST / Spectra-0（lab260，Apache-2.0，约 0.32B）：ASAS-Arena 上错误率最低（平均 1.3–1.6），也是唯一许可宽松的强模型；但没发表论文，训练数据不公开（可能包含测试集），而且这个排行榜就是它们的作者在维护。
- **许可：强的模型都是非商用或无法核实。** AntiDeepfake（CC BY-NC-SA 4.0；MMS 版还继承 MMS 的 CC BY-NC 4.0）、DF Arena（非商用）、SONAR-Full 和 AASIST3（CC BY-NC 4.0）。可商用的：AASIST、RawGAT-ST、wav2vec2-AASIST、TCM、XLSR-Mamba（MIT），Nes2Net（Apache-2.0），NII P3（BSD-3-Clause，在 Zenodo 上），Spectra（Apache-2.0）；都更弱，或训练数据不明。生产上用，需要向 NII 或 Idiap 买商用许可，或者用可商用的数据自己训练。
- **电话音频**：没有开源模型报告过 8 kHz G.711 通话上的结果。最接近的参照是 ASVspoof 2021 LA（评测数据经过电话编解码和传输）和 ASVspoof 5（有编解码条件）。AntiDeepfake 后训练时见过 ASVspoof 2021 LA。真正的检验是我们 `evaluate.py` 的电话信道条件。
- **各家结果很难直接比**：AntiDeepfake 用了大多数官方测试集训练（ASVspoof 2019/2021/5、CodecFake、DFADD、MLAAD、SpoofCeleb、基于 LibriTTS 的数据集……），干净的只有 In-the-Wild、ADD 2023、DEEP-VOICE、FakeOrReal、Deepfake-Eval-2024 和 ASAS-Arena 的部分数据集。ASAS-Arena 只给每个文件的前 4 秒打分，所以它报的 XLS-R-1B-AntiDeepfake 在 In-the-Wild 上是 4.24%，论文按整段音频是 1.35%。
- **建议下一步评估**（表中标 ⭐ 的模型；接入难度见第 6 节）：
  1. XLS-R-2B-AntiDeepfake：全家零样本 EER 总体最好；在 Zenodo 上；`inference.py` 改配置即可。
  2. 同一套代码能跑的其他成员：NDA 版（不用 RawBoost 训练；XLS-R-2B-NDA 的 Deepfake-Eval-2024 EER 全家最低，25.76%）、MMS-1B（形状和 XLS-R-1B 完全一样，FakeOrReal 上好很多）、Wav2Vec2-Large（0.32B，快）。把其中两三个的分数取平均，就是成本很低的集成。
  3. DF_Arena_1B_V_1：最强的独立开源模型；要装 transformers，权重要手动搬过去（只在 Hugging Face）；非商用。
  4. Spectra-AASIST3（或 Spectra-AASIST）：许可宽松，排行榜成绩好；正因为训练数据不明，才值得在我们自己的克隆语音上验证；只在 Hugging Face。
  - 没标 ⭐ 的：只用 ASVspoof 2019 训练的模型（第 2 节 #7–#15）和经典小模型（#16–#21），只能当弱基线；AntiDeepfake 里的 HuBERT-XL 和 Wav2Vec2-Small 是全家最弱的。

## 1. NII AntiDeepfake 全家

14 个 checkpoint 都出自同一篇论文（Ge、Wang、Liu、Yamagishi，"Post-training for Deepfake Speech Detection"，ASRU 2025，[arXiv 2506.21090](https://arxiv.org/abs/2506.21090)），用同样的 74.7k 小时数据后训练（真 56.4k 小时、假 18.3k 小时，29 个来源；详见 [xlsr_1b_antideepfake.md](xlsr_1b_antideepfake.md) 第 4.2 节），分类头也相同：对所有帧取平均，再接 Linear(D→2)。每个模型有默认版（用 RawBoost 训练）和 NDA 版（不做数据增强）。
全部在 [Zenodo record 15580543](https://zenodo.org/records/15580543) ✅，也在 Hugging Face（`nii-yamagishilab/*-anti-deepfake[-nda]`）❌；checkpoint 许可 CC BY-NC-SA 4.0，代码 BSD-3-Clause。

⭐ = 推荐优先评估；只有推荐的模型名字加粗。

### 1a 基本信息和零样本 EER

EER（%）取自论文 Table II，格式"用 RawBoost / NDA"。这些测试集都没用于预训练和后训练。ASAS-Arena 一列：ASAS-Arena 给 AntiDeepfake 打分的 12 个数据集上的平均错误率（10 个 EER，加 2 个只有伪造样本、按 1−SRR 计的数据集），每个文件只用前 4 秒。

| # | 模型 | 参数量（实测：推理 / 文件） | 文件（Zenodo） | ADD 2023 T1.2 R2 | DEEP-VOICE | FakeOrReal | FoR-norm | In-the-Wild | Deepfake-Eval-2024 | ASAS-Arena：In-the-Wild / 12 集平均（4 s） | SSL 底座的许可 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLS-R-1B-AntiDeepfake（现在用的） | 962.5M / 965.5M | `xlsr_1b.ckpt` 3.86 GB | 5.39 / 6.58 | 2.47 / 2.59 | 5.74 / 3.17 | 12.14 / 10.91 | 1.35 / 1.36 | 26.76 / 26.17 | 4.24 / 2.37（NDA 3.95 / 2.58） | XLS-R：Apache-2.0 |
| 2 | ⭐ **XLS-R-2B-AntiDeepfake** | 2.16B / 2.16B（2,159.3M / 2,162.9M） | `xlsr_2b.ckpt` 8.65 GB | 4.67 / 6.84 | 2.23 / 2.72 | 2.61 / 1.19 | 1.64 / 1.73 | 1.23 / 1.31 | 27.76 / 25.76 | 2.81 / 2.28（NDA 2.99 / 2.00） | XLS-R：Apache-2.0 |
| 3 | ⭐ **MMS-1B-AntiDeepfake** | 962.5M / 965.5M | `mms_1b.ckpt` 3.86 GB | 9.06 / 9.46 | 2.47 / 2.35 | 1.23 / 0.88 | 1.73 / 1.10 | 1.82 / 1.87 | 27.70 / 27.54 | 3.30 / 3.82（NDA 3.83 / 2.98） | MMS：CC BY-NC 4.0 |
| 4 | MMS-300M-AntiDeepfake | 315.4M / 317.4M | `mms_300m.ckpt` 1.27 GB | 7.93 / 11.22 | 2.35 / 3.06 | 1.40 / 0.48 | 5.92 / 2.70 | 2.90 / 2.00 | 32.84 / 31.39 | 4.83 / 4.31（NDA 3.77 / 4.24） | MMS：CC BY-NC 4.0 |
| 5 | ⭐ **Wav2Vec2-Large-AntiDeepfake** | 315.4M / 317.4M | `w2v_large.ckpt` 1.27 GB | 13.25 / 12.67 | 4.44 / 5.04 | 0.67 / 0.80 | 0.97 / 1.45 | 1.91 / 2.25 | 33.36 / 30.06 | 2.89 / 3.68（NDA 2.94 / 3.44） | wav2vec 2.0 Large：Apache-2.0 |
| 6 | HuBERT-XL-AntiDeepfake | 962.5M / 964.3M | `hubert_xl.ckpt` 3.86 GB | 18.91 / 35.34 | 5.68 / 14.84 | 2.48 / 3.67 | 3.18 / 15.56 | 5.23 / 17.99 | 34.10 / 47.73 | 3.46 / 5.23（NDA 10.93 / 14.92） | HuBERT：Apache-2.0 |
| 7 | Wav2Vec2-Small-AntiDeepfake | 94.4M / 95.0M | `w2v_small.ckpt` 0.38 GB | 13.02 / 19.41 | 9.79 / 16.20 | 22.03 / 1.06 | 17.89 / 6.48 | 4.23 / 4.65 | 33.33 / 31.97 | 5.31 / 6.26（NDA 4.95 / 6.23） | wav2vec 2.0 Base：Apache-2.0 |

- 每个 NDA 文件和默认版一样大（Zenodo 上的 `*_nda.ckpt`，MD5 也在那里公布）。
- "文件"参数量包括只在预训练用的部分（quantizer、`project_q`、`final_proj`、`mask_emb`；HuBERT 是 `label_embs_concat`），0.7M–3.7M，推理时去掉。
- 在 Deepfake-Eval-2024 训练集上微调后（未公开），4 秒片段：XLS-R-1B 11.85、MMS-1B 12.11、XLS-R-2B 12.14、MMS-300M 17.15、W2V-Large 19.56；50 秒片段 8.29–11.37（GitHub README）。
- NII 之后还发表了用强化学习（GRPO）微调的工作（Wang、Ge、Yamagishi，Interspeech 2026，[arXiv 2603.02914](https://arxiv.org/abs/2603.02914)）：代码和配置在仓库里（`README_grpo.md`），没找到微调后的权重。

### 1b 架构、训练数据和输出

| # | 模型 | 架构（`models/W2V_configs.py` 里的 fairseq 配置） | SSL 预训练数据 | 输出 | 输入音频 | 备注 |
|---|---|---|---|---|---|---|
| 1 | XLS-R-1B | wav2vec 2.0：7 层 CNN（layer norm）+ 48 层 Transformer，宽 1280，FFN 5120，16 头，pre-LN；平均池化 + Linear(1280→2) | XLS-R：436k 小时，128 种语言 | 2 个 logits [fake, real]；softmax 第 1 位 = P(real)；默认阈值 0.5 | 16 kHz 单声道，任意长度，整段波形归一化（零均值、单位方差）；后训练时长于 13 秒的文件随机截成 10–13 秒 | 详见 [xlsr_1b_antideepfake.md](xlsr_1b_antideepfake.md) |
| 2 | ⭐ **XLS-R-2B** | 同 #1，但宽 1920，FFN 7680（48 层，16 头） | 同 #1 | 同 #1 | 同 #1 | 论文：后训练只把略多于 90% 的数据过了一遍（其他模型大部分数据过了两遍）；每批最多 50 秒，不是 100 秒 |
| 3 | ⭐ **MMS-1B** | 和 #1 形状完全相同（张量名和大小都一样） | MMS：约 50 万小时，1,400 多种语言（MMS 论文；这里未核实） | 同 #1 | 同 #1 | MMS 底座是 CC BY-NC 4.0 |
| 4 | MMS-300M | 24 层，宽 1024，FFN 4096，16 头，pre-LN，layer-norm CNN | 同 #3 | 同 #1 | 同 #1 | — |
| 5 | ⭐ **Wav2Vec2-Large** | 形状同 #4 | README 写的是 wav2vec 2.0 Large "LV-60 + CV + SWBD + FSH"（即 large-robust，含电话语音 Switchboard 和 Fisher）；Hugging Face 卡片写的却是 `wav2vec2-large-960h-lv60-self`（不一致，未核实） | 同 #1 | 同 #1 | 如果真是 large-robust，它是全家唯一用电话语音预训练过的 |
| 6 | HuBERT-XL | HuBERT：48 层，宽 1280，FFN 5120；CNN 无 bias | Libri-Light 60k 小时（英语） | 同 #1 | 同 #1 | 学习率 5e-6，不是 1e-7；NDA 版差很多 |
| 7 | Wav2Vec2-Small | wav2vec 2.0 Base：12 层，宽 768，group-norm CNN，post-LN | LibriSpeech 960 小时 | 同 #1 | 同 #1 | 层结构和 #1 不同（要多写代码） |

## 2. 其他可下载权重的检测模型

表 2a 是基本信息和公开报告的效果，表 2b 是架构、数据、输出和输入音频，两张表的 # 一一对应。"论文"数字来自作者；"DF-Arena""ASAS-Arena"数字是排行榜重新打分的；三种来源评测方式不同，不能混用（见第 3 节）。ASAS-Arena 12 集平均：和表 1a 是同样的 12 个数据集，可以直接和 4 秒下的 AntiDeepfake 比。"—"表示没有报告。

### 2a 基本信息和公开报告的效果

| # | 模型 | 机构 | 年份 | 公开报告的效果（EER %） | 论文 | 下载 | 大小 | 许可 | 怎么运行 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ⭐ **DF_Arena_1B_V_1** | Speech DF Arena 团队（Idiap、TalTech、MBZUAI） | 2025-10 | DF-Arena：pooled 9.52（开源模型最好；商用系统 1.59–2.32），In-the-Wild 0.91，ASVspoof 2021 LA 4.66，2021 DF 1.75，FakeOrReal 2.92，ADD 2023 R2 11.54，SONAR 1.09；ASAS-Arena 12 集平均 2.83 | [arXiv 2603.06164](https://arxiv.org/abs/2603.06164)（RAPTOR） | [HF](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1) ❌ | 1.15B（实测）；`pytorch_model.bin` 4.59 GB | 自定义非商用许可（部分 MIT）；商用许可另谈 | transformers `pipeline("antispoofing", trust_remote_code=True)` |
| 2 | DF_Arena_500M_V_1 | 同 #1 | 2025-11 | DF-Arena：pooled 10.88，In-the-Wild 1.76，ADD 2023 R2 12.30；ASAS-Arena 12 集平均 5.38 | 同 #1 | [HF](https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1) ❌ | 436.2M（实测）；1.75 GB | 同 #1 | 同 #1 |
| 3 | ⭐ **Spectra-AASIST3** | lab260（MTUCI，俄罗斯） | 2026-04 | ASAS-Arena：In-the-Wild 1.20，ASVspoof 2021 LA 4.38，2021 DF 4.30，ASVspoof 5 15.09，12 集平均 1.34；卡片：In-the-Wild 0.96 | 无（未发表） | [HF](https://huggingface.co/lab260/Spectra-AASIST3) ❌ | 319.1M（实测）；1.28 GB（另有 ONNX 1.28 GB） | Apache-2.0 | 仓库里的 PyTorch 代码（`PyTorchModelHubMixin`） |
| 4 | Spectra-AASIST | 同 #3 | 2026-04 | ASAS-Arena：In-the-Wild 1.46，2021 LA 5.25，2021 DF 2.52，ASVspoof 5 14.22，12 集平均 1.38 | 无 | [HF](https://huggingface.co/lab260/Spectra-AASIST) ❌ | 316.0M（实测）；1.26 GB（另有 ONNX） | 文件头写 Apache-2.0，正文写 "MIT" | 同 #3 |
| 5 | Spectra-0 | 同 #3 | 2026-01 | ASAS-Arena：In-the-Wild 1.10，ASVspoof 5 14.82，12 集平均 1.60 | 无 | [HF](https://huggingface.co/lab260/spectra_0) ❌ | 318.3M（实测）；1.27 GB | Apache-2.0 | 同 #3 |
| 6 | NII P3（声码器数据 SSL） | NII Yamagishi Lab | 2023-09 | 论文：In-the-Wild 6.10，ASVspoof 2019 LA 1.91，2021 LA 15.92，2021 DF 5.67，WaveFake 1.30；Deepfake-Eval-2024 43.00（AntiDeepfake Table II），AUC 0.58（Deepfake-Eval-2024 论文） | [arXiv 2309.06014](https://arxiv.org/abs/2309.06014)（ICASSP 2024） | [Zenodo 8337778](https://zenodo.org/records/8337778) ✅ | `project10-cm-ssl-model-ID-P3.tar` 1.33 GB；参数量未核实（一个 XLS-R-300M 大小的 SSL） | BSD-3-Clause（Zenodo 记录里的 LICENSE）；代码 BSD-3-Clause | NII project-NN-Pytorch-scripts（fairseq） |
| 7 | wav2vec2-AASIST（SSL_Anti-spoofing） | EURECOM、NII 等（Tak 等） | 2022-02 | 论文：ASVspoof 2021 LA 0.82，2021 DF 2.85；In-the-Wild 10.46（Nes2Net 论文）、11.20（DF-Arena）；DF-Arena pooled 19.61；ASAS-Arena 12 集平均 26.90 | [arXiv 2202.12233](https://arxiv.org/abs/2202.12233)（Odyssey 2022） | [Google Drive](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB) ❓；HF 镜像 ❌ | 317.8M（实测）；1.27 GB | 代码 MIT；权重没有单独许可 | [GitHub](https://github.com/TakHemlata/SSL_Anti-spoofing)（fairseq） |
| 8 | XLSR-SLS | Zhang、Wen、Hu | 2024-10 | 论文：2021 LA 2.87，2021 DF 1.92，In-the-Wild 7.46；DF-Arena pooled 16.08（那里最好的 ASVspoof 2019 单数据集模型）；ASAS-Arena 12 集平均 19.71 | ACM MM 2024（[DOI](https://doi.org/10.1145/3664647.3681345)） | Google Drive ❓；HF 镜像 ❌ | 340.8M（实测）；1.36 GB | GitHub 仓库没有许可（HF 镜像写 MIT）：不明 | [GitHub](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)（fairseq） |
| 9 | TCM（XLSR-Conformer + TCM） | 新加坡南洋理工大学（NTU）等（Truong 等） | 2024-06 | 论文：2021 LA 1.03，2021 DF 2.06，In-the-Wild 7.79；DF-Arena pooled 16.69 | [arXiv 2406.17376](https://arxiv.org/abs/2406.17376)（Interspeech 2024） | NTU OneDrive ❓ | 约 319M（DF-Arena；未实测） | MIT | [GitHub](https://github.com/ductuantruong/tcm_add)（fairseq） |
| 10 | XLSR-Mamba | Y. Xiao、R. K. Das | 2024-11 | 论文：2021 LA 0.93，2021 DF 1.88，In-the-Wild 6.71；DF-Arena pooled 20.59（平均 14.65） | [arXiv 2411.10027](https://arxiv.org/abs/2411.10027)（IEEE SPL 2025） | [HF LA](https://huggingface.co/AustinXiao/XLSR-Mamba-LA)、[HF DF](https://huggingface.co/AustinXiao/XLSR-Mamba-DF) ❌ | 319.3M（实测）；各 1.28 GB | MIT | [GitHub](https://github.com/swagshaw/XLSR-Mamba)（fairseq + mamba_ssm） |
| 11 | Nes2Net-X | 新加坡国立大学（NUS）、NTU（Liu 等） | 2025-04 | 论文：2021 LA 1.73，2021 DF 1.65（另一个 checkpoint 1.49），In-the-Wild 5.15（用验证增强选出的 checkpoint）；DF-Arena：In-the-Wild 7.75，pooled 17.37；ASAS-Arena 12 集平均 20.51 | [arXiv 2504.05657](https://arxiv.org/abs/2504.05657)（IEEE TIFS） | Google Drive ❓；HF 镜像 ❌ | 317.9M（实测；后端 0.51M）；1.27 GB | Apache-2.0（"对商用友好"） | [GitHub](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW)（fairseq） |
| 12 | XLSR-MamBo | Ng 等 | 2026-01 | ASAS-Arena：In-the-Wild 5.47，12 集平均 13.33（那里最好的 ASVspoof 2019 单数据集模型） | [arXiv 2601.02944](https://arxiv.org/abs/2601.02944)（ACL 2026 Findings） | HF 镜像 ❌；GitHub 代码 | 319.4M（实测）；1.28 GB | MIT | [GitHub](https://github.com/saki-ciallo/XLSR-MamBo)（mamba_ssm） |
| 13 | SONAR-Full | Hidekel、Lifshitz、Cohen、Raviv | 2025-11 | ASAS-Arena：In-the-Wild 7.41，12 集平均 22.16 | [arXiv 2511.21325](https://arxiv.org/abs/2511.21325)（ICML 2026） | HF 镜像 ❌；[GitHub](https://github.com/idonithid/SONAR-Audio-DF-Detection) | 641.5M（实测）；2.57 GB | 权重 CC BY-NC 4.0；代码 MIT | GitHub 代码（fairseq） |
| 14 | ASAS-Arena 上其他 ASVspoof 2019 SSL 模型 | DPDA-GA（Truong 等，[arXiv 2509.20682](https://arxiv.org/abs/2509.20682)）、SSL feature gating（ACM MM 2025）、Fake-Mamba（[arXiv 2508.09294](https://arxiv.org/abs/2508.09294)）、AMSDF（IEEE TIFS 2024） | 2024–2025 | ASAS-Arena In-the-Wild：DPDA-GA-AASIST 5.41，SSL feature gating 5.53，AMSDF 18.44；12 集平均 SSL feature gating 20.13，AMSDF 30.31 | 见左 | HF 镜像 ❌ | 318–325M | 大多 MIT | 各自的 GitHub 代码 |
| 15 | AASIST3 | MTUCI / lab260 | 2024-08（权重 2025-08） | ASAS-Arena：In-the-Wild 29.72，12 集平均 25.90；卡片说这不是论文用的权重，模型已弃用 | [arXiv 2408.17352](https://arxiv.org/abs/2408.17352) | [HF](https://huggingface.co/lab260/AASIST3) ❌ | 321.7M（实测）；1.29 GB | CC BY-NC 4.0 | 自带代码 |
| 16 | AASIST | Clova AI、EURECOM 等（Jung 等） | 2021-10 | 论文：ASVspoof 2019 LA 0.83；In-the-Wild 43.01；Deepfake-Eval-2024 55.22（AntiDeepfake Table II），AUC 0.43 | [arXiv 2110.01200](https://arxiv.org/abs/2110.01200)（ICASSP 2022） | [GitHub](https://github.com/clovaai/aasist) `models/weights/AASIST.pth` ✅ | 0.30M（实测）；1.3 MB | MIT | GitHub 代码（纯 PyTorch） |
| 17 | AASIST-L | 同 #16 | 2021-10 | 论文：ASVspoof 2019 LA 0.99；ASAS-Arena In-the-Wild 44.45 | 同 #16 | GitHub ✅ | 0.085M；0.43 MB | MIT | 同 #16 |
| 18 | RawGAT-ST | EURECOM（Tak 等） | 2021-07 | ASVspoof 2019 LA 1.06（论文和 DF-Arena）；DF-Arena In-the-Wild 52.54 | [arXiv 2107.12710](https://arxiv.org/abs/2107.12710)（ASVspoof 2021 workshop） | [GitHub](https://github.com/eurecom-asp/RawGAT-ST-antispoofing) `Pre_trained_models/` ✅ | 0.44M；1.9 MB | MIT | GitHub 代码 |
| 19 | RawNet2（ASVspoof 2021 基线） | EURECOM、ASVspoof 组织方 | 2020-11 | DF-Arena In-the-Wild 49.19；Deepfake-Eval-2024 48.20（AntiDeepfake Table II），AUC 0.53 | [arXiv 2011.01108](https://arxiv.org/abs/2011.01108)（ICASSP 2021） | [asvspoof.org zip](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip) ❓ | 17.6M（DF-Arena） | 代码 MIT；权重未核实 | [GitHub](https://github.com/asvspoof-challenge/2021) |
| 20 | ASVspoof 5 基线 AASIST | ASVspoof 5 组织方 | 2024 | 权重附带的结果没公布（未核实） | ASVspoof 5 | [GitHub](https://github.com/asvspoof-challenge/asvspoof5) `Baseline-AASIST/models/weights/AASIST/best.pth` ✅ | 约 0.3M；1.3 MB | 仓库没有许可：不明 | GitHub 代码 |
| 21 | Whisper + MFCC MesoNet | Kawa 等 | 2023-06 | DF-Arena：In-the-Wild 26.72，2021 DF 2.11（它的训练来源） | [arXiv 2306.01428](https://arxiv.org/abs/2306.01428)（Interspeech 2023） | Google Drive ❓；HF 镜像 ❌ | 7.66M（ASAS-Arena） | MIT | [GitHub](https://github.com/piotrkawa/deepfake-whisper-features) |

LFCC-LCNN（ASVspoof 2021 的另一个基线）没列：仓库里只有一个空的 `__pretrained` 文件夹，脚本从哪里下预训练模型未核实。

### 2b 架构、训练数据和输出

| # | 模型 | 架构 | 训练数据 | 输出 | 输入音频 | 备注 |
|---|---|---|---|---|---|---|
| 1 | ⭐ **DF_Arena_1B_V_1** | XLS-R 1B（transformers `Wav2Vec2Model`，用全部 49 层隐状态）→ 每层注意力池化、每层一个 sigmoid 门控、加权求和 → BatchNorm + SELU → 4 个 Conformer 块（4 头，卷积核 31）加 class token → Linear(1280→2)；1.15B 参数（XLS-R 0.96B，Conformer 头 0.19B） | ASVspoof 2019、ASVspoof 2024（= ASVspoof 5）、Codecfake、LibriSeVoc、DFADD、CtrSVDD（歌声）、SpoofCeleb、MLAAD、EnvSDD（环境声）；离线 MUSAN + RIR，在线 RawBoost；用 FakeOrReal 开发集选模型（RAPTOR 论文） | logits [spoof, bonafide]（id 0 = spoof，1 = bonafide）；pipeline 返回预测类别的 softmax 分数和两类分数 | ⚠️ **只用 16 kHz 下前 64,600 个采样点（4.04 秒）**；短的重复补到这个长度；不做归一化；一次一段 | 构建模型时调用 `Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-1b")`，要连 Hugging Face（或本地放一份 config） |
| 2 | DF_Arena_500M_V_1 | 同 #1，底座换成 XLS-R 300M（宽 1024）；436.2M 参数 | 同 #1 | 同 #1 | 同 #1 | 名字叫 500M，实际 0.44B |
| 3 | ⭐ **Spectra-AASIST3** | XLS-R 300M（transformers）→ Linear(1024→128) + SELU → 带 Kolmogorov-Arnold（KAN）层的 AASIST，4 个推理分支；319.1M 参数 | 未公开 | 2 个 logits [spoof, bonafide]；分数 = bonafide logit | 16 kHz；预加重 0.97；卡片代码随机截 64,600 个采样点（4.04 秒），ASAS-Arena 用前 4.04 秒 | 启动时从 Hugging Face 拉 `facebook/wav2vec2-xls-r-300m` 的 config |
| 4 | Spectra-AASIST | XLS-R 300M → MLP → AASIST；316.0M | 未公开 | 同 #3；默认阈值：bonafide logit −1.140625 | 同 #3 | 同 #3 |
| 5 | Spectra-0 | XLS-R 300M → MLP → ECAPA-TDNN；318.3M | 未公开 | 同 #3 | 同 #3 | 同 #3 |
| 6 | NII P3 | 一个 XLS-R-300M 大小的 SSL，由 XLS-R 和一个在声码器重合成 VoxCeleb2 上继续预训练的 SSL 蒸馏而来，加一个轻量后端 | SSL：约 9k 小时 VoxCeleb2，用 4 个声码器重合成；CM 微调：声码器重合成的 ASVspoof 2019 LA 训练集 | CM 分数（分数文件第 2 列），越高越像真人（未核实） | 16 kHz；长度处理未核实 | 论文说 ASVspoof 2021 里的 GSM、PSTN 编解码对它仍然难；在声码器 VoxCeleb2 上微调的变体（P3-c）把 LA21hid 里 GSM/PSTN 部分的 EER 减半（权重未核实） |
| 7 | wav2vec2-AASIST | XLS-R 300M（微调）→ Linear(1024→128) → AASIST 图注意力；317.8M | ASVspoof 2019 LA 训练集 + RawBoost（LA、DF 两个 checkpoint） | 2 个 logits，第 1 位 = bona fide | 训练用 64,600 个采样点（4.04 秒）；测试时整段或 4 秒，看评测方式 | 后来大多数 SSL 后端（#8–#14）都以它为基础 |
| 8 | XLSR-SLS | XLS-R 300M，24 层全部经"敏感层选择"模块加权 → 分类器；340.8M | 同 #7 | 越高越像真人 | 同 #7 | — |
| 9 | TCM | XLS-R 300M → 投影到 144 → 4 个带时间-通道 head token 的 Conformer 块；约 319M | 同 #7 | 越高越像真人；示例阈值 −3.73 | 训练用 4 秒 | NTU |
| 10 | XLSR-Mamba | XLS-R 300M → 144 → 双列双向 Mamba（6 + 6 块）；319.3M | 同 #7（RawBoost；LA、DF 两个 checkpoint） | 越高越像真人 | 训练用 4 秒 | mamba_ssm 要 CUDA（Triton）；Mac 上要用纯 PyTorch 重写扫描 |
| 11 | Nes2Net-X | XLS-R 300M（或 WavLM）→ 0.51M 的嵌套 Res2Net 后端；317.9M | 同 #7 | 越高越像真人 | 训练用 4 秒 | NUS；还有 WavLM 版和 ASVspoof 5 分支 |
| 12 | XLSR-MamBo | XLS-R 300M → 128 → 5 × [3 个 Hydra（双向 Mamba2）+ 注意力]；319.4M | ASVspoof 2019 LA，不用 RawBoost | logit 差，越高越像真人 | ASAS-Arena：前 66,800 个采样点 | — |
| 13 | SONAR-Full | 两个 XLS-R 300M 编码器（波形；可学习 SRM 滤波得到的高频残差）+ 交叉注意力 + AASIST；641.5M | ASVspoof 2019 LA | 第 1 位 = real | 前 64,600 个采样点 | — |
| 14 | 其他 | XLS-R 300M + AASIST/TCM/Mamba/多视角后端 | ASVspoof 2019 LA（+ RawBoost） | 越高越像真人 | 4 秒 | — |
| 15 | AASIST3 | XLS-R 300M + KAN-AASIST；321.7M | 卡片：MLAAD、ASVspoof 5、ASVspoof 2019 LA | 第 1 位 = bona fide | 64,600 个采样点，预加重 | 已弃用，改用 #3 |
| 16 | AASIST | 原始波形上的 sinc 卷积 + 残差编码器 + 频谱-时间图注意力；0.30M | ASVspoof 2019 LA | 第 1 位 = bona fide | 64,600 个采样点 | — |
| 17 | AASIST-L | 更窄的 AASIST；0.085M | 同 #16 | 同 #16 | 同 #16 | — |
| 18 | RawGAT-ST | 原始波形 + 频谱-时间图注意力；0.44M | 同 #16 | 同 #16 | 同 #16 | — |
| 19 | RawNet2 | sinc 滤波器 + 残差块 + GRU；17.6M | ASVspoof 2019 LA | 同 #16 | 约 4 秒 | — |
| 20 | ASVspoof 5 AASIST | 同 #16 | ASVspoof 5 训练集 | 同 #16 | 同 #16 | 这里唯一用 ASVspoof 5（有编解码条件）训练的小模型 |
| 21 | Whisper + MFCC MesoNet | Whisper tiny.en 编码器 + MFCC → MesoInception；7.66M | ASVspoof 2021 DF 的一部分 | 越高越像真人 | Whisper 的固定 30 秒窗口 | — |

## 3. 公共测试集上谁强谁弱

### 3a 最常用于比较的测试集上的 EER（%）

来源：P = 作者论文；D = DF-Arena（基于 wav2vec2、WavLM、HuBERT 的模型按整段打分，每批补齐到最长的文件；其他模型固定 4 秒）；A = ASAS-Arena（每个文件只用前 4 秒）；T = AntiDeepfake Table II。"×" = 这个测试集（或它的来源）在模型的训练数据里，不是零样本结果；"?" = 训练数据不明。

| 模型 | In-the-Wild | ASVspoof 2021 LA | ASVspoof 2021 DF | ASVspoof 5 eval | Deepfake-Eval-2024 | ADD 2023 R2 | FakeOrReal |
|---|---|---|---|---|---|---|---|
| XLS-R-1B-AntiDeepfake | 1.35 P / 4.24 A | × | × | × | 26.76 P | 5.39 P | 5.74 P |
| XLS-R-2B-AntiDeepfake | 1.23 P / 2.81 A | × | × | × | 27.76 P | 4.67 P | 2.61 P |
| DF_Arena_1B_V_1 | 0.91 D / 0.91 A | 4.66 D | 1.75 D | 17.25 D × | — | 11.54 D | 2.92 D（它的开发集用来选模型） |
| Spectra-AASIST3 | 1.20 A ? | 4.38 A ? | 4.30 A ? | 15.09 A ? | — | —（R1：6.64 A） | — |
| NII P3 | 6.10 P | 15.92 P | 5.67 P | — | 43.00 T | — | — |
| wav2vec2-AASIST | 10.46 P / 11.20 D / 11.22 A | 0.82 P / 8.11 A | 2.85 P / 8.32 A | 16.25 D | — | 21.93 D | 7.47 D |
| XLSR-SLS | 7.46 P / 7.46 D | 2.87 P | 1.92 P | 18.76 D | — | 21.10 D | 5.08 D |
| TCM | 7.79 P / 7.79 D | 1.03 P | 2.06 P | 18.85 D | — | 22.74 D | 10.69 D |
| XLSR-Mamba | 6.71 P / 6.71 D | 0.93 P | 1.88 P | 14.40 D | — | 20.15 D（T：19.36） | 6.71 D |
| Nes2Net-X | 5.15 P / 7.75 D / 8.48 A | 1.73 P | 1.65 P | 22.06 D | — | 18.45 D | 6.32 D |
| AASIST | 43.01 D | 11.46 D | 21.07 D | 35.53 D | 55.22 T | 32.47 D | 21.64 D |
| RawNet2 | 49.19 D | 40.07 D | 40.67 D | 41.22 D | 48.20 T | 64.55 D | 65.68 D |

- ASVspoof 2019 LA 和 ASVspoof 2021 LA/DF 的真人语音来源相同（VCTK），所以对用 ASVspoof 2019 训练的模型（#7–#21），2021 两列是"跨数据集但同一家族"，不算真正的领域外。
- 同一个 checkpoint 换一种评测方式，数字差很多：wav2vec2-AASIST 在 ASVspoof 2021 LA 上论文（整段）0.82，ASAS-Arena（前 4 秒）8.11；XLS-R-1B-AntiDeepfake 在 In-the-Wild 上 1.35 对 4.24。只比较同一来源的数字。
- DF-Arena 的 "pooled" EER 对 14 个数据集用同一个阈值，是最严的指标；开源模型最好的是 DF_Arena_1B_V_1（9.52），只用 ASVspoof 2019 训练的模型最好的是 XLSR-SLS（16.08）；商用系统（Modulate、Resemble、Hiya）1.59–2.32。AntiDeepfake 不在 DF-Arena 上。
- Deepfake-Eval-2024（Chandra 等，2025）：开源音频模型的 AUC 比在原始基准上平均下降 48%（AASIST 0.43、RawNet2 0.53、P3 0.58）；用 Deepfake-Eval-2024 的 60% 数据微调后 AUC 0.88–0.92。

### 3b 训练和测试的重叠（哪些不算零样本结果）

| 模型 | 在它训练数据里的（不能当独立测试） |
|---|---|
| AntiDeepfake（全部） | ASVspoof 2019 LA（含 eval）、2021 LA、2021 DF、ASVspoof 5（含 test）、CFAD、DECRO、DFADD、Diffuse or Confuse、DiffSSD、DSD、HABLA、MLAAD v5、SpoofCeleb、VoiceMOS 2022、CVoiceFake、WaveFake、LibriTTS 及其声码器版和修复版（所以 SONAR、CD-ADD、LibriSeVoc 也有风险）、VoxCeleb2、FLEURS / FLEURS-R、Codecfake、CodecFake、AISHELL-3、CN-Celeb2、MLS。干净的：In-the-Wild、ADD 2023、DEEP-VOICE、FakeOrReal、Deepfake-Eval-2024，以及表 1a 用的 ASAS-Arena 12 个数据集 |
| DF_Arena_1B / 500M | ASVspoof 2019、ASVspoof 5、Codecfake、LibriSeVoc、DFADD、SpoofCeleb、MLAAD（另有 CtrSVDD、EnvSDD）；FakeOrReal 开发集用来选模型 |
| Spectra-* | 不明 |
| 只用 ASVspoof 2019 的模型 | ASVspoof 2019 LA（以及和 2021 LA/DF 共用的 VCTK 来源） |
| AASIST3（lab260） | MLAAD、ASVspoof 5、ASVspoof 2019 LA |

## 4. Hugging Face 社区模型（几个代表）

下载量大，但都是个人上传：训练集没说明或很小，测试集多是同一批数据的随机切分，卡片上的准确率不可信。两个排行榜都没收录它们。全部只在 Hugging Face ❌，用标准 `AutoModelForAudioClassification` 加载。下载量：近 30 天，2026-10-30 从 Hugging Face API 查得。

| 模型 | 年份 | 下载量 | 公开报告的效果 | 大小（实测） | 许可 | 架构 | 训练数据 | 输出（标签顺序） | 问题 |
|---|---|---|---|---|---|---|---|---|---|
| [mo-thecreator/Deepfake-audio-detection](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) | 2024-05 | 19,904 | 自己评测集上准确率 98.8% | 94.6M；0.38 GB | Apache-2.0 | wav2vec2-base + 分类头 | "None dataset"（没说明） | 0 fake，1 real | 数据不明 |
| [Gustking/wav2vec2-large-xlsr-deepfake-audio-classification](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification) | 2024-05 | 11,483 | Kaggle 上的 ASVspoof 2019 评测子集 EER 4.01% | 315.7M；1.26 GB | Apache-2.0 | wav2vec2-large-XLSR-53（从一个性别识别模型开始）+ 分类头 | 没说明 | 0 real，1 fake | 只有 ASVspoof 2019 子集作证据 |
| [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection) | 2024-05 | 9,785 | 没说明 | 94.6M | Apache-2.0 | wav2vec2-base + 分类头 | "audiofolder" | 0 AIVoice，1 HumanVoice | 数据不明 |
| [garystafford/wav2vec2-deepfake-voice-detector](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector) | 2025-12 | 8,975 | 187 条测试片段上的指标 | 315.7M | Apache-2.0 | 在 Gustking 上微调（上面 12 层） | 1,866 条：14 个 YouTube 视频 + ElevenLabs、Amazon Polly、Kokoro、Hume AI、Speechify、Luvvoice | 0 real，1 fake | 数据很小；卡片说 2.5–13 秒的片段效果最好 |
| [MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) | 2024-06 | 7,772 | 自己评测集上准确率 99.7% | 94.6M | Apache-2.0 | 在 motheecreator/Deepfake-audio-detection（wav2vec2-base）上微调 | "audiofolder" | 0 fake，1 real | 数据不明 |
| [MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection](https://huggingface.co/MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection) | 2024-07 | 359 | 卡片上自己切分的指标 | 86.2M | BSD-3-Clause | Audio Spectrogram Transformer（AudioSet）+ 分类头 | ASVspoof 5 | 0 Bonafide，1 Spoof | 只用 ASVspoof 5 数据 |
| [abhishtagatya/wav2vec2-base-960h-itw-deepfake](https://huggingface.co/abhishtagatya/wav2vec2-base-960h-itw-deepfake) | 2024-01 | 374 | 卡片上自己切分的指标 | 94.6M | Apache-2.0 | wav2vec2-base-960h + 分类头 | In-the-Wild | 0 bona-fide，1 spoof | 用 In-the-Wild 训练，所以不能在它上面评测 |

## 5. 其他检测模型

### 5a 音频大模型、词级检测和工具包（简表）

| 模型 | 机构、年份 | 是什么 | 下载 | 许可 | 备注 |
|---|---|---|---|---|---|
| AntiDeepfake-AudioLLM checkpoints（SLT 2026） | NII，2026-10 | 4 个 CM：XLS-R-300M + FC、Whisper-large-v2 + FC、Qwen2-Audio-7B + LoRA 加分类头，以及加生成式头 | [HF](https://huggingface.co/nii-yamagishilab/slt2026-checkpoints) ❌（[代码](https://github.com/nii-yamagishilab/AntiDeepfake-AudioLLM)） | CC BY-NC 4.0 | 用 ESDD2-CompSpoof-V2（语音混背景事件）训练；研究 CM 在听什么，不是通用检测器 |
| Whisper deepfake word detection | NII，2026-09 | 微调 Whisper large-v3，用下一个 token 预测标出哪些词是伪造的（部分伪造） | [HF](https://huggingface.co/nii-yamagishilab/whisper-deepfake-word-detection) ❌ | 卡片：CC BY 4.0（"见 LICENSE"） | Interspeech 2026，[arXiv 2602.22658](https://arxiv.org/abs/2602.22658)；针对剪辑、拼接的语音 |
| WeDefense | Lin Zhang、J. Rohdin、X. Wang（NII）、N. Evans（EURECOM）等，2026-01 | 检测和定位工具包（Apache-2.0）；预训练模型 "detection_MHFA_wav2vec2_large" | ModelScope ❌ | 工具包 Apache-2.0；权重许可和训练数据未核实 | [arXiv 2601.15240](https://arxiv.org/abs/2601.15240)；[GitHub](https://github.com/zlin0/wedefense) |
| FakeMark | NII，2025-10 | 用水印做溯源，不能检测没打水印的克隆语音 | HF ❌ | CC BY 4.0 | 不适用于来电者的克隆语音 |

### 5b 没有可下载权重（或没找到）

- DF-Arena 上的商用系统：Modulate VELMA-2、Resemble Detect 和 DETECT-3B Omni、Hiya、DLMSL SpeakSure、Whispeak、Syntra、MoLEx（A*STAR）、DF-Raptor（Idiap）。Resemble 开源了别的工具（Resemblyzer、Chatterbox、PerTh 水印），但没开源检测器。
- DF_Arena_100M_V_0 / V_1：标为 "Closed"；Hugging Face 仓库里没有权重文件。
- 在 Deepfake-Eval-2024 上微调的 AntiDeepfake 和 GRPO 微调模型（未公开）；ASAS-Arena 上的 M1（只有分数文件）。
- ALLM4ADD（Qwen-Audio + LoRA；GitHub 有代码，没找到权重）、SpeechFake 基线（ACL 2025；没找到）、BiCrossMamba-ST（Interspeech 2025；写着"代码和模型将公开"，没找到）。

## 6. 接入本项目的难度

`scripts/antideepfake/inference.py` 用纯 PyTorch 按 fairseq 的参数名重写了 XLS-R（不装 fairseq）；编码器类已经把 `dim`、`ffn`、`heads`、`layers` 作为参数，只有分类头写死为 Linear(1280→2)。`evaluate.py` 对几组真/假文件分别在原始音频和 G.711 电话信道下打分；任何能对每个文件给出 P(real) 的模型都能沿用它。

| 难度 | 模型 | 要做的事 |
|---|---|---|
| 只改配置 | XLS-R-2B（宽 1920，FFN 7680）、MMS-1B（和 XLS-R-1B 完全一样）、Wav2Vec2-Large 和 MMS-300M（24 层，1024，FFN 4096）、所有 NDA 版 | checkpoint 的张量名相同（`m_ssl.model.*`、`proj_fc.*`；已从文件头核对）；用 `download_model.py` 从 Zenodo 下载（record 15580543，换文件名和 MD5）；分类头宽度跟随 `dim`。XLS-R-2B 在 fp32 下约需 8.7 GB 内存 |
| 小改 | HuBERT-XL（CNN 无 bias）、Wav2Vec2-Small（group-norm CNN，post-LN） | 编码器里加几个小分支；按它们的结果，没太大必要 |
| 写 transformers 加载器 + 离线文件 | DF_Arena_1B/500M、Spectra-* | 它们用从 Hugging Face 拉来的 config 构建 `Wav2Vec2Model`：要把 config 存到本地，权重手动搬过去（如果许可允许，可以像 voice_clone_models 那样分卷放进私有 GitHub 仓库），`requirements.txt` 加 transformers（或用纯 PyTorch 重写分类头）；它们只看 4.04 秒，长通话要切窗口再平均分数 |
| 把后端移植到我们的 XLS-R 上 | wav2vec2-AASIST、XLSR-SLS、TCM、Nes2Net、SONAR | 前端是 fairseq 命名的 XLS-R 300M（我们的类设成 24 层、宽 1024）；AASIST、SLS、Conformer 或 Res2Net 后端（几百行）要从 MIT/Apache 代码里抄过来；权重在 Google Drive 或 OneDrive |
| 要额外的算子 | XLSR-Mamba、XLSR-MamBo、Fake-Mamba | mamba_ssm 要 CUDA 和 Triton：Mac 上要写纯 PyTorch 的扫描（ASAS-Arena 给 MamBo 做过，分数最多差 2e-4） |
| 纯 PyTorch，很小 | AASIST、AASIST-L、RawGAT-ST、RawNet2、ASVspoof 5 AASIST | 权重就在 GitHub 仓库里；容易但弱；只适合当基本对照 |
| 不同框架 | NII P3（project-NN-Pytorch-scripts，fairseq）、音频大模型 | fairseq 和情绪项目环境里的 FunASR 冲突（hydra-core、omegaconf）；要单独的环境或重写 |

## 7. 注意事项

- **许可。** 非商用：所有 AntiDeepfake checkpoint（CC BY-NC-SA 4.0；MMS-300M/1B 还继承 MMS 的 CC BY-NC 4.0）、DF Arena（自定义非商用；Idiap 提供商用许可）、SONAR-Full 和 AASIST3（CC BY-NC 4.0）、NII SLT 2026 checkpoints。宽松：AASIST、AASIST-L、RawGAT-ST、wav2vec2-AASIST、TCM、XLSR-Mamba、XLSR-MamBo（MIT），Nes2Net（Apache-2.0），NII P3（Zenodo 上 BSD-3-Clause），Spectra（Apache-2.0）。不明：XLSR-SLS 和 ASVspoof 5 基线（仓库没有许可）。
  权重许可宽松不等于训练数据没问题：ASVspoof 2019 是 ODC-By（署名，允许商用），但 Spectra 的数据不明，社区模型也很少说明。生产使用前需要法务审查。
- **排行榜是谁在维护。** ASAS-Arena 由 Kirill Borodin（MTUCI）维护，他所在的 lab260 也发布了排在最前的 Spectra 模型（因为没发表，不参与排名）。DF-Arena 由 DF_Arena_1B 的作者维护。两者都公开分数文件，数字可以复算，但都不检查训练数据是否重叠（ASAS-Arena 每个模型的 `trained_on` 字段都是空的）。
- **输入长度。** AntiDeepfake 对整个文件打分；其他模型几乎都是用 4 秒窗口（64,600 个采样点）训练，评测也常只看 4 秒。处理通话时，要么切成 4 秒窗口再平均（像 AntiDeepfake 的 `max_seconds` 那样），要么接受只看开头几秒。
- **阈值。** EER 处的 P(real) 阈值在不同测试集之间差别很大（AntiDeepfake：0.46–0.99；Spectra-AASIST 默认在 logit 上取 −1.14）。每个模型都要用目标领域的电话数据重新校准阈值。
- **电话信道。** AntiDeepfake 后训练见过 ASVspoof 2021 LA（电话编解码和传输）和 RawBoost；Wav2Vec2-Large-AntiDeepfake 的底座可能用 Switchboard、Fisher 预训练过（未核实）；DF Arena 用了 MUSAN、RIR、RawBoost，但没有电话编解码；P3 论文说 GSM、PSTN 编解码是弱点。这些都不能代替在 8 kHz G.711 音频上实测。
- **来源之间的矛盾。** XLSR-Mamba 的 ADD 2023 R2 在 AntiDeepfake Table II 是 19.36，在 DF-Arena 是 20.15；Spectra-AASIST 的卡片同时写了 Apache-2.0 和 MIT；Wav2Vec2-Large-AntiDeepfake 的底座在 GitHub README 和 Hugging Face 卡片上写的不一样；AASIST3 在 Hugging Face 上的权重不是论文用的那份。
- **Hugging Face 下载量** 是近 30 天的数，说明不了质量；下载最多的检测器（第 4 节）恰恰文档最少。

## 8. 主要来源

- AntiDeepfake：https://arxiv.org/abs/2506.21090 （v4，Table II 零样本 EER，Table III 微调）、https://github.com/nii-yamagishilab/AntiDeepfake （README、`README_grpo.md`、`models/W2V_configs.py`、`hparams/`）、https://zenodo.org/records/15580543 、https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc （14 张模型卡，含各测试集指标）；GRPO：https://arxiv.org/abs/2603.02914
- NII P3：https://arxiv.org/abs/2309.06014 、https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/10-asvspoof-vocoded-trn-ssl 、https://zenodo.org/records/8337778
- NII 其他：https://huggingface.co/nii-yamagishilab/slt2026-checkpoints 、https://huggingface.co/nii-yamagishilab/whisper-deepfake-word-detection 、https://arxiv.org/abs/2602.22658 、https://huggingface.co/nii-yamagishilab/fakemark
- Speech DF Arena：https://arxiv.org/abs/2509.02859 、https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena （2026-10-30 通过 Space 的 `/config` 接口读表）、https://github.com/Speech-Arena/speech_df_arena ；DF Arena 模型：https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1 、https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1 ；RAPTOR：https://arxiv.org/abs/2603.06164
- Speech Anti-Spoofing Arena：https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena （`cache.json`，761 行结果，2026-10-30）、https://github.com/lab260ru/speech_spoof_bench 、https://huggingface.co/SpeechAntiSpoofingBenchmarks 下的镜像模型（如 XLS-R-1B-AntiDeepfake 的提交说明："fixed first-4 s window, NOT the paper's whole-utterance protocol"）
- Spectra / AASIST3：https://huggingface.co/lab260/Spectra-AASIST3 、https://huggingface.co/lab260/Spectra-AASIST 、https://huggingface.co/lab260/spectra_0 、https://huggingface.co/lab260/AASIST3 、https://arxiv.org/abs/2408.17352
- Deepfake-Eval-2024：https://arxiv.org/abs/2503.02857 （Table 3 零样本，Table 4 微调）
- In-the-Wild：https://arxiv.org/abs/2203.16263 ；ASVspoof 2021：https://arxiv.org/abs/2109.00537 ；ASVspoof 5：https://arxiv.org/abs/2502.08857 ；ASVspoof 2019 LA（ODC-By）：https://datashare.ed.ac.uk/handle/10283/3336
- SSL 后端：wav2vec2-AASIST https://arxiv.org/abs/2202.12233 、https://github.com/TakHemlata/SSL_Anti-spoofing ；XLSR-SLS https://doi.org/10.1145/3664647.3681345 、https://github.com/QiShanZhang/SLSforASVspoof-2021-DF ；TCM https://arxiv.org/abs/2406.17376 、https://github.com/ductuantruong/tcm_add ；XLSR-Mamba https://arxiv.org/abs/2411.10027 、https://github.com/swagshaw/XLSR-Mamba ；Nes2Net https://arxiv.org/abs/2504.05657 、https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW ；XLSR-MamBo https://arxiv.org/abs/2601.02944 ；SONAR https://arxiv.org/abs/2511.21325 ；DPDA-GA https://arxiv.org/abs/2509.20682 ；Fake-Mamba https://arxiv.org/abs/2508.09294
- 经典模型：AASIST https://arxiv.org/abs/2110.01200 、https://github.com/clovaai/aasist ；RawGAT-ST https://arxiv.org/abs/2107.12710 、https://github.com/eurecom-asp/RawGAT-ST-antispoofing ；RawNet2 https://arxiv.org/abs/2011.01108 、https://github.com/asvspoof-challenge/2021 ；ASVspoof 5 基线 https://github.com/asvspoof-challenge/asvspoof5 ；Whisper MesoNet https://arxiv.org/abs/2306.01428
- 工具包和其他：WeDefense https://arxiv.org/abs/2601.15240 、https://github.com/zlin0/wedefense ；ALLM4ADD https://arxiv.org/abs/2505.11079 、https://github.com/ucas-hao/qwen_audio_for_add
- SSL 底座许可（Hugging Face）：facebook/wav2vec2-xls-r-300m、-1b、-2b（Apache-2.0）；facebook/mms-300m、mms-1b（CC BY-NC 4.0）；facebook/hubert-xlarge-ll60k、wav2vec2-base、wav2vec2-large-robust（Apache-2.0）
- 查询接口：Hugging Face `https://huggingface.co/api/models/<id>?blobs=true`；Zenodo `https://zenodo.org/api/records/<id>`；safetensors 文件头和 PyTorch zip 的 `data.pkl` 用 HTTP range 请求读取

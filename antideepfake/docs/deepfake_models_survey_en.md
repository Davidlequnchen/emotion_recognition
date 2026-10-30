# Speech Deepfake Detection Open-Source Model Survey

English translation of [deepfake_models_survey.md](deepfake_models_survey.md) (Chinese); the two have the same content. The model this project uses now is described in [xlsr_1b_antideepfake_en.md](xlsr_1b_antideepfake_en.md).

Survey date: 2026-10-30. Goal: survey speech deepfake (anti-spoofing) detectors whose weights can be downloaded, that are counterparts of, or possibly better than, XLS-R-1B-AntiDeepfake, and pick the ones worth evaluating next for the use case of this project: a bank call centre in Singapore, English, 8 kHz G.711 telephone audio, detecting cloned or synthetic caller voices.
Selection criteria: downloadable weights plus code or documentation to run them (non-commercial licences are included but flagged); speech, not singing or environmental sound; any size; the whole NII AntiDeepfake family is included. Models without downloadable weights are listed only in section 5b.

Method: queried the Hugging Face API (file lists and sizes, licences, creation dates, downloads in the last 30 days), the Zenodo API and the GitHub API; read the model cards, papers (arXiv, ISCA, ACL, PMLR) and code; read the result tables of two public leaderboards that rescore many open models on the same test sets: **Speech DF Arena** (Idiap, TalTech and others, 2025; abbreviated DF-Arena) and the **Speech Anti-Spoofing Arena** (SpeechAntiSpoofingBenchmarks, 2026; abbreviated ASAS-Arena), whose full score table was read from the `cache.json` file of the arena's Space (state of 2026-10-30).
Parameter counts: for safetensors, counted from the tensor shapes in the file header (HTTP range request); for .pt, .pth, .ckpt and .bin files (PyTorch zip format), counted from the tensor shapes in `data.pkl`, read out of the archive with HTTP range requests. These are marked "measured". No weights were downloaded. A count covers everything in the file (BatchNorm statistics included); where a file also holds parts used only in pre-training, the inference count is given separately.
Sizes are in decimal MB and GB. Parameter counts: M = million, B = billion; one decimal below 1B, two decimals from 1B up. EER = equal error rate, lower is better; all numbers are in %. "Unverified" marks what could not be confirmed from a primary source.
Reachability from the company computer (GitHub, Zenodo and PyPI reachable; Hugging Face and ModelScope not): ✅ = weights on GitHub or Zenodo; ❌ = only on Hugging Face or ModelScope; ❓ = Google Drive, OneDrive or another site, untested.

## 0. Key points

- **The AntiDeepfake family is still the best open option whose training data is known.** On every leaderboard and paper that includes it, it is in the top group of open models; among the models trained only on ASVspoof 2019 none comes close outside ASVspoof.
- **The easiest upgrade is XLS-R-2B-AntiDeepfake** (same authors, same code, Zenodo): EER lower than XLS-R-1B on 5 of 6 zero-shot test sets (In-the-Wild 1.23 vs 1.35, ADD 2023 4.67 vs 5.39, FakeOrReal-norm 1.64 vs 12.14), but not on Deepfake-Eval-2024 (27.76 vs 26.76). Cost: 2.16B parameters (2.2 times as many), an 8.65 GB file, and roughly 2.2 times the compute (estimate, not measured). In `inference.py` only the width (1920) and FFN (7680) change.
- **No open model solves real-world audio.** On Deepfake-Eval-2024 (deepfakes circulating in 2024) every AntiDeepfake model has an EER of 25.8–34.1% zero-shot; AASIST, RawNet2 and NII's earlier P3 have 43–55%. Only fine-tuning on data from the target domain helps (AntiDeepfake fine-tuned: 8.3–12.1%, weights not released). For the bank, fine-tuning on its own telephone and voice-clone data matters more than picking another model.
- **Models trained only on ASVspoof 2019 LA look excellent on paper but generalise badly**: AASIST, wav2vec2-AASIST, XLSR-SLS, TCM, XLSR-Mamba, Nes2Net, XLSR-MamBo, SONAR reach 0.1–3% on ASVspoof 2019/2021, but 5–11% on In-the-Wild and 20–45% on newer sets. On the 12 ASAS-Arena sets that AntiDeepfake never saw, their mean error is 13–42%, against 2.0–2.4% for XLS-R-1B/2B-AntiDeepfake. They are not upgrades.
- **Two other strong models exist, both with caveats:**
  - DF_Arena_1B_V_1 (Speech DF Arena team, XLS-R 1B + Conformer, 1.15B, measured): the best open model on DF-Arena (In-the-Wild 0.91%); on the 12 ASAS-Arena sets about level with XLS-R-1B-AntiDeepfake (mean 2.83 vs 2.37). Different training data, so a useful second opinion; but a non-commercial licence, Hugging Face only, and it scores only the first 4.04 s.
  - Spectra-AASIST3 / Spectra-AASIST / Spectra-0 (lab260, Apache-2.0, about 0.32B): the lowest errors on ASAS-Arena (mean 1.3–1.6) and the only strong models with a permissive licence; but unpublished, the training data is not disclosed (it may include the test sets), and their authors also run that leaderboard.
- **Licences: everything strong is non-commercial or unverifiable.** AntiDeepfake (CC BY-NC-SA 4.0; the MMS variants also inherit MMS's CC BY-NC 4.0), DF Arena (non-commercial), SONAR-Full and AASIST3 (CC BY-NC 4.0). Commercially usable: AASIST, RawGAT-ST, wav2vec2-AASIST, TCM, XLSR-Mamba (MIT), Nes2Net (Apache-2.0), NII P3 (BSD-3-Clause, Zenodo) and Spectra (Apache-2.0); all weaker or with unknown training data. Production use needs a commercial licence from NII or Idiap, or a model trained in-house on commercially licensed data.
- **Telephone audio**: no open model reports results on 8 kHz G.711 calls. The closest proxies are ASVspoof 2021 LA (its evaluation data went through telephone codecs and transmission) and ASVspoof 5 (codec conditions). AntiDeepfake saw ASVspoof 2021 LA in post-training. The telephone condition of our `evaluate.py` is the real test.
- **Benchmarks are hard to compare**: AntiDeepfake was trained on most official test sets (ASVspoof 2019/2021/5, CodecFake, DFADD, MLAAD, SpoofCeleb, the LibriTTS-based sets…); only In-the-Wild, ADD 2023, DEEP-VOICE, FakeOrReal, Deepfake-Eval-2024 and some ASAS-Arena sets are clean. ASAS-Arena scores only the first 4 s, so it reports 4.24% on In-the-Wild for XLS-R-1B-AntiDeepfake, against 1.35% for the whole utterance in the paper.
- **Worth evaluating next** (the models marked ⭐; effort in section 6):
  1. XLS-R-2B-AntiDeepfake: best overall zero-shot EER in the family; Zenodo; a configuration change in `inference.py`.
  2. The rest of the family on the same code: the NDA versions (trained without RawBoost; XLS-R-2B-NDA has the family's lowest Deepfake-Eval-2024 EER, 25.76%), MMS-1B (same shape as XLS-R-1B; much better on FakeOrReal) and Wav2Vec2-Large (0.32B, a fast option). Averaging the scores of two or three of them is a cheap ensemble.
  3. DF_Arena_1B_V_1: the strongest independent open model; needs transformers and a copy of the model files moved by hand (Hugging Face only); non-commercial.
  4. Spectra-AASIST3 (or Spectra-AASIST): permissive licence and strong leaderboard numbers; worth checking on our own voice clones precisely because its training data is unknown; Hugging Face only.
  - Not marked ⭐: the ASVspoof 2019-only models (section 2, #7–#15) and the classic small models (#16–#21), useful only as weak baselines; HuBERT-XL and Wav2Vec2-Small AntiDeepfake, the weakest of the family.

## 1. The NII AntiDeepfake family

All 14 checkpoints come from the same paper (Ge, Wang, Liu, Yamagishi, "Post-training for Deepfake Speech Detection", ASRU 2025, [arXiv 2506.21090](https://arxiv.org/abs/2506.21090)), were post-trained on the same 74.7k h of data (56.4k h real, 18.3k h fake, 29 sources; details in [xlsr_1b_antideepfake_en.md](xlsr_1b_antideepfake_en.md), section 4.2) and share one head: an average over all frames and Linear(D→2). Each model has a default version (trained with RawBoost) and an NDA version (no data augmentation).
All are on [Zenodo record 15580543](https://zenodo.org/records/15580543) ✅ and on Hugging Face (`nii-yamagishilab/*-anti-deepfake[-nda]`) ❌; checkpoint licence CC BY-NC-SA 4.0, code BSD-3-Clause.

⭐ = recommended to evaluate first; only the names of recommended models are in bold.

### 1a Basic facts and zero-shot EER

EER in % from the paper's Table II, "with RawBoost / NDA". None of these test sets was used in pre-training or post-training. ASAS-Arena: mean error over the 12 sets on which ASAS-Arena scores AntiDeepfake (10 EER + 2 spoof-only sets scored as 1−SRR), first 4 s of each file only.

| # | Model | Params (measured: inference / file) | File (Zenodo) | ADD 2023 T1.2 R2 | DEEP-VOICE | FakeOrReal | FoR-norm | In-the-Wild | Deepfake-Eval-2024 | ASAS-Arena: In-the-Wild / 12-set mean (4 s) | Licence of the SSL base |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XLS-R-1B-AntiDeepfake (used now) | 962.5M / 965.5M | `xlsr_1b.ckpt` 3.86 GB | 5.39 / 6.58 | 2.47 / 2.59 | 5.74 / 3.17 | 12.14 / 10.91 | 1.35 / 1.36 | 26.76 / 26.17 | 4.24 / 2.37 (NDA 3.95 / 2.58) | XLS-R: Apache-2.0 |
| 2 | ⭐ **XLS-R-2B-AntiDeepfake** | 2.16B / 2.16B (2,159.3M / 2,162.9M) | `xlsr_2b.ckpt` 8.65 GB | 4.67 / 6.84 | 2.23 / 2.72 | 2.61 / 1.19 | 1.64 / 1.73 | 1.23 / 1.31 | 27.76 / 25.76 | 2.81 / 2.28 (NDA 2.99 / 2.00) | XLS-R: Apache-2.0 |
| 3 | ⭐ **MMS-1B-AntiDeepfake** | 962.5M / 965.5M | `mms_1b.ckpt` 3.86 GB | 9.06 / 9.46 | 2.47 / 2.35 | 1.23 / 0.88 | 1.73 / 1.10 | 1.82 / 1.87 | 27.70 / 27.54 | 3.30 / 3.82 (NDA 3.83 / 2.98) | MMS: CC BY-NC 4.0 |
| 4 | MMS-300M-AntiDeepfake | 315.4M / 317.4M | `mms_300m.ckpt` 1.27 GB | 7.93 / 11.22 | 2.35 / 3.06 | 1.40 / 0.48 | 5.92 / 2.70 | 2.90 / 2.00 | 32.84 / 31.39 | 4.83 / 4.31 (NDA 3.77 / 4.24) | MMS: CC BY-NC 4.0 |
| 5 | ⭐ **Wav2Vec2-Large-AntiDeepfake** | 315.4M / 317.4M | `w2v_large.ckpt` 1.27 GB | 13.25 / 12.67 | 4.44 / 5.04 | 0.67 / 0.80 | 0.97 / 1.45 | 1.91 / 2.25 | 33.36 / 30.06 | 2.89 / 3.68 (NDA 2.94 / 3.44) | wav2vec 2.0 Large: Apache-2.0 |
| 6 | HuBERT-XL-AntiDeepfake | 962.5M / 964.3M | `hubert_xl.ckpt` 3.86 GB | 18.91 / 35.34 | 5.68 / 14.84 | 2.48 / 3.67 | 3.18 / 15.56 | 5.23 / 17.99 | 34.10 / 47.73 | 3.46 / 5.23 (NDA 10.93 / 14.92) | HuBERT: Apache-2.0 |
| 7 | Wav2Vec2-Small-AntiDeepfake | 94.4M / 95.0M | `w2v_small.ckpt` 0.38 GB | 13.02 / 19.41 | 9.79 / 16.20 | 22.03 / 1.06 | 17.89 / 6.48 | 4.23 / 4.65 | 33.33 / 31.97 | 5.31 / 6.26 (NDA 4.95 / 6.23) | wav2vec 2.0 Base: Apache-2.0 |

- Every NDA file has the same size as its default version (`*_nda.ckpt` on Zenodo; MD5s are published there).
- The "file" count includes the pre-training-only parts (quantizer, `project_q`, `final_proj`, `mask_emb`; for HuBERT `label_embs_concat`), 0.7M–3.7M parameters, which inference drops.
- Fine-tuned on the Deepfake-Eval-2024 training set (not released), 4 s segments: XLS-R-1B 11.85, MMS-1B 12.11, XLS-R-2B 12.14, MMS-300M 17.15, W2V-Large 19.56; with 50 s segments 8.29–11.37 (GitHub README).
- NII has since published fine-tuning by reinforcement learning (GRPO; Wang, Ge, Yamagishi, Interspeech 2026, [arXiv 2603.02914](https://arxiv.org/abs/2603.02914)): code and configurations are in the repository (`README_grpo.md`), but no fine-tuned weights were found.

### 1b Architecture, training data and output

| # | Model | Architecture (fairseq configuration in `models/W2V_configs.py`) | SSL pre-training data | Output | Input audio | Notes |
|---|---|---|---|---|---|---|
| 1 | XLS-R-1B | wav2vec 2.0: 7-layer CNN (layer norm) + 48 Transformer layers, width 1280, FFN 5120, 16 heads, pre-LN; mean pooling + Linear(1280→2) | XLS-R: 436k h, 128 languages | 2 logits [fake, real]; softmax index 1 = P(real); default threshold 0.5 | 16 kHz mono, any length, whole-waveform normalisation (zero mean, unit variance); post-training cropped files over 13 s to 10–13 s | described in detail in [xlsr_1b_antideepfake_en.md](xlsr_1b_antideepfake_en.md) |
| 2 | ⭐ **XLS-R-2B** | as #1 but width 1920, FFN 7680 (48 layers, 16 heads) | as #1 | as #1 | as #1 | the paper: post-training updated it on just over 90% of the data once (the others saw most of it twice); maximum batch 50 s instead of 100 s |
| 3 | ⭐ **MMS-1B** | exactly the shape of #1 (same tensor names and sizes) | MMS: about 500k h, over 1,400 languages (MMS paper; unverified here) | as #1 | as #1 | the MMS base is CC BY-NC 4.0 |
| 4 | MMS-300M | 24 layers, width 1024, FFN 4096, 16 heads, pre-LN, layer-norm CNN | as #3 | as #1 | as #1 | — |
| 5 | ⭐ **Wav2Vec2-Large** | same shape as #4 | the README says wav2vec 2.0 Large "LV-60 + CV + SWBD + FSH" (i.e. large-robust, which includes telephone speech: Switchboard and Fisher); the Hugging Face card names `wav2vec2-large-960h-lv60-self` instead (mismatch, unverified) | as #1 | as #1 | if it really is large-robust, it is the only family member pre-trained on telephone speech |
| 6 | HuBERT-XL | HuBERT: 48 layers, width 1280, FFN 5120; CNN without bias | Libri-Light 60k h (English) | as #1 | as #1 | trained with learning rate 5e-6 instead of 1e-7; the NDA version is much worse |
| 7 | Wav2Vec2-Small | wav2vec 2.0 Base: 12 layers, width 768, group-norm CNN, post-LN | LibriSpeech 960 h | as #1 | as #1 | different layer layout from #1 (needs more code) |

## 2. Other detectors with downloadable weights

Table 2a has the basic facts and the reported performance, table 2b the architecture, data, output and input audio; the # numbers match. "Paper" numbers come from the authors; "DF-Arena" and "ASAS-Arena" numbers were rescored by the leaderboards; the three sources use different protocols and are not interchangeable (section 3). ASAS-Arena 12-set mean: the same 12 sets as in table 1a, so it compares directly with AntiDeepfake at 4 s. A dash means not reported.

### 2a Basic facts and reported performance

| # | Model | Organisation | Year | Reported performance (EER %) | Paper | Download | Size | Licence | How to run |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ⭐ **DF_Arena_1B_V_1** | Speech DF Arena team (Idiap, TalTech, MBZUAI) | 2025-10 | DF-Arena: pooled 9.52 (best open model; commercial systems 1.59–2.32), In-the-Wild 0.91, ASVspoof 2021 LA 4.66, 2021 DF 1.75, FakeOrReal 2.92, ADD 2023 R2 11.54, SONAR 1.09; ASAS-Arena 12-set mean 2.83 | [arXiv 2603.06164](https://arxiv.org/abs/2603.06164) (RAPTOR) | [HF](https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1) ❌ | 1.15B (measured); `pytorch_model.bin` 4.59 GB | custom non-commercial licence (parts MIT); commercial licence on request | transformers `pipeline("antispoofing", trust_remote_code=True)` |
| 2 | DF_Arena_500M_V_1 | as #1 | 2025-11 | DF-Arena: pooled 10.88, In-the-Wild 1.76, ADD 2023 R2 12.30; ASAS-Arena 12-set mean 5.38 | as #1 | [HF](https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1) ❌ | 436.2M (measured); 1.75 GB | as #1 | as #1 |
| 3 | ⭐ **Spectra-AASIST3** | lab260 (MTUCI, Russia) | 2026-04 | ASAS-Arena: In-the-Wild 1.20, ASVspoof 2021 LA 4.38, 2021 DF 4.30, ASVspoof 5 15.09, 12-set mean 1.34; card: In-the-Wild 0.96 | none (unpublished) | [HF](https://huggingface.co/lab260/Spectra-AASIST3) ❌ | 319.1M (measured); 1.28 GB (+ ONNX 1.28 GB) | Apache-2.0 | own PyTorch code in the repository (`PyTorchModelHubMixin`) |
| 4 | Spectra-AASIST | as #3 | 2026-04 | ASAS-Arena: In-the-Wild 1.46, 2021 LA 5.25, 2021 DF 2.52, ASVspoof 5 14.22, 12-set mean 1.38 | none | [HF](https://huggingface.co/lab260/Spectra-AASIST) ❌ | 316.0M (measured); 1.26 GB (+ ONNX) | Apache-2.0 in the header, "MIT" in the text | as #3 |
| 5 | Spectra-0 | as #3 | 2026-01 | ASAS-Arena: In-the-Wild 1.10, ASVspoof 5 14.82, 12-set mean 1.60 | none | [HF](https://huggingface.co/lab260/spectra_0) ❌ | 318.3M (measured); 1.27 GB | Apache-2.0 | as #3 |
| 6 | NII P3 (vocoded-data SSL) | NII Yamagishi Lab | 2023-09 | paper: In-the-Wild 6.10, ASVspoof 2019 LA 1.91, 2021 LA 15.92, 2021 DF 5.67, WaveFake 1.30; Deepfake-Eval-2024 43.00 (AntiDeepfake Table II), AUC 0.58 (Deepfake-Eval-2024 paper) | [arXiv 2309.06014](https://arxiv.org/abs/2309.06014) (ICASSP 2024) | [Zenodo 8337778](https://zenodo.org/records/8337778) ✅ | `project10-cm-ssl-model-ID-P3.tar` 1.33 GB; parameters unverified (one XLS-R-300M-sized SSL) | BSD-3-Clause (LICENSE in the Zenodo record); code BSD-3-Clause | NII project-NN-Pytorch-scripts (fairseq) |
| 7 | wav2vec2-AASIST (SSL_Anti-spoofing) | EURECOM, NII and others (Tak et al.) | 2022-02 | paper: ASVspoof 2021 LA 0.82, 2021 DF 2.85; In-the-Wild 10.46 (Nes2Net paper), 11.20 (DF-Arena); DF-Arena pooled 19.61; ASAS-Arena 12-set mean 26.90 | [arXiv 2202.12233](https://arxiv.org/abs/2202.12233) (Odyssey 2022) | [Google Drive](https://drive.google.com/drive/folders/1c4ywztEVlYVijfwbGLl9OEa1SNtFKppB) ❓; HF mirror ❌ | 317.8M (measured); 1.27 GB | code MIT; weights carry no separate licence | [GitHub](https://github.com/TakHemlata/SSL_Anti-spoofing) (fairseq) |
| 8 | XLSR-SLS | Zhang, Wen, Hu | 2024-10 | paper: 2021 LA 2.87, 2021 DF 1.92, In-the-Wild 7.46; DF-Arena pooled 16.08 (best ASVspoof 2019-only model there); ASAS-Arena 12-set mean 19.71 | ACM MM 2024 ([DOI](https://doi.org/10.1145/3664647.3681345)) | Google Drive ❓; HF mirror ❌ | 340.8M (measured); 1.36 GB | the GitHub repository has no licence (the HF mirror says MIT): unclear | [GitHub](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF) (fairseq) |
| 9 | TCM (XLSR-Conformer + TCM) | NTU Singapore and others (Truong et al.) | 2024-06 | paper: 2021 LA 1.03, 2021 DF 2.06, In-the-Wild 7.79; DF-Arena pooled 16.69 | [arXiv 2406.17376](https://arxiv.org/abs/2406.17376) (Interspeech 2024) | NTU OneDrive ❓ | about 319M (DF-Arena; not measured) | MIT | [GitHub](https://github.com/ductuantruong/tcm_add) (fairseq) |
| 10 | XLSR-Mamba | Y. Xiao, R. K. Das | 2024-11 | paper: 2021 LA 0.93, 2021 DF 1.88, In-the-Wild 6.71; DF-Arena pooled 20.59 (average 14.65) | [arXiv 2411.10027](https://arxiv.org/abs/2411.10027) (IEEE SPL 2025) | [HF LA](https://huggingface.co/AustinXiao/XLSR-Mamba-LA), [HF DF](https://huggingface.co/AustinXiao/XLSR-Mamba-DF) ❌ | 319.3M (measured); 1.28 GB each | MIT | [GitHub](https://github.com/swagshaw/XLSR-Mamba) (fairseq + mamba_ssm) |
| 11 | Nes2Net-X | NUS, NTU (Liu et al.) | 2025-04 | paper: 2021 LA 1.73, 2021 DF 1.65 (another checkpoint 1.49), In-the-Wild 5.15 (a checkpoint selected with validation augmentation); DF-Arena: In-the-Wild 7.75, pooled 17.37; ASAS-Arena 12-set mean 20.51 | [arXiv 2504.05657](https://arxiv.org/abs/2504.05657) (IEEE TIFS) | Google Drive ❓; HF mirror ❌ | 317.9M (measured; back end 0.51M); 1.27 GB | Apache-2.0 ("friendly to commercial use") | [GitHub](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW) (fairseq) |
| 12 | XLSR-MamBo | Ng et al. | 2026-01 | ASAS-Arena: In-the-Wild 5.47, 12-set mean 13.33 (best ASVspoof 2019-only model there) | [arXiv 2601.02944](https://arxiv.org/abs/2601.02944) (ACL 2026 Findings) | HF mirror ❌; GitHub code | 319.4M (measured); 1.28 GB | MIT | [GitHub](https://github.com/saki-ciallo/XLSR-MamBo) (mamba_ssm) |
| 13 | SONAR-Full | Hidekel, Lifshitz, Cohen, Raviv | 2025-11 | ASAS-Arena: In-the-Wild 7.41, 12-set mean 22.16 | [arXiv 2511.21325](https://arxiv.org/abs/2511.21325) (ICML 2026) | HF mirror ❌; [GitHub](https://github.com/idonithid/SONAR-Audio-DF-Detection) | 641.5M (measured); 2.57 GB | weights CC BY-NC 4.0; code MIT | GitHub code (fairseq) |
| 14 | Other ASVspoof 2019 SSL models on ASAS-Arena | DPDA-GA (Truong et al., [arXiv 2509.20682](https://arxiv.org/abs/2509.20682)), SSL feature gating (ACM MM 2025), Fake-Mamba ([arXiv 2508.09294](https://arxiv.org/abs/2508.09294)), AMSDF (IEEE TIFS 2024) | 2024–2025 | ASAS-Arena In-the-Wild: DPDA-GA-AASIST 5.41, SSL feature gating 5.53, AMSDF 18.44; 12-set mean of SSL feature gating 20.13, AMSDF 30.31 | see the left | HF mirrors ❌ | 318–325M | mostly MIT | their GitHub code |
| 15 | AASIST3 | MTUCI / lab260 | 2024-08 (weights 2025-08) | ASAS-Arena: In-the-Wild 29.72, 12-set mean 25.90; the card says these are not the weights of the paper and that the model is deprecated | [arXiv 2408.17352](https://arxiv.org/abs/2408.17352) | [HF](https://huggingface.co/lab260/AASIST3) ❌ | 321.7M (measured); 1.29 GB | CC BY-NC 4.0 | own code |
| 16 | AASIST | Clova AI, EURECOM and others (Jung et al.) | 2021-10 | paper: ASVspoof 2019 LA 0.83; In-the-Wild 43.01; Deepfake-Eval-2024 55.22 (AntiDeepfake Table II), AUC 0.43 | [arXiv 2110.01200](https://arxiv.org/abs/2110.01200) (ICASSP 2022) | [GitHub](https://github.com/clovaai/aasist) `models/weights/AASIST.pth` ✅ | 0.30M (measured); 1.3 MB | MIT | GitHub code (plain PyTorch) |
| 17 | AASIST-L | as #16 | 2021-10 | paper: ASVspoof 2019 LA 0.99; ASAS-Arena In-the-Wild 44.45 | as #16 | GitHub ✅ | 0.085M; 0.43 MB | MIT | as #16 |
| 18 | RawGAT-ST | EURECOM (Tak et al.) | 2021-07 | ASVspoof 2019 LA 1.06 (paper and DF-Arena); DF-Arena In-the-Wild 52.54 | [arXiv 2107.12710](https://arxiv.org/abs/2107.12710) (ASVspoof 2021 workshop) | [GitHub](https://github.com/eurecom-asp/RawGAT-ST-antispoofing) `Pre_trained_models/` ✅ | 0.44M; 1.9 MB | MIT | GitHub code |
| 19 | RawNet2 (ASVspoof 2021 baseline) | EURECOM, ASVspoof organisers | 2020-11 | DF-Arena In-the-Wild 49.19; Deepfake-Eval-2024 48.20 (AntiDeepfake Table II), AUC 0.53 | [arXiv 2011.01108](https://arxiv.org/abs/2011.01108) (ICASSP 2021) | [asvspoof.org zip](https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip) ❓ | 17.6M (DF-Arena) | code MIT; weights unverified | [GitHub](https://github.com/asvspoof-challenge/2021) |
| 20 | ASVspoof 5 baseline AASIST | ASVspoof 5 organisers | 2024 | not published with the weights (unverified) | ASVspoof 5 | [GitHub](https://github.com/asvspoof-challenge/asvspoof5) `Baseline-AASIST/models/weights/AASIST/best.pth` ✅ | about 0.3M; 1.3 MB | repository without a licence: unclear | GitHub code |
| 21 | Whisper + MFCC MesoNet | Kawa et al. | 2023-06 | DF-Arena: In-the-Wild 26.72, 2021 DF 2.11 (its training source) | [arXiv 2306.01428](https://arxiv.org/abs/2306.01428) (Interspeech 2023) | Google Drive ❓; HF mirror ❌ | 7.66M (ASAS-Arena) | MIT | [GitHub](https://github.com/piotrkawa/deepfake-whisper-features) |

LFCC-LCNN (the other ASVspoof 2021 baseline) is not listed: its repository only has an empty `__pretrained` folder, and where its scripts fetch pre-trained models from is unverified.

### 2b Architecture, training data and output

| # | Model | Architecture | Training data | Output | Input audio | Notes |
|---|---|---|---|---|---|---|
| 1 | ⭐ **DF_Arena_1B_V_1** | XLS-R 1B (transformers `Wav2Vec2Model`, all 49 hidden states) → attention pooling per layer, a sigmoid gate per layer, weighted sum → BatchNorm + SELU → 4 Conformer blocks (4 heads, kernel 31) with a class token → Linear(1280→2); 1.15B parameters (XLS-R 0.96B, Conformer head 0.19B) | ASVspoof 2019, ASVspoof 2024 (= ASVspoof 5), Codecfake, LibriSeVoc, DFADD, CtrSVDD (singing), SpoofCeleb, MLAAD, EnvSDD (environmental sound); MUSAN + RIR offline, RawBoost online; model selection on the FakeOrReal dev set (RAPTOR paper) | logits [spoof, bonafide] (id 0 = spoof, 1 = bonafide); the pipeline returns the softmax score of the predicted label and both scores | ⚠️ **only the first 64,600 samples (4.04 s) at 16 kHz**; shorter clips are repeated to that length; no normalisation; one clip per call | the code calls `Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-1b")` when it builds the model, which needs Hugging Face (or a local copy of the config) |
| 2 | DF_Arena_500M_V_1 | as #1 with XLS-R 300M (width 1024); 436.2M parameters | as #1 | as #1 | as #1 | despite the name, 0.44B |
| 3 | ⭐ **Spectra-AASIST3** | XLS-R 300M (transformers) → Linear(1024→128) + SELU → AASIST with Kolmogorov-Arnold (KAN) layers, 4 inference branches; 319.1M parameters | not disclosed | 2 logits [spoof, bonafide]; score = bonafide logit | 16 kHz; pre-emphasis 0.97; the card's code takes a random 64,600-sample (4.04 s) crop, ASAS-Arena the first 4.04 s | fetches the `facebook/wav2vec2-xls-r-300m` config from Hugging Face at start-up |
| 4 | Spectra-AASIST | XLS-R 300M → MLP → AASIST; 316.0M | not disclosed | as #3; default threshold −1.140625 on the bonafide logit | as #3 | as #3 |
| 5 | Spectra-0 | XLS-R 300M → MLP → ECAPA-TDNN; 318.3M | not disclosed | as #3 | as #3 | as #3 |
| 6 | NII P3 | an XLS-R-300M-sized SSL distilled from XLS-R and from an SSL continually pre-trained on vocoded VoxCeleb2, + a light back end | SSL: about 9k h of VoxCeleb2 re-synthesised by 4 vocoders; CM fine-tuning: vocoded ASVspoof 2019 LA training set | CM score (column 2 of the score file), higher = more bona fide (unverified) | 16 kHz; length handling unverified | the paper says GSM and PSTN codecs in ASVspoof 2021 remain hard for it; a variant fine-tuned on vocoded VoxCeleb2 (P3-c) halved the EER on the GSM/PSTN part of LA21hid (weights unverified) |
| 7 | wav2vec2-AASIST | XLS-R 300M (fine-tuned) → Linear(1024→128) → AASIST graph attention; 317.8M | ASVspoof 2019 LA training set + RawBoost (separate LA and DF checkpoints) | 2 logits, index 1 = bona fide | 64,600 samples (4.04 s) in training; full length or 4 s at test, depending on the evaluation | the base model of most later SSL back ends (#8–#14) |
| 8 | XLSR-SLS | XLS-R 300M, all 24 layers weighted by a "sensitive layer selection" module → classifier; 340.8M | as #7 | higher = more bona fide | as #7 | — |
| 9 | TCM | XLS-R 300M → projection to 144 → 4 Conformer blocks with temporal-channel head tokens; about 319M | as #7 | higher = more bona fide; example threshold −3.73 | 4 s in training | NTU Singapore |
| 10 | XLSR-Mamba | XLS-R 300M → 144 → dual-column bidirectional Mamba (6 + 6 blocks); 319.3M | as #7 (RawBoost; separate LA and DF checkpoints) | higher = more bona fide | 4 s in training | mamba_ssm needs CUDA (Triton); on a Mac the scan must be rewritten in plain PyTorch |
| 11 | Nes2Net-X | XLS-R 300M (or WavLM) → nested Res2Net back end of 0.51M; 317.9M | as #7 | higher = more bona fide | 4 s in training | NUS Singapore; there are also WavLM versions and an ASVspoof 5 branch |
| 12 | XLSR-MamBo | XLS-R 300M → 128 → 5 × [3 Hydra (bidirectional Mamba2) + attention]; 319.4M | ASVspoof 2019 LA, no RawBoost | logit difference, higher = bona fide | ASAS-Arena: first 66,800 samples | — |
| 13 | SONAR-Full | two XLS-R 300M encoders (waveform; high-frequency residual from learnable SRM filters) + cross-attention + AASIST; 641.5M | ASVspoof 2019 LA | index 1 = real | first 64,600 samples | — |
| 14 | Others | XLS-R 300M + AASIST/TCM/Mamba/multi-view back ends | ASVspoof 2019 LA (+ RawBoost) | higher = bona fide | 4 s | — |
| 15 | AASIST3 | XLS-R 300M + KAN-AASIST; 321.7M | card: MLAAD, ASVspoof 5, ASVspoof 2019 LA | index 1 = bona fide | 64,600 samples, pre-emphasis | deprecated in favour of #3 |
| 16 | AASIST | sinc convolution on the raw waveform + residual encoder + spectro-temporal graph attention; 0.30M | ASVspoof 2019 LA | index 1 = bona fide | 64,600 samples | — |
| 17 | AASIST-L | a narrower AASIST; 0.085M | as #16 | as #16 | as #16 | — |
| 18 | RawGAT-ST | raw waveform + spectro-temporal graph attention; 0.44M | as #16 | as #16 | as #16 | — |
| 19 | RawNet2 | sinc filters + residual blocks + GRU; 17.6M | ASVspoof 2019 LA | as #16 | about 4 s | — |
| 20 | ASVspoof 5 AASIST | as #16 | ASVspoof 5 training set | as #16 | as #16 | the only small model here trained on ASVspoof 5 (codec conditions) |
| 21 | Whisper + MFCC MesoNet | Whisper tiny.en encoder + MFCC → MesoInception; 7.66M | part of ASVspoof 2021 DF | higher = bona fide | Whisper's fixed 30 s window | — |

## 3. Common test sets: who beats whom

### 3a EER (%) on the test sets most used for comparison

Sources: P = the authors' paper; D = DF-Arena (Speech DF Arena; models built on wav2vec2, WavLM or HuBERT are scored at full length, padded to the longest file in each batch; the others at a fixed 4 s); A = ASAS-Arena (first 4 s of every file); T = AntiDeepfake Table II. "×" = the test set (or its source) was in the model's training data, so the number is not a zero-shot result; "?" = training data unknown.

| Model | In-the-Wild | ASVspoof 2021 LA | ASVspoof 2021 DF | ASVspoof 5 eval | Deepfake-Eval-2024 | ADD 2023 R2 | FakeOrReal |
|---|---|---|---|---|---|---|---|
| XLS-R-1B-AntiDeepfake | 1.35 P / 4.24 A | × | × | × | 26.76 P | 5.39 P | 5.74 P |
| XLS-R-2B-AntiDeepfake | 1.23 P / 2.81 A | × | × | × | 27.76 P | 4.67 P | 2.61 P |
| DF_Arena_1B_V_1 | 0.91 D / 0.91 A | 4.66 D | 1.75 D | 17.25 D × | — | 11.54 D | 2.92 D (its dev set was used for model selection) |
| Spectra-AASIST3 | 1.20 A ? | 4.38 A ? | 4.30 A ? | 15.09 A ? | — | — (R1: 6.64 A) | — |
| NII P3 | 6.10 P | 15.92 P | 5.67 P | — | 43.00 T | — | — |
| wav2vec2-AASIST | 10.46 P / 11.20 D / 11.22 A | 0.82 P / 8.11 A | 2.85 P / 8.32 A | 16.25 D | — | 21.93 D | 7.47 D |
| XLSR-SLS | 7.46 P / 7.46 D | 2.87 P | 1.92 P | 18.76 D | — | 21.10 D | 5.08 D |
| TCM | 7.79 P / 7.79 D | 1.03 P | 2.06 P | 18.85 D | — | 22.74 D | 10.69 D |
| XLSR-Mamba | 6.71 P / 6.71 D | 0.93 P | 1.88 P | 14.40 D | — | 20.15 D (T: 19.36) | 6.71 D |
| Nes2Net-X | 5.15 P / 7.75 D / 8.48 A | 1.73 P | 1.65 P | 22.06 D | — | 18.45 D | 6.32 D |
| AASIST | 43.01 D | 11.46 D | 21.07 D | 35.53 D | 55.22 T | 32.47 D | 21.64 D |
| RawNet2 | 49.19 D | 40.07 D | 40.67 D | 41.22 D | 48.20 T | 64.55 D | 65.68 D |

- ASVspoof 2019 LA and ASVspoof 2021 LA/DF share their bona fide source (VCTK) with the ASVspoof 2019 training set, so for the ASVspoof 2019-trained models (#7–#21) the 2021 columns are "cross-dataset but same family", not truly out of domain.
- The same checkpoint gives very different numbers under different protocols: wav2vec2-AASIST on ASVspoof 2021 LA is 0.82 (paper, full length) and 8.11 (ASAS-Arena, first 4 s); XLS-R-1B-AntiDeepfake on In-the-Wild 1.35 and 4.24. Only compare numbers from the same source.
- The DF-Arena "pooled" EER uses one threshold for all 14 sets and is the hardest metric; the best open model is DF_Arena_1B_V_1 (9.52), the best ASVspoof 2019-only model XLSR-SLS (16.08); commercial systems (Modulate, Resemble, Hiya) score 1.59–2.32. AntiDeepfake is not on DF-Arena.
- Deepfake-Eval-2024 (Chandra et al. 2025): AUC drops by 48% on average for the open audio models (AASIST 0.43, RawNet2 0.53, P3 0.58) compared with their original benchmarks; fine-tuned on 60% of Deepfake-Eval-2024 they reach AUC 0.88–0.92.

### 3b Training and test overlap (what is not a zero-shot result)

| Model | In its training data (do not use as an independent test) |
|---|---|
| AntiDeepfake (all) | ASVspoof 2019 LA (incl. eval), 2021 LA, 2021 DF, ASVspoof 5 (incl. test), CFAD, DECRO, DFADD, Diffuse or Confuse, DiffSSD, DSD, HABLA, MLAAD v5, SpoofCeleb, VoiceMOS 2022, CVoiceFake, WaveFake, LibriTTS and its vocoded and restored versions (so also SONAR, CD-ADD and LibriSeVoc are risky), VoxCeleb2, FLEURS / FLEURS-R, Codecfake, CodecFake, AISHELL-3, CN-Celeb2, MLS. Clean: In-the-Wild, ADD 2023, DEEP-VOICE, FakeOrReal, Deepfake-Eval-2024, and the 12 ASAS-Arena sets of table 1a |
| DF_Arena_1B / 500M | ASVspoof 2019, ASVspoof 5, Codecfake, LibriSeVoc, DFADD, SpoofCeleb, MLAAD (+ CtrSVDD, EnvSDD); FakeOrReal dev for model selection |
| Spectra-* | unknown |
| ASVspoof 2019-only models | ASVspoof 2019 LA (and the VCTK source shared with 2021 LA/DF) |
| AASIST3 (lab260) | MLAAD, ASVspoof 5, ASVspoof 2019 LA |

## 4. Community checkpoints on Hugging Face (a few representative ones)

Much downloaded, but all uploaded by individuals: undocumented or tiny training sets, test sets that are random splits of the same data, and accuracies on the cards that cannot be trusted. None of them is on either leaderboard. All are Hugging Face only ❌ and load with the standard `AutoModelForAudioClassification`. Downloads: last 30 days, from the Hugging Face API on 2026-10-30.

| Model | Year | Downloads | Reported performance | Size (measured) | Licence | Architecture | Training data | Output (label order) | Problems |
|---|---|---|---|---|---|---|---|---|---|
| [mo-thecreator/Deepfake-audio-detection](https://huggingface.co/mo-thecreator/Deepfake-audio-detection) | 2024-05 | 19,904 | accuracy 98.8% on its evaluation set | 94.6M; 0.38 GB | Apache-2.0 | wav2vec2-base + classification head | "None dataset" (not described) | 0 fake, 1 real | data unknown |
| [Gustking/wav2vec2-large-xlsr-deepfake-audio-classification](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification) | 2024-05 | 11,483 | EER 4.01% on a Kaggle copy of the ASVspoof 2019 evaluation subset | 315.7M; 1.26 GB | Apache-2.0 | wav2vec2-large-XLSR-53 (started from a gender-recognition model) + head | not described | 0 real, 1 fake | only an ASVspoof 2019 subset as evidence |
| [Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection) | 2024-05 | 9,785 | not described | 94.6M | Apache-2.0 | wav2vec2-base + head | "audiofolder" | 0 AIVoice, 1 HumanVoice | data unknown |
| [garystafford/wav2vec2-deepfake-voice-detector](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector) | 2025-12 | 8,975 | metrics on 187 test clips | 315.7M | Apache-2.0 | fine-tune of Gustking (upper 12 layers) | 1,866 clips: 14 YouTube videos + ElevenLabs, Amazon Polly, Kokoro, Hume AI, Speechify, Luvvoice | 0 real, 1 fake | tiny data; card: best on 2.5–13 s clips |
| [MelodyMachine/Deepfake-audio-detection-V2](https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2) | 2024-06 | 7,772 | accuracy 99.7% on its evaluation set | 94.6M | Apache-2.0 | fine-tune of motheecreator/Deepfake-audio-detection (wav2vec2-base) | "audiofolder" | 0 fake, 1 real | data unknown |
| [MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection](https://huggingface.co/MattyB95/AST-ASVspoof5-Synthetic-Voice-Detection) | 2024-07 | 359 | card metrics on its split | 86.2M | BSD-3-Clause | Audio Spectrogram Transformer (AudioSet) + head | ASVspoof 5 | 0 Bonafide, 1 Spoof | ASVspoof 5 data only |
| [abhishtagatya/wav2vec2-base-960h-itw-deepfake](https://huggingface.co/abhishtagatya/wav2vec2-base-960h-itw-deepfake) | 2024-01 | 374 | card metrics on its split | 94.6M | Apache-2.0 | wav2vec2-base-960h + head | In-the-Wild | 0 bona-fide, 1 spoof | trained on In-the-Wild, so it cannot be evaluated there |

## 5. Other detectors

### 5a Audio LLMs, word-level detection and toolkits (brief)

| Model | Organisation, year | What it is | Download | Licence | Notes |
|---|---|---|---|---|---|
| AntiDeepfake-AudioLLM checkpoints (SLT 2026) | NII, 2026-10 | four CMs: XLS-R-300M + FC, Whisper-large-v2 + FC, Qwen2-Audio-7B + LoRA with a classification head, and with a generative head | [HF](https://huggingface.co/nii-yamagishilab/slt2026-checkpoints) ❌ ([code](https://github.com/nii-yamagishilab/AntiDeepfake-AudioLLM)) | CC BY-NC 4.0 | trained on ESDD2-CompSpoof-V2 (speech mixed with background events); a study of what CMs listen to, not a general detector |
| Whisper deepfake word detection | NII, 2026-09 | Whisper large-v3 fine-tuned to mark which words are fake (partial fakes) by next-token prediction | [HF](https://huggingface.co/nii-yamagishilab/whisper-deepfake-word-detection) ❌ | card: CC BY 4.0 ("see LICENSE") | Interspeech 2026, [arXiv 2602.22658](https://arxiv.org/abs/2602.22658); for edited or spliced speech |
| WeDefense | Lin Zhang, J. Rohdin, X. Wang (NII), N. Evans (EURECOM) and others, 2026-01 | toolkit (Apache-2.0) for detection and localisation; pre-trained "detection_MHFA_wav2vec2_large" | ModelScope ❌ | toolkit Apache-2.0; weight licence and training data unverified | [arXiv 2601.15240](https://arxiv.org/abs/2601.15240); [GitHub](https://github.com/zlin0/wedefense) |
| FakeMark | NII, 2025-10 | watermarking for attribution, not a detector for unmarked clones | HF ❌ | CC BY 4.0 | not applicable to callers' cloned voices |

### 5b No downloadable weights (or not found)

- Commercial systems on DF-Arena: Modulate VELMA-2, Resemble Detect and DETECT-3B Omni, Hiya, DLMSL SpeakSure, Whispeak, Syntra, MoLEx (A*STAR), DF-Raptor (Idiap). Resemble open-sources other tools (Resemblyzer, Chatterbox, the PerTh watermark), but not its detector.
- DF_Arena_100M_V_0 / V_1: listed as "Closed"; their Hugging Face repositories hold no weight file.
- AntiDeepfake fine-tuned on Deepfake-Eval-2024 and the GRPO fine-tuned models (not released); M1 on ASAS-Arena (score files only).
- ALLM4ADD (Qwen-Audio + LoRA; code on GitHub, no weights found), SpeechFake baselines (ACL 2025; not found), BiCrossMamba-ST (Interspeech 2025; "code and models will be made available", not found).

## 6. Effort to add a model to this project

`scripts/antideepfake/inference.py` re-implements XLS-R in plain PyTorch with fairseq's parameter names (no fairseq); its encoder class already takes `dim`, `ffn`, `heads` and `layers` as arguments, and only the head is fixed at Linear(1280→2). `evaluate.py` scores groups of real and fake files clean and through the G.711 telephone channel; any model that returns P(real) per file can reuse it.

| Effort | Models | What it takes |
|---|---|---|
| Configuration only | XLS-R-2B (width 1920, FFN 7680), MMS-1B (identical to XLS-R-1B), Wav2Vec2-Large and MMS-300M (24 layers, 1024, FFN 4096), all NDA versions | the checkpoints use the same tensor names (`m_ssl.model.*`, `proj_fc.*`; checked in the headers); download from Zenodo with `download_model.py` (record 15580543, other file name and MD5); make the head width follow `dim`. XLS-R-2B needs about 8.7 GB of memory in fp32 |
| Small change | HuBERT-XL (CNN without bias), Wav2Vec2-Small (group-norm CNN, post-LN) | small branches in the encoder; little reason to do it given their results |
| A transformers loader + offline files | DF_Arena_1B/500M, Spectra-* | they build `Wav2Vec2Model` from a config fetched from Hugging Face: save that config locally, copy the weights by hand (e.g. as split parts in a private GitHub repository, as for voice_clone_models, if the licence allows it), add transformers to `requirements.txt` (or re-implement the head in plain PyTorch); they score only 4.04 s, so long calls must be split into windows and the scores averaged |
| Port a back end onto our XLS-R | wav2vec2-AASIST, XLSR-SLS, TCM, Nes2Net, SONAR | the front end is XLS-R 300M with fairseq names (our class with 24 layers, width 1024); the AASIST, SLS, Conformer or Res2Net back end (a few hundred lines) must be copied from the MIT/Apache code; weights on Google Drive or OneDrive |
| Extra kernels | XLSR-Mamba, XLSR-MamBo, Fake-Mamba | mamba_ssm needs CUDA and Triton: a plain-PyTorch scan for the Mac (ASAS-Arena did this for MamBo, max score difference 2e-4) |
| Plain PyTorch, tiny | AASIST, AASIST-L, RawGAT-ST, RawNet2, ASVspoof 5 AASIST | weights in the GitHub repositories; easy but weak; useful only as a sanity baseline |
| Different framework | NII P3 (project-NN-Pytorch-scripts, fairseq), audio LLMs | fairseq conflicts with FunASR in the emotion environment (hydra-core, omegaconf); would need its own environment or a re-implementation |

## 7. Notes

- **Licences.** Non-commercial: all AntiDeepfake checkpoints (CC BY-NC-SA 4.0; MMS-300M/1B also inherit the MMS licence CC BY-NC 4.0), DF Arena (custom non-commercial; Idiap offers a commercial licence), SONAR-Full and AASIST3 (CC BY-NC 4.0), the NII SLT 2026 checkpoints. Permissive: AASIST, AASIST-L, RawGAT-ST, wav2vec2-AASIST, TCM, XLSR-Mamba, XLSR-MamBo (MIT), Nes2Net (Apache-2.0), NII P3 (BSD-3-Clause on Zenodo), Spectra (Apache-2.0). Unclear: XLSR-SLS and the ASVspoof 5 baseline (repositories without a licence).
  A permissive weight licence does not settle the training data: ASVspoof 2019 is under ODC-By (attribution, commercial use allowed), but Spectra's data are unknown and community uploads rarely document theirs. Legal review is needed before any production use.
- **Who runs the leaderboard.** ASAS-Arena is maintained by Kirill Borodin (MTUCI), whose lab260 also publishes the Spectra models that top it (unranked, as unpublished). DF-Arena is run by the authors of DF_Arena_1B. Both publish score files, so the numbers can be recomputed, but neither checks training data for overlap (ASAS-Arena's `trained_on` field is empty for every model).
- **Input length.** AntiDeepfake scores the whole file; almost every other model was trained on, and is often scored on, a 4 s window (64,600 samples). For calls, either split into 4 s windows and average (as `max_seconds` does for AntiDeepfake) or accept that only the first seconds count.
- **Thresholds.** The P(real) threshold at the EER varies between test sets (AntiDeepfake: 0.46–0.99; Spectra-AASIST's default −1.14 on a logit). Every model needs a threshold calibrated on telephone data from the target domain.
- **Telephone channel.** AntiDeepfake saw ASVspoof 2021 LA (telephone codecs and transmission) and RawBoost in post-training; Wav2Vec2-Large-AntiDeepfake may have a base pre-trained on Switchboard and Fisher (unverified); DF Arena used MUSAN, RIR and RawBoost but no telephone codec; the P3 paper reports GSM and PSTN codecs as a weak point. None of this replaces measuring on 8 kHz G.711 audio.
- **Discrepancies between sources.** ADD 2023 R2 for XLSR-Mamba is 19.36 in AntiDeepfake's Table II and 20.15 on DF-Arena; Spectra-AASIST's card says both Apache-2.0 and MIT; Wav2Vec2-Large-AntiDeepfake's base differs between the GitHub README and the Hugging Face card; AASIST3's Hugging Face weights are not those of its paper.
- **Hugging Face downloads** are counts for the last 30 days and say nothing about quality; the most downloaded detectors (section 4) are the least documented.

## 8. Main sources

- AntiDeepfake: https://arxiv.org/abs/2506.21090 (v4, Table II zero-shot EER, Table III fine-tuning), https://github.com/nii-yamagishilab/AntiDeepfake (README, `README_grpo.md`, `models/W2V_configs.py`, `hparams/`), https://zenodo.org/records/15580543 , https://huggingface.co/collections/nii-yamagishilab/antideepfake-685a1788fc514998e841cdfc (14 model cards with per-test-set metrics); GRPO: https://arxiv.org/abs/2603.02914
- NII P3: https://arxiv.org/abs/2309.06014 , https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/10-asvspoof-vocoded-trn-ssl , https://zenodo.org/records/8337778
- NII other: https://huggingface.co/nii-yamagishilab/slt2026-checkpoints , https://huggingface.co/nii-yamagishilab/whisper-deepfake-word-detection , https://arxiv.org/abs/2602.22658 , https://huggingface.co/nii-yamagishilab/fakemark
- Speech DF Arena: https://arxiv.org/abs/2509.02859 , https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena (tables read through the Space's `/config` endpoint on 2026-10-30), https://github.com/Speech-Arena/speech_df_arena ; DF Arena models: https://huggingface.co/Speech-Arena-2025/DF_Arena_1B_V_1 , https://huggingface.co/Speech-Arena-2025/DF_Arena_500M_V_1 ; RAPTOR: https://arxiv.org/abs/2603.06164
- Speech Anti-Spoofing Arena: https://huggingface.co/spaces/SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena (`cache.json`, 761 result rows, 2026-10-30), https://github.com/lab260ru/speech_spoof_bench , mirrored models under https://huggingface.co/SpeechAntiSpoofingBenchmarks (e.g. the XLS-R-1B-AntiDeepfake submission: "fixed first-4 s window, NOT the paper's whole-utterance protocol")
- Spectra / AASIST3: https://huggingface.co/lab260/Spectra-AASIST3 , https://huggingface.co/lab260/Spectra-AASIST , https://huggingface.co/lab260/spectra_0 , https://huggingface.co/lab260/AASIST3 , https://arxiv.org/abs/2408.17352
- Deepfake-Eval-2024: https://arxiv.org/abs/2503.02857 (Table 3 zero-shot, Table 4 fine-tuned)
- In-the-Wild: https://arxiv.org/abs/2203.16263 ; ASVspoof 2021: https://arxiv.org/abs/2109.00537 ; ASVspoof 5: https://arxiv.org/abs/2502.08857 ; ASVspoof 2019 LA (ODC-By): https://datashare.ed.ac.uk/handle/10283/3336
- SSL back ends: wav2vec2-AASIST https://arxiv.org/abs/2202.12233 , https://github.com/TakHemlata/SSL_Anti-spoofing ; XLSR-SLS https://doi.org/10.1145/3664647.3681345 , https://github.com/QiShanZhang/SLSforASVspoof-2021-DF ; TCM https://arxiv.org/abs/2406.17376 , https://github.com/ductuantruong/tcm_add ; XLSR-Mamba https://arxiv.org/abs/2411.10027 , https://github.com/swagshaw/XLSR-Mamba ; Nes2Net https://arxiv.org/abs/2504.05657 , https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW ; XLSR-MamBo https://arxiv.org/abs/2601.02944 ; SONAR https://arxiv.org/abs/2511.21325 ; DPDA-GA https://arxiv.org/abs/2509.20682 ; Fake-Mamba https://arxiv.org/abs/2508.09294
- Classic: AASIST https://arxiv.org/abs/2110.01200 , https://github.com/clovaai/aasist ; RawGAT-ST https://arxiv.org/abs/2107.12710 , https://github.com/eurecom-asp/RawGAT-ST-antispoofing ; RawNet2 https://arxiv.org/abs/2011.01108 , https://github.com/asvspoof-challenge/2021 ; ASVspoof 5 baselines https://github.com/asvspoof-challenge/asvspoof5 ; Whisper MesoNet https://arxiv.org/abs/2306.01428
- Toolkits and others: WeDefense https://arxiv.org/abs/2601.15240 , https://github.com/zlin0/wedefense ; ALLM4ADD https://arxiv.org/abs/2505.11079 , https://github.com/ucas-hao/qwen_audio_for_add
- SSL base licences (Hugging Face): facebook/wav2vec2-xls-r-300m, -1b, -2b (Apache-2.0); facebook/mms-300m, mms-1b (CC BY-NC 4.0); facebook/hubert-xlarge-ll60k, wav2vec2-base, wav2vec2-large-robust (Apache-2.0)
- Query endpoints: Hugging Face `https://huggingface.co/api/models/<id>?blobs=true`; Zenodo `https://zenodo.org/api/records/<id>`; safetensors headers and PyTorch zip `data.pkl` read with HTTP range requests

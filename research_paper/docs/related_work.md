# Related Work

Compiled 2026-11-03 from web research (arXiv, ACL Anthology, ISCA, IEEE). Every row has a link.
"unverified" = could not confirm from the paper page itself (venue or number taken from a search snippet or a summary only).
Caution: one automatic page summary (VocalAffectBench PDF) invented details; that paper was then read page by page, so its numbers below are checked. Other numbers come from arXiv abstract pages or HTML full text.

---

## 0. Closest prior work to our own result (read first)

| Paper | Venue / year | Finding | Link |
|---|---|---|---|
| **VocalAffectBench** (Debaupte et al., Besimple AI) | arXiv Aug/Sep 2026 (v2) | 280 acted English clips, 52 speakers, 7 classes, scripts written to fit several emotions (lexically ambiguous). Raw audio only. gemini-3.5-flash best at **44.3%**; **gpt-realtime-2 lowest at 27.9%** (chance 14.3%); mean of 6 baselines 35.1%. Neutral predicted 761/1,679 times (**45.3% of outputs**); gpt-realtime outputs **69.6% neutral**, neutral recall 87.5%. Fearful/surprised recall 15%. Single reviewer label, no inter-rater data, no dedicated open SER models, no repeat runs. | https://arxiv.org/abs/2608.28932 |
| LISTEN (Chen et al.) | arXiv 2510.10444; EACL 2026 (per our notes; venue not re-checked = unverified) | 6 LALMs (Qwen2.5-Omni-7B, Qwen3-Omni-30B, Baichuan-Omni-1.5, Gemini 2.5 Flash/Pro, ...). Neutral-text, audio-only accuracy: Gemini 2.5 Pro 34.9%, Qwen2.5-Omni 34.0%, Gemini 2.5 Flash 25.6%, Baichuan 16.5%. Models default to "neutral" when lexical cues are neutral; small gains under cue agreement; fail under cue conflict. | https://arxiv.org/abs/2510.10444 |
| VoxEmo (Zhang, Chou, Narayanan, Hain) | arXiv Mar 2026, submitted to Interspeech 2026 | 35 corpora, 15 languages; Qwen2-Audio and Audio Flamingo 3. CREMA-D zero-shot Q2A 61.9 Macro-F1 vs 71.0 supervised. Prompt choice changes Macro-F1 by >20 points on 11 datasets; +acoustic-caption prompt makes Q2A parse failures reach 93.2%. Soft-label (KLD/JSD) protocol; 5-prompt ensemble "emulates annotator disagreement"; zero-shot LLMs trail on hard labels but align with human label distributions. No closed models. | https://arxiv.org/abs/2603.08936 |
| EmoBox (Ma et al.) | Interspeech 2024 | 32 datasets, 14 languages, intra- and cross-corpus splits, 10 SSL models. Cross-corpus test sets balanced using emotion2vec. | https://arxiv.org/abs/2406.07162 |

Implication: "closed audio LLMs, GPT Realtime worst and neutral-biased, on lexically neutral acted speech" already has a direct precedent (VocalAffectBench, one in-house corpus). Our contribution must rest on: 6 public corpora, head-to-head with 5 open dedicated SER checkpoints, per-corpus analysis, and new analyses (below).

---

## A. Limitations of audio LLMs / LALMs for emotion recognition

### A1. Lexical over paralinguistic reliance

| Paper | Venue / year | Key finding | Link |
|---|---|---|---|
| LISTEN | EACL 2026 (unverified) | See section 0; "transcribe rather than listen". | https://arxiv.org/abs/2510.10444 |
| Corrêa et al., EMIS dataset | Submitted to ICASSP 2026 | On emotionally incongruent synthetic speech, SLMs "rely predominantly on textual semantics rather than speech emotion". | https://arxiv.org/abs/2510.25054 |
| Chen, Ho, Topaz, Kostic, Hirschberg, CREMA-ASIS | EMNLP 2026 Findings | LALMs rarely predict acoustic-semantic incongruity and are "predominantly influenced by semantic information"; fine-tuning helps. Built on CREMA-D. | https://arxiv.org/abs/2608.28966 |
| VoxParadox (Pang, Chaubey, Soleymani) | ICML 2026 | 2,000 items, 10 paralinguistic tasks with misleading transcripts. Audio Flamingo 3: 17.40% -> 65.20% after their PCLM + DPO; Step-Audio-R1 54.51% on MMSU paralinguistic subset but 17.45% on VoxParadox. | https://arxiv.org/abs/2605.27772 |
| Jing & Laban, incongruence score | arXiv Oct 2026 | Text-audio incongruence raises error odds "whenever audio is present, in every corpus" (eNTERFACE'05, MELD, CMU-MOSEI); supervised classifiers on the same features show the opposite. | https://arxiv.org/abs/2610.40883 |
| Koduru et al., "Heard but Not Heeded" | arXiv Sep 2026 | 4 open ALMs on Expresso: speaking style is strongly encoded in the late audio encoder but "consistently degraded before reaching the output"; gap between what is encoded and what is used. | https://arxiv.org/abs/2609.00727 |
| Tsangko, Triantafyllopoulos, Schuller | arXiv Jun 2026 | Adding eGeMAPS-based acoustic concept tokens helps on FAU-Aibo/IEMOCAP; shuffled tokens hurt; predictions do not collapse, so models are partly grounded in audio. | https://arxiv.org/abs/2606.07309 |

### A2. Output stochasticity / inconsistency across runs (the paper the user recalls)

No paper found that measures plain run-to-run label flips of a closed audio LLM on SER with a fixed prompt. The closest matches:

| Paper | Venue / year | Key finding | Link |
|---|---|---|---|
| **VoxEmo** (most likely the recalled paper) | Interspeech 2026 submission | Abstract: moving to open generation "introduces zero-shot stochasticity, making evaluation highly sensitive to prompts"; proposes a prompt-ensemble "that emulates annotator disagreement"; LLM outputs "align with human subjective distributions". Variation is across prompts, not repeated sampling. | https://arxiv.org/abs/2603.08936 |
| Zhang et al., "Scaling Ambiguity" | arXiv Jan 2026 | Gemini 2.5 Pro sampled at temperature 0.1–1.0 and with "annotator persona" prompts to act as synthetic annotators; ~6–10 samples per utterance approximate the human distribution (IEMOCAP, MSP-Podcast); helps on low-ambiguity items, not on high-ambiguity ones; synthetic-only training is worst. | https://arxiv.org/abs/2601.14620 |
| Jia et al., "Decoding Ambiguous Emotions with Test-Time Scaling" | arXiv Feb 2026 | 8 ALMs incl. GPT-4o, Gemini 2.0 Flash, Gemini 2.5 Pro; IEMOCAP, MSP-Podcast, CREMA-D; Best-of-N / verifier scaling. Multiple diverse outputs capture ambiguity better than one; error grows with human-vote entropy. W-ALM-v: 31.1% relative JS reduction on MSP-Podcast. | https://arxiv.org/abs/2602.03873 |
| Yu et al., distributional reasoning | arXiv Mar 2026 (Interspeech submission) | Treats SER as distributional reasoning aligned with human vote distributions (IEMOCAP, CREMA-D) via SFT/DPO/GRPO. | https://arxiv.org/abs/2603.08230 |
| Music-emotion annotation with GPT-4o | arXiv Aug 2025 | Not speech and text metadata only: at T=0, 385/400 identical over 3 runs; Fleiss' kappa 0.561 (humans) vs 0.558 (humans + GPT-4o), "variability within the range of natural human disagreement". Possibly the source of the "like human listeners" phrasing. | https://arxiv.org/abs/2508.12626 |
| Chiang et al., ALLMs as judges of speaking style | EMNLP 2025 Findings | Gemini-2.5-pro vs human judges on emotion/volume/pace/pitch: agreement "comparable to the agreement between human evaluators". | https://aclanthology.org/2025.findings-emnlp.25/ |

### A3. Prompt sensitivity (wording, label order, label set)

| Paper | Venue / year | Key finding | Link |
|---|---|---|---|
| VoxEmo | Interspeech 2026 submission | >20-point Macro-F1 swings across prompts on 11 datasets; parse failures up to 93.2%. | https://arxiv.org/abs/2603.08936 |
| Lin et al., "Hearing the Order" | Interspeech 2026 (per arXiv page) | Shuffling answer options changes LALM performance by up to 24% and changes model rankings; "no model is immune"; permutation methods mitigate. | https://arxiv.org/abs/2510.00628 |
| Lin et al., GPT-4o Voice Mode | arXiv Feb 2025 | GPT-4o refuses speaker ID, age classification, deepfake detection; refusal rate varies across datasets. | https://arxiv.org/abs/2502.09940 |
| Murzaku & Rambow, OmniVox | arXiv Mar 2025 (COLM 2025 per summary, unverified) | Zero-shot omni-LLMs competitive with fine-tuned audio models on IEMOCAP/MELD with "acoustic prompting"; results depend on prompt style (minimal vs CoT vs acoustic). | https://arxiv.org/abs/2503.21480 |

### A4. Neutral bias, calibration, no probabilities

| Paper | Venue / year | Key finding | Link |
|---|---|---|---|
| VocalAffectBench | arXiv 2026 | 45.3% of outputs neutral; gpt-realtime 69.6% neutral; models output a verbal "confidence" but it was not evaluated. | https://arxiv.org/abs/2608.28932 |
| LISTEN | EACL 2026 (unverified) | Defaults to neutral under neutral lexical content. | https://arxiv.org/abs/2510.10444 |
| Huang, Pol, Zhang (Voxtral ICL) | arXiv Jun 2026 | Vanilla few-shot ICL collapses predictions to neutral (89.3% accuracy, but fails on emotional states). | https://arxiv.org/abs/2606.14784 |
| Kuan, Huang, Lee, uncertainty for ALLMs | arXiv Apr 2026 | First systematic uncertainty-estimation study for audio LLMs (entropy, semantic entropy, P(True)); no emotion task. | https://arxiv.org/abs/2604.25591 |
| Bellver-Soler et al., ASOM | ICASSP 2027 (per arXiv comment) | Non-generative 6.2M "reader" gives probability distributions; beats 6 zero-shot ALMs by 0.072 on MSP-Podcast; motivated by ALMs not giving probabilities. | https://arxiv.org/abs/2610.22401 |

### A5. Not using audio / text dominance / hallucination

| Paper | Venue / year | Key finding | Link |
|---|---|---|---|
| C2SER (Zhao et al.) | IEEE/ACM TASLP (per arXiv page) | "ALMs often suffer from hallucinations, resulting in misclassifications or irrelevant outputs" in SER; fixes with Whisper + Emotion2Vec-S and CoT. | https://arxiv.org/abs/2502.18186 |
| Piao et al., "When Demonstrations Fail" | IEEE SLT 2026 | Shuffled demo labels leave SER (CREMA-D) performance unchanged, so models do not learn audio-label mapping from demos. | https://arxiv.org/abs/2603.20433 |
| Koduru et al. / VoxParadox / EMIS | see A1 | Encoded paralinguistics is lost; text wins conflicts. | see A1 |
| AHELM (Lee et al., Stanford CRFM) | arXiv Aug 2025 | ASR + text-LM baseline ranks 6th overall despite having only speech-to-text, so many benchmark items do not need audio. | https://arxiv.org/abs/2508.21376 |

### A6. Paralinguistic weakness in general benchmarks

| Benchmark | Venue / year | Relevant finding | Link |
|---|---|---|---|
| AIR-Bench | ACL 2024 | 19 foundation tasks incl. emotion, ~19k MCQ + 2k chat; GPT-4 judge. | https://arxiv.org/abs/2402.07729 |
| Dynamic-SUPERB Phase-2 | ICLR 2025 | 180 tasks; Qwen2-Audio-7B-Instruct strong on emotion, but no model handles most tasks. | https://arxiv.org/abs/2411.05361 |
| AudioBench | arXiv 2024–2025 (NAACL 2025, unverified) | 8 tasks / 26 datasets incl. paralinguistic voice understanding; no model best across tasks. | https://arxiv.org/abs/2406.16020 |
| MMAU | arXiv 2024 (ICLR 2025, unverified) | 10k clips; Gemini Pro 1.5 52.97%, Qwen2-Audio 52.50%. | https://arxiv.org/abs/2410.19168 |
| MMSU | ICLR 2026 | 5,000 items, 47 tasks incl. paralinguistics/prosody; 14 SpeechLLMs leave large room for improvement. | https://arxiv.org/abs/2506.04779 |
| CP-Bench (Wang et al., A*STAR) | EMNLP 2025 Findings | Speech-LLMs fail to combine verbal and non-verbal cues in in-the-wild empathetic QA. | https://arxiv.org/abs/2509.16589 |
| Vox-Profile (Feng et al.) | DMLR (per arXiv page) | Speaker/speech trait benchmark (emotion, age, accent...) with WavLM/Whisper heads; source of our Vox-Profile models. | https://arxiv.org/abs/2505.14648 |
| EmoNet-Voice | arXiv Jun 2025 | Synthetic 40-category benchmark; high-arousal anger ~95%, sadness vs distress 63%. | https://arxiv.org/abs/2506.09827 |
| SpeechParaling-Bench | arXiv Apr 2026 | Paralinguistic generation; failing to understand user paralinguistic cues causes 43.3% of dialogue failures. | https://arxiv.org/abs/2604.20842 |
| "SpeechR" | not found | unverified / no matching paper located. | — |

---

## B. Few-shot / in-context learning with audio demonstrations for SER

| Paper | Venue / year | Setting and finding | Link |
|---|---|---|---|
| **Piao et al., "When Demonstrations Fail"** | IEEE SLT 2026 | 6 LALMs (Qwen2-Audio, DeSTA2.5, BLSP-Emo, Qwen2.5-Omni, Phi-4-MM, **Gemini 2.5 Flash**); tasks ASR, **SER on CREMA-D**, gender, MMAU; k = 1–8 random demos (task-matched, random order). Demos "improve format compliance but fail to improve the core task performance"; with shuffled labels "SER performance remains comparable". No speaker matching, no retrieval. | https://arxiv.org/abs/2603.20433 |
| **Wong, Huzaifah, Chen, Aw (A*STAR)**, "Speech in-context learning of paralinguistic tasks" | IEEE ASRU 2025 | Stutter detection, pronunciation, **SER**; open models "already exhibit some degree of speech ICL on paralinguistic tasks"; ensemble over multiple ICL runs with different exemplars to beat context limits. Numbers not checked (no open full text found). | https://doi.org/10.1109/ASRU65441.2025.11434787 (record: https://oar.a-star.edu.sg/communities-collections/articles/22939) |
| Ihori et al. (NTT), few-shot personalisation | ASRU 2025 | Own 0.6B speech-LM meta-trained (MetaICL) with 0–7 **same-speaker** enrollment pairs; Japanese, 800 speakers; +2–3 pt with ≤2 shots, **+8.2 pt at 7-shot** over zero-shot. Needs training; not off-the-shelf LLMs. | https://arxiv.org/abs/2509.08344 |
| Huang, Pol, Zhang | arXiv Jun 2026 | Voxtral, audio+transcript demos, **acoustic-similarity retrieval** (pitch/loudness/rate) vs random; random few-shot collapses to neutral; retrieval raises macro-F1 0.30 -> 0.49 (German VR speech, 794 segments). | https://arxiv.org/abs/2606.14784 |
| Hsu, Chang, Li, Lee | Interspeech 2024 | Textless speech LM "lacks the ICL capability" until warm-up training; first speech LM doing unseen classification by ICL. | https://arxiv.org/abs/2310.12477 |
| Wang et al., "Can Whisper perform speech-based ICL?" | ICASSP 2024 | SICL for ASR adaptation; kNN demo selection gives 36.4% average relative WER reduction. Not emotion. | https://arxiv.org/abs/2309.07081 |
| Audio Flamingo (Kong et al.) | ICML 2024 | Claims fast adaptation "via in-context learning and retrieval". | https://arxiv.org/abs/2402.01831 |
| MiMo-Audio (Xiaomi) | arXiv Dec 2025 | ">100M hours" pretraining gives few-shot ability on unseen audio tasks; emotion few-shot not stated in abstract. | https://arxiv.org/abs/2512.23808 |
| FSA-GRPO (Zheng et al.) | arXiv May 2026 | RL post-training to make auditory LLMs use demos; 53.9% relative WER reduction on child ASR. Not emotion. | https://arxiv.org/abs/2606.02615 |
| Chen et al., "Beyond Single-Audio" | EMNLP 2024 Findings | ALLMs struggle when several audio clips are in one prompt (a precondition for audio ICL). | https://arxiv.org/abs/2409.18680 |
| Jiang et al., Many-shot multimodal ICL | arXiv 2024 | GPT-4o / Gemini 1.5 Pro gain from up to ~2,000 image demos; images only. | https://arxiv.org/abs/2405.09798 |

Novelty verdict for B: **partially covered**. Piao et al. (SLT 2026) already tests random k-shot SER on CREMA-D with Gemini 2.5 Flash and finds demos teach format, not the mapping. Wong et al. (ASRU 2025) shows some paralinguistic ICL in open models. Ihori et al. shows same-speaker demos help, but only after meta-training. **Open gap:** a controlled factorial study on frontier closed models (GPT Realtime, Gemini 3.x) and open LALMs, over several public acted corpora, that crosses k (0/1/3/5), **same vs different speaker**, **class-balanced vs random**, **retrieved (acoustic/embedding kNN) vs random**, with controls (shuffled labels, text-only label lists, demo order), and a comparison with dedicated SER models and their cost. Speaker-matched demos on lexically-neutral corpora (where the cascade/lexical route gives nothing) are not covered by any paper found. The paper must engage directly with Piao et al.; a null result ("demos fix format and neutral bias but not accuracy") is still publishable if the controls are strong.

---

## C. Novelty check for other candidate angles

| Angle | Closest prior work (venue/year) | Verdict |
|---|---|---|
| (i) Contamination / lineage audit of released SER checkpoints | Tseng et al., test-set contamination for LLM-based ASR (arXiv 2025) https://arxiv.org/abs/2505.22251 ; Baah-Mensah & Asamoah, speaker-leakage audit, accuracy drops 6.9–63.1 pt under speaker independence (arXiv Oct 2026) https://arxiv.org/abs/2610.33952 . Lineage facts: emotion2vec+ seed is fine-tuned on EmoBox academic data, base/large on filtered pseudo-labelled data (large: 40k of 160k h) https://huggingface.co/emotion2vec/emotion2vec_plus_large ; SenseVoice pseudo-labels SER with emotion2vec_plus_large and evaluates on CREMA-D, MELD, IEMOCAP, MSP-Podcast, CASIA, MER2023, ESD (FunAudioLLM) https://arxiv.org/abs/2407.04051 ; MERaLiON-SER trains on CREMA-D, M3ED, ESD, MELD and keeps segments where emotion2vec+ seed and base agree https://arxiv.org/abs/2511.04914 ; C2SER labels its internal corpus with Emotion2Vec https://arxiv.org/abs/2502.18186 | **Open.** No paper audits which public test corpora are inside released SER checkpoints, or traces the emotion2vec pseudo-label lineage across SenseVoice/MERaLiON/C2SER and its effect on "zero-shot" claims. |
| (ii) Acted (intended) vs perceived labels | Giordano et al., "Label Authority in SER" (Emozionalmente, Italian, frozen XLS-R/HuBERT/WavLM; arXiv Oct 2026) https://arxiv.org/abs/2610.26446 ; VoxEmo soft labels; CHUCKLE (intended–perceived agreement curriculum on CREMA-D; arXiv 2025) https://arxiv.org/abs/2510.09382 ; CREMA-D original: human audio-only recognition of intended emotion 40.9% (Cao et al., IEEE TAC 2014; number from secondary source, unverified) | **Partially covered.** Concept and one-corpus study exist. Still open: scoring off-the-shelf SER models and audio LLMs against intended vs perceived (majority and vote distribution) on several corpora, and showing ranking changes. |
| (iii) Telephone / narrowband codec robustness | EMO-Codec (Ren et al., APSIPA ASC 2024; legacy + neural codecs degrade SER, esp. sad/fear/disgust) https://arxiv.org/abs/2407.15458 ; Codec-SUPERB @ SLT 2024 (emotion2vec on RAVDESS after codec resynthesis) https://arxiv.org/abs/2409.14085 ; Albahri et al., AMR/AMR-WB/AMR-WB+ lower SER accuracy, handcrafted features (c. 2016, venue unverified) https://www.researchgate.net/publication/283181412 | **Mostly open** for G.711 / AMR-NB / 8 kHz on modern released SER checkpoints and audio LLMs; good practical (call-centre) angle. |
| (iv) Input duration / 30 s padding in Whisper-based SER | Padding-free Whisper inference for ASR (IEEE 2025/26; details unverified, page blocked) https://ieeexplore.ieee.org/abstract/document/11462351/ ; Whisper attentive-pooling SER (arXiv Feb 2026) https://arxiv.org/abs/2602.06000 | **Open but small.** No SER-specific padding study found; best as one analysis section, not a full paper. |
| (v) Calibration (ECE), soft-label evaluation | Chou et al., calibration of multi-label SER, Interspeech 2023 https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html ; VoxEmo (KLD/JSD) ; Jia et al. TTS (JS, BC) https://arxiv.org/abs/2602.03873 ; Kuan et al. ALLM uncertainty https://arxiv.org/abs/2604.25591 | **Partially covered.** Open: ECE of released zero-shot SER checkpoints plus verbalised / sampled confidence of closed audio LLMs, against vote distributions (CREMA-D has 8–12 votes per clip). |
| (vi) Cross-lingual generalisation (Mandarin held out) | EmoBox cross-corpus; VoxEmo (15 languages); Han et al., cross-lingual SER humans vs SSL, ICASSP 2025 https://arxiv.org/abs/2409.16920 ; zero-shot cross-lingual SER (arXiv Jun 2026) https://arxiv.org/abs/2606.06200 | **Well covered.** Only adds value as a held-out check inside a bigger paper. |
| (vii-a) CoT / reasoning for SER | C2SER (TASLP); Audio-CoT https://arxiv.org/abs/2501.07246 ; Jia et al. test-time scaling; Yu et al. distributional reasoning https://arxiv.org/abs/2603.08230 | **Partially covered**; little on closed models with lexically neutral audio. |
| (vii-b) Cascaded ASR+LLM vs end-to-end | AHELM (ASR+LM baseline 6th overall); LISTEN text-only condition | **Partially covered.** On lexically neutral corpora a cascade should sit near chance — a cheap control that proves the audio route is used. |
| (vii-c) Run-to-run consistency of closed audio LLMs for SER | VoxEmo (prompt variation), Scaling Ambiguity (temperature sampling) | **Open-ish.** No direct measurement of label flip rate / Fleiss' kappa across repeated identical calls vs human inter-rater kappa found. Cheap to add (CREMA-D has per-clip votes). |
| (vii-d) LLM-as-judge agreement | Chiang et al., EMNLP 2025 Findings https://aclanthology.org/2025.findings-emnlp.25/ | **Covered** for speaking style; low priority. |

## Suggested framing

Strongest combination: (1) zero-shot benchmark on 6 public corpora with open SER vs closed LLMs (cite VocalAffectBench / LISTEN / VoxEmo as consistent evidence); (2) controlled audio ICL study (B) with speaker-matched and retrieved demos and label-shuffle controls, positioned against Piao et al. 2026; (3) a lineage/contamination audit (C-i) and intended-vs-perceived re-scoring (C-ii) as the "what are we measuring" section; optional (4) run-to-run consistency vs human kappa and (5) telephone codec robustness.

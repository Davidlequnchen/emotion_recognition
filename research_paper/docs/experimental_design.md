# Experimental Design (finalised 2026-11-06)

The design of the benchmark behind the paper: which datasets, which models, how they are scored, what runs where
and how long it takes, and which analyses the results support. Local only (`research_paper/` is git-ignored).

Sources: `emotion_recognition/docs/datasets_survey_en.md`, `docs/label_mapping.md`, `docs/models_survey_en.md`,
`docs/audio_llms_survey_en.md`, `scripts/common/eval_datasets.py`, `config/models.yaml`, `config/evaluation.yaml`,
`config/llm.yaml`, `scripts/llm_eval/prompts.yaml`, `scripts/models/*.py`, `presentation/facts.md`, `README.md`,
`reports/summary.md` (state of 2026-11-06), and [research_angles.md](research_angles.md),
[related_work.md](related_work.md), [venues.md](venues.md).

**Measured here (2026-11-06)**, from the loaders (`LOADERS[key]().clips`, which apply the plan-B subsets) and file
headers only (`soundfile.info`, `nice -n 15`, no model loaded): clips, hours, sample rates, clip-length counts;
from the metadata files: THAI-SER actor genders and raters per clip, Emozionalmente speakers and raters, CREMA-D
audio-only votes per clip; with `du -sk`: size on disk (SAVEE measured again the same way after its return to the
set). Run times of finished runs: `runtime.seconds` in `outputs/<model>/<dataset>_predictions.json`, or the
report's Hardware line where the JSON has none. Everything else is quoted from the files above.
"Unverified" marks a figure that no project file confirms.

Units: model sizes in M (million) and B (billion) parameters; disk sizes in decimal GB.

---

## 1. Overview

### 1.1 Research questions

| # | Question | What answers it |
|---|---|---|
| RQ1 | How do dedicated SER models compare with audio LLMs, zero-shot, on acted, crowd-sourced and TV speech? | 10 models × 13 datasets (the paper may show 12, section 7.1), the same clips and labels (section 6 main results) |
| RQ2 | Do audio LLMs rely on the words rather than the voice (lexical vs acoustic)? | datasets with parallel, lexically uninformative text vs datasets whose words fit the emotion; within-dataset contrasts (THAI-SER scripted vs improvised) |
| RQ3 | Which ground truth do models reproduce: the emotion the actor intended, or the one listeners perceived? | the 4 datasets with per-clip listener labels (CREMA-D, MCAE-SPPS, THAI-SER, Emozionalmente) |
| RQ4 | How much of a model's score comes from having seen the dataset in training (seen vs unseen)? | the training-data flags of `config/models.yaml`; 4 datasets unseen by every model with published training data |

### 1.2 Protocol in one paragraph

Every model is run **zero-shot, as released** (no fine-tuning, no calibration), on the same clips (THAI-SER limited
to its high-agreement subset, MEAD to a stratified 8,000-clip subset, MELD to its test split, every other dataset in
full: 13 datasets, 56,264 clips, 52.24 h) and scored against the same labels. All labels are mapped to one **7-class space:
angry, disgusted, fearful, happy, neutral, sad, surprised** (the emotion2vec+ class names). Mapping rules: renames
only (e.g. `joy` = happy, `fear` = fearful, `rabbia` = angry); **calm = neutral** (RAVDESS, MESS); **pleasant
surprise = surprised** (TESS); a class with no counterpart is **dropped with its clips** (THAI-SER *frustrated*,
MEAD *contempt*, the 5 JL-Corpus secondary emotions); one **composite label**, JL-Corpus *excited*, counts a
prediction of happy **or** surprised as correct. The ground truth is the **acted (intended) emotion**; MELD, which
has no actors portraying emotions, uses its annotators' label. A dedicated SER model's prediction is its most
probable class **among the dataset's own emotions**; an audio LLM is asked to choose one of **all seven** labels,
and an answer outside the dataset's emotions, or no readable answer, counts as wrong. Metrics: **accuracy** (vs the
acted emotion), **UA** (unweighted accuracy, the mean per-class recall), **macro-F1**, and, for the 4 datasets with
listener labels, **acted-or-listener accuracy** (correct if the prediction matches the acted emotion or the
listeners' majority, any emotion of a tied vote); also accuracy on the 4 emotions every dataset has (angry, happy,
neutral, sad). **Chance level** = the probability that a uniform guess among the dataset's classes is correct
(1/K, except JL-Corpus 24.0% because *excited* accepts 2 of 5 classes).

### 1.3 Chance levels

| Dataset | Classes (K) | Chance, SER models (choose among the dataset's K) | Chance, audio LLMs (choose among 7; computed here, unverified against the reports) |
|---|---|---|---|
| CREMA-D | 6 | 16.7% | 14.3% |
| MCAE-SPPS | 7 | 14.3% | 14.3% |
| THAI-SER | 4 | 25.0% | 14.3% |
| nEMO | 6 | 16.7% | 14.3% |
| MESS | 4 | 25.0% | 14.3% |
| RAVDESS | 7 | 14.3% | 14.3% |
| TESS | 7 | 14.3% | 14.3% |
| JL-Corpus | 5 (+ composite *excited*) | 24.0% | 17.1% |
| MEAD | 7 | 14.3% | 14.3% |
| MELD | 7 | 14.3% | 14.3% |
| EMOVO | 7 | 14.3% | 14.3% |
| Emozionalmente | 7 | 14.3% | 14.3% |
| SAVEE | 7 | 14.3% | 14.3% |

Because the LLMs choose among 7 labels, their accuracy compares best with each SER model's "accuracy with all model
classes" (in each per-dataset report), not with the restricted-choice accuracy (README, section 8).

### 1.4 Metric definitions (as implemented in `scripts/evaluation/evaluate.py`)

| Metric | Definition | Datasets |
|---|---|---|
| Accuracy | prediction = acted emotion (for *excited*: happy or surprised) | all 13 |
| UA | mean recall over the acted emotions | all 13 |
| Macro-F1 | mean F1 over the predicted emotions | all 13 |
| Acted-or-listener accuracy | prediction = acted emotion **or** listeners' majority (a tie counts every tied emotion) | CREMA-D, MCAE-SPPS, THAI-SER, Emozionalmente |
| Listeners vs acted | how often the untied listener majority equals the acted emotion (human reference) | the same 4 |
| Model vs listeners | how often the model's prediction equals the untied listener majority | the same 4 |
| 4-class accuracy | only clips of angry, happy, neutral, sad; prediction = most probable of these 4 | all 13 (computed from the prediction files) |
| Neutral rate | share of predictions that are *neutral* (planned, section 6) | all 13 |

---

## 2. Datasets

**13 datasets** are evaluated (2026-11-06); **ESD alone is archived, out of scope** (section 2.5). SAVEE was
archived and brought back the same day: every model evaluates it, and whether the paper shows it is open (the paper
may use 12, a 3 × 4 grid in figures, with SAVEE the candidate to drop; section 7.1). Plan B (decided 2026-11-06, `subsets` in
`config/evaluation.yaml`): THAI-SER high-agreement subset (agreement ≥ 0.71) and MEAD stratified 8,000 clips; the
subset is part of the loader, so every model and script sees the same clips.

### 2.1 Basic facts

| Dataset | Year | Language / accent | Speech type | Speakers (gender) | Listeners per clip / in total |
|---|---|---|---|---|---|
| CREMA-D | 2014 | American English | acted (actors); crowd-rated | 91 (48 M, 43 F, aged 20–74) | audio-only votes per clip 4–12, mean 9.21 (measured, `tabulatedVotes.csv`); 2,443 crowd raters in total, each rating 30 audio-only + 30 face + 30 audio-visual clips |
| MCAE-SPPS | 2025 (DataPort) / 2026 (Sci Data) | Mandarin | acted, professional | 6 (3 M, 3 F) | 40 native listeners per clip / 720 in total |
| THAI-SER | 2021 (data) / 2025 (paper) | Thai (tonal) | acted, professional; scripted + improvised | 200 (112 F, 88 M; measured, `labels.csv`) | crowd: 3–11 per clip in our subset, mean 5.14 (measured from `annotated`; the survey quotes 3–8); total raters unverified |
| nEMO | 2024 | Polish | acted, mostly non-professional (3 of 9 are professional voice actors), recorded at home | 9 (4 F, 5 M) | screening only (about 92% of recordings kept); no per-clip labels, counts not published |
| MESS | 2019 | American English | acted (speaker background unverified) | 6 (3 M, 3 F) | 10 per clip in the paper (category + activation + pleasantness); **not in the public release** (on request) |
| RAVDESS | 2018 | North American English | acted, professional | 24 (12 F, 12 M) | 10 ratings per file / 247 raters; published as per-file recognition rates, not votes (not used) |
| TESS | 2020 (Borealis release cited; original recording year unverified) | Canadian English | acted | 2 actresses (aged 26 and 64) | none published (unverified) |
| JL-Corpus | 2018 | New Zealand English | acted, trained voice actors | 4 (2 F, 2 M) | perception test with 120 people (about 70% recognised for primary emotions); per-clip results not published |
| MEAD | 2020 | English (accent unverified) | acted, audio-visual (talking-face corpus) | 60 in the corpus; 48 in the public Part0 used here (27 M, 21 W; counted from folder names) | video evaluation with 100 participants; no per-clip labels |
| MELD | 2019 | English (*Friends*) | TV dialogue (scripted sitcom, not emotion-elicited) | many (count unverified) | 3 annotators per utterance who saw video + transcript (Fleiss' κ 0.43); their majority **is** the ground truth |
| EMOVO | 2014 | Italian | acted, professional | 6 (3 F, 3 M) | two groups of 24 listeners on 2 sentences per actor; no per-clip labels |
| Emozionalmente | 2025 | Italian | crowd-sourced, non-professional speakers, home equipment | 431 (299 F, 131 M, 1 other; measured, `users.csv`) | 5 per clip / 829 in total (measured: 34,510 ratings); authors' human UAR 66% |
| SAVEE | 2009 | British English | acted, audio-visual | 4 (all M; speakers DC, JE, JK, KL, 120 clips each) | 10 evaluators in the original evaluation (survey; the official site could not be re-checked); no per-clip labels in the copy used |

### 2.2 Labels and text design

| Dataset | Emotion classes in the corpus | → Mapped classes (dropped) | Intensity levels | Text design |
|---|---|---|---|---|
| CREMA-D | angry, disgust, fear, happy, neutral, sad (6) | all 6 kept; no surprised | only sentence IEO has low/medium/high; all other clips unspecified (XX) | **parallel, text-neutral**: 12 sentences × 6 emotions |
| MCAE-SPPS | neutral, happiness, anger, fear, sadness, disgust, surprise (7) | one-to-one with the 7 classes | perceived intensity ratings 1–9 (not acted levels) | **parallel**: 200 subject-personal-pronoun sentences × 7 emotions × 6 actors; validated (kept if the target got ≥ 43% of votes) |
| THAI-SER | neutral, angry, happy, sad, frustrated (5) | 4 kept; **frustrated dropped** (5,270 of 27,854 clips) | 2 acted levels (low/high) in the scripted part only (our subset: 2,502 low, 2,510 high) | scripted part **parallel** (3 sentences × 5 emotions × 2 intensities × 2 takes; 5,012 clips in our subset); improvised part **not parallel** (6,836 clips in our subset) |
| nEMO | anger, fear, happiness, sadness, surprised, neutral (6) | all 6 kept; no disgusted | none | **parallel**: 90 phoneme-coverage sentences × 6 emotions × 9 speakers |
| MESS | angry, calm, happy, sad (4) | **calm = neutral** | none (perceived activation/pleasantness, not public) | **parallel**: 90 sentences × 4 emotions × 6 speakers (public version 1,800 of 2,160 designed) |
| RAVDESS | neutral, calm, happy, sad, angry, fearful, disgust, surprised (8) | 7; **calm = neutral** (neutral 288 = 96 neutral + 192 calm) | 2 (normal, strong; neutral only normal) | **parallel, text-neutral**: 2 sentences ("Kids are talking by the door", "Dogs are sitting by the door") |
| TESS | angry, disgust, fear, happy, neutral, pleasant surprise, sad (7) | 7; **pleasant surprise = surprised** | none | **parallel**: carrier phrase "Say the word ___" with 200 words |
| JL-Corpus | 5 primary (angry, happy, neutral, sad, excited) + 5 secondary (anxious, apologetic, assertive, concerned, encouraging) | primary only; **excited = happy or surprised** (composite); **secondary dropped** (1,200 clips) | none | primary: **parallel**, 15 sentences in each emotion; secondary use other sentences, some with emotional words |
| MEAD | angry, disgusted, contempt, fear, happy, neutral, sad, surprised (8) | 7; **contempt dropped**; fear = fearful | **3 acted levels** (weak, medium, strong; neutral one level) | **partly emotion-related wording**: per emotion 3 common + 7 emotion-related + 20 general sentences; different lists per emotion |
| MELD | anger, disgust, fear, joy, neutral, sadness, surprise (7) | all 7 (renames) | none | **natural dialogue**: words often carry the emotion; very unbalanced (48% neutral, 2% fear) |
| EMOVO | rabbia, disgusto, paura, gioia, neutrale, tristezza, sorpresa (7) | all 7 (renames) | none | **parallel**: 14 sentences (some nonsense) × 7 emotions |
| Emozionalmente | anger, disgust, fear, joy, neutrality, sadness, surprise (7) | all 7 (renames) | none | **parallel**: 18 sentences (measured), 986 clips per emotion |
| SAVEE | anger, disgust, fear, happiness, neutral, sadness, surprise (7) | all 7 (renames; `sa` = sad, `su` = surprised) | none | **mostly not parallel**: 15 sentences per emotion, only 3 shared across emotions, some emotion-specific; neutral 30 sentences (120 clips, the other emotions 60 each) |

"Parallel" = the same sentences are spoken in every emotion, so the words cannot predict the label.

### 2.3 Clips, audio, licence and access

Clips, hours, mean length and sample rate measured 2026-11-06 (header only); per dataset they match the survey
tables exactly, and so do the totals of the 13 datasets (56,264 clips, 52.24 h; SAVEE measured again with its loader
and `soundfile.info`: 480 clips, 0.51 h). "Size on disk now" = `du -sk` of `data/<dataset>/` today, in decimal GB.

| Dataset | Clips evaluated | Hours | Mean s/clip | Clips < 1 s / > 15 s / > 30 s | Size on disk now | Sample rate (as stored) |
|---|---|---|---|---|---|---|
| CREMA-D | 7,442 (all) | 5.26 | 2.54 | 0 / 0 / 0 | 0.62 GB | 16 kHz |
| MCAE-SPPS | 6,675 (all) | 2.30 | 1.24 | 1,797 / 0 / 0 | 0.82 GB | 48 kHz |
| THAI-SER | **11,848**: high-agreement subset (agreement ≥ 0.71) of the 22,584 clips of the 4 kept classes; neutral 2,999, happy 3,665, angry 2,883, sad 2,301 | 17.36 (all 22,584: 33.19) | 5.28 | 468 / 210 / 34 (longest 91.0 s) | 4.68 GB (extracted FLAC + `labels.csv`; the 12 GB Parquet files are deleted, so the survey's "16 GB" is outdated) | 44.1 kHz |
| nEMO | 4,481 (all) | 3.07 | 2.47 | 14 / 0 / 0 | 1.60 GB (with the Parquet files) | 24 kHz |
| MESS | 1,800 (all public) | 1.33 | 2.66 | 0 / 0 / 0 | 0.23 GB | 24,414 Hz |
| RAVDESS | 1,440 (all speech) | 1.48 | 3.70 | 0 / 0 / 0 | 0.59 GB | 48 kHz (5 stereo files) |
| TESS | 2,800 (all) | 1.60 | 2.06 | 0 / 0 / 0 | 0.28 GB | 24,414 Hz (1 file 96 kHz) |
| JL-Corpus | 1,200 (5 primary emotions of 2,400) | 0.70 | 2.09 | 0 / 0 / 0 | 0.97 GB (whole Kaggle bundle) | 44.1 kHz |
| MEAD | **8,000**: stratified subset of the 27,514 clips without contempt (every actor/emotion/level cell, 912 cells, keeps its share) | 9.51 (all 27,514: 32.73) | 4.28 | 2 / 4 / 0 | 4.35 GB (all 31,729 converted WAVs) | 16 kHz (converted from 48 kHz AAC) |
| MELD | **2,608**: test split (2,610) less 2 clips with wrong time stamps | 2.31 | 3.18 | 225 / 4 / 0 | 0.28 GB | 16 kHz (converted from video) |
| EMOVO | 588 (all) | 0.51 | 3.12 | 0 / 0 / 0 | 0.35 GB | 48 kHz |
| Emozionalmente | 6,902 (all) | 6.30 | 3.29 | 2 / 0 / 0 | 0.74 GB | 16 kHz |
| SAVEE | 480 (all) | 0.51 | 3.84 | 0 / 0 / 0 (1.6–7.1 s) | 0.16 GB (unpacked Kaggle copy) | 44.1 kHz |
| **Total (13)** | **56,264** (86,514 without the subsets) | **52.24** (about 91.3 without the subsets) | 3.34 | 2,508 / 218 / 34 | **15.67 GB** | every model resamples to its own rate (16 kHz; GPT Realtime 24 kHz PCM16) |

Clips shorter than 3 s (Vox-Profile's cards call predictions under 3 s unreliable): 31,249 of 56,264 (measured);
100% of MCAE-SPPS, TESS and JL-Corpus, 83% of CREMA-D, 81% of MESS, 80% of nEMO; SAVEE 105 of 480 (22%).

| Dataset | Licence | Source used | `public_audio` | `api_audio` |
|---|---|---|---|---|
| CREMA-D | ODbL 1.0 (contents DbCL) | official (GitHub, Git LFS, pinned commit) | yes | yes |
| MCAE-SPPS | CC BY-NC 4.0 | official (OSF 9jyzc) | yes | yes |
| THAI-SER | CC BY-SA 4.0 | official (Hugging Face airesearch/thai-ser) | yes | yes |
| nEMO | CC BY-NC-SA 4.0 | official (Hugging Face amu-cai/nEMO) | yes | yes |
| MESS | CC BY 4.0 | official (Zenodo 3813437) | yes | yes |
| RAVDESS | CC BY-NC-SA 4.0 | official (Zenodo 1188976) | yes | yes |
| TESS | CC BY-NC 4.0 | official (Borealis) | yes | yes |
| JL-Corpus | CC0 1.0 | official (Kaggle, uploaded by the author) | yes | yes |
| MEAD | no data licence published (terms of use; the repo's MIT covers code) | official (Google Drive Part0) | **no** | yes |
| MELD | GPL-3.0 repository; clips from a copyrighted TV series (fair use claimed) | official (archive on Hugging Face, CSV from GitHub) | **no** | yes |
| EMOVO | no licence published; official site gone | **unofficial mirror** (DagsHub) | **no** | yes |
| Emozionalmente | CC BY 4.0 | official (Zenodo 12616095, v1.1) | yes | yes |
| SAVEE | free for research after registration; the licence forbids redistribution | **unofficial Kaggle copy** (ejlok1), not an authorised mirror; checked only against file names and counts | **no** | yes |

`public_audio`: the clips may appear on a public page; `api_audio`: the clips may be sent to the API models for
this research evaluation (only ESD is refused; decided 2026-11-06).

### 2.4 Training-data overlap (seen = the model probably heard these clips)

From `training_data` in `config/models.yaml` and the survey matrices. emotion2vec+ is counted conservatively: all of
EmoBox (through emotion2vec+ seed) plus emotion2vec's unlabelled pre-training (MEAD, MELD). The open and API audio
LLMs publish no training data ("not published" ≠ unseen). Deferred models are listed for later reference.

| Dataset | emotion2vec+ large | MERaLiON-SER-v1 | Melati | Vox-Profile (both) | Qwen2.5-Omni / Qwen2-Audio / Gemma 4 / GPT Realtime / Gemini | Deferred: AF3 | Deferred: Kimi-Audio |
|---|---|---|---|---|---|---|---|
| CREMA-D | **seen** (EmoBox) | **seen** | – | – | not published | – (explicitly excluded in AF1/AF2) | – |
| MCAE-SPPS | – | – | – | – | not published | – | – |
| THAI-SER | – | – | – | – | not published | – | – |
| nEMO | – | – | – (held-out test on Melati's card) | – | not published | – | – |
| MESS | – | – | – | – | not published | – | – |
| RAVDESS | **seen** (EmoBox) | – | – | – | not published | – | **seen** (SFT) |
| TESS | **seen** (EmoBox) | – | – | – | not published | **seen** (all clips) | – |
| JL-Corpus | **seen** (EmoBox) | – | **seen** | – | not published | **seen** (all clips) | – |
| MEAD | **seen** (EmoBox + pre-training) | – | – | – | not published | – | – |
| MELD | **seen** (EmoBox + pre-training) | **seen** (train split) | – | – | not published; the Qwen models report MELD results, so MELD train was probably used (unverified) | **seen** (train split; test unverified) | **seen** (SFT) |
| EMOVO | **seen** (EmoBox) | – | – | – | not published | – | – |
| Emozionalmente | **seen** (EmoBox) | – | **seen** | – | not published | – | – |
| SAVEE | **seen** (EmoBox) | – | – | – | not published (LISTEN evaluated Gemini 2.5 on SAVEE: a test, not training) | – | **seen** (SFT, 0.1 h) |

**Unseen by every model with published training data: MCAE-SPPS, MESS, THAI-SER, nEMO.**

### 2.5 Archived / out of scope

| Dataset | Why out of scope | Status |
|---|---|---|
| ESD (2021, English + Mandarin, 20 speakers, 35,000 clips, 29.1 h; 3.43 GB on disk now) | licence is research-only, forbids any transfer to third parties and counts "testing commercial systems" as commercial use; no listening test | data stays in `data/esd/`; `exclude: [esd]` in `config/evaluation.yaml`; never sent to an API (`api_audio: False`); old emotion2vec+ result in `reports/archive/` |

SAVEE was moved here on 2026-11-06 and brought back into the evaluation later the same day (sections 2.1–2.4);
whether the paper shows it is open (section 7.1). `config/evaluation.yaml` excludes only ESD and lists `savee` in
its `order`, matching this.

---

## 3. What each dataset brings to the paper

| Dataset | Angle it supports | Acted / naturalistic | Seen / unseen | Language family | Listener labels per clip |
|---|---|---|---|---|---|
| CREMA-D | intended vs perceived (crowd votes; the untied audio-only majority matches the acted emotion for only 45.6% of 6,798 clips); text-neutral sentences for lexical vs acoustic; used by LISTEN, VoxEmo, AF1/AF2 | acted | seen (emotion2vec+, MERaLiON) | Indo-European (Germanic) | **yes** (≈ 9 votes) |
| MCAE-SPPS | clean held-out test: 40 listeners per clip, newest corpus (no audio LLM known to have used it), Mandarin; 200 sentences give lexical variety. Note: the clips were validated, so the untied listener majority equals the acted emotion for **100%** of clips; intended vs perceived must use the vote shares, not the majority | acted | **unseen** | Sino-Tibetan (tonal) | **yes** (40 votes) |
| THAI-SER | largest; unseen; 200 actors; crowd majority + agreement (majority ≠ assigned for 2,698 of 11,848 subset clips: 1,676 frustrated, 904 neutral); tonal language; scripted (parallel) vs improvised within one corpus for lexical vs acoustic; 2 acted intensities; mixed recording conditions (studio rooms A/B, Zoom); used by VoxEmo | acted (scripted + improvised) | **unseen** | Kra-Dai (tonal) | **yes** (3–11 votes) |
| nEMO | unseen Polish corpus; mostly non-professional speakers at home; used by VoxEmo and as Melati's held-out test | acted | **unseen** | Indo-European (Slavic) | no |
| MESS | unseen English; calm (not neutral) as 4th class, so tests how models treat low-arousal calm | acted | **unseen** | Indo-European (Germanic) | no (on request) |
| RAVDESS | classic acted English benchmark (LISTEN, AF1/AF2 zero-shot, EmoBox); 2 intensities; calm vs neutral | acted | seen (emotion2vec+) | Indo-European (Germanic) | no |
| TESS | classic acted English benchmark (LISTEN, EmoBox); 2 speakers, models often near 100% | acted | seen (emotion2vec+) | Indo-European (Germanic) | no |
| JL-Corpus | classic acted English (EmoBox, AF1/AF3 training); composite *excited* tests label-space mismatch | acted | seen (emotion2vec+, Melati) | Indo-European (Germanic) | no |
| MEAD | intensity levels (weak/medium/strong) for a per-intensity analysis; emotion-related sentences give lexical cues; used by VoxEmo | acted | seen (emotion2vec+) | Indo-European (Germanic) | no |
| MELD | in-the-wild TV dialogue, noisy multi-rater labels, words carry emotion; the standard audio-LLM reference result (Qwen cards, Kimi-Audio); heavily seen | naturalistic (TV) | seen (emotion2vec+, MERaLiON; probably LLMs) | Indo-European (Germanic) | annotator majority = ground truth |
| EMOVO | professional Italian actors; pairs with Emozionalmente for professional vs non-professional within one language; used by VoxEmo | acted | seen (emotion2vec+) | Indo-European (Romance) | no |
| Emozionalmente | crowd-sourced non-professional speakers + 5 listener votes per clip (intended vs perceived); professional vs non-professional Italian with EMOVO; used by VoxEmo | crowd-sourced acted | seen (emotion2vec+, Melati) | Indo-European (Romance) | **yes** (5 votes) |
| SAVEE | the 5th of LISTEN's corpora (direct comparison with LISTEN's audio-LLM results); British accent; male speakers only; mostly non-parallel text, so words may help. Weakest case for the paper: licence needs registration and forbids redistribution, unofficial Kaggle copy, only 4 speakers | acted | seen (emotion2vec+; deferred Kimi-Audio) | Indo-European (Germanic) | no |

Category counts: 12 acted (one of them crowd-sourced) + 1 naturalistic; 4 unseen + 9 seen; 8 English (incl. 1 New
Zealand and 1 British), 2 Italian, 1 Mandarin, 1 Thai, 1 Polish; 4 with per-clip listener labels. Of LISTEN's 5
corpora, CREMA-D, TESS, RAVDESS and SAVEE are in the set (ESD archived). Without SAVEE (12 datasets): 11 acted, 8
seen, 7 English.

---

## 4. Models

Ten models: 5 dedicated SER models, 3 open-weight audio LLMs run locally, 2 API audio LLMs. All run as released.

### 4.1 Identity, size and licence

| Model | Type | Architecture and size | Released | Licence |
|---|---|---|---|---|
| emotion2vec+ large | SER model | data2vec 2.0-style encoder (CNN + 12 Transformer layers, 1024-dim) + linear head; **162.1M** (card says ~300M; 2/3 of the 1.95 GB file is optimizer state) | 2024-05 | FunASR Model License v1.1 (HF; no commercial restriction; attribution) / Apache-2.0 (ModelScope) |
| MERaLiON-SER-v1 | SER model | Whisper-medium encoder (frozen, LoRA merged) **307.2M** + attention pooling + modified ECAPA-TDNN downstream **6.0M** + heads 0.13M = **313.4M** in use (file 770.0M with an unused Whisper decoder) | 2025-10 | MERaLiON Public Licence (MIT + attribution terms; commercial use allowed) |
| Melati | SER model | Whisper-large-v3 encoder (frozen, fp16) 637.0M + layer-mix head 0.33M = **637.3M** | 2026-10 | non-commercial research only |
| Vox-Profile Whisper | SER model | Whisper-large-v3 encoder (750 positions) 636.0M + LoRA 3.07M (LoRA B = 0, no effect) + heads 0.80M = **639.9M** | 2025-05 | OpenRAIL; card excludes commercial use and surveillance |
| Vox-Profile WavLM | SER model | WavLM-large (fully fine-tuned) 315.5M + heads 0.73M = **316.2M** | 2025-05 | as Vox-Profile Whisper |
| Qwen2.5-Omni 7B | open audio LLM | Thinker only: audio encoder 639.6M + Qwen2.5-7B LM (+ vision encoder 676.6M loaded, unused) = **8.93B**; Talker and token2wav not loaded | 2025-03 | Apache-2.0 |
| Qwen2-Audio 7B Instruct | open audio LLM | Whisper-large-v3-style encoder 637.0M + Qwen-7B LM = **8.40B** | 2024-07 | Apache-2.0 |
| Gemma 4 E4B (it) | open audio LLM | audio encoder 304.8M; **8.00B** with per-layer embeddings (**4.5B effective**) | not recorded in the project files (unverified) | Apache-2.0 (card links Google's Gemma 4 licence page) |
| GPT Realtime | API audio LLM | `gpt-realtime` (OpenAI Realtime API, WebSocket); size not disclosed | not recorded in the project files (unverified) | proprietary API |
| Gemini | API audio LLM | **placeholder** `gemini-2.5-flash` in `config/models.yaml` (to be replaced by the current Gemini audio model before the first run); size not disclosed | placeholder model: Gemini 2.5 report 2025-07 (arXiv 2507.06261) | proprietary API |

### 4.2 Labels, input, training data and where it runs

| Model | Native labels → 7-class mapping | Input handling | Training data (overlap: section 2.4) | Where it runs |
|---|---|---|---|---|
| emotion2vec+ large | 9: the 7 + *other* + `<unk>`; prediction = most probable of the dataset's emotions (other/unk never chosen) | 16 kHz mono (FunASR resamples, averages stereo); **real length**, no window | pseudo-labelled 42,526 h (sources not published) on top of seed (EmoBox 201 h) | Mac **CPU** (`mac_device: cpu`: MPS recompiles kernels per clip length and cannot run clips > ~20 s); env `emotion_recognition` |
| MERaLiON-SER-v1 | exactly the 7 | 16 kHz; **padded with silence to a fixed 30 s window**, pooled over the whole window **including the padding** (official behaviour, kept); first 30 s only | CREMA-D, ESD, MELD, M3ED, Singaporean SG-ECMT (partly labelled with emotion2vec+ seed/base agreement) and SGTV | Mac MPS; env `emotion_recognition` |
| Melati | exactly the 7 | 16 kHz; Whisper features padded to **30 s**, encoder layers averaged over the clip's own frames only; clips < 0.1 s padded; longer audio cut; raw argmax (the card's podcast-tagging thresholds not applied) | 28,895 clips of 9 acted corpora, incl. JL-Corpus and Emozionalmente | Mac MPS; env `emotion_recognition` |
| Vox-Profile Whisper | 9: the 7 + Contempt + Other (never chosen) | 16 kHz mono; at most 15 s, padded with silence to a **fixed 15 s window**, mean over the clip's own frames; card: trained on 3–15 s, < 3 s unreliable (31,249 of our clips are < 3 s) | MSP-Podcast v1.12 only | Mac MPS; env `emotion_recognition` |
| Vox-Profile WavLM | as Vox-Profile Whisper | 16 kHz mono, zero-mean/unit-variance, **real length** up to 15 s (cut), no padding | MSP-Podcast v1.12 only | Mac MPS; env `emotion_recognition` |
| Qwen2.5-Omni 7B | free text; prompt lists the 7 labels; answer parsed to one label, else `invalid` | 16 kHz mono; variable length (processor cap 300 s); audio **before** the user prompt | not published | Mac MPS, bfloat16, sdpa; env `emotion_recognition_llm` (also runs in the main env) |
| Qwen2-Audio 7B Instruct | as above | 16 kHz mono; **30 s window**: longer clips cut to the first 30 s (34 THAI-SER clips are > 30 s); audio before the prompt | not published | Mac MPS, bfloat16; env `emotion_recognition_llm` |
| Gemma 4 E4B | as above | 16 kHz mono; at most 750 audio tokens of 40 ms = **30 s** (cut); audio **after** the user prompt (card); thinking off | not published | Mac MPS, bfloat16; env `emotion_recognition_llm` only (needs PyTorch ≥ 2.6) |
| GPT Realtime | as above | **24 kHz 16-bit mono PCM**, base64, one WebSocket session per clip, text output only, no turn detection | not published | API, from the **company computer**; key in `.env` |
| Gemini | as above | 16 kHz mono 16-bit WAV, inline data (planned; loader is a placeholder) | not published | API, from the **company computer**; paid plan (free tier may train on inputs) |

The SER models' runs of 2026-10-28/29 used env `emotion_recognition`; phase 1 of plan B (section 5) runs all five in
env `emotion_recognition_llm`.

### 4.3 Prompts and decoding (audio LLMs)

| Setting | Local LLMs (Qwen2.5-Omni, Qwen2-Audio, Gemma 4) | GPT Realtime | Gemini (planned) |
|---|---|---|---|
| Prompts | the same system and user prompts for every model and dataset (`scripts/llm_eval/prompts.yaml`), labels "angry, disgusted, fearful, happy, neutral, sad, surprised"; the system prompt says the words are neutral and asks to judge the voice | same | same (system prompt as system instruction) |
| Decoding | greedy (`do_sample=False`, no repetition penalty), `max_new_tokens: 24`, bfloat16 | the API does not allow temperature 0 (answers can vary between runs) | temperature 0; thinking budget at its minimum |
| Invalid answers | a follow-up in the same conversation ("Answer with exactly one word from this list ..."), **once** (`retries_invalid: 1`); then `invalid` | asked again once (`retries_invalid: 1`), then `invalid`; network errors retried 3 times | same as GPT Realtime |
| Scoring | an answer that is none of the dataset's emotions, or `invalid`, is wrong (own *other answer* column) | same | same |
| Cache | every answer saved at once in `outputs/<model>/responses.jsonl`; a stopped run resumes; changing the prompts starts a new set | same; nothing is paid twice | same |
| Concurrency | one clip at a time | one request at a time | to be decided |

---

## 5. Run plan (plan B)

Three phases, every model on every clip of the 13 datasets (56,264 clips per model; ESD never run). Hours by
section 5.1 (planned = steady-state speeds measured 2026-11-06; actual = measured run time of finished runs).

| Phase | Models | Where | Clips | Estimated time | Cost | Status |
|---|---|---|---|---|---|---|
| **1** | the 5 SER models: emotion2vec+ large, MERaLiON-SER-v1, Melati, Vox-Profile Whisper, Vox-Profile WavLM | Mac M4 Max, env `emotion_recognition_llm` | 5 × 56,264 = 281,320 model-clips; 131,114 done, **150,206 to run** (emotion2vec+ 10,578; each other model 34,907) | done 8.07 h (actual); **remaining 7.39 h** (~7.9 h with the ×1.07 factor) | none (local) | **started 2026-11-06** (`models:` in `config/evaluation.yaml`); emotion2vec+ MEAD done 13:53; resumed after a manual pause |
| **2** | the 3 local audio LLMs: Qwen2.5-Omni 7B, Qwen2-Audio 7B Instruct, Gemma 4 E4B | Mac M4 Max, env `emotion_recognition_llm` | 3 × 56,264 = 168,792 model-clips | **36.5 h** by the table (~36 h planned, ~39 h with ×1.07); may run one model at a time: Gemma 4 ~10 h (9.8), Qwen2.5-Omni ~11 h (11.0), Qwen2-Audio ~16 h (15.7) | none (local) | after phase 1 |
| **3** | the 2 API models: GPT Realtime, Gemini | **company computer** (concurrency set there by the user) | 2 × 56,264 = 112,528 requests | GPT Realtime ~86 h at one request at a time (measured 5.5 s per clip), shorter with parallel requests; Gemini **not measured** (time per request unknown) | GPT Realtime ≈ **$106**; Gemini ≈ **$10–13**; total ≈ **$116–119** (estimates) | not started; Gemini model name still a placeholder |

Local total (phases 1 + 2): **done 8.07 h (actual) + remaining 43.85 h (plan)**; with the conservative ×1.07 the
remaining part is about 47 h.

### 5.1 Local models on the Mac M4 Max (48 GB): hours per model × dataset (phases 1 and 2)

"✓" = finished, the number is the **actual** run time; a number without ✓ is the **planned** time (steady-state
speed measured 2026-11-06). Totals are summed from unrounded seconds. The planned hours are steady-state; a
conservative factor of ~1.07 overall (model loading, the longest clips, other load on the shared Mac) turns the
remaining 43.85 h into about 47 h.

| Dataset | e2v+ | MERaLiON | Melati | VP-Whisper | VP-WavLM | Qwen2.5-Omni | Qwen2-Audio | Gemma 4 | Row: done (actual, h) | Row: remaining (plan, h) |
|---|---|---|---|---|---|---|---|---|---|---|
| THAI-SER | 0.88 ✓ (full run; subset reused, 0 h) | 0.52 | 0.80 | 0.46 | 0.87 | 2.60 | 3.57 | 2.33 | 0.88 | 11.15 |
| MEAD | 0.34 ✓ | 0.35 | 0.55 | 0.31 | 0.39 | 1.62 | 2.30 | 1.41 | 0.34 | 6.93 |
| Emozionalmente | 0.17 | 0.30 | 0.51 | 0.27 | 0.20 | 1.31 | 1.90 | 1.14 | 0 | 5.80 |
| CREMA-D | 0.04 ✓ (MPS) | 0.55 ✓ | 0.76 ✓ | 0.42 ✓ | 0.07 ✓ | 1.30 | 1.97 | 1.04 | 1.84 | 4.31 |
| MCAE-SPPS | 0.37 ✓ (MPS) | 0.49 ✓ | 0.69 ✓ | 0.38 ✓ | 0.24 ✓ | 1.08 | 1.67 | 0.89 | 2.16 | 3.64 |
| nEMO | 0.09 ✓ | 0.20 | 0.34 | 0.18 | 0.16 | 0.81 | 1.19 | 0.71 | 0.09 | 3.59 |
| MELD | 0.07 | 0.11 | 0.19 | 0.10 | 0.09 | 0.61 | 0.77 | 0.70 | 0 | 2.64 |
| TESS | 0.38 ✓ | 0.20 ✓ | 0.29 ✓ | 0.16 ✓ | 0.13 ✓ | 0.48 | 0.73 | 0.41 | 1.16 | 1.62 |
| MESS | 0.35 ✓ | 0.13 ✓ | 0.19 ✓ | 0.10 ✓ | 0.13 ✓ | 0.33 | 0.48 | 0.30 | 0.89 | 1.11 |
| RAVDESS | 0.04 ✓ | 0.10 ✓ | 0.15 ✓ | 0.08 ✓ | 0.02 ✓ | 0.29 | 0.41 | 0.26 | 0.39 | 0.96 |
| JL-Corpus | 0.03 ✓ | 0.09 ✓ | 0.12 ✓ | 0.07 ✓ | 0.01 ✓ | 0.22 | 0.32 | 0.19 | 0.32 | 0.73 |
| EMOVO | 0.01 | 0.03 | 0.04 | 0.02 | 0.02 | 0.17 | 0.19 | 0.23 | 0 | 0.71 |
| SAVEE | 0.01 | 0.02 | 0.04 | 0.02 | 0.04 | 0.15 | 0.16 | 0.22 | 0 | 0.66 |
| **Column: done (actual, h)** | **2.51** | **1.55** | **2.20** | **1.21** | **0.60** | 0 | 0 | 0 | **8.07** | |
| **Column: remaining (plan, h)** | **0.26** | **1.53** | **2.47** | **1.36** | **1.77** | **10.97** | **15.66** | **9.83** | | **43.85** |

Grand totals: **done 8.07 h (actual)**, all from the 5 SER models; **remaining 43.85 h (plan)** = SER models 7.39 h
(phase 1) + audio LLMs 36.46 h (phase 2); together 51.92 h.

Where the actual times come from:

- `runtime.seconds` in `outputs/<model>/<dataset>_predictions.json`: every finished MERaLiON, Melati and
  Vox-Profile run (2026-10-28/29, MPS) and emotion2vec+ JL-Corpus, MESS (2026-10-28, CPU) and MEAD (2026-11-06,
  CPU, 1,221.5 s, 153 ms per clip).
- emotion2vec+ CREMA-D, MCAE-SPPS, nEMO, RAVDESS, TESS: the JSON has no runtime, so the report's Hardware line
  (`reports/emotion2vec_plus_large/<dataset>.md`, ms per clip × clips). **CREMA-D (18 ms per clip) and MCAE-SPPS
  (201 ms per clip) ran on MPS**, before `mac_device: cpu`; every other emotion2vec+ run used the CPU.
- emotion2vec+ THAI-SER: 0.88 h is the full run on all 22,584 clips (2026-10-23, CPU, 52.5 min, 140 ms per clip;
  `reports/archive/emotion2vec_plus_large/thai_ser_full.md`); the subset result reused those predictions (subset
  reused, 0 h). It is counted in the done totals (without it: 7.19 h).
- The CPU speed of emotion2vec+ varied between runs (MESS 697 ms and TESS 487 ms per clip on 2026-10-28, MEAD 153 ms
  per clip today), probably other load on the shared Mac (unverified).

### 5.2 Phase 3: API models (company computer)

| Model | Clips | Run time | Cost (plan B, estimate) | Notes |
|---|---|---|---|---|
| GPT Realtime (`gpt-realtime`) | every clip of the 13 datasets: 56,264 (ESD never sent) | ~86 h (3.6 days) one request at a time (measured 5.5 s per clip) | ≈ **$106** ($106.15) | parallel requests shorten it, within the rate limit; the user sets the concurrency on the company computer |
| Gemini | every clip: 56,264 | **not measured** (time per request unknown) | ≈ **$9.87** (gemini-2.5-flash), $12.02 (gemini-3-flash-preview), **$13.21** (gemini-3.5-flash), thinking off | model name in `config/models.yaml` still the placeholder `gemini-2.5-flash` (to be set before the run); loader `scripts/models/gemini.py` not implemented yet |
| **Total API** | 112,528 requests | | ≈ **$116–119** | |

Figures from `docs/audio_llms_survey_en.md` (§3c, plan B: 56,264 clips, 52.2 h; GPT Realtime 1.88M audio / 12.01M
total tokens, +2% retries; Gemini 6.02M / 16.15M). Cost formula (§3b): 1.02 × (audio s × tokens/s × audio price +
N × 175 × text-in price + N × 5 × text-out price) / 10⁶; the fixed text prompt dominates the token count.

- **Cache reuse**: the cached answers of the earlier 200-clip runs (`outputs/<model>/responses.jsonl`, kept on the
  company computer; there is none on the Mac) are reused with the same prompts, so those clips are not paid again.
- **Data needed on the company computer**: the audio of all 13 datasets (ESD never); for THAI-SER, `labels.csv` plus
  the extracted audio (the subset is chosen by the loader from `labels.csv`). MEAD, MELD, EMOVO and SAVEE
  (`public_audio: False`) must not go into the public repo: they need a private channel, e.g. a private data
  repository.

---

## 6. Analyses

| Analysis | Question | Data | Measure | Status |
|---|---|---|---|---|
| Main results | RQ1: ranking of 10 models on 13 datasets (12 if SAVEE is dropped from the paper) | all | accuracy, UA, macro-F1, 4-class accuracy, vs chance; LLMs also vs SER "accuracy with all model classes" | **main** (running) |
| Seen vs unseen | RQ4: how much of each score comes from training overlap | flags of section 2.4; the 4 unseen-by-all datasets as the clean test set | per-model gap seen vs unseen (already 20–52 points for the SER models, per research_angles.md); lineage (e.g. MERaLiON labels partly from emotion2vec+) | **main** |
| Intended vs perceived | RQ3: do models follow actors or listeners? does the ranking change? | CREMA-D (45.6% listener–actor agreement), THAI-SER (77.2%), Emozionalmente (human UAR 66%); MCAE-SPPS only through vote shares (majority = acted for 100%) | acted vs acted-or-listener accuracy, model-vs-listener agreement, share of disputed clips where the model follows the actor (77–82% for actor-trained models on 3,699 disputed CREMA-D clips, 62–65% listener-following for Vox-Profile, per research_angles.md) | **main** |
| Lexical vs acoustic | RQ2: do LLMs use the words? | parallel corpora vs corpora with emotion-bearing words (MEAD, MELD); within THAI-SER scripted (5,012) vs improvised (6,836) | accuracy gap per model type; LLM gap vs SER gap | **main** |
| Neutral bias | do models (especially LLMs) default to *neutral*? | all | neutral rate vs share of neutral clips, neutral recall/precision; compare with LISTEN and VocalAffectBench (45.3% neutral outputs; gpt-realtime 69.6%) | **main** (from the prediction files) |
| Per-intensity | does accuracy rise with acted intensity? | MEAD 3 levels (in the loader's `intensity`, not in reports); also THAI-SER scripted low/high (2,502 / 2,510), RAVDESS normal/strong, CREMA-D IEO low/medium/high (from file names) | accuracy and neutral rate per level | **main** (small script on the prediction CSVs) |
| Professional vs non-professional | does amateur acting hurt? | EMOVO vs Emozionalmente (Italian); nEMO (3 of 9 professional) | accuracy per corpus/speaker group | possible |
| Clip length / padding | do fixed-window models lose on short clips? | all; 31,249 clips < 3 s, 34 > 30 s | accuracy by duration bin per model | possible |
| Prompt sensitivity | does wording change LLM results? | subset | spread over prompt variants | **later** |
| Repeat runs | run-to-run label flips (GPT Realtime has no temperature 0) vs human disagreement | subset with listener votes | flip rate, Fleiss' κ across runs vs vote entropy | **later** |
| Telephone simulation | loss on narrowband audio | public corpora through G.711 / AMR-NB / 8 kHz | accuracy drop per model | **later** |
| Bootstrap CIs and paired differences | how stable are paired model differences? | CREMA-D, RAVDESS and MELD; optional extensions: THAI-SER, Emozionalmente and MEAD | paired speaker-cluster bootstrap; dialogue clusters for MELD | **implemented**, 10,000 percentile resamples; episode sensitivity/extensions later |
| Cascade control (ASR + text LLM) | how much can the words alone do per corpus? | all | accuracy from transcripts only | **later** (optional) |

Statistical-analysis scope agreed on 2026-11-11: begin with CREMA-D (91 speakers), RAVDESS (24 speakers) and
MELD (280 test dialogues). MELD supports dialogue-level stability, not a claim about unseen actors; recurring
characters and episodes may require a coarser episode-level sensitivity analysis. THAI-SER (200 actors),
Emozionalmente (431 actor IDs) and MEAD (48 actors in the current subset) are useful extensions. These group counts
describe the current saved-prediction coverage and can be recovered with `scripts/analysis/probe_metadata.py`.
TESS and DailyTalk have only two actors, and MESS identities are inferred rather than author-verified, so they are
not first choices for new-speaker uncertainty. Keep the full-test-set scores as the point estimates; resampling
estimates their uncertainty under the selected grouping and does not rerun the models.

Implemented in [scripts/bootstrap.py](../scripts/bootstrap.py). Run `python scripts/bootstrap.py` from
`research_paper/` to regenerate [the Chinese report](../analysis_report/bootstrap_zh.md),
[the English report](../analysis_report/bootstrap_en.md), `output/bootstrap.csv` and its source/parameter manifest.
The four existing metrics have individual-model 95% percentile intervals; every available pair has a directly
bootstrapped A−B interval on common clips and shared cluster draws. These fixed-prediction intervals describe
speaker/dialogue-composition uncertainty, not proven unseen-speaker generalization, API repeatability, or
multiplicity-adjusted significance. New compatible saved results are discovered when the script is rerun.

---

## 7. Decisions and open items

### 7.1 The dataset set (2026-11-06)

| Decision | Reason |
|---|---|
| **13 evaluated; the paper may use 12 (3 × 4 grid): SAVEE is the candidate to drop** (open); ESD alone archived (section 2.5) | SAVEE: licence requires registration and forbids redistribution, only an unofficial Kaggle copy, 4 male speakers; but it is the 5th of LISTEN's corpora (direct comparison). It costs little: 0.66 h of local compute left, 480 clips per API model |
| **THAI-SER stays** (high-agreement subset) | the largest corpus (11.2 h of the 43.9 h remaining local plan), but one of only 4 datasets unseen by every model; 200 actors; crowd majority + agreement (RQ3); tonal language; scripted vs improvised (RQ2); used by VoxEmo |
| EMOVO stays | despite the unofficial mirror and no published licence: the only professional Italian corpus (pairs with Emozionalmente), used by VoxEmo; 0.7 h local |

### 7.2 Models deferred

| Model | Why deferred | Notes |
|---|---|---|
| Audio Flamingo 3 (8.27B, NVIDIA OneWay Noncommercial) | training data contain all of JL-Corpus and TESS, and MELD train | easiest to add (transformers native); flag JL-Corpus, TESS seen, MELD possibly seen |
| Qwen3-Omni-30B-A3B-Instruct (Thinker 31.72B, MoE ~3B active; Apache-2.0) | 63.4 GB in bf16: not on the Mac; needs 1×H100 80 GB, or the L40 with a community FP8/4-bit checkpoint (report as quantised) | recommended first among the additions |
| Qwen3.5-Omni Plus / Flash (API only, proprietary) | needs a new streaming loader (Model Studio, `stream=True`) and an international key | plan-B cost $31.25 (Plus) + $8.69 (Flash); qwen3.8-omni-flash +$1.84 |
| Kimi-Audio-7B-Instruct (9.77B, MIT) | CUDA only (flash-attn, torch 2.6), separate environment; SFT data include RAVDESS, MELD, SAVEE | flag RAVDESS, MELD and SAVEE as seen |

### 7.3 Other open items

| Item | Status |
|---|---|
| Gemini model name | `api_model: gemini-2.5-flash` is a **placeholder** in `config/models.yaml`; the loader `scripts/models/gemini.py` is not implemented yet. Pick the current Gemini audio model (and pin its version and date) before the first run |
| GPT Realtime version | pin the snapshot the API reports (saved per answer in `responses.jsonl`) and the run dates |
| Paper A vs Paper B, venue | see [research_angles.md](research_angles.md) and [venues.md](venues.md) (first choice ACL 2027, ARR deadline 2027-01-04) |
| Company approval for publication | open (research_angles.md) |

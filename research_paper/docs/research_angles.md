# Research Angles

Ideas for a paper built on our speech emotion recognition (SER) benchmark: 5 dedicated SER models and 5 audio LLMs on
13 public emotional-speech datasets. Brainstorming stage; rewritten in plain language on 2026-11-06. Chinese version:
[research_angles_zh.md](research_angles_zh.md). See also [related_work.md](related_work.md), [venues.md](venues.md),
[experimental_design.md](experimental_design.md).

**Ground rules.** Public data only, no company data, nothing that could become a product or a patent: we only test
existing models and methods. "Unverified" = not confirmed from a primary source; "estimate" = our calculation;
"preliminary" = from unfinished runs.

**Legend.** 🥇 main contribution 1 · 🥈 main contribution 2 (same weight, more risk) · 🧱 foundation (needed, not new
alone) · 🥉 supporting section with its own results · 🔹 small finding (a paragraph or table) · 🔍 being analysed
elsewhere · 💤 on hold or set aside · 📎 one or two sentences · 🆕 new in this version

## Priority overview

"Mac time" = our Mac M4 Max (48 GB) running the three open audio LLMs (Qwen2-Audio, Qwen2.5-Omni, Gemma 4). API cost
does not matter (user's decision).

| # | Angle | Role | Risk | Mac time | Status |
|---|---|---|---|---|---|
| A | Audio examples in the prompt (in-context learning, ICL) | 🥇 | medium | ≈ 14 h core (17 h with margin) + ≈ 15 h extras | design ready |
| B | Automatic prompt rewriting with GEPA | 🥈 | high | ≈ 30–50 h in total | design ready |
| 0 | Zero-shot benchmark, 10 models × 13 datasets | 🧱 | low | phase 2 ≈ 36 h | running |
| C | Training-data overlap and model family trees | 🥉 | low | none | preliminary numbers |
| D | Same clip, same prompt: does the answer change? | 🥉 | low-medium | 0–6 h per model | design ready |
| E | Words-only control (speech-to-text → text LLM) | 🔹 | low | ≈ 1 h | later |
| F | 🆕 Label probabilities, neutral-bias removal, three readouts | 🔹 | low | none extra | new |
| G | Actor's intended emotion vs what listeners heard | 🔹 | low | none | partly done |
| H | Are confidence numbers honest? (calibration) | 🔹 | low | none | later |
| I | JEV-style structured outputs | 💤 set aside for now (2026-11-06) | — | — | — |
| J | Telephone audio | 💤 | high | 1 SER run per codec | on hold |
| K | Clip length and padding | 📎 | low | none | mention only |

---

## The paper in short

**In one sentence:** can we make audio LLMs hear emotion better just by changing the prompt?

**Why it matters.** It is already known that audio LLMs trail SER models and say "neutral" too often: LISTEN
([EACL 2026](https://aclanthology.org/2026.eacl-long.274/)), VoxEmo ([arXiv 2603.08936](https://arxiv.org/abs/2603.08936)),
EmoBox ([Interspeech 2024](https://arxiv.org/abs/2406.07162)), VocalAffectBench
([arXiv 2608.28932](https://arxiv.org/abs/2608.28932)). So a plain benchmark is not enough. The open question is
where the gap lives. **In the prompt**: the model heard the emotion, but a habit ("when unsure, say neutral") hides
it. Or **in hearing**: the audio encoder never passed the emotion on. Hints point both ways: open audio LLMs keep
speaking style in the audio encoder but lose it before the answer ("Heard but Not Heeded",
[arXiv 2609.00727](https://arxiv.org/abs/2609.00727)), and describing acoustic cues in the prompt helps
([arXiv 2606.07309](https://arxiv.org/abs/2606.07309)); but random examples do not help SER (Piao et al.,
[SLT 2026](https://arxiv.org/abs/2603.20433)), and GEPA on omni models gave only −0.3 to +4.1 points on audio-visual
QA ([arXiv 2610.20085](https://arxiv.org/abs/2610.20085)).

**How.** Two prompt-side tests: 🥇 **examples** (A) and 🥈 **rewritten instructions** (B). Both run on the same test
clips, against the SER models, with the same measures: accuracy, UA (unweighted accuracy: the mean of per-class
accuracies, so a big class cannot dominate), macro-F1, **neutral rate** (share of "neutral" answers), **accuracy
without neutral** (do gains only come from saying "neutral" less?) and the **prior-correction baseline** (F). C, D and
E check that the numbers can be trusted.

| RQ | Question | Section |
|---|---|---|
| RQ1 | How big is the zero-shot gap per corpus, and how much is the "neutral" habit? | 🧱 0 |
| RQ2 | Which audio examples help, and does the model link each example's sound to its label? | 🥇 A |
| RQ3 | Do optimised instructions help? Does a coach that *hears* the errors beat one that only *reads* them? | 🥈 B |
| RQ4 | Do gains carry over to other corpora, languages and models? Do A and B add up? | A + B |
| RQ5 | What are we measuring: training overlap, word clues, run-to-run noise? | C, D, E |

**Contributions.** (1) The first controlled study of audio ICL for SER, on 5 parallel corpora (the words give no
clue; 4 unseen by every model with published training data), 3 open and 2 API audio LLMs. (2) The first use of GEPA on
audio LLMs for a paralinguistic task (how something is said, not what), including a coach that can hear. (3) A
10 × 13 benchmark with seen vs unseen corpora, a noise floor and a words-only control.

**Titles**: "Can You Prompt an Audio LLM to Hear Emotion? Demonstrations vs Optimised Instructions on 13 Corpora";
"Steering Audio LLMs Through the Prompt for Speech Emotion Recognition". **Outline** (8 pages, ACL long): setup; zero-
shot gap (RQ1); examples (RQ2); instructions (RQ3); transfer (RQ4); what are we measuring (RQ5, actor vs listener
labels); Limitations. **Maybe later**: C, G, H, J as a second paper for a speech venue (Interspeech or TASLP; see
[venues.md](venues.md)).

**Closest work and what it leaves open**

| Work | What it did | Open for us |
|---|---|---|
| LISTEN ([EACL 2026](https://aclanthology.org/2026.eacl-long.274/)) | 6 audio LLMs; models "transcribe rather than listen" | no SER models, examples or prompt optimisation; gold transcripts |
| VoxEmo ([arXiv 2603.08936](https://arxiv.org/abs/2603.08936)) | 35 corpora; > 20-point macro-F1 swings across prompts; 5-prompt vote | no closed models, no optimisation, no repeated calls |
| VocalAffectBench ([arXiv 2608.28932](https://arxiv.org/abs/2608.28932)) | gpt-realtime-2 27.9%, 45.3% of answers "neutral" | one in-house corpus (280 clips) |
| EmoBox ([Interspeech 2024](https://arxiv.org/abs/2406.07162)) | 32 corpora, frozen features | no LLMs |
| Piao et al. ([SLT 2026](https://arxiv.org/abs/2603.20433)) | 1–8 random examples on CREMA-D, 6 audio LLMs; examples fix format, not the task | no same-speaker, balanced or retrieved examples |
| Wong et al., A*STAR ([ASRU 2025](https://doi.org/10.1109/ASRU65441.2025.11434787)) | "some degree" of paralinguistic ICL | no controls, no closed models |
| AudioICL-Bench ([SLT 2026](https://arxiv.org/abs/2609.11252)) | audio LLMs learn new sound→label pairs when hearing is easy | no emotion |
| GEPA for audio-visual QA ([arXiv 2610.20085](https://arxiv.org/abs/2610.20085)) | −0.32 / +0.95 / +4.12 points | not emotion, no hearing coach |
| GEPA ([ICLR 2026 oral](https://arxiv.org/abs/2507.19457)) | text tasks; beats GRPO by 6% with up to 35× fewer model calls | never on audio |

---

## Shared setup for A and B

**Models.** Mac: Qwen2-Audio-7B-Instruct (8.40B), Qwen2.5-Omni-7B (Thinker, 8.93B loaded), Gemma 4 E4B-it (8.00B,
4.5B effective). API (company computer): GPT Realtime and Gemini (gemini-3.8-flash). The 5 SER models are the
reference and are not prompted.

**Corpora.** The 5 parallel corpora: every sentence is spoken in every emotion, so the words cannot give the answer.
The other 8 corpora are for transfer tests.

| Corpus | Language | Speakers | Classes | Why |
|---|---|---|---|---|
| CREMA-D | English | 91 | 6 | Piao et al.'s corpus; 4–12 listener votes per clip |
| MCAE-SPPS | Mandarin | 6 | 7 | unseen, newest (2025/26); 40 listeners per clip |
| nEMO | Polish | 9 | 6 | unseen; mostly non-professional speakers |
| THAI-SER (scripted, high-agreement, 5,012 clips) | Thai | 200 | 4 | unseen; tonal language |
| MESS | English | 6 | 4 (calm = neutral) | unseen; calm vs neutral |

**Splits** (🆕, one file per corpus; no speaker in two parts): train (example pool and GEPA feedback) / validation
(GEPA scoring) / test speakers = CREMA-D 61 / 15 / 15; THAI-SER 140 / 30 / 30; nEMO 5 / 2 / 2; MCAE-SPPS and MESS
4 / 1 / 1 (2 folds, swapping the validation and test speaker).

**Test clips.** 500 per corpus, balanced by emotion and speaker, **the same clips in every condition**, so each
comparison is paired (McNemar test, paired bootstrap). One accuracy on 500 clips has a standard error of about
2 points; over 2,500 clips a paired test sees differences of about 2–3 points.

**Mac speed** (measured 2026-11-06, zero-shot, one clip at a time): Gemma 4 **0.47 s**, Qwen2.5-Omni **0.66 s**,
Qwen2-Audio **1.00 s** per clip. One second of audio ≈ 25 tokens, so a 2.5 s example clip adds roughly 0.13 s to 30% of
a call (two estimates, within ~30% of each other; [experimental_design.md](experimental_design.md) §5.1). Qwen2-Audio
pads every clip to 30 s (`padding="max_length"`), about +0.15 s per example (unverified).

| Model | k = 0 (measured) | k = 1 | k = 3 | k = 5 | k = 7 |
|---|---|---|---|---|---|
| Gemma 4 E4B | 0.47 s | ≈ 0.6 s | ≈ 0.9 s | ≈ 1.2 s | ≈ 1.5 s |
| Qwen2.5-Omni | 0.66 s | ≈ 0.8 s | ≈ 1.1–1.3 s | ≈ 1.3–1.7 s | ≈ 1.6–2.1 s |
| Qwen2-Audio | 1.00 s | ≈ 1.3 s | ≈ 1.8–1.9 s | ≈ 2.3–2.5 s | ≈ 2.8–3.1 s |

k = number of example clips (≈ 2.5 s each, capped at 4 s). MPS recompiles for every new input length, so budget ×1.2;
padding examples to a fixed 3.0 s and reusing a shared example prefix reduce this.

---

## 🥇 A. Audio examples in the prompt (controlled in-context learning)

**In one sentence:** the model hears a few labelled clips before the test clip, and controls show whether it really
learns from them or just copies the label list.

**Why it matters.** Piao et al. found that random examples do not help. Nobody tried what a person would find useful:
**the same speaker** in each emotion. Emotion is relative: a high voice is normal for one person and excited for
another. If the model can use "this is how *this* speaker sounds when calm", that is new. If it cannot, our controls
say **why**.

**What "k-shot" means.** k = the number of labelled example clips before the test clip; k = 0 is the plain benchmark.
Nothing is trained, the examples only sit in the prompt:

```text
[system]    ... Answer with exactly one word: angry, disgusted, fearful, happy, neutral, sad, surprised.
[user]      Example 1: <audio: example clip 1>  Which emotion does the voice express?
[assistant] angry
[user]      Example 2: <audio: example clip 2>  Which emotion does the voice express?
[assistant] sad
[user]      <audio: TEST clip>  Which emotion does the voice express?
[assistant] → scored
```

Our zero-shot prompt already gets the format right (no invalid answers in the 18-clip pilot of 2026-11-05), so a gain
cannot come from fixing the format, unlike in Piao et al.

### How

| Factor | What changes | What it tells us |
|---|---|---|
| **k** | 0, 1, 3, 5, one per class | more examples should help more, if they carry information |
| **Balanced vs random** | one per emotion vs k random | random examples tilt the model toward labels it just saw ([Zhao et al. 2021](https://arxiv.org/abs/2102.09690)) |
| **Same vs other speaker** | the test speaker's other recordings vs another speaker of the same gender | can it use the speaker's normal voice as a reference? Helps after special training ([Ihori et al., ASRU 2025](https://arxiv.org/abs/2509.08344): +8.2 points at 7 shots), never tested on off-the-shelf LLMs. **Main novelty of A** |
| **Retrieved vs random** | the clips that sound most like the test clip | macro-F1 0.30 → 0.49 for Voxtral on one corpus ([Huang et al. 2026](https://arxiv.org/abs/2606.14784)); works for Whisper ASR ([Wang et al., ICASSP 2024](https://arxiv.org/abs/2309.07081)) |
| **Shuffled labels** (control) | same audio, labels swapped | no drop = the model ignores which sound goes with which label |
| **Labels without audio** (control) | "Example 1: [no audio] → angry" | "seeing labels" vs "hearing examples" |
| **Order** (control) | same examples reordered | how often the answer just copies the last example's label |

**Two traps.** (1) **Odd one out**: if the examples are the test speaker saying the *test sentence* in every other
emotion, the correct emotion needs a different sentence and gives itself away. So we use a **same-speaker reference
set** (🆕): the test speaker saying **one other sentence** in each emotion; the other-speaker set is built the same
way, so the two differ **only** in who speaks. (2) **Retrieval can leak the label** if the search uses an emotion model
(emotion2vec+). So we report a **copy baseline** (🆕): "take the majority label of the retrieved clips", without the
LLM. The LLM must beat it. We also retrieve with a non-emotion embedding (speaker or eGeMAPS features).

**Staged design** (each test clip gets its own random examples, fixed seed; calls per model):

| Stage | Conditions | Clips | Calls | Mac hours (Gemma / Omni / Qwen2-Audio) |
|---|---|---|---|---|
| 0. Multi-audio check | 20 prompts with 3 clips and a checkable question ("transcribe clip 2") | — | ≈ 50 | minutes |
| **1. Core** (RQ2) | C0 zero-shot (reused); C1 random k = 3, other speakers (repeats Piao); C2 balanced, other speaker; C3 balanced, **same speaker**; C4 = C3 with shuffled labels | 5 × 500 | 10,000 | 3.2 / 4.0 / 6.5 h; **13.7 h** (≈ 17 h with ×1.2) |
| 2. Extras | random k = 1, 5; retrieved k = 3 (two embeddings + copy baseline); labels without audio; reordered; **neutral anchor** (🆕: one same-speaker neutral clip); 2 per class | 3 × 500 | 10,500 | 3.3 / 4.3 / 6.9 h; 14.5 h (8–11 h for the best two) |
| 3. APIs | Stage 1 + chosen extras | as above | ≈ 20,500 | company computer; GPT Realtime ≈ 5.5 s per clip, run in parallel |
| Optional | majority vote over 3 example sets (Wong et al.) | 2 × 500 | 3,000 | 1–2.5 h per model |

Stage 1 answers the core questions: do examples help (C1, C2 vs C0)? Does the speaker matter (C3 vs C2)? Is the
sound-to-label link used (C4 vs C3)? Stage 2 uses CREMA-D, MCAE-SPPS and THAI-SER.

**Several clips in one prompt?** Qwen2-Audio **yes** ([card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct);
each clip padded to 30 s). Qwen2.5-Omni **yes** ([card](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)). Gemma 4
**probably** ([card](https://huggingface.co/google/gemma-4-E4B-it) is silent; our processor supports it; its local
layers see only 512 tokens back, so early examples reach only the global layers). Gemini **probably** (up to 9.5 h of
audio, 20 MB inline; [docs](https://aistudio.google.com/docs/audio)). GPT Realtime **to check** with one test call.
Audio LLMs often struggle with several clips ("Beyond Single-Audio", [EMNLP 2024 Findings](https://arxiv.org/abs/2409.18680)),
hence Stage 0.

**The two closest papers.** Piao et al., "When Demonstrations Fail" ([SLT 2026](https://arxiv.org/abs/2603.20433)):
6 audio LLMs incl. Qwen2-Audio, Qwen2.5-Omni and Gemini 2.5 Flash; examples "improve format compliance but fail to
improve the core task performance"; with shuffled labels "SER performance remains comparable"; hiding the example
labels drives SER to near zero, hiding the example audio does not. Our C1 + C4 on CREMA-D repeat their key result; the
rest is new. Wong, Huzaifah, Chen, Aw (A*STAR) ([ASRU 2025](https://doi.org/10.1109/ASRU65441.2025.11434787),
[record](https://oar.a-star.edu.sg/communities-collections/articles/22939)): open models "already exhibit some degree
of speech in-context learning" on stutter, pronunciation and SER; details not public, so **get the paper from IEEE
Xplore before writing**. Both findings can be true; our controls decide. A*STAR is in Singapore: likely reviewers.

**Possible results (made up).** Speaker reference works: C3 beats C2 on unseen corpora, C4 falls back to C0.
Examples only shift habits: the neutral rate drops, but prior correction (F) gives the same gain. Nothing helps: the
bottleneck is hearing, not the prompt.

**Compute and time on our Mac:** Stage 1 ≈ 14 h (two nights); Stage 2 ≈ 15 h.

**Risks:** "no gain" is likely, but with these controls still a clear answer; multi-audio may break on Gemma 4 (Stage 0
catches it).

---

## 🥈 B. Automatic prompt rewriting with GEPA

**In one sentence:** a second LLM reads the audio LLM's mistakes and rewrites the instructions, round after round; does
this help it hear emotion?

**Why it matters.** Prompt wording alone moves SER macro-F1 by more than 20 points (VoxEmo), and audio LLMs say
"neutral" far too often (in our pilot Gemma 4 answered mostly "neutral" although it transcribed the speech correctly).
A prompt can change exactly these things. **Nobody has tried** a prompt optimiser on audio LLMs for SER or any
paralinguistic task (searched 2026-11-06).

**What GEPA is, in plain words** (Agrawal et al., [ICLR 2026 oral](https://arxiv.org/abs/2507.19457)). It improves the
text of the prompt; the weights never change. One round: (1) run the audio LLM with the current prompt on a few
training clips; (2) a **reflection LM**, a second LLM acting as a coach, reads the prompt, the answers and feedback
("correct: angry; you said neutral") and writes a better prompt; (3) if it does better, score it on the validation
set; (4) keep the **Pareto front**: every prompt that is the best on at least one validation clip. Think of a team of
specialists: one prompt is great on angry clips, another on sad ones; GEPA keeps them all and builds on them instead
of polishing one. (5) Stop when the **rollout budget** is used up (a rollout = one audio-LLM call on one clip).

Paper numbers (text tasks, Qwen3-8B and GPT-4.1 mini): **1,839–7,051 rollouts** per run, mostly for validation (only
**79–737** training rollouts); **17–90 coach calls**; beat GRPO (24,000 rollouts) by 6% on average (up to 20%) and
MIPROv2 by more than 10%; a prompt optimised on one model still gave +9.0% on another (vs +12.2% native). DSPy
recommends a stronger coach than the task model ([GEPA in depth](https://dspy.ai/current/diving-deeper/gepa-in-depth/)).

**Known to work for audio? No.** The closest study ([arXiv 2610.20085](https://arxiv.org/abs/2610.20085), audio-visual
QA, 2,500 trials) found Qwen3-Omni Instruct **−0.32**, Qwen3-Omni Thinking **+0.95**, Gemini 2.5 Flash **+4.12**
points. So "no gain" is a real possibility; a controlled negative result still pairs well with A ("neither examples
nor instructions fix hearing").

**What it can change**: instructions, label definitions, cue descriptions ("anger: loud, tense, fast"; see
[Juslin & Laukka 2003](https://doi.org/10.1037/0033-2909.129.5.770)), decision rules ("calm counts as neutral"),
language notes ("Thai is tonal"). **What it cannot**: what the audio encoder delivers.

### How

| Coach | Sees per example | Risk |
|---|---|---|
| **R-text**: strong text LLM (API) | prompt, answer, acted label, **listener votes** (🆕, "angry 5/9, neutral 4/9"), transcript, pitch/loudness/speed vs the speaker's average (eGeMAPS) | cannot hear; clichéd cues |
| **R-audio**: audio LLM (Gemini Pro or a GPT audio model) | the same **plus the error clips** and one correct clip | hears emotion badly itself (gpt-realtime-2 27.9%); audio goes to an API (allowed for all 13 corpora, never ESD) |
| R-self (optional): the audio LLM itself | as R-text | weak 7–8B coach |

- **Sets**: feedback ≈ 200 balanced clips (train speakers); scoring 105–150 balanced clips (validation speakers), so
  "always neutral" cannot win. Feedback never contains validation or test labels.
- **Score** 1 / 0 per clip; minibatch of 7 (one per emotion), so the coach always sees confusions.
- **Optimise on** CREMA-D and MCAE-SPPS (THAI-SER if time allows). **Budgets** 1,000 / **2,500 (main)** / 7,000
  rollouts. Three seeds on one cell (CREMA-D, Qwen2-Audio, R-text) to see the spread.
- **Baselines**: our prompt (`prompts.yaml`); hand edits (+ label definitions, + cue descriptions, + "do not default to
  neutral", short chain of thought); VoxEmo's 5-prompt vote; random paraphrase search at the same budget (does the
  coach matter?); best ICL condition and GEPA + best examples (RQ4); prior correction (F); SER models.
- **Transfer (RQ4)**: to all 13 corpora (the 4 unseen are key); English vs Mandarin source tested on Thai, Polish,
  Italian; one model's prompt on the other models and APIs; 7-class prompts on 4-class corpora.

**Compute and time on our Mac** (the main worry). Long prompts and recompiling give ≈ 0.8 / 1.05 / 1.5 s per rollout
(Gemma / Omni / Qwen2-Audio): 2,500 rollouts = 0.6 / 0.7 / 1.0 h; 7,000 = 1.6 / 2.0 / 2.9 h. Testing one prompt on
13 × 500 clips = 1.4 / 1.9 / 2.7 h. Coach calls cost a few dollars.

| Block | Runs | Mac hours |
|---|---|---|
| Pilot | 1 small run (CREMA-D, Gemma 4, R-text) | 0.5 h |
| Main grid | 3 models × 2 corpora × 2 coaches = 12 runs (+ 2 seeds) | 12–15 h |
| Test 12 prompts on 13 corpora | 12 × 6,500 calls | ≈ 24 h (≈ 15 h with 300 clips per corpus) |
| Cross-model transfer | best prompt per model on the others | 8–13 h |
| **Total** | | **≈ 30–50 h**, about a week of nights and weekends |

Engineering: a custom adapter for the [gepa](https://github.com/gepa-ai/gepa) library; GEPA starts after phase 2. GPT
Realtime needs ≈ 4 h per 2,500 rollouts unless run in parallel. Not feasible on the Mac: large grids, 30B models
(Qwen3-Omni), GRPO.

**Risks:** may not help 7–8B models; clichéd cue lists; the R-audio coach hears emotion badly.

---

## Do we need to compare against GRPO?

**In one sentence:** no; cite published results, add one small LoRA fine-tuning run only if a GPU is free, and name
GRPO as future work.

**Why reviewers might ask.** GEPA's headline is "beats GRPO". GRPO is reinforcement learning: the model answers each
clip several times, gets a reward (1 if right), and its weights are updated. But our question is prompting vs
*trained SER models*, and those already serve as the "trained" reference.

**What it takes.** R1-AQA ([arXiv 2503.11197](https://arxiv.org/abs/2503.11197)): GRPO on Qwen2-Audio, **8 × H800**,
64.5% on MMAU test-mini. Omni-R1 ([arXiv 2505.09439](https://arxiv.org/abs/2505.09439)): much of the gain came from
text reasoning. Emotion RL: EMO-RL ([EMNLP 2025 Findings](https://arxiv.org/abs/2509.15654)), EmotionThinker
([ICLR 2026 oral](https://arxiv.org/abs/2601.15668)), R1-Omni ([arXiv 2503.05379](https://arxiv.org/abs/2503.05379)),
Yu et al. ([arXiv 2603.08230](https://arxiv.org/abs/2603.08230)), FSA-GRPO ([arXiv 2606.02615](https://arxiv.org/abs/2606.02615)).
On our Mac it is not realistic: ≈ 17 GB for the model alone, no vLLM on Apple GPUs, and 24,000 rollouts are 7–10 h of
sampling before any training.

**Decision.** Always: cite the RL numbers (marked as other setups) and list GRPO as future work. Only if a CUDA GPU is
free by early December: one **LoRA-SFT** run (ordinary fine-tuning on the label) on Qwen2-Audio with **the same
labelled clips GEPA sees**, ≈ 1–2 GPU hours. It asks "same data: prompt or weights?". For one right/wrong label, GRPO
is close to training a classifier, so SFT is the simpler reference.

---

## 🧱 0. The zero-shot benchmark

**In one sentence:** 10 models × 13 corpora with one fixed prompt; everything else builds on it.

**Why it matters.** Not new alone, but it measures the gap A and B try to close (RQ1), and its per-clip outputs feed C,
D, F, G and H. **Status**: 3 SER models done on all 13; Vox-Profile running; the 3 local LLMs next (phase 2, ≈ 36 h);
GPT Realtime and Gemini in phase 3 on the company computer.

**🆕 What phase 2 saves besides the answer**, from the same model call (no extra runs; `config/llm.yaml`):

- the **7 label probabilities**: how likely each label is as the first word of the answer, rescaled to sum to 1;
- the **mean-pooled hidden states**: the model's internal numbers over the audio part, averaged into one vector per
  clip, from 4 layers (float16, ≈ 20–33 kB per clip, git-ignored).

So analysis F and the linear-probe comparison need no extra runs.

---

## 🥉 C. What do SER benchmarks measure? Training-data overlap and family trees

**In one sentence:** emotion2vec+ scores 28 points higher on corpora it trained on than on new ones, so many
"zero-shot" numbers are partly memory.

**Why it matters.** Contamination = the test data was in the training data, like a student who saw the exam. Without
a seen/unseen split, nobody can judge the gains of A and B. Nobody has audited released SER models this way (closest:
ASR test sets, [arXiv 2505.22251](https://arxiv.org/abs/2505.22251); speaker leakage,
[arXiv 2610.33952](https://arxiv.org/abs/2610.33952)).

**How.**

- **Seen vs unseen.** Only **MCAE-SPPS, MESS, THAI-SER and nEMO** are unseen by every model with published training
  data ([experimental_design.md](experimental_design.md) §2.4). emotion2vec+ saw the other 9; MERaLiON saw CREMA-D and
  MELD; Melati saw JL-Corpus and Emozionalmente; Vox-Profile none. The audio LLMs publish no training data, and "not
  published" is not "unseen".
- **Family trees.** Several models learned from labels made by emotion2vec, acting as a teacher: emotion2vec+
  ([card](https://huggingface.co/emotion2vec/emotion2vec_plus_large)), SenseVoice ([arXiv 2407.04051](https://arxiv.org/abs/2407.04051)),
  MERaLiON-SER ([arXiv 2511.04914](https://arxiv.org/abs/2511.04914)), C²SER ([arXiv 2502.18186](https://arxiv.org/abs/2502.18186)).
  A model that never saw CREMA-D can still inherit the habits of a teacher that did.
- 🆕 **Release-date test** for the LLMs: a corpus published after a model came out cannot be in its training data
  (MCAE-SPPS, 2025/26, is newer than Qwen2-Audio, 2024-07).

**Preliminary numbers** (3 of 5 SER models, `reports/*/summary.json`, 2026-11-06, mean accuracy):

| Model | 9 corpora seen by emotion2vec+ | 4 unseen corpora | Difference |
|---|---|---|---|
| emotion2vec+ large (saw the 9) | 84.5% | 56.2% | **+28.3** |
| MERaLiON-SER-v1 (saw 2 of 9) | 54.0% | 60.4% | −6.4 |
| Melati (saw 2 of 9) | 56.8% | 56.4% | +0.4 |

For the other models the 9 corpora are not easier, so the gap is exposure, not difficulty (chance-adjusted: 0.82 vs
0.45). Single corpora agree: CREMA-D 86.5% and 84.7% (seen) vs 51.4% (Melati, unseen); JL-Corpus 99.5% (Melati, seen)
vs 75.9%; Emozionalmente 86.9% and 85.0% (seen) vs 47.4%. Family is not fate: on MESS, MERaLiON (74.5%) far beats its
teacher emotion2vec+ (40.8%). Final design: a "who saw what" table, a family-tree figure, and the 4 unseen corpora as
the recommended clean test set.

**Compute and time:** none. **Risks:** the unseen group has two 4-class corpora (chance 20.2% vs 15.6%); the seen
group includes MELD (TV speech); no significance tests yet.

---

## 🥉 D. Same clip, same prompt: does the answer change?

**In one sentence:** send each clip 10 times, count how often the answer changes, and check whether the model wavers
on the clips where listeners disagree.

**Why it matters.** Both APIs are random by design: GPT Realtime has no temperature setting at all, and
gemini-3.8-flash runs at temperature 1.0 by default with thinking that cannot be turned off (lowest level: LOW). So
one API run is noisy, and since neither API returns logprobs, repeating the call is also our only way to get label
probabilities for them (see F). Gains from A or B must beat this **noise floor**. Nobody has compared audio-LLM answer flips with
per-clip listener votes (close: "Scaling Ambiguity", [arXiv 2601.14620](https://arxiv.org/abs/2601.14620); errors grow
with listener disagreement, [arXiv 2602.03873](https://arxiv.org/abs/2602.03873); GPT-4o on music metadata,
[arXiv 2508.12626](https://arxiv.org/abs/2508.12626)).

**How.** APIs: 10 identical calls per clip. Open models: the label probabilities from F, plus 10 real samples on one
model as a check. Humans: votes in CREMA-D (mean 9.2 per clip), MCAE-SPPS (40), THAI-SER (3–11), Emozionalmente (5).
Measures:

- **flip rate** = 1 − share of the most common answer (10 × angry = 0; 6 angry + 4 neutral = 0.4);
- **wavering vs disagreement**: Spearman correlation, and AUROC for spotting disputed clips (0.5 = guessing,
  1 = perfect);
- **Jensen–Shannon divergence** between model and vote distributions (0 = identical);
- **Fleiss' κ** (agreement beyond chance: 0 = chance, 1 = always agree) across the 10 runs, next to the listeners' κ;
- one call vs majority of 10; do flips cluster on "neutral"?

**Compute and time.** 2,000 clips (500 × 4 corpora). APIs: 20,000 calls per model (GPT Realtime ≈ $38). Open models:
probabilities free from phase 2; 10 samples ≈ 2.6 / 3.7 / 5.6 h.

**Risks:** the link with human disagreement may be weak; then D stays a noise floor.

---

## 🔹 E. Words-only control: speech-to-text, then a text LLM

**In one sentence:** turn each clip into text and let a text LLM guess the emotion: how much do the words alone give
away?

**Why it matters.** A model that "solves" a corpus from the words is reading, not listening. LISTEN showed this with
gold transcripts (text-only answers "neutral" 82.5–96.6% of the time). New for us: a real Whisper → text-LLM pipeline
on all 13 corpora, including Mandarin, Thai, Polish and Italian; a per-corpus **word-clue score** for corpora that are
not fully parallel (MEAD, MELD, SAVEE, THAI-SER improvised); and a check on A and B. AHELM ranked such a pipeline 6th
among audio models ([arXiv 2508.21376](https://arxiv.org/abs/2508.21376)).

**Compute and time:** Whisper on ≈ 6,500 clips, about an hour. **Risks:** low; speech-to-text quality on Thai and
Polish.

---

## 🔹 F. 🆕 Label probabilities, removing the neutral bias, and three readouts

**In one sentence:** besides the one word the model says, we read how likely it finds each label, which shows
whether it "heard it but habitually said neutral".

**Why it matters.** Now the model says one word, e.g. "neutral". With the saved probabilities we also see its inner
ranking, e.g. neutral 45%, angry 40%, the rest 15%. That answers: did it hear the anger but fall back on "neutral" out
of habit? It is the cleanest test of "habit vs hearing", and the yardstick for A and B.

**How.**

1. **Label probabilities**, saved for every clip in phase 2: no sampling, no parsing errors. When the model says
   "neutral", how often is the right label second?
2. **Prior correction** (removing the habitual lean). Measure the model's average probability per label on the train
   split, e.g. "neutral" gets 50% where a fair share is 14%. Divide each clip's probabilities by these averages and
   take the top label again ("contextual calibration", [Zhao et al., ICML 2021](https://arxiv.org/abs/2102.09690)).
   If this cheap fix gives the same gain as A or B, those methods only removed a habit.
3. **Three readouts per clip**: (a) the **spoken answer**; (b) the **top label probability**, with and without prior
   correction; (c) a **linear probe**: a tiny classifier (one linear layer) trained on the saved hidden states of
   train speakers and tested on test speakers. It asks: "is the emotion stored inside the model, even if it does not
   say it?"

**Reading the result.** If (c) comes close to the SER models while (a) is far behind, the model inside **did** hear
the emotion and only fails when "saying" it (as in "Heard but Not Heeded"). If (b) with correction closes most of the
gap, it is mostly the neutral habit. If even (c) is weak, the emotion never got through the audio encoder, and no
prompt can fix it.

**API models** give no hidden states, so no (c). They give no answer probabilities either (checked 2026-11-06, see
[api_logprobs.md](api_logprobs.md)). GPT Realtime has no logprobs for its answer: its only `include` option gives the
probabilities of the input-audio transcript, made by a separate model. gemini-3.8-flash returns no logprobs at all;
Google dropped them for 3.x models on purpose. So for the APIs we use D's repeated sampling instead (e.g. N = 10 on a
subset): 6 "angry" out of 10 ≈ 60%.

**Compute and time:** nothing extra; probes train in minutes on the CPU. **Risks:** labels that share a first token
need multi-token scoring; probes on corpora with few speakers (MCAE-SPPS, MESS) may overfit.

---

## 🔹 G. Actor's intended emotion vs what listeners heard

**In one sentence:** do models reproduce what the actor meant or what listeners heard, and does the ranking change?

**Why it matters.** Acted labels are not the whole truth; a model that "fails" a clip may hear what people hear.
CREMA-D: the listener majority matches the actor for only 45.6% of 6,798 clips; on 3,699 disputed clips, actor-trained
models follow the actor in 77–82%, podcast-trained Vox-Profile follows the listeners in 62–65%. MCAE-SPPS: majority =
acted label for **100%** of clips, so use the **vote shares**. THAI-SER: even in the high-agreement subset the majority
differs from the label for **2,698** of 11,848 clips (1,676 "frustrated", 904 "neutral"). Emozionalmente: human UAR
66%. Closest work: Label Authority in SER ([arXiv 2610.26446](https://arxiv.org/abs/2610.26446)).

**Compute and time:** none. **Risks:** low; one paragraph and a table.

---

## 🔹 H. Are the confidence numbers honest? (calibration)

**In one sentence:** when a model says "80% angry", is it right about 80% of the time, and do its probabilities look
like the listeners' vote shares?

**Why it matters.** A model that is confidently wrong is worse than one that knows when it is unsure. **How**: compare
SER outputs, F and D with the votes of 4 corpora, using **ECE** (expected calibration error: the average gap between
"how sure" and "how often right"; 0 = honest) and Jensen–Shannon divergence. Partly covered:
[Chou et al., Interspeech 2023](https://www.isca-archive.org/interspeech_2023/chou23_interspeech.html),
[Kuan et al. 2026](https://arxiv.org/abs/2604.25591).

**Compute and time:** none (uses D and F). **Risks:** low; one table.

---

## 💤 I. JEV-style structured outputs: set aside for now (2026-11-06)

JEV is a closed text and image product without audio, and a JEV-style speech model already exists (ASOM,
[arXiv 2610.22401](https://arxiv.org/abs/2610.22401)). What we keep is analysis F (the three-readout comparison).
Background: [jev_analysis.md](jev_analysis.md).

## 💤 J. Telephone audio

Loss on phone-quality audio (G.711, AMR-NB, 8 kHz)? Mostly open (EMO-Codec, [APSIPA 2024](https://arxiv.org/abs/2407.15458));
our G.711 simulation exists. On hold: the user doubts a clear result, and it does not fit the prompting story.

## 📎 K. Clip length and padding

Whisper-based SER models pad every clip to 30 s (or 15 s), and MERaLiON averages over the padding; 31,249 of our 56,264
clips are under 3 s. One sentence in Limitations.

**Dropped**: prompt sensitivity alone (VoxEmo; "Hearing the Order", [arXiv 2510.00628](https://arxiv.org/abs/2510.00628));
cross-language alone; chain of thought (now a B baseline); LLM-as-judge ([EMNLP 2025 Findings](https://aclanthology.org/2025.findings-emnlp.25/));
giving the LLM a SER model's prediction as a hint (product-like).

---

## What still has to be built

| Item | For | Status |
|---|---|---|
| Finish the benchmark (Vox-Profile, phase 2 ≈ 36 h, phase 3 APIs) | all | running |
| Save 7 label probabilities and mean-pooled hidden states per clip (3 local LLMs) | F, D, H | 🆕 set, runs with phase 2 |
| Logprobs check for GPT Realtime and Gemini ([api_logprobs.md](api_logprobs.md)) | F, D | 🆕 done: neither gives them; use repeated sampling |
| Pin the GPT Realtime model: `gpt-realtime` shuts down on 2027-01-20; the official successor is `gpt-realtime-2.1` | all | 🆕 decision pending with the user |
| Neutral rate, accuracy without neutral, bootstrap and paired tests in `evaluate.py` | all | open |
| Split files and 500-clip test lists; multi-audio loaders; example pickers | A, B, F | new |
| Prior correction and linear probes | F | new |
| GEPA adapter around `Predictor`; coach access; transcripts, eGeMAPS, vote tables | B, E, G | new |
| LoRA-SFT reference on a CUDA GPU | GRPO question | optional |
| Release code, splits, prompts, per-clip predictions; no restricted audio | paper | at submission |

## Open decisions

1. ACL 2027 (ARR deadline 2027-01-04) or EMNLP 2027 (ARR May 2027) for more time.
2. Which text and audio LLM as coaches; company policy per API (all 13 corpora allowed; ESD never).
3. All three open models or the best two for A Stage 2 and B (Gemma 4's constant "neutral" is the most interesting case).
4. 500 clips per corpus or full corpora for the final tables.
5. LoRA-SFT only with a CUDA GPU by early December.
6. SAVEE in or out ([experimental_design.md](experimental_design.md) §7.1).
7. GEPA on GPT Realtime, or only transfer prompts to it.
8. Company approval process and timeline.

## Timeline to ACL 2027 (estimate)

| Weeks | Work |
|---|---|
| Nov 6–13 | phase 2 on the Mac (with probabilities and hidden states); splits; multi-audio loaders; Stage 0; phase 3 |
| Nov 13–20 | A Stage 1; GEPA adapter and pilot; first F analysis |
| Nov 20 – Dec 4 | B grid and transfer; A Stage 2; D on the APIs |
| Dec 4–18 | C, E, G tables; figures; statistics |
| Dec 18 – Jan 4 | writing, Limitations, checklist; designated service contributor ([venues.md](venues.md) §2.3) |

Tight but possible if the Mac is free from about 10 November; if A or B slips, EMNLP 2027 reuses the plan.

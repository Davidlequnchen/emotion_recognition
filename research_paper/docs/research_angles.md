# Research angles: what context helps audio LLMs judge emotion?

Rewritten 2026-11-09. This is an **exploration plan, not an established paper claim**. Small pilots should determine
which question deserves further work. Chinese version: [research_angles_zh.md](research_angles_zh.md). See also
[datasets_survey.md](../../emotion_recognition/docs/datasets_survey.md), [related_work.md](related_work.md) and
[venues.md](venues.md). [experimental_design.md](experimental_design.md) retains the older design; check the new
scope here before implementing it.

The project adopts 15 datasets: the original 13 have SER baselines; audited DailyTalk and EmotionTalk await baselines. There are five dedicated SER models. The three open-weight audio LLMs are
Gemma 4 E4B, Qwen2.5-Omni 7B and Qwen2-Audio 7B. API models are possible later checks. Use public research data
permitted for each experiment, without company data.

## 1. What are we trying to find out?

**For the same target audio, how does additional evidence change the model's emotion judgment? Is that change
reasonable, and does it agree with human judgments under the same evidence?**

Poor SER performance by audio LLMs and limited gains from a few demonstrations have precedents. More models and
corpora do not necessarily answer a new question. We prioritise two specific kinds of additional evidence:

| Evidence | Plain-language meaning | Difference from ordinary few-shot prompting |
|---|---|---|
| **Speaker reference** | Hear how this person speaks calmly before judging the target | Provides the current speaker's vocal baseline rather than arbitrary examples from strangers; a neutral label is extra supervision |
| **Real dialogue history** | Hear what happened before the target utterance | Comes from the target's conversation, without gold emotion labels; it is not labelled-demonstration ICL |

Both provide information through the input, but they test different questions. A retains the ICL direction while
reporting **labelled references** separately from **history without emotion labels**. The target is perceived
expressed emotion, not an unverifiable claim about a character's true internal state.

### Current research questions

| RQ | Question | Main evidence |
|---|---|---|
| **RQ1: What helps?** | Which targets benefit from a same-speaker reference, real past dialogue or textual history? | Paired A1/A2 conditions; UAR, macro-F1, class recalls and prediction changes |
| **RQ2: Are changes reasonable?** | Do humans change their judgments with context, and do models change accordingly rather than merely becoming confident or saying neutral less? | Multiple human votes on the same targets under matching evidence; these new annotations have not been collected |
| **RQ3: Does it generalise?** | Does the finding depend on particular references, speakers, a television series or one model? | Independent speakers/dialogues, repeated reference choices, another corpus and different models |
| **Conditional extension: Can we improve it?** | Can instruction optimisation address a demonstrated failure, and does letting the coach hear errors help? | B: the same coach with/without audio, matched search budgets and independent tests |

The zero-shot benchmark is the starting point. C/D support reliability; E/G help interpret evidence and labels.
**They need not each become a main contribution.**

### Priority and current status

| Direction | Role now | Status |
|---|---|---|
| **A1: Speaker references** | One candidate core question | Pilot design; reference inputs not implemented |
| **A2: Real dialogue history** | One candidate core question; start with MELD | Three-model MELD single-utterance baseline complete; context conditions not implemented |
| **B: GEPA and feedback** | Improvement experiment after a clear finding | Keep the question; defer the large grid |
| **0: 13-corpus benchmark** | Existing foundation | Five SER models complete; audio LLM evaluation in progress |
| **C: Training provenance** | Necessary background stratification | Check public sources; performance alone does not establish contamination |
| **D: Repetition and disagreement** | Reliability, with a possible ambiguity analysis | Start with representative subsets, not ten calls on every clip |
| **E: Text controls** | Essential for A2, useful for A1 | Design alongside the pilots |
| **F: Probes / label scores** | Optional supporting analysis or appendix | Probe implemented; not mandatory; prior correction not implemented |
| **G: What labels mean** | Interpretation and annotation checks | Some listener data exist; MELD evidence-conditioned votes require new collection |
| **H: Calibration** | Optional | Only if the main question needs it |
| **I/J: Structured outputs / telephone quality** | Deferred | Outside this pilot |
| **K: Length and padding** | Input documentation and limitations | No separate project now |

## 2. A1: Can a person's calm voice provide a useful reference?

A person may normally speak loudly, quickly or at a high pitch; the same change may mean something different for
another person. The hypothesis is that **knowing this speaker's vocal baseline may help more than examples from
unrelated speakers**. It is not assumed to work.

Choose one controlled corpus, such as CREMA-D or MCAE-SPPS, after checking reference availability, label quality and
sentence structure. Do not begin with the previous five-corpus grid. MELD has neutral-labelled utterances, but no
standard calm recording made for every character; it is a less convenient starting point for this question.

### Minimal paired conditions

Keep targets and candidate labels fixed; change only the reference:

| Condition | Input | Purpose |
|---|---|---|
| S0 | Target audio | Baseline |
| S1 | Another speaker's neutral reference + target | Effect of one additional labelled recording |
| S2 | The target speaker's neutral reference + target | Speaker matching, compared with S1 |
| S3 | The same neutral text cue without reference audio + target | Whether the cue alone explains gains |

Match reference sentences, duration and recording conditions where possible. Reference and target sentences must
not overlap. Use several predetermined reference choices and report the spread. If extending to multiple emotion
examples, add pairing shuffles, labels-only and order controls at the same audio/label budget. More examples need
not help monotonically.

**Same-speaker support is an explicit condition.** Prompt/rule/hyperparameter development still uses independent
speakers. In formal A1 evaluation, another recording of the test speaker is deliberately allowed calibration
input. Support and query recordings must be distinct, and the target's emotion cannot determine the reference.
This is not zero-shot evaluation without speaker information. If gold labels select neutral support, disclose that
extra annotation requirement.

**Avoid answer leakage.** Do not make all wrong-emotion examples share the target sentence while the correct one
uses a different sentence. Sentence matching must not reveal the answer. If emotion-embedding retrieval is added,
also evaluate directly copying the retrieved labels. Retrieval and voting are not required in this first pilot.

### When is it worth expanding?

- S2 reliably exceeds S1/S3 and replicates on new speakers or another corpus: evidence that speaker references add value.
- Only neutral predictions increase, without non-neutral recall gains: examine label preference before claiming better recognition.
- No clear gain: use effect intervals to assess whether practically relevant gains were excluded; non-significance alone is not evidence of no effect.

Speaker-personalised ICL has precedents, including [Ihori et al., ASRU 2025](https://arxiv.org/abs/2509.08344).
Differences between existing frozen LLMs and specially trained systems need checking; they do not establish a
first-of-its-kind contribution by themselves.

## 3. A2: Does real past dialogue help interpret the target utterance?

Start with MELD: locally it has utterance audio, `Dialogue_ID`, `Utterance_ID`, `Speaker` and transcripts. It comes
from the television series *Friends*. It has conversation structure, but remains scripted and acted rather than
fully spontaneous interaction.

### A concrete example in our local data

| MELD test item | Content | Existing label |
|---|---|---|
| Dialogue 252, utterance 0 | Chandler complains that Ross should not repeat himself | anger |
| Dialogue 252, utterance 1 | Ross replies “Fine!” (about 1.1 s) | anger |
| Dialogue 275, utterance 0, a different recording | Ross says “Fine.” | neutral |

The word does not uniquely determine emotion. The two target recordings differ, so **their label difference cannot
be attributed to context alone**. Instead, hold `dia252_utt1.wav` fixed and vary what precedes it:

| Condition | Input | Purpose |
|---|---|---|
| H0 | Target audio only | Single-utterance baseline |
| H1 | Real past audio turns + target | Effect of real audio history |
| H2 | Transcripts of the same history + target audio | Whether linguistic content explains the benefit |
| H3 | Duration/turn-count matched audio history from another dialogue + target | Relevance and sensitivity to unrelated information |

Begin with a fixed short window, for example the immediately preceding turn. Decide any three-turn extension on
design data. Mark speakers and the target explicitly; ask for the **target's emotion only**. Use past turns, no
future turns, and no gold emotions of history or target. Keep target audio, candidates and scoring fixed across
conditions. H3 measures sensitivity; it is not automatically a test with an objectively wrong new answer.

Use consistent anonymous speaker markers so that only history conditions do not additionally reveal character
names or show-specific knowledge. If target markers change the prompt, recompute H0 in the matching format rather
than reusing a baseline cache with a different input format.

**What do the four comparisons establish?** H0 is the no-history control; H3 uses unrelated-dialogue history, not another no-history condition or a reordering of the same history. H1 versus H0 measures the total contribution of audio history, including words, prosody, speaker, timing and background. H2 versus H0 tests whether written history also helps. H1 minus H2 is not a pure acoustic contribution: human transcripts are clearer than implicit recognition and modalities are processed differently. If H2 reproduces H1's gain, text history is sufficient to explain it under these conditions; this does not prove the model ignores speech. A later stronger check could compare identical written history with/without its original audio, always retaining target audio. Start with the four-condition pilot.

### Do existing labels show how humans change with context?

No. Hearing “Fine” alone may suggest acceptance or irritation. Learning that it answers a complaint may change the
interpretation; alternatively, the voice may already be clear enough. That is a change to measure, not assume.

MELD has one set of labels involving video and transcripts. It does not provide independent votes for H0/H1/H2 on
the same target. Initial model scores against the existing labels are useful, but a gain does not establish better
alignment with humans under the same evidence.

If the pilot merits further work, collect votes on a small target subset under two or more evidence conditions.
Different rater groups receive the same targets with different evidence and the same question. Retain class vote
counts. Sample multiple dialogues/classes, ordinary turns and short responses, not just model errors. The sample,
number of raters and budget remain decisions. Assess both agreement in changes and residual human disagreement.
Use the votes for each evidence condition as its reference rather than forcing every condition to share the old
hard label. Avoid asking someone who already heard the history to act as an uninformed no-history listener.

Changing history can legitimately change perceived emotion. **An answer change does not automatically mean the
model was misled.** Human judgments under the same conditions are needed to distinguish appropriate reinterpretation
from interference. Labels also do not reveal an unobservable true psychological state.

### Data and comparison boundaries

- Only MELD official test is local now. It supports interface checks and exploration. Do not repeatedly use it to choose prompts/history windows and then call it an untouched final test. Prepare train/dev or independent design data; validate on data not used for design and on another corpus.
- IEMOCAP and MSP-Conversation are [survey candidates](../../emotion_recognition/docs/datasets_survey.md), not downloaded or added to the current 13 evaluation datasets. IEMOCAP is closer to utterance classification; MSP-Conversation mainly has continuous V/A/D traces and needs a separate task, not an invented official seven-class mapping.
  DailyTalk and EmotionTalk have both been fully audited and retained for English and Mandarin past-turn pilots. Public ECSS categories exactly match inherited text labels and are not a second independent perceptual gold set. EmotionTalk offers 18,306 dyadic history targets, including 1,830 test targets; use audio gold and completed past turns, noting validation/test speaker-ID overlap. RESD remains metadata-only without audio download. See survey section 3b; none was automatically added to formal evaluation.
- The original five SER baselines see only target audio. History can test changes within an LLM, but extra LLM inputs do not establish equal-information superiority over those SER models.
- Characters, laughter, music, duration, class priors and possible familiarity with the show may influence MELD results. Stratify and replicate on another conversation source.

## 4. How small should the pilot be, and when do we expand?

| Step | Work | Decision afterwards |
|---|---|---|
| 0 | Check multiple-audio handling and target identification with verifiable questions and target-position changes | Fix the input before interpreting emotion results; processor acceptance is insufficient |
| 1 | Complete MELD single-utterance baselines: three models, 2,608 clips; check protocol, invalid answers, UAR/macro-F1/confusions | Establish reusable H0; one metric reversal does not establish a dialogue advantage |
| 2 | Compare H0–H3 on targets spanning multiple dialogues; separately run S0–S3 on one small controlled-corpus sample | Treat this as interface/phenomenon exploration, not final statistical design |
| 3 | Decide whether to collect matched-evidence human votes or use a more suitable dataset | Human benefit can be informative even if the model does not improve |
| 4 | Select one question to expand, adding independent data and model checks | Prioritise B only after identifying a clear failure |

**Implementation status (2026-11-09).** `config/evaluation.yaml` selects local MELD and three open LLMs, with
`limit: null`, run sequentially. Existing `evaluate.py` still sends **one independent clip at a time**, without
demonstrations or dialogue history. H1–H3/S1–S3 input construction, cache identities and result alignment are not
implemented. Running that configuration does not complete the context pilot. New conditions need separate
experiment identities to avoid overwriting or reusing the baseline incorrectly.

Use this run and multiple-audio smoke tests for timing; the old full-grid few-shot estimates are no longer a
scheduling commitment.

### Next run: scope and preflight, 2026-11-09

Omni's MELD Accuracy/UAR/macro-F1 are **51.27/43.45/40.43%**, all higher point estimates than the five current SER baselines. Gemma/Qwen2-Audio have macro-F1 30.86/29.35% but neutral prediction rates 54.45/10.58%. This suggests model-specific behaviour, without establishing significance or context use. All-neutral Accuracy is 48.16%, so Accuracy alone is insufficient.

Our Qwen2-Audio Instruct Accuracy is 30.71%, versus 55.3% reported for MELD in its technical report. Checkpoint, prompt and input/scoring protocols are not fully matched; do a small official-format sanity check before expanding rather than attributing the difference directly to capability loss. [Technical report](https://arxiv.org/html/2407.10759v1)

| Scope | Clips | Purpose |
|---|---:|---|
| Development targets from EmotionTalk train | Initially 200–300 | Omni H0–H3 interface, throughput and phenomenon checks |
| EmotionTalk official test | 1,929 | Baseline after freezing design; 1,830 permit dyadic history under audited rules |
| DailyTalk validation | 1,197 | Development only; fear/disgust/anger have just 1/3/4 clips |
| DailyTalk formal holdout | To freeze | No official test in published files; split by dialogue before evaluation |

EmotionTalk validation/test share some speaker IDs, so use train for speaker-independent development. Recompute the common target intersection after freezing stricter boundaries/windows: do not subtract a full-test H0 score from a smaller history-cohort score. DailyTalk has one byte-identical WAV pair across two training dialogues; keep those dialogues together in a new split or preregister deduplication.

**Scale and time.** This Mac's MELD inference took Gemma 27.3, Omni 31.0 and Qwen2-Audio 43.8 minutes. Separate extrapolations by clip count/audio duration give roughly 28–33 hours for all three models over both complete corpora; EmotionTalk test roughly 1.3–1.8 hours, DailyTalk validation 47–48 minutes. These are neither confidence intervals nor guarantees. Measure actual development-subset throughput first.

**Validate inputs first.** New evaluation loaders and H1–H3 are not connected yet. For the first pilot use one fully completed past turn, `history.end <= target.start`, ordered by start time, always preserving the complete target. EmotionTalk's existing three-turn CSV lists use end-time order and some histories overlap; do not blindly concatenate them as start-ordered non-overlapping speech. On test, 135/1,830 three-turn-plus-target combinations exceed 30 seconds. Gemma/Qwen2-Audio truncate, so reserve the complete target first and reduce older history using a common rule across models.

Freeze target/history IDs, anonymous speakers, target boundaries, transcript cleaning, order and length budgets. Include them plus condition/seed in cache identities. Supply no history emotion/intensity, captions or target gold. A target-transcript-only check can supplement development. Once interfaces/prompts are frozen, run new-corpus baselines and compare context on common targets; then replicate useful findings across models and independent dialogues. No new inference was launched in this review.

**Acting and reasonable judgment changes are separate issues.** MELD is TV performance, DailyTalk scripted conversation, EmotionTalk actor improvisation. None is fully unprompted everyday interaction. Replacing the dataset cannot substitute for independent matched-evidence human votes on the same target. “Fine” may already be clear from voice alone; measure whether history changes judgment. Released gold supports task scores but does not establish human-aligned model changes.

## 5. B: Can GEPA improve a demonstrated failure?

GEPA lets a coach inspect instructions, answers and feedback, then propose revised instructions without changing
model weights. It is an existing optimiser ([GEPA, ICLR 2026](https://arxiv.org/abs/2507.19457)); applying it to SER
does not automatically constitute a main contribution. Related multimodal-interface work exists
([AVQA interfaces](https://arxiv.org/abs/2610.20085), currently submitted to ICASSP 2027, not verified accepted).

The sharper question is: **does a coach that hears error audio learn different, more transferable rules than the
same coach reading textual feedback?**

| Condition | Coach input |
|---|---|
| B-text | Instructions, predictions, training labels, the same transcripts and structured feedback |
| B-audio | Exactly B-text, plus error audio |
| Rewrite control | Ordinary rewrite search at a comparable candidate/evaluation budget |

Use **the same multimodal coach model** for B-text/B-audio; vary audio access only. Match feedback counts, search
budgets, candidate selection and validation data; record calls and input costs. Feedback and selection cannot use
final test labels. Neither coach receives the answers to target test items.

Start with a small budget, then test held-out speakers/dialogues and corpus transfer. Validation-only gains or
changed neutral preference do not establish general acoustic rules. Prompts may change input processing as well
as output behaviour; do not assume their effects are restricted to the final readout. GRPO, LoRA and a large B grid
are not mandatory in this round.

## 6. How does this differ from nearby work?

| Work | Already covers | Consequence for our plan |
|---|---|---|
| [When Demonstrations Fail](https://arxiv.org/html/2603.20433v2), accepted SLT 2026 | Six LALMs, four tasks; CREMA-D supplies emotion demonstrations, evaluation uses Speech-IFEval subsets; balanced examples, label shuffling and attention knockout | Ordinary ICL failure, label reliance or the first controlled SER ICL study cannot simply be claimed as our discovery |
| [Ihori et al.](https://arxiv.org/abs/2509.08344), ASRU 2025 | Speaker-personalised ICL with dedicated training | Check the A1 setting difference; do not claim speaker references have never been studied |
| [MELD](https://aclanthology.org/P19-1050/), ACL 2019 | Conversational emotion data and context benefits | Adding earlier turns is not new by itself |
| [EAA](https://www.isca-archive.org/interspeech_2025/du25b_interspeech.html), Interspeech 2025 | Context-aware instruction tuning with the current and preceding utterances | A2 should investigate evidence and perception changes, not merely context gains |
| [CP-Bench](https://aclanthology.org/2025.findings-emnlp.760/), Findings EMNLP 2025 | Contextual content/paralinguistic QA, including a small conflict-emotion-word analysis | Specify the fixed-target, changed-history, matched-human-vote design; do not claim the first context study |
| [EmotionTalk](https://aclanthology.org/2026.findings-acl.440/), Findings ACL 2026 | Mandarin dyadic dialogue with separate text/audio/video/audiovisual perceptual labels; raters can replay history | Modality-specific perception comparisons already exist; verify visible history and do not treat current audio labels as independent H0/H1 ratings |
| [AVQA interfaces](https://arxiv.org/abs/2610.20085), submitted ICASSP 2027 | GEPA/COPRO and feedback interfaces for multimodal systems | B needs a specific feedback question and transferable evidence, not just a tool combination |

These are the nearest checked examples, not a comprehensive novelty proof. Recheck the literature and versions
before fixing a submission claim.

## 7. Supporting analyses: one necessary question each

### 0. Benchmark: what is the starting point?

Keep completed five-SER results and ongoing audio LLM results. Match candidates and scoring; report Accuracy, UAR,
macro-F1, class recall and neutral rate. A model need not lead or trail on every metric. A MELD metric reversal is
a clue. Preserve old prompt/protocol outputs, but do not mix them into the current formal table.

### C. Provenance: what do models publicly declare seeing?

Record corpus/version/split, backbone and pseudo-label sources. Distinguish **corpus-level training exposure**,
**test-file leakage** and **unknown**; source information is in the
[model survey](../../emotion_recognition/docs/models_survey.md). Corpus use does not establish test leakage; seen/unseen performance also
mixes language, labels, recording and domain differences, so it does not identify memorisation. Absence from a
published list does not prove non-exposure. Later dataset publication is a timeline clue; recording, acquisition
and model versions still need checking. Teacher relationships can motivate correlation checks, not establish an
individual error's cause. This section limits claims rather than presupposing contamination.

### D. Repetition: are changes stable?

On representative subsets, fix inputs, record decoding settings and repeat calls; compare single calls, empirical
distributions and voting. Greedy local repetition generally does not create a useful sampling distribution.
Generation randomness is not human perceptual disagreement. Studying their relationship requires class-prior
controls and simple baselines. `token_p_*` is a first-token label-score proxy, not a substitute for repeated answer
distributions. Estimate paired changes with speaker/dialogue grouping; many turns in one conversation are not
independent observations. Non-significance alone does not establish no effect.

### E. Text: what content explains a benefit?

Compare target and history transcripts. Separate human transcripts, an idealised semantic condition, from ASR
transcripts with recognition errors. Text use is not automatically a failure: conversational emotion can depend
on meaning. It limits claims that a gain comes from acoustic information. Start with needed pilot items rather
than assuming an ASR pass over all 13 corpora.

### G. Labels: acted instruction or listener perception?

The requested portrayal and perceived emotion of the same recording can differ. On corpora with votes, score the
same targets against each definition; report disagreement, class strata and neutral baselines. A neutral-biased
model can look human-like on neutral-heavy listener subsets without better perception. MELD labels involve video
and transcripts, not audio-only listener votes; other corpora are not uniformly actor-label-only either. A2's
matched-evidence human votes are new material, not present in existing exports.

### F/H. Optional: probes, label scores and calibration

[Probe code and guide](../../emotion_recognition/scripts/analysis/README.md) are implemented: freeze saved vectors,
train a small supervised linear classifier, select C/layer with validation and evaluate on test. This tests whether
representations support supervised linear prediction; it does not locate the cause of native generation failure.
It is not zero-shot performance and cannot prove the model understood emotion but only failed in its last layer.
Keep it optional, possibly in an appendix; having code does not require every combination to be run.

`probe_p_*` comes from the new classifier; original LLM `p_*` is answer one-hot; `token_p_*`/`token_mass` are
first-token proxies. Calibration and prior correction need their own rule and independent data. Prior correction
is not implemented, and equal class shares should not be assumed correct. Checked API logprob boundaries are in
[api_logprobs.md](api_logprobs.md); models and interfaces may change, so recheck before execution.

### I/J/K. Deferred work and limitations

Structured JEV-style outputs and telephone codecs are outside this round; see [jev_analysis.md](jev_analysis.md)
and the old design for background. Record length, cropping, padding, background sounds and multiple speakers.
A model silently truncating longer history does not receive equivalent information.

## 8. What would make a paper worth writing?

Seek evidence that changes the interpretation, not just another score table. Candidate findings include:

- Same-speaker references help while matched labels from another speaker do not: supports a role for individual vocal baselines.
- Humans benefit from true history but models do not change accordingly: limits these models' use of dialogue evidence.
- Model/human changes agree, but textual history explains the benefit: locate the claim in semantic context, not pure acoustic improvement.
- A benefit replicates on another corpus and different models: more persuasive than a development-only improvement.

These are possible outcomes. If pilots repeat ordinary ICL failure or yield intervals too wide to distinguish
explanations, stop expanding, revise the question or defer it. A negative result needs a clear comparator,
adequate precision and a meaningful capability boundary; controls alone do not make it a contribution.

**Working titles:** *What Context Helps Audio LLMs Judge Emotion?*; *Speaker References or Dialogue History?
Studying Emotion Judgments in Audio LLMs*. Let findings narrow the title and final RQs. Do not present A/B/C/D as
four established contributions, or promise an acceptance tier in advance.

## 9. Decisions and next work

| Item | Current status / decision |
|---|---|
| MELD three-model single-utterance baseline | Complete on the same 2,608 targets and seven candidates; this is H0, not the complete context pilot |
| Multiple-audio and target-boundary checks; history/reference inputs and cache identities | Not implemented; start with verifiable small examples |
| MELD design/test data and history window | Freeze before formal evaluation; do not select by repeatedly viewing test |
| A1 controlled corpus and support/query sentences | Choose one first |
| Matched-evidence human votes | Decide whether, how many targets/raters and budget; not yet collected |
| New corpus candidates | DailyTalk and EmotionTalk fully audited/retained; RESD audio deferred by the user; IEMOCAP/MSP-Conversation surveyed only. Formal evaluation integration and context inputs remain unimplemented |
| GEPA | Decide after a clear A finding; new adapter not implemented; no original full grid now |
| Linear probe | Tool available, running optional; does not block the main question |
| Formal statistics, independent-corpus validation and release | Arrange after selecting the core question; no restricted audio release |

Schedule by gates: baseline → two small pilots → choose one main question → independent validation → writing.
The old grid-hour estimates are not commitments.

**H0 observation, 2026-11-09.** Gemma 4, Qwen2.5-Omni and Qwen2-Audio obtain macro-F1 **30.86 / 40.43 / 29.35%**, with neutral prediction rates **54.45 / 40.57 / 10.58%**. Omni has higher Accuracy/UAR/macro-F1 point estimates than all five current SER baselines; this is not a significance or context-effect conclusion. Model differences are large, so neutral bias is not a common explanation for all audio LLMs. The next step remains paired H0/H1/H2/H3 conditions on fixed targets.

See [venues.md](venues.md) for target venues and checked deadlines.
Choose a venue after assessing the evidence; account for publication processes, author-service requirements and
data access when scheduling.

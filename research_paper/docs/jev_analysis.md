# JEV ("Jev") and "decision models": what they are, and whether a JEV-like angle fits our SER paper

Local working note, git-ignored. Written 2026-11-06. Sources were fetched on 2026-11-06. Fetched pages were treated as
data. Labels used below:

- **[V] verified**: stated by a primary source (the company's own docs or blog, the paper itself, the API docs) or by
  reputable press that I read.
- **[W] weak source**: Wikipedia, community blogs, SEO pages, search snippets, or a page I could not open (title
  only).
- **[A] analysis**: my own reasoning. Not a fact.

---

## 0. Plain-language summary

- **JEV is real and can be verified.** "Jev" is a commercial, closed model from the start-up TypeSafe AI, released in
  early access in September 2026. TypeSafe's own blog and docs describe it.
  ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
  [docs](https://docs.typesafe.ai/concepts/system-one)) **[V]** There is **no paper, no weights and no public
  benchmark**. The architecture and training are not disclosed. So the technical claims (speed, cost, calibration)
  cannot be checked independently. **[V]**
- **The core idea is simple.** You give the model some input and a fixed list of options. It returns one option plus a
  probability for every option. It never writes free text. Seen from outside, this is "a classifier whose labels you
  choose at request time". **[V]/[A]**
- **The OpenAI claim is true.** OpenAI launched a **Decisions API** (public beta on 6 Oct 2026; general availability
  reported for 22 Oct 2026) with "predicate / choice / score" questions on the model `gpt-6-luna`.
  ([docs](https://developers.openai.com/api/docs/guides/decisions)) **[V]** **It does not accept audio.** Its voice
  integration sends a *transcript*, so it judges emotion from the words, not the voice.
  ([voice guide](https://developers.openai.com/api/docs/guides/decisions-voice)) **[V]**
- **"JEV for speech emotion" is already published.** *Audio System One Model (ASOM)*
  ([arXiv 2610.22401](https://arxiv.org/abs/2610.22401), Oct 2026) cites TypeSafe/Jev and builds exactly this for
  speech emotion. It also scores 6 audio LLMs by reading answer-token probabilities. Label-logit reading, classification
  heads, LoRA SFT and GRPO for audio-LLM SER are also published (section 2). **[V]** So as a *main* paper angle,
  "make an audio LLM JEV-like" is **not novel**. **[A]**
- **What is still open, and cheap for us:** on the *same* three open audio LLMs, across our 13 datasets with
  seen/unseen control, compare (1) free-text answers, (2) label-probability answers (with and without a simple prior
  correction), and (3) a linear probe on hidden states. Then ask how much of the gap to dedicated SER models comes from
  *decoding* and how much from *representation*. No paper I found does this at this breadth with neutral-rate and
  calibration reporting. **[A]**
- **Verdict:** use it as a **supporting analysis** in the prompt/ICL paper, not as the main angle. **[A]**
- **Practical tip:** phase 2 (the 36 h local audio-LLM run) has not started yet. If we save the label-token
  probabilities and a few pooled hidden states *in the same forward pass*, we get almost all of experiments E1 and E2
  for little extra compute, and we never need to re-run phase 2. **[A]**

---

## 1. What JEV is

### 1.1 Verified facts [V]

| Item | Fact | Source |
|---|---|---|
| Maker | TypeSafe AI. Founder: Diogo Almeida. | [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Release | Early access announced on 15 Sep 2026 | [TechCrunch, 18 Sep 2026](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| What it is | A "System One model": "unstructured state in, typed probabilistic decisions out". It does not write replies, code or explanations. | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs: System One](https://docs.typesafe.ai/concepts/system-one) |
| Question types | Choice (unordered options), Score (ordered levels), Noul (true/false). Box was added in Jev 1.14. | [docs: models](https://docs.typesafe.ai/models.md) |
| Output of a Choice question | `choice` (the most probable option), `probabilities` (the full distribution, sums to 1), `confidence` (0–1). Up to 255 options. | [docs: Choice](https://docs.typesafe.ai/primitives/choice.md) |
| Inputs | Text, JSON, and PNG/JPEG images. "It does not natively accept audio or video (yet)." Context of 64k per request (262k in a slower long-context mode). | [docs: models](https://docs.typesafe.ai/models.md) |
| Versions | Jev 1.14.0 (14 Oct 2026) and 1.13.0, both on `POST /v1/systemone` | [docs: models](https://docs.typesafe.ai/models.md) |
| Training | Called "Reinforcement Learning for Calibrated Decisions (RLCD)". No details given. TechCrunch reports it is trained only on synthetic data. | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| Architecture | Not disclosed. The blog says "a new model architecture" and a "parallel sampler". TechCrunch says "transformer-based" and that the company is "secretive". | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| Calibration claim | "Calibrated: higher confidence means higher accuracy." The docs add that calibration "does not guarantee that an individual answer is correct". | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs](https://docs.typesafe.ai/concepts/system-one) |
| "Can't hallucinate" | This means outputs always match the schema, so there are no type errors. TypeSafe says its 0% figure "is not empirical", only guaranteed by construction. | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Benchmarks | Only internal "workflow evals". The reference answers are predictions from large external LLMs, not ground truth. TypeSafe says bias "could exist". Its FAQ leaves "How does Jev perform against public benchmarks?" open. | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) |
| Speed / price claims | 70–500 ms end to end. $0.042 per million input tokens. Output is free. Homepage: "193.6x Faster, 244.6x Cheaper". | [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [homepage](https://typesafe.ai/) |
| No customer fine-tuning | "Not fine-tuned or LoRA-adapted with customer data". You customise it through the question and option descriptions. | [docs: models](https://docs.typesafe.ai/models.md) |
| Open code | Only an MIT "System One adapter". It emulates the Jev API using *other* LLMs' structured outputs, for comparison. It is not Jev. | [GitHub](https://github.com/typesafe-ai/system-one-adapter-python) |
| Independent views | Two developer case studies: one 5–18x faster, one 10–20x cheaper (but "Gemini was slightly more accurate"). One sceptical quote: users must pick probability thresholds themselves. | [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| Paper / weights / replication | None found. The closest academic follow-up is ASOM, a speech-emotion "System One" model. It is not a replication of Jev. | [arXiv 2610.22401](https://arxiv.org/abs/2610.22401) |

### 1.2 Claims from weak sources [W]

| Claim | Where | Comment |
|---|---|---|
| The name refers to W. S. Jevons (the Jevons paradox) | [Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) | Plausible. Not found in the TypeSafe pages I read. |
| CEO spent ~4 years at OpenAI on RLHF, InstructGPT, ChatGPT and GPT-4 | [Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) | Consistent with TechCrunch's "ChatGPT inventor" headline. |
| $40M seed at $200M valuation (DCVC) | [Forbes](https://www.forbes.com/sites/the-prompt/2026/09/15/this-200-million-startup-wants-to-fix-ais-overconfidence-problem/) (HTTP 403; title and search snippet only) | Not read. |
| $870M at a $7.5B valuation (a16z) | [TechCrunch, 9 Oct 2026](https://techcrunch.com/2026/10/09/the-maker-of-non-text-ai-model-jev-valued-at-7-5b-just-weeks-after-launch/) (headline only) | Not read. Irrelevant to the science. |
| "40–400x cheaper" | [Wikipedia](https://en.wikipedia.org/wiki/Jev_(AI_model)) | **Disagrees** with the primary source ("244.6x cheaper" on the [homepage](https://typesafe.ai/); "40x–200x faster" in the [blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). One search snippet even said "444.6x". Use only the primary numbers, and only as marketing claims. |
| Hugging Face blog "What Is Jev AI? A Decision Layer for AI Agents" | [HF blog by user dddffgassa](https://huggingface.co/blog/dddffgassa/jev-ai), 25 Sep 2026 | **Low credibility.** It is a community post by an anonymous account, in a marketing tone. It has no technical details and no citations beyond TypeSafe's own docs. It links to third-party "Jev" sites (bestjevai.com, jevaimodel.dev) that are not TypeSafe domains, and warns about credential/billing confusion, which suggests look-alike services exist. Its description of the primitives (Choice, Score, Noul) matches the docs, but it adds nothing primary. It says Jev does not take images. The [current docs](https://docs.typesafe.ai/models.md) say it does (JPEG/PNG), so the blog is out of date. Do not cite it in a paper, and do not enter credentials on the sites it links. |

Inconsistencies noticed **[V]**:

- The [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) says images are not supported
  "(yet…)". The [models docs](https://docs.typesafe.ai/models.md) list PNG/JPEG. The docs are newer (Jev 1.14), so
  the docs win.
- The blog page shows two dates (15 Sep and a later "published" date). Press and docs agree on the 15 Sep 2026 launch.
- The blog says "Reinforcement Learning **for** Calibrated Decisions". TechCrunch writes "reinforcement learning
  **from** calibrated decisions". Same idea, different wording.

### 1.3 The core idea in plain words [A, based on the V facts above]

A normal LLM writes text. To get a decision from it, you ask in words, hope it answers in the right format, parse the
text, and still get no trustworthy probability. Jev turns this around:

1. You give the input ("state") and a question with a **fixed list of options** (with short descriptions).
2. The model returns a **probability for each option**. The answer is the most probable option, and invalid answers
   cannot happen.
3. TypeSafe claims the training (RLCD) makes these probabilities **calibrated**, and that many questions are answered
   **in parallel in one pass**, which makes it fast and cheap.

For SER, this is exactly the "deterministic class decision instead of free generation" the user has in mind.

### 1.4 How it differs from nearby techniques

| Technique | What it does | Invalid outputs? | Gives a probability per label? | Needs training? | Labels set at request time? | How it relates to Jev |
|---|---|---|---|---|---|---|
| Constrained / grammar decoding (e.g. Outlines, [Willard & Louf 2023](https://arxiv.org/abs/2307.09702)) | Masks tokens during generation so the output must match a regex or grammar | No | Not by itself (you must also read logprobs) | No | Yes | It fixes the *format* only. The decision is still the generator's greedy or sampled choice. |
| JSON-schema structured outputs, function calling ([OpenAI Structured Outputs, Aug 2024](https://openai.com/index/introducing-structured-outputs-in-the-api/)) | Generation constrained to a JSON schema | No | Only if the model *writes* numbers (verbalised confidence, often poorly calibrated: [Tam et al. 2024](https://arxiv.org/abs/2408.02442) on format costs; [Calibrating Verbalized Probabilities for LLMs, arXiv 2410.06707](https://arxiv.org/abs/2410.06707)) | No | Yes | TypeSafe's own adapter emulates Jev this way, and calls it "slower and more expensive" ([blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [adapter](https://github.com/typesafe-ai/system-one-adapter-python)). |
| Label-token logit scoring / verbalisers ([PET, Schick & Schütze](https://arxiv.org/abs/2001.07676); [Calibrate Before Use, Zhao et al. 2021](https://arxiv.org/abs/2102.09690); [Surface Form Competition, Holtzman et al. 2021](https://arxiv.org/abs/2104.08315)) | One forward pass. Read the probability of each label word (first token, or the full sequence) and normalise over the labels | No | Yes (raw, often biased toward some labels; can be corrected) | No | Yes | **The closest no-training equivalent of a Jev "Choice" answer.** Jev's claimed extra is training for calibration. |
| Classification head on LLM hidden states (`AutoModelForSequenceClassification` in [transformers](https://huggingface.co/docs/transformers/model_doc/auto); [LLM2Vec](https://arxiv.org/abs/2404.05961) for LLMs as encoders) | A linear or MLP layer on a pooled hidden state, trained with cross-entropy | No | Yes | Yes (a labelled set per label set) | **No** (the labels are fixed in the head) | Often the most accurate for one fixed task. Jev and ASOM avoid per-class parameters so that labels can change. |
| Reward / decision models (RLHF reward models) | Score how good a (prompt, response) pair is, as one scalar | – | One scalar, not a distribution over options | Yes | – | Like a single Jev "Score" question with a learned rubric. A decision model generalises this to many typed questions. |
| **Jev** | Typed questions in, a probability distribution per question out, in parallel | No (by construction) | Yes, with calibration claimed | Already trained, closed | Yes | Closed product. Mechanism not disclosed. |

### 1.5 The "OpenAI recently added a decision API" claim

**Verified [V].** It refers to the OpenAI **Decisions API**.

| Item | Fact | Source |
|---|---|---|
| Launch | Public beta announced on 6 Oct 2026 | [OpenAI community announcement](https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877); [The Decoder, 7 Oct 2026](https://the-decoder.com/openai-launches-decisions-api-that-reduces-complex-evaluations-to-yes-no-or-pick-one/) |
| General availability | Announced in a later community thread; secondary sources give 22 Oct 2026 | [GA thread](https://community.openai.com/t/decisions-api-is-now-generally-available-with-new-updates/1408934) (title seen, thread not read) **[W for the exact date]** |
| Question types | Predicate (probability that a condition is true), Choice (`choice` + `probabilities` + `confidence`), Score (probability-weighted level index + distribution, 2–10 levels). Up to 400 questions per request. | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| Model | Only `gpt-6-luna` | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| Inputs | Text, images, JSON, conversation history. **"Not supported: audio, files"** | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| Voice | The GPT-Live integration "sends the conversation transcript to Decisions". Example choices include an "expression" (happy, sad, concerned, neutral), judged from the transcript. Acoustic cues are not mentioned. | [Decisions voice guide](https://developers.openai.com/api/docs/guides/decisions-voice) |
| Price | $0.10 per 1M input tokens. No output-token charge. | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| vs Structured Outputs | Use Decisions for typed selections. Use Structured Outputs to generate your own JSON object. | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| Calibration | The docs make no explicit calibration claim. | [OpenAI docs](https://developers.openai.com/api/docs/guides/decisions) |
| Link to Jev | The Decoder calls it "likely OpenAI's response to the mini-trend of 'decision models' that Jev kicked off" (press opinion) | [The Decoder](https://the-decoder.com/openai-launches-decisions-api-that-reduces-complex-evaluations-to-yes-no-or-pick-one/) **[W]** |

[W] One fetch of the community thread reported a user test in which `choice` put too much probability on the top
option (a 70/30 coin came out about 98/2). A second fetch could not find that post. **Treat it as unconfirmed.**

[A] What this means for us: neither Jev nor OpenAI Decisions takes audio. OpenAI's voice route judges emotion from
transcripts, which is exactly the "words over voice" failure that our lexically neutral datasets are built to expose.
Neither can be used directly in our benchmark. The OpenAI route could at most be a transcript-only control, which our
planned cascade control (ASR + text LLM) already covers.

---

## 2. Related work for audio LLMs / SER, and the novelty gap

All entries below were opened on arXiv or the venue site unless marked. Some are already in
[related_work.md](related_work.md) (marked "in RW").

| Approach | Paper | What it shows | Link |
|---|---|---|---|
| **JEV-like model for speech emotion** | Bellver-Soler et al., *Audio System One Model: Typed Probabilistic Perception of Emotion in Speech* (Oct 2026; in RW) | A 6.2M-parameter "reader" on frozen Whisper-large-v3 and Qwen2.5-1.5B (criteria text) encoders, trained on listener vote distributions. It answers many typed questions in one pass with no per-class parameters. It beats 6 zero-shot audio LLMs by 0.072 macro-F1 and an audio LLM fine-tuned with LoRA on the same data by 0.026 on MSP-Podcast, and is tested on 21 unseen corpora. **It cites TypeSafe/Jev explicitly.** The audio LLMs (Qwen2.5-Omni 3B/7B, Qwen2-Audio-7B-Instruct, AF3, Phi-4-MM, Voxtral) were scored by reading answer-token probabilities ("instead of decoding"). Code promised on acceptance. | [arXiv 2610.22401](https://arxiv.org/abs/2610.22401) |
| Label-token logits from a speech LLM | Halim et al., *Token-Level Logits Matter*, Interspeech 2025 | LTU-AS on IEMOCAP (4 classes): emotion distributions taken from label-token logits beat the distributions the model writes as text, and the logits are less sensitive to the prompt. | [ISCA archive](https://www.isca-archive.org/interspeech_2025/halim25_interspeech.pdf) |
| Classification head vs generated labels | Watawana et al., *From Generative to Discriminative: Lightweight Adaptation of SpeechLLMs for Multimodal Emotion Recognition* (Sep/Oct 2026) | Whisper-medium + 3B LLM: a linear head on LLM hidden states beats label-token generation in 10 of 14 configurations (IEMOCAP, MELD). | [arXiv 2609.20081](https://arxiv.org/abs/2609.20081) |
| Probes: encoded but not used | Koduru et al., *Heard but Not Heeded* (Sep 2026; in RW) | Qwen2-Audio-7B, Qwen2.5-Omni-7B and others encode speaking style strongly (linear probes), but it fades before the output: "a gap between what models encode and what they use". | [arXiv 2609.00727](https://arxiv.org/abs/2609.00727) |
| Probes + causal edits | Peng & Nuchged, *Represented but Ignored* (2026) | In 4 audio LLMs, prosody is decodable in late LLM states but only partly expressed in answers. Hidden-state interventions shift the answers. | [arXiv 2608.19211](https://arxiv.org/abs/2608.19211) |
| Emotion neurons | Zhao, Schuller, Sisman, ACL 2026 | Emotion-sensitive neurons in Qwen2.5-Omni, Kimi-Audio and AF3. Steering them shifts predictions. | [arXiv 2601.03115](https://arxiv.org/abs/2601.03115) |
| Words over voice, neutral default | Chen et al., *LISTEN*, EACL 2026 (in RW) | "Models predict 'neutral' when lexical cues are neutral or absent." | [arXiv 2510.10444](https://arxiv.org/abs/2510.10444) |
| Words over voice | Chen et al., acoustic–semantic incongruity, EMNLP 2026 Findings | Audio LLMs are guided mainly by semantics. SFT helps on incongruent cases. | [arXiv 2608.28966](https://arxiv.org/abs/2608.28966) |
| Acoustic cue tokens | Tsangko, Triantafyllopoulos, Schuller (Jun 2026) | eGeMAPS-derived cue tokens in the prompt raise UAR (IEMOCAP, FAU-Aibo). Wrong tokens lower it. | [arXiv 2606.07309](https://arxiv.org/abs/2606.07309) |
| LoRA SFT of Qwen2-Audio for SER | Zhang et al., *VoxEmo* (Mar 2026; in RW) | LoRA (r=8, q/k/v/o) on Qwen2-Audio-7B-Instruct and AF3. 35 corpora, including 10 of ours. Soft labels are **parsed from a generated JSON**, not read from logits. Large prompt sensitivity. | [arXiv 2603.08936](https://arxiv.org/abs/2603.08936) |
| LoRA + Q-Former, constrained generation | EmoQ (2025) | HuBERT + Q-Former + Qwen2.5-7B with LoRA. Predicts by constrained generation. | [arXiv 2509.15775](https://arxiv.org/abs/2509.15775) |
| GRPO for SER | Li et al., *EMO-RL*, EMNLP 2025 Findings | GRPO on Qwen2-Audio-7B-Instruct with an emotion-similarity-weighted reward. MELD, IEMOCAP, cross-dataset tests. | [arXiv 2509.15654](https://arxiv.org/abs/2509.15654) |
| GRPO + reasoning for SER | Wang et al., *EmotionThinker*, ICLR 2026 | Prosody-aware GRPO with a reasoning-quality reward. Explains its predictions. | [arXiv 2601.15668](https://arxiv.org/abs/2601.15668) |
| GRPO on Qwen2-Audio (general audio QA) | Li et al., R1-AQA (2025) | GRPO beats SFT on MMAU with 38k samples. | [arXiv 2503.11197](https://arxiv.org/abs/2503.11197) |
| Uncertainty for audio LLMs | Kuan, Huang, Lee (Apr 2026; in RW) | Entropy, semantic entropy and P(True) on audio LLMs. No emotion task. | [arXiv 2604.25591](https://arxiv.org/abs/2604.25591) |
| Option-order sensitivity | Lin et al., *Hearing the Order* (in RW, not re-checked today) | Shuffling the answer options changes audio-LLM accuracy by up to 24%. | [arXiv 2510.00628](https://arxiv.org/abs/2510.00628) |
| Calibration of LLM classifiers (text) | [Zhao et al. 2021](https://arxiv.org/abs/2102.09690) (contextual calibration with a "content-free" input); [Holtzman et al. 2021](https://arxiv.org/abs/2104.08315) (surface-form competition); [Guo et al. 2017](https://arxiv.org/abs/1706.04599) (ECE, temperature scaling) | Standard tools for the "label-probability" route. | links in column 1 |

**Novelty judgement [A]:**

| Possible claim | Already done? | Verdict |
|---|---|---|
| "A JEV-like decision model for speech emotion" | Yes: ASOM (Oct 2026) | Not novel |
| "Read label probabilities instead of generating" for audio-LLM SER | Yes: Halim 2025 (LTU-AS, IEMOCAP), ASOM (as a baseline method) | Not novel as a method |
| "A classification head / probe on audio-LLM hidden states" | Yes: Watawana 2026, Koduru 2026, Peng 2026 | Not novel as a method |
| "LoRA SFT of Qwen2-Audio for SER" | Yes: VoxEmo, ASOM (Omni-3B LoRA) | Not novel |
| "GRPO for SER" | Yes: EMO-RL, EmotionThinker | Not novel |
| **"On the same open audio LLMs, across 13 corpora (4 unseen by all models), 7 classes and lexically neutral speech: how much of the gap to dedicated SER models is decoding (format, label prior, neutral bias) vs representation?"**, reported with neutral rate, invalid rate and ECE | Partly. Each piece exists, but only on 1–2 corpora (IEMOCAP/MELD/Expresso/MSP-Podcast) or as a side baseline. None decomposes the neutral bias this way, and none uses a seen/unseen control. | **Open, but incremental.** Good as one analysis section. |

---

## 3. Concrete ways to get JEV-like decisions from our audio LLMs

Shared numbers: our local audio LLMs run at about 0.5–1.0 s per clip with greedy generation (measured). Phase 2 is
planned at about 36.5 h for 3 models × 56,264 clips ([experimental_design.md](experimental_design.md) §5). All time
and memory figures below are **estimates [A]** unless marked measured. Each one should be checked with a short pilot
(about 50 clips, or about 20 training steps).

### 3.1 Without training (feasible now on the Mac)

**(a) Read the 7 label probabilities from the first answer-token logits**

- *What changes:* we use the same prompt and the same forward pass. At the first answer position, take the logits of
  the first token of each label ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"). Merge
  case and leading-space variants with log-sum-exp, then apply softmax over the 7 labels.
- *Expected effect:* invalid answers drop to 0 by construction, and no parsing is needed. We get a probability vector,
  so ECE, NLL, Brier and selective accuracy become possible. The argmax mostly equals the greedy answer whenever the
  greedy answer is already a label. So **accuracy changes little** [A]; the gain is probabilities plus the options
  (c) and the prior correction below. Halim et al. found logit distributions less prompt-sensitive than text
  ([Interspeech 2025](https://www.isca-archive.org/interspeech_2025/halim25_interspeech.pdf)).
- *Prior correction (cheap, possibly useful against "neutral"):* estimate each model's label prior from "content-free"
  audio (silence, low-level noise; about 20 clips), or from the mean prediction over a dataset, and divide it out.
  This follows contextual calibration ([Zhao et al. 2021](https://arxiv.org/abs/2102.09690)). If the neutral bias is
  partly a *prior* in the output layer, this lowers the neutral rate without training. If it is not, that is a finding
  too. [A]
- *Compute:* close to zero extra if captured during phase 2 (`output_scores=True, return_dict_in_generate=True` in
  `generate`; check that each model's wrapper passes these through, especially Qwen2.5-Omni's thinker). Use only the
  first attempt, not the retry turn. Done after phase 2, it means one extra prefill per clip (estimate 0.3–0.8 s per
  clip, i.e. about 25–35 h again for 3 models), so capture it during phase 2.
- *Risks:* two labels may share a first token in some tokenizer (check all three tokenizers; Gemma 4's has not been
  checked). The chat template may put a token (space, quote, "The") before the label, so check that the answer
  position is right. Raw probabilities will be over-confident until calibrated.

**(b) Score full label sequences**

- *What changes:* compute log P(label | prompt, audio) for each of the 7 label strings, as a sum (and as a
  length-normalised mean) of token log-probabilities.
- *Expected effect:* fixes first-token collisions and multi-token labels ("disgusted" may be split into several
  tokens). Surface-form effects remain, so compare sum vs mean ([Holtzman et al. 2021](https://arxiv.org/abs/2104.08315)).
- *Compute:* one prefill with a cached prefix plus 7 short continuations. Estimate +10–40% over (a). Only worth it if
  (a) has collisions.
- *Risks:* more code. Length normalisation choices change results, so report both.

**(c) Logit-based abstention / thresholds**

- *What changes:* answer only when max p ≥ τ (or the margin top1 − top2 ≥ δ). Otherwise output "uncertain". Report
  accuracy-coverage curves (AURC).
- *Expected effect:* shows whether audio-LLM confidence is *useful* (does accuracy rise with confidence?), compared with
  the dedicated SER models' softmax. This is the "calibrated decision" claim of Jev and OpenAI Decisions, tested on
  open audio LLMs. [A]
- *Compute:* none (post-hoc on the saved probabilities).
- *Risks:* thresholds tuned on test data inflate results. Tune on seen datasets and test on the 4 unseen ones.

### 3.2 With training

**(d) LoRA SFT with label targets**

- *What changes:* the model learns to output the label word (cross-entropy on the label token only), and we read the
  logits as in (a).
- *Expected effect:* large in-domain gains are likely (VoxEmo, ASOM's LoRA baseline). Cross-corpus gains are smaller.
  The neutral bias probably drops because the label prior is learned. [A]
- *Compute on the 48 GB Mac:* a 7–8B model in bf16 needs about 16–17 GB for weights, plus activations. With gradient
  checkpointing, short clips (3–5 s) and batch size 1 with accumulation, it **probably fits but is tight**, close to
  the MPS working-set limit [A]. MPS training of a 7B model is slow: my rough estimate is 1–3 s per sample per step,
  i.e. 3–8 h per epoch of 10k clips, unmeasured, so run a 20-step pilot first. **Better local option:
  Gemma 4 E4B**, the smallest of our three. A community tool, *gemma-tuner-multimodal*, claims audio LoRA for
  Gemma 3n and Gemma 4 E2B/E4B on Apple Silicon ([wiki](https://github.com/mattmireles/gemma-tuner-multimodal/wiki); **[W]**, no
  benchmarks given). **MLX:** MLX-LM's LoRA is text-only. MLX-VLM supports some omni models (Gemma 3n audio), but its
  docs do not show audio LoRA, and Qwen2-Audio and Qwen2.5-Omni are not listed
  ([mlx-vlm](https://github.com/Blaizzy/mlx-vlm)). So MLX is **not ready** for our models today. On one NVIDIA
  A100/H100: probably a few GPU-hours per run [A].
- *Risks:* contamination (never train on a test corpus; use leave-one-dataset-out). Catastrophic forgetting matters less
  for a classification probe study. Licences: academic use only, which is fine for the paper.

**(e) Linear / MLP head on frozen audio-LLM hidden states (a probe)**

- *What changes:* save pooled hidden states once (mean over audio-token positions, plus the last prompt token) at a
  few layers (e.g. the projector output and layers 0, L/4, L/2, 3L/4, L). Then train logistic regression on the CPU.
- *Expected effect:* this measures how much emotion information the model *has*, with no decoding at all. Based on
  Koduru 2026, Peng 2026 and Watawana 2026, the probe will likely beat free-text answers clearly, especially on "voice
  only" cues [A].
- *Compute:* hidden states can be saved during phase 2's forward pass (no extra pass). Storage estimate: hidden size
  about 3.5–4k, fp16, 5 layers × 2 poolings ≈ 80 KB per clip ≈ 4.5 GB per model ≈ 13–14 GB for 3 models. Probe
  training takes minutes on the CPU.
- *Risks:* a probe is supervised, while the zero-shot answers are not. Present it as an "upper bound of linearly
  available information" and evaluate it **cross-corpus** (train on the other datasets, test on the held-out one,
  especially the 4 unseen ones). Use speaker-independent folds inside each dataset. As a reference, run the same probe
  on emotion2vec+ features.

**(f) RL (GRPO) with an exact-match reward**

- *What changes:* sample several answers per clip and reward the correct label (EMO-RL, EmotionThinker, R1-AQA).
- *Expected effect [A]:* for a **one-word, closed-set** answer, GRPO with a 0/1 reward roughly optimises expected
  accuracy, which plain cross-entropy SFT on the label does more cheaply and more stably. RL pays off mainly when the
  model writes reasoning first, which is not our setting. Low value per compute hour for this question.
- *Compute:* group sampling (4–8 samples per clip) plus a reference model. Not realistic on the Mac. On NVIDIA, several
  times the cost of (d).
- *Risks:* reward hacking toward the majority class, instability (EMO-RL reports convergence problems), and a large
  engineering effort.

### 3.3 Summary table

| Option | Training | Fixes invalid output | Gives probabilities | Likely accuracy effect | Mac cost | Value for the paper |
|---|---|---|---|---|---|---|
| (a) first-token label logits | No | Yes | Yes | Small (by itself) | ~0 if captured in phase 2 | **High** |
| (a)+ prior correction | No (about 20 calibration clips) | Yes | Yes | Possibly lower neutral rate | ~0 | **High** |
| (b) full-sequence scoring | No | Yes | Yes | Small | +10–40% over (a) | Medium (only if collisions) |
| (c) abstention | No | – | uses (a) | Selective accuracy | 0 | Medium–high |
| (d) LoRA SFT | Yes | Yes | Yes | Large in-domain, smaller cross-corpus | Tight / slow; Gemma 4 E4B feasible | Medium (later, NVIDIA) |
| (e) linear probe | Yes (tiny) | Yes | Yes | Likely large | Storage plus minutes | **High** |
| (f) GRPO | Yes | – | – | ≈ (d) for one-word labels | Not feasible | Low |

---

## 4. Is it a worthwhile paper angle?

**Verdict [A]:** **not as the main angle.** ASOM already owns "JEV-like SER", and each method is published. **Yes as a
supporting analysis** in the prompt/ICL paper, under a question such as:

> *Free-text answers vs label-probability decisions vs a linear probe on the same audio LLM: how much of the gap to
> dedicated SER models is decoding, and how much is representation?*

Why it fits:

- It explains *our* known failure modes (neutral over-prediction, prompt dependence, no probabilities) with a
  mechanism instead of just reporting them.
- It costs almost nothing if captured during phase 2.
- It is pure evaluation and analysis with standard published methods, so it does not disclose a patentable method.
  It does not train a commercial-style decision model (RLCD-like work would be closer to TypeSafe's commercial area).

### 4.1 Experiment E1: decoding ablation (no training)

- **Models:** Qwen2.5-Omni-7B, Qwen2-Audio-7B-Instruct, Gemma 4 E4B (local). Dedicated SER models as reference.
- **Readouts per clip:** (1) free text (current pipeline); (2) label-logit argmax; (3) label-logit plus prior
  correction (content-free audio prior, and a per-dataset mean prior as a variant); (4) optional full-sequence scoring.
- **Datasets:** all 13 (crema_d, mcae_spps, mess, ravdess, tess, jl_corpus, thai_ser, nemo, mead, meld, emovo,
  emozionalmente, savee). **Report the unseen group (MCAE-SPPS, MESS, THAI-SER, nEMO) separately.**
- **Metrics:** accuracy, UA, macro-F1, neutral rate (predicted neutral share vs true neutral share), invalid rate
  (text only), ECE (15 bins), NLL, Brier, AURC; also the agreement between readouts (1) and (2).
- **Prompt-sensitivity control:** 3 paraphrased prompts plus 2 option orders, on a small subset (e.g. RAVDESS + nEMO,
  about 2.5k clips). Measure the label-flip rate for text vs logits (does Halim's finding hold for our models?).
- **Baselines:** the dedicated SER models' softmax (ECE, AURC on the same clips); majority class; uniform.
- **Effort [A]:** 2–3 person-days of code (capture, mapping, calibration, metrics) plus 2 days of analysis. Compute:
  phase 2 as planned plus maybe 5–10% for saving scores, plus about 2–4 h for the prompt-sensitivity subset.

### 4.2 Experiment E2: representation probe (tiny training)

- **Features:** pooled hidden states at about 5 depths (including the projector output) for the 3 audio LLMs, saved in
  phase 2. emotion2vec+ features as reference.
- **Protocols:** (i) leave-one-dataset-out across 13 corpora (main; tests transfer); (ii) speaker-independent 5-fold
  within each dataset (in-domain ceiling).
- **Readout:** multinomial logistic regression (also try a small MLP); label mapping as in our 7-class scheme.
- **Key plot:** for each model, the accuracy ladder *free text → logits → prior-corrected logits → best-layer probe →
  dedicated SER*, split into seen and unseen datasets. Each step is a separate "gap": format, prior, unused
  information, representation.
- **Controls:** a probe on a text-only input (transcript, or the same prompt without audio) to show that information
  comes from the voice; layer-wise curves; probe on shuffled labels (chance check).
- **Effort [A]:** 1–2 person-days for extraction code (same pass as E1), 2–3 days of analysis. Compute: storage of about
  13–14 GB and minutes of CPU per probe.
- **Optional E3 (later, NVIDIA):** LoRA SFT (d) of one model with leave-one-dataset-out, to compare "probe vs LoRA". 4–6
  person-days and an estimated 10–40 GPU-hours. Only if E2 shows a big unused-information gap.

### 4.3 What to watch

- The experimental design says "zero-shot, as released, no calibration". Readouts (2) to (4) keep the model zero-shot,
  but calibration and probes must be clearly marked as **post-hoc analyses**, separate from the main benchmark table.
- Do not re-run finished work. This only adds capture to phase 2, which has not started.
- API models: whether GPT Realtime and Gemini expose usable label logprobs for audio input was **not checked here**. If
  they do not, they stay text-only in E1.

---

## 5. Source list (all opened on 2026-11-06 unless noted)

Primary, JEV: [TypeSafe blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) ·
[TypeSafe homepage](https://typesafe.ai/) · [Docs: System One](https://docs.typesafe.ai/concepts/system-one) ·
[Docs: Models](https://docs.typesafe.ai/models.md) · [Docs: Choice](https://docs.typesafe.ai/primitives/choice.md) ·
[System One adapter (GitHub)](https://github.com/typesafe-ai/system-one-adapter-python)

Press: [TechCrunch 18 Sep 2026](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) ·
[The Decoder 7 Oct 2026](https://the-decoder.com/openai-launches-decisions-api-that-reduces-complex-evaluations-to-yes-no-or-pick-one/) ·
Forbes and TechCrunch 9 Oct (headlines only, see 1.2)

Weak: [Wikipedia: Jev (AI model)](https://en.wikipedia.org/wiki/Jev_(AI_model)) ·
[HF community blog (dddffgassa)](https://huggingface.co/blog/dddffgassa/jev-ai)

Primary, OpenAI: [Decisions guide](https://developers.openai.com/api/docs/guides/decisions) ·
[Decisions voice guide](https://developers.openai.com/api/docs/guides/decisions-voice) ·
[Community beta announcement](https://community.openai.com/t/decisions-api-is-now-available-in-public-beta/1403877) ·
[OpenAI blog post](https://openai.com/index/decisions-api/) (HTTP 403, not read)

Papers: listed with links in section 2 and section 1.4.

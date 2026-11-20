# Data Preparation for Internal Call-Centre Audio

Status: discussion draft, 2026-11-13. This document records the curation and labelling workflow. The segmentation
part (§2–§4) is implemented in [`preparation_tools/`](../../preparation_tools/README.md) and tested on public and
synthetic audio only, not yet on real calls. Method agreed on 2026-11-13: word timestamps from forced alignment of the existing transcripts
(torchaudio), and cut points chosen for each whole turn by dynamic programming (§2–§4). All numeric settings are
starting values for a pilot, not established model requirements.
Call audio, transcripts, identifiers and derived internal data remain in the company environment, outside this repository.

## 1. Scope and current inputs

Two SER workstreams have different goals:

- **Academic research** (`research_paper/`): public datasets only, with research questions and evaluation findings;
  no internal call data, systems or business details.
- **Internal domain adaptation** (this document): practical accuracy on customer speech from real calls, including
  telephone audio quality and the target emotion distribution. Select and adapt models against internally labelled
  held-out data; the public-benchmark winner is a candidate, not an automatic production choice.

What we already have (confirmed 2026-11-13):

| Input | Detail |
|---|---|
| Recordings | Raw MP3 calls, **mono** (customer and agent on one channel), **English** |
| Diarization | Done upstream with VibeVoice, which gives every turn's start and end timestamps and its transcript. Each customer turn is cut from the call with those timestamps; agent (CCP) turns are removed |
| Short turns | Turns shorter than 2 s are already filtered out |
| Transcripts | One existing transcript **per turn** (from VibeVoice), already cut to the turn. **No word timestamps** |
| Call time | The VibeVoice turn timestamps give each turn's start in the call (`turn_start_in_call_s` in `turns.csv`), so every segment maps back to its time in the call |

Do not repeat diarization or ASR. The task here is to split long customer turns into shorter pieces while keeping
the audio, the existing text and the original call timeline consistent. The workflow stays customer-only, including
any later unlabelled-data adaptation.

Long turns can contain several emotions, but duration alone does not prove a change. Similarly, a short turn can
contain useful anger, laughter or other affect; it should not automatically be treated as neutral. Segmentation
provides practical annotation units, not a guarantee that every unit has one emotion.

**What mono recording means for this workflow.** The agent's voice can still be audible inside a customer turn:
short backchannels ("mm-hmm", "okay"), overlapping speech, or a few words at the turn edges where diarization is
imprecise. Neither VAD nor the aligner can tell the two speakers apart. Three consequences, handled below:

1. A pause candidate that contains agent speech is not a real silence; the boundary score marks it down (§4.3).
2. Agent words are not in the customer transcript, so the aligner has no text for them; this lowers alignment
   confidence in that region, and no cut is allowed next to low-confidence words (§2.4).
3. Listeners label the customer only; clips where the agent dominates go to the bad-clip option.

The figures in `docs/figures/data_prep_*.png` and the corresponding presentation illustrations were made for the
2026-11-02 proposal. They are historical schematics and do not define the rules below.

## 2. Word timestamps from the existing transcript (forced alignment)

### 2.1 Why alignment, and what it is

To cut a turn and cut its transcript at exactly the same place, we need to know **when each word was spoken**.
The transcript has no word times, so we add them with **forced alignment**: the model is given the audio *and the
words that were said*, and only has to find where each word sits in time. It does not recognise speech again, so it
does not produce a new transcript, and it is much simpler and faster than ASR.

**Never split text by character count, word count or percentage of audio duration.** A proportional split puts
words in the wrong segment whenever the speaking rate changes, which is exactly the case for slow, pausy speakers.

### 2.2 Tool choice: `torchaudio.functional.forced_align` with `WAV2VEC2_ASR_BASE_960H`

Decision (2026-11-13): use torchaudio's built-in CTC forced-alignment function with torchaudio's English wav2vec 2.0
model. Comparison of the options:

| Option | Model licence | Extra installs | Verdict |
|---|---|---|---|
| **torchaudio `forced_align` + `WAV2VEC2_ASR_BASE_960H`** | MIT ("originally published by the authors of wav2vec 2.0 under MIT License") | None: torchaudio 2.7.1 is already in `emotion_recognition_llm` | **Default** |
| torchaudio `forced_align` + `WAV2VEC2_ASR_LARGE_LV60K_960H` | MIT | None | Upgrade if the base model aligns poorly on our calls (larger and slower) |
| torchaudio `forced_align` + `MMS_FA` | **CC BY-NC 4.0** (non-commercial) | None | **Not for this internal workstream**: internal business use is commercial use |
| WhisperX `align()` | Uses the same `WAV2VEC2_ASR_BASE_960H` for English | Heavy: pins `torch~=2.8.0` (we use 2.7.1), plus pyannote-audio, faster-whisper, ctranslate2 and more | No gain for English: same acoustic model, more dependencies |
| Montreal Forced Aligner | Check the licence of the pretrained model used | Separate install, pronunciation dictionary | Fallback only, if both wav2vec 2.0 models fail the pilot |

Why `MMS_FA` is not the default even though it is just as convenient: the same `forced_align` call works with it,
and it is multilingual, but its weights are published under CC BY-NC 4.0. Our calls are English, so the English MIT
model loses nothing and avoids the licence problem.

Notes:

- **Long-term support.** torchaudio marked `forced_align` as deprecated in 2.8, then reversed the decision: it is
  kept in later releases (pytorch/audio issue #3902). Version 2.7.1 has it without any warning.
- **Weights.** The model (about 95M parameters) is downloaded from download.pytorch.org on first use. If the company
  environment cannot reach that host, mirror the checkpoint the same way as the other model weights.
- **Sample rate.** The model expects 16 kHz. Telephone audio (often 8 kHz) is resampled to 16 kHz for alignment
  only; upsampling does not restore missing frequencies, but alignment only needs approximate word positions.
- **Precision needed.** The model outputs one frame per 20 ms. We cut inside pauses, not inside words, so an error
  of a few frames on a word edge does not matter.
- **Domain.** The model was fine-tuned on LibriSpeech (read audiobooks, 16 kHz). Telephone conversation is out of
  domain. Forced alignment is far more robust than recognition because the text is given, but the pilot must check
  alignment quality on real calls (§8).

### 2.3 Steps for one turn

1. **Keep the source unchanged.** The original audio and transcript are never edited.
2. **Make an alignment copy of the text.** The model only knows the letters A–Z and the apostrophe. Upper-case the
   text, write numbers and symbols as words ("$50" → "FIFTY DOLLARS", "24/7" → "TWENTY FOUR SEVEN"), and drop
   punctuation. Keep a word-by-word mapping back to the original text, so the output text keeps its original
   spelling and punctuation.
3. **Handle words that cannot be spelled.** Redaction placeholders (for example `[NAME]`, `***`), annotations in
   brackets or parentheses (including multi-word ones such as `(inaudible speech)`), numbers with several possible
   readings (`-50`, `1/2`, `10-15`, `555-1234`) and unknown symbols get no letters. Option A: leave them out of alignment; they inherit the gap between their neighbours and are never
   used as cut points. Option B (as WhisperX does): map them to a wildcard token that matches any speech.
4. **Align.** Compute the model's frame scores (emissions) for the turn, run `forced_align` on the letter sequence,
   merge consecutive frames into letter spans, and group letters into words. Each word gets a start time, an end time
   and a confidence (the mean score of its letters).
5. **Derive the gaps.** For every pair of neighbouring words, `gap = start(next word) − end(word)`. These gaps are
   the only places where a cut may happen (§4).

### 2.4 Quality checks

- **Whole-turn check.** If the mean word confidence of a turn is low, or a large share of words could not be
  aligned, the transcript and audio probably disagree (ASR errors, agent speech, noise). Do not split the turn: keep
  it whole in the `alignment_issues` group (§4.7). Never invent timings.
- **Local check.** A gap next to a low-confidence word is not used as a cut point, because that word's edges are
  uncertain. This also keeps cuts away from regions where agent speech confused the aligner.
- **Output check.** After cutting, every original word appears in exactly one segment, in order, and no cut falls
  inside a word.

Confidence thresholds are set in the pilot from the score distribution on real calls; they are not universal.

## 3. VAD confirms silence; it does not choose the cuts

[Silero VAD](https://github.com/snakers4/silero-vad) gives a **speech probability for every 32 ms** of audio. It is
MIT-licensed, runs on a CPU (under 1 ms per chunk), and supports 8 kHz directly, so telephone audio needs no
resampling for VAD.

**Why we do not use its own segments.** Its `get_speech_timestamps()` closes a speech chunk after any silence longer
than `min_silence_duration_ms` (default 100 ms). A slow speaker who pauses every few words would come out as dozens
of 1–2 s pieces. Its `max_speech_duration_s` option splits an over-long chunk at the longest silence inside it, but
it works front to back, one chunk at a time, so the last piece can be a very short tail. It also knows nothing about
words or sentences.

**How we use it.** Only its frame probabilities, to answer one question for each word gap from §2: *is this gap
really quiet?* Low probability across the gap means real silence. A gap with high probability is not a clean pause:
it may hold a breath, a sigh, a laugh, crying, background voices or the agent's backchannel. Cutting there would split
a sound that may carry the emotion, so such a gap gets a lower boundary score (§4.3).

VAD can also mark low-volume speech, crying, laughter and sighs as non-speech. **Non-speech is never deleted.** Pauses
and non-verbal sounds inside a segment stay in the audio: hesitation, sighing and laughing are part of the emotion.
Edge trimming is off by default, because VAD can miss quiet speech, crying or sighs at the start or end of a turn.
If it is turned on after checking real calls, it removes only long silence at the very start and end of an aligned
long turn, keeps about 0.2 s of margin, never cuts into an aligned word, and never applies to turns exported whole.

## 4. Choosing the cut points: whole-turn dynamic programming

### 4.1 The idea in one paragraph

The usual approach asks "is this pause long enough to cut?" and needs a fixed pause threshold. That threshold is
wrong for someone: too small for slow speakers (too many pieces), too large for fast speakers (no cut at all). We ask a
different question: **"into how many pieces should this turn be cut, and at which word gaps, so that the pieces have
good lengths and the cuts fall at the best pauses?"** We write down a cost for every possible way of cutting the turn
and pick the cheapest one. Dynamic programming finds that cheapest way exactly and quickly, without trying every
combination one by one.

The only fixed numbers are **segment-length limits**, which come from what the models and the listeners need. How
strong a pause is, is judged **relative to the same speaker**, so no absolute pause threshold is hard-coded.

### 4.2 Which turns are cut, and how long the pieces should be

All durations are elapsed time in the continuous turn audio, including internal pauses.

| Rule | Starting value | Reason |
|---|---|---|
| English turns that are never cut | ≤ 10 s | A 10 s piece is one or two sentences; a listener can give it one label |
| Target segment length | 7 s (good range 4–10 s) | Close to sentence length, where SER models and listeners work best |
| Hard window | 2–15 s | No segment may fall outside it. 2 s is the upstream minimum turn length; 15 s is Vox-Profile Whisper's fixed window |

Model limits behind these numbers: Vox-Profile Whisper was trained on 3–15 s clips, cuts audio after 15 s, and its
card calls predictions under 3 s unreliable; Whisper-encoder models (MERaLiON-SER-v1, Melati) cut audio after 30 s;
emotion2vec+ cannot run on the Mac GPU above 20.48 s; Qwen2-Audio and Gemma 4 truncate after 30 s. A 15 s maximum
stays inside all of these. In the 13 acted/TV datasets of our public benchmark, 31,249 of 56,264 clips are shorter
than 3 s, so these models mostly know sentence-length input.

The window is a hard limit only; the length cost (§4.4) is what keeps pieces near 7 s. A 2 s or 12 s piece already
costs 0.51 and a 15 s piece 1.31, so the dynamic programme uses the edges of the window only when nothing better
exists. The wider window therefore costs little quality and replaces the earlier two-pass rule (3–12 s, then 2–15 s
if nothing fitted) with a single pass. On the synthetic DailyTalk calls, both windows give exactly the same cuts.

A turn of 10–15 s goes through the same algorithm, but "no cut" is one of its options, so it is only cut when a
pause makes that worthwhile. With only poor pauses (`s` about 0.2) it stays whole up to about 13 s.

### 4.3 How good is each candidate cut? (boundary score, 0 to 1)

Every gap between two neighbouring words is a candidate, even a very short one. This matters for fast speakers who
barely pause: the algorithm still has somewhere to cut, just at a low score. Each candidate gets a score `s` between
0 (bad place to cut) and 1 (ideal place) from three cues:

```text
s = 0.5 × p  +  0.3 × q  +  0.2 × v
```

- **p, relative pause length.** First find this speaker's *typical pause* in this turn: the median of all gaps longer
  than 0.1 s (if there are fewer than three such gaps, use 0.3 s). Then `p = min(gap / (2 × typical pause), 1)`.
  A pause twice as long as the speaker's usual pause gets full marks. A slow speaker whose typical pause is 0.6 s gets
  full marks only at 1.2 s, so their many ordinary pauses score about 0.5 and do not attract cuts. A fast speaker
  whose typical pause is 0.2 s gets full marks at 0.4 s. **This is what replaces a hard-coded pause threshold.**
- **q, text cue.** `q = 1` if the word before the gap ends a sentence (`.` `?` `!`), `0.5` after a comma, semicolon
  or colon, `0` otherwise. Sentence ends are usually also ends of a thought and of an intonation phrase. If the
  transcripts carry no punctuation, set its weight to 0 and use `s = 0.7 × p + 0.3 × v`.
- **v, real silence.** `v = 1 − (mean Silero speech probability inside the gap)`. A quiet gap scores near 1; a gap
  holding a sigh, laugh or agent backchannel scores low.

The weights 0.5 / 0.3 / 0.2 are starting values for the pilot. The cut itself is placed in the middle of the quiet part of the chosen gap: among the 32 ms frames whose VAD probability is within 0.1 of the lowest one in the gap, the one nearest the gap centre. This splits the pause evenly between the two segments and keeps the cut away from word onsets, so neither neighbouring word loses its first or last sound.

### 4.4 The cost of one way of cutting

A "way of cutting" (a partition) is a set of chosen gaps. It turns the turn into consecutive segments. Its cost has
two parts:

```text
cost = Σ over segments  L(duration)   +   λ × Σ over cuts  (1 − s)

L(d) = ((d − 7 s) / 7 s)²   if 2 s ≤ d ≤ 15 s
L(d) = ∞                    otherwise (not allowed)
```

- **Length cost `L`.** Zero at the 7 s target and growing the further a segment is from it: about 0.003 at 7.4 s,
  0.12 at 4.6 s, 0.24 at 10.4 s, 0.33 at 3 s, 0.51 at 2 s and at 12 s, 1.31 at 15 s. Outside 2–15 s the partition
  is not allowed at all.
- **Cut cost `λ × (1 − s)`.** Every cut has a price. A perfect pause (`s = 1`) is free; a poor one (`s = 0.2`) costs
  0.8. `λ` (start: 1.0) sets how much boundary quality matters compared with even lengths.

**Why this cannot over-fragment a slow speaker.** Three things work together: (1) the hard 2 s minimum allows at most
one piece per 2 s, and a 2 s piece already costs 0.51; (2) the length cost prefers pieces of about 7 s, so a 40 s turn naturally wants 5–6 pieces, not 20;
(3) each extra cut costs `λ × (1 − s)`, so a cut is made only when it improves the lengths by more than it costs.
Many pauses do not mean many cuts; they only mean more choice of *where* the 5–6 cuts go.

**Why it still cuts a fast speaker.** If a turn is longer than 15 s, every allowed partition needs at least one cut,
so the algorithm must cut somewhere and picks the best available gap, even a short one. Because poor cuts are
expensive, it makes few cuts; the length cost still keeps the pieces near 7 s.

### 4.5 How dynamic programming finds the cheapest partition

Put the candidate gaps in time order and add the turn start (`t₀ = 0`) and turn end (`t_M = T`) as fixed ends. For
every candidate `j`, store `best[j]`: the lowest possible cost of cutting the audio from the start up to `t_j`, with
a cut at `t_j`.

1. `best[0] = 0`.
2. For each later candidate `j`, try every earlier candidate `i` as the start of the last segment `[t_i, t_j]`:
   `best[i] + L(t_j − t_i) + λ × (1 − s_j)`. Keep the smallest value as `best[j]` and remember which `i` gave it.
   The turn end gets `s = 1`, so it adds no cut cost.
3. `best[M]` is the cost of the best partition of the whole turn. Follow the remembered `i` values back from the end
   to read off the chosen cuts.

This is exact: it gives the same answer as comparing every possible partition, because the best partition up to `t_j`
always extends the best partition up to some earlier `t_i`. It is also fast: a 40 s turn has roughly 100 words, so
about 100 × 100 / 2 = 5,000 comparisons, which takes milliseconds. Only `i` within 15 s of `j` need checking. The
result is deterministic: the same audio and transcript always give the same cuts.

Reference sketch (for explanation; the implementation will add input checks and logging):

```python
def length_cost(d, target=7.0, lo=2.0, hi=15.0):
    if d < lo or d > hi:
        return float("inf")                      # not allowed
    return ((d - target) / target) ** 2

def best_cuts(times, scores, lam=1.0):
    """times: [0, candidate gaps..., T] in seconds; scores: boundary score per time (turn end = 1)."""
    n = len(times)
    best = [0.0] + [float("inf")] * (n - 1)       # best[j]: cheapest cost up to times[j]
    prev = [None] * n
    for j in range(1, n):
        for i in range(j):
            cost = best[i] + length_cost(times[j] - times[i]) + lam * (1 - scores[j])
            if cost < best[j]:
                best[j], prev[j] = cost, i
    if best[-1] == float("inf"):
        return None                               # no allowed partition: boundary_issues (§4.7)
    cuts, j = [], prev[n - 1]
    while j:                                      # walk back to the start (index 0)
        cuts.append(times[j])
        j = prev[j]
    return sorted(cuts)
```

### 4.6 Worked example: a slow speaker, 40 s turn

The speaker pauses 25 times; their typical pause is 0.6 s, so full marks start at 1.2 s. The strongest candidates
(the other ~18 pauses are ordinary 0.4–0.6 s pauses with scores around 0.2–0.45):

| Gap at | Gap length | Before it | VAD in gap | p | q | v | Score s |
|---|---|---|---|---|---|---|---|
| 7.4 s | 1.2 s | "…charged twice." | quiet | 1.00 | 1 | 0.95 | **0.99** |
| 12.1 s | 0.6 s | no punctuation | quiet | 0.50 | 0 | 0.90 | 0.43 |
| 15.3 s | 1.0 s | comma | quiet | 0.83 | 0.5 | 0.90 | **0.75** |
| 21.8 s | 1.4 s | full stop | a sigh | 1.00 | 1 | 0.20 | 0.84 |
| 23.0 s | 0.9 s | full stop | quiet | 0.75 | 1 | 0.95 | **0.87** |
| 29.6 s | 1.3 s | "?" | quiet | 1.00 | 1 | 0.95 | **0.99** |
| 34.2 s | 0.7 s | comma | quiet | 0.58 | 0.5 | 0.90 | 0.62 |

Two of the many allowed partitions, compared:

| Partition | Segment lengths | Length cost | Cut cost | Total |
|---|---|---|---|---|
| **A: cuts at 7.4, 15.3, 23.0, 29.6** | 7.4, 7.9, 7.7, 6.6, 10.4 | 0.27 | 0.41 | **0.68** |
| B: A plus a cut at 34.2 | 7.4, 7.9, 7.7, 6.6, 4.6, 5.8 | 0.18 | 0.79 | 0.97 |

The dynamic programme returns A: **5 segments of 6.6–10.4 s**, all at sentence ends or long quiet pauses. The extra
cut in B evens out the last piece a little, but a 0.62 boundary costs more than it saves. The 21.8 s gap is long and
ends a sentence, but it holds a sigh, so the cut goes 1.2 s later at the quiet 23.0 s gap and the sigh stays intact.
The 2–15 s window does not change this result: A and B keep their totals, and no partition with a 2–3 s or 12–15 s
piece is cheaper (the unit tests check this against all 116,674 allowed partitions).

For comparison, on the same turn: Silero's own segments with a 300 ms silence rule give about 25 pieces of mostly
1–2 s. A greedy rule ("cut at the first pause after 7 s") takes whichever ordinary 0.4–0.6 s pause comes first after
each 7 s mark, so most of its cuts land mid-thought, and it never weighs an early cut against a better one a second
later; its last piece is whatever is left over.

### 4.7 Fast speakers, failures and the four output groups

There is no fallback pass. Every turn ends in exactly one of four groups, checked in this order:

1. Transcript detected as Spanish or mixed, any length → **`non_english`**, exported whole.
2. English turn ≤ 10 s → **`final`**, exported whole.
3. Longer English turn whose alignment failed or failed the check (§2.4), or with an empty transcript →
   **`alignment_issues`**, exported whole.
4. Longer English turn with no partition inside 2–15 s → **`boundary_issues`**, exported whole.
5. Otherwise → **`final`**, cut by the dynamic programme (or whole, when "no cut" is cheapest).

| Case | What happens |
|---|---|
| Fast speaker, 20 s, almost no pauses | The turn needs one cut (20 > 15 s). Splitting into 2 × 10 s costs about 0.37 in length plus one cut; 3 × 6.7 s has a smaller length cost but needs two poor cuts (~0.8 each), so the result is **2 segments**, cut at the best gap near the middle (a sentence end there wins over a poor pause). |
| Turn of 10–15 s with no good pause | "No cut" wins up to about 13 s; the turn stays whole in `final`. A longer one gets one cut. |
| No allowed partition (for example a 20 s stretch without any usable word gap) | Keep the turn whole in `boundary_issues`, reason "no partition within 2–15 s". |
| Alignment failed (§2.4) | Do not split; keep the turn whole in `alignment_issues`, with the failed check as the reason (for example "mean confidence 0.41 < 0.5"). Audio could be cut at VAD silences alone, but the text could not follow it reliably. |
| Spanish or mixed transcript | Checked first, whatever the length; kept whole in `non_english`, because the English model must not align Spanish text. |
| Very slow speech with long silences | Flag `low_speech_ratio` (VAD speech time / duration) for review; never join speech islands or delete silence to make a piece fit. |

### 4.8 Settings to tune in the pilot

| Setting | Start | Turning it up | Turning it down |
|---|---|---|---|
| `keep_whole_max` | 10 s | Fewer turns are cut | More turns are cut |
| Target length | 7 s | Longer segments, fewer of them | Shorter segments; consider 5 s if listeners often hear two emotions in one segment |
| Hard window | 2–15 s | — | — (hard limits from upstream and model limits; change only with evidence) |
| `λ` | 1.0 | Cuts follow the best pauses even at uneven lengths | Lengths more even, pause quality matters less |
| Weights of p / q / v | 0.5 / 0.3 / 0.2 | — | Use 0.7 / 0 / 0.3 without punctuation |
| Typical-pause rule | median of gaps > 0.1 s; 0.3 s if fewer than 3 | — | — |

Each segment keeps its turn ID, its index in the turn, its start and end in the turn and in the call, the scores of
its two boundaries, its group and the reason for it (§6), so the settings can be changed later and every label stays
traceable.

Implementation (2026-11-13): [`preparation_tools/`](../../preparation_tools/README.md) implements §2–§4 and the §6 delivery layout offline; its `config/segment_turns.yaml` holds the settings above.

### 4.9 Rules that apply to every segment

- **No overlap.** Neighbouring segments share a boundary but no audio, so the same words are never labelled twice and
  cannot end up in both training and test data.
- **Segments stay inside one turn.** No piece crosses into another customer turn or into the removed agent audio.
- **The same segmenter in training and in production.** Clips seen by the model in production must be cut the same
  way as the labelled clips, otherwise we create a new mismatch.
- **We do not cut at "emotion changes".** Before labelling we do not know where the emotion changes, and using a
  model to guess change points would build that model's bias into the labels. Natural 6–8 s pieces show a change as
  different labels on neighbouring segments.

## 5. Preserve the emotion timeline and context

Each exported segment is a continuous interval within one original customer turn. Keep ordered segment boundaries
and predictions in turn and call time. Do not replace the sequence with a duration-weighted average: an optional
turn/call summary should remain a separate derived view so a neutral → angry → sad progression is still inspectable.
Gaps between turns remain gaps; removing CCP audio does not justify closing time on the original call timeline.

A segment can still contain an emotion transition, even within a sentence. The Label Studio setup keeps the
listener's task simple: choose one emotion, or discard the clip if the audio is unusable; there is no separate
mixed/transition option. A segment with two emotions therefore shows up as disagreement between listeners. If the
pilot finds much disagreement of this kind, shorten the target segment length (§4.8) rather than adding options.

Distinguish the **target interval being labelled** from optional preceding/following **listening context**.
Context does not inherit the target label, and its transcript must not be silently included in the target text.
Start with non-overlapping target clips; if later inference uses overlapping context, retain the target intervals
and document how predictions map back to them. The pilot must check whether the selected labelling setup supports
the required context and review workflow; a flat classification task alone does not implement timeline editing.

## 6. Delivery to Label Studio

The independent [`label_studio/` subproject](../../label_studio/README.md) owns the annotation setup and import
interface; this document defines the upstream segmentation and delivery contract.

Use Label Studio for **emotion classification of prepared audio**, with the matching existing transcript as task
data. This is not a speech-transcription task: listeners should not be asked to transcribe every clip again.
The segmenter delivers four groups with the same layout, one folder per call inside each group, inside the company
environment:

```text
batch/
  final/                    clean results: the only group for labelling, benchmarking and fine-tuning
    manifest.csv            one row per target segment, with identity, paths, timing and context links
    <call_id>/
      <segment_id>.wav      <call_id>_t<turn order>_s<segment index>: name order = time order in the call
      <segment_id>.txt      the original transcript text of that segment
  non_english/              same layout
  alignment_issues/         same layout
  boundary_issues/          same layout
  review/<turn_id>.json     audit of every turn: word times, gap scores, chosen cuts, group and reason
  call_timeline.csv         every exported item of every group, by call and order, with the links
  failures.csv              turns that raised an error
```

| Group | Content (§4.7) | Purpose |
|---|---|---|
| `final` | English turns ≤ 10 s exported whole, and longer English turns cut by the dynamic programme | Human labelling, benchmarking and fine-tuning |
| `non_english` | Spanish or mixed turns, any length, exported whole | Kept, not labelled |
| `alignment_issues` | Long English turns whose alignment failed or whose transcript is empty, exported whole | Kept, not labelled; also a sample for checking alignment on real calls |
| `boundary_issues` | Long English turns with no partition inside 2–15 s, exported whole | Kept, not labelled; shows where the window is too tight |

Only `final` is used for labelling, benchmarking and fine-tuning. The other three groups are kept because unlabelled
audio — long, non-English or hard to align — can still serve later self-supervised continued pre-training (e.g. of
an emotion2vec-style encoder, or of a Whisper encoder using the transcripts); no such step is implemented.

**Context links.** Conversational datasets such as DailyTalk and MELD keep each utterance's place in its dialogue,
so a model can be given the preceding turns as context. The delivery does the same for calls. Turns are grouped by
call (`turns.csv`, else one input folder per call) and ordered by turn index, else by start time. Every exported item
of every group gets `order_in_call`, its previous and next item (`prev_segment_id`, `prev_group`,
`next_segment_id`, `next_group`) and the call-time gaps to them (`gap_to_prev_s`, `gap_to_next_s`). Links cross
groups: a labelled segment in `final` can have a Spanish turn from `non_english` as its neighbour. A gap between two
customer turns usually holds the agent's turn, which is not in the data. `call_timeline.csv` lists all items of all
groups in call order and is the single file for in-context experiments. As in §5, context does not inherit the
target label.

An import adapter can map each row of `final/manifest.csv` to one Label Studio classification task, exposing the
audio and text through a company-approved local or internal serving method. The exact Label Studio task schema,
configuration, storage setup and export mapping belong to the separate subproject and are not specified by this
discussion draft. Keep the interface independent of the annotation tool. The existing offline
[`labeling_tool/`](../../labeling_tool/README.md) remains an option; it expects an `audio/` folder and same-stem
`.txt` files alongside the audio, so it requires a small packaging step from the per-call folders above.

Manifest fields (every group has the same columns):

| Group | Fields / purpose |
|---|---|
| Identity | `segment_id`, `group`, `reason`, pseudonymized `call_id`, `customer_id`, `turn_id`, `turn_index`, `turn_order`, `segment_index` |
| Files and text | `audio_path`, `transcript_path`, `review_path` (word times and the alignment-text mapping), source transcript and its SHA-256 |
| Time | `turn_start_in_call_s`, `segment_start_in_turn_s`, `segment_end_in_turn_s`, `segment_start_in_call_s`, `segment_end_in_call_s`, `duration_s` |
| Context | `order_in_call`, `prev_segment_id`, `prev_group`, `next_segment_id`, `next_group`, `gap_to_prev_s`, `gap_to_next_s` |
| Coverage and quality | `speech_duration_s`, `speech_ratio`, `low_speech_ratio`, alignment status and confidence, boundary scores |
| Provenance | source-audio reference and SHA-256, sample rate, segmentation settings/version |
| Annotation linkage (added later) | split, Label Studio task ID when available, label-set version, target/event intervals |

Validate the clock origin of any reused timestamps. For a contiguous turn extraction, call time is turn start plus
the within-turn offset; if upstream extraction removed internal intervals, recover a piecewise source-time mapping
instead of applying a false constant offset. Verify MP3 decoding and extraction offsets on examples. Missing
original-call offsets must be recorded as unresolved, not reconstructed by placing customer clips back-to-back.

Prefer decoding each source once and exporting lossless working clips; resample for the selected model's required
input without assuming that upsampling restores missing telephone frequencies. Preserve source references and
timing through conversion. Boundary changes need new IDs/revisions and invalidation or re-review of affected labels,
rather than silently moving an already labelled target.

## 7. Labelling, model selection and adaptation

Agree on a useful internal label set and written examples before scaling annotation. Options include the existing
seven emotions plus unknown, or call-centre categories such as frustrated; neither is confirmed here. Keep
unjudgeable audio (the discard option) separate from emotion choices. Use independent listeners for seed/evaluation data
and hide model hints there; preserve individual votes and review disagreements instead of forcing consensus.

Sample across calls, customers, languages, recording conditions and emotion candidates, with a random component
to check selection bias. Separate training, validation and test by customer, keeping every segment and context
from the same call in one split. If reliable customer linkage is unavailable, call-disjoint splitting is only a
fallback and must be documented as insufficient to establish customer independence.

Choose the model and segmentation settings on internal validation data; reserve an independent internally labelled
test set for final evaluation. Public results help form the shortlist. Compare zero-shot and adapted models using
per-class recall, macro-F1/UAR and the actual business-relevant errors, alongside annotation agreement and cost.
Use validation data for active-learning stopping and repeated tuning; do not repeatedly optimize against the test set.

The existing presentation's A–D supervised adaptation options and E unlabelled-data option remain possible routes,
not commitments to emotion2vec+ or MERaLiON. Start from the model that performs best under the internal evaluation
and operational constraints. If continued pre-training needs different window lengths, define that separately;
classification labels and their target intervals must remain traceable. No agent speech is introduced by this workflow.

## 8. Pilot before implementing the full pipeline

Select a manageable sample of customer turns spanning short/long duration, slow speech, frequent pauses, no clear
pauses, low volume, expressive vocalizations, noisy audio and likely within-turn emotion changes. Compare at least
a pause-triggered baseline (Silero's own segments) with the whole-turn dynamic programming of §4 on the same turns.
Keep protected internal data out of repository examples and logs.

Review boundary quality (clipped words, short tails, excessive fragmentation), transcript/audio agreement,
unaligned-word rates, speech occupancy, retained emotional events and the frequency of manual exceptions. Then
compare listener agreement, bad-clip rates, labelling time and internal model performance. Check alignment on real
calls: listen to a sample of word boundaries from `WAV2VEC2_ASR_BASE_960H`, and try the larger
`WAV2VEC2_ASR_LARGE_LV60K_960H` if the base model misplaces many words. Word confidences of different aligners are
not calibrated probabilities and are not compared across models.

The pilot should settle the duration settings, scoring weights, alignment model, review thresholds, label
set, context policy and Label Studio configuration. No library has yet been tested on internal clips in this
document, and no executable pipeline or final segmentation policy is claimed.

## Sources and existing project context

- [Silero VAD official repository](https://github.com/snakers4/silero-vad) and
  [timestamp utility source](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py): speech-activity detection and its post-processing parameters.
- [torchaudio CTC forced alignment tutorial](https://docs.pytorch.org/audio/2.8/tutorials/ctc_forced_alignment_api_tutorial.html) and
  [`forced_align` reference](https://docs.pytorch.org/audio/2.8/generated/torchaudio.functional.forced_align.html): the alignment function used here.
- [`WAV2VEC2_ASR_BASE_960H`](https://docs.pytorch.org/audio/2.8/generated/torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.html) and
  [`WAV2VEC2_ASR_LARGE_LV60K_960H`](https://docs.pytorch.org/audio/2.8/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H.html): English models, MIT licence.
- [`MMS_FA`](https://docs.pytorch.org/audio/2.8/generated/torchaudio.pipelines.MMS_FA.html): multilingual alignment model, CC BY-NC 4.0, so not used here.
- [pytorch/audio issue #3902](https://github.com/pytorch/audio/issues/3902): `forced_align` was deprecated in 2.8 and then kept.
- [WhisperX alignment source](https://github.com/m-bain/whisperX/blob/main/whisperx/alignment.py) and
  [its dependency list](https://github.com/m-bain/whisperX/blob/main/pyproject.toml): same English model, wildcard handling of unknown characters, heavier dependencies.
- [Montreal Forced Aligner: alignment workflow](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html): fallback aligner using an acoustic model and pronunciation dictionary.
- [Research-paper scope](../../research_paper/README.md), [offline labelling tool](../../labeling_tool/README.md),
  [emotion2vec+ adaptation notes](finetuning_emotion2vec_plus_large_en.md) and `presentation/s_future.js`.
- Historical figures: `docs/figures/make_data_preparation_figures.py`; these are not the updated segmentation specification.

# Facts for the turn-segmentation deck

Every number on a slide, with its source. Collected 2026-11-13. Paths are relative to the repository root.
`DOC` = `emotion_recognition/docs/data_preparation_call_centre_audio.md`, `PT` = `preparation_tools/`,
`CFG` = `PT/config/segment_turns.yaml`. Numbers marked *computed* were produced by this folder's scripts from repository
code and data (no new model runs except Silero VAD); `assets/` holds their outputs.

Numbers follow the design of 2026-11-13 (late): language check first, four output groups (`final/`, `non_english/`,
`alignment_issues/`, `boundary_issues/`) grouped per call, one DP window of 2–15 s (no relaxed pass), and the
integration run on 5 synthetic DailyTalk calls with the improved cut placement (v3; `assets/integration_run.log`, copy of
the run log with paths removed).
Slide numbers below are the shown slides 1–36; the hidden appendix slides are A1–A6.

## Problem statement (slides 2–3)

| Fact | Value | Source |
|---|---|---|
| emotion2vec+ large input | real length, no window | `emotion_recognition/docs/models_survey_en.md` (input-length note, line 25); `research_paper/docs/experimental_design.md` §4.2 |
| emotion2vec+ on the Mac GPU | fails above 20.48 s (327,685 samples; 20.4 s runs, 20.5 s fails, measured); scripts fall back to the CPU | `emotion_recognition/docs/emotion2vec_plus_large_en.md` "Length and devices"; `emotion_recognition/scripts/models/emotion2vec.py` docstring of `Predictor` |
| emotion2vec+ memory | grows roughly with length²; about 2.3 GB extra at 60 s on the CPU (0.3 GB at 16 s, 9.3 GB at 120 s) | `emotion2vec_plus_large_en.md` line 26 and "Length and devices" (notes only) |
| MERaLiON-SER-v1, Melati | fixed 30 s Whisper window: shorter padded, longer cut | `models_survey_en.md` table 1b #2, #3; `scripts/models/meralion_ser.py`, `melati.py` docstrings; `experimental_design.md` §4.2 |
| Vox-Profile Whisper | fixed 15 s window; trained on 3–15 s clips; < 3 s unreliable (card) | `models_survey_en.md` table 1b #4; `scripts/models/vox_profile.py` (`MAX_SECONDS = 15`, notes) |
| Vox-Profile WavLM | real length, at most 15 s (cut by our loader); trained on 3–15 s | `models_survey_en.md` table 1b #5; `experimental_design.md` §4.2; `vox_profile.py` (`wav[:MAX_SECONDS * SR]`) |
| Qwen2-Audio 7B, Gemma 4 E4B | at most 30 s, longer audio cut (Gemma: 750 audio tokens × 40 ms) | `scripts/models/qwen2_audio.py`, `gemma4_audio.py` (`MAX_SECONDS = 30`); `experimental_design.md` §4.2 |
| Qwen2.5-Omni 7B | variable length, processor cap 300 s | `models_survey_en.md` (audio-LLM table, Qwen2.5-Omni row); `experimental_design.md` §4.2; loader `MAX_SECONDS = None` |
| API models (GPT Realtime, Gemini) | input-length limit **not checked**; left off the table, said so in the notes | — |
| Benchmark clips under 3 s | 31,249 of 56,264 clips in the 13 acted/TV datasets | DOC §4.2 |
| Turns of 40–50 s or longer exist but are not common | user statement (relayed by the main session) | — |
| Segment illustration (neutral → frustrated → angry) | illustration, labelled "not real data" on the slide | — |

## Procedure and rules (slides 4–5)

| Fact | Value | Source |
|---|---|---|
| Routing (slides 4, 7, 9, 28) | language first: es/mixed of any length → `non_english/` (whole); English ≤ 10 s → `final/` (whole); English > 10 s → alignment; failed check or empty text → `alignment_issues/`; no partition in 2–15 s → `boundary_issues/`; else cut → `final/`; errors → `failures.csv` | `PT/segmenter/pipeline.py` docstring and `GROUPS`; `CFG` `paths.groups`; main-session design note of 2026-11-13 |
| Upstream steps 1–5 | whole mono MP3 call → VibeVoice-ASR (diarization, turn start/end, transcript) → keep customer turns, drop agent (CCP) turns → cut turn audio with its transcript → drop turns < 2 s | user description relayed by the main session; DOC §1 table (mono MP3, English, diarized customer turns, < 2 s filtered, transcript per turn, no word timestamps) |
| VibeVoice-ASR | one model for ASR, speaker diarization and timestamps ("who, when, what"); 9B parameters; up to 60 min; MIT | arXiv 2601.18184 (VibeVoice-ASR Technical Report, Peng et al., 2026); https://huggingface.co/microsoft/VibeVoice-ASR (checked 2026-11-13) |
| turns.csv | `turn_id, call_id, turn_index, turn_start_in_call_s[, customer_id]`; missing → `call_time_status = unresolved` | `PT/README.md` "Input"; `PT/segmenter/pipeline.py` (`export_turn`) |
| Minority of Spanish calls | "English with a minority in Spanish" | `PT/README.md` "Spanish turns" |
| Rules | no re-ASR or diarization; offline; commercial-use licences; deterministic; never delete audio inside a turn; audio and text cut together | DOC §1, §2.2, §3, §4.5, §4.9; `PT/README.md` |

## Pipeline and step slides (6–14)

| Fact | Value | Source |
|---|---|---|
| Decode formats | WAV, MP3, FLAC via soundfile (libsndfile 1.2.2) | `PT/segmenter/audio_io.py`; `PT/LICENSES.md` |
| Analysis copy | mono, 16 kHz; source kept for export at its own rate and channels | `audio_io.py` (`analysis_copy`, `write_clip`) |
| Clip subtype | source subtype if WAV supports it, else PCM_16 | `audio_io.py` (`wav_subtype`) |
| Silero frame | 512 samples = 32 ms at 16 kHz | `PT/segmenter/vad.py` (`FRAME_SAMPLES`) |
| Keep-whole limit | 10 s, tested on the full turn (decision span = whole turn while trim is off) | `CFG` `keep_whole_max_s: 10`; `pipeline.py` `segment_turn` |
| Edge trim | off by default; if on: > 1.0 s edge silence, 0.2 s margin, only shortens the decision span; only aligned long turns lose edge silence, never into a word; whole-turn exports never trimmed | `CFG` `vad.trim_edges: false`, `trim_min_silence_s`, `trim_margin_s`; `pipeline.py` `trim_span`, `segment_turn` |
| Language rule | Spanish share ≥ 0.8 → es; ≤ 0.2 or fewer than 2 cue words → en; else mixed; accents and ¿ ¡ count; capitalised accented words not counted | `PT/segmenter/language.py`; `CFG` `language` |
| Language outcome | every es/mixed turn, any length → `non_english/` whole (reason `language: es`) | `pipeline.py`; `output/non_english/manifest.csv` (5 rows, reason `language: es`) |
| Language examples | test strings (not shown on the slide; in notes only as rules) | `PT/tests/test_text.py` |
| Aligner | torchaudio `forced_align` + `WAV2VEC2_ASR_BASE_960H`; 94.4M parameters; English; fine-tuned on LibriSpeech 960 h; MIT | DOC §2.2; `PT/LICENSES.md`; integration log ("94.4M parameters"); torchaudio bundle docs (checked 2026-11-13) |
| Word-time example | Yeah. 0.16–0.50 s, Umm 1.24–1.34 s, it 1.56–1.60 s (d418_long_en) | `output/review/d418_long_en.json` (`words[0:3]`, rounded) |
| Tool comparison table | default / LV60K upgrade / MMS_FA CC BY-NC 4.0 / WhisperX (same model, torch ~2.8, pyannote, faster-whisper) / MFA fallback | DOC §2.2 table |
| Labels | 29: blank, `|`, 26 letters, apostrophe | `PT/model_weights/manifest.json` (`labels`) |
| Frame step | 20 ms (320 samples) | DOC §2.2; `PT/segmenter/alignment.py` (`HOP = 320`) |
| Word confidence | mean letter probability | `alignment.py` (`align`) |
| Trellis figure | illustrative scores (seeded), true best path for them (numpy Viterbi in `scripts/make_figures.py`) | *computed* |
| Text normalisation examples | `$50` → FIFTY DOLLARS; `3:05` → THREE OH FIVE; `24/7` → TWENTY FOUR SEVEN; Café → CAFE; don't → DON'T; `[NAME]`, `***`, `(inaudible speech)`, `A12B`, `4420013`, `555-1234`, `1/2`, `-50` → wildcard | `PT/segmenter/textnorm.py`; `PT/tests/test_text.py`; main-session update of 2026-11-13 (several readings → wildcard; multi-word parenthesised annotations = one token) |
| Wildcard | best non-blank score per frame (as WhisperX); never next to a cut | `alignment.py`; `pipeline.py` `choose_cuts` |
| Long audio | above 64 s: 60 s chunks with 2 s context (notes) | `CFG` `emission_chunk_s: 60`, `emission_context_s: 2`; `alignment.py` (`emissions`) |
| Thresholds | word 0.3; turn mean 0.5; weak/unspelled share 0.4 | `CFG` `alignment` |
| Confidence ranges | mean per turn: correct 0.90–0.95, 8 kHz μ-law 0.88–0.94, 30% replaced 0.63–0.74, wrong 0.27–0.36; weak share ≤ 0.02, ≤ 0.02, 0.18–0.28, 0.51–0.63 | `CFG` comment above the thresholds; `PT/README.md` "Settings" |

## Silero VAD (slides 15–18)

| Fact | Value | Source |
|---|---|---|
| Model structure (16 kHz) | STFT (129 bins) → 4 × (Conv1d k=3 + ReLU) → LSTMCell 128 → Conv1d + sigmoid | *computed*: printed the TorchScript modules of `silero_vad.jit` (silero-vad 6.2.3) |
| Parameters | 243,585 (16 kHz model; the 8 kHz model has 219,009) | *computed* (`assets/data_stats.json` `silero_16k_parameters`) |
| JIT file size | 2.27 MB, inside the wheel | `PT/LICENSES.md` |
| Speed, rates | < 1 ms per chunk on one CPU thread; 8 kHz and 16 kHz | Silero README (github.com/snakers4/silero-vad, checked 2026-11-13); DOC §3 |
| Licence | MIT | Silero README; `PT/LICENSES.md` |
| get_speech_timestamps defaults | threshold 0.5; neg_threshold = threshold − 0.15 = 0.35; min_silence_duration_ms 100; min_speech_duration_ms 250; speech_pad_ms 30; max_speech_duration_s ∞ (split at the longest silence) | `silero_vad/utils_vad.py` of silero-vad 6.2.3 (read from the installed wheel) |
| Silero pieces on the 5 long English turns (32.0–38.8 s) | 100 ms rule: 41 pieces, median 2.8 s, 21 under 3 s; 300 ms rule: 28 pieces, median 5.1 s, 6 under 3 s | *computed* by `scripts/make_data_figures.py` with `get_speech_timestamps_from_probs` on the pipeline's own probabilities (`assets/data_stats.json` `methods`) |
| Figure turn (Silero slide, comparison figure) | d1562_long_en, 38.8 s: Silero 11 (100 ms) / 7 (300 ms) pieces; fixed chop 6; greedy 6; our method 5 | `data_stats.json` `vad_turn` |
| Quietness, cut placement | v = 1 − mean p in the gap; cut in the middle of the quiet part: among frames with p within 0.1 of the gap's lowest, the one nearest the gap centre (was: the single lowest frame) | `vad.py` `gap_silence`, `QUIET_TOLERANCE = 0.1` |
| Cut position in the gap | 0.48–0.60 of the gap length for all 16 cuts (was 0.43–0.92 with the lowest-frame rule) | *computed* (`data_stats.json` `cut_position_in_gap`); old range from the main session |
| Zoom example | d418_long_en, gap after "cameras." 1.66 s, v 0.90, s 0.98, cut 5.552 s in the middle of the quiet part (an utterance join) | `review/d418_long_en.json`; `data_stats.json` `zoom` |
| Method-document estimate | Silero's own segments with a 300 ms rule: about 25 pieces of mostly 1–2 s on the 40 s example | DOC §4.6 |

## Scoring and dynamic programming (slides 19–28)

| Fact | Value | Source |
|---|---|---|
| Score | s = 0.5 p + 0.3 q + 0.2 v; without punctuation 0.7 p + 0.3 v | DOC §4.3; `CFG` `scoring`; `PT/segmenter/boundaries.py` |
| p | min(gap / (2 × typical), 1); typical = median of gaps > 0.1 s, 0.3 s if fewer than 3 | DOC §4.3; `boundaries.py` `typical_pause` |
| q | 1 after . ? !; 0.5 after , ; :; 0 otherwise (abbreviations such as "Mr." are not sentence ends) | DOC §4.3; `textnorm.py` `text_cue` |
| Slow/fast example | typical 0.6 s → full marks 1.2 s, ordinary pause p 0.5; typical 0.2 s → full marks 0.4 s | DOC §4.3; `PT/tests/test_dp.py` `test_relative_pause` |
| Weighted sum is our design | the form s = 0.5p + 0.3q + 0.2v and its weights are not from a paper; starting values for the pilot | DOC §4.3 ("starting values for the pilot") |
| Cue literature | Shriberg et al. 2000: prosodic + lexical combination significantly better than word-only; pause features highly informative (abstract). Wightman et al. 1992: rate-normalised lengthening distinguishes at least four boundary types (abstract). Swerts 1997: pause, pitch range and boundary tone studied as a function of discourse boundary strength (abstract; the direction "longer pauses at stronger boundaries" is from secondary summaries, so the slide only says "pauses cue discourse boundaries"). Goldman-Eisler 1968: pauses longer when interpreting than describing (secondary summaries; the slide says "varies with speaker and task"). Liu et al. 2006: sentence-boundary detection combining textual and prosodic knowledge sources (abstract) | abstracts via Semantic Scholar / OpenAlex; Crossref for bibliographic data (checked 2026-11-13) |
| Cost | Σ L(d) + λ Σ (1 − s); L(d) = ((d − 7)/7)² on [2, 15] s, ∞ outside; λ = 1.0 | `CFG` `segmentation` (min_s 2, max_s 15, target_s 7, cut_weight_lambda 1.0) |
| Window rationale | 2 s = upstream minimum turn length; 15 s = Vox-Profile Whisper window, below emotion2vec+'s 20.48 s Mac-GPU limit and the 30 s windows | `CFG` comment above `min_s` |
| L values | 0 at 7 s; 0.18 at 4 and 10 s; 0.51 at 2 and 12 s; 1.31 at 15 s; 0.003 at 7.4 s; 0.12 at 4.6 s; 0.24 at 10.4 s | arithmetic of L(d) |
| Poor cut = long piece | λ(1 − 0.2) = 0.8 = L(d) at d = 7 + 7√0.8 = 13.26 s | *computed* (`assets/worked_example.json` `poor_cut_equals_length_s`) |
| Good pause example | s = 0.9 costs 0.1; bringing a piece from 10 s to 7 s saves 0.18 in length price | arithmetic |
| Old window gives the same cuts | re-running the DP with 3–12 s on the 5 long turns gives identical cuts (5 of 5, run v3) | re-checked on the v3 review files with `boundaries.best_partition` |
| Recursion, window, end score | best[j] = min_i best[i] + L(t_j − t_i) + λ(1 − s_j); only t_i within 15 s (max_s); best[0] = 0; turn end s = 1 | DOC §4.5; `boundaries.py` `best_partition` |
| Complexity | ≈ 100 words in 40 s → ≈ 5,000 comparisons, milliseconds | DOC §4.5 |
| Brute-force check | DP matches exhaustive search over 116,674 allowed partitions (2–15 s) | `test_dp.py` (assert > 100,000; count from the main session) |
| Worked example | 25 pauses; 7 strong (7.4 s 0.99, 12.1 s 0.43, 15.3 s 0.75, 21.8 s 0.84 with a sigh v 0.20, 23.0 s 0.87, 29.6 s 0.99, 34.2 s 0.62); 18 ordinary 0.4–0.6 s at 0.2–0.45 | DOC §4.6 table; `test_dp.py` `STRONG`, `FILLER_TIMES` |
| DP result (2–15 s) | cuts 7.4, 15.3, 23.0, 29.6; best[T] = 0.677; runner-up 7.4, 15.3, 21.8, 29.6 at 0.707 | *computed* (`assets/worked_example.json` `dp`, `runner_up`); DOC §4.6 |
| DP slide example | from 12.1 s to 29.6 s = 17.5 s > 15 s (not allowed); from 15.3 s = 14.3 s (allowed, L = 1.09) | arithmetic |
| Partition A | lengths 7.4, 7.9, 7.7, 6.6, 10.4; length 0.269, cut 0.408, total 0.677 (doc: 0.27, 0.41, 0.68) | *computed* (`worked_example.json` `A`); DOC §4.6 |
| Partition B | + cut at 34.2; lengths … 4.6, 5.8; length 0.180, cut 0.787, total 0.967 (doc: 0.18, 0.79, 0.97) | *computed* (`worked_example.json` `B`); DOC §4.6 |
| Greedy, worked example (notes only) | cuts 7.4, 15.3, 23.0, 31.2 (31.2 s: s = 0.37); lengths 7.4, 7.9, 7.7, 8.2, 8.8; length 0.125, cut 1.028, total 1.154 | *computed* by `scripts/make_figures.py` `greedy()` on the `test_dp.py` data; rule wording from DOC §4.6 |
| Fast speaker (2–15 s) | 20 s, all gaps s = 0.2 (0.35 s grid): one cut at 10.15 s, total 1.168; with a 0.6 gap at 9.1 s the cut moves to 9.1 s | *computed* with `boundaries.best_partition(lo=2, hi=15)` on the `test_dp.py` case |
| Fast-speaker panel (idealised) | 2 × 10 s: length 0.367 + one cut 0.8 = 1.17; 3 × 6.67 s: 0.007 + 1.6 = 1.61; one cut at 13 s: L(13) = 0.73 alone | arithmetic |
| 10–15 s turn | "no cut" is an option; a turn > 15 s must be cut | DP definition; `CFG` |
| Routing table (slide 28) | groups and reasons as in the routing row above; `low_speech_ratio` stays a per-segment flag (speech ratio < 0.4) | `pipeline.py`; `CFG` `vad.low_speech_ratio: 0.4`; manifest columns |

## Outputs, context, validation (slides 29–36) and appendix (A2–A3)

| Fact | Value | Source |
|---|---|---|
| Output layout | `final/`, `non_english/`, `alignment_issues/`, `boundary_issues/`, each `<call_id>/<segment_id>.wav + .txt` and `manifest.csv`; `review/<turn_id>.json`; `call_timeline.csv`; `failures.csv` only on failure | `output/` of the run; `CFG` `paths`; `pipeline.py` |
| Segment id | `<call_id>_t<turn order, 3 digits>_s<segment, 2 digits>`, from 0, e.g. `dailytalk_d418_t000_s01` | file names in `output/final/dailytalk_d418/` |
| Manifest columns | segment_id, group, reason, call_id, customer_id, turn_id, turn_index, turn_order, segment_index, segment_count, order_in_call, prev_segment_id, prev_group, next_segment_id, next_group, gap_to_prev_s, gap_to_next_s, audio_path, transcript_path, review_path, source_*, turn/segment times in turn and call, call_time_status, duration_s, speech_duration_s, speech_ratio, low_speech_ratio, language, segmentation_status, alignment_*, start/end_boundary_score, word_count, segmentation_version | header of `output/final/manifest.csv` |
| call_timeline.csv columns | call_id, order_in_call, segment_id, group, reason, turn_id, turn_order, segment_index, segment_count, paths, language, duration and times, call_time_status, gap_to_prev_s, gap_to_next_s, prev/next segment id and group | header of `output/call_timeline.csv` |
| Context call (slide 30) | dailytalk_d1562: t000_s00–s04 final, t001_s00 non_english (reason `language: es`, gap 4.0 s), t002_s00 final (gap 4.0 s) | `call_timeline.csv`; `data_stats.json` `context_call` |
| Synthetic calls | 5 DailyTalk dialogues; customer = the speaker with most audio; turns: long English (consecutive utterances, 40% long pauses 0.9–1.4 s, else 0.15–0.35 s, room noise −60 dBFS, 25–35 s target), Spanish-text turn (long > 11 s audio in 3 calls), short English; agent gap 4 s; seed 7; one MP3 | `PT/tests/integration_dailytalk.py`; run log |
| Long turn lengths | 32.2, 32.0, 33.4, 38.8, 32.2 s → "32–39 s" | run log |
| Groups | final 10 turns → 26 clips (5 cut into 21 segments + 5 whole short); non_english 5 turns (3 longer than 10 s: 23.9, 13.4, 14.5 s); alignment_issues 0; boundary_issues 0 | run log v3; manifests |
| Segments of cut turns | 21, 4.91–11.95 s, median 8.15 s, mean 8.03 s; all within 2–15 s | run log v3 ("final segments of cut turns") |
| Cut positions | 16 cuts: 10 inside the inserted pause at an utterance join, 6 at sentence ends (q = 1) inside an utterance → 16/16 | run log v3 (per-turn notes); `data_stats.json` `methods.ours` |
| Level at the cut | median −41 dB, max −37 dB (20 ms vs speech RMS) | run log v3 |
| Coverage | every word exactly once; WAV clips exact slices | asserts in `check_outputs` (passed) |
| Links | 5 of 5 calls correct; each call crosses groups twice; call_timeline.csv 31 rows = 31 manifest rows | run log v3 |
| Alignment confidence of the run | 0.87–0.97 mean per long turn | run log |
| Speed | 3.2–4.8 s per long turn on CPU, 4 threads (run v3) | run log v3 |
| Peak memory | 1.2 GB footprint, measured in the earlier run (not re-measured) | earlier run log (`/usr/bin/time -l`), see notes |
| Unit tests | 67 pass | main session, 2026-11-13 (after the cut-placement change) |
| Listening slides | d418_long_en 32.2 s: 4 segments, bounds 0.0–5.6–13.4–23.6–32.2 s, all 3 cuts at joins (s 0.97–0.98); d1562_long_en 38.8 s: 5 segments, bounds 0.0–7.7–16.2–21.4–28.7–38.8 s, 3 cuts at sentence ends + 1 at a join | manifest; review JSONs; `data_stats.json` `listen`; `assets/audio/manifest.json` |
| Embedded audio | 11 MP3s (2 whole turns + 9 segments), 16 kHz mono 64 kbps, 1.15 MB, no other processing; DailyTalk CC BY-SA 4.0 adaptations | `scripts/make_audio.py`; `assets/audio/manifest.json` |
| Method comparison (slide 34) | over the 5 long turns (v3): Silero 100 ms 41, 2.80 s, 21 < 3 s, 0 in a word, 92% at join/sentence end, mean s 0.93; Silero 300 ms 28, 5.10 s, 6 < 3 s, 0 in a word, 100% at join/sentence end, mean s 0.95; fixed 7 s 26, 7.00 s, 0 < 3 s, 15 in a word, 24% at join/sentence end, mean s 0.24; greedy 26, 7.15 s, 1 < 3 s, 0 in a word, 10% at join/sentence end, mean s 0.37; ours 21, 8.15 s, 0 < 3 s, 0 in a word, 100% at join/sentence end, mean s 0.95 | *computed* by `scripts/make_data_figures.py` (`data_stats.json` `methods`); Silero cut = middle of the silence between chunks; score of the gap containing the cut, 0 inside a word |
| Weight sensitivity (slide 35) | 90 settings (pause 0.3–0.7, text 0–0.5, silence 0.1–0.3, normalised): join-or-sentence-end share 78%–100% (default 100%), 45 settings as good as default, 20–24 pieces, none < 3 s, cuts identical to default median 80% (min 56%); pause only 74% (24 pieces, 37% same cuts); punctuation only 100% (21, 62%); silence only 100% (19, 71%) | *computed* by `scripts/sensitivity.py` (`assets/sensitivity.json`) |
| Settings table (A2) | keep_whole 10; target 7; window 2/15; λ 1.0; 0.5/0.3/0.2 (0.7/0/0.3); typical pause rule; thresholds 0.3/0.5/0.4; trim off | `CFG` |
| Download source (A3) | GitHub Release `preparation-tools-segmenter-v1` (default) or PyPI + download.pytorch.org; size + SHA-256 checks; resumable | `PT/download.py`; `PT/publish.py` |
| Sizes (A3) | wheels any/ 11.5 MB (silero-vad 11.3 MB); Linux set 2.3 MB; Mac set 1.5 MB; checkpoint 377.7 MB; ≈ 391 MB per platform | `PT/dependencies/manifest.json`, `PT/model_weights/manifest.json` |
| Licences (A3) | wav2vec2 checkpoint MIT; silero-vad MIT; soundfile BSD-3 with libsndfile LGPL-2.1; PyYAML MIT; cffi MIT-0; pycparser BSD-3; packaging Apache-2.0 OR BSD-2; MMS_FA and VoxPopuli CC BY-NC 4.0 (not used) | `PT/LICENSES.md` |
| Limitations | not tested on real calls; provisional thresholds; MP3 offsets; mono agent speech; no Spanish aligner; out-of-domain base model; issue folders unused (possible self-supervised pre-training, not implemented) | `PT/README.md`; DOC §2.2, §8; design note |

## R. References (all checked on 2026-11-13)

| Key | Entry | Checked at |
|---|---|---|
| vibevoice | Z. Peng, J. Yu, Y. Chang, Z. Wang, L. Dong, et al. VibeVoice-ASR Technical Report. arXiv:2601.18184, 2026 | arxiv.org/abs/2601.18184; huggingface.co/microsoft/VibeVoice-ASR (MIT, cites the report) |
| silero | Silero Team. Silero VAD: pre-trained enterprise-grade Voice Activity Detector. GitHub, 2024 (MIT) | github.com/snakers4/silero-vad (README citation block) |
| wav2vec2 | A. Baevski, H. Zhou, A. Mohamed, M. Auli. wav2vec 2.0. NeurIPS 2020; arXiv:2006.11477 | arXiv page; papers.nips.cc abstract page |
| ctc | A. Graves, S. Fernández, F. Gomez, J. Schmidhuber. Connectionist Temporal Classification. ICML 2006, 369–376; doi 10.1145/1143844.1143891 | doi.org resolves to ACM |
| librispeech | V. Panayotov, G. Chen, D. Povey, S. Khudanpur. LibriSpeech. ICASSP 2015, 5206–5210; doi 10.1109/ICASSP.2015.7178964 | doi.org resolves to IEEE |
| torchaudio | J. Hwang, M. Hira, C. Chen, X. Zhang, Z. Ni, et al. TorchAudio 2.1. IEEE ASRU 2023; arXiv:2310.17864; doi 10.1109/ASRU57964.2023.10389648 | arXiv page; Crossref/OpenAlex |
| tutorial | X. Zhang, M. Hira. CTC forced alignment API tutorial, TorchAudio 2.8 docs (uses MMS_FA; same API) | docs.pytorch.org page |
| viterbi | A. J. Viterbi. Error bounds for convolutional codes … IEEE Trans. Inf. Theory 13(2), 260–269, 1967; doi 10.1109/TIT.1967.1054010 | doi.org resolves to IEEE |
| bellman | R. Bellman. Dynamic Programming. Princeton University Press, 1957 | press.princeton.edu page (reprint ISBN 9780691146683) |
| jackson | B. Jackson, J. D. Scargle, D. Barnes, et al. An algorithm for optimal partitioning of data on an interval. IEEE SPL 12(2), 105–108, 2005; registered doi 10.1109/LSP.2001.838216 | Crossref (title, volume, pages, 2005); doi.org resolves to IEEE document 1381461 |
| pelt | R. Killick, P. Fearnhead, I. A. Eckley. Optimal detection of changepoints with a linear computational cost. JASA 107(500), 1590–1598, 2012; doi 10.1080/01621459.2012.737745 | doi.org resolves to T&F |
| ctcseg | L. Kürzinger, D. Winkelbauer, L. Li, T. Watzel, G. Rigoll. CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition. SPECOM 2020, LNCS 12335, 267–278; doi 10.1007/978-3-030-60276-5_27 | Crossref; arXiv 2007.09127 |
| whisperx | M. Bain, J. Huh, T. Han, A. Zisserman. WhisperX. Interspeech 2023, 4489–4493; doi 10.21437/Interspeech.2023-78 | Crossref; ISCA archive |
| mfa | M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, M. Sonderegger. Montreal Forced Aligner. Interspeech 2017, 498–502; doi 10.21437/Interspeech.2017-1386 | Crossref; ISCA archive |
| dailytalk | K. Lee, K. Park, D. Kim. DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech. ICASSP 2023; doi 10.1109/ICASSP49357.2023.10095751; data CC BY-SA 4.0 | Crossref; github.com/keonlee9420/DailyTalk (licence) |
| meld | S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, R. Mihalcea. MELD. ACL 2019, 527–536; doi 10.18653/v1/P19-1050 | Crossref; doi.org resolves to ACL Anthology |
| shriberg | E. Shriberg, A. Stolcke, D. Hakkani-Tür, G. Tür. Prosody-based automatic segmentation of speech into sentences and topics. Speech Communication 32(1–2), 127–154, 2000; doi 10.1016/S0167-6393(00)00028-5 | Crossref; abstract via Semantic Scholar |
| wightman | C. W. Wightman, S. Shattuck-Hufnagel, M. Ostendorf, P. J. Price. Segmental durations in the vicinity of prosodic phrase boundaries. JASA 91(3), 1707–1717, 1992; doi 10.1121/1.402450 | Crossref; abstract via OpenAlex |
| swerts | M. Swerts. Prosodic features at discourse boundaries of different strength. JASA 101(1), 514–521, 1997; doi 10.1121/1.418114 | Crossref; abstract via OpenAlex |
| goldman | F. Goldman-Eisler. Psycholinguistics: Experiments in Spontaneous Speech. Academic Press, London, 1968 (no DOI; cited without link) | MPI library record and bibliographies found by web search |
| liu | Y. Liu, E. Shriberg, A. Stolcke, D. Hillard, M. Ostendorf, M. Harper. Enriching speech recognition with automatic detection of sentence boundaries and disfluencies. IEEE TASLP 14(5), 1526–1540, 2006; doi 10.1109/TASL.2006.878255 | Crossref; abstract via OpenAlex |

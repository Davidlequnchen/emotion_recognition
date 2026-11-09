# Informal test scripts

Quick checks that are not part of the pipelines in `scripts/`. Run them from the `emotion_recognition/` folder.
The OpenAI checks read the key from the `OPENAI_API_KEY` environment variable or from a line `OPENAI_API_KEY=...` in
the `.env` file at the repository root (git-ignored), and never print it.

| Script | What it does | Cost |
|---|---|---|
| `test_dataset_setup.py` | Tests data-submodule initialization, verified reuse of old files, preservation of external/archived local data, failed setup recovery, path boundaries and original-source/Release dispatch. Uses temporary local Git repositories and tiny files | free (local; no inference or remote downloads) |
| `test_model_setup.py` | Tests model submodule initialization, safe preservation and reuse of existing weights, failed initialization recovery, path boundaries and GitHub download dispatch. Uses temporary local Git repositories and tiny fixtures | free (local; no inference or remote downloads) |
| `test_dataset_label_choices.py` | Offline regression tests for dataset-specific choices, retries, cache/protocol isolation, result reuse, and consistent summary/viewer/plot selection. Uses fake models and API calls plus temporary files; no weights, keys or existing outputs are needed | free (no inference or network) |
| `check_openai.py` | Lists the models the key can use (free), then sends one test clip to an audio model (`gpt-audio-mini`) through Chat Completions | one request, a fraction of a cent |
| `check_openai_realtime.py` | Sends the same clip to each Realtime model (`gpt-realtime`, `gpt-realtime-2.1`) over WebSocket, as 24 kHz 16-bit mono PCM, and prints the answer and token usage | one request per model, a fraction of a cent each |
| `check_local_audio_llms.py` | Loads one open-weight audio LLM (`MODEL` at the top: Qwen2.5-Omni, Qwen2-Audio or Gemma 4) through its loader and runs it on 18 clips (CREMA-D, MCAE-SPPS, TESS): the filled chat template, each answer, ms per clip, device memory, invalid answers. Answers go to a temporary folder, never to `outputs/` | free (local) |
| `check_existing_models.py` | After a change of the environment: loads each of the five speech emotion models and compares its predictions on 3 CREMA-D clips with the stored `outputs/<model>/crema_d_predictions.csv` | free (local) |
| `time_models.py` | Times each local model (the five speech emotion models and the three audio LLMs, loaded as `evaluate.py` loads them) on 16 clips of each dataset, the first 2 as warm-up: median and mean seconds per clip and the mean clip duration, to plan a full evaluation. Waits while the machine is busy; the LLM answers go to a temporary folder, never to `outputs/` | free (local) |
| `check_llm_extras.py` | Checks the label probabilities and hidden states of the three local audio LLMs (`save_label_probs`, `save_hidden_states` in `config/llm.yaml`) on 20 clips (data/samples, RAVDESS, THAI-SER): each clip with and without them (alternating order), the answers compared, the probabilities summed, their most probable label against the answer, label_mass, seconds per clip, and the features reloaded with `load_features`. Answers and features go to a temporary folder, never to `outputs/` or `features/` | free (local) |
| `check_llm_nan_retry.py` | Checks the NaN retries of the local audio LLMs (`nan_retries` in `config/llm.yaml`) with Gemma 4 on 4 clips of data/samples, without a real NaN: generate() is wrapped to return NaN logits for 1, 2 and 4 calls. Recovery after retry 1 (memory cache emptied) and retry 2 (weights reloaded), the give-up after 3 retries (`invalid` with an "error", no probabilities or features) with the run going on, the answers against a clean run, and the answer cache and features index after a simulated interruption and resume. Answers and features go to a temporary folder, never to `outputs/` or `features/` | free (local) |
| `time_steady.py` | Steady-state speed of the models whose speed on MPS depends on the input length (the three audio LLMs and Vox-Profile WavLM): one process per model runs 400 clips in a row (200 of THAI-SER, then 200 of MEAD), logs per clip the seconds, the duration and whether its input length is new (a one-time compile), and prints the mean seconds per clip in four windows. Waits while the machine is busy; the LLM answers go to a temporary folder, never to `outputs/` | free (local) |

Run the offline regression suite in the `emotion_recognition_llm` environment:

```bash
python -m unittest discover -s test_scripts -p 'test_dataset_label_choices.py' -v
```

Results on 2026-10-27 with `data/samples/1001_DFA_ANG_XX.wav` (acted emotion: angry): `gpt-audio-mini`,
`gpt-realtime` and `gpt-realtime-2.1` all answered `neutral`. The clip took 22 audio tokens (about 10 per second).
A Realtime session adds about 110 text tokens of default instructions unless the session sets its own, and
`gpt-realtime-2.1` used 44 output tokens for a one-word answer (its reasoning is counted as output).

Results on 2026-11-05 (M4 Max, MPS, bfloat16, `check_local_audio_llms.py`, 18 clips, the machine shared with other
jobs): Qwen2-Audio 13 of 18 correct, Qwen2.5-Omni 7, Gemma 4 E4B 4 (mostly `neutral`), no invalid answer; about
0.6–0.9 s per clip after the warm-up, 16–19 GB of GPU memory. Qwen2-Audio gave the same 18 answers in float16 and
bfloat16, and in both environments (PyTorch 2.5.1 and 2.7.1); Qwen2.5-Omni differed on one clip between them. Gemma 4
transcribes the clips correctly, so it hears them; it judges the words more than the voice. `check_existing_models.py`:
all five models match the stored predictions (largest difference 5e-5, the CSV's rounding) in both environments.

Results on 2026-11-06 (`check_llm_nan_retry.py`, Gemma 4 E4B on MPS, 4 clips): all checks pass. Clip 1 recovered at
retry 1 with exactly the clean run's answer, label probabilities and hidden states; clip 2 recovered at retry 2 (one
reload, 13–20 s when memory is free); clip 3 was given up after 3 retries and the run went on; after an interruption
and a resume every clip has one valid answer and one features row. In a control run without NaN, the first one or two
generate() calls after a model load gave slightly different label probabilities (up to 0.02) and hidden states (up to
2 of a largest value of 112) on MPS, with the same answers; the clips run right after a reload differ from the clean
run by as much.

Run the model setup tests with Python 3.9+ and Git (no model environment needed):

```bash
python3 -m unittest discover -s test_scripts -p 'test_model_setup.py' -v
```

Run the dataset setup tests with Python 3.9+ and Git:

```bash
python3 -m unittest discover -s test_scripts -p 'test_dataset_setup.py' -v
```

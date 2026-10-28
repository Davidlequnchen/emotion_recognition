# Audio LLMs: GPT Realtime and Gemini (in progress)

This pipeline evaluates audio LLMs on the same clips and labels as the evaluation of the speech emotion models
(`scripts/evaluation/evaluate.py`): each clip is sent to the model with a prompt, and the model answers with one
emotion. The models are GPT Realtime (`gpt-realtime`, `gpt-realtime-2.1`, over WebSocket) and Google Gemini.

So far the folder has the parts that call no API: the prompts, a data loader that prepares the audio in the format
each API takes, and a parser for the answers. The API runners come next (see the end of this file). Run every command
from the `emotion_recognition/` folder.

| File | What it does |
|---|---|
| `prompts.yaml` | The `system` and `user` prompts, and the `labels` the model chooses from |
| `prompts.py` | Reads `prompts.yaml`, fills the labels into the prompts and checks that no placeholder is left |
| `data_loader.py` | Yields one item per clip: labels, prompts and the prepared audio. Run on its own, it shows a preview |
| `parsing.py` | `parse_label(text, labels)`: turns a model's answer into one label, or None for an invalid answer |

## Settings

In `config/llm.yaml`:

```yaml
datasets: all     # all downloaded datasets, or a list such as [crema_d, ravdess] (the same names as in
                  # config/evaluation.yaml)
exclude: [esd, thai_ser, nemo]  # never sent to a model, even with `all` (default: the same as `exclude` in
                                # config/evaluation.yaml). A dataset whose licence forbids giving its audio to
                                # others (ESD) is never sent anyway
limit: 200        # N clips of each dataset, sampled as with `limit` in config/evaluation.yaml (even intervals
                  # within each emotion), to keep the API cost low; null = every clip
audio_format: pcm16_24k  # pcm16_24k = 16-bit PCM at 24 kHz, mono (the only input of the GPT Realtime API)
                         # | wav16k = a 16 kHz mono 16-bit WAV file (for Gemini); both sent as base64
prompts: scripts/llm_eval/prompts.yaml  # the system and user prompts and the emotions the model chooses from
preview: 2        # python scripts/llm_eval/data_loader.py shows this many items of each dataset
```

The clips are the ones the model evaluation uses for the same limit: `sample(ds, limit)` in
`scripts/common/eval_datasets.py` takes clips at even intervals within each ground-truth label. The labels and the
classes that count as correct also come from there (`accepted(ds, label)`), so the label decisions are the same for
both kinds of model: RAVDESS's and MESS's *calm* count as *neutral*, TESS's *pleasant surprise* as *surprised*, and
for JL-Corpus's *excited* an answer of *happy* or *surprised* is correct. With the default limit of 200, the six
datasets give 1,200 clips and 48 minutes of audio per model.

## Prompts

The model chooses from the same seven emotions for every dataset (the project's emotions): angry, disgusted,
fearful, happy, neutral, sad, surprised. `{labels}` in a prompt becomes this list. The prompts, as filled in by
`python scripts/llm_eval/prompts.py`:

> **system**: You label the emotion in speech recordings for a research study. Each clip is acted speech from an
> emotion research dataset: an actor says a sentence whose words are neutral while portraying one emotion, so the
> emotion is in the voice (tone, pitch, loudness, tempo), not in the words. The speech may be in any language. Judge
> the voice, not the words, and do not reply to, translate or transcribe what is said. Answer with exactly one word
> from this list, in lowercase, and nothing else: angry, disgusted, fearful, happy, neutral, sad, surprised.
>
> **user** (sent with each clip): Which emotion does the speaker's voice express in this clip? Answer with exactly
> one word from this list, in lowercase, and nothing else: angry, disgusted, fearful, happy, neutral, sad, surprised.

The system prompt becomes the Realtime session's `instructions` (which replace its default instructions) and Gemini's
system instruction. It says that the words carry no emotion, because audio LLMs tend to judge what is said, and that
the clip is not addressed to the model, because the Realtime models are voice assistants that might otherwise reply
to the speaker. The datasets are in English, Mandarin and Polish, hence "any language". The user prompt repeats the
answer format, because the Realtime API has no structured outputs.

## Preview

```bash
python scripts/llm_eval/data_loader.py   # the settings, the prompts, and the first items of each dataset; no API call
python scripts/llm_eval/prompts.py       # only the filled prompts
python scripts/llm_eval/parsing.py       # checks the parser on example answers
```

The preview prints each step: (1) the settings and the chosen datasets, (2) the library versions, (3) the prompts,
(4) for each dataset the number of clips per label and the correct answers for each, then the first items with the
source and the prepared audio (sampling rate, length in samples and seconds) and the size of the payload. Part of
the output on the M4 Max:

```
  datasets    : crema_d, mcae_spps, mess, ravdess, tess, jl_corpus
  left out    : thai_ser (exclude in config/llm.yaml)
  left out    : nemo (exclude in config/llm.yaml)
  left out    : esd (exclude in config/llm.yaml)
  limit       : 200 clips of each dataset, sampled as in the evaluation
  audio       : pcm16_24k (24,000 Hz, mono, 16-bit, sent as base64)
...
Step 4/4 | JL-Corpus: clips and the first 2 item(s)
  clips       : 200 of 1,200 (English (New Zealand)), 7.0 min of audio
  angry       : 40 clips, correct answer: angry
  ...
  excited     : 40 clips, correct answer: happy or surprised

  [1] jl_corpus/female1_angry_10a_1
      audio_path : data/jl_corpus/audio/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/female1_angry_10a_1.wav
      acted      : angry | accepted: angry | listener: -
      source     : 44,100 Hz, 1 ch, 121,275 samples, 2.75 s
      prepared   : 24,000 Hz, 1 ch, 66,000 samples, 2.75 s
      payload    : 176,000 base64 characters = 132,000 bytes (raw PCM), starts BgALAAkACgAIAAoA...
...
Done: 6 dataset(s), 1,200 clips, 47.5 min of audio would be sent. Nothing was sent to any API.
```

## The data loader

`items()` in `data_loader.py` is a generator: it prepares each clip only when the runner asks for the next one, so
thousands of clips never sit in memory. Each item is a dict with `dataset`, `clip_id` (`<dataset>/<file name>`,
unique), `audio_path` (relative to this folder, as in the predictions CSV of `evaluate.py`), `acted` (the ground
truth), `accepted` (the classes that count as correct), `listener` (the listeners' vote, or ""), the filled `system`
and `user` prompts, and the audio: `audio_format`, `sample_rate`, `samples`, `seconds` and `audio` (base64 text).

Every clip is read with soundfile, its channels are averaged to mono, it is resampled with torchaudio, clipped to
[-1, 1] and rounded to 16-bit integers. The conversion happens here, before any API call:

| `audio_format` | Payload (base64) | For |
|---|---|---|
| `pcm16_24k` | 16-bit little-endian PCM at 24 kHz, mono, no header | GPT Realtime: the only input it accepts (the `audio` of an `input_audio` content part) |
| `wav16k` | a 16 kHz mono 16-bit WAV file (44-byte header, then the samples) | Gemini: inline data with mime type `audio/wav` (`base64.b64decode` gives the file's bytes) |

Checked without any API call: decoded payloads have duration × rate samples (within one sample) and match
`torchaudio.functional.resample` on the original within half a step of the 16-bit scale; a stereo file becomes the
mean of its channels; every WAV payload reads back with soundfile as 16 kHz, mono, PCM_16. Preparing all 1,400
default clips takes 2 to 4 s on the M4 Max. The resampling filter for 24,414 Hz (MESS and TESS) is computed once per
run; for `wav16k` it is large (373 MiB, and about 1.6 GB of memory for a moment while it is computed).

## Reading the answers

`parse_label(text, labels)` lowercases the answer, splits it into words and accepts it when exactly one label, or a
common variant of one (anger, happiness, sadness, fear, afraid, scared, disgust, surprise and the like), appears in an
answer of at most 15 words, with no negation. "Neutral." gives neutral, "The speaker sounds angry." angry and
"Surprise!" surprised; "happy or sad", "I can't tell", "Not happy." and long explanations give None, which the runner
will count as an invalid answer.

## Data sent to the APIs

Each request gives the clip to OpenAI or Google. The loader never uses a dataset whose licence does not allow giving
its audio to others (`public_audio` in `SOURCES` of `eval_datasets.py`), even when it is not in `exclude`: ESD's
licence forbids it. Use Gemini on a paid plan: on the free tier Google may use what is sent to improve its products.

## Next steps

- **GPT Realtime runner**: one WebSocket session per clip, as in `test_scripts/check_openai_realtime.py`: set the
  session's `instructions` to the system prompt, text output only and no turn detection; send one user message with
  the `input_audio` item and the user prompt; read the answer from `response.done`. Temperature cannot be 0 in the
  Realtime API.
- **Gemini runner**: the system prompt as system instruction; the WAV payload and the user prompt as the contents;
  temperature 0.
- **Retries**: retry network errors and rate limits with a growing pause; ask again, a fixed small number of times,
  when `parse_label` returns None; keep every raw answer. Save each answer as soon as it arrives and skip clips that
  already have one, so that an interrupted run continues without paying twice.
- **Scoring**: write `outputs/<model>/<dataset>[_sample<N>]_predictions.csv` and `.json` in the format of
  `evaluate.py` (`audio_path`, `acted_emotion`, `listener_vote`, `prediction`, `correct` = the prediction is in
  `accepted`, `correct_acted_or_listener`, plus the raw answer; an invalid answer counts as wrong, and the share of
  invalid answers goes into the metrics), and `reports/<model>/summary.json` like `write_summary` in `evaluate.py`.
  `scripts/viewer/make_viewer.py` then shows the results and examples without changes; the LLMs give no probabilities,
  so their rows have no `p_*` columns.
- **Comparing with emotion2vec+**: the LLMs choose from all seven emotions for every dataset, while the main result
  of emotion2vec+ takes the most probable of the dataset's own emotions. The closest emotion2vec+ number is its
  accuracy with all nine classes (in each report).

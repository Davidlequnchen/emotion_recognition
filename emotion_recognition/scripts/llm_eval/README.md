# Audio LLMs: prompts, audio and answers

The audio LLMs run through `scripts/evaluation/evaluate.py`, like the speech emotion models, through their loaders
`scripts/models/gpt_realtime.py` (GPT Realtime, over WebSocket) and `scripts/models/gemini.py` (Gemini, a placeholder
for now); how to run them: section 8 of the [README](../../README.md#8-audio-llms). They get the same clips and labels
as the other models. This folder has the parts they share, which call no API. Run every command from the
`emotion_recognition/` folder.

| File | What it does |
|---|---|
| `prompts.yaml` | The `system` and `user` prompts, and the `labels` the model chooses from |
| `prompts.py` | Reads `prompts.yaml`, fills the labels into the prompts and checks that no placeholder is left |
| `data_loader.py` | `prepare_audio(path, format)`: one clip as base64 in the format of the API (used by the loaders). Run on its own, it shows a preview of the items a run would send |
| `parsing.py` | `parse_label(text, labels)`: turns a model's answer into one label, or None for an invalid answer |

## Settings

In `config/llm.yaml`: `prompts` (the prompts file), `retries`, `retries_invalid` and `timeout_s` (read by the
loaders), and the settings of the preview (`datasets`, `exclude`, `limit`, `audio_format`, `preview`). In a run, the
datasets come from `config/evaluation.yaml` and the number of clips from the model's `limit` in `config/models.yaml`
(200).

The clips are the ones the model evaluation uses for the same limit: `sample(ds, limit)` in
`scripts/common/eval_datasets.py` takes clips at even intervals within each ground-truth label. The labels and the
classes that count as correct also come from there (`accepted(ds, label)`), so the label decisions are the same for
both kinds of model: RAVDESS's and MESS's *calm* count as *neutral*, TESS's *pleasant surprise* as *surprised*, and
for JL-Corpus's *excited* an answer of *happy* or *surprised* is correct. With the limit of 200, the six
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

`items()` in `data_loader.py` (used by the preview) is a generator: it prepares each clip only when it is asked for, so
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
"Surprise!" surprised; "happy or sad", "I can't tell", "Not happy." and long explanations give None, which
counts as an invalid answer (asked again once, then the class `invalid`).

## Data sent to the APIs

Each request gives the clip to OpenAI or Google. The loader never uses a dataset whose licence does not allow giving
its audio to others (`public_audio` in `SOURCES` of `eval_datasets.py`), even when it is not in `exclude`: ESD's
licence forbids it. Use Gemini on a paid plan: on the free tier Google may use what is sent to improve its products.

## How a run uses them

`gpt_realtime.py` prepares each clip with `prepare_audio(path, "pcm16_24k")`, opens one WebSocket session per clip
(the system prompt as the session `instructions`, text output, no turn detection), sends the audio with the user
prompt and reads the answer from `response.done`. `parse_label` turns it into a label; the loader returns a one-hot
vector over the seven labels plus `invalid`, and `evaluate.py` scores it like any model, with an answer outside the
dataset's emotions counted as wrong. Every answer is kept in `outputs/<model>/responses.jsonl` (the cache), so no
clip is paid twice. The Gemini loader will do the same with `prepare_audio(path, "wav16k")` and the google-genai SDK.

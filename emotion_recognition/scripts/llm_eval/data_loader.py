"""
Data for the audio-LLM evaluation: one item per clip, with the prompts and the audio ready to send to the model.

For the datasets chosen in the `llm` section of config/config.yaml, the loader takes the same clips and labels as the
emotion2vec+ evaluation (scripts/common/eval_datasets.py: its loaders, `sample` for llm.limit and `accepted` for the
classes that count as correct). items() yields one item per clip and prepares each clip only when it is asked for
(a generator), so thousands of clips never sit in memory. Each item is a dict:
    dataset        dataset key, e.g. "crema_d"
    clip_id        "<dataset>/<file name without extension>", unique
    audio_path     audio file, relative to the emotion_recognition/ folder (as in the predictions CSV of evaluate.py)
    acted          ground truth: the emotion the speaker was asked to portray, as a label of the seven emotions or a
                   composite label of the dataset (JL-Corpus: excited)
    accepted       the model classes that count as correct for it, sorted: usually just `acted`; excited: happy,
                   surprised (`accepted` in eval_datasets.py)
    listener       the listeners' majority vote ("a or b" for a tie); "" if the dataset has none
    system, user   the prompts, with the labels filled in (prompts.yaml, read by prompts.py)
    audio_format   pcm16_24k or wav16k (llm.audio_format)
    sample_rate    sampling rate of the prepared audio (24000 or 16000)
    samples        length of the prepared audio in samples; seconds: the same in seconds
    audio          the prepared audio as base64 text:
                     pcm16_24k  16-bit little-endian PCM at 24 kHz, mono, without a header: the only input of the
                                GPT Realtime API (the `audio` of an input_audio content part)
                     wav16k     a 16 kHz mono 16-bit WAV file, for Gemini (inline data, mime type audio/wav);
                                base64.b64decode(item["audio"]) gives the file's bytes
Each clip is read with soundfile, its channels are averaged to mono, it is resampled with torchaudio, clipped to
[-1, 1] and rounded to 16-bit integers.

Usage (from the emotion_recognition/ folder):
    python scripts/llm_eval/data_loader.py
prints step by step what the loader does and shows the first items of each dataset (llm.preview), without calling any
API. The API runners will import items() from this file.
"""
import base64
import functools
import io
import itertools
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # eval_datasets.py is in scripts/common/
# The clips (sample) and the classes that count as correct (accepted) come from the evaluation of emotion2vec+, so the
# LLMs are scored on the same clips with the same rules
from eval_datasets import LOADERS, SOURCES, accepted, sample
from prompts import load_prompts  # scripts/llm_eval/prompts.py

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "config.yaml"
RATES = {"pcm16_24k": 24000, "wav16k": 16000}  # llm.audio_format -> sampling rate of the audio sent to the model


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def select_datasets(settings):
    """The datasets to use (llm.datasets without llm.exclude), and those left out with the reason."""
    wanted = settings["datasets"]
    if wanted == "all":
        keys = [k for k in LOADERS if (ROOT / "data" / k).is_dir()]
    else:
        keys = [wanted] if isinstance(wanted, str) else list(wanted)
    exclude = settings["exclude"] or []
    unknown = [k for k in [*keys, *exclude] if k not in LOADERS]
    if unknown:
        sys.exit(f"ERROR: unknown dataset(s) {unknown} in the llm section of config/config.yaml; use: "
                 f"{', '.join(LOADERS)} or all")
    left_out = {k: "llm.exclude" for k in keys if k in exclude}
    # Every request gives the clip to OpenAI or Google, so a dataset whose licence does not allow giving its audio to
    # others is never used (public_audio in SOURCES of eval_datasets.py; ESD's licence forbids it)
    left_out |= {k: "its licence does not allow giving its audio to others" for k in keys
                 if k not in left_out and not SOURCES.get(k, {}).get("public_audio")}
    keys = [k for k in keys if k not in left_out]
    missing = [k for k in keys if not (ROOT / "data" / k).is_dir()]
    if missing:
        sys.exit(f"ERROR: not downloaded: {', '.join(missing)}. Download it with scripts/common/download_data.py.")
    if not keys:
        sys.exit(f"ERROR: no dataset left to use ({'; '.join(f'{k}: {r}' for k, r in left_out.items())}); check "
                 "`llm.datasets` and `llm.exclude` in config/config.yaml.")
    return keys, left_out


@functools.lru_cache(maxsize=None)
def resampler(rate_in, rate_out):
    """One torchaudio resampler per pair of rates, the same filter as torchaudio.functional.resample on float32
    audio, but computed once instead of for every clip. For 24,414 Hz (MESS, TESS) the filter is large: 62 MiB for
    24 kHz, 373 MiB for 16 kHz (computing it needs about 1.6 GB for a moment)."""
    return torchaudio.transforms.Resample(rate_in, rate_out, dtype=torch.float32)


def prepare_audio(path, audio_format):
    """One clip in the given format: (base64 text, number of samples, sampling rate)."""
    rate = RATES[audio_format]
    wav, sr = sf.read(path, dtype="float32", always_2d=True)  # samples x channels
    wav = torch.from_numpy(wav.mean(axis=1))                   # average the channels to mono
    if sr != rate:
        wav = resampler(sr, rate)(wav)                         # resample to the rate the model takes
    pcm = np.rint(np.clip(wav.numpy(), -1, 1) * 32767).astype("<i2")  # 16-bit little-endian integers
    if audio_format == "pcm16_24k":
        data = pcm.tobytes()                                   # raw PCM samples, no header
    else:
        buffer = io.BytesIO()
        sf.write(buffer, pcm, rate, format="WAV", subtype="PCM_16")  # a WAV file in memory
        data = buffer.getvalue()
    return base64.b64encode(data).decode("ascii"), len(pcm), rate


def dataset_items(ds, clips, prompts, audio_format):
    """One item per clip of one dataset (see the docstring of this file), each prepared when it is asked for."""
    for clip in clips:
        audio, samples, rate = prepare_audio(clip["audio"], audio_format)
        yield {
            "dataset": ds.key,
            "clip_id": f"{ds.key}/{clip['audio'].stem}",
            "audio_path": clip["audio"].relative_to(ROOT).as_posix(),
            "acted": clip["acted"],
            "accepted": sorted(accepted(ds, clip["acted"])),
            "listener": clip["listener"],
            "system": prompts["system"],
            "user": prompts["user"],
            "audio_format": audio_format,
            "sample_rate": rate,
            "samples": samples,
            "seconds": samples / rate,
            "audio": audio,
        }


def items(cfg=None, audio_format=None):
    """Every item for the `llm` settings, dataset by dataset (a generator). cfg: the parsed config/config.yaml (None =
    read it); audio_format: None = llm.audio_format (a runner can ask for the format its API takes)."""
    settings = (cfg or yaml.safe_load(CONFIG.read_text()))["llm"]
    audio_format = audio_format or settings["audio_format"]
    if audio_format not in RATES:
        sys.exit(f"ERROR: unknown llm.audio_format {audio_format!r}; use {' or '.join(RATES)}")
    prompts = load_prompts(ROOT / settings["prompts"])
    keys, _ = select_datasets(settings)
    for key in keys:
        ds = LOADERS[key]()
        yield from dataset_items(ds, sample(ds, settings["limit"]), prompts, audio_format)


def short(text, n=70):
    return text if len(text) <= n else text[:n - 3] + "..."


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the settings (config/config.yaml, section `llm`) and choose the datasets
    # ------------------------------------------------------------------
    step("Step 1/4 | Load configuration")
    settings = yaml.safe_load(CONFIG.read_text())["llm"]
    limit, audio_format, n_show = settings["limit"], settings["audio_format"], settings["preview"]
    if audio_format not in RATES:
        sys.exit(f"ERROR: unknown llm.audio_format {audio_format!r}; use {' or '.join(RATES)}")
    keys, left_out = select_datasets(settings)
    print(f"  config      : {CONFIG}")
    print(f"  datasets    : {', '.join(keys)}")
    for key, reason in left_out.items():
        print(f"  left out    : {key} ({reason})")
    sampled = f"{limit:,} clips of each dataset, sampled as in the evaluation" if limit else "every clip"
    print(f"  limit       : {sampled}")
    print(f"  audio       : {audio_format} ({RATES[audio_format]:,} Hz, mono, 16-bit, sent as base64)")
    print(f"  preview     : the first {n_show} item(s) of each dataset; no API is called")

    # ------------------------------------------------------------------
    # Step 2: Check the libraries that read and convert the audio
    # ------------------------------------------------------------------
    step("Step 2/4 | Check environment")
    print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
    print(f"  NumPy       : {np.__version__}")
    print(f"  soundfile   : {sf.__version__} (libsndfile {sf.__libsndfile_version__})")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  torchaudio  : {torchaudio.__version__}")

    # ------------------------------------------------------------------
    # Step 3: Load the prompts and fill in the labels (prompts.yaml)
    # ------------------------------------------------------------------
    step("Step 3/4 | Load the prompts")
    prompts = load_prompts(ROOT / settings["prompts"])
    print(f"  file        : {settings['prompts']}")
    print(f"  labels      : {', '.join(prompts['labels'])}")
    for k in ("system", "user"):
        print(f"  {k:<11} : ({len(prompts[k])} characters)\n    {prompts[k]}")

    # ------------------------------------------------------------------
    # Step 4: For each dataset, take the clips as the evaluation does and prepare the first items
    # ------------------------------------------------------------------
    total, seconds = 0, 0.0
    for key in keys:
        ds = LOADERS[key]()
        step(f"Step 4/4 | {ds.name}: clips and the first {n_show} item(s)")
        clips = sample(ds, limit)
        duration = sum(sf.info(c["audio"]).duration for c in clips)  # reads only the file headers
        total, seconds = total + len(clips), seconds + duration
        print(f"  clips       : {len(clips):,} of {len(ds.clips):,} ({ds.language}), {duration / 60:.1f} min of audio")
        rank = [*ds.classes, *ds.accept]  # the dataset's classes, then its composite labels such as excited
        for label in sorted({c["acted"] for c in clips}, key=lambda a: rank.index(a) if a in rank else len(rank)):
            n = sum(c["acted"] == label for c in clips)
            print(f"  {label:<11} : {n:,} clip{'s' * (n != 1)}, correct answer: "
                  f"{' or '.join(sorted(accepted(ds, label)))}")
        print(f"  listeners   : {'yes' if any(c['listener'] for c in clips) else 'no labels in this dataset'}")
        for i, item in enumerate(itertools.islice(dataset_items(ds, clips, prompts, audio_format), n_show), 1):
            source = sf.info(ROOT / item["audio_path"])
            raw = len(base64.b64decode(item["audio"]))
            print(f"\n  [{i}] {item['clip_id']}")
            print(f"      audio_path : {item['audio_path']}")
            print(f"      acted      : {item['acted']} | accepted: {', '.join(item['accepted'])} | "
                  f"listener: {item['listener'] or '-'}")
            print(f"      source     : {source.samplerate:,} Hz, {source.channels} ch, {source.frames:,} samples, "
                  f"{source.duration:.2f} s")
            print(f"      prepared   : {item['sample_rate']:,} Hz, 1 ch, {item['samples']:,} samples, "
                  f"{item['seconds']:.2f} s")
            print(f"      payload    : {len(item['audio']):,} base64 characters = {raw:,} bytes "
                  f"({'raw PCM' if audio_format == 'pcm16_24k' else 'WAV file with its header'}), "
                  f"starts {item['audio'][:16]}...")
            print(f"      system     : {short(item['system'])}")
            print(f"      user       : {short(item['user'])}")
    print(f"\nDone: {len(keys)} dataset(s), {total:,} clips, {seconds / 60:.1f} min of audio would be sent. "
          "Nothing was sent to any API.")


if __name__ == "__main__":
    main()

"""
Informal steady-state timing of the models whose speed on MPS depends on the input length, to plan a full evaluation.

On MPS the first clip of each new input length pays a one-time compile; a clip whose length was seen before runs at
the steady speed. time_models.py times only 16 clips per dataset, so most of its clips pay that compile. Here one
process runs each model over CLIPS_PER_DATASET clips of each dataset of DATASETS in a row, as a full evaluation does.

What it does, step by step:
    1. loads each dataset of DATASETS (scripts/common/eval_datasets.py) and takes the first CLIPS_PER_DATASET clips of
       sample(ds, SAMPLE) (clips at even intervals, as evaluate.py takes them with `limit`);
    2. for each model of MODELS: waits until the machine is free enough (load average at most MAX_LOAD, free memory
       at least MIN_FREE_PERCENT), loads it through its loader as scripts/evaluation/evaluate.py does, on its device
       (mac_device in config/models.yaml, else mps);
    3. runs it on every clip, one at a time, in one process, and logs per clip: seconds, clip duration, the input
       shapes (LLM: the token count of each generate() call and the audio feature shapes; other models: the samples
       fed) and whether that input shape is new in this process; the load average every LOAD_EVERY clips;
    4. prints the mean seconds per clip in WINDOWS and the share of clips with a new input shape, then frees the model.
The audio LLMs write their answers to a fresh temporary folder (entry["cache_file"]), deleted afterwards, never to
outputs/. Every timing goes to RESULTS (a JSON file in the temporary folder). Nothing leaves the machine.

Usage (from the emotion_recognition/ folder, in the emotion_recognition_llm environment):
    OMP_NUM_THREADS=4 nice -n 10 python test_scripts/time_steady.py
"""
import gc
import importlib
import inspect
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
import transformers.modeling_utils  # noqa: F401 (imported here, before vox_profile.py builds its model inside a
                                   # torch.device context, where the first import of it fails in torch 2.7.1)
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # as in evaluate.py: every model loads from its own folder
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
MODELS = ["vox_profile_wavlm", "gemma4_e4b", "qwen2_audio_7b", "qwen2_5_omni_7b"]  # keys in config/models.yaml
DATASETS = ["thai_ser", "mead"]  # run in this order, one after the other, in the same process
SAMPLE = 2000              # sample(ds, SAMPLE): clips at even intervals over the dataset
CLIPS_PER_DATASET = 200    # the first ones of that sample
WINDOWS = [(1, 50), (51, 200), (201, 250), (251, 400)]  # clip ranges (1-based, inclusive) of the summary
LOAD_EVERY = 50            # record the load average every this many clips
MAX_LOAD = 12              # wait before loading a model while the 1-minute load average is above this
MIN_FREE_PERCENT = 25      # ... or while less than this share of the memory is free (memory_pressure)
WAIT_SECONDS = 120         # time between two checks
RESULTS = Path(tempfile.gettempdir()) / "time_steady" / "results.json"

sys.path.insert(0, str(PROJECT / "scripts" / "common"))
sys.path.insert(0, str(PROJECT / "scripts" / "models"))


def machine():
    """(1-minute load average, free memory in percent) of this Mac."""
    load = float(re.findall(r"load averages?: ([\d.,]+)", subprocess.run(["uptime"], capture_output=True,
                                                                         text=True).stdout)[0].replace(",", "."))
    pressure = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return load, int(re.findall(r"(\d+)%", pressure)[0])


def wait_for_machine():
    """Wait until the machine is free enough to load a model; return its state then."""
    while True:
        load, free = machine()
        if load <= MAX_LOAD and free >= MIN_FREE_PERCENT:
            return load, free
        print(f"  busy        : load {load:.1f}, {free}% memory free; checking again in {WAIT_SECONDS} s")
        time.sleep(WAIT_SECONDS)


def shape_of(value):
    """The shape of a tensor, else None."""
    return list(value.shape) if torch.is_tensor(value) else None


def run(predict, llm, clips, load, free):
    """Every clip in a row through predict(); the per-clip log and the machine state every LOAD_EVERY clips."""
    # The input shapes of each call: the LLMs' generate() (tokens in, tokens out, audio features), the other
    # models' forward pass (the module that predict() calls)
    calls, handles = [], []
    if llm:
        model = predict.model
        original = model.generate

        def generate(*args, **kwargs):
            out = original(*args, **kwargs)
            shapes = {k: shape_of(v) for k, v in kwargs.items() if shape_of(v) is not None}
            calls.append({"shapes": shapes, "new_tokens": out.shape[1] - kwargs["input_ids"].shape[1]})
            return out
        model.generate = generate
    else:
        model = predict.model

        def hook(module, args, kwargs):
            tensors = [shape_of(a) for a in list(args) + list(kwargs.values()) if shape_of(a) is not None]
            calls.append({"shapes": {"inputs": tensors}})
        handles.append(model.register_forward_pre_hook(hook, with_kwargs=True))

    # Step 3: every clip in a row, in one process
    record = {"started": datetime.now().isoformat(), "machine": [{"clip": 0, "load": load, "free": free}],
              "clips": []}
    seen = set()
    for i, (key, audio, dur) in enumerate(clips, 1):
        calls.clear()
        t0 = time.perf_counter()
        predict(str(audio))
        seconds = time.perf_counter() - t0
        keys = [json.dumps(c["shapes"], sort_keys=True) for c in calls]
        new = any(k not in seen for k in keys)
        seen.update(keys)
        row = {"i": i, "dataset": key, "duration": round(dur, 3), "seconds": round(seconds, 4), "new_shape": new,
               "calls": calls[:] if llm else [c["shapes"] for c in calls]}
        record["clips"].append(row)
        if i % LOAD_EVERY == 0:
            load, free = machine()
            record["machine"].append({"clip": i, "load": load, "free": free})
            recent = record["clips"][-LOAD_EVERY:]
            print(f"  clips {i - LOAD_EVERY + 1:>3}-{i:<3}: mean {statistics.mean(r['seconds'] for r in recent):.3f}"
                  f" s, new shapes {sum(r['new_shape'] for r in recent)}/{LOAD_EVERY}; load {load:.1f}, "
                  f"{free}% free")

    if llm:
        del model.generate  # the class's own generate() again
    for h in handles:
        h.remove()
    return record


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import eval_datasets

    registry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}  # a stopped run goes on where it stopped
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: the versions and the clips (thai_ser's then mead's, as consecutive clips of a real run)
    print(f"Step 1 | torch {torch.__version__}, MPS {torch.backends.mps.is_available()}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    print(f"  results     : {RESULTS}")
    clips = []
    for key in DATASETS:
        picked = eval_datasets.sample(eval_datasets.LOADERS[key](), SAMPLE)[:CLIPS_PER_DATASET]
        clips += [(key, c["audio"], sf.info(str(c["audio"])).duration) for c in picked]
        print(f"  {key:<15}: {len(picked)} clips, mean {statistics.mean(d for k, _, d in clips if k == key):.2f} s")

    for m in MODELS:
        if m in results:
            print(f"\nStep 2 | {m}: already timed; skipped")
            continue
        entry = registry[m] | {"key": m}

        # Step 2: wait for the machine, then load the model as evaluate.py does
        print(f"\nStep 2 | {entry['name']} ({m})")
        load, free = wait_for_machine()
        device = torch.device(entry.get("mac_device", "mps"))
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
        scratch = Path(tempfile.mkdtemp(prefix=f"steady_{m}_"))
        if entry.get("llm"):
            entry["cache_file"] = scratch / "responses.jsonl"  # fresh: no cached answer, nothing in outputs/
            entry["features_dir"] = scratch / "features"  # keep standalone check features out of dataset runs
        print(f"  machine     : load {load:.1f}, {free}% memory free; device {device}, "
              f"{torch.get_num_threads()} threads")
        loader = importlib.import_module(entry["loader"])
        t0 = time.perf_counter()
        if "entry" in inspect.signature(loader.load).parameters:
            predict = loader.load(PROJECT / entry["dir"], device, entry=entry)
        else:
            predict = loader.load(PROJECT / entry["dir"], device)
        load_s = time.perf_counter() - t0
        print(f"  loaded in   : {load_s:.1f} s")

        record = run(predict, entry.get("llm"), clips, load, free)
        record.update(device=str(device), threads=torch.get_num_threads(), load_seconds=round(load_s, 1))

        # Step 4: the summary, then free the model before the next one
        record["windows"] = {}
        for a, b in WINDOWS:
            rows = record["clips"][a - 1:b]
            record["windows"][f"{a}-{b}"] = {
                "mean": round(statistics.mean(r["seconds"] for r in rows), 4),
                "mean_not_new": round(statistics.mean([r["seconds"] for r in rows if not r["new_shape"]] or [0]), 4),
                "new_share": round(sum(r["new_shape"] for r in rows) / len(rows), 3),
                "mean_duration": round(statistics.mean(r["duration"] for r in rows), 3)}
            print(f"  {a}-{b:<8}: {record['windows'][f'{a}-{b}']}")
        results[m] = record
        RESULTS.write_text(json.dumps(results, indent=1))
        del predict
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        shutil.rmtree(scratch)
        print(f"  freed; load average now {machine()[0]:.1f}")
    print(f"\nDone: {RESULTS}")


if __name__ == "__main__":
    main()

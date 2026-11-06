"""
Informal timing of every local model on a few clips of each dataset, on this machine, to plan a full evaluation.

What it does, step by step:
    1. loads each dataset of DATASETS (scripts/common/eval_datasets.py), takes CLIPS clips spread over it (sample(), as
       evaluate.py takes them with `limit`) and reads the duration of every clip of the dataset (file headers only);
    2. for each model of MODELS: waits until the machine is free enough (load average below MAX_LOAD, free memory
       at least MIN_FREE_PERCENT), loads it through its loader as scripts/evaluation/evaluate.py does
       (loader.load(folder, device, entry=...)), on its device (mac_device in config/models.yaml, else mps);
    3. runs it on the clips of each dataset, one at a time, and times each clip; the first WARMUP clips of each
       dataset are warm-up and not counted;
    4. prints per model and dataset the median and mean seconds per clip and the mean clip duration, then frees the
       model before the next one.
The audio LLMs write their answers to a fresh temporary folder (entry["cache_file"]), never to outputs/, so no cached
answer is reused and no real result is touched. Every timing goes to RESULTS (a JSON file in the temporary folder).
Nothing leaves the machine.

Usage (from the emotion_recognition/ folder, in the environment that has every model, e.g. emotion_recognition_llm):
    OMP_NUM_THREADS=4 nice -n 10 python test_scripts/time_models.py
"""
import gc
import importlib
import inspect
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # as in evaluate.py: every model loads from its own folder
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
MODELS = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm",
          "qwen2_5_omni_7b", "qwen2_audio_7b", "gemma4_e4b"]  # keys in config/models.yaml
DATASETS = ["crema_d", "thai_ser", "mead", "meld", "emovo", "emozionalmente", "nemo", "savee"]  # savee: if downloaded
CLIPS = 16          # clips per dataset
WARMUP = 2          # first clips of each dataset that are not counted
CPU_THREADS = 12    # PyTorch threads of a model on the CPU: torch's default on the M4 Max (12 performance cores),
                    # what evaluate.py uses for emotion2vec+; the other models keep OMP_NUM_THREADS (set in the shell)
MAX_LOAD = 12       # wait before loading a model while the 1-minute load average is above this
MIN_FREE_PERCENT = 25  # ... or while less than this share of the memory is free (memory_pressure)
WAIT_SECONDS = 120  # time between two checks
RESULTS = Path(tempfile.gettempdir()) / "time_models" / "results.json"

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


def datasets():
    """dataset key -> (picked clips, mean duration of all clips in s, number of clips); a missing dataset is left out."""
    import eval_datasets

    importlib.reload(eval_datasets)  # a dataset added since the start (e.g. SAVEE) is seen
    out = {}
    for key in DATASETS:
        if key not in eval_datasets.LOADERS or not (PROJECT / "data" / key).exists():
            print(f"  {key:<15}: not available (no loader or no data/{key}/); skipped")
            continue
        ds = eval_datasets.LOADERS[key]()
        durations = [sf.info(str(c["audio"])).duration for c in ds.clips]
        clips = eval_datasets.sample(ds, CLIPS)
        out[key] = (clips, sum(durations) / len(durations), len(ds.clips))
        print(f"  {key:<15}: {len(ds.clips):>6,} clips, mean {out[key][1]:.2f} s; {len(clips)} picked")
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    registry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}  # a stopped run goes on where it stopped
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: the versions and the datasets
    print(f"Step 1 | torch {torch.__version__}, MPS {torch.backends.mps.is_available()}, "
          f"default threads {torch.get_num_threads()} (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")
    print(f"  results     : {RESULTS}")
    sets = datasets()
    results["_datasets"] = {k: {"clips": n, "mean_seconds": round(d, 3)} for k, (_, d, n) in sets.items()}

    for m in MODELS:
        if m in results and set(sets) <= set(results[m]["datasets"]):
            print(f"\nStep 2 | {m}: already timed; skipped")
            continue
        entry = registry[m] | {"key": m}
        if "savee" not in sets and (PROJECT / "data" / "savee").exists():  # SAVEE may have arrived meanwhile
            sets = datasets()
            results["_datasets"] = {k: {"clips": n, "mean_seconds": round(d, 3)} for k, (_, d, n) in sets.items()}

        # Step 2: wait for the machine, then load the model as evaluate.py does
        print(f"\nStep 2 | {entry['name']} ({m})")
        load, free = wait_for_machine()
        device = torch.device(entry.get("mac_device", "mps"))
        torch.set_num_threads(CPU_THREADS if device.type == "cpu" else int(os.environ.get("OMP_NUM_THREADS", 4)))
        scratch = Path(tempfile.mkdtemp(prefix=f"timing_{m}_"))
        if entry.get("llm"):
            entry["cache_file"] = scratch / "responses.jsonl"  # fresh: no cached answer, nothing in outputs/
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
        record = {"device": str(device), "threads": torch.get_num_threads(), "load_average": load,
                  "free_percent": free, "load_seconds": round(load_s, 1), "started": datetime.now().isoformat(),
                  "datasets": {}}

        # Step 3: run it on the clips of each dataset
        for key, (clips, _, _) in sets.items():
            rows = []
            for i, clip in enumerate(clips):
                dur = sf.info(str(clip["audio"])).duration
                t0 = time.perf_counter()
                predict(str(clip["audio"]))
                seconds = time.perf_counter() - t0
                extra = {}
                if entry.get("llm"):  # the requests this clip took (a follow-up after an answer that is not a label)
                    rel = Path(clip["audio"]).resolve().relative_to(PROJECT).as_posix()
                    extra = {"requests": predict.cache[rel]["requests"], "answer": predict.cache[rel]["answers"][-1]}
                rows.append({"duration": round(dur, 3), "seconds": round(seconds, 4), **extra})
            timed = rows[WARMUP:]
            sec = [r["seconds"] for r in timed]
            summary = {"median": round(statistics.median(sec), 4), "mean": round(statistics.mean(sec), 4),
                       "mean_duration": round(statistics.mean(r["duration"] for r in timed), 3),
                       "warmup": [r["seconds"] for r in rows[:WARMUP]], "clips": rows}
            record["datasets"][key] = summary
            print(f"  {key:<15}: median {summary['median']:.3f} s, mean {summary['mean']:.3f} s per clip "
                  f"(mean clip {summary['mean_duration']:.2f} s; warm-up {', '.join(f'{s:.2f}' for s in summary['warmup'])} s)")
        record["load_average_after"] = machine()[0]
        results[m] = record
        RESULTS.write_text(json.dumps(results, indent=1))

        # Step 4: free the model before the next one
        del predict
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        for f in scratch.glob("*"):
            f.unlink()
        scratch.rmdir()
        print(f"  freed; load average now {record['load_average_after']:.1f}")
    print(f"\nDone: {RESULTS}")


if __name__ == "__main__":
    main()

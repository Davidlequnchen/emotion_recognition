"""
Informal check of the by-products of the local audio LLMs (save_label_probs, save_hidden_states in config/llm.yaml;
scripts/models/audio_llm.py): that they leave the answers unchanged, what they cost, and that the features reload.

What it does, step by step, for each model of MODELS (one at a time, freed before the next):
    1. waits until enough memory is free (MIN_FREE_PERCENT), then loads the model through its loader as
       scripts/evaluation/evaluate.py does, with its answers and features in a fresh temporary folder (never outputs/
       or features/);
    2. runs every clip twice in a row, once without the by-products (the generate() call with the logits only, for
       the NaN check) and once with them, each mode with its own answer cache; the order alternates from clip to clip
       so that neither mode always gets the warm second run. The first clip is a warm-up of both modes, not timed;
    3. compares the answers of the two modes (they must be identical), checks that the label probabilities add up to
       1, how often their most probable label is the parsed answer, the label_mass, and the seconds per clip;
    4. reloads the features with load_features: one row per clip, the expected shape, finite values, the pooling.
Clips: the 6 CREMA-D clips of data/samples/ and CLIPS_PER_DATASET clips of each dataset of DATASETS (sample() of
eval_datasets.py, as evaluate.py takes them with `limit`). The summary of each model goes to RESULTS (temporary
folder). Nothing leaves the machine.

Usage (from the emotion_recognition/ folder, in the emotion_recognition_llm environment):
    OMP_NUM_THREADS=4 nice -n 10 python test_scripts/check_llm_extras.py
"""
import gc
import importlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # as in evaluate.py: every model loads from its own folder
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
MODELS = ["qwen2_audio_7b", "qwen2_5_omni_7b", "gemma4_e4b"]  # keys in config/models.yaml
DATASETS = ["ravdess", "thai_ser"]  # besides data/samples/
CLIPS_PER_DATASET = 7
MIN_FREE_PERCENT = 40      # wait before loading a model while less memory is free (memory_pressure)
WAIT_SECONDS = 120
RESULTS = Path(tempfile.gettempdir()) / "check_llm_extras" / "results.json"

sys.path.insert(0, str(PROJECT / "scripts" / "common"))
sys.path.insert(0, str(PROJECT / "scripts" / "models"))


def free_memory():
    """Free memory in percent (memory_pressure)."""
    last = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return int(re.findall(r"(\d+)%", last)[0])


def run_mode(predict, path, extras, caches):
    """One clip through predict() with or without the by-products, with that mode's own answer cache."""
    predict.save_probs = predict.save_hidden = extras
    predict.cache_file, predict.cache = caches[extras]
    predict(path)
    return predict.cache[str(Path(path).resolve().relative_to(PROJECT).as_posix())]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    import eval_datasets
    from audio_llm import load_features

    registry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    print(f"Step 0 | torch {torch.__version__}, MPS {torch.backends.mps.is_available()}, results {RESULTS}")
    clips = sorted((PROJECT / "data" / "samples").glob("*.wav"))
    for key in DATASETS:
        clips += [c["audio"] for c in eval_datasets.sample(eval_datasets.LOADERS[key](), CLIPS_PER_DATASET)]
    print(f"  clips       : {len(clips)}")

    for m in MODELS:
        if m in results:
            print(f"\n{m}: already checked; skipped")
            continue
        # Step 1: wait for memory, then load the model with its answers and features in a temporary folder
        print(f"\nStep 1 | {m}")
        while free_memory() < MIN_FREE_PERCENT:
            print(f"  busy        : {free_memory()}% memory free; checking again in {WAIT_SECONDS} s")
            time.sleep(WAIT_SECONDS)
        print(f"  memory      : {free_memory()}% free")
        scratch = Path(tempfile.mkdtemp(prefix=f"extras_{m}_"))
        entry = registry[m] | {"key": m, "cache_file": scratch / "new.jsonl", "features_dir": scratch / "features"}
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
        loader = importlib.import_module(entry["loader"])
        predict = loader.load(PROJECT / entry["dir"], torch.device("mps"), entry=entry)
        caches = {False: (scratch / "old.jsonl", {}), True: (scratch / "new.jsonl", {})}

        # Step 2: each clip without and with the by-products, alternating which goes first; clip 0 = warm-up
        warm = clips[0]
        run_mode(predict, warm, False, caches)
        run_mode(predict, warm, True, caches)
        pairs = []
        for i, clip in enumerate(clips[1:]):
            first = i % 2 == 0
            a = run_mode(predict, clip, not first, caches)
            b = run_mode(predict, clip, first, caches)
            old, new = (a, b) if first else (b, a)
            pairs.append((old, new))

        # Step 3: same answers; probabilities; agreement; time
        same = [o["answers"] == n["answers"] for o, n in pairs]
        sums = [sum(n["label_probs"].values()) for _, n in pairs]
        top = [max(n["label_probs"], key=n["label_probs"].get) for _, n in pairs]
        agree = [t == n["label"] for t, (_, n) in zip(top, pairs)]
        summary = {
            "clips": len(pairs), "identical_answers": sum(same),
            "differences": [(o["audio_path"], o["answers"], n["answers"]) for (o, n), s in zip(pairs, same) if not s],
            "prob_sum_min_max": [min(sums), max(sums)],
            "argmax_agrees": sum(agree),
            "disagreements": [(n["audio_path"], n["answers"], n["label_probs"]) for (_, n), a in zip(pairs, agree)
                              if not a],
            "label_mass_mean_min": [round(statistics.mean(n["label_mass"] for _, n in pairs), 4),
                                    min(n["label_mass"] for _, n in pairs)],
            "seconds_old_new_mean": [round(statistics.mean(o["seconds"] for o, _ in pairs), 3),
                                     round(statistics.mean(n["seconds"] for _, n in pairs), 3)],
            "answers": [n["answers"] for _, n in pairs],
        }
        # The probabilities must be what evaluate.py's predictions CSV gets
        assert set(predict.extra(str(clips[1]))) == {f"token_p_{lab}" for lab in predict.choices} | {"token_mass"}

        # Step 4: reload the features: one row per clip run with the by-products, finite values
        features = {}
        for folder in sorted((scratch / "features").iterdir()):
            before = len(load_features(folder)[1])
            run_mode(predict, clips[1], True, caches)  # answered already: no new row
            vectors, index, meta = load_features(folder)
            values = np.asarray(vectors, dtype=np.float32)
            features[folder.name] = {"rows": len(index), "rows_before_cached_call": before, "shape": list(vectors.shape),
                                     "layers": meta["layers"], "num_layers": meta["num_layers"],
                                     "pooling": sorted({r["pooling"] for r in index}),
                                     "tokens": [r["tokens"] for r in index], "finite": bool(np.isfinite(values).all()),
                                     "max_abs": float(np.abs(values).max()),
                                     "bytes_per_row": int(np.prod(meta["row_shape"])) * 2}
        summary["features"] = features
        print(json.dumps({k: v for k, v in summary.items() if k != "answers"}, indent=1))
        print("  answers     :", summary["answers"])
        results[m] = summary
        RESULTS.write_text(json.dumps(results, indent=1))

        # Free the model before the next one
        del predict
        gc.collect()
        torch.mps.empty_cache()
        shutil.rmtree(scratch)
        print(f"  freed; {free_memory()}% memory free")
    print(f"\nDone: {RESULTS}")


if __name__ == "__main__":
    main()

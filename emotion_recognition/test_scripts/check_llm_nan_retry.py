"""
Informal check of the NaN retries of the local audio LLMs (nan_retries in config/llm.yaml; scripts/models/audio_llm.py),
without a real NaN: the model class's generate() is wrapped so that its next N calls return NaN logits and an empty
answer (what a real NaN gave), set before each clip.

What it does, step by step, with MODEL (Gemma 4 E4B) on the CLIPS clips of data/samples/:
    1. waits until enough memory is free (MIN_FREE_PERCENT), then loads the model through its loader as
       scripts/evaluation/evaluate.py does, with its answers and features in a fresh temporary folder (never outputs/
       or features/);
    2. a warm-up pass, then the clean run (no NaN): the answers, label probabilities and features of every clip, as
       the reference (the first generate() calls after a model load give slightly different numbers on MPS);
    3. NaN run, with a new answer cache and features folder: clip 1 gets 1 NaN call (recovers at retry 1, the memory
       cache emptied), clip 2 gets 2 (recovers at retry 2, the weights reloaded), clip 3 gets 4 (given up after 3
       retries), clip 4 none, but the run is interrupted (KeyboardInterrupt) right after its features row was
       written, before its answer; then half a row of junk bytes is appended to hidden.f16, as a kill would leave;
    4. resume as a new process would (cache and features read again from disk), no NaN: clips 1-2 come from the
       cache, clip 3 (given up: its line has an "error") and clip 4 run again;
    5. checks: the answers equal the clean run's (clip 1, recovered without a reload: also exactly its probabilities
       and vector; after a reload they are close, not always equal, as after any model load), clip 3 was invalid with
       its error and no probabilities or features, the run went on after it, the reload count, and after the resume
       one valid answer and one features row per clip.
Nothing leaves the machine.

Usage (from the emotion_recognition/ folder, in the emotion_recognition_llm environment):
    OMP_NUM_THREADS=4 nice -n 10 python test_scripts/check_llm_nan_retry.py
"""
import importlib
import json
import os
import re
import shutil
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
MODEL = "gemma4_e4b"  # key in config/models.yaml
CLIPS = 4             # the first clips of data/samples/
NAN_CALLS = [1, 2, 4, 0]  # NaN generate() calls for each clip in step 3
MIN_FREE_PERCENT = 40     # wait before loading the model while less memory is free (memory_pressure)
WAIT_SECONDS = 120

sys.path.insert(0, str(PROJECT / "scripts" / "common"))
sys.path.insert(0, str(PROJECT / "scripts" / "models"))

POISON = {"left": 0, "calls": 0}  # NaN calls still to give; generate() calls so far


def free_memory():
    """Free memory in percent (memory_pressure)."""
    last = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return int(re.findall(r"(\d+)%", last)[0])


def poison(model_class, pad_id):
    """Wrap model_class.generate (on the class, so a reloaded model has it too): while POISON["left"] > 0, a call
    returns NaN logits at every step and only an end token after the prompt (an empty answer)."""
    real = model_class.generate

    def generate(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        POISON["calls"] += 1
        if POISON["left"] > 0:
            POISON["left"] -= 1
            result.logits = tuple(torch.full_like(step, float("nan")) for step in result.logits)
            prompt = kwargs["input_ids"].shape[1]
            sequences = result.sequences[:, :prompt + 1].clone()
            sequences[:, prompt] = pad_id
            result.sequences = sequences
        return result

    model_class.generate = generate


def use(predict, folder):
    """Point the predictor at an answer cache and features folder, read again from disk as a new process would."""
    predict.cache_file, predict.features_dir = folder / "responses.jsonl", folder / "features"
    predict.cache, predict.features = predict.read_cache(), {}


def main():
    sys.stdout.reconfigure(line_buffering=True)
    from audio_llm import load_features

    # Step 0: versions, clips
    registry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())
    clips = sorted((PROJECT / "data" / "samples").glob("*.wav"))[:CLIPS]
    rels = [c.resolve().relative_to(PROJECT).as_posix() for c in clips]
    print(f"Step 0 | torch {torch.__version__}, MPS {torch.backends.mps.is_available()}, clips {len(clips)}")

    # Step 1: wait for memory, then load the model with answers and features in a temporary folder
    print(f"\nStep 1 | load {MODEL}")
    while free_memory() < MIN_FREE_PERCENT:
        print(f"  busy        : {free_memory()}% memory free; checking again in {WAIT_SECONDS} s")
        time.sleep(WAIT_SECONDS)
    print(f"  memory      : {free_memory()}% free")
    scratch = Path(tempfile.mkdtemp(prefix=f"nan_retry_{MODEL}_"))
    entry = registry[MODEL] | {"key": MODEL, "cache_file": scratch / "clean" / "responses.jsonl",
                               "features_dir": scratch / "clean" / "features"}
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    loader = importlib.import_module(entry["loader"])
    builds = []  # every model load (the first one and the reloads)
    real_build = loader.build

    def build(*args):  # counts the loads; before each, waits for memory (on a reload the old model is freed already)
        while free_memory() < MIN_FREE_PERCENT:
            print(f"  busy        : {free_memory()}% memory free; checking again in {WAIT_SECONDS} s")
            time.sleep(WAIT_SECONDS)
        print(f"  model load  : {len(builds) + 1}, {free_memory()}% memory free, MPS allocated "
              f"{torch.mps.current_allocated_memory() / 1e9:.1f} GB (0 on a reload: the old model is freed)")
        builds.append(1)
        return real_build(*args)

    loader.build = build
    predict = loader.load(PROJECT / entry["dir"], torch.device("mps"), entry=entry)
    poison(type(predict.model), predict.processor.tokenizer.pad_token_id)
    print(f"  scratch     : {scratch}; nan_retries {predict.nan_retries}")

    # Step 2: clean run, the reference, after a warm-up pass: the first generate() calls after a model load give
    # slightly different probabilities and hidden states on MPS (same answers; seen in a control run without NaN)
    print("\nStep 2 | warm-up pass, then the clean run")
    use(predict, scratch / "warmup")
    for clip in clips:
        predict(clip)
    use(predict, scratch / "clean")
    for clip in clips:
        predict(clip)
    clean = {r: predict.cache[r] for r in rels}
    clean_vectors, clean_index, _ = load_features(scratch / "clean" / "features" / "samples")
    clean_rows = {row["audio_path"]: np.asarray(clean_vectors[i], dtype=np.float32)
                  for i, row in enumerate(clean_index)}
    for r in rels:
        print(f"  clean       : {Path(r).name}: {clean[r]['answers']} -> {clean[r]['label']}")

    # Step 3: NaN run; clip 4 interrupted after its features row
    print("\nStep 3 | NaN run")
    nan_dir = scratch / "nan"
    use(predict, nan_dir)
    builds_before, probs = len(builds), {}
    real_save = predict.save_features
    for clip, rel, n in zip(clips, rels, NAN_CALLS):
        POISON["left"] = n
        if rel == rels[-1]:  # interrupted after its features row was written, before its answer
            def save_then_stop(*args):
                real_save(*args)
                raise KeyboardInterrupt
            predict.save_features = save_then_stop
        try:
            probs[rel] = predict(clip)
            probs[rel] = (probs[rel], predict.extra(str(clip)))  # one-hot, extra columns (none when given up)
        except KeyboardInterrupt:
            print(f"  interrupted : {Path(rel).name}, after its features row")
        predict.save_features = real_save
        print(f"  NaN calls   : {Path(rel).name}: {n} given, {POISON['left']} left unused")
    reloads = len(builds) - builds_before
    gave_up = list(predict.gave_up)
    lines = [json.loads(line) for line in (nan_dir / "responses.jsonl").read_text().splitlines()]
    with open(nan_dir / "features" / "samples" / "hidden.f16", "ab") as fh:  # half a row, as a kill would leave
        fh.write(b"\x00" * (predict.features["samples"].row_bytes() // 2))

    # Step 4: resume as a new process would, no NaN
    print("\nStep 4 | resume")
    use(predict, nan_dir)
    ran_before = predict.ran
    for clip in clips:
        predict(clip)
    ran_resume = predict.ran - ran_before
    final = predict.read_cache()
    vectors, index, _ = load_features(nan_dir / "features" / "samples")

    # Step 5: checks
    print("\nStep 5 | checks")
    given_up = lines[2] if len(lines) > 2 else {}
    # Per clip after the resume: how far the label probabilities and the vectors are from the clean run
    rows = {row["audio_path"]: i for i, row in enumerate(index)}
    dp = {r: max(abs(final[r]["label_probs"][lab] - clean[r]["label_probs"][lab]) for lab in predict.choices)
          for r in rels}
    dv = {r: float(np.abs(np.asarray(vectors[rows[r]], dtype=np.float32) - clean_rows[r]).max()) for r in rels}
    scale = max(float(np.abs(v).max()) for v in clean_rows.values())
    checks = {
        "step 3: clips 1, 2 recovered with the clean answers": all(lines[i]["answers"] == clean[rels[i]]["answers"]
                                                                   for i in range(2)),
        "step 3: clip 1 (retry 1, cache emptied) exactly the clean label probs and vector": dp[rels[0]] == 0
            and dv[rels[0]] == 0,
        "step 3: reloads = 1 (clip 2) + 2 (clip 3)": reloads == 3,
        "step 3: clip 3 given up (gave_up list)": gave_up == [rels[2]],
        "step 3: clip 3 line: invalid, error, no label probs": given_up.get("label") == "invalid"
            and given_up.get("error") == "nan after 3 retries" and "label_probs" not in given_up,
        "step 3: clip 3 one-hot = invalid, no extra columns": probs[rels[2]][0][predict.labels.index("invalid")] == 1
            and probs[rels[2]][1] == {},
        "step 3: 3 answer lines (clip 4 interrupted)": len(lines) == 3,
        "step 4: 2 clips run again (3 and 4)": ran_resume == 2,
        "step 4: one valid answer per clip": sorted(final) == sorted(rels),
        "step 4: answers = clean run": all(final[r]["answers"] == clean[r]["answers"] for r in rels),
        # After a reload the first calls are not warm (see step 2): close to the clean run, not always equal
        "step 4: label probs within 0.05 of the clean run": max(dp.values()) <= 0.05,
        "step 4: vectors within 5% of the largest value of the clean run": max(dv.values()) <= 0.05 * scale,
        "step 4: one features row per clip, in index order": [row["audio_path"] for row in index] == [
            rels[0], rels[1], rels[3], rels[2]] and [row["row"] for row in index] == list(range(len(rels))),
        "step 4: hidden.f16 holds exactly those rows": (nan_dir / "features" / "samples" / "hidden.f16").stat().st_size
            == len(index) * predict.features["samples"].row_bytes(),
    }
    for name, ok in checks.items():
        print(f"  {'OK  ' if ok else 'FAIL'}        : {name}")
    print(f"  answers     : clean {[clean[r]['answers'] for r in rels]}")
    print(f"  given up    : {json.dumps(given_up)}")
    for r in rels:
        print(f"  per clip    : {Path(r).name}: label probs max diff {dp[r]:.2e}, vectors max diff {dv[r]:.3g} "
              f"(largest clean value {scale:.4g})")
    print(f"  calls       : {POISON['calls']} generate() calls, {len(builds)} model loads")
    print(f"\n{'ALL OK' if all(checks.values()) else 'SOME CHECKS FAILED'}")

    # Free the model and the temporary folder
    del predict
    shutil.rmtree(scratch)


if __name__ == "__main__":
    main()

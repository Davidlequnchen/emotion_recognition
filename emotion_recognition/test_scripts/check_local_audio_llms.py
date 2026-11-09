"""
Informal check of one open-weight audio LLM (Qwen2.5-Omni, Qwen2-Audio or Gemma 4) through its loader in
scripts/models/: load it from models/<key>/, run it on a few clips and print the answers, the speed and the memory.

What it does, step by step:
    1. reads the model's registry entry (config/models.yaml) and prints the library versions and the device;
    2. loads the model through its loader (load time; memory of the device afterwards);
    3. runs it on CLIPS (clips of each dataset, taken as evaluate.py takes them with `limit`): per clip the acted
       emotion, the raw answer(s), the parsed label and the time; the first clip is the warm-up;
    4. prints the summary: load time, ms per clip after the warm-up, peak device memory, invalid answers, accuracy.
The answers go to a fresh temporary folder (printed), never to outputs/, so no real results are written and nothing
is cached between runs. Nothing leaves the machine.

Usage (from the emotion_recognition/ folder; edit the constants below to test another model, device or dtype):
    python test_scripts/check_local_audio_llms.py
"""
import gc
import platform
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
MODEL = "qwen2_audio_7b"   # qwen2_5_omni_7b | qwen2_audio_7b | gemma4_e4b (keys in config/models.yaml)
DEVICE = "mps"             # mps | cpu | cuda:0
DTYPE = None               # None: the loader's choice (bfloat16 on a GPU); or "float16" / "float32" to compare
CLIPS = {"crema_d": 14, "mcae_spps": 2, "tess": 2}  # dataset -> number of clips

sys.path.insert(0, str(PROJECT / "scripts" / "models"))
sys.path.insert(0, str(PROJECT / "scripts" / "common"))


def memory(device):
    """Memory the device holds now, in GB (MPS: what the Metal driver has allocated; CUDA: reserved)."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e9
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device) / 1e9
    return 0.0


def main(model=MODEL, device=DEVICE, dtype=DTYPE, clips=CLIPS):
    sys.stdout.reconfigure(line_buffering=True)
    import importlib

    import transformers

    import audio_llm
    from eval_datasets import LOADERS, sample

    # Step 1: the registry entry, the versions and the device
    entry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())[model]
    device = torch.device(device)
    print(f"Step 1 | {entry['name']} ({entry['hf_repo']}), loader scripts/models/{entry['loader']}.py")
    print(f"  Python {platform.python_version()}, torch {torch.__version__}, transformers {transformers.__version__}")
    print(f"  device {device}, dtype {dtype or audio_llm.dtype_for(device)}")
    if dtype:  # override the loader's choice for this check only
        audio_llm.dtype_for = lambda _: getattr(torch, dtype)
    scratch = Path(tempfile.mkdtemp(prefix=f"{model}_"))
    entry = entry | {"key": model, "cache_file": scratch / "responses.jsonl", "features_dir": scratch / "features"}

    # Step 2: load the model
    print("\nStep 2 | load")
    t0 = time.perf_counter()
    predict = importlib.import_module(entry["loader"]).load(PROJECT / entry["dir"], device, entry)
    load_s = time.perf_counter() - t0
    peak = memory(device)
    print(f"  loaded in {load_s:.1f} s; device memory {peak:.1f} GB; labels {predict.labels}")
    for n in predict.notes:
        print(f"  note: {n}")
    # What the model reads: its chat template filled with the prompts (the audio placeholder is expanded later)
    text = predict.processor.apply_chat_template(predict.family.conversation(predict.system, predict.user, []),
                                                 tokenize=False, add_generation_prompt=True, **predict.template_kwargs)
    print(f"  chat template:\n{text}")

    # Step 3: run it on the clips
    print("\nStep 3 | clips")
    items = [(key, c) for key, n in clips.items() for c in sample(LOADERS[key](), n)]
    times, invalid, correct = [], 0, 0
    for i, (key, clip) in enumerate(items):
        t0 = time.perf_counter()
        probs = predict(str(clip["audio"]))
        seconds = time.perf_counter() - t0
        times.append(seconds)
        peak = max(peak, memory(device))
        record = predict.cache[str(Path(clip["audio"]).resolve().relative_to(PROJECT))]
        label = predict.labels[probs.argmax()]
        invalid += label == "invalid"
        correct += label == clip["acted"]
        print(f"  {i + 1:>2} {key:<10} {clip['acted']:<10} -> {label:<10} {seconds * 1000:>6.0f} ms"
              f"{' (warm-up)' if i == 0 else ''}  answers {record['answers']}")

    # Step 4: summary
    after = times[1:] or times
    print(f"\nStep 4 | summary of {entry['name']} on {device} ({dtype or audio_llm.dtype_for(device)})")
    print(f"  load            : {load_s:.1f} s")
    print(f"  ms per clip     : {sum(after) / len(after) * 1000:.0f} (after the warm-up of {times[0] * 1000:.0f} ms)")
    print(f"  device memory   : {peak:.1f} GB at most")
    print(f"  invalid answers : {invalid} of {len(items)}")
    print(f"  acted emotion   : {correct} of {len(items)} correct (informal)")
    print(f"  answers         : {entry['cache_file']}")
    del predict
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()

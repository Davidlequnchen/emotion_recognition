"""Local, resumable EmotionThinker / Qwen prompt comparisons on the adopted datasets.

Results live separately from the established benchmark. Default: a 14-clip pilot
on CUDA, NOT a full benchmark. --limit 0 evaluates every adopted clip.
EmotionThinker inference on MPS/CPU is disabled at the user's request; manifests
can still be inspected without loading a model. See the research report.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
for folder in ("common", "models", "llm_eval"):
    sys.path.insert(0, str(ROOT / "scripts" / folder))
from parsing import parse_label
from prompts import REMINDER, load_prompts

VERSION = 2
MODELS = {
    "emotionthinker": ("qwen2_5_omni", "local_models/emotionthinker", "ddwang2000/EmotionThinker"),
    "qwen2_5_omni_7b": ("qwen2_5_omni", "models/qwen2_5_omni_7b", "Qwen/Qwen2.5-Omni-7B"),
    "qwen2_audio_7b": ("qwen2_audio", "models/qwen2_audio_7b", "Qwen/Qwen2-Audio-7B-Instruct"),
}
OFFICIAL_SYSTEM = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."
)
OFFICIAL_USER = (
    "<audio>What is the emotion expressed in this audio clip? Please choose one from the following options: "
    "neutral, happy, sad, angry, contempt or disgust, confused, whisper, surprise, fear."
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pilot_clips(ds, limit):
    """Evenly spaced clips within each truth class; pilots are class balanced."""
    if limit == 0 or limit >= len(ds.clips):
        return ds.clips
    groups = {label: [c for c in ds.clips if c["acted"] == label]
              for label in dict.fromkeys(c["acted"] for c in ds.clips)}
    quotas = {label: 0 for label in groups}
    for _ in range(limit):
        label = min((k for k in groups if quotas[k] < len(groups[k])), key=lambda k: quotas[k])
        quotas[label] += 1
    return [clips[i * len(clips) // quotas[label]] for label, clips in groups.items()
            for i in range(quotas[label])]


def parse_answer(text, labels, protocol):
    if protocol == "project":
        return parse_label(text, labels)
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    # Never search an explanation for an emotion; no answer or several answers is invalid.
    if len(answers) != 1:
        return None
    answer = answers[0].strip().lower().strip(" .!\"'")
    if answer == "contempt or disgust":
        return "disgusted" if "disgusted" in labels else None
    return parse_label(answer, labels)


def prompt_for(protocol, classes):
    project = load_prompts(ROOT / "scripts/llm_eval/prompts.yaml", labels=classes)
    if protocol == "project":
        return project["system"], project["user"], project["labels"]
    user = OFFICIAL_USER if protocol == "official_demo" else (
        "<audio>What is the emotion expressed in this audio clip? Please choose one from the following options: "
        + ", ".join(project["labels"]) + "."
    )
    return OFFICIAL_SYSTEM, user, project["labels"]


def metrics(records, ds):
    import numpy as np
    from sklearn.metrics import f1_score
    correct = [r["prediction"] in ds.accept.get(r["truth"], [r["truth"]]) for r in records]
    counts, hits = Counter(r["truth"] for r in records), Counter()
    for row, ok in zip(records, correct):
        hits[row["truth"]] += ok
    truth = [r["prediction"] if ok else ds.accept.get(r["truth"], [r["truth"]])[0]
             for r, ok in zip(records, correct)]
    pred = [r["prediction"] for r in records]
    per_class = f1_score(truth, pred, labels=ds.classes, average=None, zero_division=0)
    support = np.array([truth.count(label) for label in ds.classes])
    return {"n": len(records), "accuracy": sum(correct) / len(records),
            "uar": sum(hits[k] / counts[k] for k in counts) / len(counts),
            "macro_f1": float(per_class.mean()),
            "weighted_f1": float(np.average(per_class, weights=support)),
            "invalid": pred.count("invalid"), "truncated": sum(r["truncated"] for r in records),
            "seconds": sum(r["seconds"] for r in records),
            "per_class": {k: {"n": counts[k], "recall": hits[k] / counts[k]} for k in counts}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=MODELS, default="emotionthinker")
    ap.add_argument("--protocol", choices=["project", "dataset_reasoning", "official_demo"], default="dataset_reasoning")
    ap.add_argument("--datasets", nargs="+", default=["meld", "ravdess", "savee"])
    ap.add_argument("--limit", type=int, default=14, help="clips/dataset; 0 = full adopted split")
    ap.add_argument("--device", choices=["mps", "cuda:0", "cpu"], default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int)
    ap.add_argument("--manifest-only", action="store_true", help="write selected clip IDs without loading a model")
    ap.add_argument("--output", type=Path, default=ROOT.parent / "research_paper/output/emotion_reasoning_benchmark")
    args = ap.parse_args()
    if args.limit < 0 or (args.max_new_tokens is not None and args.max_new_tokens < 1):
        ap.error("limits must be nonnegative; max-new-tokens must be positive")
    if args.model == "emotionthinker" and args.device != "cuda:0" and not args.manifest_only:
        ap.error("EmotionThinker inference on this Mac/CPU is disabled at the user's request; use NVIDIA --device cuda:0")
    if not args.manifest_only and args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            ap.error("NVIDIA CUDA is unavailable; no model will be loaded. Use a CUDA server, or --manifest-only for a read-only check")
    from eval_datasets import LOADERS
    keys = [k for k in LOADERS if k != "esd" and (ROOT / "data" / k).is_dir()] if args.datasets == ["all"] else args.datasets
    if any(k == "esd" or k not in LOADERS for k in keys):
        ap.error("choose adopted dataset keys; ESD is excluded")
    family_name, model_rel, model_id = MODELS[args.model]
    model_dir = ROOT / model_rel
    max_tokens = args.max_new_tokens or (24 if args.protocol == "project" else 2048)
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_dir.glob("*.json")}
    files["weights"] = [(p.name, p.stat().st_size) for p in sorted(model_dir.glob("*.safetensors"))]
    model_fingerprint = fingerprint(files)
    model = processor = None
    summaries = []
    for key in keys:
        ds = LOADERS[key]()
        selected = pilot_clips(ds, args.limit)
        system, user, labels = prompt_for(args.protocol, ds.classes)
        spec = {"version": VERSION, "model": model_id, "model_fingerprint": model_fingerprint,
                "protocol": args.protocol, "system": system, "user": user, "labels": labels,
                "max_new_tokens": max_tokens, "do_sample": False, "use_cache": True,
                "device": args.device, "dtype": "float32" if args.device == "cpu" else "bfloat16",
                "sample_rate": 16000, "retry_invalid": 1 if args.protocol == "project" else 0,
                "parser": "short_label" if args.protocol == "project" else "answer_tag_only",
                "reminder": REMINDER.format(labels=", ".join(labels)) if args.protocol == "project" else None,
                "max_audio_seconds": 30 if family_name == "qwen2_audio" else None}
        run_hash = fingerprint(spec)[:16]
        target = args.output / args.model / args.protocol / run_hash
        target.mkdir(parents=True, exist_ok=True)
        (target / "protocol.json").write_text(json.dumps(spec, indent=2))
        manifest = [{"audio_path": str(c["audio"].relative_to(ROOT)), "truth": c["acted"],
                     "audio_bytes": c["audio"].stat().st_size, "audio_mtime_ns": c["audio"].stat().st_mtime_ns}
                    for c in selected]
        suffix = "full" if args.limit == 0 else f"sample{args.limit}"
        manifest_file = target / f"{key}_{suffix}_manifest.json"
        manifest_file.write_text(json.dumps({"dataset": key, "subset": ds.subset, "classes": ds.classes,
            "accept": ds.accept, "n_adopted": len(ds.clips), "n_selected": len(manifest),
            "selection": "all_adopted" if len(selected) == len(ds.clips) else "class_balanced_even_spacing",
            "clips": manifest}, indent=2))
        if args.manifest_only:
            print(key, len(manifest), manifest_file, flush=True)
            continue
        cache_file = target / f"{key}_responses.jsonl"
        cached = {}
        if cache_file.exists():
            # A partially written last row is ignored; earlier malformed rows are errors.
            lines = cache_file.read_text().splitlines()
            valid = []
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if i != len(lines) - 1:
                        raise
                    break
                cached[record["audio_path"]] = record
                valid.append(line)
            cache_file.write_text("".join(line + "\n" for line in valid))
        records = []
        for i, (clip, identity) in enumerate(zip(selected, manifest)):
            record = cached.get(identity["audio_path"])
            if record is None or any(record.get(k) != v for k, v in identity.items()):
                import torch
                from audio_io import load_audio
                if model is None:
                    if args.device == "mps":
                        from audio_llm import plain_stop_check_on_mps
                        plain_stop_check_on_mps()
                    family = importlib.import_module(family_name)
                    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
                    model, processor = family.build(model_dir, torch.device(args.device), dtype)
                    print("Loaded", model_id, "parameters", sum(p.numel() for p in model.parameters()), flush=True)
                wav = load_audio(clip["audio"])
                if spec["max_audio_seconds"]:
                    wav = wav[:16000 * spec["max_audio_seconds"]]
                answers, turns, token_count, truncated = [], [], 0, False
                started = time.perf_counter()
                for attempt in range(spec["retry_invalid"] + 1):
                    text = processor.apply_chat_template(family.conversation(system, user, turns),
                                                         tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=text, audio=[wav], sampling_rate=16000, return_tensors="pt")
                    inputs = {k: v.to(args.device, dtype) if v.is_floating_point() else v.to(args.device)
                              for k, v in inputs.items()}
                    stops = list(dict.fromkeys([processor.tokenizer.eos_token_id,
                        *[processor.tokenizer.convert_tokens_to_ids(t) for t in getattr(family, "STOP_TOKENS", [])]]))
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                            temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0,
                            use_cache=True, return_dict_in_generate=False, output_scores=False,
                            eos_token_id=[t for t in stops if t is not None],
                            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
                    new = output[0, inputs["input_ids"].shape[1]:]
                    answer = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
                    token_count += len(new)
                    truncated |= len(new) == max_tokens
                    answers.append(answer)
                    label = parse_answer(answer, labels, args.protocol)
                    if label is not None:
                        break
                    turns.append((answer, spec["reminder"]))
                if args.device == "mps":
                    torch.mps.synchronize()
                record = {**identity, "prediction": label or "invalid", "answers": answers,
                          "seconds": round(time.perf_counter() - started, 4), "new_tokens": token_count,
                          "truncated": truncated, "run_hash": run_hash,
                          "date": datetime.now(ZoneInfo("Asia/Singapore")).isoformat()}
                with cache_file.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"{key} {i+1}/{len(selected)} {record['prediction']} truth={clip['acted']} "
                      f"{record['seconds']:.1f}s {token_count} tokens", flush=True)
            records.append(record)
        result = {"model": args.model, "dataset": key, "protocol": spec, "run_hash": run_hash,
                  "scope": "full_adopted" if len(selected) == len(ds.clips) else "pilot",
                  "subset": ds.subset, "metrics": metrics(records, ds)}
        (target / f"{key}_{suffix}_metrics.json").write_text(json.dumps(result, indent=2))
        summaries.append(result)
        print(json.dumps({"dataset": key, **result["metrics"]}), flush=True)
    if summaries:
        index = args.output / args.model / f"{args.protocol}_{suffix}_summary.json"
        index.write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

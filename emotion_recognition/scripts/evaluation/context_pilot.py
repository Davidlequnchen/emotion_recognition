"""Prepare or run a fixed-target H0–H3 context pilot. No subcommand means prepare, never inference.

Preparation/analysis use only the standard library and PyYAML. Model/audio libraries are imported only by `run`.
Artifacts live separately from ordinary single-utterance evaluation outputs and cannot replace its caches.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/common"))
DEFAULT_CONFIG = ROOT / "config/context_pilot.yaml"
CONDITIONS = ("H0", "H1", "H2", "H3")
CONDITION_DOC = {
    "H0": "target audio only",
    "H1": "true completed past audio + same target audio",
    "H2": "transcript of the same past turn + same target audio",
    "H3": "unrelated, speaker/duration-matched train audio + same target audio",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    if config.get("dataset") != "emotiontalk" or config.get("split") != "train":
        raise ValueError("This pilot is fixed to EmotionTalk train; use separate configs/protocols for extensions")
    if config.get("model") != "qwen2_5_omni_7b":
        raise ValueError("This pilot's multi-audio adapter supports qwen2_5_omni_7b only")
    if config.get("conditions") != list(CONDITIONS) or config["history"]["turns"] != 1:
        raise ValueError("The initial pilot requires H0–H3 and one complete history turn")
    if not isinstance(config.get("samples"), int) or config["samples"] <= 0:
        raise ValueError("samples must be a positive integer")
    if config["selection"]["strategy"] != "balanced_by_gold":
        raise ValueError("Only the explicitly class-balanced pilot selection is implemented")
    canonical = {"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"}
    if len(config["labels"]) != 7 or set(config["labels"]) != canonical:
        raise ValueError("Use seven distinct canonical emotion candidates")
    if not config["unrelated"]["match_speaker"] or not config["unrelated"]["exclude_selected_targets"]:
        raise ValueError("H3 must match published speaker ID and exclude selected targets")
    if not 0 < config["unrelated"]["duration_relative_tolerance"] < 1:
        raise ValueError("H3 duration tolerance must be between zero and one")
    if config["selection"]["min_target_seconds"] <= 0 or config["selection"]["max_target_seconds"] <= 0:
        raise ValueError("Target durations must be positive")
    if config["selection"]["max_per_dialogue"] < 1:
        raise ValueError("max_per_dialogue must be positive")


def select_cases(records, config):
    """Deterministic gold-stratified targets; histories/donors are never selected by emotion labels."""
    from dialogue_datasets import completed_history
    validate_config(config)
    labels, selection, history = config["labels"], config["selection"], config["history"]
    by_dialogue = defaultdict(list)
    for row in records:
        if row["split"] != "train":
            raise ValueError("The development pool must contain only train records")
        by_dialogue[row["dialogue"]].append(row)
    pool = []
    for target in sorted(records, key=lambda r: r["id"]):
        if target["gold"] not in labels or not target["transcript"].strip():
            continue
        if not selection["min_target_seconds"] <= target["duration"] <= selection["max_target_seconds"]:
            continue
        past = completed_history(target, by_dialogue[target["dialogue"]], max_turns=1)
        if not past or not past[0]["transcript"].strip():
            continue
        previous = past[0]
        if previous["duration"] > history["max_seconds"]:
            continue
        if previous["duration"] + target["duration"] > history["max_total_seconds"]:
            continue
        pool.append({"target": target, "history": previous})
    rng = random.Random(config["seed"])
    rng.shuffle(pool)
    quotas = {lab: config["samples"] // len(labels) + int(i < config["samples"] % len(labels))
              for i, lab in enumerate(labels)}
    counts, dialogue_counts, chosen = Counter(), Counter(), []
    for case in pool:
        target = case["target"]
        lab, dialogue = target["gold"], target["dialogue"]
        if counts[lab] >= quotas[lab] or dialogue_counts[dialogue] >= selection["max_per_dialogue"]:
            continue
        chosen.append(case)
        counts[lab] += 1
        dialogue_counts[dialogue] += 1
    if len(chosen) != config["samples"]:
        raise ValueError(f"Not enough eligible targets with the configured caps: {dict(counts)}, quotas {quotas}")
    selected_ids = {c["target"]["id"] for c in chosen}
    tolerance = config["unrelated"]["duration_relative_tolerance"]
    donors_used = set()
    for case in chosen:
        target, previous = case["target"], case["history"]
        candidates = [r for r in records if r["id"] not in selected_ids and r["id"] not in donors_used
                      and r["dialogue"] != target["dialogue"] and r["speaker"] == previous["speaker"]
                      and 0 < r["duration"] <= history["max_seconds"]
                      and r["duration"] + target["duration"] <= history["max_total_seconds"]
                      and abs(r["duration"] / previous["duration"] - 1) <= tolerance]
        if not candidates:
            raise ValueError(f"No unused, speaker/duration-matched H3 donor for {target['id']}")
        # Stable seeded tie-breaks; neither target/history/donor emotion is inspected here.
        donor = min(candidates, key=lambda r: (abs(math.log(r["duration"] / previous["duration"])),
                                               digest([config["seed"], target["id"], r["id"]])))
        case["unrelated"] = donor
        donors_used.add(donor["id"])
    rng.shuffle(chosen)
    return chosen, {"eligible_targets": len(pool), "selected_by_gold": dict(counts),
                    "dialogues": len(dialogue_counts), "max_targets_per_dialogue": max(dialogue_counts.values())}


def public_utterance(record, audio_meta):
    """Whitelist inputs; labels, votes, confidence and captions are deliberately absent."""
    return {"id": record["id"], "audio": relative(record["audio"]), "speaker": record["speaker"],
            "dialogue": record["dialogue"], "split": record["split"], "start": record["start"],
            "end": record["end"], "duration": record["duration"], "transcript": record["transcript"],
            **audio_meta}


def build_messages(case, condition, protocol):
    """Actual model input. H1/H3 text is identical; only their history audio differs."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    prompts, target, past = protocol["prompts"], case["target"], case["history"]
    speaker = "Speaker A" if past["speaker"] == target["speaker"] else "Speaker B"
    content = []
    if condition in ("H1", "H3"):
        previous = past if condition == "H1" else case["unrelated"]
        content += [{"type": "text", "text": f"HISTORY — previous utterance by {speaker}:"},
                    {"type": "audio", "audio": previous["audio"]}]
    elif condition == "H2":
        content.append({"type": "text", "text": f"HISTORY — transcript of the previous utterance by {speaker}:\n"
                        + json.dumps(past["transcript"], ensure_ascii=False)})
    content += [{"type": "text", "text": "TARGET — utterance by Speaker A:"},
                {"type": "audio", "audio": target["audio"]},
                {"type": "text", "text": prompts["question"]}]
    return [{"role": "system", "content": [{"type": "text", "text": prompts["system"]}]},
            {"role": "user", "content": content}]


def prepare(config_path=DEFAULT_CONFIG):
    from dialogue_datasets import load_dialogue_records
    config = yaml.safe_load(Path(config_path).read_text())
    validate_config(config)
    records = load_dialogue_records(config["dataset"], config["split"], primary_only=True)
    selected, selection_audit = select_cases(records, config)
    registry = yaml.safe_load((ROOT / "config/models.yaml").read_text())
    model = registry[config["model"]]
    model_manifest = ROOT / "models/manifest.json"
    identity = json.loads(model_manifest.read_text())["models"][config["model"]]
    processor_files = ("config.json", "generation_config.json", "preprocessor_config.json", "chat_template.json",
                       "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json",
                       "special_tokens_map.json")
    processor_identity = {relative(path): file_hash(path) for name in processor_files
                          if (path := ROOT / model["dir"] / name).is_file()}
    protocol = {"schema_version": 1, "dataset": config["dataset"], "split": config["split"],
                "model": config["model"], "model_id": model["hf_repo"], "model_dir": model["dir"],
                "model_manifest_hash": digest(identity), "labels": config["labels"],
                "processor_files_sha256": processor_identity,
                "dataset_source": json.loads((ROOT / "data/emotiontalk/source_manifest.json").read_text()),
                "dataset_index_sha256": file_hash(ROOT / "data/emotiontalk/utterances.csv"),
                "prompts": {k: v.replace("{labels}", ", ".join(config["labels"]))
                            for k, v in config["prompts"].items()}, "config": config,
                "audio": {"sample_rate": 16000, "channels": 1, "truncate": False,
                          "packing": "separate audio items; target is last, not concatenated"},
                "implementation": {name: file_hash(ROOT / name) for name in (
                    "scripts/evaluation/context_pilot.py", "scripts/models/omni_context.py",
                    "scripts/common/dialogue_datasets.py", "scripts/models/audio_io.py",
                    "scripts/models/audio_llm.py", "scripts/models/qwen2_5_omni.py",
                    "scripts/llm_eval/parsing.py")}}
    metadata = {}
    cases = []
    for case in selected:
        for record in case.values():
            path = Path(record["audio"])
            if path not in metadata:
                with wave.open(str(path), "rb") as audio:
                    actual = audio.getnframes() / audio.getframerate()
                if abs(actual - record["duration"]) > 0.02:
                    raise ValueError(f"Duration changed for {path}")
                metadata[path] = {"audio_sha256": file_hash(path), "audio_bytes": path.stat().st_size}
        exported = {k: public_utterance(v, metadata[Path(v["audio"])]) for k, v in case.items()}
        exported["gold"] = case["target"]["gold"]  # analysis only; build_messages never serializes this field
        cases.append(exported)
    payload = {"protocol": protocol, "cases": cases, "selection_audit": selection_audit}
    prepared_hash = digest(payload)
    folder = ROOT / config["out"] / f"{config['dataset']}_{config['split']}_{len(cases)}_{prepared_hash[:12]}"
    manifest = payload | {"prepared_hash": prepared_hash, "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                          "status": "prepared_only_not_evaluated"}
    write_json(folder / "manifest.json", manifest)
    write_json(folder / "config.json", config)
    rows = [{"id": c["target"]["id"], "dialogue": c["target"]["dialogue"], "speaker": c["target"]["speaker"],
             "split": c["target"]["split"], "gold": c["gold"], "target_audio": c["target"]["audio"],
             "target_seconds": c["target"]["duration"], "history_id": c["history"]["id"],
             "history_audio": c["history"]["audio"], "history_seconds": c["history"]["duration"],
             "unrelated_id": c["unrelated"]["id"], "unrelated_dialogue": c["unrelated"]["dialogue"],
             "unrelated_audio": c["unrelated"]["audio"], "unrelated_seconds": c["unrelated"]["duration"],
             "history_transcript": c["history"]["transcript"]} for c in cases]
    write_csv(folder / "targets.csv", rows, list(rows[0]))
    with (folder / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            for condition in CONDITIONS:
                stream.write(json.dumps({"id": case["target"]["id"], "condition": condition,
                    "messages": build_messages(case, condition, protocol)}, ensure_ascii=False) + "\n")
    write_json(folder / "prompt_preview.json", {c: build_messages(cases[0], c, protocol) for c in CONDITIONS})
    print(f"PREPARED ONLY: {len(cases)} train targets × 4 = {len(cases) * 4} requests; no model loaded")
    print(json.dumps(selection_audit, ensure_ascii=False))
    print(f"Manifest: {folder / 'manifest.json'}")
    return folder / "manifest.json"


def load_manifest(path, check_audio=False, check_implementation=False):
    manifest = json.loads(Path(path).read_text())
    payload = {k: manifest[k] for k in ("protocol", "cases", "selection_audit")}
    if digest(payload) != manifest["prepared_hash"]:
        raise ValueError("Prepared manifest was changed; prepare a new version instead")
    protocol = manifest["protocol"]
    if check_implementation:
        for name, expected in protocol["implementation"].items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Implementation changed: {name}; prepare again before inference")
        model_info = json.loads((ROOT / "models/manifest.json").read_text())["models"][protocol["model"]]
        if digest(model_info) != protocol["model_manifest_hash"]:
            raise ValueError("Model manifest changed; prepare again before inference")
        for name, expected in protocol.get("processor_files_sha256", {}).items():
            if file_hash(ROOT / name) != expected:
                raise ValueError(f"Model/processor configuration changed: {name}; prepare again")
    if check_audio:
        seen = set()
        for case in manifest["cases"]:
            for role in ("target", "history", "unrelated"):
                record = case[role]
                if record["audio"] in seen:
                    continue
                path = ROOT / record["audio"]
                if path.stat().st_size != record["audio_bytes"] or file_hash(path) != record["audio_sha256"]:
                    raise ValueError(f"Prepared audio changed: {path}")
                seen.add(record["audio"])
    return manifest


def read_responses(path):
    """Read-only journal recovery: tolerate only a torn final line; run repairs it before appending."""
    path = Path(path)
    if not path.exists():
        return [], b""
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    rows, valid = [], b""
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise ValueError(f"Corrupt response journal at line {i + 1}: {path}")
            break
        rows.append(row)
        valid += line.rstrip(b"\r\n") + b"\n"
    return rows, valid


def scores(rows, labels):
    support, predicted, tp = Counter(), Counter(), Counter()
    for row in rows:
        support[row["gold"]] += 1
        predicted[row["prediction"]] += 1
        if row["gold"] == row["prediction"]:
            tp[row["gold"]] += 1
    per_class = {}
    for label in labels:
        precision = tp[label] / predicted[label] if predicted[label] else 0.0
        recall = tp[label] / support[label] if support[label] else 0.0
        per_class[label] = {"support": support[label], "precision": precision, "recall": recall,
                            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    n = len(rows)
    return {"clips": n, "accuracy": sum(tp.values()) / n if n else None,
            "unweighted_accuracy": sum(c["recall"] for c in per_class.values()) / len(labels) if n else None,
            "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels) if n else None,
            "neutral_prediction_rate": predicted["neutral"] / n if n else None,
            "invalid": predicted["invalid"], "per_class": per_class,
            "confusion": {lab: {pred: sum(r["gold"] == lab and r["prediction"] == pred for r in rows)
                               for pred in [*labels, "invalid"]} for lab in labels}}


def analyze(manifest_path, run_dir):
    manifest = load_manifest(manifest_path)
    run_dir = Path(run_dir)
    execution = json.loads((run_dir / "execution.json").read_text())
    if execution["prepared_hash"] != manifest["prepared_hash"]:
        raise ValueError("The response directory belongs to a different prepared pilot")
    if digest({k: v for k, v in execution.items() if k != "execution_hash"}) != execution["execution_hash"]:
        raise ValueError("Execution identity was changed")
    raw, _ = read_responses(run_dir / "responses.jsonl")
    cases = {c["target"]["id"]: c for c in manifest["cases"]}
    latest = {}
    for row in raw:
        if row.get("execution_hash") != execution["execution_hash"]:
            raise ValueError("Mixed execution identities in response journal")
        if row["id"] not in cases or row["condition"] not in CONDITIONS:
            raise ValueError("Response refers to a target/condition outside the frozen sample")
        messages = build_messages(cases[row["id"]], row["condition"], manifest["protocol"])
        expected = digest([execution["execution_hash"], row["id"], row["condition"], messages])
        if row.get("request_hash") != expected:
            raise ValueError("Response request identity does not match the frozen inputs")
        if "error" not in row:
            if row["prediction"] not in [*manifest["protocol"]["labels"], "invalid"]:
                raise ValueError("Response prediction is not a candidate or invalid")
            latest[row["id"], row["condition"]] = row
    rows = []
    for (key, condition), answer in latest.items():
        case = cases[key]
        rows.append({"id": key, "condition": condition, "dialogue": case["target"]["dialogue"],
                     "speaker": case["target"]["speaker"], "split": "train", "gold": case["gold"],
                     "prediction": answer["prediction"], "correct": answer["prediction"] == case["gold"],
                     "target_audio": case["target"]["audio"], "history_id": case["history"]["id"],
                     "unrelated_id": case["unrelated"]["id"], "answers_json": json.dumps(answer["answers"], ensure_ascii=False),
                     "seconds": answer["seconds"], "request_hash": answer["request_hash"],
                     **{f"token_p_{lab}": answer.get("label_probs", {}).get(lab, "") for lab in manifest["protocol"]["labels"]},
                     "token_mass": answer.get("label_mass", "")})
    paired_ids = [key for key in cases if all((key, c) in latest for c in CONDITIONS)]
    labels = manifest["protocol"]["labels"]
    metrics = {c: scores([r for r in rows if r["condition"] == c and r["id"] in paired_ids], labels)
               for c in CONDITIONS}
    comparisons = {}
    for c in CONDITIONS[1:]:
        transitions = [{"id": key, "condition": c, "gold": cases[key]["gold"],
                        "h0": latest[key, "H0"]["prediction"], "prediction": latest[key, c]["prediction"]}
                       for key in paired_ids]
        comparisons[c] = {"changed": sum(r["h0"] != r["prediction"] for r in transitions),
                          "wrong_to_correct": sum(r["h0"] != r["gold"] == r["prediction"] for r in transitions),
                          "correct_to_wrong": sum(r["h0"] == r["gold"] != r["prediction"] for r in transitions),
                          **{f"delta_{metric}": metrics[c][metric] - metrics["H0"][metric] if paired_ids else None
                             for metric in ("accuracy", "unweighted_accuracy", "macro_f1")}}
    summary = {"prepared_hash": manifest["prepared_hash"], "execution": execution,
               "planned_targets": len(cases), "completed_per_condition": dict(Counter(r["condition"] for r in rows)),
               "paired_complete_targets": len(paired_ids), "complete": len(paired_ids) == len(cases),
               "metrics_on_common_targets": metrics, "versus_h0_on_common_targets": comparisons,
               "interpretation": "Exploratory class-balanced train pilot, not held-out natural-distribution performance. "
                                 "H1/H2 are not a pure acoustic/semantic decomposition; gold is fixed across conditions. "
                                 "First-token scores are proxies, not calibrated class probabilities."}
    if rows:
        write_csv(run_dir / "predictions.csv", rows, list(rows[0]))
    write_json(run_dir / "summary.json", summary)
    lines = ["# EmotionTalk H0–H3 pilot", "", f"Paired targets: {len(paired_ids)} / {len(cases)}; "
             f"complete: {summary['complete']}", "", "Metrics below use the identical completed target intersection.", "",
             "| Condition | Accuracy | UAR | Macro F1 | Neutral predictions | Invalid |",
             "|---|---:|---:|---:|---:|---:|"]
    pct = lambda v: f"{100 * v:.2f}%" if v is not None else "—"
    for c, metric in metrics.items():
        lines.append(f"| {c} | " + " | ".join(pct(metric[k]) for k in
                    ("accuracy", "unweighted_accuracy", "macro_f1", "neutral_prediction_rate"))
                    + f" | {metric['invalid']} |")
    lines += ["", summary["interpretation"], "", "This report does not establish statistical significance or human agreement."]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Analysis: {run_dir / 'summary.json'}; paired targets {len(paired_ids)}/{len(cases)}")
    return summary


def run(manifest_path, device=None):
    # This is the only boundary that imports model libraries or loads weights.
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest = load_manifest(manifest_path, check_audio=True, check_implementation=True)
    sys.path.insert(0, str(ROOT / "scripts/models"))
    from omni_context import OmniContextPredictor
    protocol = manifest["protocol"]
    device = device or protocol["config"]["device"]
    import torch
    from audio_llm import dtype_for
    actual_device = device
    if actual_device == "auto":
        actual_device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if actual_device == "cpu":
            raise RuntimeError("No CUDA/MPS device; pass --device cpu explicitly if intended")
    packages = ("torch", "transformers", "torchaudio", "numpy", "soundfile", "tokenizers", "safetensors", "accelerate")
    versions = {package: importlib.metadata.version(package) for package in packages}
    execution = {"prepared_hash": manifest["prepared_hash"], "versions": versions, "requested_device": device,
                 "device": actual_device, "dtype": str(dtype_for(actual_device)).replace("torch.", ""),
                 "python": sys.version.split()[0]}
    execution["execution_hash"] = digest(execution)
    run_dir = Path(manifest_path).parent / "runs" / execution["execution_hash"][:12]
    write_json(run_dir / "execution.json", execution)
    journal = run_dir / "responses.jsonl"
    old, valid_bytes = read_responses(journal)
    if journal.exists() and journal.read_bytes() != valid_bytes:
        journal.with_suffix(".interrupted-tail.jsonl").write_bytes(journal.read_bytes())
        journal.write_bytes(valid_bytes)
    cached = {(r["id"], r["condition"]): r for r in old
              if r.get("execution_hash") == execution["execution_hash"] and "error" not in r}
    requests = []
    for i, case in enumerate(manifest["cases"]):
        order = CONDITIONS[i % 4:] + CONDITIONS[:i % 4]
        for c in order:
            messages = build_messages(case, c, protocol)
            request_hash = digest([execution["execution_hash"], case["target"]["id"], c, messages])
            previous = cached.get((case["target"]["id"], c))
            if previous and previous.get("request_hash") == request_hash:
                continue
            requests.append((case, c, messages, request_hash))
    print(f"Run folder: {run_dir}; remaining requests: {len(requests)} / {len(manifest['cases']) * 4}", flush=True)
    if not requests:
        return analyze(manifest_path, run_dir)
    predictor = OmniContextPredictor(ROOT / protocol["model_dir"], actual_device, protocol)
    for i, (case, condition, messages, request_hash) in enumerate(requests, 1):
        record = {"id": case["target"]["id"], "condition": condition,
                  "request_hash": request_hash, "execution_hash": execution["execution_hash"]}
        try:
            record.update(predictor(messages))
        except Exception as exc:
            record.update(error=type(exc).__name__ + ": " + str(exc))
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            raise  # hardware/model errors stop; never turn a failed request into a scored wrong answer
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        print(f"[{i}/{len(requests)}] {case['target']['id']} {condition}: {record['prediction']}", flush=True)
    return analyze(manifest_path, run_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    prep = commands.add_parser("prepare", help="freeze targets/prompts/inputs without importing model libraries")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute = commands.add_parser("run", help="explicitly load Omni and run/resume the prepared pilot")
    execute.add_argument("--manifest", type=Path, required=True)
    execute.add_argument("--device", choices=["auto", "mps", "cuda:0", "cpu"])
    analysis = commands.add_parser("analyze", help="rebuild CSV/JSON/Markdown from saved responses only")
    analysis.add_argument("--manifest", type=Path, required=True)
    analysis.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.manifest, args.device)
    elif args.command == "analyze":
        analyze(args.manifest, args.run_dir)
    else:
        prepare(getattr(args, "config", DEFAULT_CONFIG))


if __name__ == "__main__":
    main()

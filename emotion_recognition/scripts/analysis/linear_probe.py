"""CPU-only linear probes of saved audio-LLM features; export paper results, never feature vectors.

Run from emotion_recognition/: python scripts/analysis/linear_probe.py --dry-run
See scripts/analysis/README.md. No model weights, torch, audio decoding or network are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from probe_core import fit_linear_probe, make_group_splits, score_predictions
from probe_metadata import infer_groups

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "llm_eval"))
from prompts import configured_protocol

SPLITS = ("train", "validation", "test")
DEFAULTS = {
    "models": ["gemma4_e4b", "qwen2_5_omni_7b", "qwen2_audio_7b"],
    "datasets": "all", "seeds": [42], "layers": "all", "c_values": [.01, .1, 1., 10.],
    "class_weight": "balanced", "max_iter": 1000, "num_threads": 4,
    "train_validation_test": [.6, .2, .2], "group_counts": {}, "split_manifests": {},
    "require_current_protocol": True, "require_audio_pooling": True,
    "export_model_parameters": True, "hash_features": True,
    "allow_test_only_diagnostics": False, "prediction_batch_size": 1024,
    "require_convergence": True,
    "output_dir": "outputs/linear_probe",
}


def canonical_path(value):
    """Require the exact portable path form used by evaluation; aliases must not hide duplicate examples."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"invalid audio_path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or ":" in path.parts[0]:
        raise ValueError(f"audio_path must be canonical and project-relative: {value!r}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_feature_store(folder, expected_protocol, model_id, require_audio_pooling=True):
    """Strict, read-only (vectors memmap, index rows, metadata).

    model_id is the registry key; expected_protocol.model_id is the HF repository. Unlike the inference writer's
    recovery helper, this reader never truncates files or silently ignores incomplete/duplicate rows.
    """
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if meta.get("model") != model_id or meta.get("hf_repo") != expected_protocol.get("model_id"):
        raise ValueError("feature model identity differs from the prediction protocol")
    if (meta.get("prompts") != expected_protocol.get("prompt_hash")
            or meta.get("label_scope") != expected_protocol.get("label_scope")
            or meta.get("labels") != expected_protocol.get("labels")):
        raise ValueError("feature prompt hash, label scope or candidate labels differ from predictions")
    shape, layers, num_layers = meta.get("row_shape"), meta.get("layers"), meta.get("num_layers")
    if (meta.get("dtype") != "float16" or not isinstance(shape, list) or len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)):
        raise ValueError("features require dtype=float16 and positive [layers, hidden_size] row_shape")
    if (type(num_layers) is not int or num_layers < 1 or not isinstance(layers, list) or len(layers) != shape[0]
            or any(type(layer) is not int or not 1 <= layer <= num_layers for layer in layers)
            or len(set(layers)) != len(layers)):
        raise ValueError("feature layers must be unique, in range and match row_shape")
    rows, seen = [], set()
    for line in (folder / "index.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        path = canonical_path(row.get("audio_path"))
        if type(row.get("row")) is not int or row["row"] != len(rows) or path in seen:
            raise ValueError("feature index must have consecutive rows and unique audio paths")
        if row.get("pooling") not in ("audio", "prompt") or type(row.get("tokens")) is not int or row["tokens"] < 1:
            raise ValueError("feature index has invalid pooling/tokens")
        if require_audio_pooling and row["pooling"] != "audio":
            raise ValueError("pooling=prompt fallback found; cannot treat this run as an audio-only representation probe")
        seen.add(path)
        rows.append(row)
    if not rows:
        raise ValueError("feature index is empty")
    data = folder / "hidden.f16"
    expected_bytes = len(rows) * int(np.prod(shape)) * 2
    if data.stat().st_size != expected_bytes:
        raise ValueError("feature byte count does not match the index; incomplete or extra feature rows")
    vectors = np.memmap(data, dtype=np.float16, mode="r", shape=(len(rows), *shape))
    for start in range(0, len(rows), 1024):
        if not np.isfinite(vectors[start:start + 1024]).all():
            raise ValueError("features contain non-finite values")
    return vectors, rows, meta


def read_predictions(root, model, dataset, entry, require_current=True):
    path = root / "outputs" / model / f"{dataset}_predictions.csv"
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta.get("key") != dataset or meta.get("model") != model:
        raise ValueError("prediction metadata model/dataset does not match its file location")
    protocol = meta.get("inference_protocol")
    if not protocol or not protocol.get("prompt_hash") or protocol.get("model_id") != entry["hf_repo"]:
        raise ValueError("missing/incompatible inference_protocol; legacy evaluation is not a formal probe input")
    choices = protocol.get("labels")
    if (protocol.get("label_scope") != "dataset" or not isinstance(choices, list)
            or any(not isinstance(label, str) for label in choices) or len(set(choices)) != len(choices)
            or set(choices) != set(meta["classes"])):
        raise ValueError("inference_protocol must be dataset-scoped with exactly the dataset's candidate labels")
    if meta.get("total_clips", meta["clips"]) != meta["clips"]:
        raise ValueError("sampled predictions are not a full formal probe input (clips != total_clips)")
    if require_current and protocol != configured_protocol(entry, meta["classes"], root=root, config=root / "config/llm.yaml"):
        raise ValueError("prediction inference_protocol differs from the current model/prompt/candidates")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len(rows) != meta["clips"]:
        raise ValueError("prediction CSV row count does not match completed metadata")
    paths = [canonical_path(row["audio_path"]) for row in rows]
    if len(set(paths)) != len(paths) or any(not row.get("acted_emotion") for row in rows):
        raise ValueError("duplicate prediction audio paths or missing targets")
    return path, meta, {row["audio_path"]: row for row in rows}


def validate_splits(assignments, paths, labels, groups):
    if set(assignments) != set(paths) or set(assignments.values()) != set(SPLITS):
        raise ValueError("split manifest must assign every eligible clip exactly once to train/validation/test")
    ownership = {}
    for path in paths:
        group, split = groups[path]["group"], assignments[path]
        if group in ownership and ownership[group] != split:
            raise ValueError(f"group leakage across partitions: {group}")
        ownership[group] = split
    classes = set(labels.values())
    for split in SPLITS:
        if {labels[path] for path in paths if assignments[path] == split} != classes:
            raise ValueError(f"{split} does not cover every supervised class")


def split_assignments(config, root, dataset, paths, labels, groups, seed):
    supplied = config["split_manifests"].get(dataset)
    if supplied:
        manifest = root / supplied
        with manifest.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        records = [row for row in records if not row.get("seed") or int(row["seed"]) == seed]
        assignments = {}
        for row in records:
            path = canonical_path(row["audio_path"])
            if path not in labels:  # A full-corpus manifest can also list excluded composite targets.
                raise ValueError(f"manifest contains an unknown or excluded clip: {path}")
            if path in assignments or row.get("group") != groups[path]["group"]:
                raise ValueError("duplicate manifest path or group identity differs from audited dataset metadata")
            assignments[path] = row["split"]
        source = {"kind": "explicit_manifest", "path": str(manifest), "sha256": sha256(manifest)}
    else:
        counts = config["group_counts"].get(dataset)
        if counts and sum(counts) != len({groups[path]["group"] for path in paths}):
            raise ValueError("configured group_counts do not match this corpus; review the corpus/split protocol")
        assignments = make_group_splits(paths, [labels[path] for path in paths],
                                        [groups[path]["group"] for path in paths], seed,
                                        fractions=config["train_validation_test"], group_counts=counts)
        source = {"kind": "seeded_group_stratification", "group_counts": counts,
                  "fractions": config["train_validation_test"]}
    validate_splits(assignments, paths, labels, groups)
    return assignments, source


def environment(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    packages = {}
    for name in ("numpy", "scikit-learn", "scipy", "PyYAML", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "code_commit": git("rev-parse", "HEAD"), "working_tree_status": git("status", "--porcelain"),
            "device": "cpu", "created_utc": datetime.now(timezone.utc).isoformat()}


def probabilities(model, vectors, row_indices, layer_index, column_indices, batch_size):
    result = []
    for start in range(0, len(row_indices), batch_size):
        data = np.asarray(vectors[row_indices[start:start + batch_size], layer_index, :], dtype=np.float32)
        result.append(model.predict_proba(data)[:, column_indices])
    return np.concatenate(result)


def export_model(path, fitted, classes):
    steps = fitted.named_steps
    scaler, classifier = steps["scaler"], steps["classifier"]
    write_json(path, {"classes": classes, "estimator_class_order": classifier.classes_.tolist(),
                      "coef": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
                      "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                      "note": "Derived linear parameters, not raw feature vectors; preprocessing fit on train only."})


def run(config, project_root=ROOT, output_dir=None, dry_run=False):
    """Fit CPU probes and return a JSON-compatible run summary. Never modify input outputs or features."""
    root = Path(project_root).resolve()
    config = DEFAULTS | config
    if (not config["models"] or not config["seeds"] or not config["c_values"]
            or config["num_threads"] < 1 or config["prediction_batch_size"] < 1):
        raise ValueError("models, seeds, c_values must be nonempty; threads and batch size must be positive")
    registry = yaml.safe_load((root / "config/models.yaml").read_text())
    models = list(dict.fromkeys(config["models"]))
    for model in models:
        if model not in registry or not registry[model].get("llm") or registry[model].get("api"):
            raise ValueError(f"{model}: select a local audio LLM with saved features")
    datasets = config["datasets"]
    if datasets == "all":
        datasets = sorted({path.name.removesuffix("_predictions.json") for model in models
                           for path in (root / "outputs" / model).glob("*_predictions.json")
                           if "_sample" not in path.name})
    if isinstance(datasets, str):
        datasets = [datasets]
    if not datasets:
        raise ValueError("no datasets found; finish the LLM evaluation or explicitly select datasets")
    seeds = list(dict.fromkeys(int(seed) for seed in config["seeds"]))
    destination = Path(output_dir).resolve() if output_dir else (
        root / config["output_dir"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).resolve()
    protected = [root / "features", root / "data", root / "models", root / "outputs"]
    if any(destination == folder or destination in folder.parents for folder in protected):
        raise ValueError("output directory would overlap an input root")
    if any(destination.is_relative_to(folder) for folder in protected[:3]):
        raise ValueError("probe exports cannot be placed inside features, data or model weights")
    if not dry_run and destination.exists() and any(destination.iterdir()):
        raise ValueError("output directory is not empty; choose a new run directory")
    summary = {"schema_version": 1, "status": "dry_run" if dry_run else "complete",
               "config": config, "environment": environment(root), "output_dir": str(destination),
               "results": [], "skipped": [], "failed": [], "ready": []}
    if not dry_run:
        write_json(destination / "run_config.json", summary | {"results": [], "ready": []})
        # Keep literal source settings as JSON strings; hashes alone cannot reconstruct a prompt.
        sources = {}
        for relative in ("config/evaluation.yaml", "config/llm.yaml", "config/models.yaml"):
            path = root / relative
            if path.is_file():
                sources[relative] = path.read_text(encoding="utf-8")
        llm = yaml.safe_load((root / "config/llm.yaml").read_text())
        prompt_path = root / llm["prompts"]
        sources[llm["prompts"]] = prompt_path.read_text(encoding="utf-8")
        write_json(destination / "source_configuration.json", sources)

    for dataset in datasets:
        if dataset == "tess":
            summary["skipped"].append({"dataset": dataset, "reason": "only two speakers: no speaker-disjoint train/validation/test split"})
            continue
        if dataset == "meld" and not (config["allow_test_only_diagnostics"] and config["split_manifests"].get(dataset)):
            summary["skipped"].append({"dataset": dataset, "reason": "available MELD features cover official test only; supervised training would consume the official test. Supply an explicit dialogue manifest and opt into diagnostic-only analysis, or collect separate train/dev features."})
            continue
        inputs = {}
        for model in models:
            try:
                inputs[model] = read_predictions(root, model, dataset, registry[model], config["require_current_protocol"])
            except (ValueError, OSError, KeyError) as error:
                summary["failed"].append({"model": model, "dataset": dataset, "reason": str(error)})
        if not inputs:
            continue
        first_path, first_meta, first_rows = next(iter(inputs.values()))
        composite = set(first_meta.get("accept", {}))
        paths = sorted(path for path, row in first_rows.items() if row["acted_emotion"] not in composite)
        excluded = [{"audio_path": path, "target": row["acted_emotion"], "reason": "composite_target_excluded",
                     "accepted_labels": "|".join(first_meta["accept"][row["acted_emotion"]])}
                    for path, row in sorted(first_rows.items()) if row["acted_emotion"] in composite]
        labels = {path: first_rows[path]["acted_emotion"] for path in paths}
        classes = [label for label in first_meta["classes"] if label in set(labels.values())]
        try:
            if len(classes) < 2 or set(labels.values()) != set(classes):
                raise ValueError("supervised targets must be at least two unambiguous candidate classes")
            groups = infer_groups(dataset, paths, root)
            for model, (_, meta, rows) in inputs.items():
                if (meta["classes"] != first_meta["classes"] or meta.get("accept", {}) != first_meta.get("accept", {})
                        or set(rows) != set(first_rows) or any(
                        rows[path]["acted_emotion"] != first_rows[path]["acted_emotion"] for path in rows)):
                    raise ValueError(f"{model}: clip membership/targets/candidates differ; cannot make paired splits")
            splits_by_seed = {seed: split_assignments(config, root, dataset, paths, labels, groups, seed) for seed in seeds}
        except (ValueError, OSError, KeyError) as error:
            summary["failed"].append({"dataset": dataset, "reason": str(error)})
            continue
        manifest_hashes = {}
        for seed, (assignments, source) in splits_by_seed.items():
            records = [{"audio_path": path, "y_true": labels[path], **groups[path], "split": assignments[path], "seed": seed}
                       for path in paths]
            digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            manifest_hashes[seed] = digest
            if not dry_run:
                manifest = destination / "splits" / dataset / f"seed_{seed}"
                write_csv(manifest.with_suffix(".csv"), records)
                write_json(manifest.with_suffix(".json"), {"sha256": digest, "source": source, "classes": classes,
                           "inference_candidates": first_meta["classes"], "clips": len(paths),
                           "counts": dict(Counter(assignments.values())), "excluded_composite_clips": len(excluded),
                           "evaluation_scope": "diagnostic_only_official_test_dialogues" if dataset == "meld" else "speaker_disjoint"})
                write_csv(manifest.parent / "excluded_targets.csv", excluded,
                          ["audio_path", "target", "reason", "accepted_labels"])
        for model, (prediction_path, prediction_meta, prediction_rows) in inputs.items():
            protocol = prediction_meta["inference_protocol"]
            folder = root / "features" / model / dataset / protocol["prompt_hash"]
            try:
                vectors, index, feature_meta = read_feature_store(folder, protocol, model, config["require_audio_pooling"])
                indices = {row["audio_path"]: row["row"] for row in index}
                missing, extra = set(prediction_rows) - set(indices), set(indices) - set(prediction_rows)
                coverage = {"prediction_clips": prediction_meta["clips"],
                            "dataset_total_clips": prediction_meta.get("total_clips", prediction_meta["clips"]),
                            "feature_rows": len(index), "eligible_clips": len(paths),
                            "excluded_composite_clips": len(excluded), "complete": not missing and not extra,
                            "missing_audio_paths": sorted(missing), "extra_audio_paths": sorted(extra),
                            "pooling_counts": dict(Counter(row["pooling"] for row in index))}
                if not dry_run:
                    write_json(destination / model / dataset / "coverage.json", coverage)
                if set(indices) != set(prediction_rows):
                    raise ValueError(f"feature coverage differs from predictions: {len(missing)} missing, {len(extra)} extra; no per-model resplitting or silent row drop")
                available = feature_meta["layers"]
                layers = available if config["layers"] == "all" else available[-1:] if config["layers"] == "last" else list(config["layers"])
                if not layers or len(set(layers)) != len(layers) or any(layer not in available for layer in layers):
                    raise ValueError(f"requested layers must be a nonempty subset of saved layers {available}")
                summary["ready"].append({"model": model, "dataset": dataset, "layers": layers, "clips": len(paths)})
                if dry_run:
                    del vectors
                    continue
                provenance = {"feature_meta": feature_meta, "feature_meta_sha256": sha256(folder / "meta.json"),
                              "feature_index_sha256": sha256(folder / "index.jsonl"),
                              "feature_bytes": (folder / "hidden.f16").stat().st_size,
                              "feature_sha256": sha256(folder / "hidden.f16") if config["hash_features"] else None,
                              "prediction_csv_sha256": sha256(prediction_path),
                              "prediction_json_sha256": sha256(prediction_path.with_suffix(".json")),
                              "inference_protocol": protocol, "coverage": coverage}
                ordered_indices = np.asarray([indices[path] for path in paths])
                y = np.asarray([labels[path] for path in paths])
                index_by_path = {row["audio_path"]: row for row in index}
                for seed, (assignments, _) in splits_by_seed.items():
                    partitions = np.asarray([assignments[path] for path in paths])
                    selected_dir = destination / model / dataset / f"seed_{seed}"
                    fitted_layers = []
                    with threadpool_limits(limits=config["num_threads"]):
                        for layer in layers:
                            layer_index = available.index(layer)
                            fitted = fit_linear_probe(
                                np.asarray(vectors[ordered_indices[partitions == "train"], layer_index, :], dtype=np.float32),
                                y[partitions == "train"].tolist(),
                                np.asarray(vectors[ordered_indices[partitions == "validation"], layer_index, :], dtype=np.float32),
                                y[partitions == "validation"].tolist(), classes, config["c_values"],
                                class_weight=config["class_weight"], max_iter=config["max_iter"], seed=seed)
                            if config["require_convergence"] and not fitted["converged"]:
                                raise ValueError(f"selected C did not converge at layer {layer}; increase max_iter before reporting formal results")
                            fitted_layers.append((layer, fitted))
                        # Choose layer solely by validation UAR, before inspecting test performance.
                        best_layer, best_fit = max(fitted_layers, key=lambda item: max(
                            row["validation_uar"] for row in item[1]["validation_search"] if row["C"] == item[1]["selected_C"]))
                        for layer, fitted in fitted_layers:
                            probs = probabilities(fitted["model"], vectors, ordered_indices, available.index(layer),
                                                  fitted["probability_column_indices"], config["prediction_batch_size"])
                            predictions = np.asarray(classes)[probs.argmax(axis=1)]
                            metrics = {split: score_predictions(y[partitions == split].tolist(), probs[partitions == split], classes)
                                       for split in SPLITS}
                            rows = [{"audio_path": path, "split": assignments[path], "group": groups[path]["group"],
                                     "group_kind": groups[path]["group_kind"], "y_true": labels[path],
                                     "prediction": str(predictions[i]), "correct": bool(predictions[i] == labels[path]),
                                     "baseline_prediction": prediction_rows[path].get("prediction", ""),
                                     "listener_vote": prediction_rows[path].get("listener_vote", ""),
                                     "pooling": index_by_path[path]["pooling"], "tokens": index_by_path[path]["tokens"],
                                     **{f"probe_p_{label}": float(probs[i, c]) for c, label in enumerate(classes)}}
                                    for i, path in enumerate(paths)]
                            layer_dir = selected_dir / f"layer_{layer}"
                            write_csv(layer_dir / "predictions.csv", rows)
                            write_json(layer_dir / "metrics.json", metrics)
                            write_json(layer_dir / "config.json", {"model": model, "dataset": dataset, "seed": seed,
                                       "layer": layer, "classes": classes, "selected_C": fitted["selected_C"],
                                       "validation_search": fitted["validation_search"], "converged": fitted["converged"],
                                       "convergence_warnings": fitted["convergence_warnings"], "manifest_sha256": manifest_hashes[seed],
                                       "class_weight": config["class_weight"], "max_iter": config["max_iter"],
                                       "selection_metric": "validation_uar", "scaler_fit": "train_only",
                                       "training_fit": "train_only_no_train_validation_refit", "provenance": provenance})
                            if config["export_model_parameters"]:
                                export_model(layer_dir / "model_parameters.json", fitted["model"], classes)
                            record = {"model": model, "dataset": dataset, "seed": seed, "layer": layer,
                                      "C": fitted["selected_C"], "selected": layer == best_layer,
                                      "validation_uar": metrics["validation"]["uar"], "test_accuracy": metrics["test"]["accuracy"],
                                      "test_uar": metrics["test"]["uar"], "test_macro_f1": metrics["test"]["macro_f1"],
                                      "test_log_loss": metrics["test"]["log_loss"], "converged": fitted["converged"],
                                      "test_clips": metrics["test"]["n_samples"], "manifest_sha256": manifest_hashes[seed],
                                      "evaluation_scope": "diagnostic_only_official_test_dialogues" if dataset == "meld" else "speaker_disjoint"}
                            summary["results"].append(record)
                            if layer == best_layer:
                                write_csv(selected_dir / "selected_predictions.csv", rows)
                                write_json(selected_dir / "selected.json", record | {"metrics": metrics})
                    print(f"{model}/{dataset}/seed={seed}: selected layer {best_layer}, C={best_fit['selected_C']}", flush=True)
                    write_json(destination / "summary.json", summary)
                del vectors
            except (ValueError, OSError, KeyError) as error:
                summary["failed"].append({"model": model, "dataset": dataset, "reason": str(error)})
    if summary["failed"]:
        summary["status"] = "partial" if summary["results"] else "failed"
    if not dry_run:
        write_json(destination / "summary.json", summary)
        write_csv(destination / "summary.csv", summary["results"], ["model", "dataset", "seed", "layer", "C", "selected",
                  "validation_uar", "test_accuracy", "test_uar", "test_macro_f1", "test_log_loss", "converged",
                  "test_clips", "manifest_sha256", "evaluation_scope"])
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/linear_probe.yaml")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="audit inputs and splits without fitting or writing exports")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    for key in ("models", "datasets", "seeds"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    result = run(config, output_dir=args.output_dir, dry_run=args.dry_run)
    print(json.dumps({key: value for key, value in result.items() if key not in ("config", "environment", "results")},
                     ensure_ascii=False, indent=2))
    print(f"Completed layer runs: {len(result['results'])}")
    if result["failed"] or not args.dry_run and not result["results"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

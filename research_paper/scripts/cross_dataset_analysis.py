"""Language, conversational source and duration diagnostics of saved SER predictions.

Run from the repository root: python research_paper/scripts/cross_dataset_analysis.py
No inference, API requests or prediction edits. Audio headers are cached separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import spearmanr
import yaml

import consensus
from duration_inventory import build_inventory, write_inventory
from duration_stats import fixed_effect_slope

CONVERSATIONAL = frozenset({"meld", "emotiontalk", "dailytalk", "multidialog"})
BIN_EDGES = np.array([0., 1., 2., 3., 5., 10., 15., 30., np.inf])
BIN_LABELS = ("0–1", "1–2", "2–3", "3–5", "5–10", "10–15", "15–30", "≥30")
METRICS = ("accuracy", "uar", "macro_f1", "weighted_f1")
KNOWN_CAPS = {"emotion2vec_plus_large": None, "meralion_ser_v1": 30, "melati_emotion": 30,
              "vox_profile_whisper": 15, "vox_profile_wavlm": 15, "qwen2_5_omni_7b": 300,
              "qwen2_audio_7b": 30, "gemma4_e4b": 30}
sys.path.insert(0, str(consensus.PROJECT / "scripts/common"))
from training_provenance import training_metadata


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No analysis rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_inventory(project, selected, output, *, refresh=False):
    """Reuse only the exact source cohort, metadata and audio-header snapshot."""
    path, meta = output / "audio_inventory.csv", output / "audio_inventory_manifest.json"
    if refresh or not path.exists() or not meta.exists():
        rows, manifest = build_inventory(project, selected)
        write_inventory(rows, manifest, output)
        return rows
    manifest = json.loads(meta.read_text())
    if consensus.sha256(path) != manifest["inventory_csv_sha256"]:
        raise ValueError("Audio inventory CSV changed; use --refresh-inventory")
    current = {s["path"]: s["sha256"] for runs in selected.values() for run in runs.values() for s in run.sources}
    recorded = {s["path"]: s["sha256"] for s in manifest["prediction_sources"]}
    if current != recorded:
        raise ValueError("Prediction selection changed; use --refresh-inventory")
    for source in manifest["grouping_sources"]:
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise ValueError("Grouping metadata changed; use --refresh-inventory")
    implementation = {"duration_inventory.py": Path(__file__).with_name("duration_inventory.py"),
                      "probe_metadata.py": project / "scripts/analysis/probe_metadata.py"}
    for name, digest in manifest["implementation"].items():
        if consensus.sha256(implementation[name]) != digest:
            raise ValueError("Inventory implementation changed; use --refresh-inventory")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("sampling_rate", "frames", "channels", "bytes", "mtime_ns"):
            row[key] = int(row[key])
        row["duration_seconds"] = float(row["duration_seconds"])
        for key in ("empty_audio", "known_silent_recording"):
            row[key] = row[key] == "True"
        current_stat = (project / row["audio_path"]).stat()
        if (current_stat.st_size, current_stat.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise ValueError(f"Audio changed: {row['audio_path']}; use --refresh-inventory")
    return rows


def score_run(run, paths):
    """Original benchmark metrics, including composite labels and invalid errors."""
    classes = run.classes
    labels = run.metadata.get("labels", classes + list(run.accept))
    gold = np.asarray([run.rows[p]["gold"] for p in paths], dtype=object)
    predictions = [run.rows[p]["prediction"] for p in paths]
    correct = np.asarray([p in run.accept.get(g, [g]) for g, p in zip(gold, predictions)], dtype=float)
    invalid = np.asarray([p is None for p in predictions], dtype=bool)
    uar = np.mean([correct[gold == label].mean() if np.any(gold == label) else 0. for label in labels])
    lookup = {label: i for i, label in enumerate(classes)}
    pi = np.asarray([lookup[p] if p is not None else len(classes) for p in predictions])
    truth = np.asarray([lookup[p] if ok else lookup[run.accept.get(g, [g])[0]]
                        for g, p, ok in zip(gold, predictions, correct)])
    matrix = np.zeros((len(classes), len(classes) + 1), dtype=int)
    np.add.at(matrix, (truth, pi), 1)
    tp = matrix.diagonal()
    support, predicted = matrix.sum(1), matrix[:, :len(classes)].sum(0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(classes)), where=support + predicted != 0)
    scored = [i for i, label in enumerate(classes) if label in labels or (pi == i).any()]
    metrics = {"accuracy": float(correct.mean()), "uar": float(uar), "macro_f1": float(f1[scored].mean()),
               "weighted_f1": float(np.average(f1, weights=support))}
    for name, key in (("accuracy", "accuracy"), ("uar", "unweighted_accuracy"),
                      ("macro_f1", "macro_f1"), ("weighted_f1", "weighted_f1")):
        claimed = run.metadata["metrics"].get(key)
        if claimed is not None and abs(claimed - metrics[name]) > 1e-6:
            raise ValueError(f"{run.model}/{run.dataset}: recomputed {name} disagrees with saved metadata")
    return metrics, gold, correct, invalid, predictions


def constant_baselines(run, paths):
    values = {label: sum(label in run.accept.get(run.rows[p]["gold"], [run.rows[p]["gold"]]) for p in paths)
              / len(paths) for label in run.classes}
    best = max(values, key=values.get)
    return {"best_constant_label": best, "best_constant_accuracy": values[best],
            "neutral_accuracy": values.get("neutral", 0.)}


def balanced_weights(dataset, gold):
    """Each dataset has equal total weight; within it, each original gold row does too."""
    counts = Counter(zip(dataset, gold))
    classes = defaultdict(set)
    for d, g in counts:
        classes[d].add(g)
    return np.asarray([1 / (len(classes[d]) * counts[d, g]) for d, g in zip(dataset, gold)])


def cluster_identity(row):
    # EmotionTalk's 84 dialogues come from only two source groups, not 84 independent people.
    if row["dataset"] == "emotiontalk" and row["source_group_id"]:
        return "emotiontalk:source:" + row["source_group_id"]
    return row["group"]


def listener_metrics(run, paths):
    with run.path.with_suffix(".csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = {r["audio_path"]: r for r in csv.DictReader(stream)}
    votes = [(rows[p].get("listener_vote") or "").split(" or ") for p in paths]
    chosen = [(p, v[0]) for p, v in zip(paths, votes) if len(v) == 1 and v[0]]
    if not chosen:
        return {}
    gold = np.asarray([v for _, v in chosen], dtype=object)
    correct = np.asarray([run.rows[p]["prediction"] == v for p, v in chosen])
    return {"listener_n": len(chosen), "listener_accuracy": float(correct.mean()),
            "listener_uar": float(np.mean([correct[gold == g].mean() for g in sorted(set(gold))])),
            "listener_neutral_baseline": float(np.mean(gold == "neutral"))}


def analyse(selected, registry, inventory):
    indexed = {(r["dataset"], r["audio_path"]): r for r in inventory}
    dataset_rows, model_rows, bin_rows, effects, dataset_effects, correlations = [], [], [], [], [], []
    arrays = defaultdict(list)
    for dataset in consensus.DATASETS:
        entries = [r for r in inventory if r["dataset"] == dataset]
        reference = next(runs[dataset] for runs in selected.values() if dataset in runs)
        durations = np.asarray([r["duration_seconds"] for r in entries])
        language = reference.metadata.get("language", "")
        if "English" in language or language in ("British", "American"):
            language = "English"
        dataset_rows.append({"dataset": dataset, "name": consensus.NAMES[dataset], "language": language,
            "conversation": dataset in CONVERSATIONAL, "n": len(entries), "mean_seconds": float(durations.mean()),
            **{f"q{q}_seconds": float(np.percentile(durations, q)) for q in (5, 25, 75, 95)},
            "median_seconds": float(np.median(durations)), "max_seconds": float(durations.max()),
            "pct_under1": float(np.mean(durations < 1)), "pct_under3": float(np.mean(durations < 3)),
            "pct_over10": float(np.mean(durations > 10)), "pct_over15": float(np.mean(durations > 15)),
            "pct_over30": float(np.mean(durations > 30)), "groups": len({r["group"] for r in entries}),
            **constant_baselines(reference, [r["audio_path"] for r in entries])})
    dataset_index = {r["dataset"]: r for r in dataset_rows}
    for model in registry:
        for dataset in consensus.DATASETS:
            run = selected.get(model, {}).get(dataset)
            if run is None:
                continue
            paths = sorted(run.rows)
            entries = [indexed[dataset, p] for p in paths]
            if any(r["gold"] != run.rows[p]["gold"] for r, p in zip(entries, paths)):
                raise ValueError("Inventory ground truth differs from predictions")
            metrics, gold, correct, invalid, predictions = score_run(run, paths)
            durations = np.asarray([r["duration_seconds"] for r in entries])
            ids = np.searchsorted(BIN_EDGES[1:], durations, side="right")
            provenance = training_metadata(dataset, registry[model])
            cap = KNOWN_CAPS.get(model)
            row = {"dataset": dataset, "model": model, "n": len(paths), **metrics,
                   "invalid_count": int(invalid.sum()), "invalid_rate": float(invalid.mean()),
                   "valid_only_accuracy": float(correct[~invalid].mean()) if (~invalid).any() else None,
                   "neutral_pred_rate": predictions.count("neutral") / len(paths),
                   "angry_pred_rate": predictions.count("angry") / len(paths),
                   "training_status": provenance["training_status"],
                   "known_cap_seconds": cap, "cap_established": model in KNOWN_CAPS,
                   "beyond_cap_count": int(np.sum(durations > cap)) if cap else 0 if model in KNOWN_CAPS else None,
                   **constant_baselines(run, paths), **listener_metrics(run, paths)}
            model_rows.append(row)
            for b, label in enumerate(BIN_LABELS):
                use = ids == b
                covered = set(gold[use])
                all_classes = set(run.metadata.get("labels", run.classes + list(run.accept)))
                bin_rows.append({"dataset": dataset, "model": model, "bin_id": b, "bin_label": label,
                    "lower_seconds": float(BIN_EDGES[b]),
                    "upper_seconds": None if b == 7 else float(BIN_EDGES[b + 1]), "n": int(use.sum()),
                    "correct": int(correct[use].sum()), "invalid": int(invalid[use].sum()),
                    "accuracy": float(correct[use].mean()) if use.any() else None,
                    "invalid_rate": float(invalid[use].mean()) if use.any() else None,
                    "gold_classes_present": len(covered), "gold_classes_total": len(all_classes),
                    "uar": float(np.mean([correct[use & (gold == g)].mean() for g in sorted(all_classes)]))
                           if all_classes <= covered else None})
            positive = durations > 0
            if not np.all(positive):
                print(f"{model}/{dataset}: excluding {np.sum(~positive)} zero-length headers from log-duration fits")
            duration, y, truth = durations[positive], correct[positive], gold[positive]
            chosen_entries = [e for e, keep in zip(entries, positive) if keep]
            groups = [cluster_identity(e) for e in chosen_entries]
            result = fixed_effect_slope(duration, y, truth, groups, balanced_weights([dataset] * len(y), truth))
            if result["nclusters"] < 20:
                result |= {"ci_lower_pp": None, "ci_upper_pp": None, "stderr_pp": None, "pvalue": None,
                           "note": result["note"] + " Per-dataset interval suppressed: fewer than 20 source clusters."}
            dataset_effects.append({"dataset": dataset, "model": model, **result})
            arrays[model].extend((e, float(ok)) for e, ok in zip(chosen_entries, y))
    group_rows = []
    full_models = [m for m in registry if set(selected.get(m, {})) == set(consensus.DATASETS)
                   and all(len(r.rows) == dataset_index[d]["n"] for d, r in selected[m].items())]
    for model in full_models:
        rows = [r for r in model_rows if r["model"] == model]
        for grouping, levels in (("context", ("conversational", "non_conversational")),
                                 ("language", ("English", "non_English"))):
            for level in levels:
                chosen = [r for r in rows if ((r["dataset"] in CONVERSATIONAL) == (level == "conversational")
                          if grouping == "context" else
                          (dataset_index[r["dataset"]]["language"] == "English") == (level == "English"))]
                group_rows.append({"model": model, "grouping": grouping, "level": level,
                    "n_datasets": len(chosen), "n_clips": sum(r["n"] for r in chosen),
                    **{metric: float(np.mean([r[metric] for r in chosen])) for metric in METRICS}})
    for model in registry:
        records = arrays.get(model)
        if not records:
            continue
        entries, correct = zip(*records)
        duration = np.asarray([e["duration_seconds"] for e in entries])
        correct = np.asarray(correct)
        datasets = np.asarray([e["dataset"] for e in entries], dtype=object)
        gold = np.asarray([e["gold"] for e in entries], dtype=object)
        strata = [f"{d}:{g}" for d, g in zip(datasets, gold)]
        groups = [cluster_identity(e) for e in entries]
        balanced = balanced_weights(datasets, gold)
        scope = "16 datasets" if model in full_models else ", ".join(consensus.NAMES[d] for d in dict.fromkeys(datasets)) + " only"
        for kind, clusters in (("dataset", datasets), ("speaker_dialogue", groups)):
            for fit in ("pooled", "within_dataset_gold", "balanced_within_dataset_gold"):
                fit_strata = ["all"] * len(correct) if fit == "pooled" else strata
                weights = balanced if fit == "balanced_within_dataset_gold" else None
                effects.append({"model": model, "scope": scope, "fit": fit, "cluster_kind": kind,
                                **fixed_effect_slope(duration, correct, fit_strata, clusters, weights)})
        use = (duration >= 1) & (duration <= 15)
        if use.any():
            effects.append({"model": model, "scope": scope, "fit": "balanced_within_dataset_gold_1to15",
                "cluster_kind": "dataset", **fixed_effect_slope(duration[use], correct[use], np.asarray(strata)[use],
                datasets[use], balanced_weights(datasets[use], gold[use]))})
        rows = [r for r in model_rows if r["model"] == model]
        for metric in ("accuracy", "uar", "macro_f1"):
            x = [dataset_index[r["dataset"]]["median_seconds"] for r in rows]
            y = [r[metric] for r in rows]
            rho, pvalue = spearmanr(x, y) if len(rows) >= 3 else (None, None)
            correlations.append({"model": model, "scope": scope, "metric": metric, "n_datasets": len(rows),
                                 "spearman_rho": float(rho) if rho is not None else None,
                                 "pvalue": float(pvalue) if pvalue is not None else None})
    gap_effects = []
    lookup = {model: {(e["dataset"], e["audio_path"]): y for e, y in records} for model, records in arrays.items()}
    for model, records in arrays.items():
        for reference in ("emotion2vec_plus_large", "meralion_ser_v1"):
            if model == reference:
                continue
            shared = [(e, y, lookup[reference][e["dataset"], e["audio_path"]]) for e, y in records
                      if (e["dataset"], e["audio_path"]) in lookup[reference]]
            ds = [e["dataset"] for e, _, _ in shared]
            gold = [e["gold"] for e, _, _ in shared]
            # Linear rescaling lets the same tested WLS routine fit a paired outcome in [-1, 1].
            result = fixed_effect_slope([e["duration_seconds"] for e, _, _ in shared],
                [(a - b + 1) / 2 for _, a, b in shared], [f"{d}:{g}" for d, g in zip(ds, gold)],
                ds, balanced_weights(ds, gold))
            for key in ("slope", "slope_pp", "stderr_pp", "ci_lower_pp", "ci_upper_pp"):
                if result[key] is not None:
                    result[key] *= 2
            gap_effects.append({"model": model, "reference": reference,
                "scope": "16 datasets" if model in full_models else "MELD only", **result})
    return {"dataset_rows": dataset_rows, "model_rows": model_rows, "group_rows": group_rows,
            "bin_rows": bin_rows, "effects": effects, "dataset_effects": dataset_effects,
            "gap_effects": gap_effects, "correlations": correlations, "full_models": full_models}


def run(*, refresh_inventory=False, plots=True, data_only=False):
    project, paper = consensus.PROJECT, consensus.REPO / "research_paper"
    output = paper / "output"
    output.mkdir(exist_ok=True)
    registry = yaml.safe_load((project / "config/models.yaml").read_text())
    selected, selection = consensus.load_runs(project, registry, datasets=consensus.DATASETS, exclude=["esd"])
    inventory = load_inventory(project, selected, output, refresh=refresh_inventory)
    paths = [Path(__file__), Path(__file__).with_name("duration_stats.py"),
             Path(__file__).with_name("duration_inventory.py"), Path(consensus.__file__),
             project / "scripts/common/training_provenance.py", project / "config/models.yaml"]
    for filename in ("cross_dataset_plot.py", "cross_dataset_report.py"):
        path = Path(__file__).with_name(filename)
        if path.exists():
            paths.append(path)
    implementation = {p.relative_to(consensus.REPO).as_posix(): consensus.sha256(p) for p in paths}
    data = analyse(selected, registry, inventory)
    sources = {s["path"]: s for runs in selected.values() for run in runs.values() for s in run.sources}
    manifest = {"analysis": "cross_dataset_language_context_duration_v1",
        "generated_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M SGT"),
        "selected_run_count": sum(map(len, selected.values())), "model_order": [m for m in registry if m in selected],
        "model_names": {m: consensus.SHORT_NAMES.get(m, registry[m].get("name", m)) for m in selected},
        "full_models": data.pop("full_models"), "numpy": np.__version__, "selection": selection,
        "prediction_sources": sorted(sources.values(), key=lambda r: r["path"]),
        "inventory_sha256": consensus.sha256(output / "audio_inventory.csv"),
        "implementation": implementation,
        "parameters": {"duration_bins_seconds": [0, 1, 2, 3, 5, 10, 15, 30, "inf"],
            "display_min_bin_n": 50, "predictor": "log2(original clip duration)",
            "primary_fit": "dataset-by-original-gold fixed intercepts; equal dataset and equal gold-class total weights",
            "primary_interval": "dataset-cluster CR1, t(G-1); 16 clusters for complete models; no multiplicity adjustment",
            "sensitivity_interval": "speaker/dialogue clusters, EmotionTalk source groups; no new-speaker guarantee",
            "outcome": "saved emotion prediction correct under original accepted-label rules",
            "zero_header_policy": "retained in summaries/bins; excluded explicitly from log-duration fits",
            "missing_coverage": "Gemma MELD only; excluded from multi-dataset group means"},
        "interpretation": "Observational fixed-prediction analysis; no causal claims; raw duration includes silence"}
    data["manifest"] = manifest
    for name, digest in implementation.items():
        if consensus.sha256(consensus.REPO / name) != digest:
            raise RuntimeError("Analysis implementation changed during computation; rerun after edits finish")
    for source in sources.values():
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError("Source predictions changed during analysis")
    for name in ("dataset_rows", "model_rows", "group_rows", "bin_rows", "effects", "dataset_effects", "gap_effects", "correlations"):
        write_csv(output / f"cross_dataset_{name}.csv", data[name])
    if not data_only:
        if plots:
            from cross_dataset_plot import plot_analysis
            figures = plot_analysis(inventory, data["dataset_rows"], data["group_rows"], data["bin_rows"],
                                    data["effects"], data["dataset_effects"], paper / "analysis_report/figures")
            manifest["figures"] = [p.relative_to(paper).as_posix() for p in figures]
        from cross_dataset_report import render_report
        for language in ("zh", "en"):
            (paper / "analysis_report" / f"cross_dataset_{language}.md").write_text(render_report(data, language), encoding="utf-8")
    (output / "cross_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # The compact analysis snapshot also allows independent figure/report inspection without re-reading audio.
    (output / "cross_dataset_analysis.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Cross-dataset analysis: {manifest['selected_run_count']} runs; {len(inventory):,} headers; "
          f"{len(data['effects'])} duration fits. Reports: {paper / 'analysis_report'}")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-inventory", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    run(refresh_inventory=args.refresh_inventory, plots=not args.no_plots, data_only=args.data_only)

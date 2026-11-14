"""Emotion-classification accuracy versus native emotion-strength annotations.

No model/API calls. Unknown strengths and neutral references are not assigned
synthetic levels. Run: python research_paper/scripts/intensity_analysis.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import t
import yaml

import consensus
from cross_dataset_analysis import balanced_weights, cluster_identity, load_inventory, write_csv
from duration_stats import fixed_effect_linear
from intensity_inventory import SCALES, build_intensity_inventory, write_intensity_inventory


def load_strengths(project, inventory, output, *, refresh=False):
    path, manifest_path = output / "intensity_inventory.csv", output / "intensity_inventory_manifest.json"
    if refresh or not path.exists() or not manifest_path.exists():
        rows, manifest = build_intensity_inventory(project, inventory)
        write_intensity_inventory(rows, manifest, output)
        return rows, json.loads(manifest_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if consensus.sha256(path) != manifest["inventory_csv_sha256"]:
        raise ValueError("Intensity inventory changed; use --refresh-intensity")
    for source in manifest["metadata_sources"]:
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise ValueError("Intensity metadata changed; use --refresh-intensity")
    if consensus.sha256(Path(__file__).with_name("intensity_inventory.py")) != manifest["implementation_sha256"]:
        raise ValueError("Intensity parser changed; use --refresh-intensity")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if {(r["dataset"], r["audio_path"], r["gold"]) for r in rows} != {
            (r["dataset"], r["audio_path"], r["gold"]) for r in inventory}:
        raise ValueError("Strength inventory does not match the evaluation cohort")
    for row in rows:
        row["intensity_available"] = row["intensity_available"] == "True"
        row["intensity_applicable"] = row["intensity_applicable"] == "True"
        row["intensity_level"] = float(row["intensity_level"]) if row["intensity_level"] else None
    return rows, manifest


def paired_difference(difference, gold, groups):
    """Class-balanced high-minus-low accuracy on complete published take keys."""
    values = np.asarray(difference, dtype=float)
    weights = balanced_weights(["corpus"] * len(values), gold)
    estimate = float(np.average(values, weights=weights))
    unique = sorted(set(groups))
    result = {"n_pairs": len(values), "difference_pp": estimate * 100,
              "nclusters": len(unique), "ci_lower_pp": None, "ci_upper_pp": None}
    if len(unique) >= 20 and len(values) > 1:
        scores = np.asarray([np.sum(weights[np.asarray(groups) == group] *
                            (values[np.asarray(groups) == group] - estimate)) for group in unique])
        variance = len(unique) / (len(unique) - 1) * float(scores @ scores) / weights.sum() ** 2
        margin = float(t.ppf(.975, len(unique) - 1)) * np.sqrt(max(0., variance))
        result.update(ci_lower_pp=100 * (estimate - margin), ci_upper_pp=100 * (estimate + margin))
    return result


def analyse(selected, inventory, strengths):
    original = {(r["dataset"], r["audio_path"]): r for r in inventory}
    by_dataset = defaultdict(list)
    for row in strengths:
        by_dataset[row["dataset"]].append(row)
    scope_rows, level_rows, class_rows, effect_rows, matched_rows = [], [], [], [], []
    for dataset, spec in SCALES.items():
        all_rows = by_dataset[dataset]
        known = [r for r in all_rows if r["intensity_available"]]
        eligible = [r for r in known if r["intensity_applicable"]]
        continuous = "continuous" in spec["kind"]
        native_levels = sorted({r["intensity_level"] for r in eligible})
        scope_rows.append({"dataset": dataset, "n_total": len(all_rows), "n_known": len(known),
            "n_non_neutral_known": len(eligible), "scale_name": spec["scale"],
            "scale_kind": "continuous" if continuous else "ordinal", "native_kind": spec["kind"],
            "levels": "1–9 listener means (continuous)" if continuous else ",".join(f"{v:g}" for v in native_levels),
            "pair_rule": spec.get("pair_rule", "No matched ordinal take keys"), "source": spec["source"],
            "source_url": spec["url"], "notes": spec["notes"]})
        # For the continuous listener score, quartiles are visual bins only. Regression uses the raw score.
        cuts = np.quantile([r["intensity_level"] for r in eligible], [.25, .5, .75]) if continuous else None
        for model, runs in selected.items():
            if dataset not in runs:
                continue
            run = runs[dataset]
            rows = [r for r in eligible if r["audio_path"] in run.rows]
            if not rows:
                continue
            gold = np.asarray([r["gold"] for r in rows], dtype=object)
            correct = np.asarray([run.rows[r["audio_path"]]["prediction"] in run.accept.get(r["gold"], [r["gold"]])
                                  for r in rows], dtype=float)
            predictions = np.asarray([run.rows[r["audio_path"]]["prediction"] for r in rows], dtype=object)
            invalid = np.asarray([run.rows[r["audio_path"]]["prediction"] is None for r in rows])
            strength = np.asarray([r["intensity_level"] for r in rows])
            headers = [original[dataset, r["audio_path"]] for r in rows]
            duration = np.asarray([r["duration_seconds"] for r in headers])
            groups = [cluster_identity(r) for r in headers]
            ids = np.searchsorted(cuts, strength, side="right") if continuous else np.asarray([
                native_levels.index(level) for level in strength])
            n_levels = 4 if continuous else len(native_levels)
            for i in range(n_levels):
                mask = ids == i
                covered = set(gold[mask])
                labels = set(gold)
                if continuous:
                    label = f"Q{i + 1}"
                    level = float(np.median(strength[mask])) if mask.any() else None
                else:
                    label = next(r["intensity_name"] for r in rows if r["intensity_level"] == native_levels[i])
                    level = native_levels[i]
                level_rows.append({"dataset": dataset, "model": model, "scale_name": spec["scale"],
                    "scale_kind": "continuous" if continuous else "ordinal", "level": level,
                    "level_label": label, "level_id": i, "n": int(mask.sum()),
                    "correct": int(correct[mask].sum()), "invalid": int(invalid[mask].sum()),
                    "accuracy": float(correct[mask].mean()) if mask.any() else None,
                    "uar": float(np.mean([correct[mask & (gold == label)].mean() for label in sorted(labels)]))
                           if labels <= covered else None,
                    "invalid_rate": float(invalid[mask].mean()) if mask.any() else None,
                    "predicted_angry_rate": float(np.mean(predictions[mask] == "angry")) if mask.any() else None,
                    "predicted_disgusted_rate": float(np.mean(predictions[mask] == "disgusted")) if mask.any() else None,
                    "median_duration_seconds": float(np.median(duration[mask])) if mask.any() else None})
                for truth in sorted(covered):
                    chosen = mask & (gold == truth)
                    errors = Counter(str(p) if p is not None else "<invalid>"
                                     for p, ok in zip(predictions[chosen], correct[chosen]) if not ok)
                    common_error, common_count = errors.most_common(1)[0] if errors else (None, 0)
                    class_rows.append({"dataset": dataset, "model": model, "level_id": i,
                        "level_label": label, "level": level, "gold": truth,
                        "n": int(chosen.sum()), "correct": int(correct[chosen].sum()),
                        "invalid": int(invalid[chosen].sum()), "accuracy": float(correct[chosen].mean()),
                        "most_common_error": common_error, "most_common_error_count": common_count})
            weights = balanced_weights([dataset] * len(rows), gold)
            for fit in ("class_balanced", "class_balanced_duration_adjusted"):
                use = duration > 0 if fit.endswith("duration_adjusted") else np.ones(len(rows), dtype=bool)
                result = fixed_effect_linear(strength[use], correct[use], gold[use], np.asarray(groups)[use], weights[use],
                    covariates=np.log2(duration[use]) if fit.endswith("duration_adjusted") else None)
                if result["nclusters"] < 20:
                    result |= {"ci_lower_pp": None, "ci_upper_pp": None, "stderr_pp": None, "pvalue": None,
                               "note": result["note"] + " CI suppressed: fewer than 20 source clusters."}
                effect_rows.append({"dataset": dataset, "model": model, "fit": fit,
                    "scale_kind": "continuous" if continuous else "ordinal",
                    "unit": "per listener-rating point" if continuous else "per native ordinal step", **result})
            if continuous:
                continue
            cases = defaultdict(list)
            for i, row in enumerate(rows):
                if row["intensity_pair_key"]:
                    cases[row["intensity_pair_key"]].append(i)
            differences, pair_gold, pair_groups = [], [], []
            for case, positions in cases.items():
                levels = {float(strength[i]) for i in positions}
                if set(native_levels) - levels:
                    continue
                if len({gold[i] for i in positions}) != 1 or len({groups[i] for i in positions}) != 1:
                    raise ValueError(f"{case}: matched take key spans emotions or source clusters")
                low = [i for i in positions if strength[i] == native_levels[0]]
                high = [i for i in positions if strength[i] == native_levels[-1]]
                differences.append(float(correct[high].mean() - correct[low].mean()))
                pair_gold.append(gold[positions[0]])
                pair_groups.append(groups[positions[0]])
            if differences:
                matched_rows.append({"dataset": dataset, "model": model,
                    **paired_difference(differences, pair_gold, pair_groups),
                    "low_level": native_levels[0], "high_level": native_levels[-1], "pair_rule": spec["pair_rule"]})
    return {"scope_rows": scope_rows, "level_rows": level_rows, "class_rows": class_rows,
            "effect_rows": effect_rows, "matched_rows": matched_rows}


def run(*, refresh_intensity=False, plots=True, data_only=False):
    project, paper = consensus.PROJECT, consensus.REPO / "research_paper"
    output = paper / "output"
    registry = yaml.safe_load((project / "config/models.yaml").read_text())
    selected, selection = consensus.load_runs(project, registry, datasets=consensus.DATASETS, exclude=["esd"])
    inventory = load_inventory(project, selected, output)
    strengths, intensity_manifest = load_strengths(project, inventory, output, refresh=refresh_intensity)
    files = [Path(__file__), Path(__file__).with_name("intensity_inventory.py"), Path(__file__).with_name("duration_stats.py"),
             Path(__file__).with_name("cross_dataset_analysis.py"), Path(consensus.__file__), project / "config/models.yaml"]
    for name in ("intensity_plot.py", "intensity_report.py", "cross_dataset_plot.py"):
        path = Path(__file__).with_name(name)
        if path.exists():
            files.append(path)
    implementation = {p.relative_to(consensus.REPO).as_posix(): consensus.sha256(p) for p in files}
    data = analyse(selected, inventory, strengths)
    sources = {s["path"]: s for runs in selected.values() for run in runs.values() for s in run.sources}
    manifest = {"analysis": "native_intensity_classification_v1",
        "generated_at_singapore": datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M SGT"),
        "selected_run_count": sum(map(len, selected.values())), "selection": selection,
        "model_order": [m for m in registry if m in selected],
        "model_names": {m: consensus.SHORT_NAMES.get(m, registry[m].get("name", m)) for m in selected},
        "native_meralion_audit": intensity_manifest["meralion_capabilities"],
        "intensity_inventory_sha256": consensus.sha256(output / "intensity_inventory.csv"),
        "audio_inventory_sha256": consensus.sha256(output / "audio_inventory.csv"),
        "prediction_sources": sorted(sources.values(), key=lambda r: r["path"]),
        "implementation": implementation,
        "parameters": {"primary_scope": "known native intensity, non-neutral original gold, original saved predictions",
            "intensity_scales": "separate native units; never pooled across datasets",
            "continuous_display": "MCAE-SPPS native listener mean; quartiles for plotting only, raw 1-9 score for regression",
            "fixed_effects": "original gold class intercepts; equal gold-class total weights within each dataset",
            "duration_control": "log2(original clip duration) as additional covariate",
            "intervals": "source-cluster CR1 and t(G-1); withheld below 20 source clusters; exploratory, not multiplicity-adjusted",
            "matched_cases": "complete native levels on published take keys; class-balanced high-minus-low accuracy",
            "display_min_level_n": 20},
        "notes": ["Classification sensitivity to native strength is not a test of predicted intensity accuracy.",
            "Neutral references and unspecified XX are not weak-emotion examples.",
            "MERaLiON native VAD outputs are absent from saved predictions; no additional inference was performed.",
            "Perceived intensity can covary with rater agreement; acting-level comparisons are not randomized interventions."]}
    data["manifest"] = manifest
    for name, digest in implementation.items():
        if consensus.sha256(consensus.REPO / name) != digest:
            raise RuntimeError("Analysis implementation changed during computation; rerun after edits finish")
    for source in sources.values():
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError("Source predictions changed while analysing intensity")
    for key in ("scope_rows", "level_rows", "class_rows", "effect_rows", "matched_rows"):
        write_csv(output / f"intensity_{key}.csv", data[key])
    if not data_only:
        if plots:
            from intensity_plot import plot_intensity
            paths = plot_intensity(data["level_rows"], data["effect_rows"], data["matched_rows"],
                                   data["scope_rows"], paper / "analysis_report/figures")
            manifest["figures"] = [p.relative_to(paper).as_posix() for p in paths]
        from intensity_report import render_report
        for language in ("zh", "en"):
            (paper / "analysis_report" / f"intensity_{language}.md").write_text(render_report(data, language), encoding="utf-8")
    (output / "intensity_analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "intensity_analysis.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Intensity analysis: {len(data['scope_rows'])} native scales; {len(data['effect_rows'])} adjusted fits; "
          f"{len(data['matched_rows'])} model/take comparisons.")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-intensity", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    run(refresh_intensity=args.refresh_intensity, plots=not args.no_plots, data_only=args.data_only)

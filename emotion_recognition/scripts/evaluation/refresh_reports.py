"""Validate saved prediction artifacts and rebuild reports, without inference or source-file edits.

Run from emotion_recognition/:
    python scripts/evaluation/refresh_reports.py --model gemini-3.8-flash --plots --viewer
    python scripts/evaluation/refresh_reports.py --all-models

Every selected CSV/JSON pair is validated before any report is written. --plots and --viewer
optionally rebuild the existing comparison figures and HTML; they never call a model/API.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for folder in ("common", "llm_eval", "viewer", "evaluation"):
    sys.path.insert(0, str(ROOT / "scripts" / folder))
from prediction_io import parse_correct, read_prediction_rows, write_derived_metrics
import evaluate as reporting

METRICS = {"accuracy", "unweighted_accuracy", "macro_f1", "weighted_f1", "accuracy_all_classes",
           "accuracy_acted_or_listener", "listeners_vs_acted", "model_vs_listeners"}
METRIC_TOLERANCE = 1e-6  # existing metadata rounds to six decimals


@dataclass
class ValidatedRun:
    model: str
    stem: str
    metadata: dict
    ds: object
    records: list
    metrics: dict
    matrix: np.ndarray
    lenient_matrix: np.ndarray
    rows: list
    lenient_rows: list
    other: bool
    has_listener: bool
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    resolved_support: np.ndarray
    row_recall: np.ndarray
    lenient_recall: list
    scored: list
    audit: dict


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distinct_strings(value, name):
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() == x and x for x in value):
        raise ValueError(f"{name} must be a non-empty list of labels")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} contains duplicate labels")
    return list(value)


def compare_metric(name, claimed, computed, source):
    if computed is None:
        if claimed is not None:
            raise ValueError(f"{source}: {name} cannot be verified without the required labels")
        return
    if isinstance(claimed, bool) or not isinstance(claimed, (int, float)) or not math.isfinite(claimed):
        raise ValueError(f"{source}: {name} must be a finite numeric metric")
    if not math.isclose(claimed, computed, abs_tol=METRIC_TOLERANCE, rel_tol=0):
        raise ValueError(f"{source}: {name} mismatch: JSON {claimed}, CSV recomputation {computed:.12f}")


def probability_audit(records, metadata, entry, header):
    """Check supplied numbers without creating scores or requiring unavailable probabilities."""
    model_classes = distinct_strings(metadata["model_classes"], "model_classes")
    columns = [f"p_{lab.strip('<>')}" for lab in model_classes]
    if len(set(columns)) != len(columns):
        raise ValueError("model_classes map to duplicate p_* column names")
    supplied = [key for key in header if key.startswith("p_")]
    if not supplied:
        return "not supplied"
    if set(supplied) != set(columns):
        raise ValueError("p_* columns must exactly match model_classes when supplied")
    one_hot = True
    for number, row in enumerate(records, 2):
        try:
            values = [float(row[column]) for column in columns]
        except (TypeError, ValueError):
            raise ValueError(f"CSV row {number}: invalid p_* value") from None
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
            raise ValueError(f"CSV row {number}: p_* values must be finite and between zero and one")
        # Existing evaluation exports scores to four decimal places.
        if abs(sum(values) - 1) > max(1e-6, len(values) * 0.00005001):
            raise ValueError(f"CSV row {number}: p_* values do not sum to one within export rounding")
        one_hot &= all(v in (0.0, 1.0) for v in values) and values.count(1.0) == 1
        all_prediction = row.get("prediction_all_classes")
        if all_prediction and values[model_classes.index(all_prediction)] < max(values) - 0.00010001:
            raise ValueError(f"CSV row {number}: prediction_all_classes disagrees with supplied p_* values")
        selected = [values[model_classes.index(label)] if label in model_classes else 0.0
                    for label in metadata["classes"]]
        if reporting.is_llm(entry) and any(value > 0 for value in selected) != (row["prediction"] in metadata["classes"]):
            raise ValueError(f"CSV row {number}: candidate probability mass disagrees with the LLM answer")
        if row["prediction"] in metadata["classes"]:
            predicted = selected[metadata["classes"].index(row["prediction"])]
            if predicted < max(selected) - 0.00010001:
                raise ValueError(f"CSV row {number}: dataset prediction disagrees with supplied p_* values")
    return "one-hot answer encoding" if one_hot and reporting.is_llm(entry) else "supplied scores (not recalibrated)"


def validate_pair(json_path, model, entry):
    """Read and verify one result; creates no output directories or reports."""
    json_path = Path(json_path).resolve()
    csv_path = json_path.with_suffix(".csv")
    metadata = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if not isinstance(metadata, dict) or metadata.get("model") != model:
        raise ValueError(f"{json_path.name}: model identity must match outputs/{model}")
    key = metadata.get("key")
    if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
        raise ValueError(f"{json_path.name}: invalid dataset key")
    stem = json_path.name.removesuffix("_predictions.json")
    if stem != key and not stem.startswith(key + "_"):
        raise ValueError(f"{json_path.name}: file stem does not match dataset key {key}")
    classes = distinct_strings(metadata.get("classes"), "classes")
    model_classes = distinct_strings(metadata.get("model_classes"), "model_classes")
    accept = metadata.get("accept") or {}
    if not isinstance(accept, dict):
        raise ValueError("accept must be a mapping of composite labels")
    for composite, targets in accept.items():
        if not isinstance(composite, str) or not composite or composite in classes:
            raise ValueError("accept keys must be separate composite ground-truth labels")
        if not set(distinct_strings(targets, f"accept[{composite}]")) <= set(classes):
            raise ValueError(f"accept[{composite}] names a class outside the candidates")
    row_labels = distinct_strings(metadata.get("labels", classes + list(accept)), "labels")
    if not set(row_labels) <= set(classes) | set(accept):
        raise ValueError("labels contain ground truth outside classes/accept")
    if type(metadata.get("clips")) is not int or metadata["clips"] <= 0:
        raise ValueError("clips must be a positive integer")
    total = metadata.get("total_clips", metadata["clips"])
    if type(total) is not int or total < metadata["clips"]:
        raise ValueError("total_clips must be an integer at least as large as clips")
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"{csv_path.name}: empty or duplicated column names")
    truth_column = metadata.get("ground_truth")
    required = {"audio_path", "prediction", "correct", truth_column}
    if not isinstance(truth_column, str) or not required <= set(header):
        raise ValueError(f"{csv_path.name}: missing required prediction/ground-truth columns")
    records = read_prediction_rows(csv_path)
    if len(records) != metadata["clips"]:
        raise ValueError(f"{csv_path.name}: {len(records)} rows but JSON declares {metadata['clips']}")
    claimed_metrics = metadata.get("metrics") or {}
    if not isinstance(claimed_metrics, dict) or not {"accuracy", "unweighted_accuracy", "macro_f1"} <= set(claimed_metrics):
        raise ValueError("metrics must include accuracy, unweighted_accuracy and macro_f1")
    if set(claimed_metrics) - METRICS:
        raise ValueError(f"Unsupported metrics cannot be verified: {sorted(set(claimed_metrics) - METRICS)}")
    expected_protocol = reporting.current_protocol(entry, classes)
    if expected_protocol is not None and metadata.get("inference_protocol") != expected_protocol:
        raise ValueError(f"{json_path.name}: candidate/prompt/model protocol does not match current configuration")
    if expected_protocol is None and metadata.get("inference_protocol") is not None:
        raise ValueError(f"{json_path.name}: unexpected LLM protocol for a non-LLM registry entry")
    has_listener = "listener_vote" in header
    if has_listener and "correct_acted_or_listener" not in header:
        raise ValueError("Listener-labelled rows require correct_acted_or_listener")
    if "accuracy_all_classes" in claimed_metrics and "prediction_all_classes" not in header:
        raise ValueError("accuracy_all_classes requires prediction_all_classes")
    ds = SimpleNamespace(key=key, name=metadata.get("dataset", key), language=metadata.get("language", ""),
                         classes=classes, accept=accept)
    truth, prediction, all_predictions, heard, paths = [], [], [], [], set()
    allowed_predictions = set(classes) | ({reporting.OTHER, "invalid"} if reporting.is_llm(entry) else set())
    for number, row in enumerate(records, 2):
        if None in row or any(row.get(column) is None for column in required):
            raise ValueError(f"CSV row {number}: malformed column count")
        path = PurePosixPath(row["audio_path"].strip())
        if not row["audio_path"].strip() or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"CSV row {number}: audio_path must be a non-empty relative path")
        normalized = path.as_posix()
        if normalized in paths:
            raise ValueError(f"CSV row {number}: duplicated audio_path {normalized}")
        paths.add(normalized)
        label, answer = row[truth_column], row["prediction"]
        if label not in row_labels or answer not in allowed_predictions:
            raise ValueError(f"CSV row {number}: ground truth or prediction outside declared labels/candidates")
        all_answer = row.get("prediction_all_classes", answer)
        if all_answer not in model_classes:
            raise ValueError(f"CSV row {number}: prediction_all_classes outside model_classes")
        votes = set(row.get("listener_vote", "").split(" or ")) - {""}
        # Listener annotations may legitimately use a wider vocabulary than actor candidates
        # (THAI-SER keeps 'frustrated'). Keep them verbatim for the existing lenient/human rules.
        if any(not re.fullmatch(r"[a-z]+", vote) for vote in votes):
            raise ValueError(f"CSV row {number}: listener votes must be canonical lowercase label words")
        strict = answer in reporting.accepted(ds, label)
        lenient = strict or answer in votes
        if parse_correct(row["correct"]) != strict:
            raise ValueError(f"CSV row {number}: correct flag disagrees with the labels")
        if has_listener and parse_correct(row["correct_acted_or_listener"]) != lenient:
            raise ValueError(f"CSV row {number}: lenient correctness flag disagrees with the labels")
        truth.append(label)
        prediction.append(answer)
        all_predictions.append(all_answer)
        heard.append(votes)
    probability_kind = probability_audit(records, metadata, entry, header)
    has_other = any(answer not in classes for answer in prediction)
    columns = reporting.columns_of(ds, has_other)
    ti = np.array([row_labels.index(label) for label in truth])
    pi = np.array([classes.index(answer) if answer in classes else len(classes) for answer in prediction])
    strict = np.array([answer in reporting.accepted(ds, label) for answer, label in zip(prediction, truth)])
    lenient = strict | np.array([answer in votes for answer, votes in zip(prediction, heard)])
    matrix = np.zeros((len(row_labels), len(columns)), dtype=int)
    np.add.at(matrix, (ti, pi), 1)
    lenient_rows = row_labels + [label for label in classes if label not in row_labels]
    lenient_matrix = np.zeros((len(lenient_rows), len(columns)), dtype=int)
    vote_rows = np.array([lenient_rows.index(answer) if answer in lenient_rows else -1 for answer in prediction])
    np.add.at(lenient_matrix, (np.where(lenient & ~strict, vote_rows, ti), pi), 1)
    good = reporting.correct_cells(ds, row_labels, has_other)
    recall = reporting.ratio((matrix * good).sum(1), matrix.sum(1))
    resolved = np.where(strict, pi, [classes.index(accept.get(label, [label])[0]) for label in truth])
    resolved_matrix = np.zeros((len(classes), len(columns)), dtype=int)
    np.add.at(resolved_matrix, (resolved, pi), 1)
    precision = reporting.ratio(resolved_matrix.diagonal(), resolved_matrix[:, :len(classes)].sum(0))
    resolved_recall = reporting.ratio(resolved_matrix.diagonal(), resolved_matrix.sum(1))
    f1 = reporting.ratio(2 * precision * resolved_recall, precision + resolved_recall)
    scored = [i for i, label in enumerate(classes) if label in row_labels or (pi == i).any()]
    voted = [i for i, votes in enumerate(heard) if len(votes) == 1]
    metrics = {"accuracy": float(strict.mean()), "unweighted_accuracy": float(recall.mean()),
               "macro_f1": float(f1[scored].mean()),
               "weighted_f1": float(np.average(f1, weights=resolved_matrix.sum(1))),
               "accuracy_all_classes": float(np.mean([p in reporting.accepted(ds, a)
                                                      for p, a in zip(all_predictions, truth)]))}
    if has_listener:
        metrics.update(accuracy_acted_or_listener=float(lenient.mean()),
                       listeners_vs_acted=float(np.mean([next(iter(heard[i])) in reporting.accepted(ds, truth[i])
                                                        for i in voted])) if voted else None,
                       model_vs_listeners=float(np.mean([prediction[i] == next(iter(heard[i])) for i in voted]))
                       if voted else None)
    for name, claimed in claimed_metrics.items():
        compare_metric(name, claimed, metrics.get(name), json_path.name)
    chance = sum(len(reporting.accepted(ds, label) & set(classes)) for label in truth) / len(truth) / len(classes)
    if "chance" in metadata:
        compare_metric("chance", metadata["chance"], chance, json_path.name)
    artifacts = [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                 for path in (csv_path, json_path)]
    prediction_counts, truth_counts = Counter(prediction), Counter(truth)
    invalid_count = sum(count for label, count in prediction_counts.items() if label not in classes)
    audit = {"validated_at_utc": datetime.now(timezone.utc).isoformat(), "model": model, "key": key,
             "clips": len(records), "unique_audio_paths": len(paths), "source_artifacts": artifacts,
             "source_metrics": claimed_metrics, "recomputed_metrics": metrics, "metric_tolerance": METRIC_TOLERANCE,
             "derived_metrics": sorted(set(metrics) - set(claimed_metrics)),
             "probability_columns": probability_kind, "candidate_protocol": metadata.get("inference_protocol"),
             "protocol_matches_configuration": expected_protocol is not None,
             "prediction_diagnostics": {
                 "invalid_count": invalid_count, "invalid_rate": invalid_count / len(records),
                 "prediction_counts": {**{label: prediction_counts[label] for label in classes},
                                       "invalid": invalid_count},
                 "ground_truth_counts": {label: truth_counts[label] for label in row_labels}},
             "verification_scope": "CSV/JSON internal consistency; no inference, API calls or source edits. "
                                   "This does not independently authenticate the original remote requests."}
    audit["scope_notes"] = (["The known empty CREMA-D recording 1076_MTI_SAD_XX.wav is retained in the saved "
                             "predictions, matching the existing project evaluation scope."]
                            if key == "crema_d" and any(PurePosixPath(row["audio_path"]).name == "1076_MTI_SAD_XX.wav"
                                                        for row in records) else [])
    return ValidatedRun(model, stem, metadata, ds, records, metrics, matrix, lenient_matrix, row_labels,
                        lenient_rows, has_other, has_listener, precision, resolved_recall, f1,
                        resolved_matrix.sum(1), recall,
                        [float(lenient[ti == i].mean()) if (ti == i).any() else 0 for i in range(len(row_labels))],
                        scored, audit)


def scientific_notes(run):
    """Extract reproducibility facts without copying gateway names or external personal paths."""
    text = "\n".join(str(note) for note in run.metadata.get("notes", []))
    notes = []
    if re.search(r"wav\s*24k|24\s*khz", text, re.I):
        notes.append("The original notes report one 24 kHz audio clip per request.")
    concurrent = re.search(r"up to\s+(\d+)\s+clips?\s+evaluated\s+concurrently", text, re.I)
    if concurrent:
        notes.append(f"The original notes report up to {concurrent.group(1)} concurrent clip evaluations.")
        notes.append("The recorded ms_per_clip is total wall-clock time divided by the number of clips, "
                     "not individual request latency; it is not directly comparable with local GPU inference timing.")
    if "responseschema" in text.lower() or "literal json string" in text.lower():
        notes.append("The original notes describe schema-constrained answers validated as literal JSON strings.")
    if run.audit["probability_columns"] == "one-hot answer encoding":
        notes.append("Saved p_* columns encode the returned answer as one-hot values; they are not calibrated "
                     "API class probabilities, and no confidence plot is inferred from them.")
    elif run.audit["probability_columns"] == "not supplied":
        notes.append("No probability columns were supplied; none are reconstructed.")
    return notes


def render_report(run, entry, root=ROOT):
    """Render an already validated record using existing confusion-matrix/table helpers."""
    root = Path(root)
    folder = root / "reports" / run.model
    folder.mkdir(parents=True, exist_ok=True)
    strict_path = folder / "figures" / f"{run.stem}_confusion.png"
    lenient_path = folder / "figures" / f"{run.stem}_confusion_acted_or_listener.png"
    name = run.metadata.get("model_name", entry["name"])
    pct = lambda value: f"{100 * value:.2f}%" if value is not None else "—"
    diagnostics = run.audit["prediction_diagnostics"]
    reporting.plot_confusion(run.matrix, run.ds, run.rows, f"{name} on {run.ds.name}",
                            f"Saved predictions; accuracy {pct(run.metrics['accuracy'])}.",
                            "Ground-truth emotion (clips)", strict_path, run.other)
    if run.has_listener:
        reporting.plot_confusion(run.lenient_matrix, run.ds, run.lenient_rows, f"{name} on {run.ds.name}",
                                f"Acted emotion or listener vote; accuracy {pct(run.metrics['accuracy_acted_or_listener'])}.",
                                "Ground truth, or listener vote if matched (clips)", lenient_path, run.other)
    source = root / "outputs" / run.model / f"{run.stem}_predictions"
    link = lambda suffix: f"../../outputs/{run.model}/{run.stem}_predictions{suffix}"
    lines = [f"# {name} on {run.ds.name}", "",
             "Regenerated offline from saved CSV/JSON by `scripts/evaluation/refresh_reports.py`. "
             "This refresh performs no model inference or API requests.", "",
             "## Summary", "", "| Metric | Recomputed value |", "|---|---:|",
             f"| Clips | {len(run.records):,} |", f"| Accuracy: acted emotion | {pct(run.metrics['accuracy'])} |",
             f"| UAR: mean ground-truth-row recall | {pct(run.metrics['unweighted_accuracy'])} |",
             f"| Macro F1 | {pct(run.metrics['macro_f1'])} |",
             f"| Weighted F1: true-support weighted | {pct(run.metrics['weighted_f1'])} |",
             f"| Accuracy: all model classes | {pct(run.metrics['accuracy_all_classes'])} |",
             f"| Invalid / outside-candidate answers | {diagnostics['invalid_count']:,} "
             f"({pct(diagnostics['invalid_rate'])}) |"]
    for metric, label in (("accuracy_acted_or_listener", "Accuracy: acted emotion or listener vote"),
                          ("listeners_vs_acted", "Single listener vote vs acted emotion"),
                          ("model_vs_listeners", "Model vs single listener vote")):
        if metric in run.metrics:
            lines.append(f"| {label} | {pct(run.metrics[metric])} |")
    lines += ["", "## Imported run and verification", "",
              f"- Source: [prediction CSV]({link('.csv')}) and [metadata JSON]({link('.json')}); "
              "both remain byte-for-byte unchanged.",
              f"- Original execution date recorded in JSON: `{run.metadata.get('date', 'not supplied')}`.",
              f"- Dataset scope recorded in JSON: **{run.ds.name}**, {run.ds.language}; "
              f"{run.metadata['clips']:,} / {run.metadata.get('total_clips', run.metadata['clips']):,} clips.",
              f"- Ground-truth column: `{run.metadata['ground_truth']}`. Candidate classes: {', '.join(run.ds.classes)}.",
              "- Row counts, unique audio paths, correctness flags and saved metrics were checked against the CSV. "
              "Metric agreement tolerance is 0.000001 (six-decimal metadata).",
              "- Weighted F1 weights each class's F1 by its true support, using the same resolved truth as macro F1; "
              "invalid answers remain errors. New metrics absent from the source JSON are recorded in the offline "
              "audit and summaries; the source JSON is preserved."]
    protocol = run.metadata.get("inference_protocol")
    if protocol:
        lines.append(f"- Saved candidate/prompt/model protocol matches current configuration: "
                     f"scope `{protocol['label_scope']}`, model `{protocol['model_id']}`, "
                     f"prompt hash `{protocol['prompt_hash']}`.")
    runtime = run.metadata.get("runtime") or {}
    recorded_runtime = {key: runtime[key] for key in ("device", "hardware", "seconds", "ms_per_clip", "on_cpu")
                        if key in runtime}
    if recorded_runtime:
        lines.append("- Original execution/client runtime metadata (reported by the imported run, not measured "
                     "by this refresh): `" + json.dumps(recorded_runtime, ensure_ascii=False) + "`.")
    lines += [f"- {note}" for note in scientific_notes(run)]
    if reporting.is_llm(entry):
        journal = source.parent / "responses.jsonl"
        lines.append("- " + ("A raw responses journal is present; this refresh validates CSV/JSON only."
                     if journal.exists() else "Raw responses were not included in this import. Metrics can be "
                     "recomputed from the saved labels, but original generated answers and remote requests cannot "
                     "be independently reconstructed."))
    if run.ds.accept:
        lines.append("- Composite labels follow the existing accepted-label rule: " + "; ".join(
                     f"{label} accepts {' or '.join(accepted)}" for label, accepted in run.ds.accept.items()) + ". "
                     "F1 resolves a correct composite to the predicted class, otherwise to its first accepted class. "
                     "Weighted F1 uses the resulting resolved-truth support, which can depend on the predictions.")
    small_classes = {label: count for label, count in diagnostics["ground_truth_counts"].items()
                     if 0 < count <= 5}
    if small_classes:
        lines.append("- Small original ground-truth classes (at most 5 clips): " + ", ".join(
                     f"{label} ({count:,} {'clip' if count == 1 else 'clips'})"
                     for label, count in small_classes.items()) + ". "
                     "UAR gives each original ground-truth row equal weight; recalls from these small supports "
                     "should be interpreted with care.")
    lines += ["", "## Prediction distribution", "",
              "Counts and shares use all saved samples. The invalid bucket combines `invalid` and "
              "`other answer`: answers outside the dataset's candidate classes remain errors.", "",
              "| Predicted label | Clips | Share of all clips |", "|---|---:|---:|",
              *[f"| {label if label != 'invalid' else 'invalid / outside candidates'} | {count:,} | "
                f"{pct(count / len(run.records))} |"
                for label, count in diagnostics["prediction_counts"].items()]]
    lines += ["", "## Results per emotion", "",
              "| Ground-truth emotion | Clips | Recall | Acted-or-listener recall |",
              "|---|---:|---:|---:|",
              *[f"| {reporting.row_name(run.ds, label)} | {run.matrix[i].sum():,} | {pct(run.row_recall[i])} | "
                + (pct(run.lenient_recall[i]) if run.has_listener else "—") + " |" for i, label in enumerate(run.rows)],
              "", "Precision/F1 use the same resolved-truth convention as the evaluation script. Weighted F1 uses "
              "the resolved-truth clip counts below as weights.", "",
              "| Predicted emotion | Resolved-truth clips | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|",
              *[f"| {label} | {run.resolved_support[i]:,} | "
                + " | ".join(pct(values[i]) if i in run.scored else "—"
                              for values in (run.precision, run.recall, run.f1)) + " |"
                for i, label in enumerate(run.ds.classes)],
              "", "## Confusion matrix: acted emotion", "",
              f"![Acted-emotion confusion matrix](figures/{strict_path.name})", "",
              *reporting.confusion_table(run.matrix, run.ds, run.rows, "Ground truth \\ Predicted", run.other), ""]
    if run.has_listener:
        lines += ["## Confusion matrix: acted emotion or listener vote", "",
                  "A prediction accepted only through the listener vote is placed in that predicted emotion's row; "
                  "other clips keep their original ground-truth row. Row supports can therefore change.", "",
                  f"![Acted-or-listener confusion matrix](figures/{lenient_path.name})", "",
                  *reporting.confusion_table(run.lenient_matrix, run.ds, run.lenient_rows,
                                            "Ground truth \\ Predicted", run.other), ""]
    lines += ["## Caveats", "", f"- {reporting.training_caveat(run.ds.key, entry, '')}",
              *[f"- {note}" for note in run.audit["scope_notes"]],
              "- Internal CSV/JSON consistency does not authenticate the original model or remote request logs.", ""]
    audit = dict(run.audit)
    audit["source_artifacts"] = [{**item, "path": Path(item["path"]).relative_to(root).as_posix()}
                                 for item in audit["source_artifacts"]]
    audit_path = folder / f"{run.stem}_offline_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines += [f"[Offline validation audit]({audit_path.name})", ""]
    output = folder / f"{run.stem}.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"  offline report -> {output.relative_to(root)}")
    return output


@contextmanager
def reporting_paths(root):
    """Existing summary helpers use module paths; scope them to the requested report root."""
    paths = {"ROOT": root, "CONFIG": root / "config/evaluation.yaml",
             "MODELS_CONFIG": root / "config/models.yaml", "LLM_CONFIG": root / "config/llm.yaml"}
    previous = {name: getattr(reporting, name) for name in paths}
    try:
        for name, value in paths.items():
            setattr(reporting, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(reporting, name, value)


def refresh(model_keys, *, root=ROOT, plots=False, viewer=False):
    root = Path(root).resolve()
    if root != ROOT and (plots or viewer):
        raise ValueError("Optional plots/viewer rebuilding uses the project root; custom roots are report-only")
    with reporting_paths(root):
        return _refresh(model_keys, root=root, plots=plots, viewer=viewer)


def _refresh(model_keys, *, root, plots, viewer):
    root = Path(root)
    registry = yaml.safe_load((root / "config/models.yaml").read_text()) or {}
    settings = yaml.safe_load((root / "config/evaluation.yaml").read_text()) or {}
    exclude = settings.get("exclude") or []
    if model_keys is None:
        model_keys = [model for model in registry
                      if any((root / "outputs" / model).glob("*_predictions.json"))]
    validated = []
    for model in dict.fromkeys(model_keys):
        if model not in registry or not re.fullmatch(r"[a-zA-Z0-9_.-]+", model):
            raise ValueError(f"Unknown model registry key: {model}")
        files = sorted((root / "outputs" / model).glob("*_predictions.json"))
        if not files:
            raise ValueError(f"No saved prediction JSONs in outputs/{model}")
        for path in files:
            key = json.loads(path.read_text(encoding="utf-8-sig")).get("key")
            if key not in exclude:
                validated.append(validate_pair(path, model, registry[model]))
    if not validated:
        raise ValueError("No selected prediction pairs remain after dataset exclusions")
    # Validation is a complete preflight; a bad later pair cannot leave a partial refresh behind.
    for run in validated:
        for artifact in run.audit["source_artifacts"]:
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise ValueError("A source artifact changed during validation; retry after copying is complete")
    reports = [render_report(run, registry[run.model], root) for run in validated]
    write_derived_metrics(root, [run.audit for run in validated])
    figure = settings.get("summary_figure") or {}
    for model in dict.fromkeys(run.model for run in validated):
        reporting.write_summary(model, registry[model], note="Dataset reports regenerated offline from saved artifacts; "
                                "this refresh performs no inference or API requests.",
                                figure_order=figure.get("order") or [],
                                acted_or_listener=figure.get("acted_or_listener") or [], exclude=exclude)
    reporting.write_comparison(registry, exclude=exclude)
    if plots:
        import plot_results
        plot_results.main()
    if viewer:
        import make_viewer
        make_viewer.main()
    for run in validated:
        for artifact in run.audit["source_artifacts"]:
            if sha256(artifact["path"]) != artifact["sha256"]:
                raise RuntimeError("Source prediction artifacts changed while refreshing reports")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", action="append", help="registry key; repeat for multiple imports")
    selection.add_argument("--all-models", action="store_true", help="validate and refresh every registered model with saved predictions")
    parser.add_argument("--plots", action="store_true", help="also rebuild comparison PNG/SVG figures")
    parser.add_argument("--viewer", action="store_true", help="also rebuild HTML; use --plots to refresh embedded figures")
    args = parser.parse_args()
    try:
        refresh(args.model, plots=args.plots, viewer=args.viewer)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

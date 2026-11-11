"""Read saved prediction CSVs without changing the source artifacts.

CSV exports from other machines can have a UTF-8 BOM and Yes/No correctness
columns. Present the same canonical True/False strings to all result consumers.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def parse_correct(value):
    """Parse an explicit correctness flag; reject unknown values instead of scoring them wrong."""
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise ValueError(f"Unrecognized prediction correctness value: {value!r}")


def read_prediction_rows(path):
    """Return rows with a BOM-free header and normalized correctness fields, in memory only."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for line, row in enumerate(rows, 2):
        for key in row:
            if isinstance(key, str) and key.startswith("correct"):
                try:
                    row[key] = str(parse_correct(row[key]))
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line}, {key}: {exc}") from exc
    return rows


def load_reported_metrics(prediction_path, run):
    """Saved metrics plus verified derived metrics, without editing source artifacts.

    Older prediction JSONs have no weighted_f1. Read the compact derived-metrics index (or an older companion audit)
    only while both source hashes still match. Other saved metrics keep their original values.
    A missing, stale or malformed audit leaves the new metric unavailable rather than inventing a value.
    """
    metrics = dict(run.get("metrics", {}))
    if "weighted_f1" in metrics:
        return metrics
    prediction_path = Path(prediction_path).resolve()
    if prediction_path.parent.parent.name != "outputs":
        return metrics
    root = prediction_path.parent.parent.parent
    model = prediction_path.parent.name
    stem = prediction_path.name.removesuffix("_predictions.json")
    audit_path = root / "reports" / model / f"{stem}_offline_audit.json"
    try:
        index_path = root / "reports" / "derived_metrics.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        if not isinstance(index, dict) or not isinstance(index.get("runs", {}), dict):
            return metrics
        audit = index.get("runs", {}).get(prediction_path.relative_to(root).as_posix())
        if audit is None:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (run.get("model") != model or audit["model"] != model or audit["key"] != run["key"]
                or audit["clips"] != run["clips"]):
            return metrics
        weighted = audit["recomputed_metrics"]["weighted_f1"]
        if isinstance(weighted, bool) or not isinstance(weighted, (int, float)) or not math.isfinite(weighted) \
                or not 0 <= weighted <= 1:
            return metrics
        artifacts = {item["path"]: item for item in audit["source_artifacts"]}
        for source in (prediction_path, prediction_path.with_suffix(".csv")):
            recorded = artifacts[source.relative_to(root).as_posix()]
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != recorded["sha256"]:
                return metrics
    except (OSError, ValueError, KeyError, TypeError):
        return metrics
    metrics["weighted_f1"] = float(weighted)
    return metrics


def write_derived_metrics(root, audits):
    """Keep derived scores and source hashes in one index instead of one new file per run."""
    root = Path(root).resolve()
    path = root / "reports" / "derived_metrics.json"
    index = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}
    if not isinstance(index, dict) or not isinstance(index.get("runs"), dict):
        raise ValueError(f"Invalid derived-metrics index: {path}")
    for audit in audits:
        sources = []
        for item in audit["source_artifacts"]:
            source = Path(item["path"])
            relative = source.relative_to(root).as_posix() if source.is_absolute() else source.as_posix()
            sources.append({"path": relative, "sha256": item["sha256"]})
        prediction = next(item["path"] for item in sources if item["path"].endswith("_predictions.json"))
        index["runs"][prediction] = {
            **{name: audit[name] for name in ("model", "key", "clips")},
            "recomputed_metrics": {"weighted_f1": audit["recomputed_metrics"]["weighted_f1"]},
            "source_artifacts": sources,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

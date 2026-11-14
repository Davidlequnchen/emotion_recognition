"""Read the headers and published identities of the exact saved evaluation clips.

No waveform decoding, dataset loaders, models or APIs are used. Run from the
repository root: ``python research_paper/scripts/duration_inventory.py``.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf

import consensus

sys.path.insert(0, str(consensus.PROJECT / "scripts/analysis"))
from probe_metadata import DATASET_KEYS, infer_groups


FIELDS = ("dataset", "audio_path", "gold", "duration_seconds", "sampling_rate", "frames",
          "channels", "bytes", "mtime_ns", "empty_audio", "known_silent_recording",
          "group", "group_kind", "group_source", "speaker_id", "source_group_id")
DIALOGUE_DATASETS = frozenset({"emotiontalk", "dailytalk", "multidialog"})
KNOWN_SILENT = "data/crema_d/AudioWAV/1076_MTI_SAD_XX.wav"


def _digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _cohorts(selected):
    cohorts, rules = {}, {}
    for model, datasets in selected.items():
        for dataset, run in datasets.items():
            current_rule = (run.classes, run.accept)
            if dataset in rules and current_rule != rules[dataset]:
                raise ValueError(f"{dataset}: class or accepted-label rules differ across saved runs")
            rules[dataset] = current_rule
            cohort = cohorts.setdefault(dataset, {})
            for path, record in run.rows.items():
                path = consensus.canonical_path(path)
                if path in cohort and cohort[path] != record["gold"]:
                    raise ValueError(f"{dataset}: saved ground truth differs for {path}")
                cohort[path] = record["gold"]
    if not cohorts:
        raise ValueError("No saved prediction cohorts to inventory")
    return cohorts


def _dialogue_groups(project, dataset, paths):
    source = project / "data" / dataset / "utterances.csv"
    wanted = set(paths)
    mappings = {}
    before = consensus.sha256(source)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"audio_path", "dialogue_id", "speaker_id"} <= set(reader.fieldnames or []):
            raise ValueError(f"{source}: missing audio_path/dialogue_id/speaker_id")
        for row in reader:
            relative = row["audio_path"]
            if not relative.startswith(f"data/{dataset}/"):
                relative = f"data/{dataset}/{relative}"
            path = consensus.canonical_path(relative)
            if path not in wanted:
                continue
            if path in mappings:
                raise ValueError(f"{source}: duplicate selected audio path {path}")
            identity = row["dialogue_id"].strip()
            if not identity:
                raise ValueError(f"{source}: empty dialogue ID for {path}")
            mappings[path] = {"group": f"{dataset}:dialogue:{identity}", "group_kind": "dialogue",
                "group_source": f"data/{dataset}/utterances.csv: audio_path -> dialogue_id; "
                                "dialogue clusters do not guarantee speaker independence",
                "speaker_id": row["speaker_id"], "source_group_id": row.get("group_id", "")}
    if wanted != set(mappings):
        raise ValueError(f"{source}: {len(wanted - set(mappings))} selected paths lack dialogue metadata")
    if consensus.sha256(source) != before:
        raise RuntimeError(f"{source}: metadata changed during inventory")
    return mappings, [{"path": source.relative_to(project).as_posix(), "sha256": before}]


def _identities(project, dataset, paths):
    if dataset in DIALOGUE_DATASETS:
        return _dialogue_groups(project, dataset, paths)
    if dataset not in DATASET_KEYS:
        raise ValueError(f"{dataset}: no audited speaker/dialogue identity rule")
    sources = []
    if dataset == "thai_ser":
        candidates = [project / "data/thai_ser/labels.csv"]
    elif dataset == "emozionalmente":
        candidates = [project / "data/emozionalmente/audio/emozionalmente/metadata/samples.csv",
                      project / "data/emozionalmente/metadata/samples.csv"]
    else:
        candidates = []
    if candidates:
        source = next(path for path in candidates if path.is_file())
        sources.append({"path": source.relative_to(project).as_posix(), "sha256": consensus.sha256(source)})
    mapping = infer_groups(dataset, paths, project)
    for source in sources:
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError(f"{source['path']}: grouping metadata changed during parsing")
    return {path: {"group": record["group"], "group_kind": record["group_kind"],
                   "group_source": record["source"],
                   "speaker_id": record["group"].split(":", 2)[2] if record["group_kind"] == "speaker" else "",
                   "source_group_id": ""} for path, record in mapping.items()}, sources


def _header(project, dataset, path, gold, identity):
    audio = project / path
    before = audio.stat()
    info = sf.info(str(audio))
    after = audio.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"{audio}: audio changed while its header was read")
    if info.samplerate <= 0 or info.frames < 0 or info.channels <= 0:
        raise ValueError(f"{audio}: invalid audio header")
    return {"dataset": dataset, "audio_path": path, "gold": gold,
            "duration_seconds": info.frames / info.samplerate, "sampling_rate": info.samplerate,
            "frames": info.frames, "channels": info.channels, "bytes": before.st_size,
            "mtime_ns": before.st_mtime_ns, "empty_audio": info.frames == 0,
            "known_silent_recording": path == KNOWN_SILENT, **identity}


def build_inventory(project, selected, *, workers=8):
    """Return one row per union audio path in each validated dataset, plus provenance.

    ``selected`` has the model/dataset/Run structure returned by
    ``consensus.load_runs``. Matching paths across models must have identical
    gold labels and label rules. Dialogue speaker_id values are retained exactly
    as published and must not be assumed globally unique across dialogues.
    """
    project = Path(project).resolve()
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Use 1 to 8 header-reading workers")
    cohorts = _cohorts(selected)
    jobs, grouping_sources = [], {}
    for dataset in sorted(cohorts, key=lambda key: (consensus.DATASETS.index(key)
                          if key in consensus.DATASETS else len(consensus.DATASETS), key)):
        paths = sorted(cohorts[dataset])
        mapping, sources = _identities(project, dataset, paths)
        grouping_sources.update((source["path"], source) for source in sources)
        jobs.extend((project, dataset, path, cohorts[dataset][path], mapping[path]) for path in paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(lambda args: _header(*args), jobs))
    for row in rows:
        current = (project / row["audio_path"]).stat()
        if (current.st_size, current.st_mtime_ns) != (row["bytes"], row["mtime_ns"]):
            raise RuntimeError(f"{row['audio_path']}: audio changed before inventory completion")
    for source in grouping_sources.values():
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError(f"{source['path']}: grouping metadata changed")
    sources = {source["path"]: source for runs in selected.values()
               for run in runs.values() for source in run.sources}
    for source in sources.values():
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError(f"{source['path']}: saved predictions changed")
    manifest = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "method": "soundfile.info headers only; duration = frames / sampling_rate; no audio decoding",
                "soundfile": sf.__version__, "clips": len(rows),
                "implementation": {"duration_inventory.py": consensus.sha256(Path(__file__)),
                    "probe_metadata.py": consensus.sha256(Path(sys.modules[infer_groups.__module__].__file__))},
                "prediction_sources": sorted(sources.values(), key=lambda item: item["path"]),
                "grouping_sources": sorted(grouping_sources.values(), key=lambda item: item["path"]),
                "datasets": {dataset: {"clips": len(values),
                    "cohort_sha256": _digest(sorted(values.items())),
                    "groups": len({row["group"] for row in rows if row["dataset"] == dataset}),
                    "group_kinds": sorted({row["group_kind"] for row in rows if row["dataset"] == dataset})}
                    for dataset, values in cohorts.items()},
                "notes": ["Cohorts are the union of saved paths per dataset; matched model comparisons must intersect them.",
                    "The known silent CREMA-D recording retains its actual header duration; silence cannot be inferred from headers.",
                    "Conversational group IDs are dialogues, not proof of independent speakers; published speaker_id may be local.",
                    "Audio content is not hashed; bytes/mtime/header signatures detect changes during this inventory."]}
    return rows, manifest


def write_inventory(rows, manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "audio_inventory.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = manifest | {"inventory_csv_sha256": consensus.sha256(path)}
    (output_dir / "audio_inventory_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    project = consensus.PROJECT
    registry = yaml.safe_load((project / "config/models.yaml").read_text())
    selected, _ = consensus.load_runs(project, registry, datasets=consensus.DATASETS, exclude=["esd"])
    rows, manifest = build_inventory(project, selected, workers=args.workers)
    target = write_inventory(rows, manifest, project.parent / "research_paper/output")
    print(f"Header inventory: {len(rows):,} clips from {len(manifest['datasets'])} datasets -> {target}")


if __name__ == "__main__":
    main()

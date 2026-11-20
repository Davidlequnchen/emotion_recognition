"""Manifest I/O, call-disjoint splits and reviewed Label Studio label joins.

CSV audio paths are relative to the CSV's own directory. A group should identify
the largest known leakage unit, usually a customer or a call, never a segment.
"""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Iterable

from .audit import AudioAudit


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ClipRecord:
    path: Path
    label: str | None = None
    group: str = ""
    split: str = ""
    clip_id: str = ""


def _audio_path(value, directory, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: path must be nonempty")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    path = path.resolve(strict=True)
    if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"{context}: expected an audio file: {path}")
    return path


def read_manifest(path, labeled=True, labels=None) -> list[ClipRecord]:
    """Read path,group[,label,split,clip_id] without inferring label decisions."""
    path = Path(path).expanduser().resolve(strict=True)
    records = []
    provided_ids = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "group"} | ({"label"} if labeled else set())
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path}: missing CSV columns {sorted(required - set(reader.fieldnames or []))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate CSV column names")
        for index, row in enumerate(reader, 2):
            context = f"{path}:{index}"
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{context}: malformed CSV row")
            group = row["group"].strip()
            if not group:
                raise ValueError(f"{context}: group must identify a call/customer")
            label = row.get("label", "").strip() if labeled else None
            if labeled and (not label or (labels is not None and label not in labels)):
                raise ValueError(f"{context}: label {label!r} is not a configured class")
            split = row.get("split", "").strip()
            if split and split not in SPLITS:
                raise ValueError(f"{context}: split must be train, validation or test")
            audio_path = _audio_path(row["path"], path.parent, context)
            clip_id = row.get("clip_id", "").strip()
            if clip_id and clip_id in provided_ids:
                raise ValueError(f"{context}: duplicate explicit clip_id: {clip_id}")
            if clip_id:
                provided_ids.add(clip_id)
            records.append(ClipRecord(audio_path, label, group, split,
                                      clip_id or audio_path.stem))
    if not records:
        raise ValueError(f"{path}: manifest contains no clips")
    _validate_unique(records)
    return records


def _validate_unique(records):
    paths = set()
    for row in records:
        if row.path in paths:
            raise ValueError(f"Duplicate audio path in manifest: {row.path}")
        paths.add(row.path)


def assert_disjoint_splits(splits, check_content=True, audit=None):
    """Reject shared groups, paths and byte-identical audio across partitions.

    SHA256 digests come from the audio audit (an in-memory one when audit is None). check_content=False checks
    groups and paths only, for callers that check content later on the clips they actually use.
    """
    groups, paths, contents = {}, {}, {}
    for name, rows in splits.items():
        for row in rows:
            for identity, seen, description in ((row.group, groups, "group"),
                                                 (row.path, paths, "audio path")):
                if identity in seen and seen[identity] != name:
                    raise ValueError(f"Split leakage: {description} {identity!r} occurs in "
                                     f"{seen[identity]} and {name}")
                seen[identity] = name
    if not check_content:
        return
    audit = audit or AudioAudit()
    for name, rows in splits.items():
        for row, digest in zip(rows, audit.digests(row.path for row in rows)):
            if digest in contents and contents[digest] != name:
                raise ValueError(f"Split leakage: byte-identical audio {row.path} occurs in "
                                 f"{contents[digest]} and {name}")
            contents[digest] = name


def split_records(records, validation_fraction=0.2, test_fraction=0.1, seed=42, check_content=True, audit=None):
    """Honor explicit splits or deterministically split entire groups.

    Automatic fractions apply to group counts. Unequal call sizes can therefore
    produce different clip fractions; actual split sizes are the audit record.
    check_content and audit are passed to assert_disjoint_splits.
    """
    records = list(records)
    if not records:
        raise ValueError("Cannot split an empty manifest")
    _validate_unique(records)
    if not (math.isfinite(validation_fraction) and math.isfinite(test_fraction)
            and 0 <= validation_fraction < 1 and 0 <= test_fraction < 1
            and 0 < validation_fraction + test_fraction < 1):
        raise ValueError("Split fractions must be finite, nonnegative and sum to a value between 0 and 1")
    explicit = [bool(row.split) for row in records]
    if any(explicit) and not all(explicit):
        raise ValueError("Either all rows or no rows must provide an explicit split")
    result = {name: [] for name in SPLITS}
    if all(explicit):
        for row in records:
            if row.split not in result:
                raise ValueError(f"Unknown split {row.split!r}")
            result[row.split].append(row)
    else:
        groups = sorted({row.group for row in records})
        if "" in groups:
            raise ValueError("Every clip needs a nonempty call/customer group before splitting")
        random.Random(seed).shuffle(groups)
        counts = [max(1, round(len(groups) * fraction)) if fraction else 0
                  for fraction in (validation_fraction, test_fraction)]
        if sum(counts) >= len(groups):
            raise ValueError("Not enough independent groups for train/validation/test splits")
        validation_count, test_count = counts
        assignments = {group: "validation" for group in groups[:validation_count]}
        assignments.update({group: "test" for group in groups[validation_count:validation_count + test_count]})
        for row in records:
            name = assignments.get(row.group, "train")
            result[name].append(replace(row, split=name))
    if not result["train"] or (validation_fraction and not result["validation"]) or (test_fraction and not result["test"]):
        raise ValueError("Required train/validation/test split is empty; supply more independent groups")
    for name in result:
        result[name].sort(key=lambda row: (row.group, str(row.path)))
    assert_disjoint_splits(result, check_content, audit)
    return result


def select_training_subset(records, max_clips, labels, seed=42):
    """Select at most max_clips after splitting, retaining every configured class."""
    records = sorted(records, key=lambda row: (row.label or "", row.group, str(row.path)))
    labels = list(labels)
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("labels must contain distinct class names")
    invalid = {row.label for row in records} - set(labels)
    if invalid:
        raise ValueError(f"Unexpected training labels: {sorted(invalid, key=str)}")
    buckets = {label: [row for row in records if row.label == label] for label in labels}
    missing = [label for label, rows in buckets.items() if not rows]
    if missing:
        raise ValueError(f"Training split has no examples for configured classes: {missing}; revise group split or classes")
    if max_clips is None:
        return records
    if isinstance(max_clips, bool) or not isinstance(max_clips, int) or max_clips < len(labels):
        raise ValueError("max_train_clips must be null or an integer at least as large as the class count")
    if max_clips >= len(records):
        return records
    rng = random.Random(seed)
    selected, remainder = [], []
    for label in labels:
        rng.shuffle(buckets[label])
        selected.append(buckets[label][0])
        remainder.extend(buckets[label][1:])
    rng.shuffle(remainder)
    selected.extend(remainder[:max_clips - len(selected)])
    return sorted(selected, key=lambda row: (row.group, str(row.path)))


def read_unlabeled(source, exclude_records: Iterable[ClipRecord] = (), training_only=True, check_content=True,
                   audit=None):
    """Read a group-aware CSV or recursively discover a curated audio directory.

    For directory discovery, a nested parent folder is the group (call/customer).
    Top-level clips use their stem. Use a CSV whenever the folder arrangement does
    not encode your true leakage unit. Held-out groups/files/content are excluded.
    Explicit CSV validation/test rows are excluded by default; prediction may set
    training_only=False to retain all rows. Original explicit splits are checked
    before filtering, so a shared train/validation group can never be hidden.
    check_content=False skips the byte-content checks (groups and paths only); SSL
    checks content after sampling, on the clips it actually uses.
    """
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_file():
        records = read_manifest(source, labeled=False)
        if any(row.split for row in records):
            if not all(row.split for row in records):
                raise ValueError("Either all rows or no rows must provide an explicit split")
            assert_disjoint_splits({name: [row for row in records if row.split == name]
                                    for name in SPLITS}, check_content, audit)
            if training_only:
                records = [row for row in records if row.split == "train"]
    else:
        records = []
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            relative = path.relative_to(source)
            group = str(relative.parent) if len(relative.parts) > 1 else path.stem
            records.append(ClipRecord(path.resolve(), group=group, clip_id=path.stem))
        _validate_unique(records)
    records = exclude_holdouts(records, exclude_records, check_content, audit)
    if not records:
        raise ValueError(f"{source}: no unlabeled clips remain after holdout exclusions")
    return records


def exclude_holdouts(records, holdouts: Iterable[ClipRecord], check_content=True, audit=None):
    """Drop clips sharing a group, a path or (check_content) byte-identical audio with a held-out clip."""
    holdouts = list(holdouts)
    paths, groups = {row.path for row in holdouts}, {row.group for row in holdouts}
    records = [row for row in records if row.path not in paths and row.group not in groups]
    if check_content and holdouts and records:
        audit = audit or AudioAudit()
        digests = set(audit.digests(row.path for row in holdouts))
        records = [row for row, digest in zip(records, audit.digests(row.path for row in records))
                   if digest not in digests]
    return records


def write_manifest(records, output_path):
    """Write auditable absolute paths without overwriting an existing manifest."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "group", "split", "clip_id"])
        writer.writeheader()
        writer.writerows({"path": str(row.path), "label": row.label or "", "group": row.group,
                         "split": row.split, "clip_id": row.clip_id} for row in records)


def build_labeled_manifest(votes_path, clips_manifest, output_path, labels):
    """Join normalized Label Studio votes by clip_id, requiring unanimous votes.

    The upstream normalizer preserves individual votes and does not adjudicate.
    Mixed decisions raise an error. Unlabeled clips and unanimous bad_clip or
    unknown decisions are excluded. No pseudo-labels enter this supervised set.
    """
    clips = read_manifest(clips_manifest, labeled=False)
    by_id = {}
    for row in clips:
        if row.clip_id in by_id:
            raise ValueError(f"Duplicate clip_id in clips manifest: {row.clip_id}")
        by_id[row.clip_id] = row
    votes = json.loads(Path(votes_path).read_text(encoding="utf-8-sig"))
    if not isinstance(votes, list):
        raise ValueError("Normalized Label Studio votes must be a JSON array")
    decisions = defaultdict(list)
    for vote in votes:
        if not isinstance(vote, dict) or not isinstance(vote.get("clip_id"), str):
            raise ValueError("Every normalized vote needs a clip_id")
        clip_id = vote["clip_id"]
        if clip_id not in by_id:
            raise ValueError(f"Vote clip_id is absent from clips manifest: {clip_id}")
        decision = vote.get("decision")
        if not isinstance(decision, str) or not decision:
            raise ValueError(f"Vote for {clip_id} needs a normalized decision")
        expected_status = "bad_clip" if decision == "bad_clip" else "accepted"
        if vote.get("status") != expected_status:
            raise ValueError(f"Vote for {clip_id} has inconsistent status/decision")
        decisions[clip_id].append(decision)
    output = []
    summary = Counter()
    for clip_id, row in by_id.items():
        values = set(decisions[clip_id])
        if not values:
            summary["unlabeled"] += 1
            continue
        if len(values) != 1:
            raise ValueError(f"Conflicting Label Studio decisions for {clip_id}: {sorted(values)}; adjudicate first")
        label = values.pop()
        if label in {"bad_clip", "unknown", "<unk>"}:
            summary[label] += 1
            continue
        if label not in labels:
            raise ValueError(f"Label {label!r} for {clip_id} is not in configured classes")
        output.append(replace(row, label=label))
        summary["accepted"] += 1
    if not output:
        raise ValueError("No unanimously labeled clips remain")
    write_manifest(output, output_path)
    return dict(summary)

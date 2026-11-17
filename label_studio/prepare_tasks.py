#!/usr/bin/env python3
"""Select one round of clips for human emotion labelling, copy their audio and write the Label Studio tasks file.

Clips come from several curated batches. Three model predictions per clip decide its stratum, so that a round holds
far more emotional clips than a random draw would. Annotators never see the predictions: they are kept only in the
round's manifest.parquet and summary.json.
Settings: label_studio/config/prepare_tasks.yaml. Run from the repository root: python3 label_studio/prepare_tasks.py
"""

from collections import Counter, defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import unicodedata
from urllib.parse import quote

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


CONFIG = Path(__file__).resolve().parent / "config" / "prepare_tasks.yaml"
REQUIRED = {"curated_root", "batches", "output_root"}
DEFAULTS = {
    "manifest_name": "manifest_with_predictions.parquet",
    "columns": {"clip_id": "clip_id", "audio": "audio_clip_file", "text": "text",
                "long_clip": "is_longer_than_10_seconds", "call_id": "external_call_id",
                "predictions": ["pred_e2v", "pred_gpt", "pred_gemini"]},
    "quotas": {"random": 150, "emotion": 560, "conflict": 140, "neutral": 150},
    "max_clips_per_call": 3,
    "files_per_folder": 100,
    "seed": 42,
}
EMOTIONS = ("angry", "disgusted", "fearful", "happy", "sad", "surprised")  # the 6 non-neutral classes
VOTES = set(EMOTIONS) | {"neutral"}
NO_VOTE = {"", "invalid", "other", "unknown", "none"}  # the model gave no usable label
TIERS = ("agree3", "agree2", "single")   # an emotion's target is split evenly over these, in this order
REFILL = ("agree2", "single", "agree3")  # a tier's shortfall is taken from the same emotion's tiers, in this order
EXTRA_COLUMNS = ("source_batch", "layer", "target_class", "stratum", "task_index", "part", "copied_audio")
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
# Label Studio stores an upload as upload/<project>/<8 hex>-<name> in a 100-character field and cuts longer
# names, which then no longer match tasks_NNN.json. Keep names at or below this length.
MAX_NAME_LENGTH = 70
ROUND = re.compile(r"tasks_(\d{3,})")


def step(number, title):
    print(f"\n== Step {number}: {title}")


def show(header, rows):
    """Print a small table: first column left-aligned, the others right-aligned."""
    lines = [[str(cell) for cell in line] for line in [header, *rows]]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    for line in lines:
        print("  " + "  ".join(cell.ljust(w) if i == 0 else cell.rjust(w)
                               for i, (cell, w) in enumerate(zip(line, widths))))


def listed(items, limit=10):
    more = f"\n  … and {len(items) - limit} more" if len(items) > limit else ""
    return "\n  " + "\n  ".join(items[:limit]) + more


def django_name(name):
    """The file name Label Studio (Django get_valid_filename) keeps for an uploaded file."""
    return re.sub(r"(?u)[^-\w.]", "", str(name).strip().replace(" ", "_"))


def name_keys(name):
    """Case-folded names Label Studio may store; a Mac may send accents decomposed (NFD), which Django drops."""
    forms = (name, django_name(name), django_name(unicodedata.normalize("NFD", name)))
    return {form.casefold() for form in forms}


def positive_int(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def plain_name(value):
    return isinstance(value, str) and value.strip() == value and value not in ("", ".", "..") \
        and "/" not in value and "\\" not in value


def load_settings(path):
    return check_settings(yaml.safe_load(Path(path).read_text(encoding="utf-8")), path)


def check_settings(settings, source="settings"):
    """Merge the defaults into the settings and check every value; returns a new dict (safe to call twice)."""
    if not isinstance(settings, dict):
        raise ValueError(f"{source}: expected 'key: value' settings")
    missing, unknown = REQUIRED - settings.keys(), settings.keys() - REQUIRED - DEFAULTS.keys()
    if missing or unknown:
        raise ValueError(f"{source}: missing keys {sorted(missing)}; unknown keys {sorted(unknown)}")
    merged = {**DEFAULTS, **settings}
    for key in ("columns", "quotas"):
        if not isinstance(merged[key], dict) or merged[key].keys() - DEFAULTS[key].keys():
            raise ValueError(f"{source}: {key} must be a mapping with the keys {sorted(DEFAULTS[key])}")
        merged[key] = {**DEFAULTS[key], **merged[key]}
    for key in ("curated_root", "output_root"):
        if not isinstance(merged[key], str) or not merged[key].strip():
            raise ValueError(f"{source}: {key} must be a folder path")
    if not plain_name(merged["manifest_name"]):
        raise ValueError(f"{source}: manifest_name must be a file name, such as manifest_with_predictions.parquet")
    batches = merged["batches"]
    if not isinstance(batches, list) or not batches or not all(plain_name(b) for b in batches) \
            or len(set(batches)) != len(batches):
        raise ValueError(f"{source}: batches must be a list of distinct folder names under curated_root, "
                         "such as [batch_01, batch_02]")
    columns = merged["columns"]
    if not all(isinstance(columns[k], str) and columns[k] for k in DEFAULTS["columns"] if k != "predictions"):
        raise ValueError(f"{source}: every entry in columns must be a column name")
    predictions = columns["predictions"]
    if not isinstance(predictions, list) or not all(isinstance(p, str) and p for p in predictions) \
            or len(predictions) != 3 or len(set(predictions)) != 3:
        raise ValueError(f"{source}: columns.predictions must list 3 different column names")
    quotas = merged["quotas"]
    if not all(positive_int(quotas[k], 0) for k in quotas) or sum(quotas.values()) == 0:
        raise ValueError(f"{source}: quotas must be whole numbers of 0 or more, with a sum above 0")
    for key in ("max_clips_per_call", "files_per_folder"):
        if not positive_int(merged[key]):
            raise ValueError(f"{source}: {key} must be a positive whole number")
    if not isinstance(merged["seed"], int) or isinstance(merged["seed"], bool):
        raise ValueError(f"{source}: seed must be a whole number")
    return merged


def normalize(value):
    """A prediction as stripped lowercase text; null and NaN become ''."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def vote(value):
    """'Angry ' -> 'angry'; null, NaN, '', invalid, other, unknown, none -> None (no vote); else ValueError."""
    label = normalize(value)
    if label in NO_VOTE:
        return None
    if label not in VOTES:
        raise ValueError(f"unexpected label {label!r}")
    return label


def stratum(votes):
    """Stratum of one clip from its normalized votes (None = no vote); only valid votes count."""
    valid = [label for label in votes if label is not None]
    if len(valid) < 2:
        return "unassigned"
    emotional = Counter(label for label in valid if label != "neutral")
    if not emotional:
        return "neutral"
    label, count = emotional.most_common(1)[0]
    if count >= 2:
        return f"agree{count}:{label}"
    return f"single:{label}" if len(emotional) == 1 else "conflict"


def strata_names():
    return [f"{tier}:{emotion}" for emotion in EMOTIONS for tier in TIERS] + ["conflict", "neutral", "unassigned"]


def long_clip_flag(value, context):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().casefold() in {"true", "false", "1", "0"}:
        return value.strip().casefold() in {"true", "1"}
    raise ValueError(f"{context}: expected true/false or 1/0; missing or invalid long-clip flags are not allowed")


def clip_id_text(value, context):
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: clip ID must be a non-empty string or integer")
    return value


def call_id_text(value):
    """Call IDs as text; null, NaN and blank become ''."""
    if value is None or isinstance(value, bool) or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def transcript(value, context):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{context}: transcript must be text or null")
    return value.strip()


def audio_path(value, folder, roots, context):
    """Where the audio should be: a bare name in <batch>/audio/, a relative path from the batch folder."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: audio file must be a non-empty file name or path")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = (folder / "audio" / path) if len(path.parts) == 1 else (folder / path)
    path = Path(os.path.normpath(path))
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError(f"{context}: audio path is outside curated_root: {value}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"{context}: unsupported audio extension {path.suffix!r}: {value}")
    return path


def check_names(pool):
    """Label Studio matches uploads by file name, so names must differ ignoring case and the characters it drops."""
    too_long = [f"{clip['batch']}/{clip['name']}" for clip in pool if len(django_name(clip["name"])) > MAX_NAME_LENGTH]
    if too_long:
        raise ValueError(f"{len(too_long)} audio file names are longer than {MAX_NAME_LENGTH} characters; Label "
                         f"Studio would shorten them on upload:" + listed(too_long))
    groups = defaultdict(dict)
    for clip in pool:
        for key in name_keys(clip["name"]):
            groups[key][clip["clip_id"]] = f"{clip['batch']}/{clip['name']}"
    clashes = sorted({" = ".join(sorted(group.values())) for group in groups.values() if len(group) > 1})
    if clashes:
        raise ValueError(f"{len(clashes)} audio file names are not unique among the eligible clips (ignoring case "
                         "and the characters Label Studio drops):" + listed(clashes))


def split(total, parts):
    """Split total into parts as evenly as possible; the first parts get the remainder."""
    return [total // parts + (index < total % parts) for index in range(parts)]


def emotion_targets(total):
    """{'angry': {'agree3': 32, 'agree2': 31, 'single': 31}, …}; emotions in EMOTIONS order get the remainder."""
    return {emotion: dict(zip(TIERS, split(count, len(TIERS))))
            for emotion, count in zip(EMOTIONS, split(total, len(EMOTIONS)))}


def earlier_rounds(output):
    """Next round number, the earlier round folders and every clip ID already used in them."""
    rounds, used = {}, set()
    if output.is_dir():
        leftovers = sorted(path.name for path in output.glob("tasks_*.partial"))
        if leftovers:
            raise ValueError(f"Unfinished round folder {output / leftovers[0]}: an earlier run was interrupted. "
                             "Delete that folder, then run again")
        for folder in sorted(output.iterdir()):
            match = ROUND.fullmatch(folder.name)
            if not match or not folder.is_dir():
                continue
            tasks_file = folder / f"{folder.name}.json"
            try:
                ids = [task["data"]["clip_id"] for task in json.loads(tasks_file.read_text(encoding="utf-8"))]
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"Cannot read the clip IDs of earlier round {tasks_file}: {exc}") from None
            rounds[int(match.group(1))] = (folder.name, len(ids))
            used.update(ids)
    return 1 + max(rounds, default=0), [rounds[number] for number in sorted(rounds)], used


def read_manifest(path, needed):
    """The whole manifest table (all columns are kept in the round's manifest) and the needed columns as rows."""
    try:
        table = pq.read_table(path)
    except pa.ArrowException as exc:
        raise ValueError(f"Cannot read Parquet manifest {path}: {exc}") from None
    names = table.column_names
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate column names")
    missing = [column for column in dict.fromkeys(needed) if column not in names]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; available columns: {', '.join(names)}")
    clash = [column for column in EXTRA_COLUMNS if column in names]
    if clash:
        raise ValueError(f"{path}: columns {clash} are added by this script and must not already exist")
    return table, table.select(list(dict.fromkeys(needed))).to_pylist()


class Draw:
    """Takes clips at random, at most once each and at most `cap` per call, recording why each was taken."""

    def __init__(self, cap, rng):
        self.cap, self.rng = cap, rng
        self.taken, self.per_call = [], Counter()
        self.ids = set()

    def take(self, candidates, count, layer, target=None):
        if count <= 0:
            return 0
        order = list(candidates)  # candidates are in clip_id order, so the shuffle is reproducible
        self.rng.shuffle(order)
        drawn = 0
        for clip in order:
            if drawn == count:
                break
            if clip["clip_id"] in self.ids or self.per_call[clip["call"]] >= self.cap:
                continue  # already taken, or its call is full
            clip["layer"], clip["target_class"] = layer, target
            self.taken.append(clip)
            self.ids.add(clip["clip_id"])
            self.per_call[clip["call"]] += 1
            drawn += 1
        return drawn


def select(pool, quotas, cap, rng):
    """Draw one round: random, emotion (with the shortfall cascade), conflict, neutral, then fill; shuffle."""
    strata = defaultdict(list)
    for clip in pool:
        strata[clip["stratum"]].append(clip)
    draw = Draw(cap, rng)
    targets = {"random": quotas["random"], "emotion": emotion_targets(quotas["emotion"]),
               "conflict": quotas["conflict"], "neutral": quotas["neutral"]}
    drawn = {"random": 0, "emotion": {e: dict.fromkeys(TIERS, 0) for e in EMOTIONS},
             "conflict": 0, "neutral": 0, "fill": 0}
    moves = []

    # (a) Random layer: a uniform sample of the whole pool, as a check on the real distribution.
    drawn["random"] = draw.take(pool, quotas["random"], "random")
    if drawn["random"] < quotas["random"]:
        moves.append({"from": "random", "missing": quotas["random"] - drawn["random"], "to": "fill"})

    # (b) Emotion layer: per emotion and tier; a tier shortfall comes from the same emotion's other tiers,
    # and what the emotion still lacks moves to the conflict target.
    to_conflict = 0
    for emotion in EMOTIONS:
        short = 0
        for tier in TIERS:
            target = targets["emotion"][emotion][tier]
            got = draw.take(strata[f"{tier}:{emotion}"], target, "emotion", emotion)
            drawn["emotion"][emotion][tier] += got
            if got < target:
                moves.append({"from": f"emotion {tier}:{emotion}", "missing": target - got,
                              "to": f"other tiers of {emotion}"})
                short += target - got
        for tier in REFILL:
            got = draw.take(strata[f"{tier}:{emotion}"], short, "emotion", emotion)
            drawn["emotion"][emotion][tier] += got
            short -= got
        if short:
            moves.append({"from": f"emotion {emotion}", "missing": short, "to": "conflict"})
            to_conflict += short

    # (c) Conflict and (d) all-neutral layers; their shortfalls go to the fill.
    targets["conflict_with_moves"] = quotas["conflict"] + to_conflict
    for layer, target in (("conflict", targets["conflict_with_moves"]), ("neutral", quotas["neutral"])):
        drawn[layer] = draw.take(strata[layer], target, layer, "neutral" if layer == "neutral" else None)
        if drawn[layer] < target:
            moves.append({"from": layer, "missing": target - drawn[layer], "to": "fill"})

    # (e) Fill: whatever is still missing, uniformly from the rest of the pool.
    targets["fill"] = sum(quotas.values()) - len(draw.taken)
    drawn["fill"] = draw.take(pool, targets["fill"], "fill")

    # Mix the layers, so the task order says nothing about how a clip was chosen.
    selected = list(draw.taken)
    rng.shuffle(selected)
    return selected, {"targets": targets, "drawn": drawn, "moves": moves, "calls": len(draw.per_call)}


def check_files(selected, curated):
    """Every selected audio file must exist and really lie inside curated_root (no symlink escapes)."""
    missing, outside = [], []
    for clip in selected:
        label = f"{clip['path']} (clip {clip['clip_id']})"
        try:
            real = clip["path"].resolve(strict=True)
        except OSError:
            missing.append(label)
            continue
        if not real.is_relative_to(curated):
            outside.append(f"{label} -> {real}")
        elif not real.is_file():
            missing.append(label)
        clip["source"] = real
    if missing:
        raise ValueError(f"{len(missing)} selected audio files are missing; nothing was created:" + listed(missing))
    if outside:
        raise ValueError(f"{len(outside)} selected audio files resolve outside curated_root (symlink?); "
                         "nothing was created:" + listed(outside))


def place(selected, name, size):
    """Task order -> task_index, part folder and the copy's path relative to output_root."""
    width = max(2, len(str(math.ceil(len(selected) / size))))
    for index, clip in enumerate(selected, 1):
        clip["task_index"] = index
        clip["part"] = f"part_{(index - 1) // size + 1:0{width}d}"
        clip["copied_audio"] = f"{name}/audio/{clip['part']}/{clip['name']}"
    return dict(sorted(Counter(clip["part"] for clip in selected).items()))


def as_text(table, names):
    for name in names:
        if name in table.column_names:
            table = table.set_column(table.column_names.index(name), name, table.column(name).cast(pa.string()))
    return table


def round_manifest(selected, tables):
    """Selected rows with all their original columns, plus how and where each clip was used; in task order.

    Returns the table and the columns kept as text because the batches disagree on their type."""
    by_batch = defaultdict(list)
    for clip in selected:
        by_batch[clip["batch"]].append(clip)
    parts = []
    for batch, clips in by_batch.items():
        table = tables[batch].take(pa.array([clip["row"] for clip in clips], type=pa.int64()))
        values = {"source_batch": [batch] * len(clips)}
        values.update({column: [clip[column] for clip in clips] for column in EXTRA_COLUMNS[1:]})
        for column in EXTRA_COLUMNS:
            kind = pa.int64() if column == "task_index" else pa.string()
            table = table.append_column(column, pa.array(values[column], type=kind))
        parts.append(table)
    mixed = []
    try:
        table = pa.concat_tables(parts, promote_options="permissive")
    except pa.ArrowException:
        # E.g. numbers in one batch and text in another: keep such columns as text rather than stop.
        types = defaultdict(set)
        for part in parts:
            for field in part.schema:
                types[field.name].add(str(field.type))
        mixed = sorted(name for name, kinds in types.items() if len(kinds) > 1)
        try:
            table = pa.concat_tables([as_text(part, mixed) for part in parts], promote_options="permissive")
        except pa.ArrowException as exc:
            raise ValueError(f"The batch manifests have incompatible column types: {exc}") from None
    return table.sort_by("task_index").replace_schema_metadata(None), mixed


def write_round(output, name, selected, manifest, summary):
    """Copy the audio and write the files into <name>.partial/, then rename it; remove it on any error."""
    final, partial = output / name, output / f"{name}.partial"
    if final.exists():
        raise ValueError(f"{final} already exists; it is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        tasks = []
        for index, clip in enumerate(selected, 1):
            target = partial / "audio" / clip["part"] / clip["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip["source"], target)  # a copy: the original stays untouched
            # Only what annotators may see: no predictions, strata or votes.
            tasks.append({"data": {"audio": "/data/local-files/?d=" + quote(clip["copied_audio"], safe="/"),
                                   "transcript": clip["text"], "clip_id": clip["clip_id"]}})
            if index % 200 == 0:
                print(f"  copied {index}/{len(selected)}")
        text = json.dumps(tasks, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (partial / f"{name}.json").write_text(text, encoding="utf-8")
        pq.write_table(manifest, partial / "manifest.parquet")
        (partial / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        if final.exists():
            raise ValueError(f"{final} appeared while this run was writing; nothing was changed")
        os.rename(partial, final)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(settings):
    # 1. Settings and folders: the two roots apart, every batch folder and manifest really inside curated_root.
    step(1, "Check settings and folders")
    settings = check_settings(settings)
    columns, quotas = settings["columns"], settings["quotas"]
    total = sum(quotas.values())
    given = Path(os.path.normpath(Path(settings["curated_root"]).expanduser().absolute()))
    curated = given.resolve()
    if not curated.is_dir():
        raise ValueError(f"curated_root is not a folder: {curated}")
    output = Path(settings["output_root"]).expanduser().resolve()
    if output.is_relative_to(curated) or curated.is_relative_to(output):
        raise ValueError(f"output_root and curated_root must be separate folders, neither inside the other: "
                         f"{output}, {curated}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output_root is not a folder: {output}")
    folders = {}
    for batch in settings["batches"]:
        folder = (curated / batch).resolve()
        manifest = (folder / settings["manifest_name"]).resolve()
        if not folder.is_dir() or not folder.is_relative_to(curated):
            raise ValueError(f"Batch folder not found inside curated_root: {curated / batch}")
        if not manifest.is_file() or not manifest.is_relative_to(curated):
            raise ValueError(f"{batch}: {settings['manifest_name']} not found in {folder}")
        folders[batch] = (folder, manifest)
    print(f"Curated root: {curated} ({len(folders)} batches)\nOutput root: {output}")
    print("Quotas: " + ", ".join(f"{layer} {count}" for layer, count in quotas.items())
          + f" = {total} clips per round; at most {settings['max_clips_per_call']} clips per call; "
          f"{settings['files_per_folder']} files per part folder")

    # 2. The next round number; clips of earlier rounds are never chosen again.
    step(2, "Find the next round")
    number, rounds, used = earlier_rounds(output)
    name = f"tasks_{number:03d}"
    rng_seed = f"{settings['seed']}:{number}"
    for folder_name, count in rounds:
        print(f"  earlier round {folder_name}: {count} clips")
    print(f"This run makes round {number} ({name}); random seed {rng_seed!r}")

    # 3. Read every manifest; check clip IDs, long-clip flags and prediction labels on all rows.
    step(3, "Read the batch manifests")
    needed = [columns[key] for key in ("clip_id", "audio", "text", "long_clip", "call_id")] + columns["predictions"]
    tables, counts, pool, seen = {}, {}, [], {}
    duplicates, unexpected = [], defaultdict(dict)
    for batch, (folder, manifest) in folders.items():
        tables[batch], rows = read_manifest(manifest, needed)
        count = counts[batch] = {"rows": len(rows), "long_excluded": 0, "earlier_rounds": 0, "eligible": 0}
        for index, row in enumerate(rows):
            context = f"{batch}, row {index + 1}"
            clip_id = clip_id_text(row[columns["clip_id"]], context)
            if clip_id in seen:
                duplicates.append(f"{clip_id} ({seen[clip_id]}, {batch})")
            seen.setdefault(clip_id, batch)
            votes = []
            for column in columns["predictions"]:
                try:
                    votes.append(vote(row[column]))
                except ValueError:
                    unexpected[column].setdefault(normalize(row[column]), batch)
                    votes.append(None)
            if long_clip_flag(row[columns["long_clip"]], f"{context}, {columns['long_clip']}"):
                count["long_excluded"] += 1
            elif clip_id in used:
                count["earlier_rounds"] += 1
            else:
                count["eligible"] += 1
                pool.append({"clip_id": clip_id, "batch": batch, "row": index, "folder": folder, "votes": votes,
                             "audio": row[columns["audio"]], "text": row[columns["text"]],
                             "call": call_id_text(row[columns["call_id"]])})
    if duplicates:
        raise ValueError(f"{len(duplicates)} clip IDs appear more than once in the listed batches; clip_id must "
                         "be unique:" + listed(duplicates))
    if unexpected:
        lines = [f"{column}: " + ", ".join(repr(label) for label in list(labels)[:10])
                 + (f" ({len(labels)} different values)" if len(labels) > 10 else "")
                 for column, labels in unexpected.items()]
        raise ValueError("Unexpected labels in the prediction columns (merge bug?). Valid: "
                         f"{', '.join(EMOTIONS)}, neutral; no vote: empty, invalid, other, unknown, none."
                         + listed(lines))
    show(["batch", "rows", "long excluded", "earlier rounds", "eligible"],
         [[batch, c["rows"], c["long_excluded"], c["earlier_rounds"], c["eligible"]] for batch, c in counts.items()]
         + [["total"] + [sum(c[key] for c in counts.values()) for key in counts[next(iter(counts))]]])

    # 4. The eligible pool: call IDs, audio paths, transcripts, file names; then the strata.
    step(4, "Check the eligible clips and sort them into strata")
    pool.sort(key=lambda clip: clip["clip_id"])  # canonical order: manifest row order must not matter
    roots = {given, curated}
    no_call = []
    for clip in pool:
        context = f"{clip['batch']}, clip {clip['clip_id']}"
        if not clip["call"]:
            no_call.append(context)
        clip["path"] = audio_path(clip["audio"], clip["folder"], roots, context)
        clip["name"] = clip["path"].name
        clip["text"] = transcript(clip["text"], context)
        clip["stratum"] = stratum(clip["votes"])
    if no_call:
        raise ValueError(f"{len(no_call)} eligible clips have an empty call ID ({columns['call_id']}):"
                         + listed(no_call))
    check_names(pool)
    votes = {column: Counter(clip["votes"][i] or "no vote" for clip in pool)
             for i, column in enumerate(columns["predictions"])}
    print("Votes of the eligible clips per model:")
    show(["label", *columns["predictions"]],
         [[label, *(votes[column][label] for column in columns["predictions"])]
          for label in [*EMOTIONS, "neutral", "no vote"]])
    sizes = Counter(clip["stratum"] for clip in pool)
    pool_sizes = {key: sizes[key] for key in strata_names()}
    print("Eligible clips per stratum:")
    show(["emotion", *TIERS], [[e, *(sizes[f"{t}:{e}"] for t in TIERS)] for e in EMOTIONS])
    print(f"  conflict {sizes['conflict']}; neutral {sizes['neutral']}; unassigned (fewer than 2 votes) "
          f"{sizes['unassigned']}; eligible pool {len(pool)}")
    if len(pool) < total:
        raise ValueError(f"The eligible pool has {len(pool)} clips, fewer than one round of {total} (the sum of "
                         "quotas). Add batches or lower the quotas; nothing was created")

    # 5. Draw the round.
    step(5, "Select the clips")
    selected, report = select(pool, quotas, settings["max_clips_per_call"], random.Random(rng_seed))
    targets, drawn = report["targets"], report["drawn"]
    if len(selected) < total:
        calls = len({clip["call"] for clip in pool})
        raise ValueError(f"Only {len(selected)} of {total} clips could be selected: the pool has {len(pool)} clips "
                         f"from {calls} calls, and at most max_clips_per_call={settings['max_clips_per_call']} "
                         "clips per call are used. Raise max_clips_per_call, add batches or lower the quotas; "
                         "nothing was created")
    emotion_total = sum(sum(tiers.values()) for tiers in drawn["emotion"].values())
    lines = [["random", targets["random"], drawn["random"], ""],
             ["emotion", quotas["emotion"], emotion_total, ""]]
    for emotion in EMOTIONS:
        got = drawn["emotion"][emotion]
        lines.append([f"  {emotion}", sum(targets["emotion"][emotion].values()), sum(got.values()),
                      " / ".join(str(got[tier]) for tier in TIERS)])
    lines += [["conflict", targets["conflict_with_moves"], drawn["conflict"],
              f"{quotas['conflict']} + {targets['conflict_with_moves'] - quotas['conflict']} moved"],
              ["neutral", targets["neutral"], drawn["neutral"], ""],
              ["fill", targets["fill"], drawn["fill"], "rest of the pool"],
              ["total", total, len(selected), f"from {report['calls']} calls"]]
    show(["layer", "target", "drawn", "agree3 / agree2 / single"], lines)
    print("Shortfalls:" if report["moves"] else "Shortfalls: none")
    for move in report["moves"]:
        print(f"  {move['from']}: {move['missing']} missing -> {move['to']}")
    blanks = sum(1 for clip in selected if not clip["text"])
    print(f"Blank transcripts in this round: {blanks} (written as \"\")")

    # 6. Every selected file must exist, and the round's manifest must build, before anything is created.
    step(6, "Check the selected audio files and build the round's manifest")
    check_files(selected, curated)
    size_mb = sum(clip["source"].stat().st_size for clip in selected) / 1e6
    print(f"All {len(selected)} files found; {size_mb:.1f} MB to copy")
    parts = place(selected, name, settings["files_per_folder"])
    manifest, mixed = round_manifest(selected, tables)
    if mixed:
        print(f"Note: the batches disagree on the type of {mixed}; manifest.parquet keeps these columns as text")

    # 7. Copy and write into tasks_NNN.partial/, then rename it to tasks_NNN/.
    step(7, f"Write {name}")
    summary = {"round": number, "folder": name, "created": datetime.now().astimezone().isoformat(timespec="seconds"),
               "seed": settings["seed"], "random_seed": rng_seed, "settings": settings,
               "earlier_rounds": [folder_name for folder_name, _ in rounds], "batches": counts,
               "eligible_pool": len(pool), "pool_sizes": pool_sizes, "votes": {c: dict(v) for c, v in votes.items()},
               "targets": targets, "drawn": drawn, "shortfalls": report["moves"], "calls": report["calls"],
               "transcript_blanks": blanks, "tasks": len(selected), "parts": parts,
               "columns_kept_as_text": mixed}
    final = write_round(output, name, selected, manifest, summary)
    names, counts_per_part = list(parts), list(parts.values())
    print(f"  audio/{names[0]} … {names[-1]}: {counts_per_part[0]} files each"
          + (f" ({names[-1]}: {counts_per_part[-1]})" if counts_per_part[-1] != counts_per_part[0] else ""))
    print(f"Wrote {final}: {name}.json ({len(selected)} tasks), manifest.parquet, summary.json")

    # 8. What to do in Label Studio.
    step(8, "Next")
    print(f"Server Label Studio (upload the audio):\n"
          f"  1. Labeling template: config_audio_only.xml.\n"
          f"  2. Import → Upload Files: upload one part folder at a time, {final}/audio/{names[0]} … "
          f"{names[-1]}.\n"
          f"  3. In label_studio/config/attach_task_data.yaml set tasks_file: {final / (name + '.json')}\n"
          f"     then run: python3 label_studio/attach_task_data.py (dry run first).\n"
          f"  4. Switch the template to config_with_transcript.xml, then let the annotators start.\n"
          f"Local Label Studio: the Local Files document root is {output};\n"
          f"  Source path {final}, import method Tasks, file filter ^{name}\\.json$.")


def main():
    try:
        run(load_settings(CONFIG))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()

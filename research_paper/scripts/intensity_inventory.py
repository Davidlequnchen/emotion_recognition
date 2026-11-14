"""Preserve native emotion-strength annotations on the saved benchmark cohort.

Intensity scales are dataset-specific, never pooled or inferred from loudness,
class probability, rater agreement, or a model's arousal output. No inference.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import consensus


FIELDS = ("dataset", "audio_path", "gold", "intensity_available", "intensity_applicable",
          "intensity_dataset_scale", "intensity_scale_kind", "intensity_level", "intensity_name",
          "intensity_raw", "intensity_source", "intensity_missing_reason", "intensity_applicability_reason",
          "intensity_pair_key")
SCALES = {
    "crema_d": {"scale": "crema_d_acted_LO_MD_HI", "kind": "acted_ordinal", "levels": [1, 2, 3],
        "source": "filename fourth field LO/MD/HI; XX means unspecified",
        "pair_rule": "Actor + published IEO sentence ID + emotion; identical sentence across acted levels.",
        "url": "https://github.com/CheyneyComputerScience/CREMA-D",
        "notes": "Only IEO has LO/MD/HI; all neutral and other sentence clips are XX. Native perceived VoiceLevel is a separate scale, not substituted."},
    "ravdess": {"scale": "ravdess_acted_normal_strong", "kind": "acted_ordinal", "levels": [1, 2],
        "source": "filename fourth numeric field: 01 normal, 02 strong",
        "pair_rule": "Actor + native emotion + published statement ID + repetition; identical statement across levels.",
        "url": "https://zenodo.org/records/1188976",
        "notes": "Native neutral has only normal intensity. Calm maps to benchmark neutral; both are excluded from primary non-neutral intensity comparisons."},
    "mead": {"scale": "mead_acted_level_1_3", "kind": "acted_ordinal", "levels": [1, 2, 3],
        "source": "audio/<actor>/<emotion>/level_<1|2|3>/<sentence>.wav",
        "pair_rule": "Actor + emotion + file index only. The corpus describes shared sentences across levels, but index-to-transcript equivalence was not independently verified locally.",
        "url": "https://github.com/uniBruce/Mead",
        "notes": "Neutral is a single level_1 reference, not weak emotional acting; current 8,000-clip stratified scope is retained."},
    "thai_ser": {"scale": "thai_ser_script_Low_High", "kind": "acted_ordinal", "levels": [1, 2],
        "source": "data/thai_ser/labels.csv: script_intensity",
        "pair_rule": "Session + actor + exact published script_sent text + script_trial + assigned emotion; intensity excluded from key.",
        "url": "https://huggingface.co/datasets/airesearch/thai-ser",
        "notes": "Raw values are Low/High; the author README describes normal/strong. Improvised turns have no script intensity; agreement is not intensity."},
    "mcae_spps": {"scale": "mcae_spps_listener_mean_1_9", "kind": "perceived_continuous_mean", "range": [1, 9],
        "source": "data/mcae_spps/Corpus information.xlsx: Intensity",
        "url": "https://osf.io/9jyzc/",
        "notes": "Mean of 40 listeners' 1-9 emotional-strength ratings. Intensity is not meanIntensity (the latter is acoustic dB). No low/high bins are invented."},
    "dailytalk": {"scale": "dailytalk_ECSS_1_3", "kind": "annotated_ordinal", "levels": [1, 2, 3],
        "source": "data/dailytalk/utterances.csv: ecss_intensity",
        "url": "https://github.com/walker-hyf/ECSS",
        "notes": "Additional public ECSS intensity annotation, not randomized acting strength. Published levels1-3 are retained verbatim; neutral is excluded from primary non-neutral comparisons."},
}
CANONICAL = {"neutral": "neutral", "none": "neutral", "no emotion": "neutral", "calm": "neutral",
             "angry": "angry", "anger": "angry", "happy": "happy", "happiness": "happy",
             "sad": "sad", "sadness": "sad", "fear": "fearful", "fearful": "fearful",
             "disgust": "disgusted", "disgusted": "disgusted", "surprise": "surprised", "surprised": "surprised"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _canonical(value):
    label = CANONICAL.get(str(value).strip().lower())
    if label is None:
        raise ValueError(f"Unknown source emotion {value!r}")
    return label


def _csv_index(path, key):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        identity = row[key]
        if not identity or identity in result:
            raise ValueError(f"{path}: empty or duplicated metadata key {identity!r}")
        result[identity] = row
    return result


def _mcae_index(path):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook["Sheet1"]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
        columns = {str(value): i for i, value in enumerate(header)}
        if not {"Filename", "Intensity", "emotion"} <= set(columns):
            raise ValueError(f"{path}: missing Filename/Intensity/emotion columns")
        result = {}
        for values in rows:
            if all(value is None for value in values):
                continue
            number = values[columns["Filename"]]
            if not isinstance(number, (str, int, float)) or str(number).strip() == "":
                raise ValueError(f"{path}: missing filename")
            filename = str(int(number)).zfill(5) + ".wav"
            if filename in result:
                raise ValueError(f"{path}: duplicated filename {filename}")
            value = float(values[columns["Intensity"]])
            if not math.isfinite(value) or not 1 <= value <= 9:
                raise ValueError(f"{path}: intensity outside 1-9")
            result[filename] = {"value": value, "gold": _canonical(values[columns["emotion"]])}
        return result
    finally:
        workbook.close()


def _filename_intensity(dataset, audio_path, gold):
    path = Path(audio_path)
    if dataset == "crema_d":
        match = re.fullmatch(r"(?P<actor>\d{4})_(?P<sentence>[A-Z]{3})_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<intensity>LO|MD|HI|XX)", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized CREMA-D filename: {audio_path}")
        native_gold = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}[match["emotion"]]
        raw = match["intensity"]
        if raw == "XX":
            return None, "", raw, "unspecified_XX", "", native_gold
        if match["sentence"] != "IEO" or native_gold == "neutral":
            raise ValueError(f"Unexpected explicit CREMA-D intensity scope: {audio_path}")
        value, name = {"LO": (1, "low"), "MD": (2, "medium"), "HI": (3, "high")}[raw]
        pair = f"crema_d:{match['actor']}:{match['sentence']}:{match['emotion']}"
    elif dataset == "ravdess":
        match = re.fullmatch(r"03-01-(?P<emotion>0[1-8])-(?P<intensity>0[12])-(?P<sentence>0[12])-(?P<repeat>0[12])-(?P<actor>0[1-9]|1\d|2[0-4])", path.stem)
        if match is None:
            raise ValueError(f"Unrecognized RAVDESS speech filename: {audio_path}")
        native_gold = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fearful", "07": "disgusted", "08": "surprised"}[match["emotion"]]
        raw = match["intensity"]
        if match["emotion"] == "01" and raw != "01":
            raise ValueError("RAVDESS native neutral cannot have strong intensity")
        value, name = {"01": (1, "normal"), "02": (2, "strong")}[raw]
        pair = f"ravdess:{match['actor']}:{match['emotion']}:{match['sentence']}:{match['repeat']}"
    else:
        match = re.fullmatch(r"(?P<actor>[MW]\d{3})/(?P<emotion>angry|disgusted|fear|happy|sad|surprised|neutral)/level_(?P<intensity>[123])/(?P<sentence>\d{3})\.wav", "/".join(path.parts[-4:]))
        if match is None:
            raise ValueError(f"Unrecognized MEAD path: {audio_path}")
        native_gold = _canonical(match["emotion"])
        raw = match["intensity"]
        if native_gold == "neutral" and raw != "1":
            raise ValueError("MEAD neutral must retain its single level_1 reference")
        value, name = {"1": (1, "weak"), "2": (2, "medium"), "3": (3, "strong")}[raw]
        pair = f"mead:{match['actor']}:{match['emotion']}:{match['sentence']}"
    return value, name, raw, "", pair, native_gold


def audit_meralion_capabilities(project):
    """Inspect saved column names and local model/adapter code, never load tensors."""
    project = Path(project)
    native = project / "models/meralion_ser_v1/modeling_ser_whisper_ecapa.py"
    adapter = project / "scripts/models/meralion_ser.py"
    if not native.is_file() or not adapter.is_file():
        return {"available": False, "reason": "Native model or adapter source unavailable"}
    model_text, adapter_text = native.read_text(), adapter.read_text()
    headers = {}
    dimensional = []
    for path in sorted((project / "outputs/meralion_ser_v1").glob("*_predictions.csv")):
        with path.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream))
        headers[path.name] = header
        dimensional += [column for column in header if re.search(r"valence|arousal|dominance|^dims?$|^dim_", column, re.I)]
    return {"available": True, "native_output": {"logits": "seven categorical emotion scores",
            "dims": ["valence", "arousal", "dominance"], "dimensional_range": [0, 1],
            "generic_emotion_intensity_head": False},
            "code_returns_dims": '"dims": dim_output' in model_text,
            "adapter_selects_logits_only": '["logits"]' in adapter_text and '["dims"]' not in adapter_text,
            "saved_dimensional_columns": sorted(set(dimensional)), "saved_csv_headers": headers,
            "new_inference_required_to_recover_dims": not dimensional,
            "sources": [{"path": p.relative_to(project).as_posix(), "sha256": consensus.sha256(p)} for p in (native, adapter)],
            "model_card": "https://huggingface.co/MERaLiON/MERaLiON-SER-v1",
            "interpretation": "VAD is not an emotion-strength label. Arousal is calm-to-active; dominance weak-to-strong is control, not expressiveness. Neither class probability nor existing CSVs reconstruct native dims."}


def build_intensity_inventory(project, inventory_rows):
    project = Path(project).resolve()
    inventory_rows = list(inventory_rows)
    datasets = {row["dataset"] for row in inventory_rows}
    paths = {"thai_ser": project / "data/thai_ser/labels.csv",
             "mcae_spps": project / "data/mcae_spps/Corpus information.xlsx",
             "dailytalk": project / "data/dailytalk/utterances.csv"}
    sources = {dataset: {"path": path.relative_to(project).as_posix(), "sha256": consensus.sha256(path)}
               for dataset, path in paths.items() if dataset in datasets}
    metadata = {}
    for dataset in sources:
        metadata[dataset] = (_mcae_index(paths[dataset]) if dataset == "mcae_spps" else
                             _csv_index(paths[dataset], "audio_id" if dataset == "thai_ser" else "audio_path"))
    seen, rows = set(), []
    for original in inventory_rows:
        dataset, path, gold = original["dataset"], consensus.canonical_path(original["audio_path"]), original["gold"]
        if (dataset, path) in seen:
            raise ValueError(f"Duplicated inventory sample: {dataset}/{path}")
        seen.add((dataset, path))
        spec = SCALES.get(dataset)
        value, name, raw, reason, pair, native_gold = None, "", "", "no_native_intensity_in_current_inventory", "", gold
        if dataset in {"crema_d", "ravdess", "mead"}:
            value, name, raw, reason, pair, native_gold = _filename_intensity(dataset, path, gold)
        elif dataset == "mcae_spps":
            item = metadata[dataset].get(Path(path).name)
            if item is None:
                raise ValueError(f"MCAE intensity missing for {path}")
            value, name, raw, reason, native_gold = item["value"], "listener mean (1-9)", str(item["value"]), "", item["gold"]
        elif dataset == "thai_ser":
            item = metadata[dataset].get(Path(path).name)
            if item is None:
                raise ValueError(f"THAI-SER metadata missing for {path}")
            native_gold, raw = _canonical(item["assigned_emo"]), item["script_intensity"]
            if raw:
                if item["turn_type"] != "script" or raw not in {"Low", "High"}:
                    raise ValueError(f"Invalid THAI-SER script intensity for {path}")
                value, name, reason = {"Low": 1, "High": 2}[raw], raw.lower(), ""
                paired_fields = [item[k] for k in ["session_id", "actor_id", "script_sent", "script_trial", "assigned_emo"]]
                pair = "thai_ser:" + _digest(paired_fields)[:24]
            else:
                reason = "improvised_no_script_intensity"
        elif dataset == "dailytalk":
            relative = path.removeprefix("data/dailytalk/")
            item = metadata[dataset].get(relative)
            if item is None:
                raise ValueError(f"DailyTalk metadata missing for {path}")
            native_gold = _canonical(item["ecss_emotion"] or item["original_emotion"])
            raw = item["ecss_intensity"]
            if raw:
                if raw not in {"1", "2", "3"}:
                    raise ValueError(f"DailyTalk ECSS intensity outside 1-3: {path}")
                value, name, reason = int(raw), f"ECSS level {raw}", ""
            else:
                reason = "no_public_ecss_intensity"
        if native_gold != gold:
            raise ValueError(f"{path}: intensity metadata gold {native_gold!r} differs from inventory {gold!r}")
        rows.append({"dataset": dataset, "audio_path": path, "gold": gold,
            "intensity_available": value is not None, "intensity_applicable": value is not None and gold != "neutral",
            "intensity_dataset_scale": spec["scale"] if spec else "", "intensity_scale_kind": spec["kind"] if spec else "",
            "intensity_level": value, "intensity_name": name, "intensity_raw": raw,
            "intensity_source": spec["source"] if spec else "", "intensity_missing_reason": reason,
            "intensity_applicability_reason": ("no_native_annotation" if value is None else
                                               "benchmark_neutral_excluded" if gold == "neutral" else ""),
            "intensity_pair_key": pair})
    for dataset, source in sources.items():
        if consensus.sha256(paths[dataset]) != source["sha256"]:
            raise RuntimeError(f"{paths[dataset]}: intensity metadata changed")
    summary = {}
    for dataset in sorted(datasets):
        subset = [row for row in rows if row["dataset"] == dataset]
        available = [row for row in subset if row["intensity_available"]]
        summary[dataset] = {"clips": len(subset), "available": len(available),
            "non_neutral_applicable": sum(row["intensity_applicable"] for row in subset),
            "native_level_counts": dict(sorted(Counter(str(row["intensity_level"]) for row in available).items())),
            "missing_reasons": dict(Counter(row["intensity_missing_reason"] for row in subset if not row["intensity_available"]))}
    manifest = {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "clips": len(rows), "cohort_sha256": _digest(sorted((r["dataset"], r["audio_path"], r["gold"]) for r in rows)),
        "implementation_sha256": consensus.sha256(Path(__file__)), "scales": SCALES,
        "metadata_sources": list(sources.values()), "datasets": summary,
        "meralion_capabilities": audit_meralion_capabilities(project),
        "notes": ["Ordinal numbers are comparable only within their native dataset scale; no cross-dataset intensity regression is licensed by these labels.",
            "Intensity availability preserves neutral annotations, but intensity_applicable excludes benchmark-neutral labels in the primary emotional-strength comparison.",
            "Primary CREMA scale is acted LO/MD/HI, not perceived VoiceLevel. MCAE meanIntensity(dB), rater agreement and confidence weights are never substituted for emotion strength.",
            "Pair keys are documented separately per dataset. CREMA/RAVDESS/THAI use published sentence/text IDs; MEAD uses file index with unverified local lexical equivalence. They are not independent samples or a guarantee all levels survive the configured cohort."]}
    return rows, manifest


def write_intensity_inventory(rows, manifest, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "intensity_inventory.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = manifest | {"inventory_csv_sha256": consensus.sha256(csv_path)}
    (output_dir / "intensity_inventory_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return csv_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=consensus.REPO / "research_paper/output/audio_inventory.csv")
    args = parser.parse_args()
    with args.inventory.open(encoding="utf-8", newline="") as stream:
        inventory_rows = list(csv.DictReader(stream))
    rows, manifest = build_intensity_inventory(consensus.PROJECT, inventory_rows)
    target = write_intensity_inventory(rows, manifest, consensus.REPO / "research_paper/output")
    print(f"Intensity inventory: {len(rows):,} clips; {sum(row['intensity_available'] for row in rows):,} with native annotations -> {target}")


if __name__ == "__main__":
    main()

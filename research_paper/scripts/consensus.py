"""Compare fixed model groups on saved predictions; no models, audio or APIs are run.

From the repository root: python research_paper/scripts/consensus.py
From research_paper/: python scripts/consensus.py
Outputs: analysis_report/consensus_zh.md, analysis_report/consensus_en.md and output/consensus.csv.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "emotion_recognition"
for folder in ("common", "llm_eval"):
    sys.path.insert(0, str(PROJECT / "scripts" / folder))
from prompts import configured_protocol

# Prespecified groups: never search the same results for the best-performing combination.
GROUPS = {
    "SER5": {"models": ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion",
                         "vox_profile_whisper", "vox_profile_wavlm"], "require_all": True},
    "AudioLLM5": {"models": ["gpt_realtime", "gemini-3.8-flash", "qwen2_5_omni_7b",
                              "qwen2_audio_7b", "gemma4_e4b"], "require_all": False},
    "Selected3": {"models": ["emotion2vec_plus_large", "gemini-3.8-flash", "qwen2_5_omni_7b"],
                  "require_all": False},
}
RULES = {"unanimous": {"unanimous": True}, "majority": {"fraction": [1, 2], "strict": True}}
DATASETS = ("crema_d", "mcae_spps", "thai_ser", "nemo", "mess", "ravdess", "tess", "jl_corpus",
            "mead", "meld", "emovo", "emozionalmente", "savee", "emotiontalk", "dailytalk", "multidialog")
GROUP_NAMES = {"zh": {"SER5": "五个 SER 模型", "AudioLLM5": "Audio LLM（含 API）", "Selected3": "精选模型"},
               "en": {"SER5": "Five SER models", "AudioLLM5": "Audio LLMs (including APIs)", "Selected3": "Selected models"}}
NAMES = dict(zip(DATASETS, ("CREMA-D", "MCAE-SPPS", "THAI-SER", "nEMO", "MESS", "RAVDESS", "TESS",
                          "JL-Corpus", "MEAD", "MELD", "EMOVO", "Emozionalmente", "SAVEE",
                          "EmotionTalk", "DailyTalk", "MultiDialog")))
SHORT_NAMES = {"emotion2vec_plus_large": "emotion2vec+", "meralion_ser_v1": "MERaLiON", "melati_emotion": "Melati",
               "vox_profile_whisper": "Vox-Profile Whisper", "vox_profile_wavlm": "Vox-Profile WavLM",
               "gpt_realtime": "GPT Realtime", "gemini-3.8-flash": "Gemini 3.8 Flash", "qwen2_5_omni_7b": "Qwen2.5-Omni",
               "qwen2_audio_7b": "Qwen2-Audio", "gemma4_e4b": "Gemma 4"}
INVALID = frozenset({"invalid", "other answer"})
SAFE_NAME = re.compile(r"[a-zA-Z0-9_.-]+")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid audio_path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or ":" in path.parts[0]:
        raise ValueError(f"audio_path must be canonical and project-relative: {value!r}")
    return value


def labels(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(v, str) or not v or v.strip() != v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"{name} must contain distinct nonempty labels")
    return list(value)


@dataclass
class Run:
    model: str
    dataset: str
    path: Path
    metadata: dict
    classes: list
    accept: dict
    rows: dict
    sources: list


def read_run(path, model, registry, root):
    """Strict prediction/identity validation; correctness is recomputed, never read from a flag."""
    metadata_hash = sha256(path)
    metadata = json.loads(path.read_text(encoding="utf-8-sig"))
    key = metadata.get("key")
    if metadata.get("model") != model or not isinstance(key, str) or key in (".", "..") or not SAFE_NAME.fullmatch(key):
        raise ValueError(f"{path}: metadata model/dataset identity is invalid")
    stem = path.name.removesuffix("_predictions.json")
    if stem != key and not stem.startswith(key + "_"):
        raise ValueError(f"{path}: filename does not match dataset identity")
    classes = labels(metadata.get("classes"), "classes")
    accept = metadata.get("accept") or {}
    if not isinstance(accept, dict):
        raise ValueError(f"{path}: accept must be a mapping")
    for label, accepted in accept.items():
        if label in classes or not isinstance(label, str) or not label:
            raise ValueError(f"{path}: invalid composite ground-truth label")
        if not set(labels(accepted, f"accept[{label}]")) <= set(classes):
            raise ValueError(f"{path}: composite accepts a label outside the candidate set")
    expected = configured_protocol(registry[model], classes, config=root / "config/llm.yaml", root=root)
    if expected is not None and metadata.get("inference_protocol") != expected:
        return None, "incompatible_inference_protocol"
    if expected is None and metadata.get("inference_protocol") is not None:
        raise ValueError(f"{path}: non-LLM entry unexpectedly declares an LLM protocol")
    csv_path = path.with_suffix(".csv")
    before = {path: metadata_hash, csv_path: sha256(csv_path)}
    truth_column = metadata.get("ground_truth", "acted_emotion")
    rows = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {"audio_path", "prediction", truth_column} <= set(fields):
            raise ValueError(f"{csv_path}: missing or duplicated required columns")
        for number, row in enumerate(reader, 2):
            if None in row or any(row.get(k) is None for k in ("audio_path", "prediction", truth_column)):
                raise ValueError(f"{csv_path}:{number}: malformed row")
            audio = canonical_path(row["audio_path"])
            if audio in rows:
                raise ValueError(f"{csv_path}:{number}: duplicate audio_path {audio}")
            truth, prediction = row[truth_column], row["prediction"]
            if truth not in set(classes) | set(accept):
                raise ValueError(f"{csv_path}:{number}: unknown ground-truth label {truth!r}")
            if prediction not in classes and (prediction not in INVALID or not (
                    registry[model].get("api") or registry[model].get("llm"))):
                raise ValueError(f"{csv_path}:{number}: unknown prediction {prediction!r}")
            rows[audio] = {"gold": truth, "prediction": None if prediction in INVALID else prediction}
    if type(metadata.get("clips")) is not int or not rows or len(rows) != metadata["clips"]:
        raise ValueError(f"{path}: CSV count does not match positive metadata clips")
    sources = []
    for source, digest in before.items():
        if sha256(source) != digest:
            raise ValueError(f"{source}: source changed during validation")
        sources.append({"path": source.relative_to(root).as_posix(), "sha256": digest,
                        "bytes": source.stat().st_size})
    return Run(model, key, path, metadata, classes, accept, rows, sources), None


def load_runs(root, registry, datasets="all", exclude=()):
    """Select the largest current-protocol run per registered model/dataset, as the viewer does."""
    selected, audit = {}, []
    for path in sorted((root / "outputs").glob("*/*_predictions.json")):
        model = path.parent.name
        if model not in registry:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8-sig"))
        key = metadata.get("key")
        if key in exclude or (datasets != "all" and key not in datasets) or "metrics" not in metadata:
            continue
        run, reason = read_run(path, model, registry, root)
        record = {"model": model, "dataset": key, "path": path.relative_to(root).as_posix()}
        if reason:
            audit.append(record | {"status": "excluded", "reason": reason})
            continue
        existing = selected.get(model, {}).get(key)
        if existing is None or len(run.rows) > len(existing.rows):
            if existing is not None:
                audit.append({"model": model, "dataset": key, "path": existing.path.relative_to(root).as_posix(),
                              "status": "excluded", "reason": "smaller_run"})
            selected.setdefault(model, {})[key] = run
        else:
            audit.append(record | {"status": "excluded", "reason": "smaller_or_equal_run"})
    for model, runs in selected.items():
        for key, run in runs.items():
            audit.append({"model": model, "dataset": key, "path": run.path.relative_to(root).as_posix(),
                          "status": "selected", "clips": len(run.rows)})
    return selected, audit


def prepare_cohort(dataset, models, available):
    runs = [available[model][dataset] for model in models]
    first = runs[0]
    for run in runs[1:]:
        if run.classes != first.classes or run.accept != first.accept:
            raise ValueError(f"{dataset}: candidate/accepted-label rules differ across {models}")
        if run.metadata.get("ground_truth", "acted_emotion") != first.metadata.get("ground_truth", "acted_emotion"):
            raise ValueError(f"{dataset}: ground-truth columns differ across models")
    common = sorted(set.intersection(*(set(run.rows) for run in runs)))
    if not common:
        raise ValueError(f"{dataset}: selected models have no common audio paths")
    for audio in common:
        if len({run.rows[audio]["gold"] for run in runs}) != 1:
            raise ValueError(f"{dataset}: ground-truth mismatch at {audio}")
    return runs, common


def decision(counts, m, spec):
    """Invalids abstain, but the denominator remains the entire selected model cohort."""
    if not counts:
        return None
    winning = max(counts.values())
    leaders = [label for label, votes in counts.items() if votes == winning]
    if len(leaders) != 1:
        return None
    if spec.get("unanimous"):
        return leaders[0] if winning == m else None
    fraction = Fraction(*spec["fraction"])
    share = Fraction(winning, m)
    accepted = share > fraction if spec.get("strict", False) else share >= fraction
    return leaders[0] if accepted else None


def analyze_group(dataset, group, runs, common):
    """Two acceptance filters, evaluated on exactly the same shared examples."""
    n, m = len(common), len(runs)
    totals = {rule: Counter() for rule in RULES}
    all_correct = unanimous_correct = invalid = ties = 0
    for audio in common:
        gold = runs[0].rows[audio]["gold"]
        accepted_gold = runs[0].accept.get(gold, [gold])
        predictions = [run.rows[audio]["prediction"] for run in runs]
        votes = Counter(label for label in predictions if label is not None)
        individual_correct = all(label in accepted_gold for label in predictions)
        all_correct += individual_correct
        invalid += any(label is None for label in predictions)
        ties += bool(votes) and list(votes.values()).count(max(votes.values())) > 1
        unanimous_correct += len(votes) == 1 and sum(votes.values()) == m and individual_correct
        for rule, spec in RULES.items():
            proposed = decision(votes, m, spec)
            totals[rule]["accepted"] += proposed is not None
            totals[rule]["accepted_correct"] += proposed in accepted_gold if proposed is not None else False
    rows = []
    for rule, counts in totals.items():
        accepted, correct = counts["accepted"], counts["accepted_correct"]
        rows.append({"dataset": dataset, "group": group, "rule": rule, "common_samples": n,
                     "union_samples": len(set.union(*(set(run.rows) for run in runs))),
                     "accepted": accepted, "accepted_correct": correct, "coverage": accepted / n,
                     "conditional_precision": correct / accepted if accepted else None,
                     "accepted_correct_fraction": correct / n, "all_models_individually_correct": all_correct,
                     "unanimous_and_correct": unanimous_correct, "samples_with_invalid_votes": invalid,
                     "samples_with_vote_ties": ties})
    return rows


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def progress_sentence(selected, language):
    """Separate each model's completed datasets from a consensus group's missing members."""
    zh = language == "zh"
    families = [(["gemini-3.8-flash"], None), (["gpt_realtime"], None),
                (GROUPS["SER5"]["models"], "五个 SER 模型" if zh else "Each of the five SER models"),
                (["qwen2_5_omni_7b", "qwen2_audio_7b", "gemma4_e4b"],
                 "三个本地 Audio LLM" if zh else "Each of the three local audio LLMs")]
    parts = []
    for models, family_name in families:
        counts = {}
        for model in models:
            runs = selected.get(model, {})
            full = sum(len(run.rows) == run.metadata.get("total_clips", len(run.rows)) for run in runs.values())
            counts[model] = (full, len(runs) - full)
        grouped = family_name and len(set(counts.values())) == 1
        entries = [(family_name, next(iter(counts.values())))] if grouped else [
            (SHORT_NAMES.get(model, model), counts[model]) for model in models]
        for name, (full, partial) in entries:
            value = f"{name}{'各 ' if grouped and zh else ' '}{full}/{len(DATASETS)}"
            if partial:
                value += (f"（另有 {partial} 个数据集仅有部分样本）" if zh else
                          f" (plus {partial} {'dataset' if partial == 1 else 'datasets'} with partial samples)")
            parts.append(value)
    return (("模型完成情况（已完成当前评估范围的数据集数 / 16）：" if zh else
             "Model completion (datasets with a full run in the current evaluation scope / 16): ")
            + ("；" if zh else "; ").join(parts) + ("。" if zh else "."))


def render_report(rows, selected_count, excluded, language, *, selected, matched_meld=False):
    zh = language == "zh"
    indexed = {(r["group"], r["dataset"], r["rule"]): r for r in rows}
    complete = {group: sum(indexed[group, dataset, "unanimous"]["complete"] for dataset in DATASETS)
                for group in GROUPS}
    meld = {group: indexed[group, "meld", "unanimous"] for group in GROUPS}
    values_ready = matched_meld and all(r["complete"] and r["coverage"] is not None and r["conditional_precision"] is not None for r in meld.values())
    selected_more = values_ready and meld["Selected3"]["coverage"] > max(meld[g]["coverage"] for g in ("SER5", "AudioLLM5"))
    selected_lower = values_ready and meld["Selected3"]["conditional_precision"] < max(meld[g]["conditional_precision"] for g in ("SER5", "AudioLLM5"))
    filtered = sum(r["status"] == "excluded" and r["reason"] == "incompatible_inference_protocol" for r in excluded)
    if zh:
        lines = ["# 模型共识能帮助筛选标签吗？", "",
                 f"这次直接使用 {selected_count} 组兼容的现有预测，比较 16 个公开数据集，没有重新运行模型。",
                 progress_sentence(selected, language),
                 f"整组成员齐备的数据集数：五个 SER 模型 {complete['SER5']}；Audio LLM 组 {complete['AudioLLM5']}；"
                 f"精选模型组 {complete['Selected3']}。‘组齐备’要求目标成员都有结果；下表会分别列出现有和待补成员。",
                 ("同一批 MELD 样本上，" if matched_meld else "各组当前 MELD 样本范围尚未对齐；")
                 + "全部一致的候选标签正确率／覆盖率分别为："
                 + "；".join(f"{GROUP_NAMES['zh'][g]} {percent(meld[g]['conditional_precision'])}／{percent(meld[g]['coverage'])}"
                             for g in GROUPS) + "。",
                 "精选模型保留了更多样本，但候选标签正确率不是三组最高。" if selected_more and selected_lower else
                 "比较时需要同时看保留了多少样本，以及保留标签的正确率；成员未齐备的结果仍需后续更新。",
                 "这些公开数据与公司电话存在领域差异；用于 pseudo-label 前，应先用小批人工通话标注校准组合和阈值。", "",
                 "覆盖率是共同样本中有多少被接受；候选标签正确率是被接受的标签中有多少正确。"
                 "表中的‘过半数’要求严格大于当前模型数的一半；invalid 不投票但仍占分母，最高票并列时不接受。"
                 "只有两个模型时，过半数与全部一致相同；五个模型时，过半数至少需要三票。"
                 "没有接受样本时，正确率记为 —。", ""]
    else:
        lines = ["# Can model agreement help select pseudo-labels?", "",
                 f"This analysis reuses {selected_count} compatible saved prediction runs across 16 public datasets; no model was rerun.",
                 progress_sentence(selected, language),
                 f"Datasets with every target member available: five-SER group {complete['SER5']}; audio-LLM group "
                 f"{complete['AudioLLM5']}; selected-model group {complete['Selected3']}. Group completeness requires results "
                 "from every target member; available and missing members are listed below.",
                 ("On the same MELD samples, " if matched_meld else "MELD sample sets are not yet aligned across groups; ")
                 + "unanimous accepted-label accuracy / coverage is: "
                 + "; ".join(f"{GROUP_NAMES['en'][g]} {percent(meld[g]['conditional_precision'])} / {percent(meld[g]['coverage'])}"
                             for g in GROUPS) + ".",
                 "The selected models retain more samples, while their accepted-label accuracy is not the highest of the three groups."
                 if selected_more and selected_lower else
                 "Compare both the retained share and the accuracy of retained labels; unfinished groups still need an update.",
                 "Public-data results do not establish accuracy on company calls; calibrate the group and threshold on a small manually labelled call sample first.", "",
                 "Coverage is the accepted share of common samples; accepted-label accuracy is the correct share of accepted labels. "
                 "Majority means strictly more than half of the current model count. Invalid answers abstain without reducing "
                 "that denominator; top-vote ties are rejected. With two models, majority equals unanimity; with five models, "
                 "majority requires at least three votes. Accuracy is — when no samples are accepted.", ""]
    for group, spec in GROUPS.items():
        target = len(spec["models"])
        lines += [f"## {GROUP_NAMES[language][group]}", "", ("目标成员：" if zh else "Target members: ")
                  + ", ".join(SHORT_NAMES.get(m, m) for m in spec["models"]) + ("。" if zh else "."), ""]
        patterns = defaultdict(list)
        for dataset in DATASETS:
            patterns[tuple(json.loads(indexed[group, dataset, "unanimous"]["members"]))].append(NAMES[dataset])
        for members, datasets in patterns.items():
            description = (("全组" if zh else "all target members") if list(members) == spec["models"] else
                           ", ".join(SHORT_NAMES.get(m, m) for m in members) or ("暂无结果" if zh else "no results yet"))
            if len(datasets) == 1:
                scope = datasets[0]
            elif set(datasets) == {"EmotionTalk", "DailyTalk", "MultiDialog"}:
                scope = "3 个新增对话数据集" if zh else "the three added dialogue datasets"
            elif set(datasets) == {name for key, name in NAMES.items() if key != "meld"}:
                scope = "MELD 以外的 15 个数据集" if zh else "the 15 datasets other than MELD"
            else:
                scope = f"{len(datasets)} 个数据集" if zh else f"{len(datasets)} datasets"
            missing = [SHORT_NAMES.get(m, m) for m in spec["models"] if m not in members]
            lines.append(("当前 " if zh else "") + scope + ("：" if zh else ": ") + description
                         + (("；待补：" if zh else "; awaiting: ") + ", ".join(missing) if missing and members else "")
                         + ("。" if zh else "."))
        lines += ["", ("缺任何成员就等待完整结果。" if spec["require_all"] and zh else
                       "This group waits until every member is available." if spec["require_all"] else
                       "* 表示这组仍缺其他成员；已有成员的结果已纳入。至少有两个成员时，先按现有成员计算共识。" if zh else
                       "* marks a group missing other members; available members are included. Consensus is calculated when at least two members are available."), "",
                  "| 数据集 | 参与 / 目标成员 | 共同样本 | 全部一致覆盖率 | 一致正确率 | 过半数覆盖率 | 过半数正确率 |" if zh else
                  "| Dataset | Participating / target members | Common samples | Unanimous coverage | Unanimous accuracy | Majority coverage | Majority accuracy |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for dataset in DATASETS:
            a, b = (indexed[group, dataset, rule] for rule in RULES)
            pending = a["common_samples"] is None
            membership = f"{a['model_count']}/{target}" + ("*" if not a["complete"] else "")
            samples = (("待其他模型" if zh else "Awaiting other models") if a["model_count"] else
                       ("暂无成员结果" if zh else "No member results")) if pending else f"{a['common_samples']:,}"
            lines.append(f"| {NAMES[dataset]} | {membership} | {samples} | {percent(a['coverage'])} | "
                         f"{percent(a['conditional_precision'])} | {percent(b['coverage'])} | {percent(b['conditional_precision'])} |")
        lines.append("")
    if zh:
        lines += [f"已过滤 {filtered} 组旧推理协议；各组均按相同 audio_path 的交集计分，"
                  "复合标签沿用原来的接受规则。表演与对话的语言、类别分布、录音来源不同，"
                  "上述差异不能只归因于对话上下文。模型可能共享训练来源和错误，不能把同意票数当作独立证据。", "",
                  "[完整计数、成员和源文件记录](../output/consensus.csv)。后续模型完成后，重新运行脚本即可更新这两份报告和 CSV。"]
    else:
        lines += [f"Excluded {filtered} runs with old inference protocols. Each group is scored on its common audio-path "
                  "intersection; composite gold labels keep their existing acceptance rules. Performed and dialogue corpora "
                  "also differ in language, class distribution and recording source, so their differences cannot be attributed "
                  "to context alone. Shared training sources and errors mean agreement votes are not independent evidence.", "",
                  "[Full counts, membership and source files](../output/consensus.csv). Rerun the script after more model results arrive to update both reports and the CSV."]
    return "\n".join(lines) + "\n"


def run(project=PROJECT, paper=None):
    project = Path(project).resolve()
    paper = Path(paper).resolve() if paper is not None else project.parent / "research_paper"
    registry = yaml.safe_load((project / "config/models.yaml").read_text()) or {}
    required = {m for spec in GROUPS.values() for m in spec["models"]}
    if not required <= set(registry):
        raise ValueError("The model registry is missing a predefined group member")
    selected, selection = load_runs(project, registry, datasets=DATASETS, exclude=["esd"])
    jobs, rows = [], []
    # All validation happens before any output is written.
    for group, spec in GROUPS.items():
        for dataset in DATASETS:
            models = [m for m in spec["models"] if dataset in selected.get(m, {})]
            complete = len(models) == len(spec["models"])
            common_fields = {"dataset": dataset, "group": group, "model_count": len(models),
                             "target_count": len(spec["models"]), "complete": complete,
                             "members": json.dumps(models, separators=(",", ":")),
                             "missing_members": json.dumps([m for m in spec["models"] if m not in models], separators=(",", ":")),
                             "sources": json.dumps({m: {"files": selected[m][dataset].sources,
                                 "subset": selected[m][dataset].metadata.get("subset"),
                                 "clips": len(selected[m][dataset].rows),
                                 "inference_protocol": selected[m][dataset].metadata.get("inference_protocol")}
                                 for m in models}, separators=(",", ":"))}
            eligible = len(models) >= 2 and (complete or not spec["require_all"])
            if eligible:
                runs, common = prepare_cohort(dataset, models, selected)
                jobs.append((common_fields, runs, common))
            else:
                for rule in RULES:
                    rows.append(common_fields | {"status": "pending", "rule": rule, "common_samples": None,
                        "union_samples": None, "accepted": None, "accepted_correct": None, "coverage": None,
                        "conditional_precision": None, "accepted_correct_fraction": None,
                        "all_models_individually_correct": None, "unanimous_and_correct": None,
                        "samples_with_invalid_votes": None, "samples_with_vote_ties": None})
    for fields, runs, common in jobs:
        rows.extend(fields | {"status": "complete" if fields["complete"] else "partial"} | values
                    for values in analyze_group(fields["dataset"], fields["group"], runs, common))
    order = {(group, dataset, rule): i for i, (group, dataset, rule) in enumerate(
             (g, d, r) for g in GROUPS for d in DATASETS for r in RULES)}
    rows.sort(key=lambda r: order[r["group"], r["dataset"], r["rule"]])
    sources = {a["path"]: a for runs in selected.values() for run in runs.values() for a in run.sources}
    for artifact in sources.values():
        if sha256(project / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Source predictions changed during analysis; rerun on a stable snapshot")
    selected_count = sum(len(runs) for runs in selected.values())
    meld_samples = [common for fields, _, common in jobs if fields["dataset"] == "meld"]
    matched_meld = len(meld_samples) == len(GROUPS) and all(samples == meld_samples[0] for samples in meld_samples)
    reports = {language: render_report(rows, selected_count, selection, language, selected=selected, matched_meld=matched_meld)
               for language in ("zh", "en")}
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "output").mkdir(exist_ok=True)
    (paper / "analysis_report").mkdir(exist_ok=True)
    for language, content in reports.items():
        (paper / "analysis_report" / f"consensus_{language}.md").write_text(content, encoding="utf-8")
    with (paper / "output/consensus.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Consensus: {selected_count} compatible runs; {len(jobs)} available group/dataset comparisons.")
    print(f"Reports: {paper / 'analysis_report/consensus_zh.md'}, {paper / 'analysis_report/consensus_en.md'}")
    return rows


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        run()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

"""Compare fixed model groups on saved predictions; no models, audio or APIs are run.

From the repository root: python research_paper/scripts/consensus.py
From research_paper/: python scripts/consensus.py
Outputs: bilingual reports, accuracy–coverage figures, consensus/threshold/class/baseline CSVs and a manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zoneinfo import ZoneInfo

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
# Supplemental fixed groups, chosen by model family rather than by observed accuracy.
# Keep the original three groups/CSV stable so older results remain directly comparable.
EXTRA_GROUPS = {
    "SER_API7": {"models": GROUPS["SER5"]["models"] + ["gpt_realtime", "gemini-3.8-flash"],
                 "require_all": True},
    "All10": {"models": GROUPS["SER5"]["models"] + GROUPS["AudioLLM5"]["models"],
              "require_all": True},
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


def render_report(rows, selected_count, excluded, language, *, selected, matched_meld=False,
                  threshold_rows=(), generated_at=None, plots=False):
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
    if generated_at:
        lines[2:2] = [("更新时间：" if zh else "Updated: ") + generated_at + ".", ""]
    lines += render_figure_guide(threshold_rows, language)
    ser_unanimous = [r for r in threshold_rows if r["group"] == "SER5" and r["min_votes"] == 5]
    if ser_unanimous:
        high_precision = sum(r["conditional_precision"] is not None and r["conditional_precision"] >= .95
                             for r in ser_unanimous)
        lines += [(f"当前五 SER 全部一致的观测正确率在 {high_precision}/{len(ser_unanimous)} 个数据集上达到 95%，"
                   "说明共识在部分领域有筛选价值，但不能跨数据集统一保证高准确率。增加成员也不保证更准确。"
                   "下文补充 4/5 票、跨家族固定组、错误数与类别分布，所有阈值均保留，不用测试标签自动选最佳规则。" if zh else
                   f"Five-SER unanimity reaches an observed 95% accepted-label accuracy on {high_precision}/{len(ser_unanimous)} "
                   "datasets. Agreement can select reliable labels in some domains, but does not guarantee high accuracy "
                   "across datasets, and adding members need not improve it. The supplement adds 4/5 votes, fixed mixed-family "
                   "groups, error counts and class distributions; every threshold is retained without selecting a best rule "
                   "on these evaluation labels."), ""]
    lines += (["下列结果是在各数据集既定评估范围内计算，不代表整个原始语料；THAI-SER 使用 high_agreement，"
               "MEAD 使用 stratified_8000。组成员齐备不等于成员已预测整个原始语料。"
               "Audio LLM 组在不同数据集的参与成员不同，不汇总为一个固定 ensemble；Selected3 是沿用旧分析的组合，"
               "名称不表示它在本轮表现最佳。", ""] if zh else
              ["Results use each dataset's configured evaluation scope, not necessarily the entire source corpus: "
               "THAI-SER uses high_agreement and MEAD uses stratified_8000. Member availability does not imply "
               "full-source-corpus coverage. AudioLLM5 membership varies across datasets, so it is not pooled as a fixed "
               "ensemble. Selected3 is the legacy group name, not a claim that it is optimal in this refresh.", ""])
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
    lines += render_details(threshold_rows, language, plots=plots)
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


def render_figure_guide(rows, language):
    """Keep the rule, two denominators and model-group comparison explicit."""
    zh = language == "zh"
    indexed = {(r["group"], r["dataset"], r["min_votes"]): r for r in rows}
    if zh:
        lines = ["## 图怎么读：一个点是一组模型加一条投票规则", "",
                 "横轴 Samples auto-labelled (%) 表示：同一批可比较音频中，有多少比例通过投票规则，"
                 "可以作为自动标签候选。它等于接受数 / 共同样本数。纵轴 Correct auto-labels (%) 表示："
                 "这些候选标签里，有多少与数据集参考标签相符。它等于正确接受数 / 接受数，"
                 "不是整个数据集的准确率，也不是模型自己给出的置信度。这里的‘自动标注’是对基准数据的模拟筛选。", "",
                 "例如横轴 20%、纵轴 97% 的点，可以读作：每 1,000 条音频约筛出 200 条候选标签，"
                 "其中约 194 条正确、6 条错误，其余 800 条未通过筛选。参考标签来自数据集；内部未标注音频"
                 "没有现成的这个纵轴，需要独立人工核验。", "",
                 "固定模型组合之后，还需要指定至少几票同意，才能确定一个结果。五个 SER 的三个点分别是"
                 "至少 3/5、至少 4/5、5/5 一致；不是同一规则被重复测了三次，也没有改变共同样本分母。"
                 "3/5 包含 4 票和 5 票样本，4/5 包含 5 票样本。收紧票数后覆盖率一定不增加，"
                 "但正确率不保证上升。连线仅帮助看离散票数的变化，不表示任意覆盖率都有可实现的规则。", ""]
    else:
        lines = ["## Reading the figures: one model group plus one vote rule defines a point", "",
                 "The x-axis, Samples auto-labelled (%), is the share of the common audio cohort that passes the vote "
                 "rule: accepted / common samples. The y-axis, Correct auto-labels (%), is the share of those selected "
                 "labels matching the dataset reference: correct accepted / accepted. This is neither full-dataset "
                 "accuracy nor a model's confidence score. Auto-labelling here is a simulated selection on benchmark data.", "",
                 "A point at 20% x and 97% y means that out of 1,000 clips, about 200 receive candidate labels: "
                 "194 correct and six wrong, with 800 unselected. Unlabelled internal audio has no known y-value; "
                 "independent human verification is needed.", "",
                 "A model group alone does not define the result: the minimum vote count must also be fixed. "
                 "The three five-SER points mean at least 3/5, at least 4/5 and all 5/5 agreeing. They are not repetitions "
                 "of one rule, and their common-cohort denominator is unchanged. The 3/5 rule includes four- and five-vote "
                 "cases; 4/5 includes five-vote cases. Raising the threshold cannot increase coverage, but need not improve "
                 "accuracy. Lines only guide the eye between discrete rules; intermediate coverage is not an evaluated rule.", ""]
    example = [indexed.get(("SER5", "crema_d", k)) for k in (3, 4, 5)]
    if all(r is not None for r in example):
        lines += [f"CREMA-D，始终使用同一批 {example[0]['common_samples']:,} 条音频：" if zh else
                  f"CREMA-D, always the same {example[0]['common_samples']:,} clips:", "",
                  "| 五 SER 规则 | 筛出数 | 正确 / 错误 | 横轴：筛出比例 | 纵轴：筛出标签正确率 |" if zh else
                  "| Five-SER rule | Selected | Correct / wrong | x: selected share | y: selected-label accuracy |",
                  "|---|---:|---:|---:|---:|"]
        for r in example:
            lines.append(f"| ≥{r['min_votes']}/5 | {r['accepted']:,} | "
                         f"{r['accepted_correct']:,} / {r['accepted_errors']:,} | "
                         f"{percent(r['coverage'])} | {percent(r['conditional_precision'])} |")
        lines.append("")
    if zh:
        lines += ["颜色只表示模型成员，不表示性能高低：蓝色 SER5 是五个 SER；橙色 Selected3 是 "
                  "emotion2vec+、Gemini、Qwen2.5-Omni；绿色 SER+API7 是五 SER 加 GPT Realtime、Gemini；"
                  "粉色 All10 是全部十个模型。Selected3 是沿用的组名，不代表最优组合。", "",
                  "比较不同颜色时要先对齐规则。过半数分别是 3/5、2/3、4/7、6/10，要求的同意比例不同，"
                  "通过筛选的样本和投票标签都可能不同。例如五 SER 投出 3 票正确、2 票错误时，"
                  "如果两个 API 都投给原来的错误标签，七模型会变为 3 票正确、4 票错误。"
                  "因此加入模型可能改变答案，过半数接受集也不一定变小。", "",
                  "在相同共同样本上比较全体一致时，7/7 是 5/5 的子集，10/10 又是 7/7 和 3/3 的子集；"
                  "覆盖率因此不会增加。但只有新增模型更容易淘汰错误标签、而不是正确标签时，"
                  "筛出标签的正确率才会上升；相关错误、共同偏好都可能破坏这个条件。", ""]
    else:
        lines += ["Colour identifies members, not a quality ranking: blue SER5 is the five SER models; orange "
                  "Selected3 is emotion2vec+, Gemini and Qwen2.5-Omni; green SER+API7 adds GPT Realtime and Gemini to "
                  "SER5; pink All10 includes all ten models. Selected3 is a legacy name, not an optimality claim.", "",
                  "Align the rules before comparing colours. Strict majority means 3/5, 2/3, 4/7 or 6/10: different "
                  "agreement fractions, potentially different selected samples and labels. If five SER models vote three "
                  "for the correct label and two for a wrong label, two APIs voting for that wrong label turn the result "
                  "into three correct votes against four wrong votes. Adding models can change the answer, and majority "
                  "acceptance sets need not shrink.", "",
                  "On the same common cohort, 7/7 unanimity is a subset of 5/5, and 10/10 is a subset of both 7/7 and "
                  "3/3: coverage cannot increase. Selected-label accuracy improves only when added models preferentially "
                  "reject wrong labels rather than correct ones; shared errors or label preferences can prevent that.", ""]
    majority_a, majority_b = (indexed.get((g, "crema_d", k)) for g, k in (("SER5", 3), ("SER_API7", 4)))
    if majority_a and majority_b and majority_a["cohort_sha256"] == majority_b["cohort_sha256"]:
        lines += [(f"例如 CREMA-D，蓝色 3/5 筛出 {majority_a['accepted']:,} 条，正确率 "
                   f"{percent(majority_a['conditional_precision'])}；绿色 4/7 筛出 {majority_b['accepted']:,} 条，"
                   f"正确率 {percent(majority_b['conditional_precision'])}。这不是在蓝色已接受标签上做同一种更严格筛选，"
                   "而是换了投票组合和最低票数。" if zh else
                   f"For example, CREMA-D blue 3/5 selects {majority_a['accepted']:,} clips at "
                   f"{percent(majority_a['conditional_precision'])} accuracy; green 4/7 selects "
                   f"{majority_b['accepted']:,} at {percent(majority_b['conditional_precision'])}. This changes the voters "
                   "and minimum vote count rather than simply applying a tighter version of the same filter."), ""]
    a, b = (indexed.get((g, "crema_d", k)) for g, k in (("SER5", 5), ("SER_API7", 7)))
    if a and b and a["cohort_sha256"] == b["cohort_sha256"]:
        kept_correct = b["accepted_correct"] / a["accepted_correct"] if a["accepted_correct"] else None
        kept_errors = b["accepted_errors"] / a["accepted_errors"] if a["accepted_errors"] else None
        lines += [(f"CREMA-D 的实际例子：5/5 接受 {a['accepted']:,} 条（正确 {a['accepted_correct']:,}、错误 "
                   f"{a['accepted_errors']:,}）；要求两个 API 也同意后，7/7 剩下 {b['accepted']:,} 条"
                   f"（正确 {b['accepted_correct']:,}、错误 {b['accepted_errors']:,}）。新增条件保留原先正确样本的 "
                   f"{percent(kept_correct)}，却保留错误样本的 {percent(kept_errors)}，所以正确率从 "
                   f"{percent(a['conditional_precision'])} 变为 {percent(b['conditional_precision'])}。" if zh else
                   f"An observed CREMA-D example: 5/5 selects {a['accepted']:,} clips ({a['accepted_correct']:,} correct, "
                   f"{a['accepted_errors']:,} wrong). Requiring both APIs to agree leaves {b['accepted']:,} "
                   f"({b['accepted_correct']:,} correct, {b['accepted_errors']:,} wrong). The added condition retains "
                   f"{percent(kept_correct)} of the original correct cases but {percent(kept_errors)} of the errors, "
                   f"changing precision from {percent(a['conditional_precision'])} to {percent(b['conditional_precision'])}."), ""]
    meld = [indexed.get((g, "meld", k)) for g, k in (("SER5", 5), ("Selected3", 3),
                                                     ("SER_API7", 7), ("All10", 10))]
    if all(r is not None for r in meld) and len({r["cohort_sha256"] for r in meld}) == 1:
        lines += ["MELD 的公平端点比较：全部一致，且共同样本完全相同。" if zh else
                  "Matched MELD endpoints: unanimity on the identical common cohort.", "",
                  "| 组合 | 规则 | 筛出数 | 正确 / 错误 | 筛出比例 | 筛出标签正确率 |" if zh else
                  "| Group | Rule | Selected | Correct / wrong | Selected share | Selected-label accuracy |",
                  "|---|---|---:|---:|---:|---:|"]
        for r in meld:
            lines.append(f"| {r['group']} | {r['min_votes']}/{r['model_count']} | {r['accepted']:,} | "
                         f"{r['accepted_correct']:,} / {r['accepted_errors']:,} | {percent(r['coverage'])} | "
                         f"{percent(r['conditional_precision'])} |")
        lines += ["", ("粉色曲线不是其他曲线的平均值，每个点都重新对十个模型计票。MELD 的 10/10 只筛出 "
                       f"{meld[3]['accepted']:,} 条，其中 {meld[3]['accepted_errors']:,} 条仍错误；"
                       "它以更低覆盖率换来与 Selected3 的 3/3 相近的观测正确率，而非自动成为最优组合。"
                       "粉色曲线中间的点是 6/10 到 9/10，不能与其他颜色的全体一致端点混作同一规则比较。" if zh else
                       "The pink curve is not an average of the others: every point recounts all ten model votes. "
                       f"MELD 10/10 selects only {meld[3]['accepted']:,} clips, including {meld[3]['accepted_errors']:,} errors. "
                       "It trades much lower coverage for an observed accuracy close to Selected3 at 3/3, rather than "
                       "automatically becoming optimal. Its other points are 6/10 through 9/10 and should not be treated "
                       "as the same rule as another group's unanimous endpoint."), ""]
    return lines


def render_details(rows, language, *, plots=False):
    """Explain the operational trade-off without choosing a rule on evaluation labels."""
    zh = language == "zh"
    indexed = {(r["group"], r["dataset"], r["min_votes"]): r for r in rows}
    lines = ["## 投票阈值、接受数与错误数" if zh else "## Vote thresholds, accepted labels and errors", "",
             ("补充文件枚举从严格过半到全部一致的每个整数票数，没有根据这些测试结果挑选最佳组合或阈值。"
              "下表补上五 SER 的 4/5 票；错误数可以直接反映自动接受标签的代价。" if zh else
              "The supplement enumerates every integer threshold from strict majority to unanimity; it does not "
              "select an optimal group or threshold on these evaluation labels. This table adds the 4/5 SER threshold "
              "and makes the cost of erroneous accepted labels explicit."), "",
             "| 数据集 | ≥4/5 接受数 | 错误数 | 覆盖率 | 正确率 | 5/5 接受数 | 错误数 |" if zh else
             "| Dataset | ≥4/5 accepted | Errors | Coverage | Accuracy | 5/5 accepted | Errors |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        a, b = (indexed.get(("SER5", dataset, k)) for k in (4, 5))
        if a is not None and b is not None:
            lines.append(f"| {NAMES[dataset]} | {a['accepted']:,} | {a['accepted_errors']:,} | "
                         f"{percent(a['coverage'])} | {percent(a['conditional_precision'])} | "
                         f"{b['accepted']:,} | {b['accepted_errors']:,} |")
    lines += ["", "## 跨模型家族的固定组合" if zh else "## Fixed groups across model families", "",
              ("SER+API7 = 五个 SER + GPT Realtime + Gemini；All10 = 五个 SER + 五个 Audio LLM/API。"
               "这两个补充组合缺任何成员都等待，不缩小组合。下表为全部一致结果；完整整数阈值见 CSV。"
               "不同组只有在共同样本路径完全相同时才能直接比较。" if zh else
               "SER+API7 = five SER models + GPT Realtime + Gemini; All10 = five SER models + all five audio LLM/API "
               "models. Both supplemental groups wait for every member. The table reports unanimity; the CSV contains "
               "all integer thresholds. Direct group comparisons require identical common audio-path cohorts."), "",
              "| 数据集 | SER+API7 共同样本 | 7/7 接受数 / 错误数 | 覆盖率 | 正确率 | All10 共同样本 | 10/10 接受数 / 错误数 | 覆盖率 | 正确率 |" if zh else
              "| Dataset | SER+API7 common | 7/7 accepted / errors | Coverage | Accuracy | All10 common | 10/10 accepted / errors | Coverage | Accuracy |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in DATASETS:
        cells = []
        for group, k in (("SER_API7", 7), ("All10", 10)):
            r = indexed.get((group, dataset, k))
            cells.extend([f"{r['common_samples']:,}", f"{r['accepted']:,} / {r['accepted_errors']:,}",
                          percent(r["coverage"]), percent(r["conditional_precision"]) ] if r else
                         [("待齐备" if zh else "Pending"), "—", "—", "—"])
        lines.append(f"| {NAMES[dataset]} | " + " | ".join(cells) + " |")
    if plots:
        lines += ["", "## 图表：分开看票数与模型组合" if zh else "## Figures: separate vote thresholds from group membership", "",
                  ("票数图只画五 SER，逐点标明至少几票同意。MELD 细图将四种组合分成独立面板。"
                   "柱状图沿用 benchmark 的数据集顺序、彩色柱和数值标注；文字使用更大的黑色字体。"
                   "全体一致柱状图比较固定完整组合，不混入缺成员的 AudioLLM5。" if zh else
                   "The five-SER threshold plot labels each minimum vote count; MELD uses a separate panel for each group. "
                   "Bar charts follow the benchmark's dataset order, coloured bars and value labels, with larger black text. "
                   "Unanimity bars compare complete fixed groups, excluding the variable-membership AudioLLM5."), "",
                  "![Five-SER vote thresholds](figures/consensus_accuracy_coverage.png)", "",
                  "![MELD model groups and vote thresholds](figures/consensus_meld_accuracy_coverage.png)", "",
                  "![Accuracy of unanimous auto-labels](figures/consensus_unanimous_accuracy_by_dataset.png)", "",
                  "![Share of clips with unanimous auto-labels](figures/consensus_unanimous_coverage_by_dataset.png)", "",
                  "![Five-SER accuracy by minimum votes](figures/consensus_ser5_accuracy_by_votes.png)", "",
                  "![Five-SER auto-labelling share by minimum votes](figures/consensus_ser5_coverage_by_votes.png)", "",
                  ("柱顶数字按整数百分比显示，精确数值见表格和 CSV。n/a 表示组合成员尚未齐备；"
                   "筛出数为 0 时，覆盖率是 0%，正确率未定义，不能画成 0% 正确率。" if zh else
                   "Bar labels round to whole percentages; tables and CSVs retain full precision. n/a indicates missing "
                   "group members. With zero selected clips, coverage is 0% but accuracy is undefined, not 0%."), ""]
    lines += ["", "## 用于标注与论文时如何解释" if zh else "## Interpretation for labelling and the paper", ""]
    if zh:
        lines += ["覆盖率 = accepted / common_samples；候选标签正确率 = accepted_correct / accepted；"
                  "selective risk = 1 − 正确率；正确接受占比 = accepted_correct / common_samples。"
                  "另导出 accepted / union_samples，显示模型样本交集缺失的影响；union 也只是现有预测的并集，"
                  "不能替代整个待标注池的分母。", "",
                  "逐预测类别文件同时给出接受数、错误数、该类别 precision 及其在接受标签中的占比，"
                  "用于检查高总体正确率是否主要来自 neutral 等类别；单模型基线也在完全相同的组内交集上重新计分。"
                  "这些是固定预测的观测点估计，没有置信区间或准确率保证，小样本上的 100% 不能等同于零风险。", "",
                  "内部标注可先用独立人工标注样本确定允许的错误率，再校准固定模型组合和票数，"
                  "在未参与校准的样本上按类别、语言和录音条件验证。接受样本仍应抽检，拒绝样本交人工处理。"
                  "公开数据中的覆盖率只是潜在自动接受比例，实际节省的人力还取决于内部错误率、抽检和纠错成本。", "",
                  "论文中可将这一部分作为 selective prediction 的 accuracy–coverage / risk–coverage 分析，"
                  "比较固定组合与阈值在各公开数据集上的可靠性，不据此宣称新算法或内部标注收益。"
                  "这一分析框架已有先例：[Geifman & El-Yaniv, NeurIPS 2017]"
                  "(https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf)。"
                  "正式推断还需按 speaker/dialogue 分组估计不确定性，并披露已知训练来源；当前实验不提供这些区间。", ""]
    else:
        lines += ["Coverage = accepted / common_samples; accepted-label accuracy = accepted_correct / accepted; "
                  "selective risk = 1 − accuracy; correct accepted fraction = accepted_correct / common_samples. "
                  "accepted / union_samples also exposes missing model overlap, but the union is still only the available "
                  "prediction pool, not the entire intended labelling pool.", "",
                  "The per-predicted-class file reports accepted/error counts, class precision and share of accepted labels "
                  "to reveal dominance by neutral or another class. Individual-model baselines are recomputed on exactly "
                  "the same within-group cohort. These are observed fixed-prediction point estimates, without confidence "
                  "intervals or accuracy guarantees; 100% on a small accepted set does not establish zero risk.", "",
                  "For internal labelling, set a tolerated error rate using independently labelled calibration data, freeze "
                  "the model group and vote threshold, then validate by class, language and recording condition on held-out "
                  "samples. Audit accepted labels and route rejected samples to humans. Public-data coverage is potential "
                  "automatic acceptance, while realised effort savings also depend on internal errors, auditing and correction.", "",
                  "For the paper, this can be a supporting selective-prediction accuracy–coverage / risk–coverage analysis "
                  "of fixed groups and thresholds across public datasets, without claiming a new algorithm or internal "
                  "labelling benefit. The framework has precedent: [Geifman & El-Yaniv, NeurIPS 2017]"
                  "(https://papers.neurips.cc/paper/2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf). "
                  "Formal inference would additionally need speaker/dialogue-cluster uncertainty estimates and training "
                  "provenance disclosure; those intervals are not supplied here.", ""]
    lines += [("补充结果：" if zh else "Supplementary results: ")
              + "[vote thresholds](../output/consensus_thresholds.csv) · "
              "[per predicted class](../output/consensus_by_class.csv) · "
              "[matched individual baselines](../output/consensus_baselines.csv) · "
              "[reproducibility manifest](../output/consensus_manifest.json)", ""]
    return lines


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError(f"No rows to export: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(project=PROJECT, paper=None, *, plots=False):
    from consensus_details import build_details, THRESHOLD_FIELDS, CLASS_FIELDS, BASELINE_FIELDS

    project = Path(project).resolve()
    paper = Path(paper).resolve() if paper is not None else project.parent / "research_paper"
    implementation = {path.name: path for path in [Path(__file__).resolve(),
                      Path(__file__).with_name("consensus_details.py"), Path(__file__).with_name("consensus_plot.py"),
                      Path(sys.modules[configured_protocol.__module__].__file__).resolve()]}
    configuration = {"config/models.yaml": project / "config/models.yaml",
                     "config/llm.yaml": project / "config/llm.yaml",
                     "config/evaluation.yaml": project / "config/evaluation.yaml"}
    if configuration["config/llm.yaml"].exists():
        llm = yaml.safe_load(configuration["config/llm.yaml"].read_text()) or {}
        if llm.get("prompts"):
            configuration[llm["prompts"]] = project / llm["prompts"]
    tracked = {**implementation, **{key: path for key, path in configuration.items() if path.exists()}}
    code_config_hashes = {key: sha256(path) for key, path in tracked.items()}
    registry = yaml.safe_load((project / "config/models.yaml").read_text()) or {}
    required = {m for spec in GROUPS.values() for m in spec["models"]}
    if not required <= set(registry):
        raise ValueError("The model registry is missing a predefined group member")
    selected, selection = load_runs(project, registry, datasets=DATASETS, exclude=["esd"])
    jobs, rows, cohorts = [], [], []
    analysis_groups = GROUPS | EXTRA_GROUPS
    # All validation happens before any output is written.
    for group, spec in analysis_groups.items():
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
            cohorts.append({key: value for key, value in common_fields.items() if key != "sources"}
                           | {"eligible": eligible,
                              "common_samples": len(common) if eligible else None,
                              "union_samples": len(set.union(*(set(r.rows) for r in runs))) if eligible else None})
            if not eligible and group in GROUPS:
                for rule in RULES:
                    rows.append(common_fields | {"status": "pending", "rule": rule, "common_samples": None,
                        "union_samples": None, "accepted": None, "accepted_correct": None, "coverage": None,
                        "conditional_precision": None, "accepted_correct_fraction": None,
                        "all_models_individually_correct": None, "unanimous_and_correct": None,
                        "samples_with_invalid_votes": None, "samples_with_vote_ties": None})
    for fields, runs, common in jobs:
        if fields["group"] in GROUPS:
            rows.extend(fields | {"status": "complete" if fields["complete"] else "partial"} | values
                        for values in analyze_group(fields["dataset"], fields["group"], runs, common))
    order = {(group, dataset, rule): i for i, (group, dataset, rule) in enumerate(
             (g, d, r) for g in GROUPS for d in DATASETS for r in RULES)}
    rows.sort(key=lambda r: order[r["group"], r["dataset"], r["rule"]])
    thresholds, by_class, baselines = build_details(jobs)
    cohort_hashes = {(row["group"], row["dataset"]): row["cohort_sha256"] for row in thresholds}
    for cohort in cohorts:
        cohort["cohort_sha256"] = cohort_hashes.get((cohort["group"], cohort["dataset"]))
    sources = {a["path"]: a for runs in selected.values() for run in runs.values() for a in run.sources}
    for artifact in sources.values():
        if sha256(project / artifact["path"]) != artifact["sha256"]:
            raise RuntimeError("Source predictions changed during analysis; rerun on a stable snapshot")
    for key, path in tracked.items():
        if sha256(path) != code_config_hashes[key]:
            raise RuntimeError(f"Code or configuration changed during analysis: {path}; rerun")
    selected_count = sum(len(runs) for runs in selected.values())
    meld_samples = [common for fields, _, common in jobs
                    if fields["dataset"] == "meld" and fields["group"] in GROUPS]
    matched_meld = len(meld_samples) == len(GROUPS) and all(samples == meld_samples[0] for samples in meld_samples)
    generated = datetime.now(timezone.utc)
    timestamp = generated.astimezone(ZoneInfo("Asia/Singapore")).strftime("%Y-%m-%d %H:%M SGT")
    reports = {language: render_report(rows, selected_count, selection, language, selected=selected,
                                      matched_meld=matched_meld, threshold_rows=thresholds,
                                      generated_at=timestamp, plots=plots)
               for language in ("zh", "en")}
    manifest = {"analysis": "model_consensus_v2", "generated_at_utc": generated.isoformat(timespec="seconds"),
                "generated_at_singapore": timestamp, "python": sys.version.split()[0], "pyyaml": yaml.__version__,
                "selected_run_count": selected_count, "datasets": list(DATASETS), "groups": analysis_groups,
                "rules": RULES, "threshold_sweep": "Every integer k from floor(m/2)+1 through m; invalids retain m",
                "coverage_denominator": "within-group common audio-path intersection",
                "uncertainty": "Point estimates only; no confidence intervals or accuracy guarantees",
                "code_configuration_sha256": code_config_hashes, "selection": selection,
                "sources": sorted(sources.values(), key=lambda source: source["path"]), "cohorts": cohorts,
                "run_scopes": [{"model": model, "dataset": dataset, "clips": len(run.rows),
                                "total_clips": run.metadata.get("total_clips"),
                                "subset": run.metadata.get("subset"), "date": run.metadata.get("date")}
                               for model, runs in selected.items() for dataset, run in runs.items()]}
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "output").mkdir(exist_ok=True)
    (paper / "analysis_report").mkdir(exist_ok=True)
    if plots:
        import matplotlib
        from consensus_plot import plot_thresholds
        figure_paths = plot_thresholds(thresholds, paper / "analysis_report/figures")
        manifest["figures"] = [path.relative_to(paper).as_posix() for path in figure_paths]
        manifest["matplotlib"] = matplotlib.__version__
    for language, content in reports.items():
        (paper / "analysis_report" / f"consensus_{language}.md").write_text(content, encoding="utf-8")
    for name, values, schema in (("consensus", rows, None), ("consensus_thresholds", thresholds, THRESHOLD_FIELDS),
                                 ("consensus_by_class", by_class, CLASS_FIELDS),
                                 ("consensus_baselines", baselines, BASELINE_FIELDS)):
        write_csv(paper / "output" / f"{name}.csv", values, fieldnames=schema)
    (paper / "output/consensus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8")
    print(f"Consensus: {selected_count} compatible runs; {len(jobs)} available group/dataset comparisons.")
    print(f"Reports: {paper / 'analysis_report/consensus_zh.md'}, {paper / 'analysis_report/consensus_en.md'}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Write reports/CSVs/manifest without matplotlib figures")
    args = parser.parse_args()
    try:
        run(plots=not args.no_plots)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

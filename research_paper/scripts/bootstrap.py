"""Reproducible paired cluster bootstrap of saved SER/audio-LLM predictions.

From the repository root: python research_paper/scripts/bootstrap.py
From research_paper/: python scripts/bootstrap.py
Writes output/bootstrap.csv, output/bootstrap_manifest.json and bilingual
analysis_report/bootstrap_{zh,en}.md. No models, audio or APIs are run.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import sys

import numpy as np
import yaml

import consensus

sys.path.insert(0, str(consensus.PROJECT / "scripts" / "analysis"))
from probe_metadata import infer_groups
from prediction_io import load_reported_metrics

DATASETS = ("crema_d", "ravdess", "meld")
METRICS = ("accuracy", "unweighted_accuracy", "macro_f1", "weighted_f1")
METRIC_NAMES = ("Accuracy", "UAR", "Macro-F1", "Weighted-F1")
FOCUS_PAIRS = (("emotion2vec_plus_large", "gemini-3.8-flash"),
               ("emotion2vec_plus_large", "qwen2_5_omni_7b"),
               ("gemini-3.8-flash", "qwen2_5_omni_7b"))
DEFAULT_SEED = 20261111
DEFAULT_RESAMPLES = 10000
ALGORITHM = "paired_cluster_percentile_v1"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=denominator != 0)


def scores_from_counts(counts):
    """Pooled clip metrics, with a fixed class set and invalid predictions in the last column.

    Input shape (..., K, K+1); output (..., 4) follows METRICS. Absent classes
    keep recall/F1=0, rather than changing the metric's class set per resample.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim < 2 or counts.shape[-1] != counts.shape[-2] + 1 or np.any(counts < 0):
        raise ValueError("Counts must be nonnegative (..., K, K+1) confusion matrices")
    k = counts.shape[-2]
    support = counts.sum(axis=-1)
    predicted = counts[..., :k].sum(axis=-2)
    true_positive = np.diagonal(counts, axis1=-2, axis2=-1)
    total = support.sum(axis=-1)
    if np.any(total <= 0):
        raise ValueError("Cannot score an empty resample")
    recalls = ratio(true_positive, support)
    f1 = ratio(2 * true_positive, support + predicted)
    return np.stack((true_positive.sum(axis=-1) / total, recalls.mean(axis=-1),
                     f1.mean(axis=-1), (f1 * support).sum(axis=-1) / total), axis=-1)


def cluster_counts(run, paths, inverse, n_groups):
    """Sum each group's complete confusion counts; a repeated group repeats every clip."""
    if run.accept:
        raise ValueError("This bootstrap scope does not support composite gold labels")
    lookup = {label: i for i, label in enumerate(run.classes)}
    k = len(lookup)
    counts = np.zeros((n_groups, k, k + 1), dtype=np.int64)
    gold = [lookup[run.rows[path]["gold"]] for path in paths]
    predictions = [k if run.rows[path]["prediction"] is None else lookup[run.rows[path]["prediction"]]
                   for path in paths]
    np.add.at(counts, (inverse, gold, predictions), 1)
    return counts


def bootstrap_weights(n_groups, n_resamples, seed, dataset, cohort_hash):
    if n_groups < 2 or n_resamples < 1:
        raise ValueError("Bootstrap needs at least two groups and one resample")
    # Neither model names nor predictions affect the stream: adding a model does
    # not change existing comparisons on the same cohort. Never use Python hash().
    entropy = int(digest([ALGORITHM, seed, dataset, cohort_hash]), 16)
    rng = np.random.Generator(np.random.PCG64(entropy))
    return rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_resamples)


def interval(values, confidence=.95):
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    values = np.asarray(values)
    if not np.isfinite(values).all() or not values.size:
        raise ValueError("Bootstrap distribution must be nonempty and finite")
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(values, [tail, 1 - tail], axis=0, method="linear")
    return lower, upper


def paired_interval(draws_a, draws_b, confidence=.95):
    if np.shape(draws_a) != np.shape(draws_b):
        raise ValueError("Paired distributions must have the same shape")
    return interval(np.asarray(draws_a) - np.asarray(draws_b), confidence)


@dataclass
class Cohort:
    key: str
    paths: list
    inverse: np.ndarray
    weights: np.ndarray
    info: dict


def make_cohort(dataset, paths, reference, project, n_resamples, seed):
    paths = sorted(paths)
    identities = infer_groups(dataset, paths, project)
    kinds = {item["group_kind"] for item in identities.values()}
    expected_kind = "dialogue" if dataset == "meld" else "speaker"
    if kinds != {expected_kind}:
        raise ValueError(f"{dataset}: expected {expected_kind} groups")
    group_ids = [identities[path]["group"] for path in paths]
    groups, inverse, sizes = np.unique(group_ids, return_inverse=True, return_counts=True)
    cohort_hash = digest([[path, identities[path]["group"], reference.rows[path]["gold"]] for path in paths])
    weights = bootstrap_weights(len(groups), n_resamples, seed, dataset, cohort_hash)
    info = {"dataset": dataset, "n_samples": len(paths), "n_clusters": len(groups), "group_kind": expected_kind,
            "classes": reference.classes, "cohort_sha256": cohort_hash, "audio_paths_sha256": digest(paths),
            "group_ids_sha256": digest(groups.tolist()),
            "grouping_sources": sorted({item["source"] for item in identities.values()}),
            "cluster_sizes": {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max())},
            "resampled_n_min": int((weights @ sizes).min()), "resampled_n_max": int((weights @ sizes).max())}
    return Cohort(cohort_hash, paths, inverse, weights, info)


def distribution(run, cohort):
    counts = cluster_counts(run, cohort.paths, cohort.inverse, cohort.info["n_clusters"])
    original = counts.sum(axis=0)
    if np.any(original.sum(axis=-1) == 0):
        raise ValueError(f"{run.dataset}/{run.model}: original cohort lacks a candidate gold class")
    draws = (cohort.weights.astype(np.float64) @ counts.reshape(len(counts), -1)).reshape(
        len(cohort.weights), *original.shape)
    missing_class_draws = int(np.any(draws.sum(axis=-1) == 0, axis=-1).sum())
    return scores_from_counts(original), scores_from_counts(draws), missing_class_draws


def run_scope(run):
    total = run.metadata.get("total_clips")
    if total is None:
        return "total_not_declared"
    if type(total) is not int or total < len(run.rows):
        raise ValueError(f"{run.path}: invalid total_clips")
    return "partial_declared_scope" if len(run.rows) < total else "full_declared_scope"


def result_rows(kind, dataset, run_a, run_b, cohort, point_a, draws_a, confidence,
                point_b=None, draws_b=None, missing_class_draws=0):
    point = point_a if run_b is None else point_a - point_b
    lower, upper = (interval(draws_a, confidence) if run_b is None
                    else paired_interval(draws_a, draws_b, confidence))
    base = {"kind": kind, "dataset": dataset, "model_a": run_a.model, "model_b": run_b.model if run_b else "",
            "n_samples": len(cohort.paths), "n_clusters": cohort.info["n_clusters"],
            "group_kind": cohort.info["group_kind"], "cohort_sha256": cohort.key,
            "run_samples_a": len(run_a.rows), "run_samples_b": len(run_b.rows) if run_b else None,
            "common_fraction_a": len(cohort.paths) / len(run_a.rows),
            "common_fraction_b": len(cohort.paths) / len(run_b.rows) if run_b else None,
            "run_scope_a": run_scope(run_a), "run_scope_b": run_scope(run_b) if run_b else "",
            "missing_class_resamples": missing_class_draws}
    return [base | {"metric": metric, "estimate": float(point[i]), "ci_lower": float(lower[i]),
                    "ci_upper": float(upper[i]), "score_a": float(point_a[i]),
                    "score_b": float(point_b[i]) if run_b else None}
            for i, metric in enumerate(METRICS)]


def name(model, registry):
    return consensus.SHORT_NAMES.get(model, registry.get(model, {}).get("name", model))


def formatted(row, difference=False):
    sign = "+" if difference else ""
    return (f"{100 * row['estimate']:{sign}.2f} "
            f"[{100 * row['ci_lower']:{sign}.2f}, {100 * row['ci_upper']:{sign}.2f}]")


def oriented_row(index, dataset, model_a, model_b, metric):
    key = (dataset, model_a, model_b, metric)
    if key in index:
        return index[key]
    reversed_row = index.get((dataset, model_b, model_a, metric))
    if reversed_row is None:
        return None
    oriented = reversed_row | {"model_a": model_a, "model_b": model_b,
                           "estimate": -reversed_row["estimate"],
                           "ci_lower": -reversed_row["ci_upper"], "ci_upper": -reversed_row["ci_lower"],
                           "score_a": reversed_row["score_b"], "score_b": reversed_row["score_a"]}
    for field in ("run_samples", "common_fraction", "run_scope"):
        oriented[f"{field}_a"], oriented[f"{field}_b"] = reversed_row[f"{field}_b"], reversed_row[f"{field}_a"]
    return oriented


def render_report(rows, manifest, registry, language):
    zh = language == "zh"
    params = manifest["parameters"]
    coverage = manifest["coverage"]
    singles = [row for row in rows if row["kind"] == "single"]
    differences = [row for row in rows if row["kind"] == "paired_difference"]
    single_index = {(row["dataset"], row["model_a"], row["metric"]): row for row in singles}
    paired_index = {(row["dataset"], row["model_a"], row["model_b"], row["metric"]): row for row in differences}
    level = f"{100 * params['confidence']:g}%"
    coverage_text = "; ".join(f"{consensus.NAMES[dataset]} {len(info['models'])}" for dataset, info in coverage.items())
    if zh:
        lines = [f"# Bootstrap：得分与模型差距的 {level} 置信区间", "",
                 f"直接使用现有预测，未重新运行模型。当前纳入模型数：{coverage_text}。",
                 f"按组有放回抽样 {params['n_resamples']:,} 次，随机种子 `{params['seed']}`；"
                 f"使用 percentile 方法，取两端各 {(1 - params['confidence']) * 50:g}% 分位点。", "",
                 "全体保存样本的分数仍是点估计。Bootstrap 反复改变说话人／对话的构成，观察分数和模型差距如何波动。"
                 "区间较窄表示在这种组构成变化下较稳定。95% 是区间构造方法的名义覆盖率："
                 "若重复取得类似数据并构造区间，目标是约 95% 的区间覆盖总体分数；它不是这一条区间内真实分数的概率。", "",
                 "## 抽样范围", "", "| 数据集 | 分组单位 | 当前片段数 | 当前组数 | 可用模型 |",
                 "|---|---|---:|---:|---:|"]
    else:
        lines = [f"# Bootstrap: {level} confidence intervals for scores and model differences", "",
                 f"Saved predictions only; no models were rerun. Models currently included: {coverage_text}.",
                 f"Resample whole clusters with replacement {params['n_resamples']:,} times; seed `{params['seed']}`. "
                 f"Percentile intervals use the {(1 - params['confidence']) * 50:g}th and "
                 f"{100 - (1 - params['confidence']) * 50:g}th percentiles.", "",
                 "Scores on all saved samples remain the point estimates. Bootstrap changes the mixture of speakers or "
                 "dialogues and measures how scores and model differences vary. Narrower intervals indicate greater "
                 "stability under that resampling scheme. The 95% level is nominal coverage of the procedure, not a "
                 "95% probability that a fixed parameter lies inside this particular interval.", "",
                 "## Resampling scope", "", "| Dataset | Cluster unit | Current clips | Current clusters | Models |",
                 "|---|---|---:|---:|---:|"]
    for dataset, info in coverage.items():
        unit = ("对话" if dataset == "meld" else "说话人") if zh else ("Dialogue" if dataset == "meld" else "Speaker")
        samples = sorted(set(info["sample_counts"].values()))
        clusters = sorted(set(info["cluster_counts"].values()))
        lines.append(f"| {consensus.NAMES[dataset]} | {unit} | " + ", ".join(f"{n:,}" for n in samples)
                     + " | " + ", ".join(map(str, clusters)) + f" | {len(info['models'])} |")
    lines += ["", ("每次抽取与原来相同数量的组，抽中的组带上全部片段；重复抽到的组重复计数。"
                   "按片段汇总后计算指标，既不重新预测，也不把各说话人的分数直接平均。" if zh else
                   "Each draw samples the original number of clusters and includes every clip of each selected cluster; "
                   "repeated clusters repeat all their clips. Metrics are pooled over clips, not averaged over speaker scores."), "",
              "## 单模型得分" if zh else "## Individual model scores", "",
              (f"单元格为分数 [{level} CI]，均以百分数表示。N / G 为片段数／组数。每个模型使用其保存预测的全部片段。"
               "带 * 的行表示只覆盖已声明评估范围中的部分样本。" if zh else
               f"Cells are score [{level} CI], in percent. N / G means clips / clusters. Each model uses all its saved "
               "predictions; * marks a partial run relative to its declared evaluation scope."), ""]
    for dataset, info in coverage.items():
        lines += [f"### {consensus.NAMES[dataset]}", "",
                  "| 模型 | N / G | Accuracy | UAR | Macro-F1 | Weighted-F1 |" if zh else
                  "| Model | N / G | Accuracy | UAR | Macro-F1 | Weighted-F1 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for model in info["models"]:
            record = single_index[dataset, model, "accuracy"]
            marker = "*" if record["run_scope_a"] == "partial_declared_scope" else ""
            values = " | ".join(formatted(single_index[dataset, model, metric]) for metric in METRICS)
            lines.append(f"| {name(model, registry)}{marker} | {record['n_samples']:,} / {record['n_clusters']} | {values} |")
        missing = [name(model, registry) for model in info["missing_models"]]
        if missing:
            lines += ["", ("暂无当前协议的结果：" if zh else "No current-protocol results: ") + ", ".join(missing) + "."]
        lines.append("")
    lines += ["## 重点模型的配对差值" if zh else "## Paired differences for the selected models", "",
              ("这里沿用精选模型组合，展示 emotion2vec+、Gemini、Qwen2.5-Omni 之间的 UAR 与 Macro-F1 差值；"
               "未按本次分数挑选比较对象。CSV 包含所有可用模型对、四项指标。数值是 A−B，单位为百分点。"
               "CI 跨过 0，表示本次分析不足以确定差距方向；不跨 0 也不是经多重比较校正的显著性检验。" if zh else
               "These comparisons use the existing selected-model set: emotion2vec+, Gemini and Qwen2.5-Omni. They "
               "were not selected by the scores in this analysis. The CSV includes every available model pair and "
               "all four metrics. Values are A−B in percentage points. An interval crossing zero does not resolve "
               "the direction of the difference; excluding zero is not a multiplicity-adjusted significance test."), "",
              "| 数据集 | A − B | 共同 N / G | Δ UAR | Δ Macro-F1 |" if zh else
              "| Dataset | A − B | Common N / G | Δ UAR | Δ Macro-F1 |",
              "|---|---|---:|---:|---:|"]
    focus = []
    for dataset in coverage:
        for model_a, model_b in FOCUS_PAIRS:
            a = oriented_row(paired_index, dataset, model_a, model_b, "unweighted_accuracy")
            b = oriented_row(paired_index, dataset, model_a, model_b, "macro_f1")
            if b is None:
                continue
            focus.append(b)
            lines.append(f"| {consensus.NAMES[dataset]} | {name(model_a, registry)} − {name(model_b, registry)} | "
                         f"{b['n_samples']:,} / {b['n_clusters']} | {formatted(a, True)} | {formatted(b, True)} |")
    if not focus:
        lines.append("| — | — | — | — | — |")
    lines += ["", "当前结论：" if zh else "Current findings:", ""]
    for row in focus:
        a, b = name(row["model_a"], registry), name(row["model_b"], registry)
        if row["ci_lower"] > 0:
            conclusion = (f"区间支持 {a} 的 Macro-F1 高于 {b}。" if zh else
                          f"The interval supports higher Macro-F1 for {a} than {b}.")
        elif row["ci_upper"] < 0:
            conclusion = (f"区间支持 {b} 的 Macro-F1 高于 {a}。" if zh else
                          f"The interval supports higher Macro-F1 for {b} than {a}.")
        else:
            conclusion = (f"{a} 与 {b} 的 Macro-F1 区间差值包含 0，差距方向尚不明确。" if zh else
                          f"The Macro-F1 difference between {a} and {b} includes zero; its direction remains unresolved.")
        lines.append(f"- {consensus.NAMES[row['dataset']]}: {conclusion}")
        uar = oriented_row(paired_index, row["dataset"], row["model_a"], row["model_b"], "unweighted_accuracy")
        if uar["ci_lower"] <= 0 <= uar["ci_upper"]:
            lines.append(f"- {consensus.NAMES[row['dataset']]}: " +
                         (f"{a} 与 {b} 的 UAR 差值区间包含 0，尚不能确定谁更高。" if zh else
                          f"The UAR difference between {a} and {b} includes zero; neither has a resolved advantage."))
    missing_draws = sum(info["missing_class_resamples"] for info in manifest["cohorts"].values())
    if zh:
        lines += ["", "## 如何解读及限制", "",
                  "- 配对比较只使用双方相同 audio_path 的交集，核对金标／类别规则一致，并对双方使用同一次抽样。"
                  "差值区间直接来自每次抽样的 A−B，不是两个单模型区间相减；两个单模型区间重叠也不决定差值是否可靠。",
                  "- 类别集合固定。invalid 作为错误保留，计入样本数及漏判，不增加一个宏平均类别。"
                  "某次抽样缺少一类时，该类 recall/F1 按 0 处理；weighted-F1 使用当次抽样的真实类别占比。"
                  f"本次有缺类的重采样数为 {missing_draws}（按不同样本交集计数）。",
                  "- 这些是固定模型在当前数据集组构成变化下的近似区间，不包含训练／API 重跑波动。"
                  "模型可能见过相关训练数据，因此不能据此证明未见说话人的泛化。RAVDESS 仅有 24 位说话人，区间仍需谨慎解读。",
                  "- MELD 按对话分组；不同对话可能共享演员或 episode，所以不等于新演员／新 episode 泛化。"
                  "未做 episode 层级敏感性分析，也未校正多重比较。不能将这些区间直接外推到公司电话。", "",
                  "## 复现与更新", "", "在 `research_paper/` 下运行：", "",
                  "```bash", f"python scripts/bootstrap.py --resamples {params['n_resamples']} --seed {params['seed']}", "```", "",
                  "默认自动发现当前协议下的全部可用模型，并选每个模型／数据集最大的已完成保存结果。"
                  "新增模型结果后重跑即可更新。相同输入、参数及 NumPy 版本可复现；抽样流由种子和样本交集确定，"
                  "加入其他模型不改变既有比较。所有结果集中在 [bootstrap.csv](../output/bootstrap.csv)，"
                  "参数、源文件 SHA256、分组来源与诊断在 [manifest](../output/bootstrap_manifest.json)。"
                  "CSV 的 estimate／ci_lower／ci_upper 用原始分数尺度保存；本报告乘 100 后显示为百分数或百分点。", "",
                  "方法参考：[SciPy bootstrap 文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)"
                  "（percentile 与 paired 定义）；[Cameron & Miller, 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf)"
                  "（整组抽样及少组数的限制；该文讨论回归，本项目仅采用其整组重采样原则）。"]
    else:
        lines += ["", "## Interpretation and limitations", "",
                  "- Each comparison uses the two models' common audio paths, identical gold/candidate rules and the "
                  "same resampled clusters. Its interval comes from replicate-wise A−B, not subtraction of marginal "
                  "interval endpoints. Overlapping individual-model intervals do not determine the paired result.",
                  "- The class set is fixed. Invalid answers remain errors, count in N/false negatives, and are not an "
                  "extra macro-average class. Recall/F1 is zero for a class absent in a draw. Weighted-F1 uses each "
                  f"draw's gold supports. Resamples missing a class: {missing_draws}, counted over distinct cohorts.",
                  "- These approximate intervals concern fixed predictions under changes in the observed cluster mixture; "
                  "they exclude training/API rerun variability. Possible training exposure prevents claims of unseen-speaker "
                  "generalization. RAVDESS has only 24 speakers, so interpret its intervals cautiously.",
                  "- MELD clusters dialogues. Recurring actors/episodes can create dependence across dialogues. This is "
                  "neither unseen-actor nor unseen-episode inference. Episode sensitivity and multiplicity corrections "
                  "were not performed. The intervals do not establish accuracy on company calls.", "",
                  "## Reproduce and update", "", "From `research_paper/`:", "", "```bash",
                  f"python scripts/bootstrap.py --resamples {params['n_resamples']} --seed {params['seed']}", "```", "",
                  "The default discovers all available current-protocol models and selects the largest finished saved "
                  "run per model/dataset. Rerun after new predictions arrive. Identical inputs, parameters and NumPy "
                  "version reproduce the output. Streams depend on the seed and cohort, so adding another model does "
                  "not change existing comparisons. All results are in [bootstrap.csv](../output/bootstrap.csv); "
                  "parameters, source SHA256s, grouping provenance and diagnostics are in the "
                  "[manifest](../output/bootstrap_manifest.json). CSV estimates and bounds use the raw score scale; "
                  "this report multiplies them by 100 to show percentages or percentage points.", "",
                  "Method references: [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) "
                  "for percentile/paired definitions; [Cameron & Miller, 2015](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf) "
                  "for whole-cluster resampling and few-cluster cautions. That paper concerns regression; this analysis "
                  "uses its cluster-resampling principle only."]
    return "\n".join(lines) + "\n"


def run(project=consensus.PROJECT, paper=None, *, datasets=DATASETS, models=None,
        n_resamples=DEFAULT_RESAMPLES, seed=DEFAULT_SEED, confidence=.95):
    project = Path(project).resolve()
    paper = Path(paper).resolve() if paper is not None else project.parent / "research_paper"
    if (type(n_resamples) is not int or n_resamples < 100 or type(seed) is not int or seed < 0
            or not 0 < confidence < 1):
        raise ValueError("Use at least 100 resamples, a nonnegative integer seed, and 0 < confidence < 1")
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
        raise ValueError(f"Choose distinct datasets from {DATASETS}")
    implementation_paths = {
        "research_paper/scripts/bootstrap.py": Path(__file__).resolve(),
        "research_paper/scripts/consensus.py": Path(consensus.__file__).resolve(),
        "emotion_recognition/scripts/analysis/probe_metadata.py": Path(sys.modules[infer_groups.__module__].__file__).resolve(),
        "emotion_recognition/scripts/llm_eval/prompts.py": Path(sys.modules[consensus.configured_protocol.__module__].__file__).resolve(),
        "emotion_recognition/config/models.yaml": project / "config/models.yaml",
    }
    llm_config = project / "config/llm.yaml"
    if llm_config.is_file():
        implementation_paths["emotion_recognition/config/llm.yaml"] = llm_config
        prompt_path = project / (yaml.safe_load(llm_config.read_text()) or {}).get("prompts", "scripts/llm_eval/prompts.yaml")
        if prompt_path.is_file():
            prompt_label = ("emotion_recognition/" + prompt_path.relative_to(project).as_posix()
                            if prompt_path.is_relative_to(project) else prompt_path.as_posix())
            implementation_paths[prompt_label] = prompt_path
    implementation_before = {label: consensus.sha256(path) for label, path in implementation_paths.items()}
    registry = yaml.safe_load((project / "config/models.yaml").read_text()) or {}
    if models is not None and (not models or len(set(models)) != len(models) or not set(models) <= set(registry)):
        raise ValueError("Choose distinct registered models")
    model_order = [model for model in registry if models is None or model in models]
    selected, selection = consensus.load_runs(project, registry, datasets=datasets, exclude=["esd"])
    selected = {model: runs for model, runs in selected.items() if model in model_order}
    manifest = {"schema_version": 1, "algorithm": ALGORITHM,
                "parameters": {"n_resamples": n_resamples, "seed": seed, "confidence": confidence,
                               "datasets": list(datasets), "models": list(models) if models is not None else "all",
                               "metrics": list(METRICS), "interval": "percentile", "quantile_method": "linear",
                               "rng": "PCG64", "class_policy": "fixed candidates; zero_division=0",
                               "multiplicity_adjustment": None},
                "software": {"python": platform.python_version(), "numpy": np.__version__, "pyyaml": yaml.__version__},
                "coverage": {}, "cohorts": {}, "runs": {}, "selection": selection}
    rows = []
    # Validate every dataset and pair before writing any result. Cache by cohort,
    # so current full runs share both random weights and score distributions.
    for dataset in datasets:
        available = [model for model in model_order if dataset in selected.get(model, {})]
        if not available:
            raise ValueError(f"{dataset}: no compatible saved predictions")
        info = {"models": available, "missing_models": [m for m in model_order if m not in available],
                "sample_counts": {}, "cluster_counts": {}}
        manifest["coverage"][dataset] = info
        cohorts, distributions = {}, {}

        def evaluate(run, paths):
            signature = digest(paths)
            if signature not in cohorts:
                cohorts[signature] = make_cohort(dataset, paths, run, project, n_resamples, seed)
            cohort = cohorts[signature]
            if (signature, run.model) not in distributions:
                distributions[signature, run.model] = distribution(run, cohort)
            return cohort, distributions[signature, run.model]

        for model in available:
            current = selected[model][dataset]
            paths = sorted(current.rows)
            cohort, (point, draws, missing) = evaluate(current, paths)
            reported = load_reported_metrics(current.path, current.metadata)
            for i, metric in enumerate(METRICS):
                if metric in reported and abs(float(reported[metric]) - point[i]) > 1.5e-6:
                    raise ValueError(f"{dataset}/{model}: recomputed {metric} differs from recorded metric")
            info["sample_counts"][model] = len(paths)
            info["cluster_counts"][model] = cohort.info["n_clusters"]
            manifest["runs"][f"{dataset}/{model}"] = {
                "sources": current.sources, "classes": current.classes, "ground_truth": current.metadata.get("ground_truth"),
                "subset": current.metadata.get("subset"), "scope": run_scope(current),
                "inference_protocol": current.metadata.get("inference_protocol"), "cohort_sha256": cohort.key}
            rows.extend(result_rows("single", dataset, current, None, cohort, point, draws, confidence,
                                    missing_class_draws=missing))
        for model_a, model_b in itertools.combinations(available, 2):
            runs, common = consensus.prepare_cohort(dataset, [model_a, model_b], selected)
            cohort_a, (point_a, draws_a, missing_a) = evaluate(runs[0], common)
            cohort_b, (point_b, draws_b, missing_b) = evaluate(runs[1], common)
            if cohort_a.key != cohort_b.key:
                raise ValueError("Paired models must share their cohort")
            rows.extend(result_rows("paired_difference", dataset, runs[0], runs[1], cohort_a,
                                    point_a, draws_a, confidence, point_b, draws_b, max(missing_a, missing_b)))
        for cohort in cohorts.values():
            missing_counts = [item[2] for (signature, _), item in distributions.items()
                              if cohorts[signature].key == cohort.key]
            manifest["cohorts"][cohort.key] = cohort.info | {"missing_class_resamples": max(missing_counts)}
        print(f"{consensus.NAMES[dataset]}: {len(available)} models, {len(cohorts)} sample cohort(s), "
              f"{len(available) * (len(available) - 1) // 2} paired comparisons")
    source_files = {source["path"]: source for runs in selected.values() for current in runs.values()
                    for source in current.sources}
    for source in source_files.values():
        if consensus.sha256(project / source["path"]) != source["sha256"]:
            raise RuntimeError("Source predictions changed during bootstrap; rerun on a stable snapshot")
    for label, path in implementation_paths.items():
        if consensus.sha256(path) != implementation_before[label]:
            raise RuntimeError(f"Implementation/configuration changed during bootstrap: {label}")
    manifest["implementation_sources"] = [{"path": label, "sha256": implementation_before[label]}
                                          for label in implementation_paths]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = stream.getvalue()
    manifest["results"] = {"csv": "output/bootstrap.csv", "rows": len(rows),
                           "single_model_runs": sum(r["kind"] == "single" for r in rows) // len(METRICS),
                           "model_pairs": sum(r["kind"] == "paired_difference" for r in rows) // len(METRICS),
                           "sha256": hashlib.sha256(csv_text.encode()).hexdigest()}
    reports = {language: render_report(rows, manifest, registry, language) for language in ("zh", "en")}
    (paper / "output").mkdir(parents=True, exist_ok=True)
    (paper / "analysis_report").mkdir(parents=True, exist_ok=True)
    (paper / "output/bootstrap.csv").write_text(csv_text, encoding="utf-8")
    (paper / "output/bootstrap_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for language, content in reports.items():
        (paper / f"analysis_report/bootstrap_{language}.md").write_text(content, encoding="utf-8")
    print(f"Wrote {len(rows)} metric rows to {paper / 'output/bootstrap.csv'} and bilingual bootstrap reports.")
    return rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=.95)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", help="Registered model keys; default: every compatible available model")
    args = parser.parse_args()
    try:
        run(datasets=args.datasets, models=args.models, n_resamples=args.resamples,
            seed=args.seed, confidence=args.confidence)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

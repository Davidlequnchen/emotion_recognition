"""Render the saved-prediction duration/domain analysis in English or Chinese.

This module only formats supplied analysis results. It does not read audio,
select runs, fit models, or turn observational associations into causal claims.
"""
from __future__ import annotations

from collections import Counter
import math


SER_MODELS = ("emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion",
              "vox_profile_whisper", "vox_profile_wavlm")
OMNI = "qwen2_5_omni_7b"
QWEN_AUDIO = "qwen2_audio_7b"
GEMMA = "gemma4_e4b"
CONVERSATIONS = ("meld", "emotiontalk", "dailytalk", "multidialog")
DEFAULT_NAMES = {
    "emotion2vec_plus_large": "emotion2vec+ large", "meralion_ser_v1": "MERaLiON-SER-v1",
    "melati_emotion": "Melati", "vox_profile_whisper": "Vox-Profile Whisper",
    "vox_profile_wavlm": "Vox-Profile WavLM", OMNI: "Qwen2.5-Omni 7B",
    QWEN_AUDIO: "Qwen2-Audio 7B Instruct", GEMMA: "Gemma 4 E4B",
    "gpt_realtime": "GPT Realtime", "gemini-3.8-flash": "Gemini 3.8 Flash",
}


def _number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _percent(value, decimals=2):
    value = _number(value)
    return "—" if value is None else f"{100 * value:.{decimals}f}%"


def _decimal(value, decimals=2, signed=False):
    value = _number(value)
    if value is None:
        return "—"
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def _pvalue(value):
    value = _number(value)
    return "—" if value is None else ("<0.001" if value < .001 else f"{value:.3f}")


def _integer(value):
    value = _number(value)
    return "—" if value is None else f"{int(value):,}"


def _mean(values):
    values = [number for value in values if (number := _number(value)) is not None]
    return sum(values) / len(values) if values else None


def _escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(map(_escape, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(_escape, row)) + " |" for row in rows])


def _ci(row):
    lo, hi = _number(row.get("ci_lower_pp")), _number(row.get("ci_upper_pp"))
    return "—" if lo is None or hi is None else f"[{lo:+.2f}, {hi:+.2f}]"


def _slope_ci(row):
    return f"{_decimal(row.get('slope_pp'), signed=True)} {_ci(row)}"


def _evidence(row, zh):
    if _number(row.get("slope_pp")) is None:
        return "无法识别" if zh else "Not identified"
    lo, hi = _number(row.get("ci_lower_pp")), _number(row.get("ci_upper_pp"))
    if lo is None or hi is None:
        return "无可用区间" if zh else "Interval unavailable"
    if lo > 0:
        return "正关联，区间不跨 0" if zh else "Positive; interval excludes 0"
    if hi < 0:
        return "负关联，区间不跨 0" if zh else "Negative; interval excludes 0"
    return "方向未确定，区间跨 0" if zh else "Direction unresolved; interval crosses 0"


def render_report(data, language):
    """Return Markdown for ``language='zh'`` or ``'en'`` from analysis dictionaries.

    Rate and metric inputs are fractions; duration slopes and their confidence
    limits already use percentage points per doubling. Missing optional values
    remain explicit em dashes rather than silently becoming zero.
    """
    if language not in ("zh", "en"):
        raise ValueError("language must be 'zh' or 'en'")
    zh = language == "zh"
    manifest = data["manifest"]
    datasets = list(data["dataset_rows"])
    models = list(data["model_rows"])
    groups = list(data.get("group_rows", []))
    effects = list(data.get("effects", []))
    dataset_effects = list(data.get("dataset_effects", []))
    gaps = list(data.get("gap_effects", []))
    correlations = list(data.get("correlations", []))
    names = DEFAULT_NAMES | manifest.get("model_names", {})
    order = list(manifest.get("model_order", dict.fromkeys(row["model"] for row in models)))
    full = list(manifest.get("full_models", [model for model in order
                if sum(row["model"] == model for row in models) == len(datasets)]))
    ds = {row["dataset"]: row for row in datasets}
    mr = {(row["dataset"], row["model"]): row for row in models}
    total_n = sum(int(row["n"]) for row in datasets)
    present_conversations = [key for key in CONVERSATIONS if key in ds]
    model_name = lambda model: names.get(model, model)
    dataset_name = lambda key: ds.get(key, {}).get("name", key)

    def row_for(model, fit, cluster="dataset"):
        candidates = [row for row in effects if row["model"] == model and row["fit"] == fit
                      and row["cluster_kind"] == cluster]
        scope = "16 datasets" if model in full else "MELD only"
        return next((row for row in candidates if row.get("scope") == scope),
                    candidates[0] if candidates else {})

    def best_ser(key, metric):
        candidates = [mr[(key, model)] for model in SER_MODELS if (key, model) in mr
                      and _number(mr[(key, model)].get(metric)) is not None]
        return max(candidates, key=lambda row: row[metric]) if candidates else {}

    def group_cell(model, grouping, level):
        row = next((r for r in groups if r["model"] == model and r["grouping"] == grouping
                    and r["level"] == level), {})
        return " / ".join(_percent(row.get(metric), 1)
                          for metric in ("accuracy", "uar", "macro_f1"))

    def correlation_cell(model, metric):
        row = next((r for r in correlations if r["model"] == model and r["metric"] == metric
                    and r.get("scope") == "16 datasets"), {})
        return f"{_decimal(row.get('spearman_rho'))} ({_pvalue(row.get('pvalue'))})"

    out = []
    out.append("# 十六数据集的模型差异、语言、对话类型与音频时长分析" if zh else
               "# Model differences, language, conversational setting and audio duration across sixteen datasets")
    generated = manifest.get("generated_at_singapore", "—")
    selected = manifest.get("selected_run_count", len(models))
    out.append((f"生成时间：{generated}（Singapore）。本报告复核 {len(datasets)} 个数据集、"
                f"{total_n:,} 条不同音频及 {selected} 个采用的 model–dataset runs。"
                f"{len(full)} 个模型覆盖全部数据集；Gemma 4 当前只提供 MELD 结果。"
                "所有数字来自已经保存的预测与音频文件元数据，本次分析没有重新运行模型或调用 API。") if zh else
               (f"Generated: {generated} (Singapore). This report audits {len(datasets)} datasets, "
                f"{total_n:,} distinct clips and {selected} selected model–dataset runs. "
                f"{len(full)} models cover all datasets; Gemma 4 currently contributes MELD only. "
                "All results use saved predictions and audio-file metadata; this analysis runs no new inference or API requests."))
    out.append("## 1. 先校正对四个对话数据集的观察" if zh else
               "## 1. First, refine the observation about the four conversational datasets")
    wins_accuracy = sum(_number(mr.get((key, OMNI), {}).get("accuracy")) is not None
                        and bool(best_ser(key, "accuracy"))
                        and mr[key, OMNI]["accuracy"] > best_ser(key, "accuracy")["accuracy"]
                        for key in present_conversations)
    wins_f1 = sum(_number(mr.get((key, OMNI), {}).get("macro_f1")) is not None
                  and bool(best_ser(key, "macro_f1"))
                  and mr[key, OMNI]["macro_f1"] > best_ser(key, "macro_f1")["macro_f1"]
                  for key in present_conversations)
    out.append((f"Qwen2.5-Omni 的优势取决于评价指标：在这 {len(present_conversations)} 个数据集上，"
                f"它的 Accuracy 超过全部五个 SER 模型的有 {wins_accuracy} 个，macro-F1 超过全部 SER 的有 {wins_f1} 个。"
                "因此，不能把四个数据集上的较高 Accuracy 概括为四次全面领先。下表中 Accuracy 和 macro-F1 "
                "各自与该指标最好的 SER 相比，两个最佳 SER 可以不同。") if zh else
               (f"Qwen2.5-Omni's advantage depends on the metric: it exceeds every one of the five SER models "
                f"on Accuracy in {wins_accuracy}/{len(present_conversations)} datasets and on macro-F1 in "
                f"{wins_f1}/{len(present_conversations)}. A high Accuracy therefore does not imply four across-the-board wins. "
                "The best SER is selected separately for each displayed metric and can be a different model."))
    conversation_rows = []
    for key in present_conversations:
        omni, ba, bf = mr.get((key, OMNI), {}), best_ser(key, "accuracy"), best_ser(key, "macro_f1")
        conversation_rows.append([dataset_name(key), _percent(omni.get("accuracy")),
                                  f"{_percent(ba.get('accuracy'))} ({model_name(ba.get('model', '—'))})",
                                  _percent(omni.get("uar")), _percent(omni.get("macro_f1")),
                                  f"{_percent(bf.get('macro_f1'))} ({model_name(bf.get('model', '—'))})",
                                  _percent(ds[key].get("neutral_accuracy"))])
    out.append(_table((["数据集", "Omni Accuracy", "最佳 SER Accuracy", "Omni UAR", "Omni macro-F1",
                        "最佳 SER macro-F1", "始终预测 neutral"] if zh else
                       ["Dataset", "Omni Accuracy", "Best SER Accuracy", "Omni UAR", "Omni macro-F1",
                        "Best SER macro-F1", "Always neutral"]), conversation_rows))
    baseline_sentences = []
    for key in ("dailytalk", "multidialog"):
        if key not in ds:
            continue
        rows = [row for row in models if row["dataset"] == key]
        best = max(rows, key=lambda row: row["accuracy"]) if rows else {}
        neutral = _number(ds[key].get("neutral_accuracy"))
        exceeds = neutral is not None and _number(best.get("accuracy")) is not None and neutral > best["accuracy"]
        if zh:
            baseline_sentences.append(f"{dataset_name(key)} 的 neutral 常量基线为 {_percent(neutral)}，"
                                      f"{'高于所有已完成模型' if exceeds else '应与模型一并比较'}"
                                      f"（最好模型 {_percent(best.get('accuracy'))}）。")
        else:
            baseline_sentences.append(f"The always-neutral baseline is {_percent(neutral)} on {dataset_name(key)}, "
                                      f"{'higher than every completed model' if exceeds else 'an essential comparator'} "
                                      f"(best model: {_percent(best.get('accuracy'))}).")
    out.append(" ".join(baseline_sentences) + (" Accuracy 是全部样本中预测正确的比例；类别不均衡时可被多数类支配。"
               "UAR 是各真实类别 recall 的等权平均，macro-F1 是各类别 F1 的等权平均，weighted-F1 按真实类别数量加权。"
               "较好的 macro-F1 可以与较低 Accuracy 同时出现，意味着少数类与多数类之间的取舍不同，"
               "并不表示整体已足够可靠或每个情绪类别都已解决。" if zh else
               " Accuracy is the fraction of all clips classified correctly and can be dominated by a majority class. "
               "UAR averages recall equally across gold classes; macro-F1 averages class F1 equally; weighted-F1 weights it by gold support. "
               "Better macro-F1 can coexist with lower Accuracy because majority/minority-class tradeoffs differ. "
               "Neither result establishes reliable performance for every emotion."))
    jl = mr.get(("jl_corpus", QWEN_AUDIO), {})
    jl_best = max((row for row in models if row["dataset"] == "jl_corpus"),
                  key=lambda row: row["accuracy"], default={})
    out.append((f"另一个需要校正的例子是 JL-Corpus：Qwen2-Audio Accuracy 为 {_percent(jl.get('accuracy'))}，"
                f"该数据集最高为 {_percent(jl_best.get('accuracy'))}（{model_name(jl_best.get('model', '—'))}）。"
                "它在部分表演语料上的表现相对更好，不等于在这些语料上均与领先 SER 持平。") if zh else
               (f"JL-Corpus is another useful correction: Qwen2-Audio Accuracy is {_percent(jl.get('accuracy'))}, "
                f"versus a best observed {_percent(jl_best.get('accuracy'))} from {model_name(jl_best.get('model', '—'))}. "
                "Its relative strength on some acted corpora does not mean it matches the leading SER model throughout that group."))

    out.append("## 2. 实际音频时长分布与输入窗口" if zh else
               "## 2. Actual duration distributions and model input windows")
    out.append("![十六数据集原始音频时长分布](figures/duration_distributions.png)" if zh else
               "![Raw audio duration distributions across sixteen datasets](figures/duration_distributions.png)")
    duration_rows = [[row.get("name", row["dataset"]), row.get("language", "—"),
                      ("对话" if row["conversation"] else "非对话") if zh else
                      ("Conversational" if row["conversation"] else "Other"), _integer(row["n"]),
                      _decimal(row.get("median_seconds")),
                      f"{_decimal(row.get('q25_seconds'))}–{_decimal(row.get('q75_seconds'))}",
                      _decimal(row.get("q95_seconds")), _percent(row.get("pct_under3"), 1),
                      _percent(row.get("pct_over10"), 1)] for row in datasets]
    out.append(_table((["数据集", "语言", "类型", "N", "中位数 s", "Q25–Q75 s", "Q95 s", "<3 s", ">10 s"] if zh else
                       ["Dataset", "Language", "Setting", "N", "Median s", "Q25–Q75 s", "Q95 s", "<3 s", ">10 s"]),
                      duration_rows))
    finite_durations = [row for row in datasets if _number(row.get("median_seconds")) is not None]
    shortest = min(finite_durations, key=lambda row: row["median_seconds"], default={})
    longest = max(finite_durations, key=lambda row: row["median_seconds"], default={})
    under3 = sum(row["n"] * row["pct_under3"] for row in datasets) / total_n if total_n else None
    over10 = sum(row["n"] * row["pct_over10"] for row in datasets) / total_n if total_n else None
    out.append((f"最短与最长的数据集中位数分别是 {shortest.get('name', '—')} "
                f"{_decimal(shortest.get('median_seconds'))} s 和 {longest.get('name', '—')} "
                f"{_decimal(longest.get('median_seconds'))} s；按音频条数汇总，{_percent(under3, 1)} 短于 3 s，"
                f"{_percent(over10, 1)} 长于 10 s。分位数展示分布，而不是用单个平均值代表整套语料。"
                "这里的时长由实际文件帧数与采样率计算，包含静音、停顿和背景声；它不是语音活动检测后的有效说话时长。") if zh else
               (f"Dataset medians range from {_decimal(shortest.get('median_seconds'))} s on {shortest.get('name', '—')} "
                f"to {_decimal(longest.get('median_seconds'))} s on {longest.get('name', '—')}. Across clips, "
                f"{_percent(under3, 1)} are shorter than 3 s and {_percent(over10, 1)} exceed 10 s. Quantiles expose "
                "the distribution instead of replacing a corpus with one average. Durations come from file frames and sample rates; "
                "they include silence, pauses and background sound and are not voice-activity-detected speaking time."))
    out.append("本地 loader 的有效窗口不同：Vox-Profile 两个模型最多取前 15 s；Qwen2-Audio、Gemma 4、"
               "MERaLiON 和 Melati 最多使用前 30 s。Qwen2.5-Omni 的 loader 不显式裁切，但所附 processor 配置为 300 s 窗口；"
               "emotion2vec+ 未设显式固定截断。padding 与 pooling 也不同：MERaLiON 会把短音频补至 30 s 并对整段 pooling，"
               "Melati/Vox-Profile Whisper 只汇总实际帧。API 结果来自另一执行器，现存结果不足以核实服务端的完整裁切策略。"
               "因此，图中的原始时长不等于每个模型实际听到的长度，15/30 s 以上的延长对某些模型只会改变原始时长标签。" if zh else
               "Effective local windows differ: both Vox-Profile models take at most the first 15 s; Qwen2-Audio, Gemma 4, "
               "MERaLiON and Melati use at most the first 30 s. Qwen2.5-Omni has no explicit loader crop, but its bundled processor "
               "configuration has a 300 s window; emotion2vec+ has no explicit fixed crop. Padding and pooling also differ: "
               "MERaLiON pads to 30 s and pools the whole representation, while Melati/Vox-Profile Whisper pool actual frames. "
               "The API outputs were imported from another runner and do not verify the complete server-side crop policy. "
               "Raw file duration therefore differs from effective heard duration; extra raw seconds beyond a model's cap need not add input evidence.")

    out.append("## 3. 英语与非英语、对话与非对话：等数据集比较" if zh else
               "## 3. English/non-English and conversational/other settings: equal-dataset comparisons")
    out.append("![语言与对话类型分组表现](figures/language_context_performance.png)" if zh else
               "![Performance grouped by language and conversational setting](figures/language_context_performance.png)")
    out.append((f"下面只比较覆盖全部 {len(datasets)} 个数据集的 {len(full)} 个模型，每套数据集在组内权重相同，"
                "而非让大数据集贡献更多样本权重；Gemma 4 不参加这项比较。每格依次为 Accuracy / UAR / macro-F1。"
                "这些是所选语料的描述统计，类别数、类别语义、标注方式和训练接触情况并未被分组消除，"
                "不同组的平均值不是同一任务难度下的语言效应。") if zh else
               (f"This comparison includes only the {len(full)} models covering all {len(datasets)} datasets and gives each "
                "dataset equal weight within a group; larger corpora do not dominate through clip count. Gemma 4 is excluded. "
                "Each cell is Accuracy / UAR / macro-F1. These are descriptive means for the selected corpora. "
                "Grouping does not remove differences in class inventory, label meaning, annotation or training exposure, "
                "so a group difference is not an isolated language effect at matched task difficulty."))
    out.append(_table((["模型", "英语", "非英语", "对话", "非对话"] if zh else
                       ["Model", "English", "Non-English", "Conversational", "Other"]),
                      [[model_name(model), group_cell(model, "language", "English"),
                        group_cell(model, "language", "non_English"),
                        group_cell(model, "context", "conversational"),
                        group_cell(model, "context", "non_conversational")] for model in full]))
    out.append("所有模型本次均输入单条 utterance，没有前后轮次、说话人历史或视频；“对话数据集”只描述录音来源，"
               "不表示模型在做带上下文的对话情绪识别。MELD 来自影视对话；DailyTalk 与 MultiDialog 为依据脚本表演的对话，"
               "EmotionTalk 也包含基于情境的演员表演，不能把四者都称为 spontaneous speech。DailyTalk 的情绪类别继承文本对话标签，"
               "MultiDialog 包含表演意图及人工质检，标签来源与纯声学感知并不完全相同。英语组与非英语组也不是同一语料的翻译版本。"
               "这些差异使语言理解、表演方式、标签来源和音频时长相互交织。" if zh else
               "Every model receives one utterance without neighboring turns, speaker history or video. A conversational corpus "
               "does not mean that contextual conversational emotion recognition was evaluated. MELD uses television dialogue; "
               "DailyTalk and MultiDialog contain performed scripted conversations, and EmotionTalk includes scenario-based acting. "
               "The four are not all spontaneous speech. DailyTalk's emotion classes inherit text-dialogue labels; MultiDialog includes "
               "intended delivery and human quality control. These labels do not all measure purely acoustic perception. "
               "English and non-English groups are also different corpora rather than translated versions of the same recordings. "
               "Language understanding, performance style, label origin and duration are consequently entangled.")

    out.append("## 4. 核心检验：较长音频是否与更高正确率相关？" if zh else
               "## 4. Primary test: is longer audio associated with higher correctness?")
    out.append("![控制数据集与真实类别后的时长关联及敏感性](figures/duration_effects.png)" if zh else
               "![Duration associations adjusted for dataset and gold class, with sensitivities](figures/duration_effects.png)")
    out.append("主分析拟合 `correct = 数据集×真实类别截距 + β·log2(原始时长)`，采用加权组内去均值。"
               "权重使每个数据集的总权重相同，并使一个数据集内各真实类别的总权重相同；这样既避免大语料主导，"
               "也避免 neutral 等多数类单独决定斜率。代价是稀有类别中每条样本会获得较大权重，所以必须同时查看未平衡估计与逐数据集结果。"
               "`β` 报告为原始时长翻倍对应正确率变化的百分点（pp）：例如 +2 pp 表示组内 2 s 与 4 s、4 s 与 8 s "
               "具有同一线性关联单位。它不是预测概率，也不表示将同一条音频加长就能提高 2 pp。" if zh else
               "The primary fit is `correct = dataset×gold-class intercept + β·log2(raw duration)`, estimated by weighted "
               "within-stratum demeaning. Weights give every dataset the same total weight and every gold class within a dataset "
               "the same total weight. This prevents large corpora and majority classes such as neutral from determining the slope. "
               "The tradeoff is greater per-clip weight for rare classes, making unbalanced estimates and dataset-specific results necessary checks. "
               "β is reported in percentage points (pp) of correctness per doubling: +2 pp uses the same association unit for 2→4 s and 4→8 s. "
               "It is not a probability prediction or evidence that lengthening a particular clip improves correctness by 2 pp.")
    out.append("主要 95% 区间使用以数据集为 cluster 的 CR1 标准误与 `t(G−1)` 分布；完整模型只有 16 个数据集单位，"
               "这比把数万条音频视为独立证据更保守，但仍依赖数据集 cluster 的建模假设，不自动推广到任意新语料。"
               "Gemma 4 只有 MELD 一个数据集，因此保留其描述性斜率，不生成数据集聚类区间。区间与 p 值未做多重比较校正，"
               "不应按单个显著结果挑选结论。" if zh else
               "Primary 95% intervals use dataset-clustered CR1 standard errors and t(G−1). Complete models have only sixteen "
               "dataset units, rather than tens of thousands of independent clips. This still depends on the cluster assumptions "
               "and does not guarantee generalization to arbitrary new corpora. Gemma 4 has one dataset, MELD, so its descriptive "
               "slope is retained without a dataset-clustered interval. Intervals and p-values are not multiplicity-adjusted; "
               "individual favorable results must not be selected as the conclusion.")
    primary_rows = []
    positive, negative, unresolved, no_interval = [], [], [], []
    for model in order:
        row = row_for(model, "balanced_within_dataset_gold")
        if not row:
            continue
        local = [r for r in dataset_effects if r["model"] == model
                 and _number(r.get("slope_pp")) is not None and r.get("status") != "not_identified"]
        npositive = sum(r["slope_pp"] > 0 for r in local)
        primary_rows.append([model_name(model), row.get("scope", "—"),
                             _decimal(row.get("slope_pp"), signed=True), _ci(row),
                             _pvalue(row.get("pvalue")), _integer(row.get("nclusters")),
                             f"{npositive}/{len(local)}", _evidence(row, zh)])
        if model in full:
            lo, hi = _number(row.get("ci_lower_pp")), _number(row.get("ci_upper_pp"))
            (no_interval if lo is None or hi is None else positive if lo > 0 else negative if hi < 0
             else unresolved).append(model_name(model))
    out.append(_table((["模型", "范围", "β pp/翻倍", "95% CI", "p", "数据集 clusters", "正斜率数据集", "解释"] if zh else
                       ["Model", "Scope", "β pp/doubling", "95% CI", "p", "Dataset clusters", "Positive dataset slopes", "Interpretation"]),
                      primary_rows))
    if zh:
        result_text = (f"在完整模型的主分析中，{len(positive)} 个模型的区间完全高于 0，{len(negative)} 个完全低于 0，"
                       f"{len(unresolved)} 个跨 0，{len(no_interval)} 个无可用区间。")
        if positive:
            result_text += "正关联区间不跨 0 的模型为 " + "、".join(positive) + "。"
        if unresolved:
            result_text += "区间跨 0 的模型，包含正点估计的情况，都应表述为方向尚未确定，而非已证实“长音频更好”。"
    else:
        result_text = (f"Among complete models, primary intervals are wholly positive for {len(positive)}, wholly negative "
                       f"for {len(negative)}, cross zero for {len(unresolved)}, and unavailable for {len(no_interval)}. ")
        if positive:
            result_text += "Positive intervals excluding zero occur for " + ", ".join(positive) + ". "
        if unresolved:
            result_text += "An interval crossing zero, including one around a positive estimate, leaves the direction unresolved rather than confirming a longer-is-better hypothesis."
    out.append(result_text)
    sensitivity_rows = []
    for model in order:
        balanced = row_for(model, "balanced_within_dataset_gold")
        if not balanced:
            continue
        sensitivity_rows.append([model_name(model), _slope_ci(row_for(model, "pooled")),
                                 _slope_ci(row_for(model, "within_dataset_gold")),
                                 _slope_ci(balanced),
                                 _slope_ci(row_for(model, "balanced_within_dataset_gold", "speaker_dialogue"))])
    out.append(_table((["模型", "合并样本", "组内未平衡", "组内平衡（主分析）", "说话人/对话聚类敏感性"] if zh else
                       ["Model", "Pooled", "Within strata, unbalanced", "Within strata, balanced (primary)", "Speaker/dialogue clustering sensitivity"]),
                      sensitivity_rows))
    out.append("每格为斜率与 95% CI，单位均为 pp/时长翻倍。Pooled 不控制数据集或类别，可出现 Simpson reversal："
               "较长录音来自更难语料，会让总斜率为负，即使同一数据集同一类别内为正，反向情况也可能出现。"
               "说话人/对话 cluster 使用带数据集命名空间的可恢复 ID，只检查同人/同对话相关性对区间的影响；"
               "它没有重新划分训练测试、没有测试新说话人，也不能替代主分析的跨数据集不确定性。" if zh else
               "Cells show slope and 95% CI in pp/doubling. The pooled fit does not control dataset or class and can exhibit "
               "a Simpson reversal: long clips may come from difficult corpora, producing a negative pooled slope despite "
               "positive within-dataset/class associations, or vice versa. Speaker/dialogue clustering uses recoverable IDs "
               "namespaced by dataset. It checks sensitivity to dependence within those groups; it does not create a new "
               "train/test split, evaluate new speakers, or replace the primary uncertainty across datasets.")
    out.append("![逐数据集时长关联的异质性](figures/duration_within_dataset_slopes.png)" if zh else
               "![Heterogeneity in duration associations across datasets](figures/duration_within_dataset_slopes.png)")
    out.append("主表中的“正斜率数据集”仅为可识别点估计的方向计数，不是显著结果计数，也不保证每套数据集均有同方向关联。"
               "逐数据集图展示这种异质性；单一共同斜率只是摘要，不能代替各领域、各类别的错误分析。"
               "时长变化范围很窄的语料可产生很大的单位斜率，例如 RAVDESS；超过 100 pp/翻倍的系数"
               "不是预测正确率超过 100%，也不能外推到未观察到的时长翻倍。" if zh else
               "The positive-dataset count describes the signs of identifiable point estimates, not the number of significant "
               "findings, and does not imply one direction in every corpus. The dataset-specific figure exposes this heterogeneity. "
               "One shared slope is a summary rather than a replacement for domain- and class-specific error analysis. "
               "A narrow observed duration range, as in RAVDESS, can produce a large unit slope. A coefficient above "
               "100 pp/doubling is not accuracy above 100% and must not be extrapolated to an unobserved doubling.")
    if gaps:
        out.append("模型之间的差异还需直接比较斜率，不能以“一个显著、另一个不显著”代替差异检验。下表对同一条音频的正确性差值建模；"
                   "正值表示相对 reference 的优势随时长增加，但它仍是调整后的观察性关联。" if zh else
                   "Between-model differences require a direct slope comparison; one significant slope and one non-significant "
                   "slope do not establish a difference. The following fits use paired per-clip correctness differences. "
                   "Positive values mean an advantage relative to the reference grows with duration, still as an adjusted observational association.")
        out.append(_table((["模型", "Reference", "范围", "差异 β pp/翻倍", "95% CI"] if zh else
                           ["Model", "Reference", "Scope", "Gap β pp/doubling", "95% CI"]),
                          [[model_name(row["model"]), model_name(row["reference"]), row.get("scope", "—"),
                            _decimal(row.get("slope_pp"), signed=True), _ci(row)] for row in gaps]))

    out.append("## 5. 分箱曲线与数据集均值相关性能说明什么" if zh else
               "## 5. What raw bins and correlations of dataset medians can tell us")
    out.append("![按原始时长分箱的实际 Accuracy](figures/accuracy_by_duration.png)" if zh else
               "![Observed Accuracy by raw-duration bin](figures/accuracy_by_duration.png)")
    out.append("时长分箱图的 x 轴是原始音频时长区间，y 轴是该区间内预测正确的样本比例；只有至少 50 条音频的点才展示。"
               "它便于定位短音频或长音频的错误集中位置，但分箱后的真实类别和数据集混合比例仍会变化，"
               "不是在相同音频上增加信息的实验。不能把曲线的上升或下降直接解释为时长造成的改善或退化。" if zh else
               "The bin plot puts raw-duration intervals on x and the fraction of correctly classified clips in each bin on y; "
               "only points supported by at least fifty clips are shown. It can locate concentrations of short- or long-clip errors, "
               "but gold-class and dataset composition still change across bins. It is not an experiment that adds information "
               "to the same recording, so an upward or downward curve does not establish a causal duration effect.")
    out.append(_table((["模型", "中位时长 vs Accuracy ρ (p)", "vs UAR ρ (p)", "vs macro-F1 ρ (p)"] if zh else
                       ["Model", "Median duration vs Accuracy ρ (p)", "vs UAR ρ (p)", "vs macro-F1 ρ (p)"]),
                      [[model_name(model), correlation_cell(model, "accuracy"), correlation_cell(model, "uar"),
                        correlation_cell(model, "macro_f1")] for model in full]))
    out.append("这些 Spearman 相关性把每套数据集作为一个点，仅有 16 个点，衡量“典型时长较长的语料是否更容易”。"
               "它与逐音频、控制数据集与真实类别的主检验回答不同问题；较高相关性可由语料难度、标签先验和语言构成造成。"
               "p 值为未校正的探索性描述，不能据此选择部署时长阈值；Gemma 4 只有一个点，不计算跨数据集相关性。" if zh else
               "These Spearman correlations treat each corpus as one point, only sixteen points, asking whether corpora with "
               "longer typical clips are easier. That differs from the clip-level within-dataset/gold-class question. "
               "Corpus difficulty, label priors and language composition can generate such correlations. The unadjusted p-values "
               "are exploratory and do not establish a deployment duration threshold. Gemma 4 has one point and receives no cross-dataset correlation.")

    out.append("## 6. 模型差异的可检验解释与当前证据边界" if zh else
               "## 6. Testable explanations for model differences and the limits of current evidence")
    diagnostic_rows = []
    for model in order:
        rows = [row for row in models if row["model"] == model]
        invalids = [_number(row.get("invalid_rate")) for row in rows]
        invalids = [value for value in invalids if value is not None]
        differences = [row["valid_only_accuracy"] - row["accuracy"] for row in rows
                       if _number(row.get("valid_only_accuracy")) is not None]
        diagnostic_rows.append([model_name(model), len(rows), _percent(_mean(invalids)),
                                _percent(max(invalids) if invalids else None),
                                _decimal(100 * _mean(differences), signed=True) if differences else "—"])
    out.append(_table((["模型", "数据集数", "平均 invalid", "最高 invalid", "仅有效输出 Accuracy − 总 Accuracy (pp)"] if zh else
                       ["Model", "Datasets", "Mean invalid", "Maximum invalid", "Valid-only Accuracy − overall (pp)"]),
                      diagnostic_rows))
    out.append("上表为各模型已完成数据集的等权均值，Gemma 4 的一套语料均值不能与其余模型的十六套语料均值公平排名。"
               "invalid 仍计入主指标的错误；valid-only Accuracy 只描述有效回答子集，可能选择性保留较容易样本，"
               "不是修复格式后的预期 Accuracy。SER 从候选类分数中选最高者，Audio LLM 在自由生成后按协议解析单词，"
               "所以格式/候选集遵从也会影响后者。需要把这类协议损失与情绪辨别错误分开验证。" if zh else
               "These are equal-dataset averages over each model's completed runs. Gemma 4's one-corpus mean cannot fairly "
               "rank against sixteen-corpus means. Invalid outputs remain errors in the primary metrics. Valid-only Accuracy "
               "conditions on a potentially easier selected subset; it is not the expected Accuracy after fixing formatting. "
               "SER selects the highest score among candidates, while Audio LLMs generate text followed by protocol-specific "
               "one-word parsing. Candidate/format compliance can therefore affect the latter and must be distinguished from emotion discrimination.")
    # Prediction-prior diagnostics are compared with the same dataset's neutral baseline.
    bias_rows = []
    for key in present_conversations:
        for model in (OMNI, QWEN_AUDIO):
            row = mr.get((key, model))
            if row:
                bias_rows.append([dataset_name(key), model_name(model), _percent(ds[key].get("neutral_accuracy")),
                                  _percent(row.get("neutral_pred_rate")), _percent(row.get("angry_pred_rate")),
                                  _percent(row.get("invalid_rate"))])
    out.append(_table((["数据集", "模型", "neutral 基线", "预测 neutral", "预测 angry", "invalid"] if zh else
                       ["Dataset", "Model", "Neutral baseline", "Predict neutral", "Predict angry", "Invalid"]), bias_rows))
    out.append("预测频率显示模型输出的类别先验是否与语料相匹配；频繁预测 neutral 本身不等于偏差，必须同时看真实分布、"
               "逐类 recall 与混淆情况。DailyTalk/MultiDialog 的常量基线说明较高 Accuracy 可来自多数类；"
               "反过来，Audio LLM 在某些短句语料上偏向单一情绪或产生较多 invalid，也可拉低整体分数。"
               "目前这些是描述性诊断，不能单凭频率确定其错误机制。" if zh else
               "Prediction frequencies reveal alignment between an output prior and corpus labels. Frequent neutral predictions "
               "are not inherently biased; gold prevalence, class recall and confusions must also be inspected. The DailyTalk/MultiDialog "
               "constant baselines show how a high Accuracy can arise from a majority class. Conversely, a single-emotion prior "
               "or invalid outputs can reduce scores on some short-utterance corpora. Frequencies alone do not identify the mechanism.")
    listener_rows = [row for row in models if _number(row.get("listener_accuracy")) is not None]
    if listener_rows:
        out.append("对于另有 listener 标签的语料，下面补充模型与听者标注的一致性；它与表演意图标签的评分是两个参考目标，"
                   "不能把差异全部归因于模型能力。两者标注覆盖的样本数量也须一并保留。"
                   "这里只使用单一、非空的 listener vote；并列票不参与这项补充评分。" if zh else
                   "Where listener labels are available, the following adds agreement with perceived emotion. Intended-delivery "
                   "and listener labels are different reference targets; their score difference cannot all be attributed to model ability. "
                   "The listener-labeled sample count is retained. Only a single nonempty listener vote is scored here; tied votes are excluded.")
        out.append(_table((["数据集", "模型", "Listener N", "Listener Accuracy", "Listener UAR"] if zh else
                           ["Dataset", "Model", "Listener N", "Listener Accuracy", "Listener UAR"]),
                          [[dataset_name(row["dataset"]), model_name(row["model"]), _integer(row.get("listener_n")),
                            _percent(row.get("listener_accuracy")), _percent(row.get("listener_uar"))]
                           for row in listener_rows]))
    out.append("**声学信息、词义与标签来源。** 对话中的词义有时能帮助判断情绪，固定文本的表演语料则更依赖韵律和音色。"
               "但“Audio LLM 必然依赖文本、SER 必然只用声学”过于简单。VoxEmo 的多语料实验发现，提示方式可大幅改变 "
               "Qwen2-Audio 的表现，词义信息的收益也随表演/自然语料而异；另一项控制词汇不变、改变情绪表达的研究说明，"
               "模型内部可包含情绪信息，却未必在回答中稳定利用。两者支持进一步测试声学/语义读出，而没有证明本项目差异的成因。"
               "[VoxEmo](https://arxiv.org/html/2603.08936v1)；"
               "[Encoded but Not Express](https://aclanthology.org/2026.aacl-long.262/)。" if zh else
               "**Acoustics, lexical meaning and label origin.** Words may help emotion judgments in conversation, while acted "
               "fixed-text speech places more weight on delivery. A strict Audio-LLM=lexical / SER=acoustic division is too simple. "
               "VoxEmo finds substantial Qwen2-Audio prompt sensitivity and corpus-dependent benefits from lexical information. "
               "A controlled study changing emotional delivery while preserving words shows that emotional information can be "
               "represented without being reliably used in answers. These motivate acoustic/semantic readout tests rather than "
               "establishing the cause of our rankings. [VoxEmo](https://arxiv.org/html/2603.08936v1); "
               "[Encoded but Not Express](https://aclanthology.org/2026.aacl-long.262/).")
    out.append("**Qwen2-Audio 与 Omni 不只差在音频长度。** 两者都使用 Whisper-large-v3 初始化音频编码器，"
               "但语言模型、预训练和后训练不同：Qwen2-Audio 使用 Qwen-7B，Omni 使用 Qwen2.5 并进行多模态训练。"
               "原论文 MELD 指标分别为 55.3% 与 57.0%，不能直接与本项目 Instruct checkpoint、提示、解析与候选标签协议下的结果对齐。"
               "这也提醒我们，MELD 出现在论文评测中既不是本地复现证明，也不能凭分数推断训练泄漏。"
               "[Qwen2-Audio 技术报告](https://arxiv.org/html/2407.10759v1)；"
               "[Qwen2.5-Omni 技术报告](https://arxiv.org/html/2503.20215v1)。" if zh else
               "**Qwen2-Audio and Omni differ in more than duration.** Both initialize their audio encoders from Whisper-large-v3, "
               "but their language backbones and training differ: Qwen2-Audio uses Qwen-7B, while Omni uses Qwen2.5 and multimodal "
               "training. Their technical reports list MELD Accuracy of 55.3% and 57.0%, respectively. These are not directly "
               "matched to our Instruct checkpoint, prompt, parser and candidate-label protocol. A published MELD evaluation is "
               "neither proof of a local reproduction nor evidence of training leakage inferred from a score. "
               "[Qwen2-Audio report](https://arxiv.org/html/2407.10759v1); "
               "[Qwen2.5-Omni report](https://arxiv.org/html/2503.20215v1).")
    tess, mead = mr.get(("tess", QWEN_AUDIO), {}), mr.get(("mead", QWEN_AUDIO), {})
    out.append((f"具体到 TESS（Qwen2-Audio Accuracy {_percent(tess.get('accuracy'))}）与 MEAD"
                f"（{_percent(mead.get('accuracy'))}）的差异，训练数据接触或泄漏只是尚未证实的假设，"
                "不能从高分推断前者被训练、从低分推断后者未被训练。Qwen2-Audio 作者声明排除其评测数据，"
                "这是有范围的作者声明，并非对本项目十六套语料及全部继承组件的独立逐文件审计。"
                "语音材料、说话人、表达方式、标签与解析协议也可能影响这一差异，当前数据不能区分这些解释。") if zh else
               (f"For the TESS (Qwen2-Audio Accuracy {_percent(tess.get('accuracy'))}) versus MEAD "
                f"({_percent(mead.get('accuracy'))}) contrast, training exposure or leakage is an unverified hypothesis. "
                "High and low scores do not establish which corpus entered training. Qwen2-Audio's authors state that their "
                "evaluation data were excluded; this scoped statement is not an independent file-level audit of all sixteen "
                "corpora and inherited components. Speech material, speakers, delivery, labels and parsing are alternative "
                "explanations that the current observations do not distinguish."))
    out.append("**非英语并非一个统一失败原因。** Qwen2-Audio 官方列出的语言包括 Italian，因此不能把 EMOVO/Emozionalmente "
               "的低分解释成“模型不支持意大利语”。语言支持也不保证情绪、口音、表演风格或标签映射的跨域泛化；"
               "官方示例没有列出某语言，也不是该语言完全未出现在训练中的证据。"
               "[Qwen2-Audio 官方说明](https://qwenlm.github.io/blog/qwen2-audio/)。" if zh else
               "**Non-English is not one uniform failure mechanism.** Italian appears in Qwen2-Audio's official language examples, "
               "so low EMOVO/Emozionalmente scores cannot simply be explained as unsupported Italian. Language support does not "
               "guarantee transfer across emotion, accent, acting style or label mapping. Absence from an illustrative language "
               "list also does not prove absence from training. [Official Qwen2-Audio description](https://qwenlm.github.io/blog/qwen2-audio/).")
    statuses = Counter(str(row.get("training_status", "unknown")) for row in models)
    status_text = "; ".join(f"{key}: {count}" for key, count in sorted(statuses.items()))
    out.append(("**训练来源不能忽略。** 领先 SER 与若干表演语料存在已记录的训练、开发或上游教师关系，"
                "包括部分 CREMA-D、RAVDESS、SAVEE、JL-Corpus 等；每个模型的关系与阶段不同。"
                "这使跨模型排名同时混合迁移能力和已接触领域的适配。当前 run 的来源状态计数为 " + status_text + "。"
                "这些状态是来源审查，不是根据分数推测；没有记录不等于确认未见，参与训练流程也不等于本项目具体测试文件泄漏。"
                "详见[项目训练来源审查](../../emotion_recognition/docs/training_provenance.md)。") if zh else
               ("**Training provenance matters.** Leading SER systems have documented training, development or upstream-teacher "
                "links to several acted corpora, including some of CREMA-D, RAVDESS, SAVEE and JL-Corpus; stages differ by model. "
                "Rankings can therefore mix transfer with adaptation to previously encountered domains. Current run-level provenance "
                "status counts are " + status_text + ". These come from source review, not score-based inference. "
                "No documented link does not certify unseen data, and participation in a training pipeline does not establish "
                "leakage of these specific test clips. See the [project provenance audit](../../emotion_recognition/docs/training_provenance.md)."))

    out.append("原生强度标签、控制情绪类别/时长后的正确率，以及 MEAD 的逐类错误另见[强度分析](intensity_zh.md)。" if zh else
               "See the [intensity analysis](intensity_en.md) for native strength labels, class/duration-adjusted accuracy and class-specific MEAD errors.")
    out.append("## 7. 下一步：能区分解释的小规模控制实验" if zh else
               "## 7. Next steps: small controlled experiments that distinguish explanations")
    out.append("以下是建议实验，尚未执行，不属于本报告的新模型结果。应预先固定样本、评价规则与比较方式，并保留每类样本，"
               "优先覆盖四个对话语料与 Qwen2-Audio 表现差异最大的几个语料。" if zh else
               "The following are proposed experiments, not executed runs or new model findings in this report. Prespecify clips, "
               "scoring and comparisons; preserve class coverage and prioritize the conversational corpora plus domains with "
               "the largest Qwen2-Audio performance differences.")
    proposals = ([
        "**同一音频的 crop / pad 对照。** 对配对样本比较原音频、保留情绪线索的 1/2/3/5/10 s 裁切和仅补静音版本；固定 loudness、采样率与提示。裁切会同时改变词义和韵律，补静音不能代表增加语义信息，应分别解释，并显式记录各模型真正使用的窗口。",
        "**自由生成与候选标签约束。** 同一批样本比较现有解析协议与候选标签 logit/受约束解码，分别记录 invalid、逐类混淆与总体分数。对不提供这种能力的 API 单独报告，避免把不等价接口当成同一干预。",
        "**词义与类别先验对照。** 采用有声学感知标注的样本，配对比较音频、仅转写文本和受控声学/文本线索，报告 neutral 常量基线及每类 recall；核查 intended emotion 与 listener emotion 的差异。语音合成或变声会引入新混杂，不能直接当作自然语音的因果证据。",
        "**真正留出的部署验证。** 在来源可审查的新领域、独立说话人和目标内部数据上，先用一部分人工标签校准每个 consensus 规则，再在独立部分验证 Accuracy/UAR、接受率和逐类错误。把自动标注可接受的误差与人工抽检量预先定好，而不是根据这十六套基准挑选最有利规则。",
    ] if zh else [
        "**Paired crop/pad controls.** Compare original clips, 1/2/3/5/10 s crops retaining emotion cues, and silence-only padding, holding loudness, sample rate and prompt fixed. Cropping changes lexical and prosodic information together; padding adds no semantic evidence. Interpret them separately and log effective model windows.",
        "**Free generation versus candidate constraints.** On identical clips, compare current parsing with candidate-label logits or constrained decoding; report invalids, class confusions and overall metrics separately. APIs lacking this capability require a separate comparison rather than an assumed equivalent intervention.",
        "**Lexical and class-prior controls.** Use perceptually labeled clips to compare audio, transcript-only input and controlled acoustic/text cues. Include neutral baselines, per-class recall and intended-versus-listener labels. Synthesis or voice transformation adds confounds and is not automatically causal evidence for natural speech.",
        "**A genuinely held-out deployment check.** In provenance-reviewed new domains, independent speakers and the internal target data, calibrate prespecified consensus rules on one manually labeled subset, then validate Accuracy/UAR, acceptance rate and class errors independently. Set tolerable auto-label error and manual-audit effort in advance rather than selecting favorable rules on these sixteen benchmarks.",
    ])
    out.append("\n\n".join(f"{i}. {text}" for i, text in enumerate(proposals, 1)))
    out.append("当前结果可以形成论文的跨语料稳健性与错误分析，并帮助设计内部标注策略；"
               "是否适合自动接受标签，最终取决于目标域独立验证的错误率与类别覆盖，而不是单个基准上的高分或时长相关性。" if zh else
               "These results support a paper's cross-corpus robustness and error analysis and guide the internal labeling study. "
               "Automatic acceptance ultimately depends on independently validated error and class coverage in the target domain, "
               "rather than a high benchmark score or duration correlation alone.")
    return "\n\n".join(out) + "\n"

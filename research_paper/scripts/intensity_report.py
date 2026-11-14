"""Bilingual interpretation of native intensity strata; rendering only, no I/O."""
from __future__ import annotations

import consensus


def _number(value, digits=2, signed=False):
    if value is None:
        return "—"
    return format(float(value), f"{ '+' if signed else ''}.{digits}f")


def _percent(value):
    return "—" if value is None else f"{100 * float(value):.2f}%"


def _effect(row):
    if not row or row.get("slope_pp") is None:
        return "—"
    point = _number(row["slope_pp"], signed=True)
    if (row.get("nclusters", 0) >= 20 and row.get("ci_lower_pp") is not None
            and row.get("ci_upper_pp") is not None):
        return point + f" [{_number(row['ci_lower_pp'], signed=True)}, {_number(row['ci_upper_pp'], signed=True)}]"
    return point + " [—]"


def _paired(row):
    if row.get("difference_pp") is None:
        return "—"
    point = _number(row["difference_pp"], signed=True)
    if (row.get("nclusters", 0) >= 20 and row.get("ci_lower_pp") is not None
            and row.get("ci_upper_pp") is not None):
        return point + f" [{_number(row['ci_lower_pp'], signed=True)}, {_number(row['ci_upper_pp'], signed=True)}]"
    return point + " [—]"


def _mead_class_diagnostics(data, zh):
    """Describe saved Qwen2-Audio errors without attributing their cause."""
    model = "qwen2_audio_7b"
    rows = [row for row in data.get("class_rows", [])
            if row["dataset"] == "mead" and row["model"] == model]
    if not rows:
        return []
    levels = {}
    for row in rows:
        levels.setdefault(str(row["level_id"]), row)
    ids = sorted(levels, key=lambda key: (float(levels[key]["level"]), key))
    classes = sorted({row["gold"] for row in rows})
    row_index = {(row["gold"], str(row["level_id"])): row for row in rows}
    totals = {key: sum(int(row["n"]) for row in rows if str(row["level_id"]) == key) for key in ids}
    lines = ["", "### MEAD：Qwen2-Audio 的类别混淆" if zh else
             "### MEAD: class-specific Qwen2-Audio errors", "",
             "下面的逐类 Recall 是该参考类别中预测正确的比例，不是类别置信度。括号列出正确数/样本数。" if zh else
             "Class Recall below is the proportion correctly classified within each reference class, not class "
             "confidence. Parentheses show correct clips / class clips.", "",
             "| " + " | ".join((["参考情绪"] if zh else ["Reference emotion"]) +
                                     [f"{levels[key]['level_label']} (n={totals[key]:,})" for key in ids]) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(ids)) + "|"]
    for gold in classes:
        cells = []
        for key in ids:
            row = row_index.get((gold, key))
            cells.append(f"{_percent(row['accuracy'])} ({int(row['correct'])}/{int(row['n'])})" if row else "—")
        lines.append("| " + " | ".join([gold, *cells]) + " |")
    predictions = {str(row["level_id"]): row for row in data["level_rows"]
                   if row["dataset"] == "mead" and row["model"] == model}
    concentration = []
    for key in ids:
        row = predictions.get(key)
        if (row and row.get("predicted_angry_rate") is not None
                and row.get("predicted_disgusted_rate") is not None):
            concentration.append(f"{levels[key]['level_label']}: angry {_percent(row['predicted_angry_rate'])}, "
                                 f"angry + disgusted {_percent(row['predicted_angry_rate'] + row['predicted_disgusted_rate'])}")
    if concentration:
        lines += ["", ("预测分布随等级变化：" if zh else "Predicted-label distribution by level: ")
                  + "; ".join(concentration) + "."]
    high_key = ids[-1]
    high_prediction = predictions.get(high_key)
    angry = row_index.get(("angry", high_key))
    n = totals[high_key]
    if angry and n and high_prediction and high_prediction.get("predicted_angry_rate") is not None:
        angry_n = int(angry["n"])
        non_angry_n = n - angry_n
        non_angry_correct = sum(int(row["correct"]) for row in rows
                               if str(row["level_id"]) == high_key and row["gold"] != "angry")
        invalid = sum(int(row["invalid"]) for row in rows if str(row["level_id"]) == high_key)
        # Rates are exact proportions of this level's retained prediction cohort.
        wrong_angry = round(float(high_prediction["predicted_angry_rate"]) * n) - int(angry["correct"])
        if non_angry_n:
            lines += ["", (f"最高等级真实 angry 只有 {angry_n}/{n}（{_percent(angry_n / n)}），"
                           f"但 {_percent(high_prediction['predicted_angry_rate'])} 的输出是 angry。"
                           f"其余 {non_angry_n} 条非 angry 音频中，{wrong_angry} 条（{_percent(wrong_angry / non_angry_n)}）"
                           f"被判为 angry，正确数仅 {non_angry_correct}/{non_angry_n}（{_percent(non_angry_correct / non_angry_n)}）。"
                           f"该等级 invalid 为 {invalid}/{n}（{_percent(invalid / n)}），因此格式无效不能解释主要错误。"
                           "更强表达下 angry 的识别改善，并不代表其他情绪同步改善。这里观察到的是输出类别集中；"
                           "这些预测不能确定训练数据来源、泄漏或其他因果机制。" if zh else
                           f"At the highest level, true angry clips account for only {angry_n}/{n} "
                           f"({_percent(angry_n / n)}), while {_percent(high_prediction['predicted_angry_rate'])} of outputs "
                           f"are angry. Among the other {non_angry_n} clips, {wrong_angry} "
                           f"({_percent(wrong_angry / non_angry_n)}) are labelled angry, with only "
                           f"{non_angry_correct}/{non_angry_n} ({_percent(non_angry_correct / non_angry_n)}) classified "
                           f"correctly. Invalid answers account for {invalid}/{n} ({_percent(invalid / n)}), so answer "
                           "format does not explain most errors. Improved angry recognition at higher strength does not "
                           "mean other emotions improve together. This is concentration of predicted categories; the "
                           "predictions cannot identify training-data sources, leakage or another causal mechanism.")]
    return lines


def render_report(data, language):
    """Return Markdown for the data contract supplied by intensity_analysis.py."""
    zh = language == "zh"
    manifest = data["manifest"]
    scope = data["scope_rows"]
    levels = data["level_rows"]
    effects = data["effect_rows"]
    matched = data.get("matched_rows", [])
    model_names = manifest.get("model_names", {})
    model_keys = {row["model"] for row in levels} | {row["model"] for row in effects}
    models = [model for model in manifest.get("model_order", []) if model in model_keys]
    models += sorted(model_keys - set(models))
    def model_name(model):
        return model_names.get(model, consensus.SHORT_NAMES.get(model, model))
    def dataset_name(dataset):
        return consensus.NAMES.get(dataset, dataset)
    effect_index = {(row["dataset"], row["model"], row["fit"]): row for row in effects}
    total = sum(int(row["n_total"]) for row in scope)
    known = sum(int(row["n_known"]) for row in scope)
    eligible = sum(int(row["n_non_neutral_known"]) for row in scope)
    selected_totals = {}
    for row in manifest.get("selection", []):
        if (isinstance(row, dict) and row.get("status") == "selected"
                and row.get("clips") is not None):
            selected_totals[row["dataset"]] = max(
                selected_totals.get(row["dataset"], 0), int(row["clips"]))
    other_datasets = set(selected_totals) - {row["dataset"] for row in scope}
    timestamp = manifest.get("generated_at_singapore", "")
    if zh:
        lines = ["# 情绪强度：更强烈的表达是否更容易分类？", "",
                 f"更新时间：{timestamp}。使用现有 {manifest.get('selected_run_count', '—')} 组模型结果；没有重新推理。", "",
                 "本报告区分两个问题。第一，带有更高原生强度标签的音频，现有模型是否更容易预测对情绪类别？"
                 "第二，模型能否准确预测情绪强度本身？这次可以用保存的类别预测分析第一个问题，"
                 "不能把分类正确率的上升当成第二个问题已经解决。", "",
                 "强度不是类别概率、音量或时长的另一种写法。情绪分类器给 angry 的概率高，表示它偏向这个类别，"
                 "并不表示表达得更强烈；强烈的悲伤也未必有高激活度。本次只读取数据集明确提供的强度字段，"
                 "没有从模型置信度、声学能量或标注一致率补造强度标签。", "",
                 "## 原生标注覆盖范围", "",
                 f"下表只覆盖 {len(scope)} 套可分析强度的语料，共 {total:,} 条音频；其中 {known:,} 条具有可读取的原生强度标注。主分析使用其中 "
                 f"{eligible:,} 条非 neutral 样本。未知强度不填成‘低’，neutral 的原始等级保留在 inventory，"
                 "但不混入主分析。原模型的 invalid 回答仍计为分类错误，不因模型回答格式而移除样本。", "",
                 "| 数据集 | 总样本 | 强度已知 | 主分析：非 neutral | 原生尺度 / 类型 | 原生等级 |",
                 "|---|---:|---:|---:|---|---|"]
    else:
        lines = ["# Emotion intensity: are stronger expressions easier to classify?", "",
                 f"Updated: {timestamp}. Saved results from {manifest.get('selected_run_count', '—')} model–dataset runs; no inference rerun.", "",
                 "There are two distinct questions. First, are existing emotion-category predictions more often correct "
                 "when a clip has a higher native intensity annotation? Second, can a model accurately predict intensity "
                 "itself? Saved category predictions support the first analysis here. Better category accuracy at higher "
                 "strength does not demonstrate successful intensity estimation.", "",
                 "Intensity is not another name for class probability, loudness or clip duration. A high angry-class "
                 "probability means the classifier prefers angry; it does not measure how strongly anger is expressed. "
                 "Strong sadness need not be highly activated. We use explicit dataset annotations rather than inventing "
                 "strength from classifier confidence, acoustic energy or rater agreement.", "",
                 "## Native annotation coverage", "",
                 f"The table covers only the {len(scope)} intensity-analysable corpora, containing {total:,} clips; "
                 f"{known:,} have readable native intensity annotations. The primary analysis uses "
                 f"the {eligible:,} non-neutral clips among them. Missing intensity is not filled as low. Native neutral "
                 "levels remain in the inventory but are not mixed into the primary emotional-strength comparison. "
                 "Invalid model answers remain classification errors; answer-format failures do not remove examples.", "",
                 "| Dataset | All clips | Intensity known | Non-neutral analysis | Native scale / kind | Native levels |",
                 "|---|---:|---:|---:|---|---|"]
    for row in scope:
        lines.append(f"| {dataset_name(row['dataset'])} | {int(row['n_total']):,} | {int(row['n_known']):,} | "
                     f"{int(row['n_non_neutral_known']):,} | {row['scale_name']} / {row['scale_kind']} | {row['levels']} |")
    if other_datasets:
        lines += ["", (f"完整 benchmark 是 {len(selected_totals)} 套语料、{sum(selected_totals.values()):,} 条音频。"
                       f"其余 {len(other_datasets)} 套没有找到可用的原生情绪强度标注，因此不进入本强度分析；"
                       "这不表示那些语料缺少类别预测，也不表示模型得分为零。" if zh else
                       f"The full benchmark contains {len(selected_totals)} corpora and {sum(selected_totals.values()):,} clips. "
                       f"The other {len(other_datasets)} corpora have no available native emotion-intensity annotation "
                       "and therefore do not enter this intensity analysis. Their category predictions still exist; "
                       "they are not zero-scoring corpora.")]
    missing = [model_name(key) for key in manifest.get("model_order", []) if key not in model_keys]
    if missing:
        lines += ["", ("没有当前可用强度分层结果的模型：" if zh else "Models without current intensity-stratum results: ")
                  + ", ".join(missing) + ("。这表示缺少匹配预测或原生强度标注，不表示模型得分为零。" if zh else
                    ". This indicates missing matching predictions or intensity annotations, not a score of zero.")]
    if zh:
        lines += ["", "六种尺度必须分别解释。CREMA-D 的 LO/MD/HI 只覆盖 IEO 句子；XX 是未指定。"
                  "RAVDESS 是 normal/strong，native neutral 只有 normal；benchmark 还把 calm 合并为 neutral。"
                  "MEAD 的情绪有三级，neutral 只有 level_1 参考。THAI-SER 的 Low/High 只对 scripted 部分存在，"
                  "不能给 improvised 音频指定同样的等级。", "",
                  "MCAE-SPPS 使用 40 位听众的平均强度评分，原尺度为 1–9，分析保留原始连续数值；"
                  "它不是元数据里单位为 dB 的 meanIntensity。图中的分位数组仅用于展示，连续效应不使用人为制造的"
                  "低/中/高标签。DailyTalk 使用 ECSS 的 1–3 级附加标注，是观察到的评分而非本实验随机分配的表演条件。"
                  "作者论文将其描述为 weak/medium/strong，但它不创造新的独立音频情绪类别 gold。"
                  "参见 [ECSS 原论文](https://arxiv.org/html/2312.11947v1) 和 [作者数据格式](https://github.com/walker-hyf/ECSS)。", "",
                  "## 怎样控制类别与时长", "",
                  "每个模型、每个数据集分别拟合，结果不跨原生尺度合并。先控制参考情绪类别，使用类别平衡权重，"
                  "再在第二个拟合中加入音频时长。表里的系数用百分点表示，具体原生单位写在各数据集标题下。"
                  "这种调整帮助识别类别构成和时长带来的混杂；它不会把观察性数据变成随机实验。"
                  "等级 1→2 与 2→3 的数值编码也不证明心理强度恰好等距。", "",
                  "区间按数据中的 speaker/dialogue 聚类方案计算；少于 20 个聚类时不展示置信区间，记为 [—]，"
                  "即使点估计仍可计算。少数聚类、稀少类别、很少的高强度样本，都会限制稳定性。"
                  "对话聚类并不保证说话人独立，多个模型与多个尺度的探索也没有变成多重比较校正后的确认性检验。", "",
                  "## 各模型在原生强度层的表现", "",
                  "等级列显示分类 Accuracy；括号中的 n 是该层音频数。效应列分别为类别平衡、以及再调整时长后的"
                  "百分点变化。它们描述正确分类与强度的关系，不是强度预测准确率。"]
    else:
        lines += ["", "The six scales require separate interpretations. CREMA-D LO/MD/HI covers only the IEO sentence; "
                  "XX is unspecified. RAVDESS uses normal/strong, with native neutral only normal; this benchmark also "
                  "merges calm into neutral. MEAD has three emotional levels but only a level_1 neutral reference. "
                  "THAI-SER Low/High exists only for scripted speech and must not be assigned to improvised clips.", "",
                  "MCAE-SPPS supplies the mean of 40 listeners' strength ratings on 1–9; the continuous analysis retains "
                  "the original values. This is not its meanIntensity acoustic-dB field. Display quantiles do not become "
                  "invented native low/medium/high labels in the continuous fit. DailyTalk uses additional public ECSS "
                  "1–3 annotations, an observed rating rather than an acting condition randomized in our experiment. "
                  "The authors describe weak/medium/strong levels, but these do not create an independent audio-only "
                  "category-gold set. See the [ECSS paper](https://arxiv.org/html/2312.11947v1) and "
                  "[author-supplied format](https://github.com/walker-hyf/ECSS).", "",
                  "## Controlling emotion class and duration", "",
                  "Fit each model and dataset separately, without pooling native scales. The first fit controls the "
                  "reference emotion class and balances class weights; the second additionally controls clip duration. "
                  "Effects are percentage points per native unit, specified below each dataset heading. This adjustment "
                  "helps separate class composition and duration from the observed strength association; it does not "
                  "turn observational data into a randomized experiment. Numeric ordinal coding also does not establish "
                  "equal psychological distances from level 1 to 2 and 2 to 3.", "",
                  "Intervals follow the recorded speaker/dialogue clustering scheme. With fewer than 20 clusters, "
                  "intervals are withheld as [—], even when a point estimate can be calculated. Few clusters, rare "
                  "classes and sparse high-intensity cases limit stability. Dialogue clustering does not guarantee "
                  "independent speakers. Exploratory comparisons across models and scales are not multiplicity-adjusted "
                  "confirmatory tests.", "",
                  "## Model results at native intensity strata", "",
                  "Level columns report category Accuracy; n is the number of clips in that stratum. The two effect "
                  "columns give class-balanced and additionally duration-adjusted percentage-point changes. They measure "
                  "association with correct classification, not intensity-prediction accuracy."]
    focus_models = [model for model in ("qwen2_audio_7b", "qwen2_5_omni_7b") if model in models]
    if focus_models:
        lines += ["", "### Qwen2-Audio 与 Qwen2.5-Omni 的跨语料观察" if zh else
                  "### Qwen2-Audio and Qwen2.5-Omni across corpora", "",
                  "下列数值来自类别平衡并调整时长后的拟合，单位随原生尺度改变，不能跨数据集直接排序。"
                  "这两个模型的表现并不遵循一个普遍的‘强度越高越准确’规则：" if zh else
                  "The following estimates balance emotion classes and adjust for duration. Their native units differ "
                  "across corpora, so the magnitudes cannot be ranked across datasets. These models do not follow a "
                  "universal rule that stronger emotion always improves classification:", ""]
        for coverage in scope:
            dataset = coverage["dataset"]
            values = [effect_index.get((dataset, model, "class_balanced_duration_adjusted")) for model in focus_models]
            values = [row for row in values if row and row.get("slope_pp") is not None]
            if not values:
                continue
            estimates = ", ".join(f"{model_name(row['model'])} {_number(row['slope_pp'], signed=True)} pp" for row in values)
            unit = values[0]["unit"]
            lines.append(f"- {dataset_name(dataset)} ({unit}): {estimates}.")
        mead_levels = [row for row in levels if row["dataset"] == "mead" and row["model"] in focus_models]
        if mead_levels:
            ranges = []
            for model in focus_models:
                ordered = sorted((row for row in mead_levels if row["model"] == model), key=lambda row: float(row["level"]))
                if ordered:
                    ranges.append(f"{model_name(model)} {_percent(ordered[0]['accuracy'])} → {_percent(ordered[-1]['accuracy'])}")
            lines += ["", ("MEAD 虽呈正向强度关联，最低至最高等级的绝对分类正确率仍只有 " if zh else
                           "Despite positive intensity associations in MEAD, absolute category accuracy from the lowest "
                           "to highest level is only ") + "; ".join(ranges) +
                      ("。因此，强度敏感性不能解释或消除模型在 MEAD 的整体低正确率。" if zh else
                       ". Intensity sensitivity therefore does not explain away or eliminate poor overall MEAD accuracy.")]
        if any(row["dataset"] == "mcae_spps" for row in effects):
            lines += ["", "MCAE-SPPS 的正向点估计只有 6 位说话人的支持，因此不展示聚类置信区间；"
                      "连续听众评分的关联也可能反映情绪表达是否清楚可辨，不能当成强度的因果作用。" if zh else
                      "MCAE-SPPS positive point estimates are supported by only six speakers, so cluster intervals are "
                      "withheld. Continuous listener scores may also track how recognisable an expression is; this is "
                      "not evidence for a causal intensity effect."]
    lines += _mead_class_diagnostics(data, zh)
    daily = next((row for row in scope if row["dataset"] == "dailytalk"), None)
    if daily:
        daily_levels = {str(row["level_id"]): row for row in levels if row["dataset"] == "dailytalk"}
        strongest = max(daily_levels.values(), key=lambda row: float(row["level"])) if daily_levels else None
        daily_clusters = max((int(row["nclusters"]) for row in effects if row["dataset"] == "dailytalk"), default=0)
        strongest_count = strongest["n"] if strongest else "—"
        lines += ["", (f"DailyTalk 的主分析只有 {int(daily['n_non_neutral_known']):,} 条，最高等级只有 {strongest_count} 条；"
                       "当前子集中 fearful 仅 1 条、disgusted 仅 3 条。类别平衡把这些极小类别赋予与大类别相同的总权重，"
                       "因此很大的斜率可能由少数样本强烈影响，不能稳定外推。"
                       f"{daily_clusters} 个 dialogue 聚类仍只来自两位演员，按 dialogue 计算的区间不能支持新说话人的泛化结论。" if zh else
                       f"DailyTalk contributes only {int(daily['n_non_neutral_known']):,} primary clips, with {strongest_count} at its highest level. "
                       "The current subset has just one fearful and three disgusted clips. Class balancing gives these tiny "
                       "classes the same total weight as large classes, so a few observations can strongly influence very "
                       "large slopes; these are not stable extrapolations. "
                       f"Its {daily_clusters} dialogue clusters still come from only two actors. Dialogue-cluster intervals "
                       "do not establish generalisation to new speakers.")]
    level_datasets = {row["dataset"] for row in levels}
    for coverage in scope:
        dataset = coverage["dataset"]
        if dataset not in level_datasets:
            continue
        dataset_levels = [row for row in levels if row["dataset"] == dataset]
        level_map = {}
        for row in dataset_levels:
            level_map.setdefault(str(row["level_id"]), row)
        ids = sorted(level_map, key=lambda key: (float(level_map[key]["level"]), key))
        header = []
        for level_id in ids:
            first = level_map[level_id]
            counts = {int(row["n"]) for row in dataset_levels if str(row["level_id"]) == level_id}
            label = first["level_label"]
            header.append(f"{label} (n={next(iter(counts)):,})" if len(counts) == 1 else str(label))
        unit = next((row["unit"] for row in effects if row["dataset"] == dataset), "native unit")
        lines += ["", f"### {dataset_name(dataset)}", "",
                  ("原生效应单位：" if zh else "Native effect unit: ") + unit + ("。" if zh else "."), "",
                  "| " + " | ".join((["模型"] if zh else ["Model"]) + header
                           + (["类别平衡效应 [区间]", "再调整时长 [区间]", "聚类数"] if zh else
                              ["Class-balanced effect [interval]", "+ duration [interval]", "Clusters"])) + " |",
                  "|" + "|".join(["---"] + ["---:"] * (len(header) + 3)) + "|"]
        row_map = {(row["model"], str(row["level_id"])): row for row in dataset_levels}
        for model in models:
            if not any(key[0] == model for key in row_map):
                continue
            cells = []
            for level_id in ids:
                row = row_map.get((model, level_id))
                cells.append(_percent(row["accuracy"]) if row else "—")
            a = effect_index.get((dataset, model, "class_balanced"))
            b = effect_index.get((dataset, model, "class_balanced_duration_adjusted"))
            clusters = b.get("nclusters", "—") if b else a.get("nclusters", "—") if a else "—"
            lines.append("| " + " | ".join([model_name(model), *cells, _effect(a), _effect(b), str(clusters)]) + " |")
    lines += ["", "![Emotion-category accuracy by native intensity](figures/emotion_intensity_accuracy.png)", "",
              "![Class- and duration-adjusted intensity associations](figures/emotion_intensity_associations.png)", ""]
    if zh:
        lines += ["## 同键高低强度对照", "",
                  "另检查同一内容键内较高与较低强度的分类正确率差。CREMA-D 的键固定演员、IEO 句子与情绪；"
                  "RAVDESS 固定演员、原生情绪、句子与重复序号；THAI-SER 使用同一 session、actor、明确的"
                  "script_sent 文本、trial 与 assigned emotion。MEAD 的键仅由演员、情绪和文件序号构成，"
                  "本地尚未逐条验证跨 level 的词句完全一致，因此只能称同序号配对，不能当作严格词句控制。", "",
                  "只有保留下来的完整配对参与这个比较。MEAD 的 8,000 条分层子集可能在各强度留下不同句子序号，"
                  "所以完整配对会远少于其强度已知样本数；不能为增加配对而补读未评测预测。"
                  "MCAE 的听众均分及 DailyTalk 的观察性评级也没有天然同句多强度录音配对。"]
    else:
        lines += ["## Higher-versus-lower intensity within recorded keys", "",
                  "A supplementary comparison measures the category-correctness difference between higher and lower "
                  "levels within a content key. CREMA-D fixes actor, IEO sentence and emotion. RAVDESS fixes actor, "
                  "native emotion, statement and repetition. THAI-SER fixes session, actor, exact script_sent text, "
                  "trial and assigned emotion. MEAD keys contain actor, emotion and file index only: cross-level "
                  "lexical equivalence has not been independently verified locally, so these are index-matched cases, "
                  "not a verified lexical control.", "",
                  "Only complete retained pairs enter this comparison. The 8,000-clip MEAD stratified subset may retain "
                  "different sentence indices at different levels, leaving far fewer complete pairs than known-intensity "
                  "clips. We do not add unscored predictions to increase pair counts. MCAE listener means and DailyTalk "
                  "observed ratings do not provide natural same-utterance multi-intensity recordings."]
    mead_matches = [row for row in matched if row["dataset"] == "mead"]
    if mead_matches:
        pair_count = max(int(row["n_pairs"]) for row in mead_matches)
        actors = max(int(row["nclusters"]) for row in mead_matches)
        lines += ["", (f"当前 MEAD 实际只有 {pair_count} 个完整配对、来自 {actors} 位演员，而且词句键未逐条验证。"
                       "这些结果只能作为个案描述，不能支持一般的强度效应或新说话人结论；不展示置信区间。" if zh else
                       f"MEAD currently has only {pair_count} complete pairs from {actors} actors, and lexical keys have "
                       "not been verified individually. These are descriptive cases, not support for a general intensity "
                       "effect or a new-speaker conclusion; no interval is shown.")]
    if matched:
        lines += ["", "| 数据集 | 模型 | 完整配对数 | 高 − 低：百分点 [区间] | 聚类数 |" if zh else
                  "| Dataset | Model | Complete pairs | Higher − lower: pp [interval] | Clusters |",
                  "|---|---|---:|---:|---:|"]
        for row in matched:
            lines.append(f"| {dataset_name(row['dataset'])} | {model_name(row['model'])} | "
                         f"{int(row['n_pairs']):,} | {_paired(row)} | {row['nclusters']} |")
        lines += ["", "![Within-key higher-minus-lower intensity differences](figures/emotion_intensity_matched.png)"]
    else:
        lines += ["", "当前没有可报告的完整配对。" if zh else "No complete-pair result is currently reportable."]
    if zh:
        lines += ["", "## MERaLiON 能否直接预测强度？", "",
                  "本地官方模型有两个输出：七类情绪 logits，以及经 sigmoid 映射到 [0,1] 的三维 "
                  "Valence、Arousal、Dominance。维度头的真实权重存在，但没有一个名为通用‘emotion intensity’的独立头。"
                  "Arousal 表示平静到活跃，Dominance 的 weak/strong 描述控制感；两者都不能直接当成上述六套强度量表。"
                  "见 [本地模型说明](../../emotion_recognition/models/meralion_ser_v1/README.md) 和 "
                  "[模型 forward](../../emotion_recognition/models/meralion_ser_v1/modeling_ser_whisper_ecapa.py)。", "",
                  "当前 [项目适配器](../../emotion_recognition/scripts/models/meralion_ser.py) 只取 logits 并保存情绪概率，"
                  "现有 MERaLiON 预测 CSV 没有 valence/arousal/dominance 列。被丢弃的 dims 不能从 p_* 列反推。"
                  "本次没有重新运行模型，因此没有直接的强度预测准确率、VAD 回归误差或维度相关性结果。", "",
                  "后续若评估维度预测，需要另行保存原生 dims，并取得与目标维度、标注条件和尺度匹配的参考值。"
                  "可以在同一情绪类别内探索 arousal 与原生强度等级的秩相关，但这只是代理关系检验，"
                  "不是把 arousal 校准为跨情绪、跨语料通用强度。这些是后续建议，本次未执行。", "",
                  "## 结论边界", "",
                  "同一数据集内，在控制情绪类别和时长后仍为正的效应，支持‘较强表达在当前标注条件下更容易分类’"
                  "这一观察。它不证明增加表达强度会因果地改善模型，不解释全部语言或对话差异，也不证明模型已经理解"
                  "或可量化强度。训练暴露、表演习惯、录音条件、标签可靠性与样本筛选仍可能同时变化。", "",
                  "强度分析应与时长分析一起读：声音更强烈时，音长、停顿和词句表达也可能变化；只比较六套数据集的"
                  "平均时长或平均强度，无法区分这些因素。保留所有模型和尺度的结果比仅挑最支持假设的例子更有解释力。"]
    else:
        lines += ["", "## Can MERaLiON directly estimate intensity?", "",
                  "The local official model has two outputs: seven-category emotion logits and sigmoid-bounded [0,1] "
                  "Valence, Arousal and Dominance. The dimensional-head weights exist, but there is no separate generic "
                  "emotion-intensity head. Arousal runs from calm to active; dominance concerns control, despite its "
                  "weak/strong wording. Neither is interchangeable with the six native strength scales. See the "
                  "[local model card](../../emotion_recognition/models/meralion_ser_v1/README.md) and "
                  "[model forward code](../../emotion_recognition/models/meralion_ser_v1/modeling_ser_whisper_ecapa.py).", "",
                  "The current [project adapter](../../emotion_recognition/scripts/models/meralion_ser.py) selects logits "
                  "and saves categorical probabilities only. Existing MERaLiON prediction CSVs have no valence, arousal "
                  "or dominance columns. Discarded dims cannot be recovered from p_* values. No model was rerun, so this "
                  "report supplies no direct intensity-estimation accuracy, VAD regression error or dimensional correlation.", "",
                  "A future direct evaluation would need newly saved native dims and references matching the intended "
                  "dimension, annotation condition and scale. Within one emotion class, a rank correlation between "
                  "arousal and native strength could test a proxy relationship; it would not calibrate arousal as universal "
                  "emotion intensity across classes or corpora. These are proposals, not experiments performed here.", "",
                  "## Limits of the conclusion", "",
                  "A positive within-dataset association surviving emotion-class and duration adjustment supports an "
                  "observation that stronger expressions are easier to classify under the present annotations. It does "
                  "not establish that increasing intensity causally improves the model, explain every language or "
                  "conversation difference, or demonstrate that the model understands and measures intensity. Training "
                  "exposure, acting style, recording conditions, annotation reliability and sample selection may still vary.", "",
                  "Read this alongside duration: stronger delivery may change speaking length, pauses and expression. "
                  "Comparing corpus-average duration or intensity cannot separate those factors. Reporting all model/scale "
                  "results is more informative than choosing only examples that favour the hypothesis."]
    lines += ["", ("复现与来源：" if zh else "Reproduction and sources: ")
              + "[native inventory](../output/intensity_inventory.csv) · "
              "[inventory/source manifest](../output/intensity_inventory_manifest.json) · "
              "[analysis script](../scripts/intensity_analysis.py)", "",
              ("结合 [跨语料语言、对话与时长分析](cross_dataset_zh.md) 阅读，区分整体数据集差异和语料内强度关联。" if zh else
               "Read alongside the [cross-corpus language, conversation and duration analysis](cross_dataset_en.md) "
               "to distinguish overall dataset differences from within-corpus intensity associations."), ""]
    if manifest.get("notes"):
        lines += ["### " + ("运行记录中的补充说明" if zh else "Additional recorded analysis notes"), ""]
        lines.extend("- " + str(note) for note in manifest["notes"])
        lines.append("")
    return "\n".join(lines)

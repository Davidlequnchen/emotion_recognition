"""
Evaluate emotion2vec+ on the speech emotion datasets and write a Markdown report for each.

Runs the model on every clip of a dataset, compares each prediction with the dataset's labels and writes:
    reports/<dataset>_<model>.md                      accuracy, per-emotion results, confusion matrices
    reports/figures/<dataset>_<model>_confusion*.png  confusion-matrix figures used in the report
    outputs/<dataset>_<model>_predictions.csv          one row per clip (read by scripts/make_viewer.py)
    outputs/<dataset>_<model>_predictions.json         what the CSV contains, and the metrics
    reports/<model>_summary.md                         every evaluated dataset side by side
The datasets are defined in scripts/eval_datasets.py; download them first with scripts/download_data.py.

Usage:
    python scripts/evaluate.py
Settings (which datasets, an optional sample size, device) are in the `evaluation` section of config/config.yaml.
"""
import collections
import csv
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

from eval_datasets import LOADERS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "config.yaml"
# Figure colours: a one-hue blue ramp (light = small share, dark = large share) and neutral inks
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def hardware(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":  # e.g. "Apple M4 Max"
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return platform.processor() or platform.machine()


def ratio(a, b):
    """Element-wise a / b, with 0 where b is 0."""
    return np.divide(a, b, out=np.zeros(len(a)), where=b > 0)


def confusion_table(cm, names, corner):
    """A confusion matrix as Markdown table rows: row percentages, diagonal in bold."""
    rows = [f"| {corner} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, name in enumerate(names):
        share = ratio(cm[k], np.full(len(names), cm[k].sum()))
        rows.append(f"| {name} ({cm[k].sum():,}) | " + " | ".join(
            f"**{100 * s:.1f}%**" if j == k else f"{100 * s:.1f}%" for j, s in enumerate(share)) + " |")
    return rows


def top_confusions(cm, names, n=3):
    """The n largest off-diagonal row shares, as text such as "fearful predicted as sad (14.9%)"."""
    cells = sorted(((cm[r, c] / cm[r].sum(), names[r], names[c]) for r in range(len(names))
                    for c in range(len(names)) if r != c and cm[r].sum()), reverse=True)[:n]
    return ", ".join(f"{a} predicted as {b} ({100 * v:.1f}%)" for v, a, b in cells)


def text_on(rgb):
    """White or ink, whichever has more contrast with the cell colour (WCAG relative luminance)."""
    lum = sum(w * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
              for w, v in zip((0.2126, 0.7152, 0.0722), rgb))
    return "#ffffff" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.0533 else INK


def plot_confusion(cm, names, title, subtitle, ylabel, path):
    """Row-normalised confusion matrix as a heatmap; every cell shows its percentage."""
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file; works on servers without a display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fonts), "DejaVu Sans")
    share = 100 * ratio(cm.ravel(), np.repeat(cm.sum(1), len(names))).reshape(cm.shape)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200, facecolor=SURFACE)
    # A 2-px gap in the surface colour separates the cells (no borders)
    mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=100, edgecolors=SURFACE, linewidth=2)
    for r in range(len(names)):
        for c in range(len(names)):
            ax.text(c + 0.5, r + 0.5, f"{share[r, c]:.1f}%", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if r == c else "normal", color=text_on(cmap(share[r, c] / 100)[:3]))
    ax.invert_yaxis()  # first emotion at the top, like the table
    ax.set_aspect("equal")  # square cells
    ax.set_xticks(np.arange(len(names)) + 0.5, names)
    ax.set_yticks(np.arange(len(names)) + 0.5, [f"{n} ({cm[k].sum():,})" for k, n in enumerate(names)])
    ax.tick_params(length=0, colors=INK_2, labelsize=10.5)
    ax.set_xlabel("Predicted emotion", color=INK_2, fontsize=11, labelpad=8)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=11, labelpad=8)
    ax.spines[:].set_visible(False)
    # Title and subtitle centred over the matrix
    ax.set_title(title, loc="center", color=INK, fontsize=13, fontweight="bold", pad=28)
    ax.text(0.5, 1.03, subtitle, transform=ax.transAxes, ha="center", color=INK_2, fontsize=9.5)
    bar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.04, shrink=0.85, ticks=[0, 25, 50, 75, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=INK_2, labelsize=10)
    bar.set_label("Share of the row (%)", color=INK_2, fontsize=10.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.25)  # keep every label inside the image
    plt.close(fig)


class Predictor:
    """Runs the model on one clip and returns its class probabilities. MPS cannot run emotion2vec's first
    convolution on clips longer than about 20 s (more than 65,536 output steps), so such clips run on a CPU
    copy of the model instead."""

    def __init__(self, load, device):
        self.load, self.model, self.cpu_model, self.on_cpu = load, load(device), None, 0

    def __call__(self, path):
        try:
            return self.model.generate(path, granularity="utterance", extract_embedding=False)[0]["scores"]
        except NotImplementedError:
            if self.cpu_model is None:
                self.cpu_model = self.load(torch.device("cpu"))
            self.on_cpu += 1
            return self.cpu_model.generate(path, granularity="utterance", extract_embedding=False)[0]["scores"]


def evaluate(ds, predict, labels, model_name, device, limit, versions):
    """Steps 4-6 for one dataset: load it, run the model on every clip, write the outputs."""
    # ------------------------------------------------------------------
    # Step 4: Load the dataset (scripts/eval_datasets.py) and its labels
    # ------------------------------------------------------------------
    step(f"Step 4/6 | Load {ds.name}")
    clips = ds.clips
    if limit and limit < len(clips):  # evaluation.limit: clips taken at even intervals, so the sample covers the whole dataset
        clips = clips[::len(clips) // limit][:limit]
    sampled = len(clips) < len(ds.clips)
    acted = [c["acted"] for c in clips]
    listener = [c["listener"] for c in clips]
    has_listener = any(listener)
    print(f"  clips       : {len(clips):,}" + (f" of {len(ds.clips):,}, taken at even intervals" if sampled else "")
          + f" ({ds.language})")
    for name in ds.classes:
        print(f"  {name:<11} : {acted.count(name):,} clips")
    print(f"  listeners   : {'yes' if has_listener else 'no labels in this dataset'}")

    # ------------------------------------------------------------------
    # Step 5: Run the model on every clip (one probability per class)
    # ------------------------------------------------------------------
    step(f"Step 5/6 | Run inference on {len(clips):,} {ds.name} clips")
    predict(str(clips[0]["audio"]))  # warm-up
    scores = np.zeros((len(clips), len(labels)))
    t0, on_cpu = time.perf_counter(), predict.on_cpu
    for i, clip in enumerate(clips, 1):
        scores[i - 1] = predict(str(clip["audio"]))
        if i % 500 == 0 or i == len(clips):
            elapsed = time.perf_counter() - t0
            print(f"  {i:>6,}/{len(clips):,} clips | {elapsed / i * 1000:.0f} ms per clip | "
                  f"about {elapsed / i * (len(clips) - i) / 60:.1f} min left")
    runtime, on_cpu = time.perf_counter() - t0, predict.on_cpu - on_cpu
    if on_cpu:
        print(f"  {on_cpu:,} clips longer than about 20 s ran on the CPU (too long for MPS)")

    # ------------------------------------------------------------------
    # Step 6: Compare with the labels and write the outputs
    # ------------------------------------------------------------------
    step(f"Step 6/6 | Compute accuracy and write the {ds.name} report")
    names, n_cls = ds.classes, len(ds.classes)
    columns = [labels.index(n) for n in names]            # where the dataset's emotions sit in the model output
    truth = np.array([names.index(a) for a in acted])
    pred = scores[:, columns].argmax(1)                   # most probable of the dataset's emotions
    pred9 = scores.argmax(1)                              # most probable of all 9 model classes
    # Two ways to count a prediction as correct (the second only when the dataset has listeners' labels):
    #   strict  - it matches the acted emotion
    #   lenient - it matches the acted emotion OR the listeners' vote (any emotion of a tied vote)
    strict = pred == truth
    heard = [set(v.split(" or ")) if v else set() for v in listener]
    lenient = strict | np.array([names[p] in h for p, h in zip(pred, heard)])
    cm = np.zeros((n_cls, n_cls), dtype=int)
    np.add.at(cm, (truth, pred), 1)                       # rows: acted emotion, columns: prediction
    # Lenient matrix: a clip goes into the row of the label its prediction matched, otherwise its acted emotion
    cm_lenient = np.zeros((n_cls, n_cls), dtype=int)
    np.add.at(cm_lenient, (np.where(lenient, pred, truth), pred), 1)
    recall = ratio(cm.diagonal(), cm.sum(1))
    precision = ratio(cm.diagonal(), cm.sum(0))
    f1 = ratio(2 * precision * recall, precision + recall)
    acc, acc_lenient = strict.mean(), lenient.mean()
    lenient_by_emotion = [lenient[truth == k].mean() if (truth == k).any() else 0.0 for k in range(n_cls)]
    rescued = collections.Counter((acted[i], names[pred[i]]) for i in np.flatnonzero(lenient & ~strict))
    acc9 = np.mean([labels[p] == a for p, a in zip(pred9, acted)])
    extra = {lab: int((pred9 == k).sum()) for k, lab in enumerate(labels) if lab not in names}
    # Human reference: how often the listeners' single majority vote matches the acted emotion (ties left out)
    voted = [i for i, v in enumerate(listener) if v and " or " not in v]
    human = np.mean([listener[i] == acted[i] for i in voted]) if voted else None
    agree = np.mean([names[pred[i]] == listener[i] for i in voted]) if voted else None
    print(f"  accuracy, acted emotion             : {acc:.1%}")
    if has_listener:
        print(f"  accuracy, acted emotion or listener : {acc_lenient:.1%}")
    print(f"  unweighted accuracy (UA)            : {recall.mean():.1%}")
    print(f"  macro F1                            : {f1.mean():.1%}")
    print(f"  accuracy, all 9 classes             : {acc9:.1%}")
    if human is not None:
        print(f"  listeners vs acted emotion          : {human:.1%}")

    # Per-clip predictions (CSV) and a description with the metrics (JSON, read by make_viewer.py)
    suffix = f"_sample{len(clips)}" if sampled else ""  # a sample never overwrites the results of a full run
    stem = f"{ds.key}_{model_name}{suffix}"
    predictions = ROOT / "outputs" / f"{stem}_predictions.csv"
    predictions.parent.mkdir(exist_ok=True)
    prob_columns = [f"p_{lab.strip('<>')}" for lab in labels]  # "<unk>" -> "p_unk"
    with open(predictions, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "acted_emotion"] + ["listener_vote"] * has_listener
                        + ["prediction", "prediction_9_classes", "correct"]
                        + ["correct_acted_or_listener"] * has_listener + prob_columns)
        for i, clip in enumerate(clips):
            writer.writerow([clip["audio"].relative_to(ROOT).as_posix(), acted[i]] + [listener[i]] * has_listener
                            + [names[pred[i]], labels[pred9[i]], bool(strict[i])] + [bool(lenient[i])] * has_listener
                            + [f"{p:.4f}" for p in scores[i]])
    metrics = {"accuracy": acc, "unweighted_accuracy": recall.mean(), "macro_f1": f1.mean(), "accuracy_9_classes": acc9}
    if has_listener:
        metrics |= {"accuracy_acted_or_listener": acc_lenient, "listeners_vs_acted": human, "model_vs_listeners": agree}
    columns_doc = {
        "audio_path": "audio file, relative to the repository root",
        "acted_emotion": f"ground truth: {ds.ground_truth}",
        "listener_vote": f"{ds.listener}; 'a or b' is a tie",
        "prediction": f"model prediction: the most probable of the {n_cls} emotions evaluated ({', '.join(names)})",
        "prediction_9_classes": "model prediction: the most probable of all 9 model classes",
        "correct": "whether prediction equals acted_emotion",
        "correct_acted_or_listener": "whether prediction equals acted_emotion or the listener_vote (any emotion of a tie)",
        "p_*": "model probability of each of its 9 classes (they add up to 1)",
    }
    meta = {
        "dataset": ds.name, "key": ds.key, "language": ds.language, "model": model_name, "date": str(date.today()),
        "clips": len(clips), "total_clips": len(ds.clips), "classes": names, "in_emobox": ds.in_emobox,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items() if v is not None},
        "ground_truth": "acted_emotion",
        # Correctness columns and what the prediction is compared with (the viewer can switch between them)
        "correct_columns": {"correct": "acted emotion"} | ({"correct_acted_or_listener": "acted emotion or listener vote"}
                                                           if has_listener else {}),
        "columns": {k: v for k, v in columns_doc.items() if has_listener or "listener" not in k},
    }
    predictions.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Confusion-matrix figures and the Markdown report
    figures = ROOT / "reports" / "figures"
    figure = figures / f"{stem}_confusion.png"
    figure_lenient = figures / f"{stem}_confusion_acted_or_listener.png"
    plot_confusion(cm, names, f"{model_name} on {ds.name}",
                   f"Correct = matches the acted emotion. {len(clips):,} clips, accuracy {100 * acc:.1f}%.",
                   "Acted emotion (clips)", figure)
    if has_listener:
        plot_confusion(cm_lenient, names, f"{model_name} on {ds.name}",
                       f"Correct = matches the acted emotion or the listeners' vote. Accuracy {100 * acc_lenient:.1f}%.",
                       "Acted emotion, or listener vote if matched (clips)", figure_lenient)
    pct = lambda x: f"{100 * x:.1f}%"
    image = lambda path, alt: f"![{alt}]({path.relative_to(ROOT / 'reports').as_posix()})"
    report = [
        f"# {model_name} on {ds.name}",
        "",
        f"Generated by `scripts/evaluate.py` on {date.today()}."
        + (f" **Sample**: {len(clips):,} of the {len(ds.clips):,} clips, taken at even intervals in file order "
           "(`evaluation.limit` in config/config.yaml)." if sampled else ""),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Clips evaluated | {len(clips):,} ({ds.language}) |",
        f"| **Accuracy: prediction matches the acted emotion** | **{pct(acc)}** |",
        *([f"| **Accuracy: prediction matches the acted emotion or the listeners' vote** | **{pct(acc_lenient)}** |"]
          if has_listener else []),
        f"| Unweighted accuracy (mean recall of the {n_cls} acted emotions) | {pct(recall.mean())} |",
        f"| Macro F1 (acted emotion) | {pct(f1.mean())} |",
        f"| Accuracy, all 9 model classes (acted emotion) | {pct(acc9)} |",
        f"| Chance level ({n_cls} emotions) | {pct(1 / n_cls)} |",
        *([f"| Listeners' majority vote vs acted emotion | {pct(human)} |"] if human is not None else []),
        "",
        "## Setup",
        "",
        f"- **Model**: `{model_name}` as released (no fine-tuning), FunASR {versions['funasr']}, one prediction per clip.",
        f"- **Data**: {len(clips):,} clips of {ds.name} ({ds.language}) from `data/{ds.key}/`"
        + (f", a sample of its {len(ds.clips):,} clips taken at even intervals in file order." if sampled else "."),
        f"- **Ground truth (acted emotion)**: {ds.ground_truth}.",
        *([f"- **Listeners' vote**: {ds.listener}."] if has_listener else []),
        *[f"- {note}" for note in ds.notes],
        f"- **{n_cls} emotions**: the prediction is the most probable of the {n_cls} emotions in this dataset "
        f"({', '.join(names)}); the model's other classes ({', '.join(f'`{lab}`' for lab in extra)}) are ignored. "
        "**All 9 classes**: the prediction is the most probable of all nine classes, so choosing a class that the "
        "dataset does not have counts as wrong.",
        *(["- **Two ways to count a prediction as correct.** *Acted emotion*: it must match the acted emotion. *Acted "
           "emotion or listeners' vote*: it also counts when it matches the listeners' majority vote (for a tied vote, "
           "any of the tied emotions). Both are reported."] if has_listener else []),
        f"- **Hardware**: {hardware(device)}, device `{device}`, PyTorch {torch.__version__}. "
        f"Inference took {runtime / 60:.1f} min ({runtime / len(clips) * 1000:.0f} ms per clip)."
        + (f" {on_cpu:,} clips longer than about 20 s ran on the CPU, because MPS cannot process them."
           if on_cpu else ""),
        f"- **Per-clip predictions**: `{predictions.relative_to(ROOT)}` (not in git; re-run the script to recreate it). "
        "The JSON file next to it describes the columns.",
        "",
        "## Results per emotion",
        "",
        "Rows: acted emotion. Precision, recall and F1 compare the prediction with the acted emotion"
        + ("; the last column also accepts the listeners' vote." if has_listener else "."),
        "",
        "| Emotion | Clips | Precision | Recall | F1 |" + (" Correct, acted or listeners' vote |" if has_listener else ""),
        "|---|---|---|---|---|" + ("---|" if has_listener else ""),
        *[f"| {n} | {cm[k].sum():,} | {pct(precision[k])} | {pct(recall[k])} | {pct(f1[k])} |"
          + (f" {pct(lenient_by_emotion[k])} |" if has_listener else "") for k, n in enumerate(names)],
        "",
        "## Confusion matrix: acted emotion",
        "",
        image(figure, f"Confusion matrix of {model_name} on {ds.name}, acted emotion"),
        "",
        "The same numbers as a table. Rows: acted emotion. Columns: predicted emotion. Each row adds up to 100%.",
        "",
        *confusion_table(cm, names, "Acted \\ Predicted"),
        "",
        f"Most frequent confusions: {top_confusions(cm, names)}.",
        "",
    ]
    if has_listener:
        report += [
            "## Confusion matrix: acted emotion or listeners' vote",
            "",
            image(figure_lenient, f"Confusion matrix of {model_name} on {ds.name}, acted emotion or listeners' vote"),
            "",
            "Here a prediction is correct when it matches the acted emotion or the listeners' vote. A clip goes into "
            "the row of the label its prediction matched, otherwise into the row of its acted emotion. The diagonal "
            "therefore holds every correct prediction, and the rows differ in size from the matrix above. "
            f"{sum(rescued.values()):,} clips ({pct(sum(rescued.values()) / len(clips))}) count as correct only because "
            "of the listeners' vote" + ("; the most common are " + ", ".join(
                f"acted {a} predicted as {p} ({n:,})" for (a, p), n in rescued.most_common(3)) if rescued else "") + ".",
            "",
            *confusion_table(cm_lenient, names, "Ground truth \\ Predicted"),
            "",
            f"Most frequent confusions: {top_confusions(cm_lenient, names)}.",
            "",
        ]
    report += [
        f"## Predictions outside the {n_cls} emotions",
        "",
        "With all nine classes, the model chose a class that this dataset does not have for "
        f"{sum(extra.values()):,} clips ({pct(sum(extra.values()) / len(clips))}): "
        + ", ".join(f"`{lab}` {n:,}" for lab, n in extra.items()) + ".",
        "",
    ]
    if human is not None:
        report += [
            "## Comparison with human listeners",
            "",
            f"The listeners' majority vote matches the acted emotion for {pct(human)} of the {len(voted):,} clips with a "
            f"single (untied) vote. The model agrees with that vote for {pct(agree)} of the same clips.",
            "",
        ]
    report += ["## Caveats", "", *[f"- {c}" for c in ds.caveats]]
    if has_listener:
        report.append("- **Acted labels are not perceived emotions.** The ground truth is what the actor intended; "
                      "listeners sometimes heard something else, as the listeners' numbers above show.")
    report.append("")
    out = ROOT / "reports" / f"{stem}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nDone: report -> {out.relative_to(ROOT)}, predictions -> {predictions.relative_to(ROOT)}")


def write_summary(model_name):
    """reports/<model>_summary.md: one row per evaluated dataset, read from the JSON files in outputs/. A dataset with
    a full run and a sample (evaluation.limit) is shown with the run that has the most clips."""
    order = list(LOADERS)
    out = ROOT / "outputs"
    best = {}
    for p in [*out.glob(f"*_{model_name}_predictions.json"), *out.glob(f"*_{model_name}_sample*_predictions.json")]:
        run = json.loads(p.read_text(encoding="utf-8")) | {"stem": p.name.removesuffix("_predictions.json")}
        if "metrics" in run and run["clips"] > best.get(run["key"], {"clips": 0})["clips"]:
            best[run["key"]] = run
    runs = sorted(best.values(), key=lambda r: order.index(r["key"]) if r["key"] in order else len(order))
    sampled = [r for r in runs if r.get("total_clips", r["clips"]) > r["clips"]]
    pct = lambda x: f"{100 * x:.1f}%" if x is not None else "–"
    clips = lambda r: f"{r['clips']:,}" + (f" of {r['total_clips']:,} (sample)" if r in sampled else "")
    rows = [f"| [{r['dataset']}]({r['stem']}.md) | {r['language']} | {clips(r)} | {len(r['classes'])} | "
            f"{pct(1 / len(r['classes']))} | **{pct(r['metrics']['accuracy'])}** | "
            f"{pct(r['metrics'].get('accuracy_acted_or_listener'))} | {pct(r['metrics']['unweighted_accuracy'])} | "
            f"{pct(r['metrics']['macro_f1'])} | {'yes' if r['in_emobox'] else 'no'} |" for r in runs]
    summary = [
        f"# {model_name}: results on every dataset",
        "",
        f"Generated by `scripts/evaluate.py` on {date.today()}. Each dataset links to its full report.",
        "",
        "| Dataset | Language | Clips | Emotions | Chance | Accuracy: acted emotion | Accuracy: acted or listeners' vote "
        "| Unweighted accuracy | Macro F1 | In EmoBox |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "- **Accuracy: acted emotion** compares the prediction with the emotion the speaker was asked to portray. "
        "The prediction is the most probable of the emotions that the dataset has, so chance level depends on how "
        "many emotions it has.",
        "- **Accuracy: acted or listeners' vote** also accepts the listeners' majority vote; it exists only for "
        "datasets that publish listeners' labels.",
        "- **In EmoBox**: the dataset was probably part of emotion2vec+'s training data, so its result is optimistic. "
        "For the others, earlier exposure cannot be ruled out completely either (see each report).",
        *(["- **(sample)**: evaluated on clips taken at even intervals (`evaluation.limit`), not on every clip."]
          if sampled else []),
        "",
    ]
    path = ROOT / "reports" / f"{model_name}_summary.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    print(f"Summary of {len(runs)} dataset(s) -> {path.relative_to(ROOT)}")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the project configuration (config/config.yaml, section `evaluation`)
    # ------------------------------------------------------------------
    step("Step 1/6 | Load configuration")
    cfg = yaml.safe_load(CONFIG.read_text())
    settings = cfg["evaluation"]
    mode = cfg["mode"]
    device = torch.device(settings["devices"][mode])
    limit = settings["limit"]
    model_name = cfg["model"]
    model_dir = ROOT / cfg["models"][model_name]["dir"]
    wanted = settings["datasets"]
    if wanted == "all":
        keys = [k for k in LOADERS if (ROOT / "data" / k).is_dir()]
    else:
        keys = [wanted] if isinstance(wanted, str) else list(wanted)
    unknown = [k for k in keys if k not in LOADERS]
    missing = [k for k in keys if k in LOADERS and not (ROOT / "data" / k).is_dir()]
    if unknown:
        sys.exit(f"ERROR: unknown dataset(s) {unknown} in config/config.yaml; use: {', '.join(LOADERS)} or all")
    if not keys or missing:
        sys.exit(f"ERROR: not downloaded: {', '.join(missing) or 'any dataset'}. Download it with "
                 "scripts/download_data.py (set `download.dataset` in config/config.yaml).")
    print(f"  config      : {CONFIG}")
    print(f"  mode        : {mode}")
    print(f"  device      : {device}")
    print(f"  model       : {model_name} ({model_dir})")
    print(f"  datasets    : {', '.join(keys)}")
    print(f"  limit       : {f'{limit:,} clips of each dataset, taken at even intervals' if limit else 'every clip'}")

    # ------------------------------------------------------------------
    # Step 2: Check the software stack and the device
    # ------------------------------------------------------------------
    step("Step 2/6 | Check environment and device")
    import funasr  # imported here because it takes a few seconds
    from funasr import AutoModel

    logging.getLogger().setLevel(logging.ERROR)  # importing FunASR turns on INFO logs; keep the output readable
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("ERROR: CUDA was requested but is not available (see README, Troubleshooting).")
    if device.type == "mps" and not torch.backends.mps.is_available():
        sys.exit("ERROR: MPS was requested but is not available (see README, Troubleshooting).")
    print(f"  Python      : {platform.python_version()}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  FunASR      : {funasr.__version__}")
    print(f"  hardware    : {hardware(device)}")

    # ------------------------------------------------------------------
    # Step 3: Load the model onto the device (same settings as scripts/inference.py)
    # ------------------------------------------------------------------
    step("Step 3/6 | Load model")
    t0 = time.perf_counter()
    load = lambda dev: AutoModel(
        model=str(model_dir),
        device=str(dev),
        ncpu=torch.get_num_threads(),
        excludes="modality_encoders.AUDIO.decoder.",  # pre-training decoder: not in model.pt, unused for inference
        disable_update=True,
        disable_pbar=True,
    )
    predict = Predictor(load, device)
    labels = [t.split("/")[-1] for t in (model_dir / "tokens.txt").read_text(encoding="utf-8").split()]
    print(f"  loaded in   : {time.perf_counter() - t0:.1f} s")
    print(f"  classes     : {', '.join(labels)}")

    # Steps 4-6 for every dataset, then a summary of all evaluated datasets
    for key in keys:
        evaluate(LOADERS[key](), predict, labels, model_name, device, limit, {"funasr": funasr.__version__})
    write_summary(model_name)


if __name__ == "__main__":
    main()

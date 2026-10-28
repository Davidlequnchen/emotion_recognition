"""
Evaluate emotion2vec+ on the speech emotion datasets and write a Markdown report for each.

Runs the model on every clip of a dataset, compares each prediction with the dataset's labels and writes:
    reports/<dataset>_<model>.md                      accuracy, per-emotion results, confusion matrices
    reports/figures/<dataset>_<model>_confusion*.png  confusion-matrix figures used in the report
    outputs/<dataset>_<model>_predictions.csv          one row per clip (read by scripts/viewer/make_viewer.py)
    outputs/<dataset>_<model>_predictions.json         what the CSV contains, and the metrics
    reports/<model>_summary.md                         every evaluated dataset side by side (.json: the same numbers)
    reports/figures/<model>_summary_confusion.png      every dataset's confusion matrix in one figure
The datasets are defined in scripts/common/eval_datasets.py; download them first with
scripts/common/download_data.py.

Usage (from the emotion_recognition/ folder):
    python scripts/emotion2vec/evaluate.py
Settings (which datasets, an optional sample size, device) are in the `evaluation` section of config/config.yaml.
A dataset whose results for the same model and sample size are already in outputs/ is skipped, unless
`evaluation.overwrite` is true. The summary covers every dataset with results in outputs/. Datasets listed in
`evaluation.exclude` are never evaluated and left out of the summary.
"""
import collections
import csv
import json
import logging
import math
import platform
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # eval_datasets.py is in scripts/common/
from eval_datasets import LOADERS, accepted, sample, truth_labels

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
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


def correct_cells(ds, rows):
    """Which cells of a confusion matrix count as correct (rows: ground-truth labels, columns: ds.classes): the
    column of the row's own label, or for a composite label such as excited every class it accepts."""
    return np.array([[name in accepted(ds, row) for name in ds.classes] for row in rows])


def row_name(ds, label):
    """A row label; a composite label also names the classes it accepts, e.g. "excited (happy or surprised)"."""
    return f"{label} ({' or '.join(ds.accept[label])})" if label in ds.accept else label


def confusion_table(cm, ds, rows, corner):
    """A confusion matrix as Markdown table rows: row percentages, the cells that count as correct in bold."""
    names, good = ds.classes, correct_cells(ds, rows)
    lines = [f"| {corner} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, row in enumerate(rows):
        share = ratio(cm[k], np.full(len(names), cm[k].sum()))
        lines.append(f"| {row_name(ds, row)} ({cm[k].sum():,}) | " + " | ".join(
            f"**{100 * s:.1f}%**" if good[k, j] else f"{100 * s:.1f}%" for j, s in enumerate(share)) + " |")
    return lines


def top_confusions(cm, ds, rows, n=3):
    """The n largest row shares among the cells that count as wrong, as text such as "fearful predicted as sad
    (14.9%)"."""
    names, good = ds.classes, correct_cells(ds, rows)
    cells = sorted(((cm[r, c] / cm[r].sum(), rows[r], names[c]) for r in range(len(rows))
                    for c in range(len(names)) if not good[r, c] and cm[r].sum()), reverse=True)[:n]
    return ", ".join(f"{a} predicted as {b} ({100 * v:.1f}%)" for v, a, b in cells)


def text_on(rgb):
    """White or ink, whichever has more contrast with the cell colour (WCAG relative luminance)."""
    lum = sum(w * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
              for w, v in zip((0.2126, 0.7152, 0.0722), rgb))
    return "#ffffff" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.0533 else INK


def plot_confusion(cm, ds, rows, title, subtitle, ylabel, path):
    """Row-normalised confusion matrix as a heatmap (rows: ground-truth labels, columns: ds.classes); every cell
    shows its percentage, in bold where the prediction counts as correct. In the row of a composite label (e.g.
    JL-Corpus's excited) the cells that count as correct are also outlined, as bold text is hard to see in a light
    cell."""
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file; works on servers without a display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fonts), "DejaVu Sans")
    names, good = ds.classes, correct_cells(ds, rows)
    share = 100 * ratio(cm.ravel(), np.repeat(cm.sum(1), len(names))).reshape(cm.shape)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=200, facecolor=SURFACE)
    # A 2-px gap in the surface colour separates the cells (no borders)
    mesh = ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=100, edgecolors=SURFACE, linewidth=2)
    # Composite labels: an ink outline around each cell that counts as correct, as wide as the frame and inside a wider
    # gap in the surface colour, so that it shows on light and dark cells (drawn before the frame, which stays on top)
    for r, c in zip(*np.nonzero(good & np.array([[row in ds.accept] for row in rows]))):
        for colour, width in ((SURFACE, 6.5), (INK, 2.5)):
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor=colour, linewidth=width))
    # A thick frame around the grid of cells (clip_on=False draws its full width at the edge of the axes)
    ax.add_patch(plt.Rectangle((0, 0), len(names), len(rows), fill=False, edgecolor=INK_2, linewidth=2.5, clip_on=False))
    for r in range(len(rows)):
        for c in range(len(names)):
            ax.text(c + 0.5, r + 0.5, f"{share[r, c]:.1f}%", ha="center", va="center", fontsize=7.5,
                    fontweight="bold" if good[r, c] else "normal", color=text_on(cmap(share[r, c] / 100)[:3]))
    ax.invert_yaxis()  # first emotion at the top, like the table
    ax.set_aspect("equal")  # square cells
    crowded = len(names) > 6  # with 7 or more emotions the names would touch: tilt them
    ax.set_xticks(np.arange(len(names)) + 0.5, names, rotation=30 if crowded else 0,
                  ha="right" if crowded else "center", rotation_mode="anchor")
    # Row labels with their number of clips; a composite label names the classes it accepts on a second line
    ax.set_yticks(np.arange(len(rows)) + 0.5, [f"{n} ({cm[k].sum():,})" + (
        f"\n({' or '.join(ds.accept[n])})" if n in ds.accept else "") for k, n in enumerate(rows)])
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


def stitch(paths, path, columns=3, col_gap=110, row_gap=80, margin=50):
    """Put figures in a grid, row by row. The empty margin of each figure is cut off first, so the gaps set the
    spacing; each figure is centred across its cell and aligned to its top, so the titles of a row line up."""
    from PIL import Image, ImageChops
    images = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        images.append(im.crop(ImageChops.difference(im, Image.new("RGB", im.size, SURFACE)).getbbox()))
    columns = min(columns, len(images))
    rows = -(-len(images) // columns)
    w, h = max(im.width for im in images), max(im.height for im in images)
    grid = Image.new("RGB", (2 * margin + columns * w + (columns - 1) * col_gap,
                             2 * margin + rows * h + (rows - 1) * row_gap), SURFACE)
    for k, im in enumerate(images):
        grid.paste(im, (margin + (k % columns) * (w + col_gap) + (w - im.width) // 2,
                        margin + (k // columns) * (h + row_gap)))
    grid.save(path, optimize=True)


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
    # Step 4: Load the dataset (scripts/common/eval_datasets.py) and its labels
    # ------------------------------------------------------------------
    step(f"Step 4/6 | Load {ds.name}")
    # evaluation.limit: clips taken at even intervals, emotion by emotion (in file order within each), so that
    # every emotion is sampled even when the file order repeats (TESS: the 7 emotions of each word in a row)
    clips = sample(ds, limit)
    sampled = len(clips) < len(ds.clips)
    acted = [c["acted"] for c in clips]
    listener = [c["listener"] for c in clips]
    has_listener = any(listener)
    rows = truth_labels(ds)  # the ground-truth labels: the rows of the confusion matrices
    composite = [r for r in rows if r in ds.accept]  # labels that accept several classes, e.g. JL-Corpus's excited
    print(f"  clips       : {len(clips):,}" + (f" of {len(ds.clips):,}, taken at even intervals within each emotion"
                                       if sampled else "")
          + f" ({ds.language})")
    for name in rows:
        print(f"  {name:<11} : {acted.count(name):,} clips"
              + (f" (correct: {' or '.join(ds.accept[name])})" if name in composite else ""))
    if composite:
        print(f"  predictions : {', '.join(ds.classes)}")
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
    names, n_cls = ds.classes, len(ds.classes)            # the classes a prediction is chosen from
    columns = [labels.index(n) for n in names]            # where the dataset's emotions sit in the model output
    truth = np.array([rows.index(a) for a in acted])      # each clip's row: its acted emotion
    pred = scores[:, columns].argmax(1)                   # most probable of the dataset's emotions
    pred9 = scores.argmax(1)                              # most probable of all 9 model classes
    good = correct_cells(ds, rows)                        # rows x classes: True where the prediction counts as correct
    # Two ways to count a prediction as correct (the second only when the dataset has listeners' labels):
    #   strict  - it matches the acted emotion (for a composite label such as excited: one of the classes it accepts)
    #   lenient - it matches the acted emotion OR the listeners' vote (any emotion of a tied vote)
    strict = good[truth, pred]
    heard = [set(v.split(" or ")) if v else set() for v in listener]
    lenient = strict | np.array([names[p] in h for p, h in zip(pred, heard)])
    cm = np.zeros((len(rows), n_cls), dtype=int)
    np.add.at(cm, (truth, pred), 1)                       # rows: acted emotion, columns: prediction
    # Lenient matrix: a clip that counts as correct only because of the listeners' vote goes into the row of that vote
    # (its prediction), every other clip into the row of its acted emotion; a class without a row gets one here
    rows_lenient = rows + [n for n in names if n not in rows]
    cm_lenient = np.zeros((len(rows_lenient), n_cls), dtype=int)
    np.add.at(cm_lenient, (np.where(lenient & ~strict, [rows_lenient.index(names[p]) for p in pred], truth), pred), 1)
    recall = ratio((cm * good).sum(1), cm.sum(1))         # per acted emotion: the share of its clips that is correct
    # Precision, recall and F1 per class compare the prediction with a resolved truth: the prediction when it counts
    # as correct, otherwise the acted emotion, or for a composite label the first class it accepts (so without
    # composite labels, simply the acted emotion)
    resolved = np.where(strict, pred, [names.index(ds.accept.get(a, [a])[0]) for a in acted])
    cm_resolved = np.zeros((n_cls, n_cls), dtype=int)
    np.add.at(cm_resolved, (resolved, pred), 1)
    precision = ratio(cm_resolved.diagonal(), cm_resolved.sum(0))
    recall_resolved = ratio(cm_resolved.diagonal(), cm_resolved.sum(1))
    f1 = ratio(2 * precision * recall_resolved, precision + recall_resolved)
    # Macro F1 averages the classes' F1. A class that no clip is labelled with (surprised in JL-Corpus, a possible
    # prediction only because excited accepts it) is left out while the model never predicts it: its F1 is undefined
    scored = [k for k, n in enumerate(names) if n in rows or (pred == k).any()]
    unscored = [n for k, n in enumerate(names) if k not in scored]
    macro_f1 = f1[scored].mean()
    acc, acc_lenient = strict.mean(), lenient.mean()
    lenient_by_emotion = [lenient[truth == k].mean() if (truth == k).any() else 0.0 for k in range(len(rows))]
    rescued = collections.Counter((acted[i], names[pred[i]]) for i in np.flatnonzero(lenient & ~strict))
    acc9 = np.mean([labels[p] in accepted(ds, a) for p, a in zip(pred9, acted)])
    extra = {lab: int((pred9 == k).sum()) for k, lab in enumerate(labels) if lab not in names}
    # Chance level: a random guess among the classes is right with probability (accepted classes) / (classes)
    chance = np.mean([len(accepted(ds, a) & set(names)) for a in acted]) / n_cls
    # Human reference: how often the listeners' single majority vote matches the acted emotion (ties left out)
    voted = [i for i, v in enumerate(listener) if v and " or " not in v]
    human = np.mean([listener[i] in accepted(ds, acted[i]) for i in voted]) if voted else None
    agree = np.mean([names[pred[i]] == listener[i] for i in voted]) if voted else None
    print(f"  accuracy, acted emotion             : {acc:.1%}")
    if has_listener:
        print(f"  accuracy, acted emotion or listener : {acc_lenient:.1%}")
    print(f"  unweighted accuracy (UA)            : {recall.mean():.1%}")
    print(f"  macro F1                            : {macro_f1:.1%}"
          + (f" ({', '.join(unscored)} left out: never predicted, no clip has this label)" if unscored else ""))
    print(f"  accuracy, all 9 classes             : {acc9:.1%}")
    print(f"  chance level                        : {chance:.1%}")
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
    metrics = {"accuracy": acc, "unweighted_accuracy": recall.mean(), "macro_f1": macro_f1, "accuracy_9_classes": acc9}
    if has_listener:
        metrics |= {"accuracy_acted_or_listener": acc_lenient, "listeners_vs_acted": human, "model_vs_listeners": agree}
    rule = "; ".join(f"{r}: {' or '.join(ds.accept[r])}" for r in composite)  # e.g. "excited: happy or surprised"
    columns_doc = {
        "audio_path": "audio file, relative to the emotion_recognition/ folder",
        "acted_emotion": f"ground truth: {ds.ground_truth}",
        "listener_vote": f"{ds.listener}; 'a or b' is a tie",
        "prediction": f"model prediction: the most probable of the {n_cls} emotions evaluated ({', '.join(names)})",
        "prediction_9_classes": "model prediction: the most probable of all 9 model classes",
        "correct": "whether prediction equals acted_emotion" + (f" or is accepted for it ({rule})" if composite else ""),
        "correct_acted_or_listener": "whether prediction equals acted_emotion or the listener_vote (any emotion of a tie)"
                                     + (" or is accepted for acted_emotion" if composite else ""),
        "p_*": "model probability of each of its 9 classes (they add up to 1)",
    }
    meta = {
        "dataset": ds.name, "key": ds.key, "language": ds.language, "model": model_name, "date": str(date.today()),
        "clips": len(clips), "total_clips": len(ds.clips), "classes": names,
        "labels": rows,       # the ground-truth labels, in the order of the confusion-matrix rows
        "accept": ds.accept,  # composite labels: the classes that count as correct for them
        "chance": round(float(chance), 6), "in_emobox": ds.in_emobox,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items() if v is not None},
        "ground_truth": "acted_emotion",
        # Correctness columns and what the prediction is compared with (the viewer can switch between them)
        "correct_columns": {"correct": "acted emotion"} | ({"correct_acted_or_listener": "acted emotion or listener vote"}
                                                           if has_listener else {}),
        "columns": {k: v for k, v in columns_doc.items() if has_listener or "listener" not in k},
    }

    # Confusion-matrix figures and the Markdown report
    figures = ROOT / "reports" / "figures"
    figure = figures / f"{stem}_confusion.png"
    figure_lenient = figures / f"{stem}_confusion_acted_or_listener.png"
    plot_confusion(cm, ds, rows, f"{model_name} on {ds.name}",
                   (f"Bold = correct ({rule})." if composite else "Correct = matches the acted emotion.")
                   + f" {len(clips):,} clips, accuracy {100 * acc:.1f}%.",
                   "Acted emotion (clips)", figure)
    if has_listener:
        plot_confusion(cm_lenient, ds, rows_lenient, f"{model_name} on {ds.name}",
                       f"Correct = matches the acted emotion or the listeners' vote. Accuracy {100 * acc_lenient:.1f}%.",
                       "Acted emotion, or listener vote if matched (clips)", figure_lenient)
    pct = lambda x: f"{100 * x:.1f}%"
    image = lambda path, alt: f"![{alt}]({path.relative_to(ROOT / 'reports').as_posix()})"
    # Results per emotion: one table; with composite labels, one per acted emotion (the recall that unweighted accuracy
    # averages) and one per predicted emotion (precision, recall and F1 against the resolved truth, for macro F1)
    lenient_column = (" Correct, acted or listeners' vote |", "---|") if has_listener else ("", "")
    lenient_cell = lambda k: f" {pct(lenient_by_emotion[k])} |" if has_listener else ""
    f1_scope = (f", over {len(scored)} of the {n_cls} predicted emotions" if unscored
                else f", over the {n_cls} predicted emotions" if composite else "")
    if not composite:
        per_emotion = [
            "Rows: acted emotion. Precision, recall and F1 compare the prediction with the acted emotion"
            + ("; the last column also accepts the listeners' vote." if has_listener else "."),
            "",
            "| Emotion | Clips | Precision | Recall | F1 |" + lenient_column[0],
            "|---|---|---|---|---|" + lenient_column[1],
            *[f"| {r} | {cm[k].sum():,} | {pct(precision[names.index(r)])} | {pct(recall[k])} | "
              f"{pct(f1[names.index(r)])} |" + lenient_cell(k) for k, r in enumerate(rows)],
        ]
    else:
        per_emotion = [
            "Rows: acted emotion. Recall: the share of the row's clips whose prediction counts as correct; unweighted "
            "accuracy is the mean of this column" + ("; the last column also accepts the listeners' vote." if has_listener
                                                     else "."),
            "",
            "| Emotion | Clips | Recall |" + lenient_column[0],
            "|---|---|---|" + lenient_column[1],
            *[f"| {row_name(ds, r)} | {cm[k].sum():,} | {pct(recall[k])} |" + lenient_cell(k) for k, r in enumerate(rows)],
            "",
            "Precision, recall and F1 of each predicted emotion compare the prediction with a resolved truth: the "
            "prediction when it counts as correct, otherwise the acted emotion, and for "
            + ", ".join(f"{r} the first class it accepts ({ds.accept[r][0]})" for r in composite)
            + f". So a correct {composite[0]} clip counts as a clip of the emotion predicted, and a wrong one as "
            f"{ds.accept[composite[0]][0]}. Clips: the clips of each resolved truth. Macro F1 is the mean of the "
            "F1 column" + (f"; {', '.join(unscored)} is left out, because no clip has this label and the model never "
                           "predicted it, so its F1 is undefined." if unscored else "."),
            "",
            "| Predicted emotion | Clips | Precision | Recall | F1 |",
            "|---|---|---|---|---|",
            *[f"| {n} | {cm_resolved[k].sum():,} | " + (" | ".join(pct(x[k]) for x in (precision, recall_resolved, f1))
                                                         if k in scored else "– | – | –") + " |" for k, n in enumerate(names)],
        ]
    report = [
        f"# {model_name} on {ds.name}",
        "",
        f"Generated by `scripts/emotion2vec/evaluate.py` on {date.today()}."
        + (f" **Sample**: {len(clips):,} of the {len(ds.clips):,} clips, taken at even intervals within each emotion "
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
        f"| Unweighted accuracy (mean recall of the {len(rows)} acted emotions) | {pct(recall.mean())} |",
        f"| Macro F1 (acted emotion{f1_scope}) | {pct(macro_f1)} |",
        f"| Accuracy, all 9 model classes (acted emotion) | {pct(acc9)} |",
        f"| Chance level ({n_cls} emotions"
        + "".join(f"; {r} accepts {len(accepted(ds, r) & set(names))}" for r in composite) + f") | {pct(chance)} |",
        *([f"| Listeners' majority vote vs acted emotion | {pct(human)} |"] if human is not None else []),
        "",
        "## Setup",
        "",
        f"- **Model**: `{model_name}` as released (no fine-tuning), FunASR {versions['funasr']}, one prediction per clip.",
        f"- **Data**: {len(clips):,} clips of {ds.name} ({ds.language}) from `data/{ds.key}/`"
        + (f", a sample of its {len(ds.clips):,} clips taken at even intervals within each emotion." if sampled else "."),
        f"- **Ground truth (acted emotion)**: {ds.ground_truth}.",
        *([f"- **Listeners' vote**: {ds.listener}."] if has_listener else []),
        *[f"- {note}" for note in ds.notes],
        f"- **{n_cls} emotions**: the prediction is the most probable of the {n_cls} emotions in this dataset "
        f"({', '.join(names)}); the model's other classes ({', '.join(f'`{lab}`' for lab in extra)}) are ignored. "
        "**All 9 classes**: the prediction is the most probable of all nine classes, so choosing a class that the "
        "dataset does not have counts as wrong.",
        *([f"- **Composite label{'s' if len(composite) > 1 else ''}**: "
           + "; ".join(f"*{r}* has no model class, so a prediction of {' or '.join(ds.accept[r])} counts as correct for "
                       "it" for r in composite)
           + " (also with all 9 classes). Each such label has its own row in the confusion matrices, whose bold cells "
           "count as correct; its recall is the share of its clips predicted correctly, and chance level "
           f"({pct(chance)} instead of {pct(1 / n_cls)}) counts every class it accepts."] if composite else []),
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
        *per_emotion,
        "",
        "## Confusion matrix: acted emotion",
        "",
        image(figure, f"Confusion matrix of {model_name} on {ds.name}, acted emotion"),
        "",
        "The same numbers as a table. Rows: acted emotion. Columns: predicted emotion. Each row adds up to 100%."
        + (" Bold: the cells that count as correct." if composite else ""),
        "",
        *confusion_table(cm, ds, rows, "Acted \\ Predicted"),
        "",
        f"Most frequent confusions: {top_confusions(cm, ds, rows)}.",
        "",
    ]
    if has_listener:
        report += [
            "## Confusion matrix: acted emotion or listeners' vote",
            "",
            image(figure_lenient, f"Confusion matrix of {model_name} on {ds.name}, acted emotion or listeners' vote"),
            "",
            "Here a prediction is correct when it matches the acted emotion or the listeners' vote. A clip goes into "
            "the row of the label its prediction matched, otherwise into the row of its acted emotion. "
            + ("The bold cells therefore hold" if composite else "The diagonal therefore holds")
            + " every correct prediction, and the rows differ in size from the matrix above. "
            f"{sum(rescued.values()):,} clips ({pct(sum(rescued.values()) / len(clips))}) count as correct only because "
            "of the listeners' vote" + ("; the most common are " + ", ".join(
                f"acted {a} predicted as {p} ({n:,})" for (a, p), n in rescued.most_common(3)) if rescued else "") + ".",
            "",
            *confusion_table(cm_lenient, ds, rows_lenient, "Ground truth \\ Predicted"),
            "",
            f"Most frequent confusions: {top_confusions(cm_lenient, ds, rows_lenient)}.",
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
    # Written last, so that it exists only for a finished run (see finished_run)
    predictions.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone: report -> {out.relative_to(ROOT)}, predictions -> {predictions.relative_to(ROOT)}")


def shared_scores(run, shared):
    """Clips, accuracy and unweighted accuracy of one run on the emotions every dataset has: clips of other emotions
    are left out, and the prediction is the most probable of the shared emotions."""
    with open(ROOT / "outputs" / f"{run['stem']}_predictions.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["acted_emotion"] in shared]
    pred = []
    for r in rows:
        if r["prediction"] in shared:  # most probable of the dataset's emotions, so also of the shared ones
            pred.append(r["prediction"])
        else:  # use the probabilities in the file; they have 4 decimals, so a tie counts as wrong
            p = [float(r[f"p_{e}"]) for e in shared]
            pred.append(shared[p.index(max(p))] if p.count(max(p)) == 1 else None)
    truth = np.array([r["acted_emotion"] for r in rows])
    hit = np.array(pred) == truth
    return len(rows), hit.mean(), np.mean([hit[truth == e].mean() for e in shared if (truth == e).any()])


def write_summary(model_name, figure_order=(), acted_or_listener=(), exclude=()):
    """reports/<model>_summary.md: one row per evaluated dataset, read from the JSON files in outputs/. A dataset with
    a full run and a sample (evaluation.limit) is shown with the run that has the most clips. The summary figure shows
    the datasets in figure_order first (the others follow in table order); those in acted_or_listener show their
    "acted emotion or listeners' vote" confusion matrix. Datasets in exclude (evaluation.exclude) are left out."""
    order = list(LOADERS)
    out = ROOT / "outputs"
    best = {}
    for p in [*out.glob(f"*_{model_name}_predictions.json"), *out.glob(f"*_{model_name}_sample*_predictions.json")]:
        run = json.loads(p.read_text(encoding="utf-8")) | {"stem": p.name.removesuffix("_predictions.json")}
        if "metrics" in run and run["key"] not in exclude and run["clips"] > best.get(run["key"], {"clips": 0})["clips"]:
            best[run["key"]] = run
    runs = sorted(best.values(), key=lambda r: order.index(r["key"]) if r["key"] in order else len(order))
    sampled = [r for r in runs if r.get("total_clips", r["clips"]) > r["clips"]]
    pct = lambda x: f"{100 * x:.1f}%" if x is not None else "–"
    clips = lambda r: f"{r['clips']:,}" + (f" of {r['total_clips']:,} (sample)" if r in sampled else "")
    chance = lambda r: r.get("chance", 1 / len(r["classes"]))  # older predictions files have no "chance"
    # Ground-truth labels in the order of the confusion-matrix rows. Older predictions files have no "labels": their
    # classes and composite labels stand in, those that clips in the CSV have (JL-Corpus has no clip acted surprised)
    for r in runs:
        if "labels" not in r:
            with open(out / f"{r['stem']}_predictions.csv", newline="", encoding="utf-8") as fh:
                found = {row[r["ground_truth"]] for row in csv.DictReader(fh)}
            r["labels"] = [x for x in [*r["classes"], *r.get("accept", {})] if x in found]

    def labels(r):
        """The labels as text; a composite label names the classes it accepts, e.g. "excited (happy or surprised)".
        No-break spaces keep three labels together: GitHub squeezes a wide table to its narrowest, and the column then
        wraps between such groups instead of after every label."""
        names = [f"{x} ({' or '.join(r['accept'][x])})" if x in r.get("accept", {}) else x for x in r["labels"]]
        return ", ".join(",\u00a0".join(names[k:k + 3]) for k in range(0, len(names), 3))
    emobox = lambda r: "yes" if r["in_emobox"] else "no"
    # Composite labels, e.g. "JL-Corpus's *excited* (happy or surprised)"
    composite = [f"{r['dataset']}'s *{label}* ({' or '.join(accept)})" for r in runs
                 for label, accept in r.get("accept", {}).items()]
    rows = [f"| [{r['dataset']}]({r['stem']}.md) | {r['language']} | {labels(r)} | {emobox(r)} | {clips(r)} | "
            f"{len(r['classes'])} | {pct(chance(r))} | **{pct(r['metrics']['accuracy'])}** | "
            f"{pct(r['metrics'].get('accuracy_acted_or_listener'))} | {pct(r['metrics']['unweighted_accuracy'])} | "
            f"{pct(r['metrics']['macro_f1'])} |" for r in runs]
    summary = [
        f"# {model_name}: results on every dataset",
        "",
        f"Generated by `scripts/emotion2vec/evaluate.py` on {date.today()}. Each dataset links to its full report.",
        "",
        "| Dataset | Language | Ground-truth labels | In emotion2vec+ training data (EmoBox) | Clips | Emotions | Chance "
        "| Accuracy: acted emotion | Accuracy: acted or listeners' vote | Unweighted accuracy | Macro F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "- **Ground-truth labels**: the emotions the speakers were asked to portray, as emotion2vec+ class names (each "
        "report gives the dataset's own names), in the order of the rows of the confusion matrices.",
        *([f"- **Composite labels** (no model class; a prediction of any class in brackets counts as correct): "
           f"{', '.join(composite)}. Chance level counts these classes, so it is higher than 1 in the number of "
           "emotions."] if composite else []),
        "- **In emotion2vec+ training data (EmoBox)**: *yes* means that the dataset is in EmoBox, whose data fine-tuned "
        "emotion2vec+ seed, the first model in the chain that led to emotion2vec+ large; the model has probably seen "
        "these clips, so the result is optimistic. *no*: not in EmoBox, although earlier exposure cannot be ruled out "
        "completely (see each report).",
        "- **Accuracy: acted emotion** compares the prediction with the emotion the speaker was asked to portray. "
        "The prediction is the most probable of the emotions that the dataset has, so chance level depends on how "
        "many emotions it has.",
        "- **Accuracy: acted or listeners' vote** also accepts the listeners' majority vote; it exists only for "
        "datasets that publish listeners' labels.",
        *(["- **(sample)**: evaluated on clips taken at even intervals within each emotion (`evaluation.limit`), not on "
           "every clip."]
          if sampled else []),
        "",
    ]
    # Every dataset's confusion matrix (the figures of the reports) stitched into one figure
    figures = ROOT / "reports" / "figures"
    rank = {key: k for k, key in enumerate(figure_order)}
    shown = sorted(runs, key=lambda r: rank.get(r["key"], len(rank)))
    lenient = [r for r in shown if r["key"] in acted_or_listener
               and (figures / f"{r['stem']}_confusion_acted_or_listener.png").exists()]
    panels = [figures / f"{r['stem']}_confusion{'_acted_or_listener' if r in lenient else ''}.png" for r in shown]
    panels = [p for p in panels if p.exists()]
    grid = figures / f"{model_name}_summary_confusion.png" if panels else None
    if panels:
        stitch(panels, grid, columns=min(3, math.ceil(math.sqrt(len(panels)))))  # 4 panels: 2 x 2, 6: 3 x 2
        summary += [
            "## Confusion matrices",
            "",
            f"![Confusion matrices of {model_name} on every dataset]({grid.relative_to(ROOT / 'reports').as_posix()})",
            "",
            f"One matrix per dataset, from each dataset's report: {', '.join(r['dataset'] for r in shown)} (order: "
            "`evaluation.summary_figure` in config/config.yaml). Rows: acted emotion; columns: predicted emotion; each "
            "row adds up to 100%."
            + (f" {', '.join(r['dataset'] for r in lenient)}: the matrix where a prediction also counts as correct when "
               "it matches the listeners' vote, so a clip whose prediction matched that vote sits in the row of that "
               "vote." if lenient else "")
            + (" A composite label has its own row, whose bold cells count as correct." if composite else ""),
            "",
        ]
    # The same comparison on the emotions that every dataset has, so that the choice of emotions does not matter
    shared = sorted(set.intersection(*(set(r["classes"]) for r in runs))) if len(runs) > 1 else []
    scored = sorted(((r, *shared_scores(r, shared)) for r in runs), key=lambda s: -s[2]) if shared else []
    if shared:
        summary += [
            f"## Only the emotions every dataset has: {', '.join(shared)}",
            "",
            "The datasets have different emotions, and some emotions are much harder than others, so the table above "
            f"also reflects which emotions each dataset has. Here every dataset is scored on the same {len(shared)} "
            f"emotions (chance {pct(1 / len(shared))}): clips of other emotions are left out, and the prediction is "
            f"the most probable of these {len(shared)}.",
            "",
            "| Dataset | Language | Ground-truth labels | In emotion2vec+ training data (EmoBox) | Clips | Accuracy "
            "| Unweighted accuracy |",
            "|---|---|---|---|---|---|---|",
            *[f"| [{r['dataset']}]({r['stem']}.md) | {r['language']} | {labels(r)} | {emobox(r)} | "
              f"{n:,}{' (sample)' if r in sampled else ''} | **{pct(acc)}** | {pct(ua)} |" for r, n, acc, ua in scored],
            "",
            "Computed from the predictions files in `outputs/`. Their probabilities have 4 decimals, so a clip whose "
            "shared emotions are tied counts as wrong.",
            "",
        ]
    path = ROOT / "reports" / f"{model_name}_summary.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    # The same numbers as JSON, for the viewer page (scripts/viewer/make_viewer.py)
    data = {
        "model": model_name, "date": str(date.today()),
        "datasets": [{"dataset": r["dataset"], "key": r["key"], "language": r["language"], "clips": r["clips"],
                      "total_clips": r.get("total_clips", r["clips"]), "emotions": len(r["classes"]),
                      "labels": r["labels"],  # ground-truth labels, in the order of the confusion-matrix rows
                      "chance": round(float(chance(r)), 6), **({"accept": r["accept"]} if r.get("accept") else {}),
                      "in_emobox": r["in_emobox"], "metrics": r["metrics"]} for r in runs],
        "figure": grid.relative_to(ROOT / "reports").as_posix() if grid else None,
        "figure_datasets": [r["dataset"] for r in shown] if grid else [],
        "acted_or_listener": [r["dataset"] for r in lenient] if grid else [],
        "shared_emotions": shared,
        "shared": [{"dataset": r["dataset"], "key": r["key"], "language": r["language"], "clips": n,
                    "sample": r in sampled, "accuracy": round(float(acc), 6), "unweighted_accuracy": round(float(ua), 6),
                    "in_emobox": r["in_emobox"]} for r, n, acc, ua in scored],
    }
    path.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Summary of {len(runs)} dataset(s) -> {path.relative_to(ROOT)} (and .json)")


def finished_run(key, model_name, limit):
    """The predictions JSON of an earlier finished run that a run with these settings would replace, or None.
    With evaluation.limit smaller than the dataset, a run writes <key>_<model>_sample<limit>_*, otherwise
    <key>_<model>_*. The JSON is written last, so it exists only for a finished run."""
    out = ROOT / "outputs"
    sample = out / f"{key}_{model_name}_sample{limit}_predictions.json"
    full = out / f"{key}_{model_name}_predictions.json"
    if limit and sample.exists():
        return sample
    # A full run also counts when the limit is at least the size of the dataset (then there is no sample)
    if full.exists() and (not limit or json.loads(full.read_text(encoding="utf-8"))["clips"] <= limit):
        return full
    return None


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
    limit, overwrite, figure = settings["limit"], settings["overwrite"], settings["summary_figure"]
    exclude = settings["exclude"] or []
    model_name = cfg["model"]
    model_dir = ROOT / cfg["models"][model_name]["dir"]
    wanted = settings["datasets"]
    if wanted == "all":
        keys = [k for k in LOADERS if (ROOT / "data" / k).is_dir()]
    else:
        keys = [wanted] if isinstance(wanted, str) else list(wanted)
    unknown = [k for k in [*keys, *exclude] if k not in LOADERS]
    if unknown:
        sys.exit(f"ERROR: unknown dataset(s) {unknown} in config/config.yaml; use: {', '.join(LOADERS)} or all")
    # Datasets in evaluation.exclude are never evaluated (their data can stay in data/)
    excluded = [k for k in keys if k in exclude]
    keys = [k for k in keys if k not in exclude]
    if excluded and not keys:
        sys.exit(f"ERROR: every dataset to evaluate is in `evaluation.exclude` ({', '.join(excluded)}).")
    # Datasets that already have results for this model and limit in outputs/ are skipped, unless overwrite is true
    done = {} if overwrite else {k: run for k in keys if (run := finished_run(k, model_name, limit))}
    todo = [k for k in keys if k not in done]
    missing = [k for k in todo if not (ROOT / "data" / k).is_dir()]
    if not keys or missing:
        sys.exit(f"ERROR: not downloaded: {', '.join(missing) or 'any dataset'}. Download it with "
                 "scripts/common/download_data.py (set `download.dataset` in config/config.yaml).")
    print(f"  config      : {CONFIG}")
    print(f"  mode        : {mode}")
    print(f"  device      : {device}")
    print(f"  model       : {model_name} ({model_dir})")
    print(f"  datasets    : {', '.join(keys)}")
    if excluded:
        print(f"  excluded    : {', '.join(excluded)} (evaluation.exclude)")
    print(f"  limit       : {f'{limit:,} clips of each dataset, taken at even intervals within each emotion' if limit else 'every clip'}")
    print(f"  overwrite   : {'yes, evaluate again' if overwrite else 'no, skip datasets that already have results'}")
    for key, run in done.items():
        print(f"  skipped     : {key} (results in {run.relative_to(ROOT)})")
    if not todo:
        print("\nEvery dataset already has results, so only the summary is written. To evaluate again, set "
              "`evaluation.overwrite: true` in config/config.yaml.")
        write_summary(model_name, figure["order"], figure["acted_or_listener"], exclude)
        return
    print(f"  to evaluate : {', '.join(todo)}")

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
    # Step 3: Load the model onto the device (same settings as scripts/emotion2vec/inference.py)
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

    # Steps 4-6 for every dataset to evaluate, then a summary of every dataset with results in outputs/
    for key in todo:
        evaluate(LOADERS[key](), predict, labels, model_name, device, limit, {"funasr": funasr.__version__})
    write_summary(model_name, figure["order"], figure["acted_or_listener"], exclude)


if __name__ == "__main__":
    main()

"""
Evaluate speech emotion recognition models on the speech emotion datasets and write a Markdown report for each.

Runs every chosen model on every clip of every chosen dataset, compares each prediction with the dataset's labels
and writes, per model (<model> is its key in the model registry, config/models.yaml):
    outputs/<model>/<dataset>_predictions.csv          one row per clip (read by scripts/viewer/make_viewer.py)
    outputs/<model>/<dataset>_predictions.json         what the CSV contains, the metrics and the runtime
    reports/<model>/<dataset>.md                       accuracy, per-emotion results, confusion matrices
    reports/<model>/figures/<dataset>_confusion*.png   confusion-matrix figures used in the report
    reports/<model>/summary.md                         every evaluated dataset side by side (.json: the same numbers)
    reports/<model>/figures/summary_confusion.png      every dataset's confusion matrix in one figure
and for every model with results:
    reports/summary.md                                 every model on every dataset (.json: the same numbers)
A sample run (`limit` in config/evaluation.yaml, or the model's own `limit` in config/models.yaml, which wins) adds
_sample<N> to the dataset's file names, e.g. reports/<model>/crema_d_sample14.md.

Each model runs through its loader, scripts/models/<loader>.py: load(model_dir, device) returns a predictor that,
called with an audio file, returns the probabilities of the model's classes; it also gives the classes (labels),
library versions and notes for the report. Clips run one at a time. A loader may also
define TRAINING_NOTE, a sentence added to the training-data caveat of the model's reports, and take a third
argument, `entry`: the model's registry entry, with its key added as entry["key"].
The datasets are defined in scripts/common/eval_datasets.py; download them first with scripts/common/download_data.py
and the models with scripts/common/download_model.py. The local models never touch the network: every one loads
from its folder in models/.

Audio LLMs (`api: true` in the registry: GPT Realtime, Gemini) run through their loaders in the same way, but call
their API for every clip (paid), so they run only with `api_models: true` in config/evaluation.yaml. Instead of
weights, the loader's ready() is checked (e.g. that there is a key). They return a one-hot vector of their answer, so
an answer that is none of the dataset's emotions (or no readable answer) gives a probability of 0 for every one of
them: such a clip counts as wrong and gets its own last column, "other answer", in the confusion matrices. Datasets
whose licence does not allow giving their audio to others (ESD) are never sent to an API.

Usage (from the emotion_recognition/ folder):
    python scripts/evaluation/evaluate.py
Settings (hardware mode, which models and datasets, an optional sample size, devices) are in
config/evaluation.yaml;
the models are registered in config/models.yaml. A model and dataset whose results for the same sample size are
already in outputs/<model>/ are skipped, unless `overwrite` is true; a model whose folder has no weights yet is skipped
with a message. The summaries cover every result in outputs/; datasets listed in `exclude` are never evaluated and
left out of the summaries.

reports/summary.json (fractions, rounded to 6 decimals):
    date             the day it was written
    models           [{model, name, training_source, training_data, summary}]: every model with results, in registry
                     order; summary is the path of its summary report, relative to reports/
    datasets         [{key, dataset, language, chance}]: every dataset with results, in eval_datasets.LOADERS order
    shared_emotions  the emotions every dataset has ([] with a single dataset)
    results          {model: {dataset key: {clips, total_clips, sample, in_training_data, accuracy,
                     unweighted_accuracy, macro_f1, shared_clips, shared_accuracy, shared_unweighted_accuracy}}}:
                     the run shown in the model's summary (the one with the most clips); the shared_* keys only when
                     there are shared emotions
"""
import collections
import csv
import functools
import gc
import importlib
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

# The evaluation never touches the network: every model loads from its own folder in models/, as on a server without
# Hugging Face. Set before a model loader imports transformers or FunASR (a value set in the shell is kept).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "evaluation.yaml"     # this script's settings
MODELS_CONFIG = ROOT / "config" / "models.yaml"  # the model registry
sys.path.insert(0, str(ROOT / "scripts" / "common"))  # eval_datasets.py
sys.path.insert(0, str(ROOT / "scripts" / "models"))  # the model loaders, imported by name (`loader` in the registry)
from eval_datasets import LOADERS, SOURCES, accepted, sample, truth_labels

# What the script reads of config/evaluation.yaml, and of each model in config/models.yaml
SETTINGS = ["mode", "models", "datasets", "exclude", "limit", "overwrite", "devices", "summary_figure"]
REGISTRY = ["name", "loader", "dir", "training_data", "training_source"]  # an API model has api_model instead of dir
OTHER = "other answer"  # confusion-matrix column of the clips whose answer is none of the dataset's emotions (LLMs)
WEIGHTS = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}  # a model folder with such a file has its weights
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


def as_list(value):
    """A setting that is one name or a list of names, as a list."""
    return [value] if isinstance(value, str) else list(value or [])


def unbracketed(text):
    """Text to put in brackets, with its own brackets as commas: "EmoBox (through emotion2vec+ seed)" -> "EmoBox,
    through emotion2vec+ seed"."""
    return re.sub(r"\s*\(([^()]*)\)", r", \1", text)


def training_caveat(key, entry, note):
    """Whether the dataset is in the model's training data (training_data and training_source in the registry), as
    the first caveat of a report; note: the loader's TRAINING_NOTE, if it has one."""
    source = unbracketed(entry["training_source"])
    if entry.get("api") and key not in entry["training_data"]:  # an API model whose training data is not published
        text = f"**{entry['name']}'s training data**: {entry['training_source']}."
    elif key in entry["training_data"]:
        text = (f"**In {entry['name']}'s training data** ({source}), so the model has probably seen these clips; "
                "treat the result as optimistic.")
    else:
        text = f"**Not in {entry['name']}'s training data** ({source})."
    return f"{text} {note}" if note else text


def loader_note(entry):
    """The TRAINING_NOTE of a model's loader (a sentence on its training data for the reports), or ""."""
    if not (ROOT / "scripts" / "models" / f"{entry['loader']}.py").exists():
        return ""
    try:
        return getattr(importlib.import_module(entry["loader"]), "TRAINING_NOTE", "")
    except ImportError as e:  # a library the loader needs is missing; the summary can be written without the note
        print(f"  (scripts/models/{entry['loader']}.py could not be imported: {e})")
        return ""


def not_ready(entry):
    """Why a model cannot run yet (no loader, or no weights in its folder; for an API model: what its loader's ready()
    reports, e.g. no key), or None when it can."""
    loader = ROOT / "scripts" / "models" / f"{entry['loader']}.py"
    if not loader.exists():
        return f"its loader {loader.relative_to(ROOT)} does not exist yet"
    if entry.get("api"):
        try:
            module = importlib.import_module(entry["loader"])
        except ImportError as e:
            return f"its loader {loader.relative_to(ROOT)} could not be imported ({e})"
        return module.ready() if hasattr(module, "ready") else None
    folder = ROOT / entry["dir"]
    files = [p for p in folder.rglob("*") if not any(part.startswith(".") for part in p.relative_to(folder).parts)
             ] if folder.is_dir() else []  # hidden folders such as .cache/ do not count
    if any(p.suffix in WEIGHTS for p in files):
        return None
    if any(p.suffix == ".part000" for p in files):
        return f"its weights in {entry['dir']}/ are still in parts: run python scripts/common/join_parts.py"
    return (f"no weights in {entry['dir']}/ yet: run python scripts/common/download_model.py (or "
            "scripts/common/join_parts.py if the weights came as parts through git)")


@functools.cache
def dataset(key):
    """The EvalSet of a dataset (scripts/common/eval_datasets.py), read once and reused for every model."""
    return LOADERS[key]()


def columns_of(ds, other=False):
    """The columns of a confusion matrix: the dataset's classes, and with other=True a last column OTHER for the
    clips whose answer is none of them (audio LLMs)."""
    return ds.classes + [OTHER] * other


def correct_cells(ds, rows, other=False):
    """Which cells of a confusion matrix count as correct (rows: ground-truth labels, columns: columns_of(ds, other)):
    the column of the row's own label, or for a composite label such as excited every class it accepts."""
    return np.array([[name in accepted(ds, row) for name in ds.classes] + [False] * other for row in rows])


def row_name(ds, label):
    """A row label; a composite label also names the classes it accepts, e.g. "excited (happy or surprised)"."""
    return f"{label} ({' or '.join(ds.accept[label])})" if label in ds.accept else label


def confusion_table(cm, ds, rows, corner, other=False):
    """A confusion matrix as Markdown table rows: row percentages, the cells that count as correct in bold."""
    names, good = columns_of(ds, other), correct_cells(ds, rows, other)
    lines = [f"| {corner} | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k, row in enumerate(rows):
        share = ratio(cm[k], np.full(len(names), cm[k].sum()))
        lines.append(f"| {row_name(ds, row)} ({cm[k].sum():,}) | " + " | ".join(
            f"**{100 * s:.1f}%**" if good[k, j] else f"{100 * s:.1f}%" for j, s in enumerate(share)) + " |")
    return lines


def top_confusions(cm, ds, rows, n=3, other=False):
    """The n largest row shares among the cells that count as wrong, as text such as "fearful predicted as sad
    (14.9%)"."""
    names, good = columns_of(ds, other), correct_cells(ds, rows, other)
    cells = sorted(((cm[r, c] / cm[r].sum(), rows[r], names[c]) for r in range(len(rows))
                    for c in range(len(names)) if not good[r, c] and cm[r].sum()), reverse=True)[:n]
    return ", ".join(f"{a} predicted as {b} ({100 * v:.1f}%)" for v, a, b in cells)


def text_on(rgb):
    """White or ink, whichever has more contrast with the cell colour (WCAG relative luminance)."""
    lum = sum(w * (v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
              for w, v in zip((0.2126, 0.7152, 0.0722), rgb))
    return "#ffffff" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.0533 else INK


def plot_confusion(cm, ds, rows, title, subtitle, ylabel, path, other=False):
    """Row-normalised confusion matrix as a heatmap (rows: ground-truth labels, columns: columns_of(ds, other)); every
    cell shows its percentage, in bold where the prediction counts as correct. In the row of a composite label (e.g.
    JL-Corpus's excited) the cells that count as correct are also outlined, as bold text is hard to see in a light
    cell."""
    import matplotlib
    matplotlib.use("Agg")  # draw straight to a file; works on servers without a display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fonts), "DejaVu Sans")
    names, good = columns_of(ds, other), correct_cells(ds, rows, other)
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
    crowded = len(names) > 6 or other  # with 7 or more emotions, or "other answer", the names would touch: tilt them
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


def evaluate(ds, predict, model, entry, note, device, limit):
    """Steps 4-6 for one model and dataset: load the dataset, run the model on every clip, write the outputs."""
    name, labels = entry["name"], list(predict.labels)
    # ------------------------------------------------------------------
    # Step 4: Load the dataset (scripts/common/eval_datasets.py) and its labels
    # ------------------------------------------------------------------
    step(f"Step 4/8 | {name}: load {ds.name}")
    # limit: clips taken at even intervals, emotion by emotion (in file order within each), so that every emotion
    # is sampled even when the file order repeats (TESS: the 7 emotions of each word in a row)
    clips = sample(ds, limit)
    sampled = len(clips) < len(ds.clips)
    acted = [c["acted"] for c in clips]
    listener = [c["listener"] for c in clips]
    has_listener = any(listener)
    rows = truth_labels(ds)  # the ground-truth labels: the rows of the confusion matrices
    composite = [r for r in rows if r in ds.accept]  # labels that accept several classes, e.g. JL-Corpus's excited
    absent = [n for n in ds.classes if n not in labels]  # emotions of the dataset that the model has no class for
    print(f"  clips       : {len(clips):,}" + (f" of {len(ds.clips):,}, taken at even intervals within each emotion"
                                       if sampled else "")
          + f" ({ds.language})")
    for label in rows:
        print(f"  {label:<11} : {acted.count(label):,} clip{'s' * (acted.count(label) != 1)}"
              + (f" (correct: {' or '.join(ds.accept[label])})" if label in composite else ""))
    if composite:
        print(f"  predictions : {', '.join(ds.classes)}")
    print(f"  listeners   : {'yes' if has_listener else 'no labels in this dataset'}")
    if absent:
        print(f"  no class    : the model has no class for {', '.join(absent)}, so it never predicts "
              f"{'it' if len(absent) == 1 else 'them'}")

    # ------------------------------------------------------------------
    # Step 5: Run the model on every clip (one probability per model class)
    # ------------------------------------------------------------------
    step(f"Step 5/8 | {name}: run inference on {len(clips):,} {ds.name} clips")
    first = np.asarray(predict(str(clips[0]["audio"])))  # warm-up, and a check of what the loader returns
    if first.shape != (len(labels),):
        sys.exit(f"ERROR: scripts/models/{entry['loader']}.py returned probabilities of shape {first.shape} for "
                 f"{len(labels)} labels; it must return one probability per label.")
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
        print(f"  {on_cpu:,} clips ran on the CPU, because {device} could not run them")

    # ------------------------------------------------------------------
    # Step 6: Compare with the labels and write the outputs
    # ------------------------------------------------------------------
    step(f"Step 6/8 | {name}: compute accuracy and write the {ds.name} report")
    names, n_cls, n_all = ds.classes, len(ds.classes), len(labels)  # the classes a prediction is chosen from
    # The model's probability of each of the dataset's emotions (0 for an emotion it has no class for)
    probs = np.column_stack([scores[:, labels.index(n)] if n in labels else np.zeros(len(clips)) for n in names])
    # An API model (audio LLM) whose probability is 0 for every one of the dataset's emotions answered an emotion the
    # dataset does not have, or gave no readable answer (its one-hot vector is elsewhere). Such a clip counts as wrong
    # and gets the last column of the confusion matrices, OTHER, which exists only when there are such clips. Local
    # models keep the most probable of the dataset's emotions, even when all of them are close to 0 (emotion2vec+
    # on TESS with almost all of its probability on `other`), so their results stay exactly as before
    other = ~(probs > 0).any(1) if entry.get("api") else np.zeros(len(clips), dtype=bool)
    has_other = bool(other.any())
    cols = columns_of(ds, has_other)                      # the columns of the confusion matrices
    truth = np.array([rows.index(a) for a in acted])      # each clip's row: its acted emotion
    pred = np.where(other, n_cls, probs.argmax(1))        # most probable of the dataset's emotions, or OTHER
    pred_all = scores.argmax(1)                           # most probable of all the model's classes
    good = correct_cells(ds, rows, has_other)             # rows x columns: True where the prediction counts as correct
    # Two ways to count a prediction as correct (the second only when the dataset has listeners' labels):
    #   strict  - it matches the acted emotion (for a composite label such as excited: one of the classes it accepts)
    #   lenient - it matches the acted emotion OR the listeners' vote (any emotion of a tied vote)
    strict = good[truth, pred]
    heard = [set(v.split(" or ")) if v else set() for v in listener]
    lenient = strict | np.array([cols[p] in h for p, h in zip(pred, heard)])
    cm = np.zeros((len(rows), len(cols)), dtype=int)
    np.add.at(cm, (truth, pred), 1)                       # rows: acted emotion, columns: prediction
    # Lenient matrix: a clip that counts as correct only because of the listeners' vote goes into the row of that vote
    # (its prediction), every other clip into the row of its acted emotion; a class without a row gets one here
    rows_lenient = rows + [n for n in names if n not in rows]
    cm_lenient = np.zeros((len(rows_lenient), len(cols)), dtype=int)
    vote_row = [rows_lenient.index(cols[p]) if cols[p] in rows_lenient else -1 for p in pred]  # -1: OTHER, never used
    np.add.at(cm_lenient, (np.where(lenient & ~strict, vote_row, truth), pred), 1)
    recall = ratio((cm * good).sum(1), cm.sum(1))         # per acted emotion: the share of its clips that is correct
    # Precision, recall and F1 per class compare the prediction with a resolved truth: the prediction when it counts
    # as correct, otherwise the acted emotion, or for a composite label the first class it accepts (so without
    # composite labels, simply the acted emotion)
    resolved = np.where(strict, pred, [names.index(ds.accept.get(a, [a])[0]) for a in acted])
    cm_resolved = np.zeros((n_cls, len(cols)), dtype=int)  # columns: the prediction, OTHER last if there is one
    np.add.at(cm_resolved, (resolved, pred), 1)
    precision = ratio(cm_resolved.diagonal(), cm_resolved[:, :n_cls].sum(0))  # over the dataset's emotions only
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
    acc_all = np.mean([labels[p] in accepted(ds, a) for p, a in zip(pred_all, acted)])
    extra = {lab: int((pred_all == k).sum()) for k, lab in enumerate(labels) if lab not in names}
    # Chance level: a random guess among the classes is right with probability (accepted classes) / (classes)
    chance = np.mean([len(accepted(ds, a) & set(names)) for a in acted]) / n_cls
    # Human reference: how often the listeners' single majority vote matches the acted emotion (ties left out)
    voted = [i for i, v in enumerate(listener) if v and " or " not in v]
    human = np.mean([listener[i] in accepted(ds, acted[i]) for i in voted]) if voted else None
    agree = np.mean([cols[pred[i]] == listener[i] for i in voted]) if voted else None
    if has_other:
        print(f"  other answer                        : {other.sum():,} clips: the answer is none of the dataset's "
              "emotions (or no readable answer); they count as wrong")
    print(f"  accuracy, acted emotion             : {acc:.1%}")
    if has_listener:
        print(f"  accuracy, acted emotion or listener : {acc_lenient:.1%}")
    print(f"  unweighted accuracy (UA)            : {recall.mean():.1%}")
    print(f"  macro F1                            : {macro_f1:.1%}"
          + (f" ({', '.join(unscored)} left out: never predicted, no clip has this label)" if unscored else ""))
    print(f"  {f'accuracy, all {n_all} model classes':<36}: {acc_all:.1%}")
    print(f"  chance level                        : {chance:.1%}")
    if human is not None:
        print(f"  listeners vs acted emotion          : {human:.1%}")

    # Per-clip predictions (CSV) and a description with the metrics (JSON, read by make_viewer.py)
    suffix = f"_sample{len(clips)}" if sampled else ""  # a sample never overwrites the results of a full run
    stem = f"{ds.key}{suffix}"
    predictions = ROOT / "outputs" / model / f"{stem}_predictions.csv"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    prob_columns = [f"p_{lab.strip('<>')}" for lab in labels]  # "<unk>" -> "p_unk"
    with open(predictions, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "acted_emotion"] + ["listener_vote"] * has_listener
                        + ["prediction", "prediction_all_classes", "correct"]
                        + ["correct_acted_or_listener"] * has_listener + prob_columns)
        for i, clip in enumerate(clips):
            writer.writerow([clip["audio"].relative_to(ROOT).as_posix(), acted[i]] + [listener[i]] * has_listener
                            + [cols[pred[i]], labels[pred_all[i]], bool(strict[i])] + [bool(lenient[i])] * has_listener
                            + [f"{p:.4f}" for p in scores[i]])
    metrics = {"accuracy": acc, "unweighted_accuracy": recall.mean(), "macro_f1": macro_f1,
               "accuracy_all_classes": acc_all}
    if has_listener:
        metrics |= {"accuracy_acted_or_listener": acc_lenient, "listeners_vs_acted": human, "model_vs_listeners": agree}
    rule = "; ".join(f"{r}: {' or '.join(ds.accept[r])}" for r in composite)  # e.g. "excited: happy or surprised"
    columns_doc = {
        "audio_path": "audio file, relative to the emotion_recognition/ folder",
        "acted_emotion": f"ground truth: {ds.ground_truth}",
        "listener_vote": f"{ds.listener}; 'a or b' is a tie",
        "prediction": f"model prediction: the most probable of the {n_cls} emotions evaluated ({', '.join(names)})"
                      + (f"; '{OTHER}' when the model's answer is none of them" if has_other else ""),
        "prediction_all_classes": f"model prediction: the most probable of all {n_all} model classes",
        "correct": "whether prediction equals acted_emotion" + (f" or is accepted for it ({rule})" if composite else ""),
        "correct_acted_or_listener": "whether prediction equals acted_emotion or the listener_vote (any emotion of a tie)"
                                     + (" or is accepted for acted_emotion" if composite else ""),
        "p_*": f"model probability of each of its {n_all} classes (they add up to 1)",
    }
    in_training = ds.key in entry["training_data"]
    meta = {
        "dataset": ds.name, "key": ds.key, "language": ds.language, "model": model, "model_name": name,
        "date": str(date.today()), "clips": len(clips), "total_clips": len(ds.clips), "classes": names,
        "labels": rows,       # the ground-truth labels, in the order of the confusion-matrix rows
        "accept": ds.accept,  # composite labels: the classes that count as correct for them
        "chance": round(float(chance), 6),
        "in_training_data": in_training, "training_source": entry["training_source"],  # from the model registry
        "model_classes": labels,  # the model's classes, in the order of the p_ columns
        "metrics": {k: round(float(v), 6) for k, v in metrics.items() if v is not None},
        "ground_truth": "acted_emotion",
        # Correctness columns and what the prediction is compared with (the viewer can switch between them)
        "correct_columns": {"correct": "acted emotion"} | ({"correct_acted_or_listener": "acted emotion or listener vote"}
                                                           if has_listener else {}),
        "columns": {k: v for k, v in columns_doc.items() if has_listener or "listener" not in k},
        "runtime": {"device": str(device), "hardware": hardware(device), "seconds": round(runtime, 1),
                    "ms_per_clip": round(runtime / len(clips) * 1000, 1), "on_cpu": on_cpu},
    }

    # Confusion-matrix figures and the Markdown report
    reports = ROOT / "reports" / model
    figure = reports / "figures" / f"{stem}_confusion.png"
    figure_lenient = reports / "figures" / f"{stem}_confusion_acted_or_listener.png"
    plot_confusion(cm, ds, rows, f"{name} on {ds.name}",
                   (f"Bold = correct ({rule})." if composite else "Correct = matches the acted emotion.")
                   + f" {len(clips):,} clips, accuracy {100 * acc:.1f}%.",
                   "Acted emotion (clips)", figure, has_other)
    if has_listener:
        plot_confusion(cm_lenient, ds, rows_lenient, f"{name} on {ds.name}",
                       f"Correct = matches the acted emotion or the listeners' vote. Accuracy {100 * acc_lenient:.1f}%.",
                       "Acted emotion, or listener vote if matched (clips)", figure_lenient, has_other)
    pct = lambda x: f"{100 * x:.1f}%"
    image = lambda path, alt: f"![{alt}]({path.relative_to(reports).as_posix()})"
    sentence = lambda text: text if text.endswith((".", "!", "?")) else text + "."
    versions = ", ".join(f"{lib} {v}" for lib, v in predict.versions.items())
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
        f"# {name} on {ds.name}",
        "",
        f"Generated by `scripts/evaluation/evaluate.py` on {date.today()}."
        + (f" **Sample**: {len(clips):,} of the {len(ds.clips):,} clips, taken at even intervals within each emotion "
           + ("(the model's `limit` in config/models.yaml)." if "limit" in entry
              else "(`limit` in config/evaluation.yaml).") if sampled else ""),
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
        f"| Accuracy, all {n_all} model classes (acted emotion) | {pct(acc_all)} |",
        f"| Chance level ({n_cls} emotions"
        + "".join(f"; {r} accepts {len(accepted(ds, r) & set(names))}" for r in composite) + f") | {pct(chance)} |",
        *([f"| Listeners' majority vote vs acted emotion | {pct(human)} |"] if human is not None else []),
        "",
        "## Setup",
        "",
        " ".join([f"- **Model**: {name} (`{entry.get('hf_repo', entry.get('api_model', model))}`) as released (no "
                  "fine-tuning)" + (f", {versions}" if versions else "")
                  + f", {'called through' if entry.get('api') else 'loaded by'} `scripts/models/{entry['loader']}.py`.",
                  *(sentence(n) for n in predict.notes)]),
        f"- **Data**: {len(clips):,} clips of {ds.name} ({ds.language}) from `data/{ds.key}/`"
        + (f", a sample of its {len(ds.clips):,} clips taken at even intervals within each emotion." if sampled else "."),
        f"- **Ground truth (acted emotion)**: {ds.ground_truth}.",
        *([f"- **Listeners' vote**: {ds.listener}."] if has_listener else []),
        *[f"- {n}" for n in ds.notes],
        (f"- **{n_cls} emotions**: the prediction is the model's answer when it is one of the {n_cls} emotions in "
         f"this dataset ({', '.join(names)}); any other answer ({', '.join(f'`{lab}`' for lab in extra)}) counts as "
         f"wrong, so the accuracy is the same with all {n_all} model classes." if entry.get("api") else
         f"- **{n_cls} emotions**: the prediction is the most probable of the {n_cls} emotions in this dataset "
         f"({', '.join(names)})"
         + (f"; the model's other classes ({', '.join(f'`{lab}`' for lab in extra)}) are ignored." if extra
            else "; the model has no other classes.")
         + f" **All {n_all} model classes**: the prediction is the most probable of all {n_all} model classes, so "
         "choosing a class that the dataset does not have counts as wrong."),
        *([f"- **Other answer**: for {other.sum():,} clips the model's answer is none of the {n_cls} emotions (an "
           "emotion this dataset does not have, or no readable answer). They count as wrong and have their own last "
           f"column, *{OTHER}*, in the confusion matrices; precision, recall and F1 are over the {n_cls} emotions."]
          if has_other else []),
        *([f"- **No model class for {' and '.join(absent)}**: the model cannot predict "
           f"{'this emotion' if len(absent) == 1 else 'these emotions'}, so every clip acted "
           f"{' or '.join(absent)} counts as wrong."] if absent else []),
        *([f"- **Composite label{'s' if len(composite) > 1 else ''}**: "
           + "; ".join(f"*{r}* has no model class, so a prediction of {' or '.join(ds.accept[r])} counts as correct for "
                       "it" for r in composite)
           + f" (also with all {n_all} model classes). Each such label has its own row in the confusion matrices, whose "
           "bold cells count as correct; its recall is the share of its clips predicted correctly, and chance level "
           f"({pct(chance)} instead of {pct(1 / n_cls)}) counts every class it accepts."] if composite else []),
        *(["- **Two ways to count a prediction as correct.** *Acted emotion*: it must match the acted emotion. *Acted "
           "emotion or listeners' vote*: it also counts when it matches the listeners' majority vote (for a tied vote, "
           "any of the tied emotions). Both are reported."] if has_listener else []),
        (f"- **API**: one request at a time from {hardware(device)}. The run took "
         f"{f'{runtime:.0f} s' if runtime < 60 else f'{runtime / 60:.1f} min'} ({runtime / len(clips) * 1000:.0f} ms "
         "per clip; answers saved by an earlier run take no time)." if entry.get("api") else
         f"- **Hardware**: {hardware(device)}, device `{device}`, PyTorch {torch.__version__}. "
         f"Inference took {f'{runtime:.0f} s' if runtime < 60 else f'{runtime / 60:.1f} min'} "
         f"({runtime / len(clips) * 1000:.0f} ms per clip)."
         + (f" {on_cpu:,} clips ran on the CPU, because `{device}` could not process them." if on_cpu else "")),
        f"- **Per-clip predictions**: `{predictions.relative_to(ROOT)}` (in git, so the results can be checked clip by clip). "
        "The JSON file next to it describes the columns.",
        "",
        "## Results per emotion",
        "",
        *per_emotion,
        "",
        "## Confusion matrix: acted emotion",
        "",
        image(figure, f"Confusion matrix of {name} on {ds.name}, acted emotion"),
        "",
        "The same numbers as a table. Rows: acted emotion. Columns: predicted emotion. Each row adds up to 100%."
        + (" Bold: the cells that count as correct." if composite else "")
        + (f" *{OTHER}*: the answer is none of the dataset's emotions (wrong)." if has_other else ""),
        "",
        *confusion_table(cm, ds, rows, "Acted \\ Predicted", has_other),
        "",
        f"Most frequent confusions: {top_confusions(cm, ds, rows, other=has_other)}.",
        "",
    ]
    if has_listener:
        report += [
            "## Confusion matrix: acted emotion or listeners' vote",
            "",
            image(figure_lenient, f"Confusion matrix of {name} on {ds.name}, acted emotion or listeners' vote"),
            "",
            "Here a prediction is correct when it matches the acted emotion or the listeners' vote. A clip goes into "
            "the row of the label its prediction matched, otherwise into the row of its acted emotion. "
            + ("The bold cells therefore hold" if composite else "The diagonal therefore holds")
            + " every correct prediction, and the rows differ in size from the matrix above. "
            f"{sum(rescued.values()):,} clips ({pct(sum(rescued.values()) / len(clips))}) count as correct only because "
            "of the listeners' vote" + ("; the most common are " + ", ".join(
                f"acted {a} predicted as {p} ({n:,})" for (a, p), n in rescued.most_common(3)) if rescued else "") + ".",
            "",
            *confusion_table(cm_lenient, ds, rows_lenient, "Ground truth \\ Predicted", has_other),
            "",
            f"Most frequent confusions: {top_confusions(cm_lenient, ds, rows_lenient, other=has_other)}.",
            "",
        ]
    if extra:
        report += [
            f"## Predictions outside the {n_cls} emotions",
            "",
            f"With all {n_all} model classes, the model chose a class that this dataset does not have for "
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
    report += ["## Caveats", "", f"- {training_caveat(ds.key, entry, note)}", *[f"- {c}" for c in ds.caveats]]
    if has_listener:
        report.append("- **Acted labels are not perceived emotions.** The ground truth is what the actor intended; "
                      "listeners sometimes heard something else, as the listeners' numbers above show.")
    report.append("")
    out = reports / f"{stem}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    # Written last, so that it exists only for a finished run (see finished_run)
    predictions.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nDone: report -> {out.relative_to(ROOT)}, predictions -> {predictions.relative_to(ROOT)}")


def best_runs(model, exclude=()):
    """Per dataset key, in eval_datasets.LOADERS order: the predictions JSON of the run in outputs/<model>/ with the
    most clips (a full run rather than a sample), with its file stem added. Datasets in exclude (`exclude` in
    config/evaluation.yaml) are left out."""
    out, order = ROOT / "outputs" / model, list(LOADERS)
    best = {}
    for p in sorted(out.glob("*_predictions.json")):
        run = json.loads(p.read_text(encoding="utf-8")) | {"stem": p.name.removesuffix("_predictions.json")}
        if "metrics" in run and run["key"] not in exclude and run["clips"] > best.get(run["key"], {"clips": 0})["clips"]:
            best[run["key"]] = run
    # Ground-truth labels in the order of the confusion-matrix rows. Older predictions files have no "labels": their
    # classes and composite labels stand in, those that clips in the CSV have (JL-Corpus has no clip acted surprised)
    for r in best.values():
        if "labels" not in r:
            with open(out / f"{r['stem']}_predictions.csv", newline="", encoding="utf-8") as fh:
                found = {row[r["ground_truth"]] for row in csv.DictReader(fh)}
            r["labels"] = [x for x in [*r["classes"], *r.get("accept", {})] if x in found]
    return dict(sorted(best.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order)))


def shared_scores(model, run, shared):
    """Clips, accuracy and unweighted accuracy of one run on the emotions every dataset has: clips of other emotions
    are left out, and the prediction is the most probable of the shared emotions."""
    with open(ROOT / "outputs" / model / f"{run['stem']}_predictions.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["acted_emotion"] in shared]
    if not rows:  # a small sample can miss them
        return 0, None, None
    pred = []
    for r in rows:
        if r["prediction"] in shared:  # most probable of the dataset's emotions, so also of the shared ones
            pred.append(r["prediction"])
        else:  # use the probabilities in the file; they have 4 decimals, so a tie counts as wrong
            p = [float(r.get(f"p_{e}", 0)) for e in shared]  # 0 for an emotion the model has no class for
            pred.append(shared[p.index(max(p))] if p.count(max(p)) == 1 else None)
    truth = np.array([r["acted_emotion"] for r in rows])
    hit = np.array(pred) == truth
    return len(rows), hit.mean(), np.mean([hit[truth == e].mean() for e in shared if (truth == e).any()])


def write_summary(model, entry, note, figure_order=(), acted_or_listener=(), exclude=()):
    """reports/<model>/summary.md: one row per evaluated dataset, read from the JSON files in outputs/<model>/. A
    dataset with a full run and a sample (`limit`) is shown with the run that has the most clips. The datasets in the
    model's training data come first, then the others, each group sorted by accuracy. The summary figure shows the
    datasets in figure_order first (the others follow in eval_datasets.LOADERS order); those in acted_or_listener show
    their "acted emotion or listeners' vote" confusion matrix. Datasets in exclude are left out. figure_order and
    acted_or_listener come from `summary_figure`, exclude from `exclude`, in config/evaluation.yaml."""
    runs = list(best_runs(model, exclude).values())
    if not runs:
        print(f"  no results in outputs/{model}/ yet, so no summary")
        return
    name, trained = entry["name"], set(entry["training_data"])
    reports = ROOT / "reports" / model
    sampled = [r for r in runs if r.get("total_clips", r["clips"]) > r["clips"]]
    pct = lambda x: f"{100 * x:.1f}%" if x is not None else "–"
    clips = lambda r: f"{r['clips']:,}" + (f" of {r['total_clips']:,} (sample)" if r in sampled else "")
    chance = lambda r: r.get("chance", 1 / len(r["classes"]))  # older predictions files have no "chance"
    yes_no = lambda r: "yes" if r["key"] in trained else "not known" if entry.get("api") else "no"
    # The datasets in the model's training data first, then the others, each group by accuracy (best first)
    ranked = sorted(runs, key=lambda r: (r["key"] not in trained, -r["metrics"]["accuracy"]))

    def labels(r):
        """The labels as text; a composite label names the classes it accepts, e.g. "excited (happy or surprised)".
        No-break spaces keep three labels together: GitHub squeezes a wide table to its narrowest, and the column then
        wraps between such groups instead of after every label."""
        names = [f"{x} ({' or '.join(r['accept'][x])})" if x in r.get("accept", {}) else x for x in r["labels"]]
        return ", ".join(", ".join(names[k:k + 3]) for k in range(0, len(names), 3))
    # Composite labels, e.g. "JL-Corpus's *excited* (happy or surprised)"
    composite = [f"{r['dataset']}'s *{label}* ({' or '.join(accept)})" for r in runs
                 for label, accept in r.get("accept", {}).items()]
    column = f"In {name}'s training data"
    rows = [f"| [{r['dataset']}]({r['stem']}.md) | {r['language']} | {labels(r)} | {yes_no(r)} | {clips(r)} | "
            f"{len(r['classes'])} | {pct(chance(r))} | **{pct(r['metrics']['accuracy'])}** | "
            f"{pct(r['metrics'].get('accuracy_acted_or_listener'))} | {pct(r['metrics']['unweighted_accuracy'])} | "
            f"{pct(r['metrics']['macro_f1'])} |" for r in ranked]
    summary = [
        f"# {name}: results on every dataset",
        "",
        f"Generated by `scripts/evaluation/evaluate.py` on {date.today()}. Each dataset links to its full report. The "
        "datasets in the model's training data come first, then the others, each group sorted by accuracy.",
        "",
        f"| Dataset | Language | Ground-truth labels | {column} | Clips | Emotions | Chance "
        "| Accuracy: acted emotion | Accuracy: acted or listeners' vote | Unweighted accuracy | Macro F1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "- **Ground-truth labels**: the emotions the speakers were asked to portray, as the project's class names (each "
        "report gives the dataset's own names), in the order of the rows of the confusion matrices.",
        *([f"- **Composite labels** (no model class; a prediction of any class in brackets counts as correct): "
           f"{', '.join(composite)}. Chance level counts these classes, so it is higher than 1 in the number of "
           "emotions."] if composite else []),
        f"- **{column}**: *yes* means that the dataset is part of the data the model was trained on "
        f"({unbracketed(entry['training_source'])}), so the model has probably seen these clips and the result is "
        "optimistic. " + ("*not known*: the training data is not published." if entry.get("api") else "*no*: it is not.")
        + (f" {note}" if note else ""),
        "- **Accuracy: acted emotion** compares the prediction with the emotion the speaker was asked to portray. "
        "The prediction is the most probable of the emotions that the dataset has, so chance level depends on how "
        "many emotions it has.",
        "- **Accuracy: acted or listeners' vote** also accepts the listeners' majority vote; it exists only for "
        "datasets that publish listeners' labels.",
        *(["- **(sample)**: evaluated on clips taken at even intervals within each emotion (`limit` in "
           "config/evaluation.yaml), not on every clip."]
          if sampled else []),
        "",
    ]
    # Every dataset's confusion matrix (the figures of the reports) stitched into one figure
    figures = reports / "figures"
    rank = {key: k for k, key in enumerate(figure_order)}
    shown = [r for r in sorted(runs, key=lambda r: rank.get(r["key"], len(rank)))
             if (figures / f"{r['stem']}_confusion.png").exists()]
    lenient = [r for r in shown if r["key"] in acted_or_listener
               and (figures / f"{r['stem']}_confusion_acted_or_listener.png").exists()]
    panels = [figures / f"{r['stem']}_confusion{'_acted_or_listener' if r in lenient else ''}.png" for r in shown]
    grid = figures / "summary_confusion.png" if panels else None
    if panels:
        stitch(panels, grid, columns=min(3, math.ceil(math.sqrt(len(panels)))))  # 4 panels: 2 x 2, 6: 3 x 2
        summary += [
            "## Confusion matrices",
            "",
            f"![Confusion matrices of {name} on every dataset]({grid.relative_to(reports).as_posix()})",
            "",
            f"One matrix per dataset, from each dataset's report: {', '.join(r['dataset'] for r in shown)} (order: "
            "`summary_figure` in config/evaluation.yaml). Rows: acted emotion; columns: predicted emotion; each "
            "row adds up to 100%."
            + (f" {', '.join(r['dataset'] for r in lenient)}: the matrix where a prediction also counts as correct when "
               "it matches the listeners' vote, so a clip whose prediction matched that vote sits in the row of that "
               "vote." if lenient else "")
            + (" A composite label has its own row, whose bold cells count as correct." if composite else ""),
            "",
        ]
    # The same comparison on the emotions that every dataset has, so that the choice of emotions does not matter
    shared = sorted(set.intersection(*(set(r["classes"]) for r in runs))) if len(runs) > 1 else []
    scored = sorted(((r, *shared_scores(model, r, shared)) for r in runs),
                    key=lambda s: (s[0]["key"] not in trained, -(s[2] or 0))) if shared else []
    if shared:
        summary += [
            f"## Only the emotions every dataset has: {', '.join(shared)}",
            "",
            "The datasets have different emotions, and some emotions are much harder than others, so the table above "
            f"also reflects which emotions each dataset has. Here every dataset is scored on the same {len(shared)} "
            f"emotions (chance {pct(1 / len(shared))}): clips of other emotions are left out, and the prediction is "
            f"the most probable of these {len(shared)}.",
            "",
            f"| Dataset | Language | Ground-truth labels | {column} | Clips | Accuracy | Unweighted accuracy |",
            "|---|---|---|---|---|---|---|",
            *[f"| [{r['dataset']}]({r['stem']}.md) | {r['language']} | {labels(r)} | {yes_no(r)} | "
              f"{n:,}{' (sample)' if r in sampled else ''} | **{pct(acc)}** | {pct(ua)} |" for r, n, acc, ua in scored],
            "",
            "Computed from the predictions files in `outputs/`. Their probabilities have 4 decimals, so a clip whose "
            "shared emotions are tied counts as wrong.",
            "",
        ]
    path = reports / "summary.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    # The same numbers as JSON, for the viewer page (scripts/viewer/make_viewer.py)
    rounded = lambda x: round(float(x), 6) if x is not None else None
    data = {
        "model": model, "name": name, "date": str(date.today()), "training_source": entry["training_source"],
        "datasets": [{"dataset": r["dataset"], "key": r["key"], "language": r["language"], "clips": r["clips"],
                      "total_clips": r.get("total_clips", r["clips"]), "emotions": len(r["classes"]),
                      "labels": r["labels"],  # ground-truth labels, in the order of the confusion-matrix rows
                      "chance": round(float(chance(r)), 6), **({"accept": r["accept"]} if r.get("accept") else {}),
                      "in_training_data": r["key"] in trained, "metrics": r["metrics"]} for r in ranked],
        "figure": grid.relative_to(reports).as_posix() if grid else None,
        "figure_datasets": [r["dataset"] for r in shown] if grid else [],
        "acted_or_listener": [r["dataset"] for r in lenient] if grid else [],
        "shared_emotions": shared,
        "shared": [{"dataset": r["dataset"], "key": r["key"], "language": r["language"], "clips": n,
                    "sample": r in sampled, "accuracy": rounded(acc), "unweighted_accuracy": rounded(ua),
                    "in_training_data": r["key"] in trained} for r, n, acc, ua in scored],
    }
    path.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  summary of {len(runs)} dataset(s) -> {path.relative_to(ROOT)} (and .json)")


def write_comparison(registry, exclude=()):
    """reports/summary.md (and .json, see the docstring of this file): every registered model with results in
    outputs/<model>/ on every dataset, with the same run per model and dataset as the model's summary (the one with
    the most clips). Rows: datasets, in eval_datasets.LOADERS order; columns: models, in registry order."""
    runs = {m: best_runs(m, exclude) for m in registry}
    models = [m for m in registry if runs[m]]
    if not models:
        print("  no results in outputs/ yet, so no comparison")
        return
    keys = [k for k in LOADERS if any(k in runs[m] for m in models)]
    facts = {k: max((runs[m][k] for m in models if k in runs[m]), key=lambda r: r["clips"]) for k in keys}  # name, ...
    shared = sorted(set.intersection(*(set(facts[k]["classes"]) for k in keys))) if len(keys) > 1 else []
    values = {}  # (model, dataset) -> metric -> value
    for m in models:
        for k, r in runs[m].items():
            values[m, k] = dict(r["metrics"])
            if shared:
                values[m, k] |= dict(zip(("shared_clips", "shared_accuracy", "shared_unweighted_accuracy"),
                                         shared_scores(m, r, shared)))
    sampled = lambda r: r.get("total_clips", r["clips"]) > r["clips"]
    trained = lambda m, k: k in registry[m]["training_data"]
    chance = lambda r: r.get("chance", 1 / len(r["classes"]))  # older predictions files have no "chance"
    pct = lambda x: f"{100 * x:.1f}%"
    samples = any(sampled(r) for m in models for r in runs[m].values())

    def table(metric, chance_of):
        """Rows: datasets; columns: models. The best value of a row in bold, † where the dataset is in the model's
        training data, and the number of clips of a sample."""
        lines = ["| Dataset | Language | Chance | " + " | ".join(f"[{registry[m]['name']}]({m}/summary.md)"
                                                          for m in models) + " |",
                 "|---|---|---|" + "---|" * len(models)]
        for k in keys:
            got = {m: values[m, k][metric] for m in models if values.get((m, k), {}).get(metric) is not None}
            best = max((round(100 * v, 1) for v in got.values()), default=None)  # compared as shown, to 0.1%
            cells = []
            for m in models:
                if m not in got:
                    cells.append("–")
                    continue
                cell = f"**{pct(got[m])}**" if len(got) > 1 and round(100 * got[m], 1) == best else pct(got[m])
                cells.append(cell + "†" * trained(m, k)
                             + (f" ({runs[m][k]['clips']:,} clips)" if sampled(runs[m][k]) else ""))
            lines.append(f"| {facts[k]['dataset']} | {facts[k]['language']} | {pct(chance_of(k))} | "
                         + " | ".join(cells) + " |")
        return lines
    dataset_chance = lambda k: chance(facts[k])
    summary = [
        "# Every model on every dataset",
        "",
        f"Generated by `scripts/evaluation/evaluate.py` on {date.today()}. Every model runs as released on the same "
        "clips and is scored against the same labels. Each model's name links to its summary, which links to a report "
        "per dataset.",
        "",
        "- **†**: the dataset is in the model's training data, so the model has probably seen these clips; treat the "
        "result as optimistic.",
        "- **Bold**: the best model on the dataset.",
        *(["- **(N clips)**: a sample of the dataset (`limit` in config/evaluation.yaml), not every clip."]
          if samples else []),
        "",
        "## Accuracy: prediction matches the acted emotion",
        "",
        "The prediction is the most probable of the dataset's emotions, so chance level depends on how many emotions "
        "the dataset has.",
        "",
        *table("accuracy", dataset_chance),
        "",
        "## Unweighted accuracy",
        "",
        "The mean recall of the acted emotions, so every emotion counts the same, however many clips it has.",
        "",
        *table("unweighted_accuracy", dataset_chance),
        "",
        "## Macro F1",
        "",
        "The mean F1 of the predicted emotions (each report explains how a composite label such as JL-Corpus's "
        "*excited* counts).",
        "",
        *table("macro_f1", dataset_chance),
        "",
    ]
    if shared:
        summary += [
            f"## Only the emotions every dataset has: {', '.join(shared)}",
            "",
            f"Accuracy with every dataset scored on the same {len(shared)} emotions: clips of other emotions are left "
            f"out, and the prediction is the most probable of these {len(shared)}, so the choice of emotions does not "
            "matter. Computed from the predictions files in `outputs/`.",
            "",
            *table("shared_accuracy", lambda k: 1 / len(shared)),
            "",
        ]
    summary += [
        "## Training data",
        "",
        *[f"- [{registry[m]['name']}]({m}/summary.md): {registry[m]['training_source']}. "
          + (f"In its training data: {', '.join(facts[k]['dataset'] for k in keys if trained(m, k))}."
             if any(trained(m, k) for k in keys) else "Whether these datasets are in its training data is not known."
             if registry[m].get("api") else "None of these datasets is in its training data.")
          for m in models],
        "",
    ]
    path = ROOT / "reports" / "summary.md"
    path.write_text("\n".join(summary), encoding="utf-8")
    rounded = lambda x: round(float(x), 6) if x is not None else None
    data = {
        "date": str(date.today()),
        "models": [{"model": m, "name": registry[m]["name"], "training_source": registry[m]["training_source"],
                    "training_data": registry[m]["training_data"], "summary": f"{m}/summary.md"} for m in models],
        "datasets": [{"key": k, "dataset": facts[k]["dataset"], "language": facts[k]["language"],
                      "chance": rounded(chance(facts[k]))} for k in keys],
        "shared_emotions": shared,
        "results": {m: {k: {"clips": r["clips"], "total_clips": r.get("total_clips", r["clips"]), "sample": sampled(r),
                            "in_training_data": trained(m, k),
                            **{x: (v if x == "shared_clips" else rounded(v)) for x, v in values[m, k].items()
                               if x in ("accuracy", "unweighted_accuracy", "macro_f1", "shared_clips",
                                        "shared_accuracy", "shared_unweighted_accuracy")}}
                        for k, r in runs[m].items()} for m in models},
    }
    path.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  {len(models)} model(s) on {len(keys)} dataset(s) -> {path.relative_to(ROOT)} (and .json)")


def finished_run(model, key, limit):
    """The predictions JSON of an earlier finished run that a run with these settings would replace, or None.
    With `limit` smaller than the dataset, a run writes outputs/<model>/<key>_sample<limit>_*, otherwise
    outputs/<model>/<key>_*. The JSON is written last, so it exists only for a finished run."""
    out = ROOT / "outputs" / model
    sample = out / f"{key}_sample{limit}_predictions.json"
    full = out / f"{key}_predictions.json"
    if limit and sample.exists():
        return sample
    # A full run also counts when the limit is at least the size of the dataset (then there is no sample)
    if full.exists() and (not limit or json.loads(full.read_text(encoding="utf-8"))["clips"] <= limit):
        return full
    return None


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the settings (config/evaluation.yaml) and the model registry (config/models.yaml)
    # ------------------------------------------------------------------
    step("Step 1/8 | Load configuration")
    settings = yaml.safe_load(CONFIG.read_text()) or {}
    registry = yaml.safe_load(MODELS_CONFIG.read_text()) or {}
    missing = [k for k in SETTINGS if k not in settings]
    if missing:
        sys.exit(f"ERROR: config/evaluation.yaml lacks {', '.join(missing)}.")
    # The hardware mode: auto = cuda when PyTorch sees an NVIDIA GPU, otherwise mac
    mode, why = settings["mode"], ""
    if mode == "auto":
        mode = "cuda" if torch.cuda.is_available() else "mac"
        why = " (auto: CUDA is available)" if mode == "cuda" else " (auto: CUDA is not available)"
    if mode not in settings["devices"]:
        sys.exit(f"ERROR: mode {mode} has no device in `devices` of config/evaluation.yaml "
                 f"(it has: {', '.join(settings['devices'])}).")
    for m, entry in registry.items():
        needed = [k if k != "dir" or not entry.get("api") else "api_model" for k in REGISTRY]  # API: no folder
        lacks = [k for k in needed if k not in entry]
        if lacks:
            sys.exit(f"ERROR: model {m} in config/models.yaml lacks {', '.join(lacks)}.")
    # API models (api: true) are paid, so they run only with `api_models: true`, also when `models` lists them
    api_on = settings.get("api_models") is True
    models = list(registry) if settings["models"] == "all" else as_list(settings["models"])
    unknown = [m for m in models if m not in registry]
    if unknown or not models:
        sys.exit(f"ERROR: unknown model(s) {unknown} in `models` of config/evaluation.yaml; use all or "
                 f"models registered in config/models.yaml: {', '.join(registry)}")
    paid = [m for m in models if registry[m].get("api")]
    if paid and not api_on:
        models = [m for m in models if m not in paid]
        print(f"  API models  : {', '.join(paid)} skipped: they call a paid API; set `api_models: true` in "
              "config/evaluation.yaml to run them")
        if not models:
            sys.exit("ERROR: no model left to evaluate: every model in `models` is an API model and `api_models` "
                     "is false in config/evaluation.yaml.")
    wanted, exclude = settings["datasets"], as_list(settings["exclude"])
    keys = [k for k in LOADERS if (ROOT / "data" / k).is_dir()] if wanted == "all" else as_list(wanted)
    figure = settings["summary_figure"] or {}
    order, acted_or_listener = as_list(figure.get("order")), as_list(figure.get("acted_or_listener"))
    named = [*keys, *exclude, *order, *acted_or_listener,
             *(k for m in models for k in registry[m]["training_data"])]
    unknown = sorted({k for k in named if k not in LOADERS})
    if unknown:
        sys.exit(f"ERROR: unknown dataset(s) {unknown} in config/evaluation.yaml (or in training_data in "
                 f"config/models.yaml); use: {', '.join(LOADERS)} or all")
    limit, overwrite = settings["limit"], settings["overwrite"]
    if limit is not None and (not isinstance(limit, int) or limit < 1):
        sys.exit(f"ERROR: `limit` in config/evaluation.yaml must be a positive number or null, not {limit!r}.")
    # Each model's limit: its own `limit` in config/models.yaml, otherwise the one above
    limits = {m: registry[m].get("limit", limit) for m in models}
    for m, lim in limits.items():
        if lim is not None and (not isinstance(lim, int) or lim < 1):
            sys.exit(f"ERROR: `limit` of {m} in config/models.yaml must be a positive number or null, not {lim!r}.")
    # Datasets in exclude are never evaluated (their data can stay in data/)
    excluded = [k for k in keys if k in exclude]
    keys = [k for k in keys if k not in exclude]
    if excluded and not keys:
        sys.exit(f"ERROR: every dataset to evaluate is in `exclude` in config/evaluation.yaml ({', '.join(excluded)}).")
    # Each model's device: devices for the mode, or on the Mac the model's own mac_device (config/models.yaml)
    devices = {}
    for m in models:
        if registry[m].get("api"):  # runs at the API provider; nothing runs on a device here
            devices[m] = torch.device("cpu")
            continue
        wish = settings["devices"][mode]
        if mode == "mac" and registry[m].get("mac_device"):
            wish = registry[m]["mac_device"]
        try:
            devices[m] = torch.device(wish)
        except RuntimeError:
            sys.exit(f"ERROR: unknown device {wish!r} for {m} (`devices` in config/evaluation.yaml, or its mac_device "
                     "in config/models.yaml); use cpu, mps or cuda:<index>.")
    # Models and datasets that already have results for this limit in outputs/<model>/ are skipped, unless overwrite
    # is true; so is a model that cannot run yet (no loader or no weights)
    done = {m: {k: run for k in keys if not overwrite and (run := finished_run(m, k, limits[m]))} for m in models}
    todo = {m: [k for k in keys if k not in done[m]] for m in models}
    # An API model is never given a dataset whose licence does not allow giving its audio to others (ESD)
    private = {m: [k for k in todo[m] if not SOURCES.get(k, {}).get("public_audio")]
               for m in models if registry[m].get("api")}
    todo = {m: [k for k in ks if k not in private.get(m, [])] for m, ks in todo.items()}
    waiting = {m: reason for m in models if todo[m] and (reason := not_ready(registry[m]))}
    todo = {m: ks for m, ks in todo.items() if ks and m not in waiting}
    missing = [k for k in keys if any(k in ks for ks in todo.values()) and not (ROOT / "data" / k).is_dir()]
    if not keys or missing:
        sys.exit(f"ERROR: not downloaded: {', '.join(missing) or 'any dataset'}. Download it with "
                 "scripts/common/download_data.py (set `dataset` in config/download_data.yaml).")
    print(f"  config      : {CONFIG}")
    print(f"  registry    : {MODELS_CONFIG}")
    print(f"  mode        : {mode}{why}")
    for m in models:
        where = (f"API model {registry[m]['api_model']}" if registry[m].get("api")
                 else f"device {devices[m]} ({registry[m]['dir']})")
        print(f"  model       : {m} = {registry[m]['name']}, {where}")
    print(f"  datasets    : {', '.join(keys)}")
    if excluded:
        print(f"  excluded    : {', '.join(excluded)} (exclude in config/evaluation.yaml)")
    print(f"  limit       : {f'{limit:,} clips of each dataset, taken at even intervals within each emotion' if limit else 'every clip'}")
    for m, lim in limits.items():
        if lim != limit:
            print(f"  limit       : {m}: {f'{lim:,} clips of each dataset' if lim else 'every clip'} (its own `limit` "
                  "in config/models.yaml)")
    for m, ks in private.items():
        if ks:
            print(f"  never sent  : {', '.join(ks)} to {m}: the licence does not allow giving the audio to others")
    print(f"  overwrite   : {'yes, evaluate again' if overwrite else 'no, skip what already has results'}")
    for m in models:
        if done[m]:
            print(f"  skipped     : {m} on {', '.join(done[m])} (results in outputs/{m}/)")
    for m, reason in waiting.items():
        print(f"  not ready   : {m}: {reason}")
    for m, ks in todo.items():
        print(f"  to evaluate : {m} on {', '.join(ks)}")
    if not todo:
        print("\nNothing to evaluate, so only the summaries are written. To evaluate again, set "
              "`overwrite: true` in config/evaluation.yaml.")
    else:
        # ------------------------------------------------------------------
        # Step 2: Check the software stack and the devices
        # ------------------------------------------------------------------
        step("Step 2/8 | Check environment and devices")
        print(f"  Python      : {platform.python_version()} ({platform.system()} {platform.machine()})")
        print(f"  PyTorch     : {torch.__version__}")
        print(f"  offline     : HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']}, "
              f"TRANSFORMERS_OFFLINE={os.environ['TRANSFORMERS_OFFLINE']} (models load from models/ only)")
        for dev in dict.fromkeys(devices[m] for m in todo if not registry[m].get("api")):
            if dev.type == "cuda" and not torch.cuda.is_available():
                sys.exit("ERROR: CUDA was requested but is not available (see README, Troubleshooting).")
            if dev.type == "mps" and not torch.backends.mps.is_available():
                sys.exit("ERROR: MPS was requested but is not available (see README, Troubleshooting).")
            users = ", ".join(m for m in todo if devices[m] == dev and not registry[m].get("api"))
            print(f"  device      : {dev} = {hardware(dev)}"
                  + (f", {torch.get_num_threads()} threads" if dev.type == "cpu" else "") + f" (for {users})")

    # Steps 3-7 for every model to evaluate: load it, run it on its datasets, write its summary, free its memory
    for i, (m, ks) in enumerate(todo.items(), 1):
        entry = registry[m]
        # ------------------------------------------------------------------
        # Step 3: Load the model through its loader (scripts/models/<loader>.py), from its folder only
        # ------------------------------------------------------------------
        step(f"Step 3/8 | Load {entry['name']} (model {i} of {len(todo)})")
        loader = importlib.import_module(entry["loader"])
        t0 = time.perf_counter()
        folder = ROOT / entry["dir"] if "dir" in entry else None  # an API model has no folder
        # A loader that takes `entry` gets the model's registry entry, with its key (API models: api_model, cache)
        if "entry" in inspect.signature(loader.load).parameters:
            predict = loader.load(folder, devices[m], entry=entry | {"key": m})
        else:
            predict = loader.load(folder, devices[m])
        labels = list(predict.labels)
        if len(set(labels)) != len(labels):
            sys.exit(f"ERROR: scripts/models/{entry['loader']}.py gives repeated labels: {labels}")
        print(f"  loader      : scripts/models/{entry['loader']}.py")
        print(f"  folder      : {entry['dir']}" if folder else f"  API model   : {entry['api_model']}")
        print(f"  device      : {devices[m]}" if folder else "  device      : none here (runs at the API provider)")
        print(f"  limit       : {f'{limits[m]:,} clips of each dataset' if limits[m] else 'every clip'}")
        print(f"  versions    : {', '.join(f'{lib} {v}' for lib, v in predict.versions.items())}")
        print(f"  loaded in   : {time.perf_counter() - t0:.1f} s")
        print(f"  classes     : {len(labels)} -> {', '.join(labels)}")
        for n in predict.notes:
            print(f"  note        : {n}")
        note = getattr(loader, "TRAINING_NOTE", "")
        # Steps 4-6 for every dataset to evaluate
        for key in ks:
            evaluate(dataset(key), predict, m, entry, note, devices[m], limits[m])
        # ------------------------------------------------------------------
        # Step 7: The model's summary: every dataset with results in outputs/<model>/
        # ------------------------------------------------------------------
        step(f"Step 7/8 | Summary of {entry['name']}")
        write_summary(m, entry, note, order, acted_or_listener, exclude)
        # Free the model's memory before the next one
        del predict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    # The summaries of the chosen models that had nothing to evaluate (their results may still have changed)
    for m in models:
        if m not in todo:
            step(f"Step 7/8 | Summary of {registry[m]['name']}")
            write_summary(m, registry[m], loader_note(registry[m]), order, acted_or_listener, exclude)

    # ------------------------------------------------------------------
    # Step 8: Compare every model with results in outputs/ (also those not chosen this time)
    # ------------------------------------------------------------------
    step("Step 8/8 | Summary of every model")
    write_comparison(registry, exclude)

    # Bar plots of every model on every dataset (a separate script, scripts/viewer/plot_results.py), if `plots: true`
    if settings.get("plots"):
        sys.path.insert(0, str(ROOT / "scripts" / "viewer"))
        import plot_results
        plot_results.main()


if __name__ == "__main__":
    main()

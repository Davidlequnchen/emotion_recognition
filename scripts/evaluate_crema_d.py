"""
Evaluate emotion2vec+ on the whole CREMA-D dataset and write a Markdown report.

Runs the model on every clip in data/crema_d/AudioWAV, compares each prediction with the acted
emotion in the file name (the ground truth) and writes:
    reports/crema_d_<model>.md                      accuracy, per-emotion results, confusion matrix
    reports/figures/crema_d_<model>_confusion*.png  confusion-matrix figures used in the report
    outputs/crema_d_<model>_predictions.csv          one row per clip: labels, prediction, probabilities
    outputs/crema_d_<model>_predictions.json         what the CSV contains (read by scripts/make_viewer.py)

Download the dataset first: python scripts/download_data.py --subset full

Usage:
    python scripts/evaluate_crema_d.py                 # device from config/config.yaml (MPS on the Mac)
    python scripts/evaluate_crema_d.py --device cpu    # override the device
    python scripts/evaluate_crema_d.py --limit 300     # quick test on the first 300 clips
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crema_d"
# CREMA-D emotion codes: the third part of each file name, and the letters used in summaryTable.csv
EMOTIONS = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
VOTES = {"A": "angry", "D": "disgusted", "F": "fearful", "H": "happy", "N": "neutral", "S": "sad"}
SILENT = "1076_MTI_SAD_XX.wav"  # the one CREMA-D WAV without audio (listed in the dataset's README)
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


def listener_vote(code):
    """summaryTable letters -> emotion names. A tie such as "N:S" becomes "neutral or sad"."""
    return " or ".join(VOTES[c] for c in code.split(":")) if code else ""


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml", help="project config file")
    parser.add_argument("--mode", choices=["mac", "cuda"], help="override `mode` from the config")
    parser.add_argument("--device", help="override the device, e.g. mps, cpu, cuda:1")
    parser.add_argument("--limit", type=int, help="evaluate only the first N clips (quick test)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep prints in order, also when redirected to a log

    # ------------------------------------------------------------------
    # Step 1: Read the project configuration (config/config.yaml)
    # ------------------------------------------------------------------
    step("Step 1/6 | Load configuration")
    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = args.mode or cfg["mode"]
    device = torch.device(args.device or cfg["devices"][mode])
    model_name = cfg["model"]
    model_dir = ROOT / cfg["models"][model_name]["dir"]
    print(f"  mode        : {mode}")
    print(f"  device      : {device}")
    print(f"  model       : {model_name} ({model_dir})")
    print(f"  dataset     : {DATA}")

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
    model = AutoModel(
        model=str(model_dir),
        device=str(device),
        ncpu=torch.get_num_threads(),
        excludes="modality_encoders.AUDIO.decoder.",  # pre-training decoder: not in model.pt, unused for inference
        disable_update=True,
        disable_pbar=True,
    )
    labels = [t.split("/")[-1] for t in (model_dir / "tokens.txt").read_text(encoding="utf-8").split()]
    print(f"  loaded in   : {time.perf_counter() - t0:.1f} s")
    print(f"  classes     : {', '.join(labels)}")

    # ------------------------------------------------------------------
    # Step 4: List the clips and read their labels
    # ------------------------------------------------------------------
    step("Step 4/6 | Load CREMA-D and its labels")
    files = sorted((DATA / "AudioWAV").glob("*.wav"))[: args.limit]
    if not files:
        sys.exit(f"ERROR: no clips in {DATA / 'AudioWAV'}. Download them: python scripts/download_data.py --subset full")
    acted = [EMOTIONS[f.stem.split("_")[2]] for f in files]  # ground truth, e.g. 1001_DFA_ANG_XX -> angry
    actors = {f.stem.split("_")[0] for f in files}
    with open(DATA / "processedResults" / "summaryTable.csv", newline="") as fh:
        votes = {row["FileName"]: row["VoiceVote"] for row in csv.DictReader(fh)}
    listener = [listener_vote(votes.get(f.stem, "")) for f in files]  # listeners' majority vote, voice only
    print(f"  clips       : {len(files)} from {len(actors)} actors")
    for name in EMOTIONS.values():
        print(f"  {name:<11} : {acted.count(name)} clips")

    # ------------------------------------------------------------------
    # Step 5: Run the model on every clip (one probability per class)
    # ------------------------------------------------------------------
    step(f"Step 5/6 | Run inference on {len(files)} clips")
    model.generate(str(files[0]), granularity="utterance", extract_embedding=False)  # warm-up
    scores = np.zeros((len(files), len(labels)))
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        scores[i - 1] = model.generate(str(f), granularity="utterance", extract_embedding=False)[0]["scores"]
        if i % 500 == 0 or i == len(files):
            elapsed = time.perf_counter() - t0
            print(f"  {i:>5}/{len(files)} clips | {elapsed / i * 1000:.0f} ms per clip | "
                  f"about {elapsed / i * (len(files) - i) / 60:.1f} min left")
    runtime = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Step 6: Compare with the ground truth and write the outputs
    # ------------------------------------------------------------------
    step("Step 6/6 | Compute accuracy and write the report")
    names = list(EMOTIONS.values())                       # the 6 CREMA-D emotions
    columns = [labels.index(n) for n in names]            # where they sit in the model output
    truth = np.array([names.index(a) for a in acted])
    pred6 = scores[:, columns].argmax(1)                  # most probable of the 6 CREMA-D emotions
    pred9 = scores.argmax(1)                              # most probable of all 9 model classes
    # Two ways to count a prediction as correct:
    #   strict  - it matches the acted emotion (the usual CREMA-D evaluation)
    #   lenient - it matches the acted emotion OR the listeners' vote (any emotion of a tied vote)
    strict = pred6 == truth
    heard = [set(v.split(" or ")) if v else set() for v in listener]
    lenient = strict | np.array([names[p] in h for p, h in zip(pred6, heard)])
    cm = np.zeros((6, 6), dtype=int)
    np.add.at(cm, (truth, pred6), 1)                      # rows: acted emotion, columns: prediction
    # Lenient matrix: a clip goes into the row of the label its prediction matched, otherwise its acted emotion
    cm_lenient = np.zeros((6, 6), dtype=int)
    np.add.at(cm_lenient, (np.where(lenient, pred6, truth), pred6), 1)
    recall = ratio(cm.diagonal(), cm.sum(1))
    precision = ratio(cm.diagonal(), cm.sum(0))
    f1 = ratio(2 * precision * recall, precision + recall)
    acc6, acc_lenient = strict.mean(), lenient.mean()
    lenient_by_emotion = [lenient[truth == k].mean() if (truth == k).any() else 0.0 for k in range(6)]
    rescued = collections.Counter((acted[i], names[pred6[i]]) for i in np.flatnonzero(lenient & ~strict))
    acc9 = np.mean([labels[p] == a for p, a in zip(pred9, acted)])
    extra = {lab: int((pred9 == k).sum()) for k, lab in enumerate(labels) if lab not in names}
    # Human reference: the listeners' majority vote from the voice alone (tied votes are left out)
    voted = [i for i, v in enumerate(listener) if v in names]
    human = np.mean([listener[i] == acted[i] for i in voted])
    agree = np.mean([names[pred6[i]] == listener[i] for i in voted])
    print(f"  accuracy, acted emotion             : {acc6:.1%}")
    print(f"  accuracy, acted emotion or listener : {acc_lenient:.1%}")
    print(f"  unweighted accuracy (UA)            : {recall.mean():.1%}")
    print(f"  macro F1                            : {f1.mean():.1%}")
    print(f"  accuracy, all 9 classes             : {acc9:.1%}")
    print(f"  listeners vs acted emotion          : {human:.1%}")

    # Per-clip predictions (CSV) and a description of its columns (JSON, read by scripts/make_viewer.py)
    suffix = f"_first{args.limit}" if args.limit else ""
    predictions = ROOT / "outputs" / f"crema_d_{model_name}{suffix}_predictions.csv"
    predictions.parent.mkdir(exist_ok=True)
    prob_columns = [f"p_{lab.strip('<>')}" for lab in labels]  # "<unk>" -> "p_unk"
    with open(predictions, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "acted_emotion", "listener_vote", "prediction", "prediction_9_classes", "correct",
                         "correct_acted_or_listener"] + prob_columns)
        for i, f in enumerate(files):
            writer.writerow([f.relative_to(ROOT).as_posix(), acted[i], listener[i], names[pred6[i]], labels[pred9[i]],
                             bool(strict[i]), bool(lenient[i])] + [f"{p:.4f}" for p in scores[i]])
    meta = {
        "dataset": "CREMA-D",
        "model": model_name,
        "date": str(date.today()),
        "clips": len(files),
        "accuracy": round(float(acc6), 4),
        "accuracy_acted_or_listener": round(float(acc_lenient), 4),
        "ground_truth": "acted_emotion",
        # Correctness columns and what the prediction is compared with (the viewer can switch between them)
        "correct_columns": {"correct": "acted emotion", "correct_acted_or_listener": "acted emotion or listener vote"},
        "columns": {
            "audio_path": "audio file, relative to the repository root",
            "acted_emotion": "the emotion the actor was asked to portray (ground truth)",
            "listener_vote": "majority vote of crowd-sourced listeners who heard the voice only; 'a or b' is a tie",
            "prediction": "model prediction: the most probable of the 6 CREMA-D emotions",
            "prediction_9_classes": "model prediction: the most probable of all 9 model classes",
            "correct": "whether prediction equals acted_emotion",
            "correct_acted_or_listener": "whether prediction equals acted_emotion or the listener_vote (any emotion of a tie)",
            "p_*": "model probability of each of its 9 classes (they add up to 1)",
        },
    }
    predictions.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Confusion-matrix figures (one per way of counting) and the Markdown report
    figures = ROOT / "reports" / "figures"
    figure = figures / f"crema_d_{model_name}{suffix}_confusion.png"
    figure_lenient = figures / f"crema_d_{model_name}{suffix}_confusion_acted_or_listener.png"
    plot_confusion(cm, names, f"{model_name} on CREMA-D",
                   f"Correct = matches the acted emotion. {len(files):,} clips, accuracy {100 * acc6:.1f}%.",
                   "Acted emotion (clips)", figure)
    plot_confusion(cm_lenient, names, f"{model_name} on CREMA-D",
                   f"Correct = matches the acted emotion or the listeners' vote. Accuracy {100 * acc_lenient:.1f}%.",
                   "Acted emotion, or listener vote if matched (clips)", figure_lenient)
    pct = lambda x: f"{100 * x:.1f}%"
    image = lambda path, alt: f"![{alt}]({path.relative_to(ROOT / 'reports').as_posix()})"
    report = [
        f"# {model_name} on CREMA-D",
        "",
        f"Generated by `scripts/evaluate_crema_d.py` on {date.today()}"
        + (f" (quick test: first {args.limit} clips only)." if args.limit else "."),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Clips evaluated | {len(files):,} ({len(actors)} actors) |",
        f"| **Accuracy: prediction matches the acted emotion** | **{pct(acc6)}** |",
        f"| **Accuracy: prediction matches the acted emotion or the listeners' vote** | **{pct(acc_lenient)}** |",
        f"| Unweighted accuracy (mean recall of the 6 acted emotions) | {pct(recall.mean())} |",
        f"| Macro F1 (acted emotion) | {pct(f1.mean())} |",
        f"| Accuracy, all 9 model classes (acted emotion) | {pct(acc9)} |",
        f"| Human listeners, voice only: majority vote vs acted emotion | {pct(human)} |",
        "",
        "## Setup",
        "",
        f"- **Model**: `{model_name}` as released (no fine-tuning), FunASR {funasr.__version__}, one prediction per clip.",
        f"- **Data**: {len(files):,} clips from `data/crema_d/AudioWAV` (16 kHz mono WAV, CREMA-D commit `1658cd3`).",
        "- **Ground truth**: the emotion each actor was asked to portray, taken from the file name "
        "(`ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`).",
        "- **Label mapping**: " + ", ".join(f"`{c}` = {n}" for c, n in EMOTIONS.items()) + ".",
        "- **6 emotions**: the prediction is the most probable of the six CREMA-D emotions; the model's other classes "
        f"({', '.join(f'`{lab}`' for lab in extra)}) are ignored. **All 9 classes**: the prediction is the most "
        "probable of all nine classes, so choosing a class that CREMA-D does not have counts as wrong.",
        "- **Two ways to count a prediction as correct.** *Acted emotion*: it must match the emotion the actor "
        "portrayed (the usual way to report CREMA-D). *Acted emotion or listeners' vote*: it also counts when it "
        "matches the majority vote of the listeners who heard the voice only (for a tied vote, any of the tied "
        "emotions). The second credits the model when it agrees with what people heard. Both are reported.",
        f"- **Hardware**: {hardware(device)}, device `{device}`, PyTorch {torch.__version__}. "
        f"Inference took {runtime / 60:.1f} min ({runtime / len(files) * 1000:.0f} ms per clip).",
        f"- **Per-clip predictions**: `{predictions.relative_to(ROOT)}` (not in git; re-run the script to recreate it). "
        "Its columns are described in the JSON file next to it and in `docs/crema_d.md`.",
        "",
        "## Results per emotion",
        "",
        "Rows: acted emotion. Precision, recall and F1 compare the prediction with the acted emotion; the last column "
        "also accepts the listeners' vote.",
        "",
        "| Emotion | Clips | Precision | Recall | F1 | Correct, acted or listeners' vote |",
        "|---|---|---|---|---|---|",
        *[f"| {n} | {cm[k].sum():,} | {pct(precision[k])} | {pct(recall[k])} | {pct(f1[k])} | {pct(lenient_by_emotion[k])} |"
          for k, n in enumerate(names)],
        "",
        "## Confusion matrix: acted emotion",
        "",
        image(figure, f"Confusion matrix of {model_name} on CREMA-D, acted emotion"),
        "",
        "The same numbers as a table. Rows: acted emotion. Columns: predicted emotion. Each row adds up to 100%.",
        "",
        *confusion_table(cm, names, "Acted \\ Predicted"),
        "",
        f"Most frequent confusions: {top_confusions(cm, names)}.",
        "",
        "## Confusion matrix: acted emotion or listeners' vote",
        "",
        image(figure_lenient, f"Confusion matrix of {model_name} on CREMA-D, acted emotion or listeners' vote"),
        "",
        "Here a prediction is correct when it matches the acted emotion or the listeners' vote. A clip goes into the "
        "row of the label its prediction matched, otherwise into the row of its acted emotion. The diagonal therefore "
        "holds every correct prediction, and the rows differ in size from the matrix above. "
        f"{sum(rescued.values()):,} clips ({pct(sum(rescued.values()) / len(files))}) count as correct only because "
        "of the listeners' vote; the most common are "
        + ", ".join(f"acted {a} predicted as {p} ({n})" for (a, p), n in rescued.most_common(3)) + ".",
        "",
        *confusion_table(cm_lenient, names, "Ground truth \\ Predicted"),
        "",
        f"Most frequent confusions: {top_confusions(cm_lenient, names)}.",
        "",
        "## Predictions outside the six emotions",
        "",
        "With all nine classes, the model chose a class that CREMA-D does not have for "
        f"{sum(extra.values()):,} clips ({pct(sum(extra.values()) / len(files))}): "
        + ", ".join(f"`{lab}` {n:,}" for lab, n in extra.items()) + ".",
        "",
        "## Comparison with human listeners",
        "",
        "CREMA-D's crowd-sourced listeners also rated every clip from the voice alone "
        "(`processedResults/summaryTable.csv`). Their majority vote matches the acted emotion for "
        f"{pct(human)} of the {len(voted):,} clips without a tied vote. The model agrees with that majority vote "
        f"for {pct(agree)} of the same clips.",
        "",
        "## Caveats",
        "",
        "- **Not a held-out test.** CREMA-D is one of the datasets in [EmoBox](https://github.com/emo-box/EmoBox). "
        "EmoBox data fine-tuned emotion2vec+ seed, the first model in the chain that led to emotion2vec+ large "
        "(see the model card). The model has probably seen these clips, or labels derived from them, during "
        "training, so these numbers are likely higher than on unseen data.",
        "- **Acted labels are not perceived emotions.** The ground truth is what the actor intended. Listeners often "
        "heard something else, as the comparison above shows.",
        *([f"- **One clip is silent.** `{SILENT}` contains no audio (a known CREMA-D issue); it is kept so that "
           "every clip is counted."] if any(f.name == SILENT for f in files) else []),
        "",
    ]
    out = ROOT / "reports" / f"crema_d_{model_name}{suffix}.md"
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nDone: report -> {out.relative_to(ROOT)}, figures -> {figures.relative_to(ROOT)}/, "
          f"predictions -> {predictions.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

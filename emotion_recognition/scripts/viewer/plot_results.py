"""
Dataset panels and compact score matrices for comparing models, for papers and slides.

Reads the results that the evaluations left in outputs/ and draws, for each metric in config/plots.yaml:
    reports/figures/<metric>_by_dataset.png (300 dpi) and .pdf   one panel per dataset, one bar per model
    reports/figures/overview_<metric>.png and .pdf                dataset-by-model annotated score matrix
Bar patterns distinguish directly documented, indirectly supported and possible training-source evidence.
Unhatched means no corpus-specific evidence, not verified unseen; source links do not prove test-clip leakage.
The dashed line in each panel: chance level. A value marked * comes from a sample, not every clip.

Which results it reads: every outputs/<model>/<dataset>[_sample<N>]_predictions.json that has `metrics` (for a model
and dataset with several compatible runs, the one with the most clips). Registered audio LLMs require the current
inference_protocol; legacy prompt/candidate results stay on disk but are excluded. So any model counts, whether it ran locally
(scripts/evaluation/evaluate.py) or through an API (GPT Realtime, Gemini): an API evaluation only has to write the same
kind of JSON into outputs/<its key>/. The fields used are:
    dataset (display name), key (dataset key), model_name (display name of the model), clips, total_clips,
    chance (a fraction), metrics {accuracy, unweighted_accuracy, macro_f1, ...}
Training markers/status/source are resolved from config/models.yaml, not historical in_training_data booleans.
Models are ordered as in config/models.yaml, then any other model folder in alphabetical order; datasets in the order
of scripts/common/eval_datasets.py.

Usage (from the emotion_recognition/ folder):
    python scripts/viewer/plot_results.py
Settings: config/plots.yaml. scripts/evaluation/evaluate.py runs it at its end when `plots: true` is set in
config/evaluation.yaml.
"""
import json
import hashlib
import math
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "plots.yaml"
MODELS_CONFIG = ROOT / "config" / "models.yaml"
sys.path.insert(0, str(ROOT / "scripts" / "common"))  # eval_datasets.py
sys.path.insert(0, str(ROOT / "scripts" / "llm_eval"))
from eval_datasets import LOADERS
from prompts import configured_protocol
from training_provenance import (EXPOSURE_LABEL, EXPOSURE_LEVELS, EXPOSURE_NOTE, training_hatch,
                                 training_marker, with_training_provenance)

# One colour per model, in this order (Okabe-Ito: distinguishable with colour-vision deficiencies and in grey print)
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
METRIC_NAMES = {"accuracy": "Accuracy", "unweighted_accuracy": "UAR",
                "macro_f1": "Macro F1", "accuracy_all_classes": "Accuracy (all model classes)",
                "accuracy_acted_or_listener": "Accuracy (acted emotion or listeners' vote)"}
INK, INK_2, GRID = "#111111", "#444444", "#d9d9d9"


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def as_list(value):
    return [value] if isinstance(value, str) else list(value or [])


def load_results(models, datasets, exclude, registry=None, root=None):
    """{model key: {dataset key: run}}: the finished run with the most clips per model and dataset."""
    root = Path(root or ROOT)
    models = models if models == "all" else as_list(models)
    datasets = datasets if datasets == "all" else as_list(datasets)
    exclude = as_list(exclude)
    if registry is None:
        registry = yaml.safe_load((root / "config" / "models.yaml").read_text()) or {}
    best = {}
    for path in sorted((root / "outputs").glob("*/*_predictions.json")):
        model = path.parent.name
        if model == "archive" or (models != "all" and model not in models):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        key = run.get("key")
        if "metrics" not in run or key in exclude or (datasets != "all" and key not in datasets):
            continue
        expected = configured_protocol(registry.get(model, {}), run.get("classes", []),
                                       config=root / "config" / "llm.yaml", root=root)
        if expected is not None and run.get("inference_protocol") != expected:
            continue
        run = with_training_provenance(run, registry.get(model))
        if run.get("clips", 0) > best.get(model, {}).get(key, {}).get("clips", 0):
            best.setdefault(model, {})[key] = run
    return best


def results_fingerprint(results):
    """Identify the exact result metadata behind figures, so a viewer cannot embed stale pre-protocol plots."""
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # draw straight to files; works on servers without a display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fonts), "DejaVu Sans"),
        "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 13,
        "legend.fontsize": 13, "axes.edgecolor": INK_2, "axes.linewidth": 1.0, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed real fonts (journals ask for it)
    })
    return plt


def legend_handles(plt, names, colours, chance=True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colours[m], edgecolor=INK, linewidth=1.2, label=names[m]) for m in names]
    for status, level in EXPOSURE_LEVELS.items():
        handles.append(Patch(facecolor="white", edgecolor="#9a9fa6" if status == "possible_exposure" else INK,
                             hatch=training_hatch(status), linewidth=1.0,
                             label=level["label"] + (" (no hatch)" if not level["hatch"] else "")))
    if chance:  # macro F1 has no chance line
        handles.append(Line2D([0], [0], color=INK_2, linestyle="--", linewidth=1.6, label="chance level"))
    return handles


def chance_of(run):
    """Chance level of a run; older predictions files have no "chance": 1 in the number of emotions."""
    return run.get("chance") or (1 / len(run["classes"]) if run.get("classes") else None)


def bar_label(ax, x, value, run):
    text = f"{100 * value:.1f}" + ("*" if run["clips"] < run.get("total_clips", run["clips"]) else "")
    ax.text(x, 100 * value + 1.5, text, ha="center", va="bottom", fontsize=12, color=INK)


def per_dataset_figure(plt, metric, results, models, names, colours, datasets, columns, out):
    """One panel per dataset, one bar per model."""
    rows = math.ceil(len(datasets) / columns)
    legend_count = len(models) + len(EXPOSURE_LEVELS) + int(metric != "macro_f1")
    legend_columns = min(4, legend_count)
    # Use physical margins: a percentage of a tall canvas used to leave several inches below the panels.
    legend_rows = math.ceil(legend_count / legend_columns)
    legend_height = 0.32 * legend_rows + 0.64
    height = 4.0 * rows + legend_height + 1.0
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, height), squeeze=False)
    for k, key in enumerate(datasets):
        ax = axes[k // columns][k % columns]
        runs = [results.get(m, {}).get(key) for m in models]
        first = next(r for r in runs if r)
        for i, (m, run) in enumerate(zip(models, runs)):
            if run is None or metric not in run["metrics"]:
                ax.text(i, 2, "n/a", ha="center", va="bottom", fontsize=11, color=INK_2)
                continue
            value = run["metrics"][metric]
            status = run.get("training_status", "not_established")
            hatch = training_hatch(status)
            ax.bar(i, 100 * value, width=0.72, color=colours[m], edgecolor=INK, linewidth=1.2,
                   hatch=hatch if status != "possible_exposure" else None, zorder=3)
            if status == "possible_exposure":
                # A light cross-pattern is a possible association, not a confirmed-exposure hatch.
                ax.bar(i, 100 * value, width=0.72, facecolor="none", edgecolor="#9a9fa6", linewidth=0,
                       hatch=hatch, zorder=3.1)
            bar_label(ax, i, value, run)
        chance = chance_of(first)
        if chance and metric != "macro_f1":
            ax.axhline(100 * chance, color=INK_2, linestyle="--", linewidth=1.6, zorder=4)
        title = textwrap.fill(first["dataset"], width=37)
        if first.get("language"):
            title += "\n" + first["language"]
        ax.set_title(title, pad=8, linespacing=1.05)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if k % columns == 0:
            ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)} (%)")
    for k in range(len(datasets), rows * columns):
        axes[k // columns][k % columns].axis("off")
    fig.suptitle(f"{METRIC_NAMES.get(metric, metric)} by dataset", fontsize=21,
                 fontweight="bold", color=INK, y=1 - 0.10 / height)
    fig.legend(handles=legend_handles(plt, names, colours, metric != "macro_f1"), loc="lower center",
               ncol=legend_columns, frameon=False, bbox_to_anchor=(0.5, 0.54 / height), borderaxespad=0,
               columnspacing=1.7, handletextpad=0.6, labelspacing=0.55)
    fig.text(0.5, 0.08 / height, EXPOSURE_NOTE, ha="center", va="bottom", fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.045, right=0.992, top=1 - 1.28 / height,
                        bottom=(legend_height + 0.16) / height, wspace=0.19, hspace=0.34)
    return save(fig, plt, out / f"{metric}_by_dataset")


def overview_figure(plt, metric, results, models, names, colours, datasets, out):
    """A readable matrix that grows by one column, rather than thirteen bars, for each added model."""
    import numpy as np
    from matplotlib.patches import Rectangle

    has_chance = metric != "macro_f1"
    ncols = len(models) + int(has_chance)
    values = np.full((len(datasets), ncols), np.nan)
    sampled = False
    first = {key: next(results[m][key] for m in models if key in results.get(m, {})) for key in datasets}
    for row, key in enumerate(datasets):
        for col, model in enumerate(models):
            run = results.get(model, {}).get(key)
            if run is not None and metric in run["metrics"]:
                values[row, col] = 100 * run["metrics"][metric]

    width, height = max(11.5, 1.60 * ncols + 3.6), max(5.0, 0.55 * len(datasets) + 1.65)
    fig, ax = plt.subplots(figsize=(width, height))
    cmap = plt.get_cmap("cividis").copy()
    cmap.set_bad("#f1f2f4")
    plotted = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
    headers = [names[m].replace("emotion2vec+ ", "emotion2vec+\n")
               .replace("MERaLiON-", "MERaLiON\n").replace("Vox-Profile ", "Vox-Profile\n")
               for m in models]
    headers = ["\n".join(textwrap.fill(line, width=15) for line in name.splitlines()) for name in headers]
    if has_chance:
        headers.append("Chance\nreference")
    ax.set_xticks(range(ncols), headers, fontsize=12.5)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=11)
    ax.set_yticks(range(len(datasets)), [first[key]["dataset"].replace(" (", "\n(") for key in datasets],
                  fontsize=12.5)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(datasets), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.spines[:].set_visible(False)
    # Match the panel colours without requiring a long model legend beneath the overview.
    for col, model in enumerate(models):
        ax.add_patch(Rectangle((col - 0.5, -0.64), 1, 0.07, facecolor=colours[model], edgecolor="none",
                               clip_on=False))
    for row, key in enumerate(datasets):
        for col, model in enumerate(models):
            value, run = values[row, col], results.get(model, {}).get(key)
            if np.isnan(value):
                text, colour = "—", "#858b94"
            else:
                is_sample = run["clips"] < run.get("total_clips", run["clips"])
                sampled |= is_sample
                text = f"{value:.1f}" + training_marker(run) + ("*" if is_sample else "")
                r, g, b, _ = cmap(value / 100)
                colour = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.49 else INK
            ax.text(col, row, text, ha="center", va="center", fontsize=13, color=colour)
        if has_chance:
            chance = chance_of(first[key])
            ax.text(len(models), row, f"{100 * chance:.1f}" if chance is not None else "—",
                    ha="center", va="center", fontsize=12.5, color=INK_2)
    if has_chance:
        ax.axvline(len(models) - 0.5, color="#c8ccd1", linewidth=1.5)
    colourbar = fig.colorbar(plotted, ax=ax, fraction=0.026, pad=0.025, shrink=0.94)
    colourbar.set_label(f"{METRIC_NAMES.get(metric, metric)} (%)", fontsize=17, labelpad=11)
    colourbar.ax.tick_params(labelsize=12)
    notes = [f"{level['marker']} {level['label']}" for level in EXPOSURE_LEVELS.values() if level["marker"]]
    notes += ["* Sampled evaluation"] if sampled else []
    notes.append("— No compatible result")
    fig.suptitle(f"{METRIC_NAMES.get(metric, metric)} across models and datasets", fontsize=20,
                 fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.025, "    ".join(notes) + "\n" + EXPOSURE_NOTE,
             ha="center", va="bottom", fontsize=11.5, color=INK_2)
    fig.subplots_adjust(left=3.15 / width, right=1 - 0.85 / width,
                        top=1 - 1.0 / height, bottom=1.10 / height)
    return save(fig, plt, out / f"overview_{metric}")


def save(fig, plt, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 300), (".pdf", None)):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stem


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # ------------------------------------------------------------------
    # Step 1: Read the settings (config/plots.yaml) and the order of the models (config/models.yaml)
    # ------------------------------------------------------------------
    step("Step 1/3 | Load configuration")
    settings = yaml.safe_load(CONFIG.read_text()) or {}
    registry = yaml.safe_load(MODELS_CONFIG.read_text()) or {}
    metrics, columns = as_list(settings.get("metrics", ["accuracy"])), int(settings.get("columns", 4))
    wanted_models = settings.get("models", "all")
    wanted_models = wanted_models if wanted_models == "all" else as_list(wanted_models)
    wanted_datasets = settings.get("datasets", "all")
    wanted_datasets = wanted_datasets if wanted_datasets == "all" else as_list(wanted_datasets)
    exclude, out = as_list(settings.get("exclude")), ROOT / settings.get("out", "reports/figures")
    print(f"  config      : {CONFIG}")
    print(f"  metrics     : {', '.join(metrics)}")
    print(f"  out         : {out.relative_to(ROOT)}")

    # ------------------------------------------------------------------
    # Step 2: Collect the results in outputs/<model>/ (local models and API models alike)
    # ------------------------------------------------------------------
    step("Step 2/3 | Collect results")
    results = load_results(wanted_models, wanted_datasets, exclude, registry)
    if not results:
        sys.exit("ERROR: no results in outputs/<model>/ yet; run scripts/evaluation/evaluate.py first.")
    models = [m for m in registry if m in results] + sorted(m for m in results if m not in registry)
    datasets = [k for k in LOADERS if any(k in results[m] for m in models)]
    datasets += sorted({k for m in models for k in results[m]} - set(datasets))
    names = {m: next(iter(results[m].values())).get("model_name", m) for m in models}
    colours = {m: COLOURS[i % len(COLOURS)] for i, m in enumerate(models)}
    for m in models:
        sampled = [k for k, r in results[m].items() if r["clips"] < r.get("total_clips", r["clips"])]
        print(f"  {names[m]:<22}: {len(results[m])} dataset(s)" + (f" ({', '.join(sampled)} on a sample)" if sampled
                                                                     else ""))
    print(f"  datasets    : {', '.join(datasets)}")

    # ------------------------------------------------------------------
    # Step 3: Draw the figures, PNG (300 dpi) and PDF
    # ------------------------------------------------------------------
    step("Step 3/3 | Draw the figures")
    plt = setup_matplotlib()
    rendered = []
    for metric in metrics:
        if not any(metric in r["metrics"] for m in models for r in results[m].values()):
            print(f"  {metric}: no result has this metric, skipped")
            continue
        for stem in (per_dataset_figure(plt, metric, results, models, names, colours, datasets, columns, out),
                     overview_figure(plt, metric, results, models, names, colours, datasets, out)):
            print(f"  {stem.relative_to(ROOT)}.png and .pdf")
        rendered.append(metric)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_manifest.json").write_text(json.dumps({"fingerprint": results_fingerprint(results),
                                                         "metrics": rendered}, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()

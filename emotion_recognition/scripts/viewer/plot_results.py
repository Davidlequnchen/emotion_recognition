"""
Bar plots that compare every model on every dataset, for papers and slides.

Reads the results that the evaluations left in outputs/ and draws, for each metric in config/plots.yaml:
    reports/figures/<metric>_by_dataset.png (300 dpi) and .pdf   one panel per dataset, one bar per model
    reports/figures/overview_<metric>.png and .pdf                every dataset side by side, a group of bars each
Bars with a hatch: the dataset is in that model's training data, so its result is optimistic. The dashed line in each
panel: chance level. A value marked * comes from a sample of the dataset (evaluation limit), not every clip.

Which results it reads: every outputs/<model>/<dataset>[_sample<N>]_predictions.json that has `metrics` (for a model
and dataset with several runs, the one with the most clips). So any model counts, whether it ran locally
(scripts/evaluation/evaluate.py) or through an API (GPT Realtime, Gemini): an API evaluation only has to write the same
kind of JSON into outputs/<its key>/. The fields used are:
    dataset (display name), key (dataset key), model_name (display name of the model), clips, total_clips,
    chance (a fraction), in_training_data (true or false), metrics {accuracy, unweighted_accuracy, macro_f1, ...}
Models are ordered as in config/models.yaml, then any other model folder in alphabetical order; datasets in the order
of scripts/common/eval_datasets.py.

Usage (from the emotion_recognition/ folder):
    python scripts/viewer/plot_results.py
Settings: config/plots.yaml. scripts/evaluation/evaluate.py runs it at its end when `plots: true` is set in
config/evaluation.yaml.
"""
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "plots.yaml"
MODELS_CONFIG = ROOT / "config" / "models.yaml"
sys.path.insert(0, str(ROOT / "scripts" / "common"))  # eval_datasets.py
from eval_datasets import LOADERS

# One colour per model, in this order (Okabe-Ito: distinguishable with colour-vision deficiencies and in grey print)
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"]
METRIC_NAMES = {"accuracy": "Accuracy (acted emotion)", "unweighted_accuracy": "Unweighted accuracy",
                "macro_f1": "Macro F1", "accuracy_all_classes": "Accuracy (all model classes)",
                "accuracy_acted_or_listener": "Accuracy (acted emotion or listeners' vote)"}
INK, INK_2, GRID = "#111111", "#444444", "#d9d9d9"


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def as_list(value):
    return [value] if isinstance(value, str) else list(value or [])


def load_results(models, datasets, exclude):
    """{model key: {dataset key: run}}: the finished run with the most clips per model and dataset."""
    best = {}
    for path in sorted((ROOT / "outputs").glob("*/*_predictions.json")):
        model = path.parent.name
        if model == "archive" or (models != "all" and model not in models):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        key = run.get("key")
        if "metrics" not in run or key in exclude or (datasets != "all" and key not in datasets):
            continue
        if run.get("clips", 0) > best.get(model, {}).get(key, {}).get("clips", 0):
            best.setdefault(model, {})[key] = run
    return best


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # draw straight to files; works on servers without a display
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({
        "font.family": next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fonts), "DejaVu Sans"),
        "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13,
        "legend.fontsize": 14, "axes.edgecolor": INK_2, "axes.linewidth": 1.0, "axes.titleweight": "bold",
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed real fonts (journals ask for it)
    })
    return plt


def legend_handles(plt, names, colours, any_trained):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colours[m], edgecolor=INK, linewidth=1.2, label=names[m]) for m in names]
    if any_trained:
        handles.append(Patch(facecolor="white", edgecolor=INK, hatch="///", linewidth=1.2,
                             label="dataset in the model's training data"))
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
    fig, axes = plt.subplots(rows, columns, figsize=(5.6 * columns, 4.9 * rows + 1.2), squeeze=False)
    trained = False
    for k, key in enumerate(datasets):
        ax = axes[k // columns][k % columns]
        runs = [results.get(m, {}).get(key) for m in models]
        first = next(r for r in runs if r)
        for i, (m, run) in enumerate(zip(models, runs)):
            if run is None or metric not in run["metrics"]:
                ax.text(i, 2, "n/a", ha="center", va="bottom", fontsize=11, color=INK_2)
                continue
            value, hatch = run["metrics"][metric], "///" if run.get("in_training_data") else None
            trained |= bool(hatch)
            ax.bar(i, 100 * value, width=0.72, color=colours[m], edgecolor=INK, linewidth=1.2, hatch=hatch, zorder=3)
            bar_label(ax, i, value, run)
        chance = chance_of(first)
        if chance and metric != "macro_f1":
            ax.axhline(100 * chance, color=INK_2, linestyle="--", linewidth=1.6, zorder=4)
        ax.set_title(f"{first['dataset']} – {first['language']}" if first.get("language") else first["dataset"])
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
    fig.suptitle(f"{METRIC_NAMES.get(metric, metric)} of every model on every dataset", fontsize=20,
                 fontweight="bold", color=INK, y=0.995)
    fig.legend(handles=legend_handles(plt, names, colours, trained), loc="lower center",
               ncol=min(4, len(models) + 2), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09 + 0.03 * (len(models) > 4), 1, 0.96))
    return save(fig, plt, out / f"{metric}_by_dataset")


def overview_figure(plt, metric, results, models, names, colours, datasets, out):
    """Every dataset side by side: a group of bars (one per model) for each dataset."""
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(12, 2.3 * len(datasets) + 2), 6.4))
    trained = False
    for i, m in enumerate(models):
        for j, key in enumerate(datasets):
            run = results.get(m, {}).get(key)
            if run is None or metric not in run["metrics"]:
                continue
            x, value = j - 0.4 + width * (i + 0.5), run["metrics"][metric]
            hatch = "///" if run.get("in_training_data") else None
            trained |= bool(hatch)
            ax.bar(x, 100 * value, width=width * 0.92, color=colours[m], edgecolor=INK, linewidth=1.1, hatch=hatch,
                   zorder=3)
    first = {key: next(results[m][key] for m in models if key in results.get(m, {})) for key in datasets}
    ax.set_xticks(range(len(datasets)), [first[key]["dataset"] for key in datasets])
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)} (%)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if metric != "macro_f1":  # chance level of each dataset, as a short dashed line over its group
        for j, key in enumerate(datasets):
            if chance_of(first[key]):
                ax.hlines(100 * chance_of(first[key]), j - 0.45, j + 0.45, colors=INK_2, linestyles="--",
                          linewidth=1.6, zorder=4)
    ax.set_title(f"{METRIC_NAMES.get(metric, metric)}: every model on every dataset", fontsize=18, pad=12)
    ax.legend(handles=legend_handles(plt, names, colours, trained), loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=min(4, len(models) + 2), frameon=False)
    fig.tight_layout()
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
    registry = list((yaml.safe_load(MODELS_CONFIG.read_text()) or {}).keys())
    metrics, columns = as_list(settings.get("metrics", ["accuracy"])), int(settings.get("columns", 3))
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
    results = load_results(wanted_models, wanted_datasets, exclude)
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
    for metric in metrics:
        if not any(metric in r["metrics"] for m in models for r in results[m].values()):
            print(f"  {metric}: no result has this metric, skipped")
            continue
        for stem in (per_dataset_figure(plt, metric, results, models, names, colours, datasets, columns, out),
                     overview_figure(plt, metric, results, models, names, colours, datasets, out)):
            print(f"  {stem.relative_to(ROOT)}.png and .pdf")
    print("Done.")


if __name__ == "__main__":
    main()

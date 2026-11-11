"""
Dataset panels and compact score matrices for comparing models, for papers and slides.

Reads the results that the evaluations left in outputs/ and draws, for each metric in config/plots.yaml:
    reports/figures/<metric>_by_dataset.png (300 dpi) and .svg   one panel per dataset, one bar per model
    reports/figures/overview_<metric>.png and .svg                dataset-by-model annotated score matrix
One hatch marks all documented training involvement, including upstream teacher use.
The detailed source/stage evidence stays in the registry and reports rather than the figure.
The dashed line in each panel: chance level. A value marked * comes from a sample, not every clip.
Panel labels are rounded to whole percentage points for print legibility; overview matrices keep one decimal.
This is display rounding only: saved predictions, metric values and bar heights retain their precision.
Panel titles show only dataset and language; subset/cohort scopes stay in the formal reports and metadata.

Which results it reads: every outputs/<model>/<dataset>[_sample<N>]_predictions.json that has `metrics` (for a model
and dataset with several compatible runs, the one with the most clips). Registered audio LLMs require the current
inference_protocol; legacy prompt/candidate results stay on disk but are excluded. So any model counts, whether it ran locally
(scripts/evaluation/evaluate.py) or through an API (GPT Realtime, Gemini): an API evaluation only has to write the same
kind of JSON into outputs/<its key>/. The fields used are:
    dataset (display name), key (dataset key), model_name (display name of the model), clips, total_clips,
    chance (a fraction), metrics {accuracy, unweighted_accuracy, macro_f1, ...}
Training markers/status/source are resolved from config/models.yaml, not historical in_training_data booleans.
Models follow config/models.yaml; dataset_order in config/plots.yaml controls panels and overview rows.
Other available datasets append in loader order, then alphabetical order, so new results are never dropped.

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
from prediction_io import load_reported_metrics
from training_provenance import with_training_provenance

# One colour per model, in this order (Okabe-Ito: distinguishable with colour-vision deficiencies and in grey print)
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999",
           "#332288", "#AA4499"]
METRIC_NAMES = {"accuracy": "Accuracy", "unweighted_accuracy": "UAR",
                "macro_f1": "Macro F1", "weighted_f1": "Weighted F1",
                "accuracy_all_classes": "Accuracy (all model classes)",
                "accuracy_acted_or_listener": "Accuracy (acted emotion or listeners' vote)"}
F1_METRICS = {"macro_f1", "weighted_f1"}
INK, INK_2, GRID = "#111111", "#444444", "#d9d9d9"
TRAINING_HATCH, TRAINING_LABEL = "///", "Part of the training process"
PLOT_STYLE = "ordered_slim_panels_v6"
PANEL_LAYOUT = {
    "panel_width": 4.4, "panel_height": 2.15, "bar_width": 0.56, "bar_spacing": 0.70,
    "x_margin": 0.42, "wspace": 0.14, "hspace": 0.28, "title_pad": 8.0, "ymax": 118.0,
    "y_tick_labels": "left", "show_title": True,
}
DATASET_DISPLAY = {
    "crema_d": ("CREMA-D", "English"), "mcae_spps": ("MCAE-SPPS", "Mandarin"),
    "thai_ser": ("THAI-SER", "Thai"), "nemo": ("nEMO", "Polish"), "mess": ("MESS", "English"),
    "ravdess": ("RAVDESS", "English"), "tess": ("TESS", "English"), "jl_corpus": ("JL-Corpus", "English"),
    "mead": ("MEAD", "English"), "meld": ("MELD", "English"), "emovo": ("EMOVO", "Italian"),
    "emozionalmente": ("Emozionalmente", "Italian"), "savee": ("SAVEE", "English"),
    "emotiontalk": ("EmotionTalk", "Mandarin"), "dailytalk": ("DailyTalk", "English"),
    "multidialog": ("MultiDialog", "English"),
}


def dataset_display(key, run):
    """Presentation-only short name/language; never replace the run's recorded evaluation scope."""
    if key in DATASET_DISPLAY:
        return DATASET_DISPLAY[key]
    name = " ".join(str(run.get("dataset", key)).split(" (", 1)[0].split())
    language = " ".join(str(run.get("language", "")).split())
    if "english" in language.lower() or language.lower() in {"british", "american"}:
        language = "English"
    elif "mandarin" in language.lower():
        language = "Mandarin"
    return name, language


def panel_title(key, run):
    name, language = dataset_display(key, run)
    return f"{name} ({language})" if language else name


def part_of_training(run):
    """Both direct and supported upstream use count; possible/unknown do not assert use."""
    status = run.get("training_status")
    if status:
        return status in ("confirmed_exposure", "probable_exposure")
    return bool(run.get("in_training_data"))  # external records without registry metadata


def chart_models(results, registry, wanted="all", placeholders=None):
    """Reserve configured model slots without fabricating a result or a zero-valued bar."""
    placeholders = placeholders or {}
    allowed = None if wanted == "all" else set(as_list(wanted))
    keys = {m for m in results if allowed is None or m in allowed}
    keys |= {m for m in placeholders if allowed is None or m in allowed}
    models = [m for m in registry if m in keys] + sorted(keys - set(registry))
    names = {m: placeholders.get(m) or registry.get(m, {}).get("name") or
             next(iter(results.get(m, {}).values()), {}).get("model_name", m) for m in models}
    return models, names


def chart_datasets(results, preferred=()):
    """Apply the requested display order, then append every other available result key."""
    preferred = as_list(preferred)
    if len(preferred) != len(set(preferred)):
        raise ValueError("dataset_order contains duplicate dataset keys")
    available = {key for model in results.values() for key in model}
    ordered = [key for key in preferred if key in available]
    ordered += [key for key in LOADERS if key in available and key not in ordered]
    return ordered + sorted(available - set(ordered))


def panel_layout(settings=None):
    """Physical panel dimensions and bar geometry, shared by the CLI and offline previews."""
    settings = settings or {}
    if not isinstance(settings, dict) or set(settings) - set(PANEL_LAYOUT):
        raise ValueError("Unknown panel layout setting")
    layout = PANEL_LAYOUT | settings
    for key in ("panel_width", "panel_height", "bar_width", "bar_spacing", "x_margin", "title_pad", "ymax"):
        if isinstance(layout[key], bool) or not isinstance(layout[key], (int, float)) or not math.isfinite(layout[key]) or layout[key] <= 0:
            raise ValueError(f"layout.{key} must be a positive finite number")
    for key in ("wspace", "hspace"):
        if not isinstance(layout[key], (int, float)) or not math.isfinite(layout[key]) or layout[key] < 0:
            raise ValueError(f"layout.{key} must be a nonnegative finite number")
    if layout["bar_width"] >= layout["bar_spacing"] or layout["ymax"] < 100:
        raise ValueError("Bars must have positive gaps and ymax must include 100 percent")
    if layout["y_tick_labels"] not in ("left", "all") or not isinstance(layout["show_title"], bool):
        raise ValueError("layout requires y_tick_labels: left/all and show_title: true/false")
    return layout


def presentation_fingerprint(settings):
    """Version figure order/layout separately from the immutable prediction/result fingerprint."""
    content = {"style": PLOT_STYLE, "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "metrics": as_list(settings.get("metrics", ["accuracy"])), "columns": int(settings.get("columns", 4)),
               "models": settings.get("models", "all"), "datasets": settings.get("datasets", "all"),
               "exclude": as_list(settings.get("exclude")), "placeholders": settings.get("placeholders") or {},
               "dataset_order": as_list(settings.get("dataset_order")), "layout": panel_layout(settings.get("layout"))}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


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
        run = run | {"metrics": load_reported_metrics(path, run)}
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
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 19, "xtick.labelsize": 14, "ytick.labelsize": 14,
        "legend.fontsize": 14, "axes.edgecolor": INK_2, "axes.linewidth": 1.0, "axes.titleweight": "bold",
        "svg.fonttype": "none", "svg.hashsalt": "emotion-recognition",
    })
    return plt


def legend_handles(plt, names, colours, chance=True, pending=()):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="white" if m in pending else colours[m], edgecolor=colours[m] if m in pending else INK,
                     linewidth=1.2, label=names[m] + (" (pending)" if m in pending else "")) for m in names]
    handles.append(Patch(facecolor="white", edgecolor=INK, hatch=TRAINING_HATCH, linewidth=1.0,
                         label=TRAINING_LABEL))
    if chance:  # F1 metrics do not use the accuracy chance reference.
        handles.append(Line2D([0], [0], color=INK_2, linestyle="--", linewidth=1.6, label="chance level"))
    return handles


def chance_of(run):
    """Chance level of a run; older predictions files have no "chance": 1 in the number of emotions."""
    return run.get("chance") or (1 / len(run["classes"]) if run.get("classes") else None)


def bar_label(ax, x, value, run):
    text = f"{100 * value:.0f}" + ("*" if run["clips"] < run.get("total_clips", run["clips"]) else "")
    ax.text(x, 100 * value + 1.4, text, ha="center", va="bottom", fontsize=15.5, color=INK)


def per_dataset_figure(plt, metric, results, models, names, colours, datasets, columns, out, layout=None):
    """One panel per dataset, with a fixed bottom legend (ten models + two keys fit two rows)."""
    layout = panel_layout(layout)
    rows = math.ceil(len(datasets) / columns)
    has_chance = metric not in F1_METRICS
    legend_count = len(models) + 1 + int(has_chance)
    legend_columns = min(6, legend_count)
    pending = {m for m in models if not results.get(m)}
    legend_rows = math.ceil(legend_count / legend_columns)
    # Compact handles leave room for unbroken names: two item rows are also two text lines.
    legend_height = 0.31 * legend_rows + 0.08
    top_margin = 0.82 if layout["show_title"] else 0.36
    height = layout["panel_height"] * rows + legend_height + top_margin
    fig, axes = plt.subplots(rows, columns, figsize=(layout["panel_width"] * columns, height), squeeze=False)
    positions = [i * layout["bar_spacing"] for i in range(len(models))]
    for k, key in enumerate(datasets):
        ax = axes[k // columns][k % columns]
        runs = [results.get(m, {}).get(key) for m in models]
        first = next(r for r in runs if r)
        for i, (m, run) in enumerate(zip(models, runs)):
            if run is None or run["metrics"].get(metric) is None:
                ax.text(positions[i], 0.025, "—" if m in pending else "n/a", transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=11, color=colours[m] if m in pending else INK_2)
                continue
            value = run["metrics"][metric]
            ax.bar(positions[i], 100 * value, width=layout["bar_width"], color=colours[m], edgecolor=INK, linewidth=1.0,
                   hatch=TRAINING_HATCH if part_of_training(run) else None, zorder=3)
            bar_label(ax, positions[i], value, run)
        chance = chance_of(first)
        if chance and has_chance:
            ax.axhline(100 * chance, color=INK_2, linestyle="--", linewidth=1.6, zorder=4)
        ax.set_title(panel_title(key, first), pad=layout["title_pad"])
        ax.set_xticks([])
        ax.set_xlim(-layout["x_margin"], positions[-1] + layout["x_margin"])
        ax.set_ylim(0, layout["ymax"])
        ax.set_yticks(range(0, 101, 20))
        if layout["y_tick_labels"] == "left" and k % columns:
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        if k % columns == 0:
            ax.set_ylabel(f"{METRIC_NAMES.get(metric, metric)} (%)")
    for k in range(len(datasets), rows * columns):
        axes[k // columns][k % columns].axis("off")
    if layout["show_title"]:
        fig.suptitle(f"{METRIC_NAMES.get(metric, metric)} by dataset", fontsize=22,
                     fontweight="bold", color=INK, y=1 - 0.04 / height)
    fig.subplots_adjust(left=0.052, right=0.995, top=1 - top_margin / height,
                        bottom=(legend_height + 0.20) / height,
                        wspace=layout["wspace"], hspace=layout["hspace"])
    handles = legend_handles(plt, names, colours, has_chance, pending)
    fig.legend(handles=handles, loc="lower center", ncol=legend_columns, frameon=False,
               bbox_to_anchor=(0.5, 0.035 / height), borderaxespad=0, borderpad=0.2,
               handlelength=1.4, columnspacing=0.8, handletextpad=0.35, labelspacing=0.45)
    return save(fig, plt, out / f"{metric}_by_dataset")


def overview_figure(plt, metric, results, models, names, colours, datasets, out):
    """A readable matrix that grows by one column, rather than thirteen bars, for each added model."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle

    has_chance = metric not in F1_METRICS
    ncols = len(models) + int(has_chance)
    values = np.full((len(datasets), ncols), np.nan)
    sampled = False
    first = {key: next(results[m][key] for m in models if key in results.get(m, {})) for key in datasets}
    for row, key in enumerate(datasets):
        for col, model in enumerate(models):
            run = results.get(model, {}).get(key)
            if run is not None and run["metrics"].get(metric) is not None:
                values[row, col] = 100 * run["metrics"][metric]

    width, height = max(11.5, 1.23 * ncols + 3.5), max(5.0, 0.44 * len(datasets) + 1.15)
    fig, ax = plt.subplots(figsize=(width, height))
    # A single light-to-dark hue keeps ordered scores readable in colour and grey print.
    # Trim the near-white/darkest endpoints; missing results remain a separate neutral grey.
    cmap = LinearSegmentedColormap.from_list(
        "overview_blues", plt.get_cmap("Blues")(np.linspace(0.06, 0.96, 256))
    ).with_extremes(bad="#f1f2f4")
    # Keep cells as vector paths in SVG rather than embedding an imshow bitmap.
    plotted = ax.pcolormesh(np.arange(ncols + 1) - 0.5, np.arange(len(datasets) + 1) - 0.5,
                           np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100)
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(len(datasets) - 0.5, -0.5)
    headers = [names[m].replace("emotion2vec+ ", "emotion2vec+\n")
               .replace("MERaLiON-", "MERaLiON\n").replace("Vox-Profile ", "Vox-Profile\n")
               for m in models]
    headers = ["\n".join(textwrap.fill(line, width=12) for line in name.splitlines()) for name in headers]
    short_headers = {"qwen2_5_omni_7b": "Qwen2.5-\nOmni 7B", "qwen2_audio_7b": "Qwen2-Audio\n7B Instruct",
                     "gemma4_e4b": "Gemma 4\nE4B", "gpt_realtime": "GPT\nRealtime",
                     "gemini-3.8-flash": "Gemini 3.8\nFlash"}
    headers = [short_headers.get(model, header) for model, header in zip(models, headers)]
    if has_chance:
        headers.append("Chance\nreference")
    ax.set_xticks(range(ncols), headers, fontsize=13)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=9)
    ax.set_yticks(range(len(datasets)), [dataset_display(key, first[key])[0] for key in datasets],
                  fontsize=14)
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
                text = f"{value:.1f}" + ("†" if part_of_training(run) else "") + ("*" if is_sample else "")
                # Compare actual contrast after linearising sRGB, rather than a score/RGB threshold.
                rgb = np.array(cmap(value / 100)[:3])
                linear_rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
                luminance = float(linear_rgb @ [0.2126, 0.7152, 0.0722])
                ink_luminance = ((17 / 255 + 0.055) / 1.055) ** 2.4  # INK: #111111
                colour = "white" if 1.05 / (luminance + 0.05) >= (luminance + 0.05) / (ink_luminance + 0.05) else INK
            ax.text(col, row, text, ha="center", va="center", fontsize=14, color=colour)
        if has_chance:
            chance = chance_of(first[key])
            ax.text(len(models), row, f"{100 * chance:.1f}" if chance is not None else "—",
                    ha="center", va="center", fontsize=14, color=INK_2)
    if has_chance:
        ax.axvline(len(models) - 0.5, color="#c8ccd1", linewidth=1.5)
    colourbar = fig.colorbar(plotted, ax=ax, fraction=0.026, pad=0.025, shrink=0.94)
    colourbar.solids.set_rasterized(False)
    colourbar.set_label(f"{METRIC_NAMES.get(metric, metric)} (%)", fontsize=19, labelpad=9)
    colourbar.ax.tick_params(labelsize=13)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=INK, linestyle="none", marker=r"$\dagger$", markersize=14,
                      label=TRAINING_LABEL),
               Line2D([], [], color="#858b94", linestyle="none", marker="_", markersize=13,
                      label="Result pending / unavailable")]
    if sampled:
        handles.append(Line2D([], [], color=INK, linestyle="none", marker=r"$*$", markersize=13,
                              label="Sampled evaluation"))
    fig.suptitle(f"{METRIC_NAMES.get(metric, metric)} across models and datasets", fontsize=21,
                 fontweight="bold", color=INK, y=0.985)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.005), borderaxespad=0, fontsize=14, columnspacing=1.8, handletextpad=0.3)
    fig.subplots_adjust(left=2.75 / width, right=1 - 0.70 / width,
                        top=1 - 1.18 / height, bottom=0.40 / height)
    return save(fig, plt, out / f"overview_{metric}")


def save(fig, plt, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix, dpi in ((".png", 300), (".svg", None)):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, bbox_inches="tight", facecolor="white",
                    **({"metadata": {"Date": None}} if suffix == ".svg" else {}))
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
    layout = panel_layout(settings.get("layout"))
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
    placeholders = settings.get("placeholders") or {}
    models, names = chart_models(results, registry, wanted_models, placeholders)
    datasets = chart_datasets(results, settings.get("dataset_order"))
    colours = {m: COLOURS[i % len(COLOURS)] for i, m in enumerate(models)}
    for m in models:
        sampled = [k for k, r in results.get(m, {}).items() if r["clips"] < r.get("total_clips", r["clips"])]
        print(f"  {names[m]:<22}: {len(results.get(m, {}))} dataset(s)" +
              (" (reserved; results pending)" if not results.get(m) else "") +
              (f" ({', '.join(sampled)} on a sample)" if sampled else ""))
    print(f"  datasets    : {', '.join(datasets)}")

    # ------------------------------------------------------------------
    # Step 3: Draw the figures, PNG (300 dpi) and SVG
    # ------------------------------------------------------------------
    step("Step 3/3 | Draw the figures")
    plt = setup_matplotlib()
    rendered = []
    for metric in metrics:
        if not any(r["metrics"].get(metric) is not None for m in models for r in results.get(m, {}).values()):
            print(f"  {metric}: no result has this metric, skipped")
            continue
        for stem in (per_dataset_figure(plt, metric, results, models, names, colours, datasets, columns, out, layout),
                     overview_figure(plt, metric, results, models, names, colours, datasets, out)):
            print(f"  {stem.relative_to(ROOT)}.png and .svg")
        rendered.append(metric)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results_manifest.json").write_text(json.dumps({"fingerprint": results_fingerprint(results),
                                                         "metrics": rendered, "models": models,
                                                         "pending_models": [m for m in models if not results.get(m)],
                                                         "dataset_order": datasets, "layout": layout,
                                                         "presentation_fingerprint": presentation_fingerprint(settings),
                                                         "style": PLOT_STYLE}, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()

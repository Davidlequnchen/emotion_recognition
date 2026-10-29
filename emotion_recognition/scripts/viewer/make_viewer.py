"""
Build a static HTML page with the results of every evaluated model and example clips the models got right and wrong.

The page has three tabs. Results (the default): a switch between all models side by side (one row per dataset, one
column per model, one metric at a time) and one model at a time (its summary tables, with the datasets in its training
data first, and its confusion matrices). Values are coloured by level: 80% or more, 60 to 80%, under 60%. Both views
read reports/<model>/summary.json. Examples: pick a dataset and a model, play a clip and compare its labels (e.g. the
acted emotion and the listeners' vote) with that model's prediction and probabilities, and with every other model's
prediction for the same clip. Datasets: the source, licence and credit of each dataset. The open tab is in the URL
hash (#results, #examples, #datasets), so a tab can be linked and the browser's back button works.
It is a single self-contained file (outputs/viewer.html; audio embedded as 16 kHz mono MP3) that opens in any browser.

The same run also writes the public version for GitHub Pages (`site`, e.g. site/index.html): the same page,
but only with the clips of datasets whose licence allows putting them on a public web page (`public_audio` in
SOURCES of scripts/common/eval_datasets.py; a dataset missing there counts as not public). The results of all
datasets are shown. Datasets in `exclude` are left out of both pages: no results, no clips.

Input, written by the evaluation, scripts/evaluation/evaluate.py:
    reports/<model>/summary.json               one model on every dataset: name, training data, metrics, figure
    outputs/<model>/<dataset>_predictions.csv  one row per clip; the .json next to it describes the run
Models come in the order of the model registry, config/models.yaml; a model folder that is not registered comes
last. New models and datasets appear automatically. If a model has a full run and a sample (`limit` in
config/evaluation.yaml) of a dataset, the page uses the one with the most clips. A predictions CSV needs:
    audio_path       audio file, relative to the emotion_recognition/ folder (the same in every model's CSV)
    prediction       the model's answer; other columns whose names start with "prediction" (e.g.
                     prediction_all_classes) are shown in the details
    correct...       one or more True/False columns whose names start with "correct", e.g. correct and
                     correct_acted_or_listener; the first one decides which clips count as right or wrong
                     when picking examples, and the page can switch between them
    <ground truth>   the column named by "ground_truth" in the JSON file next to the CSV
    p_<class>        (optional) the model's probability for each class
Any other column (e.g. listener_vote) is shown as an extra label of the clip. The JSON file next to the CSV gives
"dataset", "key", "model", "ground_truth" and, optionally, "in_training_data", "correct_columns" (a display name for
each correct... column), "columns" (a description of each column) and "accept" (composite labels, e.g.
{"excited": ["happy", "surprised"]}: a ground truth of excited also matches a prediction of happy or surprised).

Example clips are shared by the models. For each dataset, every model with results on it adds up to `correct`
clips it got right and `wrong` clips it got wrong, at random and spread over the ground-truth emotions, and the
page shows every model's prediction for each of these clips. All models take their clips from the same shuffled list
of the dataset's clips, so their picks overlap and the page stays small.

Usage (from the emotion_recognition/ folder):
    python scripts/viewer/make_viewer.py
Settings (which predictions, datasets to leave out, how many clips, seed, linked or embedded audio, output files)
are in config/viewer.yaml; `site`, `exclude`, `correct` and `wrong` above are among them.
"""
import base64
import csv
import functools
import io
import json
import os
import random
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import yaml
from PIL import Image, features

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # eval_datasets.py is in scripts/common/
from eval_datasets import SOURCES

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "viewer.yaml"         # this script's settings
MODELS_CONFIG = ROOT / "config" / "models.yaml"  # the model registry (the order of the models)
TEMPLATE = ROOT / "scripts" / "viewer" / "templates" / "viewer.html"
SR = 16000  # sampling rate of the embedded clips
MP3 = "MP3" in sf.available_formats()  # libsndfile 1.1+ writes MP3; without it the clips are embedded as WAV
MP3_LEVEL = 0.8  # soundfile compression level, 0 (best) to 1 (smallest): 0.8 = a constant 40 kbit/s at 16 kHz


def step(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def rel(path):
    """A path as printed: relative to the project folder when it is inside it."""
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def embed_image(path, width=None):
    """A figure as a data URI: lossless WebP (the same pixels, about 2.5 times smaller than the PNG), or a PNG if this
    Pillow has no WebP; width: shrink a wider figure to this many pixels first (the bar plots are drawn at 300 dpi)."""
    image = Image.open(path)
    if width and image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    elif not features.check("webp"):
        return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"
    buf = io.BytesIO()
    image.save(buf, "WEBP" if features.check("webp") else "PNG", lossless=True)
    kind = "webp" if features.check("webp") else "png"
    return f"data:image/{kind};base64,{base64.b64encode(buf.getvalue()).decode()}"


def load_plots():
    """The bar plots of scripts/viewer/plot_results.py (metrics and folder from config/plots.yaml), embedded at 3,000
    pixels wide: [{metric, name, by_dataset, overview}] for every metric whose figures exist."""
    settings = yaml.safe_load((ROOT / "config" / "plots.yaml").read_text()) or {}
    folder, plots = ROOT / settings.get("out", "reports/figures"), []
    names = {"accuracy": "Accuracy", "unweighted_accuracy": "Unweighted accuracy", "macro_f1": "Macro F1",
             "accuracy_all_classes": "Accuracy, all model classes",
             "accuracy_acted_or_listener": "Accuracy, acted or listeners' vote"}
    for metric in settings.get("metrics", []):
        grid, overview = folder / f"{metric}_by_dataset.png", folder / f"overview_{metric}.png"
        if grid.is_file():
            plots.append({"metric": metric, "name": names.get(metric, metric), "by_dataset": embed_image(grid, 3000),
                          "overview": embed_image(overview, 3000) if overview.is_file() else None})
    return plots


def load_summary(path, exclude):
    """reports/<model>/summary.json without the excluded datasets, its confusion-matrix figure embedded."""
    summary = read_json(path)
    hidden = {d["dataset"] for d in summary["datasets"] if d["key"] in exclude}  # display names, as in the figure
    summary["datasets"] = [d for d in summary["datasets"] if d["key"] not in exclude]
    summary["shared"] = [d for d in summary.get("shared", []) if d["key"] not in exclude]
    summary["acted_or_listener"] = [name for name in summary.get("acted_or_listener", []) if name not in hidden]
    figure = path.parent / summary["figure"] if summary.get("figure") else None
    if figure and hidden & set(summary.get("figure_datasets", [])):
        print(f"  WARNING     : the figure of {rel(path)} shows an excluded dataset, so the page leaves it out; "
              "run python scripts/evaluation/evaluate.py again")
        figure = None
    elif figure and not figure.is_file():
        print(f"  WARNING     : {rel(figure)} is missing, so the page has no confusion matrices for this model")
        figure = None
    summary["figure"] = embed_image(figure) if figure else None
    summary["figure_datasets"] = summary.get("figure_datasets", []) if figure else []
    return summary


def load_run(csv_path, meta):
    """One predictions CSV (meta: the .json next to it) -> the run: metadata, rows by audio path, accuracies."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"ERROR: {rel(csv_path)} has no clips. Evaluate again: python scripts/evaluation/evaluate.py")
    checks = [c for c in rows[0] if c.startswith("correct")]  # e.g. correct, correct_acted_or_listener
    missing = {"audio_path", "prediction", meta["ground_truth"]} - set(rows[0])
    if missing or not checks:
        sys.exit(f"ERROR: {rel(csv_path)} lacks {sorted(missing) or 'a correct... column'}; "
                 "see the docstring of this script.")
    names = meta.get("correct_columns", {})
    criteria = [{"column": c, "name": names.get(c, c.replace("_", " ")),
                 "accuracy": sum(r[c] == "True" for r in rows) / len(rows)} for c in checks]
    return {"meta": meta, "rows": {r["audio_path"]: r for r in rows}, "checks": checks, "criteria": criteria}


def spread(paths, emotion, n):
    """The first n of paths, taking the ground-truth emotions in turn, so that every emotion appears."""
    queues = {}
    for path in paths:
        queues.setdefault(emotion(path), []).append(path)
    chosen = []
    while len(chosen) < n and any(queues.values()):
        chosen += [queue.pop(0) for queue in queues.values() if queue][: n - len(chosen)]
    return chosen


def pick_clips(key, runs, n_correct, n_wrong, seed):
    """The example clips of one dataset (runs: model -> run): each model adds up to n_correct clips it got right and
    n_wrong it got wrong (by its first correct... column), spread over the ground-truth emotions. Every model walks
    through the same shuffled list of the dataset's clips, so each model's picks are random and the models' picks
    overlap as much as possible."""
    order = sorted(set().union(*(run["rows"] for run in runs.values())))
    random.Random(f"{seed}-{key}").shuffle(order)  # one order per dataset, whatever the other datasets
    picked, counts = set(), {}
    for model, run in runs.items():
        rows, check = run["rows"], run["checks"][0]
        emotion = lambda path: rows[path][run["meta"]["ground_truth"]]
        right = spread([path for path in order if path in rows and rows[path][check] == "True"], emotion, n_correct)
        wrong = spread([path for path in order if path in rows and rows[path][check] != "True"], emotion, n_wrong)
        picked.update(right + wrong)
        counts[model] = (len(right), len(wrong))
    return picked, counts


def make_clip(path, runs):
    """One example clip: its labels (the dataset's, e.g. acted_emotion and listener_vote) and every model's
    prediction, correctness and probabilities (a model whose run lacks the clip is left out)."""
    rows = {model: run["rows"][path] for model, run in runs.items() if path in run["rows"]}
    audio = ROOT / path
    if not audio.is_file():
        sys.exit(f"ERROR: audio file not found: {path}. Download the dataset: python scripts/common/download_data.py")
    first = next(iter(rows.values()))
    return {
        "file": audio.name,
        "path": audio,  # becomes the audio source in write_page
        "duration": round(sf.info(audio).duration, 1),
        "labels": {k: v for k, v in first.items()
                   if k != "audio_path" and not k.startswith(("prediction", "correct", "p_"))},
        "models": {model: {
            "predictions": {k: v for k, v in r.items() if k.startswith("prediction")},
            "correct": {c: r[c] == "True" for c in runs[model]["checks"]},
            "probs": {k[2:]: round(float(v), 4) for k, v in r.items() if k.startswith("p_") and v},
        } for model, r in rows.items()},
    }


@functools.cache
def resampler(rate):
    """torchaudio's resampler from `rate` to 16 kHz (the one scripts/models/audio_io.py uses), built once per rate:
    building its filter is the slow part, about 0.2 s for the 24,414 Hz of TESS and MESS."""
    return torchaudio.transforms.Resample(rate, SR)


@functools.cache
def embedded_audio(path):
    """A clip as a data URI: 16 kHz mono MP3 at a constant 40 kbit/s (clear speech, about 7 times smaller than 16-bit
    WAV), or 16 kHz mono 16-bit WAV if libsndfile cannot write MP3."""
    wav, rate = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(wav.mean(axis=1))  # mono
    wav = (resampler(rate)(wav) if rate != SR else wav).numpy()
    buf = io.BytesIO()
    if MP3:
        sf.write(buf, wav, SR, format="MP3", bitrate_mode="CONSTANT", compression_level=MP3_LEVEL)
    else:
        sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return f"data:audio/{'mpeg' if MP3 else 'wav'};base64,{base64.b64encode(buf.getvalue()).decode()}"


def write_page(path, data, datasets, link_audio):
    """Fill the page template. Audio is embedded as data URIs (the page works anywhere, even sent to someone
    else), or with link_audio linked by a path relative to the page (small file, works only in this repository)."""
    def source(audio):
        return Path(os.path.relpath(audio, path.parent)).as_posix() if link_audio else embedded_audio(audio)
    page = data | {"datasets": [ds | {"clips": [{k: v for k, v in clip.items() if k != "path"} | {"audio": source(clip["path"])}
                                                for clip in ds["clips"]]} for ds in datasets]}
    # Every "<" written as its JSON escape (backslash, u003c) keeps the JSON from closing its <script> tag early
    text = json.dumps(page, separators=(",", ":")).replace("<", "\\u003c")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", text), encoding="utf-8")
    clips = sum(len(ds["clips"]) for ds in datasets)
    print(f"  page        : {rel(path)} ({path.stat().st_size / 1e6:.1f} MB, {clips} clips of {len(datasets)} dataset(s))")


def main():
    # Step 1: settings, config/viewer.yaml; the model order comes from the model registry, config/models.yaml
    step("Step 1/5 | Load configuration")
    settings = yaml.safe_load(CONFIG.read_text())
    n_correct, n_wrong, link_audio = settings["correct"], settings["wrong"], settings["link_audio"]
    seed = settings["seed"] if settings["seed"] is not None else random.randrange(1_000_000)  # printed, to rebuild the page
    exclude = set(settings["exclude"] or [])  # left out of both pages: no results, no clips
    out = (ROOT / settings["out"]).resolve()
    site = (ROOT / settings["site"]).resolve() if settings["site"] else None
    registry = list(yaml.safe_load(MODELS_CONFIG.read_text()) or {})
    print(f"  config      : {CONFIG}")
    print(f"  clips       : up to {n_correct} right and {n_wrong} wrong per dataset and model (correct, wrong)")
    print(f"  seed        : {seed} (to get the same clips again, set `seed: {seed}` in config/viewer.yaml)")
    print(f"  excluded    : {', '.join(sorted(exclude)) or 'none'} (exclude)")
    print(f"  pages       : {rel(out)}" + (f", and {rel(site)} for the public" if site else ""))

    # Step 2: find every model (a folder with a summary or predictions), in registry order; read each one's summary
    # (name, training data, metrics, figure) without the excluded datasets
    step("Step 2/5 | Read the summary of every model (reports/<model>/summary.json)")
    if settings["predictions"] == "all":
        files = sorted((ROOT / "outputs").glob("*/*_predictions.csv"))  # not outputs/archive/<model>/ (one level deeper)
    else:
        files = [ROOT / f for f in settings["predictions"]]
    metas = {}  # predictions CSV -> the JSON next to it, for the finished runs of datasets that are not excluded
    for f in files:
        if not f.is_file():
            sys.exit(f"ERROR: {rel(f)} (`predictions` in config/viewer.yaml) does not exist")
        meta = read_json(f.with_suffix(".json")) if f.with_suffix(".json").is_file() else {}
        if "metrics" not in meta:  # the evaluation writes the JSON, with the metrics, after the CSV
            print(f"  skipped     : {rel(f)}, no finished .json next to it (a run in progress?)")
        elif meta["key"] not in exclude:
            metas[f] = meta | {"model": meta.get("model") or f.parent.name}
    found = ({f.parent.name for f in (ROOT / "reports").glob("*/summary.json")}
             | {meta["model"] for meta in metas.values()}) - {"archive"}
    if not found:
        sys.exit("ERROR: no results found. Run an evaluation first: python scripts/evaluation/evaluate.py")
    models = [m for m in registry if m in found] + sorted(found - set(registry))  # unregistered model folders last
    page_models, order = [], []  # order: every dataset with results, in order of first appearance
    for model in models:
        path = ROOT / "reports" / model / "summary.json"
        summary = load_summary(path, exclude) if path.is_file() else None
        meta = next((meta for meta in metas.values() if meta["model"] == model), {})
        name = (summary or {}).get("name") or meta.get("model_name") or model
        source = (summary or {}).get("training_source") or meta.get("training_source") or ""
        page_models.append({"key": model, "name": name, "training_source": source, "summary": summary})
        if summary:
            order += [d["key"] for d in summary["datasets"] if d["key"] not in order]
            trained = [d["dataset"] for d in summary["datasets"] if d.get("in_training_data")]
            print(f"  {name:<20}: {len(summary['datasets'])} datasets; in its training data: {', '.join(trained) or 'none'}; "
                  f"confusion matrices: {'yes' if summary['figure'] else 'none'}")
        else:
            print(f"  {name:<20}: no summary yet (run python scripts/evaluation/evaluate.py), so example clips only")
    names = {m["key"]: m["name"] for m in page_models}

    # Step 3: the predictions; one run per model and dataset, the one with the most clips (a full run, not a sample)
    step("Step 3/5 | Read the predictions (outputs/<model>/)")
    best = {}  # (model, dataset key) -> (clips, csv)
    for f, meta in metas.items():
        run, clips = (meta["model"], meta["key"]), meta.get("clips", 0)
        if run not in best or clips > best[run][0]:
            best[run] = (clips, f)
    runs = {}  # dataset key -> {model: run}, models in page order
    for (model, key), (_, f) in sorted(best.items(), key=lambda item: (models.index(item[0][0]), item[0][1])):
        run = load_run(f, metas[f])
        runs.setdefault(key, {})[model] = run
        accuracies = ", ".join(f"{100 * c['accuracy']:.1f}% ({c['name']})" for c in run["criteria"])
        print(f"  run         : {run['meta']['dataset']} / {names[model]}, {len(run['rows']):,} clips, accuracy {accuracies}")
    order += sorted(k for k in runs if k not in order)

    # Step 4: the example clips of each dataset, shared by its models
    step("Step 4/5 | Pick example clips (shared by the models)")
    datasets = []
    for key in [k for k in order if k in runs]:
        picked, counts = pick_clips(key, runs[key], n_correct, n_wrong, seed)
        first = next(iter(runs[key].values()))["meta"]
        clips = sorted((make_clip(p, runs[key]) for p in picked),
                       key=lambda c: (c["labels"].get(first["ground_truth"], ""), c["file"]))  # by ground truth, then file
        datasets.append({
            "key": key, "dataset": first["dataset"], "language": first.get("language", ""),
            "ground_truth": first["ground_truth"], "accept": first.get("accept") or {},
            "runs": {model: {"total": len(run["rows"]), "total_clips": run["meta"].get("total_clips") or len(run["rows"]),
                             "criteria": run["criteria"], "in_training_data": run["meta"].get("in_training_data"),
                             "columns": run["meta"].get("columns", {})} for model, run in runs[key].items()},
            "clips": clips,
        })
        picks = ", ".join(f"{names[model]} {right} + {wrong}" for model, (right, wrong) in counts.items())
        print(f"  {first['dataset']:<11} : {len(clips)} clips (right + wrong per model: {picks})")

    # Step 5: the page for this machine, then the public one (only clips that may be shared; audio always embedded)
    step("Step 5/5 | Write the pages")
    embedded = "16 kHz mono MP3, 40 kbit/s" if MP3 else "16 kHz mono 16-bit WAV (this libsndfile cannot write MP3)"
    print("  audio       : " + (f"linked to the original files (link_audio); the public page embeds {embedded}"
                                if link_audio else f"embedded as {embedded}"))
    sources = {k: SOURCES[k] for k in order if k in SOURCES}
    plots = load_plots()  # the bar plots of plot_results.py, if they have been drawn
    print(f"  bar plots   : {', '.join(p['metric'] for p in plots) or 'none (run scripts/viewer/plot_results.py)'}")
    data = {"seed": seed, "picks": {"correct": n_correct, "wrong": n_wrong}, "models": page_models, "sources": sources,
            "plots": plots}
    write_page(out, data, datasets, link_audio)
    if site:
        public = [d for d in datasets if SOURCES.get(d["key"], {}).get("public_audio")]
        write_page(site, data, public, link_audio=False)
        left_out = [d["dataset"] for d in datasets if d not in public]
        if left_out:
            print(f"                no clips of {', '.join(left_out)} on the public page: the licence does not allow it")
    print("Done. Open the page in a browser.")


if __name__ == "__main__":
    main()

"""
Build a static HTML page with the evaluation results and random clips a model got right and wrong.

The page has three tabs. Results (the default): the summary of every evaluated dataset (the tables, with each
dataset's language, ground-truth labels and whether it is in emotion2vec+'s training data, and the confusion matrices
of reports/<model>_summary.md, read from the .json next to it); clicking a dataset opens its examples. Examples:
play each example clip and compare its labels (e.g. the acted emotion and the listeners' vote) with the model's
prediction and probabilities. Datasets: the source, licence and credit of each dataset. The open tab is in the URL
hash (#results, #examples, #datasets), so a tab can be linked and the browser's back button works.
It is a single self-contained file (outputs/viewer.html, audio embedded) that opens in any browser.

The same run also writes the public version for GitHub Pages (viewer.site, e.g. site/index.html): the same page,
with embedded audio, but only with the clips of datasets whose licence allows putting them on a public web page
(`public_audio` in SOURCES of scripts/common/eval_datasets.py; ESD's does not). The results of all datasets
are shown.

Input: the per-clip predictions written by an evaluation script, scripts/emotion2vec/evaluate.py.
Every predictions CSV becomes one "run" (one dataset + one model) that you can switch between in
the page, so new datasets and models appear automatically. If a dataset and model have a full run and a sample
(evaluation.limit), the page uses the one with the most clips. A predictions CSV needs:
    audio_path       audio file, relative to the emotion_recognition/ folder
    prediction       the model's answer
    correct...       one or more True/False columns whose names start with "correct", e.g. correct and
                     correct_acted_or_listener; the first one decides which clips count as right or wrong
                     when picking examples, and the page can switch between them
    <ground truth>   the column named by "ground_truth" in the JSON file next to the CSV
    p_<class>        (optional) the model's probability for each class
Any other column (e.g. listener_vote) is shown as an extra label. The JSON file next to the CSV
gives "dataset", "model", "ground_truth" and, optionally, "correct_columns" (a display name for
each correct... column), "columns" (a description of each column) and "accept" (composite labels, e.g.
{"excited": ["happy", "surprised"]}: a ground truth of excited also matches a prediction of happy or surprised).

Usage (from the emotion_recognition/ folder):
    python scripts/viewer/make_viewer.py
Settings (which predictions, datasets to leave out, how many clips, seed, linked or embedded audio, output files)
are in the `viewer` section of config/config.yaml.
"""
import base64
import csv
import json
import os
import random
import sys
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # eval_datasets.py is in scripts/common/
from eval_datasets import SOURCES

ROOT = Path(__file__).resolve().parents[2]  # project root (emotion_recognition/)
CONFIG = ROOT / "config" / "config.yaml"
TEMPLATE = ROOT / "scripts" / "viewer" / "templates" / "viewer.html"
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


def load_run(csv_path, n_correct, n_wrong, rng):
    """One predictions CSV -> one run: its metadata, accuracies and randomly chosen correct and wrong clips."""
    meta = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    checks = [c for c in rows[0] if c.startswith("correct")]  # e.g. correct, correct_acted_or_listener
    missing = {"audio_path", "prediction", meta["ground_truth"]} - set(rows[0])
    if missing or not checks:
        sys.exit(f"ERROR: {csv_path.name} lacks {sorted(missing) or 'a correct... column'}; see the docstring of this script.")
    names = meta.get("correct_columns", {})
    criteria = [{"column": c, "name": names.get(c, c.replace("_", " ")),
                 "accuracy": sum(r[c] == "True" for r in rows) / len(rows)} for c in checks]

    right = [r for r in rows if r[checks[0]] == "True"]
    wrong = [r for r in rows if r[checks[0]] != "True"]
    examples = []
    for r in rng.sample(right, min(n_correct, len(right))) + rng.sample(wrong, min(n_wrong, len(wrong))):
        audio = ROOT / r["audio_path"]
        examples.append({
            "file": audio.name,
            "path": audio,  # becomes the audio source in write_page
            "duration": round(sf.info(audio).duration, 1),
            "correct": {c: r[c] == "True" for c in checks},
            "labels": {k: v for k, v in r.items() if k != "audio_path" and k not in checks and not k.startswith("p_")},
            "probs": {k[2:]: float(v) for k, v in r.items() if k.startswith("p_")},
        })
    return {**meta, "total": len(rows), "criteria": criteria, "examples": examples}


def load_summaries():
    """The summary of each model (reports/<model>_summary.json, written by scripts/emotion2vec/evaluate.py), figure
    embedded."""
    summaries = []
    for f in sorted((ROOT / "reports").glob("*_summary.json")):
        summary = json.loads(f.read_text(encoding="utf-8"))
        if summary.get("figure"):
            png = (ROOT / "reports" / summary["figure"]).read_bytes()
            summary["figure"] = f"data:image/png;base64,{base64.b64encode(png).decode()}"
        summaries.append(summary)
    return summaries


def write_page(path, runs, summaries, seed, link_audio):
    """Fill the page template. Audio is embedded as data URIs (the page works anywhere, even sent to someone
    else), or with link_audio linked by a path relative to the page (small file, works only in this repository)."""
    def source(audio):
        if link_audio:
            return Path(os.path.relpath(audio, path.parent)).as_posix()
        return f"data:{AUDIO_TYPES[audio.suffix.lower()]};base64,{base64.b64encode(audio.read_bytes()).decode()}"
    page_runs = [run | {"examples": [{k: v for k, v in ex.items() if k != "path"} | {"audio": source(ex["path"])}
                                     for ex in run["examples"]]} for run in runs]
    keys = {r["key"] for r in runs if "key" in r} | {d["key"] for s in summaries for d in s["datasets"]}
    sources = {k: SOURCES[k] for k in sorted(keys) if k in SOURCES}
    # "<\/" keeps the JSON from closing its <script> tag early
    data = json.dumps({"seed": seed, "runs": page_runs, "summaries": summaries, "sources": sources})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data.replace("</", "<\\/")),
                    encoding="utf-8")
    print(f"  page        : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} "
          f"({path.stat().st_size / 1e6:.1f} MB, clips of {len(runs)} dataset(s))")


def main():
    # Settings: the `viewer` section of config/config.yaml
    settings = yaml.safe_load(CONFIG.read_text())["viewer"]
    n_correct, n_wrong, link_audio = settings["correct"], settings["wrong"], settings["link_audio"]
    seed = settings["seed"] if settings["seed"] is not None else random.randrange(1_000_000)  # printed, to rebuild the page
    out = (ROOT / settings["out"]).resolve()
    site = (ROOT / settings["site"]).resolve() if settings["site"] else None
    if settings["predictions"] == "all":
        best = {}  # one run per dataset and model: the one with the most clips (a full run rather than a sample)
        for f in (ROOT / "outputs").glob("*_predictions.csv"):
            meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
            run, clips = (meta["dataset"], meta["model"]), meta.get("clips", 0)
            if run not in best or clips > best[run][0]:
                best[run] = (clips, f)
        files = sorted(f for _, f in best.values())
    else:
        files = [ROOT / f for f in settings["predictions"]]
    # viewer.exclude: datasets whose clips stay off both pages
    exclude = settings["exclude"] or []
    files = [f for f in files if json.loads(f.with_suffix(".json").read_text(encoding="utf-8")).get("key") not in exclude]
    if not files:
        sys.exit("ERROR: no predictions found. Run an evaluation first: python scripts/emotion2vec/evaluate.py")

    print("Building the example viewer")
    if exclude:
        print(f"  excluded    : {', '.join(exclude)} (viewer.exclude)")
    print(f"  seed        : {seed} (to get the same clips again, set `viewer.seed: {seed}` in config/config.yaml)")
    rng = random.Random(seed)
    runs = []
    for f in files:
        run = load_run(f.resolve(), n_correct, n_wrong, rng)
        first = run["criteria"][0]["column"]
        shown = sum(ex["correct"][first] for ex in run["examples"])
        accuracies = ", ".join(f"{100 * c['accuracy']:.1f}% ({c['name']})" for c in run["criteria"])
        print(f"  run         : {run['dataset']} / {run['model']}, {run['total']:,} clips, accuracy {accuracies}")
        print(f"                showing {shown} correct + {len(run['examples']) - shown} wrong ({run['criteria'][0]['name']})")
        runs.append(run)

    summaries = load_summaries()
    print(f"  summary     : {', '.join(s['model'] for s in summaries) or 'none yet (run scripts/emotion2vec/evaluate.py)'}")

    # The page for this machine, then the public one: only clips that may be shared, audio always embedded
    write_page(out, runs, summaries, seed, link_audio)
    if site:
        public = [r for r in runs if SOURCES.get(r.get("key"), {}).get("public_audio")]
        left_out = [r["dataset"] for r in runs if r not in public]
        write_page(site, public, summaries, seed, link_audio=False)
        if left_out:
            print(f"                no clips of {', '.join(left_out)}: its licence does not allow a public page")
    print("Done. Open the page in a browser.")


if __name__ == "__main__":
    main()

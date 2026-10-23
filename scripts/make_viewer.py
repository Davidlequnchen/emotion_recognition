"""
Build a static HTML page that shows random clips a model got right and wrong.

The page lets you play each clip and compare its labels (e.g. the acted emotion and the listeners'
vote) with the model's prediction and probabilities. It is a single self-contained file
(outputs/viewer.html, audio embedded) that opens in any browser.

Input: the per-clip predictions written by an evaluation script, e.g. scripts/evaluate_crema_d.py.
Every predictions CSV becomes one "run" (one dataset + one model) that you can switch between in
the page, so new datasets and models appear automatically. A predictions CSV needs:
    audio_path       audio file, relative to the repository root
    prediction       the model's answer
    correct...       one or more True/False columns whose names start with "correct", e.g. correct and
                     correct_acted_or_listener; the first one decides which clips count as right or wrong
                     when picking examples, and the page can switch between them
    <ground truth>   the column named by "ground_truth" in the JSON file next to the CSV
    p_<class>        (optional) the model's probability for each class
Any other column (e.g. listener_vote) is shown as an extra label. The JSON file next to the CSV
gives "dataset", "model", "ground_truth" and, optionally, "correct_columns" (a display name for
each correct... column) and "columns" (a description of each column).

Usage:
    python scripts/make_viewer.py                      # counts and seed from config/config.yaml
    python scripts/make_viewer.py --correct 5 --wrong 15 --seed 42
    python scripts/make_viewer.py --predictions outputs/crema_d_emotion2vec_plus_large_predictions.csv
"""
import argparse
import base64
import csv
import json
import os
import random
import sys
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "templates" / "viewer.html"
AUDIO_TYPES = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


def load_run(csv_path, n_correct, n_wrong, rng, out, link_audio):
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
        if link_audio:  # a path relative to the page: small file, but only works inside this repository
            src = Path(os.path.relpath(audio, out.parent)).as_posix()
        else:           # embedded: the page works anywhere, even sent to someone else
            src = f"data:{AUDIO_TYPES[audio.suffix.lower()]};base64,{base64.b64encode(audio.read_bytes()).decode()}"
        examples.append({
            "file": audio.name,
            "audio": src,
            "duration": round(sf.info(audio).duration, 1),
            "correct": {c: r[c] == "True" for c in checks},
            "labels": {k: v for k, v in r.items() if k != "audio_path" and k not in checks and not k.startswith("p_")},
            "probs": {k[2:]: float(v) for k, v in r.items() if k.startswith("p_")},
        })
    return {**meta, "total": len(rows), "criteria": criteria, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "config" / "config.yaml", help="project config file")
    parser.add_argument("--predictions", nargs="+", type=Path,
                        help="predictions CSV file(s) (default: every outputs/*_predictions.csv)")
    parser.add_argument("--correct", type=int, help="correct clips to show per run (overrides the config)")
    parser.add_argument("--wrong", type=int, help="wrong clips to show per run (overrides the config)")
    parser.add_argument("--seed", type=int, help="random seed; the same seed gives the same clips (overrides the config)")
    parser.add_argument("--link-audio", action="store_true",
                        help="link to the audio files instead of embedding them (smaller page, works only in this repository)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "viewer.html", help="output HTML file")
    args = parser.parse_args()

    # Settings: command line first, then the `viewer` section of the config, then defaults
    cfg = yaml.safe_load(Path(args.config).read_text()).get("viewer", {})
    n_correct = args.correct if args.correct is not None else cfg.get("correct", 10)
    n_wrong = args.wrong if args.wrong is not None else cfg.get("wrong", 10)
    seed = args.seed if args.seed is not None else cfg.get("seed")
    seed = seed if seed is not None else random.randrange(1_000_000)  # printed, so the page can be rebuilt
    files = args.predictions or sorted((ROOT / "outputs").glob("*_predictions.csv"))
    if not files:
        sys.exit("ERROR: no predictions found. Run an evaluation first: python scripts/evaluate_crema_d.py")

    print("Building the example viewer")
    print(f"  seed        : {seed} (rebuild the same page with --seed {seed})")
    rng = random.Random(seed)
    args.out = args.out.resolve()
    runs = []
    for f in files:
        run = load_run(f.resolve(), n_correct, n_wrong, rng, args.out, args.link_audio)
        first = run["criteria"][0]["column"]
        shown = sum(ex["correct"][first] for ex in run["examples"])
        accuracies = ", ".join(f"{100 * c['accuracy']:.1f}% ({c['name']})" for c in run["criteria"])
        print(f"  run         : {run['dataset']} / {run['model']}, {run['total']:,} clips, accuracy {accuracies}")
        print(f"                showing {shown} correct + {len(run['examples']) - shown} wrong ({run['criteria'][0]['name']})")
        runs.append(run)

    # Put the data into the page template; "<\/" keeps the JSON from closing its <script> tag early
    data = json.dumps({"seed": seed, "runs": runs}).replace("</", "<\\/")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data), encoding="utf-8")
    print(f"Done: {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} "
          f"({args.out.stat().st_size / 1e6:.1f} MB). Open it in a browser.")


if __name__ == "__main__":
    main()

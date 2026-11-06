"""
Informal check after a change of the Python environment (e.g. a new transformers or torchvision): do the speech
emotion models still load and give the predictions stored in outputs/<model>/crema_d_predictions.csv?

For each model in MODELS, one at a time: load it through its loader on the device that evaluate.py would use on the
Mac (devices.mac in config/evaluation.yaml, or the model's mac_device), run it on the first CLIPS clips of the
stored CREMA-D predictions, and compare the most probable class and the largest probability difference with the CSV
(which rounds to 4 decimals, so differences up to about 1e-4 are rounding). Nothing is written.

Usage (from the emotion_recognition/ folder; edit the constants below):
    python test_scripts/check_existing_models.py
"""
import csv
import gc
import importlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml

PROJECT = Path(__file__).resolve().parents[1]  # emotion_recognition/
MODELS = ["emotion2vec_plus_large", "meralion_ser_v1", "melati_emotion", "vox_profile_whisper", "vox_profile_wavlm"]
CLIPS = 3  # the first N clips of each model's stored CREMA-D predictions

sys.path.insert(0, str(PROJECT / "scripts" / "models"))
sys.path.insert(0, str(PROJECT / "scripts" / "common"))


def main():
    sys.stdout.reconfigure(line_buffering=True)
    registry = yaml.safe_load((PROJECT / "config" / "models.yaml").read_text())
    mac = yaml.safe_load((PROJECT / "config" / "evaluation.yaml").read_text())["devices"]["mac"]
    print(f"Step 1 | torch {torch.__version__}, transformers {transformers.__version__}")
    failed = []
    for key in MODELS:
        entry = registry[key]
        device = torch.device(entry.get("mac_device", mac))
        rows = list(csv.DictReader(open(PROJECT / "outputs" / key / "crema_d_predictions.csv", newline="")))[:CLIPS]
        print(f"\nStep 2 | {entry['name']} on {device}")
        loader = importlib.import_module(entry["loader"])
        args = {"entry": entry | {"key": key}} if "entry" in inspect.signature(loader.load).parameters else {}
        predict = loader.load(PROJECT / entry["dir"], device, **args)
        for row in rows:
            probs = np.asarray(predict(str(PROJECT / row["audio_path"])))
            stored = np.array([float(row[f"p_{label.strip('<>')}"]) for label in predict.labels])
            top = predict.labels[int(probs.argmax())]
            diff = float(np.abs(probs - stored).max())
            same = top == row["prediction_all_classes"] and diff < 2e-3
            failed += [] if same else [f"{key}: {row['audio_path']}"]
            print(f"  {Path(row['audio_path']).name:<24} stored {row['prediction_all_classes']:<10} now {top:<10} "
                  f"largest probability difference {diff:.1e} {'OK' if same else 'DIFFERENT'}")
        del predict
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    print(f"\nStep 3 | {'all the same' if not failed else 'DIFFERENT: ' + ', '.join(failed)}")


if __name__ == "__main__":
    main()

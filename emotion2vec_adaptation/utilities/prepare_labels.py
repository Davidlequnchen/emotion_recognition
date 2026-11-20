"""Join normalized Label Studio decisions to clip paths; settings in config/labels.yaml."""
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utilities.config import load_settings, resolve_path
from adaptation.data import build_labeled_manifest

if __name__ == "__main__":
    cfg = load_settings("labels")
    settings = cfg["labels_import"]
    summary = build_labeled_manifest(resolve_path(settings["votes"]), resolve_path(settings["clips_manifest"]),
                                     resolve_path(settings["output_manifest"]), cfg["labels"])
    print(summary)

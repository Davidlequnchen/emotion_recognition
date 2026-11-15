#!/bin/bash
# One command to regenerate the intensity analysis, its figures and reports, and this deck from the saved
# predictions in emotion_recognition/outputs/ (for example after new Gemma 4 E4B runs). No model is run.
#   research_paper/presentation/intensity/rebuild.sh
# Steps:
#   0. refresh the shared audio inventory (research_paper/output/audio_inventory.csv) only if the selected
#      prediction files changed; this is the same function the cross-dataset analysis uses
#   1. research_paper/scripts/intensity_analysis.py: statistics, figures (intensity_plot.py), intensity_en.md / _zh.md
#   2. node build.js: intensity_analysis.pptx (stops if a font is below the size rules)
#   3. ./render.sh: render/<deck>.pdf and render/p-NN.png for checking (needs LibreOffice; waits for free memory)
# Python: the emotion_recognition_llm conda environment (override with PYTHON=...).
set -euo pipefail
DECK_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DECK_DIR/../../.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/emotion_recognition_llm/bin/python}"
cd "$REPO"

echo "== Step 0/3 | Audio inventory up to date?"
"$PYTHON" - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "research_paper/scripts")
import yaml
import consensus
from cross_dataset_analysis import load_inventory
output = Path("research_paper/output")
registry = yaml.safe_load((consensus.PROJECT / "config/models.yaml").read_text())
selected, _ = consensus.load_runs(consensus.PROJECT, registry, datasets=consensus.DATASETS, exclude=["esd"])
current = {s["path"]: s["sha256"] for runs in selected.values() for run in runs.values() for s in run.sources}
recorded = {s["path"]: s["sha256"] for s in json.loads((output / "audio_inventory_manifest.json").read_text())["prediction_sources"]}
if current == recorded:
    print("  unchanged")
else:
    print("  prediction selection changed: refreshing the shared audio inventory")
    print("  (also re-run research_paper/scripts/cross_dataset_analysis.py and the other analyses afterwards)")
    load_inventory(consensus.PROJECT, selected, output, refresh=True)
PY

echo "== Step 1/3 | Intensity analysis, figures and reports"
"$PYTHON" research_paper/scripts/intensity_analysis.py

echo "== Step 2/3 | Build the deck"
cd "$DECK_DIR"
[ -d node_modules ] || npm install --no-audit --no-fund
node build.js

echo "== Step 3/3 | Render for checking"
if command -v soffice >/dev/null && command -v pdftoppm >/dev/null; then
  ./render.sh
else
  echo "  LibreOffice or poppler missing: skipped"
fi
echo "Done. Re-read the slide titles against the new numbers (README.md, 'After a rebuild')."

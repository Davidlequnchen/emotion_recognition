#!/bin/bash
# Rebuild the whole consensus analysis and this deck from saved predictions: ./rebuild.sh
# Run in the emotion_recognition_llm environment (conda activate emotion_recognition_llm). No model, audio or API is run.
#   1. research_paper/scripts/consensus.py  CSVs, manifest, both reports and the four paper figures
#   2. scripts/make_assets.py                slide figures (fig/), assets/numbers.json, facts.md
#   3. node build.js                         consensus_analysis.pptx (stops if a font is too small)
# Optional afterwards: ./render.sh (LibreOffice previews in render/).
set -e
cd "$(dirname "$0")"
echo "== 1/3 consensus analysis"
python ../../scripts/consensus.py
echo "== 2/3 slide figures and numbers"
python scripts/make_assets.py
echo "== 3/3 deck"
[ -d node_modules ] || npm install --no-audit --no-fund
node build.js

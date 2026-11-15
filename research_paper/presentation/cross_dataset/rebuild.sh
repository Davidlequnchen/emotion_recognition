#!/bin/bash
# Regenerate everything behind the cross-dataset deck with one command, e.g. after Gemma 4 E4B finishes:
#     research_paper/presentation/cross_dataset/rebuild.sh
# 1. research_paper/scripts/cross_dataset_analysis.py reads the saved predictions (no model runs, no API calls) and
#    writes research_paper/output/cross_dataset_*, the figures in research_paper/analysis_report/figures/ and the
#    English and Chinese reports research_paper/analysis_report/cross_dataset_{en,zh}.md
# 2. scripts/make_facts.py writes assets/facts.json and facts.md (every number with its source file)
# 3. node build.js writes cross_dataset_analysis.pptx; it stops if a font is too small and prints
#    "CHECK WORDING" for any slide claim that the new numbers no longer support
# 4. render.sh writes render/ (PDF and PNG previews, hidden appendix included) for checking
set -e
DECK="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$DECK/../../.." && pwd)"
PY="${PYTHON:-/opt/homebrew/Caskroom/miniconda/base/envs/emotion_recognition_llm/bin/python}"

echo "== 1/4 cross-dataset analysis (saved predictions only)"
cd "$REPO"
# New prediction files (e.g. Gemma 4) change the run selection; the analysis then asks for a fresh audio-header
# inventory, which re-reads only the audio headers. Any other failure stops the rebuild.
if ! log=$("$PY" research_paper/scripts/cross_dataset_analysis.py 2>&1); then
  if grep -q "refresh-inventory" <<<"$log"; then
    echo "   prediction selection changed: refreshing the audio-header inventory"
    "$PY" research_paper/scripts/cross_dataset_analysis.py --refresh-inventory
  else
    echo "$log"; exit 1
  fi
else
  echo "$log" | tail -1
fi

echo "== 2/4 facts"
"$PY" "$DECK/scripts/make_facts.py" | tail -2

echo "== 3/4 deck"
cd "$DECK"
[ -d node_modules ] || npm install --silent
node build.js

echo "== 4/4 previews"
./render.sh

#!/bin/bash
# Render the deck to a PDF and PNG previews for checking: ./render.sh [deck.pptx]
# Needs LibreOffice (soffice) and poppler (pdftoppm). Output: render/<deck>.pdf and render/p-NN.png (100 dpi),
# hidden appendix slides included.
# LibreOffice needs about 1 GB of memory; on macOS the script waits while free memory is below 25%.
set -e
cd "$(dirname "$0")"
DECK="${1:-turn_segmentation.pptx}"
if command -v memory_pressure >/dev/null; then
  while free=$(memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); [ "${free:-100}" -lt 25 ]; do
    echo "free memory ${free}% < 25%, waiting 30 s"; sleep 30
  done
fi
rm -rf render && mkdir -p render
# ExportHiddenSlides: the appendix slides are hidden in the slide show; LibreOffice skips them in a PDF unless asked
soffice --headless --convert-to 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}' --outdir render "$DECK" >/dev/null 2>&1
pdftoppm -r 100 -png render/*.pdf render/p
ls render/p-*.png | wc -l

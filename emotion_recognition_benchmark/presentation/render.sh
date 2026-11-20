#!/bin/bash
# Render a pptx to PNG pages for checking: ./render.sh deck.pptx
set -e
cd "$(dirname "$0")"
rm -rf render && mkdir -p render
soffice --headless --convert-to pdf --outdir render "$1" >/dev/null 2>&1
pdftoppm -r 50 -png render/*.pdf render/p
ls render | wc -l

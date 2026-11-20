# Tile rendered pages into contact sheets: python sheet.py [first] [count] [cols]
import sys, glob
from PIL import Image, ImageDraw
pages = sorted(glob.glob("render/p-*.png"))
a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 2
sel = pages[a - 1:a - 1 + n]
w, h = Image.open(sel[0]).size
rows = (len(sel) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w + (cols + 1) * 8, rows * h + (rows + 1) * 8), "#888")
for i, f in enumerate(sel):
    im = Image.open(f); sheet.paste(im, (8 + (i % cols) * (w + 8), 8 + (i // cols) * (h + 8)))
sheet.save(f"render/sheet_{a}.png"); print(f"render/sheet_{a}.png", sheet.size)

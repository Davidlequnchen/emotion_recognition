"""Shared figure style for the emotion2vec adaptation deck (same colours as lib.js; no grey text).

Every figure is drawn at the size it gets on the slide (inches) and saved at 300 dpi, so a font size here is the
font size on the slide. The smallest text is 16 pt. Copied from preparation_tools/presentation/scripts/style.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DECK = HERE.parent                      # emotion2vec_adaptation/presentation/
IMG = DECK / "img"                      # every image the deck uses
DPI = 300

# Deck colours (lib.js). Text is always INK or white; light colours are fills only.
INK = "#1B2A41"
LINK = "#00508C"
ACCENT = "#0072B2"      # blue: published / reused
ACCENT2 = "#D55E00"     # orange: internal data, trainable
DARK_ORANGE = "#9C3D00"
TEACHER = "#6A3D9A"     # purple: EMA teacher
FILL_BLUE = "#E3EEF8"
FILL_ORANGE = "#FBE6D8"
GRID = "#C5CED9"        # grid lines only (never text)

# Calibri, as in the deck, when it is available (it ships with PowerPoint on macOS); otherwise a sans fallback.
CALIBRI = [Path("/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts") / f
           for f in ("Calibri.ttf", "Calibrib.ttf", "Calibrii.ttf", "Calibriz.ttf")]
FAMILY = "DejaVu Sans"
if all(p.is_file() for p in CALIBRI):
    for p in CALIBRI:
        fm.fontManager.addfont(str(p))
    FAMILY = "Calibri"

plt.rcParams.update({
    "font.family": FAMILY, "font.size": 18, "axes.titlesize": 19, "axes.labelsize": 18,
    "xtick.labelsize": 17, "ytick.labelsize": 18, "legend.fontsize": 17,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.minor.width": 0.9, "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "svg.fonttype": "none", "svg.hashsalt": "emotion2vec-adaptation-deck",
})


def save(fig, name):
    """Write img/<name>.png at 300 dpi with the exact figure size (no tight cropping, so slide sizes hold)."""
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / f"{name}.png", dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  wrote img/{name}.png ({w:.2f} x {h:.2f} in at {DPI} dpi)")

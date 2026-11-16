"""Shared figure style for the turn-segmentation deck (same colours as lib.js; no grey text).

Every figure is drawn at the size it gets on the slide (inches), so a font size here is the font size on the
slide. The smallest text is 16 pt; axis labels and annotations are 17-20 pt.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DECK = HERE.parent                      # preparation_tools/presentation/
TOOLS = DECK.parent                     # preparation_tools/
FIG = DECK / "fig"
ASSETS = DECK / "assets"
DPI = 300

# Deck colours (lib.js). Text is always INK or white; light colours are fills only.
INK = "#1B2A41"
ACCENT = "#0072B2"      # blue: our method, chosen cuts
ACCENT2 = "#D55E00"     # orange: alternative / rejected
GOOD = "#2E8B57"
BAD = "#C0392B"
WARN = "#B7791F"
FILL_BLUE = "#DCEAF6"
FILL_ORANGE = "#FBE3D3"
FILL_GREEN = "#DDEFE4"
FILL_RED = "#F6DCDA"
GRID = "#C9D3DE"        # grid lines only (never text)
WAVE = "#5B7FA6"        # waveform fill (not text)

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
    "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 17,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold",
})


def save(fig, name):
    """Write fig/<name>.png at 300 dpi with the exact figure size (no tight cropping, so slide sizes hold)."""
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, dpi=DPI, metadata={"Software": None})
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"  wrote fig/{name}.png ({w:.2f} x {h:.2f} in at {DPI} dpi)")

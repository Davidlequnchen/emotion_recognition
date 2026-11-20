"""Typeset the deck equations as PNG images (matplotlib mathtext, STIX fonts, dark ink), without arguments.

The images are cropped tightly and saved at 300 dpi; lib.js places them at native size, so SIZE below is the font
size on the slide (sub- and superscripts are about 70% of it). Run: python scripts/render_equations.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # files only, no window
matplotlib.rcParams.update({"mathtext.fontset": "stix"})
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

IMG = Path(__file__).resolve().parent.parent / "img"
INK = "#1B2A41"
SIZE = 30   # pt on the slide (equations)
SYM_SIZE = 24   # pt on the slide (symbols in the symbol tables; indices about 17 pt)

# Notation follows adaptation/ssl.py (forward, update_teacher) and docs/guide.md, section "当前目标的精确定义".
EQUATIONS = {
    # default target (target: final): the teacher's last block, layer-normalised per frame (no learned scale)
    "eq_final": r"$y_{b,t}=\mathrm{LN}\!\left(z^{\mathrm{T}}_{b,t}\right)$",
    # layer norm over the D channels of one frame
    "eq_ln": r"$\mathrm{LN}(z)_d=\dfrac{z_d-\bar{z}}{\sqrt{\sigma_z^2+\epsilon}}$",
    # control target (target: ffn_mean): mean over the top K shared blocks of per-clip, per-channel normalised FFN outputs
    "eq_target": r"$y_{b,t}=\dfrac{1}{K}\sum_{k=1}^{K}\,\mathrm{IN}_{\mathcal{V}_b}\!\left(h^{\mathrm{T}}_{k,b,t}\right)$",
    # instance normalisation over the clip's valid frames only (temporal_instance_normalize)
    "eq_instnorm": r"$\mathrm{IN}_{\mathcal{V}}(h)_t=\dfrac{h_t-\mu_{\mathcal{V}}}{\sqrt{\sigma^2_{\mathcal{V}}+\epsilon}}$",
    # student prediction on the masked input
    "eq_pred": r"$p_{b,t}=g_{\phi}\!\left(f_{\theta^{\mathrm{S}}}(\tilde{x}_b)_t\right)$",
    # frame loss: mean squared error over masked valid frames and channels
    "eq_loss": (r"$\mathcal{L}_{\mathrm{frame}}=\dfrac{1}{|\mathcal{M}|\,D}\sum_{(b,t)\in\mathcal{M}}\;\sum_{d=1}^{D}"
                r"\left(p_{b,t,d}-\mathrm{sg}\left(y_{b,t,d}\right)\right)^2$"),
    # total loss: frame loss, optional pooled utterance loss (utterance_weight) and optional L2-SP anchor (anchor_weight)
    "eq_total": (r"$\mathcal{L}=w\,\mathcal{L}_{\mathrm{frame}}+\lambda\,\mathcal{L}_{\mathrm{utt}}"
                 r"+\mu\,\left\|\theta^{\mathrm{S}}-\theta^{0}\right\|^2$"),
    # gradient step on student and predictor
    "eq_update": r"$\left(\theta^{\mathrm{S}},\,\phi\right)\;\leftarrow\;\mathrm{AdamW}\!\left(\nabla_{\theta^{\mathrm{S}},\,\phi}\;\mathcal{L}\right)$",
    # EMA teacher update, once per optimizer step
    "eq_ema": r"$\theta^{\mathrm{T}}\;\leftarrow\;\tau\,\theta^{\mathrm{T}}+\left(1-\tau\right)\theta^{\mathrm{S}}$",
}

# Symbols for the symbol tables (placed over the first table column by lib.js symtable())
SYMBOLS = {
    "sym_bt": r"$b,\ t$",
    "sym_h": r"$h^{\mathrm{T}}_{k,b,t}$",
    "sym_K": r"$K$",
    "sym_V": r"$\mathcal{V}_b$",
    "sym_musigma": r"$\mu_{\mathcal{V}},\ \sigma^2_{\mathcal{V}}$",
    "sym_eps": r"$\epsilon$",
    "sym_xtilde": r"$\tilde{x}_b$",
    "sym_f": r"$f_{\theta^{\mathrm{S}}}$",
    "sym_g": r"$g_{\phi}$",
    "sym_p": r"$p_{b,t}$",
    "sym_M": r"$\mathcal{M}$",
    "sym_D": r"$D$",
    "sym_sg": r"$\mathrm{sg}(\cdot)$",
    "sym_lambda": r"$\lambda\,\mathcal{L}_{\mathrm{utt}}$",
    "sym_w": r"$w$",
    "sym_mu": r"$\mu,\ \theta^{0}$",
    "sym_z": r"$z^{\mathrm{T}}_{b,t}$",
    "sym_LN": r"$\mathrm{LN}$",
    "sym_IN": r"$\mathrm{IN}_{\mathcal{V}_b}$",
    "sym_thetaTtau": r"$\theta^{\mathrm{T}},\ \tau$",
    "sym_thetaS": r"$\theta^{\mathrm{S}},\ \phi$",
    "sym_thetaT": r"$\theta^{\mathrm{T}}$",
    "sym_tau": r"$\tau$",
}


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    for table, size in ((EQUATIONS, SIZE), (SYMBOLS, SYM_SIZE)):
        for name, expression in table.items():
            target = IMG / f"{name}.png"
            math_to_image(expression, target, prop=FontProperties(size=size), dpi=300, format="png", color=INK)
            print(f"  wrote img/{name}.png")
    print(f"typeset {len(EQUATIONS)} equations at {SIZE} pt and {len(SYMBOLS)} symbols at {SYM_SIZE} pt (STIX)")


if __name__ == "__main__":
    main()

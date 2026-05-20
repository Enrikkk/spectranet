#!/usr/bin/env python3
"""DEPRECATED — do not run.

The SpectraNet architecture figure is now hand-drawn in drawio
(source: paper/SpectraNet-Architecture.drawio, export:
paper/SpectraNet-Architecture-crop.pdf, deployed: figures/architecture.pdf).
This matplotlib generator is preserved for historical reference only; running
it will overwrite the hand-drawn figure with a stale rendering that depicts
the multi-resolution head + KAN sub-layer (both removed in Session 22).
"""
import sys

sys.exit(
    "make_architecture_figure.py is deprecated. The architecture figure is now "
    "hand-drawn (paper/SpectraNet-Architecture-crop.pdf, deployed to "
    "figures/architecture.pdf). Do not regenerate."
)

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath

# ── Style ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
})

COL_INPUT  = "#ECECEC"
COL_LIFT   = "#E2E2E2"
COL_ENC    = "#D6E4F0"
COL_BN     = "#E5D4F0"
COL_DEC    = "#FCE5CD"
COL_HEAD   = "#D4EDD4"
COL_KAN    = "#E5F2E5"
COL_RES    = "#FFF2CC"
COL_OUT    = "#DDDDDD"
COL_SKIP   = "#7A7A7A"
COL_FLOW   = "#222222"
COL_AR     = "#9E2A2B"


def box(ax, x, y, w, h, text, fc, fontsize=7.0, ec="black", lw=0.7):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color=COL_FLOW, ls="-", lw=0.8):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=7,
        color=color, linewidth=lw, linestyle=ls,
        shrinkA=0.5, shrinkB=0.5,
    )
    ax.add_patch(a)


def routed_arrow(ax, points, color=COL_FLOW, ls="-", lw=0.8):
    """Polyline with an arrowhead on the last segment (waypoints in data coords)."""
    # Draw all but the last segment as plain lines.
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        if (x1, y1) == points[-2] and (x2, y2) == points[-1]:
            arrow(ax, x1, y1, x2, y2, color=color, ls=ls, lw=lw)
        else:
            ax.plot([x1, x2], [y1, y2], color=color, linestyle=ls, linewidth=lw,
                    solid_capstyle="round")


def label(ax, x, y, s, fontsize=6.5, color=COL_FLOW, style="italic",
          ha="center", va="center"):
    ax.text(x, y, s, fontsize=fontsize, color=color, style=style, ha=ha, va=va)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(-0.6, 17)
    ax.set_ylim(-0.4, 9.0)
    ax.set_aspect("equal")
    ax.axis("off")

    BW, BH = 2.2, 0.65   # main backbone box

    # ── Input + lift (top-left) ───────────────────────────────────────────
    box(ax, 0.30, 8.10, BW, BH,
        r"Input window $\omega_{t-9:t}$" "\n" r"$(B,\,64,\,64,\,10)$",
        COL_INPUT)
    box(ax, 3.10, 8.10, BW, BH,
        "Grid concat $+$ lift\n" r"$(B,\,64,\,64,\,32)$",
        COL_LIFT)
    arrow(ax, 2.50, 8.42, 3.10, 8.42)

    # ── Encoder column ────────────────────────────────────────────────────
    enc_x = 0.30
    enc_specs = [
        (6.85, "L0   SpecConv  $M{=}12$",  r"$(64{\times}64,\,32\,\mathrm{ch})$"),
        (5.75, "L1   SpecConv  $M{=}6$",   r"$(32{\times}32,\,64\,\mathrm{ch})$"),
        (4.65, "L2   SpecConv  $M{=}3$",   r"$(16{\times}16,\,128\,\mathrm{ch})$"),
    ]
    for y, line1, line2 in enc_specs:
        box(ax, enc_x, y, BW, BH, f"{line1}\n{line2}", COL_ENC)
    enc_cx = enc_x + BW / 2

    # Lift down into L0
    arrow(ax, 4.20, 8.10, 4.20, 7.75)
    arrow(ax, 4.20, 7.75, enc_cx, 7.75)
    arrow(ax, enc_cx, 7.75, enc_cx, 7.50)

    # Encoder downflow (between rows)
    arrow(ax, enc_cx, 6.85, enc_cx, 6.40)
    label(ax, enc_cx + 0.55, 6.625, r"$\downarrow 2{\times}$", fontsize=6.5)
    arrow(ax, enc_cx, 5.75, enc_cx, 5.30)
    label(ax, enc_cx + 0.55, 5.525, r"$\downarrow 2{\times}$", fontsize=6.5)

    # ── Bottleneck (bottom of U) ──────────────────────────────────────────
    bn_x, bn_y = 4.20, 3.20
    box(ax, bn_x, bn_y, BW, BH,
        r"Bottleneck   $M{=}1$" "\n" r"$(8{\times}8,\,128\,\mathrm{ch})$",
        COL_BN)
    bn_cx = bn_x + BW / 2

    # L2 → BN (straight down then right; ↓2× lives on the vertical leg)
    arrow(ax, enc_cx, 4.65, enc_cx, 3.85)
    label(ax, enc_cx + 0.55, 4.25, r"$\downarrow 2{\times}$", fontsize=6.5)
    arrow(ax, enc_cx, 3.85, bn_cx, 3.85)
    arrow(ax, bn_cx, 3.85, bn_cx, 3.20 + BH)

    # ── Decoder column ────────────────────────────────────────────────────
    dec_x = 8.10
    dec_specs = [
        (6.85, "D0   SpecConv",  r"$(64{\times}64,\,32\,\mathrm{ch})$"),
        (5.75, "D1   SpecConv",  r"$(32{\times}32,\,64\,\mathrm{ch})$"),
        (4.65, "D2   SpecConv",  r"$(16{\times}16,\,128\,\mathrm{ch})$"),
    ]
    for y, line1, line2 in dec_specs:
        box(ax, dec_x, y, BW, BH, f"{line1}\n{line2}", COL_DEC)
    dec_cx = dec_x + BW / 2

    # BN → D2 (right then up; ↑2× lives on the vertical leg)
    arrow(ax, bn_x + BW, 3.85, dec_cx, 3.85)
    arrow(ax, dec_cx, 3.85, dec_cx, 4.65)
    label(ax, dec_cx + 0.55, 4.25, r"$\uparrow 2{\times}$", fontsize=6.5)

    # Decoder upflow
    arrow(ax, dec_cx, 5.30, dec_cx, 5.75)
    label(ax, dec_cx + 0.55, 5.525, r"$\uparrow 2{\times}$", fontsize=6.5)
    arrow(ax, dec_cx, 6.40, dec_cx, 6.85)
    label(ax, dec_cx + 0.55, 6.625, r"$\uparrow 2{\times}$", fontsize=6.5)

    # ── Skip connections (additive merge), encoder → decoder ─────────────
    skip_ys = [7.175, 6.075, 4.975]
    for sy in skip_ys:
        arrow(ax, enc_x + BW, sy, dec_x, sy,
              color=COL_SKIP, ls=(0, (3, 2)), lw=0.7)
    label(ax, (enc_x + BW + dec_x) / 2, 7.45,
          r"skip $\oplus$  (additive merge)", color=COL_SKIP, fontsize=6.5)

    # ── Right-column output flow: 1×1 linear → ⊕ → ω̂_{t+1} ───────────────
    HBW, HBH = 2.4, 0.65
    head_x = 11.50
    box(ax, head_x, 6.85, HBW, HBH,
        r"$1{\times}1$ linear projection" "\n" r"$(64{\times}64,\,1\,\mathrm{ch})$",
        COL_HEAD)
    head_cx = head_x + HBW / 2

    # D0 → linear projection (direct, top of decoder)
    arrow(ax, dec_x + BW, 7.175, head_x, 7.175)

    # ⊕ residual sum, below the linear projection box
    sum_x, sum_y = head_cx, 4.65
    sum_r = 0.16
    sc = Circle((sum_x, sum_y), sum_r, facecolor="white",
                edgecolor="black", linewidth=0.7)
    ax.add_patch(sc)
    ax.text(sum_x, sum_y, r"$+$", ha="center", va="center", fontsize=8.5)

    # linear → ⊕
    arrow(ax, sum_x, 6.85, sum_x, sum_y + sum_r)

    # ω_t (last input frame): a small box just LEFT of ⊕, in the gap between
    # the decoder column (ends at x=10.30) and the head column (starts at 11.50).
    # Place above the sum so it doesn't visually collide with D2.
    omega_box_x = 10.40
    omega_box_y = 4.40
    omega_box_w = 1.05
    omega_box_h = 0.50
    box(ax, omega_box_x, omega_box_y, omega_box_w, omega_box_h,
        r"$\omega_t$",
        COL_RES, fontsize=8.0)
    label(ax, omega_box_x + omega_box_w / 2, omega_box_y - 0.18,
          r"last input frame", fontsize=5.8, style="italic", color="#555")
    arrow(ax, omega_box_x + omega_box_w, omega_box_y + omega_box_h / 2,
          sum_x - sum_r, sum_y)

    # ⊕ → ω̂_{t+1}
    box(ax, head_x, 3.20, HBW, HBH,
        r"$\widehat{\omega}_{t+1}$  (one-step prediction)",
        COL_OUT, fontsize=7.5)
    arrow(ax, sum_x, sum_y - sum_r, sum_x, 3.20 + HBH)

    # ── AR rollout feedback (output → bottom → left → up → input) ────────
    # Routed entirely outside the box columns: down from output, left along a
    # bottom rail under the diagram, up the left margin, and into the input
    # box from the left side.
    ar_y_floor = 1.55
    ar_x_left = -0.30
    input_left_x = 0.30           # left edge of input box
    input_mid_y = 8.10 + BH / 2   # mid-height of input box
    routed_arrow(ax, [
        (head_cx, 3.20),                  # bottom of output box
        (head_cx, ar_y_floor),            # drop below diagram
        (ar_x_left, ar_y_floor),          # run left along bottom rail
        (ar_x_left, input_mid_y),         # up the left margin
        (input_left_x, input_mid_y),      # enter input box from the left
    ], color=COL_AR, ls=(0, (4, 2)), lw=0.9)
    label(ax, (head_cx + ar_x_left) / 2, ar_y_floor - 0.32,
          r"AR rollout: append $\widehat{\omega}_{t+1}$, drop $\omega_{t-9}$",
          color=COL_AR, fontsize=6.7, style="italic")

    # ── Section labels ────────────────────────────────────────────────────
    ax.text(enc_cx, 2.40, "Encoder", fontsize=7.5, color="#3B5C84",
            ha="center", va="center")
    ax.text(bn_cx, 2.40, "Bottleneck", fontsize=7.5, color="#5C3B7C",
            ha="center", va="center")
    ax.text(dec_cx, 2.40, "Decoder", fontsize=7.5, color="#A05C00",
            ha="center", va="center")
    ax.text(head_cx, 2.40, "Output projection", fontsize=7.5,
            color="#3B7C3B", ha="center", va="center")

    # ── Two-step penalty footnote ────────────────────────────────────────
    label(ax, head_cx, 0.85,
          r"Semigroup-Consistency Loss: $f_\theta\!\circ\!f_\theta$ penalty at training time, $\lambda{=}0.1$",
          fontsize=6.0, color="#444", style="italic")

    out = Path(__file__).resolve().parents[1] / "figures" / "architecture.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Schematic of the ACoAtt-V2 attribute-correlation block (PAPER section 4.4).

Top panel   : the six kinematic attributes -> three physically coupled pairs ->
              bidirectional correlation -> six re-calibrated embeddings ->
              concatenation -> trajectory embedding.
Bottom panel: the correlation-attention mechanism itself, which for the pair
              (a1, a2) is  e'_a1 = LN([e_a1 ‖ MHA(q=a1, k=v=a2)])  and
              symmetrically for a2.

Difference from the FlightBERT original, shown in the labels: the meteorological
channels are attached to the VELOCITY they physically influence (wind to the
horizontal components, temperature to the vertical one) rather than being a single
global context vector.

Run:  python plot_acoat_block.py
Out:  docs/figures/acoat_v2_block.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/figures/acoat_v2_block.png")

C_PAIR = "#dbe7f5", "#3b6ea5"
C_MHA = "#fdf3d9", "#c8a23a"
C_EMB = "#e3f2e6", "#4a8f5c"
C_IN = "#f2f2f2", "#777777"


def box(ax, x, y, w, h, text, fc, ec, fs=8.0, bold=False, pad=0.5, ls="solid"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={pad},rounding_size=0.6",
                                linewidth=1.3, edgecolor=ec, facecolor=fc,
                                linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color="#1b1b1b", fontweight="bold" if bold else "normal",
            zorder=3, linespacing=1.35)


def arrow(ax, p, q, color="#555555", lw=1.3, ls="solid", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls, zorder=1,
                                 shrinkA=1.0, shrinkB=1.0,
                                 connectionstyle=f"arc3,rad={rad}"))


def panel(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.8,rounding_size=1.0",
                                linewidth=1.1, edgecolor="#bbbbbb", facecolor="#fcfcfc",
                                linestyle="--", zorder=0))
    ax.text(x + 2, y + h - 2.2, title, fontsize=9.5, fontweight="bold",
            color="#333333", va="top")


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 11.2), dpi=150)
    ax.set_xlim(0, 124)
    ax.set_ylim(0, 124)
    ax.axis("off")

    ax.text(62, 121.8, "ACoAtt-V2 — Attribute Correlation Attention (PAPER §4.4)",
            ha="center", va="center", fontsize=12.5, fontweight="bold")
    ax.text(62, 118.4, "each attribute = its Gray-Code bits ⊕ window-relative state; "
            "physically coupled pairs are re-calibrated in both directions",
            ha="center", va="center", fontsize=8.0, color="#555555")

    # ======================= top panel ======================= #
    panel(ax, 2, 58, 120, 62, "Attribute embedding and correlation")

    # six attributes; the meteo contribution is shown in the label of the velocity
    # it perturbs instead of as separate boxes (keeps the diagram unambiguous)
    attrs = [
        ("λ  (Lon)", 112.0, "pos"),
        ("vx (Vx)  + u_wind", 106.5, "vel"),
        ("φ  (Lat)", 101.0, "pos"),
        ("vy (Vy)  + v_wind", 95.5, "vel"),
        ("h  (Alt)", 90.0, "pos"),
        ("vz (Vz)  + T", 84.5, "vel"),
    ]
    for text, y, kind in attrs:
        fc, ec = ("#f7dede", "#b05a5a") if kind == "vel" else C_IN
        box(ax, 5, y, 24, 4.2, text, fc, ec, fs=7.4, bold=True)

    # three coupled pairs -> one CoAtt box each -> one re-calibrated embedding each
    pairs = [(0, 1, "λ"), (2, 3, "φ"), (4, 5, "h")]
    for (i, j, sym), ey in zip(pairs, [108.5, 97.5, 86.5]):
        y1, y2 = attrs[i][1] + 2.1, attrs[j][1] + 2.1
        cy = (y1 + y2) / 2
        box(ax, 36, cy - 3.6, 21, 7.2, "CoAtt\n(bidirectional)", *C_PAIR, fs=7.4, bold=True)
        arrow(ax, (29.6, y1), (35.0, cy + 1.6), rad=-0.22)
        arrow(ax, (29.6, y2), (35.0, cy - 1.6), rad=0.22)
        arrow(ax, (57.6, cy), (64.0, ey + 2.1))
        box(ax, 64, ey, 17, 4.2, f"e*_{sym}", *C_EMB, fs=8.0, bold=True)

    # concat -> trajectory embedding -> fusion
    arrow(ax, (81.6, 110.6), (88.0, 103.0), rad=-0.12)
    arrow(ax, (81.6, 99.6), (88.0, 101.5))
    arrow(ax, (81.6, 88.6), (88.0, 100.0), rad=0.12)
    box(ax, 88, 99.5, 13, 4.2, "concat", "#eeeeee", "#888888", fs=7.6)
    arrow(ax, (94.5, 99.4), (94.5, 95.6))
    box(ax, 88, 79.0, 13, 16.0, "", "#f6fbf7", C_EMB[1], pad=0.4)
    for k in range(4):
        box(ax, 89.5, 80.6 + k * 3.6, 10, 2.8, "", "#cfe8d6", C_EMB[1], pad=0.25)
    ax.text(94.5, 77.0, "Trajectory embedding\n[B, T, 144]", ha="center", va="top",
            fontsize=7.6, color="#2b5f3a", fontweight="bold")
    box(ax, 105, 84.5, 15, 9.0, "GELU +\nLayerNorm\nLinear 144→144", *C_IN, fs=6.9)
    arrow(ax, (101.6, 89.0), (104.4, 89.0))

    ax.text(31, 79.0, "the velocity attributes are the only ones that receive a meteo channel,\n"
            "because wind perturbs the horizontal motion and temperature the vertical one",
            ha="center", va="top", fontsize=7.0, color="#b05a5a", style="italic")

    # ======================= bottom panel ======================= #
    panel(ax, 2, 4, 120, 52, "Correlation attention mechanism (one pair, both directions)")

    rows = [
        (44.0, "a1", "a2", "e'_a1"),
        (18.0, "a2", "a1", "e'_a2"),
    ]
    for y0, self_name, other_name, out_name in rows:
        box(ax, 5, y0, 14, 5.4, f"e_{self_name}", *C_IN, fs=8.2, bold=True)
        box(ax, 25, y0 - 1.0, 26, 7.4,
            f"Multi-head Attention\nq = {self_name},  k = v = {other_name}",
            *C_MHA, fs=7.0, bold=True)
        arrow(ax, (19.6, y0 + 2.7), (24.0, y0 + 2.7))
        box(ax, 58, y0, 17, 5.4, "concat", "#eeeeee", "#888888", fs=7.6)
        arrow(ax, (51.6, y0 + 2.7), (57.0, y0 + 2.7))
        arrow(ax, (19.6, y0 + 2.7), (57.0, y0 + 4.4), color="#999999", ls="dotted", rad=0.0)
        box(ax, 82, y0 - 1.0, 22, 7.4, "LayerNorm\nLinear 2d→d", *C_IN, fs=7.2)
        arrow(ax, (75.6, y0 + 2.7), (81.0, y0 + 2.7))
        box(ax, 108, y0, 12, 5.4, out_name, *C_EMB, fs=8.2, bold=True)
        arrow(ax, (104.6, y0 + 2.7), (107.4, y0 + 2.7))

    # the cross wiring: each row reads the OTHER attribute as key/value
    arrow(ax, (12.0, 44.0), (24.0, 20.7), rad=-0.25)
    arrow(ax, (12.0, 23.4), (24.0, 46.7), rad=0.25)
    ax.text(20, 33.5, "each attribute attends\nto the other", fontsize=6.8, color="#555555",
            ha="center", va="center", style="italic")
    ax.text(66, 9.5, "dotted = the attribute's own vector, concatenated with the "
            "attention output before the projection",
            ha="center", va="center", fontsize=7.0, color="#777777", style="italic")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

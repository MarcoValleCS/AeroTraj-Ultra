# coding=utf-8
"""
Schematic of the training objective (PAPER section 4.11), in the same visual
language as the model schematic: pastel boxes per role, bit-vectors as rows of
squares, a summation node, one legend.

The binary-encoding lineage is recognised by its Hybrid Constrained Loss: a
classification term on the Gray-Code bits AND a regression term on the DECODED
trajectory, which is only possible because the decoder is differentiable.  This
draws that structure together with the two terms this project adds (the residual
on a physics prior, and the kinematic-consistency constraint) and the per-horizon
normalisation the report identifies as its most transferable trick.

Run:  python plot_loss.py
Out:  docs/figures/hybrid_loss.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from plot_style import PALETTE, arrow, bitrow, box, legend, rc

OUT = Path("docs/figures/hybrid_loss.png")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 8.6), dpi=150)
    ax.set_xlim(0, 148)
    ax.set_ylim(0, 92)
    ax.axis("off")

    ax.text(74, 89.6, "AeroTraj-Ultra — hybrid training objective",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#111111")
    ax.text(74, 86.4, "five weighted terms; the per-horizon normalisation is what stops "
            "the 300 s horizon from dominating the 5 s one",
            ha="center", va="center", fontsize=8.6, color="#555555")

    # -------------------------------------------------- model outputs (left) --
    ax.text(2, 80.5, "model outputs", ha="left", va="bottom", fontsize=9.2,
            fontweight="bold", color="#333333")
    box(ax, 2, 68.0, 38, 10.4, "Gray-Code bit logits",
        "6 attributes × Δ_h bits per horizon\n(57–58 bits with the 10 s grid)",
        role="purple", fs=8.6, fs_body=7.2)
    bitrow(ax, 4.0, 69.4, 22, cell=0.8, gap=0.26)
    box(ax, 2, 54.0, 38, 9.4, "STE-decoded trajectory",
        "straight-through decode of the same bits\n→  ŷ_gc ∈ ℝ^(B×H×6)",
        role="purple", fs=8.6, fs_body=7.2)
    box(ax, 2, 40.0, 38, 9.4, "fused prediction  ŷ",
        "ŷ = ŷ_cont + α ⊙ tanh(…) ⊙ σ_h\n→  ℝ^(B×H×6)",
        role="red", fs=8.6, fs_body=7.2)
    box(ax, 2, 26.0, 38, 9.4, "physics prior  P",
        "three learned scalars only\n→  ℝ^(B×H×6)",
        role="orange", fs=8.6, fs_body=7.2)
    for y0, y1 in [(68.0, 63.4), (54.0, 49.4), (40.0, 35.4), (26.0, 21.4)]:
        arrow(ax, (21, y0), (21, y1), color="#8a8a8a", lw=1.0)

    # ------------------------------------------------------ target (right) ----
    ax.text(108, 80.5, "target", ha="left", va="bottom", fontsize=9.2,
            fontweight="bold", color="#333333")
    box(ax, 108, 68.0, 38, 10.4, "ground truth  y",
        "the observed future state\n→  ℝ^(B×H×6)", role="green", fs=8.6, fs_body=7.2)
    box(ax, 108, 54.0, 38, 9.4, "Gray-Code encoding of y",
        "per-horizon step deltas on the\nsame lattice as the decoder",
        role="green", fs=8.6, fs_body=7.2)
    bitrow(ax, 110.0, 55.4, 22, cell=0.8, gap=0.26)
    arrow(ax, (127, 68.0), (127, 63.4), color="#8a8a8a", lw=1.0)
    ax.text(127, 52.2, "the SAME codec as the decoder:\nno train/test mismatch",
            ha="center", va="top", fontsize=7.0, color="#82b366", style="italic")

    # ------------------------------------------------------------- loss terms --
    ax.text(44, 80.5, "objective", ha="left", va="bottom", fontsize=9.2,
            fontweight="bold", color="#333333")
    terms = [
        (70.5, "L_BCE  × 0.4", "binary cross-entropy between the predicted bit logits "
         "and the Gray-Code encoding of the target Δ", "yellow", True),
        (58.5, "L_STE  × 0.25", "Smooth-L1(β=0.5) on the DECODED trajectory vs the truth "
         "— the Hybrid Constrained Loss", "yellow", True),
        (46.5, "L_pos  × 2.5", "Smooth-L1(β=0.25) on the fused ŷ, divided by "
         "σ_attr ⊙ h_norm(h),  h_norm = 0.04 + 0.96·h/300 s", "orange", False),
        (34.5, "L_vel  × 0.6", "Smooth-L1(β=0.5) on the velocities, divided by "
         "[15, 15, 1.5]", "orange", False),
        (22.5, "L_kin  × 0.25", "trapezoidal integral consistency between consecutive "
         "horizons: Δx must match ½(v_h + v_h+1)·Δt_h", "orange", False),
    ]
    for y0, title, body, role, hard in terms:
        box(ax, 44, y0, 62, 9.6, title, body, role=role, fs=8.8, fs_body=7.0,
            bold=hard)

    # two buses: outputs on the left, target on the right
    ax.plot([41.5, 41.5], [21.5, 74.5], color="#b9b9b9", lw=1.0, zorder=1)
    ax.plot([106.5, 106.5], [21.5, 74.5], color="#b9b9b9", lw=1.0, zorder=1)
    for y0, *_ in terms:
        arrow(ax, (41.5, y0 + 4.8), (43.4, y0 + 4.8), color="#b9b9b9", lw=1.0, head=8)
        arrow(ax, (106.5, y0 + 4.8), (104.6, y0 + 4.8), color="#b9b9b9", lw=1.0, head=8)
    arrow(ax, (21, 68.0), (41.5, 74.5), color="#c9c9c9", lw=1.0, head=8, rad=-0.05)
    arrow(ax, (21, 40.0), (41.5, 26.5), color="#c9c9c9", lw=1.0, head=8, rad=0.05)
    arrow(ax, (127, 68.0), (106.5, 74.5), color="#c9c9c9", lw=1.0, head=8, rad=0.05)

    # ------------------------------------------------------------- summation --
    fc, ec = rc("red")
    ax.add_patch(Circle((76, 12.0), 5.2, facecolor=fc, edgecolor=ec, lw=1.4, zorder=3))
    ax.text(76, 12.0, "Σ", ha="center", va="center", fontsize=17, color="#7a2f2c",
            fontweight="bold", zorder=4)
    for y0, *_ in terms:
        arrow(ax, (75, y0), (75.6, 17.4), color="#d8a6a4", lw=1.0, head=7)
    box(ax, 84, 5.6, 62, 12.8, "total objective",
        "L = 2.5·L_pos + 0.6·L_vel + 0.4·L_BCE + 0.25·L_STE + 0.25·L_kin\n"
        "(+ 0.5·L_hyp when the multi-hypothesis head is enabled)",
        role="red", fs=9.0, fs_body=7.4)
    arrow(ax, (81.2, 12.0), (83.4, 12.0))

    legend(ax, 118, 26.0, [
        ("inherited from the binary-encoding lineage", "yellow"),
        ("added by this project", "orange"),
        ("model output / target", "purple"),
    ], fs=7.6, title="the five terms")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

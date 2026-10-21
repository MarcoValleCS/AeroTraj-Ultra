# coding=utf-8
"""
Architecture overview figure for AeroTraj-Ultra.

Renders, as a real image, the block diagram that README section 2.1 used to carry
as ASCII art: the Gray-Code encoder -> ACoAtt-V2 -> conv stem -> Pre-LN
Transformer encoder -> attention pooling spine, the non-autoregressive
multi-horizon decoder, the physics kinematic prior, the three output heads and the
MAF-Net trust-region fusion.

Run:  python plot_architecture.py
Out:  docs/figures/architecture_overview.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path("docs/figures/architecture_overview.png")

# palette: (face, edge)
C_INP = "#f6f6f6", "#666666"
C_ENC = "#e8f0fb", "#3b6ea5"      # encoder path
C_DEC = "#e9f6ee", "#3f8f5e"      # decoder path
C_PRI = "#fdf0e3", "#c8823a"      # physics prior
C_HEAD = "#f3ecfa", "#7a4fa3"     # heads
C_OUT = "#fdecec", "#b04848"      # output


def box(ax, x, y, w, h, title, body="", fc="#ffffff", ec="#444444",
        fs_title=9.5, fs_body=7.9, lw=1.4):
    """Rounded box with the title top-aligned and the body below it."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.55,rounding_size=1.6",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                fontsize=fs_title, fontweight="bold", color="#1b1b1b", zorder=3)
        ax.text(x + w / 2, y + h - 5.2, body, ha="center", va="top",
                fontsize=fs_body, color="#3a3a3a", zorder=3, linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs_title, fontweight="bold", color="#1b1b1b", zorder=3,
                linespacing=1.5)


def arrow(ax, p, q, color="#555555", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color,
        zorder=1, shrinkA=1.0, shrinkB=1.0, connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 10.4), dpi=150)
    ax.set_xlim(0, 132)
    ax.set_ylim(-70, 133)
    ax.axis("off")

    ax.text(66, 128.5, "AeroTraj-Ultra  ·  architecture overview",
            ha="center", va="center", fontsize=13.5, fontweight="bold", color="#111111")
    ax.text(66, 124.6,
            "single forward pass for all 13 horizons (5–300 s) · no autoregressive rollout · "
            "every prediction is a residual on the physics prior",
            ha="center", va="center", fontsize=8.6, color="#555555")

    # ---------------- encoder spine ---------------- #
    sx, sw = 26, 62
    box(ax, sx, 108, sw, 13, "input  x_obs  [B, 12, 9]",
        "(lon, lat, alt, vx, vy, vz) + (u_wind, v_wind, T)  —  12 steps × 5 s = 60 s of history",
        fc=C_INP[0], ec=C_INP[1], fs_title=10)
    arrow(ax, (sx + sw / 2, 108), (sx + sw / 2, 102))

    box(ax, sx, 90, sw, 12, "Gray-Code encoder  (85-bit lattice)",
        "absolute state → Gray code  ·  19/18/12 bits for lon / lat / alt",
        fc=C_ENC[0], ec=C_ENC[1])
    arrow(ax, (sx + sw / 2, 90), (sx + sw / 2, 86))

    box(ax, sx, 72, sw, 14, "ACoAtt-V2  ·  attribute correlation attention",
        "6 attribute embeddings, cross-attention on the coupled pairs\n"
        "lon ↔ vx, lat ↔ vy, alt ↔ vz   (meteo attached to the velocity it perturbs)",
        fc=C_ENC[0], ec=C_ENC[1])
    arrow(ax, (sx + sw / 2, 72), (sx + sw / 2, 66))

    box(ax, sx, 54, sw, 12, "Conv1d stem (k=3)  +  sinusoidal PE",
        "159 → 144 channels  ·  pre-LN Transformer encoder, 2 layers, 6 heads, d = 144",
        fc=C_ENC[0], ec=C_ENC[1])
    arrow(ax, (sx + sw / 2, 54), (sx + sw / 2, 48))

    box(ax, sx, 36, sw, 12, "Attention feature aggregation",
        "softmax pooling of the 12 encoder states  →  context  c ∈ ℝ^(B×144)",
        fc=C_ENC[0], ec=C_ENC[1])
    arrow(ax, (sx + sw / 2, 36), (sx + sw / 2, 32))

    # ---------------- decoder and physics prior ---------------- #
    box(ax, sx, 16, sw, 16, "Non-autoregressive multi-horizon decoder",
        "prefix: 55-bit Gray-Code step deltas → Conv1d\n"
        "queries: horizon embedding + Fourier time features + context\n"
        "pre-LN Transformer decoder (2 layers)  →  D ∈ ℝ^(B×13×144)",
        fc=C_DEC[0], ec=C_DEC[1])
    box(ax, 96, 15, 34, 17, "Physics kinematic prior",
        "closed-form damped coordinated turn +\n"
        "damped vertical rate on WGS-84\n"
        "3 learned time constants: τ_turn, τ_vz, τ_acc\n"
        "→  P ∈ ℝ^(B×13×6)",
        fc=C_PRI[0], ec=C_PRI[1], fs_title=9.0, fs_body=7.6)
    ax.text(113, 32.6, "input: x_obs", ha="center", va="bottom", fontsize=7.4,
            color=C_PRI[1], style="italic", zorder=3)

    # ---------------- heads ---------------- #
    box(ax, 2, -14, 44, 14, "Head A  ·  Gray-Code MBC (55 logits)",
        "sigmoid → straight-through Gray-Code decode\n→  pred_gc",
        fc=C_HEAD[0], ec=C_HEAD[1])
    box(ax, 86, -14, 44, 14, "Head B  ·  continuous residual",
        "zero-initialised output layer\ntrainable δ  →  pred_cont = P + δ",
        fc=C_HEAD[0], ec=C_HEAD[1])

    arrow(ax, (sx + 14, 16), (24, 0), rad=-0.12)
    arrow(ax, (96 + 10, 13), (108, 0), rad=0.12)

    box(ax, 22, -40, 66, 15, "Head C  ·  MAF-Net trust region",
        "α = σ( MLP([D ‖ kinematic displacement]) )\n"
        "pred = pred_cont + α ⊙ tanh( (pred_gc − pred_cont) / σ_h ) ⊙ σ_h",
        fc=C_HEAD[0], ec=C_HEAD[1], fs_title=9.5)
    arrow(ax, (24, -14), (62, -25), rad=-0.10)
    arrow(ax, (108, -14), (72, -25), rad=0.10)

    box(ax, 40, -60, 30, 12, "prediction\n[B, 13, 6]",
        fc=C_OUT[0], ec=C_OUT[1], fs_title=9.5)
    arrow(ax, (55, -40), (55, -48))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
The model schematic (PAPER section 4.1), drawn in the visual language of the
published figures of the binary-encoding lineage: pastel boxes per role, dashed
stage containers, bit-vectors as rows of squares, one legend.

Data path:

  input window (12 x 5 s of ADS-B state)
    -> Gray-Code encoding of the absolute state
    -> ACoAtt-V2 attribute correlation (6 attributes, 3 coupled pairs)
    -> conv stem + pre-LN Transformer encoder -> attention pooling -> context
    -> non-autoregressive multi-horizon decoder (prefix + H horizon queries)
    -> three heads (Gray-Code, continuous residual, MAF trust region)
    -> fused prediction [B, H, 6]
  with the closed-form physics prior anchoring both residual heads.

Run:  python plot_architecture.py
Out:  docs/figures/architecture_overview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# the backend must be chosen BEFORE pyplot is imported: PGF for the LaTeX source,
# Agg for the raster that goes into the repository
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--format", choices=["png", "pgf"], default="png")
_args, _ = _ap.parse_known_args()
if _args.format == "pgf":
    from plot_style import use_pgf_backend

    use_pgf_backend()
else:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from plot_style import arrow, bitrow, box, group, legend, save_figure  # noqa: E402

OUT = Path("docs/figures/architecture_overview.png")


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.8, 11.6), dpi=150)
    ax.set_xlim(0, 152)
    ax.set_ylim(0, 122)
    ax.axis("off")

    ax.text(76, 119.6, "AeroTraj-Ultra — model schematic",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color="#111111")
    ax.text(76, 116.2, "one forward pass emits the whole horizon vector (13 points 5–300 s, "
            "or 6 points 10–300 s); every prediction is a residual on a closed-form "
            "kinematic propagator",
            ha="center", va="center", fontsize=8.8, color="#555555")

    # ------------------------------------------------------------------ input --
    group(ax, 2, 78, 33, 24, "Input")
    box(ax, 4, 92, 29, 7.0, "x_obs   [B, 12, 9]",
        "12 steps = 60 s @ 5 s, or 120 s @ 10 s", role="yellow", fs=9.0, fs_body=7.4)
    ax.text(4.6, 90.4, "longitude · latitude · altitude\nvx · vy · vz · wind · temperature",
            ha="left", va="top", fontsize=6.9, color="#6b6b6b", linespacing=1.35)
    bitrow(ax, 4.6, 83.4, 24, cell=0.78, gap=0.26)
    ax.text(4.6, 81.6, "9 channels, 12 steps", ha="left", va="top", fontsize=6.6,
            color="#8a8a8a", style="italic")

    # ------------------------------------------------------- embedding stage --
    group(ax, 39, 78, 71, 24, "Trajectory embedding  ·  Gray-Code + ACoAtt-V2")
    box(ax, 41, 92.5, 30, 7.0, "Gray-Code encoder",
        "absolute state → 85-bit lattice", role="blue", fs=8.6, fs_body=7.2)
    bitrow(ax, 42.0, 89.2, 20, cell=0.66, gap=0.22)
    arrow(ax, (71.4, 96.0), (76.6, 96.0))
    box(ax, 77, 92.5, 31, 7.0, "ACoAtt-V2  attribute correlation",
        "6 attributes · pairs lon↔vx, lat↔vy, alt↔vz", role="blue", fs=8.6, fs_body=7.2)
    box(ax, 41, 81.5, 30, 6.4, "Conv1d stem (k=3)",
        "159 → 144 channels", role="blue", fs=8.6, fs_body=7.2)
    box(ax, 77, 81.5, 31, 6.4, "pre-LN Transformer encoder",
        "2 layers · 6 heads · d = 144", role="blue", fs=8.6, fs_body=7.2)
    arrow(ax, (56, 92.5), (56, 88.2))
    arrow(ax, (71.4, 84.7), (76.6, 84.7))
    arrow(ax, (92.5, 92.5), (92.5, 88.2))

    # ----------------------------------------------------------- backbone ----
    arrow(ax, (56, 81.5), (56, 74.4))
    group(ax, 39, 55, 71, 17, "Context")
    box(ax, 41, 64.5, 30, 6.4, "attention pooling",
        "softmax over the 12 encoder states", role="blue", fs=8.6, fs_body=7.2)
    box(ax, 77, 64.5, 31, 6.4, "context   c ∈ ℝ^(B×144)",
        "one vector per window", role="teal", fs=8.6, fs_body=7.2)
    arrow(ax, (71.4, 67.7), (76.6, 67.7))
    # x_obs also feeds the prior: routed along the outside of the stage
    arrow(ax, (35.4, 95.5), (114.0, 57.0), color="#d79b00", ls="dashed", rad=-0.22)

    # --------------------------------------------- decoder + prior + heads ----
    group(ax, 2, 4, 76, 47, "Non-autoregressive multi-horizon decoder")
    box(ax, 4, 39.5, 72, 9.4, "decoder input sequence",
        "prefix: 55–58 bit Gray-Code step deltas → Conv1d   (11 tokens)\n"
        "queries: horizon embedding ⊕ Fourier time features ⊕ context   (H = 13, or 6 at 10 s)",
        role="green", fs=9.0, fs_body=7.3)
    arrow(ax, (56, 64.5), (56, 49.4))
    box(ax, 4, 28.5, 72, 7.4, "pre-LN Transformer decoder",
        "2 layers · 6 heads · self-attention over prefix + queries · "
        "cross-attention to the encoder memory",
        role="green", fs=9.0, fs_body=7.3)
    arrow(ax, (40, 39.5), (40, 36.4))
    box(ax, 4, 19.5, 72, 6.4, "decoder features   D ∈ ℝ^(B×H×144)",
        "the same representation feeds all three heads", role="teal",
        fs=9.0, fs_body=7.3)
    arrow(ax, (40, 28.5), (40, 26.4))
    box(ax, 4, 7.5, 34, 8.6, "Head A — Gray-Code MBC",
        "55 logits → straight-through\ndecode →  ŷ_gc",
        role="purple", fs=8.4, fs_body=7.2)
    box(ax, 42, 7.5, 34, 8.6, "Head B — continuous residual",
        "zero-initialised output layer\ntrainable δ → ŷ_cont = P + δ",
        role="purple", fs=8.4, fs_body=7.2)
    arrow(ax, (24, 19.5), (21, 16.6), rad=-0.1)
    arrow(ax, (56, 19.5), (59, 16.6), rad=0.1)

    group(ax, 82, 28, 68, 30, "Physics kinematic prior")
    box(ax, 84, 44.0, 64, 11.0, "closed-form damped propagator",
        "damped coordinated turn + damped vertical rate, WGS-84 metres-per-degree\n"
        "only three learned scalars:  τ_turn, τ_vz, τ_acc — everything else is analytic",
        role="orange", fs=9.0, fs_body=7.3)
    arrow(ax, (116, 44.0), (116, 40.4))
    box(ax, 84, 32.0, 64, 8.0, "P ∈ ℝ^(B×H×6)",
        "parameter-free anchor: the network starts exactly here (zero-init residual)",
        role="orange", fs=9.0, fs_body=7.3)
    arrow(ax, (116, 32.0), (116, 27.4), ls="dashed", color="#d79b00")
    arrow(ax, (116, 27.4), (80.0, 11.8), ls="dashed", color="#d79b00", rad=0.10)

    box(ax, 82, 15.5, 68, 9.6, "Head C — MAF-Net trust region",
        "α = σ( MLP([D ‖ kinematic displacement]) )\n"
        "ŷ = ŷ_cont + α ⊙ tanh( (ŷ_gc − ŷ_cont) / σ_h ) ⊙ σ_h",
        role="purple", fs=8.4, fs_body=7.2)
    arrow(ax, (38, 7.5), (88, 15.2), rad=-0.10, color="#9673a6")
    arrow(ax, (76, 7.5), (98, 15.2), rad=0.10, color="#9673a6")

    box(ax, 100, 4.0, 50, 8.4, "prediction   ŷ ∈ ℝ^(B×H×6)",
        "compared with the truth in earth-centred coordinates → 3-D ECEF MDE",
        role="red", fs=9.0, fs_body=7.2)
    arrow(ax, (116, 15.5), (125, 12.6))

    # ---------------------------------------------------------------- legend --
    legend(ax, 114, 114.0, [
        ("input / raw state", "yellow"),
        ("Gray-Code + encoder (learned)", "blue"),
        ("decoder (learned)", "green"),
        ("physics prior (analytic anchor)", "orange"),
        ("output heads and fusion", "purple"),
    ], fs=7.7, title="colour = role")
    ax.text(114, 95.0, "■ 1     □ 0", fontsize=8.2, color="#333333", va="top",
            fontweight="bold")
    ax.text(114, 92.6, "bit-vector convention", fontsize=7.0, color="#6b6b6b", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.4)
    save_figure(fig, OUT, _args.format)


if __name__ == "__main__":
    main()

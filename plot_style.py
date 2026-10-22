# coding=utf-8
"""
Shared visual language for the report's schematics.

The published figures this project is modelled on (FlightBERT Fig. 1 and the
Attribute-Correlation block) are draw.io diagrams, and their readability comes
from a consistent, small design system rather than from the drawing tool:

  * a pastel fill + darker border per ROLE (input, encoder, decoder, loss, output),
    which is what makes the eye follow the data path;
  * rounded rectangles, thin borders, no shadows;
  * dashed containers with a bold caption to group a stage;
  * thin dark-grey arrows with small solid heads;
  * bit-vectors drawn as rows of little squares (□ 0 / ■ 1), which is the visual
    signature of the binary-encoding lineage;
  * generous, grid-aligned spacing, and one legend per figure.

Everything here is matplotlib, so it renders offline and stays reproducible from
the repository.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import matplotlib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# draw.io pastel palette, as used by the reference figures
PALETTE = {
    "input":   ("#ffffff", "#666666"),
    "gray":    ("#f5f5f5", "#666666"),
    "yellow":  ("#fff2cc", "#d6b656"),
    "blue":    ("#dae8fc", "#6c8ebf"),
    "green":   ("#d5e8d4", "#82b366"),
    "orange":  ("#ffe6cc", "#d79b00"),
    "red":     ("#f8cecc", "#b85450"),
    "purple":  ("#e1d5e7", "#9673a6"),
    "teal":    ("#d5f5f5", "#3f9a9a"),
}
INK = "#1f1f1f"
ARROW = "#4a4a4a"
MUTED = "#6b6b6b"

BOXSTYLE = "round,pad=0.35,rounding_size=0.7"


def rc(key: str) -> Tuple[str, str]:
    return PALETTE[key]


def box(ax, x, y, w, h, title: str = "", body: str = "", role: str = "gray",
        fs: float = 8.5, fs_body: float = 7.6, bold: bool = True,
        lw: float = 1.2, zorder: int = 3, align: str = "center"):
    """Rounded box with an optional body line under the title."""
    fc, ec = rc(role)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=BOXSTYLE, linewidth=lw,
                                edgecolor=ec, facecolor=fc, zorder=zorder))
    cx = x + w / 2 if align == "center" else x + 1.4
    if body:
        ax.text(cx, y + h - 1.5, title, ha=align, va="top", fontsize=fs,
                color=INK, fontweight="bold" if bold else "normal", zorder=zorder + 1,
                linespacing=1.35)
        ax.text(cx, y + h - 4.1, body, ha=align, va="top", fontsize=fs_body,
                color="#3a3a3a", zorder=zorder + 1, linespacing=1.4)
    else:
        ax.text(cx, y + h / 2, title, ha=align, va="center", fontsize=fs,
                color=INK, fontweight="bold" if bold else "normal",
                zorder=zorder + 1, linespacing=1.35)


def arrow(ax, p, q, color: str = ARROW, lw: float = 1.25, ls: str = "solid",
          rad: float = 0.0, head: float = 11, zorder: int = 2):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1.2, shrinkB=1.2, zorder=zorder,
                                 connectionstyle=f"arc3,rad={rad}"))


def group(ax, x, y, w, h, label: str = "", ec: str = "#b9b9b9",
          fc: str = "none", fs: float = 9.0, zorder: int = 1):
    """Dashed container that groups a stage, with a bold caption."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.7,rounding_size=1.0",
                                linewidth=1.1, edgecolor=ec, facecolor=fc,
                                linestyle=(0, (5, 4)), zorder=zorder))
    if label:
        ax.text(x + 2.2, y + h - 2.4, label, ha="left", va="top", fontsize=fs,
                color="#333333", fontweight="bold", zorder=zorder + 1)


def bitrow(ax, x, y, n: int, bits: Optional[Sequence[int]] = None,
           cell: float = 1.0, gap: float = 0.32, zorder: int = 4):
    """A row of little squares: the visual signature of the binary encoding."""
    for i in range(n):
        b = 0 if bits is None else int(bits[i % len(bits)])
        fc = "#1f1f1f" if b else "#ffffff"
        ax.add_patch(Rectangle((x + i * (cell + gap), y), cell, cell,
                               linewidth=0.6, edgecolor="#5a5a5a", facecolor=fc,
                               zorder=zorder))
    return x + n * (cell + gap) - gap


def bitcolumn(ax, x, y, n: int, cell: float = 1.0, gap: float = 0.32,
              pattern: Optional[Sequence[int]] = None, zorder: int = 4):
    for i in range(n):
        b = 0 if pattern is None else int(pattern[i % len(pattern)])
        fc = "#1f1f1f" if b else "#ffffff"
        ax.add_patch(Rectangle((x, y - i * (cell + gap)), cell, cell,
                               linewidth=0.6, edgecolor="#5a5a5a", facecolor=fc,
                               zorder=zorder))
    return y - n * (cell + gap) + gap


def caption(ax, x, y, text: str, fs: float = 7.4, color: str = MUTED):
    ax.text(x, y, text, ha="center", va="top", fontsize=fs, color=color,
            style="italic", linespacing=1.45)


# ---------------------------------------------------------------- LaTeX output
PGF_PREAMBLE = r"""% Preamble needed to compile the .pgf files in this directory:
%   \usepackage{pgf}
%   \usepackage{textcomp}
%   \usepackage[utf8]{inputenc}   % or a modern engine (lualatex/xelatex)
%   \usepackage{amsmath,amssymb}  % for the math in the labels
% Then:  \input{model_schematic.pgf}
%
% The .pgf files are VECTOR LaTeX sources produced by matplotlib's PGF backend,
% so they compile to the same figure at any size and inherit the document's fonts.
% They are NOT hand-written TikZ: matplotlib emits low-level PGF paths.  For a
% figure built from TikZ primitives, see model_schematic.dot (Graphviz) or write
% the TikZ by hand from the same structure.
"""


def use_pgf_backend() -> None:
    """Select matplotlib's PGF backend BEFORE pyplot is used."""
    import matplotlib

    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
        "font.family": "serif",
        "font.size": 10,
    })


def save_figure(fig, out_png: Path, fmt: str = "png") -> None:
    """Save the raster for the repository, or the PGF source for LaTeX."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pgf":
        pgf = out_png.with_suffix(".pgf")
        fig.savefig(pgf)
        print(f"saved {pgf}  (LaTeX vector source)")
    else:
        fig.savefig(out_png)
        print(f"saved {out_png}")


def legend(ax, x, y, entries: Iterable[Tuple[str, str]], fs: float = 7.6,
           title: str = "", cell: float = 1.0):
    """Colour swatches with labels, plus the bit-square convention."""
    y0 = y
    if title:
        ax.text(x, y0 + 2.4, title, ha="left", va="bottom", fontsize=fs + 0.4,
                fontweight="bold", color="#333333")
        y0 -= 0.6
    for label, role in entries:
        fc, ec = rc(role)
        ax.add_patch(Rectangle((x, y0 - cell), cell, cell, linewidth=1.0,
                               edgecolor=ec, facecolor=fc, zorder=4))
        ax.text(x + cell + 0.9, y0 - cell / 2, label, ha="left", va="center",
                fontsize=fs, color=INK)
        y0 -= cell + 1.5
    return y0

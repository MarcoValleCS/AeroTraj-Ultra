# coding=utf-8
"""
Model schematic defined as DOT text and laid out by Graphviz.

This is the alternative to the hand-placed matplotlib schematic
(`plot_architecture.py`): the STRUCTURE lives in text (the DOT language) and the
layout is computed by `dot`, so the diagram can be diffed, version-controlled and
regenerated without any coordinates in the source.

Graphviz must be on the machine (`winget install Graphviz.Graphviz`); the binary is
looked up in PATH and in the standard install location.

Run:  python plot_model_dot.py
Out:  docs/figures/model_schematic.dot   (the source, text)
      docs/figures/model_schematic.png   (raster, for the README/paper preview)
      docs/figures/model_schematic.svg   (vector, for LaTeX \\includegraphics)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

OUT_DOT = Path("docs/figures/model_schematic.dot")

# draw.io pastel palette, same roles as the matplotlib schematic
STYLE = {
    "yellow": ("#fff2cc", "#d6b656"),
    "blue": ("#dae8fc", "#6c8ebf"),
    "green": ("#d5e8d4", "#82b366"),
    "orange": ("#ffe6cc", "#d79b00"),
    "purple": ("#e1d5e7", "#9673a6"),
    "red": ("#f8cecc", "#b85450"),
    "teal": ("#d5f5f5", "#3f9a9a"),
    "gray": ("#f5f5f5", "#666666"),
}


def node(name: str, title: str, body: str = "", role: str = "gray",
         shape: str = "box") -> str:
    fc, ec = STYLE[role]
    # inside an HTML-like label a bare '>' (as in "a -> b") TERMINATES the label,
    # so every arrow must be written as the entity &gt;
    title = title.replace("->", "-&gt;")
    body = body.replace("->", "-&gt;")
    label = f'<<B>{title}</B>'
    if body:
        label += f'<BR/><FONT POINT-SIZE="9">{body}</FONT>'
    label += ">"
    style = "rounded,filled" if shape == "box" else "filled"
    return (f'  {name} [label={label}, shape={shape}, style="{style}", '
            f'fillcolor="{fc}", color="{ec}"];\n')


def build_dot() -> str:
    s = ['digraph aerotraj {',
         '  rankdir=TB;',
         '  splines=spline;',
         '  nodesep=0.45; ranksep=0.42;',
         '  ratio=compress; size="9,11";',
         '  graph [fontname="Helvetica", bgcolor="white", pad=0.3, '
         'label="AeroTraj-Ultra - model schematic", labelloc="t", fontsize=18, '
         'fontname="Helvetica-Bold"];',
         '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
         'fontsize=11, penwidth=1.3, margin="0.16,0.10"];',
         '  edge [color="#4a4a4a", arrowsize=0.75, penwidth=1.1, '
         'fontname="Helvetica", fontsize=9];',
         '']

    # ---- encoder path (a dashed cluster: the embedding stage)
    s.append('  subgraph cluster_embed {')
    s.append('    label="Trajectory embedding  -  Gray-Code + ACoAtt-V2";')
    s.append('    style="dashed"; color="#b9b9b9"; fontname="Helvetica-Bold"; '
             'fontsize=13; penwidth=1.1;')
    s += [node("gc", "Gray-Code encoder", "absolute state -> 85-bit lattice", "blue"),
          node("acoat", "ACoAtt-V2", "6 attributes, pairs lon&lt;-&gt;vx, lat&lt;-&gt;vy, "
               "alt&lt;-&gt;vz", "blue"),
          node("conv", "Conv1d stem (k=3)", "159 -> 144 channels", "blue"),
          node("enc", "pre-LN Transformer encoder", "2 layers, 6 heads, d=144", "blue")]
    s.append('  }')

    # ---- context, decoder, prior, heads
    s += [node("x", "x_obs  [B, 12, 9]", "12 steps = 60 s @ 5 s, or 120 s @ 10 s", "yellow"),
          node("pool", "attention pooling", "softmax over the 12 states", "blue"),
          node("ctx", "context  c", "one vector per window  [B,144]", "teal"),
          node("dec", "non-autoregressive decoder",
               "prefix: 55-58 bit Gray-Code deltas -> Conv1d (11 tokens)<BR ALIGN=\"LEFT\"/>"
               "queries: horizon embedding + Fourier time + context (H = 13, or 6 at 10 s)", "green"),
          node("D", "decoder features  D", "[B, H, 144]", "teal"),
          node("prior", "physics kinematic prior",
               "damped coordinated turn + damped vertical rate<BR ALIGN=\"LEFT\"/>"
               "only three learned scalars: tau_turn, tau_vz, tau_acc", "orange"),
          node("P", "P  [B, H, 6]", "parameter-free anchor", "orange"),
          node("headA", "Head A - Gray-Code MBC", "55 logits -> STE decode -> y_gc", "purple"),
          node("headB", "Head B - continuous residual",
               "zero-init output layer -> y_cont = P + delta", "purple"),
          node("headC", "Head C - MAF-Net trust region",
               "y = y_cont + alpha (*) tanh((y_gc - y_cont)/sigma_h) (*) sigma_h", "purple"),
          node("yhat", "prediction  y  [B, H, 6]", "compared with the truth in ECEF "
               "coordinates -> 3-D MDE", "red")]

    # ---- edges
    edges = [
        ("x", "gc", ""),
        ("gc", "acoat", "85 bits"),
        ("acoat", "conv", ""),
        ("conv", "enc", ""),
        ("enc", "pool", ""),
        ("pool", "ctx", ""),
        ("ctx", "dec", "conditioning"),
        ("dec", "D", ""),
        ("x", "prior", "x_obs", 'style=dashed, color="#d79b00"'),
        ("prior", "P", ""),
        ("D", "headA", ""),
        ("D", "headB", ""),
        ("P", "headB", "anchor", 'style=dashed, color="#d79b00"'),
        ("headA", "headC", "y_gc", 'color="#9673a6"'),
        ("headB", "headC", "y_cont", 'color="#9673a6"'),
        ("headC", "yhat", ""),
    ]
    for a, b, lbl, *extra in edges:
        attrs = f'label="{lbl}", ' if lbl else ""
        attrs += (extra[0] if extra else "")
        s.append(f'  {a} -> {b} [{attrs.rstrip(", ")}];\n')

    # rank hints: keep the spine in the middle and let the two side branches
    # (the prior and the two output heads) sit next to it instead of snaking
    s.append('  {rank=same; x; gc;}\n')
    s.append('  {rank=same; prior; dec;}\n')
    s.append('  {rank=same; P; D;}\n')
    s.append('  {rank=same; headA; headB;}\n')
    s.append('  {rank=same; headC; yhat;}\n')
    s.append('}\n')
    # join with explicit newlines: the header lines above carry no trailing \n,
    # while node()/edge lines do (harmless, stripped here)
    return "\n".join(line.rstrip("\n") for line in s) + "\n"


def find_dot() -> str:
    exe = shutil.which("dot")
    if exe:
        return exe
    for c in [r"C:\Program Files\Graphviz\bin\dot.exe",
              r"C:\Program Files (x86)\Graphviz\bin\dot.exe"]:
        if os.path.exists(c):
            return c
    raise SystemExit("Graphviz 'dot' not found: winget install Graphviz.Graphviz")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    dot_src = build_dot()
    OUT_DOT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOT.write_text(dot_src, encoding="utf-8")
    print(f"[dot] wrote {OUT_DOT} ({len(dot_src)} bytes of DOT source)")

    exe = find_dot()
    print(f"[dot] using {exe}")
    for fmt, suffix in [("png", "png"), ("svg", "svg")]:
        out = OUT_DOT.with_suffix("." + suffix)
        cmd = [exe, f"-T{fmt}"]
        if fmt == "png":
            cmd.append(f"-Gdpi={args.dpi}")
        cmd += ["-o", str(out), str(OUT_DOT)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"dot failed for {fmt}: {r.stderr[:400]}")
        print(f"[dot] wrote {out} ({out.stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()

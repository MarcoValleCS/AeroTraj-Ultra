# coding=utf-8
"""
Accuracy / latency Pareto figure.

Plots the 3-D ECEF MDE (from results_bench/benchmark_results.json, unified table
from results_bench/BENCHMARK_ALL.md) against the single-aircraft latency measured
by bench_latency.py (results_bench/latency.json), on a log-latency axis.

The point of the figure: accuracy alone does not rank the models, because the
operational question is whether the prediction is ready before the update cycle
closes.  Only models on the lower-left frontier are admissible.

Run:  python plot_pareto.py
Out:  docs/figures/pareto_accuracy_latency.png   (tracked: the report embeds it)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/figures/pareto_accuracy_latency.png")

# latency.json display name  ->  (benchmark_results.json key, label, marker, label offset)
MAP = [
    ("AeroTraj-Ultra", "ours", "AeroTraj-Ultra (ours)", "o", (-102, -20)),
    ("AeroTraj-Ultra + retrieval", None, "AeroTraj-Ultra + retrieval", "o", (10, -14)),
    ("FlightBERT++ V2", "fbpp_v2", "FlightBERT++ V2", "s", (10, 4)),
    ("Deep LSTM", "lstm", "Deep LSTM", "s", (10, 4)),
    ("ASCENT", "ascent", "ASCENT (ICRA 2026)", "^", (10, 4)),
    ("CFM (20 step, 1 campione)", "cfm", "CFM / FlowATC (1 sample)", "^", (-150, 10)),
    ("DDPM (50 step, 1 campione)", "ddpm", "DDPM / MID (1 sample)", "^", (-160, 10)),
    ("CV / Doppler", "cv_doppler", "CV / Doppler", "v", (10, 4)),
]

LEGACY_MDE = 1495.8          # FlightPREV re-implemented, from BENCHMARK_ALL.md
RETRIEVAL_MDE = 957.2        # AeroTraj-Ultra + retrieval (multi-hypothesis + retrieval)


def main() -> None:
    bench = json.load(open("results_bench/benchmark_results.json", encoding="utf-8"))
    lat = json.load(open("results_bench/latency.json", encoding="utf-8"))
    b1 = lat["batch1_latency"]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=150)

    for disp, key, label, marker, off in MAP:
        if disp not in b1:
            continue
        x = b1[disp]["gpu_ms"]                  # GPU-busy time, the stable measure
        y = bench[key]["overall_mde_3d_m"] if key else RETRIEVAL_MDE
        ax.scatter(x, y, marker=marker, s=95, zorder=3, label=label)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=off,
                    fontsize=8.5, color="#333333")

    # the constant-velocity floor and the deterministic reference line
    cv = b1.get("CV / Doppler", {}).get("gpu_ms")
    if cv:
        ax.axvline(cv, ls=":", lw=1, color="#999999")

    # operational budget lines
    y0, y1 = ax.get_ylim()
    for ms, txt in [(1000.0, "1 s radar cycle"), (100.0, "100 ms"), (10.0, "10 ms")]:
        ax.axvline(ms, ls="--", lw=1, color="#cc4444", alpha=0.5)
        ax.text(ms * 1.06, y1 - 0.015 * (y1 - y0), txt, rotation=90,
                fontsize=8, color="#cc4444", va="top", ha="left")

    ax.set_xscale("log")
    ax.set_xlabel("inference latency, single aircraft [ms]  (GPU time, log scale)")
    ax.set_ylabel("3-D ECEF MDE [m]  (lower is better)")
    ax.set_title("Accuracy versus inference latency — 90 days, horizon 5–300 s\n"
                 "lower-left is better; every model trained on the same data and protocol",
                 fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}")

    # text summary: the Pareto frontier
    pts = []
    for disp, key, label, _marker, _off in MAP:
        if disp in b1:
            pts.append((b1[disp]["gpu_ms"],
                        bench[key]["overall_mde_3d_m"] if key else RETRIEVAL_MDE, label))
    pts.sort()
    print("\nFronte di Pareto (accuracy-latency, lower-left is better):")
    for x, y, lab in pts:
        dominated = any((x2 <= x and y2 <= y and (x2 < x or y2 < y)) for x2, y2, _ in pts)
        print(f"  {lab:28s} {x:8.2f} ms  {y:9.1f} m{'   <- DOMINATO' if dominated else '   <- non dominato'}")


if __name__ == "__main__":
    main()

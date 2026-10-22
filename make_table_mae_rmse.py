# coding=utf-8
"""
The results table in the format of the FlightBERT paper (Table I).

The paper reports, for every method and every prediction step, the MAE and the
RMSE **per attribute** (Lon, Lat, Alt, Vx, Vy, Vz) plus the prediction steps
themselves, grouped as "group A" (the deep-regression and the model-driven
baselines) and "group B" (the binary-encoding transformers).

This script rebuilds the same table from the results of the 365-day / 10 s
benchmark:
  * `results_bench_10s/benchmark_results.json`  -> the numbers
  * `docs/figures/bench_10s_mae_rmse.png`        -> the rendered table
  * a markdown block for the paper, the notebook and the README.

Run:  python make_table_mae_rmse.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("results_bench_10s/benchmark_results.json")
OUT_FIG = Path("docs/figures/bench_10s_mae_rmse.png")
OUT_MD = Path("results_bench_10s/table_mae_rmse.md")

CFG = json.load(open("config_365d_10s_bench.json", encoding="utf-8"))
STEPS = list(CFG["window"]["horizon_steps"])
SECS = list(CFG["window"]["horizon_seconds"])
DT = int(CFG["data"]["dt_seconds"])

# the groups of the paper: A = the deep-regression and model-driven baselines,
# B = the binary-encoding transformers, C = this project
GROUPS = [
    ("A1", "LSTM / regression", "lstm"),
    ("A2", "BiLSTM / regression", "bilstm"),
    ("A3", "Kalman Filter", "kalman"),
    ("A4", "IMM (CV/CA/CT)", "imm"),
    ("B0", "FlightBERT (T-ITS 2023)", "flightbert"),
    ("B1", "FlightBERT++ (no ACoAtt)", "fbpp_v2"),
    ("B2", "FlightBERT++ (ACoAtt)", "fbpp_v2_faithful"),
    ("B3", "FlightBERT++ (ACoAtt), MSE-only", "fbpp_v2_faithful_mse"),
    ("C1", "AeroTraj-Ultra (ours)", "ours"),
]
ATTRS = ["lon", "lat", "alt", "vx", "vy", "vz"]
ATTR_LABEL = {"lon": "Lon", "lat": "Lat", "alt": "Alt", "vx": "Vx", "vy": "Vy", "vz": "Vz"}
# the units: the paper reports the altitude in metres and the velocities in km/h;
# our vertical velocity is in m/s (the OpenSky feature), and it is reported as such
UNIT = {"lon": "deg", "lat": "deg", "alt": "m", "vx": "km/h", "vy": "km/h", "vz": "m/s"}


def scaled(value: float, attr: str) -> float:
    return value * 10.0 if attr == "alt" else value


def load() -> Dict[str, dict]:
    return json.load(open(RES, encoding="utf-8"))


def rows(data: Dict[str, dict]):
    """-> a list of (group, label, step, seconds, {metric: {attr: value}})."""
    out = []
    for gid, label, key in GROUPS:
        if key not in data:
            continue
        ph = data[key]["per_horizon"]
        for i, (st, sec) in enumerate(zip(STEPS, SECS)):
            h = ph.get(f"{sec}s")
            if h is None:
                continue
            mae = {a: scaled(h["mae"][a], a) for a in ATTRS}
            rmse = {a: scaled(h["rmse"][a], a) for a in ATTRS}
            out.append((gid, label, st, sec, mae, rmse))
    return out


def fmt(v: float, attr: str) -> str:
    if attr in ("lon", "lat"):
        return f"{v:.4f}"
    if attr == "alt":
        return f"{v:.2f}"
    if attr == "vz":
        return f"{v:.2f}"
    return f"{v:.2f}"


def markdown(data: Dict[str, dict]) -> str:
    lines = []
    lines.append("| Methods | Pred. steps | "
                 + " | ".join(f"MAE {ATTR_LABEL[a]}" for a in ATTRS) + " | "
                 + " | ".join(f"RMSE {ATTR_LABEL[a]}" for a in ATTRS) + " |")
    lines.append("|---|---:|" + "---:|" * (2 * len(ATTRS)))
    last = None
    for gid, label, st, sec, mae, rmse in rows(data):
        head = f"**{gid}** ({label})" if gid != last else ""
        last = gid
        cells = [fmt(mae[a], a) for a in ATTRS] + [fmt(rmse[a], a) for a in ATTRS]
        lines.append(f"| {head} | {st} ({sec} s) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(data: Dict[str, dict]) -> None:
    """The table, rendered in the style of the paper (small caps, rules)."""
    rr = rows(data)
    n = len(rr)
    fig_h = 1.4 + 0.30 * n
    fig, ax = plt.subplots(figsize=(13.4, fig_h), dpi=150)
    ax.axis("off")

    headers = ["Methods", "Pred.\nsteps"] + [f"MAE ↓\n{ATTR_LABEL[a]}" for a in ATTRS] \
        + [f"RMSE ↓\n{ATTR_LABEL[a]}" for a in ATTRS]
    body = []
    for gid, label, st, sec, mae, rmse in rr:
        body.append([f"{gid} ({label})", f"{st}\n({sec} s)"]
                    + [fmt(mae[a], a) for a in ATTRS]
                    + [fmt(rmse[a], a) for a in ATTRS])

    tbl = ax.table(cellText=body, colLabels=headers, loc="upper center",
                   cellLoc="center", colLoc="center",
                   colWidths=[0.17, 0.065] + [0.0639] * (2 * len(ATTRS)))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.6)
    tbl.scale(1, 1.32)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#999999")
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#f2f2f2")
            cell.set_text_props(weight="bold", fontsize=6.8)
        elif c == 0:
            cell.set_text_props(ha="left", fontsize=6.4)
            if body[r - 1][0].startswith("C1"):
                cell.set_text_props(weight="bold")
            else:
                cell.set_text_props(weight="bold" if body[r - 1][0][0] != "(" else "normal")
    ax.set_title("Results of the 365-day benchmark, group A (regression and model-driven), "
                 "group B (binary-encoding transformers) and group C (this project)\n"
                 "12 observed points at 10 s (120 s of history), prediction steps of 10 s; "
                 "same data, same budget, 60 000 held-out windows; "
                 "MAE and RMSE per attribute, in the native units",
                 fontsize=8.2, pad=12)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"saved {OUT_FIG}")


def main() -> None:
    data = load()
    md = markdown(data)
    OUT_MD.write_text(md + "\n", encoding="utf-8")
    print(f"saved {OUT_MD}")
    figure(data)
    print()
    print(md)


if __name__ == "__main__":
    main()

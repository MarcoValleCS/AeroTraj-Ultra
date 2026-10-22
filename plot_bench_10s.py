# coding=utf-8
"""
Comparison sheets for the 365-day / 10 s-grid benchmark.

Reads results_bench_10s/benchmark_results.json (the test metrics of every
model trained on the same data with the same budget) and
results_bench_10s/training_summary.json (the training history, the parameter
count and the wall-clock), plus the full-budget reference run of the project
model (results_365_10s/evaluation_summary.json).

Outputs (docs/figures/):
  bench_10s_horizon.png    the per-horizon error of every model + the relative
                           error against the best model of each horizon
  bench_10s_ranking.png    the overall ranking, the parameters and the cost
  bench_10s_table.png      the comparison table, rendered
  bench_10s_gain.png       the gain of each model over the best classical
                           baseline, per horizon (where does each model win?)

Run:  python plot_bench_10s.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results_bench_10s")
FIG = Path("docs/figures")
FULL_BUDGET = Path("results_365_10s/evaluation_summary.json")

# key -> (label, colour, marker, zorder)
STYLE = {
    "ours":        ("AeroTraj-Ultra (ours)", "#1f77b4", "o", 5),
    "ours_full":   ("AeroTraj-Ultra, full budget (180M samples)", "#0b3d63", "*", 6),
    "flightbert":  ("FlightBERT (T-ITS 2023), BCE-dominant loss", "#d62728", "s", 3),
    "flightbert_mse": ("FlightBERT (T-ITS 2023), MSE-only loss", "#e377c2", "P", 3),
    "flightbert_soft": ("FlightBERT (T-ITS 2023), roll-out without threshold", "#bcbd22", "X", 2),
    "fbpp_v2":     ("FlightBERT++ V2 (AAAI 2024), no ACoAtt", "#ff7f0e", "v", 3),
    "fbpp_v2_faithful": ("FlightBERT++ V2 (AAAI 2024), faithful", "#8c564b", "1", 4),
    "fbpp_v2_faithful_mse": ("FlightBERT++ V2 (AAAI 2024), faithful, MSE-only loss",
                             "#c49c94", "2", 3),
    "kalman":      ("Kalman filter (CV, learnable noise)", "#2ca02c", "^", 3),
    "imm":         ("IMM (CV/CA/CT, learnable noise)", "#9467bd", "D", 3),
    "lstm":        ("LSTM regression", "#8c564b", "P", 3),
    "bilstm":      ("BiLSTM regression", "#e377c2", "X", 3),
    "cv_lsq":      ("CV least-squares (no learning)", "#7f7f7f", "<", 2),
}
ORDER = ["ours_full", "ours", "fbpp_v2_faithful", "fbpp_v2_faithful_mse", "fbpp_v2",
         "flightbert_mse", "flightbert", "flightbert_soft",
         "imm", "kalman", "bilstm", "lstm", "cv_lsq"]

# the wall-clock of the training runs, recovered from the run logs (the
# training_summary.json only keeps the models of the last invocation)
TRAIN_SECONDS = {
    "ours": 21.5 * 25,
    "fbpp_v2": 14.1 * 25,
    "fbpp_v2_faithful": 24.7 * 25,
    "kalman": 18.4 * 25,
    "imm": 64.6 * 25,
    "lstm": 3.9 * 25,
    "bilstm": 24.8 * 25,
}
PARAMS = {
    "ours": 1_276_983,
    "fbpp_v2": 760_000,
    "fbpp_v2_faithful": 1_850_000,
    "kalman": 0,
    "imm": 0,
    "lstm": 380_000,
    "bilstm": 1_360_000,
}


def load() -> Dict[str, dict]:
    res = json.load(open(RES / "benchmark_results.json", encoding="utf-8"))
    ts_path = RES / "training_summary.json"
    ts = json.load(open(ts_path, encoding="utf-8")) if ts_path.exists() else {}
    models = ts.get("models", {})

    out: Dict[str, dict] = {}
    for key, r in res.items():
        entry = {
            "mde": r["overall_mde_3d_m"],
            "per_horizon": {int(h.split("s")[0]): v["mde_3d_m"]
                            for h, v in r["per_horizon"].items()},
            "ade": r.get("ade_3d_m"), "fde": r.get("fde_3d_m"),
            "params": models.get(key, {}).get("parameters") or PARAMS.get(key),
            "train_seconds": models.get(key, {}).get("train_seconds") or TRAIN_SECONDS.get(key),
            "best_epoch": models.get(key, {}).get("best_epoch"),
            "val_mde": models.get(key, {}).get("best_val_mde_m"),
        }
        out[key] = entry

    # the reference: the project model trained with the FULL budget
    if FULL_BUDGET.exists():
        fb = json.load(open(FULL_BUDGET, encoding="utf-8"))
        m = fb.get("aerotraj_ultra_test_metrics", {})
        if m.get("per_horizon"):
            out["ours_full"] = {
                "mde": fb["test_overall_mde_3d_m"]["AeroTraj_Ultra_Ours"],
                "per_horizon": {int(h.split("s")[0]): v["mde_3d_m"]
                                for h, v in m["per_horizon"].items()},
                "ade": m.get("ade_3d_m"), "fde": m.get("fde_3d_m"),
                "params": fb.get("model_parameters", {}).get("AeroTraj_Ultra_Ours")
                if isinstance(fb.get("model_parameters"), dict) else None,
                "train_seconds": None, "best_epoch": None, "val_mde": None,
            }
    return out


def horizons(data: Dict[str, dict]) -> List[int]:
    for e in data.values():
        return sorted(e["per_horizon"].keys())
    return []


def present(data: Dict[str, dict]) -> List[str]:
    return [k for k in ORDER if k in data] + [k for k in data if k not in ORDER]


# --------------------------------------------------------------------------- #
def sheet_horizon(data: Dict[str, dict], hs: List[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=150)

    ax = axes[0]
    for key in present(data):
        if key not in STYLE:
            continue
        label, colour, marker, z = STYLE[key]
        ys = [data[key]["per_horizon"].get(h, np.nan) for h in hs]
        ax.plot(range(len(hs)), ys, marker=marker, color=colour, lw=1.9 if key.startswith("ours") else 1.4,
                ms=8 if key.startswith("ours") else 5.5, label=label, zorder=z,
                ls="--" if key == "ours_full" else "-")
    ax.set_yscale("log")
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"{h} s" for h in hs])
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("3-D ECEF error [m]  (log scale)")
    ax.set_title("Error per horizon — 365 days, 10 s grid\n"
                 "every model trained on the same data with the same budget",
                 fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.93)

    # the relative error against the best model of each horizon
    ax = axes[1]
    best = [min(data[k]["per_horizon"].get(h, np.inf) for k in data) for h in hs]
    for key in present(data):
        if key not in STYLE:
            continue
        label, colour, marker, z = STYLE[key]
        ys = [100.0 * data[key]["per_horizon"].get(h, np.nan) / b for h, b in zip(hs, best)]
        ax.plot(range(len(hs)), ys, marker=marker, color=colour, lw=1.9 if key.startswith("ours") else 1.4,
                ms=8 if key.startswith("ours") else 5.5, label=label, zorder=z,
                ls="--" if key == "ours_full" else "-")
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"{h} s" for h in hs])
    ax.set_yscale("log")
    ax.axhline(100.0, color="#333333", lw=0.9, ls=":")
    ax.set_xlabel("prediction horizon")
    ax.set_ylabel("error relative to the best model of that horizon [%]")
    ax.set_title("Relative error: who wins where", fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.35)

    fig.tight_layout()
    out = FIG / "bench_10s_horizon.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def sheet_ranking(data: Dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), dpi=150,
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    keys = [k for k in present(data) if k != "ours_full"]
    keys = sorted(keys, key=lambda k: data[k]["mde"])
    labels = [STYLE.get(k, (k,))[0] for k in keys]
    colours = [STYLE.get(k, ("", "#333333"))[1] for k in keys]
    y = np.arange(len(keys))

    ax = axes[0]
    vals = [data[k]["mde"] for k in keys]
    ax.barh(y, vals, color=colours, alpha=0.9, height=0.68)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("mean 3-D error over the six horizons [m]  (log scale)")
    ax.set_title("Overall ranking — same data, same budget", fontsize=10)
    ax.grid(True, axis="x", which="both", ls=":", alpha=0.35)
    for i, k in enumerate(keys):
        ax.text(vals[i] * 1.12, i, f"{vals[i]:,.0f} m".replace(",", " "), va="center", fontsize=8)
    if "ours_full" in data:
        ax.axvline(data["ours_full"]["mde"], color=STYLE["ours_full"][1], ls="--", lw=1.4,
                   label="AeroTraj-Ultra, full budget (180M samples)")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.93)

    # the cost side: parameters and training time
    ax = axes[1]
    sub = [k for k in keys if data[k].get("params")]
    if sub:
        xs = np.arange(len(sub))
        par = [data[k]["params"] / 1e6 for k in sub]
        ax.bar(xs - 0.2, par, width=0.38, color="#6c8ebf", label="parameters [M]")
        ax.set_ylabel("parameters [M]", color="#3a6ea5")
        ax.set_xticks(xs)
        ax.set_xticklabels([STYLE.get(k, (k,))[0][:22] for k in sub], rotation=30,
                           ha="right", fontsize=7.5)
        ax2 = ax.twinx()
        sec = [(data[k].get("train_seconds") or 0) / 60.0 for k in sub]
        ax2.bar(xs + 0.2, sec, width=0.38, color="#d79b00", label="training time [min]")
        ax2.set_ylabel("training time [min]", color="#b07800")
        ax.set_title("Cost of each model", fontsize=10)
        ax.grid(True, axis="y", ls=":", alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.93)

    fig.tight_layout()
    out = FIG / "bench_10s_ranking.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def sheet_gain(data: Dict[str, dict], hs: List[int]) -> None:
    """The gain of every learned model over the best classical (no-learning) one."""
    classical = [k for k in ("cv_lsq", "kalman", "imm") if k in data]
    if not classical:
        return
    base = {h: min(data[k]["per_horizon"].get(h, np.inf) for k in classical) for h in hs}

    keys = [k for k in present(data) if k not in classical and k in data]
    mat = np.array([
        [100.0 * (1.0 - data[k]["per_horizon"].get(h, np.nan) / base[h]) for h in hs]
        for k in keys
    ])

    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(keys) + 2.6), dpi=150)
    lim = max(5.0, float(np.nanmax(np.abs(mat))) * 1.15)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(hs)))
    ax.set_xticklabels([f"{h} s" for h in hs])
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([STYLE.get(k, (k,))[0] for k in keys], fontsize=8.5)
    for i in range(len(keys)):
        for j in range(len(hs)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=8,
                        color="#1f1f1f" if abs(v) < lim * 0.55 else "white")
    ax.set_title("Error reduction with respect to the best classical filter, per horizon\n"
                 "green = better than the classical filter, red = worse",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="error reduction [%]")
    fig.tight_layout()
    out = FIG / "bench_10s_gain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def sheet_table(data: Dict[str, dict], hs: List[int]) -> None:
    keys = [k for k in present(data) if k != "cv_lsq"]
    keys = sorted(keys, key=lambda k: data[k]["mde"])
    head = ["model", "MDE [m]", "ADE [m]", "FDE [m]"] + [f"{h} s" for h in hs]
    body = []
    for k in keys:
        e = data[k]
        row = [STYLE.get(k, (k,))[0], f"{e['mde']:.0f}",
               f"{e['ade']:.0f}" if e.get("ade") else "-",
               f"{e['fde']:.0f}" if e.get("fde") else "-"]
        row += [f"{e['per_horizon'].get(h, float('nan')):.0f}" for h in hs]
        body.append(row)

    fig, ax = plt.subplots(figsize=(12.2, 0.46 * len(body) + 2.0), dpi=150)
    ax.axis("off")
    tbl = ax.table(cellText=body, colLabels=head, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.4)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold")
        elif c == 0:
            cell.set_text_props(ha="left", weight="bold" if "AeroTraj" in body[r - 1][0] else "normal")
    ax.set_title("365-day / 10 s benchmark — the same data, the same budget\n"
                 "test split: 36 days (Nov-Dec 2022), 60 000 windows, 3-D ECEF error [m]",
                 fontsize=10, pad=14)
    fig.tight_layout()
    out = FIG / "bench_10s_table.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = load()
    if not data:
        raise SystemExit("no results found in results_bench_10s/")
    hs = horizons(data)
    sheet_horizon(data, hs)
    sheet_ranking(data)
    sheet_gain(data, hs)
    sheet_table(data, hs)

    print("\n" + "=" * 78)
    print(f"{'model':46s} {'MDE':>9s}  " + "  ".join(f"{h:>6d}s" for h in hs))
    for k in sorted(data, key=lambda k: data[k]["mde"]):
        e = data[k]
        row = "  ".join(f"{e['per_horizon'].get(h, float('nan')):6.0f}" for h in hs)
        print(f"{STYLE.get(k, (k,))[0]:46s} {e['mde']:9.1f}  {row}")


if __name__ == "__main__":
    main()

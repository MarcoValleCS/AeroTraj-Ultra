# coding=utf-8
"""
Generate the markdown fragments of the 365-day / 10 s benchmark from the
results JSON, so that every number quoted in the paper, in the notebook and in
the README comes from the same source.

Run:  python make_bench_10s_report.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

RES = Path("results_bench_10s")
FULL = Path("results_365_10s/evaluation_summary.json")

LABEL = {
    "ours": "AeroTraj-Ultra (ours)",
    "ours_full": "AeroTraj-Ultra (ours), full budget",
    "flightbert": "FlightBERT (IEEE T-ITS 2023)",
    "fbpp_v2": "FlightBERT++ V2 (AAAI 2024)",
    "kalman": "Kalman filter (CV, learnable noise)",
    "imm": "IMM (CV/CA/CT, learnable noise)",
    "lstm": "LSTM regression",
    "bilstm": "BiLSTM regression",
    "cv_lsq": "CV least-squares (no learning)",
}
ORDER = ["ours_full", "ours", "flightbert", "fbpp_v2", "imm", "kalman",
         "bilstm", "lstm", "cv_lsq"]


def load() -> Dict[str, dict]:
    res = json.load(open(RES / "benchmark_results.json", encoding="utf-8"))
    ts_path = RES / "training_summary.json"
    ts = json.load(open(ts_path, encoding="utf-8")) if ts_path.exists() else {}
    models = ts.get("models", {})
    out: Dict[str, dict] = {}
    for key, r in res.items():
        out[key] = {
            "mde": r["overall_mde_3d_m"],
            "ph": {int(h.split("s")[0]): v["mde_3d_m"] for h, v in r["per_horizon"].items()},
            "ade": r.get("ade_3d_m"), "fde": r.get("fde_3d_m"),
            "params": models.get(key, {}).get("parameters"),
            "sec": models.get(key, {}).get("train_seconds"),
            "epoch": models.get(key, {}).get("best_epoch"),
            "val": models.get(key, {}).get("best_val_mde_m"),
            "noise": models.get(key, {}).get("learned_noise_model"),
        }
    if FULL.exists():
        fb = json.load(open(FULL, encoding="utf-8"))
        m = fb.get("aerotraj_ultra_test_metrics", {})
        if m.get("per_horizon"):
            out["ours_full"] = {
                "mde": fb["test_overall_mde_3d_m"]["AeroTraj_Ultra_Ours"],
                "ph": {int(h.split("s")[0]): v["mde_3d_m"] for h, v in m["per_horizon"].items()},
                "ade": m.get("ade_3d_m"), "fde": m.get("fde_3d_m"),
                "params": None, "sec": None, "epoch": None, "val": None, "noise": None,
            }
    return out


def num(v, dec=1):
    if v is None:
        return "-"
    return f"{v:,.{dec}f}".replace(",", " ")


def main() -> None:
    data = load()
    hs = sorted(next(iter(data.values()))["ph"].keys())
    keys = [k for k in ORDER if k in data] + [k for k in data if k not in ORDER]
    keys = sorted(keys, key=lambda k: data[k]["mde"])

    # ---------------------------------------------------------------- table
    print("\n### Benchmark table (3-D ECEF error [m])\n")
    print("| model | MDE mean | ADE | FDE | " + " | ".join(f"{h} s" for h in hs) + " |")
    print("|---|---:|---:|---:|" + "---:|" * len(hs))
    for k in keys:
        e = data[k]
        row = " | ".join(num(e["ph"].get(h)) for h in hs)
        print(f"| {LABEL.get(k, k)} | **{num(e['mde'])}** | {num(e.get('ade'))} | "
              f"{num(e.get('fde'))} | {row} |")

    # ------------------------------------------------------------ the costs
    print("\n### Cost of each model\n")
    print("| model | parameters | best epoch | val MDE [m] | training time |")
    print("|---|---:|---:|---:|---:|")
    for k in keys:
        e = data[k]
        if e.get("params") is None and e.get("sec") is None:
            continue
        p = f"{e['params']/1e6:.2f} M" if e.get("params") else "-"
        t = f"{e['sec']/60:.1f} min" if e.get("sec") else "-"
        print(f"| {LABEL.get(k, k)} | {p} | {e.get('epoch') or '-'} | "
              f"{num(e.get('val'))} | {t} |")

    # ------------------------------------------------- the learned noise
    print("\n### The learned noise model of the classical filters\n")
    for k in ("kalman", "imm"):
        if k in data and data[k].get("noise"):
            n = data[k]["noise"]
            print(f"* **{LABEL[k]}**")
            for kk, vv in n.items():
                print(f"  * `{kk}` = {vv}")

    # ------------------------------------------------------ the comparisons
    print("\n### Head-to-head\n")
    best = min(data, key=lambda k: data[k]["mde"])
    print(f"* best model: **{LABEL.get(best, best)}** at {num(data[best]['mde'])} m")
    classical = [k for k in ("cv_lsq", "kalman", "imm") if k in data]
    if classical:
        bc = min(classical, key=lambda k: data[k]["mde"])
        for k in keys:
            if k in classical:
                continue
            d = 100.0 * (data[k]["mde"] - data[bc]["mde"]) / data[bc]["mde"]
            print(f"* {LABEL.get(k, k)}: {d:+.1f} % w.r.t. the best classical filter "
                  f"({LABEL.get(bc, bc)})")
    if "ours" in data and "ours_full" in data:
        d = 100.0 * (data["ours"]["mde"] - data["ours_full"]["mde"]) / data["ours_full"]["mde"]
        print(f"* the same-budget run of ours is {d:+.1f} % w.r.t. the full-budget run "
              f"({num(data['ours_full']['mde'])} m vs {num(data['ours']['mde'])} m: the "
              f"budget is 18x larger)")

    # per-horizon winner
    print("\n### Who wins at each horizon\n")
    print("| horizon | best model | error [m] |")
    print("|---:|---|---:|")
    for h in hs:
        k = min(data, key=lambda kk: data[kk]["ph"].get(h, 1e18))
        print(f"| {h} s | {LABEL.get(k, k)} | {num(data[k]['ph'][h])} |")


if __name__ == "__main__":
    main()

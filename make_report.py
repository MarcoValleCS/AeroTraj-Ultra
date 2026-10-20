# coding=utf-8
"""
Builds the final comparison report (markdown) across the available runs:

  1. 40-day CPU reference run   (results/)
  2. 40-day GPU control run     (results40gpu/)
  3. 90-day GPU run             (results90/)

Per-horizon numbers come from `horizon_benchmark.csv` (all models), overall
numbers from `evaluation_summary.json` (all models).

Usage:
    python make_report.py --out results90/REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

RUNS = [
    ("40d CPU (original)", "results"),
    ("40d GPU (control)", "results40gpu"),
    ("90d GPU (nuovo)", "results90"),
]

# (label, csv column, json key in test_overall_mde_3d_m)
MODELS = [
    ("AeroTraj-Ultra (ours)", "mde_aerotraj_ultra_m", "AeroTraj_Ultra_Ours"),
    ("AeroTraj-Ultra w/o MAF gate", "mde_aerotraj_ultra_no_maf_m", "AeroTraj_Ultra_wo_MAF_GC_Gate"),
    ("Physics kinematic prior", "mde_physics_prior_m", "Physics_Turn_Climb_Prior"),
    ("FlightBERT++ V2", "mde_flightbert_pp_v2_m", "FlightBERT_PP_V2"),
    ("Deep LSTM", "mde_lstm_m", "Deep_LSTM_Regression"),
    ("CV (raw Doppler)", "mde_kalman_cv_m", "Kalman_Constant_Velocity"),
    ("CV (least-squares)", "mde_cv_least_squares_m", "CV_LeastSquares_Extrapolation"),
]

SHOW_H = [5, 30, 60, 300, 600]


def _pick_horizons(bench: Optional[pd.DataFrame]) -> List[int]:
    """Pick the reference horizons actually available in the benchmark table."""
    if bench is None or len(bench) == 0:
        return SHOW_H
    avail = sorted(int(v) for v in bench["horizon_s"].tolist())
    wanted = [h for h in SHOW_H if h in avail]
    if not wanted:
        wanted = [avail[0], avail[len(avail) // 2], avail[-1]]
    return wanted


def load_run(folder: str) -> Optional[Dict[str, Any]]:
    d = Path(folder)
    summ_p, csv_p = d / "evaluation_summary.json", d / "horizon_benchmark.csv"
    if not summ_p.exists():
        return None
    with open(summ_p, "r", encoding="utf-8") as f:
        summary = json.load(f)
    bench = pd.read_csv(csv_p) if csv_p.exists() else None
    return {"summary": summary, "bench": bench, "folder": folder}


def fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:,.1f}"


def main(out_path: str) -> None:
    runs = {name: load_run(folder) for name, folder in RUNS}
    runs = {k: v for k, v in runs.items() if v is not None}
    if not runs:
        print("[Report] Nessun run trovato.")
        return

    L: List[str] = []
    L.append("# AeroTraj-Ultra (FlightBERT++ V3) - report comparativo delle run\n")

    # ---------------- 1. Dataset ---------------- #
    L.append("## 1. Dataset\n")
    L.append("| Run | Giorni | Train/Val/Test | Punti 5 s | Segmenti | Voli unici | Finestre (stride 1) | Righe 1 s lette |")
    L.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for name, r in runs.items():
        ds = r["summary"]["dataset_summary"]
        sp = ds["splits"]
        pts = sum(v["num_5s_points"] for v in sp.values())
        segs = sum(v["num_contiguous_5s_trajectories"] for v in sp.values())
        fl = sum(v["num_unique_flights"] for v in sp.values())
        win = sum(v["total_sliding_windows_stride1"] for v in sp.values())
        raw = sum(v["raw_1s_rows_streamed"] for v in sp.values())
        L.append(
            f"| {name} | {ds['num_total_days']} | {ds['train_days']}/{ds['val_days']}/{ds['test_days']} | "
            f"{pts:,} | {segs:,} | {fl:,} | {win:,} | {raw:,} |"
        )
    L.append("")

    # ---------------- 2. Training ---------------- #
    L.append("## 2. Addestramento\n")
    L.append("| Run | Epoche | Step/epoch | Parametri | Test 5 s [m] | Test 1 min [m] | Test 5 min [m] | Test 10 min [m] | MDE media [m] |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in runs.items():
        s = r["summary"]
        hist = s.get("training_history", [])
        per = s["aerotraj_ultra_test_metrics"]["per_horizon"]
        pool = s["dataset_summary"]["splits"]["train"]["total_sliding_windows_stride1"]
        steps = "-"
        if hist:
            steps = f"{pool // 8 // 4096}"
        L.append(
            f"| {name} | {len(hist)} | {steps} | {s['model_parameters']:,} | "
            f"{per['5s']['mde_3d_m']:.1f} | {per['60s']['mde_3d_m']:.1f} | "
            f"{per['300s']['mde_3d_m']:.1f} | {per['600s']['mde_3d_m']:.1f} | "
            f"{s['aerotraj_ultra_test_metrics']['overall_mde_3d_m']:.1f} |"
        )
    L.append("")

    # ---------------- 3. Benchmark ---------------- #
    L.append("## 3. MDE 3D ECEF [m] per orizzonte\n")
    for name, r in runs.items():
        b = r["bench"]
        hs = _pick_horizons(b)
        hdr = " | ".join(f"{h // 60} min" if h >= 60 else f"{h} s" for h in hs)
        L.append(f"### {name}\n")
        L.append(f"| Modello | {hdr} | Media |")
        L.append("|---|" + "---:|" * (len(hs) + 1))
        overall = r["summary"]["test_overall_mde_3d_m"]
        for label, col, jkey in MODELS:
            ov = overall.get(jkey)
            if b is None or col not in b.columns:
                L.append(f"| {label} | " + " | ".join(["n/a"] * len(hs)) + f" | {fmt(ov)} |")
                continue
            cells = []
            for h in hs:
                sel = b[b["horizon_s"] == h]
                cells.append(fmt(float(sel[col].iloc[0])) if len(sel) else "n/a")
            L.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(ov)} |")
        L.append("")

    # ---------------- 4. Delta 40d GPU -> 90d GPU ---------------- #
    if "40d GPU (control)" in runs and "90d GPU (nuovo)" in runs:
        a = runs["40d GPU (control)"]["summary"]["test_overall_mde_3d_m"]["AeroTraj_Ultra_Ours"]
        c = runs["90d GPU (nuovo)"]["summary"]["test_overall_mde_3d_m"]["AeroTraj_Ultra_Ours"]
        L.append("## 4. Effetto dei dati (stessa ricetta di training)\n")
        L.append(f"- 40 giorni (GPU): **{a:,.1f} m**")
        L.append(f"- 90 giorni (GPU): **{c:,.1f} m**")
        L.append(f"- Variazione: **{(c / a - 1) * 100:+.1f} %**\n")
        L.append("> Attenzione: i due modelli hanno test set *diversi* (finestra cronologica diversa).")
        L.append("> Per un confronto apples-to-apples usare `eval_ckpt.py` che valuta entrambi")
        L.append("> sullo stesso split mai visto in training.\n")

    text = "\n".join(L)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text, encoding="utf-8")
    print(text)
    print(f"[Report] scritto in {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results90/REPORT.md")
    args = parser.parse_args()
    main(args.out)

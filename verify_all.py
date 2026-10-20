# coding=utf-8
"""Final state verification of all the runs performed in this project."""
import json
from pathlib import Path

RUNS = [
    ("A  40d CPU   (h 600s)", "results/evaluation_summary.json"),
    ("B  40d GPU   (h 600s)", "results40gpu/evaluation_summary.json"),
    ("C  90d GPU   (h 600s)", "results90/evaluation_summary.json"),
    ("D 180d GPU   (h 180s)", "results180/evaluation_summary.json"),
    ("E  90d GPU   (h 180s)", "results90_180s/evaluation_summary.json"),
]

print(f"{'run':24s} {'ours':>9s} {'FB++V2':>9s} {'CV':>9s} {'LSTM':>9s} {'prior':>9s}  ep/best  ES   optim")
print("-" * 106)
for tag, p in RUNS:
    if not Path(p).exists():
        print(f"{tag:24s}  (assente)")
        continue
    s = json.load(open(p, encoding="utf-8"))
    o = s["test_overall_mde_3d_m"]
    op = s.get("optimization", {})
    es = op.get("early_stopping", {})
    print(
        f"{tag:24s} {o['AeroTraj_Ultra_Ours']:9.1f} {o['FlightBERT_PP_V2']:9.1f} "
        f"{o['Kalman_Constant_Velocity']:9.1f} {o['Deep_LSTM_Regression']:9.1f} "
        f"{o['Physics_Turn_Climb_Prior']:9.1f}  "
        f"{op.get('epochs_executed','-')}/{op.get('epochs_planned','-')} @{op.get('best_epoch','-'):<3} "
        f"{str(es.get('triggered','-')):5s} {op.get('optimizer','-')}/{op.get('scheduler','-')}"
    )

print("\nDataset")
for tag, p in [("data  (40d)", "data/dataset_stats.json"), ("data90 (90d)", "data90/dataset_stats.json"),
               ("data180(180d)", "data180/dataset_stats.json")]:
    if Path(p).exists():
        d = json.load(open(p, encoding="utf-8"))
        fl = sum(v["num_unique_flights"] for v in d["splits"].values())
        pts = sum(v["num_5s_points"] for v in d["splits"].values())
        win = sum(v["total_sliding_windows_stride1"] for v in d["splits"].values())
        print(f"  {tag}: {d['num_total_days']:3d} giorni | {fl:>7,} voli | {pts/1e6:6.1f} M punti | {win/1e6:6.1f} M finestre")

print("\nSpazio su disco")
for d in ["data", "data90", "data180"]:
    p = Path(d)
    if p.exists():
        tot = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        cache = sum(f.stat().st_size for f in (p / "_cache_days").rglob("*") if f.is_file()) if (p / "_cache_days").exists() else 0
        print(f"  {d:9s} {tot/1e9:6.1f} GB totali ({cache/1e9:5.1f} GB di cache per-giorno)")

# coding=utf-8
"""
State verification of the repository: what is on disk, and does it agree with the
documents?

Checks, in order:
  1. the datasets present, with their signature and their split sizes;
  2. the training runs present, with the losses they report;
  3. the benchmarks present, with the number of models and the best one;
  4. the figures the documents reference;
  5. the disk used by the artefacts.

Run:  python verify_all.py
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------- datasets
DATASETS = [
    ("data365      (5 s grid)", "data365", "config_365d.json"),
    ("data365_10s  (10 s grid)", "data365_10s", "config_365d_10s.json"),
]

# ---------------------------------------------------------------- runs
RUNS = [
    ("365 d, 5 s, 400 k x 150", "results_365/evaluation_summary.json"),
    ("365 d, 5 s, 1.2 M x 150", "results_365/cross_budgets.json"),
    ("365 d, 10 s, 1.2 M x 150", "results_365_10s/evaluation_summary.json"),
]

# ---------------------------------------------------------------- benchmarks
BENCHES = [
    ("90 days, 5 s (12 models)", "results_bench/benchmark_results.json"),
    ("365 days, 10 s (the re-implementations)", "results_bench_10s/benchmark_results.json"),
]

FIGURES = [
    "docs/figures/architecture_overview.png",
    "docs/figures/hybrid_loss.png",
    "docs/figures/acoat_v2_block.png",
    "docs/figures/model_schematic.png",
    "docs/figures/model_schematic.svg",
    "docs/figures/model_schematic_tikz.pdf",
    "docs/figures/hybrid_loss_tikz.pdf",
    "docs/figures/trajectory_cases.png",
    "docs/figures/pareto_accuracy_latency.png",
    "docs/figures/bench_10s_horizon.png",
    "docs/figures/bench_10s_ranking.png",
    "docs/figures/bench_10s_gain.png",
    "docs/figures/bench_10s_table.png",
    "docs/figures/bench_10s_mae_rmse.png",
]


def size_of(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e9


def main() -> None:
    print("== datasets")
    for tag, d, cfg in DATASETS:
        p = Path(d)
        stats = p / "dataset_stats.json"
        if not stats.exists():
            print(f"  {tag:28s} ABSENT")
            continue
        s = json.load(open(stats, encoding="utf-8"))
        fl = sum(v.get("num_unique_flights", 0) for v in s["splits"].values())
        pts = sum(v.get("num_5s_points", 0) for v in s["splits"].values())
        print(f"  {tag:28s} {s['num_total_days']:3d} days | {fl:>7,} flights | "
              f"{pts / 1e6:6.1f} M points | dt {s['dt_seconds']} s | "
              f"sig {s.get('dataset_signature', 'n/a')} | {size_of(p):5.1f} GB")

    print("\n== training runs")
    for tag, p in RUNS:
        if not Path(p).exists():
            print(f"  {tag:34s} ABSENT")
            continue
        d = json.load(open(p, encoding="utf-8"))
        if "test_overall_mde_3d_m" in d:
            o = d["test_overall_mde_3d_m"]
            ours = o.get("AeroTraj_Ultra_Ours")
            ep = d.get("optimization", {})
            print(f"  {tag:34s} ours {ours:9.1f} m | epochs "
                  f"{ep.get('epochs_executed', '-')}/{ep.get('epochs_planned', '-')} "
                  f"@ {ep.get('best_epoch', '-')}")
        else:
            print(f"  {tag:34s} {len(d)} entries (cross-budget comparison)")

    print("\n== benchmarks")
    for tag, p in BENCHES:
        if not Path(p).exists():
            print(f"  {tag:44s} ABSENT")
            continue
        d = json.load(open(p, encoding="utf-8"))
        best = min(d.items(), key=lambda kv: kv[1].get("overall_mde_3d_m", 1e18))
        print(f"  {tag:44s} {len(d):2d} models | best {best[0]} "
              f"{best[1].get('overall_mde_3d_m', float('nan')):.1f} m")

    print("\n== figures referenced by the documents")
    missing = [f for f in FIGURES if not Path(f).exists()]
    print(f"  {len(FIGURES) - len(missing)}/{len(FIGURES)} present"
          + (f" | MISSING: {missing}" if missing else ""))

    print("\n== disk")
    for d in ["data365", "data365_10s"]:
        p = Path(d)
        if p.exists():
            cache = p / "_cache_days"
            c = size_of(cache) if cache.exists() else 0.0
            print(f"  {d:14s} {size_of(p):6.1f} GB ({c:5.1f} GB of per-day cache)")
    for d in sorted(Path(".").glob("checkpoints*")):
        if d.is_dir():
            print(f"  {d.name:14s} {size_of(d):6.1f} GB")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Unified benchmark table: all the models evaluated on the SAME OpenSky 90-day
test split (9 days, 2022-03-23 -> 2022-03-31), horizon 5-300 s, metric = 3-D
ECEF mean distance error.

Collects:
  * results_bench/benchmark_results.json   (the 10-model benchmark)
  * results_bench/legacy/legacy_results.json (FlightPREV re-implemented)
  * results_hyp/retrieval/final_eval.json  (AeroTraj-Ultra + retrieval + multi-hypothesis)

Usage: python make_benchmark_report.py --out results_bench/BENCHMARK_ALL.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

HS = [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results_bench/BENCHMARK_ALL.md")
_args = ap.parse_args()


def load(p: str) -> Optional[Dict[str, Any]]:
    q = Path(p)
    if not q.exists():
        return None
    with open(q, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    bench = load("results_bench/benchmark_results.json") or {}
    legacy = load("results_bench/legacy/legacy_results.json")
    retr = load("results_hyp/retrieval/final_eval.json")

    rows = []

    def add(name, mde, per_h, extra=""):
        r = {"model": name, "mde_mean_m": round(mde, 1), "extra": extra}
        for h in HS:
            r[f"mde_{h}s"] = round(per_h[f"{h}s"]["mde_3d_m"], 1) if per_h and f"{h}s" in per_h else None
        rows.append(r)

    for name, res in bench.items():
        add(name, res["overall_mde_3d_m"], res.get("per_horizon"))

    if legacy:
        add("FlightPREV (legacy, re-impl.)", legacy["test"]["overall_mde_3d_m"],
            legacy["test"].get("per_horizon"))
    if retr:
        add("**AeroTraj-Ultra + retrieval**", retr["test"]["mde_deterministic_m"],
            {f"{h}s": {"mde_3d_m": retr["test"]["per_horizon_mde"][f"{h}s"]} for h in HS},
            f"minADE@8={retr['test']['minade_k_m']:,.0f} m")

    df = pd.DataFrame(rows).sort_values("mde_mean_m")
    cols = ["model", "mde_mean_m"] + [f"mde_{h}s" for h in HS]
    df = df[cols]

    lines = ["# Benchmark unificato - OpenSky 90 giorni, orizzonte 5 s -> 300 s\n",
             "Test set: 9 giorni (2022-03-23 -> 2022-03-31), stessi split e stessa metrica",
             "(MDE 3-D ECEF, metri) per tutti i modelli.\n",
             df.to_markdown(index=False),
             "\n## Note\n",
             "- `FlightPREV (legacy, re-impl.)`: architettura TrajTransformer del repo",
             "  MarcoValleCS/flight ri-implementata su OpenSky, orizzonte esteso a 5-300 s.",
             "  Gli embedding categorici originali (ac_type/operator/dep/arr) non esistono nei",
             "  dati OpenSky e sono sostituiti da due embedding sull'area geografica di",
             "  origine/destinazione del segmento: e' un handicap per il modello legacy.\n",
             "- `AeroTraj-Ultra + retrieval`: modello con prior d'intenzione recuperata dalle",
             "  traiettorie passate piu' simili + testa multi-iperspettrale (K=8).",
             ]
    if retr:
        c = retr["conformal"]
        lines += ["\n## Banda di incertezza calibrata (conformal, Mondrian per fase di volo)\n",
                  "| Livello nominale | Copertura (ensemble) | Raggio medio |",
                  "|---:|---:|---:|"]
        for a in ["alpha_0.2", "alpha_0.1", "alpha_0.05"]:
            if a in c:
                lines.append(f"| {1-float(a.split('_')[1]):.0%} | {c[a]['coverage']:.3f} | "
                             f"{c[a]['mean_radius_m']:,.0f} m |")

    txt = "\n".join(lines)
    Path(_args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(_args.out).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

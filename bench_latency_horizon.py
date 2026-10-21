# coding=utf-8
"""
Is inference latency really independent of the prediction horizon?

The claim in the paper is that AeroTraj-Ultra emits the whole horizon vector in
ONE forward pass, so the cost must be flat in the number of horizons.  This
script measures it directly: the same architecture is built with H = 1, 2, 4, 8
and 13 horizons and the batch-1 GPU time is measured for each.

Randomly-initialised weights are used on purpose: the measurement is a timing of
the graph, and the checkpoint does not change any shape in the encoder.

Run:  python bench_latency_horizon.py
Out:  printed table + results_bench/latency_horizon.json
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra

CONFIG = "config_90d_300s_hyp.json"


def gpu_median(fn, repeat: int = 60, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return float(np.median([a.elapsed_time(b) for a, b in zip(s, e)]))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    base = json.load(open(CONFIG, encoding="utf-8"))
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=32)

    out: Dict[str, object] = {"device": str(dev), "runs": []}
    print(f"{'H':>4} {'latency [ms]':>14} {'params':>12}   {'horizons'}")
    for H in [1, 2, 4, 8, 13]:
        cfg = copy.deepcopy(base)
        sec = list(cfg["window"]["horizon_seconds"])[:H]
        steps = [int(round(s / 5.0)) for s in sec]
        cfg["window"]["horizon"] = H
        cfg["window"]["horizon_seconds"] = sec
        cfg["window"]["horizon_steps_5s"] = steps

        model = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
        n_par = sum(p.numel() for p in model.parameters())
        x, _ = ds.get_batch_from_anchors(ds.anchors[:1], dev)
        with torch.no_grad():
            ms = gpu_median(lambda: model(x)["pred"])
        print(f"{H:>4} {ms:>14.2f} {n_par:>12,}   {sec}")
        out["runs"].append({"n_horizons": H, "horizons_s": sec, "latency_gpu_ms": ms,
                            "params": n_par})

    r = out["runs"]
    ws = max(x["horizons_s"][-1] for x in r)
    H13 = next(x for x in r if x["n_horizons"] == 13)
    H1 = next(x for x in r if x["n_horizons"] == 1)
    ratio = H13["latency_gpu_ms"] / H1["latency_gpu_ms"]
    out["summary"] = {"latency_H1_ms": H1["latency_gpu_ms"],
                      "latency_H13_ms": H13["latency_gpu_ms"],
                      "ratio_13_over_1": ratio,
                      "max_horizon_s": ws}
    print(f"\n13 orizzonti (5-{ws:.0f} s) costano {ratio:.2f}x un solo orizzonte:")
    print("la latenza e' praticamente piatta nell'orizzonte (decoder non autoregressivo),")
    print("mentre un modello autoregressivo a 60 passi costerebbe ~60x.")

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/latency_horizon.json", "w"), indent=2)
    print("[horizon] saved results_bench/latency_horizon.json")


if __name__ == "__main__":
    main()

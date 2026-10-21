# coding=utf-8
"""
Retrieval: the two blocking dimensions INTERACT.

`block` (queries per chunk) and `cand_block` (candidates per chunk) both divide
the same work, so the number of kernel launches is ~(B/block) * (M/cand_block).
Shrinking one while the other is small explodes the launch count, which is fatal
for a launch-bound model: naive combination (block=16, cand_block=16384) was
measured 13x SLOWER than the sensible one.

This script measures the grid and verifies that every configuration returns the
IDENTICAL neighbours, so any speed difference is a pure layout effect and can be
chosen freely.

Run:  python bench_retrieval_grid.py
Out:  results_bench/opt_retrieval_grid.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from retrieval import build_pool, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
POS_WEIGHT = 10.0
POOL_STRIDE = 16
CONFIGS = [(256, 16384), (256, None), (16, None), (16, 16384),
           (64, 16384), (512, 16384), (256, 4096)]
BATCHES = [512, 4096]


def gpu_ms(fn, repeat: int = 20, warmup: int = 5) -> float:
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
    dev = torch.device("cuda")
    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=8192)
    print("[grid] costruzione del pool ...")
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    print(f"[grid] pool: {pool.M:,} candidati")
    out: Dict[str, object] = {"n_candidates": int(pool.M), "results": {}}

    with torch.inference_mode():
        for B in BATCHES:
            x, _ = ds.get_batch_from_anchors(ds.anchors[:B], dev)
            desc = window_descriptor(x, POS_WEIGHT)
            print(f"\n=== batch {B} ===")
            row = {}
            ref = None
            torch.cuda.reset_peak_memory_stats()
            for blk, cb in CONFIGS:
                torch.cuda.reset_peak_memory_stats()
                fn = lambda blk=blk, cb=cb: pool.search(desc, k=8, block=blk, cand_block=cb)
                ms = gpu_ms(fn, repeat=20)
                idx, _d = fn()
                peak = torch.cuda.max_memory_allocated() / 2**20
                if ref is None:
                    ref = idx
                nd = int((idx != ref).any(dim=1).sum().item())
                row[f"block{blk}_cand{cb}"] = {"gpu_ms": ms, "us_per_query": ms * 1e3 / B,
                                               "peak_mib": peak, "n_queries_with_diff": nd}
                print(f"  block {blk:4d}  cand_block {str(cb):>6s}   {ms:8.2f} ms"
                      f"  {ms * 1e3 / B:7.2f} us/query   picco {peak:7.1f} MiB"
                      f"   query diverse {nd}")
            out["results"][str(B)] = row

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/opt_retrieval_grid.json", "w"), indent=2)
    print("\n[grid] saved results_bench/opt_retrieval_grid.json")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Retrieval search benchmark: blocked-candidate top-k versus the single-shot version.

Checks two things that must BOTH hold:
  1. the returned neighbours are identical (the ranking must not change, or the
     retrieval context changes and the prediction changes);
  2. the cost, which is the real blocker: the single-shot version materialises
     [block, M] = 256 x 342 572 x 4 B = 351 MB per query block, i.e. 5.6 GB of
     DRAM traffic at batch 4096, which is why the end-to-end retrieval pipeline
     stopped scaling after batch ~512.

Run:  python bench_retrieval.py
Out:  results_bench/opt_retrieval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from retrieval import build_pool, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
POS_WEIGHT = 10.0
POOL_STRIDE = 16
BATCHES = [1, 64, 512, 2048, 4096]


def gpu_ms(fn, repeat: int = 30, warmup: int = 5) -> float:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=30)
    args = ap.parse_args()

    dev = torch.device("cuda")
    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=8192)
    print("[retr] costruzione del pool ...")
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    print(f"[retr] pool: {pool.M:,} candidati")

    out: Dict[str, object] = {"n_candidates": int(pool.M), "k": 8, "results": {}}
    with torch.inference_mode():
        for B in BATCHES:
            x, _ = ds.get_batch_from_anchors(ds.anchors[:B], dev)
            desc = window_descriptor(x, POS_WEIGHT)
            print(f"\n=== batch {B} ===  ({B} query x {pool.M:,} candidati)")
            row: Dict[str, object] = {}
            ref_i = ref_d = None
            for label, cb in [("single_shot", None), ("blocked_16k", 16384),
                              ("blocked_64k", 65536), ("blocked_4k", 4096)]:
                fn = lambda cb=cb: pool.search(desc, k=8, cand_block=cb)
                ms = gpu_ms(fn, repeat=args.repeat)
                idx, dd = fn()
                if ref_i is None:
                    ref_i, ref_d = idx, dd
                same = int((idx == ref_i).sum().item())
                nd = int((idx != ref_i).any(dim=1).sum().item())
                rel = float(((dd - ref_d).abs() / (ref_d.abs() + 1e-6)).max())
                row[label] = {"gpu_ms": ms, "identical_neighbours": same == idx.numel(),
                              "n_queries_with_diff": nd,
                              "max_rel_dist_diff": rel,
                              "us_per_query": ms * 1e3 / B}
                print(f"  {label:12s} {ms:9.3f} ms   {ms * 1e3 / B:9.1f} us/query"
                      f"   identici {same}/{idx.numel()}   query con differenze {nd}"
                      f"   max rel d {rel:.2e}")
            out["results"][str(B)] = row

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/opt_retrieval.json", "w"), indent=2)
    print("\n[retr] saved results_bench/opt_retrieval.json")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Retrieval: where does the time actually go?

bench_retrieval.py showed that blocking over CANDIDATES buys only ~15%, because
the dominant cost is not the peak size of the [nb, M] distance matrix but the
TOTAL volume of it (B x M x 4 B = 5.6 GB of writes at batch 4096).  This script
tests the three levers that can actually move that:

  1. `block` (queries per chunk): a smaller chunk keeps the 8.2 MB candidate
     matrix resident in the 36 MB L2 while the distance block is written;
  2. fp16 distance computation: halves the write volume (ranking only);
  3. a pruned pool with ~one candidate per flight (M 342 572 -> ~75 000).

Run:  python bench_retrieval_block.py
Out:  results_bench/opt_retrieval_block.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from retrieval import RetrievalPool, anchor_flight_ids, build_pool, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
POS_WEIGHT = 10.0
POOL_STRIDE = 16


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


def search_fp16(pool: RetrievalPool, q: torch.Tensor, k: int = 8, block: int = 256):
    """Distance in fp16 (ranking only): halves the write volume of [nb, M]."""
    cq = pool.cand_desc.half()
    sq = (cq.float() ** 2).sum(dim=1)
    B = q.shape[0]
    idx = torch.empty((B, k), dtype=torch.long, device=q.device)
    d = torch.empty((B, k), dtype=torch.float32, device=q.device)
    for s in range(0, B, block):
        qb = q[s:s + block]
        d2 = ((qb ** 2).sum(1, keepdim=True) - 2.0 * (qb.half() @ cq.T)).float() + sq[None, :]
        dd, ii = torch.topk(d2, k, dim=1, largest=False)
        idx[s:s + block] = ii
        d[s:s + block] = dd
    return idx, d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device("cuda")

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=8192)
    print("[retr] costruzione del pool (stride 16) ...")
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    print(f"[retr] pool pieno: {pool.M:,} candidati")

    # pruned pool: at most `per_flight` candidate(s) per flight
    flights = anchor_flight_ids(ds_tr)[::POOL_STRIDE]
    keep = np.ones(len(flights), dtype=bool)
    seen = set()
    for i, f in enumerate(flights):
        if int(f) in seen:
            keep[i] = False
        else:
            seen.add(int(f))
    print(f"[retr] pool potato (1 per volo): {int(keep.sum()):,} candidati "
          f"({len(seen):,} voli distinti)")
    small = RetrievalPool(pool.cand_desc[torch.as_tensor(np.where(keep)[0], device=dev)].cpu().numpy(),
                          pool.cand_val[torch.as_tensor(np.where(keep)[0], device=dev)].cpu().numpy(),
                          flights[keep], dev, pos_weight=POS_WEIGHT)

    out: Dict[str, object] = {"full_candidates": int(pool.M),
                              "pruned_candidates": int(small.M), "results": {}}

    with torch.inference_mode():
        x, _ = ds.get_batch_from_anchors(ds.anchors[:4096], dev)
        desc = window_descriptor(x, POS_WEIGHT)

        print("\n=== 1) effetto di `block` (query per blocco), pool pieno, B=4096 ===")
        ref_i = ref_d = None
        for blk in [8, 16, 32, 64, 128, 256]:
            fn = lambda blk=blk: pool.search(desc, k=8, block=blk, cand_block=None)
            ms = gpu_ms(fn, repeat=args.repeat)
            idx, dd = fn()
            if ref_i is None:
                ref_i, ref_d = idx, dd
            nd = int((idx != ref_i).any(dim=1).sum().item())
            out["results"].setdefault("block_sweep", {})[str(blk)] = {
                "gpu_ms": ms, "us_per_query": ms * 1e3 / 4096, "n_queries_with_diff": nd}
            print(f"  block {blk:4d}  {ms:8.2f} ms  {ms * 1e3 / 4096:7.2f} us/query"
                  f"  query con differenze {nd}")

        print("\n=== 2) fp16 per le distanze, B=4096 ===")
        for blk in [16, 64, 256]:
            fn = lambda blk=blk: search_fp16(pool, desc, k=8, block=blk)
            ms = gpu_ms(fn, repeat=args.repeat)
            idx, dd = fn()
            nd = int((idx != ref_i).any(dim=1).sum().item())
            out["results"].setdefault("fp16", {})[str(blk)] = {
                "gpu_ms": ms, "us_per_query": ms * 1e3 / 4096, "n_queries_with_diff": nd}
            print(f"  fp16 block {blk:4d}  {ms:8.2f} ms  {ms * 1e3 / 4096:7.2f} us/query"
                  f"  query con differenze {nd}")

        print("\n=== 3) pool potato (1 candidato per volo), B=4096 ===")
        for blk in [16, 64, 256]:
            fn = lambda blk=blk: small.search(desc, k=8, block=blk, cand_block=None)
            ms = gpu_ms(fn, repeat=args.repeat)
            idx, dd = fn()
            out["results"].setdefault("pruned", {})[str(blk)] = {
                "gpu_ms": ms, "us_per_query": ms * 1e3 / 4096,
                "n_candidates": int(small.M)}
            print(f"  potato block {blk:4d}  {ms:8.2f} ms  {ms * 1e3 / 4096:7.2f} us/query")

        print("\n=== 4) pool potato, latenza per batch ===")
        for B in [1, 64, 512, 2048, 4096]:
            xb, _ = ds.get_batch_from_anchors(ds.anchors[:B], dev)
            dq = window_descriptor(xb, POS_WEIGHT)
            a = gpu_ms(lambda: pool.search(dq, k=8, block=16, cand_block=None), repeat=args.repeat)
            b = gpu_ms(lambda: small.search(dq, k=8, block=16, cand_block=None), repeat=args.repeat)
            out["results"].setdefault("by_batch", {})[str(B)] = {
                "full_ms": a, "pruned_ms": b, "speedup": a / b}
            print(f"  B {B:5d}  pieno {a:8.2f} ms   potato {b:8.2f} ms   x{a / b:4.2f}")

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/opt_retrieval_block.json", "w"), indent=2)
    print("\n[retr] saved results_bench/opt_retrieval_block.json")


if __name__ == "__main__":
    main()

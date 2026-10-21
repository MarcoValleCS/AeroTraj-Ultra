# coding=utf-8
"""
End-to-end latency AFTER the optimisations, versus the eager baseline.

Compares, on the same weights and the same windows:
  * eager + no_grad            (the original measurement, PAPER section 8)
  * eager + inference_mode
  * torch.compile(reduce-overhead) + inference_mode   <- the recommended path

for the deterministic model and for the retrieval-conditioned one, and reports
the one-off compile cost.

Run:  python bench_latency_opt.py
Out:  results_bench/latency_opt.json + results_bench/LATENCY_OPT.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from inference import FastPredictor
from retrieval import build_pool, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
CKPT = "checkpoints_bench/ours.pt"
CKPT_RETR = "checkpoints_hyp_reg/retrieval/aerotraj_ultra_retrieval_best.pt"
POS_WEIGHT = 10.0
POOL_STRIDE = 16
BATCHES = [1, 8, 32, 128, 512, 2048, 4096]
REPEAT_1 = 200


def sync() -> None:
    torch.cuda.synchronize()


def wall_ms(fn: Callable[[], None], repeat: int, warmup: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def gpu_median(fn: Callable[[], None], repeat: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    sync()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        s[i].record()
        fn()
        e[i].record()
    sync()
    return float(np.median([a.elapsed_time(b) for a, b in zip(s, e)]))


def main() -> None:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=32)
    ds_thr = FlightSlidingWindowDataset("test", CONFIG, max_samples=8192)

    print("[opt] costruzione del pool di retrieval ...")
    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    del ds_tr
    print(f"[opt] pool: {pool.M:,} candidati")

    out: Dict[str, object] = {"device": torch.cuda.get_device_name(0),
                              "pool_candidates": int(pool.M), "batches": BATCHES}

    # ---------------- 1) single aircraft, end to end ---------------- #
    print("\n=== Latenza a batch 1 (end-to-end, H2D inclusa) ===")
    x1_cpu = ds.get_batch_from_anchors(ds.anchors[:1], dev)[0].cpu()
    rows: List[tuple] = []
    for label, cm, ckpt, retr in [
        ("eager + no_grad", "none", CKPT, False),
        ("eager + inference_mode", "none", CKPT, False),
        ("compile(reduce-overhead)", "reduce-overhead", CKPT, False),
        ("+ retrieval, eager", "none", CKPT_RETR, True),
        ("+ retrieval, compiled", "reduce-overhead", CKPT_RETR, True),
    ]:
        fp = FastPredictor(CONFIG, ckpt, device="cuda", compile_mode=cm,
                           norm_mean=ds.norm_mean, norm_std=ds.norm_std,
                           retrieval_pool=pool if retr else None)
        warm = fp.warmup(1)
        if label == "eager + no_grad":
            fn = lambda fp=fp: fp.model(x1_cpu.to(dev))
            with torch.no_grad():
                t = wall_ms(fn, REPEAT_1)
                g = gpu_median(fn, 50)
        else:
            fn = lambda fp=fp: fp(x1_cpu.to(dev))
            t = wall_ms(fn, REPEAT_1)
            g = gpu_median(fn, 50)
        rows.append((label, t, g, warm))
        print(f"  {label:26s} {t['median_ms']:7.2f} ms  p95 {t['p95_ms']:7.2f}"
              f"  GPU {g:6.2f} ms  warmup {warm:7.0f} ms")
        del fp
        torch.cuda.empty_cache()
    out["batch1"] = [{"label": l, **t, "gpu_ms": g, "warmup_ms": w} for l, t, g, w in rows]

    # ---------------- 2) throughput ---------------- #
    print("\n=== Throughput ===")
    thr: Dict[str, Dict[str, float]] = {}
    for cm, retr in [("none", False), ("reduce-overhead", False),
                     ("none", True), ("reduce-overhead", True)]:
        label = ("retrieval " if retr else "base ") + ("compiled" if cm != "none" else "eager")
        fp = FastPredictor(CONFIG, CKPT_RETR if retr else CKPT, device="cuda", compile_mode=cm,
                           norm_mean=ds.norm_mean, norm_std=ds.norm_std,
                           retrieval_pool=pool if retr else None)
        thr[label] = {}
        with torch.inference_mode():
            for B in BATCHES:
                xb = ds_thr.get_batch_from_anchors(ds_thr.anchors[:B], dev)[0]
                fp.warmup(B)
                rep = {1: 60, 8: 40, 32: 30, 128: 20, 512: 12, 2048: 6, 4096: 4}[B]
                fn = lambda fp=fp, xb=xb: fp(xb)
                t = wall_ms(fn, rep, warmup=5)
                thr[label][str(B)] = {"total_ms": t["median_ms"],
                                      "per_aircraft_us": t["median_ms"] * 1e3 / B,
                                      "aircraft_per_s": B / (t["median_ms"] / 1e3)}
                print(f"  {label:20s} batch {B:5d}  {t['median_ms']:9.2f} ms"
                      f"  {t['median_ms'] * 1e3 / B:9.1f} us/aereo"
                      f"  {B / (t['median_ms'] / 1e3):10,.0f} aerei/s")
        del fp
        torch.cuda.empty_cache()
    out["throughput"] = thr

    # ---------------- 3) where the retrieval path loses time ---------------- #
    print("\n=== Scomposizione del percorso retrieval (batch 512) ===")
    B = 512
    xb = ds_thr.get_batch_from_anchors(ds_thr.anchors[:B], dev)[0]
    diag: Dict[str, float] = {}
    with torch.inference_mode():
        desc = window_descriptor(xb, POS_WEIGHT)
        diag["search_before_compile_ms"] = gpu_median(lambda: pool.search(desc, k=8), 20, 3)
        diag["context_before_compile_ms"] = gpu_median(lambda: pool.context(desc, xb, k=8), 20, 3)
    print(f"  search, prima di compilare un modello : {diag['search_before_compile_ms']:8.2f} ms")
    print(f"  context (search + rotazione)           : {diag['context_before_compile_ms']:8.2f} ms")

    for label, cm in [("eager", "none"), ("compiled", "reduce-overhead")]:
        fp = FastPredictor(CONFIG, CKPT_RETR, device="cuda", compile_mode=cm,
                           norm_mean=ds.norm_mean, norm_std=ds.norm_std, retrieval_pool=pool)
        fp.warmup(B)
        with torch.inference_mode():
            ctx = pool.context(window_descriptor(xb, POS_WEIGHT), xb, k=8)
            diag[f"search_with_{label}_model_ms"] = gpu_median(lambda: pool.search(desc, k=8), 20, 3)
            t_fwd = gpu_median(lambda: fp.fn(xb, retr_ctx=ctx), 20, 3)
            t_all = wall_ms(lambda: fp(xb), 20, warmup=3)["median_ms"]
        diag[f"forward_only_{label}_ms"] = t_fwd
        diag[f"full_{label}_ms"] = t_all
        print(f"  [{label:8s}] search con modello presente {diag[f'search_with_{label}_model_ms']:8.2f} ms"
              f" | forward da solo {t_fwd:7.2f} ms | pipeline completa {t_all:8.2f} ms")
        del fp
        torch.cuda.empty_cache()
    out["retrieval_diagnostic"] = diag

    json.dump(out, open("results_bench/latency_opt.json", "w"), indent=2)
    print("\n[opt] saved results_bench/latency_opt.json")


if __name__ == "__main__":
    main()

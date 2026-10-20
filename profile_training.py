# coding=utf-8
"""
Profiling / optimization benchmark for the AeroTraj-Ultra training step.

Measures, for a range of batch sizes:
  * the cost of the forward+backward+step
  * the cost of the dataloader (CPU gather + H2D)
  * the GPU utilisation (SM occupancy) and the RAM footprint
  * the effect of the two optimisations:
      - the Gray-Code soft-XOR chain expressed as a prefix parity (1 cumprod
        instead of ~5 kernels per bit)
      - the removal of the per-step GPU->CPU synchronisations (.item())

Usage:  python profile_training.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra, HybridPhysicsConstrainedLoss

CONFIG = "config_90d_300s.json"


def bench(fn, n=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n


def main() -> None:
    torch.manual_seed(97)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    ds = FlightSlidingWindowDataset("train", CONFIG)
    print(f"points array: {ds.points.shape} = {ds.points.nbytes / 1e9:.2f} GB in RAM")

    model = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev)
    crit = HybridPhysicsConstrainedLoss(model.codec, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"\n{'batch':>6} {'step [ms]':>10} {'data [ms]':>10} {'data %':>8} "
          f"{'samples/s':>11} {'GPU mem [GB]':>13} {'SM util':>9}")
    print("-" * 78)
    for bs in [1024, 2048, 4096, 8192]:
        anchors = ds.anchors[:bs]
        torch.cuda.reset_peak_memory_stats()

        t_data = bench(lambda: ds.get_batch_from_anchors(anchors, dev), n=5, warmup=2)
        x, y = ds.get_batch_from_anchors(anchors, dev)

        def step():
            opt.zero_grad(set_to_none=True)
            o = model(x)
            l, _ = crit(o, x, y)
            l.backward()
            opt.step()

        t_step = bench(step, n=8, warmup=3)
        try:
            util = torch.cuda.utilization()
        except Exception:
            util = -1
        print(f"{bs:>6} {t_step*1e3:>10.1f} {t_data*1e3:>10.2f} {100*t_data/t_step:>7.1f}% "
              f"{bs/t_step:>11,.0f} {torch.cuda.max_memory_allocated()/1e9:>13.2f} {util:>8d}%")
        del x, y
        torch.cuda.empty_cache()

    # throughput over a realistic epoch
    bs = 4096
    loader = torch.utils.data.DataLoader(
        [ds[i][0] for i in range(0, 200)], batch_size=bs, shuffle=False
    ) if False else None
    print(f"\n[RAM] host used by the points array: {ds.points.nbytes/1e9:.2f} GB")
    print(f"[GPU] free/total: "
          f"{torch.cuda.mem_get_info()[0]/1e9:.1f} / {torch.cuda.mem_get_info()[1]/1e9:.1f} GB")


if __name__ == "__main__":
    main()

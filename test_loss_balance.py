"""Decisive test: the BCE and the MSE of the Hybrid Constrained loss fight each
other on the binary code (the same value can be written with different bits).
Which weight makes the model learn the HIGH bits?"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

from bench_baselines import FlightBERTBaseline
from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset

cfg = json.load(open("config_365d_10s_bench.json", encoding="utf-8"))
hsteps = list(cfg["window"]["horizon_steps"])
dev = torch.device("cuda")

ds = FlightSlidingWindowDataset("train", "config_365d_10s_bench.json")
names = ["lon", "lat", "alt", "vx", "vy", "vz"]


def loss_fn(m, x, y, w_bce, w_mse):
    _, bits = m(x)
    target = m.codec.encode_flat(y[:, 0])
    bce = F.binary_cross_entropy(bits.clamp(1e-6, 1 - 1e-6), target)
    mse = F.smooth_l1_loss(m.codec.ste_codes(bits), m.codec.target_codes(y[:, 0]), beta=0.1)
    return w_bce * bce + w_mse * mse


def run(lr, w_bce, w_mse, steps=400, bs=512):
    torch.manual_seed(97)
    m = FlightBERTBaseline(hsteps, dt_sec=10.0).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    loader = FastSlidingWindowBatchLoader(ds, batch_size=bs, shuffle=True,
                                          max_samples_per_epoch=bs * steps, device=dev)
    it = iter(loader)
    m.train()
    t0 = time.time()
    for i in range(steps):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        opt.zero_grad()
        loss = loss_fn(m, x, y, w_bce, w_mse)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.5)
        opt.step()
    m.eval()
    dl = FastSlidingWindowBatchLoader(
        FlightSlidingWindowDataset("val", "config_365d_10s_bench.json", max_samples=2048),
        batch_size=512, shuffle=False, device=dev)
    x, y = next(iter(dl))
    with torch.no_grad():
        nxt, bits = m(x)
        tgt = m.codec.encode_flat(y[:, 0])
        b = (bits > 0.5).float()
        acc = ((b == tgt).float().mean() * 100).item()
        dec = m.codec.ste_decode(bits)
        lat = torch.deg2rad(y[:, 0, 1].abs().clamp(max=80))
        e = torch.sqrt(((dec[:, 0] - y[:, 0, 0]) * 111320 * torch.cos(lat)) ** 2
                       + ((dec[:, 1] - y[:, 0, 1]) * 111132) ** 2
                       + ((dec[:, 2] - y[:, 0, 2]) * 10) ** 2)
        off = 0
        highs = []
        for k, nm in enumerate(names):
            nb = m.codec.sizes[k]
            for j in range(min(2, nb)):
                highs.append(f"{nm}[{j}]={100*(b[:, off+j]==tgt[:, off+j]).float().mean():.0f}")
            off += nb
    print(f"  lr={lr:.0e} w_bce={w_bce} w_mse={w_mse}: bits {acc:.1f} %  "
          f"one-step 3-D {e.mean():9.0f} m  ({time.time()-t0:.0f} s)")
    print(f"    high bits: {' '.join(highs)}")
    return float(e.mean())


if __name__ == "__main__":
    run(1e-4, 1.0, 0.0)      # BCE only
    run(1e-4, 1.0, 0.1)      # BCE + a small MSE

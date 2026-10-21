# coding=utf-8
"""
Does torch.compile change the PREDICTION QUALITY?

bench_opt.py shows a 6x batch-1 speed-up with torch.compile, but also a
numerical difference against eager.  This script answers the only question that
matters: does the 3-D ECEF error change?

  1. per-channel max|eager - compiled| on 'pred', so the difference can be read
     in physical units (deg / 10 m / km/h / m/s) instead of raw tensor units;
  2. the MDE and the per-horizon MDE for BOTH variants on the same test windows.

Run:  python check_compile_quality.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra
from run import compute_trajectory_metrics

CONFIG = "config_90d_300s_hyp.json"
CH = ["lon(deg)", "lat(deg)", "alt(10 m)", "vx(km/h)", "vy(km/h)", "vz(m/s)"]


def main() -> None:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=24576)
    bs = 2048

    model = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
    ck = torch.load("checkpoints_bench/ours.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)

    print("[chk] compilazione ...")
    t0 = time.perf_counter()
    cm = torch.compile(model, mode="reduce-overhead")
    with torch.inference_mode():
        cm(torch.zeros(4, 12, 9, device=dev))
    print(f"[chk] compilato in {time.perf_counter() - t0:.1f} s")

    A, Bp, T = [], [], []
    n_diff_max = 0.0
    ch_max = np.zeros(6)
    with torch.inference_mode():
        for s in range(0, len(ds.anchors), bs):
            x, y = ds.get_batch_from_anchors(ds.anchors[s:s + bs], dev)
            pe = model(x)["pred"]
            pc = cm(x)["pred"]
            d = (pe - pc).abs().amax(dim=(0, 1)).cpu().numpy()
            ch_max = np.maximum(ch_max, d)
            n_diff_max = max(n_diff_max, float((pe - pc).abs().max()))
            A.append(pe.float().cpu().numpy())
            Bp.append(pc.float().cpu().numpy())
            T.append(y.cpu().numpy())
    A = np.concatenate(A); Bp = np.concatenate(Bp); T = np.concatenate(T)

    print(f"\nfinestre di test: {len(T):,}")
    print(f"max |eager - compiled| globale : {n_diff_max:.4e}  (unita' del tensore)")
    print("per canale:")
    for i, c in enumerate(CH):
        print(f"   {c:11s} {ch_max[i]:12.4e}")

    # how many windows are actually affected, and by how much, in metres
    mlon = 111320.0 * np.cos(np.deg2rad(T[:, :, 1]))
    d_lon = (A[..., 0] - Bp[..., 0]) * mlon
    d_lat = (A[..., 1] - Bp[..., 1]) * 111132.0
    d_alt = (A[..., 2] - Bp[..., 2]) * 10.0
    dist = np.sqrt(d_lon ** 2 + d_lat ** 2 + d_alt ** 2)
    per_win = dist.max(axis=1)
    print(f"\ndistanza fisica |eager - compiled| per finestra (metri):")
    for q in [50, 90, 99, 99.9]:
        print(f"   p{q:<5} {np.percentile(per_win, q):10.3f}")
    print(f"   max   {per_win.max():10.3f}")
    for thr in [0.001, 0.01, 1.0, 10.0, 100.0]:
        n = int((per_win > thr).sum())
        print(f"   finestre con differenza > {thr:>7} m : {n:6d}  ({100 * n / len(per_win):.3f} %)")

    me = compute_trajectory_metrics(A, T, hs)
    mc = compute_trajectory_metrics(Bp, T, hs)
    print(f"\nMDE 3-D ECEF   eager    : {me['overall_mde_3d_m']:9.2f} m")
    print(f"MDE 3-D ECEF   compiled : {mc['overall_mde_3d_m']:9.2f} m")
    print(f"delta                   : {mc['overall_mde_3d_m'] - me['overall_mde_3d_m']:+.3f} m "
          f"({100 * (mc['overall_mde_3d_m'] / me['overall_mde_3d_m'] - 1):+.4f} %)")

    adv = float(np.abs(A - Bp).max())
    print("\nper orizzonte (m):")
    for h in hs:
        k = f"{int(h)}s"
        a = me["per_horizon"][k]["mde_3d_m"]; b = mc["per_horizon"][k]["mde_3d_m"]
        print(f"   {int(h):4d} s  eager {a:10.2f}   compiled {b:10.2f}   d {b - a:+8.3f}")

    json.dump({"max_abs_diff": n_diff_max, "per_channel_max": ch_max.tolist(),
               "channels": CH,
               "mde_eager": me["overall_mde_3d_m"], "mde_compiled": mc["overall_mde_3d_m"],
               "n_windows": int(len(T))},
              open("results_bench/compile_quality.json", "w"), indent=2)
    print("\n[chk] saved results_bench/compile_quality.json")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Where does a model gain over another?  Two checkpoints on the same 60 000 test
windows, split by the altitude of the aircraft and by what it does in the next
300 s (turn, altitude change), plus a paired bootstrap of the mean difference.

Run:  python eval_where_gain.py BASE.pt NEW.pt OUT.json
      (defaults: the base model and AeroTraj-Atlas at the benchmark budget)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from bench_run import horizontal_and_3d_errors
from dataloader import FlightSlidingWindowDataset
from eval_ckpt import load_model

CONFIG = "config_365d_12pt_bench.json"
DEV = torch.device("cuda")


def main() -> None:
    base_ck, new_ck, out_path = (sys.argv[1:4] if len(sys.argv) >= 4 else (
        "checkpoints_window/T12_s97/aerotraj_ultra_best.pt",
        "checkpoints_atlas/atlas_s97/aerotraj_ultra_best.pt",
        "results_atlas/where_gain_s97.json"))
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=60000)

    @torch.inference_mode()
    def errors(ck: str) -> np.ndarray:                      # [N, H] metres
        m, _, _ = load_model(ck, DEV)
        out = []
        for s in range(0, len(ds.anchors), 4096):
            x, y = ds.get_batch_from_anchors(ds.anchors[s:s + 4096], DEV)
            out.append(horizontal_and_3d_errors(m(x)["pred"].float().cpu().numpy(), y.cpu().numpy())[0])
        return np.concatenate(out)

    base, new = errors(base_ck), errors(new_ck)
    X, Y = (t.numpy() for t in ds.get_batch_from_anchors(ds.anchors, torch.device("cpu")))
    alt_m = X[:, -1, 2] * 10.0
    psi0 = np.arctan2(X[:, -1, 3], X[:, -1, 4])
    psi1 = np.arctan2(Y[:, -1, 3], Y[:, -1, 4])
    turn = np.degrees(np.abs(np.arctan2(np.sin(psi1 - psi0), np.cos(psi1 - psi0))))   # heading change at 300 s
    dalt = np.abs(Y[:, -1, 2] - X[:, -1, 2]) * 10.0                                    # altitude change at 300 s
    groups = {
        "all": np.ones(len(alt_m), bool),
        "altitude < 3 km": alt_m < 3000,
        "altitude 3-7.5 km": (alt_m >= 3000) & (alt_m < 7500),
        "altitude >= 7.5 km": alt_m >= 7500,
        "turns > 15 deg within 300 s": turn > 15,
        "turns <= 15 deg": turn <= 15,
        "altitude change > 300 m": dalt > 300,
        "level (<= 300 m)": dalt <= 300,
        "straight and level": (turn <= 15) & (dalt <= 300),
    }
    res = {"base": base_ck, "new": new_ck, "groups": {}}
    print(f"{'group':30s} {'share':>6s} {'base':>8s} {'new':>8s} {'delta':>8s} | 300 s: {'base':>8s} {'new':>8s} {'delta':>8s}")
    for k, g in groups.items():
        b, n, b3, n3 = base[g].mean(), new[g].mean(), base[g, -1].mean(), new[g, -1].mean()
        res["groups"][k] = {"share": float(g.mean()), "base_m": float(b), "new_m": float(n),
                            "base_300s_m": float(b3), "new_300s_m": float(n3)}
        print(f"{k:30s} {100 * g.mean():5.1f}% {b:8.1f} {n:8.1f} {100 * (n / b - 1):+7.1f}% |"
              f"        {b3:8.1f} {n3:8.1f} {100 * (n3 / b3 - 1):+7.1f}%")

    # paired bootstrap on the per-window means: the two models see the same windows
    rng = np.random.default_rng(0)
    d = new.mean(1) - base.mean(1)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res["paired_mean_diff_m"] = {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}
    res["share_of_windows_improved"] = float((d < 0).mean())
    print(f"\npaired difference {d.mean():+.1f} m, 95 % bootstrap CI [{lo:+.1f}, {hi:+.1f}] | "
          f"windows improved: {100 * (d < 0).mean():.1f} %")
    json.dump(res, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()

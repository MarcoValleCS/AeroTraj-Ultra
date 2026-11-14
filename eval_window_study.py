# coding=utf-8
"""
The observation-window study, on IDENTICAL test windows.

Two models with different window lengths do not have the same valid anchors: a
window needs `inp_seq_len - 1` points of history, so the test set of the 10 s
grid (12 points = 120 s) only holds windows with at least 120 s of history,
while the one of the 5 s grid (12 points = 60 s) holds every window with 60 s.
Comparing each model on its own test set therefore confounds the window length
with the window SET.  This script removes the confound:

  1. the full-budget 5 s model and the full-budget 10 s model on the same
     anchors (those of a 24-point 5 s window, where both have their history);
     the 10 s model reads every second point of the same window;
  2. the error of the 5 s model on the windows that the 120 s requirement
     removes (history between 60 s and 115 s).

Run:  python eval_window_study.py
Out:  results_window/same_windows_5s_vs_10s.json
      results_window/short_history_windows.json
The 12-vs-24-point comparison at equal resolution is `eval_cross.py --split`.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from bench_run import horizontal_and_3d_errors
from dataloader import FlightSlidingWindowDataset
from eval_ckpt import load_model

CFG_12, CFG_24 = "config_365d_12pt_bench.json", "config_365d_24pt_bench.json"
CKPT_5S = "checkpoints_365_1200k/aerotraj_ultra_best.pt"
CKPT_10S = "checkpoints_365_10s/aerotraj_ultra_best.pt"
SHARED = [10, 30, 60, 90, 180, 300]          # the horizons both grids predict
N = 60000
DEV = torch.device("cuda")


@torch.inference_mode()
def errors(model, ds, anchors, pick, pred_idx, targ_idx) -> np.ndarray:
    """Mean 3-D error [m] per horizon of `model` on `anchors` (it reads `pick(x)`)."""
    out = []
    for s in range(0, len(anchors), 4096):
        x, y = ds.get_batch_from_anchors(anchors[s:s + 4096], DEV)
        p = model(pick(x))["pred"][:, pred_idx].float().cpu().numpy()
        out.append(horizontal_and_3d_errors(p, y[:, targ_idx].cpu().numpy())[0])
    return np.concatenate(out).mean(0)


def main() -> None:
    m5, cfg5, _ = load_model(CKPT_5S, DEV)
    m10, cfg10, _ = load_model(CKPT_10S, DEV)
    h5 = [int(s) for s in cfg5["window"]["horizon_seconds"]]
    h10 = [int(s) for s in cfg10["window"]["horizon_seconds"]]
    sel5, sel10 = [h5.index(h) for h in SHARED], [h10.index(h) for h in SHARED]

    # ---- 1) the two grids on identical windows ----------------------------- #
    same = {}
    for cfg_path in (CFG_12, CFG_24):
        ds = FlightSlidingWindowDataset("test", cfg_path, max_samples=N)
        T = ds.inp_seq_len
        rows = {"5s grid, 60 s of history": errors(m5, ds, ds.anchors, lambda x: x[:, -12:], sel5, sel5)}
        if T >= 23:                               # 12 points at 10 s = the last 23 at 5 s, one in two
            rows["10s grid, 120 s of history"] = errors(
                m10, ds, ds.anchors, lambda x: x[:, -23::2], sel10, sel5)
        for name, e in rows.items():
            same[f"{name} | anchors of the {T}-pt window"] = {f"{h}s": float(v) for h, v in zip(SHARED, e)}
            print(f"{name:27s} | anchors T={T:2d} | "
                  + "  ".join(f"{h}s {v:7.1f}" for h, v in zip(SHARED, e)) + f" | mean {e.mean():7.1f}")
    json.dump(same, open("results_window/same_windows_5s_vs_10s.json", "w"), indent=2)

    # ---- 2) the windows a 120 s history requirement removes ---------------- #
    d12 = FlightSlidingWindowDataset("test", CFG_12)            # every strided window
    d24 = FlightSlidingWindowDataset("test", CFG_24)
    short = np.setdiff1d(d12.anchors, d24.anchors)
    rng = np.random.default_rng(0)
    full = lambda x: x
    idx = list(range(len(h5)))
    a = errors(m5, d12, np.sort(rng.choice(short, N, replace=False)), full, idx, idx)
    b = errors(m5, d12, np.sort(rng.choice(d24.anchors, N, replace=False)), full, idx, idx)
    print(f"\nstrided test windows: {len(d12.anchors):,} with >= 60 s of history, {len(d24.anchors):,} "
          f"with >= 120 s, {len(short):,} in between ({100 * len(short) / len(d12.anchors):.2f} %), "
          f"{len(d12.segments):,} segments")
    print(f"history <  120 s: mean {a.mean():7.1f} m, 300 s {a[-1]:7.1f} m")
    print(f"history >= 120 s: mean {b.mean():7.1f} m, 300 s {b[-1]:7.1f} m  ({a.mean() / b.mean():.2f}x)")
    json.dump({
        "n_windows_60s": int(len(d12.anchors)), "n_windows_120s": int(len(d24.anchors)),
        "n_short_history": int(len(short)), "model": CKPT_5S, "sample": N,
        "short_history_mde_m": {f"{h}s": float(v) for h, v in zip(h5, a)},
        "short_history_mean_m": float(a.mean()),
        "long_history_mde_m": {f"{h}s": float(v) for h, v in zip(h5, b)},
        "long_history_mean_m": float(b.mean()),
    }, open("results_window/short_history_windows.json", "w"), indent=2)


if __name__ == "__main__":
    main()

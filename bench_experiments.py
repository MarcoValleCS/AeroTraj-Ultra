# coding=utf-8
"""
Post-benchmark experiments on the trained checkpoints (90 days, horizon 5-300 s):

  EXP-1  Ensemble of the trained models (ours + FlightBERT++ V2 + Deep LSTM).
  EXP-2  MC-dropout ensemble of our model at inference (uncertainty + better mean).
  EXP-3  Anchor-denoising hypothesis: how much of the 5 s error is irreducible
         (quantisation of the LAST OBSERVED position) vs learnable?

Usage:  python bench_experiments.py
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch

from bench_run import (DEVICE, _raw_module, horizontal_and_3d_errors, min_ade_k)
from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import (
    AeroTrajUltra,
    ConstantVelocityKalmanBaseline,
    FlightBERT_PP_V2_Baseline,
    LSTMRegressionBaseline,
    SmoothedConstantVelocityBaseline,
)

CONFIG = "config_90d_300s.json"
MAX_TEST = 60000
N_SAMPLES = 10


def _load(module, name):
    """Load a checkpoint tolerating buffers added after it was trained
    (the Gray-Code constant tables are deterministic, so strict=False is safe)."""
    state = torch.load(f"checkpoints_bench/{name}.pt", map_location=DEVICE,
                       weights_only=False)["model_state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing:
        print(f"    [load {name}] missing (re-initialised, deterministic): {missing}")
    return module


def load_preds(name: str, cfg, ds, loader):
    """Return the test-set predictions of a trained checkpoint."""
    if name == "ours":
        m = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(DEVICE)
        _load(m, "ours")
    elif name == "fbpp_v2":
        m = FlightBERT_PP_V2_Baseline(cfg).to(DEVICE)
        _load(m, "fbpp_v2")
    elif name == "lstm":
        m = LSTMRegressionBaseline(horizon=cfg["window"]["horizon"],
                                   horizon_seconds=cfg["window"]["horizon_seconds"],
                                   norm_mean=ds.norm_mean, norm_std=ds.norm_std).to(DEVICE)
        _load(m, "lstm")
    else:
        raise ValueError(name)
    m.eval()
    preds, targs = [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            if name == "fbpp_v2":
                p, _ = m(x_b)
            else:
                out_m = m(x_b)
                p = out_m["pred"] if isinstance(out_m, dict) else out_m
            preds.append(p.cpu().numpy())
            targs.append(y_b.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targs)


def main() -> None:
    torch.manual_seed(97)
    np.random.seed(97)
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]

    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=MAX_TEST)
    loader = FastSlidingWindowBatchLoader(ds, batch_size=2048, shuffle=False, device=DEVICE)
    T = None
    out: Dict[str, object] = {}

    # ---------------- EXP-1: ensembles ---------------- #
    P = {}
    for name in ["ours", "fbpp_v2", "lstm"]:
        p, T = load_preds(name, cfg, ds, loader)
        P[name] = p
        d3, _ = horizontal_and_3d_errors(p, T)
        out[f"single:{name}"] = float(d3.mean())
        print(f"  single {name:9s} MDE = {d3.mean():9,.1f} m")

    for combo in [("ours", "fbpp_v2"), ("ours", "lstm"), ("ours", "fbpp_v2", "lstm")]:
        avg = np.mean([P[c] for c in combo], axis=0)
        d3, _ = horizontal_and_3d_errors(avg, T)
        out["ens:" + "+".join(combo)] = float(d3.mean())
        print(f"  ensemble {'+'.join(combo):24s} MDE = {d3.mean():9,.1f} m")

    # ---------------- EXP-2: MC-dropout on our model ---------------- #
    m = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(DEVICE)
    m.load_state_dict(torch.load("checkpoints_bench/ours.pt", map_location=DEVICE,
                                weights_only=False)["model_state_dict"])
    # enable dropout at inference (MC-dropout)
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.1
    m.train()   # dropout active, but no BN in this model
    mc = []
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            ps = []
            for x_b, _ in loader:
                ps.append(m(x_b)["pred"].cpu().numpy())
            mc.append(np.concatenate(ps))
    MC = np.stack(mc, axis=0)                 # [K,N,H,6]
    d3_mean, _ = horizontal_and_3d_errors(MC.mean(axis=0), T)
    out["mcdropout:mean"] = float(d3_mean.mean())
    d3_each = np.mean([horizontal_and_3d_errors(MC[k], T)[0].mean() for k in range(N_SAMPLES)])
    out["mcdropout:single"] = float(d3_each)
    out["mcdropout:minADE_k"] = float(min_ade_k(MC.transpose(1, 0, 2, 3), T, N_SAMPLES)["minADE_k_m"])
    print(f"  MC-dropout  single draw   MDE = {d3_each:9,.1f} m")
    print(f"  MC-dropout  mean of {N_SAMPLES}     MDE = {d3_mean.mean():9,.1f} m")
    print(f"  MC-dropout  minADE@{N_SAMPLES}      = {out['mcdropout:minADE_k']:9,.1f} m")

    # ---------------- EXP-3: anchor quantisation floor ---------------- #
    # The prediction is anchored on the LAST OBSERVED position, whose error is
    # the ADS-B CPR quantisation.  A least-squares line fitted to the 12-point
    # window denoises it.  How much of the 5 s error does that noise explain?
    x_all = np.concatenate([x_b.cpu().numpy() for x_b, _ in loader], axis=0)   # [N,12,9]
    t = (np.arange(x_all.shape[1]) - (x_all.shape[1] - 1)) * 5.0             # s, last = 0
    s0 = len(t)
    s1 = t.sum()
    s2 = (t ** 2).sum()
    det = s0 * s2 - s1 ** 2
    sum_x = x_all[:, :, :3].sum(axis=1)                    # [N,3]
    sum_tx = (t[None, :, None] * x_all[:, :, :3]).sum(axis=1)
    a0 = (s2 * sum_x - s1 * sum_tx) / det                   # fitted value at t=0
    a1 = (s0 * sum_tx - s1 * sum_x) / det                   # fitted slope (deg/s)

    mlon = 111320.0 * np.cos(np.radians(x_all[:, -1, 1]))
    mlat = 111132.0
    anchor_noise_m = np.sqrt(((x_all[:, -1, 0:1] - a0[:, 0:1]) * mlon[:, None]) ** 2
                             + ((x_all[:, -1, 1:2] - a0[:, 1:2]) * mlat) ** 2
                             + ((x_all[:, -1, 2:3] - a0[:, 2:3]) * 10.0) ** 2).mean(axis=1)
    print(f"\n  EXP-3  last-observed position vs least-squares fit at t=0:")
    print(f"         anchor noise (1-sigma, 3D)   = {anchor_noise_m.mean():6.1f} m "
          f"(p90 = {np.percentile(anchor_noise_m, 90):.1f} m)")

    # 5 s prediction: (a) hold the last position, (b) LS extrapolation
    t5 = 5.0
    ls5 = a0 + t5 * a1
    def _err(pred_xyz, true_xyz):
        dlon = (pred_xyz[:, 0:1] - true_xyz[:, 0:1]) * mlon[:, None]
        dlat = (pred_xyz[:, 1:2] - true_xyz[:, 1:2]) * mlat
        dz = (pred_xyz[:, 2:3] - true_xyz[:, 2:3]) * 10.0
        return np.sqrt(dlon ** 2 + dlat ** 2 + dz ** 2).mean(axis=1)
    true5 = x_all[:, -1, :3] + 0.0   # placeholder, replaced below
    # the true 5-s-ahead position is the FIRST TARGET of the window, not available
    # here -> use the model's own targets instead (y at +5 s)
    y5 = np.concatenate([y_b.cpu().numpy() for _, y_b in loader], axis=0)[:, 0, :3]
    e_hold = _err(x_all[:, -1, :3], y5)
    e_lsq = _err(ls5, y5)
    e_doppler = _err(x_all[:, -1, :3] + t5 * np.concatenate(
        [x_all[:, -1, 3:4] / 3.6 / 3600.0 * mlon[:, None],
         x_all[:, -1, 4:5] / 3.6 / 3600.0,
         x_all[:, -1, 5:6] / 3.6 / 3600.0], axis=-1), y5)
    print(f"         5 s error, holding the last position   = {e_hold.mean():6.1f} m")
    print(f"         5 s error, CV with the Doppler velocity  = {e_doppler.mean():6.1f} m")
    print(f"         5 s error, least-squares extrapolation  = {e_lsq.mean():6.1f} m")
    out["exp3_anchor_noise_m"] = float(anchor_noise_m.mean())
    out["exp3_5s_hold_m"] = float(e_hold.mean())
    out["exp3_5s_doppler_m"] = float(e_doppler.mean())
    out["exp3_5s_least_squares_m"] = float(e_lsq.mean())

    json.dump(out, open("results_bench/post_experiments.json", "w"), indent=2)
    print("\n[exp] saved to results_bench/post_experiments.json")


if __name__ == "__main__":
    main()

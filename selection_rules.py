# coding=utf-8
"""
Can the oracle gap be closed WITHOUT a learned score head?

The learned score (CE or regression) selects a hypothesis from a per-mode
embedding and reaches 1031 m, still worse than the deterministic output
(957 m).  The oracle is 613 m, so a lot of information is there.

This script tests selection rules that need NO training at all, computed from
the hypotheses themselves and from the observation:

  R1  closest to the deterministic output  (the deterministic head is the
      best predictor; the hypothesis closest to it is the most likely one)
  R2  kinematic consistency: the hypothesis whose implied turn rate / speed
      / vertical rate best matches the behaviour observed in the window
  R3  combination of R1 and R2

Usage: python selection_rules.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltraRetrieval
from retrieval import build_pool, window_descriptor

CONFIG = "config_90d_300s_regress.json"
CKPT = "checkpoints_hyp_reg/retrieval/aerotraj_ultra_retrieval_best.pt"
K = 8
POS_WEIGHT = 10.0
POOL_STRIDE = 16


def main() -> None:
    dev = torch.device("cuda")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]
    tr = cfg["training"]

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds_te = FlightSlidingWindowDataset("test", CONFIG, max_samples=int(tr["max_test_samples"]))
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    model = AeroTrajUltraRetrieval(cfg, ds_tr.norm_mean, ds_tr.norm_std, n_neighbours=K).to(dev)
    st = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()

    loader = FastSlidingWindowBatchLoader(ds_te, batch_size=2048, shuffle=False, device=dev)
    det, hyp, score, targ, obs = [], [], [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            ctx = pool.context(window_descriptor(x_b, POS_WEIGHT), x_b, k=K)
            out = model(x_b, retr_ctx=ctx)
            det.append(out["pred"].cpu().numpy())
            hyp.append(out["hyp_pred"].cpu().numpy())
            score.append(out["hyp_score"].cpu().numpy())
            targ.append(y_b.cpu().numpy())
            obs.append(x_b.cpu().numpy())

    D = np.concatenate(det)          # [N,H,6]
    Hh = np.concatenate(hyp)         # [N,K,H,6]
    S = np.concatenate(score)        # [N,K]
    T = np.concatenate(targ)         # [N,H,6]
    O = np.concatenate(obs)          # [N,T,9]

    # ---- per-mode 3D error to the truth -------------------------------- #
    lat_r = np.radians(T[:, None, :, 1])
    mlon = 111320.0 * np.cos(lat_r)
    dh = Hh - T[:, None, :, :]
    dist = np.sqrt((dh[..., 0] * mlon) ** 2 + (dh[..., 1] * 111132.0) ** 2
                   + (dh[..., 2] * 10.0) ** 2)          # [N,K,H]
    ade_k = dist.mean(axis=2)                             # [N,K]

    # ---- R1: closest to the deterministic output ----------------------- #
    dd = Hh - D[:, None, :, :]
    lat_r2 = np.radians(D[:, None, :, 1])
    mlon2 = 111320.0 * np.cos(lat_r2)
    d1 = np.sqrt((dd[..., 0] * mlon2) ** 2 + (dd[..., 1] * 111132.0) ** 2
                 + (dd[..., 2] * 10.0) ** 2).mean(axis=2)   # [N,K]

    # ---- R2: kinematic consistency with the observed behaviour ---------- #
    # observed turn rate over the last 15 s (from the Doppler velocity)
    v = O[:, :, 3:5]
    psi = np.arctan2(v[..., 0], v[..., 1])
    dpsi = np.unwrap(psi, axis=1)
    obs_turn = (dpsi[:, -1] - dpsi[:, -4]) / 15.0          # rad/s
    obs_spd = np.linalg.norm(v[:, -1, :], axis=-1)          # km/h
    obs_vs = O[:, -1, 5]                                    # m/s
    # implied by each hypothesis: turn rate and speed over the first 60 s
    tsec = np.array(hs, dtype=np.float64)
    def implied(h):
        # local-frame components are not directly available in the global output:
        # reconstruct the track angle from the global positions
        o_lon = O[:, None, -1, 0][:, :, None]             # [N,1,1]
        o_lat = O[:, None, -1, 1][:, :, None]             # [N,1,1]
        dx = (h[..., 0] - o_lon) * 111320.0 * np.cos(np.radians(o_lat))
        dy = (h[..., 1] - o_lat) * 111132.0
        trk = np.unwrap(np.arctan2(dx, dy), axis=-1)
        turn = (trk[..., 2] - trk[..., 0]) / max(tsec[2], 1.0)
        spd = np.linalg.norm(np.stack([dx, dy], axis=-1), axis=-1) / np.maximum(tsec, 1.0) * 3.6
        return turn, spd
    turn_h, spd_h = implied(Hh)
    e_turn = np.abs(turn_h - obs_turn[:, None]) * 1000.0     # mrad-like
    e_spd = np.abs(spd_h[..., 2] - obs_spd[:, None]) / 50.0
    e_vs = np.abs(Hh[..., 5].mean(axis=2) - obs_vs[:, None]) / 5.0
    r2 = e_turn + e_spd + e_vs

    rules = {
        "deterministico": None,
        "score-argmin (regress)": S.argmin(axis=1),
        "R1 piu vicino al deterministico": d1.argmin(axis=1),
        "R2 coerenza cinematica": r2.argmin(axis=1),
        "R3 R1 + R2": (d1 / d1.mean() + r2 / r2.mean()).argmin(axis=1),
        "orakle": ade_k.argmin(axis=1),
    }
    print(f"\n{'regola':34s} {'MDE [m]':>10s} {'vs det':>9s}")
    base = float(ade_k[np.arange(len(T)), rules["deterministico"] or 0].mean()) if False else None
    det_mde = float(dist[np.arange(len(T)), 0].mean()) * 0  # placeholder
    for name, sel in rules.items():
        if sel is None:
            continue
        m = float(ade_k[np.arange(len(T)), sel].mean())
        print(f"{name:34s} {m:10,.1f}")
    print(f"{'orakle (minADE@8)':34s} {float(ade_k.min(axis=1).mean()):10,.1f}")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Decisive probe: does the retrieval prior carry signal?

For every test window we retrieve the k nearest past trajectories from the
TRAINING set and build a "retrieval prediction" (the mean of the neighbours'
futures, expressed in the query's local frame).  We then compare it with

  * the AeroTraj-Ultra prediction,
  * the constant-velocity (Doppler) baseline,

per horizon.  If the retrieval prediction beats the kinematic baseline at
60-300 s, the prior carries signal and the model integration is justified;
if it does not, the whole retrieval direction is not worth the effort.

Usage:  python retrieval_probe.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltra, ConstantVelocityKalmanBaseline
from retrieval import build_index, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
K = 8
STRIDE = 32
MAX_TEST = 20000


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]
    tr = cfg["training"]

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds_te = FlightSlidingWindowDataset("test", CONFIG, max_samples=MAX_TEST)
    print(f"[probe] candidati nell'indice: {len(ds_tr.anchors)//STRIDE:,} finestre (stride {STRIDE})")

    print("[probe] building the retrieval index on the TRAIN split ...")
    index = build_index(ds_tr, stride=STRIDE, device=dev)

    # the model (deterministic output) and the kinematic baseline
    model = AeroTrajUltra(cfg, ds_tr.norm_mean, ds_tr.norm_std).to(dev)
    st = torch.load("checkpoints_hyp/aerotraj_ultra_best.pt", map_location=dev, weights_only=False)
    missing, unexpected = model.load_state_dict(st["model_state_dict"], strict=False)
    if missing:
        print(f"  [load] missing: {missing}")
    model.eval()
    cv = ConstantVelocityKalmanBaseline(hs).to(dev)

    loader = FastSlidingWindowBatchLoader(ds_te, batch_size=1024, shuffle=False, device=dev)

    rows = {"retr_mean": [], "retr_medoid": [], "model": [], "cv": [], "truth": []}
    with torch.no_grad():
        for x_b, y_b in loader:
            desc = window_descriptor(x_b)
            ctx = index.context(desc, x_b, k=K)               # [B,K,H,4]
            # the retrieval prediction: the mean over the neighbours, in the
            # query's local frame -> converted to global coordinates
            f = ctx[..., 0].mean(dim=1).unsqueeze(-1)      # [B,H,1] forward [m]
            r = ctx[..., 1].mean(dim=1).unsqueeze(-1)      # [B,H,1] right   [m]
            u = ctx[..., 2].mean(dim=1).unsqueeze(-1)      # [B,H,1] up      [m]
            lon0 = x_b[:, -1, 0:1].unsqueeze(1)          # [B,1,1]
            lat0 = x_b[:, -1, 1:2].unsqueeze(1)          # [B,1,1]
            alt0 = x_b[:, -1, 2:3].unsqueeze(1)          # [B,1,1]
            psi = torch.atan2(x_b[:, -1, 3:4], x_b[:, -1, 4:5])          # [B,1]
            sh = torch.sin(psi).unsqueeze(1)              # [B,1,1]
            ch = torch.cos(psi).unsqueeze(1)
            mlon = 111320.0 * torch.cos(torch.deg2rad(lat0.clamp(-80, 80)))   # [B,1,1]
            dx_e = f * sh + r * ch                        # [B,H,1]
            dy_n = f * ch - r * sh
            gl = torch.cat(
                [lon0 + dx_e / mlon, lat0 + dy_n / 111132.0, alt0 + u / 10.0], dim=-1
            )                                            # [B,H,3]
            # velocity from finite differences (anchored at the last observation)
            anchor = torch.cat([lon0, lat0, alt0], dim=-1)          # [B,1,3]
            prev = torch.cat([anchor, gl[:, :-1]], dim=1)          # [B,H,3]
            d = gl - prev
            dt = torch.tensor(hs, device=dev, dtype=torch.float32).view(1, -1, 1)
            vx = d[..., 0:1] * mlon / dt * 3.6
            vy = d[..., 1:2] * 111132.0 / dt * 3.6
            vz = d[..., 2:3] * 10.0 / dt
            gv = torch.cat([vx, vy, vz], dim=-1)
            retr = torch.cat([gl, gv], dim=-1)

            rows["retr_mean"].append(retr.cpu().numpy())
            rows["model"].append(model(x_b)["pred"].cpu().numpy())
            rows["cv"].append(cv(x_b).cpu().numpy())
            rows["truth"].append(y_b.cpu().numpy())

    T = np.concatenate(rows["truth"], axis=0)
    retr = np.concatenate(rows["retr_mean"], axis=0)
    mdl = np.concatenate(rows["model"], axis=0)
    cvs = np.concatenate(rows["cv"], axis=0)
    out = {"n_windows": int(len(T)), "k": K, "stride": STRIDE,
           "horizon_seconds": hs, "mde_m": {}}
    print(f"\n{'horizon':>8s} {'retrieval':>12s} {'CV-Doppler':>12s} {'AeroTraj':>12s}")
    for h_i, h in enumerate(hs):
        d_retr = dist3(retr, T, h_i)
        d_cv = dist3(cvs, T, h_i)
        d_mod = dist3(mdl, T, h_i)
        out["mde_m"][f"{int(h)}s"] = {
            "retrieval_mean": float(d_retr.mean()),
            "cv_doppler": float(d_cv.mean()),
            "aerotraj_ultra": float(d_mod.mean()),
        }
        print(f"{int(h):>6d} s {d_retr.mean():12,.1f} {d_cv.mean():12,.1f} {d_mod.mean():12,.1f}")

    all_r = np.mean([out["mde_m"][f"{int(h)}s"]["retrieval_mean"] for h in hs])
    all_c = np.mean([out["mde_m"][f"{int(h)}s"]["cv_doppler"] for h in hs])
    all_m = np.mean([out["mde_m"][f"{int(h)}s"]["aerotraj_ultra"] for h in hs])
    out["overall"] = {"retrieval_mean": all_r, "cv_doppler": all_c, "aerotraj_ultra": all_m}
    print(f"\n  MEDIA       retrieval={all_r:,.1f} m | CV={all_c:,.1f} m | modello={all_m:,.1f} m")
    print(f"  il retrieval batte il CV: {all_r < all_c}  ({(1-all_r/all_c)*100:+.1f} %)")
    json.dump(out, open("results_hyp/retrieval_probe.json", "w"), indent=2)
    print("\n[probe] saved to results_hyp/retrieval_probe.json")


def dist3(pred: np.ndarray, targ: np.ndarray, h_i: int) -> np.ndarray:
    """3D distance (m) of one horizon between [N,6] and [N,H,6] arrays."""
    lat_r = np.radians(targ[:, h_i, 1])
    mlon = 111320.0 * np.cos(lat_r)
    d = pred[:, h_i] - targ[:, h_i]
    return np.sqrt((d[:, 0] * mlon) ** 2 + (d[:, 1] * 111132.0) ** 2 + (d[:, 2] * 10.0) ** 2)


if __name__ == "__main__":
    main()

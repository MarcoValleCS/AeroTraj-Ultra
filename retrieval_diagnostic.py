# coding=utf-8
"""
Diagnostic: is there ANY signal in the retrieval?

`retrieval_probe.py` showed that the *mean* of the k retrieved futures is much
worse than the constant-velocity baseline (mode averaging: the neighbours are
often on different routes, so their mean is not a plausible trajectory).

That does NOT prove the direction is dead: the literature uses retrieval as a
CONDITIONING signal for a learned model, which can select rather than average.
The question that decides the direction is the CEILING: the oracle best-of-k
among the retrieved neighbours.  If even that is worse than the model, the
retrieval as formulated carries no exploitable information.

This script sweeps several descriptor formulations and reports, per horizon:
  * the mean-based retrieval prediction (the naive predictor),
  * the ORACLE best-of-k among the retrieved neighbours (the ceiling),
  * the constant-velocity baseline and the model, for reference.

Usage:  python retrieval_diagnostic.py
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltra, ConstantVelocityKalmanBaseline
from retrieval import build_index, local_future, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
K = 8
STRIDE = 16
MAX_TEST = 12000


# --------------------------------------------------------------------------- #
# descriptor variants
# --------------------------------------------------------------------------- #
def make_descriptor(x: torch.Tensor, variant: str) -> torch.Tensor:
    """
    All variants share the same value storage; they differ only in the KEY used
    for the nearest-neighbour search.
    """
    lon0 = x[:, -1, 0]
    lat0 = x[:, -1, 1]
    alt0 = x[:, -1, 2]
    vx = x[:, -1, 3]
    vy = x[:, -1, 4]
    spd = torch.linalg.norm(x[:, -1, 3:5], dim=-1)
    psi = torch.atan2(vx, vy)
    s, c = torch.sin(psi), torch.cos(psi)
    if variant == "pos_state":
        return torch.stack([lat0, lon0, alt0 / 500.0, s, c, spd / 50.0], dim=-1)
    if variant == "state_only":
        # kinematic state only: no absolute position
        return torch.stack([alt0 / 500.0, s, c, spd / 50.0,
                            torch.zeros_like(s), torch.zeros_like(c)], dim=-1)
    if variant == "pos_light":
        # the position is present but strongly down-weighted (route prior, not route identity)
        return torch.stack([lat0 / 10.0, lon0 / 10.0, alt0 / 500.0, s, c, spd / 50.0], dim=-1)
    if variant == "pos_heavy":
        return torch.stack([lat0, lon0, alt0 / 200.0, s, c, spd / 50.0], dim=-1)
    raise ValueError(variant)


VARIANTS = ["pos_state", "pos_light", "pos_heavy", "state_only"]


# --------------------------------------------------------------------------- #
def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds_te = FlightSlidingWindowDataset("test", CONFIG, max_samples=MAX_TEST)

    # build the candidate pool ONCE (descriptors for every variant + the futures)
    print("[diag] building the candidate pool ...")
    pool_desc: Dict[str, List[np.ndarray]] = {v: [] for v in VARIANTS}
    pool_val: List[np.ndarray] = []
    anchors = ds_tr.anchors[::STRIDE]
    for s in range(0, len(anchors), 4096):
        x, y = ds_tr.get_batch_from_anchors(anchors[s:s + 4096], dev)
        for v in VARIANTS:
            pool_desc[v].append(make_descriptor(x, v).cpu().numpy())
        from retrieval import candidate_value
        pool_val.append(candidate_value(x, y).cpu().numpy())
    pool_val = np.concatenate(pool_val, axis=0)
    for v in VARIANTS:
        pool_desc[v] = np.concatenate(pool_desc[v], axis=0)
    print(f"[diag] {pool_val.shape[0]:,} candidati")

    model = AeroTrajUltra(cfg, ds_tr.norm_mean, ds_tr.norm_std).to(dev)
    st = torch.load("checkpoints_hyp/aerotraj_ultra_best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()
    cv = ConstantVelocityKalmanBaseline(hs).to(dev)

    loader = FastSlidingWindowBatchLoader(ds_te, batch_size=1024, shuffle=False, device=dev)

    # pre-compute the truth in the local frame (per query) and the model/CV errors
    res: Dict[str, Dict[str, List[float]]] = {
        v: {"mean": [], "oracle": []} for v in VARIANTS
    }
    mod_err, cv_err, truth_local = [], [], []

    for v in VARIANTS:
        cand = torch.as_tensor(pool_desc[v], dtype=torch.float32, device=dev)
        sq = (cand ** 2).sum(dim=1)
        with torch.no_grad():
            for x_b, y_b in loader:
                q = make_descriptor(x_b, v)
                d2 = (q ** 2).sum(dim=1, keepdim=True) - 2.0 * (q @ cand.T) + sq[None, :]
                _, idx = torch.topk(d2, K, dim=1, largest=False)
                val = torch.as_tensor(pool_val, dtype=torch.float32, device=dev)[idx]  # [B,k,H*3+3]
                H = len(hs)
                fut = val[..., :H * 3].reshape(val.shape[0], val.shape[1], H, 3)
                # truth in the query's local frame
                ft = local_future(x_b, y_b)                                 # [B,H,3]
                # rotate the neighbours into the query's heading frame
                hsin = val[..., H * 3 + 1]
                hcos = val[..., H * 3 + 2]
                qpsi = torch.atan2(x_b[:, -1, 3:4], x_b[:, -1, 4:5])
                dpsi = qpsi - torch.atan2(hsin, hcos)
                cs, sn = torch.cos(dpsi).unsqueeze(-1), torch.sin(dpsi).unsqueeze(-1)
                f, r = fut[..., 0], fut[..., 1]
                f_rot = f * cs - r * sn
                r_rot = f * sn + r * cs
                # distance of every neighbour to the truth, in the query frame
                df = f_rot - ft[..., 0].unsqueeze(1)          # [B,k,H]
                dr = r_rot - ft[..., 1].unsqueeze(1)
                du = fut[..., 2] - ft[..., 2].unsqueeze(1)
                d = torch.sqrt(df ** 2 + dr ** 2 + du ** 2)                  # [B,k,H]
                res[v]["oracle"].append(d.min(dim=1).values.cpu().numpy())   # best-of-k
                res[v]["mean"].append(
                    torch.sqrt((f_rot.mean(1) - ft[..., 0]) ** 2
                               + (r_rot.mean(1) - ft[..., 1]) ** 2
                               + (fut[..., 2].mean(1) - ft[..., 2]) ** 2
                               ).cpu().numpy()
                )
                if v == VARIANTS[0]:
                    mod_err.append(dist3(model(x_b)["pred"], y_b).cpu().numpy())
                    cv_err.append(dist3(cv(x_b), y_b).cpu().numpy())
                    truth_local.append(ft.cpu().numpy())

    mod_err = np.concatenate(mod_err, axis=0)
    cv_err = np.concatenate(cv_err, axis=0)
    truth_local = np.concatenate(truth_local, axis=0)

    out = {"k": K, "stride": STRIDE, "n_windows": int(len(mod_err)), "variants": {}}
    print(f"\n{'variant':12s} {'regola':10s} " + " ".join(f"{int(h):>7d}s" for h in hs) + f" {'MEDIA':>9s}")
    for v in VARIANTS:
        for rule in ["mean", "oracle"]:
            arr = np.concatenate(res[v][rule], axis=0)         # [N,H]
            out["variants"].setdefault(v, {})[rule] = {
                f"{int(h)}s": float(arr[:, i].mean()) for i, h in enumerate(hs)
            }
            out["variants"][v][rule]["media"] = float(arr.mean())
            print(f"{v:12s} {rule:10s} " + " ".join(f"{arr[:, i].mean():8,.0f}" for i, h in enumerate(hs))
                  + f" {arr.mean():9,.0f}")
        print(f"{'':12s} {'CV':10s} " + " ".join(f"{cv_err[:, i].mean():8,.0f}" for i, h in enumerate(hs))
              + f" {cv_err.mean():9,.0f}")
        print(f"{'':12s} {'MODEL':10s} " + " ".join(f"{mod_err[:, i].mean():8,.0f}" for i, h in enumerate(hs))
              + f" {mod_err.mean():9,.0f}\n")
    out["cv_doppler"] = {"media": float(cv_err.mean()),
                          **{f"{int(h)}s": float(cv_err[:, i].mean()) for i, h in enumerate(hs)}}
    out["aerotraj_ultra"] = {"media": float(mod_err.mean()),
                             **{f"{int(h)}s": float(mod_err[:, i].mean()) for i, h in enumerate(hs)}}
    json.dump(out, open("results_hyp/retrieval_diagnostic.json", "w"), indent=2)
    print("[diag] saved to results_hyp/retrieval_diagnostic.json")


def dist3(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """3D ECEF-ish distance in metres, [B,H,6] vs [B,H,6] -> [B,H]"""
    lat_r = torch.deg2rad(targ[..., 1])
    mlon = 111320.0 * torch.cos(lat_r)
    d = pred - targ
    return torch.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)


if __name__ == "__main__":
    main()

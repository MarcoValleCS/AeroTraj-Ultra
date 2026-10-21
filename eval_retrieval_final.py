# coding=utf-8
"""
Evaluate the retrieval-conditioned model: deterministic MDE, oracle minADE@K and
the conformal calibration of the hypothesis ensemble.

Usage:  python eval_retrieval_final.py
"""
from __future__ import annotations

import json
from typing import Dict

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltraRetrieval
from retrieval import build_pool, window_descriptor
from run import compute_trajectory_metrics

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config_90d_300s_hyp.json")
ap.add_argument("--checkpoint", default="checkpoints_hyp/retrieval/aerotraj_ultra_retrieval_best.pt")
ap.add_argument("--out", default="results_hyp/retrieval/final_eval.json")
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--pos-weight", type=float, default=10.0)
ap.add_argument("--pool-stride", type=int, default=16)
_args = ap.parse_args()

CONFIG = _args.config
CKPT = _args.checkpoint
K = _args.k
POS_WEIGHT = _args.pos_weight
POOL_STRIDE = _args.pool_stride


def main() -> None:
    dev = torch.device("cuda")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]
    tr = cfg["training"]

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds_val = FlightSlidingWindowDataset("val", CONFIG, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", CONFIG, max_samples=int(tr["max_test_samples"]))

    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    model = AeroTrajUltraRetrieval(cfg, ds_tr.norm_mean, ds_tr.norm_std, n_neighbours=K).to(dev)
    st = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(st["model_state_dict"], strict=False)
    model.eval()
    print(f"[eval] checkpoint epoca {st.get('epoch')} | val MDE {st.get('val_mde_m', float('nan')):.1f} m")

    def collect(ds):
        loader = FastSlidingWindowBatchLoader(ds, batch_size=2048, shuffle=False, device=dev)
        det, hyps, scores, targs = [], [], [], []
        with torch.no_grad():
            for x_b, y_b in loader:
                desc = window_descriptor(x_b, POS_WEIGHT)
                ctx = pool.context(desc, x_b, k=K)
                out = model(x_b, retr_ctx=ctx)
                det.append(out["pred"].cpu().numpy())
                hyps.append(out["hyp_pred"].cpu().numpy())
                scores.append(out["hyp_score"].cpu().numpy())
                targs.append(y_b.cpu().numpy())
        return (np.concatenate(det), np.concatenate(hyps),
                np.concatenate(scores), np.concatenate(targs))

    print("[eval] validation ...")
    dv, hv, sv, tv = collect(ds_val)
    print("[eval] test ...")
    dt, ht, st_, tt = collect(ds_test)

    def metrics(det, hyp, score, targ):
        m = compute_trajectory_metrics(det, targ, hs)
        lat_r = np.radians(targ[:, None, :, 1])
        mlon = 111320.0 * np.cos(lat_r)
        d = hyp - targ[:, None, :, :]
        dist = np.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)
        ade_k = dist.mean(axis=2)
        fde_k = dist[:, :, -1]
        _mode = getattr(model, "hyp_score_mode", "ce")
        sel = score.argmax(axis=1) if _mode == "ce" else score.argmin(axis=1)
        idx = np.arange(len(targ))
        return {
            "mde_deterministic_m": m["overall_mde_3d_m"],
            "minade_k_m": float(ade_k.min(axis=1).mean()),
            "minfde_k_m": float(fde_k.min(axis=1).mean()),
            "mde_score_selected_m": float(ade_k[idx, sel].mean()),
            "score_mode": _mode,
            "mde_mean_over_modes_m": float(ade_k.mean()),
            "per_horizon_mde": {f"{int(h)}s": m["per_horizon"][f"{int(h)}s"]["mde_3d_m"] for h in hs},
        }

    res = {"test": metrics(dt, ht, st_, tt), "val": metrics(dv, hv, sv, tv)}

    # conformal calibration of the ensemble (Mondrian by flight phase is not
    # available here: the phase is recomputed from the observations)
    print("\n[eval] conformal calibration del set di ipotesi")
    cal = {}
    for alpha in [0.2, 0.1, 0.05]:
        dv_ens = np.stack([np.sqrt(
            (0,))], axis=0) if False else None
        # distance of the truth to the closest hypothesis
        def ens_dist(hyp, targ):
            lat_r = np.radians(targ[:, None, :, 1])
            mlon = 111320.0 * np.cos(lat_r)
            d = hyp - targ[:, None, :, :]
            return np.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2
                           + (d[..., 2] * 10.0) ** 2).min(axis=1)
        e_val = ens_dist(hv, tv)
        e_te = ens_dist(ht, tt)
        q = np.quantile(e_val, 1 - alpha, axis=0)
        cov = float((e_te <= q[None, :]).mean())
        cal[f"alpha_{alpha}"] = {"coverage": cov, "mean_radius_m": float(q.mean()),
                                 "coverage_per_horizon": {
                                     f"{int(h)}s": float(c) for h, c in zip(hs, (e_te <= q[None, :]).mean(axis=0))}}
        print(f"  alpha={alpha:<5} copertura={cov:.3f} (target {1-alpha:.2f})  raggio medio={q.mean():,.0f} m")
    res["conformal"] = cal

    json.dump(res, open(_args.out, "w"), indent=2)
    print(f"\n[eval] MDE deterministico = {res['test']['mde_deterministic_m']:.1f} m")
    print(f"[eval] minADE@{K} (orakle)  = {res['test']['minade_k_m']:.1f} m")
    print(f"[eval] selezione(score)    = {res['test']['mde_score_selected_m']:.1f} m")
    print(f"[eval] salvato in {_args.out}")


if __name__ == "__main__":
    main()

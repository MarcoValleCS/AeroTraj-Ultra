# coding=utf-8
"""
Calibration of the multi-hypothesis head: from an oracle gain to a *certified*
uncertainty.

The multi-hypothesis head reaches an oracle minADE@8 of 627.4 m against a
deterministic output of 984.9 m, but the score-based selection is WORSE than
the deterministic output (1098.7 m), so the oracle gain is not exploitable.

This script answers two questions with measurements, not assumptions:

  Q1  Is there an oracle-free selection rule that beats the deterministic head?
      Rules compared: score-argmax, consensus-medoid, score-weighted mean,
      and the deterministic output itself.  The rule is chosen on the
      VALIDATION split and then applied unchanged to the test split.

  Q2  Can the hypothesis ensemble be turned into a *calibrated* predictive set?
      The ensemble gives the SHAPE of the uncertainty (which manoeuvres are
      plausible); a Mondrian (regime-stratified) conformal calibration on the
      validation split provides the GUARANTEE and the radius.

Outputs: results_hyp/calibration.json
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltra

CONFIG = "config_90d_300s_hyp.json"
CKPT = "checkpoints_hyp/aerotraj_ultra_best.pt"
ALPHAS = [0.2, 0.1, 0.05]
N_MODES = 8


# --------------------------------------------------------------------------- #
def collect(model, loader, device):
    """-> det [N,H,6], hyp [N,K,H,6], score [N,K], targ [N,H,6], phase [N]"""
    det, hyps, scores, targs, phases = [], [], [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            out = model(x_b)
            det.append(out["pred"].cpu().numpy())
            hyps.append(out["hyp_pred"].cpu().numpy())
            scores.append(out["hyp_score"].cpu().numpy())
            targs.append(y_b.cpu().numpy())
            vz = x_b[:, -1, 5].cpu().numpy()
            alt = x_b[:, -1, 2].cpu().numpy()
            phase = np.where(vz > 0.5, 0, np.where(vz < -0.5, 2,
                            np.where(alt > 600.0, 1, 1))).astype(np.int64)  # 0 climb 1 cruise 2 descent
            phases.append(phase)
    return (np.concatenate(det), np.concatenate(hyps), np.concatenate(scores),
            np.concatenate(targs), np.concatenate(phases))


def dist3(pred: np.ndarray, targ: np.ndarray) -> np.ndarray:
    """3D distance in metres between [N,H,6] arrays -> [N,H]"""
    lat_r = np.radians(targ[..., 1])
    mlon = 111320.0 * np.cos(lat_r)
    d = pred - targ
    return np.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)


def per_mode_dist(hyp: np.ndarray, targ: np.ndarray) -> np.ndarray:
    """[N,K,H,6] vs [N,H,6] -> [N,K,H] metres"""
    lat_r = np.radians(targ[:, None, :, 1])
    mlon = 111320.0 * np.cos(lat_r)
    d = hyp - targ[:, None, :, :]
    return np.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)


# --------------------------------------------------------------------------- #
def selection_rules(det, hyp, score, targ):
    """-> dict name -> [N,H,6] selected trajectory"""
    N, K, H, _ = hyp.shape
    rules = {"deterministic": det}

    # score-argmax
    sel = score.argmax(axis=1)
    rules["score_argmax"] = hyp[np.arange(N), sel]

    # consensus medoid: the hypothesis with the smallest total distance to the others
    lat_r = np.radians(targ[:, None, :, 1])               # [N,1,H]
    mlon = 111320.0 * np.cos(lat_r)                      # [N,1,H]
    pw_sum = np.zeros((N, K))
    for i in range(K):                                  # loop keeps the memory bounded
        dh = hyp[:, i:i + 1, :, :] - hyp                 # [N,K,H,6]
        di = np.sqrt((dh[..., 0] * mlon) ** 2 + (dh[..., 1] * 111132.0) ** 2
                     + (dh[..., 2] * 10.0) ** 2)         # [N,K,H]
        pw_sum += di.sum(axis=2)
    medoid = pw_sum.argmin(axis=1)                      # [N]
    rules["consensus_medoid"] = hyp[np.arange(N), medoid]

    # score-weighted mean
    w = np.exp(score - score.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    rules["score_weighted_mean"] = np.einsum("nk,nkhc->nhc", w, hyp)

    # deterministic blended with the consensus
    rules["det_plus_medoid"] = 0.5 * (det + rules["consensus_medoid"])
    return rules


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]
    tr = cfg["training"]

    ds_val = FlightSlidingWindowDataset("val", CONFIG, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", CONFIG, max_samples=int(tr["max_test_samples"]))
    bs = int(tr["batch_size"])
    val_loader = FastSlidingWindowBatchLoader(ds_val, batch_size=bs, shuffle=False, device=dev)
    test_loader = FastSlidingWindowBatchLoader(ds_test, batch_size=bs, shuffle=False, device=dev)

    model = AeroTrajUltra(cfg, ds_val.norm_mean, ds_val.norm_std).to(dev)
    state = torch.load(CKPT, map_location=dev, weights_only=False)["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing (deterministic): {missing}")
    model.eval()

    print("[calib] collecting validation set ...")
    dv, hv, sv, tv, pv = collect(model, val_loader, dev)
    print("[calib] collecting test set ...")
    dt, ht, st, tt, pt = collect(model, test_loader, dev)

    out: Dict[str, object] = {"horizon_seconds": hs, "n_modes": N_MODES}

    # ---------------- Q1: oracle-free selection rules ---------------- #
    print("\n=== Q1  selection rules (chosen on VALIDATION, applied to TEST) ===")
    print(f"{'rule':22s} {'val MDE [m]':>13s} {'test MDE [m]':>14s} {'test vs det':>13s}")
    val_rules = selection_rules(dv, hv, sv, tv)
    test_rules = selection_rules(dt, ht, st, tt)
    val_mde = {k: float(dist3(v, tv).mean()) for k, v in val_rules.items()}
    test_mde = {k: float(dist3(v, tt).mean()) for k, v in test_rules.items()}
    base = test_mde["deterministic"]
    for k in val_rules:
        mark = "  <-- migliore su val" if k == min(val_mde, key=val_mde.get) and k != "deterministic" else ""
        print(f"{k:22s} {val_mde[k]:13,.1f} {test_mde[k]:14,.1f} "
              f"{100*(test_mde[k]/base-1):+12.1f}%{mark}")
    best_rule = min(val_mde, key=val_mde.get)
    out["selection_rules"] = {"val_mde_m": val_mde, "test_mde_m": test_mde,
                              "rule_selected_on_val": best_rule}
    print(f"  -> regola scelta su validation: '{best_rule}'")

    # oracle reference
    dm = per_mode_dist(ht, tt)
    out["oracle_minade_k_m"] = float(dm.mean(axis=2).min(axis=1).mean())
    out["oracle_minfde_k_m"] = float(dm[:, :, -1].min(axis=1).mean())
    print(f"  -> oracolo minADE@{N_MODES} (test): {out['oracle_minade_k_m']:,.1f} m")

    # ---------------- Q2: Mondrian conformal calibration ---------------- #
    print("\n=== Q2  Mondrian conformal calibration (per flight phase) ===")
    # nonconformity of the ENSEMBLE: distance of the truth to the closest hypothesis
    d_val_ens = per_mode_dist(hv, tv).min(axis=1)          # [N_val, H]
    d_test_ens = per_mode_dist(ht, tt).min(axis=1)         # [N_test, H]
    d_val_det = dist3(dv, tv)                               # [N_val, H]
    d_test_det = dist3(dt, tt)                              # [N_test, H]
    strata = {0: "climb", 1: "cruise", 2: "descent"}
    cal: Dict[str, object] = {}
    for alpha in ALPHAS:
        qs, cov, rad = {}, [], []
        for s, name in strata.items():
            mv = pv == s
            mt = pt == s
            if mv.sum() < 50:
                continue
            q = np.quantile(d_val_ens[mv], 1.0 - alpha, axis=0)          # [H]
            cov.append(float((d_test_ens[mt] <= q[None, :]).mean()))
            rad.append(float(q.mean()))
            qs[name] = q.tolist()
        # also the deterministic output calibrated the same way (single point + radius)
        qd = np.quantile(d_val_det, 1.0 - alpha, axis=0)
        covd = float((d_test_det <= qd[None, :]).mean())
        cal[f"alpha_{alpha}"] = {
            "ensemble_coverage": float(np.average(cov, weights=[1, 1, 1][:len(cov)])) if cov else None,
            "ensemble_mean_radius_m": float(np.average(rad, weights=[1, 1, 1][:len(rad)])) if rad else None,
            "deterministic_coverage": covd,
            "deterministic_mean_radius_m": float(qd.mean()),
            "quantiles_per_stratum_m": qs,
        }
        print(f"  alpha={alpha:<5} ensemble: coverage={cal[f'alpha_{alpha}']['ensemble_coverage']:.3f} "
              f"(target {1-alpha:.2f})  mean radius={cal[f'alpha_{alpha}']['ensemble_mean_radius_m']:,.0f} m"
              f"   |  deterministico: coverage={covd:.3f}  radius={qd.mean():,.0f} m")
    out["conformal"] = cal

    # per-horizon coverage at the nominal level
    alpha = 0.1
    q_all = np.quantile(d_val_ens, 1.0 - alpha, axis=0)
    cov_h = (d_test_ens <= q_all[None, :]).mean(axis=0)
    out["coverage_per_horizon_alpha_0.1"] = {f"{int(h)}s": float(c) for h, c in zip(hs, cov_h)}
    print("\n  copertura per orizzonte (alpha=0,1, ensemble + raggio calibrato):")
    for h, c in zip(hs, cov_h):
        print(f"    {int(h):>4d} s : {c:.3f}")

    json.dump(out, open("results_hyp/calibration.json", "w"), indent=2)
    print("\n[calib] saved to results_hyp/calibration.json")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Standalone checkpoint evaluator.

Evaluates a trained AeroTraj-Ultra checkpoint (and, optionally, the matching
FlightBERT++ V2 baseline checkpoint) on an arbitrary preprocessed split.

Why this exists: the 40-day and the 90-day models are trained on different
chronological windows, therefore their native test sets differ and the two
numbers are NOT directly comparable. Evaluating every checkpoint on the SAME
(held-out, never-trained-on) split makes the comparison rigorous.

Example:
    python eval_ckpt.py --checkpoint checkpoints_365_1200k/aerotraj_ultra_best.pt \
        --eval-config config_365d_1200k.json --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import (
    AeroTrajUltra,
    ConstantVelocityKalmanBaseline,
    FlightBERT_PP_V2_Baseline,
    HybridPhysicsConstrainedLoss,
    SmoothedConstantVelocityBaseline,
)
from run import compute_trajectory_metrics


def load_model(ckpt_path: str, device: torch.device) -> tuple[AeroTrajUltra, Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    norm_mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    norm_std = np.array(ckpt["norm_std"], dtype=np.float32)
    model = AeroTrajUltra(cfg, norm_mean, norm_std).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg, ckpt


def evaluate(
    ckpt_path: str,
    eval_config: str,
    split: str = "test",
    device_str: str = "cuda",
    max_samples: Optional[int] = None,
    baseline_ckpt: Optional[str] = None,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    with open(eval_config, "r", encoding="utf-8") as f:
        ev_cfg = json.load(f)
    horizon_seconds: List[int] = ev_cfg["window"]["horizon_seconds"]

    ds = FlightSlidingWindowDataset(split, eval_config, max_samples=max_samples)
    loader = FastSlidingWindowBatchLoader(ds, batch_size=batch_size, shuffle=False, device=device)

    model, train_cfg, ckpt = load_model(ckpt_path, device)
    criterion = HybridPhysicsConstrainedLoss(model.codec, train_cfg).to(device)

    preds, preds_cont, preds_kin, targs = [], [], [], []
    with torch.no_grad():
        for x_b, y_b in loader:
            out = model(x_b)
            preds.append(out["pred"].cpu().numpy())
            preds_cont.append(out["pred_cont"].cpu().numpy())
            preds_kin.append(out["pred_kin"].cpu().numpy())
            targs.append(y_b.cpu().numpy())

    res: Dict[str, Any] = {}
    res["AeroTraj-Ultra"] = compute_trajectory_metrics(np.concatenate(preds), np.concatenate(targs), horizon_seconds)
    res["AeroTraj-Ultra (no MAF gate)"] = compute_trajectory_metrics(
        np.concatenate(preds_cont), np.concatenate(targs), horizon_seconds
    )
    res["Physics Kinematic Prior"] = compute_trajectory_metrics(
        np.concatenate(preds_kin), np.concatenate(targs), horizon_seconds
    )

    cv = ConstantVelocityKalmanBaseline(horizon_seconds).to(device)
    cvs = SmoothedConstantVelocityBaseline(
        horizon_seconds, dt_sec=float(ev_cfg["data"]["dt_seconds"])
    ).to(device)
    pc, pcs = [], []
    with torch.no_grad():
        for x_b, _ in loader:
            pc.append(cv(x_b).cpu().numpy())
            pcs.append(cvs(x_b).cpu().numpy())
    res["CV (raw Doppler)"] = compute_trajectory_metrics(np.concatenate(pc), np.concatenate(targs), horizon_seconds)
    res["CV (least-squares)"] = compute_trajectory_metrics(np.concatenate(pcs), np.concatenate(targs), horizon_seconds)

    if baseline_ckpt and Path(baseline_ckpt).exists():
        fb = FlightBERT_PP_V2_Baseline(train_cfg).to(device)
        fb.load_state_dict(
            torch.load(baseline_ckpt, map_location=device, weights_only=False)["model_state_dict"]
        )
        fb.eval()
        pfb = []
        with torch.no_grad():
            for x_b, _ in loader:
                p, _ = fb(x_b)
                pfb.append(p.cpu().numpy())
        res["FlightBERT++ V2"] = compute_trajectory_metrics(
            np.concatenate(pfb), np.concatenate(targs), horizon_seconds
        )

    res["_meta"] = {
        "checkpoint": ckpt_path,
        "eval_config": eval_config,
        "split": split,
        "n_windows": int(len(ds)),
        "trained_signature": ckpt.get("run_signature"),
        "epoch": ckpt.get("epoch"),
        "train_days": train_cfg["data"]["num_total_days"],
    }
    return res


def print_table(results: Dict[str, Any], horizon_seconds: List[int]) -> None:
    keys = [k for k in results if not k.startswith("_")]
    print(f"\n{'Horizon':>8s} | " + " | ".join(f"{k[:22]:>22s}" for k in keys))
    print("-" * (11 + 25 * len(keys)))
    for h in horizon_seconds:
        hk = f"{int(h)}s"
        cells = [f"{results[k]['per_horizon'][hk]['mde_3d_m']:,.1f}" for k in keys]
        print(f"{hk:>8s} | " + " | ".join(f"{c:>22s}" for c in cells))
    print("-" * (11 + 25 * len(keys)))
    cells = [f"{results[k]['overall_mde_3d_m']:,.1f}" for k in keys]
    print(f"{'OVERALL':>8s} | " + " | ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-config", type=str, default="config_90d.json")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--baseline-ckpt", type=str, default=None)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    res = evaluate(
        checkpoint_path := args.checkpoint,
        eval_config=args.eval_config,
        split=args.split,
        device_str=args.device,
        max_samples=args.max_samples,
        baseline_ckpt=args.baseline_ckpt,
    )
    with open(args.eval_config, "r", encoding="utf-8") as f:
        hs = json.load(f)["window"]["horizon_seconds"]
    print_table(res, hs)
    print("\nMeta:", json.dumps(res["_meta"], indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[Saved] {args.out}")

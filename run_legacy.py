# coding=utf-8
"""
Train and evaluate the FlightPREV legacy model (https://github.com/MarcoValleCS/flight)
on the SAME OpenSky 90-day dataset, with the SAME horizon grid (5-300 s), the SAME
splits and the SAME metric as AeroTraj-Ultra, so that the comparison isolates
the architecture and the loss/prior formulation.

Usage:
    python run_legacy.py --config config_90d_300s.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from dataloader import FlightSlidingWindowDataset
from legacy_flightprev import (
    GRID,
    SC_GS,
    SC_VR,
    SC_XY,
    SC_Z,
    SC_Z0,
    LegacyLoss,
    TrajTransformer,
    build_features,
    cv_baseline,
    local_to_global,
    phase_label,
)
from run import compute_trajectory_metrics
from run_retrieval import AnchoredLoader


def segment_buckets(ds: FlightSlidingWindowDataset) -> Dict[str, np.ndarray]:
    """
    Per-anchor categorical features: the geographic origin and destination area
    of the flight segment (the OpenSky replacement for the dep/arr airport
    embeddings of the original model).
    """
    segs = ds.segments                                   # (start, len, flight, day)
    seg_idx = np.searchsorted(segs[:, 0], ds.anchors, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(segs) - 1)
    s0 = segs[seg_idx, 0]
    s1 = s0 + segs[seg_idx, 1] - 1
    p_start = ds.points[s0]
    p_end = ds.points[s1]
    n_lat = int(180.0 / GRID)
    n_lon = int(360.0 / GRID)
    def bucket(lon, lat):
        b_lat = np.clip(((lat + 90.0) / GRID).astype(np.int64), 0, n_lat - 1)
        b_lon = np.clip(((lon + 180.0) / GRID).astype(np.int64), 0, n_lon - 1)
        return (b_lat * n_lon + b_lon).astype(np.int64)
    return {
        "orig": bucket(p_start[:, 0], p_start[:, 1]),
        "dest": bucket(p_end[:, 0], p_end[:, 1]),
        "n_lat": n_lat,
        "n_lon": n_lon,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_90d_300s.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(42)      # the original seed
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.load(open(args.config, encoding="utf-8"))
    tr = cfg["training"]
    hs: List[int] = cfg["window"]["horizon_seconds"]

    ds_tr = FlightSlidingWindowDataset("train", args.config)
    ds_val = FlightSlidingWindowDataset("val", args.config, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", args.config, max_samples=int(tr["max_test_samples"]))
    bs = int(tr["batch_size"])

    bk = segment_buckets(ds_tr)
    vocab = {"orig": int(bk["n_lat"] * bk["n_lon"]), "dest": int(bk["n_lat"] * bk["n_lon"])}
    emb_dims = {"orig": 16, "dest": 16}      # the original used ac_type 16 / operator 12 / dep 8 / arr 8
    n_feat = 10
    n_hist = int(cfg["window"]["inp_seq_len"])
    n_hor = len(hs)

    model = TrajTransformer(
        n_feat=n_feat, n_hist=n_hist, n_horizons=n_hor, vocab_sizes=vocab, emb_dims=emb_dims,
        d_model=256, n_layers=4, n_heads=8, ff=768, dropout=0.1, predict_logvar=False,
    ).to(dev)
    crit = LegacyLoss(w_xy=1.0, w_z=0.3, w_phase=0.1, huber_delta=0.1)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[legacy] TrajTransformer (arch=transformer) | {n_par:,} parametri | "
          f"{n_hist} step di storia | {n_hor} orizzonti 5-{hs[-1]} s")
    print(f"[legacy] categorie: orig/dest x {vocab['orig']} bucket da {GRID} deg | "
          f"embedding 16+16 (l'originale usava ac_type/operator/dep/arr: 16+12+8+8)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = AnchoredLoader(ds_tr, bs, True, int(tr["max_train_samples_per_epoch"]), dev)
    val_loader = AnchoredLoader(ds_val, bs, False, None, dev)
    test_loader = AnchoredLoader(ds_test, bs, False, None, dev)
    steps_per_epoch = int(np.ceil(int(tr["max_train_samples_per_epoch"]) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        pct_start=0.2, anneal_strategy="cos")

    # ---- helpers ---------------------------------------------------- #
    def pack(x_b, y_b, anchors):
        """-> features, static, cats, targets"""
        feats = build_features(x_b)                                  # [B,T,10]
        alt0 = x_b[:, -1, 2] * 10.0
        gs0 = torch.linalg.norm(x_b[:, -1, 3:5], dim=-1) * (1000.0 / 3600.0)
        static = torch.stack([alt0 / SC_Z0, gs0 / SC_GS], dim=-1)
        pos = np.searchsorted(ds_tr.anchors, anchors)
        pos = np.clip(pos, 0, len(bk["orig"]) - 1)
        cats = torch.stack([
            torch.as_tensor(bk["orig"][pos], dtype=torch.long, device=dev),
            torch.as_tensor(bk["dest"][pos], dtype=torch.long, device=dev),
        ], dim=1)
        # targets: the CV residual
        cv_xy, cv_z = cv_baseline(x_b, hs)                           # metres
        # true future in ENU relative to the anchor
        lon0 = x_b[:, -1, 0:1].unsqueeze(1)                     # [B,1,1]
        lat0 = x_b[:, -1, 1:2].unsqueeze(1)                     # [B,1,1]
        alt0 = x_b[:, -1, 2:3].unsqueeze(1)                    # [B,1,1]
        mlon = 111320.0 * torch.cos(torch.deg2rad(lat0))
        true_e = (y_b[..., 0:1] - lon0) * mlon
        true_n = (y_b[..., 1:2] - lat0) * 111132.0
        true_xy = torch.cat([true_e, true_n], dim=-1)                # metres
        true_z = (y_b[..., 2:3] - alt0) * 10.0                       # metres
        y_xy = (true_xy - cv_xy) / SC_XY
        y_z = ((true_z - cv_z.unsqueeze(-1)) / SC_Z).squeeze(-1)
        # phase at +60 s
        i60 = int(np.argmin([abs(h - 60) for h in hs]))
        ph = phase_label(y_b[:, i60, 5], y_b[:, i60, 2])
        return feats, static, cats, y_xy, y_z, ph, cv_xy, cv_z

    def forward_pred(x_b, feats, static, cats, cv_xy, cv_z):
        out = model(feats, static, cats)
        xy = cv_xy + out["xy"] * SC_XY
        z = cv_z + out["z"] * SC_Z
        gl = local_to_global(xy, z, x_b)
        # velocities from the predicted path (anchored at the last observation)
        prev = torch.cat([torch.stack([x_b[:, -1, 0:1], x_b[:, -1, 1:2], x_b[:, -1, 2:3]], -1),
                          gl[:, :-1]], dim=1)
        d = gl - prev
        t = torch.tensor(hs, device=dev, dtype=torch.float32).view(1, -1, 1)
        mlon = 111320.0 * torch.cos(torch.deg2rad(x_b[:, -1, 1:2])).unsqueeze(1)   # [B,1,1]
        vx = d[..., 0:1] * mlon / t * 3.6
        vy = d[..., 1:2] * 111132.0 / t * 3.6
        vz = d[..., 2:3] * 10.0 / t
        return torch.cat([gl, vx, vy, vz], dim=-1), out

    # ---- training ---------------------------------------------------- #
    save_dir = Path(tr["save_dir"]) / "legacy"
    results_dir = Path(tr["results_dir"]) / "legacy"
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "flightprev_best.pt"

    best = float("inf")
    best_ep = 0
    hist: List[Dict[str, Any]] = []
    stale = 0
    es = tr.get("early_stopping", {})
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x_b, y_b, anchors in train_loader:
            feats, static, cats, y_xy, y_z, ph, cv_xy, cv_z = pack(x_b, y_b, anchors)
            opt.zero_grad(set_to_none=True)
            out = model(feats, static, cats)
            loss, _ = crit(out, y_xy, y_z, ph)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sched.step()
            losses.append(loss.detach())
        vm = evaluate(model, val_loader, pack, forward_pred, hs, dev)["overall_mde_3d_m"]
        hist.append({"epoch": ep, "train_loss": float(torch.stack(losses).mean().item()),
                     "val_mde_m": vm, "time_s": round(time.time() - t0, 1)})
        print(f"  ep {ep:02d}/{args.epochs} ({time.time()-t0:5.1f}s) "
              f"loss={float(torch.stack(losses).mean().item()):9.4f} val MDE={vm:8.1f} m", flush=True)
        if vm < best - float(es.get("min_delta", 0.5)):
            best, best_ep, stale = vm, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep, "val_mde_m": vm,
                        "vocab": vocab, "emb_dims": emb_dims, "n_feat": n_feat,
                        "n_hist": n_hist, "n_horizons": n_hor}, ckpt_path)
        else:
            stale += 1
            if int(es.get("patience", 10)) and ep >= int(es.get("min_epochs", 12)) \
                    and stale >= int(es.get("patience", 10)):
                print(f"  early stop @ ep {ep} (best {best:.1f} m @ {best_ep})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False)["model_state_dict"])
    print(f"[legacy] best epoch {best_ep} | val MDE {best:.1f} m")

    test = evaluate(model, test_loader, pack, forward_pred, hs, dev, full=True)
    print(f"\n[legacy] TEST MDE = {test['overall_mde_3d_m']:.1f} m")
    for h in hs:
        print(f"   {int(h):>4d} s : {test['per_horizon'][f'{int(h)}s']['mde_3d_m']:9,.1f} m")

    json.dump({"parameters": n_par, "history": hist, "best_epoch": best_ep,
               "best_val_mde_m": best, "test": test,
               "note": "FlightPREV re-implemented on OpenSky; dep/arr airport embeddings "
                       "replaced by geographic origin/destination area embeddings",
               "config": args.config},
              open(results_dir / "legacy_results.json", "w"), indent=2)
    print(f"[legacy] salvato in {results_dir}")


@torch.no_grad()
def evaluate(model, loader, pack, forward_pred, hs, dev, full: bool = False) -> Dict[str, Any]:
    model.eval()
    preds, targs = [], []
    for x_b, y_b, anchors in loader:
        feats, static, cats, y_xy, y_z, ph, cv_xy, cv_z = pack(x_b, y_b, anchors)
        p, _ = forward_pred(x_b, feats, static, cats, cv_xy, cv_z)
        preds.append(p.cpu().numpy())
        targs.append(y_b.cpu().numpy())
    P = np.concatenate(preds, axis=0)
    T = np.concatenate(targs, axis=0)
    m = compute_trajectory_metrics(P, T, hs)
    return {"overall_mde_3d_m": m["overall_mde_3d_m"], "per_horizon": m["per_horizon"]}


if __name__ == "__main__":
    main()

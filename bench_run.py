# coding=utf-8
"""
Benchmark harness: trains the 5 replicated literature models (plus the
project's own models and the classical baselines) on the SAME 90-day OpenSky
dataset and compares them over a 5 s -> 300 s horizon.

Models
------
  ours        AeroTraj-Ultra (FlightBERT++ V3)  - project model
  ascent      ASCENT (ICRA 2026)                - mode queries + WTA
  cfm         Conditional Flow Matching (2026)  - probabilistic
  ddpm        DDPM / MID diffusion (2020-2022)  - probabilistic, same backbone
  attn_lstm   Attention-LSTM phase-aware (2022/2021)
  imm_res     IMM(KF) + neural residual (2025) - hybrid physics+ML
  fbpp_v2     FlightBERT++ V2 (AAAI 2024)       - reference
  lstm        Deep LSTM multi-horizon           - reference
  cv_doppler  constant velocity (Doppler)       - kinematic baseline
  cv_lsq      least-squares extrapolation       - kinematic baseline
  phys_prior  Physics kinematic prior           - no-learning baseline

Usage
-----
    python bench_run.py --config config_90d_300s.json
    python bench_run.py --config config_90d_300s.json --models ascent cfm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_models import (
    ASCENT,
    AttentionLSTMPhase,
    CFMTrajectoryModel,
    DDPMTrajectoryModel,
    IMMResidualModel,
    ascent_wta_loss,
    horizon_weighted_smooth_l1,
)
from data_pipeline import PyOpenSkyTrajectoryPipeline
from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import (
    AeroTrajUltra,
    ConstantVelocityKalmanBaseline,
    FlightBERT_PP_V2_Baseline,
    HybridPhysicsConstrainedLoss,
    LSTMRegressionBaseline,
    SmoothedConstantVelocityBaseline,
)
from run import compute_trajectory_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Adapters: uniform interface over heterogeneous model families
# --------------------------------------------------------------------------- #
def _raw_module(m: nn.Module) -> nn.Module:
    """Unwrap torch.compile's OptimizedModule (state_dict keys are prefixed)."""
    return getattr(m, "_orig_mod", m)


def _maybe_compile(module: nn.Module, mode: str) -> nn.Module:
    if not mode:
        return module
    try:
        torch._dynamo.config.cache_size_limit = 64
        return torch.compile(module, mode=mode)
    except Exception as exc:  # pragma: no cover - depends on the local Triton
        print(f"  [compile] unavailable ({exc}); falling back to eager")
        return module


class Adapter:
    """kind: 'det' | 'multi' | 'gen'"""
    kind = "det"

    def __init__(self, name: str, module: nn.Module) -> None:
        self.name = name
        self.module = module

    def loss(self, x, y) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x, n_samples: int = 1) -> torch.Tensor:
        """-> [B, n_samples, H, 6]"""
        out = self.module(x)
        if isinstance(out, dict):
            out = out["pred"]
        if out.dim() == 3:
            out = out.unsqueeze(1)
        return out


class DetAdapter(Adapter):
    def __init__(self, name, module, loss_fn) -> None:
        super().__init__(name, module)
        self.loss_fn = loss_fn

    def loss(self, x, y):
        return self.loss_fn(self.module(x), y)


class ASCENTAdapter(Adapter):
    kind = "multi"

    def loss(self, x, y):
        out, score = self.module(x, return_scores=True)
        loss, best = ascent_wta_loss(out, score, y, self.horizon_seconds, self.module.n_modes)
        return loss

    def __init__(self, name, module, horizon_seconds) -> None:
        super().__init__(name, module)
        self.horizon_seconds = horizon_seconds

    @torch.no_grad()
    def predict(self, x, n_samples: int = 1):
        out, score = self.module(x, return_scores=True)
        return out, score


class GenAdapter(Adapter):
    kind = "gen"

    def __init__(self, name, module, horizon_seconds) -> None:
        super().__init__(name, module)
        self.horizon_seconds = horizon_seconds

    def loss(self, x, y):
        return self.module(x, y).mean()

    @torch.no_grad()
    def predict(self, x, n_samples: int = 1):
        return self.module.sample(x, n_samples=n_samples)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def horizontal_and_3d_errors(pred: np.ndarray, targ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """3D ECEF distance (m) and 2D horizontal distance (m), both [N,H]."""
    lat_r = np.radians(targ[..., 1])
    dx = (pred[..., 0] - targ[..., 0]) * 111320.0 * np.cos(lat_r)
    dy = (pred[..., 1] - targ[..., 1]) * 111132.0
    d2 = np.sqrt(dx**2 + dy**2)

    a = 6378.137
    b = 6356.752
    e2 = 1.0 - (b**2) / (a**2)

    def ecef(lon_deg: np.ndarray, lat_deg: np.ndarray, alt_km: np.ndarray):
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        N = a / np.sqrt(1.0 - e2 * (np.sin(lat) ** 2))
        X = (N + alt_km) * np.cos(lat) * np.cos(lon)
        Y = (N + alt_km) * np.cos(lat) * np.sin(lon)
        Z = (b**2 / a**2 * N + alt_km) * np.sin(lat)
        return X, Y, Z

    # the predicted point uses its OWN lon/lat, the target its own
    Xp, Yp, Zp = ecef(pred[..., 0], pred[..., 1], pred[..., 2] * 0.01)
    Xt, Yt, Zt = ecef(targ[..., 0], targ[..., 1], targ[..., 2] * 0.01)
    d3 = np.sqrt((Xp - Xt) ** 2 + (Yp - Yt) ** 2 + (Zp - Zt) ** 2) * 1000.0
    return d3, d2


def ade_fde(d3: np.ndarray, d2: np.ndarray) -> Dict[str, float]:
    return {
        "ade_3d_m": float(d3.mean()),
        "fde_3d_m": float(d3[:, -1].mean()),
        "ade_2d_m": float(d2.mean()),
        "fde_2d_m": float(d2[:, -1].mean()),
    }


def min_ade_k(preds: np.ndarray, targ: np.ndarray, k: int) -> Dict[str, float]:
    """preds [N,K,H,6] -> minADE@k / minFDE@k (3D, metres)."""
    lat_r = np.radians(targ[:, None, :, 1])
    mlon = 111320.0 * np.cos(lat_r)
    d = preds - targ[:, None, :, :]
    dist = np.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)
    per_mode_ade = dist.mean(axis=2)                 # [N,K]
    per_mode_fde = dist[:, :, -1]                    # [N,K]
    return {
        "minADE_k_m": float(per_mode_ade.min(axis=1).mean()),
        "minFDE_k_m": float(per_mode_fde.min(axis=1).mean()),
        "single_ADE_m": float(per_mode_ade.mean()),
        "k": int(k),
    }


# --------------------------------------------------------------------------- #
# Train / evaluate loops
# --------------------------------------------------------------------------- #
def train_model(
    adapter: Adapter,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    es_patience: int,
    es_min_delta: float,
    es_min_epochs: int,
    ckpt_path: Path,
    horizon_seconds: List[int],
    tag: str = "",
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(adapter.module.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy="cos",
    )
    best = float("inf")
    best_ep = 0
    hist: List[Dict[str, Any]] = []
    stale = 0
    for ep in range(1, epochs + 1):
        adapter.module.train()
        t0 = time.time()
        losses = []
        for x_b, y_b in train_loader:
            opt.zero_grad()
            loss = adapter.loss(x_b, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.module.parameters(), max_norm=1.5)
            opt.step()
            sched.step()
            losses.append(loss.detach())   # tensor: synced once per epoch
        val = evaluate_model(adapter, val_loader, horizon_seconds, n_samples=1)
        vm = val["overall_mde_3d_m"]
        hist.append({"epoch": ep, "train_loss": float(torch.stack(losses).mean().item()),
                     "val_mde_m": vm, "time_s": round(time.time() - t0, 1)})
        print(f"    [{tag}] ep {ep:02d}/{epochs} ({time.time()-t0:5.1f}s) "
              f"loss={float(torch.stack(losses).mean().item()):9.4f} "
              f"val MDE={vm:8.1f} m", flush=True)
        if vm < best - es_min_delta:
            best = vm
            best_ep = ep
            stale = 0
            torch.save({"model_state_dict": _raw_module(adapter.module).state_dict(),
                        "adapter": adapter.name, "horizon_seconds": horizon_seconds}, ckpt_path)
        else:
            stale += 1
            if es_patience and ep >= es_min_epochs and stale >= es_patience:
                print(f"    [{tag}] early stop @ ep {ep} (best {best:.1f} m @ {best_ep})")
                break
    if ckpt_path.exists():
        _raw_module(adapter.module).load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=False)["model_state_dict"]
        )
    return {"best_val_mde_m": best, "best_epoch": best_ep, "history": hist}


@torch.no_grad()
def evaluate_model(adapter: Adapter, loader, horizon_seconds: List[int],
                   n_samples: int = 1, max_windows: Optional[int] = None) -> Dict[str, Any]:
    adapter.module.eval()
    preds, targs, scores_all = [], [], []
    seen = 0
    for x_b, y_b in loader:
        p = adapter.predict(x_b, n_samples=n_samples)
        if isinstance(p, tuple):
            p, sc = p
            scores_all.append(sc.cpu().numpy())
        preds.append(p.cpu().numpy())
        targs.append(y_b.cpu().numpy())
        seen += y_b.shape[0]
        if max_windows is not None and seen >= max_windows:
            break
    P = np.concatenate(preds, axis=0)
    T = np.concatenate(targs, axis=0)
    if P.shape[1] == 1:
        P = P[:, 0]
        d3, d2 = horizontal_and_3d_errors(P, T)
        m = compute_trajectory_metrics(P, T, horizon_seconds)
        out = {"overall_mde_3d_m": m["overall_mde_3d_m"], "per_horizon": m["per_horizon"]}
        out.update(ade_fde(d3, d2))
    elif adapter.kind == "multi":
        # MULTI-MODAL (ASCENT): the operational prediction is the hypothesis
        # with the highest SCORE (the model must learn to rank it); the
        # mean-over-modes is meaningless because only the winner is supervised.
        # We report: score-selected (primary), oracle best (minADE@K) and the
        # mean over the modes for reference.
        S = np.concatenate(scores_all, axis=0)                 # [N,K]
        d3s = np.stack([horizontal_and_3d_errors(P[:, k], T)[0] for k in range(P.shape[1])], axis=1)
        # operational selection: argmax for a "ce" score, argmin for a
        # "regress" score (the score then predicts the error directly)
        _mode = getattr(ad.module, "hyp_score_mode", "ce")
        sel = S.argmax(axis=1) if _mode == "ce" else S.argmin(axis=1)
        idx = np.arange(len(T))
        d3_sel = d3s[idx, sel]                                  # [N,H]
        d3_mean = d3s.mean(axis=1)
        d2_sel = np.stack([horizontal_and_3d_errors(P[:, k], T)[1] for k in range(P.shape[1])], axis=1)
        d2_sel = d2_sel[idx, sel]
        out = {"overall_mde_3d_m": float(d3_sel.mean()),
               "per_horizon": {f"{int(h)}s": {"mde_3d_m": float(d3_sel[:, i].mean())}
                               for i, h in enumerate(horizon_seconds)},
               "mde_mean_over_modes_m": float(d3_mean.mean())}
        out.update(ade_fde(d3_sel, d2_sel))
        out.update(min_ade_k(P, T, P.shape[1]))
    else:
        # GENERATIVE (CFM / DDPM): the operational metric is the error of a
        # SINGLE draw (the model samples the distribution, not its mean);
        # minADE@K reports the best-of-K for the probabilistic use case.
        d3s, d2s = [], []
        for k in range(P.shape[1]):
            d3, d2 = horizontal_and_3d_errors(P[:, k], T)
            d3s.append(d3)
            d2s.append(d2)
        d3m = np.mean(np.stack(d3s), axis=0)
        d2m = np.mean(np.stack(d2s), axis=0)
        out = {"overall_mde_3d_m": float(d3m.mean()), "per_horizon": {
            f"{int(h)}s": {"mde_3d_m": float(d3m[:, i].mean())}
            for i, h in enumerate(horizon_seconds)
        }}
        out.update(ade_fde(d3m, d2m))
        out.update(min_ade_k(P, T, P.shape[1]))
    out["n_windows"] = int(len(T))
    return out


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
def build_models(cfg, ds_train, horizon_seconds, args) -> Dict[str, Adapter]:
    win = cfg["window"]
    H = int(win["horizon"])
    T = int(win["inp_seq_len"])
    nm, ns = ds_train.norm_mean, ds_train.norm_std
    out: Dict[str, Adapter] = {}
    cmode = args.compile

    # ---- project model -------------------------------------------------
    if args.run_ours:
        ours = AeroTrajUltra(cfg, nm, ns).to(DEVICE)
        crit = HybridPhysicsConstrainedLoss(ours.codec, cfg).to(DEVICE)

        class _OursAdapter(Adapter):
            def loss(self, x, y):
                o = self.module(x)
                l, _ = crit(o, x, y)
                return l

        out["ours"] = _OursAdapter("ours", _maybe_compile(ours, cmode))

    # ---- literature models --------------------------------------------
    if args.run_bench:
        ascent = ASCENT(H, horizon_seconds, T, d_model=128, n_head=8, n_modes=args.n_modes).to(DEVICE)
        out["ascent"] = ASCENTAdapter("ascent", _maybe_compile(ascent, cmode), horizon_seconds)

        cfm = CFMTrajectoryModel(H, horizon_seconds, T, d_model=128, n_head=8, n_layers=4).to(DEVICE)
        out["cfm"] = GenAdapter("cfm", _maybe_compile(cfm, cmode), horizon_seconds)

        ddpm = DDPMTrajectoryModel(H, horizon_seconds, T, d_model=128, n_head=8, n_layers=4).to(DEVICE)
        out["ddpm"] = GenAdapter("ddpm", _maybe_compile(ddpm, cmode), horizon_seconds)

        alstm = AttentionLSTMPhase(H, horizon_seconds, T, hidden_dim=192, n_layers=2).to(DEVICE)
        out["attn_lstm"] = DetAdapter(
            "attn_lstm", _maybe_compile(alstm, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

        imm = IMMResidualModel(H, horizon_seconds, T, hidden_dim=192, n_layers=2).to(DEVICE)
        out["imm_res"] = DetAdapter(
            "imm_res", _maybe_compile(imm, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

    # ---- reference DL models -------------------------------------------
    if args.run_refs:
        fb = FlightBERT_PP_V2_Baseline(cfg).to(DEVICE)

        class _FBAdapter(Adapter):
            def loss(self, x, y):
                _, logits = self.module(x)
                tgt = self.module.codec.encode_targets_from_anchor(x[:, -1, :6], y)
                return sum(
                    F.binary_cross_entropy_with_logits(l, t) for l, t in zip(logits, tgt)
                ) / 6.0

            @torch.no_grad()
            def predict(self, x, n_samples: int = 1):
                p, _ = self.module(x)
                return p.unsqueeze(1)

        out["fbpp_v2"] = _FBAdapter("fbpp_v2", _maybe_compile(fb, cmode))

        lstm = LSTMRegressionBaseline(
            horizon=H, horizon_seconds=horizon_seconds, norm_mean=nm, norm_std=ns
        ).to(DEVICE)
        scales = torch.tensor([0.02, 0.02, 4.0, 15.0, 15.0, 1.5], device=DEVICE).view(1, 1, 6)

        class _LSTMAdapter(Adapter):
            def loss(self, x, y):
                p = self.module(x)
                return F.smooth_l1_loss((p - y) / scales, torch.zeros_like(y))

        out["lstm"] = _LSTMAdapter("lstm", _maybe_compile(lstm, cmode))

    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_90d_300s.json")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models to train")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-modes", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=10, help="K samples for minADE@K")
    ap.add_argument("--max-test-windows", type=int, default=60000)
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--skip-refs", action="store_true")
    ap.add_argument("--compile", type=str, default="reduce-overhead",
                    help="torch.compile mode (reduce-overhead | default | none)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate the existing checkpoints and merge the results")
    args = ap.parse_args()
    if args.compile in ("none", "off", ""):
        args.compile = ""
    args.run_ours = not args.skip_ours
    args.run_bench = not args.skip_bench
    args.run_refs = not args.skip_refs

    torch.manual_seed(97)
    np.random.seed(97)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = json.load(open(args.config, encoding="utf-8"))
    tr = cfg["training"]
    win = cfg["window"]
    horizon_seconds: List[int] = win["horizon_seconds"]

    pipeline = PyOpenSkyTrajectoryPipeline(args.config)
    dataset_stats = pipeline.build_dataset_splits(force_rebuild=False)

    ds_train = FlightSlidingWindowDataset("train", args.config)
    ds_val = FlightSlidingWindowDataset("val", args.config, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", args.config, max_samples=int(tr["max_test_samples"]))

    bs = int(tr["batch_size"])
    train_loader = FastSlidingWindowBatchLoader(
        ds_train, batch_size=bs, shuffle=True,
        max_samples_per_epoch=int(tr["max_train_samples_per_epoch"]), device=DEVICE,
    )
    val_loader = FastSlidingWindowBatchLoader(ds_val, batch_size=bs, shuffle=False, device=DEVICE)
    test_loader = FastSlidingWindowBatchLoader(ds_test, batch_size=bs, shuffle=False, device=DEVICE)

    save_dir = Path(tr["save_dir"])
    results_dir = Path(tr["results_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[bench] device={DEVICE} | horizon {horizon_seconds[0]}s -> {horizon_seconds[-1]}s "
          f"({len(horizon_seconds)} points)")
    print(f"[bench] train pool={len(ds_train):,} | val={len(ds_val):,} | test={len(ds_test):,}")

    adapters = build_models(cfg, ds_train, horizon_seconds, args)
    if args.models:
        adapters = {k: v for k, v in adapters.items() if k in args.models}

    epochs = int(args.epochs or tr["epochs"])
    summary: Dict[str, Any] = {"config": args.config, "horizon_seconds": horizon_seconds,
                               "dataset": dataset_stats["splits"], "models": {}}

    for name, ad in adapters.items():
        if args.eval_only:
            ckpt = save_dir / f"{name}.pt"
            if not ckpt.exists():
                print(f"  [eval-only] {name}: no checkpoint at {ckpt} -> skipped")
                continue
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["model_state_dict"]
            _raw_module(ad.module).load_state_dict(state)
            print(f"  [eval-only] {name}: loaded {ckpt}")
            continue
        print(f"\n=== Training '{name}' ({ad.kind}) ===")
        t0 = time.time()
        info = train_model(
            ad, train_loader, val_loader, epochs,
            float(tr["learning_rate"]), float(tr["weight_decay"]),
            int(tr.get("early_stopping", {}).get("patience", 10)),
            float(tr.get("early_stopping", {}).get("min_delta", 0.5)),
            int(tr.get("early_stopping", {}).get("min_epochs", 12)),
            save_dir / f"{name}.pt", horizon_seconds, tag=name,
        )
        info["train_seconds"] = round(time.time() - t0, 1)
        summary["models"][name] = info
        json.dump(summary, open(results_dir / "training_summary.json", "w"), indent=2)

    # ---------------- evaluation ---------------- #
    print("\n=== Evaluating all models on the held-out test set ===")
    results_path = results_dir / "benchmark_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        # merge with the results of a previous (partial) run: a model can be
        # re-trained/ re-evaluated in isolation without losing the others
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  [merge] {len(results)} pre-existing results loaded from {results_path}")
    trained_names = set(adapters.keys())

    for name, ad in adapters.items():
        n_s = args.n_samples if ad.kind in ("gen", "multi") else 1
        res = evaluate_model(ad, test_loader, horizon_seconds, n_samples=n_s,
                             max_windows=args.max_test_windows)
        results[name] = res
        print(f"  {name:10s} MDE={res['overall_mde_3d_m']:9.1f} m  "
              f"ADE={res['ade_3d_m']:9.1f}  FDE={res['fde_3d_m']:9.1f}"
              + (f"  minADE@{n_s}={res['minADE_k_m']:9.1f}" if "minADE_k_m" in res else ""))

    # classical baselines (no training)
    cv = ConstantVelocityKalmanBaseline(horizon_seconds).to(DEVICE)
    # the smoothed CV fits a velocity over the input window, so its time axis must
    # come from the config: hardcoding 5 s made the baseline 2x wrong on a coarser
    # grid, i.e. artificially weak
    cvs = SmoothedConstantVelocityBaseline(
        horizon_seconds, dt_sec=float(cfg["data"]["dt_seconds"])).to(DEVICE)
    for nm, mod in (("cv_doppler", cv), ("cv_lsq", cvs)):
        preds, targs = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                preds.append(mod(x_b).cpu().numpy())
                targs.append(y_b.cpu().numpy())
        P, T = np.concatenate(preds), np.concatenate(targs)
        d3, d2 = horizontal_and_3d_errors(P, T)
        m = compute_trajectory_metrics(P, T, horizon_seconds)
        results[nm] = {"overall_mde_3d_m": m["overall_mde_3d_m"], "per_horizon": m["per_horizon"]}
        results[nm].update(ade_fde(d3, d2))
        print(f"  {nm:10s} MDE={results[nm]['overall_mde_3d_m']:9.1f} m")

    json.dump(results, open(results_path, "w"), indent=2)

    # merge the training summary as well
    ts_path = results_dir / "training_summary.json"
    ts: Dict[str, Any] = {}
    if ts_path.exists():
        with open(ts_path, "r", encoding="utf-8") as f:
            ts = json.load(f)
    ts["config"] = args.config
    ts["horizon_seconds"] = horizon_seconds
    ts["dataset"] = dataset_stats["splits"]
    ts.setdefault("models", {}).update(summary["models"])
    json.dump(ts, open(ts_path, "w"), indent=2)

    # ---------------- tables ---------------- #
    rows = []
    for name, r in results.items():
        row = {"model": name, "mde_mean_m": round(r["overall_mde_3d_m"], 1),
               "ade_3d_m": round(r.get("ade_3d_m", float("nan")), 1),
               "fde_3d_m": round(r.get("fde_3d_m", float("nan")), 1),
               "n_windows": r.get("n_windows", 0)}
        if "minADE_k_m" in r:
            row["minADE_k_m"] = round(r["minADE_k_m"], 1)
            row["minFDE_k_m"] = round(r["minFDE_k_m"], 1)
            row["k"] = r["k"]
        for h in horizon_seconds:
            row[f"mde_{int(h)}s"] = round(r["per_horizon"][f"{int(h)}s"]["mde_3d_m"], 1)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mde_mean_m")
    df.to_csv(results_dir / "benchmark_table.csv", index=False)

    lines = ["# Benchmark 90 giorni - orizzonte 5 s -> 300 s\n",
             f"Test set: {results_dir.name} | {dataset_stats['splits']['test']['num_days']} giorni "
             f"({dataset_stats['splits']['test']['first_day']} -> {dataset_stats['splits']['test']['last_day']}), "
             f"{results[list(results)[0]].get('n_windows', 0)} finestre.\n",
             "| Modello | MDE media [m] | ADE 3D [m] | FDE 3D [m] | minADE@K [m] | "
             + " | ".join(f"{int(h)} s" for h in horizon_seconds) + " |",
             "|---|---:|---:|---:|---:|" + "---:|" * len(horizon_seconds)]
    for _, r in df.iterrows():
        minade = f"{r['minADE_k_m']:,.1f}" if "minADE_k_m" in r else "—"
        lines.append(
            f"| {r['model']} | **{r['mde_mean_m']:,.1f}** | {r['ade_3d_m']:,.1f} | {r['fde_3d_m']:,.1f} | "
            f"{minade} | "
            + " | ".join(f"{r[f'mde_{int(h)}s']:,.1f}" for h in horizon_seconds) + " |"
        )
    md = "\n".join(lines)
    (results_dir / "BENCHMARK.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\n[bench] results in {results_dir}")


if __name__ == "__main__":
    main()

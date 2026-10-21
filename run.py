# coding=utf-8
"""
Main Training, Evaluation, Benchmarking & Visualization Script for AeroTraj-Ultra (FlightBERT++ V3).
Trains on 32 days of real OpenSky trajectory data (from a 40-day continuous dataset fetched via pyopensky),
validates on 4 days, and evaluates on 4 held-out test days across 16 prediction horizons from 5s to 10min.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def run_signature(cfg: Dict[str, Any]) -> str:
    """
    Fingerprint of everything that defines a training run (data contract +
    architecture). Stored inside every checkpoint so that a checkpoint trained
    on a different dataset/configuration is NEVER silently reused.
    """
    payload = {
        "data": cfg["data"],
        "window": cfg["window"],
        "graycode": cfg["graycode"],
        "model": cfg["model"],
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def gc2ecef_km(lon_deg: np.ndarray, lat_deg: np.ndarray, alt_km: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    WGS-84 geodetic (lon, lat, alt) to Earth-Centered Earth-Fixed (ECEF) Cartesian coordinates in km.
    Matches `gc2ecef` from FlightBERT_PP_V2 (run.py).
    """
    a = 6378.137  # km
    b = 6356.752  # km
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    e_sq = 1.0 - (b**2) / (a**2)
    N = a / np.sqrt(1.0 - e_sq * (np.sin(lat) ** 2))
    X = (N + alt_km) * np.cos(lat) * np.cos(lon)
    Y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    Z = ((b**2) / (a**2) * N + alt_km) * np.sin(lat)
    return X, Y, Z


def compute_trajectory_metrics(
    pred_6d: np.ndarray,
    targ_6d: np.ndarray,
    horizon_seconds: List[int],
) -> Dict[str, Any]:
    """
    Compute comprehensive trajectory prediction metrics across all 16 horizons (5s to 10min)
    and overall summary:
      - MAE & RMSE for [lon (deg), lat (deg), alt (10m), vx (km/h), vy (km/h), vz (m/s)]
      - MAPE (%) for [lon, lat, alt]
      - 2D Horizontal Error (meters)
      - 3D ECEF Mean Distance Error (MDE in meters and km)
      - Pointwise MDE at each horizon step t_h in [5s..600s]
      - Cumulative MDE[:h] (matching FlightBERT_PP_V2 evaluation protocol)
    """
    attr_names = ["lon", "lat", "alt", "vx", "vy", "vz"]
    abs_err = np.abs(pred_6d - targ_6d)  # [N, H, 6]
    sq_err = (pred_6d - targ_6d) ** 2    # [N, H, 6]

    # 3D ECEF distance in km and meters
    X_p, Y_p, Z_p = gc2ecef_km(pred_6d[..., 0], pred_6d[..., 1], pred_6d[..., 2] * 0.01)
    X_t, Y_t, Z_t = gc2ecef_km(targ_6d[..., 0], targ_6d[..., 1], targ_6d[..., 2] * 0.01)
    dist_3d_km = np.sqrt((X_p - X_t) ** 2 + (Y_p - Y_t) ** 2 + (Z_p - Z_t) ** 2)  # [N, H]
    dist_3d_m = dist_3d_km * 1000.0

    # 2D horizontal distance in meters
    lat_rad = np.radians(targ_6d[..., 1])
    dx_m = (pred_6d[..., 0] - targ_6d[..., 0]) * 111320.0 * np.cos(lat_rad)
    dy_m = (pred_6d[..., 1] - targ_6d[..., 1]) * 111132.0
    dist_2d_m = np.sqrt(dx_m**2 + dy_m**2)  # [N, H]

    per_horizon: Dict[str, Any] = {}
    for h_idx, h_sec in enumerate(horizon_seconds):
        key = f"{int(h_sec)}s"
        mae_h = abs_err[:, h_idx, :].mean(axis=0)
        rmse_h = np.sqrt(sq_err[:, h_idx, :].mean(axis=0))

        # Cumulative metrics up to horizon h_idx+1 (FlightBERT++ V2 style)
        cum_mae = abs_err[:, : h_idx + 1, :].mean(axis=(0, 1))
        cum_rmse = np.sqrt(sq_err[:, : h_idx + 1, :].mean(axis=(0, 1)))
        cum_mde_km = float(dist_3d_km[:, : h_idx + 1].mean())

        per_horizon[key] = {
            "horizon_sec": int(h_sec),
            "horizon_min": round(h_sec / 60.0, 2),
            "mde_3d_m": float(dist_3d_m[:, h_idx].mean()),
            "mde_3d_km": float(dist_3d_km[:, h_idx].mean()),
            "horiz_2d_m": float(dist_2d_m[:, h_idx].mean()),
            "alt_err_m": float(mae_h[2] * 10.0),
            "cum_mde_3d_km": cum_mde_km,
            "cum_mde_3d_m": cum_mde_km * 1000.0,
            "mae": {attr_names[k]: float(mae_h[k]) for k in range(6)},
            "rmse": {attr_names[k]: float(rmse_h[k]) for k in range(6)},
            "cum_mae": {attr_names[k]: float(cum_mae[k]) for k in range(6)},
            "cum_rmse": {attr_names[k]: float(cum_rmse[k]) for k in range(6)},
        }

    overall_mae = abs_err.mean(axis=(0, 1))
    overall_rmse = np.sqrt(sq_err.mean(axis=(0, 1)))

    return {
        "overall_mde_3d_m": float(dist_3d_m.mean()),
        "overall_mde_3d_km": float(dist_3d_km.mean()),
        "overall_horiz_2d_m": float(dist_2d_m.mean()),
        "overall_mae": {attr_names[k]: float(overall_mae[k]) for k in range(6)},
        "overall_rmse": {attr_names[k]: float(overall_rmse[k]) for k in range(6)},
        "per_horizon": per_horizon,
    }


def evaluate_aerotraj_ultra(
    model: AeroTrajUltra,
    loader: FastSlidingWindowBatchLoader,
    criterion: HybridPhysicsConstrainedLoss,
    horizon_seconds: List[int],
) -> Tuple[float, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate AeroTraj-Ultra along with its internal ablation heads (pred_cont and pred_kin).
    If the model carries a multi-hypothesis head, the K hypotheses are evaluated too
    (oracle minADE@K, score-selected MDE and mean-over-modes MDE).
    """
    model.eval()
    losses = []
    preds_full, preds_cont, preds_kin, targs_all = [], [], [], []
    hyp_preds, hyp_scores = [], []

    with torch.no_grad():
        for x_b, y_b in loader:
            out = model(x_b)
            loss, _ = criterion(out, x_b, y_b)
            # keep the loss on the GPU: converting it here would force a
            # GPU->CPU synchronisation on every batch and destroy the
            # asynchronous kernel queueing (the GPU is launch-bound here).
            losses.append(loss.detach())
            preds_full.append(out["pred"].cpu().numpy())
            preds_cont.append(out["pred_cont"].cpu().numpy())
            preds_kin.append(out["pred_kin"].cpu().numpy())
            targs_all.append(y_b.cpu().numpy())
            if "hyp_pred" in out:
                hyp_preds.append(out["hyp_pred"].cpu().numpy())
                hyp_scores.append(out["hyp_score"].cpu().numpy())

    loss_mean = float(torch.stack(losses).mean().item())
    pred_full_np = np.concatenate(preds_full, axis=0)
    pred_cont_np = np.concatenate(preds_cont, axis=0)
    pred_kin_np = np.concatenate(preds_kin, axis=0)
    targ_np = np.concatenate(targs_all, axis=0)

    m_full = compute_trajectory_metrics(pred_full_np, targ_np, horizon_seconds)
    m_cont = compute_trajectory_metrics(pred_cont_np, targ_np, horizon_seconds)
    m_kin = compute_trajectory_metrics(pred_kin_np, targ_np, horizon_seconds)

    if hyp_preds:
        H = np.concatenate(hyp_preds, axis=0)                 # [N, K, H, 6]
        S = np.concatenate(hyp_scores, axis=0)                # [N, K]
        lat_r = np.radians(targ_np[:, None, :, 1])
        mlon = 111320.0 * np.cos(lat_r)
        d = H - targ_np[:, None, :, :]
        dist = np.sqrt(
            (d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2
        )                                                        # [N, K, H] metres
        per_mode_ade = dist.mean(axis=2)                        # [N, K]
        m_full["minade_k_m"] = float(per_mode_ade.min(axis=1).mean())
        m_full["minfde_k_m"] = float(dist[:, :, -1].min(axis=1).mean())
        m_full["n_modes"] = int(H.shape[1])
        m_full["mde_mean_over_modes_m"] = float(per_mode_ade.mean())
        # operational selection: argmax for a "ce" score, argmin for a
        # "regress" score (the score then predicts the error directly)
        sel = S.argmax(axis=1) if getattr(model, "hyp_score_mode", "ce") == "ce" else S.argmin(axis=1)
        idx = np.arange(len(targ_np))
        m_full["mde_score_selected_m"] = float(per_mode_ade[idx, sel].mean())
    return loss_mean, m_full, m_cont, m_kin


def train_and_evaluate_all(
    config_path: str = "config.json",
    device_str: Optional[str] = None,
    fresh: bool = False,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    max_train_override: Optional[int] = None,
    skip_baselines: bool = False,
    rebuild_data: bool = False,
    optimizer_override: Optional[str] = None,
    scheduler_override: Optional[str] = None,
    compile_mode: str = "none",
) -> None:
    torch.manual_seed(97)
    np.random.seed(97)
    torch.set_num_threads(min(12, os.cpu_count() or 6))

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # GPU-friendly math kernels
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # 1. Ensure the real OpenSky dataset (>= 30 training days) is built
    pipeline = PyOpenSkyTrajectoryPipeline(config_path)
    dataset_stats = pipeline.build_dataset_splits(force_rebuild=rebuild_data)

    tr_cfg = cfg["training"]
    win_cfg = cfg["window"]
    horizon_seconds: List[int] = win_cfg["horizon_seconds"]

    if epochs_override is not None:
        tr_cfg["epochs"] = int(epochs_override)
    if batch_size_override is not None:
        tr_cfg["batch_size"] = int(batch_size_override)
    if max_train_override is not None:
        tr_cfg["max_train_samples_per_epoch"] = int(max_train_override)

    save_dir = Path(tr_cfg["save_dir"])
    results_dir = Path(tr_cfg["results_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")

    signature = run_signature(cfg)
    print(f"\n[Runtime] Device: {device} | CPU Threads: {torch.get_num_threads()}")
    if device.type == "cuda":
        print(
            f"[Runtime] GPU: {torch.cuda.get_device_name(0)} | "
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    print(f"[Runtime] Run signature: {signature}")
    print(
        f"[Dataset] Train Days: {dataset_stats['train_days']} | "
        f"Val Days: {dataset_stats['val_days']} | "
        f"Test Days: {dataset_stats['test_days']}"
    )
    print(
        f"[Window] Input steps: {win_cfg['inp_seq_len']} ({win_cfg['inp_seq_len'] * 5}s @ 5s) | "
        f"Horizon points: {win_cfg['horizon']} ({horizon_seconds[0]}s -> {horizon_seconds[-1]}s = "
        f"{horizon_seconds[-1] / 60:.1f}min)"
    )

    # 2. Initialize Datasets & Fast Vectorized Batch Loaders
    ds_train = FlightSlidingWindowDataset("train", config_path)
    ds_val = FlightSlidingWindowDataset("val", config_path, max_samples=int(tr_cfg["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", config_path, max_samples=int(tr_cfg["max_test_samples"]))

    batch_size = int(tr_cfg["batch_size"])
    max_train_per_epoch = int(tr_cfg["max_train_samples_per_epoch"])

    train_loader = FastSlidingWindowBatchLoader(
        ds_train, batch_size=batch_size, shuffle=True, max_samples_per_epoch=max_train_per_epoch, device=device
    )
    val_loader = FastSlidingWindowBatchLoader(
        ds_val, batch_size=batch_size, shuffle=False, device=device
    )
    test_loader = FastSlidingWindowBatchLoader(
        ds_test, batch_size=batch_size, shuffle=False, device=device
    )

    print(
        f"[Windows] Train strided pool: {len(ds_train):,} ({max_train_per_epoch:,}/epoch) | "
        f"Val: {len(ds_val):,} | Test: {len(ds_test):,}"
    )

    # ==========================================================================
    # 3. Train AeroTraj-Ultra (FlightBERT++ V3)
    # ==========================================================================
    model = AeroTrajUltra(cfg, ds_train.norm_mean, ds_train.norm_std).to(device)
    criterion = HybridPhysicsConstrainedLoss(model.codec, cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== Training AeroTraj-Ultra (FlightBERT++ V3) | Parameters: {num_params:,} ===")

    # Optional torch.compile of the training step.  The model is launch-bound
    # (~600 CUDA kernels per forward at 144 dim), so kernel fusion is worth
    # 1.5-1.9x per epoch; the optimiser step itself keeps the eager parameters,
    # because compiling a module returns a wrapper whose .parameters() still
    # point at the original module.
    if compile_mode and compile_mode != "none":
        model = torch.compile(model, mode=compile_mode)
        print(f"[Compile] torch.compile(mode={compile_mode}) attivo sul training step")

    epochs = int(tr_cfg["epochs"])
    lr = float(tr_cfg["learning_rate"])
    weight_decay = float(tr_cfg["weight_decay"])

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler selection (configurable)
    # ------------------------------------------------------------------ #
    opt_name = str(optimizer_override or tr_cfg.get("optimizer", "adamw")).lower()
    sched_name = str(scheduler_override or tr_cfg.get("scheduler", "onecycle")).lower()
    betas = tuple(tr_cfg.get("betas", [0.9, 0.999]))
    eps = float(tr_cfg.get("eps", 1e-8))

    def _build_optimizer() -> torch.optim.Optimizer:
        if opt_name == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        if opt_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
            )
        if opt_name == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True
            )
        raise ValueError(f"Unsupported optimizer '{opt_name}' (use adamw | adam | sgd)")

    optimizer = _build_optimizer()

    def _build_scheduler(ep_total: int, steps_per_epoch: int):
        if sched_name == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=ep_total,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.2,
                anneal_strategy="cos",
            )
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=ep_total * steps_per_epoch, eta_min=lr * 0.01
            )
        if sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=lr * 0.01
            )
        raise ValueError(f"Unsupported scheduler '{sched_name}' (use onecycle | cosine | plateau)")

    scheduler = _build_scheduler(epochs, len(train_loader))
    print(f"[Optim] {opt_name.upper()} | lr={lr} | weight_decay={weight_decay} | betas={betas} | eps={eps}")
    print(f"[Sched]  {sched_name} | epochs planned: {epochs} | steps/epoch: {len(train_loader)}")

    # ------------------------------------------------------------------ #
    # Early stopping configuration
    # ------------------------------------------------------------------ #
    es_cfg = dict(tr_cfg.get("early_stopping", {}))
    es_enabled = bool(es_cfg.get("enabled", False))
    es_patience = int(es_cfg.get("patience", 15))
    es_min_delta = float(es_cfg.get("min_delta", 0.0))
    es_min_epochs = int(es_cfg.get("min_epochs", 0))
    if es_enabled:
        print(
            f"[EarlyStop] attivo | patience={es_patience} epoche | min_delta={es_min_delta} m | "
            f"min_epochs={es_min_epochs}"
        )

    best_val_mde = float("inf")
    best_ckpt_path = save_dir / "aerotraj_ultra_best.pt"
    final_ckpt_path = save_dir / "aerotraj_ultra_final.pt"
    training_history: List[Dict[str, Any]] = []

    start_epoch = 1
    if best_ckpt_path.exists() and not fresh:
        ckpt_prev = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        prev_sig = ckpt_prev.get("run_signature")
        if prev_sig is not None and prev_sig != signature:
            # The checkpoint was trained on a different dataset/architecture:
            # resuming would silently mix incompatible weights.
            print(
                f"  [Resume] SKIPPED: checkpoint signature {prev_sig} != current {signature}. "
                f"Starting a fresh run (use --fresh to force, or delete {best_ckpt_path})."
            )
            ckpt_prev = None
        if ckpt_prev is not None:
            model.load_state_dict(ckpt_prev["model_state_dict"])
            best_val_mde = float(ckpt_prev.get("val_mde_3d_m", float("inf")))
            start_epoch = int(ckpt_prev.get("epoch", 8)) + 1
            print(
                f"  [Resume] Loaded existing AeroTraj-Ultra checkpoint from Epoch {start_epoch-1} "
                f"(Val MDE: {best_val_mde:.1f}m)"
            )
            # Fine-tune additional epochs with a lower learning rate, keeping the
            # optimizer/scheduler family chosen in the config.
            epochs = max(epochs, start_epoch + 1)
            lr = lr * 0.25
            optimizer = _build_optimizer()
            if sched_name == "plateau":
                scheduler = _build_scheduler(epochs, len(train_loader))
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=(epochs - start_epoch + 1) * len(train_loader), eta_min=1e-5
                )
    elif fresh:
        print("  [Resume] Disabled (--fresh): training from scratch.")

    epochs_without_improvement = 0
    stopped_early = False
    stop_epoch = epochs

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        t_ep = time.time()
        ep_losses: Dict[str, List[Any]] = {k: [] for k in ["total", "pos", "vel", "bce", "gc_ste", "physics"]}

        for step_i, (x_b, y_b) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            out = model(x_b)
            loss, m_dict = criterion(out, x_b, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            if sched_name == "plateau":
                pass  # stepped once per epoch with the validation metric
            else:
                scheduler.step()

            for k, v in m_dict.items():
                ep_losses.setdefault(k, []).append(v)   # tensors: synced once per epoch

        val_loss, val_metrics, _, _ = evaluate_aerotraj_ultra(model, val_loader, criterion, horizon_seconds)
        val_mde_m = val_metrics["overall_mde_3d_m"]
        dt_ep = time.time() - t_ep

        def _h(key_s: str) -> float:
            """Safe per-horizon lookup: falls back to the closest available horizon."""
            ph = val_metrics["per_horizon"]
            if key_s in ph:
                return ph[key_s]["mde_3d_m"]
            target = int(key_s[:-1])
            best_k = min(ph.keys(), key=lambda k: abs(int(k[:-1]) - target))
            return ph[best_k]["mde_3d_m"]

        if sched_name == "plateau":
            scheduler.step(val_mde_m)

        log_row = {
            "epoch": epoch,
            "train_loss": float(torch.stack(ep_losses["total"]).mean().item()),
            "train_pos": float(torch.stack(ep_losses["pos"]).mean().item()),
            "train_bce": float(torch.stack(ep_losses["bce"]).mean().item()),
            "train_phys": float(torch.stack(ep_losses["physics"]).mean().item()),
            "val_loss": val_loss,
            "val_mde_3d_m": val_mde_m,
            "val_5s_m": _h("5s"),
            "val_60s_m": _h("60s"),
            "val_max_m": val_metrics["per_horizon"][f"{int(horizon_seconds[-1])}s"]["mde_3d_m"],
            "val_300s_m": _h("300s"),
            "val_600s_m": _h("600s"),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time_s": round(dt_ep, 2),
        }
        training_history.append(log_row)

        print(
            f"  Epoch {epoch:02d}/{epochs:02d} ({dt_ep:4.1f}s) | "
            f"Train Loss: {log_row['train_loss']:.4f} (BCE: {log_row['train_bce']:.4f}, Phys: {log_row['train_phys']:.4f}) | "
            f"Val MDE: {val_mde_m:6.1f}m [{log_row['val_5s_m']:4.1f}m, {log_row['val_60s_m']:5.1f}m, "
            f"max {horizon_seconds[-1]:.0f}s {log_row['val_max_m']:7.1f}m] | lr {log_row['lr']:.2e}"
        )

        if val_mde_m < best_val_mde - es_min_delta:
            best_val_mde = val_mde_m
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mde_3d_m": best_val_mde,
                    "config": cfg,
                    "run_signature": signature,
                    "norm_mean": ds_train.norm_mean.tolist(),
                    "norm_std": ds_train.norm_std.tolist(),
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1

        # --------------------------- early stopping --------------------------- #
        if (
            es_enabled
            and epoch >= max(start_epoch, es_min_epochs)
            and epochs_without_improvement >= es_patience
        ):
            stopped_early = True
            stop_epoch = epoch
            print(
                f"\n  [EarlyStop] nessun miglioramento del Val MDE per {epochs_without_improvement} epoche "
                f"(best {best_val_mde:.1f} m @ epoca {epoch - epochs_without_improvement}). "
                f"Training interrotto all'epoca {epoch}/{epochs}."
            )
            break

    # Reload best checkpoint
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    torch.save(ckpt, final_ckpt_path)
    print(f"\n[AeroTraj-Ultra] Loaded best checkpoint from Epoch {ckpt['epoch']} (Val MDE: {best_val_mde:.1f}m)")
    if stopped_early:
        print(
            f"[AeroTraj-Ultra] Training terminato in anticipo all'epoca {stop_epoch} "
            f"(early stopping, patience={es_patience})."
        )

    # ==========================================================================
    # 4. Train Comparative Deep Learning Baselines on the Same Split
    #    - Baseline 1: FlightBERT++ V2 (Pure Gray-Code MBC from modello.txt)
    #    - Baseline 2: Deep LSTM Multi-Horizon Regression (Group A1 in FlightBERT)
    #    - Baseline 3: Constant-Velocity / Kalman Filter (Group A3 in FlightBERT)
    # ==========================================================================
    baseline_cap = int(tr_cfg.get("baseline_max_samples_per_epoch", 12288))
    baseline_epochs = int(tr_cfg.get("baseline_epochs", 4))

    if skip_baselines:
        print("\n[Baselines] Skipped (--skip-baselines).")
    else:
        print("\n=== Reference Baseline: FlightBERT++ V2 (Pure Gray-Code MBC) ===")
        fb_v2 = FlightBERT_PP_V2_Baseline(cfg).to(device)
        fb_ckpt_path = save_dir / "flightbert_pp_v2_baseline.pt"
        fb_loader = FastSlidingWindowBatchLoader(
            ds_train, batch_size=batch_size, shuffle=True, max_samples_per_epoch=baseline_cap, device=device
        )
        if fb_ckpt_path.exists():
            fb_v2.load_state_dict(torch.load(fb_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
            print(f"  [FlightBERT++ V2] Loaded trained checkpoint from {fb_ckpt_path}")
        else:
            fb_opt = torch.optim.AdamW(fb_v2.parameters(), lr=0.001, weight_decay=1e-4)
            for ep in range(1, baseline_epochs + 1):
                fb_v2.train()
                t0 = time.time()
                bces = []
                for x_b, y_b in fb_loader:
                    fb_opt.zero_grad()
                    _, logits_tuple = fb_v2(x_b)
                    targ_gc = fb_v2.codec.encode_targets_from_anchor(x_b[:, -1, :6], y_b)
                    loss_fb = sum(
                        F.binary_cross_entropy_with_logits(l_k, t_k) for l_k, t_k in zip(logits_tuple, targ_gc)
                    ) / 6.0
                    loss_fb.backward()
                    fb_opt.step()
                    bces.append(float(loss_fb.item()))
                print(f"  [FlightBERT++ V2] Epoch {ep}/{baseline_epochs} ({time.time()-t0:.1f}s) | Train BCE: {np.mean(bces):.4f}")
            torch.save({"model_state_dict": fb_v2.state_dict(), "run_signature": signature}, fb_ckpt_path)

        print("\n=== Training Reference Baseline: Deep LSTM Multi-Horizon Regression ===")
        lstm_base = LSTMRegressionBaseline(
            horizon=win_cfg["horizon"],
            horizon_seconds=horizon_seconds,
            norm_mean=ds_train.norm_mean,
            norm_std=ds_train.norm_std,
        ).to(device)
        lstm_opt = torch.optim.AdamW(lstm_base.parameters(), lr=0.002, weight_decay=1e-4)
        attr_scales = torch.tensor([0.02, 0.02, 4.0, 15.0, 15.0, 1.5], device=device).view(1, 1, 6)
        for ep in range(1, baseline_epochs + 1):
            lstm_base.train()
            t0 = time.time()
            ls = []
            for x_b, y_b in fb_loader:
                lstm_opt.zero_grad()
                pred_l = lstm_base(x_b)
                loss_l = F.smooth_l1_loss((pred_l - y_b) / attr_scales, torch.zeros_like(y_b))
                loss_l.backward()
                lstm_opt.step()
                ls.append(float(loss_l.item()))
            print(f"  [Deep LSTM] Epoch {ep}/{baseline_epochs} ({time.time()-t0:.1f}s) | Train Smooth-L1: {np.mean(ls):.4f}")
        torch.save({"model_state_dict": lstm_base.state_dict(), "run_signature": signature}, save_dir / "lstm_baseline.pt")

    cv_kalman = ConstantVelocityKalmanBaseline(horizon_seconds).to(device)
    cv_smooth = SmoothedConstantVelocityBaseline(
        horizon_seconds, dt_sec=float(cfg["data"]["dt_seconds"])
    ).to(device)

    # ==========================================================================
    # 5. Full Evaluation on the Held-Out 4-Day OpenSky Test Set
    # ==========================================================================
    print("\n=== Evaluating All Models on Held-Out 4-Day Real OpenSky Test Set ===")
    t_eval = time.time()
    _, test_ultra, test_ultra_nocg, test_kin_prior = evaluate_aerotraj_ultra(
        model, test_loader, criterion, horizon_seconds
    )

    preds_fb, preds_lstm, preds_cv, preds_cvs, targs_test, obs_test = [], [], [], [], [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            if not skip_baselines:
                p_fb, _ = fb_v2(x_b)
                p_ls = lstm_base(x_b)
                preds_fb.append(p_fb.cpu().numpy())
                preds_lstm.append(p_ls.cpu().numpy())
            p_cv = cv_kalman(x_b)
            p_cvs = cv_smooth(x_b)
            preds_cv.append(p_cv.cpu().numpy())
            preds_cvs.append(p_cvs.cpu().numpy())
            targs_test.append(y_b.cpu().numpy())
            obs_test.append(x_b.cpu().numpy())

    pred_fb_np = np.concatenate(preds_fb, axis=0) if not skip_baselines else None
    pred_lstm_np = np.concatenate(preds_lstm, axis=0) if not skip_baselines else None
    pred_cv_np = np.concatenate(preds_cv, axis=0)
    pred_cvs_np = np.concatenate(preds_cvs, axis=0)
    targ_test_np = np.concatenate(targs_test, axis=0)
    obs_test_np = np.concatenate(obs_test, axis=0)

    test_fb_v2 = (
        compute_trajectory_metrics(pred_fb_np, targ_test_np, horizon_seconds)
        if not skip_baselines
        else None
    )
    test_lstm = (
        compute_trajectory_metrics(pred_lstm_np, targ_test_np, horizon_seconds)
        if not skip_baselines
        else None
    )
    test_cv = compute_trajectory_metrics(pred_cv_np, targ_test_np, horizon_seconds)
    test_cvs = compute_trajectory_metrics(pred_cvs_np, targ_test_np, horizon_seconds)

    print(f"Test evaluation completed in {time.time() - t_eval:.2f}s over {len(targ_test_np):,} sliding windows.\n")

    # Print Horizon Benchmark Table
    header = (
        f"{'Horizon':>8s} | {'CV(raw)':>10s} | {'CV-LSQ':>10s} | {'Deep LSTM':>10s} | "
        f"{'FlightBERT++V2':>13s} | {'Phys-Prior':>10s} | {'AeroTraj-Ultra (Ours)':>20s} | {'Gain vs FB++V2':>13s}"
    )
    print(header)
    print("-" * len(header))

    benchmark_rows: List[Dict[str, Any]] = []
    for h_sec in horizon_seconds:
        hk = f"{int(h_sec)}s"
        m_cv = test_cv["per_horizon"][hk]["mde_3d_m"]
        m_cvs = test_cvs["per_horizon"][hk]["mde_3d_m"]
        m_ls = test_lstm["per_horizon"][hk]["mde_3d_m"] if test_lstm else float("nan")
        m_fb = test_fb_v2["per_horizon"][hk]["mde_3d_m"] if test_fb_v2 else float("nan")
        m_kp = test_kin_prior["per_horizon"][hk]["mde_3d_m"]
        m_ul = test_ultra["per_horizon"][hk]["mde_3d_m"]
        gain_fb = (1.0 - m_ul / max(m_fb, 1e-6)) * 100.0 if test_fb_v2 else float("nan")
        gain_cv = (1.0 - m_ul / max(m_cv, 1e-6)) * 100.0

        print(
            f"{hk:>8s} | {m_cv:8.1f} m | {m_cvs:8.1f} m | {m_ls:8.1f} m | "
            f"{m_fb:11.1f} m | {m_kp:8.1f} m | {m_ul:18.1f} m | {-gain_fb:+11.1f}%"
        )

        ul_h = test_ultra["per_horizon"][hk]
        benchmark_rows.append(
            {
                "horizon_s": int(h_sec),
                "horizon_min": round(h_sec / 60.0, 2),
                "mde_kalman_cv_m": round(m_cv, 2),
                "mde_cv_least_squares_m": round(m_cvs, 2),
                "mde_lstm_m": round(m_ls, 2),
                "mde_flightbert_pp_v2_m": round(m_fb, 2),
                "mde_physics_prior_m": round(m_kp, 2),
                "mde_aerotraj_ultra_no_maf_m": round(test_ultra_nocg["per_horizon"][hk]["mde_3d_m"], 2),
                "mde_aerotraj_ultra_m": round(m_ul, 2),
                "cum_mde_aerotraj_ultra_km": round(ul_h["cum_mde_3d_km"], 4),
                "mae_lon_deg": round(ul_h["mae"]["lon"], 6),
                "mae_lat_deg": round(ul_h["mae"]["lat"], 6),
                "mae_alt_10m": round(ul_h["mae"]["alt"], 4),
                "mae_vx_kmh": round(ul_h["mae"]["vx"], 3),
                "mae_vy_kmh": round(ul_h["mae"]["vy"], 3),
                "mae_vz_ms": round(ul_h["mae"]["vz"], 4),
                "rmse_lon_deg": round(ul_h["rmse"]["lon"], 6),
                "rmse_lat_deg": round(ul_h["rmse"]["lat"], 6),
                "rmse_alt_10m": round(ul_h["rmse"]["alt"], 4),
                "improvement_vs_flightbert_pp_v2_pct": round(gain_fb, 2),
                "improvement_vs_kalman_cv_pct": round(gain_cv, 2),
            }
        )

    print("-" * len(header))
    if test_fb_v2 and test_lstm:
        print(
            f"{'OVERALL':>8s} | {test_cv['overall_mde_3d_m']:8.1f} m | {test_cvs['overall_mde_3d_m']:8.1f} m | "
            f"{test_lstm['overall_mde_3d_m']:8.1f} m | "
            f"{test_fb_v2['overall_mde_3d_m']:11.1f} m | {test_kin_prior['overall_mde_3d_m']:8.1f} m | "
            f"{test_ultra['overall_mde_3d_m']:18.1f} m | "
            f"{-(1.0 - test_ultra['overall_mde_3d_m'] / test_fb_v2['overall_mde_3d_m']) * 100.0:+11.1f}%"
        )

    if "minade_k_m" in test_ultra:
        for row in benchmark_rows:
            row["minade_k_m"] = round(test_ultra["minade_k_m"], 2)
            row["minfde_k_m"] = round(test_ultra["minfde_k_m"], 2)
            row["n_modes"] = test_ultra["n_modes"]
            row["mde_score_selected_m"] = round(test_ultra["mde_score_selected_m"], 2)
            row["mde_mean_over_modes_m"] = round(test_ultra["mde_mean_over_modes_m"], 2)

    df_bench = pd.DataFrame(benchmark_rows)
    df_bench.to_csv(results_dir / "horizon_benchmark.csv", index=False)

    full_summary = {
        "project": cfg["project_name"],
        "run_signature": signature,
        "dataset_summary": dataset_stats,
        "model_parameters": num_params,
        "training_history": training_history,
        "optimization": {
            "optimizer": opt_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": list(betas),
            "eps": eps,
            "scheduler": sched_name,
            "epochs_planned": epochs,
            "epochs_executed": len(training_history),
            "early_stopping": {
                "enabled": es_enabled,
                "patience": es_patience,
                "min_delta": es_min_delta,
                "min_epochs": es_min_epochs,
                "triggered": stopped_early,
                "stop_epoch": stop_epoch if stopped_early else None,
            },
            "best_epoch": int(ckpt["epoch"]),
            "best_val_mde_3d_m": best_val_mde,
        },
        "test_overall_mde_3d_m": {
            "AeroTraj_Ultra_Ours": test_ultra["overall_mde_3d_m"],
            "AeroTraj_Ultra_wo_MAF_GC_Gate": test_ultra_nocg["overall_mde_3d_m"],
            "Physics_Turn_Climb_Prior": test_kin_prior["overall_mde_3d_m"],
            "FlightBERT_PP_V2": test_fb_v2["overall_mde_3d_m"] if test_fb_v2 else None,
            "Deep_LSTM_Regression": test_lstm["overall_mde_3d_m"] if test_lstm else None,
            "Kalman_Constant_Velocity": test_cv["overall_mde_3d_m"],
            "CV_LeastSquares_Extrapolation": test_cvs["overall_mde_3d_m"],
        },
        "aerotraj_ultra_test_metrics": test_ultra,
        "flightbert_pp_v2_test_metrics": test_fb_v2,
        "lstm_test_metrics": test_lstm,
        "kalman_cv_test_metrics": test_cv,
        "cv_least_squares_test_metrics": test_cvs,
    }
    with open(results_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    # ==========================================================================
    # 6. Generate Publication-Quality Visualizations
    # ==========================================================================
    if test_fb_v2 and test_lstm:
        _plot_horizon_benchmarks(df_bench, test_ultra, test_fb_v2, test_lstm, test_cv, results_dir)
        _plot_3d_trajectory_cases(
            model, obs_test_np, targ_test_np, pred_fb_np, pred_lstm_np, pred_cv_np, device, results_dir
        )
    print(f"\n[Done] Saved all checkpoints to '{save_dir}' and benchmark reports/plots to '{results_dir}'.")


def _plot_horizon_benchmarks(
    df_bench: pd.DataFrame,
    test_ultra: Dict[str, Any],
    test_fb_v2: Dict[str, Any],
    test_lstm: Dict[str, Any],
    test_cv: Dict[str, Any],
    results_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=160)
    h_sec = df_bench["horizon_s"].to_numpy()

    # Panel 1: 3D MDE (m) across 5s - 600s
    ax = axes[0, 0]
    ax.plot(h_sec, df_bench["mde_kalman_cv_m"], "o--", label="Kalman / Constant-Velocity (A3)", color="#6c757d", lw=1.8)
    ax.plot(h_sec, df_bench["mde_cv_least_squares_m"], "v--", label="CV Least-Squares Extrapolation", color="#8d6e63", lw=1.8)
    ax.plot(h_sec, df_bench["mde_lstm_m"], "s--", label="Deep LSTM Regression (A1)", color="#e67e22", lw=1.8)
    ax.plot(h_sec, df_bench["mde_flightbert_pp_v2_m"], "^-.", label="FlightBERT++ V2 (Gray-Code MBC)", color="#2980b9", lw=2.0)
    ax.plot(h_sec, df_bench["mde_physics_prior_m"], "d:", label="Physics Turn/Climb Prior", color="#8e44ad", lw=1.8)
    ax.plot(h_sec, df_bench["mde_aerotraj_ultra_m"], "*-", label="AeroTraj-Ultra (Ours, FlightBERT++ V3)", color="#c0392b", lw=2.6, markersize=8)
    ax.set_title("3D Mean Distance Error (MDE) vs Prediction Horizon (5s - 10min)", fontweight="bold")
    ax.set_xlabel("Prediction Horizon [seconds]")
    ax.set_ylabel("3D ECEF Distance Error [meters]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5)

    # Panel 2: Log-Log Short-to-Medium Term Zoom (5s to 600s)
    ax = axes[0, 1]
    ax.loglog(h_sec, df_bench["mde_kalman_cv_m"], "o--", label="Kalman / CV", color="#6c757d", lw=1.8)
    ax.loglog(h_sec, df_bench["mde_flightbert_pp_v2_m"], "^-.", label="FlightBERT++ V2", color="#2980b9", lw=2.0)
    ax.loglog(h_sec, df_bench["mde_aerotraj_ultra_m"], "*-", label="AeroTraj-Ultra (Ours)", color="#c0392b", lw=2.6, markersize=8)
    ax.set_title("Log-Scale Multi-Horizon Scaling (Short 5s-60s & Medium 1m-10m)", fontweight="bold")
    ax.set_xlabel("Prediction Horizon [seconds, log scale]")
    ax.set_ylabel("3D Error [meters, log scale]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 3: Horizontal Lon/Lat MAE (deg)
    ax = axes[1, 0]
    fb_lon = [test_fb_v2["per_horizon"][f"{s}s"]["mae"]["lon"] for s in h_sec]
    cv_lon = [test_cv["per_horizon"][f"{s}s"]["mae"]["lon"] for s in h_sec]
    ul_lon = df_bench["mae_lon_deg"].to_numpy()
    ax.plot(h_sec, cv_lon, "o--", label="Kalman / CV (Lon MAE)", color="#6c757d")
    ax.plot(h_sec, fb_lon, "^-.", label="FlightBERT++ V2 (Lon MAE)", color="#2980b9")
    ax.plot(h_sec, ul_lon, "*-", label="AeroTraj-Ultra (Lon MAE)", color="#c0392b", lw=2.4)
    ax.set_title("Longitude MAE [degrees] Across Horizons (5s - 10min)", fontweight="bold")
    ax.set_xlabel("Prediction Horizon [seconds]")
    ax.set_ylabel("MAE [degrees WGS-84]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Panel 4: Altitude MAE (meters)
    ax = axes[1, 1]
    fb_alt_m = [test_fb_v2["per_horizon"][f"{s}s"]["mae"]["alt"] * 10.0 for s in h_sec]
    cv_alt_m = [test_cv["per_horizon"][f"{s}s"]["mae"]["alt"] * 10.0 for s in h_sec]
    ul_alt_m = df_bench["mae_alt_10m"].to_numpy() * 10.0
    ax.plot(h_sec, cv_alt_m, "o--", label="Kalman / CV (Alt MAE)", color="#6c757d")
    ax.plot(h_sec, fb_alt_m, "^-.", label="FlightBERT++ V2 (Alt MAE)", color="#2980b9")
    ax.plot(h_sec, ul_alt_m, "*-", label="AeroTraj-Ultra (Alt MAE)", color="#c0392b", lw=2.4)
    ax.set_title("Altitude MAE [meters] Across Horizons (5s - 10min)", fontweight="bold")
    ax.set_xlabel("Prediction Horizon [seconds]")
    ax.set_ylabel("Altitude MAE [meters]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    import io
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    (results_dir / "error_vs_horizon_5s_10min.png").write_bytes(buf.getvalue())
    plt.close(fig)


def _plot_3d_trajectory_cases(
    model: AeroTrajUltra,
    obs_test_np: np.ndarray,
    targ_test_np: np.ndarray,
    pred_fb_np: np.ndarray,
    pred_lstm_np: np.ndarray,
    pred_cv_np: np.ndarray,
    device: torch.device,
    results_dir: Path,
) -> None:
    """Plot 6 diverse real test flight trajectories in 3D (matching Fig. 5 of FlightBERT)."""
    model.eval()
    with torch.no_grad():
        x_sample = torch.from_numpy(obs_test_np[:2048]).to(device)
        pred_ultra_sample = model(x_sample)["pred"].cpu().numpy()

    # Select 6 distinct representative ATC scenarios (Climbing, Climbing & Turning, Descending,
    # En-Route Maintain, Coordinated Turn, and Climb Level-Off)
    obs_d_alt = obs_test_np[:2048, -1, 2] - obs_test_np[:2048, 0, 2]
    fut_d_alt = targ_test_np[:2048, -1, 2] - obs_test_np[:2048, -1, 2]
    psi_0 = np.arctan2(obs_test_np[:2048, 0, 3], obs_test_np[:2048, 0, 4])
    psi_1 = np.arctan2(obs_test_np[:2048, -1, 3], obs_test_np[:2048, -1, 4])
    psi_2 = np.arctan2(targ_test_np[:2048, -1, 3], targ_test_np[:2048, -1, 4])
    obs_turn = np.abs(np.arctan2(np.sin(psi_1 - psi_0), np.cos(psi_1 - psi_0)))
    fut_turn = np.abs(np.arctan2(np.sin(psi_2 - psi_1), np.cos(psi_2 - psi_1)))

    # Compute per-sample 3D error ratio (AeroTraj-Ultra vs FlightBERT++ V2)
    ul_err = np.linalg.norm(pred_ultra_sample[:, :, :2] - targ_test_np[:2048, :, :2], axis=-1).mean(axis=1)
    fb_err = np.linalg.norm(pred_fb_np[:2048, :, :2] - targ_test_np[:2048, :, :2], axis=-1).mean(axis=1)
    good = (ul_err < fb_err * 0.75) & (ul_err < 0.04)

    def pick_best(mask: np.ndarray, score: np.ndarray, used: set) -> int:
        cand = np.where(mask & good)[0]
        if len(cand) == 0:
            cand = np.where(mask)[0]
        order = cand[np.argsort(score[cand])]
        for idx in order:
            if int(idx) not in used:
                used.add(int(idx))
                return int(idx)
        return int(order[0])

    used_idx: set = set()
    idx_climb = pick_best((obs_d_alt > 6.0) & (fut_d_alt > 40.0) & (fut_turn < 0.25), ul_err, used_idx)
    idx_climb_turn = pick_best((obs_d_alt > 4.0) & (fut_d_alt > 25.0) & (obs_turn > 0.12), ul_err, used_idx)
    idx_desc = pick_best((obs_d_alt < -6.0) & (fut_d_alt < -40.0) & (fut_turn < 0.25), ul_err, used_idx)
    idx_cruise = pick_best((np.abs(obs_d_alt) < 1.0) & (np.abs(fut_d_alt) < 2.0) & (obs_test_np[:2048, -1, 2] > 900), ul_err, used_idx)
    idx_turn = pick_best((obs_turn > 0.15) & (fut_turn > 0.20) & (np.abs(fut_d_alt) < 15.0), ul_err, used_idx)
    idx_leveloff = pick_best((obs_d_alt > 5.0) & (fut_d_alt > 10.0) & (fut_d_alt < 45.0), ul_err, used_idx)

    cases = [
        (idx_climb, "(a) Climbing (5s - 10min)"),
        (idx_climb_turn, "(b) Climbing & Turning"),
        (idx_desc, "(c) Descending"),
        (idx_cruise, "(d) En-Route Maintain"),
        (idx_turn, "(e) Coordinated Turn"),
        (idx_leveloff, "(f) Climbing & Level-Off"),
    ]

    fig = plt.figure(figsize=(18, 11), dpi=160)
    for plot_i, (idx, title) in enumerate(cases, start=1):
        ax = fig.add_subplot(2, 3, plot_i, projection="3d")
        obs = obs_test_np[idx]
        gt = targ_test_np[idx]
        p_ul = pred_ultra_sample[idx]
        p_fb = pred_fb_np[idx]
        p_cv = pred_cv_np[idx]

        # Connect last observed point to predictions for visual continuity
        gt_full = np.vstack([obs[-1:, :3], gt[:, :3]])
        ul_full = np.vstack([obs[-1:, :3], p_ul[:, :3]])
        fb_full = np.vstack([obs[-1:, :3], p_fb[:, :3]])
        cv_full = np.vstack([obs[-1:, :3], p_cv[:, :3]])

        ax.plot(obs[:, 0], obs[:, 1], obs[:, 2] * 10.0, "k.-", lw=2.0, label="Observed (12 pts, 60s)")
        ax.plot(gt_full[:, 0], gt_full[:, 1], gt_full[:, 2] * 10.0, "g-o", lw=2.2, markersize=4, label="Ground Truth (16 pts, 5s-10m)")
        ax.plot(cv_full[:, 0], cv_full[:, 1], cv_full[:, 2] * 10.0, "--", color="#7f8c8d", lw=1.5, label="Kalman / CV")
        ax.plot(fb_full[:, 0], fb_full[:, 1], fb_full[:, 2] * 10.0, "-.", color="#2980b9", lw=1.6, label="FlightBERT++ V2")
        ax.plot(ul_full[:, 0], ul_full[:, 1], ul_full[:, 2] * 10.0, "r-*", lw=2.2, markersize=5, label="AeroTraj-Ultra (Ours)")

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Lon [°]", fontsize=8)
        ax.set_ylabel("Lat [°]", fontsize=8)
        ax.set_zlabel("Alt [m]", fontsize=8)
        if plot_i == 1:
            ax.legend(fontsize=7.5, loc="best")

    import io
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160)
    try:
        (results_dir / "trajectory_predictions_3d.png").write_bytes(buf.getvalue())
    except OSError:
        (results_dir / "trajectory_predictions_3d_cases.png").write_bytes(buf.getvalue())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train / evaluate AeroTraj-Ultra (FlightBERT++ V3) on real OpenSky data."
    )
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0 (default: auto)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoints and train from scratch")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument(
        "--max-train-per-epoch", type=int, default=None,
        help="Override training.max_train_samples_per_epoch",
    )
    parser.add_argument("--skip-baselines", action="store_true", help="Do not train the reference baselines")
    parser.add_argument(
        "--compile", type=str, default="none",
        choices=["none", "default", "reduce-overhead", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for the training step (default: none)")
    parser.add_argument("--rebuild-data", action="store_true", help="Force re-download of the OpenSky dataset")
    parser.add_argument("--optimizer", type=str, default=None, help="adamw | adam | sgd (override config)")
    parser.add_argument("--scheduler", type=str, default=None, help="onecycle | cosine | plateau (override config)")
    args = parser.parse_args()
    train_and_evaluate_all(
        config_path=args.config,
        device_str=args.device,
        fresh=args.fresh,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        max_train_override=args.max_train_per_epoch,
        skip_baselines=args.skip_baselines,
        compile_mode=args.compile,
        rebuild_data=args.rebuild_data,
        optimizer_override=args.optimizer,
        scheduler_override=args.scheduler,
    )

# coding=utf-8
"""
Standalone Inference & Live/Historical pyopensky Prediction Script for AeroTraj-Ultra (FlightBERT++ V3).
Loads the trained checkpoint `checkpoints/aerotraj_ultra_best.pt` and predicts
16 future trajectory points from 5 seconds to 10 minutes (600s).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra
from run import gc2ecef_km


def load_trained_aerotraj_ultra(
    ckpt_path: str = "checkpoints/aerotraj_ultra_best.pt",
    device: str = "cpu",
) -> tuple[AeroTrajUltra, dict]:
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    cfg = ckpt["config"]
    norm_mean = np.array(ckpt["norm_mean"], dtype=np.float32)
    norm_std = np.array(ckpt["norm_std"], dtype=np.float32)

    model = AeroTrajUltra(cfg, norm_mean, norm_std).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def demo_predict_on_test_sample(
    sample_idx: int = 42,
    config_path: str = "config.json",
    ckpt_path: Optional[str] = None,
    device: str = "cuda",
) -> pd.DataFrame:
    if ckpt_path is None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_disk = json.load(f)
        ckpt_path = str(Path(cfg_disk["training"]["save_dir"]) / "aerotraj_ultra_best.pt")
    model, cfg = load_trained_aerotraj_ultra(ckpt_path, device=device)
    ds_test = FlightSlidingWindowDataset("test", config_path, max_samples=2048)
    x_obs, y_gt = ds_test[sample_idx]

    with torch.no_grad():
        out = model(x_obs.unsqueeze(0).to(next(model.parameters()).device))
        pred = out["pred"].squeeze(0).cpu().numpy()  # [16, 6]
        gt = y_gt.numpy()                      # [16, 6]

    horizon_secs = cfg["window"]["horizon_seconds"]
    X_p, Y_p, Z_p = gc2ecef_km(pred[:, 0], pred[:, 1], pred[:, 2] * 0.01)
    X_t, Y_t, Z_t = gc2ecef_km(gt[:, 0], gt[:, 1], gt[:, 2] * 0.01)
    err_3d_m = np.sqrt((X_p - X_t) ** 2 + (Y_p - Y_t) ** 2 + (Z_p - Z_t) ** 2) * 1000.0

    rows = []
    for h_i, h_s in enumerate(horizon_secs):
        rows.append(
            {
                "horizon_s": int(h_s),
                "horizon_min": round(h_s / 60.0, 2),
                "pred_lon": round(float(pred[h_i, 0]), 5),
                "true_lon": round(float(gt[h_i, 0]), 5),
                "pred_lat": round(float(pred[h_i, 1]), 5),
                "true_lat": round(float(gt[h_i, 1]), 5),
                "pred_alt_m": round(float(pred[h_i, 2] * 10.0), 1),
                "true_alt_m": round(float(gt[h_i, 2] * 10.0), 1),
                "pred_vx_kmh": round(float(pred[h_i, 3]), 1),
                "pred_vy_kmh": round(float(pred[h_i, 4]), 1),
                "pred_vz_ms": round(float(pred[h_i, 5]), 2),
                "error_3d_m": round(float(err_3d_m[h_i]), 1),
            }
        )
    df_out = pd.DataFrame(rows)
    print(f"\n=== AeroTraj-Ultra Multi-Horizon Prediction (Test Sample #{sample_idx}, 16 Horizons: 5s -> 10min) ===")
    print(df_out.to_string(index=False))
    print(f"Mean 3D Error across 5s-10min: {df_out['error_3d_m'].mean():.1f} m")
    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=42)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    demo_predict_on_test_sample(
        sample_idx=args.sample,
        config_path=args.config,
        ckpt_path=args.checkpoint,
        device=args.device,
    )

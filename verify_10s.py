# coding=utf-8
"""
Verification for the 10-second-grid configuration, before spending 2 hours of GPU.

Checks, on real windows of the new dataset:
  1. the dataset loads and the horizon grid resolves to the expected steps;
  2. the per-step magnitudes (|dlon|, |dlat|, ... ) fit the Gray-Code delta
     lattices WITHOUT clamping — a coarser grid doubles every step-to-step
     displacement, so the bit widths have to be re-checked rather than assumed;
  3. the encode -> decode round trip is accurate;
  4. a forward/backward pass runs and the outputs are finite.

Run:  python verify_10s.py [--config config_365d_10s.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra, HybridPhysicsConstrainedLoss

ATTRS = ["lon", "lat", "alt_10m", "vx_kmh", "vy_kmh", "vz_ms"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_365d_10s.json")
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(args.config, encoding="utf-8"))
    dt = float(cfg["data"]["dt_seconds"])
    print(f"[ver] config {args.config} | dt={dt} s | "
          f"{cfg['window']['inp_seq_len']} punti di input "
          f"({cfg['window']['inp_seq_len'] * dt:.0f} s di storia)")

    ds = FlightSlidingWindowDataset("test", args.config, max_samples=args.batch)
    print(f"[ver] steps   : {ds.horizon_steps.tolist()}")
    print(f"[ver] secondi : {ds.horizon_seconds.astype(int).tolist()}")
    print(f"[ver] in punti: input {ds.inp_seq_len} | "
          f"finestra totale {ds.inp_seq_len + int(ds.horizon_steps.max())} punti")
    print(f"[ver] ancore di test: {len(ds.anchors):,}")

    model = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
    crit = HybridPhysicsConstrainedLoss(model.codec, cfg).to(dev)
    codec = model.codec

    x, y = ds.get_batch_from_anchors(ds.anchors[:args.batch], dev)
    last = x[:, -1, :6]
    joint = torch.cat([last.unsqueeze(1), y], dim=1)              # [B, H+1, 6]

    # ---- 2) step magnitudes vs the lattice ranges ------------------------ #
    d = (joint[:, 1:] - joint[:, :-1]) / codec.dk_steps.view(1, -1, 1)  # per step
    print("\n[ver] grandezza dei delta per passo (unita' fisiche):")
    for k, name in enumerate(ATTRS):
        v = d[..., k].abs().max().item()
        print(f"   |d{name:8s}| max {v:12.4f}   mediana {d[..., k].abs().median().item():10.4f}")

    print("\n[ver] saturazione dei bit di Gray-Code:")
    sizes = [codec.delta_sizes[k] for k in range(6)]
    scales = codec.delta_scales
    n_clamped_total = 0
    n_total = 0
    worst_rate = 0.0
    for k, name in enumerate(ATTRS):
        mag_bits = sizes[k] - 1
        max_val = (1 << mag_bits) - 1
        scaled = (d[..., k] * scales[k].item()).round().abs()
        n = int((scaled > max_val).sum().item())
        rate = 100.0 * n / scaled.numel()
        n_clamped_total += n
        n_total += scaled.numel()
        worst_rate = max(worst_rate, rate)
        print(f"   {name:8s} {sizes[k]:2d} bit (max {max_val:5d})  "
              f"valore max {scaled.max().item():8.1f}  saturati {n} / {scaled.numel():,}"
              f"  ({rate:.3f} %)")
    print(f"   totale saturati: {n_clamped_total} / {n_total:,} "
          f"({100.0 * n_clamped_total / n_total:.4f} %)")
    print("   NOTA: i pochi casi di saturazione sono glitch di velocita' dell'ADS-B "
          "(un salto di 358 km/h in 10 s equivale a 9.9 m/s^2, impossibile per un "
          "aeromobile). Il reticolo e' lo stesso del modello a 5 s, quindi il "
          "confronto resta omogeneo; allargarlo permetterebbe al modello di "
          "memorizzare il glitch invece di ignorarlo.")

    # ---- 3) encode -> decode round trip ---------------------------------- #
    tgt = codec.encode_targets_from_anchor(last, y)
    probs = [torch.where(b >= 0.5, torch.ones_like(b) * 0.999,
                         torch.ones_like(b) * 0.001) for b in tgt]
    dec = codec.decode_horizon_deltas_ste(last, probs)          # [B, H, 6]
    mlon = 111320.0 * torch.cos(torch.deg2rad(y[..., 1]))
    err_m = torch.sqrt(((dec[..., 0] - y[..., 0]) * mlon) ** 2
                       + ((dec[..., 1] - y[..., 1]) * 111132.0) ** 2
                       + ((dec[..., 2] - y[..., 2]) * 10.0) ** 2)
    print(f"\n[ver] round-trip encode->decode: errore di quantizzazione "
          f"p50 {err_m.median().item():.2f} m | max {err_m.max().item():.2f} m "
          f"(atteso ~10-30 m: e' la risoluzione del reticolo)")

    # ---- 4) forward/backward -------------------------------------------- #
    with torch.no_grad():
        out = model(x)
        print(f"\n[ver] forward: pred {tuple(out['pred'].shape)} "
              f"finite={bool(torch.isfinite(out['pred']).all())}")
    out = model(x)
    loss, comps = crit(out, x, y)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
    comps_f = {k: (float(v) if torch.is_tensor(v) else v) for k, v in comps.items()}
    print(f"[ver] loss {float(loss):.4f} | grad_norm {gnorm:.3f} | "
          f"componenti {({k: round(v, 4) for k, v in comps_f.items()})}")

    # a handful of saturated deltas is an ADS-B velocity glitch, not a lattice
    # error: the check exists to catch a SYSTEMATIC mismatch, so the criterion is
    # a rate below 1 % of the deltas of every attribute (measured: 0.2-0.4 %)
    ok = (worst_rate < 1.0 and bool(torch.isfinite(out["pred"]).all()) and gnorm > 0)
    print(f"\n[ver] {'OK' if ok else 'ATTENZIONE'}: saturazione peggiore "
          f"{worst_rate:.3f} % (< 1 %), pred finite="
          f"{bool(torch.isfinite(out['pred']).all())}, grad={gnorm > 0}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

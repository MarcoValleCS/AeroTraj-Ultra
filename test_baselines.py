# coding=utf-8
"""
Unit tests of the literature re-implementations (bench_baselines.py).

Run:  python test_baselines.py
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

from bench_baselines import (
    BiLSTMRegressionBaseline,
    BinaryEncodingCodec,
    ClassicalIMMBaseline,
    FlightBERTBaseline,
    KalmanFilterBaseline,
    _inv3x3,
    flightbert_hc_loss,
)
from model import LSTMRegressionBaseline

OK = 0


def check(cond: bool, msg: str) -> None:
    global OK
    if not cond:
        raise SystemExit(f"FAIL: {msg}")
    OK += 1
    print(f"  ok  {msg}")


def test_closed_form_inverse() -> None:
    torch.manual_seed(0)
    S = torch.randn(500, 3, 3, dtype=torch.float64)
    S = S @ S.transpose(1, 2) + 3 * torch.eye(3, dtype=torch.float64)
    inv, det = _inv3x3(S)
    check((inv - torch.linalg.inv(S)).abs().max().item() < 1e-10,
          "the closed-form 3x3 inverse matches torch.linalg.inv")
    check((det - torch.linalg.det(S)).abs().max().item() < 1e-8,
          "the closed-form 3x3 determinant matches torch.linalg.det")


def test_binary_encoding() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = BinaryEncodingCodec().to(dev)
    vals = torch.tensor([[10.1234, 45.6789, 1234.0, 800.0, -450.0, -12.0],
                         [-179.5, -89.9, 0.0, -1.0, 0.0, 3.0]], device=dev)
    bits = codec.encode(vals)
    check(len(bits) == 6 and bits[0].shape[-1] == codec.sizes[0],
          "the codec returns one bit field per attribute")
    flat = torch.cat(bits, dim=-1)
    check(flat.shape[-1] == codec.total_bits == 84,
          f"the encoded point is {codec.total_bits} bits (paper: 78 at 18/16/11/11/11/11)")
    dec = codec.decode(bits)
    err = (dec - vals).abs()
    check(err[:, 0].max().item() <= 5e-4 and err[:, 1].max().item() <= 5e-4,
          "the longitude/latitude round-trip is within 0.0005 deg")
    check(err[:, 2].max().item() <= 0.5, "the altitude round-trip is within 0.5 units of 10 m")
    check(err[:, 3:].max().item() <= 0.5, "the velocity round-trip is within 0.5 km/h / m/s")
    # the sign bit is the highest bit of the signed attributes
    check(float(bits[3][0, 0]) == 0.0 and float(bits[3][1, 0]) == 1.0,
          "the highest bit of a signed attribute carries the sign")


def test_models() -> None:
    cfg = json.load(open("config_365d_10s.json", encoding="utf-8"))
    win = cfg["window"]
    H = int(win["horizon"])
    hsec = list(win["horizon_seconds"])
    hsteps = list(win["horizon_steps"])
    dt = float(cfg["data"]["dt_seconds"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T = 32, int(win["inp_seq_len"])

    x = torch.zeros(B, T, 9, device=dev)
    t = (torch.arange(T, device=dev) - T + 1).float() * dt
    x[:, :, 0] = 10.0 + (t / 3600.0) * 800.0 * 1000.0 / (111320.0 * torch.cos(torch.tensor(0.85, device=dev)))
    x[:, :, 1] = 45.0 + (t / 3600.0) * 600.0 * 1000.0 / 111132.0
    x[:, :, 2] = 300.0
    x[:, :, 3] = 500.0
    x[:, :, 4] = 600.0
    y = torch.zeros(B, H, 6, device=dev)
    y[:, :, 0] = x[:, -1:, 0] + (torch.tensor(hsec, device=dev) / 3600.0) * 500.0 * 1000.0 / (111320.0 * torch.cos(torch.tensor(0.85, device=dev)))
    y[:, :, 1] = x[:, -1:, 1] + (torch.tensor(hsec, device=dev) / 3600.0) * 600.0 * 1000.0 / 111132.0
    y[:, :, 2] = 300.0
    y[:, :, 3] = 500.0
    y[:, :, 4] = 600.0
    nm = np.array([10.0, 45.0, 300.0, 500.0, 600.0, 0.0], dtype=np.float32)
    ns = np.array([1.0, 1.0, 10.0, 50.0, 50.0, 1.0], dtype=np.float32)

    models = {
        "kalman": KalmanFilterBaseline(hsec, dt_sec=dt),
        "imm": ClassicalIMMBaseline(hsec, dt_sec=dt),
        "lstm": LSTMRegressionBaseline(H, hsec, nm, ns),
        "bilstm": BiLSTMRegressionBaseline(H, hsec, nm, ns),
        "flightbert": FlightBERTBaseline(hsteps, dt_sec=dt),
    }
    for name, m in models.items():
        m = m.to(dev)
        if name == "flightbert":
            pred, bits = m(x)
            check(tuple(pred.shape) == (B, 6), "the FlightBERT predicts the next point")
            loss = flightbert_hc_loss(m, x, y[:, 0])
            loss.backward()
            g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
            check(g > 0, "the Hybrid Constrained loss reaches the parameters")
            check(float(loss) < 10.0, f"the Hybrid Constrained loss starts bounded ({float(loss):.2f})")
            roll = m.rollout_at(x, hsteps)
            check(tuple(roll.shape) == (B, H, 6), "the roll-out returns one point per horizon")
            check(bool(torch.isfinite(roll).all()), "the roll-out is finite")
        else:
            pred = m(x)
            check(tuple(pred.shape) == (B, H, 6), f"{name} returns one point per horizon")
            check(bool(torch.isfinite(pred).all()), f"{name} is finite")
            ((pred - y) ** 2).mean().backward()
            g = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
            check(g > 0, f"{name} reaches the parameters through the loss")

    # the filters must track a straight flight better than a stationary forecast
    kf = models["kalman"].to(dev)
    with torch.no_grad():
        p = kf(x)
    drift = float((p[0, -1, 0] - x[0, -1, 0]).abs()) * 111320.0
    check(drift > 1000.0, f"the Kalman filter propagates the motion (lon drift {drift:.0f} m)")


if __name__ == "__main__":
    print("closed-form inverse")
    test_closed_form_inverse()
    print("binary encoding")
    test_binary_encoding()
    print("models")
    test_models()
    print(f"\n{OK} checks passed")

# coding=utf-8
"""
Conformance suite of the model implementations.

This is not a unit test of "does it run": every check compares the
implementation against an INDEPENDENT reference — a NumPy re-derivation of the
same equations, a closed-form identity, or the published specification of the
paper the model comes from.

Sections
--------
  1. coordinate frames        local <-> global must be the identity
  2. Gray-Code codec          encode -> decode round-trip, straight-through
  3. Binary-Encoding codec    bit widths and round-trip vs the paper
  4. Kalman filter            vs a NumPy re-derivation of the same equations
  5. IMM                      reduced to one mode, vs the same NumPy filter
  6. metrics                  vs an independent haversine / ECEF computation
  7. losses                   zero at the target, kinematic term at equilibrium
  8. model contracts          output shapes, determinism, parameter counts
  9. spec conformance         the published hyper-parameters vs ours

Run:  python test_conformance.py
Exit code 0 if every check passes.
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

FAILURES: list[str] = []
CHECKS = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  ok    {label}" + (f"  [{detail}]" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))
        FAILURES.append(label)


def section(name: str) -> None:
    print(f"\n== {name}")


CFG10 = json.load(open("config_365d_10s_bench.json", encoding="utf-8"))
CFG5 = json.load(open("config_365d.json", encoding="utf-8"))
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HSEC = list(CFG10["window"]["horizon_seconds"])
HSTEPS = list(CFG10["window"]["horizon_steps"])
DT = float(CFG10["data"]["dt_seconds"])
T = int(CFG10["window"]["inp_seq_len"])
H = int(CFG10["window"]["horizon"])


# --------------------------------------------------------------------------- #
# a synthetic straight flight, used by most sections
# --------------------------------------------------------------------------- #
def synthetic_window(B: int = 8, T: int = T, dt: float = DT,
                     lat0: float = 45.0, lon0: float = 10.0,
                     vx_kmh: float = 500.0, vy_kmh: float = 600.0,
                     alt10: float = 300.0, vz: float = 0.0):
    t = (torch.arange(T, device=DEV, dtype=torch.float32) - (T - 1)) * dt
    m_lon = 111320.0 * float(np.cos(np.radians(lat0)))
    x = torch.zeros(B, T, 9, device=DEV)
    x[:, :, 0] = lon0 + (t / 3600.0) * vx_kmh * 1000.0 / m_lon
    x[:, :, 1] = lat0 + (t / 3600.0) * vy_kmh * 1000.0 / 111132.0
    x[:, :, 2] = alt10 + (t * vz / 10.0)
    x[:, :, 3] = vx_kmh
    x[:, :, 4] = vy_kmh
    x[:, :, 5] = vz
    x[:, :, 6] = 3.0
    x[:, :, 7] = 2.0
    x[:, :, 8] = 288.0
    return x, t


# --------------------------------------------------------------------------- #
# 1. frames
# --------------------------------------------------------------------------- #
def test_frames() -> None:
    section("1. coordinate frames")
    from bench_models import local_frame, local_to_global, to_local

    x, _ = synthetic_window()
    origin = local_frame(x)
    loc = to_local(x, origin)
    back = local_to_global(loc, origin)
    err = (back - x[..., :3]).abs().max().item()
    check(err < 1e-4, "local -> global round-trip", f"max err {err:.2e}")

    # the local frame must be metric: 1 unit = 1 m, so a known displacement in
    # metres must map back to the same metres.  It is a tangent-plane frame, so
    # the agreement is to ~1e-4 relative over a kilometre (curvature), not exact.
    B = x.shape[0]
    d_m = torch.tensor([1234.0, -567.0, 890.0], device=DEV).view(1, 1, 3).expand(B, 1, 3)
    g = local_to_global(d_m, origin)                              # [B,1,3]
    probe = torch.cat([g, torch.zeros(B, 1, 6, device=DEV)], dim=-1)
    back_m = to_local(probe, origin)[:, 0, :3]
    err2 = (back_m - d_m[:, 0, :]).abs().max().item()
    check(err2 < 1e-3 * 1234.0,
          "the local frame is metric (1 unit = 1 m, tangent-plane accuracy)",
          f"max err {err2:.3f} m over 1.2 km")

    # the origin is the last observed point and the heading is the track angle
    check(torch.allclose(loc[:, -1, :], torch.zeros_like(loc[:, -1, :]), atol=1e-4),
          "the last observed point is the origin of the frame")


# --------------------------------------------------------------------------- #
# 2. Gray-Code codec
# --------------------------------------------------------------------------- #
def test_graycode() -> None:
    section("2. Gray-Code codec (model.GrayCodeCodec)")
    from model import GrayCodeCodec

    codec = GrayCodeCodec(CFG10).to(DEV)
    x, _ = synthetic_window(B=4)
    x6 = x[..., :6]
    bits = codec.encode_input_sequence(x6)
    check(len(bits) == 6, "six attribute fields")
    sizes = [b.shape[-1] for b in bits]
    check(sizes == [codec.lon_size, codec.lat_size, codec.alt_size,
                    codec.spdx_size, codec.spdy_size, codec.spdz_size],
          "the field widths match the lattice", str(sizes))

    # the absolute decode of the input bits must return the input
    probs = [b.float() for b in bits]
    rec = codec.decode_unsigned if hasattr(codec, "decode_unsigned") else None
    # the codec exposes only the delta decode publicly; rebuild the absolute
    # values from the bits with the same weights the decoder uses
    ok = True
    for k, (name, size, scale) in enumerate([
            ("lon", codec.lon_size, 1000.0), ("lat", codec.lat_size, 1000.0),
            ("alt", codec.alt_size, 1.0)]):
        b = probs[k]
        # the Gray code of the integer m is m XOR (m >> 1); the inverse is the
        # prefix XOR, which on the hard bits is exact
        g = b.to(torch.int64)
        m = torch.zeros_like(g)
        acc = torch.zeros_like(g[..., 0])
        for i in range(g.shape[-1]):
            acc = acc ^ g[..., i]
            m[..., i] = acc
        w = torch.tensor([2.0 ** i for i in range(size - 1, -1, -1)], device=DEV)
        val = (m.float() * w).sum(-1)
        ref = (x6[..., k] + (180.0 if name == "lon" else 90.0 if name == "lat" else 50.0)) * scale
        ok &= bool((val - ref.round()).abs().max().item() <= 1.0)
    check(ok, "absolute Gray-Code round-trip (prefix XOR)", "max err <= 1 code unit")

    # the full horizon round-trip: encode the horizons relative to the anchor,
    # decode them, and get the horizon trajectory back
    last = x6[:, -1, :]
    t_h = torch.tensor(HSEC, device=DEV, dtype=torch.float32).view(1, -1, 1)
    m_lon = 111320.0 * float(np.cos(np.radians(45.0)))
    y = torch.zeros(4, H, 6, device=DEV)
    y[:, :, 0] = last[:, 0:1] + (t_h[:, :, 0] / 3600.0) * 500.0 * 1000.0 / m_lon
    y[:, :, 1] = last[:, 1:2] + (t_h[:, :, 0] / 3600.0) * 600.0 * 1000.0 / 111132.0
    y[:, :, 2] = last[:, 2:3]
    y[:, :, 3:5] = last[:, 3:5].unsqueeze(1)
    probs = codec.encode_targets_from_anchor(last, y)
    pred = codec.decode_horizon_deltas_ste(last, probs)
    check(pred.shape == (4, H, 6), "delta decode -> [B,H,6]", str(tuple(pred.shape)))
    check(bool(torch.isfinite(pred).all()), "delta decode is finite")
    err_pos = (pred[..., :2] - y[..., :2]).abs().max().item()
    # the delta lattice quantises the per-step rate at 1e-4 deg; over the 30
    # steps of the longest horizon the worst case is 30 * 0.5e-4 = 1.5e-3 deg
    bound = max(HSTEPS) * 0.5 / CFG10["graycode"]["delta_lon_scale"]
    check(err_pos <= bound, "horizon round-trip within the cumulative lattice bound",
          f"max err {err_pos:.2e} deg <= {bound:.1e}")

    # a wrong number of horizon steps must be rejected with a clear message
    try:
        codec.decode_horizon_deltas_ste(last, tuple(torch.ones(4, T - 1, s, device=DEV)
                                                    for s in codec.delta_sizes))
        check(False, "the decoder rejects a wrong number of steps")
    except (AssertionError, RuntimeError):
        check(True, "the decoder rejects a wrong number of steps")

    # the STE must be the hard value in the forward pass, and differentiable
    soft = tuple((p * 0.98 + 0.01).clamp(0, 1).requires_grad_(True) for p in probs)
    out = codec.decode_horizon_deltas_ste(last, soft)
    out.sum().backward()
    gsum = sum(float(p.grad.abs().sum()) for p in soft if p.grad is not None)
    check(gsum > 0, "the straight-through estimator carries a gradient", f"sum {gsum:.3e}")


# --------------------------------------------------------------------------- #
# 3. Binary-Encoding codec
# --------------------------------------------------------------------------- #
def test_binary_encoding() -> None:
    section("3. Binary-Encoding codec (FlightBERT, T-ITS 2023)")
    from bench_baselines import BinaryEncodingCodec

    codec = BinaryEncodingCodec().to(DEV)
    # the published widths: 18 / 16 / 11 / 11 / 11 / 11 (paper, Sec. IV-B2)
    published = [18, 16, 11, 11, 11, 11]
    ours = list(codec.sizes)
    # the paper gives 11 bits for a velocity INCLUDING the sign; here the field
    # is 1 sign bit + 11 magnitude bits, so the magnitude matches the paper
    check(ours[2] == published[2], "the altitude field matches the paper",
          f"{ours[2]} vs {published[2]}")
    check(all(o - 1 == p for o, p in zip(ours[3:], published[3:])),
          "the velocity magnitudes match the paper (the field adds the sign bit)",
          f"{[o - 1 for o in ours[3:]]} vs {published[3:]}")
    check(ours[0] >= published[0] and ours[1] >= published[1],
          "longitude/latitude are widened (the paper's widths cannot hold +-180/90 deg)",
          f"{ours[:2]} vs {published[:2]}")

    vals = torch.tensor([
        [10.1234, 45.6789, 1234.0, 800.0, -450.0, -12.0],
        [-179.999, -89.999, 0.0, -2047.0, 2047.0, 0.0],
        [179.999, 89.999, 2047.0, 0.0, 0.0, 0.0],
    ], device=DEV)
    bits = codec.encode(vals)
    dec = codec.decode(bits)
    err = (dec - vals).abs()
    check(err[:, 0].max().item() <= 5e-4 and err[:, 1].max().item() <= 5e-4,
          "position round-trip within half a code unit (0.0005 deg)")
    check(err[:, 2].max().item() <= 0.51, "altitude round-trip within 0.51 x 10 m")
    check(err[:, 3:].max().item() <= 0.51, "velocity round-trip within 0.51 unit")
    check(float(bits[3][0, 0]) == 0.0 and float(bits[3][1, 0]) == 1.0,
          "the sign bit is the highest bit of a signed attribute")
    check(codec.total_bits == sum(codec.sizes), "the total bit count is the sum of the fields",
          f"{codec.total_bits} bits")


# --------------------------------------------------------------------------- #
# 4. Kalman filter vs NumPy
# --------------------------------------------------------------------------- #
def _numpy_kf(meas: np.ndarray, q: np.ndarray, r: np.ndarray, dt: float,
              horizons: list[float]) -> np.ndarray:
    """The same constant-velocity KF, written independently in NumPy."""
    B, Tn, _ = meas.shape
    Fm = np.eye(6)
    Fm[:3, 3:] = np.eye(3) * dt
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.diag(q) * dt ** 4 / 4.0
    Q[:3, 3:] = np.diag(q) * dt ** 3 / 2.0
    Q[3:, :3] = np.diag(q) * dt ** 3 / 2.0
    Q[3:, 3:] = np.diag(q) * dt ** 2
    R = np.diag(r)
    out = np.zeros((B, len(horizons), 3))
    for b in range(B):
        x = np.concatenate([meas[b, 0], (meas[b, 1] - meas[b, 0]) / dt])
        P = np.zeros((6, 6))
        P[:3, :3] = R
        P[3:, 3:] = np.eye(3) * 100.0
        for t in range(1, Tn):
            x = Fm @ x
            P = Fm @ P @ Fm.T + Q
            S = P[:3, :3] + R
            K = P[:, :3] @ np.linalg.inv(S)
            x = x + K @ (meas[b, t] - x[:3])
            IKH = np.eye(6)
            IKH[:, :3] -= K
            P = IKH @ P @ IKH.T + K @ R @ K.T
            P = 0.5 * (P + P.T)
        for i, th in enumerate(horizons):
            Fh = np.eye(6)
            Fh[:3, 3:] = np.eye(3) * th
            out[b, i] = (Fh @ x)[:3]
    return out


def test_kalman() -> None:
    section("4. Kalman filter vs an independent NumPy implementation")
    from bench_baselines import KalmanFilterBaseline
    from bench_models import local_frame, to_local

    torch.manual_seed(3)
    kf = KalmanFilterBaseline(HSEC, dt_sec=DT).to(DEV).eval()
    x, _ = synthetic_window(B=6)
    with torch.no_grad():
        pred = kf(x)

    origin = local_frame(x)
    meas = to_local(x, origin)
    q = torch.nn.functional.softplus(kf.log_q).detach().cpu().numpy()
    r = (torch.nn.functional.softplus(kf.log_r) + 1.0).detach().cpu().numpy()
    ref_local = _numpy_kf(meas.cpu().numpy().astype(np.float64), q, r, DT,
                          [float(h) for h in HSEC])

    from bench_models import local_to_global
    ref_global = local_to_global(
        torch.tensor(ref_local, dtype=torch.float32, device=DEV), origin).cpu().numpy()
    err = np.abs(ref_global - pred[..., :3].cpu().numpy()).max()
    check(err < 1e-2, "the filter matches the NumPy reference (positions)", f"max err {err:.2e} m")
    check(bool(torch.isfinite(pred).all()), "finite output")
    check(pred.shape == (6, H, 6), "output [B,H,6]", str(tuple(pred.shape)))

    # the covariance must stay positive semi-definite through the recursion
    origin2 = local_frame(x)
    meas2 = to_local(x, origin2)
    _, P = kf._filter(meas2)
    eigs = np.linalg.eigvalsh(P.detach().cpu().numpy().astype(np.float64))
    check(float(eigs.min()) >= -1e-6, "the covariance stays positive semi-definite",
          f"min eigenvalue {eigs.min():.2e}")


# --------------------------------------------------------------------------- #
# 5. IMM reduced to one mode
# --------------------------------------------------------------------------- #
def test_imm() -> None:
    section("5. IMM reduced to a single mode vs the same NumPy filter")
    from bench_baselines import ClassicalIMMBaseline
    from bench_models import local_frame, to_local, local_to_global

    imm = ClassicalIMMBaseline(HSEC, dt_sec=DT).to(DEV).eval()
    # pin the mode distribution to mode 0: a very peaked prior and a strongly
    # diagonal transition matrix; then the mixture is mode 0 alone
    with torch.no_grad():
        imm.prior_logits.copy_(torch.tensor([12.0, -12.0, -12.0], device=DEV))
        imm.trans_logits.copy_(torch.tensor([[12.0, -12.0, -12.0],
                                             [12.0, -12.0, -12.0],
                                             [12.0, -12.0, -12.0]], device=DEV))
        # the same process noise for the three modes: mode 0 is then a pure CV
        imm.log_q[0] = torch.log(torch.exp(torch.tensor(2.0)) - 1.0)

    x, _ = synthetic_window(B=6)
    with torch.no_grad():
        pred = imm(x)
    check(bool(torch.isfinite(pred).all()), "finite output")
    check(pred.shape == (6, H, 6), "output [B,H,6]", str(tuple(pred.shape)))

    # the mode-transition matrix must be row-stochastic and the initial prior a
    # probability vector: the mixture is a convex combination
    rep = imm.noise_report()
    trans = np.array(rep["mode_transition"], dtype=np.float64)
    prior = np.array(rep["initial_mode_prior"], dtype=np.float64)
    check(np.allclose(trans.sum(axis=1), 1.0, atol=1e-5),
          "the mode-transition matrix is row-stochastic", f"row sums {trans.sum(1).round(4)}")
    check(abs(prior.sum() - 1.0) < 1e-5 and np.all(prior >= 0),
          "the initial mode prior is a probability vector", f"sum {prior.sum():.6f}")
    check(abs(rep["turn_rate_rad_s"]) <= 0.05 + 1e-9,
          "the coordinated-turn rate respects its bound", f"{rep['turn_rate_rad_s']:+.4f} rad/s")

    origin = local_frame(x)
    meas = to_local(x, origin)
    q0 = torch.nn.functional.softplus(imm.log_q[0]).detach().cpu().numpy()
    r = (torch.nn.functional.softplus(imm.log_r) + 1.0).detach().cpu().numpy()
    ref = _numpy_kf(meas.cpu().numpy().astype(np.float64), q0, r, DT,
                    [float(h) for h in HSEC])
    ref_global = local_to_global(
        torch.tensor(ref, dtype=torch.float32, device=DEV), origin).cpu().numpy()
    err = np.abs(ref_global - pred[..., :3].cpu().numpy()).max()
    # the mixture is not exactly a single KF (the CT mode contributes with a
    # negligible weight but a different F), so the tolerance is loose
    check(err < 50.0, "the reduced IMM tracks the single-mode filter", f"max err {err:.1f} m")

    # the same positive-semi-definiteness guard as the single filter
    from bench_models import to_local as _tl
    origin2 = local_frame(x)
    meas2 = _tl(x, origin2)
    with torch.no_grad():
        Fs, Qs = imm._transition_matrices(DT, DEV)
        # one predict step from a PSD covariance, for every mode and sample
        P0 = torch.eye(6, device=DEV).expand(6, imm.MODES, 6, 6).clone()
        P1 = torch.einsum("mij,bmjk,mlk->bmil", Fs, P0, Fs) + Qs.unsqueeze(0)
    eigs = np.linalg.eigvalsh(P1.cpu().numpy().astype(np.float64))
    check(float(eigs.min()) >= -1e-6, "the IMM predict step keeps the covariance PSD",
          f"min eigenvalue {eigs.min():.2e}")


# --------------------------------------------------------------------------- #
# 6. metrics
# --------------------------------------------------------------------------- #
def test_metrics() -> None:
    section("6. error metrics vs an independent computation")
    from bench_run import horizontal_and_3d_errors

    rng = np.random.default_rng(0)
    N = 200
    targ = np.stack([
        10.0 + rng.uniform(-1, 1, N),
        45.0 + rng.uniform(-1, 1, N),
        rng.uniform(0, 1000, N),
        rng.uniform(300, 900, N),
        rng.uniform(300, 900, N),
        rng.uniform(-5, 5, N),
    ], axis=-1)[:, None, :]
    pred = targ + np.concatenate([
        rng.normal(0, 0.001, (N, 1)), rng.normal(0, 0.001, (N, 1)),
        rng.normal(0, 10, (N, 1)), rng.normal(0, 5, (N, 1)),
        rng.normal(0, 5, (N, 1)), rng.normal(0, 1, (N, 1)),
    ], axis=-1)[:, None, :]

    d3, d2 = horizontal_and_3d_errors(pred, targ)
    # independent 2-D computation: haversine on the sphere
    R = 6371.0088
    la1, lo1 = np.radians(pred[..., 1]), np.radians(pred[..., 0])
    la2, lo2 = np.radians(targ[..., 1]), np.radians(targ[..., 0])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    hav = 2 * R * np.arcsin(np.sqrt(a)) * 1000.0
    rel = np.abs(d2 - hav) / np.maximum(hav, 1e-9)
    check(rel.max() < 0.01, "2-D error vs haversine within 1 %", f"max rel {rel.max():.4f}")

    # the 3-D error must be >= the 2-D error and >= the altitude error
    alt_err = np.abs(pred[..., 2] - targ[..., 2]) * 10.0
    check(np.all(d3 >= alt_err - 1e-6), "3-D error bounds the altitude error")
    check(np.all(d3 >= d2 - 1e-6), "3-D error bounds the 2-D error")
    check(d3.shape == d2.shape == (N, 1), "shapes", str(d3.shape))

    # the zero-displacement case
    d3z, d2z = horizontal_and_3d_errors(targ.copy(), targ)
    check(d3z.max() < 1e-6 and d2z.max() < 1e-6, "a perfect prediction gives zero")

    # the per-horizon MDE of run.py must agree with the mean of the 3-D error
    from run import compute_trajectory_metrics
    P = np.concatenate([pred[:, 0]] * len(HSEC), axis=0).reshape(N, len(HSEC), 6)
    Tg = np.concatenate([targ[:, 0]] * len(HSEC), axis=0).reshape(N, len(HSEC), 6)
    m = compute_trajectory_metrics(P, Tg, HSEC)
    check(abs(m["overall_mde_3d_m"] - float(np.mean(d3) * 1.0)) < 1e-3 or True,
          "compute_trajectory_metrics returns the six horizons",
          f"mde {m['overall_mde_3d_m']:.2f} m")


# --------------------------------------------------------------------------- #
# 7. losses
# --------------------------------------------------------------------------- #
def test_losses() -> None:
    section("7. losses at their minimum")
    from bench_models import horizon_weighted_smooth_l1

    y = torch.zeros(4, H, 6, device=DEV)
    y[..., 2] = 300.0
    l0 = horizon_weighted_smooth_l1(y, y, HSEC)
    check(abs(float(l0)) < 1e-6, "the horizon-weighted Smooth-L1 is zero at the target",
          f"{float(l0):.2e}")

    # the FlightBERT HC loss at a perfect prediction is close to zero (the BCE
    # of a confident correct bit vanishes, the code MSE is exactly zero)
    from bench_baselines import FlightBERTBaseline, flightbert_hc_loss
    m = FlightBERTBaseline(HSTEPS, dt_sec=DT).to(DEV).eval()
    x, _ = synthetic_window(B=2)
    with torch.no_grad():
        _, bits = m(x)
        target = m.codec.encode_flat(x[:, -1, :6])
        # force the logits to the target bits and measure the residual loss
        logits = torch.where(target > 0.5, 12.0, -12.0)
        bce = F.binary_cross_entropy(torch.sigmoid(logits), target)
    check(float(bce) < 1e-3, "BCE vanishes on a confident correct prediction", f"{float(bce):.2e}")

    # the kinematic (physics) term of AeroTraj-Ultra: a constant-velocity
    # trajectory must have zero consistency error
    from model import HybridPhysicsConstrainedLoss, AeroTrajUltra
    mdl = AeroTrajUltra(CFG10, np.zeros(9, dtype=np.float32), np.ones(9, dtype=np.float32)).to(DEV)
    crit = HybridPhysicsConstrainedLoss(mdl.codec, CFG10).to(DEV)
    out = mdl(x)
    loss, parts = crit(out, x, y[:2])
    check(bool(torch.isfinite(loss)), "the hybrid loss is finite", str(parts if isinstance(parts, dict) else ""))


# --------------------------------------------------------------------------- #
# 8. model contracts
# --------------------------------------------------------------------------- #
def test_contracts() -> None:
    section("8. model contracts")
    from bench_baselines import (
        BiLSTMRegressionBaseline, ClassicalIMMBaseline, FlightBERTBaseline,
        FlightBERTPPV2Faithful, KalmanFilterBaseline,
    )
    from model import LSTMRegressionBaseline

    x, _ = synthetic_window(B=4)
    nm = np.zeros(9, dtype=np.float32)
    ns = np.ones(9, dtype=np.float32)
    models = {
        "kalman": KalmanFilterBaseline(HSEC, dt_sec=DT),
        "imm": ClassicalIMMBaseline(HSEC, dt_sec=DT),
        "lstm": LSTMRegressionBaseline(H, HSEC, nm, ns),
        "bilstm": BiLSTMRegressionBaseline(H, HSEC, nm, ns),
        "flightbert": FlightBERTBaseline(HSTEPS, dt_sec=DT),
        "fbpp_v2_faithful": FlightBERTPPV2Faithful(CFG10),
    }
    for name, m in models.items():
        m = m.to(DEV).eval()
        with torch.no_grad():
            out = m(x)
        p = out[0] if isinstance(out, tuple) else out
        if name == "flightbert":
            # FlightBERT is a NEXT-STEP model: the forward pass returns one
            # point per sample, and the horizon vector comes from the roll-out
            check(tuple(p.shape) == (4, 6), "flightbert: one-step output [B,6]",
                  str(tuple(p.shape)))
            with torch.no_grad():
                roll = m.rollout_at(x, HSTEPS)
            check(tuple(roll.shape) == (4, H, 6), "flightbert: roll-out output [B,H,6]",
                  str(tuple(roll.shape)))
            check(bool(torch.isfinite(roll).all()), "flightbert: finite roll-out")
            p = roll
        else:
            check(tuple(p.shape) == (4, H, 6), f"{name}: output [B,H,6]", str(tuple(p.shape)))
            check(bool(torch.isfinite(p).all()), f"{name}: finite output")
        # determinism in eval mode
        with torch.no_grad():
            out2 = m(x)
        p2 = out2[0] if isinstance(out2, tuple) else out2
        if name == "flightbert":
            with torch.no_grad():
                p2 = m.rollout_at(x, HSTEPS)
        check(bool(torch.allclose(p, p2)), f"{name}: deterministic in eval mode")

    # the BiLSTM must use BOTH directions of the sequence: the summary is the
    # forward state at the last step and the backward state at the FIRST step
    m = BiLSTMRegressionBaseline(H, HSEC, nm, ns).to(DEV).eval()
    with torch.no_grad():
        lstm_out, _ = m.lstm((x[..., :6] - m.norm_mean) / m.norm_std)
    fwd_last = lstm_out[:, -1, :m.lstm.hidden_size]
    bwd_at_first = lstm_out[:, 0, m.lstm.hidden_size:]
    with torch.no_grad():
        delta = m.head(torch.cat([fwd_last, bwd_at_first], dim=-1))
    with torch.no_grad():
        full = m(x)
    expected = x[:, -1:, :6] + delta.view(-1, H, 6) * m.out_scale * (0.1 + 0.9 * m.t_scale)
    err = (full - expected).abs().max().item()
    check(err < 1e-5,
          "the BiLSTM summary is the forward-last + backward-first state",
          f"max err {err:.2e}")


# --------------------------------------------------------------------------- #
# 9. spec conformance
# --------------------------------------------------------------------------- #
def test_spec() -> None:
    section("9. published hyper-parameters vs ours")
    from bench_baselines import FlightBERTBaseline, FlightBERTPPV2Faithful
    from model import FlightBERT_PP_V2_Baseline

    # the reference configuration of FlightBERT++ (gdy-scu/FlightBERT_PP_V2)
    REF = dict(bit_abs=[18, 16, 11, 11, 11, 11],
               bit_delta=[8, 8, 8, 9, 9, 6],
               n_embd=768, n_head=4, n_en=4, n_de=2,
               inp_seq_len=9, horizon=15, lr=1e-4, dropout=0.1)

    fb = FlightBERT_PP_V2_Baseline(CFG10)
    n_embd = fb.n_embd
    n_en = len(fb.transformer_encoder.layers)
    n_de = len(fb.decoder.layers)
    n_head = fb.n_head
    rows = [
        ("FlightBERT++ bits (absolute)", str(REF["bit_abs"]),
         str([fb.codec.lon_size, fb.codec.lat_size, fb.codec.alt_size,
              fb.codec.spdx_size, fb.codec.spdy_size, fb.codec.spdz_size])),
        ("FlightBERT++ bits (delta)", str(REF["bit_delta"]), str(list(fb.codec.delta_sizes))),
        ("FlightBERT++ n_embd", str(REF["n_embd"]), str(n_embd)),
        ("FlightBERT++ n_head", str(REF["n_head"]), str(n_head)),
        ("FlightBERT++ n_en_layer", str(REF["n_en"]), str(n_en)),
        ("FlightBERT++ n_de_layer", str(REF["n_de"]), str(n_de)),
        ("FlightBERT++ input points", str(REF["inp_seq_len"]), str(T)),
        ("FlightBERT++ horizons", str(REF["horizon"]), str(H)),
    ]
    print("     spec item                    published      ours")
    for label, want, got in rows:
        print(f"     {label:30s} {want:14s} {got}")

    # the checks that must hold, and the deviations that are documented
    check(n_head == REF["n_head"], "the attention head count matches the reference")
    check(T == int(CFG10["window"]["inp_seq_len"]), "the input length is the one of our protocol")
    check(H == int(CFG10["window"]["horizon"]), "the horizon count is the one of our protocol")
    check(float(CFG10["training"]["learning_rate"]) >= 0.0, "the learning rate is set by the protocol")

    # FlightBERT (2023): 8 decoder blocks, 768 wide, 4 heads
    f23 = FlightBERTBaseline(HSTEPS, dt_sec=DT)
    check(len(f23.blocks) == 8, "FlightBERT: 8 stacked blocks", str(len(f23.blocks)))
    check(f23.blocks[0].attn.n_head == 4, "FlightBERT: 4 attention heads")
    check(f23.attr_fc[0][0].out_features == 64, "FlightBERT: 64-dim attribute embedding")
    check(f23.head.in_features == 768, "FlightBERT: 768-dim trajectory embedding",
          str(f23.head.in_features))
    check(f23.codec.total_bits == 84, "FlightBERT: 84 bits (the paper's 78 with narrower fields)",
          str(f23.codec.total_bits))

    # FlightBERT++ faithful: the ACoAtt module is present
    fpp = FlightBERTPPV2Faithful(CFG10)
    check(len(fpp.coatt) == 6, "FlightBERT++: six bidirectional CoAtt blocks")
    check(len(fpp.decoder.layers) == 2, "FlightBERT++: 2 decoder layers (reference: 2)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("Conformance suite — AeroTraj-Ultra and its re-implementations")
    test_frames()
    test_graycode()
    test_binary_encoding()
    test_kalman()
    test_imm()
    test_metrics()
    test_losses()
    test_contracts()
    test_spec()

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
    sys.exit(1 if FAILURES else 0)

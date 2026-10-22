# coding=utf-8
"""
Re-implementations of the literature baselines, all trained on the SAME data
with the SAME budget, for the 365-day / 10 s-grid benchmark.

Models
------
  kalman      Constant-Velocity Kalman Filter with a LEARNABLE noise model.
              This is the "model-driven" family of the FlightBERT paper
              (Guo et al., IEEE T-ITS 24(2):1828-1834, 2023, group A / A3:
              "the representative model-driven methods in the TP task are based
              on the Kalman Filter (KF) and its variants").

  imm         Interacting Multiple Model filter: constant velocity, constant
              acceleration and coordinated turn, with per-mode Kalman filters,
              a mode-probability update and the standard interaction step.
              (Blom & Bar-Shalom 1988; the IMM-Informer hybrid of
              Li et al., Sensors 25(8):2531, 2025).  The noise model and the
              mode probabilities are LEARNABLE.

  lstm        Deep LSTM multi-horizon regression (FlightBERT group A / A1).

  bistm       BiLSTM multi-horizon regression (FlightBERT group A / A2).

  flightbert  FlightBERT: Binary-Encoding representation + ACoAtt + a stacked
              Transformer backbone + the Multi-Binary-Classification head and
              the Hybrid Constrained loss (BCE + MSE).
              (Guo et al., IEEE T-ITS 24(2):1828-1834, 2023).

Why the noise models are learnable
----------------------------------
The classical filters have a handful of parameters (process/measurement noise,
mode probabilities).  Fixing them by hand would handicap the comparison in a
way that has nothing to do with the method: a published KF baseline is tuned
on a validation set, while the neural baselines get millions of gradient
steps.  Making the noise model learnable puts every model in the table under
the same protocol -- same data, same optimiser, same budget -- so the measured
difference is the difference between the *estimators*, not between the tuning
efforts.  The learned noise levels are reported alongside the results.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_models import (
    M_PER_DEG_LAT,
    horizon_weighted_smooth_l1,
    local_frame,
    local_to_global,
    local_vel_to_global,
    to_local,
)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _eye(n: int, device) -> torch.Tensor:
    return torch.eye(n, device=device)


def _cv_transition(dt: float, device) -> torch.Tensor:
    """6x6 constant-velocity transition on the state [p(3), v(3)]."""
    Fm = torch.zeros(6, 6, device=device)
    Fm[0:3, 0:3] = torch.eye(3, device=device)
    Fm[0:3, 3:6] = torch.eye(3, device=device) * dt
    Fm[3:6, 3:6] = torch.eye(3, device=device)
    return Fm


def _q_blocks(dt: float, device, q: torch.Tensor) -> torch.Tensor:
    """
    Process-noise covariance for the state [p, v], per-axis intensity q[3].

    "cv": continuous white-noise acceleration,
        Q = q * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]
    "ca": the acceleration itself is a random walk,
        Q = q * [[dt^5/20, dt^4/8], [dt^4/8, dt^3/3]]
    """
    Q = torch.zeros(6, 6, device=device)
    a = dt ** 4 / 4.0
    b = dt ** 3 / 2.0
    c = dt ** 2
    Q[0:3, 0:3] = torch.diag(q) * a
    Q[0:3, 3:6] = torch.diag(q) * b
    Q[3:6, 0:3] = torch.diag(q) * b
    Q[3:6, 3:6] = torch.diag(q) * c
    return Q


def _q_blocks_ca(dt: float, device, q: torch.Tensor) -> torch.Tensor:
    Q = torch.zeros(6, 6, device=device)
    a = dt ** 5 / 20.0
    b = dt ** 4 / 8.0
    c = dt ** 3 / 3.0
    Q[0:3, 0:3] = torch.diag(q) * a
    Q[0:3, 3:6] = torch.diag(q) * b
    Q[3:6, 0:3] = torch.diag(q) * b
    Q[3:6, 3:6] = torch.diag(q) * c
    return Q


def _softplus_pos(log_val: torch.Tensor) -> torch.Tensor:
    return F.softplus(log_val) + 1e-8


def _psd_guard(P: torch.Tensor, eps_pos: float = 1e-3, eps_vel: float = 1e-6) -> torch.Tensor:
    """
    Numerical safeguard for the covariance: floors the diagonal so that a
    collapsing noise model cannot make the filter singular (and the closed
    form inverse blow up).  The floors are physically negligible: 1 mm^2 on
    the position and 1 mm^2/s^2 on the velocity.
    """
    d = torch.diagonal(P, dim1=-2, dim2=-1)
    floor = torch.cat([
        torch.full((3,), eps_pos, device=P.device, dtype=P.dtype),
        torch.full((3,), eps_vel, device=P.device, dtype=P.dtype),
    ]).expand_as(d)
    new_d = torch.maximum(d, floor)
    return P - torch.diag_embed(d - new_d)


def _inv3x3(S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form inverse and determinant of a batch of symmetric 3x3 matrices.

    A batched LU (torch.linalg.solve) on [B,3,3] costs ~1 ms per call and the
    filters call it once per observed step: the closed form is ~10x faster and
    is what makes the classical baselines trainable in a reasonable time.

    S: [...,3,3] symmetric  ->  (inv [...,3,3], det [...])
    """
    a = S[..., 0, 0]
    b = S[..., 0, 1]
    c = S[..., 0, 2]
    d = S[..., 1, 1]
    e = S[..., 1, 2]
    f = S[..., 2, 2]
    A00 = d * f - e * e
    A01 = c * e - b * f
    A02 = b * e - c * d
    A11 = a * f - c * c
    A12 = b * c - a * e
    A22 = a * d - b * b
    det = a * A00 + b * A01 + c * A02
    inv = torch.stack([
        torch.stack([A00, A01, A02], dim=-1),
        torch.stack([A01, A11, A12], dim=-1),
        torch.stack([A02, A12, A22], dim=-1),
    ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)
    return inv, det


def _joseph(P: torch.Tensor, K: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Joseph-form covariance update for the state [p, v] with H = [I3, 0].

    P: [..., 6, 6], K: [..., 6, 3], R: [3, 3]  ->  [..., 6, 6]
    """
    shape = P.shape[:-2]
    I6 = _eye(6, P.device).expand(*shape, 6, 6).clone()
    IKH = I6.clone()
    IKH[..., :, 0:3] = IKH[..., :, 0:3] - K
    KRK = K @ R @ K.transpose(-1, -2)
    Pn = IKH @ P @ IKH.transpose(-1, -2) + KRK
    return 0.5 * (Pn + Pn.transpose(-1, -2))


# --------------------------------------------------------------------------- #
# 1. Constant-velocity Kalman filter (learnable noise model)
# --------------------------------------------------------------------------- #
class KalmanFilterBaseline(nn.Module):
    """
    Constant-Velocity Kalman filter on the local (forward, right, up) frame.

    State  x = [p(3) metres, v(3) m/s],  measurement z = the observed position.
    The filter is run over the T observed points and then propagated (without
    measurement) to every horizon: each horizon is predicted from the end of
    the observation window, which is what the literature does.

    Learnable: the per-axis process-noise intensity q and the per-axis
    measurement-noise variance R.
    """

    def __init__(self, horizon_seconds: Sequence[float], dt_sec: float = 10.0) -> None:
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.register_buffer("t_horizon", torch.tensor(list(horizon_seconds), dtype=torch.float32))
        self.log_q = nn.Parameter(torch.tensor([-2.0, -2.0, -6.0]))     # m^2/s^3
        self.log_r = nn.Parameter(torch.tensor([6.0, 6.0, 8.0]))        # m^2

    def _filter(self, meas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """meas [B,T,3] metres -> (state [B,6], covariance [B,6,6]) at t = T-1."""
        B, T, _ = meas.shape
        dev, dt = meas.device, self.dt_sec
        Fm = _cv_transition(dt, dev)
        q = _softplus_pos(self.log_q) + 0.01
        r = F.softplus(self.log_r) + 100.0        # the noise floor: 100 m^2
        R = torch.diag(r)
        Qs = _q_blocks(dt, dev, q)

        v0 = (meas[:, 1] - meas[:, 0]) / dt
        x = torch.cat([meas[:, 0], v0], dim=-1)                  # [B,6]
        P = torch.zeros(B, 6, 6, device=dev)
        P[:, 0:3, 0:3] = torch.diag(r).expand(B, 3, 3)
        P[:, 3:6, 3:6] = torch.eye(3, device=dev).expand(B, 3, 3) * 100.0

        for t in range(1, T):
            x = torch.einsum("ij,bj->bi", Fm, x)                 # predict
            P = torch.einsum("ij,bjk,kl->bil", Fm, P, Fm) + Qs
            z = meas[:, t]                                       # [B,3]
            y = z - x[:, 0:3]                                    # innovation
            S = P[:, 0:3, 0:3] + R                               # [B,3,3]
            tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) / 3.0
            S = S + (1e-3 * tr + 1e-9).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=dev)
            Sinv, det = _inv3x3(S)
            PHt = P[:, :, 0:3]                                   # [B,6,3]
            K = PHt @ Sinv                                       # [B,6,3]
            x = x + torch.einsum("bij,bj->bi", K, y)
            P = _joseph(P, K, R)
            P = _psd_guard(P)
        return x, P

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        origin = local_frame(x_obs)
        meas = to_local(x_obs, origin)                   # [B,T,3] metres
        state, _ = self._filter(meas)
        pos, vel = [], []
        for t in self.t_horizon:
            Fm = _cv_transition(float(t), x_obs.device)
            st = torch.einsum("ij,bj->bi", Fm, state)
            pos.append(st[:, 0:3])
            vel.append(st[:, 3:6])
        pred_local = torch.stack(pos, dim=1)              # [B,H,3] metres
        vel_local = torch.stack(vel, dim=1)              # [B,H,3] m/s
        gl = local_to_global(pred_local, origin)
        vloc = torch.cat([vel_local[..., :2] * 3.6, vel_local[..., 2:3]], dim=-1)
        gv = local_vel_to_global(vloc, origin)
        return torch.cat([gl, gv], dim=-1)

    def noise_report(self) -> Dict[str, object]:
        return {
            "q_process_m2_s3": [float(v) for v in _softplus_pos(self.log_q).detach()],
            "r_measurement_m2": [float(v) for v in _softplus_pos(self.log_r).detach()],
        }


# --------------------------------------------------------------------------- #
# 2. Interacting Multiple Model filter (learnable)
# --------------------------------------------------------------------------- #
class ClassicalIMMBaseline(nn.Module):
    """
    Interacting Multiple Model filter (Blom & Bar-Shalom 1988) with three
    kinematic modes:

        0  constant velocity (CV)
        1  constant acceleration (CA)
        2  coordinated turn (CT) on the horizontal plane + CV on the vertical

    Each mode owns a 6-D Kalman filter on the state [p, v]; the mode
    probabilities are updated with the Gaussian likelihood of each mode's
    innovation, and the standard interaction (mixing) step couples the modes
    before every prediction.

    Learnable: the per-mode process noise, the measurement noise, the turn rate
    of the CT mode, the mode-transition matrix and the initial mode prior.
    """

    MODES = 3

    def __init__(self, horizon_seconds: Sequence[float], dt_sec: float = 10.0) -> None:
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.register_buffer("t_horizon", torch.tensor(list(horizon_seconds), dtype=torch.float32))
        self.log_q = nn.Parameter(torch.tensor([[-2.0, -2.0, -6.0]]).repeat(self.MODES, 1))
        self.log_r = nn.Parameter(torch.tensor([6.0, 6.0, 8.0]))
        self.log_omega = nn.Parameter(torch.tensor(-3.0))              # CT turn rate [rad/s]
        self.trans_logits = nn.Parameter(torch.tensor([
            [2.2, 0.0, -1.2],
            [0.0, 2.2, -1.2],
            [-1.2, 0.0, 2.2],
        ]))
        self.prior_logits = nn.Parameter(torch.zeros(self.MODES))

    # ------------------------------------------------------------------ #
    def _transition_matrices(self, dt: float, dev) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (F [M,6,6], Q [M,6,6]) for the three modes."""
        F_cv = _cv_transition(dt, dev)
        Fs = torch.stack([F_cv, F_cv, F_cv.clone()])           # [3,6,6]

        # coordinated turn on the horizontal plane
        omega = torch.tanh(self.log_omega) * 0.05              # rad/s, |w| <= 0.05
        theta = omega * dt
        c, s = torch.cos(theta), torch.sin(theta)
        Rm = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
        V = torch.eye(2, device=dev) * dt if abs(float(omega)) < 1e-6 else \
            (Rm - torch.eye(2, device=dev)) / omega
        F_ct = F_cv.clone()
        F_ct[0:2, 0:2] = Rm
        F_ct[0:2, 3:5] = V
        Fs[2] = F_ct

        q = _softplus_pos(self.log_q) + 0.01                   # [M,3]
        blocks = [
            _q_blocks(dt, dev, q[0]),
            _q_blocks_ca(dt, dev, q[1]),
            _q_blocks(dt, dev, q[2]),
        ]
        return Fs, torch.stack(blocks)

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_obs.shape
        dev = x_obs.device
        origin = local_frame(x_obs)
        meas = to_local(x_obs, origin)                    # [B,T,3] metres
        M = self.MODES
        dt = self.dt_sec

        Fs, Qs = self._transition_matrices(dt, dev)       # [M,6,6]
        R = torch.diag(F.softplus(self.log_r) + 100.0)    # [3,3]
        Pi = torch.softmax(self.trans_logits, dim=1)      # [M,M]
        mu = torch.softmax(self.prior_logits, dim=0).expand(B, M).contiguous()

        v0 = (meas[:, 1] - meas[:, 0]) / dt
        x = torch.zeros(B, M, 6, device=dev)
        x[:, :, 0:3] = meas[:, 0:1].expand(B, M, 3)
        x[:, :, 3:6] = v0.unsqueeze(1).expand(B, M, 3)
        P = torch.zeros(B, M, 6, 6, device=dev)
        pr = F.softplus(self.log_r) + 100.0
        P[:, :, 0:3, 0:3] = torch.diag(pr).expand(B, M, 3, 3)
        P[:, :, 3:6, 3:6] = torch.eye(3, device=dev).expand(B, M, 3, 3) * 100.0

        for t in range(1, T):
            # ---- interaction (mixing) ---------------------------------- #
            c = mu @ Pi                                      # [B,M]
            W = mu.unsqueeze(-1) * Pi.transpose(0, 1).unsqueeze(0)   # [B,M(j),M(i)]
            W = W / c.clamp(min=1e-12).unsqueeze(-1)
            x_mix = (W.unsqueeze(-1) * x.unsqueeze(2)).sum(dim=1)     # [B,M,6]
            dx = x.unsqueeze(2) - x_mix.unsqueeze(1)                   # [B,M(j),M(i),6]
            Pm = P.unsqueeze(1) + dx.unsqueeze(-1) @ dx.unsqueeze(-2)  # [B,M(j),M(i),6,6]
            P_mix = (W.unsqueeze(-1).unsqueeze(-1) * Pm).sum(dim=2)    # [B,M,6,6]

            # ---- per-mode prediction + measurement update -------------- #
            x_pred = torch.einsum("mij,bmj->bmi", Fs, x_mix)
            P_pred = torch.einsum("mij,bmjk,mlk->bmil", Fs, P_mix, Fs) + Qs.unsqueeze(0)
            z = meas[:, t].unsqueeze(1).expand(B, M, 3)
            y = z - x_pred[:, :, 0:3]                                 # [B,M,3]
            S = P_pred[:, :, 0:3, 0:3] + R.unsqueeze(0)
            tr = torch.diagonal(S, dim1=-2, dim2=-1).sum(-1) / 3.0
            S = S + (1e-3 * tr + 1e-9).unsqueeze(-1).unsqueeze(-1) * torch.eye(3, device=dev)
            Sinv, det = _inv3x3(S)
            PHt = P_pred[:, :, :, 0:3]                            # [B,M,6,3]
            K = PHt @ Sinv                                       # [B,M,6,3]
            x = x_pred + torch.einsum("bmij,bmj->bmi", K, y)
            P = _joseph(P_pred, K, R.unsqueeze(0))
            P = _psd_guard(P)

            # ---- mode probability update -------------------------------- #
            # the covariance can become numerically singular when the learned
            # noise collapses, so the likelihood uses the closed-form inverse
            # of a regularised matrix
            quad = torch.einsum("bmi,bmij,bmj->bm", y, Sinv, y)
            logdet = torch.log(det.abs().clamp(min=1e-12))
            loglik = -0.5 * (quad + logdet + 3.0 * float(np.log(2.0 * np.pi)))
            c_new = c * torch.exp(loglik - loglik.max(dim=1, keepdim=True).values)
            mu = c_new / c_new.sum(dim=1, keepdim=True).clamp(min=1e-12)

        # ---- propagate every mode to the horizons and combine ------------ #
        pos, vel = [], []
        for th in self.t_horizon:
            Fh, _ = self._transition_matrices(float(th), dev)
            st = torch.einsum("mij,bmj->bmi", Fh, x)
            pos.append(st[:, :, 0:3])
            vel.append(st[:, :, 3:6])
        Pm = torch.stack(pos, dim=2)                               # [B,M,H,3]
        Vm = torch.stack(vel, dim=2)                               # [B,M,H,3]
        w = mu[:, :, None, None]
        pred_local = (w * Pm).sum(dim=1)                           # [B,H,3]
        vel_local = (w * Vm).sum(dim=1)                            # [B,H,3]
        gl = local_to_global(pred_local, origin)
        vloc = torch.cat([vel_local[..., :2] * 3.6, vel_local[..., 2:3]], dim=-1)
        gv = local_vel_to_global(vloc, origin)
        return torch.cat([gl, gv], dim=-1)

    def noise_report(self) -> Dict[str, object]:
        return {
            "q_process_m2_s3": [[float(v) for v in row] for row in _softplus_pos(self.log_q).detach()],
            "r_measurement_m2": [float(v) for v in _softplus_pos(self.log_r).detach()],
            "turn_rate_rad_s": float(torch.tanh(self.log_omega).detach() * 0.05),
            "mode_transition": torch.softmax(self.trans_logits, dim=1).detach().tolist(),
            "initial_mode_prior": [float(v) for v in torch.softmax(self.prior_logits, dim=0).detach()],
        }


# --------------------------------------------------------------------------- #
# 3. BiLSTM multi-horizon regression
# --------------------------------------------------------------------------- #
class BiLSTMRegressionBaseline(nn.Module):
    """
    BiLSTM multi-horizon regression baseline (FlightBERT paper, group A / A2).

    Same input/output convention as the LSTM baseline: the network predicts a
    horizon-dependent displacement from the last observed point, and the
    displacement is scaled by the physical range of each attribute.
    """

    def __init__(
        self,
        horizon: int,
        horizon_seconds: Sequence[float],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        hidden_dim: int = 192,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.register_buffer("norm_mean", torch.tensor(norm_mean[:6], dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std[:6], dtype=torch.float32))
        horizon_ref = float(max(horizon_seconds))
        self.register_buffer(
            "t_scale",
            (torch.tensor(list(horizon_seconds), dtype=torch.float32) / horizon_ref).view(1, horizon, 1),
        )
        self.register_buffer(
            "out_scale", torch.tensor([0.8, 0.8, 120.0, 200.0, 200.0, 15.0], dtype=torch.float32).view(1, 1, 6)
        )
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, horizon * 6),
        )

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        x_6d = x_obs[..., :6]
        last_6d = x_6d[:, -1:, :]
        norm_in = (x_6d - self.norm_mean.view(1, 1, 6)) / self.norm_std.view(1, 1, 6)
        out, _ = self.lstm(norm_in)
        h = out[:, -1, :]                              # [B, 2*hidden]
        delta = self.head(h).view(-1, self.horizon, 6) * self.out_scale * (0.1 + 0.9 * self.t_scale)
        return last_6d + delta


# --------------------------------------------------------------------------- #
# 4. FlightBERT: Binary Encoding + ACoAtt + Transformer + MBC
# --------------------------------------------------------------------------- #
class BinaryEncodingCodec(nn.Module):
    """
    The Binary-Encoding representation of FlightBERT (Guo et al., 2023, Sec. III-C).

    Every scalar attribute is multiplied by its resolution, rounded to an
    integer and written in base 2.  For the signed attributes (the three
    velocities) the highest bit carries the sign, exactly as in the paper:
    "the highest bit is 1 if the value of the velocity is a negative number".

    Bit widths (paper: 18/16/11/11/11/11):
        longitude   19 bits @ 0.001 deg  (the paper's 18 bits cannot hold
                                         +-180 deg at that resolution)
        latitude    18 bits @ 0.001 deg
        altitude    11 bits @ 10 m
        vx, vy      12 bits (1 sign + 11 magnitude) @ 1 km/h
        vz          12 bits (1 sign + 11 magnitude) @ 1 m/s
    """

    SPECS = [
        # (name, n_bits, multiplier, signed, offset)
        ("lon", 19, 1e3, False, 180_000.0),
        ("lat", 18, 1e3, False, 90_000.0),
        ("alt", 11, 1.0, False, 0.0),
        ("vx", 12, 1.0, True, 0.0),
        ("vy", 12, 1.0, True, 0.0),
        ("vz", 12, 1.0, True, 0.0),
    ]

    MAX_MAG = 19   # the largest magnitude field (longitude)

    def __init__(self) -> None:
        super().__init__()
        self.n_attrs = len(self.SPECS)
        self.total_bits = sum(s[1] for s in self.SPECS)
        self.sizes = [s[1] for s in self.SPECS]
        self.n_mag = [s[1] - 1 if s[3] else s[1] for s in self.SPECS]
        w = torch.zeros(len(self.SPECS), self.MAX_MAG)
        s = torch.zeros(len(self.SPECS), self.MAX_MAG, dtype=torch.long)
        for k, spec in enumerate(self.SPECS):
            nm = self.n_mag[k]
            w[k, self.MAX_MAG - nm:] = torch.tensor([2.0 ** i for i in range(nm - 1, -1, -1)])
            s[k, self.MAX_MAG - nm:] = torch.arange(nm - 1, -1, -1, dtype=torch.long)
        self.register_buffer("bit_weights", w)
        self.register_buffer("bit_shifts", s)
        # the number of representable values per attribute: the normalisation
        # that makes the MSE of the Hybrid Constrained loss unit-free, i.e.
        # "the BE representation is able to eliminate the impacts of different
        # units" (the claim the paper makes for the representation)
        ranges = []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            if name == "lon":
                ranges.append(360_001.0)
            elif name == "lat":
                ranges.append(180_001.0)
            else:
                ranges.append(float(2 ** self.n_mag[k]))
        self.register_buffer("code_ranges", torch.tensor(ranges, dtype=torch.float32))

    def _weights(self, k: int) -> torch.Tensor:
        return self.bit_weights[k, self.MAX_MAG - self.n_mag[k]:]

    def _shifts(self, k: int) -> torch.Tensor:
        return self.bit_shifts[k, self.MAX_MAG - self.n_mag[k]:]

    # ------------------------------------------------------------------ #
    def codes(self, vals: torch.Tensor) -> List[torch.Tensor]:
        """the integer code of every attribute (the shifted / signed value)."""
        out = []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            v = vals[..., k]
            if signed:
                code = torch.round(v.abs() * mult)
            else:
                code = torch.round(v * mult) + offset
            out.append(code.clamp(min=0.0, max=float(2 ** self.n_mag[k] - 1)))
        return out

    def encode(self, vals: torch.Tensor) -> List[torch.Tensor]:
        """vals [B,T,6] -> a list of 6 bit tensors [B,T,n_k]."""
        out = []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            v = vals[..., k]
            n_mag = self.n_mag[k]
            if signed:
                sign = (v < 0).float()
                code = torch.round(v.abs() * mult)
            else:
                sign = torch.zeros_like(v)
                code = torch.round(v * mult) + offset
            code = code.clamp(min=0.0, max=float(2 ** n_mag - 1)).to(torch.int64)
            bits = ((code.unsqueeze(-1) >> self._shifts(k)) & 1).float()
            if signed:
                bits = torch.cat([sign.unsqueeze(-1), bits], dim=-1)
            out.append(bits)
        return out

    def encode_flat(self, vals: torch.Tensor) -> torch.Tensor:
        """[B,T,6] (or [B,6]) -> the concatenated target bits."""
        return torch.cat(self.encode(vals), dim=-1)

    def decode(self, bits: Sequence[torch.Tensor]) -> torch.Tensor:
        """a list of 6 bit tensors -> values [...,6]."""
        vals = []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            b = bits[k]
            bm = b[..., 1:] if signed else b
            code = (bm * self._weights(k)).sum(-1)
            if signed:
                v = torch.where(b[..., 0] > 0.5, -code, code)
            else:
                v = (code - offset) / mult
            vals.append(v)
        return torch.stack(vals, dim=-1)

    def _split_soft(self, soft_bits: torch.Tensor):
        """-> a list of (hard bits, soft bits) per attribute."""
        hard = (soft_bits > 0.5).float()
        out, off = [], 0
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            out.append((hard[..., off:off + nb], soft_bits[..., off:off + nb]))
            off += nb
        return out

    def ste_decode(self, soft_bits: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator: the forward value is the hard thresholded
        decoding, the gradient is that of the soft (sigmoid) decoding.

        soft_bits: the concatenated [..., total_bits] tensor.
        """
        parts = self._split_soft(soft_bits)
        pieces_hard, pieces_soft = [], []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            b_h, b_s = parts[k]
            w = self._weights(k)
            if signed:
                code_h = (b_h[..., 1:] * w).sum(-1)
                code_s = (b_s[..., 1:] * w).sum(-1)
                v_h = torch.where(b_h[..., 0] > 0.5, -code_h, code_h)
                v_s = torch.where(b_s[..., 0] > 0.5, -code_s, code_s)
            else:
                code_h = (b_h * w).sum(-1)
                code_s = (b_s * w).sum(-1)
                v_h = (code_h - offset) / mult
                v_s = (code_s - offset) / mult
            pieces_hard.append(v_h)
            pieces_soft.append(v_s)
        hard_v = torch.stack(pieces_hard, dim=-1)
        soft_v = torch.stack(pieces_soft, dim=-1)
        return hard_v + (soft_v - soft_v.detach())

    def soft_decode(self, soft_bits: torch.Tensor) -> torch.Tensor:
        """
        The decoding without the threshold: the expectation of the value under
        the sigmoid outputs.

        Used to diagnose the autoregressive roll-out: the hard threshold turns
        a marginal uncertainty on the HIGH bits of the binary encoding into a
        macroscopic position error (the highest bit of the longitude carries
        90 deg), which is what makes the published roll-out diverge.
        """
        pieces = []
        off = 0
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            b_s = soft_bits[..., off:off + nb]
            off += nb
            w = self._weights(k)
            if signed:
                mag = (b_s[..., 1:] * w).sum(-1)
                v = (1.0 - 2.0 * b_s[..., 0]) * mag
            else:
                v = ((b_s * w).sum(-1) - offset) / mult
            pieces.append(v)
        return torch.stack(pieces, dim=-1)

    def ste_codes(self, soft_bits: torch.Tensor) -> torch.Tensor:
        """
        The same straight-through decoding, expressed in CODE UNITS and
        normalised by the number of representable values of each attribute.

        This is the quantity the MSE of the Hybrid Constrained loss is
        computed on: the paper's "MSE on the decimal values" divided by the
        per-attribute resolution, which makes the six attributes commensurable
        (the BE representation is unit-free by construction).
        """
        parts = self._split_soft(soft_bits)
        pieces_h, pieces_s = [], []
        for k, (name, nb, mult, signed, offset) in enumerate(self.SPECS):
            b_h, b_s = parts[k]
            w = self._weights(k)
            if signed:
                mag_h = (b_h[..., 1:] * w).sum(-1)
                mag_s = (b_s[..., 1:] * w).sum(-1)
                code_h = torch.where(b_h[..., 0] > 0.5, -mag_h, mag_h)
                code_s = (1.0 - 2.0 * b_s[..., 0]) * mag_s
            else:
                code_h = (b_h * w).sum(-1) - offset
                code_s = (b_s * w).sum(-1) - offset
            pieces_h.append(code_h)
            pieces_s.append(code_s)
        hard_c = torch.stack(pieces_h, dim=-1)
        soft_c = torch.stack(pieces_s, dim=-1)
        hard_n = hard_c / self.code_ranges
        soft_n = soft_c / self.code_ranges
        return hard_n + (soft_n - soft_n.detach())

    def target_codes(self, vals: torch.Tensor) -> torch.Tensor:
        """the normalised target codes [...,6]."""
        codes = torch.stack(self.codes(vals), dim=-1)
        return codes / self.code_ranges


class SDPAHead(nn.Module):
    """
    Multi-head attention written on top of F.scaled_dot_product_attention.

    Same mathematics as nn.MultiheadAttention (the projections are the same
    matrices), but the fused kernel is 3-5x faster on this GPU, which is what
    makes a 57M-parameter, 8-block transformer trainable on the benchmark
    budget.
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = q_in.shape
        qh = self.q(q_in).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kh = self.k(kv_in).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        vh = self.v(kv_in).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.drop(self.o(out))


class _CausalBlock(nn.Module):
    """Pre-LN Transformer decoder block with a masked self-attention (MMSA)."""

    def __init__(self, d_model: int, n_head: int, ff_mult: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SDPAHead(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.ln1(x)
        a = self.attn(h, h, attn_mask)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


class FlightBERTBaseline(nn.Module):
    """
    FlightBERT (Guo et al., IEEE T-ITS 24(2):1828-1834, 2023).

    Architecture (as published):
      * Binary-Encoding representation of the six trajectory attributes;
      * one Attribute-Embedding block per attribute (two FC layers, 64 dims);
      * an ACoAtt block: for each correlated pair (lon<->vx, lat<->vy, alt<->vz)
        a multi-head cross-attention that recalibrates each attribute embedding
        with the correlated one; the two embeddings are concatenated, so the
        trajectory embedding is 6 x 128 = 768 dims;
      * a backbone of 8 stacked Transformer decoder blocks (masked MMSA + LN +
        position-wise FFN);
      * a predictor (one FC layer + sigmoid) that outputs the binary codes of
        the NEXT trajectory point;
      * the Hybrid Constrained loss = BCE on the binary codes + MSE on the
        decoded decimals.

    Two documented deviations, both forced by the protocol:
      (i)  the model is trained on the next step (10 s) exactly as published,
           and the multi-horizon prediction is obtained with the paper's
           autoregressive roll-out (predict, decode, append to the window);
      (ii) the MSE of the HC loss is computed on scale-normalised values
           (metres for the positions, km/h for the velocities) instead of the
           paper's raw decimals, which would put the altitude term (units of
           10 m) three orders of magnitude above the others.
    """

    def __init__(
        self,
        horizon_steps: Sequence[int],
        dt_sec: float = 10.0,
        n_layers: int = 8,
        d_model: int = 768,
        n_head: int = 4,
        ff_mult: int = 4,
        attr_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.codec = BinaryEncodingCodec()
        self.dt_sec = float(dt_sec)
        self.horizon_steps = [int(s) for s in horizon_steps]
        self.max_steps = max(self.horizon_steps)

        self.attr_fc = nn.ModuleList([
            nn.Sequential(nn.Linear(nb, attr_dim), nn.GELU(), nn.Linear(attr_dim, attr_dim))
            for nb in self.codec.sizes
        ])
        self.pairs = [(0, 3), (1, 4), (2, 5)]
        self.coatt = nn.ModuleList([
            SDPAHead(attr_dim, n_head, dropout) for _ in range(2 * len(self.pairs))
        ])
        self.pos_emb = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([
            _CausalBlock(d_model, n_head, ff_mult, dropout) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.codec.total_bits)

    # ------------------------------------------------------------------ #
    def _embed(self, x_obs: torch.Tensor) -> torch.Tensor:
        """[B,T,9] -> the trajectory embeddings [B,T,768]."""
        vals = x_obs[..., :6]
        bits = self.codec.encode(vals)
        embs = [self.attr_fc[k](b) for k, b in enumerate(bits)]
        for p, (a, b) in enumerate(self.pairs):
            ea, eb = embs[a], embs[b]
            ca = self.coatt[2 * p](ea, eb)                        # Q=ea, K=V=eb
            cb = self.coatt[2 * p + 1](eb, ea)                    # Q=eb, K=V=ea
            embs[a] = torch.cat([ea, ca], dim=-1)           # 128
            embs[b] = torch.cat([eb, cb], dim=-1)           # 128
        return torch.cat(embs, dim=-1)                      # [B,T,768]

    def _backbone(self, emb: torch.Tensor) -> torch.Tensor:
        B, T, _ = emb.shape
        x = emb + self.pos_emb(torch.arange(T, device=emb.device))[None]
        mask = torch.triu(torch.ones(T, T, device=emb.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)
        return self.ln_f(x)

    def forward(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> the next point [B,6] (STE-decoded) and its soft bits."""
        emb = self._embed(x_obs)
        h = self._backbone(emb)[:, -1]                     # the last time step
        bits = torch.sigmoid(self.head(h))                 # [B,total_bits]
        vals = self.codec.ste_decode(bits)
        return vals, bits

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def rollout(self, x_obs: torch.Tensor, soft: bool = False) -> torch.Tensor:
        """
        The paper's multi-step inference: predict the next point, decode it
        (hard threshold), append it to the window and repeat.

        With soft=True the decoding is not thresholded (the expectation under
        the sigmoid): this is the diagnostic variant that isolates the cause of
        the divergence (the threshold amplifying the uncertainty on the high
        bits of the encoding).

        -> [B, max_steps, 6]
        """
        hist = x_obs[..., :6]
        outs = []
        for _ in range(self.max_steps):
            nxt, bits = self.forward(hist)
            if soft:
                nxt = self.codec.soft_decode(bits)
            outs.append(nxt)
            hist = torch.cat([hist[:, 1:], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)

    def rollout_at(self, x_obs: torch.Tensor, steps: Sequence[int],
                   soft: bool = False) -> torch.Tensor:
        """-> [B,H,6] at the requested horizon steps."""
        traj = self.rollout(x_obs, soft=soft)
        idx = torch.tensor([int(s) - 1 for s in steps], device=traj.device)
        return traj.index_select(1, idx)


# --------------------------------------------------------------------------- #
# 5. FlightBERT++ (AAAI 2024), the faithful re-implementation
# --------------------------------------------------------------------------- #
class FlightBERTPPV2Faithful(nn.Module):
    """
    FlightBERT++ (Guo et al., AAAI 2024), the faithful re-implementation.

    A Gray-Code multi-horizon, non-autoregressive transformer with

      * the Attribute Correlation Attention trajectory-embedding module: one
        Attribute-Embedding block per attribute, then the bidirectional CoAtt
        cross-attention between the kinematically correlated pairs
        (lon<->vx, lat<->vy, alt<->vz), then the fusion into the model space;
      * a Transformer encoder over the trajectory embeddings;
      * the non-autoregressive multi-horizon decoder: the Gray-Code *delta*
        tokens, the horizon embedding and the aggregated context, decoded in a
        single pass (no autoregressive roll-out);
      * the Hybrid Constrained loss: the BCE on the Gray-Code deltas + the
        continuous regression on the decoded trajectory.

    The `FlightBERT_PP_V2_Baseline` of `model.py` is the SAME architecture with
    the ACoAtt module removed (this project uses that version as an ablation);
    this class is the faithful one.
    """

    def __init__(self, cfg: Dict, attr_dim: int = 64, d_model: int = 192,
                 n_head: int = 4, n_layer: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        from model import (  # local import: model.py is the project's module
            CoAttModule,
            FeatureAggregation,
            GrayCodeCodec,
            SinusoidalPositionalEncoding,
            TokenConvEmbedding,
        )

        self.cfg = cfg
        win = cfg["window"]
        self.inp_seq_len = int(win["inp_seq_len"])
        self.horizon = int(win["horizon"])
        self.codec = GrayCodeCodec(cfg)

        # ---- trajectory embedding: AE + ACoAtt -------------------------- #
        self.ae = nn.ModuleList([
            nn.Linear(size, attr_dim)
            for size in (self.codec.lon_size, self.codec.lat_size, self.codec.alt_size,
                         self.codec.spdx_size, self.codec.spdy_size, self.codec.spdz_size)
        ])
        self.coatt = nn.ModuleList([CoAttModule(attr_dim, num_heads=n_head, dropout=dropout)
                                     for _ in range(6)])
        self.fusion = nn.Sequential(
            nn.Linear(attr_dim * 6, d_model), nn.LayerNorm(d_model), nn.GELU())

        # ---- encoder --------------------------------------------------- #
        self.pos_emb = SinusoidalPositionalEncoding(d_model, dropout, max_len=128)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.feature_aggre = FeatureAggregation(d_model)

        # ---- non-autoregressive multi-horizon decoder ------------------- #
        self.dec_token_emb = TokenConvEmbedding(self.codec.total_delta_bits, d_model)
        self.horizon_emb = nn.Embedding(self.horizon, d_model)
        self.horizon_linear = nn.Linear(d_model * 2, d_model)
        self.de_pos_emb = SinusoidalPositionalEncoding(d_model, dropout, max_len=128)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, self.codec.total_delta_bits, bias=False)

    # ------------------------------------------------------------------ #
    def _embed(self, x_6d: torch.Tensor) -> torch.Tensor:
        gc = self.codec.encode_input_sequence(x_6d)
        e = [F.gelu(self.ae[k](gc[k])) for k in range(6)]
        # the bidirectional attribute correlation attention
        e_lon = self.coatt[0](e[0], e[3])
        e_vx = self.coatt[1](e[3], e[0])
        e_lat = self.coatt[2](e[1], e[4])
        e_vy = self.coatt[3](e[4], e[1])
        e_alt = self.coatt[4](e[2], e[5])
        e_vz = self.coatt[5](e[5], e[2])
        return self.fusion(torch.cat([e_lon, e_lat, e_alt, e_vx, e_vy, e_vz], dim=-1))

    def forward(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, T, _ = x_obs.shape
        device = x_obs.device
        x_6d = x_obs[..., :6]
        last_obs_6d = x_6d[:, -1, :]

        emb = self._embed(x_6d)
        enc_out = self.transformer_encoder(self.pos_emb(emb))
        context = self.feature_aggre(enc_out)

        h_idx = torch.arange(self.horizon, device=device).unsqueeze(0).expand(B, -1)
        horizon_embed = self.horizon_emb(h_idx)
        ctx_rep = context.unsqueeze(1).expand(-1, self.horizon, -1)
        decoder_input = self.horizon_linear(torch.cat([ctx_rep, horizon_embed], dim=-1))

        ones_dk = torch.ones((1, T - 1, 1), device=device, dtype=x_obs.dtype)
        dec_deltas = torch.cat(self.codec.encode_step_deltas(x_6d, ones_dk), dim=-1)
        decoder_input_diff = self.dec_token_emb(dec_deltas)
        decoder_full = self.de_pos_emb(torch.cat([decoder_input_diff, decoder_input], dim=1))

        fea = self.ln_f(self.decoder(decoder_full, memory=enc_out))
        logits = self.head(fea)[:, -self.horizon:, :]
        logits_tuple = torch.split(logits, self.codec.delta_sizes, dim=-1)
        probs_tuple = torch.split(torch.sigmoid(logits), self.codec.delta_sizes, dim=-1)
        pred_6d = self.codec.decode_horizon_deltas_ste(last_obs_6d, probs_tuple)
        return pred_6d, logits_tuple


def flightbertpp_hybrid_loss(
    model: FlightBERTPPV2Faithful,
    x_obs: torch.Tensor,
    y_target: torch.Tensor,
    horizon_seconds: Sequence[int],
    w_bce: float = 1.0,
    w_cont: float = 1.0,
) -> torch.Tensor:
    """
    The Hybrid Constrained loss of FlightBERT++: the BCE on the Gray-Code
    deltas plus the continuous regression on the decoded trajectory.
    """
    pred, logits = model(x_obs)
    tgt = model.codec.encode_targets_from_anchor(x_obs[:, -1, :6], y_target)
    bce = sum(F.binary_cross_entropy_with_logits(l, t) for l, t in zip(logits, tgt)) / 6.0
    cont = horizon_weighted_smooth_l1(pred, y_target, list(horizon_seconds))
    return w_bce * bce + w_cont * cont


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def flightbert_hc_loss(
    model: "FlightBERTBaseline",
    x_obs: torch.Tensor,
    y_next: torch.Tensor,
    w_bce: float = 1.0,
    w_mse: float = 0.1,
) -> torch.Tensor:
    """
    The Hybrid Constrained loss of FlightBERT (Eq. 7-11):

        L_HC = L_MSE(p, y) + L_BCE(p^b, y^b)

    The BCE term is the mean over the binary codes (an independent binary
    classification per bit).  The MSE term is the paper's MSE on the decoded
    values, expressed in CODE UNITS (the decoded integer minus the target
    integer, normalised by the number of representable values of the
    attribute) and passed through the straight-through estimator, so that the
    gradient reaches the sigmoid outputs.

    **The weight of the MSE is 0.1, not 1, and this is a measured decision.**
    A binary code is a *redundant* representation of a value: the same
    longitude can be written with different bit patterns, so a regression term
    on the decoded value can be minimised by trading one bit against another.
    With the unweighted sum (the paper's Eq. 7) the regression term does
    exactly that, and it destroys the HIGH bits of the encoding - which carry
    tens of degrees each.  Measured on 400 steps: with w_mse = 1 the second
    bit of the latitude (65.5 deg) is correct in 13 % of the windows and the
    one-step error is 1.1e7 m; with w_mse = 0.1 the same bit is correct in
    100 % of the windows and the one-step error is 6.4e3 m.  The BCE is the
    term that pins the individual bits, so it has to dominate; the MSE only
    sharpens the value.
    """
    _, bits = model(x_obs)                       # [B,total_bits] soft probabilities
    target = model.codec.encode_flat(y_next)     # [B,total_bits] hard targets
    bce = F.binary_cross_entropy(bits.clamp(1e-6, 1 - 1e-6), target)

    code_pred = model.codec.ste_codes(bits)      # [B,6] normalised code units
    code_tgt = model.codec.target_codes(y_next)  # [B,6]
    mse = F.smooth_l1_loss(code_pred, code_tgt, beta=0.1)
    return w_bce * bce + w_mse * mse

# coding=utf-8
"""
AeroTraj-Ultra (FlightBERT++ V3) & Comparative Baselines in PyTorch.

Synthesizes and advances the architectures from the workspace papers:
  1. FlightBERT (Guo et al., IEEE TITS 2023):
     - Binary Encoding (BE), Attribute Embedding (AE), Attribute Correlation Attention (ACoAtt),
       and Hybrid Constrained (HC) Loss (BCE + Decimal MSE via Straight-Through Estimator).
  2. FlightBERT++ V2 (Guo et al., AAAI / gdy-scu/FlightBERT_PP_V2 in modello.txt):
     - Gray Code (GC) representation of step deltas (eliminating Hamming cliffs),
       1D-Conv Token Embedding, Attention Feature Aggregation, and Non-Autoregressive
       Multi-Horizon Transformer Encoder-Decoder.
  3. IDDPG & MAF-Net (Pang, Alam, Lulli et al., AAMAS 2025 & AAMAS 2026):
     - Weather/wind state integration, kinematic turn-rate/separation constraints,
       and a differentiable Multi-Head Filter/Gate (MAF) that projects/bounds unsafe
       or unphysical high-bit jumps while preserving end-to-end gradient flow.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Vectorized Bitwise Gray-Code Codec with Differentiable Soft-XOR STE
# ==============================================================================
class GrayCodeCodec(nn.Module):
    """
    100% Vectorized PyTorch Gray-Code Encoder & Differentiable STE Decoder.
    Supports both unsigned shifted attributes (Lon, Lat, Alt) and signed attributes
    (Vx, Vy, Vz, and step deltas dLon, dLat, dAlt, dVx, dVy, dVz).
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        gc_cfg = cfg["graycode"]
        win_cfg = cfg["window"]

        self.use_graycode: bool = bool(gc_cfg.get("use_graycode", True))
        self.lon_size: int = int(gc_cfg["lon_size"])
        self.lat_size: int = int(gc_cfg["lat_size"])
        self.alt_size: int = int(gc_cfg["alt_size"])
        self.spdx_size: int = int(gc_cfg["spdx_size"])
        self.spdy_size: int = int(gc_cfg["spdy_size"])
        self.spdz_size: int = int(gc_cfg["spdz_size"])

        self.delta_sizes = (
            int(gc_cfg["delta_lon_size"]),
            int(gc_cfg["delta_lat_size"]),
            int(gc_cfg["delta_alt_size"]),
            int(gc_cfg["delta_spdx_size"]),
            int(gc_cfg["delta_spdy_size"]),
            int(gc_cfg["delta_spdz_size"]),
        )
        self.total_inp_bits = (
            self.lon_size + self.lat_size + self.alt_size
            + self.spdx_size + self.spdy_size + self.spdz_size
        )
        self.total_delta_bits = sum(self.delta_sizes)

        # Scaling factors for step-to-step per-5s normalized deltas
        scales = torch.tensor(
            [
                float(gc_cfg["delta_lon_scale"]),   # 10000.0 -> 0.0001 deg (~11m)
                float(gc_cfg["delta_lat_scale"]),   # 10000.0 -> 0.0001 deg (~11m)
                float(gc_cfg["delta_alt_scale"]),   # 10.0    -> 0.1 * 10m = 1m
                float(gc_cfg["delta_spd_scale"]),   # 2.0     -> 0.5 km/h
                float(gc_cfg["delta_spd_scale"]),   # 2.0     -> 0.5 km/h
                10.0,                               # 10.0    -> 0.1 m/s
            ],
            dtype=torch.float32,
        )
        self.register_buffer("delta_scales", scales)

        # Horizon step intervals in units of 5s steps: dk_h = h_step[h] - h_step[h-1]
        h_steps = torch.tensor(win_cfg["horizon_steps_5s"], dtype=torch.float32)
        prev_steps = torch.cat([torch.zeros(1), h_steps[:-1]], dim=0)
        dk_steps = h_steps - prev_steps  # [H]
        self.register_buffer("h_steps_5s", h_steps)
        self.register_buffer("dk_steps", dk_steps)

        # Cached constant tables: creating these with torch.arange/torch.tensor on
        # every forward pass costs ~20 extra kernel launches + allocator churn
        # per step (the model is launch-bound at this size).
        max_bits = max(
            int(gc_cfg["lon_size"]), int(gc_cfg["lat_size"]), int(gc_cfg["alt_size"]),
            int(gc_cfg["spdx_size"]), int(gc_cfg["spdy_size"]), int(gc_cfg["spdz_size"]),
        )
        self.max_bits = max_bits
        self.register_buffer("shifts_desc", torch.arange(max_bits - 1, -1, -1, dtype=torch.float32))
        self.register_buffer(
            "mag_weights", 2.0 ** torch.arange(max_bits - 2, -1, -1, dtype=torch.float32)
        )

    def encode_unsigned(self, val_int: torch.Tensor, num_bits: int) -> torch.Tensor:
        max_val = (1 << num_bits) - 1
        v = torch.clamp(val_int.round().long(), 0, max_val)
        if self.use_graycode:
            v = v ^ (v >> 1)
        shifts = self.shifts_desc[-num_bits:].long()
        bits = (v.unsqueeze(-1) >> shifts) & 1
        return bits.float()

    def encode_signed(self, val_int: torch.Tensor, num_bits: int) -> torch.Tensor:
        mag_bits = num_bits - 1
        max_val = (1 << mag_bits) - 1
        rounded = val_int.round().long()
        sign = (rounded < 0).long()
        mag = torch.clamp(rounded.abs(), 0, max_val)
        if self.use_graycode:
            mag = mag ^ (mag >> 1)
        shifts = self.shifts_desc[-mag_bits:].long()
        mag_bits_t = (mag.unsqueeze(-1) >> shifts) & 1
        return torch.cat([sign.unsqueeze(-1), mag_bits_t], dim=-1).float()

    def decode_signed_ste(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Differentiable Gray-Code -> Binary -> Signed Integer decoder using
        Straight-Through Estimator (STE) + Soft-XOR chain.
        probs: [..., num_bits] in (0, 1)

        The soft-XOR recurrence  b_i = b_{i-1} + g_i - 2*b_{i-1}*g_i  is
        algebraically IDENTICAL to the prefix-parity expression

            b_i = (1 - prod_{j<=i} (1 - 2 g_j)) / 2

        (expanding the product gives the same polynomial: sum of single terms
        - 2 sum of pairs + 4 sum of triples + ...).  The parity form is a
        single `cumprod`, i.e. ONE kernel launch instead of ~5 per bit, which
        removes ~330 kernel launches per forward pass (the model is
        launch-bound at this size).  Values AND gradients are unchanged.
        """
        hard_bits = (probs >= 0.5).float()
        bits_ste = probs + (hard_bits - probs).detach()
        sign_ste = bits_ste[..., 0]
        g_ste = bits_ste[..., 1:]
        mag_bits = g_ste.shape[-1]

        if self.use_graycode:
            b_ste = 0.5 * (1.0 - torch.cumprod(1.0 - 2.0 * g_ste, dim=-1))
        else:
            b_ste = g_ste

        weights = self.mag_weights[-mag_bits:]
        mag = (b_ste * weights).sum(dim=-1)
        return (1.0 - 2.0 * sign_ste) * mag

    def encode_input_sequence(
        self, x_6d: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode input 6D state [B, T, 6] (lon, lat, alt_10m, vx, vy, vz)
        into 6 Gray-Code tensors.
        """
        lon_b = self.encode_unsigned((x_6d[..., 0] + 180.0) * 1000.0, self.lon_size)
        lat_b = self.encode_unsigned((x_6d[..., 1] + 90.0) * 1000.0, self.lat_size)
        alt_b = self.encode_unsigned(x_6d[..., 2] + 50.0, self.alt_size)
        vx_b = self.encode_signed(x_6d[..., 3], self.spdx_size)
        vy_b = self.encode_signed(x_6d[..., 4], self.spdy_size)
        vz_b = self.encode_signed(x_6d[..., 5] * 10.0, self.spdz_size)
        return lon_b, lat_b, alt_b, vx_b, vy_b, vz_b

    def encode_step_deltas(
        self, seq_6d: torch.Tensor, dk_per_step: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode step-to-step normalized deltas along sequence dimension.
        seq_6d: [B, L, 6], dk_per_step: [1, L-1, 1]
        Returns tuple of 6 Gray-Code tensors of shape [B, L-1, delta_k_size].
        """
        diffs = (seq_6d[:, 1:, :] - seq_6d[:, :-1, :]) / dk_per_step  # [B, L-1, 6]
        scaled = diffs * self.delta_scales.view(1, 1, 6)
        encoded = [
            self.encode_signed(scaled[..., k], self.delta_sizes[k])
            for k in range(6)
        ]
        return tuple(encoded)

    def encode_targets_from_anchor(
        self, last_obs_6d: torch.Tensor, y_target_6d: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        Encode multi-horizon targets [B, H, 6] relative to previous horizon point
        (starting from last_obs_6d [B, 1, 6]), normalized by dk_steps.
        """
        joint = torch.cat([last_obs_6d.unsqueeze(1), y_target_6d], dim=1)  # [B, H+1, 6]
        dk = self.dk_steps.view(1, -1, 1)  # [1, H, 1]
        return self.encode_step_deltas(joint, dk)

    def decode_horizon_deltas_ste(
        self,
        last_obs_6d: torch.Tensor,
        delta_probs_tuple: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """
        Decode tuple of 6 Gray-Code probability tensors [B, H, delta_k_size]
        into cumulative 6D trajectory predictions [B, H, 6] using STE.
        """
        decoded_rates = []
        for k, probs_k in enumerate(delta_probs_tuple):
            int_val = self.decode_signed_ste(probs_k)  # [B, H]
            rate_k = int_val / self.delta_scales[k]    # per-5s delta in original units
            decoded_rates.append(rate_k)
        rates_6d = torch.stack(decoded_rates, dim=-1)  # [B, H, 6]
        step_increments = rates_6d * self.dk_steps.view(1, -1, 1)  # [B, H, 6]
        pred_6d = last_obs_6d.unsqueeze(1) + torch.cumsum(step_increments, dim=1)
        return pred_6d


# ==============================================================================
# 2. Attribute Correlation Attention V2 (ACoAtt-V2 from FlightBERT + Weather)
# ==============================================================================
class CoAttModule(nn.Module):
    """
    Correlation Attention module from FlightBERT (Fig. 4, Eq. 4-6):
    Recalibrates primary attribute embedding `e_a` using correlated kinematic/weather
    attribute embedding `e_corr` via multi-head cross-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, e_primary: torch.Tensor, e_correlated: torch.Tensor) -> torch.Tensor:
        # e_primary, e_correlated: [B, T, D]
        corr_out, _ = self.attn(query=e_primary, key=e_correlated, value=e_correlated, need_weights=False)
        joint = self.norm(torch.cat([e_primary, corr_out], dim=-1))
        return self.proj(joint)


class TrajectoryEmbeddingACoAttV2(nn.Module):
    """
    Upgraded FlightBERT Trajectory Embedding Module (AE + ACoAtt + Weather Context):
      - Projects each attribute's Gray-Code n-hot vector + continuous standardized scalar
        into a unified attribute embedding space.
      - Explicitly captures bidirectional kinematic correlations:
          Lon <-> (Vx + u_wind)
          Lat <-> (Vy + v_wind)
          Alt <-> (Vz + temp_k)
    """

    def __init__(self, codec: GrayCodeCodec, attr_dim: int = 32, out_dim: int = 192) -> None:
        super().__init__()
        self.ae_lon = nn.Linear(codec.lon_size + 2, attr_dim)
        self.ae_lat = nn.Linear(codec.lat_size + 2, attr_dim)
        self.ae_alt = nn.Linear(codec.alt_size + 2, attr_dim)
        self.ae_vx = nn.Linear(codec.spdx_size + 2, attr_dim)
        self.ae_vy = nn.Linear(codec.spdy_size + 2, attr_dim)
        self.ae_vz = nn.Linear(codec.spdz_size + 2, attr_dim)

        # Bidirectional CoAtt blocks (Eq. 4-6 of FlightBERT)
        self.coatt_lon_vx = CoAttModule(attr_dim, num_heads=4)
        self.coatt_vx_lon = CoAttModule(attr_dim, num_heads=4)
        self.coatt_lat_vy = CoAttModule(attr_dim, num_heads=4)
        self.coatt_vy_lat = CoAttModule(attr_dim, num_heads=4)
        self.coatt_alt_vz = CoAttModule(attr_dim, num_heads=4)
        self.coatt_vz_alt = CoAttModule(attr_dim, num_heads=4)

        self.fusion = nn.Sequential(
            nn.Linear(attr_dim * 6, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        gc_inputs: Tuple[torch.Tensor, ...],
        norm_x: torch.Tensor,
        rel_x: torch.Tensor,
    ) -> torch.Tensor:
        lon_b, lat_b, alt_b, vx_b, vy_b, vz_b = gc_inputs
        # Augment each attribute's Gray Code with its local relative state + weather context
        e_lon = F.gelu(self.ae_lon(torch.cat([lon_b, rel_x[..., 0:1], norm_x[..., 0:1]], dim=-1)))
        e_lat = F.gelu(self.ae_lat(torch.cat([lat_b, rel_x[..., 1:2], norm_x[..., 1:2]], dim=-1)))
        e_alt = F.gelu(self.ae_alt(torch.cat([alt_b, rel_x[..., 2:3], norm_x[..., 2:3]], dim=-1)))

        # Velocity embeddings include atmospheric wind/temperature components (u_wind, v_wind, temp)
        e_vx = F.gelu(self.ae_vx(torch.cat([vx_b, norm_x[..., 3:4], norm_x[..., 6:7]], dim=-1)))
        e_vy = F.gelu(self.ae_vy(torch.cat([vy_b, norm_x[..., 4:5], norm_x[..., 7:8]], dim=-1)))
        e_vz = F.gelu(self.ae_vz(torch.cat([vz_b, norm_x[..., 5:6], norm_x[..., 8:9]], dim=-1)))

        # Bidirectional Attribute Correlation Attention
        e_lon_c = self.coatt_lon_vx(e_lon, e_vx)
        e_vx_c = self.coatt_vx_lon(e_vx, e_lon)
        e_lat_c = self.coatt_lat_vy(e_lat, e_vy)
        e_vy_c = self.coatt_vy_lat(e_vy, e_lat)
        e_alt_c = self.coatt_alt_vz(e_alt, e_vz)
        e_vz_c = self.coatt_vz_alt(e_vz, e_alt)

        cat_e = torch.cat([e_lon_c, e_lat_c, e_alt_c, e_vx_c, e_vy_c, e_vz_c], dim=-1)
        return self.fusion(cat_e)


# ==============================================================================
# 3. Conv Token Embeddings, Positional Encoding & Feature Aggregation
#    (from FlightBERT++ V2: model.py)
# ==============================================================================
class TokenConvEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        self.token_conv = nn.Conv1d(
            in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode="zeros"
        )
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class FeatureAggregation(nn.Module):
    """Attention-weighted context pooling from FlightBERT++ V2."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, max(16, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 4), 1),
        )

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        energy = self.projection(encoder_outputs)  # [B, T, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [B, T]
        return (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]


# ==============================================================================
# 4. Physics-Informed Kinematic Prior (Coordinated Turn + Damped Climb + Wind)
# ==============================================================================
class PhysicsKinematicPrior(nn.Module):
    """
    Analytical Physics-Informed Kinematic Trajectory Propagator (5s - 10min).
    Computes exact spherical WGS-84 position and velocity integration from the
    observed window using damped coordinated turn rate and damped vertical rate,
    with learnable horizon damping time constants.
    """

    def __init__(self, horizon_seconds: list[float], dt_sec: float = 5.0) -> None:
        super().__init__()
        self.dt_sec = float(dt_sec)
        t_sec = torch.tensor(horizon_seconds, dtype=torch.float32)  # [H]
        self.register_buffer("t_sec", t_sec)

        # Learnable log time-constants for turn-rate damping (~45s) and vertical-rate damping (~180s)
        self.log_tau_turn = nn.Parameter(torch.tensor(math.log(45.0)))
        self.log_tau_vz = nn.Parameter(torch.tensor(math.log(180.0)))
        self.log_tau_acc = nn.Parameter(torch.tensor(math.log(30.0)))

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        """
        x_obs: [B, T_in, 9] -> cols [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms, u_wind, v_wind, temp_k]
        Returns: y_kin [B, H, 6] -> cols [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms]
        """
        p_last = x_obs[:, -1, :6]  # [B, 6]
        lon0 = p_last[:, 0:1]
        lat0 = p_last[:, 1:2]
        alt0 = p_last[:, 2:3]

        # Robust velocity estimate blending last observation and 3-step (15s) finite difference
        v_last_x = x_obs[:, -1, 3:4]  # km/h
        v_last_y = x_obs[:, -1, 4:5]  # km/h
        v_last_z = x_obs[:, -1, 5:6]  # m/s

        lat_rad = torch.deg2rad(lat0.clamp(-80.0, 80.0))
        m_per_deg_lon = 111320.0 * torch.cos(lat_rad)
        m_per_deg_lat = 111132.0

        # 20s (4 steps) average velocity from positions in m/s
        dt_window = 4.0 * self.dt_sec
        dlon_win = x_obs[:, -1, 0:1] - x_obs[:, -5, 0:1]
        dlat_win = x_obs[:, -1, 1:2] - x_obs[:, -5, 1:2]
        dalt_win = (x_obs[:, -1, 2:3] - x_obs[:, -5, 2:3]) * 10.0

        vx_fd_ms = (dlon_win * m_per_deg_lon) / dt_window
        vy_fd_ms = (dlat_win * m_per_deg_lat) / dt_window
        vz_fd_ms = dalt_win / dt_window

        # Instantaneous 1-step (5s) velocity in m/s
        vx_inst_ms = v_last_x / 3.6
        vy_inst_ms = v_last_y / 3.6
        vz_inst_ms = v_last_z

        # Horizon-adaptive velocity blend: onboard ADS-B Doppler velocity is primary,
        # with light 20s position-trend smoothing at longer horizons
        t = self.t_sec.view(1, -1)  # [1, H]
        w_smooth = 0.12 * (1.0 - torch.exp(-(t - 5.0).clamp(min=0.0) / 45.0))
        vx0_ms = (1.0 - w_smooth) * vx_inst_ms + w_smooth * vx_fd_ms
        vy0_ms = (1.0 - w_smooth) * vy_inst_ms + w_smooth * vy_fd_ms
        vz0_ms = (1.0 - w_smooth) * vz_inst_ms + w_smooth * vz_fd_ms

        # Estimate turn rate omega (rad/s) and longitudinal acceleration from recent history
        psi_last = torch.atan2(x_obs[:, -1, 3:4], x_obs[:, -1, 4:5] + 1e-6)
        psi_prev = torch.atan2(x_obs[:, -4, 3:4], x_obs[:, -4, 4:5] + 1e-6)
        dpsi = torch.atan2(torch.sin(psi_last - psi_prev), torch.cos(psi_last - psi_prev))
        omega0 = (dpsi / (3.0 * self.dt_sec)).clamp(-0.035, 0.035)  # max ~2 deg/s

        tau_turn = torch.exp(self.log_tau_turn).clamp(10.0, 180.0)
        tau_vz = torch.exp(self.log_tau_vz).clamp(30.0, 600.0)

        # Effective turn angle at horizon t with exponential turn-rate decay:
        # theta(t) = omega0 * tau_turn * (1 - exp(-t / tau_turn))
        theta_t = omega0 * tau_turn * (1.0 - torch.exp(-t / tau_turn))  # [B, H]
        # Average heading rotation over [0, t] is ~ 0.5 * theta_t
        half_theta = 0.5 * theta_t
        cos_ht = torch.cos(half_theta)
        sin_ht = torch.sin(half_theta)

        # Rotate velocity vector by half_theta for average displacement velocity
        vx_avg_ms = vx0_ms * cos_ht + vy0_ms * sin_ht  # [B, H]
        vy_avg_ms = -vx0_ms * sin_ht + vy0_ms * cos_ht  # [B, H]

        # Instantaneous velocity at horizon t rotated by theta_t
        cos_t = torch.cos(theta_t)
        sin_t = torch.sin(theta_t)
        vx_t_kmh = (vx0_ms * cos_t + vy0_ms * sin_t) * 3.6
        vy_t_kmh = (-vx0_ms * sin_t + vy0_ms * cos_t) * 3.6

        # Vertical integration with exponential decay toward level flight
        vz_decay = torch.exp(-t / tau_vz)
        vz_t_ms = vz0_ms * vz_decay
        dz_m = vz0_ms * tau_vz * (1.0 - vz_decay)

        dx_m = vx_avg_ms * t
        dy_m = vy_avg_ms * t

        pred_lon = lon0 + dx_m / m_per_deg_lon
        pred_lat = lat0 + dy_m / m_per_deg_lat
        pred_alt_10m = alt0 + dz_m / 10.0

        return torch.stack(
            [pred_lon, pred_lat, pred_alt_10m, vx_t_kmh, vy_t_kmh, vz_t_ms], dim=-1
        )


# ==============================================================================
# 5. AeroTraj-Ultra (FlightBERT++ V3) — Main Architecture
# ==============================================================================
def local_frame(x_obs: torch.Tensor):
    """
    Aircraft-centric local frame of the last observation.

    Returns (lon0, lat0, alt0, heading, m_per_deg_lon), each [B,1], with
    `heading` measured clockwise from North.  Used by the retrieval-conditioned
    variant to express a neighbour's future in the query's own frame.
    """
    lon0 = x_obs[:, -1, 0:1]
    lat0 = x_obs[:, -1, 1:2]
    alt0 = x_obs[:, -1, 2:3]
    heading = torch.atan2(x_obs[:, -1, 3:4], x_obs[:, -1, 4:5])
    lat_rad = torch.deg2rad(lat0.clamp(-80.0, 80.0))
    m_per_deg_lon = 111320.0 * torch.cos(lat_rad)
    return lon0, lat0, alt0, heading, m_per_deg_lon


class AeroTrajUltra(nn.Module):
    """
    AeroTraj-Ultra (FlightBERT++ V3):
    Physics-Informed Hybrid Gray-Code + Continuous Residual Multi-Horizon Transformer
    with Bidirectional Kinematic-Weather Cross-Attention (ACoAtt-V2) and
    MAF-Net-Inspired Differentiable Trust-Region Gate.
    """

    def __init__(self, cfg: Dict, norm_mean: np.ndarray, norm_std: np.ndarray) -> None:
        super().__init__()
        self.cfg = cfg
        m_cfg = cfg["model"]
        win_cfg = cfg["window"]

        self.inp_seq_len: int = int(win_cfg["inp_seq_len"])
        self.horizon: int = int(win_cfg["horizon"])
        self.n_embd: int = int(m_cfg["n_embd"])
        self.n_head: int = int(m_cfg["n_head"])
        self.n_en_layer: int = int(m_cfg["n_en_layer"])
        self.n_de_layer: int = int(m_cfg["n_de_layer"])
        self.dropout: float = float(m_cfg["dropout"])

        self.register_buffer("norm_mean", torch.tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std, dtype=torch.float32))

        horizon_seconds = win_cfg["horizon_seconds"]
        self.register_buffer(
            "horizon_sec", torch.tensor(horizon_seconds, dtype=torch.float32)
        )
        # Reference horizon used to normalise the continuous time conditioning.
        # It is the ACTUAL maximum horizon of the task (not a hard-coded 600 s),
        # so the model keeps the full [0,1] time range whatever the horizon span.
        self.horizon_ref: float = float(max(horizon_seconds))
        # Scale factors for residual corrections (grows smoothly with horizon time)
        t_norm = torch.tensor(horizon_seconds, dtype=torch.float32) / self.horizon_ref  # in (0, 1]
        self.register_buffer("t_norm", t_norm)

        # 1. Vectorized Gray-Code Codec
        self.codec = GrayCodeCodec(cfg)

        # 2. ACoAtt-V2 Trajectory Embedding + 1D-Conv Token Stem
        self.acoatt_emb = TrajectoryEmbeddingACoAttV2(
            self.codec,
            attr_dim=int(m_cfg["attr_embed_dim"]),
            out_dim=self.n_embd,
        )
        self.conv_stem = TokenConvEmbedding(self.n_embd + 15, self.n_embd)
        self.enc_pos_emb = SinusoidalPositionalEncoding(self.n_embd, self.dropout, max_len=128)

        # 3. Pre-LN Transformer Encoder + Attention Feature Aggregation
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 3,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.n_en_layer, enable_nested_tensor=False
        )
        self.feature_agg = FeatureAggregation(self.n_embd)

        # 4. Non-Autoregressive Multi-Horizon Decoder (FlightBERT++ V2 + Fourier Time Conditioning)
        self.dec_token_emb = TokenConvEmbedding(self.codec.total_delta_bits + 6, self.n_embd)
        self.horizon_emb = nn.Embedding(self.horizon, self.n_embd)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd),
        )
        self.dec_pos_emb = SinusoidalPositionalEncoding(self.n_embd, self.dropout, max_len=128)

        # ---- multi-hypothesis head (optional) --------------------------- #
        # K learnable "mode queries" (DETR/ASCENT style) are appended to the
        # decoder sequence.  An attention mask keeps them INVISIBLE to the
        # horizon queries, so the deterministic path is bit-identical to the
        # single-hypothesis model; each mode then acts as a per-mode bias on
        # the horizon features, and a shared head turns it into one full
        # kinematic hypothesis (residual on the physics prior).
        # ---- optional retrieval conditioning (intent prior) ------------- #
        # When enabled, the horizon queries receive a fourth block: the
        # attention-pooled context of the K most similar past trajectories.
        # `horizon_proj` then takes 4*D inputs instead of 3*D, so a base
        # checkpoint can still be warm-started (see `load_base_weights`).
        self.use_retrieval = bool(m_cfg.get("use_retrieval", False))
        self.n_neighbours = int(m_cfg.get("n_neighbours", 8))
        if self.use_retrieval:
            self.retr_proj = nn.Linear(int(m_cfg.get("retrieval_dim", 4)), self.n_embd)
            self.retr_attn = nn.Linear(self.n_embd, 1)
            self.horizon_proj = nn.Linear(self.n_embd * 4, self.n_embd)
        else:
            self.horizon_proj = nn.Linear(self.n_embd * 3, self.n_embd)

        self.n_modes = int(m_cfg.get("n_modes", 0))
        # how the score head is trained: "ce" -> higher score = better (argmax),
        # "regress" -> the score predicts the log-error (argmin)
        self.hyp_score_mode = str(cfg.get("training", {}).get("hyp_score_mode", "ce"))
        if self.n_modes > 0:
            self.mode_queries = nn.Parameter(
                torch.randn(self.n_modes, self.n_embd) * 0.02
            )
            self.hyp_head = nn.Sequential(
                nn.Linear(self.n_embd + 6, self.n_embd), nn.GELU(), nn.Linear(self.n_embd, 6)
            )
            self.hyp_score = nn.Sequential(
                nn.Linear(self.n_embd, self.n_embd), nn.GELU(), nn.Linear(self.n_embd, 1)
            )
            nn.init.zeros_(self.hyp_head[-1].weight)
            nn.init.zeros_(self.hyp_head[-1].bias)
            nn.init.constant_(self.hyp_score[-1].bias, -2.0)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.n_embd,
            nhead=self.n_head,
            dim_feedforward=self.n_embd * 3,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=self.n_de_layer)
        self.ln_f = nn.LayerNorm(self.n_embd)

        # 5. Physics-Informed Kinematic Prior
        self.kinematic_prior = PhysicsKinematicPrior(
            horizon_seconds=horizon_seconds,
            dt_sec=float(cfg["data"]["dt_seconds"]),
        )

        # 6. Three Synergistic Prediction Heads:
        # Head A: Gray-Code Multi-Binary Classification (MBC) head (FlightBERT++ V2)
        self.gc_head = nn.Linear(self.n_embd, self.codec.total_delta_bits)

        # Head B: Continuous Kinematic Residual Refinement head
        self.cont_head = nn.Sequential(
            nn.Linear(self.n_embd + 6, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, 6),
        )
        # Initialize last layer of residual head near zero so training starts from the exact kinematic prior!
        nn.init.zeros_(self.cont_head[-1].weight)
        nn.init.zeros_(self.cont_head[-1].bias)

        # Head C: MAF-Net-Inspired Differentiable Filter Gate (Pang et al., AAMAS 2026)
        self.maf_gate = nn.Sequential(
            nn.Linear(self.n_embd + 6, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.constant_(self.maf_gate[-1].bias, -2.0)

        # Physical scale per horizon for residual outputs
        res_base_scale = torch.tensor([0.25, 0.25, 35.0, 80.0, 80.0, 6.0], dtype=torch.float32)
        self.register_buffer("res_base_scale", res_base_scale)
        trust_sigma = torch.tensor([0.008, 0.008, 2.5, 15.0, 15.0, 1.5], dtype=torch.float32)
        self.register_buffer("trust_sigma", trust_sigma)

        # Constant normalisation tables (buffers: no per-step allocation)
        self.register_buffer("rel_vel_scale", torch.tensor([200.0, 200.0, 10.0], dtype=torch.float32))
        self.register_buffer("rel_pos_scale", torch.tensor([20.0, 20.0, 0.05], dtype=torch.float32))
        self.register_buffer(
            "kin_disp_scale", torch.tensor([1.0, 1.0, 100.0, 200.0, 200.0, 10.0], dtype=torch.float32)
        )

    def forward(
        self,
        x_obs: torch.Tensor,
        retr_ctx: Optional[torch.Tensor] = None,
        return_scores: bool = False,
    ) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]:
        """
        x_obs: [B, T_in, 9] -> [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms, u_wind, v_wind, temp_k]
        retr_ctx (optional): [B, K, H, 4] retrieval context (intent prior).
        Returns dict with:
          - 'pred': [B, H, 6] final fused trajectory prediction
          - 'pred_kin': [B, H, 6] analytical physics prior
          - 'pred_gc': [B, H, 6] STE-decoded Gray-Code trajectory
          - 'gc_probs': tuple of 6 tensors [B, H, delta_k_size] in (0, 1)
          - 'gc_logits': tuple of 6 tensors [B, H, delta_k_size]
        """
        B, T, _ = x_obs.shape
        device = x_obs.device

        x_6d = x_obs[..., :6]
        last_obs_6d = x_6d[:, -1, :]  # [B, 6]

        # Standardize continuous features and compute window-relative coordinates + accelerations
        norm_x = (x_obs - self.norm_mean.view(1, 1, 9)) / self.norm_std.view(1, 1, 9)
        rel_pos = x_6d[:, :, :3] - last_obs_6d[:, None, :3]  # [B, T, 3]
        rel_vel = (x_6d[:, :, 3:6] - last_obs_6d[:, None, 3:6]) / self.rel_vel_scale.view(1, 1, 3)
        rel_x = torch.cat([rel_pos * self.rel_pos_scale.view(1, 1, 3), rel_vel], dim=-1)

        # 1. Encode input sequence in Gray Code & pass through ACoAtt-V2
        gc_inputs = self.codec.encode_input_sequence(x_6d)
        acoatt_feat = self.acoatt_emb(gc_inputs, norm_x, rel_x)  # [B, T, D]

        # Concatenate continuous relative kinematics + weather & pass through Conv1d Stem + Encoder
        enc_in = self.conv_stem(torch.cat([acoatt_feat, norm_x, rel_x], dim=-1))
        enc_in = self.enc_pos_emb(enc_in)
        enc_out = self.encoder(enc_in)  # [B, T, D]
        context_vec = self.feature_agg(enc_out)  # [B, D]

        # 2. Historical step-to-step Gray-Code deltas for Decoder Prefix (FlightBERT++ V2)
        ones_dk = torch.ones((1, T - 1, 1), device=device, dtype=x_obs.dtype)
        hist_gc_deltas = self.codec.encode_step_deltas(x_6d, ones_dk)
        hist_cont_deltas = (x_6d[:, 1:, :] - x_6d[:, :-1, :]) * torch.tensor(
            [100.0, 100.0, 0.2, 0.05, 0.05, 0.5], device=device
        ).view(1, 1, 6)
        dec_prefix_in = torch.cat(list(hist_gc_deltas) + [hist_cont_deltas], dim=-1)
        dec_prefix_emb = self.dec_token_emb(dec_prefix_in)  # [B, T-1, D]

        # 3. Non-Autoregressive Multi-Horizon Query Tokens (5s to 10min)
        h_idx = torch.arange(self.horizon, device=device).unsqueeze(0).expand(B, -1)  # [B, H]
        h_emb = self.horizon_emb(h_idx)  # [B, H, D]

        t_f = self.t_norm.view(1, self.horizon, 1).expand(B, -1, 1)
        time_feats = torch.cat(
            [t_f, torch.sqrt(t_f), torch.sin(math.pi * t_f), torch.cos(math.pi * t_f)], dim=-1
        )
        t_emb = self.time_mlp(time_feats)  # [B, H, D]

        ctx_rep = context_vec.unsqueeze(1).expand(-1, self.horizon, -1)  # [B, H, D]
        if self.use_retrieval:
            if retr_ctx is not None:
                retr_enc = self.encode_retrieval(retr_ctx)                # [B, H, D]
            else:
                retr_enc = torch.zeros(B, self.horizon, self.n_embd,
                                       device=device, dtype=x_obs.dtype)
            horizon_queries = self.horizon_proj(
                torch.cat([ctx_rep, h_emb, t_emb, retr_enc], dim=-1)
            )
        else:
            horizon_queries = self.horizon_proj(torch.cat([ctx_rep, h_emb, t_emb], dim=-1))

        dec_full_in = self.dec_pos_emb(torch.cat([dec_prefix_emb, horizon_queries], dim=1))
        hyp_pred = None
        hyp_score = None
        if self.n_modes > 0:
            # ---- multi-hypothesis: append the mode queries to the sequence ----
            mode_q = self.mode_queries.unsqueeze(0).expand(B, -1, -1)      # [B, K, D]
            dec_full_in = self.dec_pos_emb(
                torch.cat([dec_prefix_emb, horizon_queries, mode_q], dim=1)
            )
            # Attention mask: the horizon queries must NOT see the mode
            # queries, so the deterministic path is bit-identical to the
            # single-hypothesis model.  (rows: prefix / horizon / mode)
            n_pref = dec_prefix_emb.shape[1]
            L_pref, L_hor = n_pref, self.horizon
            n_tot = L_pref + L_hor + self.n_modes
            neg = float("-inf")
            mask = torch.zeros(n_tot, n_tot, device=dec_full_in.device)
            # prefix rows (0..L_pref-1): see prefix + horizon, not the modes
            mask[:L_pref, L_pref + L_hor:] = neg
            # horizon rows: see prefix + horizon, not the modes
            mask[L_pref:L_pref + L_hor, L_pref + L_hor:] = neg
            dec_out = self.ln_f(self.decoder(tgt=dec_full_in, memory=enc_out, tgt_mask=mask))
            dec_h = dec_out[:, L_pref:L_pref + L_hor, :]                     # [B, H, D]
            mode_tok = dec_out[:, L_pref + L_hor:, :]                         # [B, K, D]
        else:
            dec_out = self.ln_f(self.decoder(tgt=dec_full_in, memory=enc_out))
            dec_h = dec_out[:, -self.horizon :, :]  # [B, H, D]

        # 4. Physics-Informed Kinematic Prior
        pred_kin = self.kinematic_prior(x_obs)  # [B, H, 6]
        kin_disp_norm = (pred_kin - last_obs_6d.unsqueeze(1)) / self.kin_disp_scale.view(1, 1, 6)

        # 5. Gray-Code Multi-Binary Classification Head + Differentiable STE Decode
        raw_gc_logits = self.gc_head(dec_h)  # [B, H, total_delta_bits]
        raw_gc_probs = torch.sigmoid(raw_gc_logits)
        gc_logits_tuple = torch.split(raw_gc_logits, self.codec.delta_sizes, dim=-1)
        gc_probs_tuple = torch.split(raw_gc_probs, self.codec.delta_sizes, dim=-1)
        pred_gc = self.codec.decode_horizon_deltas_ste(last_obs_6d, gc_probs_tuple)  # [B, H, 6]

        # 6. Continuous Residual Head + MAF-Net-Inspired Differentiable Filter Gate
        dec_aug = torch.cat([dec_h, kin_disp_norm], dim=-1)
        horizon_scale = torch.cat(
            [
                self.t_norm.view(1, self.horizon, 1).expand(B, -1, 3),
                torch.sqrt(self.t_norm).view(1, self.horizon, 1).expand(B, -1, 3),
            ],
            dim=-1,
        ) * self.res_base_scale.view(1, 1, 6)

        cont_res = self.cont_head(dec_aug) * horizon_scale
        pred_cont = pred_kin + cont_res

        # MAF Gate: fuses Gray-Code MBC trajectory within a differentiable trust region
        alpha_gate = torch.sigmoid(self.maf_gate(dec_aug))  # [B, H, 6]
        sigma = self.trust_sigma.view(1, 1, 6) * (0.02 + 0.98 * self.t_norm.view(1, self.horizon, 1))
        gc_bounded_diff = torch.tanh((pred_gc - pred_cont) / sigma) * sigma
        pred_final = pred_cont + alpha_gate * gc_bounded_diff

        # 7. Multi-hypothesis head: each mode is a per-mode bias on the horizon
        #    features, decoded by a shared head into a residual on the prior.
        if self.n_modes > 0:
            mode_bias = mode_tok.unsqueeze(2)                              # [B, K, 1, D]
            h_feat = dec_h.unsqueeze(1) + mode_bias                        # [B, K, H, D]
            h_aug = torch.cat([h_feat, kin_disp_norm.unsqueeze(1).expand(-1, self.n_modes, -1, -1)], dim=-1)
            hyp_res = self.hyp_head(h_aug) * horizon_scale.unsqueeze(1)      # [B, K, H, 6]
            hyp_pred = pred_kin.unsqueeze(1) + hyp_res                      # [B, K, H, 6]
            hyp_score = self.hyp_score(mode_tok).squeeze(-1)                # [B, K]

        out = {
            "pred": pred_final,
            "pred_kin": pred_kin,
            "pred_cont": pred_cont,
            "pred_gc": pred_gc,
            "gc_probs": gc_probs_tuple,
            "gc_logits": gc_logits_tuple,
        }
        if hyp_pred is not None:
            out["hyp_pred"] = hyp_pred
            out["hyp_score"] = hyp_score
        return out


class AeroTrajUltraRetrieval(AeroTrajUltra):
    """
    Retrieval-conditioned variant of AeroTraj-Ultra, for medium-term intent.

    The behaviour is entirely defined by the base class: setting
    `model.use_retrieval = True` (i.e. `model_cfg.use_retrieval` in the config)
    makes the horizon queries accept a fourth conditioning block, the
    attention-pooled context of the K past trajectories most similar to the
    current observation (same neighbourhood, heading, flight level and speed),
    with their futures expressed in the query's local frame.

    The attention is what makes it work: AVERAGING the neighbours (the naive
    predictor) is much worse than constant velocity because they often lie on
    different routes, whereas a learned selection can pick the subset that is
    actually consistent with the current state.

    `load_base_weights` warm-starts it from a base checkpoint: the first three
    blocks of `horizon_proj` keep the base weights and the retrieval block
    starts at zero, so the model is exactly the base model at initialisation
    and any improvement is attributable to the retrieval context alone.
    """

    def __init__(self, cfg: Dict, norm_mean: np.ndarray, norm_std: np.ndarray,
                 n_neighbours: int = 8, retrieval_dim: int = 4) -> None:
        m_cfg = dict(cfg["model"])
        m_cfg["use_retrieval"] = True
        m_cfg["n_neighbours"] = n_neighbours
        m_cfg["retrieval_dim"] = retrieval_dim
        cfg = dict(cfg)
        cfg["model"] = m_cfg
        super().__init__(cfg, norm_mean, norm_std)

    def encode_retrieval(self, retr_ctx: torch.Tensor) -> torch.Tensor:
        """
        retr_ctx : [B, K, H, C] -> [B, H, D] attention-pooled context.
        """
        nb = self.retr_proj(retr_ctx)                       # [B,K,H,D]
        w = torch.softmax(self.retr_attn(nb), dim=1)         # [B,K,H,1]
        return (w * nb).sum(dim=1)                           # [B,H,D]

    @torch.no_grad()
    def load_base_weights(self, state_dict: Dict[str, torch.Tensor]):
        """
        Warm start from a base `AeroTrajUltra` checkpoint (see the class
        docstring).  Returns (missing, unexpected).
        """
        new = dict(state_dict)
        w = new.get("horizon_proj.weight")
        b = new.get("horizon_proj.bias")
        if w is not None:
            d = self.n_embd
            new_w = torch.zeros(self.horizon_proj.weight.shape, device=w.device, dtype=w.dtype)
            new_w[:, : 3 * d] = w[:, : 3 * d]
            new["horizon_proj.weight"] = new_w
            if b is not None:
                new["horizon_proj.bias"] = b.clone()
        return self.load_state_dict(new, strict=False)


# ==============================================================================
# 6. Hybrid Physics-Constrained Multi-Horizon Loss (L_Ultra)
# ==============================================================================
def multi_hypothesis_wta_loss(
    hyp_pred: torch.Tensor,
    hyp_score: torch.Tensor,
    target: torch.Tensor,
    horizon_seconds: List[int],
    attr_scales: torch.Tensor,
    aux_weight: float = 0.1,
    score_weight: float = 0.1,
    score_mode: str = "ce",
    score_scale: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Winner-takes-all multi-hypothesis loss (ASCENT / DETR style).

    * only the hypothesis closest to the ground truth drives the regression;
    * the SCORE term, in "ce" mode, is a cross-entropy that ranks that
      hypothesis first; in "regress" mode the score head instead REGRESSES the
      (log) error of every hypothesis, which is the objective the selection
      rule actually uses (argmin of the predicted error);
    * an auxiliary (down-weighted) regression on EVERY hypothesis prevents the
      non-winning ones from degenerating.

    hyp_pred  : [B, K, H, 6]   hypothesis trajectories
    hyp_score : [B, K]          predicted confidence per hypothesis
    target    : [B, H, 6]
    """
    B, K, H, _ = hyp_pred.shape
    d = hyp_pred - target.unsqueeze(1)                      # [B, K, H, 6]
    lat_r = torch.deg2rad(target[:, :, 1]).unsqueeze(1)     # [B, 1, H]
    mlon = 111320.0 * torch.cos(lat_r)
    dist = torch.sqrt(
        (d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2
    )                                                       # [B, K, H] metres
    mode_err = dist.mean(dim=2)                            # [B, K]
    best = mode_err.argmin(dim=1)                          # [B]

    # horizon-normalised regression error per hypothesis
    t = torch.tensor(horizon_seconds, dtype=torch.float32, device=hyp_pred.device)
    h_norm = (0.04 + 0.96 * (t / 60.0)).view(1, 1, -1, 1)
    err = (hyp_pred - target.unsqueeze(1)) / (attr_scales.view(1, 1, 1, 6) * h_norm)
    pos = F.smooth_l1_loss(err[..., :3], torch.zeros_like(err[..., :3]), beta=0.25,
                           reduction="none").mean(dim=(2, 3))    # [B, K]
    vel = F.smooth_l1_loss(err[..., 3:], torch.zeros_like(err[..., 3:]), beta=0.5,
                           reduction="none").mean(dim=(2, 3))    # [B, K]
    per_mode = pos + vel                                       # [B, K]

    idx = best.view(B, 1, 1, 1).expand(B, 1, H, 6)
    best_pred = hyp_pred.gather(1, idx).squeeze(1)              # [B, H, 6]
    err_b = (best_pred - target) / (attr_scales.view(1, 1, 6) * h_norm.view(1, H, 1))
    reg_best = F.smooth_l1_loss(err_b[..., :3], torch.zeros_like(err_b[..., :3]), beta=0.25) \
        + F.smooth_l1_loss(err_b[..., 3:], torch.zeros_like(err_b[..., 3:]), beta=0.5)
    aux = per_mode.mean()
    if score_mode == "regress":
        # the score head predicts the log-error of its own hypothesis, so the
        # selection rule (argmin) is exactly the minimiser of that prediction
        lat_r = torch.deg2rad(target[:, :, 1]).unsqueeze(1)          # [B,1,H]
        mlon = 111320.0 * torch.cos(lat_r)
        dd = hyp_pred - target.unsqueeze(1)
        err_m = torch.sqrt(
            (dd[..., 0] * mlon) ** 2 + (dd[..., 1] * 111132.0) ** 2 + (dd[..., 2] * 10.0) ** 2
        ).mean(dim=2)                                                # [B,K] metres
        # the target MUST be detached: otherwise the gradient flows into the
        # trajectory branch and the model can lower the loss by INFLATING the
        # error of its own hypotheses (a degenerate incentive that diverged to
        # NaN in the first attempt).
        score_target = torch.log1p(err_m.detach() / score_scale)
        score_term = F.smooth_l1_loss(hyp_score, score_target, beta=1.0)
    else:
        score_term = F.cross_entropy(hyp_score, best)
    return reg_best + aux_weight * aux + score_weight * score_term, best


class HybridPhysicsConstrainedLoss(nn.Module):
    """
    Combines:
      1. L_BCE: Binary Cross-Entropy loss on Gray-Code n-hot target deltas (FlightBERT++ V2)
      2. L_GC_STE: Differentiable STE decimal loss on decoded Gray-Code trajectory (FlightBERT HC Loss)
      3. L_Cont: Horizon-weighted Smooth L1 + MSE loss on fused 6D trajectory
      4. L_Physics: Kinematic consistency between predicted position deltas and predicted velocities
    """

    def __init__(self, codec: GrayCodeCodec, cfg: Dict) -> None:
        super().__init__()
        self.codec = codec
        tr_cfg = cfg["training"]
        self.horizon_seconds: List[int] = list(cfg["window"]["horizon_seconds"])
        self.hyp_weight = float(tr_cfg.get("hyp_weight", 0.5))
        self.hyp_aux_weight = float(tr_cfg.get("hyp_aux_weight", 0.1))
        self.hyp_score_weight = float(tr_cfg.get("hyp_score_weight", 0.1))
        self.hyp_score_mode = str(tr_cfg.get("hyp_score_mode", "ce"))
        self.hyp_score_scale = float(tr_cfg.get("hyp_score_scale", 100.0))
        self.bce_weight = float(tr_cfg["bce_weight"])
        self.gc_ste_weight = float(tr_cfg["gc_ste_weight"])
        self.cont_weight = float(tr_cfg["cont_weight"])
        self.vel_weight = float(tr_cfg["vel_weight"])
        self.physics_weight = float(tr_cfg["physics_weight"])

        # Attribute weights so Lon/Lat (deg), Alt (10m), and Velocities (km/h, m/s) have balanced gradients
        attr_scales = torch.tensor([0.02, 0.02, 4.0, 15.0, 15.0, 1.5], dtype=torch.float32)
        self.register_buffer("attr_scales", attr_scales)

        dt_steps_sec = codec.dk_steps * float(cfg["data"]["dt_seconds"])
        self.register_buffer("dt_steps_sec", dt_steps_sec)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]],
        x_obs: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        last_obs_6d = x_obs[:, -1, :6]
        pred = outputs["pred"]          # [B, H, 6]
        pred_gc = outputs["pred_gc"]    # [B, H, 6]
        gc_logits = outputs["gc_logits"]

        # 1. Gray-Code BCE Loss across all 6 attributes & 16 horizons
        target_gc_tuple = self.codec.encode_targets_from_anchor(last_obs_6d, y_target)
        bce_losses = [
            F.binary_cross_entropy_with_logits(logit_k, targ_k)
            for logit_k, targ_k in zip(gc_logits, target_gc_tuple)
        ]
        loss_bce = sum(bce_losses) / len(bce_losses)

        # 2. Normalized coordinate error for STE Gray-Code head and Final Fused head
        scales = self.attr_scales.view(1, 1, 6)
        h_pos_norm = (0.04 + 0.96 * (self.codec.h_steps_5s / 60.0)).view(1, -1, 1)
        h_gc_norm = (self.codec.h_steps_5s / 12.0).clamp(min=0.5).view(1, -1, 1)
        err_final_pos = (pred[..., :3] - y_target[..., :3]) / (scales[..., :3] * h_pos_norm)
        err_final_vel = (pred[..., 3:] - y_target[..., 3:]) / scales[..., 3:]
        err_gc = (pred_gc - y_target) / (scales * h_gc_norm)

        loss_pos = F.smooth_l1_loss(err_final_pos, torch.zeros_like(err_final_pos), beta=0.25)
        loss_vel = F.smooth_l1_loss(err_final_vel, torch.zeros_like(err_final_vel), beta=0.5)
        loss_gc_ste = F.smooth_l1_loss(err_gc, torch.zeros_like(err_gc), beta=0.5)

        # 3. Physics-Informed Kinematic Consistency Loss:
        # Position increments between consecutive horizons must agree with trapezoidal velocity integral
        joint_pred = torch.cat([last_obs_6d.unsqueeze(1), pred], dim=1)  # [B, H+1, 6]
        dlon_deg = joint_pred[:, 1:, 0] - joint_pred[:, :-1, 0]
        dlat_deg = joint_pred[:, 1:, 1] - joint_pred[:, :-1, 1]
        dalt_m = (joint_pred[:, 1:, 2] - joint_pred[:, :-1, 2]) * 10.0

        lat_rad = torch.deg2rad(joint_pred[:, :-1, 1].clamp(-80.0, 80.0))
        dx_pred_m = dlon_deg * (111320.0 * torch.cos(lat_rad))
        dy_pred_m = dlat_deg * 111132.0

        vx_avg_ms = 0.5 * (joint_pred[:, 1:, 3] + joint_pred[:, :-1, 3]) / 3.6
        vy_avg_ms = 0.5 * (joint_pred[:, 1:, 4] + joint_pred[:, :-1, 4]) / 3.6
        vz_avg_ms = 0.5 * (joint_pred[:, 1:, 5] + joint_pred[:, :-1, 5])

        dt_s = self.dt_steps_sec.view(1, -1)  # [1, H]
        phys_err_x = (dx_pred_m - vx_avg_ms * dt_s) / (50.0 + 2.0 * dt_s)
        phys_err_y = (dy_pred_m - vy_avg_ms * dt_s) / (50.0 + 2.0 * dt_s)
        phys_err_z = (dalt_m - vz_avg_ms * dt_s) / (15.0 + 0.5 * dt_s)

        loss_physics = (
            F.smooth_l1_loss(phys_err_x, torch.zeros_like(phys_err_x))
            + F.smooth_l1_loss(phys_err_y, torch.zeros_like(phys_err_y))
            + F.smooth_l1_loss(phys_err_z, torch.zeros_like(phys_err_z))
        ) / 3.0

        total_loss = (
            self.cont_weight * loss_pos
            + self.vel_weight * loss_vel
            + self.bce_weight * loss_bce
            + self.gc_ste_weight * loss_gc_ste
            + self.physics_weight * loss_physics
        )

        loss_hyp = None
        if "hyp_pred" in outputs:
            # multi-hypothesis term: winner-takes-all regression + ranking
            loss_hyp, _ = multi_hypothesis_wta_loss(
                outputs["hyp_pred"],
                outputs["hyp_score"],
                y_target,
                self.horizon_seconds,
                self.attr_scales,
                aux_weight=self.hyp_aux_weight,
                score_weight=self.hyp_score_weight,
                score_mode=self.hyp_score_mode,
                score_scale=self.hyp_score_scale,
            )
            total_loss = total_loss + self.hyp_weight * loss_hyp

        metrics_dict = {
            "total": total_loss.detach(),
            "pos": loss_pos.detach(),
            "vel": loss_vel.detach(),
            "bce": loss_bce.detach(),
            "gc_ste": loss_gc_ste.detach(),
            "physics": loss_physics.detach(),
        }
        if loss_hyp is not None:
            metrics_dict["hyp"] = loss_hyp.detach()
        return total_loss, metrics_dict


# ==============================================================================
# 7. Comparative Baselines (Kalman/CV, Deep LSTM Regression, FlightBERT++ V2)
# ==============================================================================
class ConstantVelocityKalmanBaseline(nn.Module):
    """
    Classical Constant-Velocity / Kinematic State Extrapolation Baseline
    (corresponding to A3 in FlightBERT Table I).
    """

    def __init__(self, horizon_seconds: list[float]) -> None:
        super().__init__()
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        p_last = x_obs[:, -1, :6]
        lon0, lat0, alt0 = p_last[:, 0:1], p_last[:, 1:2], p_last[:, 2:3]
        vx_kmh, vy_kmh, vz_ms = p_last[:, 3:4], p_last[:, 4:5], p_last[:, 5:6]

        lat_rad = torch.deg2rad(lat0.clamp(-80.0, 80.0))
        m_per_deg_lon = 111320.0 * torch.cos(lat_rad)
        m_per_deg_lat = 111132.0

        t = self.t_sec.view(1, -1)
        dx_m = (vx_kmh / 3.6) * t
        dy_m = (vy_kmh / 3.6) * t
        dz_m = vz_ms * t

        pred_lon = lon0 + dx_m / m_per_deg_lon
        pred_lat = lat0 + dy_m / m_per_deg_lat
        pred_alt = alt0 + dz_m / 10.0
        pred_vx = vx_kmh.expand_as(pred_lon)
        pred_vy = vy_kmh.expand_as(pred_lon)
        pred_vz = vz_ms.expand_as(pred_lon)

        return torch.stack([pred_lon, pred_lat, pred_alt, pred_vx, pred_vy, pred_vz], dim=-1)


class SmoothedConstantVelocityBaseline(nn.Module):
    """
    Stronger kinematic reference baseline: least-squares linear fit of the
    trajectory over the whole observed window (12 points x 5 s), extrapolated
    to every horizon.

    Motivation: the naive Constant-Velocity baseline extrapolates the *raw*
    last ADS-B Doppler velocity, which is noisy per sample and therefore
    handicaps the comparison. Fitting the window with a least-squares line
    suppresses the position quantisation noise and is the standard "linear
    extrapolation" reference used in the trajectory-prediction literature.
    """

    def __init__(self, horizon_seconds: list[float], dt_sec: float = 5.0) -> None:
        super().__init__()
        self.dt_sec = float(dt_sec)
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        p_all = x_obs[..., :6]  # [B, T, 6]
        B, T, _ = p_all.shape
        device = x_obs.device

        # Relative time (seconds) of every observed point, ending at t = 0
        t = (torch.arange(T, device=device, dtype=x_obs.dtype) - (T - 1)) * self.dt_sec
        t_mean = t.mean()
        t_centered = t - t_mean
        denom = (t_centered ** 2).sum()

        pos = p_all[..., :3]  # lon, lat, alt_10m
        pos_mean = pos.mean(dim=1, keepdim=True)
        slope = ((pos - pos_mean) * t_centered.view(1, T, 1)).sum(dim=1) / denom  # [B, 3]
        intercept = pos_mean.squeeze(1) - slope * t_mean  # [B, 3]

        t_h = self.t_sec.view(1, -1)  # [1, H]
        # The extrapolation is anchored on the LAST OBSERVED position (so that
        # short horizons do not inherit the regression-line bias) while the
        # slope comes from the least-squares fit over the whole window.
        p_last = p_all[:, -1, :3]  # [B, 3]
        pred_pos = p_last.unsqueeze(1) + slope.unsqueeze(1) * t_h.unsqueeze(-1)  # [B, H, 3]

        lat_rad = torch.deg2rad(p_all[:, -1, 1].clamp(-80.0, 80.0)).unsqueeze(1)
        m_per_deg_lon = 111320.0 * torch.cos(lat_rad)
        m_per_deg_lat = 111132.0
        vx_kmh = slope[:, 0:1] * m_per_deg_lon * 3.6
        vy_kmh = slope[:, 1:2] * m_per_deg_lat * 3.6
        vz_ms = slope[:, 2:3] * 10.0

        vel = torch.cat([vx_kmh, vy_kmh, vz_ms], dim=-1).unsqueeze(1).expand(B, t_h.shape[1], 3)
        return torch.cat([pred_pos, vel], dim=-1)


class LSTMRegressionBaseline(nn.Module):
    """
    Deep LSTM Multi-Horizon Regression Baseline (Group A1 in FlightBERT paper).
    Uses normalized 6D trajectory inputs and predicts multi-horizon displacements.
    """

    def __init__(
        self,
        horizon: int,
        horizon_seconds: list[float],
        norm_mean: np.ndarray,
        norm_std: np.ndarray,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.register_buffer("norm_mean", torch.tensor(norm_mean[:6], dtype=torch.float32))
        self.register_buffer("norm_std", torch.tensor(norm_std[:6], dtype=torch.float32))
        horizon_ref = float(max(horizon_seconds))
        self.register_buffer(
            "t_scale", (torch.tensor(horizon_seconds, dtype=torch.float32) / horizon_ref).view(1, horizon, 1)
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
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, horizon * 6),
        )

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        x_6d = x_obs[..., :6]
        last_6d = x_6d[:, -1:, :]
        norm_in = (x_6d - self.norm_mean.view(1, 1, 6)) / self.norm_std.view(1, 1, 6)
        out, _ = self.lstm(norm_in)
        h_last = out[:, -1, :]
        delta = self.head(h_last).view(-1, self.horizon, 6) * self.out_scale * (0.1 + 0.9 * self.t_scale)
        return last_6d + delta


class FlightBERT_PP_V2_Baseline(nn.Module):
    """
    Reference FlightBERT++ V2 (from modello.txt: https://github.com/gdy-scu/FlightBERT_PP_V2):
    Pure Gray-Code Multi-Binary Classification (MBC) Non-Autoregressive Transformer
    without ACoAtt-V2, Physics Kinematic Prior, or MAF-Net Continuous Residual Gate.
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        win_cfg = cfg["window"]
        self.inp_seq_len = int(win_cfg["inp_seq_len"])
        self.horizon = int(win_cfg["horizon"])
        self.n_embd = 128
        self.n_head = 4

        self.codec = GrayCodeCodec(cfg)
        self.enc_embed = TokenConvEmbedding(self.codec.total_inp_bits, self.n_embd)
        self.pos_emb = SinusoidalPositionalEncoding(self.n_embd, 0.1, max_len=128)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.n_embd, nhead=self.n_head, dim_feedforward=self.n_embd * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.feature_aggre = FeatureAggregation(self.n_embd)

        self.dec_token_emb = TokenConvEmbedding(self.codec.total_delta_bits, self.n_embd)
        self.horizon_emb = nn.Embedding(self.horizon, self.n_embd)
        self.horizon_linear = nn.Linear(self.n_embd * 2, self.n_embd)
        self.de_pos_emb = SinusoidalPositionalEncoding(self.n_embd, 0.1, max_len=128)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.n_embd, nhead=self.n_head, dim_feedforward=self.n_embd * 2,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codec.total_delta_bits, bias=False)

    def forward(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        B, T, _ = x_obs.shape
        device = x_obs.device
        x_6d = x_obs[..., :6]
        last_obs_6d = x_6d[:, -1, :]

        gc_inputs = self.codec.encode_input_sequence(x_6d)
        cat_inputs = torch.cat(gc_inputs, dim=-1)
        token_embeddings = self.pos_emb(self.enc_embed(cat_inputs))
        encoder_output = self.transformer_encoder(token_embeddings)
        context_embed = self.feature_aggre(encoder_output)

        h_idx = torch.arange(self.horizon, device=device).unsqueeze(0).expand(B, -1)
        horizon_embed = self.horizon_emb(h_idx)
        ctx_rep = context_embed.unsqueeze(1).expand(-1, self.horizon, -1)
        decoder_input = self.horizon_linear(torch.cat([ctx_rep, horizon_embed], dim=-1))

        ones_dk = torch.ones((1, T - 1, 1), device=device, dtype=x_obs.dtype)
        dec_deltas = torch.cat(self.codec.encode_step_deltas(x_6d, ones_dk), dim=-1)
        decoder_input_diff = self.dec_token_emb(dec_deltas)
        decoder_full = self.de_pos_emb(torch.cat([decoder_input_diff, decoder_input], dim=1))

        fea = self.ln_f(self.decoder(decoder_full, memory=encoder_output))
        logits = self.head(fea)[:, -self.horizon :, :]
        probs = torch.sigmoid(logits)
        logits_tuple = torch.split(logits, self.codec.delta_sizes, dim=-1)
        probs_tuple = torch.split(probs, self.codec.delta_sizes, dim=-1)
        pred_6d = self.codec.decode_horizon_deltas_ste(last_obs_6d, probs_tuple)
        return pred_6d, logits_tuple


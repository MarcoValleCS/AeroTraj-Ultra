# coding=utf-8
"""
Benchmark models for short-term 4D aircraft trajectory prediction.

Five models replicated from the literature review (companion documents in
~/Documents/Progetto predefinito/):

  1. ASCENT            - Prutsch, Schinagl & Possegger, arXiv:2603.16550, ICRA 2026
                         Transformer encoder + DETR-style learnable MODE QUERIES,
                         PARAMETERISED kinematic outputs (speed / yaw-rate / vertical
                         rate) integrated to positions, winner-takes-all loss.
  2. CFM-Flow          - Conditional Flow Matching, FlowATC (arXiv:2609.16528) and
                         Figuet et al. (JOAS 2026).  Sequence-inpainting DiT with
                         block-causal attention + AdaLN, trained with the CFM
                         likelihood objective, sampled with the probability-flow ODE.
  3. DDPM-Diffusion     - Ho et al. 2020 DDPM / MID (Gu et al. 2022) on the SAME
                         backbone as (2), so the CFM-vs-DDPM comparison reported by
                         FlowATC can be reproduced head-to-head.
  4. AttentionLSTM     - Jia et al., Sci. Rep. 12:15533 (2022) attention-LSTM,
                         made phase-aware following Shi et al., IEEE TITS 22 (2021)
                         (flight phase as an input + hard vertical-rate constraint).
  5. IMM-Residual      - hybrid physics+ML: IMM (Blom & Bar-Shalom 1988) over
                         kinematic modes (CV / CA / CT) with per-mode Kalman filters,
                         plus a neural residual corrector trained on the innovation
                         sequence (Li et al., Sensors 25(8):2531, 2025 - IMM-Informer).

All the new models work in an AIRCRAFT-CENTRIC LOCAL FRAME (origin = last observed
position, forward axis = last observed ground track), the practice recommended by
the review (Figuet 2026, ASCENT) because it removes route/geography bias.
Predictions are mapped back to global WGS-84 for the evaluation, which is
identical for every model (including the project model).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Scales (fixed, documented): the networks never see raw metres / km/h
# --------------------------------------------------------------------------- #
POS_SCALE_M = 10000.0     # 1 network unit = 10 km
VEL_SCALE = 200.0         # 1 network unit = 200 km/h
LOCAL_SCALE = 1000.0      # local frame -> km for the generative models

M_PER_DEG_LAT = 111132.0


# --------------------------------------------------------------------------- #
# Local frame helpers
# --------------------------------------------------------------------------- #
def local_frame(x_obs: torch.Tensor):
    """-> lon0, lat0, alt0, heading, m_per_deg_lon (each [B,1])"""
    lon0 = x_obs[:, -1, 0:1]
    lat0 = x_obs[:, -1, 1:2]
    alt0 = x_obs[:, -1, 2:3]
    vx0 = x_obs[:, -1, 3:4]
    vy0 = x_obs[:, -1, 4:5]
    heading = torch.atan2(vx0, vy0)          # clockwise from North
    lat_rad = torch.deg2rad(lat0.clamp(-80.0, 80.0))
    m_per_deg_lon = 111320.0 * torch.cos(lat_rad)
    return lon0, lat0, alt0, heading, m_per_deg_lon


def repeat_origin(origin, n: int):
    """Repeat every element of the origin tuple n times along the batch dim."""
    return tuple(o.repeat_interleave(n, dim=0) for o in origin)


def _u(o: torch.Tensor) -> torch.Tensor:
    """[B,1] -> [B,1,1]: broadcast-ready against [B,T,1] / [B,H,1] tensors."""
    return o.unsqueeze(1)


def to_local(x: torch.Tensor, origin) -> torch.Tensor:
    """[B,T,>=3] global (lon, lat, alt_10m) -> local (fwd, rgt, up) in metres."""
    lon0, lat0, alt0, heading, m_lon = origin
    lon0, lat0, alt0 = _u(lon0), _u(lat0), _u(alt0)
    m_lon = _u(m_lon)
    dx_e = (x[..., 0:1] - lon0) * m_lon
    dy_n = (x[..., 1:2] - lat0) * M_PER_DEG_LAT
    dz = (x[..., 2:3] - alt0) * 10.0
    sh, ch = _u(torch.sin(heading)), _u(torch.cos(heading))
    return torch.cat([dx_e * sh + dy_n * ch, dx_e * ch - dy_n * sh, dz], dim=-1)


def local_to_global(loc: torch.Tensor, origin) -> torch.Tensor:
    """[B,H,3] local (fwd, rgt, up) metres -> global (lon, lat, alt_10m)."""
    lon0, lat0, alt0, heading, m_lon = origin
    lon0, lat0, alt0 = _u(lon0), _u(lat0), _u(alt0)
    m_lon = _u(m_lon)
    sh, ch = _u(torch.sin(heading)), _u(torch.cos(heading))
    dx_e = loc[..., 0:1] * sh + loc[..., 1:2] * ch
    dy_n = loc[..., 0:1] * ch - loc[..., 1:2] * sh
    return torch.cat(
        [lon0 + dx_e / m_lon, lat0 + dy_n / M_PER_DEG_LAT, alt0 + loc[..., 2:3] / 10.0], dim=-1
    )


def local_vel_to_global(vloc: torch.Tensor, origin) -> torch.Tensor:
    """[B,H,3] local (v_fwd, v_rgt, v_up) -> global (vx, vy, vz)."""
    heading = origin[3]
    sh, ch = _u(torch.sin(heading)), _u(torch.cos(heading))
    return torch.cat(
        [vloc[..., 0:1] * sh + vloc[..., 1:2] * ch,
         vloc[..., 0:1] * ch - vloc[..., 1:2] * sh,
         vloc[..., 2:3]], dim=-1
    )


def global_vel_from_positions(
    pos_global: torch.Tensor, x_obs: torch.Tensor, origin, t_sec: torch.Tensor
) -> torch.Tensor:
    """
    Average velocities of a predicted trajectory, anchored at the last observed
    position.  pos_global [B,H,3] (lon, lat, alt_10m) -> [B,H,3] (vx, vy, vz).
    """
    lon0, lat0, alt0, heading, m_lon = origin
    anchor = torch.cat([lon0, lat0, alt0], dim=-1)                 # [B,3]
    full = torch.cat([anchor.unsqueeze(1), pos_global], dim=1)      # [B,H+1,3]
    d = full[:, 1:] - full[:, :-1]                                 # [B,H,3]
    dt = t_sec.view(1, -1, 1)
    vx = d[..., 0:1] * _u(m_lon) / dt * 3.6
    vy = d[..., 1:2] * M_PER_DEG_LAT / dt * 3.6
    vz = d[..., 2:3] * 10.0 / dt
    return torch.cat([vx, vy, vz], dim=-1)


# --------------------------------------------------------------------------- #
# Shared building blocks
# --------------------------------------------------------------------------- #
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


def flight_phase_features(x_obs: torch.Tensor) -> torch.Tensor:
    """
    Phase-aware features (Shi, Xu & Pan, IEEE TITS 22:7242, 2021):
    one-hot over {climb, cruise, descent, level} + two vertical-margin features.
    x_obs [B,T,9] -> [B,T,6]
    """
    alt = x_obs[:, :, 2]                        # alt in 10 m units
    vz = x_obs[:, :, 5]                         # m/s
    climb = (vz > 0.5).float()
    desc = (vz < -0.5).float()
    level = 1.0 - climb - desc
    cruise = ((alt > 600.0) & (torch.abs(vz) <= 0.5)).float()
    phase = torch.stack([climb, cruise, desc, level], dim=-1)
    v_max = torch.clamp(8.0 - 4.0 * (alt / 1200.0), 1.0, 8.0)
    margin = torch.stack([vz / v_max, -vz / v_max], dim=-1)
    return torch.cat([phase, margin], dim=-1)


class PreLNEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1, ff_mult: int = 3) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.n2(x))


# --------------------------------------------------------------------------- #
# 1. ASCENT  (arXiv:2603.16550, ICRA 2026)
# --------------------------------------------------------------------------- #
class ASCENT(nn.Module):
    """
    Aircraft Sequence Encoding Transformer.

    * history normalised in the aircraft-centric frame + 3-D positional embedding
      PE(x,y,z,gamma,theta) = [x, y, z, sin(gamma), cos(gamma), sin(theta), cos(theta)]
      -> 2-layer MLP (ASCENT uses D = 128).
    * encoder: 2 self-attention blocks.
    * decoder: K DETR-style learnable MODE QUERIES -> K multimodal hypotheses.
    * PARAMETERISED outputs: per mode and per horizon the network predicts the
      kinematic state (ground speed, yaw rate, vertical rate); the positions are
      obtained by INTEGRATING those parameters, so every hypothesis is
      kinematically consistent by construction.
    * score head + winner-takes-all loss.
    """

    def __init__(
        self,
        horizon: int,
        horizon_seconds: List[int],
        inp_seq_len: int = 12,
        d_model: int = 128,
        n_head: int = 8,
        n_modes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.n_modes = n_modes
        self.d_model = d_model
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))
        prev = torch.cat([torch.zeros(1), torch.tensor(horizon_seconds[:-1], dtype=torch.float32)])
        self.register_buffer("dt_sec", torch.tensor(horizon_seconds, dtype=torch.float32) - prev)

        self.pe_mlp = nn.Sequential(
            nn.Linear(7, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.in_proj = nn.Linear(3 + 6, d_model)   # v_fwd, v_rgt, v_z + 6 phase features
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout)
        self.encoder = nn.ModuleList([PreLNEncoderBlock(d_model, n_head, dropout) for _ in range(2)])
        self.mode_queries = nn.Parameter(torch.randn(n_modes, d_model) * 0.02)
        self.mode_ctx = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.decoder = nn.ModuleList([PreLNEncoderBlock(d_model, n_head, dropout) for _ in range(2)])
        self.h_speed = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))
        self.h_yaw = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))
        self.h_vs = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, horizon))
        self.h_score = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def _history_embedding(self, x_obs: torch.Tensor, origin) -> torch.Tensor:
        spd = torch.linalg.norm(x_obs[:, :, 3:5], dim=-1, keepdim=True).clamp(min=1.0)
        sh = torch.sin(origin[3]).unsqueeze(1)
        ch = torch.cos(origin[3]).unsqueeze(1)
        v_f = (x_obs[:, :, 3:4] * sh + x_obs[:, :, 4:5] * ch) / spd
        v_r = (x_obs[:, :, 3:4] * ch - x_obs[:, :, 4:5] * sh) / spd
        feats = torch.cat([v_f, v_r, x_obs[:, :, 5:6] / 10.0, flight_phase_features(x_obs)], dim=-1)
        return self.in_proj(feats)

    def _integrate(self, speed, yaw_rate, vs):
        """
        Integrate the parameterised kinematics in the local frame.
        speed [B,K,H] km/h, yaw_rate [B,K,H] rad/s, vs [B,K,H] m/s
        -> local positions [B,K,H,3] (m) and local velocities [B,K,H,3] (km/h, m/s)
        """
        B, K, H = speed.shape
        dt = self.dt_sec.view(1, 1, H)
        v_f = speed / 3.6                                  # m/s along the track
        pos_f = torch.zeros(B, K, H, device=speed.device, dtype=speed.dtype)
        pos_r = torch.zeros_like(pos_f)
        vel_f = torch.zeros_like(pos_f)
        vel_r = torch.zeros_like(pos_f)
        cur_f = torch.zeros(B, K, device=speed.device, dtype=speed.dtype)
        cur_r = torch.zeros_like(cur_f)
        psi = torch.zeros_like(cur_f)
        z = torch.zeros_like(cur_f)
        z_hist = []
        for i in range(H):
            dti = dt[:, :, i]
            psi_mid = psi + 0.5 * yaw_rate[:, :, i] * dti
            vf_i = v_f[:, :, i]
            df = vf_i * torch.cos(psi_mid) * dti
            dr = vf_i * torch.sin(psi_mid) * dti
            cur_f = cur_f + df
            cur_r = cur_r + dr
            psi = psi + yaw_rate[:, :, i] * dti
            z = z + vs[:, :, i] * dti
            pos_f[:, :, i] = cur_f
            pos_r[:, :, i] = cur_r
            vel_f[:, :, i] = vf_i * torch.cos(psi_mid) * 3.6
            vel_r[:, :, i] = vf_i * torch.sin(psi_mid) * 3.6
            z_hist.append(z)
        z_all = torch.stack(z_hist, dim=2)            # [B,K,H]
        pos = torch.stack([pos_f, pos_r, z_all], dim=-1)
        vel = torch.stack([vel_f, vel_r, vs], dim=-1)
        return pos, vel

    def forward(self, x_obs: torch.Tensor, return_scores: bool = False):
        origin = local_frame(x_obs)
        B = x_obs.shape[0]
        h = self.pos_enc(self._history_embedding(x_obs, origin))
        for blk in self.encoder:
            h = blk(h)
        ctx = h.mean(dim=1, keepdim=True)

        q = self.mode_queries.unsqueeze(0).expand(B, -1, -1)
        q = self.mode_ctx(q) + ctx
        for blk in self.decoder:
            q = blk(q)

        speed = self.h_speed(q) * 100.0                   # km/h
        yaw_rate = self.h_yaw(q) * 0.05                   # rad/s
        vs = self.h_vs(q) * 5.0                           # m/s
        score = self.h_score(q).squeeze(-1)

        pos, vel = self._integrate(speed, yaw_rate, vs)
        Bk = B * self.n_modes
        o = repeat_origin(origin, self.n_modes)
        gl = local_to_global(pos.reshape(Bk, self.horizon, 3), o)
        gv = local_vel_to_global(vel.reshape(Bk, self.horizon, 3), o)
        out = torch.cat([gl, gv], dim=-1).reshape(B, self.n_modes, self.horizon, 6)
        return (out, score) if return_scores else out


# --------------------------------------------------------------------------- #
# 2 + 3. Flow-matching / diffusion backbone (FlowATC-style sequence inpainting)
# --------------------------------------------------------------------------- #
class AdaLNBlock(nn.Module):
    """Pre-LN transformer block with AdaLN-Zero conditioning on (t, cond)."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 3), nn.GELU(), nn.Linear(d_model * 3, d_model))
        self.mod = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask=None) -> torch.Tensor:
        s, b, g, sa, ga, gb = self.mod(c).unsqueeze(1).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sa) + b
        x = x + g * self.attn(h, h, h, need_weights=False, attn_mask=attn_mask)[0]
        h = self.n2(x) * (1 + ga) + gb
        return x + sa * self.ff(h)


class FlowBackbone(nn.Module):
    """
    Shared DiT-style backbone for CFM and DDPM.

    Tokens: [observed history (clean)] ++ [future (noisy), H tokens].
    Block-causal attention: the future block sees the whole history and is causal
    inside itself; the history block never sees the future.
    """

    def __init__(
        self,
        horizon: int,
        inp_seq_len: int = 12,
        d_model: int = 128,
        n_head: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        n_hist_feats: int = 9,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.inp_seq_len = inp_seq_len
        self.d_model = d_model
        self.hist_proj = nn.Linear(n_hist_feats, d_model)
        self.fut_proj = nn.Linear(3, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout, max_len=512)
        self.time_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cond_proj = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList([AdaLNBlock(d_model, n_head, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.out_mod = nn.Linear(d_model, 2 * d_model)
        nn.init.zeros_(self.out_mod.weight)
        nn.init.zeros_(self.out_mod.bias)
        self.out_proj = nn.Linear(d_model, 3)

    @staticmethod
    def _mask(T: int, H: int, device) -> torch.Tensor:
        """Boolean attention mask (True = blocked).

        Layout (rows x cols) = [history | future] x [history | future]:
          top-left     : history fully visible
          top-right    : history must NOT see the future
          bottom-left  : future sees the whole history
          bottom-right : future is causal inside itself
        """
        hist = torch.zeros(T, T, dtype=torch.bool, device=device)
        fut = torch.triu(torch.ones(H, H, dtype=torch.bool, device=device), diagonal=1)
        hf = torch.zeros(H, T, dtype=torch.bool, device=device)   # future rows, history cols
        fh = torch.ones(T, H, dtype=torch.bool, device=device)   # history rows, future cols
        return torch.cat([torch.cat([hist, fh], 1), torch.cat([hf, fut], 1)], 0)

    def forward(self, hist_feat, fut, t, cond):
        """hist_feat [B,T,F]; fut [B,H,3]; t [B]; cond [B,D] -> [B,H,3]"""
        B, T, _ = hist_feat.shape
        H = fut.shape[1]
        x = torch.cat([self.hist_proj(hist_feat), self.fut_proj(fut)], dim=1)
        x = self.pos_enc(x)
        c = self.cond_proj(cond) + self.time_mlp(self._time_embed(t, self.d_model))
        mask = self._mask(T, H, x.device)
        for blk in self.blocks:
            x = blk(x, c, attn_mask=mask)
        s, b = self.out_mod(c).unsqueeze(1).chunk(2, dim=-1)
        h = self.out_norm(x[:, T:])
        return self.out_proj((1.0 + s) * h + b)

    @staticmethod
    def _time_embed(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        ang = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class _GenerativeBase(nn.Module):
    """Common plumbing for the CFM / DDPM models."""

    def __init__(self, horizon, horizon_seconds, inp_seq_len=12, d_model=128,
                 n_head=8, n_layers=4):
        super().__init__()
        self.horizon = horizon
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))
        self.backbone = FlowBackbone(horizon, inp_seq_len, d_model, n_head, n_layers)

    def _hist_feat(self, x_obs, origin):
        loc = to_local(x_obs, origin) / LOCAL_SCALE
        vel = x_obs[..., 3:6] / VEL_SCALE
        return torch.cat([loc, vel, x_obs[:, :, 6:9] / 50.0], dim=-1)

    def _cond(self, x_obs, origin):
        loc = to_local(x_obs[:, -1:], origin) / LOCAL_SCALE
        vel = x_obs[:, -1:, 3:6] / VEL_SCALE
        return torch.cat([loc, vel], dim=-1).squeeze(1)

    def _norm_scale(self, x_obs) -> torch.Tensor:
        """
        Per-sample, per-horizon scale of the local displacement: the expected
        travelled distance v_ref * t.  Without it the data (tens of km) has to
        be produced by a flow/diffusion whose prior is N(0,1) -> the transport
        is unlearnable.  Normalising by the expected displacement makes the
        target O(1) at every horizon and the model learns the SHAPE of the
        trajectory (turn rate, climb) rather than its magnitude.
        """
        v_ref = torch.linalg.norm(x_obs[:, -1, 3:5], dim=-1) * (1000.0 / 3600.0)  # km/h -> m/s
        v_ref = v_ref.clamp(min=30.0)
        t = self.t_sec.view(1, -1)
        return v_ref[:, None] * t + 1.0            # [B,H] metres

    def _finish(self, x, x_obs, origin, n_samples):
        """Sampled (normalised) trajectory -> global [B,n_samples,H,6]."""
        B = x_obs.shape[0]
        scale = self._norm_scale(x_obs).repeat_interleave(n_samples, 0)      # [B*K,H]
        loc = x * scale.unsqueeze(-1)                                        # metres
        o = repeat_origin(origin, n_samples)
        gl = local_to_global(loc, o)
        gv = global_vel_from_positions(gl, x_obs.repeat_interleave(n_samples, 0), o, self.t_sec)
        return torch.cat([gl, gv], dim=-1).reshape(B, n_samples, self.horizon, 6)


class CFMTrajectoryModel(_GenerativeBase):
    """
    Conditional Flow Matching (Lipman et al. 2023; FlowATC arXiv:2609.16528;
    Figuet et al. JOAS 2026).

    Training:  x0 ~ N(0,I), x1 = true future (local frame),
               x_t = (1-t) x0 + t x1,  loss = || u_theta(x_t,t) - (x1-x0) ||^2
    Inference: probability-flow ODE  dx/dt = u_theta(x,t), Euler integration.
    """

    def __init__(self, *args, n_steps: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps

    def forward(self, x_obs: torch.Tensor, y_target: torch.Tensor = None) -> torch.Tensor:
        origin = local_frame(x_obs)
        hist = self._hist_feat(x_obs, origin)
        cond = self._cond(x_obs, origin)
        scale = self._norm_scale(x_obs)                       # [B,H] metres
        x1 = to_local(y_target, origin) / scale.unsqueeze(-1)  # normalised target, O(1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        pred = self.backbone(hist, xt, t, cond)
        return ((pred - (x1 - x0)) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def sample(self, x_obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        origin = local_frame(x_obs)
        hist = self._hist_feat(x_obs, origin)
        cond = self._cond(x_obs, origin)
        B = x_obs.shape[0]
        hist_r = hist.repeat_interleave(n_samples, 0)
        cond_r = cond.repeat_interleave(n_samples, 0)
        x = torch.randn(B * n_samples, self.horizon, 3, device=x_obs.device)
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.full((B * n_samples,), i * dt, device=x_obs.device)
            x = x + dt * self.backbone(hist_r, x, t, cond_r)
        return self._finish(x, x_obs, origin, n_samples)


class DDPMTrajectoryModel(_GenerativeBase):
    """
    DDPM (Ho et al. 2020) / MID-style diffusion on the SAME backbone as the CFM
    model, so that the FlowATC claim ("CFM beats DDPM by 11-26 % in minADE@20")
    can be checked head-to-head on this dataset.
    """

    def __init__(self, *args, n_train_steps: int = 100, n_sample_steps: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_train_steps = n_train_steps
        self.n_sample_steps = n_sample_steps
        betas = torch.linspace(1e-4, 0.02, n_train_steps)
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_acp", acp.sqrt())
        self.register_buffer("sqrt_1m_acp", (1.0 - acp).sqrt())

    def forward(self, x_obs: torch.Tensor, y_target: torch.Tensor = None) -> torch.Tensor:
        origin = local_frame(x_obs)
        hist = self._hist_feat(x_obs, origin)
        cond = self._cond(x_obs, origin)
        scale = self._norm_scale(x_obs)
        x1 = to_local(y_target, origin) / scale.unsqueeze(-1)
        B = x1.shape[0]
        t = torch.randint(0, self.n_train_steps, (B,), device=x1.device)
        noise = torch.randn_like(x1)
        xt = self.sqrt_acp[t][:, None, None] * x1 + self.sqrt_1m_acp[t][:, None, None] * noise
        pred = self.backbone(hist, xt, t.float() / self.n_train_steps, cond)
        return ((pred - noise) ** 2).mean(dim=(1, 2))

    @torch.no_grad()
    def sample(self, x_obs: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        origin = local_frame(x_obs)
        hist = self._hist_feat(x_obs, origin)
        cond = self._cond(x_obs, origin)
        B = x_obs.shape[0]
        hist_r = hist.repeat_interleave(n_samples, 0)
        cond_r = cond.repeat_interleave(n_samples, 0)
        x = torch.randn(B * n_samples, self.horizon, 3, device=x_obs.device)
        steps = torch.linspace(self.n_train_steps - 1, 0, self.n_sample_steps).long().to(x_obs.device)
        for i in range(self.n_sample_steps):
            t = steps[i]
            eps = self.backbone(hist_r, x, torch.full((B * n_samples,), t.float() / self.n_train_steps,
                                                       device=x_obs.device), cond_r)
            acp = self.alphas_cumprod[t]
            x0 = ((x - (1 - acp).sqrt() * eps) / acp.sqrt()).clamp(-8.0, 8.0)
            if i < self.n_sample_steps - 1:
                a_next = self.alphas_cumprod[steps[i + 1]]
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        return self._finish(x, x_obs, origin, n_samples)


# --------------------------------------------------------------------------- #
# 4. Attention-LSTM, phase-aware (Jia et al. 2022 + Shi et al. 2021)
# --------------------------------------------------------------------------- #
class AttentionLSTMPhase(nn.Module):
    """
    Attention-LSTM (Jia, Chen, Zhang & Han, Sci. Rep. 12:15533, 2022) made
    phase-aware (Shi, Xu & Pan, IEEE TITS 22:7242, 2021).

    * LSTM over the local-frame kinematics + phase one-hot.
    * additive attention pooling of the hidden states.
    * direct multi-step heads for position and velocity, with the predicted
      vertical rate CLIPPED to the phase-dependent envelope (hard constraint).
    """

    def __init__(self, horizon: int, horizon_seconds: List[int], inp_seq_len: int = 12,
                 hidden_dim: int = 192, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.horizon = horizon
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))
        self.lstm = nn.LSTM(input_size=3 + 6, hidden_size=hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.att = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.head_pos = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, horizon * 3))
        self.head_vel = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, horizon * 2))
        self.head_vs = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, horizon))

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        origin = local_frame(x_obs)
        spd = torch.linalg.norm(x_obs[:, :, 3:5], dim=-1, keepdim=True).clamp(min=1.0)
        sh = torch.sin(origin[3]).unsqueeze(1)
        ch = torch.cos(origin[3]).unsqueeze(1)
        v_f = (x_obs[:, :, 3:4] * sh + x_obs[:, :, 4:5] * ch) / spd
        v_r = (x_obs[:, :, 3:4] * ch - x_obs[:, :, 4:5] * sh) / spd
        feats = torch.cat([v_f, v_r, x_obs[:, :, 5:6] / 10.0, flight_phase_features(x_obs)], dim=-1)
        h, _ = self.lstm(feats)
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)
        ctx = (w.unsqueeze(-1) * h).sum(dim=1)
        z = torch.cat([ctx, h[:, -1]], dim=-1)

        pos = self.head_pos(z).reshape(-1, self.horizon, 3) * POS_SCALE_M     # metres
        vel = self.head_vel(z).reshape(-1, self.horizon, 2) * VEL_SCALE       # km/h
        vs = self.head_vs(z) * 5.0                                           # m/s

        # phase-dependent hard constraint on the vertical rate
        alt0 = x_obs[:, -1, 2]
        v_max = torch.clamp(8.0 - 4.0 * (alt0 / 1200.0), 1.0, 8.0)[:, None]
        climbing = (x_obs[:, -1, 5] > 0.5)[:, None]
        descending = (x_obs[:, -1, 5] < -0.5)[:, None]
        lo = torch.where(climbing, torch.zeros_like(v_max), -v_max)
        hi = torch.where(descending, torch.zeros_like(v_max), v_max)
        vs = torch.clamp(vs, lo, hi)

        gl = local_to_global(pos, origin)
        vloc = torch.cat([vel[..., 0:1], vel[..., 1:2], vs.unsqueeze(-1)], dim=-1)
        gv = local_vel_to_global(vloc, origin)
        return torch.cat([gl, gv], dim=-1)


# --------------------------------------------------------------------------- #
# 5. IMM over kinematic modes + neural residual corrector
# --------------------------------------------------------------------------- #
class IMMResidualModel(nn.Module):
    """
    Hybrid physics + ML: IMM (Blom & Bar-Shalom 1988) over three kinematic modes
    (CV / CA / coordinated turn) with per-mode Kalman filters, plus a neural
    residual corrector trained on the innovation sequence
    (Li et al., Sensors 25(8):2531, 2025 - IMM-Informer).

    The IMM provides a physically-consistent multi-hypothesis propagation; the
    GRU learns the systematic bias of that estimator and corrects it.
    """

    MODES = 3   # 0 = CV, 1 = CA, 2 = coordinated turn

    def __init__(self, horizon: int, horizon_seconds: List[int], inp_seq_len: int = 12,
                 hidden_dim: int = 192, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.horizon = horizon
        self.register_buffer("t_sec", torch.tensor(horizon_seconds, dtype=torch.float32))
        prev = torch.cat([torch.zeros(1), torch.tensor(horizon_seconds[:-1], dtype=torch.float32)])
        self.register_buffer("dt_sec", torch.tensor(horizon_seconds, dtype=torch.float32) - prev)
        p = torch.tensor([[0.90, 0.07, 0.03],
                          [0.07, 0.90, 0.03],
                          [0.03, 0.07, 0.90]], dtype=torch.float32)
        self.register_buffer("trans", p)
        self.gru = nn.GRU(input_size=3 + 6 + 3, hidden_size=hidden_dim, num_layers=n_layers,
                          batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, horizon * 6))

    # -- kinematic propagation of one mode -------------------------------- #
    def _propagate(self, x: torch.Tensor, v: torch.Tensor, mode: int, dt: float):
        """x,v: [B,3] in the local frame (m, m/s) -> propagated x,v."""
        if mode == 0:                                   # constant velocity
            return x + v * dt, v
        if mode == 1:                                   # constant (damped) acceleration
            a = 0.05
            return x + v * dt + 0.5 * a * dt * dt, v + a * dt
        # coordinated turn: rotate the horizontal velocity by a small rate
        vh = torch.linalg.norm(v[:, :2], dim=-1, keepdim=True)
        psi = torch.atan2(v[:, 1:2], v[:, 0:1])
        omega = torch.clamp((v[:, 1:2] / vh.clamp(min=1.0)) * 0.05, -0.05, 0.05)
        psi_mid = psi + 0.5 * omega * dt
        vx = vh * torch.cos(psi_mid)
        vy = vh * torch.sin(psi_mid)
        dx = 0.5 * (v[:, :2] + torch.cat([vx, vy], dim=-1)) * dt
        x = torch.cat([x[:, :2] + dx, x[:, 2:3] + v[:, 2:3] * dt], dim=-1)
        v = torch.cat([vx, vy, v[:, 2:3]], dim=-1)
        return x, v

    def _mode_predict(self, x_obs: torch.Tensor, origin):
        B, T, _ = x_obs.shape
        dt = 5.0
        meas = to_local(x_obs, origin)                     # [B,T,3] metres
        v_init = (meas[:, 1] - meas[:, 0]) / dt
        x_init = meas[:, 0]
        states_x = x_init.unsqueeze(1).repeat(1, self.MODES, 1).clone()
        states_v = v_init.unsqueeze(1).repeat(1, self.MODES, 1).clone()
        mu = torch.full((B, self.MODES), 1.0 / self.MODES, device=x_obs.device)
        innov_hist = []
        R = 25.0 ** 2
        for t in range(T):
            for m in range(self.MODES):
                xm, vm = self._propagate(states_x[:, m], states_v[:, m], m, dt)
                states_x[:, m] = xm
                states_v[:, m] = vm
            z = meas[:, t]
            y_modes = []
            for m in range(self.MODES):
                xm, vm = states_x[:, m], states_v[:, m]
                y = z - xm
                S = R + 4.0
                nll = (y ** 2).sum(-1) / S
                y_modes.append(y)
                k = 4.0 / (4.0 + S)
                states_x[:, m] = xm + k * y
                states_v[:, m] = vm + 0.5 * k * y / dt
                mu[:, m] = mu[:, m] * torch.exp(-0.5 * nll)
            # one innovation per timestep (fused over the modes)
            innov_hist.append(torch.stack(y_modes, dim=1).mean(dim=1))
            mu = mu / mu.sum(dim=1, keepdim=True).clamp(min=1e-9)
            mu = mu @ self.trans

        innov = torch.stack(innov_hist, dim=1)             # [B,T,3]
        # --- forward propagation of every mode, then combine with the weights ---
        preds, vels = [], []
        for m in range(self.MODES):
            cur_x, cur_v = states_x[:, m].clone(), states_v[:, m].clone()
            ps, vs = [], []
            for i in range(self.horizon):
                dti = float(self.dt_sec[i])
                cur_x, cur_v = self._propagate(cur_x, cur_v, m, dti)
                ps.append(cur_x)
                vs.append(cur_v)
            preds.append(torch.stack(ps, dim=1))
            vels.append(torch.stack(vs, dim=1))
        P = torch.stack(preds, dim=1)                     # [B,M,H,3]
        V = torch.stack(vels, dim=1)
        w = mu.unsqueeze(-1).unsqueeze(-1)                 # [B,M,1,1]
        pred_local = (w * P).sum(dim=1)                    # [B,H,3]
        vel_local = (w * V).sum(dim=1)                     # [B,H,3] m/s
        return pred_local, vel_local, innov, mu

    def forward(self, x_obs: torch.Tensor) -> torch.Tensor:
        origin = local_frame(x_obs)
        pred_local, vel_local, innov, _ = self._mode_predict(x_obs, origin)
        B = x_obs.shape[0]
        spd = torch.linalg.norm(x_obs[:, :, 3:5], dim=-1, keepdim=True).clamp(min=1.0)
        sh = torch.sin(origin[3]).unsqueeze(1)
        ch = torch.cos(origin[3]).unsqueeze(1)
        v_f = (x_obs[:, :, 3:4] * sh + x_obs[:, :, 4:5] * ch) / spd
        v_r = (x_obs[:, :, 3:4] * ch - x_obs[:, :, 4:5] * sh) / spd
        feats = torch.cat([v_f, v_r, x_obs[:, :, 5:6] / 10.0,
                           flight_phase_features(x_obs), innov / 1000.0], dim=-1)
        h, _ = self.gru(feats)
        corr = self.head(h[:, -1]).reshape(B, self.horizon, 6)
        corr_pos = corr[..., :3] * POS_SCALE_M               # metres
        corr_vel = corr[..., 3:] * VEL_SCALE                 # km/h (xy), m/s (z)
        loc = pred_local + corr_pos
        gl = local_to_global(loc, origin)
        vloc = torch.cat([vel_local[:, :, :2] * 3.6 + corr_vel[..., :2],
                          vel_local[:, :, 2:3] + corr_vel[..., 2:3]], dim=-1)
        gv = local_vel_to_global(vloc, origin)
        return torch.cat([gl, gv], dim=-1)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def horizon_weighted_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_seconds: List[int],
    pos_scale: float = 0.02,
    vel_scale: float = 15.0,
    h_norm_floor: float = 0.04,
) -> torch.Tensor:
    """Deterministic regression loss with horizon-dependent normalisation."""
    t = torch.tensor(horizon_seconds, dtype=torch.float32, device=pred.device)
    h_norm = (h_norm_floor + (1.0 - h_norm_floor) * (t / 60.0)).view(1, -1, 1)
    err_pos = (pred[..., :3] - target[..., :3]) / (pos_scale * h_norm)
    err_vel = (pred[..., 3:] - target[..., 3:]) / vel_scale
    return F.smooth_l1_loss(err_pos, torch.zeros_like(err_pos), beta=0.25) + \
        F.smooth_l1_loss(err_vel, torch.zeros_like(err_vel), beta=0.5)


def _per_mode_reg(
    out: torch.Tensor,
    target: torch.Tensor,
    horizon_seconds: List[int],
    pos_scale: float = 0.02,
    vel_scale: float = 15.0,
    h_norm_floor: float = 0.04,
) -> torch.Tensor:
    """[B,K,H,6] vs [B,H,6] -> [B,K] horizon-weighted Smooth-L1 per mode."""
    t = torch.tensor(horizon_seconds, dtype=torch.float32, device=out.device)
    h_norm = (h_norm_floor + (1.0 - h_norm_floor) * (t / 60.0)).view(1, 1, -1, 1)
    scales = torch.tensor([pos_scale] * 3 + [vel_scale] * 3,
                          device=out.device).view(1, 1, 1, 6)
    err = (out - target.unsqueeze(1)) / (scales * h_norm)
    pos = F.smooth_l1_loss(err[..., :3], torch.zeros_like(err[..., :3]), beta=0.25,
                           reduction="none").mean(dim=(2, 3))
    vel = F.smooth_l1_loss(err[..., 3:], torch.zeros_like(err[..., 3:]), beta=0.5,
                           reduction="none").mean(dim=(2, 3))
    return pos + vel


def ascent_wta_loss(
    out: torch.Tensor,
    score: torch.Tensor,
    target: torch.Tensor,
    horizon_seconds: List[int],
    n_modes: int,
    label_smooth: float = 0.0,
    aux_weight: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Winner-takes-all multi-hypothesis loss (ASCENT):
      * only the best-matching hypothesis (lowest 3D error) drives the regression,
      * a cross-entropy term ranks that hypothesis first,
      * an AUXILIARY (down-weighted) regression on every hypothesis prevents the
        non-winning modes from degenerating (the DETR "auxiliary loss" trick).
    """
    B, K, H, _ = out.shape
    d = out - target.unsqueeze(1)                       # [B,K,H,6]
    lat_r = torch.deg2rad(target[:, :, 1]).unsqueeze(1)  # [B,1,H]
    mlon = 111320.0 * torch.cos(lat_r)
    dist = torch.sqrt((d[..., 0] * mlon) ** 2 + (d[..., 1] * 111132.0) ** 2 + (d[..., 2] * 10.0) ** 2)
    mode_err = dist.mean(dim=2)                        # [B,K]
    best = mode_err.argmin(dim=1)
    idx = best.view(B, 1, 1, 1).expand(B, 1, H, 6)
    best_pred = out.gather(1, idx).squeeze(1)
    reg = horizon_weighted_smooth_l1(best_pred, target, horizon_seconds)
    aux = _per_mode_reg(out, target, horizon_seconds).mean()
    ce = F.cross_entropy(score, best, label_smoothing=label_smooth)
    return reg + aux_weight * aux + 0.1 * ce, best

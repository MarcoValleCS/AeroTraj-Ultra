# coding=utf-8
"""
FlightPREV (Marco Valle, 2024) re-implemented on the OpenSky dataset.

The original model lives in https://github.com/MarcoValleCS/flight (`src/models.py`,
arch="transformer") and predicts lat/lon/alt over 12 horizons (10-120 s) as a
RESIDUAL with respect to a constant-velocity ballistic baseline, with an
auxiliary flight-phase classification task and learned categorical embeddings
(aircraft type / operator / departure / arrival airports).

This module reproduces that architecture as faithfully as the OpenSky
ADS-B data allows, so that it can be compared with AeroTraj-Ultra on the
SAME data, with the SAME horizon grid (5-300 s) and the SAME metric.

Adaptations forced by the data (all documented, none silently):
  * The categorical slots `ac_type` / `operator` / `dep` / `arr` do not exist in
    the OpenSky competition-data parquet (no type, no operator, no airports).
    They are replaced by TWO learned embeddings over the *geographic origin*
    and *destination area* of the flight segment (a 2-degree grid), which
    preserve the "route/airport prior" role of the original embeddings but with
    much less information.  This is a HANDICAP for the legacy model.
  * The original was trained on the Italian "OpenFlight" ADS-B (April 2024,
    5 days, positions quantised at 0.01 deg ~ 1.1 km); here it is trained on
    OpenSky competition-data (90 days, 1 Hz, CPR quantisation), i.e. a
    different, larger and more precise dataset.
  * The horizons are extended from 10-120 s to 5-300 s as requested; the
    architecture is horizon-agnostic (a linear head per horizon).
  * The velocity features are the onboard Doppler ones (gs, track) as in the
    original, taken from the OpenSky state vectors.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# scales (identical to the original config.json)
# --------------------------------------------------------------------------- #
SC_XY = 10000.0     # metres -> ENU units
SC_Z = 1000.0       # metres
SC_Z0 = 12000.0     # anchor altitude
SC_GS = 250.0       # m/s
SC_VR = 25.0        # m/s
GRID = 2.0          # degrees per categorical bucket


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
def build_features(x_obs: torch.Tensor) -> torch.Tensor:
    """
    OpenSky [B, T, 9] (lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms, u, v, T)
    -> [B, T, 10] normalised features, in the spirit of the original
       (E, N, z, gs, vr, sin(track), cos(track), + meteo).
    """
    lon = x_obs[..., 0]
    lat = x_obs[..., 1]
    alt_m = x_obs[..., 2] * 10.0
    vx_kmh = x_obs[..., 3]
    vy_kmh = x_obs[..., 4]
    vz = x_obs[..., 5]
    gs_ms = torch.linalg.norm(x_obs[..., 3:5], dim=-1) * (1000.0 / 3600.0)
    trk = torch.atan2(vx_kmh, vy_kmh)
    return torch.stack([
        lon * 111320.0 * torch.cos(torch.deg2rad(lat)) / SC_XY,   # E (m)
        lat * 111132.0 / SC_XY,                                  # N (m)
        alt_m / SC_Z,
        gs_ms / SC_GS,
        vz / SC_VR,
        torch.sin(trk),
        torch.cos(trk),
        x_obs[..., 6] / 50.0,                                    # u wind
        x_obs[..., 7] / 50.0,                                    # v wind
        (x_obs[..., 8] - 220.0) / 30.0,                          # temperature
    ], dim=-1)


def cv_baseline(x_obs: torch.Tensor, horizons_s: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constant-velocity ballistic baseline (the original `baseline_cv`).
    Returns (xy [B,H,2] in ENU metres, z [B,H] in metres).
    """
    lon0 = x_obs[:, -1, 0]
    lat0 = x_obs[:, -1, 1]
    alt0 = x_obs[:, -1, 2] * 10.0
    vx_kmh = x_obs[:, -1, 3]
    vy_kmh = x_obs[:, -1, 4]
    vz = x_obs[:, -1, 5]
    gs_ms = torch.linalg.norm(x_obs[:, -1, 3:5], dim=-1) * (1000.0 / 3600.0)
    trk = torch.atan2(vx_kmh, vy_kmh)
    mlon = 111320.0 * torch.cos(torch.deg2rad(lat0))
    tau = torch.tensor(horizons_s, device=x_obs.device, dtype=x_obs.dtype).view(1, -1)
    vx = gs_ms * torch.sin(trk)
    vy = gs_ms * torch.cos(trk)
    xy = torch.stack([vx[:, None] * tau, vy[:, None] * tau], dim=-1)   # [B,H,2] E,N
    z = vz[:, None] * tau
    return xy, z


def local_to_global(xy: torch.Tensor, z: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
    """ENU (E,N) metres + relative altitude -> global (lon, lat, alt_10m)."""
    lon0 = x_obs[:, -1, 0:1].unsqueeze(1)                # [B,1,1]
    lat0 = x_obs[:, -1, 1:2].unsqueeze(1)                # [B,1,1]
    alt0 = x_obs[:, -1, 2:3].unsqueeze(1)                # [B,1,1]
    mlon = 111320.0 * torch.cos(torch.deg2rad(lat0))
    lon = lon0 + xy[..., 0:1] / mlon
    lat = lat0 + xy[..., 1:2] / 111132.0
    alt = alt0 + z.unsqueeze(-1) / 10.0
    return torch.cat([lon, lat, alt], dim=-1)


# --------------------------------------------------------------------------- #
# Model (faithful re-implementation of TrajTransformer)
# --------------------------------------------------------------------------- #
class TrajTransformer(nn.Module):
    """
    Encoder Transformer + multi-horizon head, residual w.r.t. the CV baseline,
    with a static-context token and learned categorical embeddings.
    """

    def __init__(
        self,
        n_feat: int,
        n_hist: int,
        n_horizons: int,
        vocab_sizes: Dict[str, int],
        emb_dims: Dict[str, int],
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ff: int = 768,
        dropout: float = 0.1,
        predict_logvar: bool = False,
        n_phase: int = 3,
        n_static: int = 2,
    ) -> None:
        super().__init__()
        self.n_hist = n_hist
        self.n_horizons = n_horizons
        self.emb_keys = list(vocab_sizes.keys())

        self.in_proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_hist, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.cls, std=0.02)

        self.emb = nn.ModuleDict({
            k: nn.Embedding(int(vocab_sizes[k]), int(emb_dims[k])) for k in self.emb_keys
        })
        static_in = n_static + sum(int(emb_dims[k]) for k in self.emb_keys)
        self.static_proj = nn.Sequential(nn.Linear(static_in, d_model), nn.GELU())

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)

        self.pool = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.GELU())
        self.heads = nn.ModuleDict({
            "xy": nn.Linear(d_model, n_horizons * 2),
            "z": nn.Linear(d_model, n_horizons),
            "phase": nn.Linear(d_model, n_phase),
        })
        if predict_logvar:
            self.heads["logvar"] = nn.Linear(d_model, n_horizons)

    def forward(self, x: torch.Tensor, static: torch.Tensor,
                cats: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        h = self.in_proj(x) + self.pos
        emb = torch.cat([self.emb[k](cats[:, i]) for i, k in enumerate(self.emb_keys)], dim=-1)
        s = self.static_proj(torch.cat([static, emb], dim=-1)).unsqueeze(1)
        tokens = torch.cat([self.cls.expand(B, -1, -1), s, h], dim=1)
        z = self.encoder(tokens)
        pooled = torch.cat([z[:, 0], z[:, 1], z[:, 2:].mean(dim=1)], dim=-1)
        r = self.pool(pooled)
        out = {
            "xy": self.heads["xy"](r).view(B, self.n_horizons, 2),
            "z": self.heads["z"](r),
            "phase": self.heads["phase"](r),
        }
        if "logvar" in self.heads:
            out["logvar"] = self.heads["logvar"](r).clamp(-8.0, 8.0)
        return out


# --------------------------------------------------------------------------- #
# Loss (the original: Huber on xy, weighted Huber on z, CE on the phase)
# --------------------------------------------------------------------------- #
class LegacyLoss(nn.Module):
    def __init__(self, w_xy: float = 1.0, w_z: float = 0.3, w_phase: float = 0.1,
                 huber_delta: float = 0.1):
        super().__init__()
        self.w_xy, self.w_z, self.w_phase = w_xy, w_z, w_phase
        self.delta = huber_delta

    def forward(self, out: Dict[str, torch.Tensor], y_xy: torch.Tensor,
                y_z: torch.Tensor, phase: torch.Tensor):
        l_xy = F.huber_loss(out["xy"], y_xy, delta=self.delta)
        l_z = F.huber_loss(out["z"], y_z, delta=self.delta)
        l_ph = F.cross_entropy(out["phase"], phase)
        total = self.w_xy * l_xy + self.w_z * l_z + self.w_phase * l_ph
        return total, {"xy": l_xy.detach(), "z": l_z.detach(), "phase": l_ph.detach()}


# --------------------------------------------------------------------------- #
# Phase labels (cruise / climb / descent) as in the original
# --------------------------------------------------------------------------- #
def phase_label(vz: torch.Tensor, alt_10m: torch.Tensor) -> torch.Tensor:
    """0 = cruise, 1 = climb, 2 = descent."""
    climb = (vz > 0.5).long()
    desc = (vz < -0.5).long()
    cruise = torch.where((alt_10m > 600.0) & (vz.abs() <= 0.5),
                         torch.ones_like(climb), torch.zeros_like(climb))
    ph = torch.where(climb == 1, torch.ones_like(climb), torch.where(desc == 1, 2 * torch.ones_like(climb), cruise))
    return ph.long()

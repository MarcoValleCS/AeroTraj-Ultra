# coding=utf-8
"""
Retrieval of similar past trajectories as an intent prior.

Rationale (see docs/EXPERIMENTS.md): at 60-300 s the residual error of the
learned model is not sensor noise but *intent* ("will the aircraft turn?").
The literature's largest recent gains come from exactly this prior:
  * GooDFlight (IEEE TAES 2025)  - goal-conditioned diffusion, ADE -48 %
  * RADiff (ESWA 2026)           - retrieval-augmented diffusion
  * ATP (Neurocomputing 2024)    - "local history of similar past behaviours"

This module implements the retrieval machinery:

  * a DESCRIPTOR of the observed window, used as the retrieval key:
        [ lat, lon, alt/500, sin(psi), cos(psi), |v|/50 ]
    (distance is the weighted L2 over those six components, so the key encodes
    *where* the aircraft is, *where it is heading*, *how high* and *how fast*);

  * a VALUE per candidate: the candidate's own future expressed in its local
    frame (forward, right, up in metres), plus its ground speed and heading.

At query time the neighbours' local frames are rotated into the QUERY's
heading frame, so the retrieved context is directly comparable with the
query's own kinematics.

Everything runs on the GPU with the |q|^2 - 2qc + |c|^2 trick; the candidate
set is kept in device memory (a few hundred MB) and queried in blocks.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# descriptor layout
D_LAT, D_LON, D_ALT, D_SIN, D_COS, D_SPD = 0, 1, 2, 3, 4, 5
N_DESC = 6
# value layout: H*3 (local future) + 1 (speed) + 2 (heading sin/cos)
V_FUT, V_SPD, V_HDG = 0, 1, 2


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
def window_descriptor(x: torch.Tensor, pos_weight: float = 1.0) -> torch.Tensor:
    """
    x: [B, T, 9] observations -> [B, 6] retrieval descriptor.

    `pos_weight` divides the absolute position: with a value > 1 the key stops
    being a route identity and becomes a *neighbourhood* prior, leaving the
    kinematic state (heading, level, speed) to dominate.  The diagnostic sweep
    showed this to give a better oracle (900 m vs 981 m).
    """
    lon0 = x[:, -1, 0]
    lat0 = x[:, -1, 1]
    alt0 = x[:, -1, 2]
    vx = x[:, -1, 3]
    vy = x[:, -1, 4]
    spd = torch.linalg.norm(x[:, -1, 3:5], dim=-1)
    psi = torch.atan2(vx, vy)                      # clockwise from North
    return torch.stack(
        [lat0 / pos_weight, lon0 / pos_weight, alt0 / 500.0,
         torch.sin(psi), torch.cos(psi), spd / 50.0], dim=-1
    )


def descriptor_weights(device) -> torch.Tensor:
    """Relative weight of each descriptor component in the L2 distance."""
    w = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device)
    return w


# --------------------------------------------------------------------------- #
# Local-frame future of a candidate (its own frame)
# --------------------------------------------------------------------------- #
def local_future(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [B,T,9] history, y: [B,H,6] future -> [B,H,3] future expressed in the
    aircraft's OWN local frame (forward, right, up) in metres.
    """
    lon0 = x[:, -1, 0:1].unsqueeze(1)        # [B,1,1]
    lat0 = x[:, -1, 1:2].unsqueeze(1)        # [B,1,1]
    alt0 = x[:, -1, 2:3].unsqueeze(1)        # [B,1,1]
    vx0 = x[:, -1, 3:4].unsqueeze(1)         # [B,1,1]
    vy0 = x[:, -1, 4:5].unsqueeze(1)         # [B,1,1]
    psi = torch.atan2(vx0, vy0)               # [B,1,1]
    sh, ch = torch.sin(psi), torch.cos(psi)

    dx_e = (y[..., 0:1] - lon0) * 111320.0 * torch.cos(torch.deg2rad(lat0.clamp(-80, 80)))
    dy_n = (y[..., 1:2] - lat0) * 111132.0
    dz = (y[..., 2:3] - alt0) * 10.0
    fwd = dx_e * sh + dy_n * ch
    rgt = dx_e * ch - dy_n * sh
    return torch.cat([fwd, rgt, dz], dim=-1)


def candidate_value(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    -> [B, H*3 + 3]  (local future, ground speed, heading sin/cos)
    """
    fut = local_future(x, y)                                  # [B,H,3]
    spd = torch.linalg.norm(x[:, -1, 3:5], dim=-1, keepdim=True)   # [B,1]
    psi = torch.atan2(x[:, -1, 3:4], x[:, -1, 4:5])
    extra = torch.cat([spd, torch.sin(psi), torch.cos(psi)], dim=-1)  # [B,3]
    return torch.cat([fut.reshape(x.shape[0], -1), extra], dim=-1)


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #
class RetrievalIndex:
    """
    GPU-resident index of candidate windows with their futures.

    cand_desc : [M, 6] float32 (GPU)
    cand_val  : [M, H*3+3] float32 (GPU)
    """

    def __init__(self, desc: np.ndarray, val: np.ndarray, device: torch.device):
        self.device = device
        self.cand_desc = torch.as_tensor(desc, dtype=torch.float32, device=device)
        self.cand_val = torch.as_tensor(val, dtype=torch.float32, device=device)
        self.M = self.cand_desc.shape[0]
        self.H = (self.cand_val.shape[1] - 3) // 3
        self._sq = (self.cand_desc ** 2).sum(dim=1)              # [M]
        self.w = descriptor_weights(device)

    # -- search -------------------------------------------------------- #
    def search(self, q_desc: torch.Tensor, k: int = 8, block: int = 256,
               exclude_self: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_desc [B,6] -> (idx [B,k], dist [B,k]) of the k nearest candidates.
        """
        q = q_desc * self.w
        cq = self.cand_desc * self.w
        B = q.shape[0]
        out_idx = torch.empty((B, k), dtype=torch.long, device=self.device)
        out_d = torch.empty((B, k), dtype=torch.float32, device=self.device)
        for s in range(0, B, block):
            qb = q[s:s + block]
            d2 = (qb ** 2).sum(dim=1, keepdim=True) - 2.0 * (qb @ cq.T) + self._sq[None, :]
            if exclude_self:
                # a candidate identical to the query (same window) must be dropped
                same = (d2 < 1e-8)
                d2 = d2.masked_fill(same, float("inf"))
            kk = min(k + (1 if exclude_self else 0), self.M)
            dd, ii = torch.topk(d2, kk, dim=1, largest=False)
            if exclude_self:
                dd, ii = dd[:, 1:], ii[:, 1:]
            out_idx[s:s + block] = ii
            out_d[s:s + block] = dd
        return out_idx, out_d

    # -- context -------------------------------------------------------- #
    def context(self, q_desc: torch.Tensor, q_x: torch.Tensor, k: int = 8,
                exclude_self: bool = False) -> torch.Tensor:
        """
        Retrieval context for a batch of queries.

        Returns [B, k, H, 4] with, per neighbour and per horizon:
            ( forward, right, up )  in the QUERY's heading frame, metres
            speed                  of the neighbour, km/h
        """
        idx, _ = self.search(q_desc, k=k, exclude_self=exclude_self)
        val = self.cand_val[idx]                                # [B,k,H*3+3]
        fut = val[..., :self.H * 3].reshape(val.shape[0], val.shape[1], self.H, 3)
        spd = val[..., self.H * 3:self.H * 3 + 1]              # [B,k,1]
        hsin = val[..., self.H * 3 + 1]                         # [B,k,1]
        hcos = val[..., self.H * 3 + 2]

        # rotate the neighbour's (forward, right) into the query's heading frame
        qpsi = torch.atan2(q_x[:, -1, 3:4], q_x[:, -1, 4:5])   # [B,1]
        dpsi = qpsi - torch.atan2(hsin, hcos)                   # [B,k]
        cs, sn = torch.cos(dpsi).unsqueeze(-1), torch.sin(dpsi).unsqueeze(-1)
        f, r = fut[..., 0], fut[..., 1]
        f_rot = f * cs - r * sn
        r_rot = f * sn + r * cs
        spd_b = spd.expand(-1, -1, self.H)
        return torch.stack([f_rot, r_rot, fut[..., 2], spd_b], dim=-1)   # [B,k,H,4]


# --------------------------------------------------------------------------- #
# Retrieval pool with flight-level bookkeeping
# --------------------------------------------------------------------------- #
class RetrievalPool:
    """
    Candidate pool with flight-level bookkeeping.

    `cand_flight` holds the flight id of every candidate, so that at TRAINING
    time the retrieval can exclude the candidates belonging to the same flight
    as the query.  Without this the model would learn to copy the future of a
    near-duplicate window of the same trajectory, which is impossible at
    inference (the pool only contains past data).
    """

    def __init__(self, desc: np.ndarray, val: np.ndarray,
                 cand_flight: np.ndarray, device: torch.device,
                 pos_weight: float = 1.0):
        self.device = device
        self.pos_weight = float(pos_weight)
        self.cand_desc = torch.as_tensor(desc, dtype=torch.float32, device=device)
        self.cand_val = torch.as_tensor(val, dtype=torch.float32, device=device)
        self.M = self.cand_desc.shape[0]
        self.H = (self.cand_val.shape[1] - 3) // 3
        self._sq = (self.cand_desc ** 2).sum(dim=1)
        # flight id -> candidate indices, kept in a CSR-like layout so that the
        # same-flight masking can be vectorised (a Python loop here costs ~50 s
        # per epoch because of the GPU synchronisations).
        order = np.argsort(cand_flight, kind="stable")
        sorted_f = cand_flight[order]
        uniq = np.unique(sorted_f)
        starts = np.searchsorted(sorted_f, uniq, side="left")
        bounds = np.searchsorted(sorted_f, uniq, side="right")
        self._flight_ids = torch.as_tensor(uniq, dtype=torch.long, device=device)
        self._flight_start = torch.as_tensor(starts, dtype=torch.long, device=device)
        self._flight_end = torch.as_tensor(bounds, dtype=torch.long, device=device)
        self._flight_order = torch.as_tensor(order, dtype=torch.long, device=device)
        # dense map flight id -> row in the CSR structure
        self._flight_row = {int(f): i for i, f in enumerate(uniq)}

    def _mask_same_flight(self, d2: torch.Tensor, q_flight: torch.Tensor) -> None:
        """Set to +inf the columns whose candidate belongs to the query's flight."""
        rows = torch.as_tensor(
            [self._flight_row.get(int(f), 0) for f in q_flight.tolist()],
            dtype=torch.long, device=self.device,
        )
        starts = self._flight_start[rows]
        lens = self._flight_end[rows] - starts
        total = int(lens.sum().item())
        if total == 0:
            return
        offs = torch.cumsum(lens, 0) - lens
        ar = torch.arange(total, device=self.device)
        rep = torch.repeat_interleave(torch.arange(len(lens), device=self.device), lens)
        within = ar - offs[rep]
        flat = self._flight_order[starts[rep] + within]
        d2.view(-1).scatter_(0, flat, float("inf"))

    def search(self, q_desc: torch.Tensor, k: int = 8, block: int = 256,
               q_flight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q_desc [B,6] -> (idx [B,k], dist [B,k]).
        If `q_flight` is given, the candidates of the same flight are excluded.
        """
        B = q_desc.shape[0]
        out_idx = torch.empty((B, k), dtype=torch.long, device=self.device)
        out_d = torch.empty((B, k), dtype=torch.float32, device=self.device)
        for s in range(0, B, block):
            qb = q_desc[s:s + block]
            d2 = (qb ** 2).sum(dim=1, keepdim=True) - 2.0 * (qb @ self.cand_desc.T) + self._sq[None, :]
            if q_flight is not None:
                self._mask_same_flight(d2, q_flight[s:s + block])
            kk = min(k, self.M)
            dd, ii = torch.topk(d2, kk, dim=1, largest=False)
            out_idx[s:s + block] = ii
            out_d[s:s + block] = dd
        return out_idx, out_d

    def context(self, q_desc: torch.Tensor, q_x: torch.Tensor, k: int = 8,
                q_flight: Optional[torch.Tensor] = None,
                pos_scale: float = 1000.0, spd_scale: float = 100.0) -> torch.Tensor:
        """
        -> [B, k, H, 4] : (forward, right, up) in the QUERY's heading frame [m]
                          and the neighbour's ground speed [km/h].

        The output is SCALED (positions by `pos_scale`, speed by `spd_scale`) so
        that the network sees O(1) inputs: an unscaled context (displacements up
        to 30 km) would dominate the query projection and destroy the
        conditioning coming from the horizon embedding and the time features.
        """
        idx, _ = self.search(q_desc, k=k, q_flight=q_flight)
        val = self.cand_val[idx]                                # [B,k,H*3+3]
        fut = val[..., :self.H * 3].reshape(val.shape[0], val.shape[1], self.H, 3)
        spd = val[..., self.H * 3:self.H * 3 + 1]              # [B,k,1]
        hsin = val[..., self.H * 3 + 1]
        hcos = val[..., self.H * 3 + 2]

        qpsi = torch.atan2(q_x[:, -1, 3:4], q_x[:, -1, 4:5])   # [B,1]
        dpsi = qpsi - torch.atan2(hsin, hcos)                   # [B,k]
        cs, sn = torch.cos(dpsi).unsqueeze(-1), torch.sin(dpsi).unsqueeze(-1)
        f, r = fut[..., 0], fut[..., 1]
        f_rot = f * cs - r * sn
        r_rot = f * sn + r * cs
        spd_b = (spd / spd_scale).expand(-1, -1, self.H)
        return torch.stack([f_rot / pos_scale, r_rot / pos_scale,
                            fut[..., 2] / pos_scale, spd_b], dim=-1)   # [B,k,H,4]


def anchor_flight_ids(dataset) -> np.ndarray:
    """flight id of every anchor of a `FlightSlidingWindowDataset`."""
    segs = dataset.segments                       # [N_seg,4] (start, len, flight, day)
    idx = np.searchsorted(segs[:, 0], dataset.anchors, side="right") - 1
    idx = np.clip(idx, 0, len(segs) - 1)
    return segs[idx, 2].astype(np.int64)


def build_pool(dataset, stride: int = 16, device: torch.device = torch.device("cuda"),
               batch: int = 4096, verbose: bool = True,
               pos_weight: float = 1.0) -> RetrievalPool:
    """
    Build the retrieval pool from the TRAIN split: descriptors, futures and
    flight ids of the candidate windows.
    """
    anchors = dataset.anchors[::stride]
    descs, vals = [], []
    for s in range(0, len(anchors), batch):
        x, y = dataset.get_batch_from_anchors(anchors[s:s + batch], device)
        descs.append(window_descriptor(x, pos_weight).cpu().numpy())
        vals.append(candidate_value(x, y).cpu().numpy())
        if verbose and (s // batch) % 20 == 0:
            print(f"  [pool] {min(s+batch, len(anchors))}/{len(anchors)}", flush=True)
    desc = np.concatenate(descs, axis=0).astype(np.float32)
    val = np.concatenate(vals, axis=0).astype(np.float32)
    flights = anchor_flight_ids(dataset)[::stride]
    if verbose:
        print(f"  [pool] {desc.shape[0]:,} candidati, {len(np.unique(flights)):,} voli distinti")
    return RetrievalPool(desc, val, flights, device, pos_weight=pos_weight)


# --------------------------------------------------------------------------- #
# Index building
# --------------------------------------------------------------------------- #
def build_index(
    dataset,
    stride: int = 32,
    device: torch.device = torch.device("cuda"),
    batch: int = 4096,
    verbose: bool = True,
) -> RetrievalIndex:
    """
    Build the retrieval index from a `FlightSlidingWindowDataset` (train split).
    `stride` subsamples the candidate windows (the dataset's own stride is 8).
    """
    ds = dataset
    anchors = ds.anchors[::stride]
    descs, vals = [], []
    for s in range(0, len(anchors), batch):
        chunk = anchors[s:s + batch]
        x, y = ds.get_batch_from_anchors(chunk, device)
        descs.append(window_descriptor(x).cpu().numpy())
        vals.append(candidate_value(x, y).cpu().numpy())
        if verbose and (s // batch) % 20 == 0:
            print(f"  [index] {min(s+batch, len(anchors))}/{len(anchors)}", flush=True)
    desc = np.concatenate(descs, axis=0).astype(np.float32)
    val = np.concatenate(vals, axis=0).astype(np.float32)
    if verbose:
        print(f"  [index] {desc.shape[0]:,} candidati, dim desc {desc.shape[1]}, dim value {val.shape[1]}")
    return RetrievalIndex(desc, val, device)


def save_index(path: str, desc: np.ndarray, val: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, desc=desc, val=val)


def load_index(path: str, device: torch.device = torch.device("cuda")) -> RetrievalIndex:
    z = np.load(path)
    return RetrievalIndex(z["desc"], z["val"], device)

# coding=utf-8
"""
High-Performance Sliding-Window Multi-Horizon Trajectory DataLoader.
Provides both a standard PyTorch Dataset (`FlightSlidingWindowDataset`) and
a vectorized batch iterator (`FastSlidingWindowBatchLoader`) for training and
evaluating over >= 30 days of real OpenSky 5-second trajectories with
12 input steps and 16 prediction horizon points (5s to 10min).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class FlightSlidingWindowDataset(Dataset):
    """
    Sliding-window dataset over contiguous 5-second OpenSky flight trajectories.
    For each valid anchor index `t` inside a flight segment:
      - Input sequence `x`:  `[t - inp_seq_len + 1, ..., t]`  -> shape `[inp_seq_len, 9]`
        Columns: [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms, u_wind, v_wind, temp_k]
      - Target sequence `y`: `[t + h_1, t + h_2, ..., t + h_H]` -> shape `[horizon, 6]`
        Columns: [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms]
        where `H = 16` (>= 10 horizon points) spanning 5s to 10min (600s).
    """

    def __init__(
        self,
        split: str = "train",
        config_path: str = "config.json",
        stride: Optional[int] = None,
        max_samples: Optional[int] = None,
        seed: int = 97,
    ) -> None:
        super().__init__()
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

        self.split = split
        data_dir = Path(self.cfg["data"]["data_dir"])
        win_cfg = self.cfg["window"]

        self.inp_seq_len: int = int(win_cfg["inp_seq_len"])
        self.horizon: int = int(win_cfg["horizon"])
        self.horizon_steps_5s = np.array(win_cfg["horizon_steps_5s"], dtype=np.int64)
        self.horizon_seconds = np.array(win_cfg["horizon_seconds"], dtype=np.float32)
        assert len(self.horizon_steps_5s) == self.horizon >= 10, (
            "Sliding window must have at least 10 horizon points!"
        )

        # Load trajectory points and segment boundaries
        points_path = data_dir / f"{split}_points.npy"
        segs_path = data_dir / f"{split}_segments.npy"
        stats_path = data_dir / "dataset_stats.json"

        self.points = np.load(points_path)  # float32 [N_points, 9]
        self.segments = np.load(segs_path)  # int64 [N_segments, 4]: (start, len, flight_id, day_idx)

        with open(stats_path, "r", encoding="utf-8") as f:
            self.stats = json.load(f)

        self.norm_mean = np.array(self.stats["train_norm_stats"]["mean"], dtype=np.float32)
        self.norm_std = np.array(self.stats["train_norm_stats"]["std"], dtype=np.float32)

        if stride is None:
            stride = (
                int(self.cfg["data"]["window_stride_train"])
                if split == "train"
                else int(self.cfg["data"]["window_stride_eval"])
            )
        self.stride = stride

        # Build valid anchor indices (index of the last input step t_0 in self.points)
        # Fully vectorized: for every segment we take the arithmetic progression
        # [first_anchor, last_anchor] with the given stride. With >100k segments
        # the naive Python loop becomes a real bottleneck.
        max_h_step = int(self.horizon_steps_5s.max())
        seg_starts = self.segments[:, 0].astype(np.int64)
        seg_lens = self.segments[:, 1].astype(np.int64)
        first_anchor = seg_starts + self.inp_seq_len - 1
        last_anchor = seg_starts + seg_lens - 1 - max_h_step
        valid = last_anchor >= first_anchor
        first_valid = first_anchor[valid]
        counts = (last_anchor[valid] - first_valid) // self.stride + 1
        total = int(counts.sum())
        if total > 0:
            # offsets within each progression: repeat(cumsum(counts)-counts) trick
            cum = np.cumsum(counts) - counts
            group_start = np.repeat(cum, counts)
            within = np.arange(total, dtype=np.int64) - group_start
            self.anchors = np.repeat(first_valid, counts) + within * self.stride
        else:
            self.anchors = np.empty(0, dtype=np.int64)
        self.total_strided_windows = len(self.anchors)

        if max_samples is not None and len(self.anchors) > max_samples:
            rng = np.random.default_rng(seed)
            if split == "train":
                chosen = rng.choice(len(self.anchors), size=max_samples, replace=False)
                self.anchors = np.sort(self.anchors[chosen])
            else:
                # Deterministic uniform subsample across all days/flights for evaluation
                idx = np.linspace(0, len(self.anchors) - 1, max_samples, dtype=np.int64)
                self.anchors = self.anchors[idx]

        self.inp_offsets = np.arange(-self.inp_seq_len + 1, 1, dtype=np.int64)  # [-11, -10, ..., 0]

        # Keep Torch tensors ready for fast vectorized batching
        self.points_tensor = torch.from_numpy(self.points)
        self.inp_offsets_tensor = torch.from_numpy(self.inp_offsets)
        self.horizon_steps_tensor = torch.from_numpy(self.horizon_steps_5s)

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        anchor = int(self.anchors[index])
        x = self.points_tensor[anchor + self.inp_offsets_tensor]  # [inp_seq_len, 9]
        y = self.points_tensor[anchor + self.horizon_steps_tensor, :6]  # [horizon, 6]
        return x, y

    def get_batch_from_anchors(
        self, batch_anchors: np.ndarray, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized batch extraction directly from anchor indices."""
        anchors_t = torch.from_numpy(batch_anchors).long()
        inp_idx = anchors_t.unsqueeze(1) + self.inp_offsets_tensor.unsqueeze(0)  # [B, T_in]
        tar_idx = anchors_t.unsqueeze(1) + self.horizon_steps_tensor.unsqueeze(0)  # [B, H]
        x_batch = self.points_tensor[inp_idx].to(device, non_blocking=True)  # [B, T_in, 9]
        y_batch = self.points_tensor[tar_idx, :6].to(device, non_blocking=True)  # [B, H, 6]
        return x_batch, y_batch


class FastSlidingWindowBatchLoader:
    """
    Ultra-fast vectorized batch loader over FlightSlidingWindowDataset.
    Eliminates per-sample Python overhead and supports epoch-level random subsampling
    across all 32 training days of real OpenSky trajectories.
    """

    def __init__(
        self,
        dataset: FlightSlidingWindowDataset,
        batch_size: int = 512,
        shuffle: bool = True,
        max_samples_per_epoch: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
        seed: int = 97,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_samples_per_epoch = max_samples_per_epoch
        self.device = device
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        n = len(self.dataset.anchors)
        if self.max_samples_per_epoch is not None:
            n = min(n, self.max_samples_per_epoch)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        anchors = self.dataset.anchors
        if self.shuffle:
            perm = self.rng.permutation(len(anchors))
            if self.max_samples_per_epoch is not None and len(perm) > self.max_samples_per_epoch:
                perm = perm[: self.max_samples_per_epoch]
            epoch_anchors = anchors[perm]
        else:
            epoch_anchors = anchors
            if self.max_samples_per_epoch is not None and len(epoch_anchors) > self.max_samples_per_epoch:
                epoch_anchors = epoch_anchors[: self.max_samples_per_epoch]

        for start in range(0, len(epoch_anchors), self.batch_size):
            batch_anchors = epoch_anchors[start : start + self.batch_size]
            yield self.dataset.get_batch_from_anchors(batch_anchors, self.device)

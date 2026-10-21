# coding=utf-8
"""
Optimised single-GPU inference path.

The model is launch-bound (see docs/PAPER.md §8): at batch 1 it spends ~99.94 % of
its time in per-kernel launch overhead, so the two things that matter are (a) not
building autograd bookkeeping at all and (b) letting inductor fuse the ~600 tiny
kernels.  Measured on an RTX 4070 with the 90-day checkpoint:

    variant                        batch 1      batch 512     MDE (24 576 test windows)
    eager + no_grad                24.7 ms        23.0 ms      1000.24 m
    eager + inference_mode         20.1 ms        18.3 ms      1000.24 m  (bit-identical)
    compile(reduce-overhead)        4.1 ms         6.5 ms      1000.25 m  (+0.0014 %)

`torch.compile` is not bit-identical: the Gray-Code decoder thresholds at 0.5, so
float noise is amplified into a discrete bit flip on a small tail of windows
(p50 0.79 m, p99 7.4 m, 0.06 % of windows above 100 m).  The aggregate error is
unchanged (verified per horizon, max +0.076 m), which is why it is enabled by
default here.

Also measured and REJECTED, with the reason, in docs/EXPERIMENTS.md §6.11:
  * a hand-captured torch.cuda.CUDAGraph leaves the caching allocator unusable
    ("Offset increment outside graph capture") and breaks torch.compile afterwards;
  * fp16 retrieval distances change the retrieved neighbours for 99.9 % of queries.

Usage:
    from inference import FastPredictor
    fp = FastPredictor("config_90d_300s.json", "checkpoints_90/aerotraj_ultra_best.pt")
    fp.warmup(512)                     # pay the compile cost once, up front
    out = fp(x)                        # x: [B, 12, 9] on the device
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from model import AeroTrajUltra, AeroTrajUltraRetrieval

__all__ = ["FastPredictor"]

COMPILE_MODES = ("reduce-overhead", "default", "max-autotune-no-cudagraphs", "none")


class FastPredictor:
    """Model + checkpoint + the inference-time optimisations, in one object."""

    def __init__(
        self,
        config_path: str = "config_90d_300s.json",
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        compile_mode: str = "reduce-overhead",
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
        retrieval_pool: Optional[Any] = None,
        n_neighbours: int = 8,
        pos_weight: float = 10.0,
    ) -> None:
        self.cfg = json.load(open(config_path, encoding="utf-8"))
        self.device = torch.device(device)
        self.horizon_seconds = self.cfg["window"]["horizon_seconds"]
        self.use_retrieval = retrieval_pool is not None
        self.n_neighbours = n_neighbours
        self.pos_weight = pos_weight
        self.pool = retrieval_pool

        if norm_mean is None or norm_std is None:
            from dataloader import FlightSlidingWindowDataset
            probe = FlightSlidingWindowDataset(
                "test", config_path, max_samples=8)
            norm_mean, norm_std = probe.norm_mean, probe.norm_std

        if self.use_retrieval:
            self.model = AeroTrajUltraRetrieval(
                self.cfg, norm_mean, norm_std, n_neighbours=n_neighbours).to(self.device).eval()
        else:
            self.model = AeroTrajUltra(self.cfg, norm_mean, norm_std).to(self.device).eval()

        self.loaded_from: Optional[str] = None
        if checkpoint and Path(checkpoint).exists():
            sd = torch.load(checkpoint, map_location=self.device, weights_only=False)
            sd = sd.get("model_state_dict", sd)
            from model import normalize_state_dict
            sd = normalize_state_dict(sd)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            self.loaded_from = checkpoint
            if unexpected:
                raise RuntimeError(
                    f"checkpoint {checkpoint} has {len(unexpected)} unexpected keys, "
                    f"e.g. {list(unexpected)[:3]}")
        elif checkpoint:
            raise FileNotFoundError(checkpoint)

        self.compile_mode = compile_mode
        self.fn = self.model
        self._warmed: set = set()
        if compile_mode and compile_mode != "none":
            if compile_mode not in COMPILE_MODES:
                raise ValueError(f"compile_mode must be one of {COMPILE_MODES}")
            self.fn = torch.compile(self.model, mode=compile_mode)

    # ------------------------------------------------------------------ #
    def warmup(self, batch_size: int = 1) -> float:
        """
        Run one forward so that compilation (and inductor's CUDA-graph
        preparation) happens now rather than on the first real request.
        Returns the one-off cost in milliseconds.

        Inductor specialises on the input SHAPE, so every new batch size pays its
        own compilation: the warmed shapes are tracked and each is paid once.
        (An earlier version short-circuited on a single boolean flag, which left
        every batch size other than the first to recompile *during* whatever
        measurement followed, inflating it by up to 2.5x.)
        """
        if batch_size in self._warmed:
            return 0.0
        x = torch.zeros(batch_size, self.cfg["window"]["inp_seq_len"], 9,
                        device=self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            self._run(x)
        torch.cuda.synchronize()
        self._warmed.add(batch_size)
        return (time.perf_counter() - t0) * 1e3

    def _run(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.use_retrieval:
            return self.fn(x)
        from retrieval import window_descriptor
        ctx = self.pool.context(window_descriptor(x, self.pos_weight), x, k=self.n_neighbours)
        return self.fn(x, retr_ctx=ctx)

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor, return_full: bool = False):
        """
        x: [B, 12, 9] float32 on the device (already transferred).
        Returns [B, H, 6] by default, or the full output dict.
        """
        out = self._run(x)
        return out if return_full else out["pred"]

    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        n = sum(p.numel() for p in self.model.parameters())
        return (f"FastPredictor(device={self.device.type}, params={n:,}, "
                f"retrieval={self.use_retrieval}, compile={self.compile_mode}"
                f"{', loaded ' + self.loaded_from if self.loaded_from else ''})")

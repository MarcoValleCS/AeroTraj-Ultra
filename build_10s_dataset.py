# coding=utf-8
"""
Derive a coarser dataset from an existing one by decimating in TIME.

Why this is valid.  The pipeline resamples raw ADS-B onto an exact `dt_seconds`
lattice, and within a segment consecutive points are exactly `dt_seconds` apart
by construction (a gap starts a new segment).  Taking every k-th point of such a
segment therefore produces exactly the dataset that a pipeline run with
`dt_seconds = k * 5` would produce, up to the phase of the lattice, which is
already arbitrary per segment (a flight starts at an arbitrary second).

This avoids re-downloading 40 GB and re-running a 5-hour build.  The segments'
(flight_id, day_idx) columns are preserved, so leakage control and the flight
bookkeeping keep working.

Usage:
    python build_10s_dataset.py --src data365 --dst data365_10s --factor 2

Outputs, in --dst: train/val/test_points.npy, *_segments.npy and a
dataset_stats.json whose train_norm_stats are recomputed on the decimated train
split.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np

SPLITS = ["train", "val", "test"]


def decimate(points: np.ndarray, segments: np.ndarray, factor: int):
    """
    Take every `factor`-th point of every segment.

    points   [N, 9] float32
    segments [S, 4] int64 (start, len, flight_id, day_idx)
    """
    starts = segments[:, 0].astype(np.int64)
    lens = segments[:, 1].astype(np.int64)
    counts = (lens + factor - 1) // factor
    total = int(counts.sum())
    seg_of = np.repeat(np.arange(len(segments), dtype=np.int32), counts)
    local = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts)
    keep = starts[seg_of] + local * factor
    if keep[-1] >= len(points):
        raise ValueError("decimation produced an out-of-range index")
    new_pts = np.ascontiguousarray(points[keep])
    new_starts = np.zeros(len(segments), dtype=np.int64)
    np.cumsum(counts, out=new_starts[1:])
    new_segs = np.column_stack([new_starts, counts, segments[:, 2], segments[:, 3]])
    return new_pts, new_segs.astype(np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data365")
    ap.add_argument("--dst", default="data365_10s")
    ap.add_argument("--factor", type=int, default=2, help="5 s -> factor*5 s")
    ap.add_argument("--inp-seq-len", type=int, default=12)
    ap.add_argument("--max-horizon-step", type=int, default=30)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    factor = int(args.factor)
    stats_src = json.load(open(src / "dataset_stats.json", encoding="utf-8"))
    out: Dict = dict(stats_src)
    out["grid_seconds"] = 5 * factor
    out["derived_from"] = {"data_dir": str(src), "factor": factor,
                           "note": "decimated in time; segments preserved"}
    out.setdefault("splits", {})

    for split in SPLITS:
        t0 = time.perf_counter()
        pts = np.load(src / f"{split}_points.npy")
        segs = np.load(src / f"{split}_segments.npy")
        n0, s0 = len(pts), len(segs)
        new_pts, new_segs = decimate(pts, segs, factor)
        np.save(dst / f"{split}_points.npy", new_pts)
        np.save(dst / f"{split}_segments.npy", new_segs)
        print(f"[{split}] punti {n0:,} -> {len(new_pts):,} | segmenti {s0:,} -> {len(new_segs):,} "
              f"| {time.perf_counter() - t0:.1f} s")

        if split in out["splits"]:
            sp = out["splits"][split]
            sp["num_5s_points"] = int(len(new_pts))
            sp["num_5s_points_before_decimation"] = int(n0)
            sp["points_file"] = str(dst / f"{split}_points.npy")
            sp["segments_file"] = str(dst / f"{split}_segments.npy")
        if split == "train":
            out["train_norm_stats"] = {
                "mean": new_pts.mean(axis=0).tolist(),
                "std": new_pts.std(axis=0).tolist(),
                "min": new_pts.min(axis=0).tolist(),
                "max": new_pts.max(axis=0).tolist(),
            }
        del pts, segs, new_pts, new_segs

    # recompute the stride-1 window count for the new grid
    span = args.inp_seq_len + args.max_horizon_step
    for split in SPLITS:
        segs = np.load(dst / f"{split}_segments.npy", mmap_mode="r")
        tot = int(np.sum(np.maximum(0, segs[:, 1].astype(np.int64) - span + 1)))
        if split in out["splits"]:
            out["splits"][split]["total_sliding_windows_stride1"] = tot
        print(f"[{split}] finestre (stride 1, finestra {span} punti): {tot:,}")

    out["total_pipeline_seconds"] = None
    json.dump(out, open(dst / "dataset_stats.json", "w", encoding="utf-8"), indent=2)
    total = sum((dst / f"{s}_points.npy").stat().st_size for s in SPLITS) / 1e9
    print(f"\n[done] {dst} — {total:.2f} GB, griglia {5 * factor} s")
    print("       normalizzazione ricalcolata sullo split train decimato")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Real Air Traffic Data Acquisition & Preprocessing Pipeline using pyopensky.
Streams >= 30 days (default: 40 consecutive days: 32 Train, 4 Val, 4 Test) of real
OpenSky ADS-B + Meteorological trajectories via pyopensky (S3Client / Trino / REST),
resamples to an exact 5-second grid, and builds memory-mapped trajectory stores
for sliding-window multi-horizon (5s - 10min) prediction.
"""
from __future__ import annotations

import concurrent.futures as cf
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.fs as pafs
import pyarrow.parquet as pq
from pyopensky.rest import REST
from pyopensky.s3 import S3Client
from pyopensky.trino import Trino


FEATURE_NAMES = [
    "lon",       # 0: Longitude [deg, WGS-84]
    "lat",       # 1: Latitude [deg, WGS-84]
    "alt_10m",   # 2: Altitude [10m units, i.e. meters / 10]
    "vx_kmh",    # 3: Eastward velocity Vx [km/h]
    "vy_kmh",    # 4: Northward velocity Vy [km/h]
    "vz_ms",     # 5: Vertical velocity Vz [m/s]
    "u_wind",    # 6: Eastward wind u [m/s]
    "v_wind",    # 7: Northward wind v [m/s]
    "temp_k",    # 8: Atmospheric temperature [K]
]


class PyOpenSkyTrajectoryPipeline:
    """
    Unified pyopensky data pipeline supporting:
      1. OpenSky S3 bucket archive via pyopensky.s3.S3Client ('competition-data' & 'data-samples')
      2. OpenSky Historical Trino database via pyopensky.trino.Trino
      3. OpenSky Live/Recent REST API via pyopensky.rest.REST
    """

    def __init__(self, config_path: str = "config.json") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.data_cfg = self.cfg["data"]
        self.win_cfg = self.cfg["window"]
        self.data_dir = Path(self.data_cfg["data_dir"])
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pyopensky S3Client
        self.opensky_s3 = S3Client()
        self.arrow_s3 = pafs.S3FileSystem(
            endpoint_override="s3.opensky-network.org:443",
            scheme="https",
            anonymous=True,
        )

    def list_available_opensky_days(self, bucket: str = "competition-data") -> List[str]:
        """Use pyopensky.s3.S3Client to list daily trajectory parquet files on OpenSky S3."""
        objs = list(self.opensky_s3.s3client.list_objects(bucket))
        parquet_days = sorted(
            [
                obj.object_name
                for obj in objs
                if obj.object_name and obj.object_name.endswith(".parquet")
            ]
        )
        return parquet_days

    def fetch_via_trino(
        self,
        start: str,
        stop: str,
        bounds: Tuple[float, float, float, float] = (-15.0, 30.0, 35.0, 65.0),
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical ADS-B state vectors directly from pyopensky.trino.Trino
        when OpenSky Trino credentials are configured.
        """
        trino = Trino()
        df = trino.history(start=start, stop=stop, bounds=bounds)
        return df

    def fetch_via_rest_live(
        self,
        bounds: Tuple[float, float, float, float] = (-15.0, 30.0, 35.0, 65.0),
    ) -> pd.DataFrame:
        """Fetch live state vectors from pyopensky.rest.REST."""
        rest = REST()
        return rest.states(bounds=bounds)

    def _process_raw_dataframe_to_5s_segments(
        self, df: pd.DataFrame, day_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Clean raw 1-second OpenSky trajectory dataframe, filter to region & physical bounds,
        downsample to an exact 5-second grid (dt = 5s), and extract contiguous flight segments.

        Returns:
          points_array: float32 array of shape [N_points, 9]
          segments_array: int64 array of shape [N_segments, 4] -> (start_idx, length, flight_id, day_idx)
        """
        bounds = self.data_cfg["region_bounds"]
        dt_sec = int(self.data_cfg["dt_seconds"])
        min_steps = int(self.data_cfg["min_segment_steps"])

        # Filter spatial & altitude bounds
        mask = (
            df["latitude"].between(bounds["lat_min"], bounds["lat_max"])
            & df["longitude"].between(bounds["lon_min"], bounds["lon_max"])
            & df["altitude"].between(bounds["alt_min_ft"], bounds["alt_max_ft"])
        )
        df = df.loc[mask].copy()
        if df.empty:
            return np.empty((0, 9), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

        # Convert timestamp to integer Unix seconds and keep exact multiples of dt_sec (5s)
        ts_sec = df["timestamp"].astype("int64") // 10**9
        df["ts_sec"] = ts_sec
        df = df[df["ts_sec"] % dt_sec == 0]
        if df.empty:
            return np.empty((0, 9), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

        df = df.sort_values(["flight_id", "ts_sec"]).drop_duplicates(
            subset=["flight_id", "ts_sec"], keep="first"
        )

        # Identify contiguous 5-second segments within each flight
        dt_diff = df.groupby("flight_id")["ts_sec"].diff()
        new_seg = (dt_diff != dt_sec).to_numpy()
        df["seg_id"] = np.cumsum(new_seg)

        # Filter segments with at least min_steps (e.g. >= 140 steps = 700s > 11.6 min)
        seg_sizes = df.groupby("seg_id").size()
        valid_seg_ids = seg_sizes[seg_sizes >= min_steps].index
        df = df[df["seg_id"].isin(valid_seg_ids)].copy()
        if df.empty:
            return np.empty((0, 9), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

        # Interpolate any isolated NaNs in groundspeed, track, vertical_rate within segments
        for col in ["groundspeed", "track", "vertical_rate", "u_component_of_wind", "v_component_of_wind", "temperature"]:
            if df[col].isna().any():
                df[col] = df.groupby("seg_id")[col].transform(
                    lambda s: s.interpolate(method="linear", limit=6).bfill().ffill()
                )
        df = df.dropna(subset=["latitude", "longitude", "altitude", "groundspeed", "track", "vertical_rate"])

        # Re-verify segment lengths after any dropna
        seg_sizes = df.groupby("seg_id").size()
        valid_seg_ids = seg_sizes[seg_sizes >= min_steps].index
        df = df[df["seg_id"].isin(valid_seg_ids)].reset_index(drop=True)
        if df.empty:
            return np.empty((0, 9), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

        # Physical unit conversions matching FlightBERT / FlightBERT++ V2 + Weather:
        # 1. Longitude & Latitude in degrees
        lon = df["longitude"].to_numpy(dtype=np.float64)
        lat = df["latitude"].to_numpy(dtype=np.float64)
        # 2. Altitude in 10m units (1 ft = 0.3048 m = 0.03048 * 10m)
        alt_10m = df["altitude"].to_numpy(dtype=np.float64) * 0.03048
        # 3. Velocities: groundspeed (kts -> km/h via * 1.852), track (deg CW from North)
        gs_kmh = np.clip(df["groundspeed"].to_numpy(dtype=np.float64) * 1.852, 0.0, 1300.0)
        track_rad = np.radians(df["track"].to_numpy(dtype=np.float64))
        vx_kmh = gs_kmh * np.sin(track_rad)
        vy_kmh = gs_kmh * np.cos(track_rad)
        # 4. Vertical rate (ft/min -> m/s via * 0.00508)
        vz_ms = np.clip(df["vertical_rate"].to_numpy(dtype=np.float64) * 0.00508, -40.0, 40.0)
        # 5. Meteorological context (m/s and Kelvin)
        u_wind = np.clip(df["u_component_of_wind"].to_numpy(dtype=np.float64), -120.0, 120.0)
        v_wind = np.clip(df["v_component_of_wind"].to_numpy(dtype=np.float64), -120.0, 120.0)
        temp_k = np.clip(df["temperature"].to_numpy(dtype=np.float64), 180.0, 330.0)

        points = np.column_stack(
            [lon, lat, alt_10m, vx_kmh, vy_kmh, vz_ms, u_wind, v_wind, temp_k]
        ).astype(np.float32)

        # Build segment table [start_idx, length, flight_id, day_idx]
        # Reject any segment that has an unphysical jump (> 0.05 deg in 5s)
        seg_ids = df["seg_id"].to_numpy()
        flight_ids = df["flight_id"].to_numpy(dtype=np.int64)
        _, seg_starts, seg_counts = np.unique(seg_ids, return_index=True, return_counts=True)
        order = np.argsort(seg_starts)
        seg_starts = seg_starts[order]
        seg_counts = seg_counts[order]

        clean_points_list: List[np.ndarray] = []
        clean_segs_list: List[Tuple[int, int, int, int]] = []
        curr_offset = 0

        for st, cnt in zip(seg_starts, seg_counts):
            seg_pts = points[st : st + cnt]
            # Verify no unphysical teleportation in lon/lat/alt
            dlon = np.abs(np.diff(seg_pts[:, 0]))
            dlat = np.abs(np.diff(seg_pts[:, 1]))
            dalt = np.abs(np.diff(seg_pts[:, 2]))
            if np.max(dlon) > 0.04 or np.max(dlat) > 0.04 or np.max(dalt) > 30.0:
                continue

            f_id = int(flight_ids[st])
            clean_points_list.append(seg_pts)
            clean_segs_list.append((curr_offset, int(cnt), f_id, day_idx))
            curr_offset += int(cnt)

        if not clean_points_list:
            return np.empty((0, 9), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

        return (
            np.concatenate(clean_points_list, axis=0),
            np.array(clean_segs_list, dtype=np.int64),
        )

    # ------------------------------------------------------------------ #
    # Dataset signature: detects any change of the preprocessing contract
    # so that a stale dataset is never silently reused.
    # ------------------------------------------------------------------ #
    def dataset_signature(self, selected_files: Optional[List[str]] = None) -> str:
        payload = {
            "num_total_days": int(self.data_cfg["num_total_days"]),
            "train_days": int(self.data_cfg["train_days"]),
            "val_days": int(self.data_cfg["val_days"]),
            "test_days": int(self.data_cfg["test_days"]),
            "row_groups_per_day": int(self.data_cfg.get("row_groups_per_day", 1)),
            "dt_seconds": int(self.data_cfg["dt_seconds"]),
            "min_segment_steps": int(self.data_cfg["min_segment_steps"]),
            "region_bounds": self.data_cfg["region_bounds"],
            "feature_names": FEATURE_NAMES,
            "days": selected_files or [],
        }
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]

    def _process_day(
        self,
        day_file: str,
        day_idx: int,
        cols: List[str],
        rgs_per_day: int,
        cache_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Download + preprocess a single day of OpenSky data.
        The result is cached on disk (npz) so that a rebuild is resumable and
        re-running the pipeline does not re-download the same bytes.

        Returns (points [N,9] float32, segments [M,4] int64, raw_rows_read).
        """
        cache_path = cache_dir / f"{day_file.replace('.parquet', '')}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path) as z:
                    return z["points"], z["segments"], int(z["raw_rows"])
            except Exception:  # corrupted cache entry -> recompute
                cache_path.unlink(missing_ok=True)

        s3_key = f"competition-data/{day_file}"
        pf = pq.ParquetFile(s3_key, filesystem=self.arrow_s3)
        n_rgs = min(rgs_per_day, pf.num_row_groups)

        pts_list: List[np.ndarray] = []
        seg_list: List[np.ndarray] = []
        offset = 0
        raw_rows = 0

        for rg_idx in range(n_rgs):
            table = pf.read_row_group(rg_idx, columns=cols)
            raw_rows += table.num_rows
            df = table.to_pandas()
            del table

            pts, segs = self._process_raw_dataframe_to_5s_segments(df, day_idx)
            del df
            if len(pts) > 0:
                segs[:, 0] += offset
                offset += len(pts)
                pts_list.append(pts)
                seg_list.append(segs)
            gc.collect()

        if pts_list:
            day_pts = np.concatenate(pts_list, axis=0)
            day_segs = np.concatenate(seg_list, axis=0)
        else:
            day_pts = np.empty((0, 9), dtype=np.float32)
            day_segs = np.empty((0, 4), dtype=np.int64)

        tmp_path = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, points=day_pts, segments=day_segs, raw_rows=np.int64(raw_rows))
        os.replace(tmp_path, cache_path)
        return day_pts, day_segs, raw_rows

    def build_dataset_splits(self, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Stream N consecutive days of real OpenSky trajectories via pyopensky S3:
          - Train: first `train_days` days
          - Val:   next `val_days` days
          - Test:  final `test_days` days
        Saves contiguous .npy files (points + segment table) to self.data_dir.

        Days are downloaded/processed in parallel (network bound) and cached
        individually, so an interrupted build resumes instead of restarting.
        """
        stats_file = self.data_dir / "dataset_stats.json"

        available_files = self.list_available_opensky_days("competition-data")
        num_total_days = int(self.data_cfg["num_total_days"])
        train_days = int(self.data_cfg["train_days"])
        val_days = int(self.data_cfg["val_days"])
        test_days = int(self.data_cfg["test_days"])
        rgs_per_day = int(self.data_cfg.get("row_groups_per_day", 1))

        assert train_days >= 30, "Must train on at least 30 days of real OpenSky data!"
        assert len(available_files) >= num_total_days, (
            f"OpenSky S3 exposes only {len(available_files)} daily files, "
            f"{num_total_days} requested."
        )
        selected_files = available_files[:num_total_days]
        signature = self.dataset_signature(selected_files)

        if stats_file.exists() and not force_rebuild:
            with open(stats_file, "r", encoding="utf-8") as f:
                stats = json.load(f)
            if (
                stats.get("num_total_days", 0) >= num_total_days
                and stats.get("dataset_signature") == signature
            ):
                print(
                    f"[DataPipeline] Reusing existing {stats['num_total_days']}-day dataset "
                    f"in {self.data_dir} (signature {signature})"
                )
                return stats

        splits = {
            "train": selected_files[:train_days],
            "val": selected_files[train_days : train_days + val_days],
            "test": selected_files[train_days + val_days : train_days + val_days + test_days],
        }

        cols = [
            "flight_id",
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
            "groundspeed",
            "track",
            "vertical_rate",
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
        ]

        summary_stats: Dict[str, Any] = {
            "source": "pyopensky.s3.S3Client (s3.opensky-network.org/competition-data)",
            "num_total_days": num_total_days,
            "train_days": train_days,
            "val_days": val_days,
            "test_days": test_days,
            "dt_seconds": self.data_cfg["dt_seconds"],
            "row_groups_per_day": rgs_per_day,
            "min_segment_steps": int(self.data_cfg["min_segment_steps"]),
            "dataset_signature": signature,
            "feature_names": FEATURE_NAMES,
            "splits": {},
        }

        cache_dir = self.data_dir / "_cache_days"
        cache_dir.mkdir(parents=True, exist_ok=True)
        n_workers = int(self.cfg.get("training", {}).get("num_workers", 6))

        global_day_idx = 0
        t_start_all = time.time()

        for split_name, day_files in splits.items():
            print(
                f"\n=== Processing split '{split_name}' "
                f"({len(day_files)} days x {rgs_per_day} row-groups) ==="
            )
            t_split = time.time()

            # --- Stage 1: download + preprocess days in parallel ---------
            day_results: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}

            def _task(args):
                fname, didx = args
                t0 = time.time()
                pts, segs, raw = self._process_day(fname, didx, cols, rgs_per_day, cache_dir)
                return fname, didx, pts, segs, raw, time.time() - t0

            tasks = [(fname, global_day_idx + i) for i, fname in enumerate(day_files)]
            done = 0
            with cf.ThreadPoolExecutor(max_workers=max(1, n_workers)) as executor:
                for fname, didx, pts, segs, raw, dt in executor.map(_task, tasks):
                    day_results[fname] = (pts, segs, raw)
                    done += 1
                    print(
                        f"  [{split_name.upper()} {done:03d}/{len(day_files):03d}] {fname} -> "
                        f"{len(pts):>9,} 5s-points | {len(segs):>6,} segments | {raw:>11,} raw rows "
                        f"| {dt:5.1f}s",
                        flush=True,
                    )
            global_day_idx += len(day_files)

            # --- Stage 2: assemble contiguous arrays on disk -------------
            total_pts = sum(v[0].shape[0] for v in day_results.values())
            total_segs = sum(v[1].shape[0] for v in day_results.values())
            pts_path = self.data_dir / f"{split_name}_points.npy"
            segs_path = self.data_dir / f"{split_name}_segments.npy"

            all_pts = np.lib.format.open_memmap(
                pts_path, mode="w+", dtype=np.float32, shape=(total_pts, 9)
            )
            all_segs = np.lib.format.open_memmap(
                segs_path, mode="w+", dtype=np.int64, shape=(total_segs, 4)
            )
            p_off, s_off = 0, 0
            raw_rows_processed = 0
            unique_flights = 0
            for fname in day_files:
                pts, segs, raw = day_results[fname]
                if len(pts):
                    all_pts[p_off : p_off + len(pts)] = pts
                    all_segs[s_off : s_off + len(segs)] = segs
                    # re-base segment start indices to the global concatenated array
                    all_segs[s_off : s_off + len(segs), 0] += p_off
                    p_off += len(pts)
                    s_off += len(segs)
                    unique_flights += int(len(np.unique(segs[:, 2])))
                raw_rows_processed += raw
            all_pts.flush()
            all_segs.flush()
            del all_pts, all_segs
            gc.collect()

            # --- Statistics ---------------------------------------------
            inp_len = int(self.win_cfg["inp_seq_len"])
            max_h_step = max(self.win_cfg["horizon_steps_5s"])
            min_win_span = inp_len + max_h_step
            segs_view = np.load(segs_path, mmap_mode="r")
            total_possible_windows = int(
                np.sum(np.maximum(0, segs_view[:, 1].astype(np.int64) - min_win_span + 1))
            )
            del segs_view

            summary_stats["splits"][split_name] = {
                "num_days": len(day_files),
                "first_day": day_files[0].replace(".parquet", ""),
                "last_day": day_files[-1].replace(".parquet", ""),
                "raw_1s_rows_streamed": raw_rows_processed,
                "num_contiguous_5s_trajectories": int(total_segs),
                "num_unique_flights": int(unique_flights),
                "num_5s_points": int(total_pts),
                "total_sliding_windows_stride1": total_possible_windows,
                "points_file": str(pts_path),
                "segments_file": str(segs_path),
                "build_seconds": round(time.time() - t_split, 1),
            }
            print(
                f"  -> split '{split_name}' ready: {total_pts:,} points | {total_segs:,} segments | "
                f"{unique_flights:,} flights | {total_possible_windows:,} windows (stride 1) | "
                f"{summary_stats['splits'][split_name]['build_seconds']:.1f}s"
            )

            if split_name == "train":
                # Normalization statistics computed on the TRAIN split only
                train_pts = np.load(pts_path, mmap_mode="r")
                mean_vec = np.asarray(train_pts).mean(axis=0).tolist()
                std_vec = np.maximum(np.asarray(train_pts).std(axis=0), 1e-5).tolist()
                min_vec = np.asarray(train_pts).min(axis=0).tolist()
                max_vec = np.asarray(train_pts).max(axis=0).tolist()
                summary_stats["train_norm_stats"] = {
                    "mean": mean_vec,
                    "std": std_vec,
                    "min": min_vec,
                    "max": max_vec,
                }
                del train_pts
                gc.collect()

        summary_stats["total_pipeline_seconds"] = round(time.time() - t_start_all, 2)
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(summary_stats, f, indent=2)

        print(
            f"\n[DataPipeline] Completed {num_total_days}-day OpenSky build in "
            f"{summary_stats['total_pipeline_seconds']:.1f}s!"
        )
        print(json.dumps(summary_stats["splits"], indent=2))
        return summary_stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the preprocessed OpenSky trajectory dataset.")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--rebuild", action="store_true", help="Force a full re-download/rebuild")
    args = parser.parse_args()

    pipeline = PyOpenSkyTrajectoryPipeline(args.config)
    pipeline.build_dataset_splits(force_rebuild=args.rebuild)

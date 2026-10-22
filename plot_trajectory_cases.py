# coding=utf-8
"""
Six-case 3-D trajectory comparison, in the style of the published figures of the
field (inputs / ground truth / each model, one panel per manoeuvre type).

Rendering is Plotly (WebGL-quality 3-D, clean white scenes) exported to PNG with
Kaleido; the matplotlib version of this figure was visibly the weak point of the
report.

The six windows are selected DETERMINISTICALLY from the test split by a criterion
on the ground-truth future — en-route only, and mutually exclusive — so the figure
is reproducible and cannot be cherry-picked by hand:

  (a) climbing                    sustained positive vertical rate, little turn
  (b) climbing and turning right  positive vertical rate + heading increasing
  (c) descending                  sustained negative vertical rate
  (d) maintain                    level and straight
  (e) turn                        large heading change, level
  (f) climb then maintain         vertical rate positive early, ~0 late

These are qualitative illustrations on real data, not a metric: the selection uses
the truth, exactly like the published figures they follow.

Run:
  python plot_trajectory_cases.py --config config_90d_300s.json \
      --model "AeroTraj-Ultra (ours)=config_90d_300s.json=checkpoints_bench/ours.pt" \
      --model "FlightBERT++ V2=config_90d_300s.json=checkpoints_bench/fbpp_v2.pt" \
      --model "Deep LSTM=config_90d_300s.json=checkpoints_bench/lstm.pt" \
      --out docs/figures/trajectory_cases.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataloader import FlightSlidingWindowDataset
from model import (AeroTrajUltra, FlightBERT_PP_V2_Baseline, LSTMRegressionBaseline,
                   normalize_state_dict)

CASES = [
    ("(a) Climbing", "climb"),
    ("(b) Climbing and turning right", "climb_right"),
    ("(c) Descending", "descend"),
    ("(d) Maintain", "maintain"),
    ("(e) Turn", "turn"),
    ("(f) Climb then maintain", "climb_level"),
]
MODEL_COLORS = ["#d62728", "#ff9f1c", "#9467bd", "#17becf"]
COL_IN, COL_TGT = "#1a3fa0", "#2ca02c"


# ------------------------------------------------------------------ geometry
def heading(y: np.ndarray) -> np.ndarray:
    return np.arctan2(y[..., 3], y[..., 4])


def scores(y: np.ndarray) -> Dict[str, float]:
    """
    Criterion per manoeuvre type, computed from the POSITIONS in the ground-truth
    future, not from the reported vertical rate: ADS-B vertical rate is frequently
    stale or noisy, and a window selected on it can descend while being labelled
    "climbing" (observed, and the reason this function was rewritten).
    """
    alt = y[:, 2] * 10.0
    dz = float(alt[-1] - alt[0])
    dalt = np.diff(alt)
    dlon = np.diff(y[:, 0])
    dlat = np.diff(y[:, 1])
    trk = np.unwrap(np.arctan2(dlon * np.cos(np.deg2rad(y[0, 1])), dlat))
    dhd = float(np.degrees(trk[-1] - trk[0]))
    half = max(1, len(dalt) // 2)
    return {
        "climb": 0.02 * dz - 0.5 * abs(dhd),
        "climb_right": 0.02 * dz + 0.3 * dhd if dhd > 0 else -1e9,
        "descend": -0.02 * dz - 0.5 * abs(dhd),
        "maintain": float(-np.abs(dalt).mean() - 0.06 * abs(dhd) - 0.01 * abs(dz)),
        "turn": abs(dhd) - 0.2 * abs(dz),
        "climb_level": 0.02 * (dalt[:half].sum() - dalt[half:].sum()) - 0.5 * abs(dhd),
    }


def pick_windows(ds: FlightSlidingWindowDataset, n_probe: int = 20000) -> Dict[str, int]:
    """Greedy, MUTUALLY EXCLUSIVE selection, en-route only."""
    anchors = ds.anchors[:n_probe]
    _, y = ds.get_batch_from_anchors(anchors, "cpu")
    y = y.numpy()
    feats = [scores(y[i]) for i in range(len(anchors))]
    alt = y[:, :, 2] * 10.0
    dz = alt[:, -1] - alt[:, 0]
    half = max(1, alt.shape[1] // 2)
    used: set = set()
    chosen: Dict[str, int] = {}
    for key in ["climb", "climb_right", "descend", "maintain", "turn", "climb_level"]:
        best, best_i = -1e18, -1
        for i, f in enumerate(feats):
            if i in used:
                continue
            # en-route only: without it "maintain" is won by a taxiing aircraft
            if float(alt[i].min()) < 3000.0:
                continue
            if key in ("climb", "climb_right") and dz[i] < 300.0:
                continue
            if key == "descend" and dz[i] > -300.0:
                continue
            if key == "turn" and abs(dz[i]) > 150.0:
                continue
            if key == "maintain" and abs(dz[i]) > 100.0:
                continue
            if key == "climb_level" and (alt[i, :half].max() - alt[i, 0]) < 200.0:
                continue
            if f[key] > best:
                best, best_i = f[key], i
        chosen[key] = best_i
        used.add(best_i)
    return chosen


# ------------------------------------------------------------------ rendering
def to_local(lon, lat, alt_m, lon0, lat0, alt0):
    """
    Convert (lon, lat, alt) into a LOCAL METRIC FRAME anchored at the last observed
    position: east/north metres on the horizontal plane, metres in altitude.

    The published figures of this field mix degrees and metres on the three axes,
    which compresses a real trajectory into a blob because one degree of longitude
    is ~70 km at European latitudes and the altitude spans only a few km.  Putting
    all three axes in metres makes the 3-D shape geometrically honest, so a panel
    can be read as an actual view of the manoeuvre.
    """
    east = (lon - lon0) * 111320.0 * np.cos(np.deg2rad(lat0))
    north = (lat - lat0) * 111132.0
    return east, north, alt_m - alt0


def axis_range(v: np.ndarray, floor: float) -> List[float]:
    span = max(float(np.ptp(v)), floor)
    c = 0.5 * (float(v.min()) + float(v.max()))
    pad = max(0.16 * span, 0.06 * floor)
    return [c - 0.5 * span - pad, c + 0.5 * span + pad]


def make_figure(panels, model_names: List[str]) -> go.Figure:
    fig = make_subplots(rows=3, cols=2, specs=[[{"type": "scene"}] * 2] * 3,
                        subplot_titles=[p[0] for p in panels],
                        vertical_spacing=0.045, horizontal_spacing=0.02)

    for k, (label, x_in, y_tgt, preds) in enumerate(panels):
        r, c = k // 2 + 1, k % 2 + 1
        first = k == 0
        lon0, lat0, alt0 = x_in[-1, 0], x_in[-1, 1], x_in[-1, 2] * 10.0
        ei, ni, ui = to_local(x_in[:, 0], x_in[:, 1], x_in[:, 2] * 10.0, lon0, lat0, alt0)
        et, nt, ut = to_local(y_tgt[:, 0], y_tgt[:, 1], y_tgt[:, 2] * 10.0, lon0, lat0, alt0)
        fig.add_trace(go.Scatter3d(
            x=ei, y=ni, z=ui, mode="lines+markers", name="Inputs (last 60 s)",
            line=dict(color=COL_IN, width=4),
            marker=dict(size=3.2, symbol="diamond", color=COL_IN),
            legendgroup="inputs", showlegend=first, hoverinfo="skip"), row=r, col=c)
        fig.add_trace(go.Scatter3d(
            x=et, y=nt, z=ut, mode="lines+markers", name="Ground truth",
            line=dict(color=COL_TGT, width=4, dash="dot"),
            marker=dict(size=3.2, symbol="cross", color=COL_TGT),
            legendgroup="truth", showlegend=first, hoverinfo="skip"), row=r, col=c)
        for i, (nm, p) in enumerate(preds):
            ep, np_, up = to_local(p[:, 0], p[:, 1], p[:, 2] * 10.0, lon0, lat0, alt0)
            fig.add_trace(go.Scatter3d(
                x=ep, y=np_, z=up, mode="lines+markers", name=nm,
                line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=4),
                marker=dict(size=2.6, color=MODEL_COLORS[i % len(MODEL_COLORS)]),
                legendgroup=nm, showlegend=first, hoverinfo="skip"), row=r, col=c)

        e = np.concatenate([ei, et]); n = np.concatenate([ni, nt]); u = np.concatenate([ui, ut])
        scene = dict(
            xaxis=dict(title="East [m]", range=axis_range(e, 800.0), nticks=4,
                       tickformat=".0f", showbackground=True, backgroundcolor="white",
                       gridcolor="#e6e6e6", zeroline=False),
            yaxis=dict(title="North [m]", range=axis_range(n, 800.0), nticks=4,
                       tickformat=".0f", showbackground=True, backgroundcolor="white",
                       gridcolor="#e6e6e6", zeroline=False),
            zaxis=dict(title="Altitude [m]", range=axis_range(u, 400.0), nticks=4,
                       tickformat=".0f", showbackground=True, backgroundcolor="white",
                       gridcolor="#e6e6e6", zeroline=False),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.85)),
        )
        fig.update_layout(**{f"scene{k + 1}" if k else "scene": scene})

    fig.update_layout(
        template="plotly_white",
        width=1520, height=1760,
        margin=dict(l=6, r=6, t=98, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5,
                    font=dict(size=13), bgcolor="rgba(255,255,255,0.95)",
                    bordercolor="#dddddd", borderwidth=1),
        font=dict(family="DejaVu Sans, Arial, sans-serif", size=12),
        title=dict(
            text="4-D trajectory prediction on held-out real OpenSky traffic — "
                 "one window per manoeuvre type",
            x=0.5, y=0.995, xanchor="center", font=dict(size=19)),
    )
    # NOTE: add_annotation, not update_layout(annotations=...): the latter would
    # REPLACE the subplot titles created by make_subplots.
    fig.add_annotation(
        text="local metric frame anchored at the last observed position — all three axes "
             "in metres, so each panel is a geometrically honest view of the manoeuvre. "
             "Windows are selected with the ground-truth future (as in the published "
             "figures of the field): qualitative illustration, not a metric.",
        x=0.5, y=-0.028, xref="paper", yref="paper", showarrow=False,
        font=dict(size=11, color="#666666"))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_90d_300s.json")
    ap.add_argument("--model", action="append", default=None,
                    help="NAME=config=checkpoint (repeatable)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-probe", type=int, default=20000)
    ap.add_argument("--out", default="docs/figures/trajectory_cases.png")
    ap.add_argument("--scale", type=float, default=1.6)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    ds = FlightSlidingWindowDataset(args.split, args.config, max_samples=args.n_probe)
    print(f"[cases] {len(ds.anchors):,} finestre di {args.split}")

    chosen = pick_windows(ds, args.n_probe)
    for label, key in CASES:
        print(f"  {label:32s} indice {chosen[key]}")

    models: List[Tuple[str, torch.nn.Module]] = []
    for spec in args.model or []:
        name, mcfg_path, ckpt = spec.split("=")
        mcfg = json.load(open(mcfg_path, encoding="utf-8"))
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        sd = normalize_state_dict(ck.get("model_state_dict", ck))
        if "bilstm" in ckpt.lower():
            from bench_baselines import BiLSTMRegressionBaseline
            m = BiLSTMRegressionBaseline(horizon=len(ds.horizon_seconds),
                                         horizon_seconds=list(ds.horizon_seconds),
                                         norm_mean=ds.norm_mean, norm_std=ds.norm_std).to(dev).eval()
        elif "lstm" in ckpt.lower():
            m = LSTMRegressionBaseline(horizon=len(ds.horizon_seconds),
                                       horizon_seconds=list(ds.horizon_seconds),
                                       norm_mean=ds.norm_mean, norm_std=ds.norm_std).to(dev).eval()
        elif "fbpp" in ckpt.lower() or "flightbert" in ckpt.lower():
            m = FlightBERT_PP_V2_Baseline(mcfg).to(dev).eval()
        else:
            m = AeroTrajUltra(mcfg, ds.norm_mean, ds.norm_std).to(dev).eval()
        miss, unexp = m.load_state_dict(sd, strict=False)
        print(f"[cases] {name:20s} {len(miss):2d} chiavi mancanti, {len(unexp):2d} inattese")
        models.append((name, m))

    panels = []
    for label, key in CASES:
        anchor = np.array([ds.anchors[chosen[key]]])
        x, y = ds.get_batch_from_anchors(anchor, dev)
        preds = []
        with torch.no_grad():
            for name, m in models:
                out = m(x)
                p = out[0] if isinstance(out, tuple) else out
                if isinstance(p, dict):
                    p = p["pred"]
                preds.append((name, p[0].float().cpu().numpy()))
        panels.append((label, x[0].cpu().numpy(), y[0].cpu().numpy(), preds))

    fig = make_figure(panels, [n for n, _ in models])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(args.out, scale=args.scale)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

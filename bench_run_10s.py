# coding=utf-8
"""
Benchmark harness for the 365-day / 10 s-grid comparison.

Every model below is trained on the SAME data (the 365-day OpenSky dataset
decimated to a 10 s step: 12 observed points = 120 s of history, horizons
10/30/60/90/180/300 s), with the SAME optimiser, the SAME schedule and the
SAME number of samples, and is evaluated on the same held-out test split
(36 days of November-December 2022).

Models
------
  ours        AeroTraj-Ultra (this project) - the hybrid physics+ML model
  fbpp_v2     FlightBERT++ V2 (AAAI 2024) - the previous version of this model
  flightbert  FlightBERT (Guo et al., IEEE T-ITS 2023) - Binary Encoding,
              ACoAtt, Transformer backbone, MBC head, Hybrid Constrained loss
  kalman      Constant-Velocity Kalman filter with a learnable noise model
  imm         Interacting Multiple Model filter (CV / CA / coordinated turn),
              learnable noise model and mode probabilities
  lstm        Deep LSTM multi-horizon regression
  bistm       BiLSTM multi-horizon regression

The two classical filters are *trained* (their noise model is learnable), so
that "same budget" is literally true for every row of the table.

Usage
-----
    python bench_run_10s.py --config config_365d_10s_bench.json
    python bench_run_10s.py --models kalman imm --epochs 10
    python bench_run_10s.py --eval-only
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench_baselines import (
    BiLSTMRegressionBaseline,
    ClassicalIMMBaseline,
    FlightBERTBaseline,
    FlightBERTPPV2Faithful,
    KalmanFilterBaseline,
    flightbert_hc_loss,
    flightbertpp_hybrid_loss,
)
from bench_models import horizon_weighted_smooth_l1
from bench_run import (
    Adapter,
    DetAdapter,
    _maybe_compile,
    _raw_module,
    ade_fde,
    evaluate_model,
    horizontal_and_3d_errors,
    train_model,
)
from data_pipeline import PyOpenSkyTrajectoryPipeline
from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import (
    AeroTrajUltra,
    FlightBERT_PP_V2_Baseline,
    HybridPhysicsConstrainedLoss,
    LSTMRegressionBaseline,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the per-model batch size: the FlightBERT transformer is the only model whose
# activations do not fit the 12 GB of the card at batch 2048
BATCH_SIZE = {
    "ours": 2048,
    "fbpp_v2": 1024,
    "fbpp_v2_faithful": 1024,
    "flightbert": 512,
    "flightbert_mse": 512,
    "flightbert_soft": 512,
    "kalman": 2048,
    "imm": 1024,
    "lstm": 2048,
    "bilstm": 2048,
}

PRETTY = {
    "ours": "AeroTraj-Ultra (ours)",
    "fbpp_v2": "FlightBERT++ V2 (AAAI 2024), no ACoAtt",
    "fbpp_v2_faithful": "FlightBERT++ V2 (AAAI 2024), faithful",
    "fbpp_v2_faithful_mse": "FlightBERT++ V2 (AAAI 2024), faithful, MSE-only loss",
    "flightbert": "FlightBERT (IEEE T-ITS 2023), BCE-dominant loss",
    "flightbert_mse": "FlightBERT (T-ITS 2023), MSE-only loss",
    "flightbert_soft": "FlightBERT (T-ITS 2023), roll-out without threshold",
    "kalman": "Kalman filter (CV, learnable noise)",
    "imm": "IMM (CV/CA/CT, learnable noise)",
    "lstm": "LSTM regression",
    "bilstm": "BiLSTM regression",
    "cv_lsq": "CV least-squares (no learning)",
}

# The per-model learning rate.  The FlightBERT uses the learning rate of its
# own paper (1e-4): with 1e-3 the logits of the binary head saturate, the
# sigmoid gradient vanishes and the high bits of the encoding stay wrong.
LR_OVERRIDE = {
    "flightbert": 1e-4,
    "flightbert_mse": 1e-4,
}

# The balance of the Hybrid Constrained loss, per model: (w_bce, w_cont).
#   flightbert            the BCE dominates: it is the term that pins the individual
#                         bits of the encoding (see the negative result in the paper)
#   flightbert_mse        the MSE alone
#   fbpp_v2_faithful      the MSE alone: the decoded TRAJECTORY is optimised
#                         directly and the Gray-Code bits are free to carry the error
LOSS_OVERRIDE = {
    "flightbert": (1.0, 0.1),
    "flightbert_mse": (0.0, 1.0),
    # the FlightBERT++ rows: the BCE-only variant is the one that works (it is
    # what the no-ACoAtt baseline was trained with, so the two are comparable),
    # the MSE-only variant is the ablation that shows what happens when the
    # regression term is left alone on a redundant binary code
    "fbpp_v2_faithful": (1.0, 0.0),
    "fbpp_v2_faithful_mse": (0.0, 1.0),
}


class FlightBERTAdapter(Adapter):
    """
    FlightBERT: trained on the next step, evaluated with the paper's roll-out.

    w_bce / w_mse select the balance of the Hybrid Constrained loss:
      (1.0, 0.1)  the BCE dominates  -> the individual bits are pinned
      (0.0, 1.0)  the MSE alone      -> the decoded VALUE is optimised directly
    """

    def __init__(self, name, module, horizon_steps, soft: bool = False,
                 w_bce: float = 1.0, w_mse: float = 0.1) -> None:
        super().__init__(name, module)
        self.horizon_steps = list(horizon_steps)
        self.soft = soft
        self.w_bce = w_bce
        self.w_mse = w_mse

    def loss(self, x, y):
        return flightbert_hc_loss(self.module, x, y[:, 0],
                                  w_bce=self.w_bce, w_mse=self.w_mse)

    @torch.no_grad()
    def predict(self, x, n_samples: int = 1):
        return self.module.rollout_at(x, self.horizon_steps, soft=self.soft).unsqueeze(1)


class FlightBERTPPAdapter(Adapter):
    """
    FlightBERT++ (AAAI 2024), faithful: the hybrid constrained loss.

    w_bce / w_cont select the balance: the BCE pins the individual Gray-Code
    bits, the continuous term optimises the decoded trajectory directly.
    """

    def __init__(self, name, module, horizon_seconds,
                 w_bce: float = 1.0, w_cont: float = 1.0) -> None:
        super().__init__(name, module)
        self.horizon_seconds = list(horizon_seconds)
        self.w_bce = w_bce
        self.w_cont = w_cont

    def loss(self, x, y):
        return flightbertpp_hybrid_loss(self.module, x, y, self.horizon_seconds,
                                        w_bce=self.w_bce, w_cont=self.w_cont)

    @torch.no_grad()
    def predict(self, x, n_samples: int = 1):
        p, _ = self.module(x)
        return p.unsqueeze(1)


# The models this harness can build.  A name that is not in this list is
# ignored, and the harness says so: a silently ignored model is the worst
# possible failure mode of a benchmark (it happened once: the shell truncated
# "bilstm" to "bistm" and the BiLSTM was simply never trained).
KNOWN_MODELS = [
    "ours", "fbpp_v2", "fbpp_v2_faithful", "fbpp_v2_faithful_mse",
    "flightbert", "flightbert_mse", "flightbert_soft",
    "kalman", "imm", "lstm", "bilstm",
]
MODEL_ALIASES = {"bistm": "bilstm", "bilstm_reg": "bilstm"}


# --------------------------------------------------------------------------- #
def build_models(cfg, ds_train, horizon_seconds, horizon_steps, args) -> Dict[str, Adapter]:
    win = cfg["window"]
    H = int(win["horizon"])
    T = int(win["inp_seq_len"])
    nm, ns = ds_train.norm_mean, ds_train.norm_std
    dt = float(cfg["data"]["dt_seconds"])
    out: Dict[str, Adapter] = {}
    cmode = args.compile

    if "ours" in args.models:
        ours = AeroTrajUltra(cfg, nm, ns).to(DEVICE)
        crit = HybridPhysicsConstrainedLoss(ours.codec, cfg).to(DEVICE)

        class _OursAdapter(Adapter):
            def loss(self, x, y):
                o = self.module(x)
                l, _ = crit(o, x, y)
                return l

        out["ours"] = _OursAdapter("ours", _maybe_compile(ours, cmode))

    if "fbpp_v2" in args.models:
        fb = FlightBERT_PP_V2_Baseline(cfg).to(DEVICE)

        class _FBAdapter(Adapter):
            def loss(self, x, y):
                _, logits = self.module(x)
                tgt = self.module.codec.encode_targets_from_anchor(x[:, -1, :6], y)
                return sum(
                    F.binary_cross_entropy_with_logits(l, t) for l, t in zip(logits, tgt)
                ) / 6.0

            @torch.no_grad()
            def predict(self, x, n_samples: int = 1):
                p, _ = self.module(x)
                return p.unsqueeze(1)

        out["fbpp_v2"] = _FBAdapter("fbpp_v2", _maybe_compile(fb, cmode))

    if "fbpp_v2_faithful" in args.models:
        fbf = FlightBERTPPV2Faithful(cfg).to(DEVICE)
        wb, wc = LOSS_OVERRIDE.get("fbpp_v2_faithful", (1.0, 0.0))
        out["fbpp_v2_faithful"] = FlightBERTPPAdapter(
            "fbpp_v2_faithful", _maybe_compile(fbf, cmode), horizon_seconds,
            w_bce=wb, w_cont=wc)

    if "fbpp_v2_faithful_mse" in args.models:
        fbfm = FlightBERTPPV2Faithful(cfg).to(DEVICE)
        wb, wc = LOSS_OVERRIDE.get("fbpp_v2_faithful_mse", (0.0, 1.0))
        out["fbpp_v2_faithful_mse"] = FlightBERTPPAdapter(
            "fbpp_v2_faithful_mse", fbfm, horizon_seconds, w_bce=wb, w_cont=wc)

    if "flightbert" in args.models:
        fb23 = FlightBERTBaseline(horizon_steps, dt_sec=dt).to(DEVICE)
        wb, wm = LOSS_OVERRIDE.get("flightbert", (1.0, 0.1))
        out["flightbert"] = FlightBERTAdapter(
            "flightbert", _maybe_compile(fb23, "" if cmode == "reduce-overhead" else cmode),
            horizon_steps, w_bce=wb, w_mse=wm,
        )

    if "flightbert_mse" in args.models:
        fb23m = FlightBERTBaseline(horizon_steps, dt_sec=dt).to(DEVICE)
        wb, wm = LOSS_OVERRIDE.get("flightbert_mse", (0.0, 1.0))
        out["flightbert_mse"] = FlightBERTAdapter(
            "flightbert_mse", fb23m, horizon_steps, w_bce=wb, w_mse=wm,
        )

    if "flightbert_soft" in args.models:
        fb23s = FlightBERTBaseline(horizon_steps, dt_sec=dt).to(DEVICE)
        out["flightbert_soft"] = FlightBERTAdapter(
            "flightbert_soft", fb23s, horizon_steps, soft=True)

    if "kalman" in args.models:
        kf = KalmanFilterBaseline(horizon_seconds, dt_sec=dt).to(DEVICE)
        out["kalman"] = DetAdapter(
            "kalman", _maybe_compile(kf, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

    if "imm" in args.models:
        imm = ClassicalIMMBaseline(horizon_seconds, dt_sec=dt).to(DEVICE)
        out["imm"] = DetAdapter(
            "imm", _maybe_compile(imm, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

    if "lstm" in args.models:
        lstm = LSTMRegressionBaseline(H, horizon_seconds, nm, ns).to(DEVICE)
        out["lstm"] = DetAdapter(
            "lstm", _maybe_compile(lstm, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

    if "bilstm" in args.models:
        bi = BiLSTMRegressionBaseline(H, horizon_seconds, nm, ns).to(DEVICE)
        out["bilstm"] = DetAdapter(
            "bilstm", _maybe_compile(bi, cmode),
            lambda p, t: horizon_weighted_smooth_l1(p, t, horizon_seconds),
        )

    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_365d_10s_bench.json")
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset of models to train (default: all)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-train-per-epoch", type=int, default=None)
    ap.add_argument("--max-test-windows", type=int, default=60000)
    ap.add_argument("--compile", type=str, default="none",
                    help="torch.compile mode (reduce-overhead | default | none). "
                         "The default is none: on this machine the inductor's C++ "
                         "path has no compiler, and a uniform treatment of every "
                         "model is also the fairest protocol.")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.compile in ("none", "off", ""):
        args.compile = ""

    torch.manual_seed(97)
    np.random.seed(97)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = json.load(open(args.config, encoding="utf-8"))
    tr = cfg["training"]
    win = cfg["window"]
    horizon_seconds: List[int] = list(win["horizon_seconds"])
    horizon_steps: List[int] = list(win["horizon_steps"])

    pipeline = PyOpenSkyTrajectoryPipeline(args.config)
    dataset_stats = pipeline.build_dataset_splits(force_rebuild=False)

    ds_train = FlightSlidingWindowDataset("train", args.config)
    ds_val = FlightSlidingWindowDataset("val", args.config, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", args.config, max_samples=int(tr["max_test_samples"]))

    save_dir = Path(tr["save_dir"])
    results_dir = Path(tr["results_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs or tr["epochs"])
    max_train = int(args.max_train_per_epoch or tr["max_train_samples_per_epoch"])

    print(f"\n[bench10s] device={DEVICE} | horizon {horizon_seconds} s "
          f"({len(horizon_seconds)} points) | dt={cfg['data']['dt_seconds']} s")
    print(f"[bench10s] train pool={len(ds_train):,} | val={len(ds_val):,} | test={len(ds_test):,}")
    print(f"[bench10s] budget: {epochs} epochs x {max_train:,} samples = "
          f"{epochs * max_train:,} samples per model")

    args.models = args.models or ["ours", "flightbert", "kalman", "imm", "lstm", "bilstm", "fbpp_v2"]
    args.models = [MODEL_ALIASES.get(m, m) for m in args.models]
    unknown = [m for m in args.models if m not in KNOWN_MODELS]
    if unknown:
        raise SystemExit(f"modelli sconosciuti: {unknown}\n"
                         f"disponibili: {KNOWN_MODELS}")
    print(f"[bench10s] models requested: {args.models}")
    adapters = build_models(cfg, ds_train, horizon_seconds, horizon_steps, args)
    print(f"[bench10s] adapters built   : {list(adapters.keys())}")
    missing = [m for m in args.models if m not in adapters]
    if missing:
        raise SystemExit(f"modelli richiesti ma non costruiti: {missing}")

    results_path = results_dir / "benchmark_results.json"
    results: Dict[str, Any] = {}
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  [merge] {len(results)} pre-existing results loaded")
    test_loader = FastSlidingWindowBatchLoader(
        ds_test, batch_size=2048, shuffle=False, device=DEVICE)

    summary: Dict[str, Any] = {
        "config": args.config,
        "horizon_seconds": horizon_seconds,
        "horizon_steps": horizon_steps,
        "budget": {"epochs": epochs, "samples_per_epoch": max_train,
                   "total_samples": epochs * max_train},
        "dataset": dataset_stats["splits"],
        "models": {},
    }

    for name, ad in adapters.items():
        bs = BATCH_SIZE.get(name, 1024)
        train_loader = FastSlidingWindowBatchLoader(
            ds_train, batch_size=bs, shuffle=True,
            max_samples_per_epoch=max_train, device=DEVICE)
        val_loader = FastSlidingWindowBatchLoader(
            ds_val, batch_size=bs, shuffle=False, device=DEVICE)

        ckpt_name = "flightbert" if name == "flightbert_soft" else name
        ckpt = save_dir / f"{ckpt_name}.pt"
        if args.eval_only or name == "flightbert_soft":
            # the soft roll-out is the SAME model evaluated differently
            if not ckpt.exists():
                print(f"  [eval-only] {name}: no checkpoint at {ckpt} -> skipped")
                continue
            state = torch.load(ckpt, map_location=DEVICE, weights_only=False)["model_state_dict"]
            _raw_module(ad.module).load_state_dict(state)
            print(f"  [eval-only] {name}: loaded {ckpt}")
            if args.eval_only:
                continue
            res = evaluate_model(ad, test_loader, horizon_seconds, n_samples=1,
                                 max_windows=args.max_test_windows)
            results[name] = res
            print(f"  [eval] {name:18s} MDE={res['overall_mde_3d_m']:9.1f} m  "
                  f"ADE={res['ade_3d_m']:9.1f}  FDE={res['fde_3d_m']:9.1f}")
            json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2)
            continue

        n_par = sum(p.numel() for p in ad.module.parameters())
        print(f"\n=== Training '{name}' ({ad.kind}, {n_par/1e6:.2f}M params, batch {bs}) ===")
        t0 = time.time()
        info = train_model(
            ad, train_loader, val_loader, epochs,
            float(LR_OVERRIDE.get(name, tr["learning_rate"])), float(tr["weight_decay"]),
            int(tr.get("early_stopping", {}).get("patience", 8)),
            float(tr.get("early_stopping", {}).get("min_delta", 0.5)),
            int(tr.get("early_stopping", {}).get("min_epochs", 10)),
            ckpt, horizon_seconds, tag=name,
        )
        info["train_seconds"] = round(time.time() - t0, 1)
        info["parameters"] = int(n_par)
        info["batch_size"] = bs
        if hasattr(ad.module, "noise_report"):
            info["learned_noise_model"] = ad.module.noise_report()
        summary["models"][name] = info
        # MERGED with the summary of the previous runs: a model trained in an
        # earlier invocation must not lose its record when another model is
        # trained now (the benchmark is run in several pieces)
        ts_path = results_dir / "training_summary.json"
        merged: Dict[str, Any] = {}
        if ts_path.exists():
            try:
                with open(ts_path, "r", encoding="utf-8") as f:
                    merged = json.load(f)
            except (json.JSONDecodeError, OSError):
                merged = {}
        merged.setdefault("models", {}).update(summary["models"])
        merged["config"] = args.config
        merged["horizon_seconds"] = horizon_seconds
        merged["budget"] = summary["budget"]
        json.dump(merged, open(ts_path, "w", encoding="utf-8"), indent=2)

        # ---- evaluate immediately: a run interrupted half way still leaves a
        # ---- complete table for the models that finished
        if not args.eval_only:
            res = evaluate_model(ad, test_loader, horizon_seconds, n_samples=1,
                                 max_windows=args.max_test_windows)
            results[name] = res
            print(f"  [eval] {name:12s} MDE={res['overall_mde_3d_m']:9.1f} m  "
                  f"ADE={res['ade_3d_m']:9.1f}  FDE={res['fde_3d_m']:9.1f}")
            json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2)

    # ---------------- final evaluation pass ---------------- #
    print("\n=== Final evaluation on the held-out test set ===")
    for name, ad in adapters.items():
        if name in results:
            continue
        res = evaluate_model(ad, test_loader, horizon_seconds, n_samples=1,
                             max_windows=args.max_test_windows)
        results[name] = res
        print(f"  {name:12s} MDE={res['overall_mde_3d_m']:9.1f} m  "
              f"ADE={res['ade_3d_m']:9.1f}  FDE={res['fde_3d_m']:9.1f}")
        # saved after every model: a run interrupted half way still leaves a
        # complete table for the models that finished
        json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2)

    # the reference without learning (least-squares extrapolation)
    from model import SmoothedConstantVelocityBaseline
    cvs = SmoothedConstantVelocityBaseline(
        horizon_seconds, dt_sec=float(cfg["data"]["dt_seconds"])).to(DEVICE)
    preds, targs = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            preds.append(cvs(x_b).cpu().numpy())
            targs.append(y_b.cpu().numpy())
    P, T = np.concatenate(preds), np.concatenate(targs)
    d3, d2 = horizontal_and_3d_errors(P, T)
    from run import compute_trajectory_metrics
    m = compute_trajectory_metrics(P, T, horizon_seconds)
    results["cv_lsq"] = {"overall_mde_3d_m": m["overall_mde_3d_m"],
                          "per_horizon": m["per_horizon"]}
    results["cv_lsq"].update(ade_fde(d3, d2))
    print(f"  {'cv_lsq':12s} MDE={results['cv_lsq']['overall_mde_3d_m']:9.1f} m")

    json.dump(results, open(results_path, "w", encoding="utf-8"), indent=2)

    ts_path = results_dir / "training_summary.json"
    ts: Dict[str, Any] = {}
    if ts_path.exists():
        with open(ts_path, "r", encoding="utf-8") as f:
            ts = json.load(f)
    ts["config"] = args.config
    ts["horizon_seconds"] = horizon_seconds
    ts["dataset"] = dataset_stats["splits"]
    ts.setdefault("models", {}).update(summary.get("models", {}))
    json.dump(ts, open(ts_path, "w", encoding="utf-8"), indent=2)

    # ---------------- tables ---------------- #
    rows = []
    for name, r in results.items():
        row = {"model": name, "label": PRETTY.get(name, name),
               "mde_mean_m": round(r["overall_mde_3d_m"], 1),
               "ade_3d_m": round(r.get("ade_3d_m", float("nan")), 1),
               "fde_3d_m": round(r.get("fde_3d_m", float("nan")), 1),
               "n_windows": r.get("n_windows", 0)}
        for h in horizon_seconds:
            row[f"mde_{int(h)}s"] = round(r["per_horizon"][f"{int(h)}s"]["mde_3d_m"], 1)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mde_mean_m")
    df.to_csv(results_dir / "benchmark_table.csv", index=False)

    lines = [
        "# Benchmark 365 giorni, griglia 10 s (12 punti = 120 s di storia)\n",
        f"Test set: {dataset_stats['splits']['test']['num_days']} giorni "
        f"({dataset_stats['splits']['test']['first_day']} -> {dataset_stats['splits']['test']['last_day']}), "
        f"{rows[0]['n_windows'] if rows else 0} finestre.\n",
        f"Budget comune: {epochs} epoche x {max_train:,} campioni/epoca "
        f"= {epochs * max_train:,} campioni per modello.\n",
        "| modello | MDE medio (m) | ADE 3D (m) | FDE 3D (m) | "
        + " | ".join(f"{h} s (m)" for h in horizon_seconds) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(horizon_seconds),
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['mde_mean_m']:.1f} | {r['ade_3d_m']:.1f} | {r['fde_3d_m']:.1f} | "
            + " | ".join(f"{r[f'mde_{int(h)}s']:.1f}" for h in horizon_seconds) + " |")
    open(results_dir / "BENCHMARK.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print(f"\n[bench10s] results -> {results_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Cross-dataset evaluation: isolate the effect of the TRAINING DATA from the effect
of the TEST SET.

The 365-day run is evaluated on its own test split (2022-11-26..12-31), which is
not the split used by the 90-day benchmark (2022-03-23..03-31).  Comparing those
two numbers directly would confound "more training data" with "different test
days", so this script evaluates each checkpoint on BOTH test splits.

The normalisation statistics stay those of the checkpoint (they are registered
buffers and are restored by load_state_dict), which is the correct protocol: the
model is fixed and only the evaluation data changes.

Run:  python eval_cross.py
Out:  results_365/cross_90d_vs_365d.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra, normalize_state_dict
from run import compute_trajectory_metrics

CFG_90 = "config_90d_300s.json"
CFG_365 = "config_365d.json"
DEFAULT_MODELS = [
    ("AeroTraj-Ultra 90d", CFG_90, "checkpoints_bench/ours.pt"),
    ("AeroTraj-Ultra 365d", CFG_365, "checkpoints_365/aerotraj_ultra_best.pt"),
]
SPLITS = [("test90 (2022-03-23..03-31)", CFG_90, "test"),
          ("test365 (2022-11-26..12-31)", CFG_365, "test")]
BS = 4096
MAXW = 60000


@torch.inference_mode()
def evaluate(model, ds, hs, dev):
    preds, targs = [], []
    for s in range(0, len(ds.anchors), BS):
        x, y = ds.get_batch_from_anchors(ds.anchors[s:s + BS], dev)
        preds.append(model(x)["pred"].float().cpu().numpy())
        targs.append(y.cpu().numpy())
    P = np.concatenate(preds); T = np.concatenate(targs)
    return compute_trajectory_metrics(P, T, hs), len(T)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None,
                    help="NAME=config.json=checkpoint.pt (repeatable); "
                         "when given it replaces the built-in pair")
    ap.add_argument("--max-samples", type=int, default=MAXW)
    ap.add_argument("--out", default="results_365/cross_90d_vs_365d.json")
    args = ap.parse_args()

    models = DEFAULT_MODELS
    if args.model:
        models = []
        for spec in args.model:
            parts = spec.split("=")
            if len(parts) != 3:
                raise SystemExit(f"--model must be NAME=config=ckpt, got {spec!r}")
            models.append(tuple(parts))
    maxw = int(args.max_samples)

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = {"models": {}, "note": f"same {maxw}-window budget per split, batch 4096"}

    for name, cfg_path, ckpt_path in models:
        cfg = json.load(open(cfg_path, encoding="utf-8"))
        hs = cfg["window"]["horizon_seconds"]
        # norm stats from the checkpoint's own training split
        tr = FlightSlidingWindowDataset("train", cfg_path, max_samples=8)
        model = AeroTrajUltra(cfg, tr.norm_mean, tr.norm_std).to(dev).eval()
        sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
        sd = normalize_state_dict(sd.get("model_state_dict", sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"{ckpt_path}: unexpected keys {list(unexpected)[:5]}")
        print(f"\n=== {name} ({ckpt_path}) — {len(missing)} chiavi mancanti, "
              f"norm dal checkpoint ===")

        for split_label, split_cfg, split in SPLITS:
            ds = FlightSlidingWindowDataset(split, split_cfg, max_samples=maxw)
            m, n = evaluate(model, ds, hs, dev)
            mean = float(np.mean([m["per_horizon"][f"{int(h)}s"]["mde_3d_m"] for h in hs]))
            out["models"].setdefault(name, {})[split_label] = {
                "mean_m": mean, "n_windows": int(n),
                "per_horizon": {f"{int(h)}s": m["per_horizon"][f"{int(h)}s"]["mde_3d_m"]
                                for h in hs}}
            keys = sorted(m["per_horizon"], key=lambda k: float(k[:-1]))
            detail = ", ".join(f"{k} {m['per_horizon'][k]['mde_3d_m']:.1f}"
                               for k in keys if float(k[:-1]) in (5.0, 60.0, 300.0))
            print(f"   {split_label:30s} mean {mean:8.1f} m  ({detail})")
            del ds
        del model
        torch.cuda.empty_cache()

    print("\n=== confronto a parita' di test set ===")
    names = list(out["models"])
    for split_label, _, _ in SPLITS:
        vals = [out["models"][n].get(split_label) for n in names]
        if not all(vals):
            continue
        base = vals[0]["mean_m"]
        parts = "  ".join(f"{n}: {v['mean_m']:.1f} m" + ("" if i == 0 else
                          f" ({100 * (v['mean_m'] / base - 1):+.2f} %)")
                          for i, (n, v) in enumerate(zip(names, vals)))
        print(f"   {split_label:30s} {parts}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[cross] saved {args.out}")


if __name__ == "__main__":
    main()

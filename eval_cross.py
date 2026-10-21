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

import json
import sys

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra, normalize_state_dict
from run import compute_trajectory_metrics

CFG_90 = "config_90d_300s.json"
CFG_365 = "config_365d.json"
MODELS = [
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
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out = {"models": {}, "note": "same 60k-window budget per split, batch 4096"}

    for name, cfg_path, ckpt_path in MODELS:
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
            ds = FlightSlidingWindowDataset(split, split_cfg, max_samples=MAXW)
            m, n = evaluate(model, ds, hs, dev)
            mean = float(np.mean([m["per_horizon"][f"{int(h)}s"]["mde_3d_m"] for h in hs]))
            out["models"].setdefault(name, {})[split_label] = {
                "mean_m": mean, "n_windows": int(n),
                "per_horizon": {f"{int(h)}s": m["per_horizon"][f"{int(h)}s"]["mde_3d_m"]
                                for h in hs}}
            print(f"   {split_label:30s} mean {mean:8.1f} m  "
                  f"(5s {m['per_horizon']['5s']['mde_3d_m']:6.1f}, "
                  f"60s {m['per_horizon']['60s']['mde_3d_m']:7.1f}, "
                  f"300s {m['per_horizon']['300s']['mde_3d_m']:8.1f})")
            del ds
        del model
        torch.cuda.empty_cache()

    print("\n=== effetto dei dati a parita' di test set ===")
    for split_label, _, _ in SPLITS:
        a = out["models"]["AeroTraj-Ultra 90d"][split_label]["mean_m"]
        b = out["models"]["AeroTraj-Ultra 365d"][split_label]["mean_m"]
        print(f"   {split_label:30s} 90d {a:8.1f}  ->  365d {b:8.1f}  "
              f"({100 * (b / a - 1):+.2f} %)")

    json.dump(out, open("results_365/cross_90d_vs_365d.json", "w"), indent=2)
    print("\n[cross] saved results_365/cross_90d_vs_365d.json")


if __name__ == "__main__":
    main()

# coding=utf-8
"""
Quality check for the pruned retrieval pool.

bench_retrieval_block.py shows that a pool with one candidate per flight is
~5.5x cheaper than the full stride-16 pool (26.9 ms vs 147.7 ms for 4096
queries).  The pool is an INPUT to the model, so the speed-up is only usable if
the prediction quality does not drop: this script evaluates the SAME retrieval
checkpoint on the SAME test windows with both pools.

Run:  python check_pool_quality.py
Out:  results_bench/pool_quality.json
"""
from __future__ import annotations

import json

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltraRetrieval
from retrieval import RetrievalPool, anchor_flight_ids, build_pool, window_descriptor
from run import compute_trajectory_metrics

CONFIG = "config_90d_300s_hyp.json"
CKPT = "checkpoints_hyp_reg/retrieval/aerotraj_ultra_retrieval_best.pt"
POS_WEIGHT = 10.0
POOL_STRIDE = 16
N_TEST = 12288
BS = 512


@torch.inference_mode()
def evaluate(model, ds, pool, hs, dev, block=16):
    preds, targs = [], []
    for s in range(0, len(ds.anchors), BS):
        x, y = ds.get_batch_from_anchors(ds.anchors[s:s + BS], dev)
        ctx = pool.context(window_descriptor(x, POS_WEIGHT), x, k=8)
        preds.append(model(x, retr_ctx=ctx)["pred"].float().cpu().numpy())
        targs.append(y.cpu().numpy())
    P = np.concatenate(preds); T = np.concatenate(targs)
    return compute_trajectory_metrics(P, T, hs)


def main() -> None:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]

    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=N_TEST)

    model = AeroTrajUltraRetrieval(cfg, ds.norm_mean, ds.norm_std, n_neighbours=8).to(dev).eval()
    sd = torch.load(CKPT, map_location=dev, weights_only=False)["model_state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[pool] checkpoint: {len(miss)} chiavi mancanti, {len(unexp)} inattese")
    if unexp:
        print("   inattese:", list(unexp)[:5])

    print("[pool] costruzione del pool pieno ...")
    full = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    flights = anchor_flight_ids(ds_tr)[::POOL_STRIDE]
    keep = np.ones(len(flights), dtype=bool)
    seen = set()
    for i, f in enumerate(flights):
        if int(f) in seen:
            keep[i] = False
        else:
            seen.add(int(f))
    sel = np.where(keep)[0]
    sd_idx = torch.as_tensor(sel, device=dev)
    small = RetrievalPool(full.cand_desc[sd_idx].cpu().numpy(),
                          full.cand_val[sd_idx].cpu().numpy(),
                          flights[keep], dev, pos_weight=POS_WEIGHT)
    print(f"[pool] pieno {full.M:,} candidati | potato {small.M:,} "
          f"({small.M / full.M:.2%} del pool)")

    res = {}
    for name, pool in [("full", full), ("pruned", small)]:
        m = evaluate(model, ds, pool, hs, dev)
        res[name] = {"mde": m["overall_mde_3d_m"], "per_horizon": m["per_horizon"]}
        print(f"\n{name:7s} MDE = {m['overall_mde_3d_m']:.2f} m   "
              f"(60 s {m['per_horizon']['60s']['mde_3d_m']:.1f}, "
              f"120 s {m['per_horizon']['120s']['mde_3d_m']:.1f}, "
              f"300 s {m['per_horizon']['300s']['mde_3d_m']:.1f})")

    d = res["pruned"]["mde"] - res["full"]["mde"]
    print(f"\ndelta MDE (potato - pieno) = {d:+.2f} m ({100 * d / res['full']['mde']:+.3f} %)")
    print("per orizzonte:")
    for h in hs:
        k = f"{int(h)}s"
        a = res["full"]["per_horizon"][k]["mde_3d_m"]
        b = res["pruned"]["per_horizon"][k]["mde_3d_m"]
        print(f"   {int(h):4d} s  pieno {a:9.2f}  potato {b:9.2f}  d {b - a:+8.3f}")

    json.dump({"full_mde": res["full"]["mde"], "pruned_mde": res["pruned"]["mde"],
               "delta_mde": d, "full_candidates": int(full.M), "pruned_candidates": int(small.M),
               "n_windows": int(len(ds.anchors)), "per_horizon": {
                   f"{int(h)}s": {"full": res["full"]["per_horizon"][f"{int(h)}s"]["mde_3d_m"],
                                  "pruned": res["pruned"]["per_horizon"][f"{int(h)}s"]["mde_3d_m"]}
                   for h in hs}},
              open("results_bench/pool_quality.json", "w"), indent=2)
    print("\n[pool] saved results_bench/pool_quality.json")


if __name__ == "__main__":
    main()

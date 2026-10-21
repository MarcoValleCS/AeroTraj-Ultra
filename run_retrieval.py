# coding=utf-8
"""
Train and evaluate the retrieval-conditioned variant of AeroTraj-Ultra.

`AeroTrajUltraRetrieval` is a copy of the base model whose horizon queries are
additionally conditioned on a retrieval context: the K past trajectories most
similar to the current observation (same neighbourhood, heading, flight level
and speed), with their futures expressed in the query's local frame.

At TRAINING time the retrieval excludes the candidates of the same flight as
the query (otherwise the model would learn to copy the future of a
near-duplicate window of the same trajectory - impossible at inference, where
the pool only contains past data).

Usage:
    python run_retrieval.py --config config_90d_300s_hyp.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from dataloader import FastSlidingWindowBatchLoader, FlightSlidingWindowDataset
from model import AeroTrajUltraRetrieval, HybridPhysicsConstrainedLoss
from retrieval import anchor_flight_ids, build_pool, window_descriptor
from run import compute_trajectory_metrics


class AnchoredLoader:
    """Like FastSlidingWindowBatchLoader but also yields the anchor indices."""

    def __init__(self, dataset, batch_size: int, shuffle: bool,
                 max_samples_per_epoch: Optional[int], device, seed: int = 97):
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

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, np.ndarray]]:
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
        for s in range(0, len(epoch_anchors), self.batch_size):
            chunk = epoch_anchors[s: s + self.batch_size]
            x, y = self.dataset.get_batch_from_anchors(chunk, self.device)
            yield x, y, chunk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_90d_300s_hyp.json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool-stride", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                    help="checkpoint of a base AeroTraj-Ultra used as warm start")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                    help="down-weight the absolute position in the retrieval key")
    args = ap.parse_args()

    torch.manual_seed(97)
    np.random.seed(97)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.load(open(args.config, encoding="utf-8"))
    tr = cfg["training"]
    hs = cfg["window"]["horizon_seconds"]
    lr = float(args.lr if args.lr is not None else tr["learning_rate"])

    ds_tr = FlightSlidingWindowDataset("train", args.config)
    ds_val = FlightSlidingWindowDataset("val", args.config, max_samples=int(tr["max_val_samples"]))
    ds_test = FlightSlidingWindowDataset("test", args.config, max_samples=int(tr["max_test_samples"]))
    bs = int(tr["batch_size"])

    print(f"[ret] 90 giorni | orizzonte {hs[0]}-{hs[-1]} s | K={args.k} | pool stride {args.pool_stride}")

    print("[ret] costruzione del pool di retrieval (solo TRAIN) ...")
    pool = build_pool(ds_tr, stride=args.pool_stride, device=dev,
                      pos_weight=args.pos_weight)
    tr_flight_np = anchor_flight_ids(ds_tr)
    tr_flight = torch.as_tensor(tr_flight_np, dtype=torch.long, device=dev)
    # map an anchor index -> its position in the (strided) pool, for the masking
    anchor_pos = np.searchsorted(ds_tr.anchors, ds_tr.anchors)
    anchor_flight_of_anchor = tr_flight_np[anchor_pos]

    model = AeroTrajUltraRetrieval(cfg, ds_tr.norm_mean, ds_tr.norm_std,
                                   n_neighbours=args.k).to(dev)
    crit = HybridPhysicsConstrainedLoss(model.codec, cfg).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[ret] AeroTrajUltraRetrieval: {n_par:,} parametri")

    if args.init_from:
        st = torch.load(args.init_from, map_location=dev, weights_only=False)
        sd = st.get("model_state_dict", st)
        miss, unexp = model.load_base_weights(sd)
        print(f"[ret] warm start da {args.init_from} "
              f"(mancanti: {len(miss)}, inattesi: {len(unexp)})")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tr["weight_decay"]))
    steps_per_epoch = int(np.ceil(int(tr["max_train_samples_per_epoch"]) / bs))
    if args.init_from:
        # fine tuning from a trained model: a gentle cosine, no warm-up restart
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs * steps_per_epoch, eta_min=lr * 0.05)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
            pct_start=0.2, anneal_strategy="cos")

    train_loader = AnchoredLoader(ds_tr, bs, True, int(tr["max_train_samples_per_epoch"]), dev)
    val_loader = AnchoredLoader(ds_val, bs, False, None, dev)
    test_loader = AnchoredLoader(ds_test, bs, False, None, dev)

    save_dir = Path(tr["save_dir"]) / "retrieval"
    results_dir = Path(tr["results_dir"]) / "retrieval"
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "aerotraj_ultra_retrieval_best.pt"

    def make_ctx(x_b, anchors, exclude_same_flight: bool):
        desc = window_descriptor(x_b, args.pos_weight)
        qf = None
        if exclude_same_flight:
            pos = np.searchsorted(ds_tr.anchors, anchors)
            pos = np.clip(pos, 0, len(anchor_flight_of_anchor) - 1)
            qf = torch.as_tensor(anchor_flight_of_anchor[pos], dtype=torch.long, device=dev)
        return pool.context(desc, x_b, k=args.k, q_flight=qf)

    # ---------------- training ---------------- #
    best = float("inf")
    best_ep = 0
    hist: List[Dict[str, Any]] = []
    stale = 0
    es = tr.get("early_stopping", {})
    patience = int(es.get("patience", 10))
    min_delta = float(es.get("min_delta", 0.5))
    min_epochs = int(es.get("min_epochs", 12))

    n_skipped = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for x_b, y_b, anchors in train_loader:
            ctx = make_ctx(x_b, anchors, exclude_same_flight=True)
            opt.zero_grad(set_to_none=True)
            loss, _ = crit(model(x_b, retr_ctx=ctx), x_b, y_b)
            if not torch.isfinite(loss):
                n_skipped += 1
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            opt.step()
            sched.step()
            losses.append(loss.detach())
        if n_skipped:
            print(f"  [warn] {n_skipped} batch non finiti saltati")
        vm = evaluate(model, val_loader, make_ctx, hs, dev, exclude_same_flight=False)["overall_mde_3d_m"]
        hist.append({"epoch": ep, "train_loss": float(torch.stack(losses).mean().item()),
                     "val_mde_m": vm, "time_s": round(time.time() - t0, 1)})
        print(f"  ep {ep:02d}/{args.epochs} ({time.time()-t0:5.1f}s) "
              f"loss={float(torch.stack(losses).mean().item()):8.4f} val MDE={vm:8.1f} m", flush=True)
        if vm < best - min_delta:
            best, best_ep, stale = vm, ep, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": ep,
                        "val_mde_m": vm, "config": cfg, "k": args.k}, ckpt_path)
        else:
            stale += 1
            if patience and ep >= min_epochs and stale >= patience:
                print(f"  early stop @ ep {ep} (best {best:.1f} m @ {best_ep})")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False)["model_state_dict"])
    print(f"[ret] best epoch {best_ep} | val MDE {best:.1f} m")

    # ---------------- test ---------------- #
    test = evaluate(model, test_loader, make_ctx, hs, dev, exclude_same_flight=False)
    print(f"\n[ret] TEST MDE = {test['overall_mde_3d_m']:.1f} m")
    for h in hs:
        print(f"   {int(h):>4d} s : {test['per_horizon'][f'{int(h)}s']['mde_3d_m']:9,.1f} m")

    out = {"config": args.config, "k": args.k, "pool_stride": args.pool_stride,
           "pos_weight": args.pos_weight, "init_from": args.init_from,
           "parameters": n_par, "history": hist, "best_epoch": best_ep,
           "best_val_mde_m": best, "test": test}
    json.dump(out, open(results_dir / "retrieval_results.json", "w"), indent=2)
    print(f"[ret] salvato in {results_dir}")


@torch.no_grad()
def evaluate(model, loader, make_ctx, hs, dev, exclude_same_flight: bool) -> Dict[str, Any]:
    model.eval()
    preds, targs = [], []
    for x_b, y_b, anchors in loader:
        ctx = make_ctx(x_b, anchors, exclude_same_flight)
        out = model(x_b, retr_ctx=ctx)
        preds.append(out["pred"].cpu().numpy())
        targs.append(y_b.cpu().numpy())
    P = np.concatenate(preds, axis=0)
    T = np.concatenate(targs, axis=0)
    m = compute_trajectory_metrics(P, T, hs)
    return {"overall_mde_3d_m": m["overall_mde_3d_m"], "per_horizon": m["per_horizon"]}


if __name__ == "__main__":
    main()

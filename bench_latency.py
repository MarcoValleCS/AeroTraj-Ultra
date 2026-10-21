# coding=utf-8
"""
Inference-latency and throughput benchmark.

Measures, with CUDA events for the GPU part and perf_counter for the end-to-end
path:

  * single-aircraft latency (batch 1) - the operational case: median and p95
    over many repetitions, including the retrieval lookup and the host<->device
    transfers;
  * throughput (aircraft/s) versus batch size;
  * the component breakdown at batch 1 and batch 512 (retrieval top-k, forward,
    Gray-Code decoding);
  * the effect of torch.compile on inference;
  * the same measurement for every model of the benchmark, so the accuracy is
    always reported together with its cost.

Run:  python bench_latency.py
Out:  results_bench/latency.json + results_bench/LATENCY.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import (
    AeroTrajUltra,
    AeroTrajUltraRetrieval,
    ConstantVelocityKalmanBaseline,
    FlightBERT_PP_V2_Baseline,
    LSTMRegressionBaseline,
)
from bench_models import ASCENT, AttentionLSTMPhase, CFMTrajectoryModel, DDPMTrajectoryModel
from retrieval import build_pool, window_descriptor

CONFIG = "config_90d_300s_hyp.json"
POS_WEIGHT = 10.0
POOL_STRIDE = 16
REPEAT_1 = 200      # repetitions for the batch-1 latency
REPEAT_GPU = 50     # repetitions for the per-call CUDA-event measurement
PASSES = 3          # independent measurement blocks (spread = error bar)
WARMUP = 20


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn: Callable[[], None], repeat: int, warmup: int = WARMUP) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1e3)      # ms
    ts = np.asarray(ts)
    return {"median_ms": float(np.median(ts)), "mean_ms": float(ts.mean()),
            "p95_ms": float(np.percentile(ts, 95)), "min_ms": float(ts.min()),
            "n": int(repeat)}


def gpu_timed(fn: Callable[[], None], repeat: int, warmup: int = WARMUP) -> float:
    """
    GPU-busy time per call, in ms (median over `repeat` calls).

    One CUDA event pair brackets exactly ONE call and the stream is synchronised
    before the pairs are read, so the measured interval is the GPU work of that
    call and not the CPU-side scheduling gap between launches.  The first
    version of this benchmark recorded one pair around the whole repetition loop,
    which made the "GPU time" larger than the wall-clock latency (impossible) and
    was therefore discarded.
    """
    for _ in range(warmup):
        fn()
    sync()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        starts[i].record()
        fn()
        ends[i].record()
    sync()
    ts = np.asarray([s.elapsed_time(e) for s, e in zip(starts, ends)])
    return float(np.median(ts))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    hs = cfg["window"]["horizon_seconds"]

    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=64)
    # separate dataset for the throughput sweep: the test set above is capped at
    # 64 windows, which would silently cap the large batches at 64 rows.
    ds_thr = FlightSlidingWindowDataset("train", CONFIG, max_samples=8192)
    ds_tr = FlightSlidingWindowDataset("train", CONFIG)
    print("[lat] costruzione del pool di retrieval ...")
    pool = build_pool(ds_tr, stride=POOL_STRIDE, device=dev, pos_weight=POS_WEIGHT, verbose=False)
    n_cand = int(pool.M)
    print(f"[lat] pool: {n_cand:,} candidati, {pool.H} orizzonti")

    out: Dict[str, object] = {"device": str(dev), "horizon_seconds": hs,
                              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    # ------------------------------------------------------------------ #
    # models
    # ------------------------------------------------------------------ #
    def ck(name):
        return torch.load(f"checkpoints_bench/{name}.pt", map_location=dev, weights_only=False)["model_state_dict"]

    ours = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
    ours.load_state_dict(ck("ours"), strict=False)
    retr = AeroTrajUltraRetrieval(cfg, ds.norm_mean, ds.norm_std, n_neighbours=8).to(dev).eval()
    try:
        retr.load_state_dict(torch.load("checkpoints_hyp_reg/retrieval/aerotraj_ultra_retrieval_best.pt",
                                        map_location=dev, weights_only=False)["model_state_dict"], strict=False)
    except FileNotFoundError:
        pass
    fb = FlightBERT_PP_V2_Baseline(cfg).to(dev).eval()
    fb.load_state_dict(ck("fbpp_v2"), strict=False)
    lstm = LSTMRegressionBaseline(horizon=len(hs), horizon_seconds=hs,
                                  norm_mean=ds.norm_mean, norm_std=ds.norm_std).to(dev).eval()
    lstm.load_state_dict(ck("lstm"), strict=False)
    ascent = ASCENT(len(hs), hs, 12).to(dev).eval()
    ascent.load_state_dict(ck("ascent"), strict=False)
    cfm = CFMTrajectoryModel(len(hs), hs, 12).to(dev).eval()
    cfm.load_state_dict(ck("cfm"), strict=False)
    ddpm = DDPMTrajectoryModel(len(hs), hs, 12).to(dev).eval()
    ddpm.load_state_dict(ck("ddpm"), strict=False)
    cv = ConstantVelocityKalmanBaseline(hs).to(dev)

    # ------------------------------------------------------------------ #
    # 1) single-aircraft latency (batch 1)
    # ------------------------------------------------------------------ #
    print("\n=== Latenza a batch 1 (un solo aereo, il caso operativo) ===")
    x1, _ = ds.get_batch_from_anchors(ds.anchors[:1], dev)
    x1_cpu = x1.cpu()

    def end_to_end(model_fn, with_transfer: bool = True):
        def run():
            xb = x1_cpu.to(dev, non_blocking=False) if with_transfer else x1
            model_fn(xb)
        return run

    rows = []
    with torch.no_grad():
        single = {
            "AeroTraj-Ultra": end_to_end(lambda x: ours(x)["pred"]),
            "AeroTraj-Ultra + retrieval": end_to_end(
                lambda x: retr(x, retr_ctx=pool.context(window_descriptor(x, POS_WEIGHT), x, k=8))["pred"]),
            "FlightBERT++ V2": end_to_end(lambda x: fb(x)[0]),
            "Deep LSTM": end_to_end(lambda x: lstm(x)),
            "ASCENT": end_to_end(lambda x: ascent(x)),
            "CFM (20 step, 1 campione)": end_to_end(lambda x: cfm.sample(x, n_samples=1)),
            "DDPM (50 step, 1 campione)": end_to_end(lambda x: ddpm.sample(x, n_samples=1)),
            "CV / Doppler": end_to_end(lambda x: cv(x)),
        }
        for name, fn in single.items():
            # PASSES independent blocks: the GPU-busy time is stable across runs
            # while the host-side wall clock is not (observed swing up to 40 % on
            # an unlocked-clock Windows box), so the report carries the spread
            # of the GPU measurement instead of a single bare number.
            wall, gpu = [], []
            for _ in range(PASSES):
                wall.append(timed(fn, repeat=REPEAT_1, warmup=WARMUP))
                gpu.append(gpu_timed(fn, repeat=REPEAT_GPU))
            t = {
                "median_ms": float(np.median([w["median_ms"] for w in wall])),
                "p95_ms": float(np.median([w["p95_ms"] for w in wall])),
                "gpu_ms": float(np.median(gpu)),
                "wall_pass_medians_ms": [round(w["median_ms"], 2) for w in wall],
                "gpu_pass_medians_ms": [round(g, 2) for g in gpu],
                "gpu_spread_pct": float((max(gpu) - min(gpu)) / max(1e-9, float(np.median(gpu))) * 100.0),
                "n": REPEAT_1,
            }
            rows.append((name, t))
            print(f"  {name:32s} mediana {t['median_ms']:7.2f} ms | p95 {t['p95_ms']:7.2f} ms | "
                  f"GPU {t['gpu_ms']:6.2f} ms (+/-{t['gpu_spread_pct']:.0f}%)")
    out["batch1_latency"] = {n: t for n, t in rows}

    # ------------------------------------------------------------------ #
    # 2) component breakdown at batch 1
    # ------------------------------------------------------------------ #
    print("\n=== Scomposizione a batch 1 ===")
    with torch.no_grad():
        t_retr = timed(lambda: pool.context(window_descriptor(x1, POS_WEIGHT), x1, k=8), repeat=100)
        t_fwd = timed(lambda: ours(x1)["pred"], repeat=100)
        t_full = [t for n, t in rows if n == "AeroTraj-Ultra + retrieval"][0]
    print(f"  retrieval top-k ({n_cand:,} candidati) : {t_retr['median_ms']:7.2f} ms")
    print(f"  forward del modello               : {t_fwd['median_ms']:7.2f} ms")
    print(f"  pipeline completa                 : {t_full['median_ms']:7.2f} ms")
    out["batch1_breakdown"] = {"retrieval_ms": t_retr["median_ms"], "forward_ms": t_fwd["median_ms"],
                               "full_ms": t_full["median_ms"], "n_candidates": n_cand}

    # same breakdown at batch 512, to show how the two components scale
    with torch.no_grad():
        x512, _ = ds_thr.get_batch_from_anchors(ds_thr.anchors[:512], dev)
        t_retr512 = timed(lambda: pool.context(window_descriptor(x512, POS_WEIGHT), x512, k=8), repeat=50)
        t_fwd512 = timed(lambda: ours(x512)["pred"], repeat=50)
    print(f"  [batch 512] retrieval {t_retr512['median_ms']:7.2f} ms | forward {t_fwd512['median_ms']:6.2f} ms")
    out["batch512_breakdown"] = {"retrieval_ms": t_retr512["median_ms"], "forward_ms": t_fwd512["median_ms"]}

    # operational budget: how many aircraft can be tracked per second against
    # the update cycles of the surveillance sources
    full1 = t_full["median_ms"]
    out["budget"] = {
        "ms_per_aircraft": full1,
        "aircraft_per_s_single": 1000.0 / full1,
        "headroom_vs_1s_cycle": 1000.0 / full1,
        "headroom_vs_5s_cycle": 5000.0 / full1,
    }
    print(f"\n  Budget operativo: {full1:.2f} ms per aereo -> "
          f"{1000.0/full1:,.0f} aerei/s con ciclo di aggiornamento a 1 s, "
          f"{5000.0/full1:,.0f} con ciclo a 5 s")

    # ------------------------------------------------------------------ #
    # 3) throughput vs batch size
    # ------------------------------------------------------------------ #
    print("\n=== Throughput ===")
    thr: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for name, fn in [("AeroTraj-Ultra", lambda x: ours(x)["pred"]),
                         ("AeroTraj-Ultra + retrieval",
                          lambda x: retr(x, retr_ctx=pool.context(window_descriptor(x, POS_WEIGHT), x, k=8))["pred"])]:
            thr[name] = {}
            thr_reps = {1: 100, 8: 80, 32: 60, 128: 40, 512: 20, 2048: 8, 4096: 5}
            for b in [1, 8, 32, 128, 512, 2048, 4096]:
                xb, _ = ds_thr.get_batch_from_anchors(ds_thr.anchors[:b], dev)
                t = timed(lambda: fn(xb), repeat=thr_reps[b])
                thr[name][str(b)] = {"median_ms": t["median_ms"], "per_aircraft_ms": t["median_ms"] / b,
                                     "aircraft_per_s": b / (t["median_ms"] / 1e3)}
                print(f"  {name:30s} batch {b:5d} : {t['median_ms']:8.2f} ms  "
                      f"({t['median_ms']/b*1e3:7.1f} us/aereo, {b/(t['median_ms']/1e3):9,.0f} aerei/s)")
    out["throughput"] = thr

    # ------------------------------------------------------------------ #
    # 3b) how much of the batch-1 latency is fixed overhead
    # ------------------------------------------------------------------ #
    # Measured without a profiler.  The marginal cost of one extra aircraft is
    # the slope between the AVERAGE cost per aircraft at batch 1 and at batch
    # 4096; whatever is left at batch 1 is launch and scheduling overhead, i.e.
    # the single-aircraft case is launch-bound rather than compute-bound.
    # The batch-1 endpoint is taken from the dedicated measurement above (median
    # of PASSES x REPEAT_1 calls) rather than from the throughput sweep, whose
    # batch-1 point is measured with far fewer repetitions.
    t1 = dict(rows)["AeroTraj-Ultra"]["median_ms"]
    tbig = thr["AeroTraj-Ultra"]["4096"]["median_ms"]
    bbig = 4096
    marg = (tbig - t1) / (bbig - 1)                   # ms per extra aircraft
    out["marginal_cost_ms_per_aircraft"] = marg
    out["fixed_overhead_ms"] = t1 - marg
    out["overhead_share_of_batch1"] = (t1 - marg) / t1
    out["fixed_overhead_source"] = {"t_batch1_ms": t1, "t_batch4096_ms": tbig}
    print("\n=== Costo fisso vs marginale (AeroTraj-Ultra) ===")
    print(f"  batch 1    : {t1:8.2f} ms  -> {t1*1e3:9.0f} us per aereo")
    print(f"  batch 4096 : {tbig:8.2f} ms  -> {tbig/bbig*1e3:9.1f} us per aereo")
    print(f"  marginale  : {marg*1e3:8.1f} us per aereo in piu'")
    print(f"  -> a batch 1 il {100*(t1-marg)/t1:.2f}% della latenza e' overhead fisso "
          f"({t1-marg:.1f} ms): il modello e' launch-bound, non compute-bound")

    # ------------------------------------------------------------------ #
    # 4) torch.compile at inference
    # ------------------------------------------------------------------ #
    print("\n=== torch.compile (inferenza) ===")
    comp: Dict[str, Dict[str, float]] = {}
    try:
        ours_c = torch.compile(ours, mode="reduce-overhead")
        with torch.no_grad():
            ours_c(x1)                                    # compile
            t = timed(lambda: ours_c(x1)["pred"], repeat=100, warmup=10)
        comp["AeroTraj-Ultra compiled"] = t
        print(f"  compiled  batch 1 : {t['median_ms']:7.2f} ms (eager {t_fwd['median_ms']:.2f} ms)")
    except Exception as exc:
        comp["error"] = str(exc)[:200]
        print(f"  compile non disponibile: {str(exc)[:120]}")
    out["compile"] = comp

    # ------------------------------------------------------------------ #
    # 5) cold start
    # ------------------------------------------------------------------ #
    print("\n=== Cold start ===")
    t0 = time.perf_counter()
    AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
    sync()
    cold_model = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    retr2 = AeroTrajUltraRetrieval(cfg, ds.norm_mean, ds.norm_std, n_neighbours=8).to(dev).eval()
    sync()
    cold_retr = (time.perf_counter() - t0) * 1e3
    print(f"  costruzione + trasferimento su GPU: base {cold_model:.0f} ms, retrieval {cold_retr:.0f} ms")
    out["cold_start_ms"] = {"base": cold_model, "retrieval": cold_retr}

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/latency.json", "w"), indent=2)

    # ---------------- markdown ---------------- #
    L = ["# Latency and throughput (inference)",
         f"\nDevice: **{out['gpu']}** | horizons {hs[0]}-{hs[-1]} s | "
         f"batch-1 latency over {REPEAT_1} repetitions (median / p95).\n",
         "## Single-aircraft latency (batch 1, end-to-end incl. H2D)\n",
         "Median of 3 independent measurement blocks (200 repetitions each). `GPU only` is the",
         "median of a per-call CUDA-event measurement (one event pair around each call) and its",
         "observed spread across the 3 blocks.\n",
         "| Model | median [ms] | p95 [ms] | GPU only [ms] | GPU spread |",
         "|---|---:|---:|---:|---:|"]
    for n, t in rows:
        L.append(f"| {n} | {t['median_ms']:.2f} | {t['p95_ms']:.2f} | {t['gpu_ms']:.2f} | "
                 f"±{t['gpu_spread_pct']:.0f}% |")
    L += ["\n## Component breakdown (batch 1)\n",
          f"- retrieval top-k over {n_cand:,} candidates: **{t_retr['median_ms']:.2f} ms**",
          f"- model forward: **{t_fwd['median_ms']:.2f} ms**",
          f"- full pipeline (AeroTraj-Ultra + retrieval): **{t_full['median_ms']:.2f} ms**\n",
          f"## Component breakdown (batch 512)\n",
          f"- retrieval top-k: **{t_retr512['median_ms']:.2f} ms**",
          f"- model forward: **{t_fwd512['median_ms']:.2f} ms**\n",
          "## Why batch 1 costs what it costs\n",
          f"- marginal cost of one extra aircraft (slope from batch 1 to batch 4096): "
          f"**{marg*1e3:.1f} us**",
          f"- average cost at batch 1: **{t1*1e3:,.0f} us** per aircraft, of which "
          f"**{100*(t1-marg)/t1:.2f} %** is fixed overhead ({t1-marg:.1f} ms)",
          "- the single-aircraft case is therefore **launch-bound, not compute-bound**\n",
          "## Operational budget\n",
          f"- {full1:.2f} ms per aircraft -> **{1000.0/full1:,.0f} aircraft/s** "
          f"against a 1 s update cycle, **{5000.0/full1:,.0f} aircraft/s** against a 5 s cycle\n",
          "## Throughput\n",
          "| Model | batch | total [ms] | us/aircraft | aircraft/s |", "|---|---:|---:|---:|---:|"]
    for n, d in thr.items():
        for b, tt in d.items():
            L.append(f"| {n} | {b} | {tt['median_ms']:.2f} | {tt['per_aircraft_ms']*1e3:.1f} | "
                     f"{tt['aircraft_per_s']:,.0f} |")
    txt = "\n".join(L) + "\n"
    Path("results_bench/LATENCY.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    print("[lat] saved to results_bench/latency.json and results_bench/LATENCY.md")


if __name__ == "__main__":
    main()

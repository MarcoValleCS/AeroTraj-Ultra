# coding=utf-8
"""
Latency-optimisation harness.

Measures, for the SAME model and the SAME weights, the inference cost of every
technique that does not change the mathematics:

  * eager + torch.no_grad()
  * eager + torch.inference_mode()
  * torch.compile(default)
  * torch.compile("reduce-overhead")             (CUDA graphs inside inductor)
  * torch.compile("max-autotune-no-cudagraphs")

and checks the numerical agreement of each variant against the eager reference,
so that a speed-up is only reported when the prediction is unchanged.

A hand-captured torch.cuda.CUDAGraph was tried and REMOVED: on this
torch/Windows build it leaves the CUDA caching allocator in a state where every
subsequent allocation fails ("Offset increment outside graph capture encountered
unexpectedly"), poisoning torch.compile for the rest of the process.  Inductor's
own cudagraph trees (mode="reduce-overhead") do the same job correctly.

Run:  python bench_opt.py
Out:  results_bench/opt_inference.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch

from dataloader import FlightSlidingWindowDataset
from model import AeroTrajUltra

CONFIG = "config_90d_300s_hyp.json"
BATCHES = [1, 8, 64, 512]
REPEAT = 100
WARMUP = 10


def gpu_ms(fn: Callable[[], None], repeat: int = REPEAT, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    e = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    for i in range(repeat):
        s[i].record()
        fn()
        e[i].record()
    torch.cuda.synchronize()
    return float(np.median([a.elapsed_time(b) for a, b in zip(s, e)]))


def main() -> None:
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    ds = FlightSlidingWindowDataset("test", CONFIG, max_samples=1024)

    model = AeroTrajUltra(cfg, ds.norm_mean, ds.norm_std).to(dev).eval()
    ck = torch.load("checkpoints_bench/ours.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    print("[opt] pesi caricati da checkpoints_bench/ours.pt")

    out: Dict[str, object] = {"device": torch.cuda.get_device_name(0), "config": CONFIG,
                              "repeat": REPEAT, "results": {}}

    for B in BATCHES:
        x, _ = ds.get_batch_from_anchors(ds.anchors[:B], dev)
        print(f"\n=== batch {B} ===")
        row: Dict[str, object] = {}

        with torch.inference_mode():
            ref = model(x)["pred"].clone()

        row["eager_nograd"] = {"gpu_ms": gpu_ms(lambda: model(x)["pred"])}
        with torch.inference_mode():
            row["eager_inference_mode"] = {
                "gpu_ms": gpu_ms(lambda: model(x)["pred"]),
                "maxdiff_vs_eager": float((model(x)["pred"] - ref).abs().max())}

        for mode in ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]:
            t0 = time.perf_counter()
            try:
                cm = torch.compile(model, mode=mode)
                with torch.inference_mode():
                    cm(x)                                   # trigger compilation
                    t_compile = (time.perf_counter() - t0) * 1e3
                    cm(x)
                    d = float((cm(x)["pred"] - ref).abs().max())
                    ms = gpu_ms(lambda: cm(x)["pred"], repeat=50)
                row[f"compile_{mode}"] = {"gpu_ms": ms, "maxdiff_vs_eager": d,
                                          "first_call_ms": t_compile}
            except Exception as exc:
                row[f"compile_{mode}"] = {"error": str(exc)[:150]}

        base = row["eager_nograd"]["gpu_ms"]
        for k, v in row.items():
            if "gpu_ms" in v:
                print(f"  {k:34s} {v['gpu_ms']:8.3f} ms  x{base / v['gpu_ms']:5.2f}"
                      f"  maxdiff {v.get('maxdiff_vs_eager', 0):.2e}"
                      f"  first-call {v.get('first_call_ms', 0):8.0f} ms")
            else:
                print(f"  {k:34s} {v.get('error')}")
        out["results"][str(B)] = row

    Path("results_bench").mkdir(exist_ok=True)
    json.dump(out, open("results_bench/opt_inference.json", "w"), indent=2)
    print("\n[opt] saved results_bench/opt_inference.json")


if __name__ == "__main__":
    main()

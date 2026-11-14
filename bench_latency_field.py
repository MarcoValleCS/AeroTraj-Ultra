# coding=utf-8
"""
Inference cost of a model variant on the project's own inference path
(`FastPredictor`): single-aircraft latency and batch-4096 throughput, eager and
compiled.  Used to price the airspace field against the base model.

Run:  python bench_latency_field.py NAME=config.json=checkpoint.pt [...] [--out file.json]
The GPU must be idle: a training run in the background invalidates the numbers.
"""
from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np
import torch

from inference import FastPredictor


def median_ms(fp: FastPredictor, x: torch.Tensor, reps: int, blocks: int = 3) -> float:
    meds = []
    for _ in range(blocks):
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fp(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        meds.append(statistics.median(ts))
    return statistics.median(meds)


def main() -> None:
    args = sys.argv[1:]
    out_path = args[args.index("--out") + 1] if "--out" in args else None
    specs = [a for a in args if "=" in a]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    out = {"device": torch.cuda.get_device_name(0)}
    for spec in specs:
        name, cfg_path, ckpt_path = spec.split("=")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        nm, ns = np.array(ck["norm_mean"], np.float32), np.array(ck["norm_std"], np.float32)
        T = int(json.load(open(cfg_path, encoding="utf-8"))["window"]["inp_seq_len"])
        out[name] = {"inp_seq_len": T}
        # a real window: the field reads positions, random ones would all fall on the border
        x1 = torch.tensor(nm, dtype=torch.float32).view(1, 1, 9).repeat(1, T, 1).cuda()
        for mode in ("none", "reduce-overhead"):
            fp = FastPredictor(cfg_path, ckpt_path, compile_mode=mode, norm_mean=nm, norm_std=ns)
            out[name]["parameters"] = sum(p.numel() for p in fp.model.parameters())
            for B in (1, 4096):
                x = (x1 + 0.05 * torch.randn(B, T, 9, device="cuda") * torch.tensor(ns, device="cuda")).contiguous()
                fp.warmup(B)
                for _ in range(20):
                    fp(x)
                ms = median_ms(fp, x, reps=200 if B == 1 else 30)
                out[name][f"{mode}_b{B}_ms"] = round(ms, 3)
                if B > 1:
                    out[name][f"{mode}_b{B}_aircraft_per_s"] = round(B / ms * 1e3)
                print(f"{name:8s} T={T:2d} compile={mode:15s} batch {B:5d}: {ms:8.3f} ms", flush=True)
            del fp
            torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))
    if out_path:
        json.dump(out, open(out_path, "w"), indent=2)


if __name__ == "__main__":
    main()

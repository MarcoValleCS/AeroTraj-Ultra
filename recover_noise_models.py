"""Recover the learned noise model of the classical filters from their checkpoints."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import torch

from bench_baselines import ClassicalIMMBaseline, KalmanFilterBaseline

cfg = json.load(open("config_365d_10s_bench.json", encoding="utf-8"))
hsec = list(cfg["window"]["horizon_seconds"])
dt = float(cfg["data"]["dt_seconds"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

out = {}
kf = KalmanFilterBaseline(hsec, dt_sec=dt).to(dev)
kf.load_state_dict(torch.load("checkpoints_bench_10s/kalman.pt", map_location=dev,
                              weights_only=False)["model_state_dict"])
out["kalman"] = kf.noise_report()

imm = ClassicalIMMBaseline(hsec, dt_sec=dt).to(dev)
imm.load_state_dict(torch.load("checkpoints_bench_10s/imm.pt", map_location=dev,
                               weights_only=False)["model_state_dict"])
out["imm"] = imm.noise_report()

path = "results_bench_10s/training_summary.json"
ts = json.load(open(path, encoding="utf-8")) if __import__("os").path.exists(path) else {}
ts.setdefault("models", {})
for k, v in out.items():
    ts["models"].setdefault(k, {})["learned_noise_model"] = v
json.dump(ts, open(path, "w", encoding="utf-8"), indent=2)

for k, v in out.items():
    print(f"== {k}")
    for kk, vv in v.items():
        print(f"   {kk}: {vv}")

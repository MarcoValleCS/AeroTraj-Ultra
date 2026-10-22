# coding=utf-8
"""
Recover the training history of the 1.2M run from its log.

That run died at the final checkpoint reload (the loader bug, since fixed) before
run.py could write results_365_1200k/evaluation_summary.json, so its per-epoch
history existed only in the ephemeral log.  This turns the log into a permanent
artifact, which is what makes the "the validation error saturated at epoch 137"
claim auditable.
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

LOG = os.path.join(os.environ["TEMP"], "train365_1200k.log")
OUT = "results_365_1200k/training_history.json"

raw = open(LOG, "rb").read()
try:
    text = raw.decode("utf-16")
except UnicodeError:
    text = raw.decode("utf-8", errors="replace")

pat = re.compile(
    r"Epoch\s+(\d+)/150 \(([\d.]+)s\) \| Train Loss:\s*([\d.]+) "
    r"\(BCE:\s*([\d.]+), Phys:\s*([\d.]+)\) \| Val MDE:\s*([\d.]+)m "
    r"\[([\d.]+)m,\s*([\d.]+)m, max 300s\s*([\d.]+)m\] \| lr ([\d.eE+-]+)")
hist = []
for line in text.splitlines():
    m = pat.search(line)
    if m:
        hist.append({
            "epoch": int(m.group(1)), "time_s": float(m.group(2)),
            "train_loss": float(m.group(3)), "train_bce": float(m.group(4)),
            "train_phys": float(m.group(5)), "val_mde_3d_m": float(m.group(6)),
            "val_first_horizon_m": float(m.group(7)), "val_60s_m": float(m.group(8)),
            "val_300s_m": float(m.group(9)), "lr": float(m.group(10)),
        })

best = min(hist, key=lambda e: e["val_mde_3d_m"])
ck = None
try:
    import torch
    ck = torch.load("checkpoints_365_1200k/aerotraj_ultra_best.pt",
                    map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"[warn] checkpoint non leggibile: {exc}")

out = {
    "run": "config_365d_1200k.json",
    "note": "history recovered from the training log: the run ended at the final "
            "checkpoint reload (a loader bug, since fixed) before run.py wrote its "
            "own evaluation_summary.json",
    "epochs_executed": len(hist),
    "best_epoch": best["epoch"],
    "best_val_mde_3d_m": best["val_mde_3d_m"],
    "checkpoint_epoch": int(ck["epoch"]) if ck else None,
    "checkpoint_val_mde_3d_m": float(ck["val_mde_3d_m"]) if ck else None,
    "test_metrics": "see results_365/cross_budgets.json (eval_cross.py)",
    "history": hist,
}
os.makedirs("results_365_1200k", exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=2)
print(f"epoche recuperate: {len(hist)}")
print(f"best: epoca {best['epoch']}, val {best['val_mde_3d_m']:.1f} m")
print(f"checkpoint: epoca {out['checkpoint_epoch']}, val {out['checkpoint_val_mde_3d_m']:.1f} m")
print(f"salvato {OUT}")

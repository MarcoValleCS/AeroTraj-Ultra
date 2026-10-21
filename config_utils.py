# coding=utf-8
"""
Configuration helpers shared by the data pipeline, the loader and the model.

The horizon grid is declared in SECONDS (`window.horizon_seconds`) and converted
to integer indices into the resampled sequence using `data.dt_seconds`.  Keeping
the derivation in one place is what allows the same code to serve a 5-second grid
and a 10-second one; before this, every consumer read a key literally named
`horizon_steps_5s`, which silently assumed a 5 s step.
"""
from __future__ import annotations

from typing import Dict, List


def resolve_horizon_steps(cfg: Dict) -> List[int]:
    """
    Horizon indices, in units of `data.dt_seconds`, for the declared horizons.

    Consistency is verified against `window.horizon_steps` (preferred) and
    `window.horizon_steps_5s` (legacy name, valid only for a 5 s grid), so a
    config whose declared steps disagree with its seconds/dt fails loudly instead
    of training on a silent mismatch.

    Both keys remain OPTIONAL: when absent the steps are derived from
    `horizon_seconds / dt_seconds`.  All existing 5 s configs declare
    `horizon_steps_5s` and pass the check unchanged.
    """
    dt = float(cfg["data"]["dt_seconds"])
    seconds = [float(s) for s in cfg["window"]["horizon_seconds"]]
    steps = [int(round(s / dt)) for s in seconds]
    if any(abs(s - k * dt) > 1e-6 for s, k in zip(seconds, steps)):
        raise ValueError(
            f"horizon_seconds {seconds} are not multiples of dt_seconds={dt}")
    if len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError(f"horizon steps must be strictly increasing, got {steps}")

    declared = cfg["window"].get("horizon_steps")
    if declared is not None and [int(x) for x in declared] != steps:
        raise ValueError(
            f"window.horizon_steps {list(declared)} disagrees with "
            f"horizon_seconds/dt={dt} -> {steps}")
    legacy = cfg["window"].get("horizon_steps_5s")
    if legacy is not None and [int(x) for x in legacy] != steps:
        raise ValueError(
            f"window.horizon_steps_5s {list(legacy)} is only valid for a 5 s grid; "
            f"with dt={dt} the steps are {steps}")
    return steps

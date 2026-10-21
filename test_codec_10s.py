# coding=utf-8
"""
Unit test for the Gray-Code delta codec on the 10 s configuration.

A coarser grid doubles every per-step displacement, so the delta lattices were
widened (lon/lat 10 -> 11 bits).  This checks, without any dataset:

  1. the representable RANGE of each attribute, in physical units, against the
     maximum motion the pipeline allows (0.04 deg per 5 s is its own rejection
     threshold, i.e. 0.08 deg per 10 s);
  2. the quantisation STEP, so the round-trip error can be read in metres;
  3. an exact encode -> decode round trip on random values inside the range;
  4. that values at the physical extremes are NOT clamped while clearly
     out-of-range ones are.

Run:  python test_codec_10s.py [--config config_365d_10s.json]
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from model import GrayCodeCodec

NAMES = ["d_lon", "d_lat", "d_alt_10m", "d_vx_kmh", "d_vy_kmh", "d_vz_ms"]
# Maximum REALISTIC per-step change, expressed in the INTEGER units the lattice
# stores (i.e. after multiplying by delta_scales), for a step of `dt` seconds:
#   * lon/lat: the pipeline's own rejection tolerance, 0.04 deg per 5 s, scaled;
#   * altitude: 25 m/s of climb (an extreme airliner rate) -> metres -> 10 m units;
#   * horizontal speed: 1 m/s^2 of acceleration, the practical upper bound;
#   * vertical rate: 1 m/s^2, likewise.
# `require(dt)` returns them for the configured dt.
def require(dt: float, scale_deg: float, scale_alt: float,
            scale_spd: float, scale_vz: float):
    return [
        0.04 * (dt / 5.0) * scale_deg,            # d_lon
        0.04 * (dt / 5.0) * scale_deg,            # d_lat
        (25.0 * dt / 10.0) * scale_alt,           # d_alt in 10 m units
        (1.0 * dt * 3.6) * scale_spd,             # d_vx in km/h
        (1.0 * dt * 3.6) * scale_spd,             # d_vy
        (1.0 * dt) * scale_vz,                    # d_vz in m/s
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_365d_10s.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    dt = float(cfg["data"]["dt_seconds"])
    codec = GrayCodeCodec(cfg)
    sc = codec.delta_scales
    rng = require(dt, float(sc[0]), float(sc[2]), float(sc[3]), float(sc[5]))
    print(f"[codec] config {args.config} | dt={dt} s | delta bits {codec.delta_sizes} "
          f"| totali {codec.total_delta_bits} bit")
    print(f"[codec] magnitudini massime: {[2 ** (b - 1) - 1 for b in codec.delta_sizes]}"
          f"  (in unita' intere del reticolo)")

    ok = True
    for k, name in enumerate(NAMES):
        bits = codec.delta_sizes[k]
        max_mag = 2 ** (bits - 1) - 1
        scale = float(codec.delta_scales[k])
        need = rng[k]
        head = max_mag / need
        good = head >= 1.15
        ok &= good
        print(f"  {name:10s} {bits:2d} bit  max {max_mag:5d}  "
              f"richiesto {need:7.1f}  margine x{head:5.2f}  "
              f"(passo fisico 1/{scale:g})  {'OK' if good else '!! INSUFFICIENTE'}")

    # 3) exact round trip inside the range
    print("\n[codec] round-trip encode -> decode (valori casuali nel range):")
    torch.manual_seed(0)
    for k, name in enumerate(NAMES):
        bits = codec.delta_sizes[k]
        max_mag = 2 ** (bits - 1) - 1
        scale = float(codec.delta_scales[k])
        lim = (max_mag - 1) / scale
        v = (torch.rand(20000, dtype=torch.float64) * 2 - 1) * lim
        enc = codec.encode_signed((v * scale), bits)              # [N, bits]
        probs = enc.float() * 0.998 + 0.001
        dec = codec.decode_signed_ste(probs) / scale
        err = (dec.double() - v).abs()
        step = 1.0 / scale
        good = bool((err <= step).all())
        ok &= good
        print(f"  {name:10s} err max {err.max().item():.3e}  (passo {step:.3e})  "
              f"<= 1 passo: {'OK' if good else '!! FALLITO'}")

    # 4) the decoded magnitude must NEVER exceed the lattice maximum: that is the
    #    property that matters (a wraparound would silently corrupt the target),
    #    and it holds both at the boundary and far outside it.
    print("\n[codec] limiti (nessun wraparound, valore giusto appena dentro):")
    for k, name in enumerate(NAMES):
        bits = codec.delta_sizes[k]
        max_mag = 2 ** (bits - 1) - 1
        scale = float(codec.delta_scales[k])
        lim = max_mag / scale
        vals = torch.tensor([0.9 * lim, lim, 10.0 * lim, -10.0 * lim], dtype=torch.float32)
        dec = codec.decode_signed_ste(
            codec.encode_signed(vals * scale, bits) * 0.998 + 0.001) / scale
        no_overflow = bool((dec.abs() <= lim + 1e-6).all())
        inside_ok = abs(abs(dec[0].item()) - 0.9 * lim) <= 1.0 / scale + 1e-3
        sat_ok = bool(abs(abs(dec[1].item()) - lim) <= 1e-3
                      and abs(abs(dec[2].item()) - lim) <= 1e-3)
        good = no_overflow and inside_ok and sat_ok
        ok &= good
        print(f"  {name:10s} 0.9x {dec[0].item():+11.4f} | max {dec[1].item():+11.4f} | "
              f"10x {dec[2].item():+11.4f} | nessun overflow {no_overflow} "
              f"{'OK' if good else '!! ANOMALO'}")

    print(f"\n[codec] {'TUTTO OK' if ok else 'PROBLEMI RILEVATI'}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()

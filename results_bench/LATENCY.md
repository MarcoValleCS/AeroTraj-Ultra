# Latency and throughput (inference)

Device: **NVIDIA GeForce RTX 4070** | horizons 5-300 s | batch-1 latency over 200 repetitions (median / p95).

## Single-aircraft latency (batch 1, end-to-end incl. H2D)

Median of 3 independent measurement blocks (200 repetitions each). `GPU only` is the
median of a per-call CUDA-event measurement (one event pair around each call) and its
observed spread across the 3 blocks.

| Model | median [ms] | p95 [ms] | GPU only [ms] | GPU spread |
|---|---:|---:|---:|---:|
| AeroTraj-Ultra | 19.59 | 23.39 | 19.34 | ±5% |
| AeroTraj-Ultra + retrieval | 20.12 | 23.88 | 20.19 | ±1% |
| FlightBERT++ V2 | 9.77 | 11.10 | 9.78 | ±1% |
| Deep LSTM | 0.62 | 1.25 | 0.61 | ±2% |
| ASCENT | 12.05 | 13.85 | 11.98 | ±1% |
| CFM (20 step, 1 campione) | 93.26 | 102.17 | 93.40 | ±1% |
| DDPM (50 step, 1 campione) | 262.31 | 357.12 | 260.42 | ±3% |
| CV / Doppler | 0.46 | 1.18 | 0.46 | ±8% |

## Component breakdown (batch 1)

- retrieval top-k over 342,572 candidates: **1.09 ms**
- model forward: **22.31 ms**
- full pipeline (AeroTraj-Ultra + retrieval): **20.12 ms**

## Component breakdown (batch 512)

- retrieval top-k: **20.78 ms**
- model forward: **23.47 ms**

## Why batch 1 costs what it costs

- marginal cost of one extra aircraft (slope from batch 1 to batch 4096): **12.6 us**
- average cost at batch 1: **19,591 us** per aircraft, of which **99.94 %** is fixed overhead (19.6 ms)
- the single-aircraft case is therefore **launch-bound, not compute-bound**

## Operational budget

- 20.12 ms per aircraft -> **50 aircraft/s** against a 1 s update cycle, **248 aircraft/s** against a 5 s cycle

## Throughput

| Model | batch | total [ms] | us/aircraft | aircraft/s |
|---|---:|---:|---:|---:|
| AeroTraj-Ultra | 1 | 19.64 | 19638.1 | 51 |
| AeroTraj-Ultra | 8 | 20.33 | 2541.5 | 393 |
| AeroTraj-Ultra | 32 | 20.22 | 631.9 | 1,583 |
| AeroTraj-Ultra | 128 | 20.21 | 157.9 | 6,333 |
| AeroTraj-Ultra | 512 | 20.85 | 40.7 | 24,556 |
| AeroTraj-Ultra | 2048 | 40.17 | 19.6 | 50,990 |
| AeroTraj-Ultra | 4096 | 70.99 | 17.3 | 57,702 |
| AeroTraj-Ultra + retrieval | 1 | 22.85 | 22854.7 | 44 |
| AeroTraj-Ultra + retrieval | 8 | 21.83 | 2729.1 | 366 |
| AeroTraj-Ultra + retrieval | 32 | 21.57 | 674.2 | 1,483 |
| AeroTraj-Ultra + retrieval | 128 | 21.52 | 168.1 | 5,949 |
| AeroTraj-Ultra + retrieval | 512 | 33.66 | 65.7 | 15,212 |
| AeroTraj-Ultra + retrieval | 2048 | 116.23 | 56.8 | 17,620 |
| AeroTraj-Ultra + retrieval | 4096 | 231.46 | 56.5 | 17,697 |

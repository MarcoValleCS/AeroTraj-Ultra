# AeroTraj-Ultra — project guide

The single entry point for understanding, running and extending this repository.
It is written to be read top to bottom by someone who has never seen the project, and to
be used as a reference afterwards.

**What this is.** A short-term (10 s – 5 min) flight-trajectory predictor for European
airspace, built on real ADS-B data from the OpenSky Network, with a complete experimental
record: the model, twelve re-implemented baselines, three datasets, a latency study, a
calibrated uncertainty layer and a retrieval prior over past trajectories.

**Where the evidence lives.**

| document | content |
|---|---|
| **this guide** | how the project is organised, how to run it, what the conventions are |
| [`PAPER.md`](PAPER.md) | the technical report: method, experiments, results, negative results, limitations |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | the formal specification of the model, equation by equation |
| [`MODELS.md`](MODELS.md) | every implemented model, with its source and its conformance report |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | the chronological lab notebook (Italian), every run and every number |
| [`figures/README.md`](figures/README.md) | the figures and the three ways they are produced |

---

## 1. The problem

Given the last **12 ADS-B state vectors** of an aircraft, sampled every **10 s** (120 s of
history), predict its state at six horizons:

```
10 s · 30 s · 60 s · 90 s · 180 s · 300 s
```

The state is six numbers: `[lon, lat, alt, vx, vy, vz]`, with the conventions of §9.
The prediction is **non-autoregressive**: one forward pass emits the whole horizon vector,
so the error does not accumulate through a roll-out and the latency does not grow with the
horizon.

The error metric is the **3-D ECEF distance** between the predicted and the true position,
averaged over the windows and the horizons (MDE, metres). Two secondary metrics are
reported: ADE (the mean over the horizons) and FDE (the last horizon only).

---

## 2. Quick start

```powershell
# 1. environment (Windows, one GPU; see §10 for the details)
pip install -r requirements.txt

# 2. build the dataset (downloads from OpenSky, resumable, ~4 h for 365 days)
python data_pipeline.py --config config_365d_10s.json

# 3. verify the grid before spending GPU hours (bit saturation, round-trip, forward pass)
python verify_10s.py --config config_365d_10s.json

# 4. train the model
python run.py --config config_365d_10s.json --fresh --skip-baselines --compile reduce-overhead

# 5. evaluate a checkpoint on any split
python eval_ckpt.py --checkpoint checkpoints_365_10s/aerotraj_ultra_best.pt `
                    --eval-config config_365d_10s.json --split test

# 6. the benchmark of the re-implemented literature (same data, same budget)
python bench_run_10s.py --config config_365d_10s_bench.json
python plot_bench_10s.py
python make_table_mae_rmse.py

# 6b. AeroTraj-Atlas (the airspace field) at the benchmark budget, 5 s grid
python run.py --config config_365d_12pt_atlas_bench.json --fresh --skip-baselines

# 7. the conformance and consistency checks
python test_conformance.py
python test_baselines.py
python test_codec_10s.py
python check_docs.py
```

The datasets are large and gitignored: `data365/` (20.6 GB, 5 s grid) and `data365_10s/`
(5.2 GB, 10 s grid).  Neither is needed to read the results — every number in the
documents is in `results*/`.

---

## 3. How the code is organised

The pipeline has six stages, each with its own script:

```
OpenSky S3/Trino  ──►  data_pipeline.py  ──►  data365/  (points + segment table + stats)
                                              │
                       build_10s_dataset.py ──┴──►  data365_10s/   (time decimation)
                                              │
                       dataloader.py  ─────────┘   memory-mapped sliding windows
                                              │
   model.py / bench_models.py / bench_baselines.py   the models
                                              │
   run.py / bench_run.py / bench_run_10s.py   training + evaluation
                                              │
   results*/  ──►  plot_*.py, make_*.py  ──►  docs/figures/
```

**Stage 1 — acquisition** (`data_pipeline.py`).  Streams one day of OpenSky data at a
time, resamples it into contiguous 10 s (or 5 s) trajectories, caches every day on disk
(`_cache_days/`) so an interrupted build resumes, and writes three arrays per split
(`*_points.npy`, `*_segments.npy`) plus a `dataset_stats.json` that carries a
**signature**: a hash of the structural parameters.  A dataset whose signature does not
match the configuration is never silently reused.

**Stage 2 — the coarser grid** (`build_10s_dataset.py`).  The 10 s dataset is *derived* from
the 5 s one by keeping every second point, with the segments re-indexed.  No re-download.
The derived dataset gets its own signature and records `derived_from`.

**Stage 3 — windows** (`dataloader.py`).  A window is 12 consecutive points of one flight.
The loader is memory-mapped (the 365-day array is 20 GB) and vectorised: it slices the
anchors with NumPy and copies a batch to the GPU, so the per-sample Python overhead is
zero.  The train pool is 25 M windows; an epoch samples 400 k or 1.2 M of them.

**Stage 4 — the models.**  `model.py` is the project model and the classical references;
`bench_models.py` the five models of the 90-day benchmark; `bench_baselines.py` the five
models of the 365-day / 10 s benchmark.  See [`MODELS.md`](MODELS.md).

**Stage 5 — training and evaluation.**  `run.py` trains the project model; `bench_run.py`
and `bench_run_10s.py` train the baselines with the same protocol; `eval_ckpt.py` and
`eval_cross.py` evaluate a checkpoint on any split.

**Stage 6 — reporting.**  `plot_*.py` draws the figures, `make_*.py` builds the tables,
`check_docs.py` verifies that the documents and the repository agree.

---

## 4. Data

### 4.1 Source and coverage

| | |
|---|---|
| source | OpenSky Network, `s3.opensky-network.org/competition-data` (anonymous S3, via `pyopensky`) |
| region | `lon ∈ [−15°, 35°]`, `lat ∈ [30°, 65°]`, `alt ∈ [−500, 50 000] ft` — European airspace |
| period | 2022-01-01 → 2022-12-31 (365 consecutive days) |
| flights | 438 215 unique flights, 306.8 M five-second points |
| splits | chronological, disjoint: **293 / 36 / 36** days |

The splits are chronological on purpose: a random split would put two windows of the same
flight on both sides of the boundary and inflate the score.  Midnight-crossing flights are
split at the day boundary (~10 flights per boundary).

### 4.2 The nine channels

| # | name | unit | notes |
|---:|---|---|---|
| 0 | `lon` | deg WGS-84 | |
| 1 | `lat` | deg WGS-84 | |
| 2 | `alt_10m` | 10 m | barometric altitude, `ft × 0.03048` |
| 3 | `vx_kmh` | km/h | eastward ground speed, **Doppler** (not a position difference) |
| 4 | `vy_kmh` | km/h | northward ground speed |
| 5 | `vz_ms` | m/s | vertical rate |
| 6 | `u_wind` | m/s | eastward wind (ERA5, interpolated) |
| 7 | `v_wind` | m/s | northward wind |
| 8 | `temp_k` | K | static temperature |

`(vx, vy)` come from the ADS-B ground speed and track angle, not from finite differences:
at 10 s the CPR quantisation of the position is ±75 m, which would make a difference
quotient uselessly noisy.

### 4.3 The two grids

| | 5 s grid | 10 s grid |
|---|---|---|
| dataset | `data365/` | `data365_10s/` |
| step | 5 s | **10 s** |
| window | 12 points = 60 s | 12 points = **120 s** |
| horizons | 13: 5…300 s | 6: 10, 30, 60, 90, 180, 300 s |
| config | `config_365d.json` | `config_365d_10s.json` |

The 10 s grid exists to test **how much history the model needs**: at a fixed 12-point
window, halving the sampling rate doubles the history.  The measured verdict is in
`PAPER.md` §3.1: on identical test windows the 10 s grid costs +5.3 % at 10–30 s and gains
2.0 % at 300 s (−0.8 % on the mean of the six shared horizons), and a 24-point window at 5 s
(`config_365d_24pt_bench.json` vs `config_365d_12pt_bench.json`) gains 0.67 %, less than the
seed-to-seed spread.  The default stays 12 points at 5 s; the 10 s grid remains the
protocol of the literature benchmark.  An earlier version reported −7.3 % at 300 s for the
long window: that compared the two models on different window sets (see §9, *identical
windows*).

### 4.4 Dataset signature

`dataset_signature()` hashes `num_total_days`, the split sizes, `row_groups_per_day`,
`dt_seconds`, `min_segment_steps`, the region bounds, the feature names and the list of
days.  The consequence is that a dataset built for a 5 s configuration can never be
silently consumed by a 10 s one — the failure mode that would otherwise produce a model
trained on the wrong time step.  Before touching the network, `build_dataset_splits()`
validates an on-disk dataset against the structural part of the signature, so the whole
repository works offline once the data is there.

---

## 5. The models

A one-line summary; the full catalogue, with the sources and the conformance report, is
[`MODELS.md`](MODELS.md).

| model | file | parameters | role |
|---|---|---:|---|
| **AeroTraj-Ultra** | `model.py` | 1 277 706 | the project model |
| **AeroTraj-Atlas** | `model.py` | 10 495 863 (9 134 208 of them the map) | AeroTraj-Ultra + a learned airspace field |
| FlightBERT++ V2, reference | `model.py` | 763 201 | the previous model of the project |
| Kalman filter (CV) | `bench_baselines.py` | 6 | A3 of the FlightBERT paper |
| IMM (CV/CA/CT) | `bench_baselines.py` | 25 | Blom & Bar-Shalom 1988 |
| LSTM / BiLSTM regression | `model.py` / `bench_baselines.py` | 376 k / 1 357 k | A1 / A2 of the FlightBERT paper |
| FlightBERT (2023) | `bench_baselines.py` | 57 096 276 | the binary-encoding ancestor |
| FlightBERT++ + ACoAtt | `bench_baselines.py` | 936 705 | ablation |
| ASCENT, CFM, DDPM, attention-LSTM, IMM-residual | `bench_models.py` | 0.4–1.1 M | the 90-day benchmark |
| CV (Doppler), CV least-squares | `model.py` | 0 | no-learning references |

**AeroTraj-Atlas** is AeroTraj-Ultra plus an **airspace field**: a learned map of the
airspace (a multi-resolution hash grid over longitude, latitude, a periodic heading axis and
an altitude axis) that the horizon queries read at the aircraft, along the kinematic prior,
and again along a trajectory proposed by the first decoder layer. It is not a new class: it
is `AeroTrajUltra` with the config key `model.airspace_field` (final setting
`{"n_alt": 5, "refine": true}`); without the key the model is AeroTraj-Ultra bit for bit.
Measured on the Nov–Dec 2022 split, the same 60 000 windows: **723.3 m against 865.1 m** for
AeroTraj-Ultra at the full budget (−16.4 %, better at every horizon), and **843.9 m against
1 101.2 m** at the benchmark budget (−23.4 %, mean of two seeds). Specification:
[`ARCHITECTURE.md`](ARCHITECTURE.md) §15; results: [`PAPER.md`](PAPER.md) §6.8 and the
README §4.12.

---

## 6. Training

### 6.1 The recipe

| item | value |
|---|---|
| optimiser | AdamW, lr 1e-3 (the FlightBERT rows: 1e-4, the paper's), weight decay 1e-4 |
| schedule | OneCycleLR, `pct_start = 0.2`, cosine anneal |
| batch | 2 048 (the FlightBERT: 512, a 57 M-parameter model does not fit at 2 048 on 12 GB) |
| samples / epoch | 400 k (the benchmark) or 1.2 M (the long runs) |
| epochs | 25 (the benchmark), 150 (the long runs) |
| early stopping | patience 8 / 20, `min_delta` 0.5 m, a floor of 10 / 40 epochs |
| gradient clipping | norm 1.5 |
| mixed precision | no: fp32 with TF32 matmuls |
| `torch.compile` | `reduce-overhead` when available; **off** for the benchmark, uniformly |

### 6.2 The objective

The project model minimises a sum of five terms (the exact equations are in
`ARCHITECTURE.md` §11):

| term | weight | what it does |
|---|---:|---|
| `L_pos` | 2.5 | Smooth-L1 on the position, normalised per attribute and by the horizon |
| `L_vel` | 0.6 | Smooth-L1 on the three velocity components |
| `L_BCE` | 0.4 | binary cross-entropy on the 55 Gray-Code delta bits |
| `L_STE` | 0.25 | Smooth-L1 on the *decoded* trajectory (the Hybrid Constrained term) |
| `L_kin` | 0.25 | kinematic consistency: position increments must match the integrated velocities |

The horizon normalisation `h_norm = 0.04 + 0.96·(Δt/60 s)` is what keeps the 300 s horizon
from dominating the gradient: without it the model learns the long horizons and neglects
the short ones.

### 6.3 What to expect on the clock

| run | hardware | time |
|---|---|---|
| 1.2 M samples / epoch, 150 epochs, `torch.compile` | RTX 4070 | ~2 h |
| 400 k samples / epoch, 25 epochs, eager | RTX 4070 | ~9 min (`ours`), ~14 min (FlightBERT++) |
| the whole 7-model benchmark | RTX 4070 | ~2.5 h |
| AeroTraj-Atlas, 400 k samples / epoch, 25 epochs, eager (5 s grid) | RTX 4070 | 35.7 s/epoch (AeroTraj-Ultra: 23.5) |
| AeroTraj-Atlas, 1.2 M samples / epoch, 150 epochs, `torch.compile` | RTX 4070 | 53.2 s/epoch, 2.39 h (AeroTraj-Ultra: 47.1 s, 1.98 h) |

---

## 7. Evaluation

**Protocol.**  The test split is never seen during training or model selection.  The error
is computed per horizon and then averaged; the per-attribute MAE/RMSE (the format of the
FlightBERT paper's Table I) is produced by `make_table_mae_rmse.py`.

**Cross-split evaluation.**  `eval_cross.py` evaluates a checkpoint on *both* test splits
(spring and autumn), because a single 36-day window can over- or under-state a model by
~10 % depending on the season (`PAPER.md` §6.3).  `--split LABEL=config.json=split`
(repeatable) replaces the two built-in test splits and feeds every model the last
`inp_seq_len` points of the split's windows, so models with different window lengths are
compared on identical anchors; the split must declare the longest window.  `eval_ckpt.py`
slices the window the same way and also loads checkpoints that predate the `dt`
parameterisation.

**Seed.**  `run.py` takes the seed from the configuration's `seed` key (every committed
configuration says 97).

**Metrics.**  `run.compute_trajectory_metrics` returns, per horizon, the 3-D ECEF MDE, the
2-D horizontal error, the altitude error and the per-attribute MAE/RMSE.  The 3-D distance
is computed on the ECEF coordinates, not by summing degrees and metres, which is the
mistake that makes the published figures of this field incomparable.

---

## 8. Results at a glance

The details, the tables and the discussion are in `PAPER.md` §6.

| setting | model | MDE |
|---|---|---:|
| 365 days, 5 s, 1.2 M × 150 | **AeroTraj-Atlas** | **723.3 m** (Nov–Dec; −16.4 %; March split not evaluated) |
| 365 days, 5 s, 1.2 M × 150 | AeroTraj-Ultra | **790.2 m** (March split) · **865.1 m** (Nov–Dec) |
| 365 days, 5 s, 400 k × 25, two seeds | AeroTraj-Atlas | 843.9 m (Nov–Dec, −23.4 %) |
| 365 days, 5 s, 400 k × 25, two seeds | AeroTraj-Ultra | 1 101.2 m (Nov–Dec) |
| 365 days, 5 s, 400 k × 150 | AeroTraj-Ultra | 847.3 m · 926.0 m |
| 90 days, 5 s | AeroTraj-Ultra | 991.6 m |
| 90 days, 5 s | FlightPREV (the previous model) | 1 495.8 m |
| 365 days, 10 s, same budget | AeroTraj-Ultra | **1 256.0 m** |
| 365 days, 10 s, same budget | FlightBERT++ V2 | 1 437.1 m |
| 365 days, 10 s, same budget | Kalman filter (learned noise) | 2 069.4 m |

Latency: 2.8–6.5 ms per aircraft with `torch.compile`, 19.3–20.7 ms eager
(`PAPER.md` §8). AeroTraj-Atlas, on a harness that excludes the host-to-device copy (so not
comparable with those numbers): 0.66 ms compiled against 0.65 ms for AeroTraj-Ultra, 42.2 ms
eager against 17.4 ms.

---

## 9. Conventions

These are the details that silently break a re-implementation if they are guessed wrong.

| item | convention |
|---|---|
| **state order** | `[lon, lat, alt, vx, vy, vz]` — longitude first, **not** latitude |
| **altitude** | stored in units of **10 m**; decoded to metres only for reporting |
| **velocities** | `vx, vy` in km/h (eastward, northward), `vz` in m/s (upward) |
| **local frame** | origin = the last observed point, `x` forward along the track, `y` right, `z` up, metres; the frame is a tangent plane (curvature error ~1e-4 relative over 1 km) |
| **bit order** | most significant bit first; for a signed attribute the sign is the highest bit (1 = negative) |
| **Gray code** | `m XOR (m >> 1)`; the inverse is the prefix XOR, computed in closed form as a product of parities |
| **horizon steps** | derived from the seconds and `dt` by `config_utils.resolve_horizon_steps()`, never hard-coded |
| **decoding** | straight-through: the forward pass uses the thresholded bits, the backward pass the sigmoid |
| **ECEF** | WGS-84 ellipsoid, semi-major 6378.137 km, flattening 1/298.257223563 |
| **identical windows** | compare models on identical windows: models with different window lengths have different valid anchors, and the first window of every segment is 2.6× harder — use `eval_cross.py --split` with the config of the longest window |
| **normalisation statistics** | the statistics stored with every dataset were computed in float32 and are wrong (on the 365-day set the stored latitude mean is 4.58° for a true 48.85°); they remain the **default** so that every published run stays reproducible. `python data_pipeline.py --config <cfg> --exact-norm-stats` adds float64 statistics (`train_norm_stats_exact`) without touching the stored ones, and the config key `data.norm_stats` selects them. Measured effect on accuracy: none (see §11) |

---

## 10. Environment

| item | value |
|---|---|
| Python | 3.12.7 (Anaconda) |
| PyTorch | 2.6.0 + cu126 |
| GPU | RTX 4070, 12 GB (the 57 M-parameter FlightBERT needs batch 512) |
| RAM | 34 GB (the loader is memory-mapped, so the datasets do not have to fit) |
| `torch.compile` | needs `triton-windows==3.2.0.post21` on Windows; without it the training falls back to eager (~1.5× slower) |
| figures | matplotlib (schematics), Plotly + Kaleido (3-D panels), Graphviz `dot` (the DOT variant), MiKTeX `pdflatex` (the TikZ variant) |
| optional | `pymupdf` (the PDF → PNG conversion of the TikZ figures) |

Two environment traps, both documented in `requirements.txt`:

1. **numpy 2 vs the Anaconda binaries.**  `pyopensky ≥ 2.17` requires numpy ≥ 2.0.2, but
   the matplotlib/numexpr/bottleneck wheels shipped with the base image are built against
   the numpy 1.x ABI and crash on import.  The combination in `requirements.txt` is the
   verified one; do not let pip downgrade numpy.
2. **The inductor's C++ path.**  On a machine without a C++ compiler, `torch.compile`
   fails for graphs that route part of the computation to the CPU.  The benchmark therefore
   runs with compilation off, uniformly for every model.

---

## 11. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `S3 operation failed: AccessDenied` | OpenSky's anonymous listing is unavailable | the on-disk dataset is now validated first; if there is none, the download needs working OpenSky access |
| `CUDA out of memory` with the FlightBERT | batch 512 × 57 M parameters | keep `BATCH_SIZE["flightbert"] = 512`, or reduce it |
| `torch.compile` raises `Compiler: cl is not found` | no C++ compiler for the inductor CPU path | run with `--compile none` (the benchmark's default) |
| `size mismatch for lstm.weight_hh_l1` when loading a checkpoint | the checkpoint belongs to a different width | `check_baselines.py`/`plot_trajectory_cases.py` pick the class from the file name; check which model the checkpoint is |
| the model evaluates to ~10⁷ m | the bit encoding was decoded without the high bits | see `test_loss_balance.py`: a regression-only loss on a redundant binary code fails exactly this way |
| a dataset is silently reused for a different step | should be impossible | the signature check; `verify_10s.py` prints it and aborts on a mismatch |
| the stored latitude mean of a dataset is 4.58° (altitude mean 141.7, temperature 35.4 K, std of `vx` 271.6 km/h on the 365-day set) | `data_pipeline.py` computed the mean of a float32 array along axis 0: NumPy keeps a float32 running sum, which stops growing; the error grows with the dataset size | new builds use `exact_norm_stats` (float64, by chunks); for an existing dataset `python data_pipeline.py --config <cfg> --exact-norm-stats`, then `data.norm_stats` in the config. Effect on accuracy (benchmark budget, seed 97, `results_atlas/norm_stats_ab.json`): AeroTraj-Ultra 1 091.5 → 1 095.6 m (+0.38 %), AeroTraj-Atlas 846.5 → 843.8 m (−0.32 %), both inside the seed spread; the stored statistics stay the default |
| `eval_ckpt.py` fails with missing codec buffers on an old checkpoint | strict load of a checkpoint older than the `dt` parameterisation (two codec buffers missing), including the one of the README's example command | fixed: those checkpoints now load; unexpected keys still fail |
| a 12-point checkpoint evaluated with a 24-point config | the evaluation tools did not check the window length | fixed: `eval_cross.py` and `eval_ckpt.py` give each model the last `inp_seq_len` points |
| `refine` with `n_modes > 0` raises an error | the airspace-field re-read and the multi-hypothesis head are not implemented together | use one or the other |

---

## 12. Repository map

```
    model and training
      model.py                  the model, the airspace field (AirspaceField), the Gray-Code
                                codec, the losses, the classical baselines
      config_utils.py           horizon grid <-> time step
      run.py                    training loop, evaluation, plots
      dataloader.py             memory-mapped sliding-window loader (data.norm_stats)
      data_pipeline.py          OpenSky acquisition and preprocessing (--exact-norm-stats)
      inference.py              FastPredictor: compiled + inference_mode
      predict.py                inference on pre-processed windows
      eval_ckpt.py              evaluate a checkpoint on a split
      eval_cross.py             cross-test-set evaluation
      eval_window_study.py      the window study on identical windows
      eval_where_gain.py        two checkpoints by altitude / turn / level, paired bootstrap

    benchmarks
      bench_models.py           the five models of the 90-day benchmark
      bench_baselines.py        the five re-implementations of the 10 s benchmark
      bench_run.py              harness for the 90-day benchmark
      bench_run_10s.py          harness for the 365-day / 10 s benchmark
      bench_experiments.py      post-hoc analyses
      bench_latency*.py         latency and throughput (bench_latency_field.py: a variant
                                on FastPredictor, input already on the GPU)
      bench_opt.py              inference-variant matrix
      bench_retrieval*.py       retrieval search experiments
      check_compile_quality.py  does compilation change the metric?
      check_pool_quality.py     does the pruned pool change the metric?

    retrieval and uncertainty
      retrieval.py              retrieval engine
      run_retrieval.py          training of the retrieval variant
      retrieval_probe.py        does the retrieval prior carry signal?
      retrieval_diagnostic.py   descriptor sweep and oracle ceiling
      calibrate_uncertainty.py  conformal calibration

    datasets and legacy
      build_10s_dataset.py      derive the 10 s grid by decimation
      verify_10s.py             grid verification
      verify_all.py             end-to-end state verification
      legacy_flightprev.py      FlightPREV, the previous model re-implemented
      run_legacy.py             its training and evaluation
      recover_1200k_history.py  rebuild a training history from a log
      recover_noise_models.py   rebuild the learned noise of the filters

    tests
      test_conformance.py       79 checks: frames, codecs, filters vs NumPy, metrics, losses,
                                contracts, the airspace field, the dataset statistics
      test_baselines.py         unit tests of the re-implementations
      test_codec_10s.py         the Gray-Code lattices on the 10 s grid
      test_loss_balance.py      the BCE/MSE balance of the Hybrid Constrained loss
      check_docs.py             documentation vs repository

    figures and tables
      plot_architecture.py      model schematic
      plot_loss.py              hybrid-objective schematic
      plot_acoat_block.py       ACoAtt block
      plot_model_dot.py         the model as Graphviz DOT
      plot_trajectory_cases.py  six-case 3-D comparison
      plot_pareto.py            accuracy vs latency
      plot_bench_10s.py         the comparison sheets
      plot_style.py             the shared design system
      make_table_mae_rmse.py    the MAE/RMSE table in the FlightBERT format
      make_bench_10s_report.py  the tables of the 10 s benchmark
      make_benchmark_report.py  the unified 12-model table
      make_report.py            markdown/CSV report generator

    configurations
      config.json                     the default template
      config_365d.json                365 days, 5 s, horizons 5-300 s
      config_365d_1200k.json          the same, 1.2 M samples/epoch
      config_365d_10s.json            365 days, 10 s, horizons 10-300 s
      config_365d_10s_bench.json      the benchmark budget (25 x 400 k)
      config_365d_12pt_bench.json     365 days, 5 s, 12 points, benchmark budget
      config_365d_24pt_bench.json     the same with 24 points (120 s of history)
      config_365d_12pt_atlas_bench.json  AeroTraj-Atlas, benchmark budget
      config_365d_atlas_1200k.json    AeroTraj-Atlas, full budget (1.2 M x 150)
      config_90d_300s.json            90 days, the historical benchmark
      config_90d_300s_hyp.json        the same + the multi-hypothesis head
      config_90d_300s_regress.json    the score-regression variant
      config_180d.json                the data-scaling intermediate point
      config_90d.json                 90 days, horizons 5-600 s
      config_90d_180s.json            the 90-day horizon control
      config_40d_gpu.json             the 40-day GPU control

    results (text, committed) and weights (local, gitignored) of the latest studies
      results_window/                 the window study (configs of the seed replicas in configs/)
      results_atlas/                  the AeroTraj-Atlas ablation, seeds, where-gain, latency,
                                      compile quality, norm-stats A/B (configs of every arm in configs/)
      results_365_atlas/              AeroTraj-Atlas at the full budget, and the comparison with Ultra
      checkpoints_window/, checkpoints_atlas/, checkpoints_365_atlas/   the corresponding weights
```

The configurations of the 40/90/180-day runs are kept as **recipes**: the corresponding
datasets were removed to free 14.3 GB and can be rebuilt with `data_pipeline.py`.

---

## 13. Tests

```
python test_conformance.py    # 79 checks           — the models against their sources
python test_baselines.py      # 26 checks, ~1 min   — the re-implementations, one by one
python test_codec_10s.py      # the codec on the 10 s grid
python test_loss_balance.py   # the BCE/MSE experiment (400 training steps)
python check_docs.py          # the documents against the repository
```

`test_conformance.py` is the important one: it is the executable form of
[`MODELS.md`](MODELS.md) §3.  It compares the Kalman filter against an independent NumPy
implementation, pins the IMM to a single mode and compares it with the same filter, checks
that the covariances stay positive semi-definite, that the coordinate frame is the
identity, that the codecs round-trip within their lattice, that the metrics agree with an
independent haversine computation, and that the published hyper-parameters are the ones we
implement.  Section 10 covers the AeroTraj-Atlas options: the airspace field against its own
table and the aircraft frame against the geometry; section 11 checks the dataset statistics
against a float64 reference.

---

## 14. Glossary

| term | meaning |
|---|---|
| **ADS-B** | Automatic Dependent Surveillance–Broadcast: the aircraft reports its state ~1 Hz |
| **CPR** | Compact Position Reporting: the encoding of the ADS-B position, quantised to ~5 m, whose decoding noise is the ~75 m floor of this problem |
| **MDE** | Mean Displacement Error, the mean 3-D ECEF distance over the windows and horizons |
| **ADE / FDE** | average / final displacement error over the horizon vector |
| **minADE@K** | the best of K samples — an *oracle*, not an operational metric |
| **ACoAtt** | Attribute Correlation Attention: the block that recalibrates an attribute embedding with a kinematically correlated one |
| **MBC** | Multi-Binary Classification: predicting the bits of an encoding as independent binary decisions |
| **HC loss** | Hybrid Constrained loss: BCE on the bits + a regression term on the decoded value |
| **STE** | Straight-Through Estimator: the hard value in the forward pass, the smooth gradient in the backward pass |
| **Gray code** | a binary encoding where consecutive integers differ by one bit |
| **BE** | Binary Encoding: the plain base-2 encoding used by the 2023 FlightBERT |
| **IMM** | Interacting Multiple Model: a bank of filters with mode probabilities |
| **ECE** | Earth-Centred, Earth-Fixed: the Cartesian frame in which the 3-D error is measured |
| **budget** | `epochs × samples/epoch`: what makes a comparison between models fair |

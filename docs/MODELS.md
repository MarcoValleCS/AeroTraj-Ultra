# Models — catalogue and conformance report

This document is the register of **every model implemented in this repository**: what it
is, where it comes from, what it contains, and how faithfully it reproduces its source.
It is the companion of `test_conformance.py`, which checks the claims numerically.

```
python test_conformance.py        # 79 checks, exit code 0 when everything passes
```

The results of the models on the benchmark are in `docs/PAPER.md` (§6) and in
`results_bench_10s/`; this file is about *what the implementations are*, not about how
well they score.

---

## 1. How conformance is established

There is no such thing as "the specification" for all of these models, so the verification
uses four independent levels, from the strongest to the weakest:

| level | method | applies to |
|---|---|---|
| **L1 — independent re-derivation** | the same equations written a second time, in NumPy, and compared numerically | the Kalman filter, the IMM |
| **L2 — closed-form identity** | a property that must hold exactly (a round-trip is the identity, a covariance stays PSD, a probability vector sums to one, a loss is zero at its minimum) | the coordinate frames, the codecs, the metrics, the losses, the filters |
| **L3 — published specification** | the hyper-parameters of the source paper or of its reference implementation, compared field by field | FlightBERT (T-ITS 2023), FlightBERT++ (AAAI 2024 + reference code) |
| **L4 — contract** | output shape, finiteness, determinism in `eval` mode, gradient flow | every model |

A model can only be verified at the level its source allows. A paper without a reference
implementation and without a local copy of the PDF (ASCENT, CFM, DDPM, the attention-LSTM)
is verified at L2 and L4: that *bounds* the claim we can make about it, and the table below
says so explicitly.

**What "verified" does not mean.** L1–L4 establish that the implementation *is what it
claims to be*, and that it is numerically sound. They do not establish that the
re-implementation reaches the accuracy of the published numbers: that depends on the data,
on the split and on the compute budget, and it is discussed in `docs/PAPER.md` §6.7 and
§9.2.

---

## 2. The catalogue

### 2.1 The project model

| | |
|---|---|
| **AeroTraj-Ultra** | `model.py` · `AeroTrajUltra` |
| source | this project (FlightBERT++ V2 lineage: `modello.txt`, <https://github.com/gdy-scu/FlightBERT_PP_V2>) |
| parameters | **1 276 983** (5 s grid, 13 horizons) · **1 277 706** (10 s grid, 6 horizons) |
| input | 12 points × 9 channels (6 state + wind `u`, wind `v`, temperature) |
| output | `H × 6`, non-autoregressive, one forward pass |
| specification | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| verification | L2 + L4 (frames, Gray-Code codec, hybrid loss, output contract) |

The model is specified equation by equation in `ARCHITECTURE.md`; the conformance suite
checks the parts that can be checked in isolation: the aircraft-centric frame is the
identity to machine precision, the Gray-Code encoder/decoder round-trips, the
straight-through estimator carries a non-zero gradient, the hybrid loss is finite and its
kinematic term is zero at a constant-velocity equilibrium.

| | |
|---|---|
| **AeroTraj-Atlas** | `model.py` · `AeroTrajUltra` with `model.airspace_field` + `AirspaceField` |
| source | this project; Instant-NGP (Müller, Evans, Schied & Keller, ACM Trans. Graph. 41(4), 2022) for the multi-resolution hash grid; MTR (Shi, Jiang, Dai & Schiele, NeurIPS 2022) for the map as the source of intent and the dynamic collection along a proposed trajectory |
| parameters | **10 495 863** (5 s grid, 13 horizons) = network **1 361 655** (AeroTraj-Ultra 1 276 983 + field MLP 35 424 + fourth block of `horizon_proj` 20 736 + refine MLP 28 512) + map **9 134 208** |
| input | 12 points × 9 channels, as AeroTraj-Ultra; the map is read at positions derived from them (the last observation, the kinematic prior, the proposed trajectory) |
| output | `H × 6`, non-autoregressive, one forward pass |
| specification | [`ARCHITECTURE.md`](ARCHITECTURE.md) §15 |
| verification | L2 + L4 (section 10 of the conformance suite): with the options off the parameters are those of AeroTraj-Ultra; the interpolation weights sum to one, also outside the region; the heading axis is periodic; a grid corner reads its own table row; the aircraft-frame head is zero at initialisation and a pure along/cross-track residual moves the point along/cross-track only |

AeroTraj-Atlas is not a new class: it is `AeroTrajUltra` with the config key
`model.airspace_field`, a dict of `AirspaceField` keyword arguments plus the switches `ahead`
(default true) and `refine` (default false); the final setting is
`{"n_alt": 5, "refine": true}`. The field is a learned map of the airspace — 6 levels over
longitude and latitude, a periodic heading axis of 8 sectors and an altitude axis of 5 nodes,
8 features per node, dense tables for the coarse levels and a 2^18-entry hash for the fine
ones — read at the aircraft, along the kinematic prior and, with `refine`, along a trajectory
proposed by the first decoder layer; the reads enter the horizon queries as a fourth block.
Without the key the model is AeroTraj-Ultra bit for bit (identical state dict and identical
outputs in eval and train mode; every existing checkpoint loads). `refine` together with
`n_modes > 0` raises an error: the combination is not implemented. The field has not been
combined with the retrieval variant.

**Two options implemented, measured and not adopted** (they stay in the code as documented
ablations, benchmark budget, seed 97): `model.track_frame` — the history also given in the
aircraft frame (along-track / cross-track) plus a second residual head working in that frame
as a fraction of the distance flown — is +1.6 % alone and gives no gain with the field
(902.4 vs 903.1 m); `n_vz` — a vertical-rate axis of the field, to separate departures from
arrivals — gives 868.1 vs 865.2 m (+0.3 %) with more parameters and 18 % more time.

### 2.2 The literature re-implemented for the 365-day / 10 s benchmark

These live in `bench_baselines.py` and are driven by `bench_run_10s.py`.

#### A3 — Constant-velocity Kalman filter

| | |
|---|---|
| class | `KalmanFilterBaseline` |
| source | FlightBERT paper (Guo et al., IEEE T-ITS 24(2):1828–1834, 2023), group A / A3: *"the representative model-driven methods in the TP task are based on the Kalman Filter"* |
| state | `[p(3) m, v(3) m/s]` in the aircraft-centric local frame |
| model | CV transition, continuous white-noise acceleration `Q`, position measurement, **Joseph-form** covariance update |
| parameters | **6**: per-axis process noise `q` and measurement noise `R`, learned |
| verification | **L1**: matches an independent NumPy implementation of the same filter to **1e-2 m** over all six horizons; **L2**: the covariance stays positive semi-definite (smallest eigenvalue 0.8 after the full window) |

**Deviation, declared.** The paper's Kalman baseline is tuned by hand; ours learns `q` and
`R` by gradient descent. The reason is in the paper §6.7: a hand-tuned filter would be
handicapped by the tuning effort, not by its method.

**Bug found by this verification.** The covariance propagation used `F P F` instead of
`F P Fᵀ`. The transition matrix is not symmetric, so the covariance became indefinite
(smallest eigenvalue −5·10²) within one window: the gain was wrong and the filter was
silently sub-optimal. Found by the PSD check (L2), fixed, and the model retrained.

#### A4 — Interacting Multiple Model filter

| | |
|---|---|
| class | `ClassicalIMMBaseline` |
| source | Blom & Bar-Shalom (1988); the IMM-Informer hybrid, Li et al., *Sensors* 25(8):2531, 2025 |
| modes | constant velocity · constant acceleration · coordinated turn (horizontal) + CV (vertical) |
| model | per-mode Kalman filters, Gaussian-mixture mode-probability update, standard interaction (mixing) step, exact Joseph-form updates |
| parameters | **25**: per-mode process noise (3 × 3), measurement noise (3), turn rate (1), mode-transition matrix (9), initial prior (3), learned |
| verification | **L1**: pinned to a single mode it reproduces the NumPy filter **exactly** (0.0 m difference) and its transition matrix is row-stochastic; **L2**: the predict step keeps the covariance PSD; the turn rate respects its bound |

**Deviation, declared.** The noise model, the transition matrix and the turn rate are
learned (same rationale as the Kalman filter). The coordinated-turn rate is bounded to
±0.05 rad/s; on the benchmark it saturates that bound — a documented weakness of the mode.

#### A1 — LSTM multi-horizon regression

| | |
|---|---|
| class | `LSTMRegressionBaseline` (`model.py`) |
| source | FlightBERT paper, group A / A1 ("deep LSTM regression") |
| architecture | 3 LSTM layers × 128 hidden, dropout 0.1, head `128 → 256 → H×6`, horizon-scaled displacement from the last observed point |
| parameters | **376 100** |
| verification | L4 (shape, finiteness, determinism, gradient) |

The paper does not state the depth; three layers is the "deep" reading of A1.

#### A2 — BiLSTM multi-horizon regression

| | |
|---|---|
| class | `BiLSTMRegressionBaseline` (`bench_baselines.py`) |
| source | FlightBERT paper, group A / A2 |
| architecture | 2 bidirectional LSTM layers × 192 hidden per direction, dropout 0.1, head `384 → 384 → H×6` |
| parameters | **1 356 708** |
| verification | **L2**: the sequence summary is the *forward state at the last step* and the *backward state at the first step* |

**Bug found by this verification.** The implementation pooled `out[:, -1, :]`, i.e. the
backward state of the **last** step, which has seen one element of the sequence only: the
backward direction contributed almost nothing. The correct summary of a bidirectional
recurrent network is `[h_fwd(T−1) ‖ h_bwd(0)]`. Fixed, checked to machine precision, and
the model retrained.

#### B0 — FlightBERT (IEEE T-ITS 2023)

| | |
|---|---|
| class | `FlightBERTBaseline` + `BinaryEncodingCodec` + `flightbert_hc_loss` |
| source | Guo, Zhang, Yan, Zhang & Lin, *FlightBERT: Binary Encoding Representation for Flight Trajectory Prediction*, IEEE T-ITS 24(2):1828–1834, 2023 (PDF in the repository root) |
| representation | plain binary encoding (not Gray), 19/18/11/12/12/12 bits for `lon, lat, alt, vx, vy, vz`; velocities carry a sign bit in the highest position |
| trajectory embedding | one attribute-embedding block per attribute (64 dims) + the ACoAtt recalibration, concatenated to **768** |
| backbone | **8** Transformer decoder blocks, masked self-attention, **4** heads |
| head | `768 → 84` linear + sigmoid (multi-binary classification), straight-through decoding |
| loss | Hybrid Constrained: BCE on the bits + MSE on the decoded value |
| training protocol | next step only, then the paper's **autoregressive roll-out** for the multi-horizon prediction |
| parameters | **57 096 276** |
| verification | **L3** (8 blocks, 4 heads, 64-dim attributes, 768-dim embedding, the bit widths) + L2 (codec round-trip, STE gradient) |

**Deviations, declared.**

1. **Bit widths.** The paper gives 18/16 bits for longitude/latitude at a 0.001°
   resolution; those widths *cannot* represent ±180°/±90° (18 bits hold 262 144 values, and
   ±180° at 0.001° needs 360 001). We use 19/18. For the velocities the paper's 11 bits
   *include* the sign; our field is 1 sign + 11 magnitude = 12.
2. **Loss balance.** The paper's Eq. 7 is an unweighted sum of the MSE and the BCE. That
   sum does not work: a binary code is redundant, so the regression term minimises itself
   by trading one bit against another and destroys the high bits, which carry tens of
   degrees each. Measured: with the unweighted sum the second bit of the latitude is
   correct in 13 % of the windows and the one-step error is 1.1·10⁷ m; with `w_mse = 0.1`
   the same bit is correct in 100 % and the error is 6.4·10³ m. See `test_loss_balance.py`.
3. **ACoAtt along time.** Eq. 5 of the paper applies `softmax(QKᵀ/√d)V` to the two
   correlated attribute embeddings. Applied point by point the softmax degenerates to a
   constant (a sequence of length one); we apply the correlation along the *time* axis,
   which is the only reading in which the attention does something.

#### B1 — FlightBERT++ V2 (AAAI 2024), reference configuration

| | |
|---|---|
| class | `FlightBERT_PP_V2_Baseline` (`model.py`) |
| source | Guo et al., *A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework with Gray Code Representation*, AAAI 2024, and the authors' reference implementation `gdy-scu/FlightBERT_PP_V2` (`model.py`, `config.json`) |
| representation | **Gray-Code** encoding, absolute state for the encoder and **step deltas** for the decoder |
| encoder | one conv stem over the concatenated bit fields → `n_embd`, `n_en_layer` Transformer encoder layers, attention pooling |
| decoder | horizon embedding + context query, the delta tokens as a prefix, `n_de_layer` Transformer decoder layers |
| head | `n_embd → total_delta_bits`, no bias, sigmoid, straight-through decode and cumulative sum |
| verification | **L3** against the reference `config.json` (see the table below) + L4 |

**Reference configuration vs ours** (the reference: `gdy-scu/FlightBERT_PP_V2/config.json`):

| item | reference | ours | verdict |
|---|---|---|---|
| absolute bits | 18/16/11/11/11/11 | 19/18/11/12/12/12 | **adapted**: the paper's widths cannot hold ±180° at 0.001°, and the sign needs its own bit |
| delta bits | 8/8/8/9/9/6 | 11/11/10/9/9/8 | **adapted** to the 10 s grid: the step displacements double, the lattice is widened, verified saturation 0.21 % (`verify_10s.py`) |
| `n_head` | 4 | 4 | matches |
| `n_en_layer` | 4 | 2 | **deviation**: compute-matched to the shared budget (a 4-layer, 768-wide model costs ~4 min/epoch on one RTX 4070) |
| `n_de_layer` | 2 | 2 | matches |
| `n_embd` | 768 | 128 | **deviation**: compute-matched, same reason |
| `inp_seq_len` | 9 | 12 | **protocol**: ours is a 120 s window at 10 s |
| `horizon` | 15 | 6 | **protocol**: ours is 10/30/60/90/180/300 s |
| learning rate | 1e-4 | 1e-3 | **deviation, measured**: the reference trains at 1e-4 for 50 epochs; at our budget of 25 epochs that rate does not converge (validation 1 901.3 m against 1 502.4 m at 1e-3). The schedule is ours and is declared |
| dropout | 0.1 | 0.1 | matches |
| `use_graycode` | true | true | matches |
| head bias | false | false | matches |

`n_embd = 128` with 2 + 2 layers gives **763 201** parameters.  The reference configuration
(768 wide, 4 encoder + 2 decoder layers, 48 delta bits) is ≈ **35 M**: roughly 45 times
larger.  This is a compute-matched reproduction, and every number it produces must be read
with that in mind — a wider network on the same data and budget would very likely do
better, which is exactly why the comparison is stated as "same budget" rather than "same
accuracy as the paper".

#### B2 — FlightBERT++ V2 with the ACoAtt embedding (ablation)

| | |
|---|---|
| class | `FlightBERTPPV2Faithful` |
| what it is | the delta decoder of FlightBERT++ **plus** the attribute-correlation embedding of the 2023 paper |
| parameters | **936 705** |
| verification | L3 (six bidirectional CoAtt blocks) + L4 |

**Honest labelling.** The reference implementation published with the AAAI 2024 paper does
**not** contain the ACoAtt block: it joins the six Gray-Code fields in a single conv stem.
The ACoAtt is the block of the *2023* paper. This class is therefore not "the 2024
network"; it is the ablation that measures what the block is worth on the delta decoder.

**A confound found by this verification.** The first version of this ablation used
`d_model = 192` against the 128 of B1, so the measured difference mixed the ACoAtt with a
50 % wider model. Both are now 128 wide, 2 + 2 layers deep and trained with the same
schedule, so the only difference between B1 and B2 is the ACoAtt module. Both are
retrained.

**The schedule, declared.** Every row of the benchmark is trained with the same protocol —
AdamW, OneCycle at 1e-3, 25 epochs of 400 k windows — because a comparison is only
meaningful at a common budget. The reference implementation of FlightBERT++ uses 1e-4 for
50 epochs; at half that number of epochs the 1e-4 run does not converge (validation
1 901.7 m against 1 502.4 m). The deviation is therefore the schedule, not the
architecture, and it is measured rather than assumed.

#### B3 — the same, MSE-only loss (negative control)

`fbpp_v2_faithful_mse` is B2 with the BCE removed from the loss. It exists to quantify the
failure mode described in `docs/PAPER.md` §9.2: a redundant code optimised by a regression
term only. It is an ablation, not a competitor.

### 2.3 The classical references (no learning)

| model | class | parameters | role |
|---|---|---:|---|
| CV (Doppler) | `ConstantVelocityKalmanBaseline` | 0 | extrapolates the last observed Doppler velocity |
| CV least-squares | `SmoothedConstantVelocityBaseline` | 0 | least-squares line fitted on the 12 observed points, anchored at the last position |

Both are in `model.py` and take the observation step from the config (`dt_sec`), not from a
hard-coded 5 s.

### 2.4 The models of the 90-day benchmark

These live in `bench_models.py` and were used for the benchmark of `docs/PAPER.md` §6.1.
None of them has a local copy of its paper, so their conformance is L2 + L4 only.

| model | class | parameters | architecture |
|---|---|---:|---|
| ASCENT (ICRA 2026) | `ASCENT` | 766 355 | mode queries (K = 8) + winner-takes-all, speed / turn-rate / vertical-rate integration |
| CFM (FlowATC-style) | `CFMTrajectoryModel` | 1 124 867 | conditional flow matching, 20 Euler steps |
| DDPM (MID-style) | `DDPMTrajectoryModel` | 1 124 867 | denoising diffusion, 50 sampling steps, the same backbone |
| Attention-LSTM, phase-aware | `AttentionLSTMPhase` | 718 309 | LSTM + attention over the observed window, flight-phase features |
| IMM + neural residual | `IMMResidualModel` | 384 996 | the three-mode IMM above + a GRU corrector on the innovation sequence |

**Bug found by this verification.** `IMMResidualModel` hard-coded the observation step to
5 s inside its mode propagation. On the 10 s grid every displacement would have been
doubled. The step is now a constructor argument (`obs_dt`), fed from the config, and
`bench_run.py` passes it.

---

## 3. Summary table

| model | source | params | verification | status |
|---|---|---:|---|---|
| AeroTraj-Ultra | this project | 1 277 706 | L2 + L4 | conforms to `ARCHITECTURE.md` |
| AeroTraj-Atlas | this project + Instant-NGP + MTR | 10 495 863 | L2 + L4 | conforms to `ARCHITECTURE.md` §15 |
| Kalman filter (CV) | FlightBERT A3 | 6 | L1 + L2 | conforms (learned noise declared) |
| IMM | Blom & Bar-Shalom 1988 | 25 | L1 + L2 | conforms (learned noise declared) |
| LSTM regression | FlightBERT A1 | 376 100 | L4 | conforms |
| BiLSTM regression | FlightBERT A2 | 1 356 708 | L2 + L4 | **conformed after a pooling fix** |
| FlightBERT | IEEE T-ITS 2023 | 57 096 276 | L3 + L2 | conforms with 3 declared deviations |
| FlightBERT++ V2 | AAAI 2024 + reference code | 763 201 | L3 | compute-matched, 2 declared deviations |
| FlightBERT++ + ACoAtt | ablation (2023 block) | 936 705 | L3 + L4 | ablation, not a published network |
| FlightBERT++ MSE-only | negative control | 936 705 | L4 | ablation |
| CV / CV-LSQ | classical | 0 | L2 | conforms |
| ASCENT, CFM, DDPM, Attn-LSTM, IMM-res | `bench_models.py` | 0.4–1.1 M | L2 + L4 | structurally verified, no external spec available |

---

## 4. Defects found and fixed by this verification

| # | defect | where | how it was found | impact |
|---|---|---|---|---|
| 1 | the covariance was propagated as `F P F` instead of `F P Fᵀ` | `KalmanFilterBaseline` | the PSD check (L2) | the covariance went indefinite within a window; the gain, and therefore the filter, was silently wrong |
| 2 | the BiLSTM pooled the backward state of the last step | `BiLSTMRegressionBaseline` | the summary identity (L2) | the backward direction contributed one element; fixed and retrained |
| 3 | the observation step was hard-coded to 5 s | `IMMResidualModel` | a grep for `dt = 5.0` | on the 10 s grid every displacement would be doubled |
| 4 | a dataset on disk was ignored when OpenSky was unreachable | `data_pipeline.build_dataset_splits` | the retraining run failed with an S3 `AccessDenied` | no offline reproducibility; now the local dataset is validated first |
| 5 | the ACoAtt ablation changed the model width too | `FlightBERTPPV2Faithful` | the spec table (L3) | the measured effect mixed two causes: it read −9.2 % with the confound and −1.8 % once both variants were 128 wide |
| 6 | the FlightBERT++ trained at 1e-3 against the reference's 1e-4 | `bench_run_10s.LR_OVERRIDE` | the spec table (L3) | a declared protocol deviation, now corrected |
| 7 | the delta decoder accepted a wrong number of horizon steps with a cryptic error | `GrayCodeCodec.decode_horizon_deltas_ste` | the API test | a latent misuse hazard; the shape is now asserted |
| 8 | the normalisation statistics were computed as the mean of a float32 array along axis 0, whose float32 running sum stops growing | `data_pipeline.py` | the stored latitude mean (4.58° for a true 48.85° on the 365-day set) was read while preparing a realistic input for the latency benchmark | every dataset had its own wrong constants (also altitude, temperature, std of `vx`); fixed with `exact_norm_stats` (float64, by chunks) and `--exact-norm-stats`, selected by `data.norm_stats`. Accuracy effect none (+0.38 % Ultra, −0.32 % Atlas, inside the seed spread), so the stored statistics stay the default |
| 9 | strict load of the checkpoints older than the `dt` parameterisation (two codec buffers missing) | `eval_ckpt.py` | the example command of the README failed | those checkpoints could not be evaluated; they now load, and unexpected keys still fail |
| 10 | the evaluation tools did not check the window length | `eval_cross.py`, `eval_ckpt.py` | a code audit before the 24-point experiment | a 12-point checkpoint evaluated with a 24-point config silently received 24 points; each model now gets the last `inp_seq_len` points |
| 11 | models with different window lengths were compared on different windows | the window study (`eval_cross.py --split`) | the 24-point A/B gave −6.4 % in validation and −0.28 % on identical test windows | the earlier 10 s-grid result (−7.3 % at 300 s) was a test-set artefact; models are now compared on identical windows |

---

## 5. What is *not* verified

* **The accuracy of the re-implementations against the published numbers.** Different
  data, different split, different compute budget. `docs/PAPER.md` §6.7 reports our numbers
  and §9.2 the negative results.
* **ASCENT, CFM, DDPM and the attention-LSTM against their papers.** No local copy of the
  PDFs and no reference code: only the internal consistency and the contract are checked.
* **The generative models' sampling distribution.** CFM and DDPM are evaluated as
  single-sample predictors; whether their distribution is calibrated is a different question
  and is not addressed here.
* **The retrieval variant** (`AeroTrajUltraRetrieval`). It is documented in
  `ARCHITECTURE.md` §10 and evaluated in the paper; the conformance suite does not cover it
  because it needs a retrieval pool built from the dataset.

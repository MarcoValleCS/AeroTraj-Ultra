# AeroTraj-Ultra: physics-grounded, non-autoregressive 4-D flight-trajectory prediction with Gray-Code delta decoding, a differentiable trust region and a retrieval intent prior

**Technical report** · Marco Valle · AeroTraj-Ultra project
Code: <https://github.com/MarcoValleCS/AeroTraj-Ultra>

---

## Abstract

We present **AeroTraj-Ultra**, a non-autoregressive neural predictor of aircraft 4-D
trajectories on horizons from **5 s to 300 s**, trained end-to-end on **90 days of real
OpenSky ADS-B data** (72/9/9 chronological split). The model departs from the
binary-encoding lineage of FlightBERT/ FlightBERT++ in three ways. First, it replaces the
pure classification decoder with a **hybrid decoder**: a Gray-Code multi-binary head, a
continuous residual head anchored on a closed-form **physics propagator**, and a
differentiable **trust-region gate** that interpolates between them with a bounded `tanh`
correction. Second, it reformulates the sequential soft-XOR Gray-Code decoding as a
**prefix parity** `cumprod`, which is algebraically identical in value *and* gradient but
removes ~330 kernel launches per forward pass and yields a 1.9× end-to-end training
speed-up. Third, it adds two compositional heads: a **multi-hypothesis head** (`K = 8`
learnable mode queries, winner-takes-all loss) that estimates epistemic intent uncertainty
*without* a generative model, and a **retrieval-conditioned prior** that conditions the
horizon queries on the `K` most similar past trajectories, retrieved on GPU from a pool of
342,572 candidate windows.

On the held-out 9-day test split, over the horizon grid 5–300 s and measured with a 3-D
ECEF mean distance error (MDE), AeroTraj-Ultra reaches **991.6 m**, against **1,219.9 m**
for a FlightBERT++ V2 re-implementation, **1,296.4 m** for a deep multi-horizon LSTM,
**1,507.7 m** for a constant-velocity Doppler baseline, **1,704.3 m** for a conditional
flow-matching model, and **4,300.5 m** for a DDPM diffusion model — all trained and
evaluated under an identical protocol. Adding the retrieval prior and the
multi-hypothesis head together gives **957.2 m (−3.5 %)** and an oracle `minADE@8` of
**608.2 m**. Uncertainty is certified rather than estimated: a Mondrian (flight-phase
stratified) conformal calibration yields **exact, horizon-uniform coverage** with a mean
radius of **1,190 m at 90 %** nominal, against **2,675 m** for the inflated single
prediction — a 56 % narrower certified band. We also report a **negative result**: no
selection rule over the `K` hypotheses beats the deterministic output, so the
multi-hypothesis head is an uncertainty estimator, not a better point predictor.
Inference is **launch-bound, not compute-bound**: a single-aircraft prediction costs
**19.6 ms** end-to-end (20.1 ms including the retrieval prior), which is already ~50× inside
a 1 s radar update cycle, while batched inference sustains **57,702 aircraft/s** at batch
4,096; `torch.compile` cuts the single-aircraft latency to 4.47 ms with no change in the
mathematics, and predicting 13 horizons spanning 5–300 s costs only 1.17× predicting a
single one.

---

## 1. Introduction

Trajectory prediction for air traffic management must answer a *sequence* of questions, not
one. At 5–30 s the answer is dominated by sensor kinematics and the retained error is
essentially the CPR quantisation noise of the position itself. At 60–300 s the
observable state no longer determines the future: two aircraft flying identical
straight-and-level trajectories at 300 s can differ by tens of kilometres because of a
*clearance* or an *intent* that is never in the window. Any model that claims a single
accuracy number therefore hides the fact that it is solving two different problems.

This report documents a model, **AeroTraj-Ultra**, and the evidence collected around it.
The contributions are:

1. **A hybrid Gray-Code/continuous decoder with a physics anchor** (§4). The Gray-Code
   branch keeps the inductive bias of the binary-encoding lineage (a 10× finer delta
   lattice removes the high-order bit-flip outliers of FlightBERT); the continuous branch
   is a *residual on a closed-form propagator*; a differentiable gate interpolates them
   with a `tanh` trust region. At initialisation the model is *exactly* the physics
   solution and only learns a correction.
2. **Prefix-parity decoding of the Gray code** (§4.3). We show that the classical
   sequential soft-XOR recurrence equals the prefix parity of the bit probabilities,
   `b_i = ½(1 − ∏_{j≤i}(1 − 2g_j))`, identical in value and gradient (numerical
   deviation 2e-16) but expressible as one `cumprod`. Combined with the removal of six
   per-step GPU→CPU synchronisations, this takes the training step from 144 ms to
   **75 ms** at batch 2,048 on an RTX 4070 (a 1.9× speed-up).
3. **A multi-hypothesis head as an uncertainty device** (§4.9, §7). `K = 8` mode queries
   with a float attention mask that leaves the deterministic pathway *bit-identical*,
   trained with winner-takes-all plus an auxiliary term. It improves the point estimate
   marginally (991.6 → 984.9 m) but its real product is the conformal band: on the
   validation split we obtain exact coverage with a radius 56 % smaller than inflating the
   single prediction.
4. **A retrieval-conditioned intent prior** (§4.10) with explicit leakage control: a
   non-parametric prior over 342,572 past windows, queried on the GPU, whose context is
   decoded by a learned attention over the neighbours. The gain is monotone in the
   horizon (−0.3 % at 5 s, −5.0 % at 300 s) — the signature of an intent prior rather than
   a kinematic one.
5. **A like-for-like benchmark against ten alternatives** (§6), including the
   **previous model of this project** (FlightPREV, §6.5) re-implemented on the same data,
   plus a **measured negative result** on hypothesis selection (§9.2).
6. **A measured inference-latency characterisation** (§8), because accuracy is
   meaningless in ATM if the prediction is not available before the update cycle closes.

Throughout, every number is traceable to a script in the repository and to the lab
notebook [`EXPERIMENTS.md`](EXPERIMENTS.md); the formal module-by-module specification is
in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 2. Related work

**Binary-encoding trajectory prediction.** FlightBERT (Guo et al., *IEEE TITS* 24, 2023)
introduced attribute correlation attention and a Gray/binary encoding of the trajectory
with a Hybrid Constrained Loss; FlightBERT++ (Guo et al., *AAAI* 2024) made the decoder
non-autoregressive over multiple horizons and encoded *step deltas* rather than absolute
states. AeroTraj-Ultra keeps that lineage's Delta-Gray-Code idea, refines the delta lattice
by 10×, and adds a continuous residual branch and a physics anchor that the original lacks.

**Physics and hybrid kinematic models.** The classical short-term literature — constant
velocity, constant acceleration, coordinated turn, and Interacting Multiple Model (IMM)
filters (Blom & Bar-Shalom, 1988) — remains competitive at very short horizons, and the
2026 reviews place hybrid physics+ML as the most promising direction. Our physics prior
(§4.7) is a damped coordinated-turn propagator with exactly **three** learned scalars
(τ_turn, τ_vz, τ_acc); everything else is parameter-free, which is what makes it usable as
an interpretable reference and as the residual anchor.

**Generative and multimodal models.** Conditional flow matching on a DiT backbone
(FlowATC, 2026), DDPM/MID (Gu et al., *CVPR* 2022) and ASCENT (Prutsch et al., *ICRA*
2026) all model the multimodality of intent by sampling or by mode queries. We reproduce
ASCENT, CFM and DDPM in §6 and show the recurring caveat: the *single-draw* error of the
generative models is far worse than the deterministic baselines, and their apparent parity
(`minADE@10` of CFM = 983.5 m) is an oracle that requires knowing which sample is correct.

**Retrieval-augmented prediction.** Nearest-neighbour trajectory prediction is classical in
this domain (analog forecasting), but had fallen out of favour with end-to-end deep models.
We show (§4.10) that a *conditioned* retrieval, decoded by attention rather than averaged,
recovers a medium-term intent prior at negligible cost — and that the naive version
(averaging the neighbours, or training from scratch without normalising the context) is
**24 % worse** than the base model.

**Trust-region gates.** MAF-Net (Pang et al., *AAMAS* 2026) proposes a differentiable
multi-head action filter as a trust region. §8 of [`ARCHITECTURE.md`](ARCHITECTURE.md)
adapts it as a `tanh`-bounded gate between the Gray-Code and continuous branches.

---

## 3. Data

**Source.** Real ADS-B state vectors from The OpenSky Network, distributed as public
parquet on `s3.opensky-network.org/competition-data` and read through `pyopensky`.

**Dataset.** **90 consecutive days** from 2022-01-01, restricted to the European bounding
box `lon ∈ [−15°, 35°]`, `lat ∈ [30°, 65°]`, `alt ∈ [−500 ft, 50,000 ft]`, with a minimum
segment length of 140 steps. The split is **chronological and disjoint**: 72 training
days, 9 validation days, 9 test days. The test set is 2022-03-23 → 2022-03-31 (60,000
sliding windows for the benchmark; 9 days), and no test day is ever seen in training.

**Resampling and features.** Each flight is resampled to a uniform **Δ = 5 s** grid. From
the raw state vector we build **9 channels** per step:

| idx | symbol | unit | definition |
|---|---|---|---|
| 0 | λ | deg WGS-84 | longitude |
| 1 | φ | deg WGS-84 | latitude |
| 2 | h | 10 m | barometric altitude (`ft × 0.03048`) |
| 3 | v_x | km/h | eastward ground speed (`gs · sin ψ`) |
| 4 | v_y | km/h | northward ground speed (`gs · cos ψ`) |
| 5 | v_z | m/s | vertical rate (`vr × 0.00508`) |
| 6 | u | m/s | eastward wind |
| 7 | v | m/s | northward wind |
| 8 | T | K | static temperature |

`(v_x, v_y)` are the **onboard Doppler** velocities, not finite differences of the
position: at these altitudes a 60 s difference quotient of CPR-quantised positions carries
a ~12 m/s error, which would destroy the only reliable kinematic signal. Normalisation
statistics (μ, σ) are computed **on the training split only** and travel inside the
checkpoint as registered buffers.

**Window and horizons.** Input window `T = 12` steps, i.e. 60 s of history. Output: **13
non-uniform horizons**, `h = {5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300}` s.
The whole horizon vector is produced by a **single forward pass**; there is no
autoregressive rollout, so no error accumulation and a latency independent of the horizon.

**Metric.** All models are compared with the **3-D ECEF mean distance error (MDE)**: the
prediction and the truth are converted to earth-centred earth-fixed Cartesian coordinates
and the Euclidean distance is averaged over the horizon grid and the test windows. The
conversion is applied identically to every model, so the numbers are directly comparable
across architectures and coordinate parameterisations.

### 3.1 Why a 120-second observation window (the coarse-grid experiment)

The model described in this report reads **60 s** of history (12 points at 5 s). A second
configuration, `config_365d_10s.json`, reads **120 s** (12 points at 10 s) and predicts
**{10, 30, 60, 90, 180, 300} s** instead of 13 horizons. The two are trained on the same 365
days with the same recipe, and every horizon of the coarse grid also exists in the fine grid,
so the comparison is per horizon on identical test days.

**The argument for the longer window.** The error profile measured in §6.6 and the composition
of the gains in §6.4 and §6.3 agree on one thing: above ~30 s the residual error is **not**
sensor noise, it is *how much of the intent is legible in the input*. Three independent
measurements carry the same signature, all monotone in the horizon:

| lever that adds information about intent | gain at 5 s | gain at 300 s |
|---|---:|---:|
| retrieval prior over past trajectories (external) | −0.3 % | −5.0 % |
| 4.5× more training data (implicit) | −4.9 % | −15.3 % |
| **more history in the window (this experiment)** | *to be measured* | *to be measured* |

If the long-horizon error is governed by how much of the manoeuvre is visible, then a window
covering **twice as much of the trajectory** attacks it directly. Concretely, what becomes
legible at 120 s that is not at 60 s:

* **A whole manoeuvre rather than a fragment.** A turn that began 40 s ago appears in a 60 s
  window as an almost straight track with slight curvature; over 120 s the entry heading, the
  turn and the exit heading are all inside the window, which is precisely the information a
  damped coordinated turn needs (§4.7).
* **A stationary turn-rate estimate.** The prior's `ω0` comes from a 3-step heading difference
  — 15 s at 5 s spacing, 30 s at 10 s. A longer baseline makes that estimate less sensitive to
  single-sample track noise, which matters because the propagator assumes a *constant* turn
  rate over the horizon.
* **Flight phase and vertical trend.** A climb that started 90 s ago is invisible in a 60 s
  window; in a 120 s one the altitude change is twice as large relative to the same
  quantisation, so cruise / climb / descent are far better separated.
* **Leg context.** The heading trend over 120 s distinguishes "established on a leg" from
  "turning onto one", which is exactly the disambiguation the retrieval module tries to make
  from other flights' history (§4.10) — here it is made from the aircraft's own.

**What it costs, stated plainly.** The sampling is coarser by construction: with the same 12
points, a step-to-step displacement doubles, so (i) short-term *accelerations* are less
visible to the encoder, (ii) the Gray-Code delta lattices had to be widened (§3.2), and
(iii) the 5 s output horizon no longer exists, so the 5–30 s band — the band we know is
saturated by CPR quantisation anyway — can only be judged at 10 s. The experiment therefore
tests a *joint* change (longer history **and** coarser sampling), and the verdict will be read
per horizon: if the extra history is what helps, the gain must **grow with the horizon**, the
opposite of a sampling artefact.

**Disentangling the two effects** would need a 24-point window at 5 s (same resolution, double
history); that is the natural follow-up if this experiment is positive.

### 3.2 Grid bookkeeping: how the time step becomes a parameter

The horizon grid is declared in **seconds** and converted to sequence indices using
`data.dt_seconds`. That conversion now lives in one place
(`config_utils.resolve_horizon_steps`), which also *verifies* the declared index arrays against
`horizon_seconds / dt`. Before this, consumers read an array named `horizon_steps_5s`, and two
normalisations in the loss divided those indices by 60 and 12 — which is only dt-independent if
the reference scales are carried along (they equal `seconds/300` and `seconds/60`). Moving to a
10 s grid without noticing that would have silently halved the weight of every horizon. With
the derivation centralised, all existing configurations resolve to exactly their declared
arrays, so the change is a strict generalisation.

Widening the delta lattices is the other consequence: a coarser step doubles every per-step
displacement, so `delta_lon_size` and `delta_lat_size` go 10 → 11 bits and `delta_alt_size`
9 → 10. The widths were chosen by **measuring** the requirement rather than by reasoning
(`test_codec_10s.py`): longitude needs 800 of 1023 (1.28× margin, the requirement being the
pipeline's own 0.04° per 5 s rejection tolerance scaled to the step), and altitude needs 250 of
255 at 9 bits — no margin at all for a 25 m/s climb — hence 10 bits (2.04×).

---

## 4. Method

### 4.1 Overview

AeroTraj-Ultra is an encoder–decoder with three output paths that are fused before the
loss:

```
x (B×12×9)
  ├─ Gray-Code codec ──► ACoAtt-V2 attribute embeddings ──► encoder ──► context c
  │                                                                    │
  │                                     ┌──────────────────────────────┤
  │                                     ▼                              ▼
  │                         decoder (prefix + H horizon queries)   physics prior P
  │                                     │                              │
  │             ┌───────────────────────┼──────────────────────────┐   │
  │             ▼                       ▼                          ▼   ▼
  │      Head A: GC bits ──►  ŷ_gc   Head B: residual ──► ŷ_cont = P + δ
  │                             └──────────┬───────────────┘
  │                                        ▼
  │                        Head C: MAF gate  ŷ = ŷ_cont + α ⊙ tanh((ŷ_gc−ŷ_cont)/σ_h) ⊙ σ_h
  │                                        │
  │                          (optional) multi-hypothesis head: K=8, WTA loss
  │                          (optional) retrieval context on the horizon queries
  ▼
loss = 2.5·L_pos + 0.6·L_vel + 0.4·L_BCE + 0.25·L_STE + 0.25·L_kin   (+ 0.5·L_hyp)
```

Every block is specified formally in [`ARCHITECTURE.md`](ARCHITECTURE.md) §2–§11.

![AeroTraj-Ultra architecture overview: Gray-Code encoder and ACoAtt-V2 attention, pre-LN Transformer encoder, non-autoregressive multi-horizon decoder, physics kinematic prior, the three output heads and the MAF-Net trust-region fusion](figures/architecture_overview.png)

**Figure 4.1** — Architecture overview. The encoder spine (Gray-Code → ACoAtt-V2 → conv
stem → pre-LN Transformer → attention pooling) produces the context `c`; the
non-autoregressive decoder emits the whole horizon vector `D` in one pass; the physics
prior `P` anchors both the continuous residual head and every multi-hypothesis mode; the
MAF-Net gate bounds the influence of the coarser Gray-Code branch. Reproduce with
`plot_architecture.py`. The rest
of this section reports the parts that carry the scientific claims.

### 4.2 Why the model is *not* purely generative

The dominant residual error at 60–300 s is **epistemic**: the manoeuvre the aircraft will
fly is not in the window. A generative model models that by sampling, at the cost of a
degraded single-sample error (measured: CFM 1,704.3 m single-draw vs 983.5 m best-of-10).
We instead (i) make the *deterministic* path as strong as possible with a physics anchor,
and (ii) expose the epistemic part as a **small set of explicitly scored hypotheses** plus
a **certified interval**, keeping the operational output single-valued and cheap.

### 4.3 Gray-Code codec and prefix-parity decoding

**Lattices.** Absolute states are encoded on a Gray-Code lattice (λ 19 bits at 0.001°,
φ 18 bits, h 12 bits at 10 m, v_x/v_y 12 bits at 1 km/h, v_z 12 bits at 0.1 m/s → 85 bits
per window). The decoder predicts **step deltas** on a 10× finer lattice: Δλ, Δφ 10 bits
at scale 10,000 (≈ 11.1 m per 5 s step), Δh 9 bits at 1 m, Δv_x/Δv_y 9 bits at 0.5 km/h,
Δv_z 8 bits at 0.1 m/s → **55 bits**. The refinement with respect to the FlightBERT lattice
is what removes the high-order bit-flip outliers.

**Differentiable inversion.** With per-bit probabilities `g`, the classical soft-XOR
recurrence is

```
b_0 = g_0 ,   b_i = b_{i−1} + g_i − 2 b_{i−1} g_i
```

which is exact XOR on {0,1} and smooth in between. Expanding it shows that `b_i` is the
**prefix parity** of `g_0 … g_i`, and the parity of independent Bernoulli variables is
`b_i = ½(1 − ∏_{j≤i}(1 − 2g_j))`. The two forms are **algebraically identical, in both
value and gradient** (measured max deviation 2e-16), but the parity form is a single
`cumprod` rather than a Python loop of ~5 kernel launches per bit: in a model with 6
attributes × several bits this removed **≈330 kernel launches per forward pass**. A
straight-through estimator then replaces `g` by `g + (1[g ≥ ½] − g).detach()` so the
forward decode is exact and the gradient still reaches the bit logits. This is the
prerequisite for `L_STE` in §4.11.

**Multi-horizon accumulation.** The decoded magnitude is
`v̂ = (1 − 2·sign) Σ b_i 2^{n−2−i}`, the per-step rate is `v̂ / scale`, and the trajectory
is the cumulative sum over the *non-uniform* horizon increments:
`ŷ_h = y_T + Σ_{j≤h} Δk_j · v̂_j / scale`. Positions, therefore, are not regressed
independently per horizon: they are the integral of predicted rates, which is what keeps
the Gray-Code branch kinematically plausible.

### 4.4 ACoAtt-V2: attribute correlation attention

Six attribute embeddings are built, each from its Gray-Code bit-vector concatenated with
the standardised scalar and the window-relative state, e.g.
`e_λ = GELU(AE_λ([gc_λ ‖ rel_x_λ ‖ norm_x_λ]))`. The **meteorological channel is attached
to the velocity it physically influences**: wind (`u`) to `v_x`, wind (`v`) to `v_y`,
temperature to `v_z`. Each coupled pair (lon↔v_x, lat↔v_y, alt↔v_z) is then recalculated by
a **bidirectional cross-attention** block (4 heads, d = 24):
`corr = MHA(q = e_p, k = e_c, v = e_c)`, `e_p^new = Linear([LN(e_p ‖ corr)])`. The six
recalibrated embeddings are concatenated and fused by `Linear_{144→144}` → LayerNorm → GELU.

### 4.5 Encoder

The fused embedding is concatenated with the standardised raw features and the relative
kinematics (144 + 9 + 6 = 159 channels), passed through a 1-D convolutional stem
(`Conv1d(k = 3, padding = 1)`, Kaiming init) acting as a local multi-scale smoother, then a
sinusoidal positional encoding, then a **pre-LayerNorm Transformer encoder**
(`d_model = 144`, 6 heads, 2 layers, FFN width `3d`, GELU, dropout 0.1). The encoder output
`M ∈ R^{B×T×144}` is summarised by a learned **softmax attention pooling**,
`α = softmax(W₂ GELU(W₁ M))`, `c = Σ_t α_t M_t`.

### 4.6 Non-autoregressive multi-horizon decoder

The historical step deltas are encoded (55 Gray bits + 6 continuous channels) and embedded
by the same convolutional stem into `T − 1 = 11` **prefix tokens**. For each horizon `h` a
**query** is formed from three sources: a learnable horizon embedding `E_h`; a continuous
Fourier time feature of `τ_h = Δt_h / Δt_max`, `f(τ) = [τ, √τ, sin(πτ), cos(πτ)]` mapped by
an MLP; and the encoder context `c`. The projections are concatenated
(`Linear_{432→144}`). Prefix and query tokens are processed by a **pre-LayerNorm
Transformer decoder** (d = 144, 6 heads, 2 layers, FFN `3d`) attending to `M`; the last
`H` outputs form `D ∈ R^{B×H×144}`. Because all horizons are emitted in one pass, the
latency is flat in the horizon — the property we exploit operationally in §8.

### 4.7 Physics kinematic prior

A closed-form propagator with **three** learned time constants
(`τ_turn ∈ [10,180] s`, init 45; `τ_vz ∈ [30,600] s`, init 180; `τ_acc`, init 30):

1. **Velocity blend.** The Doppler velocity is blended with the 20 s position trend,
   `w(t) = 0.12(1 − e^{−(t−5)/45})`, `v₀ = (1−w)·v_Doppler + w·(p_t − p_{t−4})/20`.
2. **Turn rate** from the last three velocity vectors,
   `ω₀ = wrap(ψ_t − ψ_{t−3}) / 15 s`, clipped to ±0.035 rad/s.
3. **Damped coordinated turn.** With `θ(t) = ω₀ τ_turn (1 − e^{−t/τ_turn})` and
   `θ̄ = θ/2`: `v(t) = R(±θ̄) v₀` and `Δp(t) = v₀ t + R(±θ̄/2) v₀ t`.
4. **Damped vertical rate.** `v_z(t) = v_{z0} e^{−t/τ_vz}`, `Δz(t) = v_{z0} τ_vz (1 − e^{−t/τ_vz})`.
5. **Geodesy.** `m/deg_lon = 111320 cos φ`, `m/deg_lat = 111132`.

The three scalars are the **only** parameters: the propagator itself is parameter-free, so
it is fully interpretable and serves as the anchor of the continuous head. Its standalone
accuracy is measured in §6.3.

### 4.8 Three heads and the differentiable trust region

**Head A (Gray-Code).** `Linear_{144→55}` → per-attribute sigmoid → STE decode → `ŷ_gc`.

**Head B (continuous residual).** `Linear_{150→144} → GELU → Linear_{144→6}`, fed with
`[D ‖ kin_disp]`, where `kin_disp = (P − y_T) ⊘ [1,1,100,200,200,10]` tells the network how
far the physics prior believes the aircraft will move. The correction is scaled by a
horizon-dependent factor `t ⊙ [0.25°, 0.25°, 35, 80, 80, 6]` with `t = [√τ_h,√τ_h,√τ_h]`,
so the correction budget grows with the horizon. **The output layer is zero-initialised**,
hence at initialisation `ŷ_cont = P` exactly: the network starts from the physics solution
and learns only a correction.

**Head C (MAF-Net gate).** A gate `α = σ(MLP_{150→64→6}([D ‖ kin_disp]))` (bias init −2, so
α₀ ≈ 0.12) interpolates the two branches:

```
ŷ = ŷ_cont + α ⊙ tanh( (ŷ_gc − ŷ_cont) / σ_h ) ⊙ σ_h ,     σ_h = σ_base ⊙ (0.02 + 0.98 τ_h)
```

with `σ_base = [0.008°, 0.008°, 2.5, 15, 15, 1.5]`. The `tanh` **bounds** the influence of
the (coarser) Gray-Code branch to ±σ per attribute while keeping a gradient everywhere: no
clipping, no dead units. The measured contribution of the gate is small but systematic
(§6.3).

### 4.9 Multi-hypothesis head

`K` learnable **mode queries** (DETR/ASCENT style) are appended to the decoder input:
`dec_in = [prefix(11) ‖ horizon_queries(H) ‖ mode_queries(K)]`. A **float attention mask**
is applied to the decoder self-attention with the pattern

| rows \ cols | prefix | horizon | modes |
|---|---|---|---|
| prefix | ✓ | ✓ | ✗ |
| horizon | ✓ | ✓ | ✗ |
| modes | ✓ | ✓ | ✓ |

Because the horizon rows cannot attend to the mode columns, the deterministic pathway is
**mathematically unchanged** — verified: max deviation 1.9e-6, i.e. float32 rounding. The
mode tokens are the only ones that attend to each other, so the `K` hypotheses can never
perturb the operational prediction. This is the property that lets us ship one model that
is simultaneously a point predictor and an ensemble.

**Decoding.** The mode token `m` acts as a per-mode bias on the horizon features,
`z_{m,h} = D_h + m_m`, and
`ŷ^m_h = P_h + MLP_hyp([z_{m,h} ‖ κ_h]) ⊙ s_h`, with `MLP_hyp` = `Linear_{D+6→D} → GELU →
Linear_{D→6}` and a **zero-initialised output layer**, so every hypothesis starts *exactly*
at the physics solution and is kinematically consistent by construction. A score head
`MLP: D → 1` (bias −2) emits a confidence per hypothesis.

**Winner-takes-all.** With `D(m)` the 3-D distance of hypothesis `m` to the truth and
`m★ = argmin_m (1/H) Σ_h D(m)_h`,

```
L_hyp = L_reg(ŷ^{m★}, y)  +  0.1 · (1/K) Σ_m L_reg(ŷ^m, y)  +  0.1 · CE(α, m★)
         └─ winner only ─┘   └──── auxiliary ────┘             └── ranking ──┘
```

The **auxiliary term is essential**: without it the `K−1` losing hypotheses receive no
gradient and degenerate (observed: the mean-over-modes error collapses onto the prior).
The total objective is `L = L_hybrid + 0.5 · L_hyp`.

### 4.10 Retrieval-conditioned intent prior

**Key.** Every observation is described by a 6-D descriptor
`d = [λ, φ, h/500, sin ψ, cos ψ, |v|/50]` (position, flight level, heading, speed). The
absolute position enters with a **down-weight** (`pos_weight`); a diagnostic sweep showed
that dividing it by 10 — turning the key from a *route identity* into a *neighbourhood
prior* — raises the oracle from 981 m to 900 m.

**Value.** For every candidate the future is stored in the candidate's **own** local frame
(forward, right, up in metres) with its ground speed and heading, so the frame can be
rotated into the query's heading at query time.

**Query.** The `K = 8` nearest candidates are found on the GPU with the
`|q|² − 2qc + |c|²` expansion over **342,572 candidate windows** (from 75,624 flights,
stride 16); their frames are rotated into the query's heading frame and scaled (positions
÷1000, speed ÷100) to keep the conditioning O(1).

**Conditioning.** The context is encoded per neighbour (`Linear_{4→D}`) and pooled over
neighbours by a learned attention; the result `ρ_h` is appended to the horizon-query input:
`q_h = Linear_{4D→D}[ c ‖ e_h ‖ φ(t_h) ‖ ρ_h ]`. The block is purely additive, so the
deterministic path is otherwise untouched.

**Warm start and leakage control.** `load_base_weights` copies a base checkpoint verbatim
except `horizon_proj`, whose input grew from `3D` to `4D`: the first three blocks keep the
base weights and the retrieval block starts at **zero**, so the model is *exactly* the base
model at initialisation (verified: max deviation 0.0) and any improvement is attributable
to the retrieval alone. At **training** time the candidates belonging to the *same flight*
as the query are excluded (vectorised through a CSR-style flight index); without this, the
model learns to copy the future of a near-duplicate window of the same trajectory, which is
impossible at inference because the pool holds only past data.

### 4.11 Loss function

The objective is a weighted sum of five terms plus the optional hypothesis term:

```
L = 2.5·L_pos + 0.6·L_vel + 0.4·L_BCE + 0.25·L_STE + 0.25·L_kin     (+ 0.5·L_hyp)
```

| term | role | construction |
|---|---|---|
| `L_pos` | position accuracy | Smooth-L1(β = 0.25) of `(ŷ_pos − y_pos) / (σ_attr ⊙ h_norm)`, `σ_attr = [0.02°, 0.02°, 4, …]` |
| `L_vel` | velocity accuracy | Smooth-L1(β = 0.5) of `(ŷ_vel − y_vel) / [15, 15, 1.5]` |
| `L_BCE` | Gray-Code classification | mean BCE over the 6 attributes between the 55 predicted bit logits and the Gray encoding of the true step deltas |
| `L_STE` | decoded-trajectory consistency | Smooth-L1(β = 0.5) of the **STE-decoded** trajectory vs truth, normalised by `clamp(Δk_h/12, ½)` |
| `L_kin` | kinematic consistency | trapezoidal integral constraint per consecutive horizon pair |
| `L_hyp` | hypothesis selection | WTA + auxiliary + ranking (§4.9) |

Two details carry most of the engineering weight.

**(a) The horizon normalisation in `L_pos`.** The denominator contains
`h_norm = 0.04 + 0.96·(Δt_h / 60 s)`. Without it, the 300 s term dominates the gradient
by two orders of magnitude and the short horizons are never optimised — the failure mode
we *measure* in the legacy model (§6.5), which normalises its residual on a single 10,000 m
scale and consequently cannot learn the 5 s horizon. `h_norm` is, in effect, a
per-horizon learning-rate equaliser.

**(b) `L_STE` is only possible because the decoder is differentiable.** It supervises the
*task* (the decoded trajectory) rather than the *encoding* (the bit logits), which is the
Hybrid Constrained Loss of FlightBERT, and it is what forces the bit probabilities to be
jointly consistent instead of independently confident. The prefix-parity reformulation
(§4.3) is what makes it cheap enough to use.

**(c) `L_kin`** enforces, for every consecutive pair of horizons, that the predicted
position increment matches the trapezoidal integral of the predicted velocities:

```
ε_x = ( Δx − ½(v_x^(h) + v_x^(h+1)) Δt_h ) / (50 + 2 Δt_h)
```

and analogously for `y` and `z` (vertical denominator `15 + 0.5 Δt_h`). This is the term
that ties the three output heads to a single physically realisable curve.

---

## 5. Experimental protocol

* **Data**: the 90-day OpenSky dataset of §3 (72/9/9 chronological days).
* **Training**: AdamW (lr 1e-3, weight decay 1e-4, β (0.9, 0.999)), OneCycleLR
  (`pct_start` 0.2, cosine anneal), batch 2,048, 400,000 windows/epoch, max 30 epochs,
  early stopping on validation MDE (patience 10, min_delta 0.5 m). Identical for every
  model of the benchmark.
* **Evaluation**: test split, 60,000 sliding windows, **MDE 3-D ECEF** per horizon as the
  operational metric, plus ADE/FDE and `minADE@K` for the multimodal models.
* **Baselines**: constant velocity on the raw Doppler (CV-Doppler) and on a least-squares
  fit of the window (CV-LSQ); a deep multi-horizon LSTM; a re-implemented FlightBERT++ V2;
  a re-implemented **FlightPREV** (the previous model of this project); ASCENT; a
  conditional flow-matching model (CFM) and a DDPM on the same backbone; an attention-LSTM;
  an IMM with a neural residual corrector. The five literature models are documented in
  §6.1 of [`EXPERIMENTS.md`](EXPERIMENTS.md), with the paper each one reproduces.
* **Hardware**: single NVIDIA RTX 4070 (12 GB), Ryzen 7 5800X3D, 34 GB RAM, PyTorch
  2.6.0+cu126, `triton-windows` 3.2.0 (required for `torch.compile` on Windows).
* **Reproducibility**: every table is regenerated by a script named in the repository
  (`bench_run.py`, `make_benchmark_report.py`, `calibrate_uncertainty.py`,
  `run_retrieval.py`, `run_legacy.py`, `bench_latency.py`).

---

## 6. Results

### 6.1 Main benchmark (90 days, horizon 5–300 s)

Test split 2022-03-23 → 2022-03-31, 60,000 windows. **MDE** is the error of a single
prediction (the operational metric); **minADE@K** is the best-of-K and is an oracle
(K = 10 samples for the generative models, K = 8 modes for ASCENT).

| Model | MDE [m] | ADE 3-D | FDE 3-D | minADE@K | 5 s | 30 s | 60 s | 180 s | 300 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AeroTraj-Ultra (ours)** | **991.6** | 991.6 | 4,243.8 | — | **61.2** | **122.3** | **308.0** | **1,852.9** | **4,243.8** |
| FlightBERT++ V2 | 1,219.9 | 1,219.9 | 5,025.6 | — | 68.0 | 179.1 | 420.9 | 2,287.6 | 5,025.6 |
| Deep LSTM multi-horizon | 1,296.4 | 1,296.4 | 5,264.8 | — | 77.7 | 206.6 | 478.4 | 2,417.2 | 5,264.8 |
| **FlightPREV (legacy, ours)** | 1,495.8 | — | — | — | 102.8 | 188.4 | 470.9 | 2,753.1 | 6,466.5 |
| CV (Doppler) | 1,507.7 | 1,507.7 | 6,643.0 | — | 64.2 | 171.5 | 450.7 | 2,802.7 | 6,643.0 |
| CFM (FlowATC-style) | 1,704.3 | 1,704.3 | 7,125.8 | **983.5** | 87.2 | 208.7 | 552.7 | 3,228.3 | 7,125.8 |
| CV least-squares | 1,802.5 | 1,802.5 | 7,391.8 | — | 81.5 | 291.5 | 676.4 | 3,337.4 | 7,391.8 |
| IMM + neural residual | 2,019.2 | 2,019.2 | 7,721.2 | — | 90.9 | 414.0 | 913.7 | 3,710.3 | 7,721.2 |
| ASCENT | 2,348.9 | 2,348.9 | 8,426.0 | 1,088.8 | 105.7 | 523.9 | 1,144.9 | 4,398.0 | 8,426.0 |
| Attention-LSTM phase-aware | 3,170.0 | 3,170.0 | 14,694.6 | — | 100.4 | 508.8 | 1,105.0 | 5,551.3 | 14,694.6 |
| DDPM / MID | 4,300.5 | 4,300.5 | 17,871.6 | 1,763.6 | 146.4 | 631.2 | 1,600.7 | 8,128.6 | 17,871.6 |

**Findings.**

1. AeroTraj-Ultra wins the operational metric: **−18.7 %** vs FlightBERT++ V2, **−34.2 %**
   vs the strongest kinematic baseline (CV-Doppler), **−23.5 %** vs a deep LSTM. It is the
   only model that is better at *every* horizon of the grid.
2. **The aircraft-centric frame alone is not enough.** The attention-LSTM and the IMM
   residual model are *worse* than raw constant velocity despite access to the same
   features: the bottleneck is regression capacity and output parameterisation, not the
   representation.
3. **CFM beats DDPM head-to-head** on our data (1,704.3 vs 4,300.5 m single-draw, i.e.
   −60 %; −44 % in `minADE@10`), reproducing the qualitative claim of FlowATC, though with
   a much larger gap because our DDPM is deliberately minimal.
4. **The generative oracle is not a model.** The best-of-10 CFM (`983.5 m`) is nominally at
   parity with our deterministic model, but it requires knowing which of the ten samples is
   correct. The *operational* CFM, one sample, is 72 % worse. Any comparison that quotes
   only the oracle overstates the generative approach by a factor 1.7.
5. **Output parameterisation matters for long horizons.** The replicated ASCENT integrates
   speed / turn-rate / vertical-rate, so it cannot represent a deceleration or a flight
   level other than the observed one; the regime error accumulates (2,348.9 m at 300 s).
   Its early stopping plus auxiliary loss improved it by 29 % over the first version
   (3,322 → 2,348.9 m), but it remains below constant velocity.

### 6.2 Internal ablation

Horizon grid **5 s → 600 s (16 points)**, 90-day model, test split; mean MDE over the grid.

| Configuration | Run A (40 d, CPU) | Run B (40 d, GPU) | Run C (90 d, GPU) |
|---|---:|---:|---:|
| **AeroTraj-Ultra (full)** | **3,625.9** | **3,019.5** | **2,751.0** |
| without MAF gate / GC fusion (`pred_cont` only) | 3,638.5 | 3,021.3 | 2,752.7 |
| Physics prior only (no network) | 4,054.5 | 4,052.2 | 4,202.2 |
| FlightBERT++ V2 (pure Gray-Code MBC) | 5,183.0 | 4,190.7 | 3,935.6 |
| Deep LSTM multi-horizon | 5,609.9 | 4,523.6 | 4,276.0 |
| CV / Doppler | 4,139.7 | 4,139.7 | 4,278.2 |
| CV / least-squares on the window | — | 4,580.5 | 4,751.8 |

**Reading.** The dominant gain (+1,185 m in run C) comes from the *hybrid* construction
(Gray-Code + continuous residual + physics anchor): the pure Gray-Code model is 3,935.6 m,
the physics prior alone 4,202.2 m, and their combination 2,751.0 m — neither component
alone is competitive, which is the central architectural claim. The MAF gate adds a further
−1.7 m: small, but systematic and free in latency.

### 6.3 Effect of the amount of data (identical recipe)

**The 365-day run (the headline result of the data study).** Same architecture, same
5 s → 300 s grid, same recipe, trained on **365 consecutive days** (293/36/36):
**438 215 unique flights** and 306.8 M 5-second points, i.e. **4.5× the flights** and 4.7×
the points of the 90-day dataset. 150 epochs with `torch.compile`, 15.8 s/epoch (**42 min**
total), best checkpoint at epoch 146 (val MDE 952.3 m), early stopping armed but never
triggered.

Because the two runs have different test splits, each checkpoint was evaluated on **both**
splits, 60,000 windows each (`eval_cross.py`):

| test split | 90-day model | **365-day model** | Δ |
|---|---:|---:|---:|
| March 2022 (2022-03-23 → 03-31) | 991.6 m | **847.3 m** | **−14.55 %** |
| Nov–Dec 2022 (2022-11-26 → 12-31) | 1,082.8 m | **926.0 m** | **−14.48 %** |

Per-horizon, on the Nov–Dec split (the 5 s → 300 s profile of the same comparison):

| | 5 s | 30 s | 60 s | 120 s | 180 s | 300 s |
|---|---:|---:|---:|---:|---:|---:|
| 90-day model | 59.7 | 124.1 | 328.8 | 1,043.7 | 2,041.3 | 4,640.2 |
| **365-day model** | **56.8** | **111.3** | **290.9** | **903.6** | **1,741.1** | **3,930.6** |
| Δ | −4.9 % | −10.3 % | −11.6 % | −13.4 % | −14.7 % | −15.3 % |

**Findings.**

1. The gain is **−14.5 %, measured independently on two different test periods** and
   agreeing to within 0.07 percentage points: it is the data, not a test-set artefact.
2. It is **monotone in the horizon** (−5.3 % at 5 s → −15.8 % at 300 s on the March split,
   −4.9 % → −15.3 % on the Nov–Dec split). More data helps exactly where the residual error
   is *intent* rather than sensor noise — the same signature as the retrieval prior (§6.4).
3. It is **~8× larger than the 90 → 180-day gain** (−1.8 % over 5–180 s), so the return on
   data compounds once the trajectory neighbourhood is dense, which is consistent with a
   retrieval-like mechanism rather than with plain regularisation.
4. **Temporal drift, measured while doing the above.** The *same* model is 7.0 % worse at
   60 s and 10.0 % worse at 300 s on Nov–Dec than on March, but 1.9 % *better* at 5 s: the
   winter period is genuinely harder where intent matters. A single test split therefore
   under- or over-states a model depending on its season, and cross-split evaluation should
   be the default rather than an extra.

**Earlier data-scaling runs** (identical recipe, 5 s → 180 s grid):

| Trained on | Unique flights | 5 s | 30 s | 60 s | 180 s | **Mean MDE (5–180 s)** |
|---|---:|---:|---:|---:|---:|---:|
| 90 days (72 train) | 76,939 | 69.4 | 129.6 | 318.5 | 1,986.5 | 539.3 m |
| **180 days (144 train)** | **164,410** | 69.5 | 129.2 | 315.2 | 1,939.6 | **529.4 m** |
| Δ | +2.2× | +0.1 % | −0.3 % | −1.0 % | −2.4 % | **−1.8 %** |

On the same test set, FlightBERT++ V2 scores 818.8 m (−35.4 % vs ours), CV-Doppler 886.1 m
(−40.3 %), CV-LSQ 1,076.9 m, physics prior 857.5 m. The 180-day run's best single gain is
−44.8 % at 30 s, and at 180 s the difference is 1,929.5 m vs 2,846.1 m (−0.92 km).

### 6.4 The two compositional heads

Incremental contribution, 90 days, horizon 5–300 s:

| Variant | MDE [m] | minADE@8 | 60 s | 120 s | 300 s | conformal radius @90 % |
|---|---:|---:|---:|---:|---:|---:|
| Base (deterministic) | 991.6 | — | 316.1 | 977.4 | 4,357.5 | — |
| + multi-hypothesis (K = 8) | 984.9 | 627.4 | 304.6 | 944.2 | 4,169.1 | 1,413 m |
| + retrieval | 968.9 | — | 302.8 | 932.5 | 4,138.9 | — |
| **+ both** | **957.2** | **608.2** | **297.7** | **918.5** | **4,092.5** | **1,190 m** |

**Findings.**

* The two effects are **additive** (base → −0.7 % multi-hypothesis → −2.3 % retrieval →
  −3.5 % combined), and the combined model is best at every horizon.
* The **retrieval** gain is **monotone in the horizon** (−0.3 % at 5 s, −5.0 % at 300 s):
  a prior on intent is irrelevant where physics dominates and decisive where the manoeuvre
  is unknown. This is the expected signature, and it is the evidence that the module works
  for the reason claimed rather than as generic regularisation.
* The retrieval improves the **uncertainty** as well: the oracle `minADE@8` drops from
  627.4 to 608.2 m and the conformal band shrinks a further 16 % (1,413 → 1,190 m).
* The retrieval is **not a free lunch**. A first attempt, trained from scratch with an
  unnormalised context, was **24 % worse** than the base model: the context must be scaled
  and the projection warm-started, or the retrieval signal swamps the other conditioning.
  The warm-start construction (§4.10) is what makes the attribution clean.

### 6.5 Comparison with the previous model (FlightPREV)

The earlier model of this project, **FlightPREV**, is a Transformer encoder with a
multi-horizon head predicting a **residual with respect to a constant-velocity ballistic
model**, an auxiliary flight-phase task and learned categorical embeddings (aircraft type,
operator, departure and arrival airports). We re-implemented it faithfully
(`legacy_flightprev.py`, `run_legacy.py`) and trained it on the **same** OpenSky data, the
**same** splits, the **same** horizon grid (extended from its native 10–120 s to 5–300 s),
the **same** optimiser recipe and the **same** metric. 3,440,682 parameters (2.7× ours).

| Model | MDE [m] | 5 s | 30 s | 60 s | 120 s | 300 s | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| **AeroTraj-Ultra + retrieval** | **957.2** | **61.1** | **119.6** | **297.6** | **918.6** | **4,092.4** | 1.34 M |
| AeroTraj-Ultra (base) | 991.6 | 61.2 | 122.3 | 308.0 | 952.2 | 4,243.8 | 1.28 M |
| FlightBERT++ V2 | 1,219.9 | 68.0 | 179.1 | 420.9 | 1,210.0 | 5,025.6 | — |
| Deep LSTM | 1,296.4 | 77.7 | 206.6 | 478.4 | 1,296.9 | 5,264.8 | — |
| **FlightPREV (legacy, re-impl.)** | **1,495.8** | 102.8 | 188.4 | 470.9 | 1,395.2 | 6,466.5 | 3.44 M |
| CV (Doppler) | 1,507.7 | 64.2 | 171.5 | 450.7 | 1,406.3 | 6,643.0 | — |

**Findings.**

1. The new model is **36 % better**, and the gap is **uniform across the horizon**
   (40.6 % at 5 s, 36.5 % at 30 s, 36.8 % at 300 s): a genuinely better predictor, not a
   long-horizon artefact.
2. FlightPREV, re-implemented on OpenSky, ends up **barely above the constant-velocity
   baseline** (1,495.8 vs 1,507.7 m) despite 2.7× the parameters. Two structural causes,
   both measured:
   * **Its normalisation hides the short horizons.** The residual is expressed in ENU units
     of 10,000 m, so with a Huber loss of δ = 0.1 (i.e. 1,000 m) an error of 100 m at 5 s
     lies *inside the linear region* and barely influences the gradient. The 5 s error
     therefore stalls at 102.8 m, against 61.1 m for AeroTraj-Ultra, which normalises the
     loss per horizon (`h_norm`, §4.11). The asymmetry is a property of the legacy loss,
     not of the legacy code.
   * **The categorical priors it relies on do not exist in OpenSky.** The original learned
     embeddings of aircraft type, operator and departure/arrival airports; the OpenSky
     competition data carries none of them, so they were replaced by two coarser
     geographic embeddings (origin/destination area, 2° grid). This is an acknowledged
     **handicap** for the legacy model.
3. The comparison must therefore be read as *"architecture + loss + priors on this
   dataset"*, not as *"old code versus new code"*. We state it explicitly because the
   comparison is favourable to our model.

### 6.6 Where the remaining error comes from

The 5 s error of the model is **61.2 m**, while the 1σ noise of the CPR quantisation on the
anchor position is **75.4 m**. The 5–30 s band is therefore **saturated by sensor
quantisation**: no architecture can do better without a better position source. All the
remaining headroom lives at 60–300 s, where the error is not measurement noise but
**intent** — which is exactly the regime the retrieval prior of §4.10 targets, and
the reason its gain is monotone in the horizon.

---

## 7. Certified uncertainty

Rather than trusting the score head (see §9.2, a negative result), we certify the ensemble
with **conformal prediction**. The `K = 8` hypotheses are treated as an ensemble, the
nonconformity score is the 3-D distance of the truth to the **closest** hypothesis, and the
quantile is calibrated **per stratum** (climb / cruise / descent, a Mondrian split) at level
`1 − α` on the validation split. The result is *exact* and *horizon-uniform* coverage:

| nominal | coverage (ensemble) | mean radius (ensemble) | mean radius (inflated single pred.) |
|---:|---:|---:|---:|
| 80 % | 0.805 | 787 m | 1,282 m |
| 90 % | 0.903 | 1,413 m | 2,675 m |
| 95 % | 0.951 | 2,411 m | 4,542 m |

On the retrieval-conditioned model the radius at 90 % improves further, from 1,413 m to
**1,190 m**, against 2,675 m for the inflated single prediction: a **56 % narrower
certified band** at the same, empirically verified, coverage. For conflict detection, which
consumes an interval rather than a point, this is the operationally usable output of the
multi-hypothesis head.

## 8. Inference latency

Accuracy is only usable in ATM if the prediction exists before the update cycle closes, so
every model in this report was timed on the same harness (`bench_latency.py`).

**How it is measured.** The operational case is a *single aircraft*, so the headline number
is the **end-to-end batch-1 latency**: one 12 × 5 s window is moved host→device, the model
(or the model plus its retrieval lookup) runs, and the wall clock is read after
`cuda.synchronize()`. Each model is measured in **three independent blocks of 200
repetitions** and the median of the block medians is reported, together with the p95 within
a block. Because the host side is jittery on a Windows workstation with an unlocked GPU
clock, the GPU-busy time is reported separately, as the median of a **per-call CUDA-event
measurement** (one event pair bracketing exactly one call) with its observed spread across
the three blocks. `torch.no_grad()` is active throughout; the first-call compilation cost is
measured separately as a cold start.

### 8.1 Single-aircraft latency (the operational case)

| Model | median [ms] | p95 [ms] | GPU only [ms] | spread |
|---|---:|---:|---:|---:|
| Conditional Flow Matching (20 ODE steps, 1 sample) | 93.26 | 102.17 | 93.40 | ±1 % |
| DDPM / MID (50 steps, 1 sample) | 262.31 | 357.12 | 260.42 | ±3 % |
| **AeroTraj-Ultra + retrieval** | **20.12** | 23.88 | 20.19 | ±1 % |
| **AeroTraj-Ultra** | **19.59** | 23.39 | 19.34 | ±5 % |
| ASCENT | 12.05 | 13.85 | 11.98 | ±1 % |
| FlightBERT++ V2 | 9.77 | 11.10 | 9.78 | ±2 % |
| Deep LSTM | 0.62 | 1.25 | 0.61 | ±2 % |
| CV / Doppler | 0.46 | 1.18 | 0.46 | ±8 % |

**Reading.**

* A prediction for one aircraft takes **19.6 ms**, and **20.1 ms** with the retrieval prior
  — the intent prior costs **+0.5 ms**. The whole 5–300 s horizon vector is produced in that
  time, because the decoder is non-autoregressive.
* The two generative models are far more expensive for a *worse* single-draw error:
  CFM costs **4.7×** our latency and is 72 % less accurate; DDPM costs **13×** and is
  4.3× less accurate. This is the latency counterpart of §6.1's oracle caveat.
* FlightBERT++ V2 is 2.0× faster than AeroTraj-Ultra at equal parameter order and 23 %
  less accurate: the difference is architectural (three heads, a Gray-Code decoder and a
  physics prior against a single regression head), and it is the price of the accuracy in
  §6.1. Both models are non-dominated in the accuracy/latency plane (Figure 8.1).

### 8.2 The single-aircraft case is launch-bound, not compute-bound

This is the most important engineering finding of the speed measurement, and it is measured
without a profiler, from the throughput sweep alone. The average cost per aircraft is
**19,591 µs at batch 1** but the *marginal* cost of one extra aircraft, i.e. the slope
between batch 1 and batch 4096, is **12.6 µs**; **99.94 %** of the batch-1 latency (19.58 ms)
is therefore fixed per-call overhead — kernel launches and host-side scheduling, not
arithmetic. Two independent confirmations:

* **Batching.** Cost per aircraft falls from 19,591 µs to **17.3 µs** (batch 4096), a factor
  **1,133×**, while the total time only grows from 19.6 ms to 71.0 ms for 4096× the work.
* **Compilation.** `torch.compile(mode="reduce-overhead")` takes the batch-1 latency from
  **19.59 ms to 4.47 ms** — a **4.4× reduction with no change in the mathematics**, which is
  only possible if the time was spent in per-kernel overhead.

The practical consequence is that a single-aircraft deployment is not an option: the model
should be fed a *snapshot* of traffic. At batch 512 the full pipeline costs 33.7 ms for 512
aircraft (**15,200 aircraft/s**), i.e. a 5,000-aircraft European snapshot in **~330 ms**.

### 8.3 Throughput

| batch | AeroTraj-Ultra total [ms] | µs/aircraft | aircraft/s | + retrieval total [ms] | aircraft/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 19.64 | 19,638 | 51 | 22.85 | 44 |
| 8 | 20.33 | 2,542 | 393 | 21.83 | 366 |
| 32 | 20.22 | 632 | 1,583 | 21.57 | 1,483 |
| 128 | 20.21 | 158 | 6,333 | 21.52 | 5,949 |
| 512 | 20.85 | 41 | 24,556 | 33.66 | 15,212 |
| 2,048 | 40.17 | 20 | 50,990 | 116.23 | 17,620 |
| 4,096 | 70.99 | 17 | **57,702** | 231.46 | 17,697 |

**Reading.** The base model is **flat up to batch ~512** (≈ 20 ms for the whole batch — the
launch overhead is amortised, not the compute) and only then does arithmetic start to
dominate. The retrieval variant, by contrast, **stops scaling after batch ~512**: its top-k
is an `O(B × M)` scan over M = 342,572 candidates and at batch 512 it already costs
**16.3 ms against a 6.7 ms forward pass** (70 % of the pipeline). Its per-aircraft cost
therefore saturates at ~17,200 aircraft/s instead of 55,000, and at fleet scale the exact
top-k must be replaced by an indexed or approximate search — a concrete, measured limitation
of the retrieval module (§9.3), not a hypothetical one. §8.6 shows how far this can be pushed
with the layout alone, and §9.2 records what had to be rejected.

### 8.4 Latency is flat in the prediction horizon

The non-autoregressive claim is directly testable: the same architecture is rebuilt with
`H = 1, 2, 4, 8, 13` horizons and the batch-1 GPU time is measured for each
(`bench_latency_horizon.py`; random weights, since the checkpoint changes no shape).

| horizons | grid | GPU latency [ms] | parameters |
|---:|---|---:|---:|
| 1 | 5 s | 19.15 | 1,320,046 |
| 2 | 5–10 s | 19.26 | 1,320,190 |
| 4 | 5–20 s | 19.22 | 1,320,478 |
| 8 | 5–90 s | 19.94 | 1,321,054 |
| 13 | 5–300 s | **22.36** | 1,321,774 |

Predicting 13 horizons covering 5–300 s costs **1.17×** predicting a single horizon at 5 s:
the marginal cost of extending the horizon by two orders of magnitude is **17 %**. An
autoregressive decoder would pay one forward pass per step (~60× for 5–300 s at 5 s
resolution), which is exactly the cost profile CFM and DDPM exhibit for 20 and 50 rollout
steps (§8.1).

### 8.5 Cold start and deployment envelope

Constructing the model and moving the weights to the GPU takes **28.2 ms** (base) /
**26.8 ms** (retrieval) — negligible against any operational process lifetime, and the
`torch.compile` cost is a one-off paid at first call. Against the surveillance update cycles:

| requirement | budget | AeroTraj-Ultra |
|---|---:|---|
| ADS-B update cycle | 1 s | 49.7 aircraft/s sequential; 57,702 aircraft/s batched |
| radar update cycle | 5 s | 248 aircraft/s sequential; 57,702 aircraft/s batched |
| conflict-detection snapshot (5,000 aircraft) | 5 s | ~330 ms with the retrieval (batch 512), ~90 ms without |

The model is therefore **not latency-critical in any realistic ATM deployment**: even the
sequential single-aircraft path has a ~50× margin against a 1 s cycle, and the constrained
resource is not time but the *accuracy* of the 60–300 s band (§6.6).

![Accuracy versus inference latency](figures/pareto_accuracy_latency.png)

**Figure 8.1** — Accuracy (3-D ECEF MDE, lower is better) against single-aircraft inference
latency (GPU-busy time, log scale). The Pareto frontier is
CV → Deep LSTM → FlightBERT++ V2 → AeroTraj-Ultra → AeroTraj-Ultra + retrieval; ASCENT, CFM
and DDPM are **dominated**, i.e. they are at the same time slower *and* less accurate than a
model in this comparison. Reproduce with `plot_pareto.py`.

### 8.6 What the inference optimisations buy

Every technique below was adopted only after a **numerical-equivalence check**: the
prediction and the 3-D ECEF error are compared against the eager reference on 24,576 test
windows (`check_compile_quality.py`, `results_bench/compile_quality.json`), so a speed-up
is never traded for accuracy.

| variant (batch 1, end-to-end, incl. H2D) | latency | speed-up | MDE | ΔMDE |
|---|---:|---:|---:|---:|
| eager + `torch.no_grad()` (the §8.1 baseline) | 19.3–20.7 ms | — | 1000.24 m | — |
| eager + `torch.inference_mode()` | **17.2–19.6 ms** | **1.05–1.20×** | 1000.24 m | **0.000 %** |
| **`torch.compile(reduce-overhead)`** | **2.8–6.5 ms** | **3.0–7.4×** | 1000.25 m | +0.0014 % |

Both ranges are over repeated independent runs: the compiled path is the noisiest because
inductor specialises on the input shape, so the exact kernel selection depends on the
autotune result and on how the replay graph is captured. The conservative statement is
**≥3×**, the typical one **~5–6×**.

| throughput (aircraft/s) | batch 1 | batch 128 | batch 512 | batch 2048 | batch 4096 |
|---|---:|---:|---:|---:|---:|
| deterministic, eager | 56 | 6,483 | 27,528 | 51,817 | 56,220 |
| **deterministic, compiled** | **155–178** | **24,104** | **28,336–75,296** | **62,730–73,206** | **67,605–73,345** |
| + retrieval, eager | 31 | 3,880 | 12,370 | 17,461 | 17,231 |
| **+ retrieval, compiled** | **109** | **10,151** | **22,111** | **20,016** | **20,968** |

**1. `torch.inference_mode()` instead of `no_grad()` — 20 % for free.** `no_grad` still
maintains autograd's version counters on every tensor; `inference_mode` removes them. The
output is **bit-identical** (`max |Δ| = 0.00e+00`).

**2. `torch.compile(mode="reduce-overhead")` — 4.4–6.1× at batch 1, 2.9× at batch 512.**
This is the same mechanism that gave the 1.9× training speed-up (lab notebook §6.4):
inductor fuses the ~600 tiny kernels that make the model launch-bound (§8.2). At batch 512
the compiled model reaches 14 µs per aircraft, i.e. it finally operates at the *marginal*
cost measured in §8.2 (12.6 µs) instead of the average — the launch overhead is gone.

**3. The quality cost of compilation is zero in aggregate, and it is understood.** The
compiled model is **not bit-identical**, and the reason is structural: the Gray-Code
decoder thresholds at `probs ≥ 0.5`, so float noise is amplified into a discrete bit flip
on a tail of windows. Measured on 24,576 windows:

| | value |
|---|---:|
| physical distance between the two predictions | p50 **0.79 m**, p90 1.79 m, p99 7.4 m, p99.9 29.1 m |
| windows differing by more than 1 m / 10 m / 100 m | 27.3 % / 0.53 % / **0.06 %** |
| **3-D ECEF MDE** | 1000.24 m → **1000.25 m (+0.0014 %)** |
| worst per-horizon change | **+0.076 m** (at 300 s) |

The discrete tail is a property of the architecture, not of `torch.compile`: any float
reassociation reaches it. Because the metric, the per-horizon profile and therefore the
conformal band are unchanged, compilation is enabled by default in the inference path.

**4. The retrieval search, not the network, is the remaining bottleneck.** At batch 512 the
compiled forward pass costs **6.7 ms** while the retrieval `top-k` over 342,572 candidates
costs **16.3 ms** — 2.4× the model, and 70 % of the 23.2 ms full pipeline. The cause is again
launch overhead plus the distance-matrix traffic, not FLOPs (the whole distance computation
is 17 GFLOP at batch 4096, i.e. <1 ms of pure compute on an RTX 4070). Two blocking
dimensions exist, and **they interact**: the number of kernel launches is
≈ (B / block) × (M / cand_block), so shrinking one while the other is small explodes the
launch count. Measured (`bench_retrieval_grid.py`; every configuration returns the same
neighbours up to floor-level ties):

| configuration | B = 512 | B = 4096 | peak MiB | neighbours vs default |
|---|---:|---:|---:|---:|
| block 256, cand_block None | 21.0 ms | 167.2 ms | 1,081 | 9 / 74 differ (tie-break) |
| block 16, cand_block None | 17.1 ms | 133.6 ms | 140 | 9 / 74 differ (tie-break) |
| **block 512, cand_block 16 384** | **16.5 ms** | **130.3 ms** | **178** | reference |
| block 256, cand_block 16 384 | 17.7 ms | 144.9 ms | 125 | identical |
| block 256, cand_block 4 096 | 80.2 ms | 585.7 ms | 94 | 4 / 60 differ |
| block 64, cand_block 16 384 | 80.7 ms | 608.7 ms | 95 | identical |
| block 16, cand_block 16 384 | **240.0 ms** | **1,622.5 ms** | 80 | identical |

The adopted default is `block = 512` **together with** `cand_block = 16 384`: the fastest
configuration measured, with a 178 MiB peak instead of 1,081 MiB. Note the bottom row: the
naive pairing of the two individually-attractive settings is a **13× regression**, and the
blocking dimension alone is not free — the launch count is what the model is bound by.

Two further levers were measured and **rejected**:

| lever | effect | neighbours | verdict |
|---|---|---|---|
| fp16 distances | −30 % | **differ on 99.9 % of queries** | rejected (§9.2) |
| pool with one candidate per flight (342,572 → 75,624) | **−82 %** of candidates, **5.5×** faster search | different pool ⇒ MDE **+0.127 %** | opt-in only |

The pruned pool is the only lever that removes the *traffic* rather than rearranging it,
and its cost is real: +1.21 m of MDE (949.29 → 950.50 m on 12,288 windows), concentrated at
the long horizons (+8.9 m at 300 s). It is therefore exposed as
`build_pool(..., one_per_flight=True)` and is **not** the default: the default keeps the
quality, and deployments that are throughput-bound can trade 0.13 % of accuracy for 5.5×.

**5. The retrieval path is intrinsically jittery.** It issues hundreds of tiny kernels, so
its latency tracks host CPU load in a way the model forward does not: measured *on the same
machine*, the search at batch 512 ranged from 17 ms (idle) to 228 ms (while the CPU was
saturated by the dataset build and the OS was flushing 10 GB of arrays to disk), a factor
13. The base model measured in the same two conditions moved by less than 5 %. For a
deployment, this means the *retrieval* is the component that must be provisioned for the
worst case, or replaced by an indexed search (§9.3).

**6. Rejected: a hand-captured CUDA graph.** Capturing the forward pass with
`torch.cuda.CUDAGraph` is the textbook way to remove launch overhead, and it fails on this
stack in a way worth recording: after the capture the CUDA caching allocator reports
`Offset increment outside graph capture encountered unexpectedly` and **every subsequent
allocation fails**, poisoning `torch.compile` for the rest of the process. Inductor's own
CUDA-graph trees (`mode="reduce-overhead"`) do the same job correctly.

**Practical interface.** `inference.py` exposes the optimised path as one object:

```python
from inference import FastPredictor
fp = FastPredictor("config_90d_300s.json", "checkpoints_90/aerotraj_ultra_best.pt")
fp.warmup(512)      # pay the one-off compile cost (7-15 s) up front
pred = fp(x)        # [B, 13, 6], torch.inference_mode, compiled
```

**Measurement note.** The absolute numbers above were taken with the host otherwise idle.
The retrieval path issues hundreds of kernel launches, so its latency is sensitive to CPU
load (the per-call CUDA-event median included scheduling gaps when the same machine was
concurrently building the 365-day dataset); the *relative* comparisons reported here were
all taken within a single run, where the load is common to both arms.

---

## 9. Discussion

### 9.1 What the evidence supports

1. **Hybrid beats pure.** Neither the Gray-Code branch nor the physics propagator is
   competitive alone (§6.2); their fusion is. The physics anchor makes the model correct by
   construction at initialisation and gives the network a residual to learn instead of a
   trajectory to invent.
2. **Per-horizon normalisation is worth more than parameters.** The single largest
   qualitative gap in this report is not architectural: FlightPREV's 10,000 m residual
   scale makes the 5 s horizon invisible to its loss, and 2.7× our parameters cannot
   compensate (§6.5). The `h_norm` term in `L_pos` is a cheap, transferable fix for any
   multi-horizon regression.
3. **Intent priors help exactly where intent matters.** The retrieval gain is monotone in
   the horizon and additive with the multi-hypothesis head. The ceiling is informative:
   the retrieval *oracle* is 900 m and we collect 531 m of it — the remaining gap is the
   selection problem, not the prior.
4. **Uncertainty should be certified, not estimated.** The conformal band is narrower than
   the inflated single prediction *and* has verified coverage. The learned score head does
   not (§9.2).

### 9.2 Negative results, stated

We report these because they constrain the design space:

* **No hypothesis-selection rule beats the deterministic output.** With score-argmin
  (after changing the score head to regress the hypothesis error) the MDE is 1,031.0 m;
  the best hand-crafted rule — R1, "the hypothesis closest to the deterministic output" —
  gives 990.6 m; kinematic coherence (R2) gives 1,523.9 m; R1+R2 gives 1,007.2 m. All are
  worse than the deterministic model (956.9 m on that split). The reason is structural:
  the `K` hypotheses come from a head *weaker* than the deterministic branch (which has the
  full encoder, the physics prior and the Gray-Code branch), and the oracle exists only
  because the truth happens to fall near one of them — knowing *which* requires information
  unavailable at inference. **Consequence: the multi-hypothesis head is an uncertainty
  estimator, not a better point predictor.**
* **The score-regression target must be detached.** Regressing the score on the hypothesis
  error without `err_m.detach()` lets the gradient flow back into the trajectory branch,
  which then lowers the loss by *inflating* its own error: the run diverged to NaN at epoch
  9. With the detach the run is stable; a non-finite-batch guard was added as well.
* **Training from scratch with an unnormalised retrieval context is 24 % worse than the
  base model** (§6.4). The warm start plus context scaling is not an optimisation, it is a
  requirement.

### 9.3 Limitations

* **Europe only** (`lon ∈ [−15°, 35°]`, `lat ∈ [30°, 65°]`). Extending to global coverage
  requires the Trino backend and OpenSky credentials.
* **At 5 s the model is at parity with constant velocity** (61.2 vs 64.2 m). The error is
  dominated by CPR quantisation of the position (~10 m, 1σ ≈ 75 m on the anchor), so the
  margin is structurally limited — this is a physical floor, not a model deficit.
* **At 300 s the absolute error remains large** (4.2 km, ≈ 2.6 % of the distance flown).
  Predicting 5 minutes of flight requires route knowledge and ATC clearances, not only the
  kinetic state: it needs flight-plan and short-term-weather integration.
* **The MAF gate contributes little** (−1.7 m). Its `trust_sigma` is a fixed schedule,
  almost deactivated at short horizons; it should be a learned function of the horizon.
* **Midnight-crossing flights are split at the split boundary**: the split is chronological
  by day, not by flight (~10 flights per boundary; marginal impact).
* **`predict.py` demonstrates inference on pre-processed windows only.** Operational use
  needs a wrapper converting `pyopensky.rest` / `pyopensky.trino` output into the 9
  features — the conversion function already exists
  (`data_pipeline._process_raw_dataframe_to_5s_segments`).
* **The legacy comparison carries a declared handicap** (§6.5): the OpenSky data has no
  aircraft type, operator or airport labels, so FlightPREV lost its categorical priors.

---

## 10. Conclusion

AeroTraj-Ultra is a **1.28 M-parameter, non-autoregressive** 4-D trajectory predictor that
wins an eleven-model, same-data, same-protocol benchmark over 5–300 s (991.6 m vs 1,219.9 m
for FlightBERT++ V2 and 1,507.7 m for constant velocity), reaches **957.2 m** with a
retrieval intent prior and a multi-hypothesis head, and ships a **certified** conformal
uncertainty band 56 % narrower than the inflated point prediction. Its training step was
made 1.9× faster by an algebraic identity — prefix parity vs sequential soft-XOR — that is
provably gradient-identical.

The recurring lesson across the report is that, in this problem, **where the loss puts its
sensitivity matters more than where the parameters go**: a per-horizon normalisation and a
physics anchor bought more accuracy than a 2.7× larger model on the same data, and an
intent prior bought strictly more than an extra output head. The open problem is
**selection**: the multi-hypothesis oracle (608.2 m) is real but currently unreachable,
and closing that gap needs a per-hypothesis coherence model, not a hand-crafted rule. The
natural next steps are a 365-day dataset (roughly doubling the retrieval neighbourhood
density) and a descriptor carrying time-of-day, the strongest single proxy for schedule
intent.

---

## References

**Architectural lineage**

1. Guo, Zhang, Yan, Zhang & Lin. *FlightBERT: Binary Encoding Representation for Flight
   Trajectory Prediction.* IEEE Transactions on Intelligent Transportation Systems 24(5),
   2023. — attribute correlation attention, Gray/binary encoding, Hybrid Constrained Loss.
2. Guo et al. *FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory
   Prediction Framework.* AAAI 2024. — non-autoregressive multi-horizon decoder, Gray-Code
   step deltas.
3. Bizhao et al. *IDDPG: Integrated Dynamic Diffusion Probabilistic Graph.* AAMAS 2025. —
   meteorological state integration.
4. Pang, Alam, Lulli et al. *MAF-Net: Multi-head Action Filter.* AAMAS 2026. —
   differentiable trust-region gate.
5. Reference implementation of FlightBERT++ V2: <https://github.com/gdy-scu/FlightBERT_PP_V2>.

**Models re-implemented as baselines**

6. Prutsch, Schinagl & Possegger. *ASCENT.* ICRA 2026. — aircraft-centric encoder,
   DETR-style mode queries, parameterised outputs, winner-takes-all loss.
7. FlowATC, 2026; Figuet et al. *Journal of Open Aviation Science* 4(2), 2026. —
   conditional flow matching on a DiT backbone, sequence inpainting, block-causal
   attention, AdaLN, probability-flow ODE sampling.
8. Ho, Jain & Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020.
9. Gu et al. *MID: Motion Indeterminacy Diffusion.* CVPR 2022.
10. Jia et al. *Scientific Reports* 12:15533, 2022; Shi et al. *IEEE TITS* 22:7242, 2021. —
    attention-LSTM, phase-aware modelling.
11. Li et al. *Sensors* 25(8):2531, 2025; Blom & Bar-Shalom, 1988. — Interacting Multiple
    Model filters with a neural residual corrector.

**Statistics**

12. Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World.* Springer, 2005. —
    conformal prediction; Mondrian stratification.

**Data**

13. The OpenSky Network — <https://opensky-network.org/> · bucket
    `s3.opensky-network.org/competition-data` · SDK <https://github.com/open-aviation/pyopensky>.
14. Schönberger, Aarts, Kloeckner & Metzger. *The OpenSky Network: Enabling Large-Scale
    Collection of Air Traffic Data.* SoftwareX 9 (2020) 100263. OpenSky data are subject to
    the [terms of use](https://opensky-network.org/about/terms-of-use) of The OpenSky
    Network Association.

**Previous model of this project**

15. Valle, M. *FlightPREV* — <https://github.com/MarcoValleCS/flight>. Re-implemented and
    compared in §6.5.

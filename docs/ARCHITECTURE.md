# AeroTraj-Ultra — formal architecture specification

This document specifies the model formally: every module, every equation, every
hyper-parameter. It is intended to be readable next to `model.py` (1 277 lines) and
sufficient to re-implement the model from scratch.

Notation follows the standard 4D-trajectory-prediction literature: a state vector
`s_t = [φ, λ, h, v_x, v_y, v_z]` (latitude, longitude, altitude, ground velocity, vertical
rate) and an observation window `x = {s_{t−T+1}, …, s_t}`.

---

## 1. Problem statement

Given a window of `T = 12` ADS-B state vectors sampled every `Δ = 5 s` (60 s of history),
predict the future state at a set of `H` non-uniform horizons

```
h = {5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300} s
```

in the 6-D space `y_k = [λ_k, φ_k, h_k, v_x,k, v_y,k, v_z,k]`.

The prediction is **non-autoregressive**: the whole horizon is produced by one forward pass.
This removes the error accumulation inherent to autoregressive decoders and makes the
latency independent of the horizon.

---

## 2. Input representation

The raw OpenSky state vector is mapped to nine channels:

| index | symbol | unit | definition |
|---|---|---|---|
| 0 | `λ` | deg WGS-84 | longitude |
| 1 | `φ` | deg WGS-84 | latitude |
| 2 | `h` | 10 m | barometric altitude (`ft × 0.03048`) |
| 3 | `v_x` | km/h | eastward ground speed (`gs·sin ψ`) |
| 4 | `v_y` | km/h | northward ground speed (`gs·cos ψ`) |
| 5 | `v_z` | m/s | vertical rate (`vr·0.00508`) |
| 6 | `u` | m/s | eastward wind |
| 7 | `v` | m/s | northward wind |
| 8 | `T` | K | static temperature |

`(v_x, v_y)` are the **onboard Doppler** velocities (ground speed and track angle), not finite
differences of the position: at these altitudes the CPR-quantised position is too noisy to
differentiate (a 60 s difference quotient has a ~12 m/s error).

Normalisation statistics (`μ`, `σ`) are computed **on the training split only** and stored in
the checkpoint; the model registers them as buffers so they travel with the weights.

Two derived quantities are formed:

* **window-relative kinematics** (`rel_x ∈ R^{B×T×6}`)

  ```
  rel_pos = [λ, φ, h]_t − [λ, φ, h]_{T}                     (degrees, degrees, 10 m)
  rel_vel = ([v_x, v_y, v_z]_t − [v_x, v_y, v_z]_{T}) / [200, 200, 10]
  rel_x   = [ rel_pos ⊙ [20, 20, 0.05] , rel_vel ]
  ```

* **standardised raw features** `norm_x = (x − μ) / σ ∈ R^{B×T×9}`.

---

## 3. Gray-Code codec

### 3.1 Lattices

Absolute positions are encoded on a Gray-Code lattice:

| attribute | bits | resolution | range |
|---|---:|---|---|
| `λ` | 19 | 0.001° ≈ 111 m | `[−180, 180]°` |
| `φ` | 18 | 0.001° ≈ 111 m | `[−90, 90]°` |
| `h` | 12 | 1 (10 m) | `[−50, 4050]` (10 m) |
| `v_x, v_y` | 12 | 1 km/h | ±2047 km/h |
| `v_z` | 12 | 0.1 m/s | ±20.4 m/s |

Total **85 bits** for the observation window.

Step-to-step *deltas* use a finer lattice, which is what the decoder actually predicts:

| delta | bits | scale | resolution per 5 s step |
|---|---:|---:|---|
| `Δλ` | 10 | 10 000 | 0.0001° ≈ 11.1 m |
| `Δφ` | 10 | 10 000 | 0.0001° ≈ 11.1 m |
| `Δh` | 9 | 10 | 1 m |
| `Δv_x` | 9 | 2 | 0.5 km/h |
| `Δv_y` | 9 | 2 | 0.5 km/h |
| `Δv_z` | 8 | 10 | 0.1 m/s |

Total **55 bits**. The 10× refinement with respect to FlightBERT's binary encoding is what
removes the high-order bit-flip outliers.

### 3.2 Encoding

For an unsigned attribute of `n` bits, the integer `m` is Gray-coded (`m ← m XOR (m >> 1)`) and
split into bits, most significant first. Signed attributes use a sign bit followed by the
Gray-coded magnitude.

### 3.3 Differentiable decoding (straight-through, prefix parity)

Given per-bit probabilities `g`, the decoder must invert the Gray code. The classical
sequential soft-XOR recurrence is

```
b_0 = g_0 ,   b_i = b_{i−1} + g_i − 2 b_{i−1} g_i          (exact XOR on {0,1}, smooth gradient)
```

Expanding the recurrence shows that `b_i` is the *prefix parity* of `g_0 … g_i`, and the parity
of a set of independent Bernoulli variables equals

$$ b_i=\frac{1}{2}\Big(1-\prod_{j\le i}(1-2g_j)\Big) $$

The two forms are **algebraically identical** (values *and* gradients), but the parity form is
a single `cumprod` instead of ~5 kernel launches per bit: a 1.9× end-to-end training speed-up
with machine-precision equivalence (2e-16), see `docs/EXPERIMENTS.md`.

The straight-through estimator replaces `g` with `g + (1[g ≥ ½] − g).detach()` before the
parity, so the forward pass is exact and the gradient flows to the bit logits.

The decoded magnitude is

$$ \hat v=\Big(1-2\,\mathrm{sign}\Big)\sum_{i=0}^{n-2} b_i\,2^{\,n-2-i} $$

and the *rate* (per 5 s step) is `v̂ / scale`. Multi-horizon decoding accumulates the rates
with the horizon step increments `Δk_h = k_h − k_{h−1}`:

$$ \hat y_h = y_T+\sum_{j\le h}\Delta k_j\,\frac{\hat v_j}{\text{scale}} $$

---

## 4. Attribute Correlation Attention (ACoAtt-V2)

Six attribute embeddings are produced, each from its Gray-Code bit-vector concatenated with
the standardised scalar and the window-relative state:

```
e_lon = GELU( AE_lon([ gc_λ , rel_x_λ , norm_x_λ]) )     AE_lon : R^{21} → R^{24}
e_lat = GELU( AE_lat([ gc_φ , rel_x_φ , norm_x_φ]) )     AE_lat : R^{20} → R^{24}
e_alt = GELU( AE_alt([ gc_h , rel_x_h , norm_x_h]) )     AE_alt : R^{14} → R^{24}
e_vx  = GELU( AE_vx ([ gc_vx, norm_x_vx, norm_x_u]) )    AE_vx  : R^{14} → R^{24}
e_vy  = GELU( AE_vy ([ gc_vy, norm_x_vy, norm_x_v]) )    AE_vy  : R^{14} → R^{24}
e_vz  = GELU( AE_vz ([ gc_vz, norm_x_vz, norm_x_T]) )    AE_vz  : R^{14} → R^{24}
```

The meteorological channel is attached to the *velocity* it physically influences: wind
perturbs the horizontal motion, temperature the vertical one.

Each physically coupled pair is then re-calibrated by a **bidirectional cross-attention**
block (`CoAttModule`, 4 heads, `d = 24`):

```
corr_out , _ = MHA(query = e_p , key = e_c , value = e_c)
joint       = LayerNorm([ e_p ‖ corr_out ])          (R^{48})
e_p^new     = Linear_{48→24}(joint)
```

applied to the six ordered pairs
`(lon↔vx)`, `(lat↔vy)`, `(alt↔vz)` in both directions. The six re-calibrated embeddings are
concatenated and fused:

$$ e=\mathrm{GELU}\big(\mathrm{LayerNorm}(\mathrm{Linear}_{144\to144}([e_{\lambda}^c\|e_{\varphi}^c\|e_h^c\|e_{v_x}^c\|e_{v_y}^c\|e_{v_z}^c])))\in\mathbb R^{B\times T\times144} $$

---

## 5. Encoder

The fused embedding is concatenated with the normalised raw features and the relative
kinematics (`144 + 9 + 6 = 159` channels) and passed through a 1-D convolution stem
(`Conv1d(k=3, padding=1)`, Kaiming init) which acts as a local multi-scale smoother, then a
sinusoidal positional encoding, then a **pre-LayerNorm Transformer encoder**:

* `d_model = 144`, `n_heads = 6`, `2` layers, feed-forward width `3·d_model`, GELU,
  dropout `0.1`, `norm_first = True`.

The encoder output `M ∈ R^{B×T×144}` is summarised by a **softmax attention pooling**
(FeatureAggregation):

$$ c=\sum_{t=1}^{T}\alpha_t M_t,\qquad
\alpha=\mathrm{softmax}\big(W_2\,\mathrm{GELU}(W_1 M)\big)\in\mathbb R^{B\times T} $$

---

## 6. Decoder (non-autoregressive, multi-horizon)

### 6.1 Prefix

The historical step deltas are Gray-Code encoded (55 bits) and concatenated with the
continuous deltas (`[Δλ·100, Δφ·100, Δh·0.2, Δv_x·0.05, Δv_y·0.05, Δv_z·0.5]`, 6 channels) and
embedded by the same 1-D convolution stem (`61 → 144`), giving `T−1 = 11` prefix tokens.

### 6.2 Queries

For every horizon `h` a query is formed from three sources:

* a **learnable horizon embedding** `E_h ∈ R^{144}` (a discrete lookup);
* a **continuous Fourier time feature** of the normalised horizon `τ_h = Δt_h / Δt_max`:

  $$ f(τ)=[\,τ,\ \sqrt{τ},\ \sin(\pi τ),\ \cos(\pi τ)\,]\in\mathbb R^4\xrightarrow{\text{MLP}}\mathbb R^{144} $$
* the encoder context `c`.

They are concatenated and projected: `Linear_{432→144}`.

### 6.3 Decoder stack

The 11 prefix tokens and the `H` query tokens are concatenated (with positional encoding) and
processed by a **pre-LayerNorm Transformer decoder** (`d=144`, 6 heads, 2 layers, FFN `3d`,
dropout 0.1), attending to the encoder memory `M`. The last `H` outputs form
`D ∈ R^{B×H×144}` (after a final LayerNorm).

---

## 7. Physics kinematic prior

A closed-form propagator with **three learned time constants**
(`τ_turn ∈ [10,180] s`, init 45 s; `τ_vz ∈ [30,600] s`, init 180 s; `τ_acc`, init 30 s):

1. **Velocity estimate.** The onboard Doppler velocity is blended with the 20 s position trend:

   $$ w(t)=0.12\big(1-e^{-(t-5)/45}\big),\qquad
   \mathbf v_0=(1-w)\,\mathbf v_{\text{Doppler}}+w\,\frac{\mathbf p_{t}-\mathbf p_{t-4}}{20} $$

2. **Turn rate.** From the last three velocity vectors,
   `ω_0 = wrap(ψ_t − ψ_{t−3}) / 15 s`, clipped to ±0.035 rad/s (≈ ±2°/s).

3. **Damped coordinated turn.** With `θ(t) = ω_0 τ_turn (1 − e^{−t/τ_turn})` and
   `θ̄(t) = θ(t)/2` (mean heading rotation over the interval):

   $$ \mathbf v(t)=R(\pm\bar\theta)\,\mathbf v_0,\qquad
   \Delta \mathbf p(t)=\mathbf v_0\,t+R(\pm\bar\theta/2)\,\mathbf v_0\,t $$

4. **Damped vertical rate.** `v_z(t) = v_{z0} e^{−t/τ_vz}`, `Δz(t) = v_{z0} τ_vz (1 − e^{−t/τ_vz})`.

5. **Geodesy.** The displacement is converted to degrees with
   `m/deg_lon = 111320 cos φ` and `m/deg_lat = 111132`.

The prior returns `P ∈ R^{B×H×6}` (`λ, φ, h, v_x, v_y, v_z`).

---

## 8. Heads and fusion

### 8.1 Head A — Gray-Code multi-binary classification

`Linear_{144→55}` → per-attribute sigmoid → STE Gray-Code decode → `pred_gc ∈ R^{B×H×6}`.

### 8.2 Head B — continuous residual on the prior

```
cont_head : Linear_{150→144} → GELU → Linear_{144→6}          (last layer zero-initialised)
```

The input is the concatenation of the decoder output `D` and the *kinematic displacement*
`(P − y_T) ⊘ [1, 1, 100, 200, 200, 10]`, which tells the network how far the prior believes
the aircraft will have moved.

The residual is scaled by a horizon-dependent factor
`t ⊙ [0.25°, 0.25°, 35, 80, 80, 6]` where `t = [√τ_h, √τ_h, √τ_h]` and `τ_h = Δt_h/Δt_max`, so
the correction budget grows with the horizon.

**The last layer is initialised to zero**, hence at initialisation `pred_cont = P` exactly:
the network starts from the physics solution and only learns a correction.

### 8.3 Head C — MAF-Net differentiable trust region

```
α = σ( MLP([D , kin_disp]) ) ∈ R^{B×H×6}          MLP: 150 → 64 → 6, bias init −2 (α₀ ≈ 0.12)
σ_h = σ_base ⊙ (0.02 + 0.98 τ_h)                  σ_base = [0.008°, 0.008°, 2.5, 15, 15, 1.5]
```

and the final prediction is the trust-region interpolation

$$ \hat y=\hat y_{\text{cont}}+\alpha\odot\tanh\!\Big(\frac{\hat y_{\text{gc}}-\hat y_{\text{cont}}}{\sigma_h}\Big)\odot\sigma_h $$

The `tanh` bounds the influence of the (coarser) Gray-Code branch to ±σ per attribute while
keeping the gradient everywhere — no clipping, no dead units.

---

## 9. Training objective

$$
\mathcal L=\underbrace{2.5}_{\text{pos}}\mathcal L_{\text{pos}}
+\underbrace{0.6}_{\text{vel}}\mathcal L_{\text{vel}}
+\underbrace{0.4}_{\text{bce}}\mathcal L_{\text{BCE}}
+\underbrace{0.25}_{\text{STE}}\mathcal L_{\text{STE}}
+\underbrace{0.25}_{\text{phys}}\mathcal L_{\text{kin}}
$$

with

* `L_pos` = Smooth-L1(β=0.25) of `(ŷ_pos − y_pos) / (σ_attr ⊙ h_norm)` with
  `σ_attr = [0.02°, 0.02°, 4, …]` and the **horizon normalisation**
  `h_norm = 0.04 + 0.96·(Δt_h / 60 s)`, which prevents the long horizons from dominating;
* `L_vel` = Smooth-L1(β=0.5) of `(ŷ_vel − y_vel) / [15, 15, 1.5]`;
* `L_BCE` = mean over the 6 attributes of the binary cross-entropy between the 55 predicted
  bit logits and the Gray-Code encoding of the true step deltas;
* `L_STE` = Smooth-L1(β=0.5) of the *decoded* (STE) Gray-Code trajectory against the truth,
  normalised by `clamp(Δk_h/12, ½)` — this is the **Hybrid Constrained Loss** of FlightBERT,
  which is only possible because the decoder is differentiable;
* `L_kin` = the **kinematic consistency** term: for every consecutive pair of horizons the
  predicted position increment must match the trapezoidal integral of the predicted velocities,

  $$ \varepsilon_x=\frac{\Delta x-\tfrac12(v_x^{(h)}+v_x^{(h+1)})\Delta t_h}{50+2\Delta t_h} $$

  and analogously for `y` and `z` (with `15 + 0.5 Δt_h` in the vertical channel).

---

## 10. Parameter budget

| module | parameters | share |
|---|---:|---:|
| decoder (2 layers + prefix/query projections) | 585 792 | 45.9 % |
| encoder (2 layers) | 418 176 | 32.7 % |
| conv stem (159→144) | 68 832 | 5.4 % |
| horizon query projection | 62 352 | 4.9 % |
| ACoAtt-V2 embedding | 45 672 | 3.6 % |
| decoder prefix embedding | 26 496 | 2.1 % |
| continuous residual head | 22 614 | 1.8 % |
| Fourier time MLP | 21 600 | 1.7 % |
| MAF-Net gate | 10 054 | 0.8 % |
| Gray-Code head | 7 975 | 0.6 % |
| attention feature aggregation | 5 257 | 0.4 % |
| horizon embedding | 1 872 | 0.1 % |
| kinematic prior (3 learned scalars) | 3 | 0.0 % |
| **total** | **1 276 983** | 100 % |

The three learned time constants of the physics prior are the *only* parameters of the
prior: the propagator itself is parameter-free, which is what makes it usable as an
interpretable reference and as a residual anchor.

---

## 11. Hyper-parameters

| group | value |
|---|---|
| window | `T = 12` samples × 5 s |
| horizons | 13 points, 5 s → 300 s (or 16 points, 5 s → 600 s) |
| `d_model` / heads / layers | 144 / 6 / 2 + 2 |
| `attr_embed_dim` | 24 |
| dropout | 0.1 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8 |
| schedule | OneCycleLR, `pct_start` 0.2, cosine anneal |
| batch | 2 048 – 4 096 |
| epochs | 30 – 80 (early stopping on validation MDE) |
| loss weights | `cont` 2.5, `vel` 0.6, `bce` 0.4, `gc_ste` 0.25, `physics` 0.25 |

---

## 12. Complexity

The model is **launch-bound** rather than compute-bound. Per forward pass: ≈ 600 CUDA kernels,
≈ 0.1 GFLOP at batch 2 048. The Gray-Code decoder alone accounted for ≈ 330 launches before
the prefix-parity reformulation of §3.3.

With the optimisations of `docs/EXPERIMENTS.md` (prefix parity, removal of the per-step
GPU→CPU synchronisations, constant tables as buffers, `torch.compile`), the step time drops
from 144 ms to 75 ms at batch 2 048 on an RTX 4070 — a 1.9× speed-up with machine-precision
equivalence.

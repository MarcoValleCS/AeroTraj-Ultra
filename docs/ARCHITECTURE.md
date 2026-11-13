# AeroTraj-Ultra — formal architecture specification

This document specifies the model formally: every module, every equation, every
hyper-parameter. It is intended to be readable next to `model.py` (1 300+ lines) and
sufficient to re-implement the model from scratch.

**The model is grid-parametric.** Two configurations are trained and reported, and the
only differences between them are the time step, the horizon set and the width of the
delta lattice:

| | 5 s grid | 10 s grid |
|---|---|---|
| config | `config_365d.json` | `config_365d_10s.json` |
| step `Δ` | 5 s | **10 s** |
| window | 12 points = 60 s | 12 points = **120 s** |
| horizons `H` | 13, 5 → 300 s | 6, 10 → 300 s |
| absolute lattice | 85 bits | 85 bits |
| delta lattice | 55 bits | **58 bits** |

Every equation below is written for the 5 s grid and holds verbatim for the 10 s one,
with `Δ`, `H` and the delta widths taken from the configuration. `config_utils.resolve_horizon_steps()`
derives the step indices from the horizon *seconds* and verifies the declared arrays, so
the time step is a parameter and never a constant.

Notation follows the standard 4D-trajectory-prediction literature: a state vector
`s_t = [λ, φ, h, v_x, v_y, v_z]` (longitude, latitude, altitude, ground velocity, vertical
rate) and an observation window `x = {s_{t−T+1}, …, s_t}`.

**AeroTraj-Atlas** — the most accurate model measured on the 365-day test split — is this model plus a learned airspace
field read along the kinematic prior (10 495 863 parameters, of which 9 134 208 are the map):
see §15.

---

## 1. Problem statement

Given a window of `T = 12` ADS-B state vectors sampled every `Δ` seconds, predict the
future state at a set of `H` non-uniform horizons

```
5 s grid   h = {5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300} s   (H = 13)
10 s grid  h = {10, 30, 60, 90, 180, 300} s                                 (H = 6)
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

| delta | bits (5 s) | bits (10 s) | scale | resolution per step (5 s grid) |
|---|---:|---:|---:|---|
| `Δλ` | 10 | **11** | 10 000 | 0.0001° ≈ 11.1 m |
| `Δφ` | 10 | **11** | 10 000 | 0.0001° ≈ 11.1 m |
| `Δh` | 9 | **10** | 10 | 1 m |
| `Δv_x` | 9 | 9 | 2 | 0.5 km/h |
| `Δv_y` | 9 | 9 | 2 | 0.5 km/h |
| `Δv_z` | 8 | 8 | 10 | 0.1 m/s |

Total **55 bits** on the 5 s grid and **58 bits** on the 10 s one.  The 10× refinement with
respect to FlightBERT's binary encoding is what removes the high-order bit-flip outliers;
the three extra bits on the coarse grid are what keeps a doubled step displacement inside
the lattice (verified: worst-case saturation 0.21 %, `verify_10s.py`).

Because the lattice quantises the per-step *rate*, the cumulative position quantisation
grows with the number of steps: ±0.5 code unit per step gives a worst case of
`max_k · 0.5 / scale` — 5.6 m at the 10 s horizon, 167 m at 300 s on the 10 s grid.

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

## 9. Multi-hypothesis head (optional)

The dominant residual error at 60–300 s is **epistemic**: the aircraft's *intent* (which
manoeuvre) is not observable from a 60 s window. Generative models capture this by sampling a
distribution; this model captures it with an additive head, at a fraction of the cost.

**Architecture.** `K` learnable *mode queries* `e_1..e_K ∈ R^{D}` (DETR/ASCENT style) are
appended to the decoder input sequence:

```
dec_in = [ prefix(11) ‖ horizon_queries(H) ‖ mode_queries(K) ]
```

A **float attention mask** `M ∈ R^{L×L}` (0 = attend, −∞ = block) is applied to the decoder
self-attention:

| rows \ columns | prefix | horizon | modes |
|---|---|---|---|
| prefix | ✓ | ✓ | ✗ |
| horizon | ✓ | ✓ | ✗ |
| modes | ✓ | ✓ | ✓ |

Because the horizon rows cannot attend to the mode columns, the *deterministic* pathway is
**mathematically unchanged** (verified: max deviation 1.9e-6, i.e. float32 rounding). The mode
tokens are the only ones that see each other, so the K hypotheses cannot influence the
operational prediction.

**Decoding.** The mode token `m_m ∈ R^{D}` acts as a per-mode bias on the horizon features:

$$ z_{m,h}=D_h+m_m,\qquad
\hat y^{m}_h=P_h+\big(\mathrm{MLP}_{hyp}([z_{m,h}\,\|\,\kappa_h])\odot s_h\big) $$

where `P_h` is the physics prior, `κ_h` the kinematic displacement and `s_h` the same
horizon-dependent scale used by the continuous residual head. `MLP_hyp` is
`Linear_{D+6→D} → GELU → Linear_{D→6}` with a **zero-initialised output layer**, so every
hypothesis starts exactly at the physics solution. The hypotheses are therefore kinematically
consistent by construction, like the deterministic output.

A **score head** `MLP_{score}: D → 1` (bias initialised at −2, i.e. `α₀ ≈ 0.12`) produces a
confidence per hypothesis.

**Loss (winner-takes-all).** With `D(m)` the 3-D distance of hypothesis `m` to the truth:

$$ m^\star=\arg\min_m \frac{1}{H}\sum_h D(m)_h $$

$$ \mathcal L_{hyp}=\underbrace{\mathcal L_{reg}(\hat y^{m^\star},y)}_{\text{only the winner}}
+\underbrace{0.1}_{\text{aux}}\frac{1}{K}\sum_m \mathcal L_{reg}(\hat y^{m},y)
+\underbrace{0.1}_{\text{rank}}\mathrm{CE}\big(\alpha,\,m^\star\big) $$

The auxiliary term is essential: without it the `K−1` losing hypotheses receive no gradient
and degenerate (observed: the mean-over-modes error collapses onto the prior). The total
objective becomes `L = L_hybrid + 0.5 · L_hyp`.

**Cost and result** (90 days, 5–300 s, RTX 4070):

| | deterministic | parameters | s/epoch | oracle minADE@8 |
|---|---:|---:|---:|---:|
| without the head | 991,6 m | 1 276 839 | 28,3 | — |
| **with the head (K=8)** | **984,9 m** | 1 321 774 (+3,5 %) | 28,5–31 (+3 %) | **627,4 m** |

The oracle gain is 36 % below the deterministic prediction and better than every generative
baseline benchmarked (CFM 983,5 m, MC-dropout 905,4 m, ASCENT 1 088,8 m, DDPM 1 763,6 m).
The score head is **not yet calibrated** (score-selected 1 098,7 m, worse than the
deterministic output), so the oracle gain is not yet operationally exploitable.

**Certified uncertainty.** Treating the `K` hypotheses as an ensemble and inflating it with a
Mondrian (flight-phase stratified) conformal radius calibrated on the validation split gives
an *exact* and *horizon-uniform* coverage with a radius ≈47 % smaller than inflating the
single deterministic prediction:

| nominal | coverage | mean radius (ensemble) | mean radius (deterministic) |
|---:|---:|---:|---:|
| 80 % | 0,805 | 787 m | 1 282 m |
| 90 % | 0,903 | 1 413 m | 2 675 m |
| 95 % | 0,951 | 2 411 m | 4 542 m |

The nonconformity score is the 3-D distance of the truth to the *closest* hypothesis,
quantile-calibrated per stratum (climb / cruise / descent) at level `1−α`. See
`calibrate_uncertainty.py` and `docs/EXPERIMENTS.md` §6.6.

---

## 10. Retrieval-conditioned variant (medium-term intent prior)

At 60–300 s the residual error is not sensor noise but **intent**: the manoeuvre the aircraft
will fly is not observable from a 60 s window. `AeroTrajUltraRetrieval` adds a non-parametric
prior over past trajectories.

**Retrieval key.** Every observation is described by a 6-D descriptor

$$ d=\big[\ \lambda,\ \varphi,\ h/500,\ \sin\psi,\ \cos\psi,\ |v|/50\ \big] $$

(position, flight level, heading, speed). The absolute position can be down-weighted
(`pos_weight`): the diagnostic sweep showed that dividing it by 10 — turning the key from a
*route identity* into a *neighbourhood prior* — raises the oracle from 981 m to 900 m.

**Retrieved value.** For every candidate the future is stored in the candidate's **own** local
frame (forward, right, up in metres) together with its ground speed and heading.

**Query.** The `K` nearest candidates are found on the GPU with the
`|q|² − 2qc + |c|²` expansion, their frames are **rotated into the query's heading frame**
(so the context is directly comparable with the query's own kinematics) and scaled
(positions ÷1000, speed ÷100) to keep the conditioning O(1).

**Conditioning.** The context is encoded per neighbour (`Linear_{4→D}`) and pooled over the
neighbours with a learned attention; the result is appended to the horizon-query input:

$$ q_h=\mathrm{Linear}_{4D\to D}\big[\,c\ \Vert\ e_h\ \Vert\ \phi(t_h)\ \Vert\ \rho_h\,\big] $$

where `ρ_h` is the retrieval context. The `tanh`-free, purely additive nature of the block
means the deterministic path is otherwise untouched.

**Warm start.** `load_base_weights` copies a base checkpoint verbatim except `horizon_proj`,
whose input grew from `3D` to `4D`: the first three blocks keep the base weights and the
retrieval block starts at zero. The model is therefore **exactly** the base model at
initialisation (verified: max deviation 0.0 on random inputs with a random context), and any
improvement is attributable to the retrieval context alone.

**Leakage control.** At training time the candidates belonging to the *same flight* as the query
are excluded from the retrieval. Without this the model learns to copy the future of a
near-duplicate window of the same trajectory, which is impossible at inference (the pool holds
only past data). The exclusion is vectorised through a CSR-style flight index.

**Measured effect** (90 days, 5–300 s, 342 572 candidate windows from 75 624 flights):

| Variant | MDE | minADE@8 | 60 s | 120 s | 300 s |
|---|---:|---:|---:|---:|---:|
| base | 991,6 | — | 316,1 | 977,4 | 4 357,5 |
| + multi-hypothesis | 984,9 | 627,4 | 304,6 | 944,2 | 4 169,1 |
| + retrieval | 968,9 | — | 302,8 | 932,5 | 4 138,9 |
| **+ both** | **957,2** | **608,2** | **297,7** | **918,5** | **4 092,5** |

The gain is monotone in the horizon (−0,3 % at 5 s, −5,0 % at 300 s): a prior on intent is
irrelevant where physics dominates and decisive where the manoeuvre is unknown.

**Cost**: +66 536 parameters (+5,2 %) and +15 s/epoch (the retrieval is a GPU top-k over
342 572 candidates).

---

## 11. Training objective

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

## 12. Parameter budget

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
| *(+ multi-hypothesis head, K=8)* | *+44 791* | *+3,5 %* |
| *(AeroTraj-Atlas: + airspace field, §15.5)* | *+9 218 880 (total 10 495 863)* | *of which map 9 134 208* |

The three learned time constants of the physics prior are the *only* parameters of the
prior: the propagator itself is parameter-free, which is what makes it usable as an
interpretable reference and as a residual anchor.

---

## 13. Hyper-parameters

| group | value |
|---|---|
| window | `T = 12` samples; 5 s (60 s of history) or 10 s (120 s) |
| horizons | 13 points, 5 s → 300 s (5 s grid) · 6 points, 10 s → 300 s (10 s grid) · 16 points, 5 s → 600 s (the 90-day variant) |
| `d_model` / heads / layers | 144 / 6 / 2 + 2 |
| `attr_embed_dim` | 24 |
| dropout | 0.1 |
| optimiser | AdamW, lr 1e-3, weight decay 1e-4, betas (0.9, 0.999), eps 1e-8 |
| schedule | OneCycleLR, `pct_start` 0.2, cosine anneal |
| batch | 2 048 – 4 096 |
| epochs | 30 – 150 (early stopping on validation MDE) |
| loss weights | `cont` 2.5, `vel` 0.6, `bce` 0.4, `gc_ste` 0.25, `physics` 0.25 |
| airspace field (AeroTraj-Atlas, §15.8) | 6 levels, 0.5 → 16 cells/deg, 8 heading sectors, 5 altitude nodes, 8 features, table `2^18`, gain 10, read ahead, `refine` (aux. weight 0.5 · `cont`) |

---

## 14. Complexity

The model is **launch-bound** rather than compute-bound. Per forward pass: ≈ 600 CUDA kernels,
≈ 0.1 GFLOP at batch 2 048. The Gray-Code decoder alone accounted for ≈ 330 launches before
the prefix-parity reformulation of §3.3.

With the optimisations of `docs/EXPERIMENTS.md` (prefix parity, removal of the per-step
GPU→CPU synchronisations, constant tables as buffers, `torch.compile`), the step time drops
from 144 ms to 75 ms at batch 2 048 on an RTX 4070 — a 1.9× speed-up with machine-precision
equivalence.

**AeroTraj-Atlas** (§15) adds three reads of the airspace field per forward pass, each an
O(1) gather of 16 table rows per level (no neighbour search). Measured on an RTX 4070 with
`FastPredictor`, input already on the GPU (`results_atlas/latency.json`; this harness does
not include the host-to-device copy):

| | AeroTraj-Ultra | AeroTraj-Atlas |
|---|---:|---:|
| 1 aircraft, eager | 17.4 ms | 42.2 ms (2.4×) |
| 1 aircraft, `torch.compile(reduce-overhead)` | 0.65 ms | 0.66 ms |
| batch 4 096, eager | 54.4 ms (75 348 aircraft/s) | 78.5 ms (52 172 aircraft/s) |
| batch 4 096, compiled | 41.1 ms (99 780 aircraft/s) | 44.8 ms (91 355 aircraft/s) |

Training: 53.2 s/epoch against 47.1 s/epoch compiled (1.13×) at the full budget.

---

## 15. Airspace field (AeroTraj-Atlas)

Above ~30 s the residual error is *intent* (the next manoeuvre), which is not observable from
the history. Road-agent predictors read it from the HD map (dynamic map collection, MTR); the
route structure of the airspace is not in the ADS-B data, so here the map is **learned** from
the trajectories and read along the trajectory the aircraft is about to fly.

AeroTraj-Atlas is not a new class: it is `AeroTrajUltra` (`model.py`) with the configuration
key `model.airspace_field`, a dict of `AirspaceField` keyword arguments plus the two switches
`ahead` and `refine`. The final model sets `{"n_alt": 5, "refine": true}`. Without the key the
model is AeroTraj-Ultra bit for bit (identical state dict and outputs; every existing
checkpoint loads).

### 15.1 The field

**Region.** The dataset bounding box `data.region_bounds`: `λ ∈ [−15, 35]°`, `φ ∈ [30, 65]°`.
A point outside the region is read at the border.

**Levels.** `l = 0 … L−1`, `L = 6`, resolution `r_l = 0.5 · 2^l` cells per degree
(0.5, 1, 2, 4, 8, 16: cells from 2° down to 0.0625°, about 7 km north–south). Grid corners per
level:

$$ n_x^{(l)}=\lceil 50\,r_l\rceil+1,\qquad n_y^{(l)}=\lceil 35\,r_l\rceil+1 $$

with the continuous grid coordinates `x = (λ − λ_min) r_l`, `y = (φ − φ_min) r_l` (clamped to
the region).

**Heading axis.** `S = 8` sectors, **periodic**. With `ψ = atan2(v_x, v_y)` (clockwise from
north), `s = (ψ / 2π mod 1) · S`; the corner `c_s + 1` wraps to `0`.

**Altitude axis** (`n_alt = A`, `0` = no axis). Nodes at

$$ h_k=h_{\max}\Big(\frac{k}{A-1}\Big)^2,\qquad h_{\max}=13\ \text{km}\ (\texttt{alt\_max}=1300\ \text{in 10 m}) $$

i.e. 0, 0.81, 3.25, 7.31, 13 km for `A = 5`: finer near the ground, where the procedures are
stacked. The coordinate is `a = sqrt(clamp(h / h_max, 0, 1)) · (A − 1)`.

**Value.** Each level stores `F = 8` features per node. The value at a point is the
multilinear interpolation of the 16 surrounding nodes (2 λ × 2 φ × 2 heading × 2 altitude):

$$ \mathcal F_l(\lambda,\varphi,\psi,h)=\sum_{c\in\{0,1\}^4} w_x w_y w_s w_a\ T_l[\mathrm{idx}_l(c)],\qquad
\mathcal F=g\,[\mathcal F_0\Vert\dots\Vert\mathcal F_{L-1}]\in\mathbb R^{48} $$

with the usual linear weights (`w = 1 − f` or `f` on each axis), which sum to one, and a fixed
gain `g = 10`.

**Storage.** A level whose `n_x n_y S A` nodes fit `2^18` entries (`log2_table = 18`) is a
**dense** table,

```
idx = c_x + n_x (c_y + n_y (c_s + S c_a))
```

otherwise it is **hashed** (Instant-NGP):

```
idx = ((c_x · 73856093) XOR (c_y · 19349663) XOR (c_s · 83492791) XOR (c_a · 2654435761)) mod 2^18
```

With the final setting levels 0–1 are dense (19 760 and 73 440 entries) and levels 2–5 are
hashed (262 144 each): 1 141 776 entries × 8 = **9 134 208 parameters**. Without the altitude
axis levels 0–3 are dense: 827 024 entries, 6 616 192 parameters.

**Initialisation and gain.** `T_l ~ U(−1e-4, 1e-4)`. Adam moves a weight by about `lr` per step
whatever the gradient scale, so the fixed output gain `g` is a learning-rate multiplier for
the table (the rarely visited fine cells need it), and the optimiser is left untouched.

### 15.2 The three reads

All three reads stop the gradient on the positions (the map is trained, the trajectory it is
read along is not differentiated through it). With `s_T` the last observed state and `P_h` the
physics prior of §7:

1. **At the aircraft**: `f_0 = F(λ_T, φ_T, ψ_T, h_T)`.
2. **Along the prior** (`ahead`, default true): `f_h = F(λ, φ, ψ, h of P_h)` for each of the
   `H` horizons. With `ahead = false`, `f_h = f_0`.
3. **Dynamic collection** (`refine`): the first decoder layer proposes a trajectory with the
   **same** residual head of §8.2,

   $$ y^{(1)}_h=P_h+\mathrm{ContHead}\big([\mathrm{LN}(z^{(1)}_h)\,\Vert\,\kappa_h]\big)\odot s_h $$

   the field is read again along it and added to the horizon tokens before the second
   decoder layer:

   $$ z^{(1)}_h\leftarrow z^{(1)}_h+\mathrm{MLP}_r\big([\mathcal F(y^{(1)}_h)\,\Vert\,\mathrm{state}\,\Vert\,\tau_h]\big) $$

   `MLP_r = Linear_{48+4→144} → GELU → Linear_{144→144}`, last layer initialised at zero.
   `refine` together with `n_modes > 0` raises an error (not implemented).

### 15.3 Query conditioning

$$ q^{\text{field}}_h=\mathrm{MLP}\big([\,f_h\,\Vert\,f_0\,\Vert\,\mathrm{state}\,\Vert\,\tau_h\,]\big),\qquad
\mathrm{MLP}=\mathrm{Linear}_{2\cdot48+4\to144}\to\mathrm{GELU}\to\mathrm{Linear}_{144\to144} $$

with `state = (norm_x_h, norm_x_vz, |v| / 900 km/h)` at the last observation and `τ_h` the
normalised horizon. It is a **fourth block** of the horizon query (§6.2):

$$ q_h=\mathrm{Linear}_{4D\to D}\big[\,c\ \Vert\ e_h\ \Vert\ \phi(t_h)\ \Vert\ q^{\text{field}}_h\,\big] $$

The physics prior is therefore computed **before** the queries (same values as before: only the
order of the computation changes).

### 15.4 Loss

With `refine`, one auxiliary term is added to §11: the positions of the proposed trajectory
`y^(1)` with the same per-horizon normalisation as `L_pos`,

$$ \mathcal L\leftarrow\mathcal L+0.5\cdot2.5\cdot\mathrm{SmoothL1}_{\beta=0.25}\Big(\frac{y^{(1)}_{pos}-y_{pos}}{\sigma_{attr}\odot h_{norm}}\Big) $$

### 15.5 Parameters

| module | parameters |
|---|---:|
| AeroTraj-Ultra (§12) | 1 276 983 |
| field MLP (`Linear_{100→144}`, `Linear_{144→144}`) | 35 424 |
| fourth block of the horizon query projection | 20 736 |
| refine MLP (`Linear_{52→144}`, `Linear_{144→144}`) | 28 512 |
| **network** | **1 361 655** |
| map (6 levels, `n_alt = 5`) | 9 134 208 |
| **total** | **10 495 863** |

9.1 M of the 10.5 M parameters are the map (about 37 MB in float32).

### 15.6 Options measured and not adopted

Both stay in the code as documented ablations; **they are not part of the model**.

* `model.track_frame`: the history is also given in the aircraft frame (along-track /
  cross-track offsets and velocities, 4 more stem channels) and a second residual head
  (`Linear_{D+8→D} → GELU → Linear_{D→4}`, zero-initialised) predicts along/cross-track
  position and velocity residuals as a fraction of the distance flown (of the speed), rotated
  back to `(λ, φ, v_x, v_y)` and added to the residual of §8.2.
* `n_vz`: a vertical-rate axis of the field, `n_vz` nodes over `[−vz_max, vz_max]` m/s,
  read with the **current** vertical rate, to separate departures from arrivals.

### 15.7 Verification

Section 10 of `test_conformance.py` checks:

* with the options off the parameters are those of AeroTraj-Ultra (old checkpoints load);
* the interpolation weights sum to one, also outside the region;
* the heading axis is periodic;
* a grid corner on a sector centre reads its own table row;
* the aircraft-frame head is zero at initialisation, and a pure along-track (cross-track)
  residual moves the point along-track (cross-track) only.

### 15.8 Configuration

| key (`model.airspace_field.*`) | default | final model |
|---|---|---|
| `n_levels` | 6 | 6 |
| `base_res` | 0.5 | 0.5 |
| `growth` | 2.0 | 2.0 |
| `n_sectors` | 8 | 8 |
| `feat_dim` | 8 | 8 |
| `log2_table` | 18 | 18 |
| `gain` | 10.0 | 10.0 |
| `n_alt` | 0 | **5** |
| `alt_max` | 1300.0 | 1300.0 |
| `n_vz` | 0 | 0 |
| `vz_max` | 10.0 | 10.0 |
| `ahead` | true | true |
| `refine` | false | **true** |

Results: `docs/PAPER.md` §6.8, `docs/EXPERIMENTS.md` §6.15.

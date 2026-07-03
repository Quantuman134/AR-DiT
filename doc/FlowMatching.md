# Flow Matching — Mathematical Specification

This document is the **single source of truth** for the flow-matching math
used in this project. It defines the time convention, the interpolant, the
training loss, classifier-free guidance (CFG), exponential moving averages
(EMA), and the Euler sampler. Every piece of code in `flow/`, `train.py`,
and `sample.py` follows this spec; if a formula in the code disagrees with
this document, the document is correct and the code is wrong.

The runtime side of training (configs, checkpoints, DDP, wandb, validation,
metrics) lives in [Train.md](Train.md). This document deals only with the
math.

---

## 1. Time convention — non-standard, read carefully

> **⚠️ This project uses a non-standard time convention.**
>
> | Direction      | This project        | SiT / Lipman et al. |
> |----------------|---------------------|---------------------|
> | `t = 0`        | **pure noise**      | clean data          |
> | `t = 1`        | **clean data**      | pure noise          |
> | Sampling sweep | `t: 0 → 1` (forward integration) | `t: 1 → 0` (reverse integration) |
>
> When cross-referencing SiT, Lipman et al. (2022), or rectified-flow
> papers, mentally substitute `t ← 1 − t`. Everything below is written
> in **this project's convention**.

### Symbols

| Symbol | Meaning                                                  | Shape          |
|--------|----------------------------------------------------------|----------------|
| `x_0`  | Sample from the **noise** distribution `N(0, I)`         | `(B, C, H, W)` |
| `x_1`  | Sample from the **data** distribution (a real image)     | `(B, C, H, W)` |
| `t`    | Continuous time, `t ∈ [0, 1]`                            | `(B,)`         |
| `x_t`      | Interpolant state at time `t`                        | `(B, C, H, W)` |
| `v_gt`     | Ground-truth velocity (target of regression)         | `(B, C, H, W)` |
| `v_θ`      | Network's predicted velocity                         | `(B, C, H, W)` |
| `y`    | Class label (with null-class index for CFG)              | `(B,)`         |

`x_0` is fresh Gaussian noise, **not** the same as the model's input variable
name `x`. Inside `models/dit.py` the input is also called `x`; that `x` is
this document's `x_t`.

---

## 2. Interpolant and ground-truth velocity

We use the **straight-line (rectified-flow) interpolant**:

```
x_t = t · x_1 + (1 − t) · x_0          # shape (B, C, H, W)
```

So at `t = 0` we have `x_t = x_0` (pure noise) and at `t = 1` we have
`x_t = x_1` (clean data) — consistent with §1.

The ground-truth velocity is the time derivative of the interpolant:

```
v_gt = dx_t / dt = x_1 − x_0           # shape (B, C, H, W)
```

This is **constant along the linear path** for each `(x_0, x_1)` pair —
that is the whole point of rectified flow.

Note the sign relative to SiT-style code: there, with `t=1` being noise,
the target is `x_0 − x_1` (data − noise) too, but their `x_0` is data and
their `x_1` is noise. Same physical quantity, opposite labelling.

---

## 3. Training loss

Given a data batch `(x_1, y)`, draw fresh `x_0 ∼ N(0, I)` and
`t ∼ U(0, 1)` independently per sample. Then:

```
x_t  = t · x_1 + (1 − t) · x_0
v_gt = x_1 − x_0
ŷ    = drop(y, p_drop)                      # CFG label dropout (§4)
L    = ‖ v_θ(x_t, t, ŷ) − v_gt ‖²          # mean over all dims
```

We use **mean squared error reduced over every element** (batch, channel,
height, width). No timestep-dependent weighting — the rectified-flow
literature shows that uniform `t` + uniform weighting is a strong default
and we want to keep the loss as plain as possible so any quality
difference between DiT and AR-DiT is attributable to the architecture.

### Why uniform `t ∼ U(0, 1)`

Rectified-flow papers (Liu et al. 2022; Lipman et al. 2022) observe that
the straight-line interpolant has a constant target velocity along each
trajectory, which in turn makes `t ∼ U(0, 1)` near-optimal in the sense
that no time region dominates the loss landscape. We adopt this without
further tuning.

---

## 4. Classifier-free guidance (CFG)

### 4.1 Training-time label dropout

During training, **per sample independently**, replace `y` with the
null-class index with probability `p_drop` (default `0.1`):

```
ŷ_i = null    with probability p_drop
ŷ_i = y_i     with probability 1 − p_drop
```

This dropout is implemented inside `LabelEmbedder` (see
[`models/dit.py`](../models/dit.py)). At eval time `p_drop = 0` and `ŷ = y`.

### 4.2 Sampling-time guided velocity

Following the standard CFG recipe, at every Euler step we evaluate the
model **twice** — once with the real label, once with the null label —
and combine:

```
v_cond   = v_θ(x_t, t, y)
v_uncond = v_θ(x_t, t, null)
v_cfg    = v_uncond + s · (v_cond − v_uncond)
```

where `s ≥ 1` is the **CFG scale**. Special cases:

- `s = 1.0` ⇒ `v_cfg = v_cond` (no guidance, conditional sampling).
- `s = 0.0` ⇒ `v_cfg = v_uncond` (unconditional sampling).
- Typical values: `s ∈ {1.0, 1.5, 2.5, 4.0}` for ablation grids.

### 4.3 Implementation note — single forward via `forward(...)`

Per the design decision recorded in [Plan.md](Plan.md), the model exposes a
**single** `forward(x, t, y)` and the sampler concatenates the conditional
and unconditional batches itself:

```python
x_in = torch.cat([x_t, x_t], dim=0)            # (2B, C, H, W)
t_in = torch.cat([t,   t  ], dim=0)            # (2B,)
y_in = torch.cat([y,   y_null], dim=0)         # (2B,)
v    = model(x_in, t_in, y_in)
v_cond, v_uncond = v.chunk(2, dim=0)
v_cfg = v_uncond + s * (v_cond - v_uncond)
```

`y_null` is a `(B,)` tensor of the null-class index. There is no
`forward_with_cfg` helper.

---

## 5. Exponential moving averages (EMA)

Throughout training we keep one or more **shadow copies** of the model
weights, updated each optimiser step:

```
θ_ema ← β · θ_ema + (1 − β) · θ
```

where `θ` is the current online parameter and `β ∈ [0, 1)` is the EMA
decay. Applied to **every** parameter (no exclusion list). Buffers
(BatchNorm running stats etc.) are not relevant for this model — DiT has
no such buffers — so EMA touches parameters only.

This project supports **multiple EMA copies in parallel**, one per decay
value. A typical config is `decays = [0.9999, 0.999]`. At validation /
sampling time, each EMA copy is evaluated independently; the FID/IS
table reports a row per (EMA, CFG-scale) pair.

### Initialisation and warmup

- Each EMA copy is initialised as a deep copy of `θ` immediately after
  model construction (before optimiser step 0).
- We do **not** use bias-corrected EMA (`θ_ema / (1 − β^step)`); the
  warmup transient is negligible after a few thousand steps and the bias
  correction is more confusing than useful for our purposes.
- EMA updates run in `torch.no_grad()` and on the same device as the
  online model. Under DDP, only **rank 0** holds the EMA copies and runs
  the update (using the unwrapped `model.module` parameters).

---

## 6. Euler sampler

Given a starting noise `x_0 ∼ N(0, I)`, a class label `y`, a CFG scale
`s`, and a number of steps `N`, the sampler integrates the velocity
field forward:

```python
def sample(model, y, num_steps=N, guidance_scale=s):
    x = torch.randn(B, C, H, W)                 # x_0 (noise)
    ts = torch.linspace(0.0, 1.0, N + 1)        # t_0=0, ..., t_N=1
    for k in range(N):
        t_k  = ts[k].expand(B)
        dt   = ts[k+1] - ts[k]                   # = 1/N for uniform grid
        v    = guided_velocity(model, x, t_k, y, guidance_scale)
        x    = x + dt * v                        # forward Euler step
    return x                                     # x_1 (clean sample)
```

Notes:

- The grid is **uniform**: `t_k = k / N`. We do not use a logit-normal or
  shifted schedule — see §7.
- Step direction is `+dt`, not `-dt`, because we integrate from
  `t = 0` (noise) to `t = 1` (data) — see §1.
- `guided_velocity` is the §4.2 combination, executed via the
  doubled-batch trick of §4.3.
- `B` (the number of samples per call) is independent of training batch
  size and is set by the sampling config.

### Why Euler-only

This project compares architectures (DiT vs. AR-DiT) under matched
compute. The sampler is shared across both arms, so it cannot bias the
comparison. Euler is the canonical baseline in the flow-matching
literature, has no implementation pitfalls, and is the cheapest to test.
Higher-order solvers (Heun, RK4, DPM-Solver) are out of scope; if
needed for a final headline-FID number, they can be added in a separate
file without touching anything else.

---

## 7. Choices we deliberately do *not* make

These are options the literature offers that we have considered and
rejected for this project's scope. Listed here so future-us doesn't have
to re-derive the reasoning.

| Choice | Rejected because |
|--------|------------------|
| Logit-normal `t` schedule (SD3) | Adds a tunable hyper-parameter that confounds the architecture comparison. |
| Loss reweighting `w(t)·‖v_θ−v_gt‖²` | Same — extra knob, no scientific gain for this question. |
| Shifted noise schedule (Karras et al.) | Same. |
| Heun / DPM-Solver | Cosmetic FID gain, doubles sampler test surface, no effect on the DiT-vs-AR-DiT delta. |
| `v_θ` with explicit variance head | Flow matching has no stochastic component to predict variance for. |
| Score-based parameterisation `s_θ` | Velocity prediction is the rectified-flow standard and matches SiT. Switching would require re-deriving the loss and CFG. |

If any of these become relevant later, this table is the place to update.

---

## 8. Reference values for tests

Tiny, hand-computable cases used by [`tests/test_flow.py`](../tests/test_flow.py).
These are part of the spec because the document defines the math; the
tests just check the implementation matches.

### 8.1 Interpolant at `t = 0` and `t = 1`

For any `x_0`, `x_1`:
- `interpolant(x_0, x_1, t=0) == x_0` exactly.
- `interpolant(x_0, x_1, t=1) == x_1` exactly.

### 8.2 Velocity target

For any `x_0`, `x_1`:
- `velocity_gt(x_0, x_1) == x_1 − x_0` exactly, independent of `t`.

### 8.3 CFG combination

With `v_cond = a`, `v_uncond = b`, scale `s`:
- `s = 1.0` ⇒ `v_cfg == a`.
- `s = 0.0` ⇒ `v_cfg == b`.
- `s = 2.0` ⇒ `v_cfg == 2a − b`.

### 8.4 EMA update

With `θ = 1.0`, `θ_ema = 0.0`, decay `β = 0.9`:
- After 1 update: `θ_ema == 0.1`.
- After 2 updates (with same `θ = 1.0`): `θ_ema == 0.19`.

### 8.5 Euler sampler — zero velocity

If the model returns `0` everywhere, the sampler returns the initial
noise unchanged: `sample(...) == x_0` (bit-exact).

### 8.6 Euler sampler — constant velocity

If the model returns a constant `v` everywhere (independent of `x`, `t`,
`y`), then `sample` with `N` uniform steps from `t=0` to `t=1` returns
`x_0 + v` (bit-exact, because `Σ dt · v = (Σ dt) · v = 1 · v`).

These six properties uniquely pin down the implementation up to
floating-point order-of-operations, and are sufficient to catch every
sign error, off-by-one, and time-direction mistake we have ever seen in
flow-matching code.

---

## 9. References

- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative
  Modeling.* ICLR 2023.
- Liu, Gong, Liu. *Flow straight and fast: learning to generate and
  transfer data with rectified flow.* ICLR 2023.
- Ma, Goldstein, Albergo, Boffi, Vanden-Eijnden, Xie. *SiT: Exploring
  Flow and Diffusion-based Generative Models with Scalable Interpolant
  Transformers.* arXiv:2401.08740, 2024.
- Ho & Salimans. *Classifier-Free Diffusion Guidance.* NeurIPS 2021
  workshop.

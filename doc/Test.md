# Testing

> ⚠️ **STATUS: NOT YET REVIEWED.**
> All test files written so far — Layer 1 (`test_components.py`),
> Layer 2 (`test_dit.py`), and the Layer 3 suites that land alongside
> Phase B (`test_flow.py`, and the upcoming `test_eval.py`,
> `test_dataset.py`) — currently pass, **but none of the test code has
> been reviewed by the project owner yet**. Treat the suite as
> *provisional*: a passing run only means the tests are internally
> consistent with the implementation under test, not that the right
> things are being tested. The plan is to **review and build other
> parts of the project first** (AR-DiT, flow-matching, training loop,
> data, eval); the test suite will be reviewed in a dedicated pass
> afterwards. Until that review:
>
> - Do **not** treat green CI as evidence of correctness.
> - Do **not** lock in any of the hand-computed numerical values as
>   "golden" — they may be revised during review.
> - When changing a component, you may still update its test, but flag
>   the change in the commit message so the upcoming review can pick it
>   up.
> - When a new Phase B / C module lands (e.g. `flow/loss.py`), its
>   tests are added alongside it under this same provisional status.

This project uses **pytest** as its test runner. Tests live in `tests/` and
are organised into four layers (1–2 already exist; 3–4 land in Phase B/C
alongside the flow-matching code).

## Test data — fully self-contained, no real datasets

Unit tests **never** download or require a real dataset. Two strategies:

1. **Synthetic tensors via `conftest.py` fixtures** — used by the vast
   majority of tests. A fixture produces a `torch.Tensor` of the desired
   shape (e.g. `(B, 3, 32, 32)`) with a fixed RNG seed. This is hermetic,
   millisecond-fast, and ideal for shape/gradient/numeric assertions where
   the *content* of the image is irrelevant.
2. **Tiny committed PNGs under `tests/fixtures/images/`** — used by
   exactly one test (`test_dataset.py`). Four tiny PNG files (~5 KB each)
   are checked into the repo so that the CIFAR-10 dataset/transform code
   can be exercised end-to-end on real encoded image bytes without
   needing the actual CIFAR-10 archive.

Real CIFAR-10 (and any other dataset) is only loaded by `train.py` /
`sample.py`, never by `pytest`.

## Philosophy

Tests fall on a spectrum from *behavioural* to *numerical*:

- **Behavioural / property tests** check that a component obeys an invariant
  (output shape, finiteness, equivariance, identity-at-init, gradient flow).
  They are robust to refactors but blind to silent numerical bugs.
- **Numerical-value tests** assert exact (or near-exact) output values for
  hand-computed inputs. They are the strongest unit tests we have, and we
  use them wherever the function is **deterministic and parameter-free**
  (no random weights to worry about).

We use numerical-value tests for `modulate`, `timestep_embedding`,
`get_2d_sincos_pos_embed`, and `unpatchify`. Everything else is covered
by behavioural tests because it depends on randomly initialised weights.

## Test layers

### Layer 1 — Component tests (`tests/test_components.py`)

Per-module unit tests. Each helper / `nn.Module` defined in
[models/dit.py](../models/dit.py) gets one or more focused tests:

| Component | Test type | What it checks |
|---|---|---|
| `modulate` | numerical | `x * (1 + scale) + shift` element-wise on a hand-laid-out example. |
| `get_2d_sincos_pos_embed` | shape + numerical | Shape `(N, D)`. First row at position `(0, 0)` is `[sin(0)…cos(0)…sin(0)…cos(0)…]`, i.e. the `sin` slots are 0 and the `cos` slots are 1. |
| `TimestepEmbedder.timestep_embedding` | numerical | Hand-computed values at `t=0` and `t=1` for `dim=4`; padding branch when `dim` is odd. |
| `TimestepEmbedder` | shape + grad | `(B,) → (B, hidden_size)`; gradients flow to MLP weights. |
| `LabelEmbedder` | behavioural | (i) `train=False, p_drop=0` is deterministic; (ii) when `force_drop_ids` is all-1s, every label maps to the null-token row of the table; (iii) shape is `(B, hidden_size)`. |
| `PatchEmbed` | shape + asserts | Output shape is `(B, N, D)` with `N = (H/P)*(W/P)`; raises when `H` or `W` is not divisible by `P`. |
| `Attention` | shape + grad | Output shape matches input; gradients flow to `qkv` and `proj`. |
| `MLP` | shape + grad | Output shape matches input; gradients flow. |
| `DiTBlock` | shape + identity-at-init | Output shape matches input; with `c=0` and adaLN-Zero init, the block reduces to the identity (`out == x`). |
| `FinalLayer` | shape + zero-at-init | Output shape `(B, N, P*P*C_out)`; with zero-init `linear` the output is exactly zero. |

### Layer 2 — Whole-model tests (`tests/test_dit.py`)

End-to-end tests on the assembled `DiT` model:

| Test | What it checks |
|---|---|
| `test_presets_construct` | All four presets (`DiT_S_2`, `DiT_B_2`, `DiT_L_2`, `DiT_XL_2`) build without error and report a sensible parameter count. |
| `test_forward_cifar` | `(2, 3, 32, 32)` input → `(2, 3, 32, 32)` output, finite values, no NaN/Inf. |
| `test_forward_64x64_p4` | Shape-agnostic config: `(2, 3, 64, 64)` with `patch_size=4`. |
| `test_forward_latent_4ch` | `in_channels=4` (latent-DiT config) round-trips correctly. |
| `test_unpatchify_inverse` | `patchify ∘ unpatchify` is the identity on a tensor with distinguishable per-patch values; locks in the `nhwpqc->nchpwq` einsum. |
| `test_zero_init_output` | At init, `model(x, t, y) == 0` exactly (adaLN-Zero contract). |
| `test_pos_embed_buffer` | `pos_embed` is a buffer, not a parameter, and is non-zero (i.e. actually filled in). |
| `test_grad_flow` | Backward pass produces no NaN/Inf gradients; every parameter receives a gradient (after one training step from zero init). |
| `test_cfg_combination_shape` | Sampler-side CFG (the model is a pure function): doubling the batch with `[y_real; y_null]` and combining `v_uncond + s*(v_cond - v_uncond)` produces a finite tensor of the right per-half shape. |
| `test_cfg_scale_one_equals_conditional` | At `cfg_scale = 1.0`, the affine combination algebraically reduces to `v_cond`. Sanity check for any future CFG implementation in the sampler. |
| `test_eval_mode_is_deterministic_with_dropout` | `eval()` mode disables CFG label dropout, so two forward passes on the same inputs produce identical outputs even when `class_dropout_prob > 0`. |
| `test_train_mode_label_dropout_fires` | `train()` with `class_dropout_prob = 1.0` always replaces labels with the null token; observable as `model(x, t, y_real) == model(x, t, y_null)` for every `y_real`. |
| `test_input_size_assertion` | Constructor raises when `input_size % patch_size != 0`. |
| `test_patchembed_assertion` | Forward raises when `H` or `W` is not divisible by `patch_size`. |
| `test_param_count_S2` | `DiT_S_2` parameter count is within ±5% of the paper's reference value, sanity-checking that no module is silently missing. |

### Layer 3 — Flow / Eval / Data tests (Phase B — not yet implemented)

These files arrive together with the Phase B implementation
(`flow/`, `eval/`, `data/`). Every assertion below corresponds to a
formula or rule in either [FlowMatching.md](FlowMatching.md) or
[Train.md](Train.md) — those documents are the spec, these tests are the
enforcement.

#### `tests/test_flow.py` — flow-matching primitives

All tests use synthetic tensors; no dataset.

| Test | What it checks | Spec ref |
|---|---|---|
| `test_interpolant_at_t0` | `interpolant(x_0, x_1, t=0) == x_0` | FlowMatching.md §8.1 |
| `test_interpolant_at_t1` | `interpolant(x_0, x_1, t=1) == x_1` | FlowMatching.md §8.1 |
| `test_velocity_gt` | `v_gt == x_1 - x_0` for arbitrary `t` | FlowMatching.md §8.2 |
| `test_loss_is_zero_at_perfect_prediction` | If `v_θ == v_gt` then `L == 0`. | FlowMatching.md §3 |
| `test_loss_is_mse_reduced_over_all_dims` | Shape `(B,C,H,W)` → scalar; matches manual MSE. | FlowMatching.md §3 |
| `test_cfg_scale_one_equals_conditional` | `s=1` ⇒ `v_cfg == v_cond` exactly. | FlowMatching.md §8.3 |
| `test_cfg_scale_zero_equals_unconditional` | `s=0` ⇒ `v_cfg == v_uncond`. | FlowMatching.md §8.3 |
| `test_cfg_scale_two_extrapolates` | `s=2` ⇒ `v_cfg == 2*v_cond - v_uncond`. | FlowMatching.md §8.3 |
| `test_ema_single_step` | `β=0.9, θ=1, θ_ema=0` → after 1 step `θ_ema=0.1`. | FlowMatching.md §8.4 |
| `test_ema_two_steps` | After 2 steps with same `θ=1`: `θ_ema=0.19`. | FlowMatching.md §8.4 |
| `test_ema_multiple_decays` | Two EMA copies with different `β` track independently. | FlowMatching.md §5 |
| `test_sampler_zero_velocity_returns_noise` | If model returns `0`, `sample(...) == x_0`. | FlowMatching.md §8.5 |
| `test_sampler_constant_velocity` | If model returns constant `v`, output == `x_0 + v` (bit-exact). | FlowMatching.md §8.6 |
| `test_sampler_time_direction` | The first integration step uses `t=0`, the last uses `t=(N-1)/N`. Pins down the non-standard time convention. | FlowMatching.md §1, §6 |
| `test_sampler_shape_and_finiteness` | End-to-end: trained-shape model produces finite `(B,C,H,W)` outputs. | FlowMatching.md §6 |

#### `tests/test_eval.py` — metrics sanity

| Test | What it checks |
|---|---|
| `test_fid_zero_for_identical_distributions` | `FID(X, X) ≈ 0` (within numerical tolerance) for two batches of identical synthetic tensors. |
| `test_fid_positive_for_different_distributions` | `FID(N(0,I), N(5,I)) > FID(N(0,I), N(0.1,I))` — monotonicity sanity. |
| `test_inception_score_runs_and_is_finite` | IS on a synthetic batch of `(N, 3, 299, 299)` returns a finite scalar with finite std. |
| `test_fid_ref_stat_cache_round_trip` | Save → load `(μ, Σ)` from `.npz` reproduces FID exactly. |

#### `tests/test_dataset.py` — CIFAR-10 dataset adapter

The **only** test that uses `tests/fixtures/images/` (4 small PNGs).

| Test | What it checks |
|---|---|
| `test_dataset_returns_correct_shape_and_range` | Output tensors are `(3, 32, 32)`, dtype `float32`, range `[-1, 1]`. |
| `test_dataset_label_type` | Labels are `int64` scalars in `[0, num_classes)`. |
| `test_dataset_length_matches_files` | Iterating returns one sample per fixture image. |

### Layer 4 — Overfit-one-batch

The gold-standard "the model can actually learn" test. Verifies the
whole training-step chain end-to-end
(`x_1 → x_0 → t → x_t → v_pred → v_gt → loss → backward → step`); a
shape-correct model with a broken gradient path would pass every other
layer but fail this one.

- File: `tests/test_overfit.py`.
- Setup: `B=4` synthetic tensors with 4 fixed labels, DiT-S/2
  (`class_dropout_prob=0.0` so the CFG null-token path is disabled),
  `700` AdamW steps at `lr=1e-4`, no weight decay.  Fresh `x_0` and `t`
  are resampled **every step** — the model has to learn the velocity
  *field*, not memorise a single lookup.
- LR is `1e-4` (not `1e-3`) to match the DiT-paper training regime and
  avoid the noisy post-cliff behaviour that `1e-3` produces on this
  setup; see the test module docstring for the full rationale.
- Assert: `final_L < 0.20 * initial_L` (i.e. loss drops by ≥80 %),
  where each endpoint is a 10-step windowed average to damp the
  Monte-Carlo noise from resampled `(x_0, t)`.  Empirically the run
  reaches ~0.15 × initial by step 700, giving ~1.3× headroom on the
  threshold.
- Marked `@pytest.mark.slow`; skipped by the default
  `pytest tests/ -q`.  Run explicitly with
  `pytest tests/test_overfit.py -q -m slow` (or the whole slow suite
  with `pytest tests/ -q -m slow`).  Runtime ~25 s on GPU.

### Layer 5 (deferred) — Golden / regression test

A change-detector that pickles the output of a fixed-seed forward pass
and asserts bit-equality on subsequent runs. Deferred until the model
has been trained and shown to learn — i.e. until we have evidence the
*current* numerical output is one we want to lock in. (Locking in a
buggy output is worse than no test at all.)

## How to run

From the project root, with the `dit` conda env active:

```bash
# Run the whole suite (quiet).
pytest tests/ -q

# Run a single layer.
pytest tests/test_components.py -q
pytest tests/test_dit.py -q

# Run a single test, verbose.
pytest tests/test_components.py::test_modulate -v
```

All tests should run in a few seconds on CPU.

## Reproducibility

`tests/conftest.py` installs a pytest fixture that seeds `torch.manual_seed(0)`
before every test. Tests that need a different seed call `torch.manual_seed`
explicitly inside the test body.

## Adding tests

When you add a new component to [models/dit.py](../models/dit.py) (or to the
forthcoming `models/ar_dit.py`), add a Layer 1 test for it in the same PR.
When you change the *interface* of an existing component, update its Layer 1
test. When you change a numeric-value reference (e.g. you change the freq
schedule of the timestep embedder), update both the implementation and the
hand-computed reference values in the same commit.

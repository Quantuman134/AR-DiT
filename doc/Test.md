# Testing

> ⚠️ **STATUS: NOT YET REVIEWED.**
> The Layer 1 / Layer 2 test suites described below have been written and
> all 38 tests currently pass, **but the test code itself has not been
> reviewed by the project owner yet**. Treat the suite as *provisional*:
> a passing run only means the tests are internally consistent with the
> current `models/dit.py`, not that the right things are being tested.
> The next priority is to **review and build other parts of the project
> first** (AR-DiT, flow-matching design, etc.); the test suite will be
> reviewed in a dedicated pass afterwards. Until that review:
>
> - Do **not** treat green CI as evidence of correctness.
> - Do **not** lock in any of the hand-computed numerical values as
>   "golden" — they may be revised during review.
> - When changing a component, you may still update its test, but flag
>   the change in the commit message so the upcoming review can pick it
>   up.

This project uses **pytest** as its test runner. Tests live in `tests/` and
are organised into two layers; two further layers are deferred and described
at the bottom of this document.

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

## Deferred layers

- **Layer 3 — Golden / regression test.** A change-detector that pickles the
  output of a fixed-seed forward pass and asserts bit-equality on subsequent
  runs. Deferred until the model has been trained and shown to learn — i.e.
  until we have evidence the *current* numerical output is one we want to
  lock in. (Locking in a buggy output is worse than no test at all.)

- **Layer 4 — Overfit-one-batch test.** Train the model on a single batch
  for ~500 steps and assert the loss collapses below a threshold. This is
  the gold-standard "the model can actually learn" test, but it depends on
  the flow-matching loss function, which is the subject of a future
  `doc/FlowMatching.md`. Once that doc lands, this test goes in
  `scripts/overfit_one_batch.py` and gets wired into CI.

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

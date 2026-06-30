# Attention Residual for DiT

This project investigates whether the **Attention Residual (AttnRes)** mechanism
proposed by Kimi/Moonshot (arXiv:2603.15031, Mar 2026) — originally designed for
decoder-only LLMs — can improve **Diffusion Transformers (DiT)** for
class-conditional image generation.

The core hypothesis: replacing DiT's standard identity residuals with
softmax-attention over depth (and a timestep-conditioned variant) will
mitigate PreNorm magnitude dilution and improve generation quality.

## Scope

We will build and compare two networks under matched compute:

1. **DiT** — a faithful re-implementation of the standard Diffusion
   Transformer (Peebles & Xie, 2023) with adaLN-Zero conditioning.
   Serves as the baseline.

2. **AR-DiT** (Attention-Residual DiT) — same backbone as DiT, but the
   inter-layer identity residual is replaced by AttnRes.
   - Block AttnRes variant (cheap, recommended default).
   - Optional: timestep-conditioned pseudo-queries `w_l(t)` —
     a diffusion-specific extension of the original method.

Both networks share the same patch embedding, positional embedding,
adaLN-Zero conditioning, and final layer, so any difference in results
is attributable to the residual mechanism alone.

## Planned project layout

```
Attention_Residual_for_DiT/
├── Attention_Residuals.pdf    # source paper
├── doc/
│   ├── Plan.md                # this file
│   ├── DiT.md                 # baseline DiT design spec
│   └── Test.md                # test strategy
├── models/
│   ├── dit.py                 # baseline DiT
│   └── ar_dit.py              # AR-DiT (attention residual)
├── tests/
│   ├── conftest.py            # pytest seeding fixture
│   ├── test_components.py     # Layer 1: per-component unit tests
│   └── test_dit.py            # Layer 2: end-to-end DiT tests
├── train.py                   # (later) training entry point
├── sample.py                  # (later) sampling / FID eval
└── configs/                   # (later) experiment configs
```

## Status

- [x] Project scoped
- [x] Baseline DiT implementation
- [~] DiT test suite (Layers 1 & 2) — **written, all 38 tests pass, but NOT YET REVIEWED**
- [ ] AR-DiT implementation
- [ ] Flow-matching training/sampling design (`doc/FlowMatching.md`)
- [ ] Overfit-one-batch & golden-output tests (Layers 3 & 4)
- [ ] Training pipeline
- [ ] ImageNet-256 experiments
- [ ] Timestep-conditioned pseudo-query extension

> ⚠️ **The test suite is provisional and has not been reviewed yet.**
> The current priority is to review and build other parts of the project
> (AR-DiT, flow-matching design, training pipeline) **first**; the test
> suite will get a dedicated review pass afterwards. A green `pytest` run
> at this stage means the tests are internally consistent with
> `models/dit.py` — it does **not** mean the right things are being
> tested. See [Test.md](Test.md) for details.

## Testing

See [Test.md](Test.md) for the full test strategy. In short: we run
`pytest tests/` against two layers — per-component unit tests and
whole-model end-to-end tests — for every code change. Layers 3 (golden
output regression) and 4 (overfit-one-batch) are deferred until the
flow-matching loss is designed; the rationale is documented in `Test.md`.

**Reminder:** the existing Layer 1 / Layer 2 tests are **not yet
reviewed** (see Status above). They are useful as a smoke-test while
building other parts, but no claim of correctness is made until the
review pass.

## Environment

All development and experiments run in a single conda environment:

| Item     | Version                  |
|----------|--------------------------|
| Conda env name | `dit`              |
| Python   | 3.12                     |
| PyTorch  | 2.6.0                    |
| CUDA     | 12.4 (matching the `+cu124` PyTorch build) |

Activate with:

```bash
conda activate dit
```

All commands in this repo (training, sampling, self-tests, e.g.
`python -m models.dit`) assume this environment is active. If the
environment is recreated, the only mandatory packages are PyTorch 2.6.0
(cu124 build) and NumPy; everything else (e.g. evaluation tooling) will
be added explicitly when introduced.

## References

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023.
- Kimi Team, *Attention Residuals*, arXiv:2603.15031, 2026.

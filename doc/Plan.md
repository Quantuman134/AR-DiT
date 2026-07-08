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
│   ├── AR_DiT.md              # AR-DiT design spec (attention residual)
│   ├── FlowMatching.md        # math: interpolant, loss, CFG, EMA, sampler
│   ├── Train.md               # runtime: configs, DDP, wandb, checkpoints
│   └── Test.md                # test strategy
├── models/
│   ├── dit.py                 # baseline DiT
│   └── ar_dit.py              # (later) AR-DiT (attention residual)
├── flow/                      # (Phase B) flow-matching primitives
│   ├── interpolant.py         # x_t, v_gt
│   ├── loss.py                # MSE flow-matching loss
│   ├── cfg.py                 # guided-velocity helper
│   ├── ema.py                 # multi-decay EMA
│   └── sampler.py             # Euler sampler
├── eval/                      # (Phase B) metrics
│   ├── fid.py                 # torchmetrics FID + reference-stat cache
│   └── inception_score.py     # torchmetrics IS
├── data/                      # (Phase B) dataset adapters
│   └── cifar10.py             # CIFAR-10 dataset + transforms
├── configs/
│   ├── model/dit_s2_cifar.yaml
│   ├── train/cifar10_train.yaml
│   └── sample/cifar10_sample.yaml
├── scripts/                   # (Phase C) bash launchers
│   ├── train_cifar10.sh
│   ├── sample_cifar10.sh
│   └── eval_checkpoint.sh
├── tests/
│   ├── conftest.py            # pytest seeding + synthetic-tensor fixtures
│   ├── fixtures/images/       # 4 small PNGs for the dataset-loader test
│   ├── test_components.py     # Layer 1: per-component unit tests
│   ├── test_dit.py            # Layer 2: end-to-end DiT tests
│   ├── test_flow.py           # (Phase B) flow primitives + sampler + EMA
│   ├── test_eval.py           # (Phase B) FID/IS sanity
│   └── test_dataset.py        # (Phase B) CIFAR-10 loader
├── secrets/                   # (gitignored) wandb token etc.
├── runs/                      # (gitignored) checkpoints, samples, logs
├── train.py                   # (Phase C) training entry point (DDP-aware)
└── sample.py                  # (Phase C) sampling / FID-IS evaluation
```

## Roadmap

Development deliverables only. Experiment work (the CIFAR-10 baseline
run, the `w_l(t)` timestep-conditioned pseudo-query extension) and
process work (the review pass over the provisional test suite) are
tracked separately, not on this roadmap.

| # | Item                                                | Status   | Notes                                                            |
|---|-----------------------------------------------------|----------|------------------------------------------------------------------|
| 1 | Baseline DiT model (`models/dit.py`)                | ✅ Done   | adaLN-Zero conditioning, matches Peebles & Xie (2023)            |
| 2 | Flow-matching primitives (`flow/`)                  | ✅ Done   | interpolant, MSE loss, CFG helper, multi-decay EMA, Euler sampler |
| 3 | Evaluation stack (`eval/`)                          | ✅ Done   | FID + Inception Score via `torchmetrics`                         |
| 4 | CIFAR-10 dataset adapter (`data/`)                  | ✅ Done   | dataset + transforms                                             |
| 5 | Training / sampling entry points                    | ⚠️ Written, not yet reviewed | `train.py`, `sample.py`, `scripts/*.sh` |
| 6 | Test suite Layers 1–4                               | ⚠️ Written, not yet reviewed | components / DiT / flow+eval+data / overfit-one-batch |
| 7 | AR-DiT model (`models/ar_dit.py`)                   | ⏳ Todo   | Block AttnRes variant — the core contribution                   |
| 8 | Layer-5 golden-output regression test               | 🔒 Deferred | Blocked on a trained checkpoint; see [Test.md](Test.md) §Layer 5 |

Legend: ✅ done · ⚠️ written but unreviewed · ⏳ todo · 🔒 deferred by design.

Supporting docs completed alongside the code above: `doc/DiT.md`,
`doc/FlowMatching.md`, `doc/Train.md`, `doc/Test.md`, and the initial
YAML configs under `configs/`.

> ⚠️ **The test suite is provisional and has not been reviewed yet.**
> A green `pytest` run at this stage means the tests are internally
> consistent with the code they exercise — it does **not** mean the
> right things are being tested. Rows 5 and 6 above are marked
> "written, not yet reviewed" for the same reason. A dedicated review
> pass will happen once AR-DiT (row 7) lands. See [Test.md](Test.md).

## Testing

See [Test.md](Test.md) for the full test strategy. In short:
`pytest tests/` runs Layers 1–3 (per-component units, whole-model DiT,
and flow / eval / dataset primitives) as a fast CPU-friendly suite;
Layer 4 (overfit-one-batch) is opt-in via `-m slow`. Layer 5 (golden
output regression) is deliberately deferred until a trained checkpoint
exists — locking in a buggy output would be worse than no test at all.

**Reminder:** these tests are **not yet reviewed** (see the Roadmap
above). They are useful as a smoke-test while building other parts, but
no claim of correctness is made until the review pass.

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

# Attention Residual for DiT

A research codebase investigating whether the **Attention Residual** mechanism
(Kimi/Moonshot, [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)) —
originally proposed for decoder-only LLMs — can improve **Diffusion
Transformers (DiT)** for class-conditional image generation.

Two networks are built under matched compute and compared on CIFAR-10 with
flow-matching training:

- **DiT** — faithful re-implementation of Peebles & Xie (2023), adaLN-Zero
  conditioning. Baseline.
- **AR-DiT** — same backbone, but the inter-layer identity residual is
  replaced by softmax-attention over depth.

Any quality delta between the two is therefore attributable to the residual
mechanism alone.

## Roadmap

Development deliverables only — experiment runs and reviews are tracked
separately.

| # | Item                                                | Status   | Notes                                                            |
|---|-----------------------------------------------------|----------|------------------------------------------------------------------|
| 1 | Baseline DiT model (`models/dit.py`)                | ✅ Done   | adaLN-Zero conditioning, matches Peebles & Xie (2023)            |
| 2 | Flow-matching primitives (`flow/`)                  | ✅ Done   | interpolant, MSE loss, CFG helper, multi-decay EMA, Euler sampler |
| 3 | Evaluation stack (`eval/`)                          | ✅ Done   | FID + Inception Score via `torchmetrics`                         |
| 4 | CIFAR-10 dataset adapter (`data/`)                  | ✅ Done   | dataset + transforms                                             |
| 5 | Training / sampling entry points                    | ⚠️ Written, not yet reviewed | `train.py`, `sample.py`, `scripts/*.sh` |
| 6 | Test suite Layers 1–4                               | ⚠️ Written, not yet reviewed | components / DiT / AR-DiT / flow+eval+data / overfit-one-batch |
| 7 | AR-DiT model (`models/ar_dit.py`)                   | ⚠️ Written, not yet reviewed | Per-sub-layer AttnRes junctions (paper-strict, v1) — the core contribution |
| 8 | Layer-5 golden-output regression test               | 🔒 Deferred | Blocked on a trained checkpoint; see [`doc/Test.md`](doc/Test.md) §Layer 5 |

Legend: ✅ done · ⚠️ written but unreviewed · ⏳ todo · 🔒 deferred by design.

See [`doc/Plan.md`](doc/Plan.md) for full context (scope, hypothesis,
project layout).

## Install

Development happens in a single conda environment named `dit` (Python 3.12,
PyTorch 2.6.0 + CUDA 12.4).

```bash
conda create -n dit python=3.12 -y
conda activate dit

# GPU box (cu124):
pip install --index-url https://download.pytorch.org/whl/cu124 \
            torch==2.6.0 torchvision==0.21.0

# ...or CPU-only:
# pip install --index-url https://download.pytorch.org/whl/cpu \
#             torch==2.6.0 torchvision==0.21.0

pip install -e ".[dev]"
```

`pip install -e ".[dev]"` pulls in the runtime deps (`torchmetrics`,
`torch-fidelity`, `scipy`, `numpy`, `Pillow`, `PyYAML`, `wandb`) plus
`pytest`.

### Weights & Biases (optional but on by default)

Training logs scalars and validation grids to
[Weights & Biases](https://wandb.ai). To enable it on a fresh checkout:

1. Put your personal API key (a single line, no trailing newline) into
   `secrets/wandb.token`. The `secrets/` folder is tracked but its
   contents are gitignored — see [`.gitignore`](.gitignore) — so the
   token stays local:

   ```bash
   echo "$WANDB_API_KEY" > secrets/wandb.token
   chmod 600 secrets/wandb.token
   ```

2. Set the project / entity / mode in your training YAML under
   `logging.wandb` (defaults live in
   [`configs/train/cifar10_train.yaml`](configs/train/cifar10_train.yaml)):

   ```yaml
   logging:
     wandb:
       enabled: true
       project: attn_residual_dit
       entity: null            # null = personal account
       token_path: secrets/wandb.token
       mode: online            # online | offline | disabled
   ```

3. To skip wandb entirely (e.g. on an offline node) set
   `logging.wandb.enabled: false` or `mode: disabled`. Training also
   degrades gracefully if the `wandb` package is missing or the token
   file cannot be read — see [`doc/Train.md`](doc/Train.md) §6 for the
   full spec.

## Usage

All entry points are driven by YAML configs under `configs/`; the bash
launchers under `scripts/` are thin `torchrun` wrappers that auto-detect the
GPU count (override with `NUM_GPUS=<n>`) and forward extra flags verbatim.

**Train** on CIFAR-10 (auto-resumes from `runs/<run_name>/ckpt/latest.pt` if
present):

```bash
./scripts/train_cifar10.sh                              # default config
./scripts/train_cifar10.sh configs/train/cifar10_train.yaml
NUM_GPUS=4 ./scripts/train_cifar10.sh --override optim.lr=2.0e-4
```

**Sample / evaluate** a checkpoint (writes samples + FID/IS to the run dir):

```bash
./scripts/sample_cifar10.sh configs/sample/cifar10_sample.yaml \
    --ckpt runs/dit_s2_cifar/ckpt/step_000200000.pt
```

For a "just give me FID" one-liner, `scripts/eval_checkpoint.sh` is a thin
alias over the sampling script with `metrics.fid=true` forced on.

Every `--override key=value` overrides a single leaf of the YAML config; see
[`doc/Train.md`](doc/Train.md) for the full runtime spec (checkpoint layout,
DDP, EMA, wandb, override syntax).

## Tests

All tests are hermetic — they use synthetic tensors or four tiny committed
PNGs, never a real dataset download.

```bash
# Fast suite — Layers 1–3 (~a few seconds, CPU-friendly).
# Skips anything marked @pytest.mark.slow by default.
pytest tests/ -q

# Layer 4 overfit-one-batch (~20 s on GPU): the gold-standard
# "the model can actually learn" test. Marked slow, so opt in explicitly.
pytest tests/ -q -m slow

# Run a single suite.
pytest tests/test_components.py -q      # Layer 1: per-component units
pytest tests/test_dit.py         -q     # Layer 2: whole DiT model
pytest tests/test_flow.py        -q     # Layer 3a: flow-matching primitives
pytest tests/test_eval.py        -q     # Layer 3b: FID / IS wrappers
pytest tests/test_dataset.py     -q     # Layer 3c: CIFAR-10 loader

# Run a single test, verbose.
pytest tests/test_components.py::test_modulate -v
```

Layer 5 (golden-output regression) is intentionally not implemented yet;
see [`doc/Test.md`](doc/Test.md) for the layered strategy and the reason
for the deferral.

## Layout

```
models/    DiT (and later AR-DiT) architecture
flow/      interpolant, flow-matching loss, CFG, EMA, Euler sampler
eval/      FID + Inception Score wrappers
data/      CIFAR-10 dataset + transforms
configs/   YAML configs (model / train / sample) + Pydantic schema
runtime/   checkpoint I/O, DDP helpers, RNG, FID-cache builder
scripts/   bash launchers (train / sample / eval)
tests/     pytest suite
doc/       design specs (DiT, FlowMatching, Train, Test, Plan)
train.py   training entry point (DDP-aware)
sample.py  sampling / FID-IS evaluation entry point
```

## References

- Peebles & Xie, *Scalable Diffusion Models with Transformers*, ICCV 2023.
- Kimi Team, *Attention Residuals*, arXiv:2603.15031, 2026.

## License

MIT.

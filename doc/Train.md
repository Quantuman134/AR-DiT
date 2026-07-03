# Training & Sampling — Runtime Specification

This document specifies the **runtime** side of training and sampling:
configs, checkpoints, distributed launch, wandb integration, validation,
and metrics. The math (interpolant, loss, CFG, EMA, sampler) lives in
[FlowMatching.md](FlowMatching.md); this document deliberately avoids
re-stating it.

---

## 1. Project layout (training-side additions)

```
Attention_Residual_for_DiT/
├── configs/
│   ├── model/
│   │   └── dit_s2_cifar.yaml         # architecture only (shared by train+sample+test)
│   ├── train/
│   │   └── cifar10_train.yaml        # training run config
│   └── sample/
│       └── cifar10_sample.yaml       # sampling / evaluation config
├── flow/
│   ├── __init__.py
│   ├── interpolant.py                # x_t, v_gt (§2 of FlowMatching.md)
│   ├── loss.py                       # MSE flow-matching loss (§3)
│   ├── cfg.py                        # guided-velocity helper (§4)
│   ├── ema.py                        # multi-decay EMA (§5)
│   └── sampler.py                    # Euler sampler (§6)
├── eval/
│   ├── __init__.py
│   ├── fid.py                        # torchmetrics FID wrapper + reference-stat cache
│   └── inception_score.py            # torchmetrics IS wrapper
├── data/
│   ├── __init__.py
│   └── cifar10.py                    # CIFAR-10 dataset + transforms
├── train.py                          # training entry point (DDP-aware)
├── sample.py                         # sampling / FID-IS evaluation entry point
├── scripts/
│   ├── train_cifar10.sh              # bash launcher: torchrun → train.py
│   ├── sample_cifar10.sh             # bash launcher → sample.py
│   └── eval_checkpoint.sh            # bash launcher: FID/IS over a saved ckpt
├── secrets/
│   └── wandb.token                   # plain text, gitignored
└── runs/                             # output root (gitignored)
    └── <run_name>/
        ├── config.yaml               # frozen copy of the run's config
        ├── ckpt/
        │   ├── step_000010000.pt
        │   ├── step_000020000.pt
        │   └── latest.pt             # symlink to most recent
        └── samples/                  # validation grid PNGs (also logged to wandb)
```

`flow/`, `eval/`, `data/`, `scripts/`, `configs/`, and `secrets/` are all
new in this milestone.

---

## 2. Configs — YAML, validated by dataclass

We use YAML for config files (human-friendly, supports comments) and a
pydantic-style or plain-`dataclass` schema in code (`configs/schema.py`)
to validate every loaded config at startup. Unknown keys raise a hard
error: silent typos are the worst kind of bug for ML configs.

### 2.1 Three-config split

The three responsibilities are deliberately separated:

| File                              | What it owns                                      |
|-----------------------------------|---------------------------------------------------|
| `configs/model/<name>.yaml`       | Architecture: `arch_name`, `input_size`, `in_channels`, `patch_size`, `num_classes`, `class_dropout_prob`. |
| `configs/train/<name>.yaml`       | Optimisation, schedule, dataset path, DDP, EMA decays, validation cadence, wandb, **plus a reference to a model config**. |
| `configs/sample/<name>.yaml`      | Sampling: `num_steps`, `guidance_scale`, batch size, ema-tag selection, FID/IS reference dataset path, output dir, **plus a reference to a model config**. |

A train-config or sample-config does not duplicate model fields; it
points at a model-config by relative path:

```yaml
# configs/train/cifar10_train.yaml
model_config: ../model/dit_s2_cifar.yaml
```

When a checkpoint is loaded for sampling, the sampler reads the
**checkpointed model config** (which was frozen at training time, see
§4.2) and ignores the model-config field of the sample-config — this
makes it impossible to mismatch architecture between train and eval.

### 2.2 Dataset path is configurable

You will move this code between machines. Datasets do not live in the
repo. Every config that touches data exposes:

```yaml
dataset:
  name: cifar10               # selects dataset class in data/
  root: /abs/path/on/this/machine/cifar10
  download: false             # never download silently — fail loudly
```

The bash launchers (§5) accept a `--dataset_root` override that wins
over the YAML field, so the same config + same launcher work on a new
machine by changing one CLI flag.

### 2.3 Schema overview

The exact field list will be finalised when `configs/schema.py` is
written; this is the intended shape so reviewers can sanity-check it
now.

```yaml
# configs/model/dit_s2_cifar.yaml
arch_name: DiT_S_2          # selects preset in models/dit.py
input_size: 32
in_channels: 3
patch_size: 2
num_classes: 10
class_dropout_prob: 0.1     # CFG label-dropout probability
```

```yaml
# configs/train/cifar10_train.yaml
model_config: ../model/dit_s2_cifar.yaml

dataset:
  name: cifar10
  root: /data/cifar10
  download: false
  num_workers: 4
  pin_memory: true

optim:
  name: adamw
  lr: 1.0e-4
  betas: [0.9, 0.999]
  weight_decay: 0.0
  grad_clip: 1.0            # null = disabled
  warmup_steps: 1000
  lr_schedule: constant     # one of: constant, cosine

train:
  total_steps: 400000
  batch_size_per_gpu: 128
  log_interval: 50          # terminal + wandb scalar interval (steps)
  ckpt_interval: 10000      # checkpoint save interval
  seed: 0
  amp_dtype: bf16           # one of: fp32, bf16

ema:
  decays: [0.9999, 0.999]   # one shadow copy per value

guidance:
  null_class_id: null       # null ⇒ auto-derive as model_config.num_classes
                            # (LabelEmbedder reserves the last embedding row
                            # for the null token). Set to an explicit int only
                            # when loading a checkpoint trained with a
                            # different null-class convention.
  # CFG label-dropout probability lives in the model config
  # (class_dropout_prob) because it's owned by LabelEmbedder.

validation:
  interval: 10000           # steps between validation passes
  num_samples: 32           # total images generated per (ema, guidance-scale) pair
  visual_log_count: 16      # how many to log to wandb as a grid
  guidance_scales: [1.0, 1.5, 4.0]
  metrics:
    fid: true
    inception_score: true
  fid_ref_stats: /data/cifar10/fid_ref_stats.npz   # cached ref stats; null = compute on first run

sampler:
  num_steps: 50             # Euler steps used during validation

logging:
  out_dir: runs/            # run subdir = runs/<run_name>/
  run_name: dit_s2_cifar_${now:%Y%m%d_%H%M%S}
  wandb:
    enabled: true
    project: attn_residual_dit
    entity: null            # null = personal account
    token_path: secrets/wandb.token
    mode: online            # online | offline | disabled
```

```yaml
# configs/sample/cifar10_sample.yaml
ckpt_path: runs/dit_s2_cifar_20260615_120000/ckpt/latest.pt
ema_tag: ema_0.9999         # which weight set to use; one of {online, ema_<decay>}

sampling:
  num_samples: 50000        # total images to generate (FID-grade = 50k)
  batch_size_per_gpu: 256
  num_steps: 50
  guidance_scale: 1.5
  seed: 0

dataset:                    # only for FID/IS reference statistics
  name: cifar10
  root: /data/cifar10
  download: false

metrics:
  fid: true
  inception_score: true
  fid_ref_stats: /data/cifar10/fid_ref_stats.npz

output:
  dir: runs/dit_s2_cifar_20260615_120000/eval/
  save_images: false        # if true, dump every generated image as PNG
  save_grid: true           # save one 8x8 grid PNG per (ema, guidance-scale) pair
```

---

## 3. Checkpoints

### 3.1 Format

Each checkpoint is a single `torch.save`d dict at
`runs/<run_name>/ckpt/step_<NNNNNNNNN>.pt`:

```python
{
  "step":             int,             # global optimiser step
  "model_state":      dict,            # online weights (model.module if DDP)
  "ema_states":       {decay_str: state_dict, ...},   # one per EMA copy
  "optim_state":      dict,
  "scaler_state":     dict | None,     # for AMP, None if fp32/bf16
  "rng_state": {
    "torch":          tensor,
    "torch_cuda":     list[tensor],
    "numpy":          dict,
    "python":         tuple,
  },
  "config_yaml":      str,             # the verbatim training YAML (resolved)
  "model_config_yaml":str,             # the verbatim model YAML
  "git_sha":          str | None,      # commit at training start (best-effort)
  "version":          1,               # bump on schema changes
}
```

A symlink `latest.pt` always points to the newest file in the directory.

### 3.2 Resume semantics

`train.py` accepts `--resume <path>`. When present:

1. Load the checkpoint dict.
2. Rebuild model + EMA copies from the **checkpointed** model YAML.
   (Hard-fail if the on-disk train YAML's `model_config` resolves to a
   different architecture — refuse to silently change shape mid-run.)
3. Restore optimiser, scaler, RNG state, EMA states.
4. Continue from `step + 1`.

The expected invariant is **bit-identical loss curve** when resuming on
the same hardware. Achieving this requires:

- RNG state restoration (`torch`, `torch.cuda`, `numpy`, `python`).
- A stateful `DataLoader` sampler — we use a custom
  `ResumableDistributedSampler` that takes a `start_step` and skips that
  many batches.

The bash launcher `scripts/train_cifar10.sh` auto-detects
`runs/<run_name>/ckpt/latest.pt` and adds `--resume` automatically when
present, so a crashed run is resumed by re-running the same command.

---

## 4. Distributed training (DDP via torchrun)

### 4.1 Single command, scales from 1 GPU to N GPUs on one node

```bash
# 1 GPU
torchrun --standalone --nproc_per_node=1 train.py --config configs/train/cifar10_train.yaml

# 4 GPUs on one node
torchrun --standalone --nproc_per_node=4 train.py --config configs/train/cifar10_train.yaml
```

The wrapper script `scripts/train_cifar10.sh` picks `nproc_per_node`
from `$NUM_GPUS` (default = `nvidia-smi -L | wc -l`) and forwards every
other CLI arg through.

Multi-node training is **out of scope** for this milestone. If we need
it later, the only change is to swap `--standalone` for a rendezvous
backend; nothing in the training loop assumes single-node.

### 4.2 Concurrency rules

- Every rank builds the model and runs the forward/backward pass.
- Only **rank 0** writes checkpoints, owns the EMA copies, runs
  validation, computes metrics, and talks to wandb.
- Validation samples are generated on rank 0 only. (We can parallelise
  generation across ranks later if validation becomes a bottleneck.)
- The data loader uses `DistributedSampler` so each rank sees a
  disjoint partition of the epoch.

### 4.3 Effective batch size

```
effective_batch = train.batch_size_per_gpu × num_gpus × grad_accum_steps
```

`grad_accum_steps` defaults to 1 and is exposed in the train YAML.
Learning rate in this project is **not** auto-scaled with batch size —
you set the LR you want and the config records it.

---

## 5. Bash launchers

Three thin wrappers under `scripts/`. Each one:

1. Resolves the project root and `cd`s into it (see §5.0).
2. Activates the `dit` conda env (or assumes it's active and warns).
3. Resolves the number of GPUs.
4. Forwards CLI args to the Python entry point.

### 5.0 Project-root resolution (shared preamble)

Every launcher must run from the repository root: relative paths
(`configs/...`, `runs/...`, `secrets/wandb.token`, the auto-resume
glob) all assume it. This is non-trivial in practice because remote
job dispatchers (SLURM, LSF, bare `ssh host bash script.sh`, cron)
drop you in `$HOME`, `$TMPDIR`, or `/` — not in the repo.

Each script therefore starts with the same three-layer preamble:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Layer 1: auto-detect the script's own location (works for direct
# invocation, ./scripts/foo.sh, bash scripts/foo.sh, and symlinks).
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Layer 2: allow an explicit override via the PROJECT_ROOT env var
# (escape hatch for symlinked scripts, scratch-dir copies, CI mounts).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"

# Layer 3: sanity check — fail loud before torchrun produces a
# confusing ModuleNotFoundError.
if [[ ! -f "train.py" ]] || [[ ! -d "configs" ]]; then
    echo "ERROR: PROJECT_ROOT=$PROJECT_ROOT does not look like the project root" >&2
    echo "       (expected train.py and configs/ to exist here)" >&2
    exit 1
fi
```

Override with `PROJECT_ROOT=/some/path ./scripts/train_cifar10.sh`
when the auto-detected path is wrong (e.g. the script was copied to a
SLURM scratch dir but the repo lives in `$WORK`).

### 5.1 `scripts/train_cifar10.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# --- shared preamble (see §5.0) ---
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
if [[ ! -f "train.py" ]] || [[ ! -d "configs" ]]; then
    echo "ERROR: PROJECT_ROOT=$PROJECT_ROOT does not look like the project root" >&2
    exit 1
fi
# --- end preamble ---

CONFIG="${1:-configs/train/cifar10_train.yaml}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

# Auto-resume if a latest.pt exists for this run.
RUN_NAME=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['logging']['run_name'])" "$CONFIG")
LATEST="runs/${RUN_NAME}/ckpt/latest.pt"
RESUME_FLAG=""
if [[ -f "$LATEST" ]]; then
    RESUME_FLAG="--resume $LATEST"
    echo "Auto-resuming from $LATEST"
fi

torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train.py \
    --config "$CONFIG" \
    ${RESUME_FLAG} \
    "${@:2}"
```

### 5.2 `scripts/sample_cifar10.sh` and `scripts/eval_checkpoint.sh`

Same structure, calling `sample.py`. `sample.py` internally dispatches
on whether the config requests just generation, just metrics, or both.
`eval_checkpoint.sh` is a one-line alias that forces `metrics.fid: true`
and points at a user-supplied checkpoint:

```bash
./scripts/eval_checkpoint.sh runs/<run_name>/ckpt/step_000200000.pt
```

---

## 6. wandb integration

### 6.1 Token handling

- The wandb API token lives in `secrets/wandb.token` (a single line).
- `secrets/` is gitignored.
- On every rank, `train.py` reads the token *only on rank 0*, calls
  `wandb.login(key=...)`, and then `wandb.init(...)`.
- If `logging.wandb.enabled = false` or the token file is missing,
  training proceeds without wandb and prints a warning once.

### 6.2 What gets logged

| Metric                  | Frequency                       | Source        |
|-------------------------|---------------------------------|---------------|
| `train/loss`            | every `log_interval` steps      | rolling mean  |
| `train/lr`              | every `log_interval` steps      | scheduler     |
| `train/grad_norm`       | every `log_interval` steps      | pre-clip      |
| `train/sec_per_step`    | every `log_interval` steps      | rolling mean  |
| `train/samples_per_sec` | every `log_interval` steps      | derived       |
| `val/fid_<ema>_<cfg>`   | every `validation.interval`     | eval/fid.py   |
| `val/is_<ema>_<cfg>`    | every `validation.interval`     | eval/is.py    |
| `val/grid_<ema>_<cfg>`  | every `validation.interval`     | image grid    |

`<ema>` is `online` or `ema_<decay>`; `<cfg>` is the cfg-scale value.

### 6.3 Terminal output (always on, wandb-independent)

A single line per `log_interval`:

```
[step 010000/400000  ep 25  4.2 it/s  loss=0.4123  lr=1.00e-4  gn=0.87  eta=18h32m]
```

A multi-line block per validation pass:

```
[val   step 010000  cfg=1.5 ema=0.9999  FID=42.13  IS=4.21  (32 samples)]
[val   step 010000  cfg=4.0 ema=0.9999  FID=38.07  IS=4.55  (32 samples)]
[val   step 010000  cfg=1.5 ema=0.999   FID=44.91  IS=4.10  (32 samples)]
...
```

---

## 7. Validation

Triggered every `validation.interval` steps and at the very end of
training. On rank 0 only:

1. For each `ema_tag` ∈ `{online} ∪ {ema_<d> for d in ema.decays}`:
   1. Swap weights into a temporary inference module.
   2. For each `guidance_scale` in `validation.guidance_scales`:
      1. Generate `validation.num_samples` images using the Euler
         sampler at `validation.sampler.num_steps`. Class labels are
         sampled uniformly from `[0, num_classes)`.
      2. Compute FID and IS (whichever are enabled).
      3. Save the first `visual_log_count` images as a grid PNG and log
         to wandb as `val/grid_<ema>_gs<scale>`.
3. Restore the online weights and resume training.

Validation runs in `torch.no_grad()` and switches the model to
`eval()` mode (which also disables CFG label dropout — see
[FlowMatching.md §4.1](FlowMatching.md)).

`num_samples = 32` is intentionally tiny: validation runs many times
during training and exists to **track relative progress**, not to
produce paper numbers. Final FID/IS reporting uses `sample.py` with
`num_samples = 50000`, which is the FID-grade community standard.

---

## 8. Metrics

Both metrics are thin wrappers over `torchmetrics.image`:

- **FID** (`eval/fid.py`): `torchmetrics.image.fid.FrechetInceptionDistance`
  with `feature=2048`, converted from the project's `[-1, 1]` range to
  `uint8` `[0, 255]` before feeding the InceptionV3 backbone.
  Reference statistics over the full training set are computed once
  and cached at `dataset.fid_ref_stats` (a `.npz` holding torchmetrics'
  running Inception-feature sums — `real_features_sum`,
  `real_features_cov_sum`, `real_features_num_samples` — from which
  `(μ, Σ)` are derived at compute time); subsequent runs load the
  cache and skip the real-set pass. The three key names track
  torchmetrics' internal attribute names as of `torchmetrics==1.9.x`;
  a mismatched cache fails loudly at load time rather than silently
  producing wrong FIDs.
- **Inception Score** (`eval/inception_score.py`):
  `torchmetrics.image.inception.InceptionScore` with the standard
  10-split protocol.

Both classes preserve the model's `dtype` for the input batch and only
cast inside the metric. They are torch-native, GPU-resident, and
deterministic given a fixed RNG seed.

### 8.1 Reference-stat cache

On the very first run with a given `dataset.fid_ref_stats` path that
does not yet exist, the FID class iterates the whole training set
once, accumulates the three Inception-v3 pool3 running sums listed in
§8, saves them to disk, and uses them from then on. `(μ, Σ)` are
determined by those three sums, so this format is equivalent to
storing `(μ, Σ)` explicitly while remaining a pixel-perfect copy of
torchmetrics' internal state — no round-trip arithmetic required.
This is a ~1-minute cost on CIFAR-10 and pays for itself the first
time validation runs.

> ⚠️ **`train.build_or_load_fid_cache` — NOT YET REVIEWED.**
> The helper that implements this cache flow (`build_or_load_fid_cache`
> in [`train.py`](../train.py)) is written and its behaviour matches
> this section, but it has **not been reviewed by the project owner
> yet**. Points to double-check in the review pass:
>
> - The rank-0-builds / all-ranks-load handshake around `dist.barrier()`
>   is the intended pattern (vs. e.g. having every rank build in
>   parallel and reduce).
> - The "no cache path configured → return an empty-real-side metric
>   with a stderr warning" branch is the desired fallback (vs. hard
>   error). It is currently kept for smoke-test convenience only.
> - The reference dataset passed in is the **non-augmented** copy of
>   the training set; confirm this matches the FID-reporting
>   convention we want to use for AR-DiT vs. DiT comparisons.
> - The hard-coded `batch_size=256` / `num_workers=2` inside the
>   builder loop are not tunable from config yet.

---

## 9. Acceptance criteria for this milestone

The training/sampling milestone is "done" when:

1. `scripts/train_cifar10.sh` launches a 1-GPU run and a 4-GPU run from
   the same config.
2. The same script auto-resumes a crashed run from `latest.pt`.
3. wandb shows `train/loss` going down and `val/fid_*` going down over
   the first ~50k steps (we are not committing to a specific FID
   number — only that the curve moves in the right direction).
4. `scripts/eval_checkpoint.sh <ckpt>` reports an FID and IS at
   `num_samples=50000` and writes a sample grid PNG.
5. All Phase-B tests in [Test.md](Test.md) pass:
   `tests/test_flow.py`, `tests/test_eval.py`, `tests/test_dataset.py`.

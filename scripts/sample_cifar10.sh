#!/usr/bin/env bash
# ----------------------------------------------------------------------
# scripts/sample_cifar10.sh
#
# Thin bash wrapper around `sample.py` for offline sampling / FID-IS
# evaluation.  See doc/Train.md §5.2 and doc/Train.md §8 for context.
#
# Responsibilities (mirror of §5.1 for the sampling side):
#   1. Resolve the project root and cd into it (§5.0 preamble).
#   2. Warn if the `dit` conda env is not active.
#   3. Detect the number of GPUs (override with NUM_GPUS=...).
#   4. Forward extra args verbatim to sample.py so any `--ckpt`,
#      `--override key=value`, `--dataset_root`, ... just work.
#
# Usage
# -----
#   ./scripts/sample_cifar10.sh
#   ./scripts/sample_cifar10.sh configs/sample/cifar10_sample.yaml
#   ./scripts/sample_cifar10.sh configs/sample/cifar10_sample.yaml \
#       --ckpt runs/dit_s2_cifar/ckpt/step_000200000.pt \
#       --override sampling.num_samples=1000
#   NUM_GPUS=4 ./scripts/sample_cifar10.sh
#
# For a "just give me FID" one-liner over a specific checkpoint, use
# `scripts/eval_checkpoint.sh` instead — it's a thin alias over this
# script that hard-codes `metrics.fid=true`.
# ----------------------------------------------------------------------

set -euo pipefail

# --- shared preamble (see doc/Train.md §5.0) --------------------------
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
if [[ ! -f "sample.py" ]] || [[ ! -d "configs" ]]; then
    echo "ERROR: PROJECT_ROOT=$PROJECT_ROOT does not look like the project root" >&2
    echo "       (expected sample.py and configs/ to exist here)" >&2
    exit 1
fi
# --- end preamble -----------------------------------------------------

# Warn (don't fail) if the `dit` conda env doesn't seem active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "dit" ]]; then
    echo "WARNING: conda env 'dit' does not appear active (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}')." >&2
    echo "         Continuing anyway; run \`conda activate dit\` first if you hit import errors." >&2
fi

CONFIG="${1:-configs/sample/cifar10_sample.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

# Detect GPU count. Override with NUM_GPUS=<n> ./scripts/sample_cifar10.sh .
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS="$(nvidia-smi -L | wc -l)"
    else
        NUM_GPUS=1
        echo "WARNING: nvidia-smi not found; defaulting to NUM_GPUS=1." >&2
    fi
fi
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "ERROR: NUM_GPUS=$NUM_GPUS is invalid (must be >= 1)." >&2
    exit 1
fi

echo "[launcher] project_root=$PROJECT_ROOT  config=$CONFIG  num_gpus=$NUM_GPUS"

# Everything after $1 is forwarded verbatim (--ckpt, --override, --dataset_root, ...).
exec torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    sample.py \
    --config "$CONFIG" \
    "${@:2}"

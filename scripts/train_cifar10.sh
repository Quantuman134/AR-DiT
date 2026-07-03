#!/usr/bin/env bash
# ----------------------------------------------------------------------
# scripts/train_cifar10.sh
#
# Thin bash wrapper around `train.py`.  See doc/Train.md §5.1 for the
# spec this script implements.
#
# Responsibilities (§5):
#   1. Resolve the project root and cd into it (§5.0 preamble).
#   2. Warn if the `dit` conda env is not active (does NOT try to
#      activate it — activating conda from a non-interactive subshell
#      is fragile across SLURM/LSF/ssh; the warning is enough).
#   3. Detect the number of GPUs (override with NUM_GPUS=...).
#   4. Auto-resume if `runs/<run_name>/ckpt/latest.pt` already exists.
#   5. Forward extra args verbatim to train.py (so `--override ...`
#      and any future flag "just work" without touching this script).
#
# Usage
# -----
#   ./scripts/train_cifar10.sh
#   ./scripts/train_cifar10.sh configs/train/cifar10_train.yaml
#   NUM_GPUS=4 ./scripts/train_cifar10.sh
#   ./scripts/train_cifar10.sh configs/train/cifar10_train.yaml \
#       --dataset_root /data/cifar10 \
#       --override optim.lr=2.0e-4
#
# Escape hatches
# --------------
#   PROJECT_ROOT=/some/path ./scripts/train_cifar10.sh
#       Use when the auto-detected repo root is wrong (e.g. the script
#       was copied to a SLURM scratch dir but the repo lives in $WORK).
#   NUM_GPUS=<n> ./scripts/train_cifar10.sh
#       Force a specific world size instead of `nvidia-smi -L | wc -l`.
# ----------------------------------------------------------------------

set -euo pipefail

# --- shared preamble (see doc/Train.md §5.0) --------------------------
# Layer 1: auto-detect the script's own location (works for direct
# invocation, ./scripts/foo.sh, bash scripts/foo.sh, and symlinks).
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Layer 2: allow an explicit override via the PROJECT_ROOT env var.
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"

# Layer 3: sanity check — fail loud before torchrun produces a
# confusing ModuleNotFoundError.
if [[ ! -f "train.py" ]] || [[ ! -d "configs" ]]; then
    echo "ERROR: PROJECT_ROOT=$PROJECT_ROOT does not look like the project root" >&2
    echo "       (expected train.py and configs/ to exist here)" >&2
    exit 1
fi
# --- end preamble -----------------------------------------------------

# Warn (don't fail) if the `dit` conda env doesn't seem active.
# CONDA_DEFAULT_ENV is set by conda's shell integration; absence is
# not fatal (users may prefer venv/direct-python), but it's a common
# source of "wrong torch version" bug reports.
if [[ "${CONDA_DEFAULT_ENV:-}" != "dit" ]]; then
    echo "WARNING: conda env 'dit' does not appear active (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-<unset>}')." >&2
    echo "         Continuing anyway; run \`conda activate dit\` first if you hit import errors." >&2
fi

CONFIG="${1:-configs/train/cifar10_train.yaml}"
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

# Detect GPU count. Override with NUM_GPUS=<n> ./scripts/train_cifar10.sh .
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

# Auto-resume if a latest.pt already exists for this run.  We read
# `logging.run_name` out of the YAML with a tiny inline python one-
# liner rather than hard-coding it — this way the launcher stays
# generic across configs (e.g. cifar10_train.yaml vs. an AR-DiT one).
RUN_NAME="$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["logging"]["run_name"])' "$CONFIG")"
LATEST="runs/${RUN_NAME}/ckpt/latest.pt"
RESUME_FLAG=()
if [[ -f "$LATEST" ]]; then
    RESUME_FLAG=(--resume "$LATEST")
    echo "[launcher] auto-resuming from $LATEST"
else
    echo "[launcher] no checkpoint at $LATEST; starting from scratch"
fi

echo "[launcher] project_root=$PROJECT_ROOT  config=$CONFIG  num_gpus=$NUM_GPUS  run_name=$RUN_NAME"

# Everything after $1 is forwarded verbatim (--override, --dataset_root, ...).
exec torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    train.py \
    --config "$CONFIG" \
    "${RESUME_FLAG[@]}" \
    "${@:2}"

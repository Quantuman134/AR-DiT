#!/usr/bin/env bash
# ----------------------------------------------------------------------
# scripts/eval_checkpoint.sh
#
# One-liner alias over `scripts/sample_cifar10.sh` that:
#   1. Takes a checkpoint path as the first positional argument.
#   2. Forces `metrics.fid=true` (that is the whole point of "eval").
#   3. Delegates the rest — preamble, GPU detection, `dit` env warning,
#      torchrun invocation — to `sample_cifar10.sh` so the two paths
#      cannot silently drift apart.
#
# See doc/Train.md §5.2.
#
# Usage
# -----
#   ./scripts/eval_checkpoint.sh runs/dit_s2_cifar/ckpt/step_000200000.pt
#   ./scripts/eval_checkpoint.sh runs/foo/ckpt/step_000400000.pt \
#       configs/sample/cifar10_sample.yaml \
#       --override sampling.guidance_scale=4.0
#
# The second positional (config YAML) is optional; it defaults to
# `configs/sample/cifar10_sample.yaml`, matching sample_cifar10.sh.
#
# Any further args (`--override ...`, `--dataset_root ...`) are
# forwarded to `sample.py` unchanged.
# ----------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <ckpt_path> [config_yaml] [extra sample.py args ...]" >&2
    echo "Example: $0 runs/dit_s2_cifar/ckpt/step_000200000.pt" >&2
    exit 2
fi

CKPT="$1"
shift

# Optional second positional: config YAML.  If the next arg starts
# with a `--`, treat it as a flag (no config override) and fall
# through to the default; otherwise consume it as the config path.
CONFIG="configs/sample/cifar10_sample.yaml"
if [[ $# -gt 0 && "$1" != --* ]]; then
    CONFIG="$1"
    shift
fi

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint file not found: $CKPT" >&2
    exit 1
fi

# Forward to sample_cifar10.sh with:
#   * an explicit --ckpt so the checkpoint path wins over whatever
#     the YAML says (see sample.py's precedence rules);
#   * --override metrics.fid=true so FID is always on for `eval`;
#   * whatever the caller passed after the ckpt/config positionals.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

exec "$SCRIPT_DIR/sample_cifar10.sh" \
    "$CONFIG" \
    --ckpt "$CKPT" \
    --override metrics.fid=true \
    "$@"

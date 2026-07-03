"""Shared runtime utilities used by both ``train.py`` and ``sample.py``.

Only truly cross-entry-point helpers live here.  Anything that is
still specific to one entry point (e.g. wandb setup, the training
loop, the sampling loop) stays in the entry point that owns it.

Modules
-------
- :mod:`runtime.dist_utils`  distributed / device helpers
- :mod:`runtime.rng`         RNG seeding + snapshot/restore helpers
- :mod:`runtime.fid_cache`   FID reference-statistics build-or-load helper
- :mod:`runtime.checkpoint`  checkpoint format version + loader
- :mod:`runtime.git`         git-HEAD SHA lookup for run provenance
"""

from runtime.checkpoint import CHECKPOINT_VERSION, load_checkpoint
from runtime.dist_utils import (
    broadcast_module_state,
    cleanup_distributed,
    setup_distributed,
)
from runtime.fid_cache import build_or_load_fid_cache
from runtime.git import get_git_sha
from runtime.rng import (
    restore_rng,
    set_seed,
    snapshot_rng,
)

__all__ = [
    "CHECKPOINT_VERSION",
    "broadcast_module_state",
    "build_or_load_fid_cache",
    "cleanup_distributed",
    "get_git_sha",
    "load_checkpoint",
    "restore_rng",
    "set_seed",
    "setup_distributed",
    "snapshot_rng",
]

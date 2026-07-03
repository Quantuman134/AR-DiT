"""RNG helpers shared by ``train.py`` and ``sample.py``.

Everything in this module operates on the *global* random-number
generators used across the codebase:

* Python's :mod:`random`
* :mod:`numpy.random`
* :mod:`torch` (CPU) and :mod:`torch.cuda` (all visible devices)

Nothing here touches ``torch.distributed``, models, datasets, or
configs — those belong elsewhere.  This module lives next to
:mod:`runtime.dist_utils` because the two are typically invoked in
sequence at process bootstrap (``setup_distributed`` then
``set_seed(seed + rank)``), but it has no import-time coupling to it.

Public surface
--------------
* :func:`set_seed`         — seed every RNG we care about.
* :func:`snapshot_rng`     — capture the *current state* of all RNGs.
* :func:`restore_rng`      — restore RNGs from a snapshot.

The snapshot/restore pair exists so that ``--resume`` produces a run
that is bit-equivalent to a non-crashed run: seeding alone is not
enough, because by step ``N`` the RNG stream has already been advanced
``N`` times (per-step ``randn_like``, timestep ``rand``, augmentation
draws, etc.).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG we care about — reproducibility per FlowMatching.md.

    Callers that want per-rank-disjoint noise (e.g. training) should
    pass ``seed + rank``; callers that want cross-rank-identical noise
    (e.g. drawing the full-``N`` validation noise once and slicing per
    shard) should pass the same seed on every rank.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_rng() -> dict[str, Any]:
    """Capture every RNG state we later need to restore on ``--resume``."""
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else []),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    """Restore RNGs from a dict produced by :func:`snapshot_rng`."""
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])

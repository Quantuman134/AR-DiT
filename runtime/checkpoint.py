"""Shared checkpoint format constants + loader.

Both ``train.py`` (for ``--resume``) and ``sample.py`` (for the input
checkpoint) load .pt files with the same version guard.  The version
constant lives here so bumping the on-disk format touches exactly one
line, and both entry points automatically pick up the new value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

# Bump when the on-disk schema written by ``save_checkpoint`` in
# ``train.py`` changes in a backward-incompatible way (Train.md §3.1).
CHECKPOINT_VERSION = 1


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a .pt checkpoint written by ``train.py``, with a version guard.

    Loads onto CPU (``map_location="cpu"``) — the caller decides which
    device to move tensors onto.  ``weights_only=False`` is required
    because the payload contains non-tensor state (config YAMLs,
    optimizer state, RNG snapshots).
    """
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version", 0) != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"checkpoint at {path} has version {payload.get('version')}, "
            f"expected {CHECKPOINT_VERSION}"
        )
    return payload

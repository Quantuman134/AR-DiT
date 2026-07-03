"""Distributed / device helpers shared by ``train.py`` and ``sample.py``.

Everything in this module is stateless: it operates on the process
environment (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``) and on the
``torch.distributed`` process group.  It never touches models,
datasets, or configs — those belong to the entry point.

RNG seeding / snapshot / restore are a separate concern and live in
:mod:`runtime.rng`; they used to live here but have no coupling to
``torch.distributed`` and were moved out for clarity.

Rank-0 responsibilities (checkpointing, EMA ownership, wandb, on-disk
grid PNG writes) are the entry point's business, not this module's.
Here we only provide the primitives:

* :func:`setup_distributed` / :func:`cleanup_distributed`
* :func:`broadcast_module_state` — used at eval time when only rank 0
  holds the "correct" weights (e.g. an EMA shadow) and every rank
  needs them for a collective-metric run.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Distributed / device setup
# ---------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int, bool, torch.device]:
    """Initialise ``torch.distributed`` if launched under torchrun.

    Returns
    -------
    (rank, world_size, local_rank, is_main, device)
        ``device`` is ``cuda:local_rank`` if CUDA is available, else CPU.
        ``is_main`` is ``True`` on rank 0 only — the rank that owns EMA,
        checkpointing, wandb, and any user-facing on-disk output.

    When neither ``RANK`` nor ``WORLD_SIZE`` is present, we fall back to
    a single-process configuration; the same code paths keep working
    without ``torchrun``.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_main = rank == 0
    return rank, world_size, local_rank, is_main, device


def cleanup_distributed() -> None:
    """Tear down the process group if one was created by :func:`setup_distributed`."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Cross-rank weight broadcast
# ---------------------------------------------------------------------------

def broadcast_module_state(module: nn.Module, src: int = 0) -> None:
    """Broadcast every parameter and buffer of ``module`` from ``src``.

    Used at eval time when only rank ``src`` holds the "correct"
    weights — for example after copying an EMA shadow (which lives
    only on rank 0) onto the online model — and every rank needs the
    same weights before a collective (FID/IS ``compute()``) is called.

    Cheap on CIFAR-scale DiT (~30M params); scales linearly with model
    size.  The module iteration order is deterministic in PyTorch (dict
    insertion order = registration order), so every rank iterates the
    same tensors in the same sequence — no name-based lookup needed.

    Buffer dtypes are unrestricted: ``dist.broadcast`` handles int,
    float, and bool tensors alike.
    """
    for p in module.parameters():
        dist.broadcast(p.data, src=src)
    for b in module.buffers():
        dist.broadcast(b.data, src=src)

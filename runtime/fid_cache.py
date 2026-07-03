"""FID reference-statistics build-or-load helper — Train.md §8.1.

Extracted from ``train.py`` so both ``train.py`` and ``sample.py`` can
call the same code path.  Behaviour is identical to the original:

- ``cache_path is None``  → return an empty-real-side metric with a warning.
- Cache exists            → every rank loads it.
- Cache does not exist    → rank 0 iterates the dataset once, saves the
                            cache, then every rank loads it.

The one-time build is intentionally single-rank (rank 0) because the
FID real-side accumulator has no cross-rank sharding story that would
also match a subsequent ``load_reference`` on a single-rank run —
building on rank 0 keeps the on-disk stats world-size-invariant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from eval.fid import FIDMetric


def build_or_load_fid_cache(
    cache_path: str | None,
    train_dataset: torch.utils.data.Dataset,
    device: torch.device,
    is_main: bool,
    world_size: int,
) -> FIDMetric | None:
    """Return a FID metric with real-side stats populated, or ``None`` if disabled.

    Parameters
    ----------
    cache_path
        Path to the ``.npz`` file holding torchmetrics' running-sum
        internal state.  If ``None``, we return a metric with an empty
        real side (validation still runs, but FID will be uncalibrated
        — this code path is a smoke-test convenience only).  A warning
        is printed once on rank 0.
    train_dataset
        Any ``Dataset`` yielding ``(image, label)`` tuples in the
        project's ``[-1, 1]`` range.  Only used when the cache is being
        built for the first time.
    device
        Where the ``FIDMetric`` should live.
    is_main
        ``True`` on rank 0 only.
    world_size
        Total number of ranks.  Used to gate a ``dist.barrier()`` when
        rank 0 finishes building the cache and other ranks are waiting.
    """
    if cache_path is None:
        if is_main:
            print("[fid] no fid_ref_stats path configured; validation FID "
                  "will run against an empty real cache and be meaningless.",
                  file=sys.stderr)
        fid = FIDMetric(feature=2048, device=device)
        return fid

    fid = FIDMetric(feature=2048, device=device)
    p = Path(cache_path)
    if p.is_file():
        if is_main:
            print(f"[fid] loading reference stats from {p}", flush=True)
        fid.load_reference(p)
    else:
        if is_main:
            print(f"[fid] cache {p} not found; building on rank 0 "
                  f"(this is a one-time ~1min cost).", flush=True)
            loader = DataLoader(
                train_dataset,
                batch_size=256,
                shuffle=False,
                num_workers=2,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )
            for imgs, _ in loader:
                fid.update_real(imgs.to(device))
            fid.save_reference(p)
            print(f"[fid] saved reference stats to {p}", flush=True)
        if world_size > 1:
            dist.barrier()
        # Every rank now loads the (freshly saved) cache.
        fid.load_reference(p)
    return fid

"""Fréchet Inception Distance — thin wrapper around torchmetrics.

Purpose
-------
Provide a single class :class:`FIDMetric` that

1. wraps :class:`torchmetrics.image.fid.FrechetInceptionDistance` at
   ``feature=2048`` (the pool3 features of Inception-V3, community
   convention for FID reporting);
2. accepts image tensors in the ``[-1, 1]`` range this project uses
   everywhere (see FlowMatching.md §1) and internally converts to the
   ``uint8 [0, 255]`` format torchmetrics expects;
3. computes the reference-set statistics **once**, caches them to a
   ``.npz`` file, and skips the ~1-minute real-set pass on subsequent
   runs — see :meth:`FIDMetric.load_reference` /
   :meth:`FIDMetric.save_reference` and doc/Train.md §8.1.

Cache file format
-----------------
The cache stores torchmetrics' running-sum internal state (three
arrays) rather than the derived ``(μ, Σ)`` pair, because those sums
are what torchmetrics uses at ``.compute()`` time — we would otherwise
need to invert ``Σ = (Σ_ff - N·μμᵀ)/(N-1)``, which is many-to-one
without also storing ``N``.

The ``.npz`` file therefore contains three arrays:

===============================  ================  ============================
Key                              Shape             Meaning
===============================  ================  ============================
``real_features_sum``            ``(feature_dim,)``   Σᵢ fᵢ  (float64)
``real_features_cov_sum``        ``(feature_dim,``    Σᵢ fᵢ fᵢᵀ  (float64)
                                  ``feature_dim)``
``real_features_num_samples``    scalar               N          (int64)
===============================  ================  ============================

.. note::
   The three key names above track torchmetrics' *internal attribute
   names* as of ``torchmetrics==1.9.x``.  If a future torchmetrics
   release renames those attributes the cache load will fail loudly at
   :func:`FIDMetric.load_reference` (``AttributeError``); nothing will
   silently break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torchmetrics.image.fid import FrechetInceptionDistance

# Torchmetrics' running-sum attribute names (v1.9.x).  Kept as a module
# constant so both save_reference and load_reference stay in sync.
_REAL_STATE_KEYS: tuple[str, ...] = (
    "real_features_sum",
    "real_features_cov_sum",
    "real_features_num_samples",
)

def _to_uint8(images: Tensor) -> Tensor:
    """Convert a batch in ``[-1, 1]`` to ``uint8`` in ``[0, 255]``.

    torchmetrics' Inception backbone requires ``uint8`` when
    ``normalize=False``.  Values are clamped defensively — a stray
    ``1.001`` from numerical noise in a sampler must not overflow.
    """
    if images.dtype == torch.uint8:
        return images
    x = (images.clamp(-1.0, 1.0) + 1.0) * 127.5
    return x.round().to(torch.uint8)

class FIDMetric:
    """Thin FID wrapper with a reference-statistics cache.

    Parameters
    ----------
    feature : int, default ``2048``
        Which InceptionV3 layer to pool from.  ``2048`` is the standard
        FID feature size and the value used in the reference paper.
    device : str or torch.device, default ``"cpu"``
        Where the underlying torchmetrics module lives.  For real
        training use ``"cuda"``.

    Notes
    -----
    All accepted image tensors have shape ``(N, 3, H, W)`` and dtype
    ``float32``/``float16``/``bfloat16`` in the range ``[-1, 1]``.  The
    only exception is when the caller has already produced ``uint8``
    tensors in ``[0, 255]``, in which case they are passed through
    verbatim.
    """

    def __init__(self, feature: int = 2048, device: str | torch.device = "cpu") -> None:
        self.feature = int(feature)
        self.device = torch.device(device)
        self._metric = FrechetInceptionDistance(
            feature=self.feature,
            normalize=False,           # we hand it uint8 ourselves
            reset_real_features=False, # keep cached real stats across compute() calls
            sync_on_compute=True,      # under DDP, all-reduce running sums at
                                       # compute() so every rank sees the same
                                       # global FID.  No-op when world_size == 1.
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Update / compute
    # ------------------------------------------------------------------ #

    def update_real(self, images: Tensor) -> None:
        """Feed a batch of *real* images to the metric.

        Only used when the reference-stat cache is being *built* — once
        loaded from disk, the real pass is skipped entirely.
        """
        self._metric.update(_to_uint8(images).to(self.device), real=True)

    def update_fake(self, images: Tensor) -> None:
        """Feed a batch of *generated* images to the metric."""
        self._metric.update(_to_uint8(images).to(self.device), real=False)

    def compute(self) -> float:
        """Return the current FID as a Python float.

        Workaround for a cuSOLVER × driver bug that makes
        ``torch.linalg.eigvals`` (called inside torchmetrics'
        ``_compute_fid``) fail with ``CUDA driver error: invalid
        argument`` on some GPU/driver combinations at the 2048×2048
        non-symmetric eigendecomposition step.

        Strategy: manually all-reduce the sharded fake-side running sums
        on CUDA using NCCL (safe; the replicated real-side sums are left
        untouched — see step 1), then migrate the metric to CPU where LAPACK's
        ``geev`` is reliable, run ``compute()`` there with torchmetrics'
        internal sync disabled, and finally restore the metric back to
        CUDA with sync re-enabled for future validation passes.
        """
        m = self._metric

        # ---- 1. Manual all-reduce on CUDA (NCCL-friendly).
        #
        # ONLY the fake-side buffers are reduced.  Each rank generates a
        # disjoint shard of the fake samples, so summing across ranks is
        # correct and gives the global fake statistics.  reset_fake()
        # zeroes these before every pass, so the reduction starts clean
        # each time.
        #
        # The real-side buffers must NOT be reduced: they are *replicated*,
        # not sharded — every rank loads the identical reference stats from
        # the same cache (see load_reference) and never updates them during
        # training.  They are therefore already global on every rank.  A
        # SUM all-reduce would multiply them by world_size, and because the
        # real buffers are never reset (reset_real_features=False,
        # reset_fake() leaves them alone), that factor compounds ×world_size
        # on every compute() call until real_features_num_samples (int64)
        # overflows to a negative value and torchmetrics' "more than one
        # sample" guard trips.  Leaving them untouched keeps the global real
        # stats correct.
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            for name in (
                "fake_features_sum",
                "fake_features_cov_sum",
                "fake_features_num_samples",
            ):
                buf = getattr(m, name)
                torch.distributed.all_reduce(buf, op=torch.distributed.ReduceOp.SUM)

        # ---- 2. Disable torchmetrics' internal sync for the CPU
        #        compute() call.  The gating flag is ``_to_sync``
        #        (an internal attribute), NOT ``sync_on_compute``.
        #        We also flip ``_should_unsync`` so torchmetrics does
        #        not try to "restore local state" from a sync that
        #        never happened.
        prev_to_sync = m._to_sync
        prev_should_unsync = m._should_unsync
        m._to_sync = False
        m._should_unsync = False

        orig_device = m.fake_features_sum.device
        try:
            # ---- 3. Migrate to CPU and run the eigen step there.
            if orig_device.type == "cuda":
                m.to("cpu")
            val = float(m.compute().item())
        finally:
            # ---- 4. Restore metric state for the next validation pass.
            if orig_device.type == "cuda":
                m.to(orig_device)
            m._to_sync = prev_to_sync
            m._should_unsync = prev_should_unsync

        return val

    def reset_fake(self) -> None:
        """Reset only the fake-side accumulators (real cache is preserved).

        This is the intended per-validation-pass reset: the reference
        statistics stay fixed, but each new sample set starts from a
        clean fake accumulator.
        """
        # Re-initialise the fake-side buffers to zero without touching
        # the real ones.  Torchmetrics has no public single-side reset,
        # so we do it by hand — same three attribute names, ``fake_``
        # prefix.
        self._metric.fake_features_sum.zero_()
        self._metric.fake_features_cov_sum.zero_()
        self._metric.fake_features_num_samples.zero_()

    # ------------------------------------------------------------------ #
    # Reference-stat cache — save / load
    # ------------------------------------------------------------------ #

    def save_reference(self, path: str | Path) -> None:
        """Save the current real-side running sums to ``path`` as ``.npz``.

        Fails loudly if no real samples have been fed yet — a zero-sample
        cache would be silently useless later.
        """
        n = int(self._metric.real_features_num_samples.item())
        if n == 0:
            raise RuntimeError(
                "FIDMetric.save_reference: no real samples have been fed "
                "via update_real(); refusing to save an empty cache."
            )
        arrays = {
            key: getattr(self._metric, key).detach().cpu().numpy()
            for key in _REAL_STATE_KEYS
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    def load_reference(self, path: str | Path) -> None:
        """Load real-side running sums from ``path`` (an ``.npz``)."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"FIDMetric.load_reference: no such file: {path}")
        with np.load(path) as data:
            missing = [k for k in _REAL_STATE_KEYS if k not in data.files]
            if missing:
                raise KeyError(
                    f"FIDMetric.load_reference: cache at {path} is missing "
                    f"required keys {missing}; found {list(data.files)}. "
                    f"Regenerate the cache."
                )
            for key in _REAL_STATE_KEYS:
                target: Tensor = getattr(self._metric, key)
                new = torch.from_numpy(data[key]).to(
                    device=target.device, dtype=target.dtype
                )
                if new.shape != target.shape:
                    raise ValueError(
                        f"FIDMetric.load_reference: shape mismatch for {key}: "
                        f"cache has {tuple(new.shape)}, metric expects "
                        f"{tuple(target.shape)}. Cache was likely built with a "
                        f"different `feature` size."
                    )
                target.copy_(new)

    # ------------------------------------------------------------------ #
    # Convenience: build reference stats from a dataset iterable
    # ------------------------------------------------------------------ #

    def build_reference_from_iterable(
        self,
        batches: Iterable[Tensor],
    ) -> None:
        """Populate the real-side stats by consuming an iterable of batches.

        Each item yielded by ``batches`` must be a ``(N, 3, H, W)`` tensor
        in ``[-1, 1]`` (the project's canonical range) or already ``uint8``.

        Typical usage::

            fid = FIDMetric(feature=2048, device="cuda")
            def real_batches():
                for imgs, _ in DataLoader(train_set, batch_size=256):
                    yield imgs
            fid.build_reference_from_iterable(real_batches())
            fid.save_reference("/data/cifar10/fid_ref_stats.npz")
        """
        for batch in batches:
            self.update_real(batch)

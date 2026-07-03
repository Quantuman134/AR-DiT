"""Inception Score — thin wrapper around torchmetrics.

Purpose
-------
Wrap :class:`torchmetrics.image.inception.InceptionScore` with

1. the standard 10-split protocol used in the reference IS paper
   (Salimans et al., 2016), so numbers are comparable to community
   values;
2. the same ``[-1, 1] → uint8`` input convention as :mod:`eval.fid`,
   so callers pass in project-native tensors and never think about the
   Inception backbone's expected range.

Notes
-----
Unlike FID, IS has *no reference distribution* — it is computed
entirely from generated samples.  There is therefore no cache to
manage; the metric is stateless across runs.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics.image.inception import InceptionScore

from eval.fid import _to_uint8  # reuse the [-1,1] -> uint8 converter

class InceptionScoreMetric:
    """Thin wrapper around torchmetrics' InceptionScore.

    Parameters
    ----------
    splits : int, default ``10``
        Number of chunks the sample set is divided into for the
        per-split KL statistic.  ``10`` is the standard IS protocol.
    device : str or torch.device, default ``"cpu"``
        Where the underlying torchmetrics module lives.

    Notes
    -----
    :meth:`compute` returns ``(mean, std)`` as a pair of Python floats
    — matching the ``mean ± std`` reporting convention used in the
    literature.
    """

    def __init__(self, splits: int = 10, device: str | torch.device = "cpu") -> None:
        self.splits = int(splits)
        self.device = torch.device(device)
        self._metric = InceptionScore(
            feature="logits_unbiased",   # torchmetrics' default; matches the reference paper
            splits=self.splits,
            normalize=False,             # we hand it uint8 ourselves
            sync_on_compute=True,        # under DDP, all-reduce accumulated
                                         # features at compute() so every rank
                                         # sees the same global IS. No-op at WS=1.
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Update / compute
    # ------------------------------------------------------------------ #

    def update(self, images: Tensor) -> None:
        """Feed a batch of generated images (``[-1, 1]`` or ``uint8``)."""
        self._metric.update(_to_uint8(images).to(self.device))

    def compute(self) -> tuple[float, float]:
        """Return ``(mean, std)`` of Inception Score across splits."""
        mean, std = self._metric.compute()
        return float(mean.item()), float(std.item())

    def reset(self) -> None:
        """Reset the internal feature accumulator."""
        self._metric.reset()

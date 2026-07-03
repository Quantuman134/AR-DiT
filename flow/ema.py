"""Exponential moving average of model parameters — see FlowMatching.md §5.

A single EMA "shadow copy" tracks one decay value ``β``.  The update
rule is::

    θ_ema ← β · θ_ema + (1 − β) · θ

applied to **every parameter** of the source model (no exclusion list).
Buffers (e.g. ``pos_embed``) are copied verbatim from the source on
every update — they are deterministic constants in this project, not
running statistics, so no smoothing is applied to them.

Multiple EMA copies in parallel (typical config:
``decays = [0.9999, 0.999]``) are managed by the training loop, which
holds a list/dict of :class:`EMA` instances and calls ``update`` on
each one per optimiser step.  We keep this file responsible for
**one** copy so it stays trivially testable.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class EMA:
    """One EMA shadow copy of a source model's parameters.

    The shadow is a deep copy of the source ``nn.Module`` at
    construction time, kept on the same device, with ``requires_grad``
    cleared on every parameter.  After construction it is updated in
    place by :meth:`update` once per optimiser step.

    The shadow module is exposed via :attr:`module` and can be used
    directly for evaluation / sampling, e.g.::

        ema = EMA(model, decay=0.9999)
        for step in range(N):
            optimiser.step()
            ema.update(model)
        ...
        with torch.no_grad():
            v = ema.module(x_t, t, y)         # sample with EMA weights

    Parameters
    ----------
    model : nn.Module
        The online model whose parameters will be tracked.  At
        construction the shadow is initialised as an exact deep copy
        — there is no bias-correction warmup (FlowMatching.md §5
        explicitly rejects it).
    decay : float
        EMA decay ``β ∈ [0, 1]``.  Higher = slower-tracking.  Typical
        values: ``0.999``, ``0.9999``.

        * ``β = 0`` ⇒ shadow == online after one update.
        * ``β = 1`` ⇒ shadow never moves from its init.
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = decay
        self.module = copy.deepcopy(model)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Apply one EMA step from ``model``'s current parameters.

        Implements ``θ_ema ← β·θ_ema + (1−β)·θ`` element-wise on every
        parameter, in place.  Buffers are copied verbatim from
        ``model`` (no smoothing) so static tables like ``pos_embed``
        stay consistent.

        The caller is responsible for ensuring ``model`` is the
        unwrapped module under DDP — i.e. pass ``model.module`` rather
        than the ``DistributedDataParallel`` wrapper.  This helper
        does not unwrap anything itself.

        Parameters
        ----------
        model : nn.Module
            The online model.  Must have the same parameter / buffer
            layout as the model passed to ``__init__`` — typically the
            *same* module continued through training.
        """
        beta = self.decay
        # Parameters: linear interpolation in place.
        ema_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            if name not in ema_params:
                raise KeyError(
                    f"online model has parameter {name!r} that the EMA "
                    f"shadow does not — module structure changed mid-run?"
                )
            ema_params[name].mul_(beta).add_(p.detach(), alpha=1.0 - beta)

        # Buffers: verbatim copy (no EMA — see module docstring).
        ema_buffers = dict(self.module.named_buffers())
        for name, b in model.named_buffers():
            if name in ema_buffers:
                ema_buffers[name].copy_(b)

    def state_dict(self) -> dict[str, Any]:
        """Return a checkpoint dict ``{decay, module_state}``.

        Suitable for ``torch.save``; restore with :meth:`load_state_dict`.
        """
        return {
            "decay": self.decay,
            "module_state": self.module.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore from a dict produced by :meth:`state_dict`.

        Hard-fails on decay mismatch — silently loading an EMA
        checkpoint with the wrong decay would invalidate the FID/IS
        comparison across runs.
        """
        if "decay" not in sd or "module_state" not in sd:
            raise KeyError(
                f"EMA state_dict must contain 'decay' and 'module_state', "
                f"got keys {list(sd.keys())}"
            )
        if sd["decay"] != self.decay:
            raise ValueError(
                f"checkpoint decay {sd['decay']} != EMA decay {self.decay}; "
                "refusing to silently change the EMA decay mid-run"
            )
        self.module.load_state_dict(sd["module_state"])

    def copy_to(self, model: nn.Module) -> None:
        """Copy this EMA's parameters & buffers onto ``model`` in place.

        Used at validation / final-eval time when we want to swap the
        online weights for an EMA snapshot.  The caller is responsible
        for saving the online weights first if it intends to restore
        them after evaluation.
        """
        model.load_state_dict(self.module.state_dict())

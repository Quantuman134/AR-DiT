"""Classifier-free-guidance helpers — see FlowMatching.md §4.

Two functions live here:

* :func:`cfg_combine` — the pure affine combination
  ``v_cfg = v_uncond + s · (v_cond − v_uncond)``.  No model, no batch
  doubling, no I/O — just the formula from §4.2.  All three Layer-3
  numerical tests in ``tests/test_flow.py`` target this function.

* :func:`guided_velocity` — the doubled-batch wrapper used by the
  Euler sampler in ``flow/sampler.py``.  It builds ``[x_t; x_t]``,
  ``[t; t]``, ``[y; y_null]``, runs ``model(...)`` **once**, splits the
  output, and calls :func:`cfg_combine`.  This is the pattern
  prescribed by §4.3 — the model has a single ``forward(x, t, y)`` and
  there is **no** ``forward_with_cfg`` method.

Keeping the algebra (``cfg_combine``) separate from the orchestration
(``guided_velocity``) lets us cover every numerical edge case
(``s = 0, 1, 2``) with parameter-free unit tests.  The wrapper is
exercised end-to-end in the sampler tests under B5.
"""

from __future__ import annotations

import torch
from torch import Tensor


def cfg_combine(v_cond: Tensor, v_uncond: Tensor, scale: float) -> Tensor:
    """Affine combination ``v_cfg = v_uncond + s · (v_cond − v_uncond)``.

    Implements FlowMatching.md §4.2.  Special cases (verified in
    ``tests/test_flow.py``):

    * ``scale == 1.0`` ⇒ ``v_cfg == v_cond`` (no guidance).
    * ``scale == 0.0`` ⇒ ``v_cfg == v_uncond`` (unconditional).
    * ``scale == 2.0`` ⇒ ``v_cfg == 2·v_cond − v_uncond``.

    Parameters
    ----------
    v_cond : Tensor of shape ``(B, C, H, W)``
        Conditional velocity ``v_θ(x_t, t, y)``.
    v_uncond : Tensor of shape ``(B, C, H, W)``
        Unconditional velocity ``v_θ(x_t, t, null)``.
    scale : float
        Guidance scale ``s``.  Typical values ``{1.0, 1.5, 2.5, 4.0}``;
        ``0.0`` and ``1.0`` are also valid (see special cases above).

    Returns
    -------
    Tensor of shape ``(B, C, H, W)``
        The guided velocity ``v_cfg``.
    """
    if v_cond.shape != v_uncond.shape:
        raise ValueError(
            f"v_cond and v_uncond must have the same shape, got "
            f"{tuple(v_cond.shape)} vs {tuple(v_uncond.shape)}"
        )
    return v_uncond + scale * (v_cond - v_uncond)


def guided_velocity(
    model: torch.nn.Module,
    x_t: Tensor,
    t: Tensor,
    y: Tensor,
    y_null: Tensor,
    scale: float,
) -> Tensor:
    """One CFG-guided velocity evaluation via the doubled-batch trick.

    Implements the snippet in FlowMatching.md §4.3 verbatim::

        x_in = torch.cat([x_t, x_t], dim=0)            # (2B, C, H, W)
        t_in = torch.cat([t,   t  ], dim=0)            # (2B,)
        y_in = torch.cat([y,   y_null], dim=0)         # (2B,)
        v    = model(x_in, t_in, y_in)
        v_cond, v_uncond = v.chunk(2, dim=0)
        v_cfg = v_uncond + s * (v_cond - v_uncond)

    A single ``model(...)`` call is used so attention can be batched
    across the conditional and unconditional halves — this is what the
    spec mandates and is faster than two separate forwards.

    Parameters
    ----------
    model : nn.Module
        Velocity network with signature ``model(x, t, y) -> v`` (i.e.
        :class:`models.dit.DiT`).
    x_t : Tensor of shape ``(B, C, H, W)``
        Current sampler state.
    t : Tensor of shape ``(B,)``
        Current sampler time, one value per batch element.
    y : Tensor of shape ``(B,)`` of dtype ``long``
        Real conditioning labels.
    y_null : Tensor of shape ``(B,)`` of dtype ``long``
        Null-class labels (``LabelEmbedder.null_id`` repeated B times).
        The caller is responsible for materialising this tensor on the
        right device — keeping it in the signature (rather than reading
        it from the model) makes the helper trivially usable with any
        velocity network, not just the one in this repo.
    scale : float
        CFG scale ``s``.

    Returns
    -------
    Tensor of shape ``(B, C, H, W)``
        The guided velocity ``v_cfg``, one per input batch element.
    """
    if x_t.ndim != 4:
        raise ValueError(f"x_t must be 4-D (B, C, H, W), got ndim={x_t.ndim}")
    B = x_t.shape[0]
    if t.shape != (B,):
        raise ValueError(f"t must have shape ({B},); got {tuple(t.shape)}")
    if y.shape != (B,):
        raise ValueError(f"y must have shape ({B},); got {tuple(y.shape)}")
    if y_null.shape != (B,):
        raise ValueError(
            f"y_null must have shape ({B},); got {tuple(y_null.shape)}"
        )

    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    y_in = torch.cat([y, y_null], dim=0)

    v = model(x_in, t_in, y_in)
    v_cond, v_uncond = v.chunk(2, dim=0)
    return cfg_combine(v_cond, v_uncond, scale)

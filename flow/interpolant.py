"""Straight-line (rectified-flow) interpolant and ground-truth velocity.

Implements §2 of ``doc/FlowMatching.md``.  This file contains only two
pure tensor functions — no state, no nn.Modules — so every test in
``tests/test_flow.py`` for these functions is a numerical-value check.

Time convention (non-standard, see FlowMatching.md §1):

    t = 0  ⇒  pure noise   (x_t = x_0)
    t = 1  ⇒  clean data   (x_t = x_1)
"""

from __future__ import annotations

import torch
from torch import Tensor


def interpolant(x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
    """Compute the straight-line interpolant ``x_t = t·x_1 + (1−t)·x_0``.

    Parameters
    ----------
    x_0 : Tensor of shape ``(B, C, H, W)``
        Sample from the noise distribution ``N(0, I)``.
    x_1 : Tensor of shape ``(B, C, H, W)``
        Sample from the data distribution.
    t : Tensor of shape ``(B,)``
        Continuous time in ``[0, 1]``, one value per batch element.
        A scalar ``t`` is **not** accepted; the caller must materialise
        the per-sample time tensor (e.g. ``t.expand(B)`` in the sampler).

    Returns
    -------
    Tensor of shape ``(B, C, H, W)``
        The interpolant state ``x_t``.

    Notes
    -----
    Endpoints are exact: at ``t = 0`` this returns ``x_0`` bit-exactly,
    and at ``t = 1`` it returns ``x_1`` bit-exactly. See FlowMatching.md
    §8.1 for the test that pins this down.
    """
    if x_0.shape != x_1.shape:
        raise ValueError(
            f"x_0 and x_1 must have the same shape, got {tuple(x_0.shape)} vs "
            f"{tuple(x_1.shape)}"
        )
    if x_0.ndim != 4:
        raise ValueError(
            f"x_0 / x_1 must be 4-D (B, C, H, W), got ndim={x_0.ndim}"
        )
    B = x_0.shape[0]
    if t.shape != (B,):
        raise ValueError(
            f"t must have shape (B,) = ({B},); got {tuple(t.shape)}. "
            "Scalar t is not accepted — broadcast it at the call site."
        )

    t_b = t.view(B, 1, 1, 1)
    return t_b * x_1 + (1.0 - t_b) * x_0


def velocity_gt(x_0: Tensor, x_1: Tensor) -> Tensor:
    """Ground-truth velocity ``v_gt = x_1 − x_0``.

    For the straight-line interpolant the velocity is **constant along
    the path** for each ``(x_0, x_1)`` pair — that's the rectified-flow
    property — so this function takes no ``t`` argument.

    Parameters
    ----------
    x_0 : Tensor of shape ``(B, C, H, W)``
        Noise sample.
    x_1 : Tensor of shape ``(B, C, H, W)``
        Data sample.

    Returns
    -------
    Tensor of shape ``(B, C, H, W)``
        The regression target for the velocity network ``v_θ``.

    Notes
    -----
    See FlowMatching.md §2 for the derivation
    (``v_gt = d/dt [t·x_1 + (1−t)·x_0] = x_1 − x_0``) and §8.2 for the
    test that pins this down.
    """
    if x_0.shape != x_1.shape:
        raise ValueError(
            f"x_0 and x_1 must have the same shape, got {tuple(x_0.shape)} vs "
            f"{tuple(x_1.shape)}"
        )
    return x_1 - x_0

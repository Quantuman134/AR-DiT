"""Flow-matching training loss — see FlowMatching.md §3.

A deliberately tiny module: just the MSE between predicted and
ground-truth velocity, reduced as a single mean over every element.
The training loop in ``train.py`` is responsible for assembling
``x_t`` (via ``flow.interpolant.interpolant``), the regression target
``v_gt`` (via ``flow.interpolant.velocity_gt``), running the model to
get ``v_pred``, and then calling this function.

We keep the loss low-level (just two velocity tensors in, scalar out)
so it is trivially reusable for AR-DiT and trivial to unit-test in
isolation.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def flow_matching_loss(v_pred: Tensor, v_target: Tensor) -> Tensor:
    """Mean squared error between predicted and ground-truth velocity.

    Implements::

        L = ‖ v_θ(x_t, t, ŷ) − v_gt ‖²       # mean over all dims

    from FlowMatching.md §3.  No timestep weighting, no per-channel
    weighting — uniform reduction over every element of the velocity
    tensor (batch, channel, height, width).

    Parameters
    ----------
    v_pred : Tensor of shape ``(B, C, H, W)``
        Network's predicted velocity, ``v_θ(x_t, t, ŷ)``.
    v_target : Tensor of shape ``(B, C, H, W)``
        Ground-truth velocity, ``v_gt = x_1 − x_0``.

    Returns
    -------
    Tensor, scalar
        The mean squared error.  Differentiable w.r.t. ``v_pred``.

    Notes
    -----
    This is exactly ``F.mse_loss(v_pred, v_target, reduction="mean")``;
    we wrap it only to give the loss a name that matches the spec and
    to keep one canonical call-site for ``train.py`` and AR-DiT to
    import.
    """
    if v_pred.shape != v_target.shape:
        raise ValueError(
            f"v_pred and v_target must have the same shape, got "
            f"{tuple(v_pred.shape)} vs {tuple(v_target.shape)}"
        )
    return F.mse_loss(v_pred, v_target, reduction="mean")

"""Forward Euler sampler for flow-matching — see FlowMatching.md §6.

We integrate ``dx/dt = v_θ(x, t, y)`` from ``t = 0`` (pure noise) to
``t = 1`` (clean data) using ``num_steps`` uniform Euler steps.  The
per-step velocity is the CFG-guided combination from §4.2, executed
via the doubled-batch trick of §4.3 (see :func:`flow.cfg.guided_velocity`).

Design notes
------------
* **Caller draws the initial noise.**  Keeping ``x_init`` as an
  argument (rather than sampling internally) puts RNG control at the
  call site.  Validation needs reproducible noise across every
  ``(EMA-tag × guidance-scale)`` pair; ``sample.py`` needs a rank-aware seed.
  ``flow/sampler.py`` should not be in the RNG business.
* **Caller passes ``y_null``.**  The sampler is model-agnostic — it
  does not read ``model.y_embedder.null_id``.  The training loop /
  ``sample.py`` builds the null-label tensor and hands it in.
* **``@torch.no_grad()`` on the whole function.**  Sampling is always
  inference; wrapping the loop guarantees no gradient buffers are
  accumulated (matters both for speed and to prevent OOM at 50k
  FID-grade sample counts).
* **Higher-order solvers are deliberately out of scope.**  See
  FlowMatching.md §6 "Why Euler-only".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from flow.cfg import guided_velocity


@torch.no_grad()
def sample(
    model: nn.Module,
    x_init: Tensor,
    y: Tensor,
    y_null: Tensor,
    num_steps: int,
    guidance_scale: float,
) -> Tensor:
    """Integrate the velocity field forward with uniform-grid Euler.

    Implements the reference snippet in FlowMatching.md §6::

        x = x_init
        ts = torch.linspace(0.0, 1.0, N + 1)
        for k in range(N):
            t_k = ts[k].expand(B)
            dt  = ts[k+1] - ts[k]                 # = 1/N
            v   = guided_velocity(model, x, t_k, y, y_null, guidance_scale)
            x   = x + dt * v
        return x

    Parameters
    ----------
    model : nn.Module
        Velocity network with signature ``model(x, t, y) -> v``
        (i.e. :class:`models.dit.DiT`).  The caller is responsible for
        putting it into the right mode (``eval()``) and picking the
        right weight set (online, or an EMA snapshot via
        :meth:`flow.ema.EMA.copy_to`).
    x_init : Tensor of shape ``(B, C, H, W)``
        Starting point ``x_0`` — typically ``torch.randn(...)`` on the
        model's device.  Drawn by the caller so RNG stays there.
    y : Tensor of shape ``(B,)`` of dtype ``long``
        Real conditioning labels.
    y_null : Tensor of shape ``(B,)`` of dtype ``long``
        Null-class labels used for the unconditional half of the
        doubled-batch CFG evaluation.  Typically
        ``torch.full((B,), model.y_embedder.null_id)`` on the same
        device.
    num_steps : int
        Number of Euler steps ``N``.  Must be ``≥ 1``.  Uniform grid:
        ``t_k = k / N``.
    guidance_scale : float
        Classifier-free-guidance scale ``s`` passed to
        :func:`guided_velocity`.  Named ``guidance_scale`` (rather than
        ``cfg_scale``) throughout the codebase to keep the letters
        "cfg" reserved for the *config* dataclass in ``configs/``.

    Returns
    -------
    Tensor of shape ``(B, C, H, W)``
        The final state ``x_N ≈ x_1`` (clean sample).

    Notes
    -----
    * Time direction is ``+dt``, **not** ``-dt`` — see FlowMatching.md
      §1.  This is the non-standard convention where ``t = 0`` is
      noise and ``t = 1`` is data.
    * The function runs under ``@torch.no_grad()``; do not rely on
      gradients being available in ``model``'s output.
    * Behaviour under DDP: the caller is responsible for calling this
      only on the ranks that need to sample (typically rank 0 for
      validation) — the sampler itself is DDP-agnostic.
    """
    if x_init.ndim != 4:
        raise ValueError(
            f"x_init must be 4-D (B, C, H, W), got ndim={x_init.ndim}"
        )
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    B = x_init.shape[0]
    if y.shape != (B,):
        raise ValueError(f"y must have shape ({B},); got {tuple(y.shape)}")
    if y_null.shape != (B,):
        raise ValueError(
            f"y_null must have shape ({B},); got {tuple(y_null.shape)}"
        )

    device = x_init.device
    dtype = x_init.dtype
    ts = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)

    x = x_init
    for k in range(num_steps):
        t_k = ts[k].expand(B)                             # (B,)
        dt = ts[k + 1] - ts[k]                            # scalar tensor
        v = guided_velocity(model, x, t_k, y, y_null, guidance_scale)
        x = x + dt * v                                    # forward Euler

    return x

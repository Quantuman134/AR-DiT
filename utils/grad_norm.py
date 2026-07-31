"""Gradient-norm inspection for training diagnostics.

Purpose
-------
Emit per-parameter-group L2 gradient norms during training so that the
zero-init cascade in ARDiTCond (and, symmetrically, the coarser dam in
AR-DiT) can be observed empirically instead of merely reasoned about.

The three DiT-family models share almost all parameter names, differing
only in a small set of E1-specific tensors.  A single regex table
(``GROUP_PATTERNS``) therefore handles all three:

*   Plain DiT: patch/timestep/label embedders, per-block attn/mlp/adaLN,
    final layer.
*   AR-DiT: the above + per-block ``attn_res_msa`` / ``attn_res_mlp``
    (each with a zero-init ``.w`` and a learnable RMSNorm scale).
*   ARDiTCond (E1): the above + the shared ``t_query_trunk`` MLP and the
    per-block per-junction heads ``W_msa`` / ``W_mlp``.

Groups whose regex matches nothing on a given model are simply omitted
from the report (not logged as zero) — a zero would be misleading, as
"the group does not exist" is a very different statement from "the group
exists and had zero gradient this step".

Granularity tiers
-----------------
* ``off``    — no work; caller should skip the helper entirely.
* ``global`` — a single scalar: total L2 across all params with a grad.
* ``group``  — the ``global`` scalar plus one scalar per group name in
  ``GROUP_PATTERNS`` (~11 lines).  This is the recommended default.
* ``block``  — the ``group`` scalars plus per-block breakdowns of every
  group whose name starts with ``blocks.`` (~4 * L extra scalars).

Wandb sectioning
----------------
Report keys use three distinct top-level prefixes so wandb renders them
in three separate collapsible sections:

*   ``grad_norm/total``                          — always logged.
*   ``grad_norm_group/<group_name>``             — from tier ``group``.
*   ``grad_norm_block/<block_idx>/<sub_name>``   — from tier ``block``.

Design constraints
------------------
*   Read-only: no ``.grad`` tensor is mutated (only reduced).
*   Rank-0 only: DDP has already all-reduced grads by the time this is
    called, so rank 0 has the correct global view.  Other ranks skip
    this entirely (caller enforces).
*   Pre-clip: called *before* ``clip_grad_norm_`` so the numbers match
    the semantics of the existing ``train/grad_norm`` scalar.
*   Cheap: one ``torch.linalg.vector_norm`` per group over a stacked
    tensor, not a Python loop over ``.norm()`` per parameter.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Valid values for ``LoggingConfig.grad_norm_inspection.granularity``.
#: Ordered from cheapest to most expensive.  Each tier includes every
#: lower tier's output.
GRAD_NORM_GRANULARITIES: tuple[str, ...] = ("off", "global", "group", "block")


#: Ordered mapping of group-name -> regex.  Kept ordered so the wandb
#: legend has a stable, semantically-meaningful sequence (embed →
#: trunk → block sub-layers → final).  A parameter is assigned to the
#: *first* group whose regex matches; there is no double-counting.
#:
#: All regexes are anchored (``^ ... $``) except where trailing suffixes
#: are intentional (e.g. any parameter *underneath* an nn.Module gets
#: matched via a trailing ``\.`` prefix on its child name).
GROUP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("patch_embed",           re.compile(r"^x_embedder\.")),
    ("t_embedder",            re.compile(r"^t_embedder\.")),
    ("y_embedder",            re.compile(r"^y_embedder\.")),
    # ARDiTCond-only shared trunk (E1).  Absent on DiT / AR-DiT.
    ("t_query_trunk",         re.compile(r"^t_query_trunk\.")),
    # Per-block sub-layers, present on all three architectures.
    ("blocks.attn",           re.compile(r"^blocks\.\d+\.attn\.")),
    ("blocks.mlp",            re.compile(r"^blocks\.\d+\.mlp\.")),
    ("blocks.adaLN",          re.compile(r"^blocks\.\d+\.adaLN_modulation\.")),
    # AR-DiT / ARDiTCond junctions.  ``.w`` is the pseudo-query
    # (zero-init in AR-DiT, bypassed in ARDiTCond); ``.rms`` is the
    # in-kernel RMSNorm scale (one-init).  Split into two groups so
    # each can be watched independently.
    ("blocks.attn_res_msa.w",     re.compile(r"^blocks\.\d+\.attn_res_msa\.w$")),
    ("blocks.attn_res_mlp.w",     re.compile(r"^blocks\.\d+\.attn_res_mlp\.w$")),
    ("blocks.attn_res_msa.rms",   re.compile(r"^blocks\.\d+\.attn_res_msa\.rms\.")),
    ("blocks.attn_res_mlp.rms",   re.compile(r"^blocks\.\d+\.attn_res_mlp\.rms\.")),
    # ARDiTCond-only per-junction linear heads (E1).  Absent on DiT /
    # AR-DiT.  Zero-init — the last dam in the E1 waterfall.
    ("blocks.W_msa",          re.compile(r"^blocks\.\d+\.W_msa\.")),
    ("blocks.W_mlp",          re.compile(r"^blocks\.\d+\.W_mlp\.")),
    ("final_layer",           re.compile(r"^final_layer\.")),
)


#: Regex used to pull the numeric block index out of any per-block
#: parameter name.  Kept separate from ``GROUP_PATTERNS`` because the
#: block tier is a *cross-cut* of the group tier.
_BLOCK_INDEX_RE: re.Pattern[str] = re.compile(r"^blocks\.(\d+)\.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify(name: str) -> str | None:
    """Return the group name for a given parameter, or ``None`` if it
    matches no group.  First match wins."""
    for group_name, pattern in GROUP_PATTERNS:
        if pattern.match(name):
            return group_name
    return None


def _block_index(name: str) -> int | None:
    """Extract the integer block index from a ``blocks.<i>.*`` name, or
    ``None`` if the name is not per-block."""
    m = _BLOCK_INDEX_RE.match(name)
    return int(m.group(1)) if m is not None else None


def _l2_of(grads: Iterable[torch.Tensor]) -> torch.Tensor:
    """L2 norm across a heterogeneously-shaped bag of gradient tensors.

    Uses ``torch.linalg.vector_norm`` per tensor followed by a single
    ``torch.linalg.vector_norm`` over the resulting stack — one CUDA
    launch per input tensor plus one small reduction, no Python-level
    scalar loops.  Returns a 0-dim tensor on the same device as the
    inputs; the caller is responsible for calling ``.item()``.
    """
    # Materialise once so we can both check emptiness and iterate twice.
    tensors = [g for g in grads]
    if not tensors:
        # Caller-visible sentinel: an empty group.  Represented as NaN
        # so it is visually distinct from a genuine "grad ≈ 0" reading.
        return torch.tensor(float("nan"))
    per_tensor = torch.stack([
        torch.linalg.vector_norm(g.detach(), ord=2) for g in tensors
    ])
    return torch.linalg.vector_norm(per_tensor, ord=2)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_grad_norm_report(
    model: nn.Module,
    *,
    granularity: str,
) -> dict[str, float]:
    """Return a flat ``{wandb_key: scalar}`` dict of gradient L2 norms.

    Must be called **after** ``loss.backward()`` and **before**
    ``optimizer.zero_grad()``.  For DDP training, must be called after
    DDP's implicit all-reduce (which happens synchronously inside
    ``backward()`` unless ``no_sync()`` is active) so the grads seen on
    rank 0 are already globally averaged.

    Args:
        model: The **unwrapped** ``nn.Module`` (not the DDP wrapper).
            The helper walks ``named_parameters()`` and only inspects
            parameters that have ``requires_grad=True`` and a non-None
            ``.grad``.
        granularity: One of ``GRAD_NORM_GRANULARITIES``.
            *   ``"off"``    → returns ``{}`` (caller should not even call
                us in this case, but the branch is here for safety).
            *   ``"global"`` → returns ``{"grad_norm/total": ...}``.
            *   ``"group"``  → the above plus one key per matched group
                under ``grad_norm_group/<group_name>``.
            *   ``"block"``  → the above plus per-block sub-splits under
                ``grad_norm_block/<block_idx:02d>/<sub_name>``.  Groups
                that do not have a ``blocks.`` prefix are unaffected.

    Returns:
        Flat dict of ``str -> float``.  Missing groups are omitted, not
        logged as zero.  All scalars are computed as post-``.item()``
        Python floats so the dict is safe to hand to ``wandb.log``.

    Raises:
        ValueError: if ``granularity`` is not in ``GRAD_NORM_GRANULARITIES``.
    """
    if granularity not in GRAD_NORM_GRANULARITIES:
        raise ValueError(
            f"granularity={granularity!r} not in {GRAD_NORM_GRANULARITIES}"
        )
    if granularity == "off":
        return {}

    # ------------------------------------------------------------------
    # Pass 1 — bucket every gradient by (group, block_idx) in a single
    # walk over named_parameters().  Skip params without .grad set.
    # ------------------------------------------------------------------
    all_grads: list[torch.Tensor] = []
    group_grads: dict[str, list[torch.Tensor]] = {}
    # (group_name, block_idx) -> list of grads.  Populated only when the
    # requested tier is "block"; otherwise left empty to save memory.
    block_grads: dict[tuple[str, int], list[torch.Tensor]] = {}

    want_block = (granularity == "block")

    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        g = param.grad
        all_grads.append(g)

        if granularity == "global":
            continue  # nothing else to bucket

        group = _classify(name)
        if group is None:
            # A parameter that no regex matches.  This is a soft signal
            # that ``GROUP_PATTERNS`` is out of date w.r.t. the current
            # model — but not a hard error; the total scalar still
            # accounts for it, and we simply do not attribute it to a
            # named group.
            continue
        group_grads.setdefault(group, []).append(g)

        if want_block:
            idx = _block_index(name)
            if idx is not None:
                block_grads.setdefault((group, idx), []).append(g)

    # ------------------------------------------------------------------
    # Pass 2 — reduce each bucket.  All tensors coming out of Pass 1
    # live on the same device (the model's) so no cross-device moves.
    # ------------------------------------------------------------------
    report: dict[str, float] = {}

    if all_grads:
        report["grad_norm/total"] = _l2_of(all_grads).item()
    else:
        report["grad_norm/total"] = float("nan")

    if granularity == "global":
        return report

    for group_name, grads in group_grads.items():
        report[f"grad_norm_group/{group_name}"] = _l2_of(grads).item()

    if want_block:
        # Sort by (block_idx, group_name) for a stable wandb legend.
        for (group_name, idx), grads in sorted(
            block_grads.items(), key=lambda kv: (kv[0][1], kv[0][0])
        ):
            # Strip the "blocks." prefix from the group name so the
            # wandb key is e.g.  grad_norm_block/00/W_msa , not
            #                    grad_norm_block/00/blocks.W_msa .
            sub = group_name.removeprefix("blocks.")
            report[f"grad_norm_block/{idx:02d}/{sub}"] = _l2_of(grads).item()

    return report


__all__ = [
    "GRAD_NORM_GRANULARITIES",
    "GROUP_PATTERNS",
    "compute_grad_norm_report",
]

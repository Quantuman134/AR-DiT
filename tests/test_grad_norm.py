"""Unit tests for utils/grad_norm.py.

Coverage
--------
*   ``_classify`` maps every parameter name in the three shipped
    architectures (DiT / AR-DiT / ARDiTCond) to some group.
*   Ordering of ``GROUP_PATTERNS`` — first-match-wins — keeps
    ``.attn_res_*.w`` and ``.attn_res_*.rms.*`` separate.
*   ``compute_grad_norm_report`` returns:
        *   ``{}`` when granularity='off';
        *   only ``grad_norm/total`` when granularity='global';
        *   ``grad_norm/total`` + ``grad_norm_group/*`` when granularity='group';
        *   also ``grad_norm_block/*`` when granularity='block'.
*   Only parameters with ``.grad`` populated contribute; params with
    ``requires_grad=False`` or ``.grad is None`` are silently skipped.
*   Groups that match no parameter are omitted from the report entirely
    (not logged as zero) — the "does not exist" vs. "exists but zero"
    distinction matters for zero-init diagnostics.
*   Bad granularity string raises ``ValueError``.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from utils.grad_norm import (
    GRAD_NORM_GRANULARITIES,
    GROUP_PATTERNS,
    compute_grad_norm_report,
)


# ---------------------------------------------------------------------------
# Fixtures — small toy models that mimic the DiT/AR-DiT/ARDiTCond
# parameter-naming conventions without pulling in the real models
# (which would drag in the whole flow-matching stack for a unit test).
# ---------------------------------------------------------------------------

def _fake_block_dit() -> nn.Module:
    """A minimal DiT-block-shaped module: attn + mlp + adaLN."""
    b = nn.Module()
    b.attn = nn.Linear(4, 4)                        # blocks.<i>.attn.*
    b.mlp = nn.Linear(4, 4)                         # blocks.<i>.mlp.*
    b.adaLN_modulation = nn.Sequential(nn.Linear(4, 4))  # blocks.<i>.adaLN_modulation.*
    return b


def _fake_block_ar_dit() -> nn.Module:
    """A minimal AR-DiT-block-shaped module: DiT block + two junctions."""
    b = _fake_block_dit()

    ar_msa = nn.Module()
    ar_msa.w = nn.Parameter(torch.zeros(4))         # blocks.<i>.attn_res_msa.w
    ar_msa.rms = nn.RMSNorm(4)                      # blocks.<i>.attn_res_msa.rms.weight
    b.attn_res_msa = ar_msa

    ar_mlp = nn.Module()
    ar_mlp.w = nn.Parameter(torch.zeros(4))
    ar_mlp.rms = nn.RMSNorm(4)
    b.attn_res_mlp = ar_mlp
    return b


def _fake_block_ar_dit_cond() -> nn.Module:
    """A minimal ARDiTCond-block-shaped module: AR-DiT block + per-junction heads."""
    b = _fake_block_ar_dit()
    b.W_msa = nn.Linear(4, 4, bias=False)           # blocks.<i>.W_msa.weight
    b.W_mlp = nn.Linear(4, 4, bias=False)
    return b


def _fake_model(kind: str, *, num_blocks: int = 2) -> nn.Module:
    """Build a fake model whose ``named_parameters()`` matches the naming
    convention of the given architecture.  ``kind`` in {'dit', 'ar_dit',
    'ar_dit_cond'}."""
    if kind == "dit":
        block_ctor = _fake_block_dit
    elif kind == "ar_dit":
        block_ctor = _fake_block_ar_dit
    elif kind == "ar_dit_cond":
        block_ctor = _fake_block_ar_dit_cond
    else:                                                # pragma: no cover
        raise ValueError(kind)

    m = nn.Module()
    m.x_embedder = nn.Linear(4, 4)                       # patch_embed
    m.t_embedder = nn.Linear(4, 4)
    m.y_embedder = nn.Embedding(3, 4)
    if kind == "ar_dit_cond":
        m.t_query_trunk = nn.Linear(4, 4)
    m.blocks = nn.ModuleList([block_ctor() for _ in range(num_blocks)])
    m.final_layer = nn.Linear(4, 4)
    return m


def _populate_grads(model: nn.Module, *, value: float = 1.0) -> None:
    """Set ``p.grad`` on every parameter to a filled tensor.  Value is a
    Python float; the resulting L2 norms are deterministic under
    ``torch.linalg.vector_norm``."""
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


# ---------------------------------------------------------------------------
# _classify — every parameter in each fake model must be classified
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["dit", "ar_dit", "ar_dit_cond"])
def test_every_parameter_matches_some_group(kind):
    """Every parameter in the three architectures must fall into some
    group.  Missing coverage would silently drop parameters from the
    report — the exact regression this test guards against."""
    m = _fake_model(kind)
    unmatched: list[str] = []
    for name, _ in m.named_parameters():
        for _, pattern in GROUP_PATTERNS:
            if pattern.match(name):
                break
        else:
            unmatched.append(name)
    assert not unmatched, f"unclassified params in {kind}: {unmatched}"


def test_first_match_wins_split_attnres_w_vs_rms():
    """The AR-DiT junction has both ``.w`` (a single parameter) and
    ``.rms.weight`` (from an nn.RMSNorm).  The regex ordering must put
    them in different groups \u2014 not both in ``blocks.attn_res_msa.w``."""
    m = _fake_model("ar_dit", num_blocks=1)
    w_group = None
    rms_group = None
    for name, _ in m.named_parameters():
        for group, pattern in GROUP_PATTERNS:
            if pattern.match(name):
                if name.endswith("attn_res_msa.w"):
                    w_group = group
                elif name.endswith("attn_res_msa.rms.weight"):
                    rms_group = group
                break
    assert w_group == "blocks.attn_res_msa.w"
    assert rms_group == "blocks.attn_res_msa.rms"


# ---------------------------------------------------------------------------
# compute_grad_norm_report — granularity tiers
# ---------------------------------------------------------------------------

def test_granularity_off_returns_empty():
    m = _fake_model("dit")
    _populate_grads(m)
    assert compute_grad_norm_report(m, granularity="off") == {}


def test_granularity_global_returns_only_total():
    m = _fake_model("dit")
    _populate_grads(m)
    r = compute_grad_norm_report(m, granularity="global")
    assert set(r.keys()) == {"grad_norm/total"}
    assert r["grad_norm/total"] > 0


def test_granularity_group_contains_total_and_group_keys():
    m = _fake_model("ar_dit_cond")
    _populate_grads(m)
    r = compute_grad_norm_report(m, granularity="group")

    assert "grad_norm/total" in r
    # E1-specific groups should be present.
    assert "grad_norm_group/t_query_trunk" in r
    assert "grad_norm_group/blocks.W_msa" in r
    assert "grad_norm_group/blocks.W_mlp" in r
    # Shared DiT-family groups should be present.
    assert "grad_norm_group/patch_embed" in r
    assert "grad_norm_group/final_layer" in r
    # No block-tier keys yet.
    assert not any(k.startswith("grad_norm_block/") for k in r)


def test_granularity_block_adds_per_block_keys():
    m = _fake_model("ar_dit_cond", num_blocks=3)
    _populate_grads(m)
    r = compute_grad_norm_report(m, granularity="block")

    # All group-tier keys still present.
    assert "grad_norm/total" in r
    assert "grad_norm_group/blocks.W_msa" in r

    # Per-block keys: one per (block_idx, sub_name) pair.  Idx is
    # zero-padded to two digits.
    block_keys = [k for k in r if k.startswith("grad_norm_block/")]
    assert len(block_keys) > 0
    # Every block_idx from 0..2 must appear at least once.
    seen_idx = {re.match(r"^grad_norm_block/(\d+)/", k).group(1)
                for k in block_keys}
    assert seen_idx == {"00", "01", "02"}
    # E1's per-block W_msa should be per-block-broken-out.
    assert any(k.endswith("/W_msa") for k in block_keys)


# ---------------------------------------------------------------------------
# Behavioural details — grad presence / absence
# ---------------------------------------------------------------------------

def test_parameters_without_grad_are_skipped():
    """If a parameter has ``.grad is None`` (e.g. very first step for
    a leaf that never received grad, or ``requires_grad=False``), the
    helper must silently skip it \u2014 not raise, not include it as zero."""
    m = _fake_model("dit")
    _populate_grads(m)
    # Drop the grad on x_embedder deliberately.
    for p in m.x_embedder.parameters():
        p.grad = None

    r = compute_grad_norm_report(m, granularity="group")
    # patch_embed must be omitted from the report entirely.
    assert "grad_norm_group/patch_embed" not in r
    # But other groups must still be present.
    assert "grad_norm_group/final_layer" in r


def test_requires_grad_false_is_skipped():
    m = _fake_model("dit")
    _populate_grads(m)
    for p in m.x_embedder.parameters():
        p.requires_grad_(False)
        p.grad = None  # simulate what optimizer.zero_grad(set_to_none=True) would leave

    r = compute_grad_norm_report(m, granularity="group")
    assert "grad_norm_group/patch_embed" not in r


def test_group_not_present_is_omitted_not_zero():
    """Plain DiT has no ``t_query_trunk`` / ``W_msa`` / ``W_mlp`` / no
    ``attn_res_*`` parameters.  Those groups must be missing from the
    report entirely \u2014 emitting them as 0.0 would falsely imply 'zero
    gradient this step' when the truth is 'group does not exist'."""
    m = _fake_model("dit")
    _populate_grads(m)
    r = compute_grad_norm_report(m, granularity="group")

    for missing_group in (
        "t_query_trunk",
        "blocks.W_msa",
        "blocks.W_mlp",
        "blocks.attn_res_msa.w",
        "blocks.attn_res_mlp.w",
        "blocks.attn_res_msa.rms",
        "blocks.attn_res_mlp.rms",
    ):
        assert f"grad_norm_group/{missing_group}" not in r


def test_bad_granularity_raises_value_error():
    m = _fake_model("dit")
    _populate_grads(m)
    with pytest.raises(ValueError):
        compute_grad_norm_report(m, granularity="per-parameter")


# ---------------------------------------------------------------------------
# Sanity — GRAD_NORM_GRANULARITIES is the canonical set
# ---------------------------------------------------------------------------

def test_grad_norm_granularities_canonical_order():
    """The tier order matters for docs and for the schema's allowlist \u2014
    cheapest to most expensive.  Guard against reordering by accident."""
    assert GRAD_NORM_GRANULARITIES == ("off", "global", "group", "block")

"""Layer 2 tests: whole-model tests for :class:`models.ar_dit.ARDiT`.

Structural mirror of tests/test_dit.py, with AR-DiT-specific additions
called for by doc/AR_DiT.md §12:

- ``test_ar_dit_forward_shape_and_dtype``: sanity forward.
- ``test_ar_dit_zero_init_output_is_zero``: doc §12 acceptance criterion —
  ``ARDiT(x, t, y) == 0`` bit-exact at init, matching baseline DiT's
  behaviour but via a different internal path (see §10).
- ``test_ar_dit_zero_init_internal_scaling``: diagnostic — the input to
  ``FinalLayer`` at step 0 equals ``v_0 / (2L + 1)`` (equal-weight mean
  over the cache ``[v_0, 0, 0, ..., 0]`` of length ``2L + 1``). Catches
  a class of bugs where AttnRes silently degenerates to identity
  residual.
- ``test_ar_dit_param_count_diff``: analytical param diff vs baseline
  DiT is exactly ``2L * 2 * D``.
- ``test_ar_dit_smoke_roundtrip``: forward + MSE + backward, no NaN,
  every trainable parameter receives a gradient.

.. warning::

   **This test file is provisional — written but not yet reviewed by the
   project owner.** A green run means the tests are internally
   consistent with the code they exercise, not that the specified
   behaviour matches the paper's intent. See ``doc/Plan.md`` Roadmap
   row 6.
"""

from __future__ import annotations

import torch

from models.ar_dit import ARDiT, ARDiTBlock
from models.dit import DiT


# ---------------------------------------------------------------------------
# Shared small-model config — matches tests/test_dit.py's tiny DiT so per-test
# cost stays negligible.
# ---------------------------------------------------------------------------
_MODEL_KWARGS = dict(
    input_size=8,
    in_channels=3,
    patch_size=2,
    hidden_size=32,
    depth=3,
    num_heads=4,
    mlp_ratio=2.0,
    num_classes=10,
    class_dropout_prob=0.0,
)


def _make_batch(B: int = 2):
    torch.manual_seed(0)
    x = torch.randn(B, _MODEL_KWARGS["in_channels"],
                    _MODEL_KWARGS["input_size"], _MODEL_KWARGS["input_size"])
    t = torch.rand(B)
    y = torch.randint(0, _MODEL_KWARGS["num_classes"], (B,))
    return x, t, y


# ---------------------------------------------------------------------------
# Forward — shape & dtype
# ---------------------------------------------------------------------------

def test_ar_dit_forward_shape_and_dtype():
    """Forward returns the same shape/dtype as the input image tensor."""
    model = ARDiT(**_MODEL_KWARGS).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


# ---------------------------------------------------------------------------
# Zero-init output — doc §12 primary acceptance criterion
# ---------------------------------------------------------------------------

def test_ar_dit_zero_init_output_is_zero():
    """``ARDiT(x, t, y) == 0`` bit-exact at initialisation.

    Mechanism (doc §10): every adaLN gate is zero at init, so
    ``v_i = 0`` for ``i >= 1``. AttnRes junctions produce an
    equal-weight mean of ``[v_0, 0, 0, ...]`` at every depth, so the
    input to ``FinalLayer`` is ``v_0 / (2L + 1)``. But
    ``FinalLayer.linear`` is zero-inited, so the model output is
    exactly zero regardless of what enters — matching baseline DiT.
    """
    model = ARDiT(**_MODEL_KWARGS).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert torch.equal(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# Zero-init internal scaling — doc §12 diagnostic
# ---------------------------------------------------------------------------

def test_ar_dit_zero_init_internal_scaling():
    """The residual-stream state entering ``FinalLayer`` at init equals
    ``v_0 / (2L + 1)``.

    ``v_0 = patch_embed(x) + pos_embed``. At init every sub-layer output
    is zero, so after ``L`` blocks the cache is
    ``[v_0, 0, 0, ..., 0]`` of length ``2L + 1``. Each junction has
    ``w = 0``, so all attention weights are uniform ``1/l``. Therefore
    the output of the very last MLP junction is the equal-weight mean
    of that cache, which is exactly ``v_0 / (2L + 1)``.

    This diagnostic complements the bit-exact zero-output test above:
    a bug that silently reverted AttnRes to identity residual would
    give ``v_0`` here (not ``v_0 / (2L + 1)``) but would still produce
    a zero model output — the bit-exact test alone cannot catch it.
    """
    model = ARDiT(**_MODEL_KWARGS).eval()
    x, _, y = _make_batch()

    # Compute the expected v_0 exactly the way the model does.
    v0 = model.x_embedder(x) + model.pos_embed                 # (B, N, D)

    # Capture the residual-stream state entering FinalLayer via forward hook.
    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, args, _output):
        captured["h_in"] = args[0].detach().clone()

    handle = model.final_layer.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = model(*_make_batch())  # y here is irrelevant — v_0 doesn't depend on it.
            # Re-run with the deterministic batch so hook capture matches v_0.
            _ = model(x, torch.rand(x.shape[0]), y)
        h_in = captured["h_in"]
    finally:
        handle.remove()

    L = _MODEL_KWARGS["depth"]
    torch.testing.assert_close(h_in, v0 / (2 * L + 1), atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# Parameter count diff — doc §12
# ---------------------------------------------------------------------------

def test_ar_dit_param_count_diff():
    """Params(ARDiT) - Params(DiT) == 2L * 2 * D exactly.

    Breakdown (doc §4):

    * 2L pseudo-queries of size D  → 2L·D scalars
    * 2L RMSNorm scales of size D  → 2L·D scalars
    * Total added                   → 2L · 2 · D scalars
    """
    ar = ARDiT(**_MODEL_KWARGS)
    dit = DiT(**_MODEL_KWARGS)
    n_ar = sum(p.numel() for p in ar.parameters())
    n_dit = sum(p.numel() for p in dit.parameters())
    L, D = _MODEL_KWARGS["depth"], _MODEL_KWARGS["hidden_size"]
    assert n_ar - n_dit == 2 * L * 2 * D


# ---------------------------------------------------------------------------
# Smoke round-trip — doc §12
# ---------------------------------------------------------------------------

def test_ar_dit_smoke_roundtrip():
    """Full forward + MSE loss + backward. No NaN; every trainable
    parameter receives a gradient.

    Runs in train mode (with class dropout still off in ``_MODEL_KWARGS``
    to keep the test deterministic) so LayerNorm/adaLN paths that
    behave differently in ``model.train()`` are also exercised.
    """
    torch.manual_seed(0)
    model = ARDiT(**_MODEL_KWARGS).train()
    x, t, y = _make_batch(B=3)
    target = torch.randn_like(x)

    out = model(x, t, y)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


# ---------------------------------------------------------------------------
# Cache length invariant (extra — not in doc §12 but worth locking)
# ---------------------------------------------------------------------------

def test_ar_dit_cache_length_invariant():
    """After a full forward, the number of tensors passed through
    ``FinalLayer`` corresponds to a cache of length ``2L + 1``.

    We check this indirectly by verifying every ``ARDiTBlock`` grew the
    (fresh) cache by exactly 2 entries.
    """
    model = ARDiT(**_MODEL_KWARGS).eval()
    x, t, y = _make_batch()

    # Count how many times each block's junctions are called via hooks.
    call_counts: dict[str, int] = {"msa": 0, "mlp": 0}

    def _msa(*_a, **_kw):
        call_counts["msa"] += 1

    def _mlp(*_a, **_kw):
        call_counts["mlp"] += 1

    handles = []
    for blk in model.blocks:
        assert isinstance(blk, ARDiTBlock)
        handles.append(blk.attn_res_msa.register_forward_hook(_msa))
        handles.append(blk.attn_res_mlp.register_forward_hook(_mlp))
    try:
        with torch.no_grad():
            _ = model(x, t, y)
    finally:
        for h in handles:
            h.remove()

    L = _MODEL_KWARGS["depth"]
    assert call_counts["msa"] == L
    assert call_counts["mlp"] == L

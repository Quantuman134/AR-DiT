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

E1 (time-conditioned pseudo-query) additions — doc/AR_DiT.md §9a.8:

- ``test_ardit_cond_forward_shape_and_dtype``: parallel to the v1
  forward test.
- ``test_ardit_cond_zero_init_output_is_zero``: parallel §9a.5 acceptance
  criterion — ``ARDiTCond(x, t, y) == 0`` bit-exact at init.
- ``test_ardit_cond_zero_init_uniform_mix``: diagnostic — the last
  block's residual state at init equals ``v_0 / (2L + 1)``, same as
  v1, because ``q_l ≡ 0`` at step 0 (§9a.5).
- ``test_ardit_cond_param_count_diff``: analytical param diff vs v1
  ``ARDiT`` is exactly ``2·D² + 2·D + 2L·D²``.
- ``test_ardit_cond_time_dependence``: smoking-gun — after perturbing
  the LAST block's ``W_mlp.weight`` away from zero, changing ``t``
  alone (holding ``x, y`` fixed) changes the model output. Confirms
  the E1 path actually depends on ``t``.
- ``test_ardit_cond_grad_flow``: forward + MSE + backward; every E1
  parameter is structurally reachable from the loss graph
  (``p.grad is not None`` and finite). A weaker check than "grad
  strictly non-zero" — which would be spec-broken at step 0 — but
  the correct shape for catching the real regression (E1 modules
  built but never wired into ``forward``).

.. warning::

   **This test file is provisional — written but not yet reviewed by the
   project owner.** A green run means the tests are internally
   consistent with the code they exercise, not that the specified
   behaviour matches the paper's intent. See ``doc/Plan.md`` Roadmap
   row 6.
"""

from __future__ import annotations

import torch

from models.ar_dit import ARDiT, ARDiTBlock, ARDiTCond, ARDiTCondBlock
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


# ===========================================================================
# E1 — Time-conditioned pseudo-query (doc/AR_DiT.md §9a) tests
# ===========================================================================
# Reuses ``_MODEL_KWARGS`` and ``_make_batch`` above so the E1 tests run
# at exactly the same small scale as the v1 tests. Every E1 test either
# mirrors a v1 test (shape, zero-init, param count, grad flow) or adds
# an E1-specific invariant (uniform-mix at init, time-dependence after
# perturbation).


def test_ardit_cond_forward_shape_and_dtype():
    """Forward returns the same shape/dtype as the input image tensor.

    Parallels :func:`test_ar_dit_forward_shape_and_dtype` — the E1
    model is API-identical to v1 so the same shape contract must hold.
    """
    model = ARDiTCond(**_MODEL_KWARGS).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_ardit_cond_zero_init_output_is_zero():
    """``ARDiTCond(x, t, y) == 0`` bit-exact at initialisation.

    §9a.5 acceptance criterion. Mechanism: ``W_msa.weight = 0`` and
    ``W_mlp.weight = 0`` in every block, plus ``attn_res_*.w = 0``
    (inherited from v1). Therefore ``q_l = W_l(tau) + w_l ≡ 0`` for
    every timestep ``t``, so every logit is zero, every softmax is
    uniform ``1/l``, the cache stays ``[v_0, 0, ..., 0]``, and
    ``FinalLayer.linear = 0`` zeros the output — same observable as
    v1, different internal wiring.
    """
    model = ARDiTCond(**_MODEL_KWARGS).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert torch.equal(out, torch.zeros_like(out))


def test_ardit_cond_zero_init_uniform_mix():
    """The residual-stream state entering ``FinalLayer`` at init equals
    ``v_0 / (2L + 1)``, regardless of the timestep ``t``.

    This is the E1-specific version of the v1 diagnostic
    :func:`test_ar_dit_zero_init_internal_scaling`. It goes further
    than v1: v1's ``w = 0`` makes ``α`` uniform trivially, but E1 also
    requires ``q_l = W_l(tau) + w_l ≡ 0``, which only holds if the
    per-junction heads are truly zero-inited (§9a.5). A regression
    where ``W_l`` was accidentally Xavier-inited would still pass the
    bit-exact ``output_is_zero`` test (because ``FinalLayer.linear``
    zeros the output regardless), but would break this diagnostic.

    We also verify the diagnostic is timestep-invariant at init — the
    step-0 uniform-mix property must hold for every ``t``.
    """
    model = ARDiTCond(**_MODEL_KWARGS).eval()
    x, _, y = _make_batch()

    v0 = model.x_embedder(x) + model.pos_embed                 # (B, N, D)
    L = _MODEL_KWARGS["depth"]
    expected = v0 / (2 * L + 1)

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, args, _output):
        captured["h_in"] = args[0].detach().clone()

    handle = model.final_layer.register_forward_hook(_hook)
    try:
        # Two different timesteps — both must produce the same
        # v_0 / (2L + 1) state, because q_l ≡ 0 at init makes the
        # depth mix independent of ``t``.
        for t_val in (0.01, 0.99):
            t_probe = torch.full((x.shape[0],), t_val)
            with torch.no_grad():
                _ = model(x, t_probe, y)
            torch.testing.assert_close(
                captured["h_in"], expected, atol=1e-6, rtol=1e-5,
                msg=f"E1 uniform-mix broken at t={t_val}",
            )
    finally:
        handle.remove()


def test_ardit_cond_param_count_diff():
    """Params(ARDiTCond) - Params(ARDiT) == 2·D² + 2·D + 2L·D² exactly.

    Breakdown (doc §9a.4):

    * Shared trunk ``TimeQueryTrunk``: two ``Linear(D, D)`` with bias
      → ``2 · (D² + D) = 2·D² + 2·D`` scalars.
    * Per-junction heads ``W_msa`` + ``W_mlp`` in every block, no bias
      → ``2L · D²`` scalars (one head per junction, 2L junctions).
    * Total added over v1: ``2·D² + 2·D + 2L·D²``.
    """
    cond = ARDiTCond(**_MODEL_KWARGS)
    ar = ARDiT(**_MODEL_KWARGS)
    n_cond = sum(p.numel() for p in cond.parameters())
    n_ar = sum(p.numel() for p in ar.parameters())
    L, D = _MODEL_KWARGS["depth"], _MODEL_KWARGS["hidden_size"]
    expected_diff = 2 * D * D + 2 * D + 2 * L * D * D
    assert n_cond - n_ar == expected_diff, (
        f"E1 param diff {n_cond - n_ar} != expected {expected_diff} "
        f"(L={L}, D={D})"
    )


def test_ardit_cond_time_dependence():
    """After the LAST block's ``W_mlp.weight`` is perturbed away from
    zero, changing ``t`` alone (holding ``x, y`` fixed) changes the
    model output. Smoking-gun test that the E1 path is functionally
    reachable, not just wired.

    Why perturb the *last* junction (not the first). At init every
    adaLN gate is zero (``adaLN-Zero``), so every sub-layer output
    ``v_i`` for ``i >= 1`` is zero. A ``t``-dependent junction output
    at block ``b`` becomes the input ``x`` of block ``b+1``'s norm →
    modulate → attn/mlp path, but is then multiplied by ``gate = 0``,
    yielding ``v = 0`` again — so the E1 signal is absorbed by the
    very next sub-layer's zero gate. Perturbing an early ``W_l`` is
    therefore not enough to observe ``t``-dependence at the model
    output.

    The last MLP junction's output is fed *directly* to ``FinalLayer``
    (no more adaLN gates in between), so its ``t``-dependence survives
    to the output as long as ``FinalLayer.linear`` is also unmuted.

    Steps:

    1. Perturb ``blocks[-1].W_mlp.weight`` to a non-zero tensor. This
       breaks ``q_{2L} ≡ 0`` for the final junction — its query now
       depends on ``t``.
    2. Perturb ``FinalLayer.linear.weight`` off zero, so that
       differences in the residual stream survive to the model output.
    3. Run the model twice with the same ``x, y`` but different ``t``;
       assert the outputs differ substantially.

    A regression where the ``q_override`` branch of
    :meth:`AttnResJunction.forward` silently fell back to using
    ``self.w`` (the v1 constant) would give identical outputs across
    ``t`` and fail this test.
    """
    torch.manual_seed(0)
    model = ARDiTCond(**_MODEL_KWARGS).eval()

    # Break q_l ≡ 0 in the LAST junction, and unmute FinalLayer.
    # Perturbation scale 1.0 chosen because 0.1 produces a signal
    # ``~ 1e-5`` — within numerical noise; 1.0 gives a signal
    # ``~ 1e-1`` at DiT-tiny scale — well above any plausible noise
    # floor without saturating softmax logits.
    D = _MODEL_KWARGS["hidden_size"]
    with torch.no_grad():
        model.blocks[-1].W_mlp.weight.copy_(torch.randn(D, D) * 1.0)
        model.final_layer.linear.weight.copy_(
            torch.randn_like(model.final_layer.linear.weight) * 1.0
        )

    # Same x, y — differ only in t.
    x, _, y = _make_batch()
    t_lo = torch.full((x.shape[0],), 0.05)
    t_hi = torch.full((x.shape[0],), 0.95)

    with torch.no_grad():
        out_lo = model(x, t_lo, y)
        out_hi = model(x, t_hi, y)

    # Outputs must differ substantially — modest tolerance floor of
    # 1e-2 to reject a near-zero delta that would indicate the E1 path
    # is silently unreachable. Observed signal at this setup: ~7e-1.
    delta = (out_hi - out_lo).abs().max().item()
    assert delta > 1e-2, (
        f"ARDiTCond output does not depend on t via the AttnRes path "
        f"(max |Δoutput| = {delta:.3e}). Suspect: q_override branch of "
        f"AttnResJunction not exercised, or W_mlp change did not propagate."
    )


def test_ardit_cond_grad_flow():
    """Full forward + MSE loss + backward. Every trainable parameter —
    including the shared trunk and every per-junction head — is
    *structurally reachable* from the loss graph (``p.grad is not None``)
    and has finite gradient values.

    Why not check ``grad != 0``. At **step 0** the zero-init story of
    §9a.5 (plus adaLN-Zero and ``FinalLayer.linear = 0``) makes the
    model bit-exactly zero-valued, so the *numerical* gradient is zero
    for every parameter upstream of ``FinalLayer.linear``. This is
    intentional — the same warm-up behaviour adaLN-Zero itself
    exhibits. A ``grad > 0`` check at step 0 would therefore be
    spec-broken.

    PyTorch's autograd, however, still traverses the *structure* of
    the computation graph on backward and allocates a zero ``.grad``
    tensor for every parameter that is on the backward path — but
    leaves ``.grad = None`` for parameters that are genuinely
    disconnected (never used in the forward). So checking
    ``p.grad is not None`` for every E1 parameter is the correct
    structural analogue of "the E1 code path is reachable": it
    distinguishes "connected, numerically zero at step 0" from
    "orphaned, never used in forward" without fighting the zero-init
    discipline.

    A regression where the E1 modules were instantiated but never
    called in ``forward`` (e.g. someone deleted the ``q_override``
    argument at the block level) would result in ``p.grad is None``
    for the E1 parameters — this test would catch it.
    """
    torch.manual_seed(0)
    model = ARDiTCond(**_MODEL_KWARGS).train()
    x, t, y = _make_batch(B=3)
    target = torch.randn_like(x)

    out = model(x, t, y)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    assert torch.isfinite(loss)

    # Generic pass: every trainable parameter has a finite, non-None grad.
    # Mirrors ``test_ar_dit_smoke_roundtrip`` for v1.
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # E1-specific pass: the new modules are on the backward graph.
    # ``.grad is not None`` here is the structural reachability check —
    # weaker than ``.grad != 0`` (which is spec-broken at step 0) but
    # exactly the right shape for what we want to guarantee: the E1
    # code path is not silently orphaned from the loss.
    e1_params: dict[str, torch.nn.Parameter] = {
        "t_query_trunk.fc1.weight": model.t_query_trunk.fc1.weight,
        "t_query_trunk.fc2.weight": model.t_query_trunk.fc2.weight,
    }
    for b_idx, blk in enumerate(model.blocks):
        assert isinstance(blk, ARDiTCondBlock)
        e1_params[f"blocks.{b_idx}.W_msa.weight"] = blk.W_msa.weight
        e1_params[f"blocks.{b_idx}.W_mlp.weight"] = blk.W_mlp.weight

    for name, p in e1_params.items():
        assert p.grad is not None, (
            f"E1 parameter {name} has no gradient — the E1 code path "
            f"may be orphaned from the loss graph."
        )
        assert torch.isfinite(p.grad).all(), (
            f"E1 parameter {name} has non-finite gradient."
        )

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

E2 (SANA-time-cond pseudo-query) additions — doc/AR_DiT.md §9b:

- ``test_ardit_cond_sana_forward_shape_and_dtype``: parallel forward
  test.
- ``test_ardit_cond_sana_zero_init_output_is_zero``: parallel §9b.5
  acceptance criterion — ``ARDiTCondSANA(x, t, y) == 0`` bit-exact at
  init.
- ``test_ardit_cond_sana_zero_init_uniform_mix``: diagnostic — the
  last block's residual state at init equals ``v_0 / (2L + 1)``,
  same as v1, because ``q_m(t) ≡ 0`` at step 0.
- ``test_ardit_cond_sana_param_count_diff``: analytical param diff
  vs v1 ``ARDiT`` is exactly ``2·D · (1 + num_time_bins)``.
- ``test_ardit_cond_sana_time_dependence``: smoking-gun — after
  perturbing the codebook entries for two different time bins away
  from zero, changing ``t`` alone (holding ``x, y`` fixed) changes
  the model output.
- ``test_ardit_cond_sana_time_quantisation``: bin-boundary contract
  — two ``t`` values falling in the same floor-quantisation bin
  produce bit-identical outputs; two ``t`` values in different bins
  (with a perturbed codebook) produce different outputs.
- ``test_ardit_cond_sana_grad_flow``: forward + MSE + backward; every
  E2 parameter is structurally reachable from the loss graph.
  Dormant ``attn_res_*.w`` on the SANA path (bypassed by the
  ``q_override_raw`` branch) is asserted to have no grad.

.. warning::

   **This test file is provisional — written but not yet reviewed by the
   project owner.** A green run means the tests are internally
   consistent with the code they exercise, not that the specified
   behaviour matches the paper's intent. See ``doc/Plan.md`` Roadmap
   row 6.
"""

from __future__ import annotations

import torch

from models.ar_dit import (
    ARDiT,
    ARDiTBlock,
    ARDiTCond,
    ARDiTCondBlock,
    ARDiTCondSANA,
    ARDiTCondSANABlock,
    SANATimeCondQuery,
)
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
    """Params(ARDiT) - Params(DiT) == 2L * 3 * D exactly.

    Breakdown (doc §4, updated for the symmetric-kernel fix in
    ``fix/e1-q-rmsnorm-softmax-saturation``):

    * 2L pseudo-queries ``w`` of size D           → 2L·D scalars
    * 2L key-path RMSNorm scales ``rms`` of D     → 2L·D scalars
    * 2L query-path RMSNorm scales ``q_rms`` of D → 2L·D scalars
    * Total added                                  → 2L · 3 · D scalars

    The ``q_rms`` scale is dormant on paper-strict AR-DiT (the
    ``q_override`` branch of :class:`AttnResJunction` is never taken
    here), but it lives on the module for a uniform state-dict shape
    shared with :class:`ARDiTCond`; the smoke test below verifies
    that dormancy explicitly.
    """
    ar = ARDiT(**_MODEL_KWARGS)
    dit = DiT(**_MODEL_KWARGS)
    n_ar = sum(p.numel() for p in ar.parameters())
    n_dit = sum(p.numel() for p in dit.parameters())
    L, D = _MODEL_KWARGS["depth"], _MODEL_KWARGS["hidden_size"]
    assert n_ar - n_dit == 2 * L * 3 * D


# ---------------------------------------------------------------------------
# Smoke round-trip — doc §12
# ---------------------------------------------------------------------------

def test_ar_dit_smoke_roundtrip():
    """Full forward + MSE loss + backward. No NaN; every trainable
    parameter receives a gradient **except the dormant ``q_rms``
    scales**, which live on the module for state-dict compatibility
    with :class:`ARDiTCond` but are never touched by the paper-strict
    ``q_override is None`` code path.

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
        # ``q_rms`` is exercised only on the ``q_override`` branch
        # (ARDiTCond); AR-DiT never routes gradient through it.
        if ".q_rms." in name:
            assert p.grad is None, (
                f"{name} unexpectedly received gradient on the v1 "
                "(q_override is None) code path"
            )
            continue
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
    # ``~ 1e-5`` — within numerical noise; 1.0 gives a signal well
    # above the tolerance floor without saturating softmax logits.
    #
    # Note on signal magnitude — after
    # `fix/e1-attn-scale-sqrt-d` introduced the ``1 / sqrt(D)`` scaled
    # dot-product factor on the ``q_override`` branch, the maximum
    # signal at this setup drops from ~7e-1 to ~3e-4. Two mechanisms
    # collaborate to shrink it and neither can be undone by scaling
    # the perturbation:
    #   * ``RMSNorm(q_override)`` is scale-invariant on the vector
    #     magnitude, so ``||RMSNorm(q)|| = sqrt(D)`` regardless of
    #     ``W_mlp`` magnitude — only the direction of ``q_override``
    #     depends on ``t``.
    #   * The ``1 / sqrt(D)`` factor caps the softmax argument at
    #     ``||RMSNorm(q)|| * ||RMSNorm(k)|| / sqrt(D) = sqrt(D)``
    #     (well below saturation for D=32), which is by design.
    # The residual signal ~3e-4 is still >>100× above fp32 numerical
    # noise, so it remains a valid detector of "E1 code path silently
    # unreachable" — that regression would give delta bit-exactly 0.
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

    # Outputs must differ substantially above numerical noise —
    # tolerance floor 1e-5 (>>100× fp32 rel-err at these output
    # magnitudes) rejects a near-zero delta that would indicate the
    # E1 path is silently unreachable. Observed signal at this setup:
    # ~3e-4 (post-`fix/e1-attn-scale-sqrt-d`).
    delta = (out_hi - out_lo).abs().max().item()
    assert delta > 1e-5, (
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


# ===========================================================================
# E2 — SANA-time-cond pseudo-query (doc/AR_DiT.md §9b) tests
# ===========================================================================
# Reuses ``_MODEL_KWARGS`` and ``_make_batch`` above, plus a small
# ``num_time_bins`` value chosen so bin-boundary tests can hit multiple
# bins with the same small ``_make_batch(B=...)`` set-up.

# Small ``num_time_bins`` for the E2 tests.  A value >= 4 is enough to
# exercise different-bin behaviour with only two probe times; keeping
# it small avoids paying for 50 D-vectors of parameters in every test
# instantiation.
_SANA_NUM_TIME_BINS: int = 5


def _sana_kwargs(**overrides):
    """Build ``ARDiTCondSANA`` kwargs from the shared v1 kwargs plus
    ``num_time_bins`` (and any per-test overrides)."""
    kw = dict(_MODEL_KWARGS)
    kw["num_time_bins"] = _SANA_NUM_TIME_BINS
    kw.update(overrides)
    return kw


def test_ardit_cond_sana_time_cond_query_quantisation_edges():
    """Unit-level test on :class:`SANATimeCondQuery`: the floor-
    quantiser maps ``t=0`` to bin 0, ``t=1`` to the last bin
    ``num_time_bins-1`` (clamped, not overflowing), and bin boundaries
    behave as documented.
    """
    D = _MODEL_KWARGS["hidden_size"]
    B_time = _SANA_NUM_TIME_BINS
    q = SANATimeCondQuery(hidden_size=D, num_time_bins=B_time)

    # Directly probe the private quantiser with an exhaustive set of
    # boundary cases.  ``_quantise`` is documented to clamp t=1.0 to
    # the last bin.  Note on fp32: use a coarse ``eps`` well above
    # single-precision quantisation of ``1 / B_time`` (~0.2 for
    # B_time=5), otherwise ``(1/B_time - 1e-9) * B_time`` rounds back
    # up to 1.0 and floor-quantisation moves the value out of bin 0.
    eps = 1e-3
    t = torch.tensor([
        0.0,                          # -> 0
        1.0 / B_time - eps,           # -> 0  (still inside bin 0)
        1.0 / B_time + eps,           # -> 1  (just past boundary)
        0.5,                          # -> floor(0.5 * B_time)
        1.0 - eps,                    # -> B_time - 1
        1.0,                          # -> B_time - 1  (clamped)
    ])
    expected = torch.tensor([
        0,
        0,
        1,
        int(0.5 * B_time),
        B_time - 1,
        B_time - 1,
    ], dtype=torch.long)
    got = q._quantise(t)
    assert torch.equal(got, expected), f"quantisation mismatch: {got.tolist()} vs {expected.tolist()}"


def test_ardit_cond_sana_forward_shape_and_dtype():
    """Forward returns the same shape/dtype as the input image tensor."""
    model = ARDiTCondSANA(**_sana_kwargs()).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_ardit_cond_sana_zero_init_output_is_zero():
    """``ARDiTCondSANA(x, t, y) == 0`` bit-exact at initialisation.

    §9b.5 acceptance criterion.  Mechanism: ``w_attn = w_mlp =
    phi_attn = phi_mlp = 0`` in :class:`SANATimeCondQuery`, so
    ``q_m(t) ≡ 0`` for every ``t``.  Under the un-scaled kernel
    ``exp(0 · RMSNorm(k)) = 1`` uniformly across sources; softmax over
    a uniform vector is uniform; and ``FinalLayer.linear = 0`` zeros
    the output.  Same observable as v1 / E1, different internal
    wiring.
    """
    model = ARDiTCondSANA(**_sana_kwargs()).eval()
    x, t, y = _make_batch()
    with torch.no_grad():
        out = model(x, t, y)
    assert torch.equal(out, torch.zeros_like(out))


def test_ardit_cond_sana_zero_init_uniform_mix():
    """The residual-stream state entering ``FinalLayer`` at init equals
    ``v_0 / (2L + 1)``, regardless of the timestep ``t``.

    E2-specific version of the v1 diagnostic.  Goes further than v1:
    v1's ``w = 0`` alone makes ``α`` uniform trivially, but E2 also
    requires ``q_m(t) ≡ 0`` for every ``t``, which only holds if all
    four :class:`SANATimeCondQuery` tensors are truly zero-inited
    (§9b.5).  A regression where ``phi_*`` were accidentally
    Xavier-inited would still pass ``output_is_zero`` (because
    ``FinalLayer.linear`` zeros the output), but would break this
    diagnostic.
    """
    model = ARDiTCondSANA(**_sana_kwargs()).eval()
    x, _, y = _make_batch()

    v0 = model.x_embedder(x) + model.pos_embed                 # (B, N, D)
    L = _MODEL_KWARGS["depth"]
    expected = v0 / (2 * L + 1)

    captured: dict[str, torch.Tensor] = {}

    def _hook(_module, args, _output):
        captured["h_in"] = args[0].detach().clone()

    handle = model.final_layer.register_forward_hook(_hook)
    try:
        for t_val in (0.01, 0.5, 0.99):
            t_probe = torch.full((x.shape[0],), t_val)
            with torch.no_grad():
                _ = model(x, t_probe, y)
            torch.testing.assert_close(
                captured["h_in"], expected, atol=1e-6, rtol=1e-5,
                msg=f"E2 uniform-mix broken at t={t_val}",
            )
    finally:
        handle.remove()


def test_ardit_cond_sana_param_count_diff():
    """Params(ARDiTCondSANA) - Params(ARDiT) == 2·D + 2·B·D exactly.

    Breakdown (doc §9b):

    * Depth-shared additive biases ``w_attn`` + ``w_mlp``: 2 · D
      scalars total (one D-vector per junction kind, shared across
      all 2L junctions).
    * Time codebooks ``phi_attn`` + ``phi_mlp`` of shape
      ``[num_time_bins, D]``: 2 · B · D scalars total.
    * ARDiTCondSANA adds no per-block per-junction linear heads
      (contrast with E1), so no ``L``-scaling term appears.
    """
    sana = ARDiTCondSANA(**_sana_kwargs())
    ar = ARDiT(**_MODEL_KWARGS)
    n_sana = sum(p.numel() for p in sana.parameters())
    n_ar = sum(p.numel() for p in ar.parameters())
    D = _MODEL_KWARGS["hidden_size"]
    B = _SANA_NUM_TIME_BINS
    expected_diff = 2 * D + 2 * B * D
    assert n_sana - n_ar == expected_diff, (
        f"E2 param diff {n_sana - n_ar} != expected {expected_diff} "
        f"(D={D}, num_time_bins={B})"
    )


def test_ardit_cond_sana_time_dependence():
    """Perturbing the SANA codebook so different bins carry different
    values, and unmuting ``FinalLayer.linear``, causes the model
    output to depend on ``t`` via the SANA path.

    Perturbation targets:

    1. ``time_cond_query.phi_mlp[b0] = random``,
       ``time_cond_query.phi_mlp[b1] = random`` (different rows)
       — so ``q_mlp`` differs between the two bins.  We deliberately
       perturb ``phi_mlp`` rather than ``phi_attn`` so the signal
       flows through the *last* junction (the MLP one), whose output
       feeds directly into ``FinalLayer`` without being multiplied by
       another adaLN-Zero gate.
    2. ``final_layer.linear.weight = random`` so residual-stream
       differences survive to the model output.

    A regression where the SANA path silently reused v1's per-
    junction ``self.w`` (bypassing the codebook lookup) would give
    identical outputs across ``t`` and fail this test.
    """
    torch.manual_seed(0)
    kwargs = _sana_kwargs()
    model = ARDiTCondSANA(**kwargs).eval()

    D = kwargs["hidden_size"]
    B_time = kwargs["num_time_bins"]

    # Pick two bin indices at the extremes of the codebook.
    b_lo, b_hi = 0, B_time - 1
    with torch.no_grad():
        model.time_cond_query.phi_mlp[b_lo].copy_(torch.randn(D) * 1.0)
        model.time_cond_query.phi_mlp[b_hi].copy_(torch.randn(D) * 1.0)
        # Also perturb the corresponding attn codebook rows to
        # increase signal — but the last-MLP-junction path is the
        # dominant contributor since it is not gated by adaLN-Zero.
        model.time_cond_query.phi_attn[b_lo].copy_(torch.randn(D) * 1.0)
        model.time_cond_query.phi_attn[b_hi].copy_(torch.randn(D) * 1.0)
        model.final_layer.linear.weight.copy_(
            torch.randn_like(model.final_layer.linear.weight) * 1.0
        )

    # Two probe times that fall in bins b_lo and b_hi respectively.
    # Bin i covers [i / B_time, (i + 1) / B_time), so mid-bin values
    # (i + 0.5) / B_time are safe.
    t_lo_val = (b_lo + 0.5) / B_time
    t_hi_val = (b_hi + 0.5) / B_time

    x, _, y = _make_batch()
    t_lo = torch.full((x.shape[0],), t_lo_val)
    t_hi = torch.full((x.shape[0],), t_hi_val)

    with torch.no_grad():
        out_lo = model(x, t_lo, y)
        out_hi = model(x, t_hi, y)

    # SANA uses the un-scaled kernel ``exp(q · RMSNorm(k))`` — no
    # 1/sqrt(D) attenuation, no query-side RMSNorm bounding — so
    # the observable signal is much larger than E1's ~3e-4 for the
    # same perturbation setup.  A loose 1e-4 floor still rejects a
    # bit-zero delta from a silently-unreachable SANA path.
    delta = (out_hi - out_lo).abs().max().item()
    assert delta > 1e-4, (
        f"ARDiTCondSANA output does not depend on t via the SANA path "
        f"(max |Δoutput| = {delta:.3e}). Suspect: q_override_raw branch "
        f"of AttnResJunction not exercised, or SANATimeCondQuery lookup "
        f"is broken."
    )


def test_ardit_cond_sana_time_quantisation_boundary():
    """Two ``t`` values that fall in the **same** floor-quantisation
    bin produce **bit-identical** outputs; two ``t`` values in
    different bins (with a perturbed codebook) produce **different**
    outputs.

    This is the contract enforced by :meth:`SANATimeCondQuery._quantise`:
    bin ``i`` covers ``[i / B_time, (i + 1) / B_time)``.  A regression
    where the quantiser used, say, ``round`` semantics or a different
    bin count would flip both assertions.
    """
    torch.manual_seed(0)
    kwargs = _sana_kwargs()
    model = ARDiTCondSANA(**kwargs).eval()

    D = kwargs["hidden_size"]
    B_time = kwargs["num_time_bins"]

    # Perturb the codebook so different bins actually differ, and
    # unmute ``final_layer.linear`` so the signal survives to output.
    with torch.no_grad():
        model.time_cond_query.phi_attn.copy_(torch.randn(B_time, D) * 0.5)
        model.time_cond_query.phi_mlp.copy_(torch.randn(B_time, D) * 0.5)
        model.final_layer.linear.weight.copy_(
            torch.randn_like(model.final_layer.linear.weight) * 1.0
        )

    x, _, y = _make_batch()

    # Two t values inside the SAME bin (bin 1, say).  Both must map
    # to bin 1 under floor(t * B_time), so their model outputs must
    # be bit-identical.
    bin_idx = 1
    t_a_val = bin_idx / B_time + 1e-4                # just past lower edge
    t_b_val = (bin_idx + 1) / B_time - 1e-4          # just before upper edge
    t_a = torch.full((x.shape[0],), t_a_val)
    t_b = torch.full((x.shape[0],), t_b_val)

    with torch.no_grad():
        out_a = model(x, t_a, y)
        out_b = model(x, t_b, y)
    assert torch.equal(out_a, out_b), (
        "Same-bin outputs are not bit-identical — the quantiser is "
        "leaking continuous t into the query."
    )

    # A t value in a DIFFERENT bin (bin 3) must produce a different
    # output.
    other_bin = 3
    assert other_bin != bin_idx and other_bin < B_time
    t_c_val = (other_bin + 0.5) / B_time
    t_c = torch.full((x.shape[0],), t_c_val)
    with torch.no_grad():
        out_c = model(x, t_c, y)
    delta = (out_c - out_a).abs().max().item()
    assert delta > 1e-4, (
        f"Different-bin outputs are indistinguishable "
        f"(max |Δoutput| = {delta:.3e}) — codebook lookup may be broken."
    )


def test_ardit_cond_sana_grad_flow():
    """Full forward + MSE loss + backward.  Every trainable parameter
    other than the dormant ``attn_res_*.w`` (bypassed by the SANA
    ``q_override_raw`` code path) is structurally reachable from the
    loss graph.  Both dormant ``.w`` and dormant ``q_rms`` scales are
    asserted to have no gradient — those are the parameters E2 does
    not use.
    """
    torch.manual_seed(0)
    model = ARDiTCondSANA(**_sana_kwargs()).train()
    x, t, y = _make_batch(B=3)
    target = torch.randn_like(x)

    out = model(x, t, y)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    assert torch.isfinite(loss)

    for name, p in model.named_parameters():
        # Dormant on the SANA path — the ``q_override_raw`` branch of
        # :meth:`AttnResJunction.forward` bypasses both ``self.w`` and
        # ``self.q_rms`` entirely.
        if name.endswith(".attn_res_msa.w") or name.endswith(".attn_res_mlp.w"):
            assert p.grad is None, (
                f"{name} unexpectedly received gradient on the SANA "
                "(q_override_raw) code path"
            )
            continue
        if ".q_rms." in name:
            assert p.grad is None, (
                f"{name} unexpectedly received gradient on the SANA "
                "(q_override_raw) code path"
            )
            continue
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # E2-specific structural check: the four SANA tensors and every
    # ARDiTCondSANABlock's junctions are on the backward graph.
    e2_params: dict[str, torch.nn.Parameter] = {
        "time_cond_query.w_attn":   model.time_cond_query.w_attn,
        "time_cond_query.w_mlp":    model.time_cond_query.w_mlp,
        "time_cond_query.phi_attn": model.time_cond_query.phi_attn,
        "time_cond_query.phi_mlp":  model.time_cond_query.phi_mlp,
    }
    for name, p in e2_params.items():
        assert p.grad is not None, (
            f"E2 parameter {name} has no gradient — the SANA code path "
            f"may be orphaned from the loss graph."
        )
        assert torch.isfinite(p.grad).all(), (
            f"E2 parameter {name} has non-finite gradient."
        )
    for blk in model.blocks:
        assert isinstance(blk, ARDiTCondSANABlock)

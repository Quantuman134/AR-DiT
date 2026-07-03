"""Layer 3 tests: flow-matching primitives (``flow/`` package).

Each test below corresponds to a numbered property in
``doc/FlowMatching.md`` §8.  See ``doc/Test.md`` (Layer 3) for the full
table; this file grows as Phase B modules land:

    B1 — flow/interpolant.py       → tests in this commit
    B2 — flow/loss.py              → tests appended later
    B3 — flow/cfg.py               → tests appended later
    B4 — flow/ema.py               → tests appended later
    B5 — flow/sampler.py           → tests appended later
"""

from __future__ import annotations

import torch

from flow.interpolant import interpolant, velocity_gt
from flow.loss import flow_matching_loss
from flow.cfg import cfg_combine
from flow.ema import EMA
from flow.sampler import sample
from models.dit import DiT_S_2


# ---------------------------------------------------------------------------
# interpolant — FlowMatching.md §2, §8.1, §8.2
# ---------------------------------------------------------------------------

def test_interpolant_at_t0() -> None:
    """At ``t = 0`` the interpolant is exactly ``x_0`` (FlowMatching.md §8.1)."""
    B, C, H, W = 4, 3, 8, 8
    x_0 = torch.randn(B, C, H, W)
    x_1 = torch.randn(B, C, H, W)
    t = torch.zeros(B)

    x_t = interpolant(x_0, x_1, t)

    assert x_t.shape == x_0.shape
    assert torch.equal(x_t, x_0)


def test_interpolant_at_t1() -> None:
    """At ``t = 1`` the interpolant is exactly ``x_1`` (FlowMatching.md §8.1)."""
    B, C, H, W = 4, 3, 8, 8
    x_0 = torch.randn(B, C, H, W)
    x_1 = torch.randn(B, C, H, W)
    t = torch.ones(B)

    x_t = interpolant(x_0, x_1, t)

    assert x_t.shape == x_1.shape
    assert torch.equal(x_t, x_1)


def test_velocity_gt() -> None:
    """``v_gt == x_1 − x_0`` for any ``x_0``, ``x_1`` (FlowMatching.md §8.2).

    The straight-line interpolant has a velocity that is constant along
    the path, so the function takes no ``t``.  We still test with several
    different random tensors to exercise the shape/broadcast logic.
    """
    B, C, H, W = 4, 3, 8, 8
    x_0 = torch.randn(B, C, H, W)
    x_1 = torch.randn(B, C, H, W)

    v = velocity_gt(x_0, x_1)

    assert v.shape == x_0.shape
    assert torch.equal(v, x_1 - x_0)


# ---------------------------------------------------------------------------
# loss — FlowMatching.md §3
# ---------------------------------------------------------------------------

def test_loss_is_zero_at_perfect_prediction() -> None:
    """If ``v_pred == v_target`` then ``L == 0`` exactly (FlowMatching.md §3).

    This is the basic sanity check on the regression loss: a perfect
    velocity predictor pays zero training cost.
    """
    B, C, H, W = 4, 3, 8, 8
    v_target = torch.randn(B, C, H, W)
    v_pred = v_target.clone()

    loss = flow_matching_loss(v_pred, v_target)

    assert loss.shape == ()  # scalar
    assert torch.equal(loss, torch.zeros_like(loss))


def test_loss_is_mse_reduced_over_all_dims() -> None:
    """The loss is plain MSE reduced over every dim (FlowMatching.md §3).

    Pinned down against a hand-rolled ``((v_pred − v_target) ** 2).mean()``
    to guarantee no hidden per-dim or per-timestep weighting creeps in.
    """
    B, C, H, W = 4, 3, 8, 8
    v_pred = torch.randn(B, C, H, W)
    v_target = torch.randn(B, C, H, W)

    loss = flow_matching_loss(v_pred, v_target)
    expected = ((v_pred - v_target) ** 2).mean()

    assert loss.shape == ()
    assert torch.allclose(loss, expected)


# ---------------------------------------------------------------------------
# cfg_combine — FlowMatching.md §4.2, §8.3
# ---------------------------------------------------------------------------

def test_cfg_scale_one_equals_conditional() -> None:
    """``s = 1`` ⇒ ``v_cfg ≈ v_cond`` (FlowMatching.md §8.3).

    At unit scale the affine combination collapses *algebraically* to
    the conditional branch, so guidance is a no-op.  We use
    ``torch.allclose`` rather than ``torch.equal`` because the formula
    ``v_uncond + 1·(v_cond − v_uncond)`` takes a different IEEE-754
    rounding path than the literal ``v_cond`` — equality holds in the
    reals, not bit-by-bit in float32.  A sign error in ``cfg_combine``
    would still break this assertion immediately.
    """
    B, C, H, W = 4, 3, 8, 8
    v_cond = torch.randn(B, C, H, W)
    v_uncond = torch.randn(B, C, H, W)

    v_cfg = cfg_combine(v_cond, v_uncond, scale=1.0)

    # ``allclose`` with a slightly relaxed ``atol`` because the formula
    # ``v_uncond + 1·(v_cond − v_uncond)`` introduces ~1 ULP of
    # cancellation noise vs the literal ``v_cond`` (max diff observed:
    # ~2.4e-7 in float32).  ``atol=1e-6`` is still 1000× below any
    # plausible algebraic bug.
    assert v_cfg.shape == v_cond.shape
    assert torch.allclose(v_cfg, v_cond, atol=1e-6)


def test_cfg_scale_zero_equals_unconditional() -> None:
    """``s = 0`` ⇒ ``v_cfg ≈ v_uncond`` (FlowMatching.md §8.3).

    At zero scale the combination collapses to the unconditional
    branch — useful for ablations and as a second algebraic anchor.
    Same float-comparison reasoning as the ``s = 1`` test: ``allclose``,
    not ``equal``, because the formula path differs from the literal.
    """
    B, C, H, W = 4, 3, 8, 8
    v_cond = torch.randn(B, C, H, W)
    v_uncond = torch.randn(B, C, H, W)

    v_cfg = cfg_combine(v_cond, v_uncond, scale=0.0)

    # See ``test_cfg_scale_one_equals_conditional`` for the rationale
    # behind ``atol=1e-6``.
    assert v_cfg.shape == v_uncond.shape
    assert torch.allclose(v_cfg, v_uncond, atol=1e-6)


def test_cfg_scale_two_extrapolates() -> None:
    """``s = 2`` ⇒ ``v_cfg == 2·v_cond − v_uncond`` (FlowMatching.md §8.3).

    The extrapolation case — the actual reason CFG works.  Asserts the
    full formula ``v_uncond + s·(v_cond − v_uncond)`` against a
    hand-rolled expression at a non-trivial scale.  ``allclose`` (not
    ``equal``) because we take a different floating-point path on each
    side.
    """
    B, C, H, W = 4, 3, 8, 8
    v_cond = torch.randn(B, C, H, W)
    v_uncond = torch.randn(B, C, H, W)

    v_cfg = cfg_combine(v_cond, v_uncond, scale=2.0)
    expected = 2.0 * v_cond - v_uncond

    assert v_cfg.shape == v_cond.shape
    assert torch.allclose(v_cfg, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# EMA — FlowMatching.md §5, §8.4
# ---------------------------------------------------------------------------

def _toy_model(value: float) -> torch.nn.Module:
    """Single 1-element parameter set to ``value`` — easiest possible
    fixture for testing the EMA update formula by hand.
    """
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    return m


def test_ema_single_step() -> None:
    """``β=0.9, θ=1, θ_ema=0`` → after 1 update ``θ_ema=0.1`` (FlowMatching.md §8.4).

    With shadow init at θ=0 (constructed from a model whose weight is 0),
    one update against an online model with weight=1.0 gives:
        θ_ema = 0.9 * 0 + 0.1 * 1 = 0.1
    """
    online = _toy_model(0.0)
    ema = EMA(online, decay=0.9)

    # Move online to θ=1.0 then update.
    with torch.no_grad():
        online.weight.fill_(1.0)
    ema.update(online)

    w = ema.module.weight
    assert torch.allclose(w, torch.tensor([[0.1]]), atol=1e-7)


def test_ema_two_steps() -> None:
    """After two updates with same θ=1.0: θ_ema = 0.9·0.1 + 0.1·1 = 0.19 (FlowMatching.md §8.4).

    Pins down that the update is in-place and stateful — running it
    twice is not the same as running it once with a doubled step.
    """
    online = _toy_model(0.0)
    ema = EMA(online, decay=0.9)

    with torch.no_grad():
        online.weight.fill_(1.0)
    ema.update(online)
    ema.update(online)

    w = ema.module.weight
    assert torch.allclose(w, torch.tensor([[0.19]]), atol=1e-7)


def test_ema_decay_one_freezes() -> None:
    """``β = 1`` ⇒ EMA never moves from its init.

    Defensive check: catches a sign / direction error in the update
    formula.  With β=1 the rule is θ_ema ← 1·θ_ema + 0·θ = θ_ema, so no
    matter what the online model does, the shadow stays put.
    """
    online = _toy_model(0.0)
    ema = EMA(online, decay=1.0)
    init_w = ema.module.weight.clone()

    for _ in range(5):
        with torch.no_grad():
            online.weight.add_(torch.randn_like(online.weight))
        ema.update(online)

    assert torch.equal(ema.module.weight, init_w)


def test_ema_multiple_decays() -> None:
    """Two EMA copies with different ``β`` track independently (FlowMatching.md §5).

    Mirrors the production setup where ``ema.decays = [0.9999, 0.999]``
    keeps two shadows in parallel.  After the same sequence of online
    updates, the two shadows must hold *different* values — if they
    don't, something is sharing state across instances.
    """
    online = _toy_model(0.0)
    ema_fast = EMA(online, decay=0.9)    # tracks online quickly
    ema_slow = EMA(online, decay=0.99)   # tracks online slowly

    with torch.no_grad():
        online.weight.fill_(1.0)
    for _ in range(3):
        ema_fast.update(online)
        ema_slow.update(online)

    w_fast = ema_fast.module.weight
    w_slow = ema_slow.module.weight

    # Fast EMA approaches 1.0 faster than slow EMA.
    assert (w_fast - 1.0).abs().item() < (w_slow - 1.0).abs().item()
    # And the two shadows are not the same object / value.
    assert not torch.equal(w_fast, w_slow)


def test_ema_state_dict_round_trip() -> None:
    """``state_dict`` → ``load_state_dict`` reproduces the EMA exactly.

    Required for checkpoint resume (Train.md §3.1: ``ema_states`` field).
    """
    online = _toy_model(0.0)
    ema = EMA(online, decay=0.9)
    with torch.no_grad():
        online.weight.fill_(1.0)
    ema.update(online)
    expected_w = ema.module.weight.clone()

    sd = ema.state_dict()

    # Build a fresh EMA from a fresh model and load the saved state.
    fresh_online = _toy_model(0.0)
    fresh_ema = EMA(fresh_online, decay=0.9)
    fresh_ema.load_state_dict(sd)

    assert torch.equal(fresh_ema.module.weight, expected_w)
    assert fresh_ema.decay == 0.9


def test_ema_copy_to_swaps_weights() -> None:
    """``copy_to(target)`` overwrites ``target``'s parameters with the EMA's.

    Used at validation time to swap online weights for an EMA snapshot
    before sampling (Train.md §7).
    """
    online = _toy_model(0.0)
    ema = EMA(online, decay=0.9)
    with torch.no_grad():
        online.weight.fill_(1.0)
    ema.update(online)               # ema.weight is now 0.1

    target = _toy_model(99.0)        # arbitrary distinct values
    ema.copy_to(target)

    assert torch.equal(target.weight, ema.module.weight)


# ---------------------------------------------------------------------------
# Euler sampler — FlowMatching.md §6, §8.5, §8.6
# ---------------------------------------------------------------------------

class _ZeroVelocity(torch.nn.Module):
    """Toy velocity model that returns exact zeros regardless of input."""

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros_like(x)


class _ConstantVelocity(torch.nn.Module):
    """Toy velocity model that returns a fixed ``(1, C, H, W)`` velocity
    broadcast to the batch, regardless of ``x``, ``t``, ``y``.
    """

    def __init__(self, v: torch.Tensor) -> None:
        super().__init__()
        # Register as a buffer so ``.to(device)`` moves it correctly.
        self.register_buffer("v", v)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        return self.v.expand_as(x)


class _TimeRecorder(torch.nn.Module):
    """Toy velocity model that records the ``t`` value seen on each call
    (first-half only — the CFG doubled batch feeds ``[t; t]``).
    Returns zero velocity so ``x`` never changes and thus does not affect
    subsequent calls.
    """

    def __init__(self) -> None:
        super().__init__()
        self.recorded_t: list[float] = []

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        # ``t`` has shape (2B,) due to the doubled-batch CFG trick;
        # the first B entries are the conditional half.
        B = x.shape[0] // 2
        self.recorded_t.append(t[0].item())
        # Sanity: the two halves must carry the same t.
        assert torch.equal(t[:B], t[B:])
        return torch.zeros_like(x)


def test_sampler_zero_velocity_returns_noise() -> None:
    """If the model returns 0 everywhere, ``sample(...) == x_init`` bit-exact
    (FlowMatching.md §8.5).

    ``x_{k+1} = x_k + dt · 0 = x_k`` for every step, so no rounding
    ever occurs — ``torch.equal`` is the right assertion here.
    """
    B, C, H, W = 2, 3, 8, 8
    model = _ZeroVelocity()
    x_init = torch.randn(B, C, H, W)
    y = torch.zeros(B, dtype=torch.long)
    y_null = torch.zeros(B, dtype=torch.long)

    out = sample(model, x_init, y, y_null, num_steps=8, guidance_scale=1.0)

    assert out.shape == x_init.shape
    assert torch.equal(out, x_init)


def test_sampler_constant_velocity() -> None:
    """If the model returns a constant ``v``, output is ``x_init + v``
    (FlowMatching.md §8.6).

    With uniform ``dt = 1/N``, the accumulated update is
    ``Σ_{k=0}^{N-1} dt · v = v``.  We pick ``N = 4`` so ``dt = 0.25``
    is representable exactly in float32; even so the running sum
    ``x + dt·v`` accrues a few ULPs of rounding, hence ``allclose``
    with a tight ``atol`` rather than ``equal``.
    """
    B, C, H, W = 2, 3, 8, 8
    v_val = torch.randn(1, C, H, W)
    model = _ConstantVelocity(v_val)
    x_init = torch.randn(B, C, H, W)
    y = torch.zeros(B, dtype=torch.long)
    y_null = torch.zeros(B, dtype=torch.long)

    out = sample(model, x_init, y, y_null, num_steps=4, guidance_scale=1.0)
    expected = x_init + v_val.expand_as(x_init)

    assert out.shape == x_init.shape
    assert torch.allclose(out, expected, atol=1e-6)


def test_sampler_time_direction() -> None:
    """First step uses ``t = 0``, last step uses ``t = (N-1)/N``
    (FlowMatching.md §1, §6).

    This test pins down the non-standard time convention (``t = 0`` is
    noise, ``t = 1`` is data) and forward integration (``+dt``, not
    ``-dt``).  A sign flip or off-by-one in the time grid would break
    it immediately.
    """
    B, C, H, W = 2, 3, 8, 8
    N = 5
    model = _TimeRecorder()
    x_init = torch.randn(B, C, H, W)
    y = torch.zeros(B, dtype=torch.long)
    y_null = torch.zeros(B, dtype=torch.long)

    _ = sample(model, x_init, y, y_null, num_steps=N, guidance_scale=1.0)

    ts_seen = model.recorded_t
    assert len(ts_seen) == N, f"expected {N} model calls, got {len(ts_seen)}"
    assert ts_seen[0] == 0.0
    assert abs(ts_seen[-1] - (N - 1) / N) < 1e-6
    # Strictly increasing time — forward integration.
    for a, b in zip(ts_seen, ts_seen[1:]):
        assert a < b


def test_sampler_shape_and_finiteness() -> None:
    """End-to-end with an initialised DiT: output is ``(B, C, H, W)`` and
    all finite (FlowMatching.md §6).

    Uses ``DiT_S_2`` at CIFAR shape with a small ``num_steps`` to keep
    the test cheap.  The model's random init means we can't assert
    numerical values, only shape and finiteness — the numerical
    guarantees live in the toy-model tests above.
    """
    B, C, H, W = 2, 3, 32, 32
    model = DiT_S_2(input_size=32, in_channels=C, num_classes=10).eval()
    x_init = torch.randn(B, C, H, W)
    y = torch.tensor([3, 7], dtype=torch.long)
    y_null = torch.full((B,), model.y_embedder.null_id, dtype=torch.long)

    out = sample(model, x_init, y, y_null, num_steps=4, guidance_scale=1.5)

    assert out.shape == (B, C, H, W)
    assert torch.isfinite(out).all()

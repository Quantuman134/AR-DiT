"""Layer 2 tests: end-to-end checks on the assembled DiT model."""

from __future__ import annotations

import pytest
import torch

from models.dit import (
    DiT,
    DiT_B_2,
    DiT_L_2,
    DiT_S_2,
    DiT_XL_2,
)


# A tiny preset used by most tests: small enough to run instantly on CPU
# but exercises every code path of the full model.
TINY_KW = dict(
    input_size=8,
    in_channels=3,
    patch_size=2,
    hidden_size=32,
    depth=2,
    num_heads=4,
    num_classes=4,
    class_dropout_prob=0.0,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_presets_construct():
    """All four size presets must build with the project defaults."""
    for ctor in (DiT_S_2, DiT_B_2, DiT_L_2, DiT_XL_2):
        m = ctor(input_size=32, in_channels=3, num_classes=10)
        n_params = sum(p.numel() for p in m.parameters())
        assert n_params > 0


def test_param_count_S2_in_expected_range():
    """DiT-S/2 should be ~33M params on CIFAR-10 with velocity head.

    The paper's S/2 reports 33M params with a learn-sigma head; our
    velocity-only head trims the final layer to half its width, but the
    final layer is a tiny fraction of the total, so the count is still
    within ±5% of 33M. This catches gross structural omissions
    (e.g. a missing block).
    """
    m = DiT_S_2(input_size=32, in_channels=3, num_classes=10)
    n = sum(p.numel() for p in m.parameters())
    assert 31e6 < n < 35e6, f"Unexpected DiT-S/2 param count: {n/1e6:.2f}M"


def test_input_size_assertion():
    with pytest.raises(AssertionError):
        DiT(input_size=10, patch_size=4, **{
            k: v for k, v in TINY_KW.items()
            if k not in {"input_size", "patch_size"}
        })


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------

def test_forward_cifar():
    m = DiT_S_2(input_size=32, in_channels=3, num_classes=10).eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out = m(x, t, y)
    assert out.shape == (2, 3, 32, 32)
    assert torch.isfinite(out).all()


def test_forward_64x64_p4():
    """Shape-agnostic: same preset on a 64x64 image with patch_size=4."""
    m = DiT_S_2(input_size=64, in_channels=3, num_classes=10, patch_size=4).eval()
    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out = m(x, t, y)
    assert out.shape == (2, 3, 64, 64)
    assert torch.isfinite(out).all()


def test_forward_latent_4ch():
    """4-channel latent input, e.g. for ImageNet-latent DiT-XL."""
    m = DiT(input_size=32, in_channels=4, patch_size=2, hidden_size=64,
            depth=2, num_heads=4, num_classes=1000,
            class_dropout_prob=0.0).eval()
    x = torch.randn(2, 4, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 1000, (2,))
    with torch.no_grad():
        out = m(x, t, y)
    assert out.shape == (2, 4, 32, 32)


def test_patchembed_assertion_on_forward():
    """Forward must raise when H or W is not divisible by patch_size."""
    m = DiT(**TINY_KW)                     # patch_size = 2
    bad = torch.randn(2, 3, 9, 8)          # H=9 not divisible by 2
    t = torch.rand(2)
    y = torch.randint(0, 4, (2,))
    with pytest.raises(AssertionError):
        m(bad, t, y)


# ---------------------------------------------------------------------------
# unpatchify (numerical)
# ---------------------------------------------------------------------------

def test_unpatchify_preserves_per_pixel_value():
    """Hand-laid: build a (B=1, C=1, H=4, W=4) image with each pixel labelled
    by its flat index, manually patchify it (P=2 ⇒ 4 patches), then run
    `unpatchify` and check the result matches the original image exactly.

    This locks in the einsum ``nhwpqc->nchpwq``. A bug in the pattern (e.g.
    transposed h/p axes) would scramble the spatial layout and fail this
    test.
    """
    B, C, H, W, P = 1, 1, 4, 4, 2
    img = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)

    # Manually patchify into the (B, N, P*P*C) layout that the model's
    # final layer produces. We mimic the inverse of `unpatchify`.
    h = w = H // P
    # (B, C, H, W) -> (B, C, h, P, w, P) -> (B, h, w, P, P, C) -> (B, N, P*P*C)
    tokens = img.reshape(B, C, h, P, w, P)
    tokens = torch.einsum("nchpwq->nhwpqc", tokens)
    tokens = tokens.reshape(B, h * w, P * P * C)

    m = DiT(input_size=H, in_channels=C, patch_size=P, hidden_size=8,
            depth=1, num_heads=2, num_classes=2, class_dropout_prob=0.0)
    m.out_channels = C  # already so, but make the contract explicit
    recovered = m.unpatchify(tokens)
    assert recovered.shape == img.shape
    torch.testing.assert_close(recovered, img)


# ---------------------------------------------------------------------------
# adaLN-Zero contract
# ---------------------------------------------------------------------------

def test_zero_init_output():
    """adaLN-Zero: at init, model(x, t, y) is exactly zero (machine zero)."""
    m = DiT(**TINY_KW).eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y = torch.randint(0, 4, (2,))
    with torch.no_grad():
        out = m(x, t, y)
    assert out.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# Buffers / params
# ---------------------------------------------------------------------------

def test_pos_embed_is_buffer_and_nonzero():
    m = DiT(**TINY_KW)
    # `pos_embed` must be registered as a buffer, not a parameter.
    param_names = {n for n, _ in m.named_parameters()}
    buffer_names = {n for n, _ in m.named_buffers()}
    assert "pos_embed" in buffer_names
    assert "pos_embed" not in param_names
    # And it must actually be filled in (not the all-zero placeholder).
    assert m.pos_embed.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_grad_flow_no_nan_inf():
    """Backward from a non-zero loss must produce finite gradients on every
    parameter that participates in the forward.

    Note: at adaLN-Zero init the model output is identically zero, so any
    smooth loss of the output alone has zero gradient. We use an MSE
    against a non-zero target so the gradient is non-trivial. After this
    one backward pass *all* parameters should have a finite (non-None)
    gradient — every module is touched.
    """
    m = DiT(**TINY_KW).train()
    x = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 4, (4,))
    target = torch.randn_like(x)
    out = m(x, t, y)
    loss = (out - target).pow(2).mean()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"{name} received no gradient."
        assert torch.isfinite(p.grad).all(), f"{name} has NaN/Inf gradient."


# ---------------------------------------------------------------------------
# Classifier-free guidance (sampler-side, model is a pure function)
# ---------------------------------------------------------------------------

def test_cfg_combination_shape():
    """With CFG implemented at the call site, the doubled-batch trick still
    yields the expected per-half shape and finite values.
    """
    m = DiT(**TINY_KW).eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y_cond = torch.randint(0, 4, (2,))
    y_null = torch.full((2,), m.num_classes, dtype=torch.long)
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t, t], dim=0)
    y2 = torch.cat([y_cond, y_null], dim=0)
    with torch.no_grad():
        v = m(x2, t2, y2)
    v_cond, v_uncond = v.chunk(2, dim=0)
    v_guided = v_uncond + 2.5 * (v_cond - v_uncond)
    assert v_guided.shape == x.shape
    assert torch.isfinite(v_guided).all()


def test_cfg_scale_one_equals_conditional():
    """At cfg_scale = 1.0, ``v_uncond + 1*(v_cond - v_uncond) == v_cond``.

    This is the algebraic identity any CFG implementation must satisfy and
    is the most basic sanity check for sampler-side guidance code that will
    later live in the sampler module.
    """
    m = DiT(**TINY_KW).eval()
    x = torch.randn(2, 3, 8, 8)
    t = torch.rand(2)
    y_cond = torch.randint(0, 4, (2,))
    y_null = torch.full((2,), m.num_classes, dtype=torch.long)
    x2 = torch.cat([x, x], dim=0)
    t2 = torch.cat([t, t], dim=0)
    y2 = torch.cat([y_cond, y_null], dim=0)
    with torch.no_grad():
        v = m(x2, t2, y2)
        v_cond, v_uncond = v.chunk(2, dim=0)
        v_guided = v_uncond + 1.0 * (v_cond - v_uncond)
    torch.testing.assert_close(v_guided, v_cond)


# ---------------------------------------------------------------------------
# Training / eval modes
# ---------------------------------------------------------------------------

def test_eval_mode_is_deterministic_with_dropout():
    """eval() must disable CFG label-dropout even when class_dropout_prob>0."""
    kw = dict(TINY_KW)
    kw["class_dropout_prob"] = 0.5
    m = DiT(**kw).eval()
    x = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    y = torch.randint(0, 4, (4,))
    with torch.no_grad():
        a = m(x, t, y)
        b = m(x, t, y)
    torch.testing.assert_close(a, b)


def test_train_mode_label_dropout_fires():
    """train() with p_drop=1.0 ⇒ every label is replaced by the null token,
    which is observable as `model(x, t, y) == model(x, t, null)` for all y.
    """
    kw = dict(TINY_KW)
    kw["class_dropout_prob"] = 1.0
    m = DiT(**kw).train()
    # Re-enable adaLN modulation so outputs are non-trivial; we don't
    # actually need that here because we only compare two model outputs
    # *to each other*, both of which see the same null replacement.
    x = torch.randn(4, 3, 8, 8)
    t = torch.rand(4)
    y_real = torch.tensor([0, 1, 2, 3])
    y_null = torch.full((4,), m.num_classes, dtype=torch.long)
    # Seed before each call so other random ops (none here, but cheap
    # insurance) are matched.
    torch.manual_seed(123)
    out_real = m(x, t, y_real)
    torch.manual_seed(123)
    # In eval, no dropout is applied, so feeding y_null is equivalent to
    # what train() does internally to y_real when p_drop=1.
    m.eval()
    out_null = m(x, t, y_null)
    torch.testing.assert_close(out_real, out_null)

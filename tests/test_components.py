"""Layer 1 tests: per-component unit tests for ``models/dit.py`` and
``models/ar_dit.py``.

See doc/Test.md for the test layering and which tests use hand-computed
numerical values vs. behavioural / property checks.

.. warning::

   **This test file is provisional — written but not yet reviewed by the
   project owner.** A passing run only means the tests are internally
   consistent; it does not certify that the specified behaviour matches
   the paper's intent. This applies to every test in this file, including
   the ``AttnResJunction`` and ``ARDiTBlock`` sections. See
   `doc/Plan.md` (Roadmap row 6) for the dedicated review pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from models.dit import (
    Attention,
    DiTBlock,
    FinalLayer,
    LabelEmbedder,
    MLP,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
)
from models.ar_dit import ARDiTBlock, AttnResJunction


# ---------------------------------------------------------------------------
# modulate (numerical)
# ---------------------------------------------------------------------------

def test_modulate_numerical():
    """Hand-computed: x * (1 + scale) + shift, broadcast over the token dim."""
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])          # (1, 2, 2)
    shift = torch.tensor([[10.0, 20.0]])                  # (1, 2)
    scale = torch.tensor([[1.0, 2.0]])                    # (1, 2)
    out = modulate(x, shift, scale)
    expected = torch.tensor([[[1.0 * 2 + 10, 2.0 * 3 + 20],
                              [3.0 * 2 + 10, 4.0 * 3 + 20]]])
    assert torch.equal(out, expected)


def test_modulate_zero_scale_zero_shift_is_identity():
    x = torch.randn(2, 5, 8)
    shift = torch.zeros(2, 8)
    scale = torch.zeros(2, 8)
    assert torch.equal(modulate(x, shift, scale), x)


# ---------------------------------------------------------------------------
# get_2d_sincos_pos_embed (shape + numerical)
# ---------------------------------------------------------------------------

def test_pos_embed_shape():
    pe = get_2d_sincos_pos_embed(embed_dim=64, grid_size=8)
    assert pe.shape == (64, 64)            # (N=64, D=64)
    assert pe.dtype == np.float32


def test_pos_embed_first_row_numerical():
    """At grid position (0, 0) both 1D embeddings reduce to [sin(0)..cos(0)..].

    The 1D builder concatenates ``[sin(out), cos(out)]`` along the last axis,
    so for ``pos=0`` we get ``D/4`` zeros (the sin half of emb_h) followed by
    ``D/4`` ones (the cos half of emb_h), then the same pattern from emb_w.
    """
    D = 16
    pe = get_2d_sincos_pos_embed(embed_dim=D, grid_size=4)
    first = pe[0]                          # position (h=0, w=0)
    expected = np.concatenate([
        np.zeros(D // 4),                   # sin(emb_h)
        np.ones(D // 4),                    # cos(emb_h)
        np.zeros(D // 4),                   # sin(emb_w)
        np.ones(D // 4),                    # cos(emb_w)
    ]).astype(np.float32)
    np.testing.assert_allclose(first, expected, atol=1e-6, rtol=1e-5)


def test_pos_embed_requires_even_dim():
    with pytest.raises(AssertionError):
        get_2d_sincos_pos_embed(embed_dim=15, grid_size=4)


# ---------------------------------------------------------------------------
# TimestepEmbedder.timestep_embedding (numerical)
# ---------------------------------------------------------------------------

def test_timestep_embedding_t0_numerical():
    """At t=0 every arg is 0 ⇒ cos half is all-1, sin half is all-0."""
    out = TimestepEmbedder.timestep_embedding(torch.tensor([0.0]), dim=4)
    expected = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_timestep_embedding_t1_numerical():
    """At t=1, freqs=[1.0, 0.01] (max_period=10000, half=2)."""
    out = TimestepEmbedder.timestep_embedding(torch.tensor([1.0]), dim=4)
    expected = torch.tensor([[
        math.cos(1.0), math.cos(0.01),
        math.sin(1.0), math.sin(0.01),
    ]])
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_timestep_embedding_odd_dim_pads_zero():
    """Odd dim ⇒ a single zero column is appended after [cos, sin]."""
    out = TimestepEmbedder.timestep_embedding(torch.tensor([0.0]), dim=5)
    assert out.shape == (1, 5)
    assert out[0, -1].item() == 0.0
    torch.testing.assert_close(out[:, :4], torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_timestep_embedder_shape_and_grad():
    emb = TimestepEmbedder(hidden_size=32, freq_dim=16)
    t = torch.rand(8)
    out = emb(t)
    assert out.shape == (8, 32)
    out.sum().backward()
    # Both MLP layers must receive gradient.
    assert emb.mlp[0].weight.grad is not None
    assert emb.mlp[2].weight.grad is not None
    assert torch.isfinite(emb.mlp[0].weight.grad).all()


# ---------------------------------------------------------------------------
# LabelEmbedder
# ---------------------------------------------------------------------------

def test_label_embedder_shape():
    emb = LabelEmbedder(num_classes=10, hidden_size=16, p_drop=0.1)
    y = torch.randint(0, 10, (4,))
    out = emb(y, train=False)
    assert out.shape == (4, 16)


def test_label_embedder_eval_is_deterministic():
    """train=False: dropout must not fire even with p_drop > 0."""
    emb = LabelEmbedder(num_classes=10, hidden_size=16, p_drop=0.5)
    y = torch.tensor([0, 1, 2, 3])
    a = emb(y, train=False)
    b = emb(y, train=False)
    torch.testing.assert_close(a, b)
    # And the result must equal a direct lookup in the table.
    torch.testing.assert_close(a, emb.embedding_table(y))


def test_label_embedder_force_drop_maps_to_null():
    """With force_drop_ids=ones, every label must map to the null-token row."""
    emb = LabelEmbedder(num_classes=10, hidden_size=16, p_drop=0.0)
    y = torch.tensor([0, 5, 9])
    force = torch.ones(3, dtype=torch.long)
    out = emb(y, train=False, force_drop_ids=force)
    null_row = emb.embedding_table.weight[emb.null_id]   # (16,)
    expected = null_row.unsqueeze(0).expand(3, -1)
    torch.testing.assert_close(out, expected)


def test_label_embedder_table_size():
    """Embedding table has num_classes + 1 rows (last = null token)."""
    emb = LabelEmbedder(num_classes=10, hidden_size=16, p_drop=0.0)
    assert emb.embedding_table.num_embeddings == 11
    assert emb.null_id == 10


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------

def test_patch_embed_shape():
    pe = PatchEmbed(in_channels=3, hidden_size=32, patch_size=4)
    x = torch.randn(2, 3, 16, 16)
    out = pe(x)
    # N = (16/4) * (16/4) = 16
    assert out.shape == (2, 16, 32)


def test_patch_embed_rectangular_input():
    pe = PatchEmbed(in_channels=3, hidden_size=32, patch_size=4)
    x = torch.randn(1, 3, 8, 16)
    out = pe(x)
    assert out.shape == (1, (8 // 4) * (16 // 4), 32)


def test_patch_embed_assertion_on_bad_size():
    pe = PatchEmbed(in_channels=3, hidden_size=32, patch_size=4)
    bad = torch.randn(1, 3, 10, 12)        # 10 not divisible by 4
    with pytest.raises(AssertionError):
        pe(bad)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def test_attention_shape_and_grad():
    attn = Attention(hidden_size=32, num_heads=4)
    x = torch.randn(2, 9, 32, requires_grad=True)
    out = attn(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert attn.qkv.weight.grad is not None
    assert attn.proj.weight.grad is not None
    assert torch.isfinite(out).all()


def test_attention_requires_divisible_heads():
    with pytest.raises(AssertionError):
        Attention(hidden_size=30, num_heads=4)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

def test_mlp_shape_and_grad():
    mlp = MLP(hidden_size=32, mlp_ratio=4.0)
    x = torch.randn(2, 9, 32, requires_grad=True)
    out = mlp(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert mlp.fc1.weight.grad is not None
    assert mlp.fc2.weight.grad is not None


# ---------------------------------------------------------------------------
# DiTBlock
# ---------------------------------------------------------------------------

def test_dit_block_shape():
    block = DiTBlock(hidden_size=32, num_heads=4)
    x = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    out = block(x, c)
    assert out.shape == x.shape


def test_dit_block_identity_when_modulation_zeroed():
    """adaLN-Zero contract: if the final Linear of adaLN_modulation is zero,
    every shift/scale/gate vector is zero ⇒ block output equals input.

    Manual zero-init here mirrors what DiT._init_weights does for every block.
    """
    block = DiTBlock(hidden_size=32, num_heads=4)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].weight)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].bias)
    x = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    out = block(x, c)
    torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------

def test_final_layer_shape():
    fl = FinalLayer(hidden_size=32, patch_size=2, out_channels=3)
    x = torch.randn(2, 16, 32)
    c = torch.randn(2, 32)
    out = fl(x, c)
    # P*P*C_out = 2*2*3 = 12
    assert out.shape == (2, 16, 12)


def test_final_layer_zero_at_init():
    """With zero-init linear weight+bias the output is exactly zero,
    regardless of the modulation values.
    """
    fl = FinalLayer(hidden_size=32, patch_size=2, out_channels=3)
    torch.nn.init.zeros_(fl.linear.weight)
    torch.nn.init.zeros_(fl.linear.bias)
    x = torch.randn(2, 16, 32)
    c = torch.randn(2, 32)
    out = fl(x, c)
    assert out.abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# AttnResJunction  (AR-DiT — see doc/AR_DiT.md §4, §12)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("l", [1, 2, 24])
def test_attnres_shape(l: int):
    """Forward on a length-``l`` cache returns ``[B, N, D]``."""
    B, N, D = 2, 9, 32
    junction = AttnResJunction(hidden_size=D)
    cache = [torch.randn(B, N, D) for _ in range(l)]
    out = junction(cache)
    assert out.shape == (B, N, D)
    assert torch.isfinite(out).all()


def test_attnres_zero_init_uniform_mix():
    """Zero-init pseudo-query ⇒ output is the equal-weight average of sources.

    This is the paper's uniform-init behaviour (doc/AR_DiT.md §10): with
    ``w = 0`` every logit is zero, so ``alpha = 1/l`` for all sources
    regardless of what ``RMSNorm`` does on the key path.
    """
    B, N, D, l = 2, 9, 32, 5
    junction = AttnResJunction(hidden_size=D)   # w is zero-init by default.
    cache = [torch.randn(B, N, D) for _ in range(l)]
    out = junction(cache)
    expected = torch.stack(cache, dim=0).mean(dim=0)   # (B, N, D)
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


def test_attnres_rmsnorm_inside_kernel_only():
    """Scaling one source by a constant ``k`` must:

    * leave the attention weights unchanged (RMSNorm inside the kernel
      cancels positive rescalings on the key path), and
    * scale that source's contribution to the output linearly (values
      enter the weighted sum un-normed).

    Concretely, if we scale source ``j`` by ``k > 0``:
    ``out_new - out_old == alpha_j * (k - 1) * v_j``.
    """
    torch.manual_seed(0)
    B, N, D, l = 2, 9, 32, 4
    j, k = 1, 3.0

    junction = AttnResJunction(hidden_size=D)
    # Break the zero-init to give the softmax non-uniform weights so the
    # test verifies "unchanged alpha" beyond the trivial 1/l case.
    with torch.no_grad():
        junction.w.copy_(torch.randn(D))

    cache_old = [torch.randn(B, N, D) for _ in range(l)]
    cache_new = [v.clone() for v in cache_old]
    cache_new[j] = cache_new[j] * k

    out_old = junction(cache_old)
    out_new = junction(cache_new)

    # --- Alpha unchanged: recompute alpha for both caches and compare. ---
    def _alpha(cache):
        sources = torch.stack(cache, dim=2)                    # [B, N, l, D]
        logits = torch.einsum("d,bnld->bnl", junction.w, junction.rms(sources))
        return torch.softmax(logits, dim=-1)                   # [B, N, l]

    alpha_old = _alpha(cache_old)
    alpha_new = _alpha(cache_new)
    torch.testing.assert_close(alpha_new, alpha_old, atol=1e-5, rtol=1e-5)

    # --- Output delta is exactly alpha_j * (k - 1) * v_j. ---
    delta_expected = alpha_old[..., j:j + 1] * (k - 1.0) * cache_old[j]
    torch.testing.assert_close(out_new - out_old, delta_expected, atol=1e-5, rtol=1e-5)


def test_attnres_softmax_normalisation():
    """Attention weights sum to 1 over the source axis for every (b, n)."""
    torch.manual_seed(0)
    B, N, D, l = 2, 9, 32, 6
    junction = AttnResJunction(hidden_size=D)
    with torch.no_grad():
        junction.w.copy_(torch.randn(D))          # non-trivial (non-uniform) weights

    cache = [torch.randn(B, N, D) for _ in range(l)]
    sources = torch.stack(cache, dim=2)
    logits = torch.einsum("d,bnld->bnl", junction.w, junction.rms(sources))
    alpha = torch.softmax(logits, dim=-1)

    ones = torch.ones(B, N)
    torch.testing.assert_close(alpha.sum(dim=-1), ones, atol=1e-6, rtol=1e-5)
    assert (alpha >= 0).all()


def test_attnres_grad_flow_at_zero_init():
    """At the zero-init state, gradients have a specific structure:

    * ``w.grad`` is non-zero (uniform softmax has a non-trivial derivative
      w.r.t. the logits, and the sources spread the values apart).
    * ``rms.weight.grad`` is *exactly zero* — with ``w = 0`` the logit is
      identically zero regardless of the RMSNorm scale ``g``, so ``g``
      receives no gradient signal at initialisation. This is a real
      property of the mechanism, worth recording explicitly.
    """
    torch.manual_seed(0)
    B, N, D, l = 2, 9, 32, 4
    junction = AttnResJunction(hidden_size=D)
    cache = [torch.randn(B, N, D) for _ in range(l)]
    out = junction(cache)
    out.sum().backward()

    assert junction.w.grad is not None
    assert junction.rms.weight.grad is not None
    assert torch.isfinite(junction.w.grad).all()
    assert torch.isfinite(junction.rms.weight.grad).all()
    assert junction.w.grad.abs().sum().item() > 0.0
    # Documented property: g gets no gradient at the exact zero-init step.
    assert junction.rms.weight.grad.abs().sum().item() == 0.0


def test_attnres_grad_flow_once_trained():
    """Once ``w`` is non-zero, gradients reach *both* learnable tensors.

    This is the operating condition after even a single optimiser step,
    so it is the practically-relevant grad-flow check.
    """
    torch.manual_seed(0)
    B, N, D, l = 2, 9, 32, 4
    junction = AttnResJunction(hidden_size=D)
    with torch.no_grad():
        junction.w.copy_(torch.randn(D))          # non-zero pseudo-query
    cache = [torch.randn(B, N, D) for _ in range(l)]
    out = junction(cache)
    out.sum().backward()

    assert junction.w.grad is not None
    assert junction.rms.weight.grad is not None
    assert torch.isfinite(junction.w.grad).all()
    assert torch.isfinite(junction.rms.weight.grad).all()
    assert junction.w.grad.abs().sum().item() > 0.0
    assert junction.rms.weight.grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# ARDiTBlock  (AR-DiT — see doc/AR_DiT.md §5)
# ---------------------------------------------------------------------------

def test_ar_dit_block_shape():
    """Forward returns ``[B, N, D]`` matching the input residual stream."""
    block = ARDiTBlock(hidden_size=32, num_heads=4)
    x = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    cache = [x.clone()]                             # v_0
    out = block(x, c, cache)
    assert out.shape == x.shape


def test_ar_dit_block_cache_grows_by_two():
    """Each call appends exactly two entries (``v_msa``, ``v_mlp``) to the cache.

    Prior cache contents must not be modified — the block is append-only.
    """
    block = ARDiTBlock(hidden_size=32, num_heads=4)
    x = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    v0 = torch.randn(2, 9, 32)
    cache = [v0]
    _ = block(x, c, cache)
    assert len(cache) == 3
    # v_0 preserved un-modified (identity check, not just close).
    assert cache[0] is v0
    # Appended entries have the sub-layer shape.
    assert cache[1].shape == (2, 9, 32)
    assert cache[2].shape == (2, 9, 32)


def test_ar_dit_block_identity_when_adaln_zeroed():
    """adaLN-Zero + AttnRes contract at init.

    If we zero the final Linear of ``adaLN_modulation`` (as ``ARDiT._init_weights``
    will do), every gate is zero ⇒ ``v_msa = v_mlp = 0`` are appended to
    the cache. Then each junction produces an equal-weight mean over its
    source pool — with ``[v_0, 0]`` the mean is ``v_0 / 2``; with
    ``[v_0, 0, 0]`` it is ``v_0 / 3``. So the block output at init is
    ``v_0 / 3``, exactly the "internal scaling of ``1/l``" behaviour
    documented in doc/AR_DiT.md §10.
    """
    block = ARDiTBlock(hidden_size=32, num_heads=4)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].weight)
    torch.nn.init.zeros_(block.adaLN_modulation[-1].bias)
    v0 = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    cache = [v0]
    out = block(v0, c, cache)                       # x on entry == v_0
    # MSA junction sees [v_0, 0]        -> mean = v_0 / 2
    # MLP junction sees [v_0, 0, 0]     -> mean = v_0 / 3
    torch.testing.assert_close(out, v0 / 3.0, atol=1e-6, rtol=1e-5)


def test_ar_dit_block_grad_flow():
    """Backward populates gradient on every learnable tensor of the block.

    Uses non-zero adaLN and non-zero pseudo-queries so both AttnRes
    junctions and both sub-layers are in a "trained" regime — this is
    the practically relevant grad-flow check.
    """
    torch.manual_seed(0)
    block = ARDiTBlock(hidden_size=32, num_heads=4)
    with torch.no_grad():
        block.attn_res_msa.w.copy_(torch.randn(32))
        block.attn_res_mlp.w.copy_(torch.randn(32))
    x = torch.randn(2, 9, 32)
    c = torch.randn(2, 32)
    cache = [x.clone()]
    out = block(x, c, cache)
    out.sum().backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_ar_dit_block_paramcount_vs_dit_block():
    """AR-DiT block adds exactly ``4 * D`` learnable scalars over baseline DiT
    (two junctions × two vectors of length D each: ``w`` and ``rms.weight``).
    """
    D, H = 32, 4
    ar_block = ARDiTBlock(hidden_size=D, num_heads=H)
    dit_block = DiTBlock(hidden_size=D, num_heads=H)
    diff = sum(p.numel() for p in ar_block.parameters()) - sum(
        p.numel() for p in dit_block.parameters()
    )
    assert diff == 4 * D

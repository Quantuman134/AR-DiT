"""Baseline Diffusion Transformer (DiT) with adaLN-Zero conditioning.

Faithful, minimal re-implementation of Peebles & Xie, "Scalable Diffusion
Models with Transformers" (ICCV 2023). The architecture is fully shape- and
config-agnostic: input resolution, channel count, patch size, and number of
classes are all constructor arguments. Default preset values (depth, width,
heads) follow the four DiT sizes from Table 2 of the paper.

**Training paradigm.** This project uses **flow matching with velocity
prediction**, not DDPM/IDDPM. Concretely, the model is a function
``v_theta(x_t, t, y)`` that predicts the velocity field of a linear
interpolant between data and noise. Output channel count therefore equals
the input channel count (no learned-variance head). Time `t` is a float in
[0, 1] at training/sampling time; it is passed verbatim to the sinusoidal
embedder, so the training loop is responsible for any rescaling.

This file is the control arm for the AR-DiT (attention-residual) experiment;
correctness and clarity are prioritised over speed.

See doc/DiT.md for the full design specification.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FiLM-style modulation used inside adaLN(-Zero).

    Args:
        x:     (B, N, D)  token features after LayerNorm.
        shift: (B, D)     per-image, per-channel shift produced from `c`.
        scale: (B, D)     per-image, per-channel scale produced from `c`.

    Returns:
        (B, N, D) tensor: ``x * (1 + scale) + shift`` with `shift`/`scale`
        broadcast across the token dimension.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """Build a fixed 2D sin-cos positional embedding.

    Same recipe as MAE / the DiT reference implementation. Produces an array
    of shape ``(grid_size * grid_size, embed_dim)``.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sin-cos."
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)              # (2, gs, gs), [w, h] order
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)

    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (N, D/2)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (N, D/2)
    return np.concatenate([emb_h, emb_w], axis=1)                         # (N, D)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega                                          # (D/2,)
    pos = pos.reshape(-1)                                                 # (N,)
    out = np.einsum("n,d->nd", pos, omega)                                # (N, D/2)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Embedding modules
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Scalar timestep -> sinusoidal features -> 2-layer MLP -> (B, D)."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # Standard small init, matches DiT reference.
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """Transformer-style sinusoidal embedding of a (B,) integer/float tensor.

        Returns a (B, dim) tensor with no learnable parameters.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)               # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)       # (B, 2*half)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.freq_dim).to(self.mlp[0].weight.dtype)
        return self.mlp(t_freq)                                           # (B, D)


class LabelEmbedder(nn.Module):
    """Class label -> learned embedding, with classifier-free-guidance dropout.

    The table has ``num_classes + 1`` rows; the last row is the "null" token
    used by CFG. During training, each label is independently replaced by the
    null id with probability ``p_drop``.
    """

    def __init__(self, num_classes: int, hidden_size: int, p_drop: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.null_id = num_classes
        self.p_drop = p_drop
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding_table.weight, std=0.02)

    def token_drop(self, y: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop = torch.rand(y.shape[0], device=y.device) < self.p_drop
        else:
            drop = force_drop_ids.to(torch.bool)
        return torch.where(drop, torch.full_like(y, self.null_id), y)

    def forward(
        self,
        y: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (train and self.p_drop > 0) or force_drop_ids is not None:
            y = self.token_drop(y, force_drop_ids=force_drop_ids)
        return self.embedding_table(y)                                    # (B, D)


class PatchEmbed(nn.Module):
    """Conv2d patch tokeniser: (B, C, H, W) -> (B, N, D), N = (H/P)*(W/P)."""

    def __init__(self, in_channels: int, hidden_size: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        # Xavier on the flattened weight matrix, matching DiT reference.
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % self.patch_size == 0 and x.shape[-2] % self.patch_size == 0, (
            f"Input H,W ({x.shape[-2]},{x.shape[-1]}) must be divisible by patch_size {self.patch_size}."
        )
        x = self.proj(x)                                                  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2).contiguous()                     # (B, N, D)
        return x


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Standard multi-head self-attention with a fused QKV projection."""

    def __init__(self, hidden_size: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads."
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                  # (3, B, H, N, d)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Use PyTorch's fused scaled-dot-product attention when available; it is
        # numerically equivalent to the manual implementation.
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)     # (B, H, N, d)
        x = x.transpose(1, 2).contiguous().reshape(B, N, D)
        return self.proj(x)


class MLP(nn.Module):
    """Standard 2-layer MLP with GELU(tanh) activation."""

    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, inner, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """One DiT transformer block with adaLN-Zero conditioning.

    For the global conditioning vector ``c`` we produce 6 modulation vectors
    (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) via a
    single SiLU + Linear(D, 6D). The final Linear is zero-initialised so the
    block reduces to the identity at step 0 (this is the "Zero" in
    adaLN-Zero).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """LayerNorm + adaLN modulation (2 outputs) + zero-init Linear -> P*P*C_out."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# ---------------------------------------------------------------------------
# DiT model
# ---------------------------------------------------------------------------

class DiT(nn.Module):
    """Diffusion Transformer with adaLN-Zero conditioning.

    The model is fully shape-agnostic: ``input_size``, ``in_channels``,
    ``patch_size`` and ``num_classes`` are all constructor arguments, so the
    same code targets CIFAR-10 (3x32x32, default), CelebA-64, ImageNet
    latents, etc. The only constraint is ``input_size % patch_size == 0``.
    """

    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        assert input_size % patch_size == 0, (
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})."
        )

        self.input_size = input_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        # Flow matching with velocity prediction: out_channels == in_channels.
        self.out_channels = in_channels

        self.x_embedder = PatchEmbed(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, p_drop=class_dropout_prob)

        # Fixed 2D sin-cos positional embedding.
        num_patches_per_side = input_size // patch_size
        self.num_patches = num_patches_per_side ** 2
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, self.num_patches, hidden_size),
            persistent=False,
        )
        pos = get_2d_sincos_pos_embed(hidden_size, num_patches_per_side)
        self.pos_embed.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        # Default Linear init: Xavier-uniform with zero bias.
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)

        # Re-zero the modulation layers (adaLN-Zero) of every block and the
        # final layer's linear. These overrides MUST run after the generic
        # `_basic` pass above.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ------------------------------------------------------------------
    # Patchify / unpatchify
    # ------------------------------------------------------------------
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, P*P*C_out) -> (B, C_out, H, W)."""
        c = self.out_channels
        p = self.patch_size
        h = w = int(math.sqrt(x.shape[1]))
        assert h * w == x.shape[1], "Token count is not a perfect square."
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the velocity field at state `x_t`.

        Args:
            x: (B, C, H, W) interpolant state at time `t`.
            t: (B,) float time values (typically in [0, 1] for flow matching;
               any rescaling is the training loop's responsibility).
            y: (B,) integer class labels in [0, num_classes]; `num_classes`
               itself is the null/uncond token.

        Returns:
            (B, C, H, W) predicted velocity v_theta(x_t, t, y).
        """
        x = self.x_embedder(x) + self.pos_embed                            # (B, N, D)
        c = self.t_embedder(t) + self.y_embedder(y, train=self.training)   # (B, D)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)                                         # (B, N, P*P*C)
        return self.unpatchify(x)                                          # (B, C, H, W)

    # Note: classifier-free guidance is *not* implemented here. The model is
    # a pure function ``v_theta(x, t, y)``; sampling code is responsible for
    # combining a conditional and an unconditional pass, e.g.
    #
    #     y_pair = torch.cat([y_real, torch.full_like(y_real, model.num_classes)])
    #     v_cond, v_uncond = model(x.repeat(2, 1, 1, 1), t.repeat(2), y_pair).chunk(2)
    #     v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
    #
    # Keeping CFG out of the model preserves a clean training/architecture
    # boundary and lets us swap CFG variants (interval CFG, per-token, etc.)
    # without touching this file.


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------
# Only (depth, hidden_size, num_heads) are fixed by these; dataset-specific
# fields (input_size, in_channels, patch_size, num_classes) must always be
# passed in by the caller.

def DiT_S_2(**kwargs) -> DiT:
    kwargs.setdefault("patch_size", 2)
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_B_2(**kwargs) -> DiT:
    kwargs.setdefault("patch_size", 2)
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L_2(**kwargs) -> DiT:
    kwargs.setdefault("patch_size", 2)
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def DiT_XL_2(**kwargs) -> DiT:
    kwargs.setdefault("patch_size", 2)
    return DiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)


# Self-tests now live in `tests/` (see doc/Test.md). Run with `pytest tests/`.

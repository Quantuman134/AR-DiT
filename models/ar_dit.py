"""Attention-Residual Diffusion Transformer (AR-DiT).

Paper-faithful port of the Attention Residual (AttnRes) mechanism of
Kimi Team, *Attention Residuals* (arXiv:2603.15031, 2026) to the DiT
backbone of Peebles & Xie (ICCV 2023).

See doc/AR_DiT.md for the full design specification. This file will
eventually host three classes:

- ``AttnResJunction``: one softmax-weighted mixture junction — the sole
  novel component (paper Eq. 2 / Eq. 4).
- ``ARDiTBlock``: DiT block with two junctions replacing the two ``+``
  operators of the baseline residual stream (§5 of the spec).
- ``ARDiT``: end-to-end model, drop-in-compatible with ``models.dit.DiT``
  (to be added).

The MHSA and MLP sub-modules, the adaLN-Zero modulation MLP, and the
patch/label/time embedders are all imported from ``models.dit`` — the
only structural change vs baseline DiT is inside the transformer stack.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from models.dit import (
    Attention,
    FinalLayer,
    LabelEmbedder,
    MLP,
    PatchEmbed,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    modulate,
)


# ---------------------------------------------------------------------------
# AttnRes junction
# ---------------------------------------------------------------------------

class AttnResJunction(nn.Module):
    """One softmax-weighted mixture junction (paper Eq. 2 / Eq. 4).

    Given a cache of ``l`` prior sub-layer outputs
    ``{v_0, v_1, ..., v_{l-1}}`` (each ``[B, N, D]``), produce

    .. math::

        \\alpha_{i \\to l} &= \\mathrm{softmax}_i\\bigl(
            w_l \\cdot \\mathrm{RMSNorm}(k_i)
        \\bigr)  \\\\
        h_l &= \\sum_{i=0}^{l-1} \\alpha_{i \\to l} \\, v_i

    with the paper's role binding ``k_i := v_i`` (Eq. 3): keys and values
    are the same tensor, but keys pass through the kernel's RMSNorm on
    the way to producing the logit, while values enter the weighted sum
    un-normed. Attention weights are computed **per-patch** — each
    ``(b, n)`` position gets its own length-``l`` softmax over source
    sub-layers (see doc/AR_DiT.md §6).

    Parameters
    ----------
    hidden_size : int
        Model dimension ``D``.

    Learnable parameters
    --------------------
    ``w`` : shape ``[D]``, initialised to zero (paper §5 — the sole stable
        initialisation, gives an equal-weight average at step 0).
    ``rms.weight`` : shape ``[D]``, RMSNorm scale, initialised to 1.

    Notes
    -----
    * The softmax is over the **source-junction axis** of length ``l``,
      not over the token axis. Each ``(b, n)`` slice is normalised
      independently.
    * No ``1/sqrt(D)`` scaling: the paper's kernel is unscaled and
      RMSNorm already bounds the key-side magnitude (see doc/AR_DiT.md
      §4).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Pseudo-query w_l  — zero-init per paper §5.
        self.w = nn.Parameter(torch.zeros(hidden_size))
        # Per-junction RMSNorm applied INSIDE the kernel to the key path.
        # nn.RMSNorm(dim) initialises the learnable scale to 1 by default.
        self.rms = nn.RMSNorm(hidden_size)

    def forward(self, cache: list[torch.Tensor]) -> torch.Tensor:
        """Compute one AttnRes junction output.

        Args:
            cache: list of length ``l``, each element a ``[B, N, D]``
                tensor. Must be non-empty; all tensors must share the
                same shape and dtype/device.

        Returns:
            ``[B, N, D]`` mixture tensor ``h_l``.
        """
        assert len(cache) > 0, "AttnResJunction: cache must be non-empty."

        # Stack the source pool along a new source axis.
        # sources: [B, N, l, D]
        sources = torch.stack(cache, dim=2)

        # Kernel: key path goes through RMSNorm; value path does not.
        # keys_normed: [B, N, l, D]
        keys_normed = self.rms(sources)

        # Logit_i = w . RMSNorm(k_i)   — per (b, n, i) dot product.
        # logits: [B, N, l]
        logits = torch.einsum("d,bnld->bnl", self.w, keys_normed)

        # Softmax over the source axis (length l).
        # alpha: [B, N, l]
        alpha = torch.softmax(logits, dim=-1)

        # Weighted sum over sources using un-normed values.
        # out: [B, N, D]
        out = torch.einsum("bnl,bnld->bnd", alpha, sources)
        return out


# ---------------------------------------------------------------------------
# AR-DiT block
# ---------------------------------------------------------------------------

class ARDiTBlock(nn.Module):
    """One AR-DiT transformer block: baseline DiT block with the two ``+``
    residuals replaced by two :class:`AttnResJunction` modules.

    Structure mirrors :class:`models.dit.DiTBlock` exactly:

    - Two LayerNorms (non-affine), one Attention, one MLP.
    - Six adaLN-Zero modulation vectors produced from the global
      conditioning vector ``c``: ``(shift_msa, scale_msa, gate_msa,
      shift_mlp, scale_mlp, gate_mlp)``.

    The only structural difference is that each sub-layer output — the
    quantity that would have been *added* to the residual stream in
    baseline DiT — is instead **appended** to a running source cache and
    fed, along with all prior sub-layer outputs, through an
    :class:`AttnResJunction`. See doc/AR_DiT.md §5.

    Cache ownership
    ---------------
    The cache is owned by the outer :class:`ARDiT` model, not by the
    block, and is passed into :meth:`forward` as a mutable list. The
    block appends exactly two entries per call (``v_msa`` then ``v_mlp``)
    and never inspects or removes prior entries. This keeps the block
    stateless w.r.t. depth — the same block type is used at every depth,
    and the source-pool size grows naturally as blocks execute in
    sequence.

    Learnable parameters
    --------------------
    Everything from :class:`DiTBlock` (norms are non-affine so contribute
    nothing) plus the two junctions' ``(w, rms.weight)`` pairs.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        # ---- Baseline DiT block components (structurally identical) ----
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # ---- AttnRes junctions (novel) ----
        self.attn_res_msa = AttnResJunction(hidden_size)
        self.attn_res_mlp = AttnResJunction(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cache: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run one AR-DiT block and grow ``cache`` by two entries in place.

        Args:
            x: ``[B, N, D]`` current residual-stream state ``h_{2b}`` at
                the input of block ``b``. Equal to the output of the
                previous block's MLP junction, or (for block 0) to
                ``v_0`` (patch-embed + positional embedding).
            c: ``[B, D]`` global conditioning vector.
            cache: mutable list of prior sub-layer outputs. Must contain
                at least ``v_0`` on entry to block 0. Appended twice by
                this call: first ``v_msa`` (=v_{2b+1}), then ``v_mlp``
                (=v_{2b+2}).

        Returns:
            ``[B, N, D]`` residual-stream state after this block —
            i.e. the output of the MLP junction, ``h_{2b+2}``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # --- MHSA sub-layer -------------------------------------------------
        # Same expression as baseline DiTBlock's `+` term: the gated,
        # modulated attention output. In baseline this would be *added*
        # to x; here we cache it as v_{2b+1} and mix via AttnRes.
        v_msa = gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        cache.append(v_msa)
        x = self.attn_res_msa(cache)                # h_{2b+1}

        # --- MLP sub-layer --------------------------------------------------
        v_mlp = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        cache.append(v_mlp)
        x = self.attn_res_mlp(cache)                # h_{2b+2}

        return x


# ---------------------------------------------------------------------------
# End-to-end AR-DiT model
# ---------------------------------------------------------------------------

class ARDiT(nn.Module):
    """Attention-Residual Diffusion Transformer.

    Drop-in replacement for :class:`models.dit.DiT`. The public API —
    ``__init__`` signature, ``forward(x, t, y) -> Tensor``, and output
    shape — is byte-identical, so any config, training loop, sampler, or
    evaluation harness that consumes ``DiT`` accepts ``ARDiT`` without
    modification.

    The only structural change is inside the transformer stack:

    - Each block is an :class:`ARDiTBlock` (two junctions replacing the
      two ``+`` residuals) instead of a :class:`~models.dit.DiTBlock`.
    - A **source cache** ``[v_0, v_1, ..., v_{2b}]`` is primed to
      ``[v_0]`` (patch-embed + positional embedding) before the first
      block and threaded through every subsequent block. Each block
      grows the cache by two entries; after ``L`` blocks the cache
      holds ``2L + 1`` tensors.
    - The final residual-stream state — the output of the last block's
      MLP junction, denoted ``h_{2L}`` in doc/AR_DiT.md — is fed to the
      :class:`~models.dit.FinalLayer` identically to baseline DiT.

    Everything before the block loop (patch/label/time embedders,
    positional embedding, class-dropout mechanics) and everything after
    the block loop (FinalLayer, unpatchify) is unchanged from baseline
    DiT — see doc/AR_DiT.md §2 and §5.
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
        self.depth = depth
        # Flow matching with velocity prediction: out_channels == in_channels.
        self.out_channels = in_channels

        # Embedders (byte-identical to baseline DiT).
        self.x_embedder = PatchEmbed(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, p_drop=class_dropout_prob)

        # Fixed 2D sin-cos positional embedding (as in baseline DiT).
        num_patches_per_side = input_size // patch_size
        self.num_patches = num_patches_per_side ** 2
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, self.num_patches, hidden_size),
            persistent=False,
        )
        pos = get_2d_sincos_pos_embed(hidden_size, num_patches_per_side)
        self.pos_embed.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        # Transformer stack — ARDiTBlock instead of DiTBlock.
        self.blocks = nn.ModuleList([
            ARDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Weight init: identical to baseline DiT.

        AttnRes-specific tensors — ``AttnResJunction.w`` (zero-init per
        paper §5) and ``AttnResJunction.rms.weight`` (RMSNorm scale, 1
        by default) — are already at their spec-mandated values from
        the sub-modules' own ``__init__``. The generic Xavier pass
        below does not touch them because ``nn.Parameter`` is not
        ``nn.Linear`` and ``nn.RMSNorm`` is not ``nn.Linear``, so no
        special guarding is needed.
        """
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
        """Predict the velocity field at state ``x_t``.

        Args:
            x: (B, C, H, W) interpolant state at time ``t``.
            t: (B,) float time values (typically in [0, 1] for flow
               matching; rescaling is the training loop's business).
            y: (B,) integer class labels in ``[0, num_classes]``;
               ``num_classes`` itself is the null / unconditional token.

        Returns:
            (B, C, H, W) predicted velocity ``v_theta(x_t, t, y)``.
        """
        v0 = self.x_embedder(x) + self.pos_embed                          # (B, N, D)
        c = self.t_embedder(t) + self.y_embedder(y, train=self.training)  # (B, D)

        # Source-pool cache owned by the model; each block appends two
        # entries. A fresh list per forward call keeps forward calls
        # independent (no state leak across mini-batches, sampling steps,
        # or eval/train transitions).
        cache: list[torch.Tensor] = [v0]
        h = v0
        for block in self.blocks:
            h = block(h, c, cache)                                        # grows cache by 2
        # After the loop: len(cache) == 2*depth + 1 and h == h_{2L}.

        h = self.final_layer(h, c)                                        # (B, N, P*P*C)
        return self.unpatchify(h)                                         # (B, C, H, W)

    # Note: classifier-free guidance is *not* implemented here — see the
    # matching note on models.dit.DiT for the reasoning. The sampling
    # code combines conditional and unconditional passes externally.


# ---------------------------------------------------------------------------
# Preset factories (parallel to models.dit.DiT_S_2 / _B_2 / _L_2 / _XL_2)
# ---------------------------------------------------------------------------
# Only (depth, hidden_size, num_heads) are fixed by these; dataset-specific
# fields (input_size, in_channels, patch_size, num_classes) must always be
# passed in by the caller. Names mirror the DiT presets 1-to-1 so a config
# can swap ``DiT_S_2`` for ``ARDiT_S_2`` and change nothing else.

def ARDiT_S_2(**kwargs) -> ARDiT:
    kwargs.setdefault("patch_size", 2)
    return ARDiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def ARDiT_B_2(**kwargs) -> ARDiT:
    kwargs.setdefault("patch_size", 2)
    return ARDiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def ARDiT_L_2(**kwargs) -> ARDiT:
    kwargs.setdefault("patch_size", 2)
    return ARDiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def ARDiT_XL_2(**kwargs) -> ARDiT:
    kwargs.setdefault("patch_size", 2)
    return ARDiT(depth=28, hidden_size=1152, num_heads=16, **kwargs)

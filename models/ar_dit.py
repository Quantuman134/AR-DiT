"""Attention-Residual Diffusion Transformer (AR-DiT).

Paper-faithful port of the Attention Residual (AttnRes) mechanism of
Kimi Team, *Attention Residuals* (arXiv:2603.15031, 2026) to the DiT
backbone of Peebles & Xie (ICCV 2023).

See doc/AR_DiT.md for the full design specification. This file hosts:

**v1 (paper-strict, §§1-8, 10-14 of the spec):**

- ``AttnResJunction``: one softmax-weighted mixture junction — the sole
  novel component (paper Eq. 2 / Eq. 4).
- ``ARDiTBlock``: DiT block with two junctions replacing the two ``+``
  operators of the baseline residual stream (§5 of the spec).
- ``ARDiT``: end-to-end model, drop-in-compatible with ``models.dit.DiT``.

**E1 — Time-conditioned pseudo-query (§9a of the spec):**

- ``TimeQueryTrunk``: shared 2-layer MLP producing the phase
  representation ``tau`` from the timestep embedding.
- ``ARDiTCondBlock``: E1 block owning two per-junction linear heads
  ``W_msa``, ``W_mlp`` that project ``tau`` into per-junction queries.
- ``ARDiTCond``: E1 end-to-end model, drop-in-compatible with ``ARDiT``.

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

    def forward(
        self,
        cache: list[torch.Tensor],
        q_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute one AttnRes junction output.

        Args:
            cache: list of length ``l``, each element a ``[B, N, D]``
                tensor. Must be non-empty; all tensors must share the
                same shape and dtype/device.
            q_override: optional per-image query of shape ``[B, D]``.
                When ``None`` (the paper-strict / v1 code path), the
                junction uses its own learned constant ``self.w`` as
                the global pseudo-query — one query for the whole
                network at this junction. When a tensor is provided
                (the E1 code path — see doc/AR_DiT.md §9a), it is used
                as the per-image query for this junction and
                ``self.w`` is bypassed entirely; the caller is
                responsible for having folded any additive bias into
                ``q_override`` upstream.

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

        # Logit_i = q . RMSNorm(k_i)   — per (b, n, i) dot product.
        # Two query modes:
        #   * v1 (q_override is None): q = self.w  of shape [D].
        #     Same query for every image in the batch — a global
        #     constant. einsum contracts the model dim ``d`` only.
        #   * E1 (q_override provided): q = q_override of shape [B, D].
        #     Per-image query — different images get different logit
        #     distributions. einsum contracts the model dim ``d`` while
        #     broadcasting the batch dim ``b``. See doc/AR_DiT.md §9a.6.
        # logits: [B, N, l] in both branches.
        if q_override is None:
            logits = torch.einsum("d,bnld->bnl", self.w, keys_normed)
        else:
            assert q_override.ndim == 2 and q_override.shape[-1] == self.hidden_size, (
                f"AttnResJunction: q_override must have shape [B, D={self.hidden_size}]; "
                f"got {tuple(q_override.shape)}."
            )
            logits = torch.einsum("bd,bnld->bnl", q_override, keys_normed)

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


# ===========================================================================
# E1 — Time-conditioned pseudo-query (doc/AR_DiT.md §9a)
# ===========================================================================
# The v1 stack above uses a constant per-junction pseudo-query ``w_l``. E1
# lifts entry E1 from the §9 ablation menu into a real model: the query
# becomes a function of the timestep, via a shared 2-layer trunk (over
# ``t_emb``) followed by a per-junction linear head. Everything else in the
# AttnRes operator is byte-identical to v1 — see §9a of the design doc.


class TimeQueryTrunk(nn.Module):
    """Shared 2-layer MLP producing the phase representation ``tau`` for E1.

    Reads the timestep embedding ``t_emb`` (output of
    :class:`~models.dit.TimestepEmbedder`, shape ``[B, D]``) and produces
    a same-shape phase representation ``tau`` consumed by **every**
    :class:`ARDiTCondBlock` in the network. The trunk is shared across
    all 2L junctions — one universal representation of "which phase of
    denoising are we in?" — while per-junction linear heads owned by the
    blocks then specialise ``tau`` into each junction's query
    (see doc/AR_DiT.md §9a.6).

    Shape
    -----
    ``Linear(D, D) -> SiLU -> Linear(D, D)`` — same shape as
    :class:`~models.dit.TimestepEmbedder`'s own MLP. Trunk depth /
    hidden expansion are recorded open questions (§9a.9); this is the
    default.

    Init
    ----
    Both linears take the model's generic Xavier-uniform init via
    :meth:`ARDiTCond._init_weights`. Xavier is chosen for consistency
    with the rest of the model; because ``W_l = 0`` at step 0 in every
    per-junction head, the trunk's output is annihilated at init anyway,
    so any finite init is safe here.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """Compute the phase representation.

        Args:
            t_emb: ``[B, D]`` timestep embedding from
                :class:`~models.dit.TimestepEmbedder`.

        Returns:
            ``tau`` of shape ``[B, D]`` — the phase representation
            consumed once per forward by every :class:`ARDiTCondBlock`.
        """
        return self.fc2(self.act(self.fc1(t_emb)))


class ARDiTCondBlock(nn.Module):
    """One E1 transformer block: :class:`ARDiTBlock` plus two per-junction
    linear heads that construct time-conditioned pseudo-queries.

    Structurally identical to :class:`ARDiTBlock` — same
    non-affine LayerNorms, same :class:`~models.dit.Attention`, same
    :class:`~models.dit.MLP`, same six-way adaLN-Zero modulation, same
    two :class:`AttnResJunction` instances. The **only** additions are:

    - ``self.W_msa: nn.Linear(D, D, bias=False)`` — per-junction linear
      head for the MHSA junction (junction index ``2b + 1``). Zero-init.
    - ``self.W_mlp: nn.Linear(D, D, bias=False)`` — per-junction linear
      head for the MLP junction (junction index ``2b + 2``). Zero-init.

    On each call the block receives the shared phase representation
    ``tau`` from the outer :class:`ARDiTCond` and constructs each
    junction's query as ``q = W_l(tau) + self.attn_res_*.w`` — i.e.
    v1's learned constant ``w_l`` is retained as an additive bias. See
    doc/AR_DiT.md §9a.1 and §9a.6.

    Cache semantics are unchanged from :class:`ARDiTBlock`: two entries
    appended per call (``v_msa`` then ``v_mlp``), never inspected or
    removed. The block stays depth-stateless.
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
        # ---- AttnRes junctions (identical to ARDiTBlock) ----
        self.attn_res_msa = AttnResJunction(hidden_size)
        self.attn_res_mlp = AttnResJunction(hidden_size)
        # ---- E1 additions: per-junction linear heads (§9a.6) ----
        # No bias — the additive constant role is played by
        # ``attn_res_*.w`` (see §9a.3). Zero-init happens in
        # ``ARDiTCond._init_weights``.
        self.W_msa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        tau: torch.Tensor,
        cache: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run one E1 block and grow ``cache`` by two entries in place.

        Args:
            x: ``[B, N, D]`` current residual-stream state ``h_{2b}``.
            c: ``[B, D]`` global conditioning vector (``t_emb + y_emb``).
                Drives adaLN-Zero, exactly as in :class:`ARDiTBlock`.
            tau: ``[B, D]`` shared phase representation from
                :class:`TimeQueryTrunk`. Same tensor across all blocks
                in a single forward.
            cache: mutable list of prior sub-layer outputs; contract
                identical to :meth:`ARDiTBlock.forward`.

        Returns:
            ``[B, N, D]`` residual-stream state after this block —
            i.e. the output of the MLP junction, ``h_{2b+2}``.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # --- MHSA sub-layer -------------------------------------------------
        # Same v_msa expression as v1; only the AttnRes call changes.
        v_msa = gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        cache.append(v_msa)
        # E1 query for the MHSA junction: W_msa(tau) + additive bias w.
        # ``self.attn_res_msa.w`` is the v1 zero-init pseudo-query,
        # kept here as the additive bias per §9a.1 / §9a.3.
        q_msa = self.W_msa(tau) + self.attn_res_msa.w   # [B, D]
        x = self.attn_res_msa(cache, q_override=q_msa)  # h_{2b+1}

        # --- MLP sub-layer --------------------------------------------------
        v_mlp = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        cache.append(v_mlp)
        q_mlp = self.W_mlp(tau) + self.attn_res_mlp.w   # [B, D]
        x = self.attn_res_mlp(cache, q_override=q_mlp)  # h_{2b+2}

        return x


class ARDiTCond(nn.Module):
    """E1 — Time-conditioned Attention-Residual Diffusion Transformer.

    Drop-in replacement for :class:`ARDiT`. The public API —
    ``__init__`` signature, ``forward(x, t, y) -> Tensor``, and output
    shape — is byte-identical, so any config, training loop, sampler, or
    evaluation harness that consumes :class:`ARDiT` accepts
    :class:`ARDiTCond` without modification.

    Differences from :class:`ARDiT` (see doc/AR_DiT.md §9a):

    - A shared :class:`TimeQueryTrunk` reads ``t_emb`` (the raw
      :class:`~models.dit.TimestepEmbedder` output, **not** the
      combined adaLN vector ``c``) and produces a phase representation
      ``tau`` of shape ``[B, D]``, computed **once** per forward.
    - Each block is an :class:`ARDiTCondBlock`, which owns two
      per-junction linear heads and constructs each junction's query as
      ``q_l = W_l(tau) + w_l``. ``tau`` is threaded into every block.
    - Initialisation: ``W_msa.weight = 0`` and ``W_mlp.weight = 0`` in
      every block (added to :meth:`_init_weights`). Combined with v1's
      ``w = 0``, this guarantees ``q_l ≡ 0`` at step 0 for every ``t``,
      preserving the ``ARDiTCond(x, t, y) == 0`` init invariant
      (§9a.5).

    Everything else — patch/label/time embedders, positional embedding,
    class-dropout mechanics, FinalLayer, unpatchify, adaLN-Zero — is
    byte-identical to :class:`ARDiT`. The class label ``y`` continues
    to reach the transformer stack **only** through adaLN-Zero via
    ``c = t_emb + y_emb``; it is deliberately kept out of the E1 query
    path (§9a.2).
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

        # Embedders (byte-identical to baseline DiT / ARDiT).
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

        # E1: shared time-query trunk (§9a.6). Reads ``t_emb`` and
        # produces ``tau``. Owned by the model; passed into every block.
        self.t_query_trunk = TimeQueryTrunk(hidden_size)

        # Transformer stack — ARDiTCondBlock instead of ARDiTBlock.
        self.blocks = nn.ModuleList([
            ARDiTCondBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Weight init: identical to :class:`ARDiT` plus E1-specific overrides.

        The generic Xavier pass handles :class:`TimeQueryTrunk`'s two
        linears — that init is arbitrary, since ``W_l = 0`` in every
        block will annihilate ``tau`` at step 0 anyway (§9a.5).

        E1-specific override: ``W_msa.weight`` and ``W_mlp.weight`` are
        set to zero in every block. Combined with the v1 ``w = 0``
        already in place from :meth:`AttnResJunction.__init__`, this
        guarantees ``q_l ≡ 0`` for every timestep ``t`` at step 0, which
        in turn preserves v1's ``ARDiT(x, t, y) == 0`` init invariant
        (see §9a.5 for the derivation).
        """
        # Default Linear init: Xavier-uniform with zero bias.
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)

        # Re-zero the modulation layers (adaLN-Zero) of every block and the
        # final layer's linear, plus the E1 per-junction heads. These
        # overrides MUST run after the generic ``_basic`` pass above.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
            # E1: per-junction heads — see §9a.5.
            nn.init.zeros_(block.W_msa.weight)
            nn.init.zeros_(block.W_mlp.weight)
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
        """Predict the velocity field at state ``x_t`` (E1 variant).

        Args:
            x: (B, C, H, W) interpolant state at time ``t``.
            t: (B,) float time values.
            y: (B,) integer class labels in ``[0, num_classes]``.

        Returns:
            (B, C, H, W) predicted velocity ``v_theta(x_t, t, y)``.
        """
        v0 = self.x_embedder(x) + self.pos_embed                          # (B, N, D)

        # Compute both branches out of the timestep embedding:
        #   * ``c``   — combined with the class embedding for adaLN-Zero;
        #               drives the six per-block modulations, unchanged.
        #   * ``tau`` — pure-time phase representation for E1 queries;
        #               fed to every ARDiTCondBlock. See §9a.2.
        t_emb = self.t_embedder(t)                                        # (B, D)
        c = t_emb + self.y_embedder(y, train=self.training)               # (B, D)
        tau = self.t_query_trunk(t_emb)                                   # (B, D)

        cache: list[torch.Tensor] = [v0]
        h = v0
        for block in self.blocks:
            h = block(h, c, tau, cache)                                   # grows cache by 2
        # After the loop: len(cache) == 2*depth + 1 and h == h_{2L}.

        h = self.final_layer(h, c)                                        # (B, N, P*P*C)
        return self.unpatchify(h)                                         # (B, C, H, W)

    # Note: classifier-free guidance is *not* implemented here — same
    # split as ARDiT / DiT. The sampling code combines conditional and
    # unconditional passes externally.


# ---------------------------------------------------------------------------
# ARDiTCond preset factories (parallel to ARDiT_S_2 / _B_2 / _L_2 / _XL_2)
# ---------------------------------------------------------------------------
# Only (depth, hidden_size, num_heads) are fixed by these; dataset-specific
# fields are always caller-supplied. Names mirror the ARDiT presets 1-to-1
# so a config can swap ``ARDiT_S_2`` for ``ARDiTCond_S_2`` and change
# nothing else.

def ARDiTCond_S_2(**kwargs) -> ARDiTCond:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCond(depth=12, hidden_size=384, num_heads=6, **kwargs)


def ARDiTCond_B_2(**kwargs) -> ARDiTCond:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCond(depth=12, hidden_size=768, num_heads=12, **kwargs)


def ARDiTCond_L_2(**kwargs) -> ARDiTCond:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCond(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def ARDiTCond_XL_2(**kwargs) -> ARDiTCond:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCond(depth=28, hidden_size=1152, num_heads=16, **kwargs)

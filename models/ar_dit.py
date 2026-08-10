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

**E2 — SANA-time-cond pseudo-query (§9b of the spec):**

- ``SANATimeCondQuery``: shared time-conditioned query module that
  owns two depth-shared learnable D-vectors ``w_attn``, ``w_mlp`` plus
  two learned time codebooks ``phi_attn``, ``phi_mlp`` of shape
  ``[num_time_bins, D]``. Continuous ``t \\in [0, 1]`` is floor-
  quantised into ``num_time_bins`` slots, and the per-image junction
  query is ``q_m(t) = w_m + phi_m[quantise(t)]``.
- ``ARDiTCondSANABlock``: E2 block that receives already-computed
  ``(q_attn, q_mlp)`` from the outer model and passes them straight
  into its two :class:`AttnResJunction` instances via the new
  ``q_override_raw`` argument (no query-side RMSNorm, no
  ``1/sqrt(D)`` scaling — bounded magnitude comes from the
  parameters themselves, not from in-kernel guards).
- ``ARDiTCondSANA``: E2 end-to-end model, drop-in-compatible with
  ``ARDiT`` / ``ARDiTCond``.

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
    ``rms.weight`` : shape ``[D]``, key-path RMSNorm scale, initialised
        to 1.
    ``q_rms.weight`` : shape ``[D]``, query-path RMSNorm scale,
        initialised to 1. **Used only when ``q_override`` is provided**
        (the E1 / ARDiTCond path); the paper-strict v1 branch (``q =
        self.w``) bypasses it entirely, since ``self.w`` is bounded by
        its own optimiser updates and RMSNorm on a zero-init constant
        would be numerically ill-conditioned.

    Notes
    -----
    * The softmax is over the **source-junction axis** of length ``l``,
      not over the token axis. Each ``(b, n)`` slice is normalised
      independently.
    * **``1/sqrt(D)`` scaling — E1 branch only.** The ``q_override``
      branch multiplies the dot-product logit by ``1 / sqrt(D)``,
      matching standard scaled-dot-product attention. Even with
      RMSNorm on both q and k, each RMSNormed vector has L2 norm
      ``sqrt(D)`` (RMSNorm rescales the per-element RMS to 1, not the
      full norm), so an un-scaled dot product has magnitude
      ``O(D * rms_q_weight * rms_k_weight)`` — dimensionally large
      and unbounded in the learnable RMSNorm weights. The
      ``1/sqrt(D)`` factor cancels the dimensional-sum growth,
      leaving the softmax argument at ``O(rms_q_weight *
      rms_k_weight)``. The v1 branch (``q = self.w``) intentionally
      omits this scaling — ``self.w`` is zero-init and
      weight-decayed, so its magnitude is small by construction and
      the paper's original unscaled formulation is preserved bit-
      for-bit.
    * **Query-side RMSNorm asymmetry** — the v1 branch uses the raw
      learned ``self.w`` un-normed; the E1 branch normalises
      ``q_override``. This asymmetry is deliberate: v1's ``w`` is a
      global constant whose magnitude is fully controlled by the
      optimiser, so it never runs away. E1's ``q_override =
      W_l(tau) + w`` is a per-image quantity produced by an upstream
      linear head whose weight matrix ``W_l`` can grow unboundedly
      during training, driving ``|q · RMSNorm(k)|`` into softmax
      saturation (one-hot attention → zero backprop through the
      kernel → the E1 query-path parameter group dies at exactly zero
      gradient). ``q_rms`` structurally prevents the vector-norm
      component of that failure by bounding ``||RMSNorm(q)|| =
      sqrt(D) * rms_q_weight``; the ``1/sqrt(D)`` scaling
      additionally kills the dimensional-sum factor. See
      doc/AR_DiT.md §9a, the `fix/e1-q-rmsnorm-softmax-saturation`
      diagnosis for the RMSNorm addition, and
      `fix/e1-attn-scale-sqrt-d` for the dimensional-scaling
      addition.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # Pseudo-query w_l  — zero-init per paper §5.
        self.w = nn.Parameter(torch.zeros(hidden_size))
        # Per-junction RMSNorm applied INSIDE the kernel to the key path.
        # nn.RMSNorm(dim) initialises the learnable scale to 1 by default.
        self.rms = nn.RMSNorm(hidden_size)
        # Per-junction RMSNorm applied INSIDE the kernel to the QUERY
        # path — symmetric with ``self.rms``.  Consumed only in the
        # ``q_override`` branch of ``forward`` (E1 / ARDiTCond); the v1
        # branch bypasses it, so on paper-strict AR-DiT this parameter
        # exists but sees zero gradient and stays at its init.
        self.q_rms = nn.RMSNorm(hidden_size)

    def forward(
        self,
        cache: list[torch.Tensor],
        q_override: torch.Tensor | None = None,
        q_override_raw: torch.Tensor | None = None,
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
                ``q_override`` upstream. The E1 branch additionally
                applies :attr:`q_rms` and a ``1/sqrt(D)`` scaling
                factor for softmax-saturation safety.
            q_override_raw: optional per-image query of shape ``[B, D]``
                for the **E2 / ARDiTCondSANA** code path (see
                doc/AR_DiT.md §9b). Semantics differ from
                ``q_override`` in two ways: the query-side RMSNorm is
                **not** applied, and the softmax logit is **not**
                scaled by ``1/sqrt(D)``. The kernel is
                ``phi(q, k) = exp(q . RMSNorm(k))`` — identical to
                v1's kernel structurally, only with a per-image
                time-conditioned ``q`` in place of the global
                constant ``self.w``. Rationale (§9b.4): E2's ``q`` is
                a directly-optimised parameter (not the output of an
                upstream matrix product), so its magnitude is
                bounded by weight decay / gradient dynamics alone —
                no in-kernel guards are needed and the paper-faithful
                un-scaled kernel is preserved.
                At most one of ``q_override`` and ``q_override_raw``
                may be non-``None``.

        Returns:
            ``[B, N, D]`` mixture tensor ``h_l``.
        """
        assert len(cache) > 0, "AttnResJunction: cache must be non-empty."
        assert not (q_override is not None and q_override_raw is not None), (
            "AttnResJunction: q_override and q_override_raw are mutually exclusive."
        )

        # Stack the source pool along a new source axis.
        # sources: [B, N, l, D]
        sources = torch.stack(cache, dim=2)

        # Kernel: key path goes through RMSNorm; value path does not.
        # keys_normed: [B, N, l, D]
        keys_normed = self.rms(sources)

        # Logit_i = q . RMSNorm(k_i)   — per (b, n, i) dot product.
        # Three query modes:
        #   * v1 (both overrides None): q = self.w  of shape [D].
        #     Same query for every image in the batch — a global
        #     constant. einsum contracts the model dim ``d`` only.
        #     ``self.w`` is used un-normed and un-scaled: it is
        #     bounded by the optimiser directly (zero-init +
        #     weight-decay), and RMSNorm on a zero-init vector would
        #     be ill-conditioned (0 / sqrt(eps) at step 0). Preserves
        #     paper-strict AR-DiT bit-for-bit.
        #   * E1 (q_override provided): q = RMSNorm(q_override) of
        #     shape [B, D].  Per-image query — different images get
        #     different logit distributions.  Query-side RMSNorm is
        #     applied here, symmetric with the key-side ``self.rms``,
        #     and the logit is additionally rescaled by
        #     ``1 / sqrt(D)`` (standard scaled-dot-product attention).
        #     Rationale — two-part defense against softmax saturation:
        #       (a) RMSNorm bounds each vector's L2 norm to
        #           ``sqrt(D) * rms_weight`` (RMSNorm rescales per-
        #           element RMS to 1, so ||x|| = sqrt(D) when weight
        #           is 1). Without (a), upstream ``W_l`` growth would
        #           blow up ``||q||`` unboundedly.
        #       (b) ``1/sqrt(D)`` cancels the dimensional-sum growth
        #           of the D-term dot product: q . k with
        #           ||q|| = ||k|| = sqrt(D) has raw magnitude ~D, so
        #           scaling by 1/sqrt(D) leaves the softmax argument
        #           at O(1) w.r.t. hidden size.
        #     Together, the kernel is
        #       phi(q, k) = exp( RMSNorm(q) . RMSNorm(k) / sqrt(D) ).
        #     einsum contracts the model dim ``d`` while broadcasting
        #     the batch dim ``b``. See doc/AR_DiT.md §9a.6.
        #   * E2 (q_override_raw provided): q = q_override_raw of
        #     shape [B, D]. Structurally identical to v1's kernel —
        #     no query-side RMSNorm, no 1/sqrt(D) scaling — only the
        #     source of ``q`` changes (per-image time-conditioned
        #     query instead of a global constant). Safe here because
        #     E2's ``q`` is a directly-optimised parameter tensor
        #     rather than the output of an upstream Linear, so its
        #     magnitude is bounded by weight decay / gradient
        #     dynamics alone and does not need in-kernel guards.
        #     einsum contracts ``d`` and broadcasts ``b``. See
        #     doc/AR_DiT.md §9b.4.
        # logits: [B, N, l] in all branches.
        if q_override is not None:
            assert q_override.ndim == 2 and q_override.shape[-1] == self.hidden_size, (
                f"AttnResJunction: q_override must have shape [B, D={self.hidden_size}]; "
                f"got {tuple(q_override.shape)}."
            )
            q_normed = self.q_rms(q_override)
            # Scaled-dot-product attention: divide by sqrt(D) to keep
            # the softmax argument at O(1) w.r.t. hidden size.
            scale = 1.0 / math.sqrt(self.hidden_size)
            logits = torch.einsum("bd,bnld->bnl", q_normed, keys_normed) * scale
        elif q_override_raw is not None:
            assert q_override_raw.ndim == 2 and q_override_raw.shape[-1] == self.hidden_size, (
                f"AttnResJunction: q_override_raw must have shape [B, D={self.hidden_size}]; "
                f"got {tuple(q_override_raw.shape)}."
            )
            logits = torch.einsum("bd,bnld->bnl", q_override_raw, keys_normed)
        else:
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
    nothing) plus the two junctions' ``(w, rms.weight, q_rms.weight)``
    triples. ``q_rms`` sees no gradient on this paper-strict AR-DiT
    path (it is used only when ``q_override`` is passed, which
    :class:`ARDiTBlock` never does); it exists on the module for a
    uniform state-dict shape shared with :class:`ARDiTCondBlock`.
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


# ===========================================================================
# E2 — SANA-time-cond pseudo-query (doc/AR_DiT.md §9b)
# ===========================================================================
# E1 replaced v1's global constant ``w_l`` with a per-junction, per-image
# query ``q_l(t) = W_l(tau) + w_l``.  Two levers changed together:
# depth-per-junction specialisation (each of the 2L junctions owns its
# own ``W_l``) *and* time dependence (via the shared trunk ``tau``).
#
# E2 keeps the second lever (time dependence) but reverses the first
# (depth-per-junction specialisation).  Instead:
#
#   * ``w_m`` is depth-shared across all junctions of type ``m ∈
#     {attn, mlp}`` — only two learnable D-vectors total for the
#     whole network, replacing v1's 2L per-junction constants.
#   * Time dependence comes from a **learned codebook**
#     ``phi_m[bin_idx, :]`` of shape ``[num_time_bins, D]`` rather
#     than from an MLP over the timestep embedding.  Continuous ``t
#     ∈ [0, 1]`` is floor-quantised to a bin index, matching the
#     discrete-schedule reality of inference-time samplers (e.g. 50
#     Euler steps → 50 codebook entries).
#
# The junction query becomes
#
#     q_m(t) = w_m + phi_m[ floor(t * num_time_bins) ]         (§9b.1)
#
# and every junction of type ``m`` in the network uses this same
# ``q_m(t)`` — one query per (kind, time-bin, image) tuple, shared
# across depth.  This is a different research hypothesis than E1: E1
# tests "should the query be time-conditioned and depth-specialised
# and produced by an MLP over t_emb?", E2 tests "is depth sharing +
# a discrete time codebook enough?".
#
# Both branches keep v1's zero-init discipline: ``w_m = phi_m = 0`` at
# init, so ``q_m(t) ≡ 0`` for every ``t`` at step 0, and the model
# output remains bit-exactly zero (§9b.5).
#
# Kernel-level rationale.  E2 uses the v1 un-scaled kernel ``phi(q,
# k) = exp(q . RMSNorm(k))`` (via the new ``q_override_raw``
# argument on ``AttnResJunction.forward``), *not* the E1 kernel
# ``phi(q, k) = exp(RMSNorm(q) . RMSNorm(k) / sqrt(D))``.  E2's
# ``q`` is a directly-optimised parameter tensor rather than the
# output of an upstream Linear head, so its magnitude is bounded by
# weight decay + gradient dynamics alone and does not need in-kernel
# guards.  This is a deliberate ablation contrast with E1 — see
# §9b.4.


class SANATimeCondQuery(nn.Module):
    """Shared time-conditioned query module for E2 (§9b).

    Owns four learnable tensors, all zero-initialised:

    * ``w_attn`` of shape ``[D]`` — depth-shared additive bias for
      every MHSA junction in the network.
    * ``w_mlp``  of shape ``[D]`` — depth-shared additive bias for
      every MLP  junction in the network.
    * ``phi_attn`` of shape ``[num_time_bins, D]`` — time codebook
      for MHSA junctions.
    * ``phi_mlp``  of shape ``[num_time_bins, D]`` — time codebook
      for MLP junctions.

    Given continuous ``t \\in [0, 1]``, the module produces the per-
    image query for junction kind ``m`` as

    .. math::

        q_m(t) = w_m + \\phi_m[\\lfloor t \\cdot B_{\\text{time}} \\rfloor]

    where ``B_time = num_time_bins``.  The floor-quantisation is
    clamped so that ``t = 1.0`` (or any tiny numerical overshoot)
    maps to the last bin ``num_time_bins - 1`` instead of an
    out-of-range index.

    Shapes
    ------
    ``forward(t: [B], kind: str) -> [B, D]``.  The same module
    instance is called twice per outer-model forward (once with
    ``kind='attn'``, once with ``kind='mlp'``) and its outputs are
    broadcast to every block that needs them.

    Init
    ----
    All four tensors are zero-init.  This preserves v1's ``q ≡ 0`` at
    step 0 property — see :meth:`ARDiTCondSANA._init_weights` and
    doc/AR_DiT.md §9b.5.
    """

    def __init__(self, hidden_size: int, num_time_bins: int):
        super().__init__()
        if num_time_bins < 1:
            raise ValueError(
                f"num_time_bins must be >= 1; got {num_time_bins}."
            )
        self.hidden_size = hidden_size
        self.num_time_bins = num_time_bins

        # Depth-shared additive biases (one D-vector per junction kind).
        # Zero-init preserves the ARDiT(x, t, y) == 0 invariant at step 0.
        self.w_attn = nn.Parameter(torch.zeros(hidden_size))
        self.w_mlp  = nn.Parameter(torch.zeros(hidden_size))

        # Time codebooks — one row per discrete inference-time bin.
        # Zero-init preserves the same invariant.
        self.phi_attn = nn.Parameter(torch.zeros(num_time_bins, hidden_size))
        self.phi_mlp  = nn.Parameter(torch.zeros(num_time_bins, hidden_size))

    def _quantise(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous ``t \\in [0, 1]`` to integer bin indices in
        ``[0, num_time_bins - 1]``.

        Uses **floor** quantisation: bin ``i`` covers
        ``[i / num_time_bins, (i + 1) / num_time_bins)``.  ``t = 0.0``
        maps to bin 0; ``t = 1.0`` (or any numerical overshoot beyond
        1) is clamped to the last bin ``num_time_bins - 1`` rather
        than triggering an out-of-range index.  Negative overshoot is
        similarly clamped to 0.

        Args:
            t: ``[B]`` float tensor of times.

        Returns:
            ``[B]`` long tensor of bin indices.
        """
        assert t.ndim == 1, f"SANATimeCondQuery: expected 1-D t; got shape {tuple(t.shape)}."
        idx = torch.floor(t * self.num_time_bins).long()
        idx = torch.clamp(idx, min=0, max=self.num_time_bins - 1)
        return idx

    def forward(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        """Return the E2 per-image junction query.

        Args:
            t: ``[B]`` float tensor of continuous times in ``[0, 1]``.
            kind: one of ``'attn'`` or ``'mlp'``.

        Returns:
            ``[B, D]`` query tensor ``q_m(t) = w_m + phi_m[quantise(t)]``.
        """
        if kind == "attn":
            w, phi = self.w_attn, self.phi_attn
        elif kind == "mlp":
            w, phi = self.w_mlp, self.phi_mlp
        else:
            raise ValueError(
                f"SANATimeCondQuery.forward: kind must be 'attn' or 'mlp'; got {kind!r}."
            )
        idx = self._quantise(t)                     # [B]
        # phi[idx] uses integer advanced indexing → shape [B, D].
        # w is [D]; broadcast-add gives [B, D].
        return w + phi[idx]


class ARDiTCondSANABlock(nn.Module):
    """One E2 transformer block: :class:`ARDiTBlock` structure with the
    two junctions driven by externally-supplied per-image queries.

    Structurally identical to :class:`ARDiTBlock` — same non-affine
    LayerNorms, same :class:`~models.dit.Attention`, same
    :class:`~models.dit.MLP`, same six-way adaLN-Zero modulation, same
    two :class:`AttnResJunction` instances.  Unlike :class:`ARDiTCondBlock`,
    this block owns **no** per-junction linear heads: the queries
    ``q_attn``, ``q_mlp`` are computed once in
    :meth:`ARDiTCondSANA.forward` (via a shared
    :class:`SANATimeCondQuery`) and threaded through every block, in
    keeping with E2's depth-sharing design (§9b.1).

    The junction's own ``self.w`` parameter is **unused** on this
    code path (the ``q_override_raw`` branch of
    :meth:`AttnResJunction.forward` bypasses it), and consequently
    sees no gradient — it exists on the module only for state-dict
    shape compatibility with v1 / E1.
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
        # ---- AttnRes junctions (identical to ARDiTBlock / ARDiTCondBlock) ----
        self.attn_res_msa = AttnResJunction(hidden_size)
        self.attn_res_mlp = AttnResJunction(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        q_attn: torch.Tensor,
        q_mlp: torch.Tensor,
        cache: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run one E2 block and grow ``cache`` by two entries in place.

        Args:
            x: ``[B, N, D]`` current residual-stream state ``h_{2b}``.
            c: ``[B, D]`` global conditioning vector (``t_emb + y_emb``);
                drives adaLN-Zero, exactly as in :class:`ARDiTBlock`.
            q_attn: ``[B, D]`` per-image query for the MHSA junction.
                Depth-shared across all blocks — the same tensor is
                passed to every block in a given forward.
            q_mlp:  ``[B, D]`` per-image query for the MLP junction.
                Depth-shared across all blocks.
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
        v_msa = gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        cache.append(v_msa)
        # E2 uses the un-scaled ``q_override_raw`` kernel — see §9b.4
        # and the docstring of :meth:`AttnResJunction.forward`.
        x = self.attn_res_msa(cache, q_override_raw=q_attn)   # h_{2b+1}

        # --- MLP sub-layer --------------------------------------------------
        v_mlp = gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        cache.append(v_mlp)
        x = self.attn_res_mlp(cache, q_override_raw=q_mlp)    # h_{2b+2}

        return x


class ARDiTCondSANA(nn.Module):
    """E2 — SANA-time-conditioned Attention-Residual Diffusion Transformer.

    Drop-in replacement for :class:`ARDiT` / :class:`ARDiTCond`.  The
    public API — ``__init__`` signature (plus a new ``num_time_bins``
    kwarg), ``forward(x, t, y) -> Tensor``, and output shape — is
    byte-identical apart from the extra kwarg, so any config-driven
    caller wired to accept a ``num_time_bins`` field works unchanged.

    Differences from :class:`ARDiT` (see doc/AR_DiT.md §9b):

    - A single :class:`SANATimeCondQuery` module (``self.time_cond_query``)
      owns the four E2 parameter tensors: two depth-shared additive
      biases ``w_attn`` / ``w_mlp`` and two time codebooks
      ``phi_attn`` / ``phi_mlp`` of shape ``[num_time_bins, D]``.
    - Each block is an :class:`ARDiTCondSANABlock`; the block owns no
      new parameters relative to v1 (no per-junction linear heads —
      that is E1's design).
    - On every forward, the model computes ``q_attn = w_attn +
      phi_attn[quantise(t)]`` and ``q_mlp = w_mlp + phi_mlp[quantise(t)]``
      **once** and threads both into every block.
    - Initialisation: all four E2 tensors are zero-init.  Combined
      with :meth:`AttnResJunction.__init__` (``rms.weight = q_rms.weight
      = 1``), this guarantees ``q_m(t) ≡ 0`` for every ``t`` at step 0
      and therefore preserves the ``ARDiT(x, t, y) == 0`` init
      invariant (§9b.5).

    Everything else — patch/label/time embedders, positional embedding,
    class-dropout mechanics, FinalLayer, unpatchify, adaLN-Zero — is
    byte-identical to :class:`ARDiT`.  The class label ``y`` continues
    to reach the transformer stack **only** through adaLN-Zero via
    ``c = t_emb + y_emb``; it is deliberately kept out of the E2 query
    path (§9b.2 mirrors §9a.2).
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
        num_time_bins: int = 50,
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
        self.num_time_bins = num_time_bins
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

        # E2: shared time-conditioned query module (§9b.1).  Owned by
        # the model; passed into every block indirectly via the two
        # ``[B, D]`` tensors it produces once per forward.
        self.time_cond_query = SANATimeCondQuery(hidden_size, num_time_bins)

        # Transformer stack — ARDiTCondSANABlock instead of ARDiTBlock.
        self.blocks = nn.ModuleList([
            ARDiTCondSANABlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Weight init: identical to :class:`ARDiT`.

        The E2-specific parameters ``w_attn``, ``w_mlp``, ``phi_attn``,
        ``phi_mlp`` are already zero-initialised by
        :class:`SANATimeCondQuery.__init__` (and, because they are
        ``nn.Parameter`` not ``nn.Linear``, are not touched by the
        generic Xavier pass below).  Combined with the v1
        ``AttnResJunction.w = 0``, this guarantees the un-used ``self.w``
        stays at zero and ``q_m(t) ≡ 0`` at step 0 for every ``t``,
        which in turn preserves ``ARDiTCondSANA(x, t, y) == 0`` bit-
        exactly at init (see doc/AR_DiT.md §9b.5).
        """
        # Default Linear init: Xavier-uniform with zero bias.
        def _basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(_basic)

        # Re-zero the modulation layers (adaLN-Zero) of every block and the
        # final layer's linear.  These overrides MUST run after the generic
        # ``_basic`` pass above.
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
        """Predict the velocity field at state ``x_t`` (E2 variant).

        Args:
            x: (B, C, H, W) interpolant state at time ``t``.
            t: (B,) float time values in ``[0, 1]``.
            y: (B,) integer class labels in ``[0, num_classes]``.

        Returns:
            (B, C, H, W) predicted velocity ``v_theta(x_t, t, y)``.
        """
        v0 = self.x_embedder(x) + self.pos_embed                          # (B, N, D)

        # Standard adaLN-Zero conditioning (same as v1 / E1).
        c = self.t_embedder(t) + self.y_embedder(y, train=self.training)  # (B, D)

        # E2 queries — computed once per forward, threaded into every block.
        # See doc/AR_DiT.md §9b.1.
        q_attn = self.time_cond_query(t, kind="attn")                      # (B, D)
        q_mlp  = self.time_cond_query(t, kind="mlp")                       # (B, D)

        cache: list[torch.Tensor] = [v0]
        h = v0
        for block in self.blocks:
            h = block(h, c, q_attn, q_mlp, cache)                          # grows cache by 2
        # After the loop: len(cache) == 2*depth + 1 and h == h_{2L}.

        h = self.final_layer(h, c)                                        # (B, N, P*P*C)
        return self.unpatchify(h)                                         # (B, C, H, W)

    # Note: classifier-free guidance is *not* implemented here — same
    # split as ARDiT / ARDiTCond / DiT.  The sampling code combines
    # conditional and unconditional passes externally.


# ---------------------------------------------------------------------------
# ARDiTCondSANA preset factories (parallel to ARDiTCond_S_2 / _B_2 / _L_2 / _XL_2)
# ---------------------------------------------------------------------------
# Only (depth, hidden_size, num_heads) are fixed by these; dataset-specific
# fields (input_size, in_channels, patch_size, num_classes,
# class_dropout_prob) and E2-specific fields (num_time_bins) are always
# caller-supplied via kwargs.  Names mirror the ARDiT / ARDiTCond presets
# 1-to-1 so a config can swap ``ARDiTCond_S_2`` for ``ARDiTCondSANA_S_2``
# and change nothing else beyond adding the ``num_time_bins`` field.

def ARDiTCondSANA_S_2(**kwargs) -> ARDiTCondSANA:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCondSANA(depth=12, hidden_size=384, num_heads=6, **kwargs)


def ARDiTCondSANA_B_2(**kwargs) -> ARDiTCondSANA:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCondSANA(depth=12, hidden_size=768, num_heads=12, **kwargs)


def ARDiTCondSANA_L_2(**kwargs) -> ARDiTCondSANA:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCondSANA(depth=24, hidden_size=1024, num_heads=16, **kwargs)


def ARDiTCondSANA_XL_2(**kwargs) -> ARDiTCondSANA:
    kwargs.setdefault("patch_size", 2)
    return ARDiTCondSANA(depth=28, hidden_size=1152, num_heads=16, **kwargs)

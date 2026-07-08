# AR-DiT — Design Document

This document specifies the **Attention-Residual Diffusion Transformer
(AR-DiT)** we will implement in `models/ar_dit.py`. It adapts the
Attention Residual (AttnRes) mechanism of Kimi Team,
*Attention Residuals* (arXiv:2603.15031, 2026) — originally proposed for
decoder-only LLMs — to the DiT-with-adaLN-Zero backbone of Peebles & Xie
(ICCV 2023).

The v1 target is a **paper-faithful port**: the only intentional
departure from the paper is the change of substrate (image patches
instead of language tokens; velocity regression instead of next-token
prediction). Nothing about the AttnRes mechanism itself is modified in
v1. Time-conditioned and activation-conditioned variants are recorded in
§9 as explicit follow-ups for the ablation study, **not** implemented
here.

This document is the sign-off point before any code is written for
`models/ar_dit.py`. Every design choice below has a paragraph explaining
what we chose and, where applicable, what we rejected and why — so the
ablation plan can trace each variant back to the decision it revisits.

---

## 1. Background — AttnRes in one page

In a standard PreNorm transformer, each sub-layer's output is added to
the residual stream:

```
h_l = h_{l-1} + f_l(h_{l-1})            (standard residual)
```

The paper argues (§1) that this identity residual causes **magnitude
dilution** as depth grows: contributions from early layers are
progressively drowned out by later ones. Their fix is to replace the
identity residual with a **learnable, softmax-weighted mixture** of
**all** previous sub-layer contributions:

```
q_l     = w_l                                       (query)
k_i     = v_i                                       (key — same tensor as value)
α_{i→l} = softmax_i ( ϕ(q_l, k_i) )                 (Eq. 2 of paper)
ϕ(q, k) = exp( q · RMSNorm(k) )                     (kernel: RMSNorm inside ϕ)
h_l     = Σ_{i=0..l-1} α_{i→l} · v_i                (Eq. 4 of paper)
```

A few things about this formula worth spelling out:

1. **Key and value are the same tensor.** Per Eq. 3 of the paper,
   `k_i = v_i` — both roles are filled by the raw sub-layer output.
   There is no separate `k` tensor and no projection.
2. **Attention weights are a query·key dot-product** (never query·value
   in name), even though `k_i` and `v_i` are the same object. The
   distinction lives in *how each is consumed*: keys pass through the
   kernel `ϕ` (which internally applies RMSNorm) to produce logits;
   values enter the weighted sum unchanged.
3. **RMSNorm lives inside the kernel `ϕ`, not on the key definition.**
   It is applied at compute time when we dot the query against the
   key — not baked into the definition of `k_i`. This distinction
   matters when we later add ablations like E3 (learnable projections),
   which would insert a `W_k, W_v` split *around* `v_i` but leave the
   kernel structure alone.

with

- `v_0 = h_0` (the embedding output — patch-embed output in our case).
- `v_i = f_i(h_{i-1})` for `i ≥ 1` (the **pre-residual** sub-layer output —
  i.e. what would have been added to the stream in standard residual).
- `w_l ∈ ℝᴰ` — a **layer-specific learnable vector** of the full model
  dimension. This is the *only* learnable parameter of the query side.
- `RMSNorm` — applied by the kernel `ϕ` to the key argument only.
  Values are used unnormalised in the weighted sum. Table 4 of the
  paper shows removing this RMSNorm degrades val loss (1.737 → 1.743),
  so it is not optional.
- **Zero-init** of `w_l`: §5 of the paper (verbatim) — *"all pseudo-query
  vectors must be initialized to zero"*. This makes AttnRes degenerate
  to an equal-weight average at step 0, which the paper found
  empirically to be the only stable initialisation.

Notational note: the paper writes attention weights as if each query
attends to prior *tokens*, but here `q = w_l` is a single learned
vector, not derived from a token. So `w_l` is called a **pseudo-query**.
The K/V come directly from `v_i`, without any projection matrix.

---

## 2. Inputs / outputs / conditioning

**Unchanged from baseline DiT** — see [DiT.md](DiT.md) §1. AR-DiT is a
drop-in replacement for `DiT` as a class-conditional velocity network
`v_θ(x_t, t, y)` for pixel-space flow matching. Same call signature,
same output shape, same adaLN-Zero conditioning via a single global
vector `c ∈ ℝᴰ`.

The only structural difference is what happens **inside the transformer
stack**: the two `+` operators in each `DiTBlock.forward` are replaced
by AttnRes junctions. Everything before the first block (PatchEmbed +
positional embedding + `t`/`y` embedders) and everything after the last
block (FinalLayer, unpatchify) is byte-identical to baseline DiT.

---

## 3. Sub-layer indexing

We treat each DiT block as **two sub-layers**: one MHSA sub-layer and
one MLP sub-layer. With `L` transformer blocks, this gives **2L
sub-layers** total, so **2L AttnRes junctions**. The source pool grows
monotonically with depth.

Let `L` be the number of DiT blocks. We define **junction index**
`l ∈ {1, 2, ..., 2L}` (1-based to match the paper). We also define
**source index** `i ∈ {0, 1, ..., 2L}` — the value cached from
sub-layer `i`, with `i = 0` reserved for the patch-embed output.

Concretely, for block `b ∈ {0, ..., L-1}`:

| Junction | Semantics                          | Source pool consumed |
|----------|------------------------------------|----------------------|
| `l = 2b + 1` | after MHSA of block `b`        | `{v_0, v_1, ..., v_{2b}}` (size `2b+1`) |
| `l = 2b + 2` | after MLP of block `b`         | `{v_0, v_1, ..., v_{2b+1}}` (size `2b+2`) |

with:

- `v_0` = patch-embed output + positional embedding (shape `[B, N, D]`)
- `v_{2b+1}` = MHSA-sub-layer output of block `b`, i.e.
  `gate_msa · attn(modulate(norm1(h_{2b}), shift_msa, scale_msa))`
- `v_{2b+2}` = MLP-sub-layer output of block `b`, i.e.
  `gate_mlp · mlp(modulate(norm2(h_{2b+1}), shift_mlp, scale_mlp))`

Junction `l`'s output `h_l` replaces what would have been
`h_{l-1} + v_l` in the standard block.

**Design decision (locked)**: `2L` junctions per model, one per
sub-layer, matching the paper's LLM formulation exactly.

---

## 4. AttnRes operator — v1 spec

For junction `l` with source pool `{v_0, v_1, ..., v_{l-1}}` where each
`v_i ∈ ℝ^{B×N×D}`, we bind `k_i := v_i` (Eq. 3 of the paper) and
compute:

```
    # Kernel logit — argument of exp() inside ϕ(q, k) = exp(q · RMSNorm(k))
    logit_i  = sum_d ( w_l[d] * RMSNorm_l(k_i)[b, n, d] )   # per-patch q·key
    α_{i→l}  = softmax_i ( logit_i )                        # over source axis (length l)
    h_l      = Σ_i α_{i→l} · v_i                            # values un-normed
```

This is the closed-form re-expression of paper Eq. 2. The paper writes
it as a fraction `ϕ(q_l, k_i) / Σ_j ϕ(q_l, k_j)`; because `ϕ` contains
`exp(...)`, that fraction is precisely `softmax_i( q_l · RMSNorm(k_i) )`.
The softmax is a **consequence** of the `exp` inside `ϕ` plus the
normalisation in Eq. 2, not an additional operator.

**Why keep the `k_i` name if `k_i = v_i`?** Following the paper's own
notation (Eq. 2 vs Eq. 3), we preserve the K/V *role* names even though
they bind to the same underlying tensor. Keys are the arguments of the
kernel `ϕ` (they pass through RMSNorm to form logits); values are the
vectors combined in the weighted sum (they are consumed un-normed).
This role separation matters for future ablations (E3: learnable
`W_k, W_v` projections would break the `k_i = v_i` binding while
leaving Eq. 2 unchanged).

**Scaling and stability**:

- **No `1/√D`**: the paper's `ϕ` is unscaled. RMSNorm on the key path
  already bounds `‖RMSNorm(k)‖ ≈ √D`, and `w_l` is zero-init and grows
  slowly, so logit magnitudes stay in a benign softmax regime. We match
  the paper — no temperature/scaling factor.
- **Numerical stability**: implement via `torch.softmax(logits, dim=source_axis)`,
  which internally does the standard max-subtraction trick. Softmax is
  over the **source-junction axis** (length `l`), *not* over the token
  axis — each `(b, n)` gets its own length-`l` softmax.

Learnable parameters of junction `l`:

- **Pseudo-query** `w_l ∈ ℝᴰ` — full model dimension. Initialised to zero.
- **RMSNorm scale** `g_l ∈ ℝᴰ` — one RMSNorm module per junction (see §7
  for the "per-junction vs shared" decision). Initialised to `1` (identity).

Non-learnable: no projection matrices. Following the paper's Eq. 3,
we bind `k_i := v_i` — the raw sub-layer outputs serve as both keys
(consumed by the kernel `ϕ` via RMSNorm) and values (consumed un-normed
in the weighted sum).

**Shapes**:

- `w_l`: `[D]`
- `RMSNorm_l.g`: `[D]`
- `logit`: `[B, N, l]`
- `α`: `[B, N, l]` — attention weights are **per-patch** (see §6).
- `h_l`: `[B, N, D]`

Total added parameters for DiT-S/2 (`D=384`, `L=12`, so `2L=24`):

- Pseudo-queries: `24 × 384 = 9,216`.
- Per-junction RMSNorm scales: `24 × 384 = 9,216`.
- Total: `18,432` — about **0.056 %** of DiT-S/2's ~33 M parameters.

---

## 5. Where the residual replacement happens (code sketch)

Baseline `DiTBlock.forward` (from [models/dit.py](../models/dit.py)):

```python
# baseline DiT
def forward(self, x, c):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ...
    x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), ...))
    x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), ...))
    return x
```

AR-DiT block (sketch — final code will live in `models/ar_dit.py`):

```python
# AR-DiT block — replaces the two `+` operators with AttnRes junctions
def forward(self, x, c, cache, attn_res_msa, attn_res_mlp):
    """
    cache  : list of {v_0, ..., v_{l-1}} maintained by the outer model
             (each element is a [B, N, D] tensor)
    attn_res_msa, attn_res_mlp : AttnResJunction modules for this block
    """
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ...

    # MHSA sub-layer — same as baseline, but result is v_{2b+1}, not added
    v_msa = gate_msa.unsqueeze(1) * self.attn(
        modulate(self.norm1(x), shift_msa, scale_msa)
    )
    cache.append(v_msa)
    x = attn_res_msa(cache)                            # AttnRes junction

    # MLP sub-layer — analogous
    v_mlp = gate_mlp.unsqueeze(1) * self.mlp(
        modulate(self.norm2(x), shift_mlp, scale_mlp)
    )
    cache.append(v_mlp)
    x = attn_res_mlp(cache)                            # AttnRes junction

    return x
```

The outer `ARDiT` model owns the `cache` list. Before the first block it
does `cache = [v_0]` where `v_0` is the patch-embed + pos-emb output.
After the last block, the last cached activation is exactly `h_{2L}`,
which is fed to the FinalLayer identically to baseline DiT.

**Memory implication**: at the deepest junction (`l = 2L`) the cache
holds `2L` tensors of `[B, N, D]`. For DiT-S/2 on CIFAR-10 (`B=128`,
`N=256`, `D=384`, `L=12`, fp32): `24 × 128 × 256 × 384 × 4 B ≈ 1.2 GB`.
This is not free but it's small relative to the MHSA activations. If we
ever need to compress this, gradient checkpointing is the natural
answer, but v1 does not need it.

---

## 6. Per-patch attention weights (Q2 decision)

**Locked decision**: attention weights `α_{i→l}` are computed
**per-patch** — each patch position `n` in each image `b` has its own
softmax-normalised depth-mixing vector.

- Chosen: `α ∈ ℝ^{B × N × l}`, computed as
  `softmax_i ( sum_d w_l[d] · RMSNorm(k_i)[b, n, d] )` with `k_i := v_i`
  per Eq. 3 (RMSNorm applied inside the kernel `ϕ`).
- Rejected alternative (**Option B, "per-image"**): pool each `v_i`
  across the patch dimension first, so `α ∈ ℝ^{B × l}` is shared across
  patches of the same image.

**Why per-patch**:

1. **Paper fidelity.** Equation 3 of the paper writes the formula for a
   single token; the accompanying PyTorch pseudocode in Fig. 2 operates
   on the full `[B, T, D]` tensor and gives per-token weights
   implicitly. Choosing Option B would be a conscious departure from
   the paper without prior evidence.
2. **Compute is trivial either way.** For DiT-S/2/CIFAR-10 the
   per-junction cost is `Σ_l B·N·l·D ≈ 944 M` scalar ops per forward
   (≈60× more than Option B) — still negligible next to DiT-S/2's own
   MHSA/MLP cost (tens of GFLOPs). Compute does not decide this.
3. **Expressiveness.** Per-patch weights let edge/centre/background
   patches choose different depth mixes if that turns out to be
   useful — the paper's motivating story ("early layers = syntactic,
   late layers = semantic" for LLMs) has a plausible image-domain
   analogue ("early = local edges, late = global semantics").
4. **Ablation-friendly.** Per-patch is the strict superset — we can
   always add a `pool_keys: bool` config knob later that recovers
   Option B for ablation.

**Ablation note (recorded, not v1)**: comparing per-patch vs per-image
attention weights is a meaningful ablation for DiT specifically —
because the LLM paper never tested this axis, we don't know a priori
whether the extra expressiveness helps images.

---

## 7. RMSNorm placement (Q3 decision)

**Locked decision**: **one RMSNorm module per junction** — 2L RMSNorm
modules total, each with its own learnable scale `g_l ∈ ℝᴰ`.

- Chosen: `AttnResJunction_l` owns its own `nn.RMSNorm(D)`.
- Rejected alternative: single globally-shared `nn.RMSNorm(D)` across
  all `2L` junctions.

**Why per-junction**:

1. **Paper fidelity.** The paper's Fig. 2 PyTorch pseudocode gives each
   junction its own `self.attn_res_norm` / `self.mlp_res_norm`.
2. **Parameter cost is negligible.** Per-junction RMSNorm adds ~9 K
   parameters to DiT-S/2 — noise floor.
3. **Different junctions see different value statistics.** `v_i` for
   `i` near 0 is the patch-embed output (roughly Gaussian, near-zero
   mean); `v_i` for large `i` is a heavily-modulated MLP output whose
   scale is driven by `gate_mlp(c)`. A shared RMSNorm would have to
   compromise between these distributions.

**Ablation note (recorded, not v1)**: shared vs per-junction RMSNorm is
a meaningful ablation. If the shared variant matches the per-junction
one, that's a small parameter-count win.

---

## 8. Value source (Q4 decision)

**Locked decision**: values are the **pre-residual sub-layer outputs**,
`v_i = f_i(h_{i-1})` for `i ≥ 1`, and `v_0 = h_0` (patch-embed +
positional embedding) — i.e. paper-strict.

- Chosen (paper-strict): cache `v_i = f_i(h_{i-1})` — the thing that
  *would have been added to the residual stream* in baseline DiT.
- Rejected alternative: cache `h_i` — the post-residual accumulated
  activation.

**Why pre-residual**:

1. **This is what the paper does (Eq. 3).** Rewriting AttnRes on top of
   the accumulated `h_i` would silently change the mechanism: sources
   would no longer be independent, and the whole "each layer's
   contribution is one point in a distribution AttnRes chooses over"
   framing collapses.
2. **Under adaLN-Zero, sub-layer outputs are individually meaningful.**
   Each `v_i` has its own gate; caching post-residual `h_i` would
   entangle already-emitted contributions.

**Practical detail**: in the AR-DiT block sketch (§5), the MHSA
sub-layer output is `gate_msa · attn(modulate(...))` — this whole
expression is `v_{2b+1}`, exactly what would have been added to the
residual in baseline DiT. Same for the MLP sub-layer.

**Ablation note (recorded, not v1)**: swapping `v_i ← h_i` (post-res)
is a valid ablation — it corresponds to "attention over accumulated
depth" vs "attention over sub-layer contributions" and answers whether
the paper's specific value definition is important, or whether the
mechanism is robust to this choice.

---

## 9. Follow-ups deliberately deferred (recorded for ablation)

The following extensions are motivated by the DiT setting and are
**not** in the v1 implementation. They are recorded here so the
ablation plan has a clear menu.

| Ext.  | Description                                        | Motivation |
|-------|----------------------------------------------------|------------|
| E1    | **Time-conditioned pseudo-query** `w_l(t)`         | Diffusion adds a strong time signal `t`; letting the depth mix depend on `t` may help the network shift its "which layers matter" prior between denoising phases. |
| E2    | **Activation-conditioned pseudo-query** `w_l(h)`   | True content-adaptive depth mixing — closer to real attention. Adds a small projection `h → q`. |
| E3    | **Learnable K/V projections**                      | Restore full attention semantics by making `k_i = W_k v_i`, `v_i = W_v v_i` learnable. Costs `2 · D · D` per junction — non-trivial for DiT-S/2 (≈4.4 M params for 24 junctions). |
| E4    | **Multi-head AttnRes**                             | Split `D → n_heads · D_h`, do the softmax per head. Same total parameter budget, more expressiveness. |
| E5    | **Multiple queries per junction (`n_q > 1`)**      | Ensemble of pseudo-queries at each junction, averaged or gated. Cheap to try. |
| E6    | **Shared RMSNorm across junctions**                | The Q3 alternative — potentially a small parameter win if it matches per-junction quality. |
| E7    | **Per-image (pooled) attention weights**           | The Q2 alternative — potentially a 60× compute win at the AttnRes op if the expressiveness of per-patch weights turns out to be unused. |
| E8    | **Post-residual values** (`v_i ← h_i`)             | The Q4 alternative — tests whether the paper's specific value choice matters for image generation. |
| E9    | **Block AttnRes** (only cross-block, identity within block) | The paper's pipeline-parallel-friendly variant. Rejected for our single-GPU setting (§?), but recorded as a compute-cost ablation baseline. |

None of these are on the v1 critical path. Each is a bounded change on
top of the v1 codebase.

---

## 10. Initialisation

**Locked decision**: `w_l = 0` for all `l`, `g_l = 1` for all RMSNorms.
Everything else identical to baseline DiT (adaLN-Zero on gates,
Xavier-uniform on MHSA/MLP weights, etc.).

The zero-init of `w_l` follows §5 of the paper verbatim: *"all
pseudo-query vectors must be initialized to zero. This ensures that
the initial attention weights α_{i→l} are uniform across source layers,
which reduces AttnRes to an **equal-weight average** at the start of
training and prevents training volatility, as we validated empirically."*
The RMSNorm scale `g_l = 1` is the standard default (LLaMA, Mistral,
etc.); the paper does not specify it, so we adopt the community default.

**What "zero-init pseudo-query" means at step 0**:

With `w_l = 0`, every logit `w_l · RMSNorm(k_i) = 0`, so `α_{i→l} = 1/l`
(uniform over the pool). At step 0, adaLN-Zero also makes every gate
zero, so `v_i ≈ 0` for `i ≥ 1` and the cache is `[v_0, 0, 0, ..., 0]`
after `2L` sub-layers. Under uniform attention, `h_l = mean(v_0, 0, ...,
0) = v_0 / l` — i.e. the patch-embed signal is passed through, scaled
down by `1/l`.

**Comparison to baseline DiT at step 0 (both intermediate activations
and model output)**:

- **Internal activations differ by a factor of `l`.** Baseline DiT at
  step 0 has `x = v_0` at every depth (identity residual preserves
  `v_0` since every `gate = 0`, giving an unnormalised sum with total
  weight `l`). AR-DiT at step 0 has `x = v_0 / l` (equal-weight
  average, total weight `1`). The paper's "equal-weight average"
  wording refers exactly to this normalised mix.
- **Model output is identical: exactly zero for both models.** Baseline
  DiT and AR-DiT both zero-init `FinalLayer.linear`, so regardless of
  what enters the final layer (`v_0` vs `v_0/l`), the model output is
  `0` at step 0. The `1/l` internal scaling has no observable
  consequence at step 0.
- **Where the `1/l` difference does surface** is in the *gradients*
  flowing back through the residual stream — specifically into
  `PatchEmbed` and adaLN modulation MLPs, which are `l`× smaller than
  the baseline's after one backward pass. Adam's per-parameter scaling
  absorbs this in practice.

**Test-plan consequence** (see §12): the strong acceptance criterion
from [DiT.md](DiT.md) §9.5 — "model output is exactly zero at init" —
applies to AR-DiT too. The reason is different (equal-weight average
× zero `FinalLayer` vs identity residual × zero `FinalLayer`), but the
observable at the model boundary is bit-identical. We test that directly.

---

## 11. File layout and module API (planned)

```
models/
├── dit.py             # baseline (unchanged)
└── ar_dit.py          # NEW
    ├── class AttnResJunction(nn.Module)     # one softmax-mix junction
    ├── class ARDiTBlock(nn.Module)          # DiT block with 2 junctions
    └── class ARDiT(nn.Module)               # end-to-end model
```

`ARDiT` subclasses no PyTorch module directly (composition, not
inheritance), but its public API — `__init__` signature and
`forward(x, t, y) -> Tensor` — is **identical** to `DiT`, so it is a
drop-in replacement in `train.py` / `sample.py` / configs.

Registration is via `models/__init__.py` `MODEL_REGISTRY` (existing
mechanism), so a new `configs/model/ar_dit_s2_cifar.yaml` selects it
declaratively.

---

## 12. Test plan (v1, revised after §10)

**Layer 1 — `tests/test_components.py` extensions** for `AttnResJunction`:

- `test_attnres_shape`: forward on random `v_i` list of length `l` for
  a few `l ∈ {1, 2, 24}`; assert output is `[B, N, D]`.
- `test_attnres_zero_init_uniform_mix`: at `w_l = 0`, junction output
  should equal `mean(v_i for i in range(l))` up to float tolerance.
  This is the paper's uniform-init behaviour (§10).
- `test_attnres_rmsnorm_inside_kernel_only`: scale one `v_i` by a
  constant `k`; check that attention weights are unchanged (RMSNorm
  inside the kernel cancels the scaling on the key path), but the
  output magnitude scales linearly in that source (values are consumed
  un-normed in the weighted sum).
- `test_attnres_softmax_normalisation`: verify `α` sums to 1 over
  source axis.
- `test_attnres_grad_flow`: backward on a random target; assert both
  `w_l.grad` and `RMSNorm.g.grad` are non-zero.

**Layer 2 — `tests/test_dit.py` extensions (or new `test_ar_dit.py`)**:

- `test_ar_dit_forward_shape_and_dtype`: parallel to DiT test.
- `test_ar_dit_zero_init_output_is_zero`: assert `ARDiT(x, t, y) == 0`
  (bit-exact, `torch.equal` against `zeros_like`) at init. This is the
  same acceptance criterion as baseline DiT (see [DiT.md](DiT.md) §9.5
  / `test_dit_zero_init_output` in [tests/test_dit.py](../tests/test_dit.py)).
  Note the internal mechanism differs — baseline achieves zero output
  via `FinalLayer.linear = 0` fed by `v_0`; AR-DiT achieves the *same*
  zero output via `FinalLayer.linear = 0` fed by `v_0 / (2L)`. The
  initialisation is **not** an identity mapping of baseline DiT (see
  §10), but the observable at the model boundary is identical.
- `test_ar_dit_zero_init_internal_scaling` *(diagnostic, optional)*:
  hook the last block's output *before* `FinalLayer`; assert it equals
  `v_0 / (2L)` up to float tolerance. This is the AttnRes-specific
  wiring check that distinguishes AR-DiT's mechanism from baseline
  DiT's — a passing `output_is_zero` test alone would not catch a bug
  where AttnRes silently degenerated to identity residual.
- `test_ar_dit_param_count`: assert exact analytical parameter-count
  diff vs baseline DiT is `2L · 2 · D` (queries + RMSNorm scales).
- `test_ar_dit_smoke_roundtrip`: full forward + MSE loss + backward,
  assert no NaN and all trainable parameters receive gradient.

**Layer 4 — overfit-one-batch** for AR-DiT, same recipe as DiT.

The provisional-test warning in `doc/Plan.md` continues to apply —
these tests should be written now for coverage, but they are subject to
the same "written but not reviewed" caveat as the rest of the suite
until the dedicated review pass.

---

## 13. Compute / memory budget summary

For DiT-S/2 on CIFAR-10 (`B=128`, `N=256`, `D=384`, `L=12`, fp32):

| Item                                     | Cost                              |
|------------------------------------------|-----------------------------------|
| Extra parameters                         | ~18 K (0.056 % of ~33 M)          |
| AttnRes ops per forward                  | ~944 M scalar ops (≪ MHSA cost)   |
| Peak cache memory (fp32, deepest junction) | ~1.2 GB                         |
| Peak cache memory (bf16, deepest junction) | ~0.6 GB                         |

All figures are per DDP-rank. Nothing here requires special-casing in
the training loop.

---

## 14. Open questions (for later, not blocking v1)

1. Does gradient checkpointing across junctions become necessary at
   DiT-XL/2 (`L=28`, `D=1152`, `N=1024` for 32×32 latent)? Cache is
   `56 · 128 · 1024 · 1152 · 4 B ≈ 33 GB` fp32 per rank — likely yes.
2. Should E1 (`w_l(t)`) share its MLP across junctions, or have one
   MLP per junction? Parameter-vs-capacity trade-off.
3. Empirical: do the learned `α_{i→l}` show interpretable patterns
   (e.g. "attention to shallow layers dominates for background patches,
   deep layers for foreground")? If yes, that's a nice qualitative
   result independent of FID gains.

---

## References

- Kimi Team, *Attention Residuals*, arXiv:2603.15031, 2026.
- Peebles & Xie, *Scalable Diffusion Models with Transformers*,
  ICCV 2023.
- Zhang & Sennrich, *Root Mean Square Layer Normalization*,
  NeurIPS 2019.

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

## 9a. E1 — Time-conditioned pseudo-query (first implemented extension)

The v1 spec above (§§ 1–8, 10–14) is the paper-strict port. §9a is the
**first extension we actually implement**: it lifts entry E1 from the
ablation menu into a real, testable model, following the same
sign-off-before-code discipline as v1.

E1 leaves every other AttnRes design choice untouched — same 2L
junctions (§3), same operator (§4), same per-patch weights (§6), same
per-junction RMSNorm (§7), same pre-residual value source (§8). The
sole change is **where each junction's pseudo-query comes from**.

### 9a.1 Mechanism

v1 uses a **constant per-junction pseudo-query**:

```
q_l = w_l                              # w_l ∈ ℝ^D, a learned constant vector.
```

E1 replaces this with a **time-conditioned pseudo-query**:

```
τ    = MLP_shared(t_emb)               # [B, D]  — one shared 2-layer MLP, computed once
q_l  = W_l · τ + w_l                   # [B, D]  — per-junction linear head + additive bias
```

with

- `t_emb ∈ ℝ^{B×D}` — the existing `TimestepEmbedder(t)` output (the
  same tensor DiT already computes at the top of `forward`).
- `MLP_shared: ℝ^D → ℝ^D` — a 2-layer MLP `Linear(D, D) → SiLU →
  Linear(D, D)`, shared across **all 2L junctions**. Learns "how to
  represent the current denoising phase".
- `W_l: ℝ^D → ℝ^D` — a per-junction linear head, `nn.Linear(D, D,
  bias=False)`. One `W_l` per AttnRes junction (2L total). Learns "how
  this specific junction wants to weight sources given the phase
  representation".
- `w_l ∈ ℝ^D` — v1's learnable constant, retained as an **additive
  bias**. Zero-init, as in v1. Keeps the "E1 = v1 + a time-dependent
  perturbation" framing crisp: at `τ = 0` (or at `W_l = 0`), E1
  degenerates to v1 exactly.

**Everything else in the AttnRes operator is byte-identical to §4**:
the kernel is still `ϕ(q, k) = exp(q · RMSNorm(k))` with `k_i := v_i`,
the softmax is still over the source-junction axis (length `l`), the
attention weights are still per-patch (`α ∈ ℝ^{B×N×l}`), and the values
are still consumed un-normed in the weighted sum.

The only observable shape change on the operator's input side is that
the query is now `[B, D]` instead of `[D]` — i.e. one query per image
(and per junction), not a global constant. Concretely the logit
einsum changes from

```
logits = einsum("d,   bnld -> bnl", w_l,          keys_normed)     # v1
```

to

```
logits = einsum("bd,  bnld -> bnl", q_l,          keys_normed)     # E1
```

with `q_l = W_l(τ) + w_l`. The output shape `[B, N, l]` is unchanged.

### 9a.2 Signal source — `t_emb`, not `c` (fork #1 decision)

**Locked decision**: the shared trunk reads **`t_emb`** — the raw
`TimestepEmbedder(t)` output — **not** the adaLN-Zero conditioning
vector `c = t_emb + y_emb`.

- Chosen: `τ = MLP_shared(t_emb)`. Depth-mixing weights depend purely
  on the timestep.
- Rejected alternative: `τ = MLP_shared(c)`. Would let the class label
  also modulate the depth mix.

**Why `t_emb` only**:

1. **Faithful to the "time-conditioned" framing.** Plan.md line 23
   describes E1 verbatim as "timestep-conditioned pseudo-queries
   `w_l(t)`". Using `c` would smuggle class information through the
   name.
2. **Cleaner ablation.** Any FID delta E1-vs-v1 is attributable to
   the time signal alone, not to a second class-conditioning path
   competing with adaLN-Zero's existing one.
3. **Two independent branches, one purpose each.** In `ARDiTCond`,
   `t_emb` fans out to two consumers: (a) the adaLN-Zero path
   (`c = t_emb + y_emb` → `adaLN_modulation(c)` per block, unchanged),
   and (b) the E1 path (`τ = MLP_shared(t_emb)` → per-junction
   query, new). Class information continues to flow into every block
   *only* through the adaLN path. This separation of concerns is the
   whole point of picking `t_emb`.

**Ablation note (recorded, not implemented in E1 v1)**: swapping
`t_emb` → `c` in the trunk input is a natural follow-up. It answers
"does the class-conditioning signal in the depth mix help further, or
does it hurt by fighting adaLN-Zero?"

### 9a.3 Query factorisation — shared trunk + per-junction head (fork #2 decision)

**Locked decision**: option **δ** — a shared 2-layer trunk producing a
common phase representation `τ`, followed by a per-junction linear
head `W_l` that projects `τ` into the junction's query.

For completeness, the design space we considered:

| Option | Formula                                     | Extra params per junction     | Total for DiT-S/2 (2L=24, D=384) |
|--------|---------------------------------------------|-------------------------------|----------------------------------|
| α      | `q_l = w_l + b_l · MLP_shared(t_emb)`       | `D` scalar bias                | ~9 K + one shared MLP            |
| β      | `q_l = W_l · t_emb + w_l`                   | `D² + D ≈ 148 K`               | ~3.5 M (no shared trunk)         |
| γ      | `q_l = MLP_l(t_emb)` (per-junction 2-layer) | `2·D² ≈ 295 K`                 | ~7 M                             |
| **δ**  | `q_l = W_l · MLP_shared(t_emb) + w_l`       | `D²` per junction + shared MLP | **~3.83 M**                      |

**Why δ**:

1. **Right factorisation.** All 2L junctions consume the same input
   `t_emb`. Having each learn its own 2-layer MLP over that input (γ)
   would waste capacity re-learning the same "what phase are we in"
   representation 2L times. Factoring it out into a shared trunk lets
   the 2L per-junction heads specialise on the more targeted question
   *"given the phase, what does this junction want?"*.
2. **Preserves v1 exactly at `W_l = 0`.** With `W_l = 0` in every
   head, `q_l = w_l` — literally the v1 pseudo-query. E1 is thus a
   strict superset of v1: any FID improvement must come from the
   *learned* time-dependence introduced by non-zero `W_l`.
3. **Cheaper than γ, more expressive than α.** ~3.83 M new params
   (~12 % on top of DiT-S/2's 33 M) — noticeable but affordable in the
   ablation-cost budget. α (additive bias only) would give every
   junction the same time-dependent perturbation, differing only by a
   scalar magnitude — too weak.
4. **Symmetric with adaLN-Zero.** adaLN's per-block modulation MLP
   already reads a shared vector (`c`) and emits per-block affine
   parameters. δ applies the same shared-trunk + per-junction-head
   pattern to the query side of AttnRes.

**Ablation note (recorded, not implemented)**: α (additive-bias-only)
is a cheap alternative. Worth trying if E1-δ turns out to help —
answers "is the per-junction-head worth its ~3.5 M param cost, or is
per-junction time modulation via a scalar gate enough?"

### 9a.4 Parameter cost

For DiT-S/2 (`D = 384`, `L = 12` ⇒ `2L = 24` junctions):

| Item                                         | Formula                | Count       |
|----------------------------------------------|------------------------|-------------|
| Shared trunk (`Linear(D,D) → SiLU → Linear(D,D)`) | `2·D² + 2·D`      | ~295 K      |
| Per-junction heads (`W_l`, no bias)          | `2L · D²`              | ~3.54 M     |
| **E1 new params**                            | (sum)                  | **~3.83 M** |
| v1 AttnRes params (for reference)            | `2L · 2·D`             | ~18 K       |
| Baseline DiT-S/2                             | —                      | ~33 M       |
| **`ARDiTCond` total**                        | (sum)                  | **~37 M**   |

Percentage overhead of E1 over baseline DiT: **~12 %**. Compare with
v1's ~0.056 % — E1 is materially larger than v1 but still an order of
magnitude smaller than an equivalent-parameter-count expansion of the
MHSA/MLP path.

### 9a.5 Initialisation and the step-0 invariant

**Locked decision**: `W_l.weight = 0`, `w_l = 0`, `MLP_shared`
Xavier-uniform (via the generic init pass), everything else identical
to `ARDiT`.

**Why zero-init on `W_l`**: v1's §12 acceptance criterion —
`ARDiT(x, t, y) == 0` bit-exactly at step 0 — must survive to E1. The
argument for v1 was: `w_l = 0` gives `α_{i→l} = 1/l` (uniform mix), all
adaLN gates are zero, so the cache is `[v_0, 0, ..., 0]`, and
`FinalLayer.linear = 0` zeros the output regardless.

For E1 the same conclusion holds *only if* `q_l ≡ 0` at step 0 for
every `t` and every `l`. With option δ:

- `q_l = W_l(τ) + w_l`.
- `W_l.weight = 0`  ⇒  `W_l(τ) = 0` for any `τ`.
- `w_l = 0`         ⇒  additive bias contributes nothing.
- Therefore `q_l ≡ 0` for every `t`, every `l`.

The `MLP_shared` trunk gets Xavier init (via `_init_weights`'s generic
pass), because its output `τ` is annihilated by `W_l = 0` at step 0 and
so any finite init is safe — Xavier is chosen for consistency with the
rest of the model.

With `q_l ≡ 0` in every junction, every logit is `0`, every softmax
becomes uniform `1/l`, the cache stays `[v_0, 0, ..., 0]` (adaLN gates
still zero at init, unchanged from v1), `FinalLayer.linear` is still
zero — so the model output is bit-exactly zero. Same observable as v1,
same test (see §9a.8).

**What "learning `W_l` away from zero" looks like**: the gradient of
the loss w.r.t. `W_l.weight` at step 0 is non-zero as long as `τ` is
non-zero (and it will be — Xavier'd trunk with a non-zero `t_emb`
input gives a non-zero `τ`). So `W_l` starts moving on step 1 exactly
as adaLN's zero-init modulation MLPs do — the same gradient flow
mechanism, applied to the E1 query path.

### 9a.6 Module placement and code shape (fork #4 decision)

**Locked decision**: the shared trunk is a **model-level** module owned
by `ARDiTCond`; per-junction heads `W_l` live on `ARDiTCondBlock`; the
inner `AttnResJunction` gains one optional `q_override` kwarg so the
same softmax-mix code path serves both v1 and E1.

Concretely:

- **`ARDiTCond.__init__`** owns `self.t_query_trunk =
  TimeQueryTrunk(D)` — a single module for the whole network.
- **`ARDiTCond.forward`** computes `τ = self.t_query_trunk(t_emb)`
  **once**, right after `t_emb` is produced, and threads it as an
  extra argument into every block. `c = t_emb + y_emb` is still built
  the same way and still drives adaLN-Zero unchanged.
- **`ARDiTCondBlock`** owns two per-junction heads: `self.W_msa =
  nn.Linear(D, D, bias=False)` (for the MSA junction, junction index
  `2b+1`) and `self.W_mlp` (for the MLP junction, junction index
  `2b+2`). Both zero-init.
- **`ARDiTCondBlock.forward(x, c, tau, cache)`** computes `q_msa =
  self.W_msa(tau) + self.attn_res_msa.w` and passes it as
  `q_override=q_msa` when calling the shared `AttnResJunction`. Same
  for MLP. The rest of the block is byte-identical to `ARDiTBlock`.
- **`AttnResJunction.forward`** gains `q_override: Tensor | None =
  None`. `None` → v1 einsum (`"d,bnld->bnl"`, using `self.w`);
  provided → E1 einsum (`"bd,bnld->bnl"`, using `q_override`). No
  duplication of the softmax-mix math.

**Why the split "trunk on model / heads on block"**: the trunk is
shared across all 2L junctions, so it belongs to the outer container.
The heads are junction-specific and there are two per block, so they
belong to the block. The junction itself is just the softmax-mix
kernel and stays unchanged in structure — this keeps §4 (the paper's
Eq. 2/4 operator) canonical.

**Why the `q_override` kwarg (not two junction classes)**: the E1
change is *entirely* in how the query is constructed; the mixing math
is identical. Duplicating `AttnResJunction` into a `AttnResJunctionCond`
just to swap one einsum string would leave the paper's core operator
maintained in two places. One kwarg with a `None` default preserves the
v1 code path byte-identically while making the E1 path opt-in.

### 9a.7 File / API layout

```
models/
├── dit.py             # (unchanged)
└── ar_dit.py          # (extended, no destructive edits)
    ├── class AttnResJunction(nn.Module)      # (existing) — forward gains q_override kwarg
    ├── class ARDiTBlock(nn.Module)           # (existing) — unchanged
    ├── class ARDiT(nn.Module)                # (existing) — unchanged
    ├── def ARDiT_{S,B,L,XL}_2                # (existing) — unchanged
    ├── class TimeQueryTrunk(nn.Module)       # NEW — shared 2-layer MLP
    ├── class ARDiTCondBlock(nn.Module)       # NEW — E1 block, owns W_msa / W_mlp
    ├── class ARDiTCond(nn.Module)            # NEW — E1 end-to-end model
    └── def ARDiTCond_{S,B,L,XL}_2            # NEW — presets parallel to ARDiT_*_2
```

**`ARDiTCond` public API is identical to `ARDiT`** — same `__init__`
signature (no new required kwargs; the trunk / heads are constructed
internally from `hidden_size` and `depth`), same `forward(x, t, y)
-> Tensor`, same output shape. This means:

- `train.py`, `sample.py`, `flow/`, `eval/`, and every existing config
  keep working unchanged.
- Selection between v1 and E1 is **purely** via config
  (`arch_name: ARDiTCond_S_2` vs `arch_name: ARDiT_S_2`). No boolean
  flags anywhere.

**Config**: new `configs/model/ar_dit_cond_s2_cifar.yaml`, a copy of
`configs/model/ar_dit_s2_cifar.yaml` with `arch_name` changed and the
header comment updated. **Registry**: the four new presets are added
to `models/__init__.py`'s `_ARCH_PRESETS` and to `__all__`.

### 9a.8 Test plan (extension of §12)

New tests in `tests/test_ar_dit.py` (appended, not replacing v1
tests). Each parallels a v1 test where one exists, plus one new
smoking-gun test unique to E1.

| Test | Assertion | Purpose |
|------|-----------|---------|
| `test_ardit_cond_forward_shape_and_dtype` | Output is `[B, C, H, W]`, `float32`. | Same shape contract as `ARDiT`. |
| `test_ardit_cond_zero_init_output_is_zero` | `torch.equal(ARDiTCond(x, t, y), zeros_like(...))` at init. | Same acceptance criterion as v1 §12; §9a.5 argues this must hold. |
| `test_ardit_cond_zero_init_uniform_mix` | Hook `FinalLayer`'s input at init; assert it equals `v_0 / (2L + 1)` for `t = 0.01` **and** `t = 0.99`. | Diagnostic — verifies the depth mix is uniform *and time-invariant* at init, distinguishing E1's step-0 mechanism (`q_l ≡ 0` for every `t`) from a bug where a non-zero `W_l` leaks. |
| `test_ardit_cond_param_count_diff` | `(ARDiTCond).num_params - (ARDiT).num_params == 2·D² + 2·D + 2L·D²`. | Analytical formula, computed directly from `(D, L)` so a config change flows through cleanly. |
| `test_ardit_cond_time_dependence` | After manually setting `blocks[-1].W_mlp.weight` to `randn(D, D) * 1.0` **and** `FinalLayer.linear.weight` to `randn(...) * 1.0`, changing `t` alone (holding `x, y` fixed) changes the model output by `> 1e-2` in max-abs. | **Smoking-gun**: E1 actually depends on `t` via the AttnRes path, not just via adaLN-Zero. See "Zero-init gradient dams" note below for why the last MLP junction, and not block 0's MSA junction, is the correct perturbation site. |
| `test_ardit_cond_grad_flow` | Forward + MSE + backward at init; assert every parameter (v1 and E1) has `p.grad is not None and torch.isfinite(p.grad).all()`. | Structural reachability check — the E1 modules are on the backward graph. See "Zero-init gradient dams" note below for why the numerical `grad != 0` version is spec-broken at step 0. |

**Note on `test_ardit_cond_time_dependence`**: baseline `ARDiT` and
v1's tests do not assert time-dependence of the AttnRes path (v1
`α_{i→l}` is time-independent by construction). This test is
E1-specific and is the strongest positive evidence that the extension
works as designed.

**Zero-init gradient dams — a design consequence worth naming**.
Adding these tests surfaced two non-obvious implications of §9a.5's
zero-init story that future ablation authors should be aware of:

1. **The adaLN-Zero *value* dam propagates through junctions.** With
   every `gate_msa = gate_mlp = 0` at init, `v_i = 0` for all `i ≥ 1`
   in the cache. A `t`-dependent junction output `h_l` becomes the
   input `x` of the *next* sub-layer, but that sub-layer's contribution
   `v_{l+1} = gate · f(...)` is then zero — so the E1 signal never
   enters any downstream cache entry, only ever the intermediate `x`
   which is then multiplied by zero. A perturbation to `W_l` at an
   *early* junction is therefore invisible at the model output at
   init; only the *last* junction (whose output feeds `FinalLayer`
   directly) produces a visible signal. This is why
   `test_ardit_cond_time_dependence` perturbs `blocks[-1].W_mlp`.
2. **The zero-init *gradient* dam propagates through parameters.**
   `FinalLayer.linear.weight = 0` makes the model bit-exactly zero at
   init, so `∂L/∂h = ∂L/∂output · linear.weight^T = 0` — every
   parameter upstream of `FinalLayer.linear` receives numerically-zero
   gradient at step 0. Additionally, since `∂L/∂τ = Σ_l W_l^T ·
   ∂L/∂q_l`, and every `W_l = 0`, the shared trunk `t_query_trunk`
   would receive zero gradient even if the `FinalLayer` dam were
   lifted. This is intentional — the same warm-up behaviour
   adaLN-Zero itself exhibits, extended by one more zero-init layer.
   Only `FinalLayer.linear.{weight,bias}` receive non-zero gradient at
   step 0; everything else unfreezes progressively over the first few
   optimiser steps. A step-0 `grad != 0` assertion on E1 params would
   therefore be spec-broken — instead the test uses `p.grad is not None`
   (structural reachability), which distinguishes "on the backward
   graph, numerically zero at init" from "orphaned, never called in
   `forward`".

The provisional-test warning from `doc/Plan.md` still applies — these
tests are written for coverage, not correctness certification, until
the dedicated review pass.

### 9a.9 Open questions (recorded, not blocking)

1. **Trunk depth.** We chose 2-layer `D → D → D` mirroring
   `TimestepEmbedder`'s style. Whether 1-layer (`Linear(D, D)` only)
   suffices, or whether an mlp_ratio=4 expansion helps, is unmeasured.
2. **Trunk input.** `t_emb` vs `c` (§9a.2's ablation).
3. **Bias on `W_l`.** We chose `bias=False` since `w_l` already plays
   that role; whether adding a redundant bias helps optimisation is
   unmeasured.
4. **α (additive-bias-only) as a cheaper alternative.** §9a.3
   footnotes this — worth a follow-up if E1-δ works.

None of these are on E1's critical path.

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

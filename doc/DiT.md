# Baseline DiT — Design Document

This document specifies the **baseline Diffusion Transformer (DiT)** we will
implement in `models/dit.py`. It follows Peebles & Xie, *Scalable Diffusion
Models with Transformers* (ICCV 2023), restricted to the **DiT-XL/2 family with
adaLN-Zero conditioning** — the variant the original paper recommends and the
one used in nearly all DiT follow-up work.

The goal is a **clean, minimal, faithful** implementation. No tricks, no
optimisations beyond what the paper specifies. This will serve as the control
arm of the AR-DiT experiment, so correctness and clarity matter more than
speed.

---

## 1. Inputs and outputs

| Symbol | Shape                  | Description                                                  |
|--------|------------------------|--------------------------------------------------------------|
| `x`    | `(B, C, H, W)`         | Interpolant state `x_t` at flow-matching time `t`.           |
| `t`    | `(B,)`                 | Float time values, conventionally `t ∈ [0, 1]`.              |
| `y`    | `(B,)`                 | Class labels in `[0, num_classes]` (last id = "null").       |
| out    | `(B, C, H, W)`         | Predicted velocity `v_θ(x_t, t, y)`. `C_out = C`.            |

We operate **directly in pixel space** — no VAE — so the architecture stays a
pure PyTorch project. The default benchmark is **CIFAR-10**: `C=3, H=W=32`,
images normalised to `[-1, 1]`.

The training paradigm is **flow matching with velocity prediction**
(rectified-flow / linear-interpolant style), not DDPM. The model is therefore
a pure function `v_θ(x_t, t, y)` and has no learned-variance head; output
channel count always equals `C`. The training/sampling math (interpolant,
loss, ODE solver) is intentionally kept out of this document and will live
in `doc/FlowMatching.md`.

The model itself is **fully shape-agnostic and config-driven**. All of
`input_size` (`H=W`), `in_channels` (`C`), `patch_size` (`P`),
`num_classes`, `hidden_size` (`D`), `depth` (`L`), and `num_heads` are
constructor arguments. To switch to e.g. CelebA-64 (`3×64×64`) or ImageNet
latent (`4×32×32`) you only change the config — no code change is required.
The only constraint is that `H` and `W` must be divisible by `P`.

---

## 2. Architecture overview

```
            ┌──────────────────────────────────────────┐
   x ──►    │ PatchEmbed (Conv2d kernel=P, stride=P)   │ ──► tokens (B, N, D)
            └──────────────────────────────────────────┘
                       + sin-cos 2D positional embedding (fixed)
                                    │
   t ─► TimestepEmbedder ─┐         │
                          ├─► c (B, D)
   y ─► LabelEmbedder ────┘         │
                                    ▼
            ┌──────────────────────────────────────────┐
            │   N_layers × DiTBlock(c)                 │
            │     (adaLN-Zero, MHSA, MLP)              │
            └──────────────────────────────────────────┘
                                    │
            ┌──────────────────────────────────────────┐
            │ FinalLayer(c)  →  Linear → unpatchify    │
            └──────────────────────────────────────────┘
                                    │
                                    ▼
                                  (B, C_out, H, W)
```

All conditioning (`t`, `y`) flows through a single vector `c ∈ R^D`,
broadcast to every block via adaLN-Zero. There is **no cross-attention**.

---

## 3. Components

### 3.1 PatchEmbed
- A single `Conv2d(C, D, kernel_size=P, stride=P, bias=True)`.
- Produces `N = (H/P) * (W/P)` tokens of width `D`.
- Default for CIFAR-10: `P = 2` (32×32 → 16×16 = 256 tokens).
  For 64×64 inputs, `P = 4` keeps the same `N=256`.

### 3.2 Positional embedding
- Fixed (non-learned) **2D sin-cos** embedding of shape `(1, N, D)`,
  added once after `PatchEmbed`.
- Frozen during training. Same scheme as MAE / DiT reference code.

### 3.3 TimestepEmbedder
- Sinusoidal embedding (Transformer-style) of dimension `D_freq` (default 256).
- Followed by `MLP: D_freq → D → D` with SiLU.
- Output: `(B, D)`.

### 3.4 LabelEmbedder
- `nn.Embedding(num_classes + 1, D)`. The extra `+1` slot is the **null class**
  used for classifier-free guidance (CFG).
- During training, each label is independently dropped to the null id with
  probability `p_drop = 0.1` (paper default). No drop at inference.
- Output: `(B, D)`.

### 3.5 Conditioning vector
```
c = TimestepEmbedder(t) + LabelEmbedder(y)        # shape (B, D)
```

### 3.6 DiTBlock (adaLN-Zero)

The heart of the model. One block computes:

```python
# c is the global conditioning vector (B, D)
# Six modulation parameters per block, produced from c:
shift_msa, scale_msa, gate_msa, \
shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(c).chunk(6, dim=-1)
# adaLN_modulation = nn.Sequential(SiLU, Linear(D, 6*D))

h = x + gate_msa * MHSA( modulate(LN(x), shift_msa, scale_msa) )
h = h + gate_mlp * MLP ( modulate(LN(h), shift_mlp, scale_mlp) )
return h
```

where
- `LN` is `LayerNorm(D, elementwise_affine=False, eps=1e-6)` (no learned γ, β).
- `modulate(x, shift, scale) = x * (1 + scale) + shift`, with `shift, scale`
  unsqueezed to broadcast over the token dimension.
- `MHSA` is a standard multi-head self-attention, `n_heads`, head dim `D/n_heads`,
  QKV via a single `Linear(D, 3D)`, output proj `Linear(D, D)`.
- `MLP` is `Linear(D, 4D) → GELU(approximate="tanh") → Linear(4D, D)`.
- **adaLN-Zero init**: the final `Linear` of `adaLN_modulation` is
  zero-initialised (weight and bias). At step 0 every block is exactly the
  identity, which the paper showed is critical for stable training.

### 3.7 FinalLayer
```python
shift, scale = adaLN_modulation_final(c).chunk(2, dim=-1)   # zero-init
x = modulate(LN(x), shift, scale)
x = Linear(D, P * P * C_out)(x)                              # zero-init
```
Then **unpatchify** back to `(B, C_out, H, W)`.

---

## 4. Model sizes

We will support the four sizes from the paper. Default for our experiments is
**DiT-S/2** on CIFAR-10 (fits comfortably on a single 24 GB GPU and converges
in hours). DiT-B/2 is the secondary size if we want a stronger result.

| Name     | Layers `L` | Width `D` | Heads | Params |
|----------|-----------:|----------:|------:|-------:|
| DiT-S/2  | 12         | 384       | 6     | ~33 M  |
| DiT-B/2  | 12         | 768       | 12    | ~130 M |
| DiT-L/2  | 24         | 1024      | 16    | ~458 M |
| DiT-XL/2 | 28         | 1152      | 16    | ~675 M |

Patch size `P` defaults to 2; configurable via the constructor.
The size presets only fix `(L, D, n_heads)` — `input_size`,
`in_channels`, `patch_size`, and `num_classes` are always passed in by
the caller, so the same preset works for CIFAR-10, CelebA-64, or any other
dataset.

---

## 5. Initialisation summary

| Module                        | Init                                |
|-------------------------------|-------------------------------------|
| `PatchEmbed.proj`             | `xavier_uniform_` on flattened weight, bias=0 |
| Positional embedding          | 2D sin-cos, fixed                   |
| `TimestepEmbedder` MLPs       | `normal_(std=0.02)`                 |
| `LabelEmbedder.embedding`     | `normal_(std=0.02)`                 |
| Linear layers (default)       | `xavier_uniform_`, bias=0           |
| `adaLN_modulation[-1]`        | **weight=0, bias=0** (adaLN-Zero)   |
| `FinalLayer.linear`           | **weight=0, bias=0**                |

---

## 6. Forward pass (pseudocode)

```python
def forward(self, x, t, y):
    x = self.patch_embed(x) + self.pos_embed             # (B, N, D)
    c = self.t_embedder(t) + self.y_embedder(y, train=self.training)  # (B, D)
    for block in self.blocks:
        x = block(x, c)                                  # (B, N, D)
    x = self.final_layer(x, c)                           # (B, N, P*P*C_out)
    x = self.unpatchify(x)                               # (B, C_out, H, W)
    return x
```

---

## 7. Loss and training (deferred)

Training, EMA, optimizer, and FID evaluation are deliberately **out of scope**
for this document — they will live in a separate `doc/FlowMatching.md` once
the model code is in place. For DiT alone we only need to guarantee:

- The model takes `(x, t, y)` and returns a tensor of shape `(B, C, H, W)`
  interpreted as the predicted velocity `v_θ(x_t, t, y)`.
- `t` is a float tensor; the model passes it through the sinusoidal embedder
  unchanged. Any rescaling (e.g. `t * 1000` to match the embedder's
  pretraining-style frequency range) is the training loop's responsibility.
- The model contains no DDPM/IDDPM-specific machinery (no β schedule, no
  variance head, no fixed `T`). Everything diffusion- or flow-specific lives
  outside `models/`.

---

## 8. File layout for this milestone

```
models/
├── __init__.py
└── dit.py        # everything in §3, plus DiT model class and size presets
```

Single file, no external dependencies beyond PyTorch and `numpy` (the latter
only for sin-cos pos-embed construction).

---

## 9. Acceptance criteria

The implementation is "done" for this milestone when:

1. `DiT_S_2()`, `DiT_B_2()`, `DiT_L_2()`, `DiT_XL_2()` constructors run with
   keyword args overriding `input_size`, `in_channels`, `patch_size`,
   `num_classes`.
2. A forward pass on the **CIFAR-10 default** `(B=2, C=3, H=W=32)` with
   random `t, y` returns shape `(2, C_out, 32, 32)`.
3. The same preset, reconfigured to `input_size=64, in_channels=3,
   patch_size=4`, runs end-to-end and returns shape `(2, C_out, 64, 64)`
   — verifying the model is shape-agnostic.
4. Parameter counts match Table 2 of the paper within ±1 % (at the original
   ImageNet-latent config `C=4, P=2, input_size=32`).
5. With adaLN-Zero correctly applied, the **untrained** model output is
   exactly zero for every input (sanity check that zero-init is wired up).
6. No `nan`/`inf` in a forward+backward pass with random data.

These are sanity checks — we are not aiming to reproduce FID at this stage.

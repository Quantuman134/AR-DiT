# TO FIX

Running log of known issues in the repository that are **out of scope for
the current working branch** but must be fixed on a dedicated branch
before they compound. Each entry is date-stamped, links to a specific
observable failure, and records what we know about root cause so far.

Adding an entry costs nothing; leaving a real issue undocumented is how
they turn into "why does this test fail on main" mysteries later.

---

## Workflow

1. **On discovery** — when you hit a broken-but-out-of-scope issue,
   append a new entry to *Open issues* using the template below.
   Record what you know now, not what you'll research later.
2. **While fixing** — do the work on a dedicated fix branch (never
   piggyback a fix onto an unrelated feature branch), so the fix's
   commits and its regression tests are reviewable in isolation.
3. **After the fix lands** — **archive the entry**. Move it, in full,
   from *Open issues* to *Closed / resolved*, and add a **Resolution**
   stanza recording:
   - the commit hash / PR that fixed it,
   - the actual root cause (which may differ from the "so far" guess),
   - which option (a / b / …) from *Proposed fix* was taken and why,
   - the date closed.

   Do **not** delete closed entries. Their forensic value — "this
   symptom looked like X but was actually Y" — is the whole point of
   keeping the log. *Open issues* should stay a small live working
   set; *Closed / resolved* is the growing institutional memory.

---

## Entry template

```
### <short title>
- **Discovered on**: <branch name> — <yyyy-mm-dd>
- **Symptom**: <one-line observable failure — pytest error, runtime
  error, wrong output, etc.>
- **Reproduce**: <exact command / file / line to see it>
- **Root cause (so far)**: <what we know, what we don't>
- **Proposed fix**: <options, none locked-in>
- **Blocking?**: <yes/no + which downstream work>
- **Owner**: <who picks it up, or "unassigned">
- **Resolution**: *(filled in when archived — see below)*
```

When archiving to *Closed / resolved*, append the following stanza to
the entry (do not remove the original fields; the diagnostic history
stays part of the record):

```
- **Resolution**:
    - **Fixed on**: <branch name> @ <commit hash or PR #> — <yyyy-mm-dd>
    - **Actual root cause**: <what was really going on, if different
      from "Root cause (so far)">
    - **Chosen fix**: <which proposed option, or a new option, plus a
      one-line rationale>
    - **Regression coverage**: <test file/name that now guards
      against recurrence — or "none" if not applicable>
```

---

## Open issues

### Shipped `cifar10_train.yaml` points at `dit_xl2_cifar.yaml`, test expects `dit_s2_cifar.yaml`
- **Discovered on**: `feature/ar-dit-cond-e1` — 2026-07-30
- **Symptom**:
  ```
  FAILED tests/test_configs.py::test_shipped_train_yaml_loads_and_embeds_model
  arch_name: 'DiT_XL_2' != 'DiT_S_2'
  ```
- **Reproduce**:
  ```
  conda run -n dit python -m pytest tests/test_configs.py::test_shipped_train_yaml_loads_and_embeds_model -q
  ```
  Fails identically on clean `master @ 3bb1801` (verified by `git stash`
  round-trip), so this is **not** an E1 regression — it landed via the
  merged PR `fix/small-fixes-and-docs` at commit `f60b9a9` ("remote
  script and config file").
- **Root cause (so far)**: `configs/train/cifar10_train.yaml` was
  edited so `model_config: ../model/dit_xl2_cifar.yaml` and
  `logging.run_name: dit_xl2_cifar`, but three surface areas were left
  inconsistent:
  1. The file's own header comment (line 2) still says *"Training
     config — DiT-S/2 on CIFAR-10"*.
  2. `tests/test_configs.py::test_shipped_train_yaml_loads_and_embeds_model`
     still asserts `t.model == load_model_config(configs/model/dit_s2_cifar.yaml)`.
  3. Intent unclear — was the pointer swap meant to become the shipped
     default (making the test stale), or is it a personal-run leak
     (making the pointer wrong)?
- **Proposed fix**: two options, pick after clarifying intent —
  - **(a)** Revert the pointer to `../model/dit_s2_cifar.yaml` and
    `run_name` to `dit_s2_cifar`. Header comment already agrees. Test
    passes as-is. This is the "S/2 is the shipped default" reading.
  - **(b)** Keep the XL/2 pointer, update the test to compare against
    `dit_xl2_cifar.yaml`, and fix the header comment to say
    *"Training config — DiT-XL/2 on CIFAR-10"*. This is the "XL/2 is
    now the shipped default" reading.
- **Blocking?**: **No.** Does not block E1 development
  (`feature/ar-dit-cond-e1`) — the failure is on `test_configs.py`,
  not on any test path E1 touches. Flagged only so the branch's own
  regression sweep (`pytest tests/test_ar_dit.py tests/test_components.py
  tests/test_dit.py tests/test_configs.py`) isn't misread as "E1 broke
  something".
- **Owner**: unassigned. To be picked up on a dedicated branch
  (suggested name: `fix/train-config-consistency`) after E1 lands.
- **Resolution**: *(pending — fill in when archived)*

---

## Closed / resolved

Entries move here in full (not deleted) once fixed, with a
**Resolution** stanza appended — see *Workflow* step 3. This section
is intended to grow; it is the project's institutional memory of
"what looked broken, what turned out to be the real problem, and how
we fixed it".

### E1 (ARDiTCond) training collapses to random-noise-tier FID at ~800k steps
- **Discovered on**: `feature/grad-norm-inspection` — 2026-08-01
- **Symptom**:
  ARDiTCond XL/2 on CIFAR-10 trained to 800k steps produced
  FID-50K ≈ **300** vs. AR-DiT baseline's FID ≈ 10 at the equivalent
  step budget (i.e. samples were near-pure noise). Per-parameter-group
  gradient-norm logging (introduced by this very branch, ironically)
  revealed that at **step ≈ 750** the query-path parameter groups
  collapsed simultaneously: `blocks.attn_res_msa.w`,
  `blocks.attn_res_mlp.w`, `blocks.attn_res_msa.rms`,
  `blocks.attn_res_mlp.rms`, `blocks.attn`, `blocks.W_msa`,
  `blocks.W_mlp`, and `t_query_trunk` all dropped from O(1e-1) to
  **exactly 0.0** within one logging interval and stayed there for the
  remaining ~77k logged steps (95–98 % zero-fraction). Meanwhile
  `blocks.mlp`, `blocks.adaLN`, `final_layer`, `patch_embed`,
  `t_embedder`, `y_embedder` stayed healthy throughout — a very
  specific *"gradient dies on the softmax-kernel path but not on the
  additive-residual path"* fingerprint.
- **Reproduce**:
  - CSVs preserved at [`tmp/wandb_export_2026-08-01T02_*.csv`](/home/hongkun/Attention_Residual_for_DiT/tmp).
  - Diagnostic summariser: [`tmp/_diag.py`](/home/hongkun/Attention_Residual_for_DiT/tmp/_diag.py).
  - Full write-up with plots + inference chain: [`tmp/ardit_cond_diagnosis.html`](/home/hongkun/Attention_Residual_for_DiT/tmp/ardit_cond_diagnosis.html).
- **Root cause (so far)**: The E1 kernel evaluates
  `logit = q_override · RMSNorm(k)` with **no bound on `‖q_override‖`**.
  Since `q_override = W_msa(tau) + w` where `W_msa` is a learned
  linear whose spectral norm grows unbounded during training,
  `|logit|` can grow into the range where softmax saturates to a
  one-hot distribution. At that point `∂α/∂logit ≈ 0`, so all upstream
  parameters that produced `q_override` (trunk → W_msa/W_mlp → w) —
  plus the key-side `RMSNorm` scale that shares the same jacobian —
  receive numerically-zero gradient and stay frozen forever. The
  v1 (AR-DiT) branch is immune because `q = self.w` is a single
  learned D-vector whose magnitude is controlled directly by the
  optimiser and never runs away.
- **Proposed fix**: three options considered —
  - **(a)** Add `1/√D` scaling to the E1 kernel path only. Cheapest
    change (one line, no new params), but bounds only the *initial*
    logit magnitude — `‖W_msa‖` could still grow ~34× and re-enter
    saturation later in a longer run.
  - **(b)** Add an in-kernel `nn.RMSNorm(D)` on the query path,
    symmetric with the existing one on the key path. Structurally
    caps `‖q‖` to O(1) forever, at the cost of D scalars per junction
    (~64k new params on XL/2, 0.014 % of the model).
  - **(c)** Both (a) and (b). Redundant.
- **Blocking?**: **Yes** — every E1 experiment beyond ~1k steps is
  invalidated until this lands.
- **Owner**: fixed on this branch (`fix/e1-q-rmsnorm-softmax-saturation`).
- **Resolution**:
    - **Fixed on**: `fix/e1-q-rmsnorm-softmax-saturation` @ (this commit) — 2026-08-04
    - **Actual root cause**: exactly as diagnosed above —
      unbounded `‖q_override‖` driving softmax saturation. The
      "dying groups vs. surviving groups" partition matched a
      saturated-softmax cutoff exactly: everything whose gradient
      passes through `α = softmax(q · RMSNorm(k))` dies; everything
      that reaches the loss via an additive residual survives.
    - **Chosen fix**: **Option (b)** — a per-junction
      `nn.RMSNorm(hidden_size)` (attribute name `q_rms`) added to
      `AttnResJunction`, applied inside the kernel on the
      `q_override` branch. The v1 branch (`q_override is None`) is
      left byte-identical: it still consumes the raw learned
      `self.w` un-normed, since `self.w` is bounded by the optimiser
      and RMSNorm-of-zero at init would be numerically undefined.
      This makes the E1 kernel
      `ϕ(q, k) = exp(RMSNorm(q) · RMSNorm(k))` — symmetric in its
      arguments and O(1)-bounded in the logit no matter how far the
      upstream `W_msa` weights drift. Option (a) was rejected as
      symptom-mitigation rather than a structural cure; option (c)
      as unnecessary belt-and-braces once (b) is in.
    - **Regression coverage**:
      - `tests/test_ar_dit.py::test_ar_dit_smoke_roundtrip` now
        asserts `q_rms.grad is None` on the paper-strict AR-DiT
        code path (dormancy invariant).
      - `tests/test_ar_dit.py::test_ar_dit_param_count_diff` now
        expects `2L · 3 · D` extra params (was `2L · 2 · D`) —
        catches accidental removal of the new module.
      - `tests/test_components.py::test_ar_dit_block_grad_flow` and
        `test_ar_dit_block_paramcount_vs_dit_block` updated
        symmetrically at block granularity.
      - Per-group gradient logging (from the parent branch) already
        provides live monitoring: a recurrence would show up as
        `blocks.attn_res_*.q_rms` moving away from an all-ones
        scale, or a sudden drop of any query-path group's grad-norm
        to exactly 0 in wandb.

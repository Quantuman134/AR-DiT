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

*(none yet)*

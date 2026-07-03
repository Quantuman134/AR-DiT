"""Layer 4 test — overfit-one-batch (see doc/Test.md §Layer 4).

The gold-standard *"the model can actually learn"* test.  Every other
test in ``tests/`` checks shapes, algebraic identities, or numerical
sanity — none of them prove the flow-matching training objective is
actually differentiable and reduces under gradient descent.  A
shape-correct model with a broken gradient path (wrong ``.detach()``,
target/prediction swapped, wrong reduction axis, timestep on the wrong
device, ...) would pass every existing test.  This one wouldn't.

Setup
-----
* ``B = 4`` synthetic clean-data tensors ``x_1`` with 4 fixed labels
  ``y``.  These two are the "batch" we overfit.
* Model: ``DiT_S_2(input_size=32, in_channels=3, num_classes=10)`` —
  same size the rest of the suite uses.
* Optimiser: AdamW, ``lr = 1e-4``, no weight decay (we want the model
  to memorise, not regularise).
* ``NUM_STEPS = 700`` gradient steps.
* At every step we resample ``x_0 ~ N(0, I)`` and ``t ~ Uniform(0, 1)``.
  We do **not** freeze ``x_0`` or ``t``: the target ``v_gt = x_1 − x_0``
  therefore changes every step, so the model cannot cheat by memorising
  a single input→output pair — it has to learn the *velocity field* that
  interpolates from arbitrary noise to the four fixed data points.

Why lr = 1e-4 (not 1e-3)
------------------------
The DiT paper trains at a constant ``lr = 1e-4`` — a value chosen for
adaLN-zero-initialised transformers, which are fragile to large early
step sizes.  A test that runs at ``lr = 1e-3`` is measuring the *wrong*
regime: a bug that only manifests at ``1e-4`` (the actual training LR)
would slip through.  Empirically ``1e-3`` also produces a sharp
"phase-transition" loss cliff around step 300 followed by a noisy tail
(non-monotone late-step behaviour), which makes the pass/fail threshold
sensitive to unrelated numerical drift.  ``1e-4`` gives a smooth,
monotone descent — the right signal for a regression test to key on.

Assertion
---------
``final_loss < 0.20 * initial_loss`` (an ≥80 % drop), where the two
endpoints are averaged over a small window of steps to damp the noise
that comes from resampling ``x_0`` and ``t``.  Empirically the run
converges to ~0.15 × initial by step 700, so the ``0.20`` threshold
leaves ~1.3× headroom — enough to absorb legitimate seed-to-seed
variance while still catching real regressions (a broken gradient path
sits near ``1.0 × initial`` throughout).  The remaining loss floor is
Monte-Carlo variance from resampled ``(x_0, t)``, not model capacity;
running for more steps does not push it substantially lower with only
4 data points.

This test is marked ``@pytest.mark.slow`` and is skipped by the default
``pytest tests/ -q`` invocation (see ``[tool.pytest.ini_options]`` in
``pyproject.toml``).  Run it explicitly with::

    pytest tests/test_overfit.py -q -m slow

Runtime is ~25 s on a single GPU, ~5–8 min on CPU.
"""

from __future__ import annotations

import pytest
import torch

from flow.interpolant import interpolant, velocity_gt
from flow.loss import flow_matching_loss
from models.dit import DiT_S_2


# --- Test hyperparameters ---------------------------------------------------
# Kept as module-level constants (not fixtures) so the intent is obvious at
# a glance; nothing else in the test suite reads these.
_BATCH_SIZE = 4
_NUM_CLASSES = 10
_IMG_HW = 32
_IN_CHANNELS = 3
_LR = 1.0e-4
_NUM_STEPS = 700

# Number of steps averaged at the beginning / end of the run when comparing
# initial vs final loss.  A single step's loss is a noisy estimator (one
# random ``(x_0, t)`` pair); averaging a small window gives a much more
# stable comparison without needing a long run.
_WINDOW = 10

# Loss must drop by at least this ratio (final < 0.20 * initial ⇒ ≥80 % drop).
# Empirically at ``lr=1e-4`` on this setup the run reaches ~0.15 × initial by
# step 700, so 0.20 leaves ~1.3× headroom — tight enough that a broken
# gradient path (which sits near 1.0 × initial the whole time) fails
# unambiguously, loose enough to absorb seed-to-seed variance from the
# resampled ``(x_0, t)`` targets.  See the module docstring for why we do
# not use ``lr=1e-3`` here even though it converges faster.
_LOSS_DROP_RATIO = 0.20


@pytest.mark.slow
def test_overfit_one_batch_dit_s2() -> None:
    """DiT-S/2 + flow-matching MSE can overfit a fixed 4-sample batch.

    Verifies the whole training-step chain end-to-end:
    ``x_1 → x_0 → t → x_t → v_pred → v_gt → loss → backward → step``.
    Any bug in any of those pieces (autograd, device, shape, sign, ...)
    manifests as the loss failing to decrease.
    """
    # Determinism.  ``conftest.py`` already seeds torch to 0 before every
    # test; we re-seed here explicitly so the test is self-contained if
    # the fixture is ever removed, and so that assertion messages below
    # are reproducible.
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Fixed batch (this is what we overfit) -----------------------------
    # ``x_1``: four random "images" standing in for real data.  Values are
    # not clamped to ``[-1, 1]``; the flow-matching loss is scale-agnostic,
    # and we deliberately want a batch that's *not* zero-centred so that
    # a "model that always predicts zero" isn't accidentally near-optimal.
    x_1 = torch.randn(_BATCH_SIZE, _IN_CHANNELS, _IMG_HW, _IMG_HW, device=device)
    y = torch.arange(_BATCH_SIZE, device=device, dtype=torch.long)  # 0,1,2,3

    # ---- Model + optimiser -------------------------------------------------
    # ``class_dropout_prob=0.0`` so the classifier-free-guidance path never
    # replaces ``y`` with the null token during this test — we want a
    # deterministic conditional signal to memorise.  (The default in
    # ``models/dit.py`` is already 0.1; overriding to 0.0 keeps the test
    # focused on the optimisation dynamics rather than the CFG mechanism,
    # which has its own tests in test_flow.py / test_dit.py.)
    model = DiT_S_2(
        input_size=_IMG_HW,
        in_channels=_IN_CHANNELS,
        num_classes=_NUM_CLASSES,
        class_dropout_prob=0.0,
    ).to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=0.0)

    # ---- Training loop -----------------------------------------------------
    losses: list[float] = []
    for _ in range(_NUM_STEPS):
        # Fresh noise & timestep every step.  This is the same recipe
        # ``train.py`` uses, and it prevents the "memorise one lookup"
        # failure mode discussed in the module docstring.
        x_0 = torch.randn_like(x_1)
        t = torch.rand(_BATCH_SIZE, device=device)  # Uniform[0, 1]

        x_t = interpolant(x_0, x_1, t)
        v_target = velocity_gt(x_0, x_1)
        v_pred = model(x_t, t, y)

        loss = flow_matching_loss(v_pred, v_target)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        losses.append(loss.item())

    # ---- Assertion ---------------------------------------------------------
    # Compare *windowed averages* rather than single steps: a single loss
    # is a noisy one-sample Monte-Carlo estimate over ``(x_0, t)``, and
    # even a well-trained model has non-zero variance across those
    # samples.  Averaging over ``_WINDOW`` steps gives a much stabler
    # estimate at either end of the run.
    initial_loss = sum(losses[:_WINDOW]) / _WINDOW
    final_loss = sum(losses[-_WINDOW:]) / _WINDOW

    assert initial_loss > 0.0, (
        f"Initial loss is non-positive ({initial_loss:.4e}); this should be "
        "impossible for a nonzero MSE and indicates a broken loss/gradient path."
    )
    assert final_loss < _LOSS_DROP_RATIO * initial_loss, (
        f"Overfit-one-batch failed: initial mean loss (steps 0..{_WINDOW}) = "
        f"{initial_loss:.4e}, final mean loss (last {_WINDOW} steps) = "
        f"{final_loss:.4e}, ratio = {final_loss / initial_loss:.3f}, "
        f"required < {_LOSS_DROP_RATIO}. "
        "The model is not learning — check the training-step chain "
        "(interpolant → velocity_gt → model forward → flow_matching_loss → "
        "backward → optim.step) and adaLN-zero initialisation."
    )

"""Layer-3 tests for the FID / IS wrappers.

These four tests match doc/Test.md §"tests/test_eval.py" verbatim:

* ``test_fid_zero_for_identical_distributions``      — FID(X, X) ≈ 0
* ``test_fid_positive_for_different_distributions``  — monotonicity in
  the mean-shift of a Gaussian
* ``test_inception_score_runs_and_is_finite``        — IS returns
  finite (mean, std) on synthetic tensors
* ``test_fid_ref_stat_cache_round_trip``             — save→load→FID
  reproduces the pre-save FID exactly

We use ``feature=64`` for FID throughout (torchmetrics' smallest
Inception layer) — the tests care about numerical *properties*, not
absolute FID values, and the smaller feature size keeps the CPU cost
of the InceptionV3 forward pass modest.

Inputs are synthesised as ``uint8`` in ``[0, 255]`` (i.e. valid image
data), then normalised into ``[-1, 1]`` to mirror the project's
canonical range and exercise the wrappers' ``_to_uint8`` conversion
path.  All tensors are ``299×299`` because that's the input size
InceptionV3 expects; passing anything else forces a resize inside
torchmetrics that only inflates the runtime.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from eval.fid import FIDMetric
from eval.inception_score import InceptionScoreMetric

# Inception-V3 native input size (torchmetrics resizes anything else).
IMG_SIZE = 299
# Small batch — tests measure numerical properties, not statistical
# power, so we don't need thousands of samples.
BATCH = 16
FEATURE_DIM_SMALL = 64  # smallest torchmetrics FID feature layer

def _uint8_batch(mean: float, std: float, n: int = BATCH, seed: int = 0) -> torch.Tensor:
    """Draw ``n`` synthetic ``uint8`` images from ``N(mean, std²)`` clamped
    to ``[0, 255]``, then rescale to ``[-1, 1]`` float32.

    The caller passes them straight into a wrapper's ``update_*`` /
    ``update`` method; the wrapper's ``_to_uint8`` will map them back
    to ``uint8`` before feeding torchmetrics.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, IMG_SIZE, IMG_SIZE, generator=g) * std + mean
    x = x.clamp(0.0, 255.0).round()          # valid image bytes
    x = (x / 127.5) - 1.0                    # -> [-1, 1]
    return x.float()

# --------------------------------------------------------------------- #
# Test 1 — FID(X, X) ≈ 0
# --------------------------------------------------------------------- #

def test_fid_zero_for_identical_distributions() -> None:
    """Two batches sampled from the *same* distribution should give
    an FID close to zero (small residual from finite-sample noise).

    We deliberately use two independent draws (not the same tensor
    twice) so the test also confirms that FID's zero-lower-bound
    behaviour is driven by distributional identity, not by literal
    tensor equality.
    """
    fid = FIDMetric(feature=FEATURE_DIM_SMALL, device="cpu")

    real = _uint8_batch(mean=127.5, std=40.0, seed=0)
    fake = _uint8_batch(mean=127.5, std=40.0, seed=1)

    fid.update_real(real)
    fid.update_fake(fake)
    value = fid.compute()

    assert torch.isfinite(torch.tensor(value)), f"FID not finite: {value}"
    # Finite-sample noise at BATCH=16 with feature=64 keeps the residual
    # small but non-zero; a comfortable upper bound is 20.  If a future
    # torchmetrics change causes this to jump above 20, that is a real
    # signal, not a flaky test.
    assert value < 20.0, (
        f"FID between two draws from the same distribution should be "
        f"small, got {value:.3f}"
    )

# --------------------------------------------------------------------- #
# Test 2 — FID monotonicity in mean-shift
# --------------------------------------------------------------------- #

def test_fid_positive_for_different_distributions() -> None:
    """FID(N(0,I), N(large,I)) > FID(N(0,I), N(small,I)).

    A larger mean-shift should produce a strictly larger FID than a
    smaller one when the reference is held fixed.  This is the
    minimum monotonicity property FID must have.
    """
    real = _uint8_batch(mean=127.5, std=40.0, seed=0)

    def _fid_against(fake_mean: float, seed: int) -> float:
        fid = FIDMetric(feature=FEATURE_DIM_SMALL, device="cpu")
        fid.update_real(real)
        fake = _uint8_batch(mean=fake_mean, std=40.0, seed=seed)
        fid.update_fake(fake)
        return fid.compute()

    fid_small_shift = _fid_against(fake_mean=127.5 + 5.0, seed=1)
    fid_large_shift = _fid_against(fake_mean=127.5 + 60.0, seed=2)

    assert fid_large_shift > fid_small_shift, (
        f"expected FID to grow with the mean-shift; got small_shift="
        f"{fid_small_shift:.3f}, large_shift={fid_large_shift:.3f}"
    )

# --------------------------------------------------------------------- #
# Test 3 — IS returns finite (mean, std)
# --------------------------------------------------------------------- #

def test_inception_score_runs_and_is_finite() -> None:
    """Inception Score on a synthetic ``(N, 3, 299, 299)`` batch
    returns a finite ``(mean, std)`` pair.

    We use ``splits=2`` here (not the default 10) purely because 10
    splits require ≥10 samples to give a well-defined per-split
    stddev, and a 10-image forward pass through InceptionV3 is not
    the point of this test — the point is that the wrapper's plumbing
    (device, uint8 conversion, compute unpack) works end-to-end.
    """
    is_metric = InceptionScoreMetric(splits=2, device="cpu")
    # Enough samples to give both splits some signal.
    imgs = _uint8_batch(mean=127.5, std=40.0, n=8, seed=0)
    is_metric.update(imgs)
    mean, std = is_metric.compute()

    assert torch.isfinite(torch.tensor(mean)), f"IS mean not finite: {mean}"
    assert torch.isfinite(torch.tensor(std)),  f"IS std not finite: {std}"
    # IS is defined as exp(KL) with KL ≥ 0, so the mean must be ≥ 1
    # up to numerical noise.
    assert mean >= 1.0 - 1e-3, f"IS mean below its theoretical floor: {mean}"

# --------------------------------------------------------------------- #
# Test 4 — reference-stat cache round-trip
# --------------------------------------------------------------------- #

def test_fid_ref_stat_cache_round_trip(tmp_path: Path) -> None:
    """Save the real-side statistics to ``.npz``, load them into a
    *fresh* :class:`FIDMetric`, feed the same fake batch, and check
    the FID matches the pre-save value bit-for-bit (float64 sums are
    lossless through numpy save/load).
    """
    real = _uint8_batch(mean=127.5, std=40.0, seed=0)
    fake = _uint8_batch(mean=127.5 + 20.0, std=40.0, seed=1)

    # First metric: build reference stats + compute a baseline FID.
    fid_a = FIDMetric(feature=FEATURE_DIM_SMALL, device="cpu")
    fid_a.update_real(real)
    fid_a.update_fake(fake)
    value_before = fid_a.compute()

    cache_path = tmp_path / "fid_ref_stats.npz"
    fid_a.save_reference(cache_path)
    assert cache_path.is_file(), "save_reference did not produce the .npz file"

    # Second metric: load the reference stats, feed the same fake
    # batch, and check the FID matches.
    fid_b = FIDMetric(feature=FEATURE_DIM_SMALL, device="cpu")
    fid_b.load_reference(cache_path)
    fid_b.update_fake(fake)
    value_after = fid_b.compute()

    # The three real-side sums are stored as float64 by torchmetrics
    # and round-trip losslessly through numpy; the fake side is
    # identical by construction; therefore the FIDs should match to
    # full float64 precision.  Use a tight-but-non-zero tolerance to
    # tolerate any last-bit noise from the .item() cast.
    assert value_before == pytest.approx(value_after, abs=1e-9), (
        f"FID before save ({value_before}) != after load ({value_after})"
    )

# --------------------------------------------------------------------- #
# Extra guardrail — save with no real samples fed must fail loudly
# --------------------------------------------------------------------- #

def test_fid_save_reference_rejects_empty_cache(tmp_path: Path) -> None:
    """Saving before any ``update_real`` call would produce a useless
    zero-sample cache; the wrapper must refuse rather than write it.
    """
    fid = FIDMetric(feature=FEATURE_DIM_SMALL, device="cpu")
    with pytest.raises(RuntimeError, match="no real samples"):
        fid.save_reference(tmp_path / "would_be_empty.npz")

"""Layer-3 test for the CIFAR-10 dataset adapter.

The three tests here match doc/Test.md §"tests/test_dataset.py" verbatim.
This is the *only* test module that reads real image bytes off disk;
every other test uses synthetic tensors.

The fixture PNGs live in ``tests/fixtures/images/class_0{0,1,2,3}/`` —
four solid-colour 32x32 PNGs produced deterministically by
``tests/fixtures/images/make_fixtures.py``.  See doc/Test.md §"How
dataset fixtures work" for the rationale.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from data.cifar10 import CIFAR10Dataset


FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
FIXTURES_SPLIT = "images"                 # -> tests/fixtures/images/<class_*>/*.png
NUM_FIXTURE_IMAGES = 4                    # class_00..class_03, one PNG each


@pytest.fixture(scope="module")
def dataset() -> CIFAR10Dataset:
    """Instantiate the dataset over the committed fixture PNGs."""
    return CIFAR10Dataset(root=FIXTURES_ROOT, split=FIXTURES_SPLIT, augment=False)


# ---------------------------------------------------------------------------
# Test 1 — output tensor contract (shape, dtype, range)
# ---------------------------------------------------------------------------

def test_dataset_returns_correct_shape_and_range(dataset: CIFAR10Dataset) -> None:
    """Every item is a ``float32`` tensor of shape ``(3, 32, 32)`` with
    values in ``[-1, 1]`` — the interval assumed by FlowMatching.md §1."""
    for i in range(len(dataset)):
        img, _ = dataset[i]
        assert isinstance(img, torch.Tensor), f"item {i}: not a tensor"
        assert img.shape == (3, 32, 32), f"item {i}: shape {tuple(img.shape)}"
        assert img.dtype == torch.float32, f"item {i}: dtype {img.dtype}"
        assert torch.isfinite(img).all(), f"item {i}: non-finite values"
        assert img.min() >= -1.0 - 1e-6, f"item {i}: min {img.min().item()}"
        assert img.max() <= 1.0 + 1e-6, f"item {i}: max {img.max().item()}"


# ---------------------------------------------------------------------------
# Test 2 — label contract (dtype, range)
# ---------------------------------------------------------------------------

def test_dataset_label_type(dataset: CIFAR10Dataset) -> None:
    """Labels are 0-D ``int64`` tensors in ``[0, num_classes)``."""
    assert dataset.num_classes == NUM_FIXTURE_IMAGES, (
        f"expected {NUM_FIXTURE_IMAGES} class-directories in the fixture, "
        f"got {dataset.num_classes}"
    )
    seen: set[int] = set()
    for i in range(len(dataset)):
        _, label = dataset[i]
        assert isinstance(label, torch.Tensor), f"item {i}: label is not a tensor"
        assert label.dtype == torch.long, f"item {i}: label dtype {label.dtype}"
        assert label.ndim == 0, f"item {i}: label ndim {label.ndim}"
        val = int(label.item())
        assert 0 <= val < dataset.num_classes, (
            f"item {i}: label {val} out of range [0, {dataset.num_classes})"
        )
        seen.add(val)
    # Since we have exactly one PNG per class-directory, every label must
    # appear exactly once — this also pins down ImageFolder's alphabetical
    # class-name → integer-label mapping.
    assert seen == set(range(NUM_FIXTURE_IMAGES)), (
        f"expected labels {set(range(NUM_FIXTURE_IMAGES))}, saw {seen}"
    )


# ---------------------------------------------------------------------------
# Test 3 — length matches the number of on-disk files
# ---------------------------------------------------------------------------

def test_dataset_length_matches_files(dataset: CIFAR10Dataset) -> None:
    """``len(dataset)`` and iteration both agree with the on-disk file
    count.  There is exactly one PNG per class-directory, so the total
    is :data:`NUM_FIXTURE_IMAGES`."""
    on_disk = sorted(
        (FIXTURES_ROOT / FIXTURES_SPLIT).glob("*/*.png")
    )
    assert len(on_disk) == NUM_FIXTURE_IMAGES, (
        f"fixture layout drifted: found {len(on_disk)} PNGs on disk, "
        f"expected {NUM_FIXTURE_IMAGES}"
    )
    assert len(dataset) == NUM_FIXTURE_IMAGES

    iterated = list(dataset)
    assert len(iterated) == NUM_FIXTURE_IMAGES


# ---------------------------------------------------------------------------
# Extra: constructor rejects a missing split directory (guardrail)
# ---------------------------------------------------------------------------

def test_dataset_missing_split_raises(tmp_path: Path) -> None:
    """A non-existent ``root/split`` fails loudly rather than silently
    returning an empty dataset."""
    with pytest.raises(FileNotFoundError):
        CIFAR10Dataset(root=tmp_path, split="does_not_exist", augment=False)

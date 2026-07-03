"""CIFAR-10 dataset adapter for flow-matching training and evaluation.

The class :class:`CIFAR10Dataset` is a thin wrapper around
:class:`torchvision.datasets.ImageFolder` that adds

1. a fixed transform pipeline that emits ``float32`` tensors of shape
   ``(3, 32, 32)`` in the range ``[-1, 1]`` (matching the convention
   assumed everywhere else in this repository — see FlowMatching.md §1
   for why the training data must live in the same interval as the
   noise ``x_0 ~ N(0, I)`` gets pushed toward);
2. an optional, minimal augmentation pipeline consisting of a single
   :class:`torchvision.transforms.RandomHorizontalFlip` at ``p=0.5``.

On-disk layout
--------------
::

    <root>/
        <split>/                       # e.g. "train", "val"
            <class_name_00>/*.png      # class name → integer label 0
            <class_name_01>/*.png      # class name → integer label 1
            ...

The integer label is assigned by :class:`ImageFolder`'s alphabetical
sort of class-directory names — same behaviour we exercise in the unit
tests via the four fixture PNGs under ``tests/fixtures/images/``.

Design notes
------------
* **``augment`` defaults to ``False``.**  Deterministic behaviour is
  the safe default: validation loaders, FID reference-statistics
  computation, and unit tests all want it off.  Training configs set
  ``dataset.augment: true`` explicitly (see
  ``configs/train/cifar10_train.yaml``).
* **Augmentation is horizontal flip only.**  CIFAR-10 classes
  (airplane, automobile, ..., truck) are all left–right symmetric, so
  a mirrored image is still a valid sample from the same class.  We
  deliberately avoid ``RandomCrop(32, padding=4)`` because it would
  make the training distribution include images with black borders,
  which the flow-matching model would then reproduce.
* **No download side-effect.**  The training config sets
  ``dataset.download: false`` and this class never downloads anything;
  if ``root/split`` does not exist the underlying ``ImageFolder`` will
  raise loudly.  Datasets are staged out-of-band; see doc/Train.md §2.2.
* **Normalisation is fused into the transform.**  ``ToTensor()``
  already produces ``float32`` in ``[0, 1]``; a subsequent
  ``Normalize(mean=0.5, std=0.5)`` shifts that to ``[-1, 1]`` in a
  single fused op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CIFAR10Dataset(Dataset):
    """CIFAR-10 stored on disk as class-subdirectories of PNG files.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory that contains one subdirectory per split.
    split : str, default ``"train"``
        Sub-directory name under ``root`` — typically ``"train"`` or
        ``"val"``.
    augment : bool, default ``False``
        If ``True``, prepend a ``RandomHorizontalFlip(p=0.5)`` to the
        transform pipeline.  Off by default so that validation, FID
        reference-stat computation and unit tests behave
        deterministically without any extra plumbing.

    Attributes
    ----------
    root : pathlib.Path
        The original root passed by the caller (not resolved).
    split : str
        The split name (e.g. ``"train"``).
    num_classes : int
        Number of class-directories found under ``root/split``.  For
        real CIFAR-10 this is ``10``; for the ``tests/fixtures/images``
        fixture it is ``4``.
    classes : list[str]
        Class-directory names, sorted alphabetically (i.e. index in
        this list == integer label).  Passed through from
        ``ImageFolder``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.augment = augment

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"CIFAR10Dataset: split directory does not exist: {split_dir}"
            )

        transform_list: list = []
        if augment:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.extend(
            [
                transforms.ToTensor(),                       # uint8 [0,255] -> float32 [0,1]
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # -> float32 [-1,1]
                                     std=(0.5, 0.5, 0.5)),
            ]
        )
        transform = transforms.Compose(transform_list)

        # ImageFolder does the class-directory scan and label mapping.
        self._backend = ImageFolder(root=str(split_dir), transform=transform)

        self.classes: list[str] = list(self._backend.classes)
        self.num_classes: int = len(self.classes)

    def __len__(self) -> int:
        return len(self._backend)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img, label = self._backend[index]
        # ImageFolder yields a Python int; convert to a 0-D int64 tensor so
        # that the collated batch has dtype torch.long without extra work
        # from the training loop.
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img, label_tensor

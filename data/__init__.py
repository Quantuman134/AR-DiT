"""Dataset adapters for training and evaluation.

Currently only CIFAR-10 is wired up (see :mod:`data.cifar10`).  Additional
datasets can live alongside it and be dispatched by the ``dataset.name``
field of the training/sampling configs.
"""

from data.cifar10 import CIFAR10Dataset

__all__ = ["CIFAR10Dataset"]

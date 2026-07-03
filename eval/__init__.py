"""Evaluation metrics for generative models.

Thin wrappers over :mod:`torchmetrics.image` — see doc/Train.md §8 for
the design rationale (why we wrap rather than reimplement) and §8.1 for
the reference-statistics cache format used by
:class:`eval.fid.FIDMetric`.
"""

from eval.fid import FIDMetric
from eval.inception_score import InceptionScoreMetric

__all__ = ["FIDMetric", "InceptionScoreMetric"]

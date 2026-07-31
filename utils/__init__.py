"""Cross-cutting training/eval utilities.

Each submodule is intentionally small, dependency-light, and unit-testable
in isolation.  Anything with a runtime side-effect (I/O, wandb, DDP setup)
should live in ``runtime/`` instead; ``utils/`` is for pure functions and
lightweight measurement helpers.
"""

from utils.grad_norm import (
    GRAD_NORM_GRANULARITIES,
    GROUP_PATTERNS,
    compute_grad_norm_report,
)

__all__ = [
    "GRAD_NORM_GRANULARITIES",
    "GROUP_PATTERNS",
    "compute_grad_norm_report",
]

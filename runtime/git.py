"""Shared git-metadata helper used by both entry points.

Extracted from ``train.py`` / ``sample.py`` so both files stamp their
outputs with the same commit id via one implementation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_sha() -> str | None:
    """Return the current repo HEAD SHA, or ``None`` if unavailable.

    Returns ``None`` on any failure (not a git checkout, ``git`` not on
    PATH, detached-worktree edge cases, …) — callers stamp this into
    ``results.json`` / checkpoints purely for provenance and must
    tolerate its absence.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip() or None
    except Exception:
        return None

"""Pytest configuration: deterministic seeding before every test.

Each test gets a fresh ``torch.manual_seed(0)`` so that tests which build a
DiT (or any other module with random init) see the same parameters every
run. Tests that need a different seed simply call ``torch.manual_seed``
inside their own body; the fixture only runs *before* the test, never
*after* it.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_all() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

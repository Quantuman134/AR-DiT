"""Models for the Attention-Residual-for-DiT project."""

from __future__ import annotations

import torch.nn as nn

from configs import ConfigError
from configs.schema import ModelConfig

from .dit import (
    DiT,
    DiT_S_2,
    DiT_B_2,
    DiT_L_2,
    DiT_XL_2,
)
from .ar_dit import (
    ARDiT,
    ARDiT_S_2,
    ARDiT_B_2,
    ARDiT_L_2,
    ARDiT_XL_2,
)

# ---------------------------------------------------------------------------
# Arch registry + config-driven factory
# ---------------------------------------------------------------------------
# Single source of truth for ``arch_name -> constructor``.  Both entry
# points (``train.py`` and ``sample.py``) go through
# :func:`build_model_from_config` so a config-shape change (new arch,
# renamed field) has exactly one place to update.

_ARCH_PRESETS = {
    "DiT_S_2":    DiT_S_2,
    "DiT_B_2":    DiT_B_2,
    "DiT_L_2":    DiT_L_2,
    "DiT_XL_2":   DiT_XL_2,
    "ARDiT_S_2":  ARDiT_S_2,
    "ARDiT_B_2":  ARDiT_B_2,
    "ARDiT_L_2":  ARDiT_L_2,
    "ARDiT_XL_2": ARDiT_XL_2,
}


def build_model_from_config(model_cfg: ModelConfig) -> nn.Module:
    """Instantiate a DiT from a validated :class:`ModelConfig`.

    Raises :class:`ConfigError` if ``arch_name`` is unknown so the failure
    mode matches the rest of config-time validation (a single, uniform
    error type callers can catch).
    """
    if model_cfg.arch_name not in _ARCH_PRESETS:
        raise ConfigError(
            f"model.arch_name={model_cfg.arch_name!r} not in "
            f"{sorted(_ARCH_PRESETS)}"
        )
    factory = _ARCH_PRESETS[model_cfg.arch_name]
    return factory(
        input_size=model_cfg.input_size,
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        num_classes=model_cfg.num_classes,
        class_dropout_prob=model_cfg.class_dropout_prob,
    )


__all__ = [
    "DiT",
    "DiT_S_2",
    "DiT_B_2",
    "DiT_L_2",
    "DiT_XL_2",
    "ARDiT",
    "ARDiT_S_2",
    "ARDiT_B_2",
    "ARDiT_L_2",
    "ARDiT_XL_2",
    "build_model_from_config",
]

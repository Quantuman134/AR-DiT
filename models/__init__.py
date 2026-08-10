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
    ARDiTCond,
    ARDiTCond_S_2,
    ARDiTCond_B_2,
    ARDiTCond_L_2,
    ARDiTCond_XL_2,
    ARDiTCondSANA,
    ARDiTCondSANA_S_2,
    ARDiTCondSANA_B_2,
    ARDiTCondSANA_L_2,
    ARDiTCondSANA_XL_2,
)

# ---------------------------------------------------------------------------
# Arch registry + config-driven factory
# ---------------------------------------------------------------------------
# Single source of truth for ``arch_name -> constructor``.  Both entry
# points (``train.py`` and ``sample.py``) go through
# :func:`build_model_from_config` so a config-shape change (new arch,
# renamed field) has exactly one place to update.

_ARCH_PRESETS = {
    "DiT_S_2":            DiT_S_2,
    "DiT_B_2":            DiT_B_2,
    "DiT_L_2":            DiT_L_2,
    "DiT_XL_2":           DiT_XL_2,
    "ARDiT_S_2":          ARDiT_S_2,
    "ARDiT_B_2":          ARDiT_B_2,
    "ARDiT_L_2":          ARDiT_L_2,
    "ARDiT_XL_2":         ARDiT_XL_2,
    "ARDiTCond_S_2":      ARDiTCond_S_2,
    "ARDiTCond_B_2":      ARDiTCond_B_2,
    "ARDiTCond_L_2":      ARDiTCond_L_2,
    "ARDiTCond_XL_2":     ARDiTCond_XL_2,
    "ARDiTCondSANA_S_2":  ARDiTCondSANA_S_2,
    "ARDiTCondSANA_B_2":  ARDiTCondSANA_B_2,
    "ARDiTCondSANA_L_2":  ARDiTCondSANA_L_2,
    "ARDiTCondSANA_XL_2": ARDiTCondSANA_XL_2,
}

# Which architectures consume the ``num_time_bins`` field of
# :class:`ModelConfig`.  Kept explicit rather than magic so a future arch
# opting into E2's discrete time codebook must announce itself here.
_ARCHES_WITH_TIME_BINS: frozenset[str] = frozenset({
    "ARDiTCondSANA_S_2",
    "ARDiTCondSANA_B_2",
    "ARDiTCondSANA_L_2",
    "ARDiTCondSANA_XL_2",
})


def build_model_from_config(model_cfg: ModelConfig) -> nn.Module:
    """Instantiate a DiT from a validated :class:`ModelConfig`.

    Raises :class:`ConfigError` if ``arch_name`` is unknown so the failure
    mode matches the rest of config-time validation (a single, uniform
    error type callers can catch).

    ``num_time_bins`` is forwarded only to architectures in
    :data:`_ARCHES_WITH_TIME_BINS` (the ``ARDiTCondSANA_*`` presets).
    Other architectures do not accept the kwarg; forwarding it
    unconditionally would produce a :class:`TypeError` from their
    constructors.
    """
    if model_cfg.arch_name not in _ARCH_PRESETS:
        raise ConfigError(
            f"model.arch_name={model_cfg.arch_name!r} not in "
            f"{sorted(_ARCH_PRESETS)}"
        )
    factory = _ARCH_PRESETS[model_cfg.arch_name]
    kwargs = dict(
        input_size=model_cfg.input_size,
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        num_classes=model_cfg.num_classes,
        class_dropout_prob=model_cfg.class_dropout_prob,
    )
    if model_cfg.arch_name in _ARCHES_WITH_TIME_BINS:
        kwargs["num_time_bins"] = model_cfg.num_time_bins
    return factory(**kwargs)


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
    "ARDiTCond",
    "ARDiTCond_S_2",
    "ARDiTCond_B_2",
    "ARDiTCond_L_2",
    "ARDiTCond_XL_2",
    "ARDiTCondSANA",
    "ARDiTCondSANA_S_2",
    "ARDiTCondSANA_B_2",
    "ARDiTCondSANA_L_2",
    "ARDiTCondSANA_XL_2",
    "build_model_from_config",
]

"""Config-schema package.

Only ``schema.py`` is a real Python module here — the sibling
directories (``model/``, ``train/``, ``sample/``) hold YAML files, not
Python code.  Re-export the public loaders for a friendly import path::

    from configs import load_train_config, load_sample_config
"""

from configs.schema import (
    ConfigError,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    UnknownKeyError,
    apply_overrides,
    load_model_config,
    load_sample_config,
    load_train_config,
)

__all__ = [
    "ConfigError",
    "UnknownKeyError",
    "ModelConfig",
    "TrainConfig",
    "SampleConfig",
    "load_model_config",
    "load_train_config",
    "load_sample_config",
    "apply_overrides",
]

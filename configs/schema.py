"""Config schema and loaders for training / sampling.

Design principles (see doc/Train.md §2):

*   YAML is the source-of-truth on disk (human-friendly, comments allowed).
*   Every loaded YAML is validated against a strict `@dataclass` schema at
    startup.  Unknown keys are a **hard error**: silent typos are the worst
    kind of bug for ML configs.
*   Enum-like fields (``optim.name``, ``lr_schedule``, ``amp_dtype``,
    ``wandb.mode``) are validated against an explicit allowlist.
*   The train-config points at a model-config by relative path; the loader
    resolves it and returns a ``TrainConfig`` with an embedded
    ``ModelConfig``.
*   The sample-config does *not* embed a model-config: the sampler reads
    the model config from the checkpoint (frozen at training time), which
    makes it impossible for train and eval to disagree on architecture.

Public API
----------
``load_model_config(path)``      -> ``ModelConfig``
``load_train_config(path)``      -> ``TrainConfig``   (with model embedded)
``load_sample_config(path)``     -> ``SampleConfig``
``apply_overrides(d, overrides)``-> ``dict``          (mutates & returns)

The overrides helper is intended to be called on the *raw* loaded dict
before it is passed to a schema loader, so a CLI flag such as
``--override optim.lr=2.0e-4`` is validated by the schema just like a
YAML value would be.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Base class for all config-validation errors raised by this module."""


class UnknownKeyError(ConfigError):
    """Raised when a YAML contains a key not declared in the schema."""


# ---------------------------------------------------------------------------
# Enum-like allowlists
# ---------------------------------------------------------------------------

_OPTIM_NAMES = ("adamw",)
_LR_SCHEDULES = ("constant", "cosine")
_AMP_DTYPES = ("fp32", "bf16")
_WANDB_MODES = ("online", "offline", "disabled")
_DATASET_NAMES = ("cifar10",)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    arch_name: str                  # e.g. "DiT_S_2"
    input_size: int                 # H = W
    in_channels: int                # 3 for CIFAR-10 pixel space
    patch_size: int                 # PatchEmbed kernel/stride
    num_classes: int                # e.g. 10 for CIFAR-10
    class_dropout_prob: float       # CFG label-dropout probability

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ConfigError(f"model.input_size must be positive, got {self.input_size}")
        if self.in_channels <= 0:
            raise ConfigError(f"model.in_channels must be positive, got {self.in_channels}")
        if self.patch_size <= 0:
            raise ConfigError(f"model.patch_size must be positive, got {self.patch_size}")
        if self.input_size % self.patch_size != 0:
            raise ConfigError(
                f"model.input_size ({self.input_size}) must be divisible by "
                f"model.patch_size ({self.patch_size})"
            )
        if self.num_classes <= 0:
            raise ConfigError(f"model.num_classes must be positive, got {self.num_classes}")
        if not 0.0 <= self.class_dropout_prob <= 1.0:
            raise ConfigError(
                f"model.class_dropout_prob must be in [0, 1], got {self.class_dropout_prob}"
            )


# ---------------------------------------------------------------------------
# TrainConfig — nested dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    download: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = True

    def __post_init__(self) -> None:
        if self.name not in _DATASET_NAMES:
            raise ConfigError(
                f"dataset.name={self.name!r} not in allowed set {_DATASET_NAMES}"
            )
        if self.num_workers < 0:
            raise ConfigError(f"dataset.num_workers must be >= 0, got {self.num_workers}")


@dataclass(frozen=True)
class OptimConfig:
    name: str
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float | None
    warmup_steps: int
    lr_schedule: str

    def __post_init__(self) -> None:
        if self.name not in _OPTIM_NAMES:
            raise ConfigError(
                f"optim.name={self.name!r} not in allowed set {_OPTIM_NAMES}"
            )
        if self.lr <= 0:
            raise ConfigError(f"optim.lr must be positive, got {self.lr}")
        if len(self.betas) != 2:
            raise ConfigError(f"optim.betas must be a length-2 sequence, got {self.betas}")
        if self.weight_decay < 0:
            raise ConfigError(f"optim.weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ConfigError(
                f"optim.grad_clip must be positive or null, got {self.grad_clip}"
            )
        if self.warmup_steps < 0:
            raise ConfigError(f"optim.warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.lr_schedule not in _LR_SCHEDULES:
            raise ConfigError(
                f"optim.lr_schedule={self.lr_schedule!r} not in {_LR_SCHEDULES}"
            )


@dataclass(frozen=True)
class TrainLoopConfig:
    total_steps: int
    batch_size_per_gpu: int
    grad_accum_steps: int = 1
    log_interval: int = 50
    ckpt_interval: int = 10000
    seed: int = 0
    amp_dtype: str = "bf16"

    def __post_init__(self) -> None:
        for name in ("total_steps", "batch_size_per_gpu", "grad_accum_steps",
                     "log_interval", "ckpt_interval"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"train.{name} must be positive, got {getattr(self, name)}")
        if self.amp_dtype not in _AMP_DTYPES:
            raise ConfigError(
                f"train.amp_dtype={self.amp_dtype!r} not in {_AMP_DTYPES}"
            )


@dataclass(frozen=True)
class EMAConfig:
    decays: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.decays) == 0:
            raise ConfigError("ema.decays must contain at least one value")
        for d in self.decays:
            if not 0.0 < d < 1.0:
                raise ConfigError(f"ema.decays entries must be in (0, 1), got {d}")


@dataclass(frozen=True)
class GuidanceTrainConfig:
    # class_dropout_prob lives in ModelConfig (owned by LabelEmbedder).
    # null_class_id may be null, meaning "auto-derive as num_classes".
    #
    # Named ``GuidanceTrainConfig`` (not ``CFGTrainConfig``) so the
    # letters "cfg" stay reserved for the *config* dataclass everywhere
    # else in the codebase.
    null_class_id: int | None = None


@dataclass(frozen=True)
class ValidationSamplerConfig:
    num_steps: int = 50

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ConfigError(
                f"validation.sampler.num_steps must be positive, got {self.num_steps}"
            )


@dataclass(frozen=True)
class MetricsConfig:
    fid: bool = True
    inception_score: bool = True


@dataclass(frozen=True)
class ValidationConfig:
    interval: int
    num_samples: int
    batch_size_per_gpu: int             # per-chunk generation size — bounds VRAM
    visual_log_count: int
    guidance_scales: tuple[float, ...]
    metrics: MetricsConfig
    sampler: ValidationSamplerConfig
    fid_ref_stats: str | None = None    # null ⇒ compute on first run

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ConfigError(f"validation.interval must be positive, got {self.interval}")
        if self.num_samples <= 0:
            raise ConfigError(
                f"validation.num_samples must be positive, got {self.num_samples}"
            )
        if self.batch_size_per_gpu <= 0:
            raise ConfigError(
                f"validation.batch_size_per_gpu must be positive, got "
                f"{self.batch_size_per_gpu}"
            )
        if not 0 <= self.visual_log_count <= self.num_samples:
            raise ConfigError(
                f"validation.visual_log_count ({self.visual_log_count}) must be in "
                f"[0, num_samples={self.num_samples}]"
            )
        if len(self.guidance_scales) == 0:
            raise ConfigError("validation.guidance_scales must contain at least one value")
        for s in self.guidance_scales:
            if s < 0:
                raise ConfigError(f"validation.guidance_scales entries must be >= 0, got {s}")


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    project: str = ""
    entity: str | None = None
    token_path: str = "secrets/wandb.token"
    mode: str = "online"

    def __post_init__(self) -> None:
        if self.mode not in _WANDB_MODES:
            raise ConfigError(f"logging.wandb.mode={self.mode!r} not in {_WANDB_MODES}")
        if self.enabled and not self.project:
            raise ConfigError("logging.wandb.project must be set when wandb.enabled=true")


@dataclass(frozen=True)
class LoggingConfig:
    out_dir: str
    run_name: str
    wandb: WandbConfig


@dataclass(frozen=True)
class TrainConfig:
    """Top-level training config (with the referenced ModelConfig embedded)."""

    # Path (as originally written in the YAML), resolved and loaded into `model` below.
    model_config: str
    dataset: DatasetConfig
    optim: OptimConfig
    train: TrainLoopConfig
    ema: EMAConfig
    guidance: GuidanceTrainConfig
    validation: ValidationConfig
    logging: LoggingConfig
    # Populated by load_train_config; not present in the YAML itself.
    model: ModelConfig | None = None


# ---------------------------------------------------------------------------
# SampleConfig — nested dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SamplingConfig:
    num_samples: int
    batch_size_per_gpu: int
    num_steps: int
    guidance_scale: float
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("num_samples", "batch_size_per_gpu", "num_steps"):
            if getattr(self, name) <= 0:
                raise ConfigError(
                    f"sampling.{name} must be positive, got {getattr(self, name)}"
                )
        if self.guidance_scale < 0:
            raise ConfigError(f"sampling.guidance_scale must be >= 0, got {self.guidance_scale}")


@dataclass(frozen=True)
class SampleDatasetConfig:
    name: str
    root: str
    download: bool = False

    def __post_init__(self) -> None:
        if self.name not in _DATASET_NAMES:
            raise ConfigError(
                f"dataset.name={self.name!r} not in allowed set {_DATASET_NAMES}"
            )


@dataclass(frozen=True)
class SampleMetricsConfig:
    fid: bool = True
    inception_score: bool = True
    fid_ref_stats: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    dir: str
    save_images: bool = False
    save_grid: bool = True


@dataclass(frozen=True)
class SampleConfig:
    """Top-level sampling / evaluation config."""
    ckpt_path: str
    ema_tag: str
    sampling: SamplingConfig
    dataset: SampleDatasetConfig
    metrics: SampleMetricsConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        if not (self.ema_tag == "online" or self.ema_tag.startswith("ema_")):
            raise ConfigError(
                f"ema_tag must be 'online' or 'ema_<decay>', got {self.ema_tag!r}"
            )


# ---------------------------------------------------------------------------
# Strict dict -> dataclass conversion
# ---------------------------------------------------------------------------

def _is_optional(tp: Any) -> bool:
    """Return True if ``tp`` is ``T | None`` / ``Optional[T]``."""
    if get_origin(tp) is Union:
        return type(None) in get_args(tp)
    return False


def _non_none_type(tp: Any) -> Any:
    """For ``T | None`` return ``T``; otherwise return ``tp`` unchanged."""
    if get_origin(tp) is Union:
        args = tuple(a for a in get_args(tp) if a is not type(None))
        if len(args) == 1:
            return args[0]
    return tp


def _from_dict_strict(cls: type, data: Any, *, path: str) -> Any:
    """Recursively convert ``data`` into an instance of dataclass ``cls``.

    Raises ``UnknownKeyError`` on any key in ``data`` not declared on ``cls``.
    Raises ``ConfigError`` on any missing required field.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")  # pragma: no cover
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path}: expected mapping for {cls.__name__}, got {type(data).__name__}"
        )

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}

    unknown = set(data.keys()) - known
    if unknown:
        raise UnknownKeyError(
            f"{path}: unknown key(s) for {cls.__name__}: "
            f"{sorted(unknown)}. Allowed: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        subpath = f"{path}.{f.name}" if path else f.name
        if f.name in data:
            raw = data[f.name]
            kwargs[f.name] = _coerce(hints[f.name], raw, path=subpath)
        else:
            if (f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING):  # type: ignore[misc]
                raise ConfigError(
                    f"{path or cls.__name__}: missing required field {f.name!r}"
                )
            # else: dataclass default will be used
    return cls(**kwargs)


def _coerce(tp: Any, raw: Any, *, path: str) -> Any:
    """Coerce a raw YAML value to the type declared on the dataclass field."""
    # Optional[T] with a null value -> None.
    if _is_optional(tp) and raw is None:
        return None
    inner = _non_none_type(tp)

    # Nested dataclass.
    if is_dataclass(inner):
        return _from_dict_strict(inner, raw, path=path)

    # Tuple / list types: honour declared element type where possible.
    origin = get_origin(inner)
    if origin in (tuple,):
        args = get_args(inner)
        if not isinstance(raw, (list, tuple)):
            raise ConfigError(
                f"{path}: expected list/tuple, got {type(raw).__name__}"
            )
        # tuple[X, ...] -> homogeneous, any length
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, path=f"{path}[{i}]")
                         for i, v in enumerate(raw))
        # tuple[A, B, ...] -> fixed length
        if len(raw) != len(args):
            raise ConfigError(
                f"{path}: expected tuple of length {len(args)}, got {len(raw)}"
            )
        return tuple(_coerce(a, v, path=f"{path}[{i}]")
                     for i, (a, v) in enumerate(zip(args, raw)))
    if origin in (list,):
        (elem_tp,) = get_args(inner) or (Any,)
        if not isinstance(raw, list):
            raise ConfigError(f"{path}: expected list, got {type(raw).__name__}")
        return [_coerce(elem_tp, v, path=f"{path}[{i}]") for i, v in enumerate(raw)]

    # Bare scalars: leave YAML's parsing as-is (yaml already gives ints/floats/bools/str).
    # We only coerce int -> float where the schema says float, since PyYAML
    # will happily hand back an int for things like `weight_decay: 0`.
    if inner is float and isinstance(raw, int) and not isinstance(raw, bool):
        return float(raw)
    return raw


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    """Load and validate a model config."""
    data = _load_yaml(path)
    return _from_dict_strict(ModelConfig, data, path="model")


def load_train_config(path: str | Path) -> TrainConfig:
    """Load and validate a train config, resolving and embedding the referenced
    model config.

    ``model_config`` in the YAML is a path relative to the train-config file's
    directory (see doc/Train.md §2.3).
    """
    train_yaml_path = Path(path).resolve()
    data = _load_yaml(train_yaml_path)

    # Resolve the model_config path *before* schema validation so an obvious
    # relative-path typo produces a clear, contextual error.
    if "model_config" not in data:
        raise ConfigError(f"{train_yaml_path}: missing required field 'model_config'")
    model_path_raw = data["model_config"]
    if not isinstance(model_path_raw, str):
        raise ConfigError(
            f"{train_yaml_path}: 'model_config' must be a string path, got "
            f"{type(model_path_raw).__name__}"
        )
    model_path = (train_yaml_path.parent / model_path_raw).resolve()
    model_cfg = load_model_config(model_path)

    cfg = _from_dict_strict(TrainConfig, {**data, "model": None}, path="")
    # dataclass is frozen; use object.__setattr__ to attach the resolved model.
    object.__setattr__(cfg, "model", model_cfg)
    return cfg


def load_sample_config(path: str | Path) -> SampleConfig:
    """Load and validate a sample / evaluation config.

    Note: the sampler reads the model architecture from the checkpoint, not
    from a model-config file, so ``SampleConfig`` deliberately has no
    ``model_config`` field.  See doc/Train.md §2.1.
    """
    data = _load_yaml(path)
    return _from_dict_strict(SampleConfig, data, path="")


# ---------------------------------------------------------------------------
# CLI overrides — apply to raw dict *before* schema validation
# ---------------------------------------------------------------------------

def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    """Apply a list of ``dotted.key=value`` overrides to a raw config dict.

    Values are parsed via ``yaml.safe_load`` so ``true``/``1.5``/``[1, 2]``
    all parse the same way they would in a YAML file.  The dict is mutated
    **and** returned.  Overriding a key that does not already exist raises
    ``ConfigError`` — this catches typos in CLI flags with the same
    strictness that the schema catches typos in YAML.

    Example
    -------
    >>> apply_overrides(cfg_dict, ["optim.lr=2.0e-4", "train.seed=42"])
    """
    if not overrides:
        return data
    for spec in overrides:
        if "=" not in spec:
            raise ConfigError(f"override {spec!r} is not of the form key.path=value")
        dotted, raw_val = spec.split("=", 1)
        keys = dotted.strip().split(".")
        if not keys or any(not k for k in keys):
            raise ConfigError(f"override {spec!r} has an empty key segment")
        try:
            value = yaml.safe_load(raw_val)
        except yaml.YAMLError as e:
            raise ConfigError(f"override {spec!r}: value is not valid YAML: {e}") from e

        node: Any = data
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ConfigError(
                    f"override {spec!r}: key path does not exist in config"
                )
            node = node[k]
        last = keys[-1]
        if not isinstance(node, dict) or last not in node:
            raise ConfigError(
                f"override {spec!r}: key path does not exist in config"
            )
        node[last] = value
    return data


__all__ = [
    "ConfigError",
    "UnknownKeyError",
    "ModelConfig",
    "DatasetConfig",
    "OptimConfig",
    "TrainLoopConfig",
    "EMAConfig",
    "GuidanceTrainConfig",
    "ValidationSamplerConfig",
    "MetricsConfig",
    "ValidationConfig",
    "WandbConfig",
    "LoggingConfig",
    "TrainConfig",
    "SamplingConfig",
    "SampleDatasetConfig",
    "SampleMetricsConfig",
    "OutputConfig",
    "SampleConfig",
    "load_model_config",
    "load_train_config",
    "load_sample_config",
    "apply_overrides",
]

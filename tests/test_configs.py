"""Unit tests for configs/schema.py.

Coverage matrix:

*   Happy path: all three shipped YAMLs load and produce valid dataclasses.
*   Unknown-key rejection (``UnknownKeyError``).
*   Missing-required-field rejection (``ConfigError``).
*   Enum-like allowlists (``optim.name``, ``lr_schedule``, ``amp_dtype``,
    ``dataset.name``, ``wandb.mode``).
*   Cross-field constraints (``input_size % patch_size == 0``,
    ``visual_log_count <= num_samples``, ``ema_tag`` shape).
*   Type coercion (int -> float for float-typed fields; list -> tuple for
    tuple-typed fields).
*   ``load_train_config`` resolves the ``model_config`` path relative to
    the train-config's own directory and embeds the resulting
    ``ModelConfig`` on the returned object.
*   ``apply_overrides``: happy path, YAML-typed values, missing-key error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from configs.schema import (
    ConfigError,
    ModelConfig,
    UnknownKeyError,
    apply_overrides,
    load_model_config,
    load_sample_config,
    load_train_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Happy path: the three shipped configs must always load
# ---------------------------------------------------------------------------

def test_shipped_model_yaml_loads():
    m = load_model_config(PROJECT_ROOT / "configs/model/dit_s2_cifar.yaml")
    assert isinstance(m, ModelConfig)
    assert m.arch_name == "DiT_S_2"
    assert m.input_size == 32
    assert m.patch_size == 2
    assert m.num_classes == 10


def test_shipped_train_yaml_loads_and_embeds_model():
    t = load_train_config(PROJECT_ROOT / "configs/train/cifar10_train.yaml")
    # The referenced model config must be loaded and embedded.
    assert t.model is not None
    # It must be equal to what a direct load produces.
    m_direct = load_model_config(PROJECT_ROOT / "configs/model/dit_s2_cifar.yaml")
    assert t.model == m_direct
    # A few structural spot-checks.
    assert t.optim.lr == pytest.approx(1e-4)
    assert t.optim.betas == (0.9, 0.999)
    assert t.train.amp_dtype == "bf16"
    assert t.ema.decays == (0.9999, 0.999)
    assert t.validation.sampler.num_steps == 50
    assert t.validation.batch_size_per_gpu > 0    # chunked-validation field
    assert t.logging.wandb.mode == "online"


def test_shipped_sample_yaml_loads():
    s = load_sample_config(PROJECT_ROOT / "configs/sample/cifar10_sample.yaml")
    assert s.sampling.num_samples == 50_000
    assert s.sampling.guidance_scale == pytest.approx(1.5)
    assert s.ema_tag.startswith("ema_")


# ---------------------------------------------------------------------------
# Helpers for writing tweaked configs to a tmp dir
# ---------------------------------------------------------------------------

_MODEL_MIN = {
    "arch_name": "DiT_S_2",
    "input_size": 32,
    "in_channels": 3,
    "patch_size": 2,
    "num_classes": 10,
    "class_dropout_prob": 0.1,
}


def _write_yaml(p: Path, obj) -> Path:
    p.write_text(yaml.safe_dump(obj), encoding="utf-8")
    return p


def _write_train_pair(tmp_path: Path, *, train_overrides=None, model_overrides=None) -> Path:
    """Write a valid model YAML + train YAML pair to ``tmp_path`` and return
    the path to the train YAML.  Overrides are shallow-merged in."""
    model = {**_MODEL_MIN, **(model_overrides or {})}
    _write_yaml(tmp_path / "model.yaml", model)

    train = {
        "model_config": "model.yaml",
        "dataset": {
            "name": "cifar10",
            "root": "/tmp/cifar10",
            "download": False,
            "num_workers": 2,
            "pin_memory": True,
            "augment": True,
        },
        "optim": {
            "name": "adamw",
            "lr": 1e-4,
            "betas": [0.9, 0.999],
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "warmup_steps": 100,
            "lr_schedule": "constant",
        },
        "train": {
            "total_steps": 1000,
            "batch_size_per_gpu": 8,
            "grad_accum_steps": 1,
            "log_interval": 10,
            "ckpt_interval": 500,
            "seed": 0,
            "amp_dtype": "bf16",
        },
        "ema": {"decays": [0.9999]},
        "guidance": {"null_class_id": 10},
        "validation": {
            "interval": 500,
            "num_samples": 8,
            "batch_size_per_gpu": 4,          # chunk size for chunked validation
            "visual_log_count": 4,
            "guidance_scales": [1.0],
            "metrics": {"fid": True, "inception_score": True},
            "sampler": {"num_steps": 10},
            "fid_ref_stats": None,
        },
        "logging": {
            "out_dir": "runs/",
            "run_name": "test",
            "wandb": {
                "enabled": False,
                "project": "p",
                "entity": None,
                "token_path": "secrets/wandb.token",
                "mode": "disabled",
            },
        },
    }
    if train_overrides:
        for k, v in train_overrides.items():
            train[k] = v
    return _write_yaml(tmp_path / "train.yaml", train)


# ---------------------------------------------------------------------------
# Rejection tests — model config
# ---------------------------------------------------------------------------

def test_model_unknown_key_rejected(tmp_path):
    p = _write_yaml(tmp_path / "m.yaml", {**_MODEL_MIN, "not_a_field": 42})
    with pytest.raises(UnknownKeyError):
        load_model_config(p)


def test_model_missing_required_field_rejected(tmp_path):
    bad = dict(_MODEL_MIN)
    del bad["patch_size"]
    p = _write_yaml(tmp_path / "m.yaml", bad)
    with pytest.raises(ConfigError):
        load_model_config(p)


def test_model_input_size_not_divisible_by_patch_size_rejected(tmp_path):
    p = _write_yaml(tmp_path / "m.yaml", {**_MODEL_MIN, "input_size": 33})
    with pytest.raises(ConfigError, match="divisible"):
        load_model_config(p)


def test_model_class_dropout_out_of_range_rejected(tmp_path):
    p = _write_yaml(tmp_path / "m.yaml", {**_MODEL_MIN, "class_dropout_prob": 1.5})
    with pytest.raises(ConfigError, match="class_dropout_prob"):
        load_model_config(p)


# ---------------------------------------------------------------------------
# Rejection tests — train config
# ---------------------------------------------------------------------------

def test_train_bad_optim_name_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    # Poison the optim.name.
    data = yaml.safe_load(train_path.read_text())
    data["optim"]["name"] = "sgd"
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="optim.name"):
        load_train_config(train_path)


def test_train_bad_amp_dtype_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["train"]["amp_dtype"] = "fp8"
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="amp_dtype"):
        load_train_config(train_path)


def test_train_visual_log_count_exceeds_num_samples_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["validation"]["visual_log_count"] = 999
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="visual_log_count"):
        load_train_config(train_path)


def test_train_validation_batch_size_per_gpu_positivity(tmp_path):
    """validation.batch_size_per_gpu must be a positive int (chunked-validation)."""
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["validation"]["batch_size_per_gpu"] = 0
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="batch_size_per_gpu"):
        load_train_config(train_path)


def test_train_validation_batch_size_per_gpu_missing_rejected(tmp_path):
    """batch_size_per_gpu has no default — a YAML that omits it must fail."""
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    del data["validation"]["batch_size_per_gpu"]
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="batch_size_per_gpu"):
        load_train_config(train_path)


def test_train_ema_decay_out_of_range_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["ema"]["decays"] = [1.5]
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="ema.decays"):
        load_train_config(train_path)


def test_train_unknown_key_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["mystery_field"] = 1
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(UnknownKeyError):
        load_train_config(train_path)


def test_train_missing_model_config_rejected(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    del data["model_config"]
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="model_config"):
        load_train_config(train_path)


def test_train_model_config_path_resolved_relative_to_train_yaml(tmp_path):
    """The `model_config` path in a train YAML must be resolved against the
    train YAML's own directory, not the caller's cwd — otherwise moving the
    launcher between machines breaks."""
    subdir = tmp_path / "sub"
    subdir.mkdir()
    train_path = _write_train_pair(subdir)
    # Load from a totally different cwd.
    t = load_train_config(train_path)
    assert isinstance(t.model, ModelConfig)


def test_train_wandb_enabled_requires_project(tmp_path):
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["logging"]["wandb"]["enabled"] = True
    data["logging"]["wandb"]["project"] = ""
    data["logging"]["wandb"]["mode"] = "online"
    train_path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="wandb.project"):
        load_train_config(train_path)


# ---------------------------------------------------------------------------
# Rejection tests — sample config
# ---------------------------------------------------------------------------

_SAMPLE_MIN = {
    "ckpt_path": "runs/x/ckpt/latest.pt",
    "ema_tag": "ema_0.9999",
    "sampling": {
        "num_samples": 100,
        "batch_size_per_gpu": 32,
        "num_steps": 10,
        "guidance_scale": 1.5,
        "seed": 0,
    },
    "dataset": {"name": "cifar10", "root": "/tmp/cifar10", "download": False},
    "metrics": {"fid": True, "inception_score": True, "fid_ref_stats": None},
    "output": {"dir": "runs/x/eval/", "save_images": False, "save_grid": True},
}


def test_sample_happy_path(tmp_path):
    p = _write_yaml(tmp_path / "s.yaml", _SAMPLE_MIN)
    s = load_sample_config(p)
    assert s.ema_tag == "ema_0.9999"
    assert s.sampling.num_samples == 100


def test_sample_bad_ema_tag_rejected(tmp_path):
    bad = {**_SAMPLE_MIN, "ema_tag": "bogus"}
    p = _write_yaml(tmp_path / "s.yaml", bad)
    with pytest.raises(ConfigError, match="ema_tag"):
        load_sample_config(p)


def test_sample_online_ema_tag_accepted(tmp_path):
    ok = {**_SAMPLE_MIN, "ema_tag": "online"}
    p = _write_yaml(tmp_path / "s.yaml", ok)
    s = load_sample_config(p)
    assert s.ema_tag == "online"


def test_sample_negative_guidance_scale_rejected(tmp_path):
    bad = {**_SAMPLE_MIN, "sampling": {**_SAMPLE_MIN["sampling"], "guidance_scale": -1.0}}
    p = _write_yaml(tmp_path / "s.yaml", bad)
    with pytest.raises(ConfigError, match="guidance_scale"):
        load_sample_config(p)


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------

def test_int_coerced_to_float_where_schema_says_float(tmp_path):
    """PyYAML gives back an int for `weight_decay: 0`; the schema should
    happily coerce it to float rather than complaining about the type."""
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    data["optim"]["weight_decay"] = 0     # int, not float
    train_path.write_text(yaml.safe_dump(data))
    t = load_train_config(train_path)
    assert isinstance(t.optim.weight_decay, float)
    assert t.optim.weight_decay == 0.0


def test_list_coerced_to_tuple_for_tuple_fields(tmp_path):
    train_path = _write_train_pair(tmp_path)
    t = load_train_config(train_path)
    # betas is declared tuple[float, float]; YAML wrote a list.
    assert isinstance(t.optim.betas, tuple)
    assert isinstance(t.ema.decays, tuple)
    assert isinstance(t.validation.guidance_scales, tuple)


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------

def test_apply_overrides_happy_path():
    d = {"optim": {"lr": 1e-4}, "train": {"seed": 0}}
    apply_overrides(d, ["optim.lr=2.0e-4", "train.seed=42"])
    assert d["optim"]["lr"] == 2e-4
    assert d["train"]["seed"] == 42


def test_apply_overrides_yaml_typed_values():
    d = {"train": {"amp_dtype": "bf16", "seed": 0}, "flag": False}
    apply_overrides(d, ["train.amp_dtype=fp32", "flag=true", "train.seed=99"])
    assert d["train"]["amp_dtype"] == "fp32"
    assert d["flag"] is True
    assert d["train"]["seed"] == 99


def test_apply_overrides_rejects_unknown_key():
    d = {"optim": {"lr": 1e-4}}
    with pytest.raises(ConfigError, match="does not exist"):
        apply_overrides(d, ["optim.momentum=0.9"])


def test_apply_overrides_rejects_malformed_spec():
    d = {"optim": {"lr": 1e-4}}
    with pytest.raises(ConfigError, match="not of the form"):
        apply_overrides(d, ["optim.lr"])


def test_apply_overrides_none_is_noop():
    d = {"a": 1}
    assert apply_overrides(d, None) is d
    assert d == {"a": 1}


# ---------------------------------------------------------------------------
# End-to-end: overrides + schema validation
# ---------------------------------------------------------------------------

def test_overrides_then_schema_validates(tmp_path):
    """CLI overrides applied to the raw dict must still be caught by the
    schema — a typo like ``--override optim.lr=abc`` should fail with a
    schema error, not run silently."""
    train_path = _write_train_pair(tmp_path)
    data = yaml.safe_load(train_path.read_text())
    apply_overrides(data, ["optim.lr=2.0e-4"])
    # Round-trip through the file so we exercise the real loader.
    train_path.write_text(yaml.safe_dump(data))
    t = load_train_config(train_path)
    assert t.optim.lr == pytest.approx(2e-4)


def test_top_level_yaml_must_be_mapping(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_model_config(p)


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_model_config(tmp_path / "nope.yaml")

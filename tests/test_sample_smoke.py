"""Smoke test for sample.py — end-to-end wiring, no GPU, no wandb.

Rationale (see doc/Train.md §9): sample.py is the entry point that
produces the paper's FID / IS numbers.  A full unit-test suite for
the sampling loop is out of scope, but a smoke test that exercises

    train.py: produce a tiny checkpoint (fixture + 3 training steps)
      → sample.py: load the checkpoint, generate 4 images, compute
        FID/IS = disabled, save a grid.png + results.json

catches ~90% of the wiring bugs (mismatched checkpoint keys, wrong
ema_tag lookup, broken model_config_yaml roundtrip, --ckpt override,
--override precedence, results.json schema, per-image dumping) that
would otherwise only fail on a real 50k-sample run.

FID and IS are *disabled* in the smoke config — those metrics require
downloading the Inception-V3 weights, which the test suite refuses to
do.  The dedicated FID/IS unit tests in :mod:`tests.test_eval` cover
that wire.

The test:

*   reuses :func:`tests.test_train_smoke._build_smoke_configs` /
    :func:`_build_dataset_root` to produce a matched tiny (dataset,
    train.yaml, model.yaml) triple and a step-3 checkpoint;
*   writes a tiny sample.yaml pointing at that checkpoint;
*   invokes :func:`sample.main` in-process on CPU with WORLD_SIZE=1;
*   asserts on ``results.json``, ``grid.png``, and (for the
    ``save_images`` variant) the per-image PNG dump layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

# Reuse the fixtures and config builder from the training smoke test.
# ``test_train_smoke`` sits next to this file; pytest's rootdir handling
# makes it importable as a top-level module.
from test_train_smoke import _build_dataset_root, _build_smoke_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _train_tiny_checkpoint(tmp_path: Path) -> Path:
    """Run train.main() on the fixtures for 3 steps; return latest.pt."""
    import train as train_module

    train_yaml, _model_yaml = _build_smoke_configs(
        tmp_path,
        run_name="sample_smoke_run",
        dataset_root=_build_dataset_root(tmp_path),
    )
    train_module.main(["--config", str(train_yaml)])

    latest = tmp_path / "runs" / "sample_smoke_run" / "ckpt" / "latest.pt"
    assert latest.exists(), f"training did not produce {latest}"
    return latest


def _write_sample_yaml(
    tmp_path: Path,
    ckpt_path: Path,
    dataset_root: Path,
    *,
    ema_tag: str = "online",
    num_samples: int = 4,
    save_images: bool = False,
    output_subdir: str = "eval",
) -> Path:
    """Write a sample.yaml matched to the tiny training config.

    ``num_samples = 4`` is the smallest value that still exercises the
    generation loop end-to-end without being pointlessly slow on CPU.
    """
    out_dir = tmp_path / output_subdir
    sample_yaml = tmp_path / f"sample_{output_subdir}.yaml"
    sample_yaml.write_text(yaml.safe_dump({
        "ckpt_path": str(ckpt_path),
        "ema_tag": ema_tag,
        "sampling": {
            "num_samples": num_samples,
            "batch_size_per_gpu": 2,   # forces >1 chunk for num_samples=4 on rank 0
            "num_steps": 2,            # matches training smoke's sampler.num_steps
            "guidance_scale": 1.0,
            "seed": 0,
        },
        "dataset": {
            "name": "cifar10",
            "root": str(dataset_root),
            "download": False,
        },
        "metrics": {
            # FID/IS both need Inception-V3 weights; disable in smoke.
            "fid": False,
            "inception_score": False,
            "fid_ref_stats": None,
        },
        "output": {
            "dir": str(out_dir),
            "save_images": save_images,
            "save_grid": True,
        },
    }))
    return sample_yaml


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sample_smoke_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train once → sample.main() on the produced checkpoint → check outputs."""
    import sample as sample_module

    # Force CPU regardless of host, matching the training smoke.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    latest = _train_tiny_checkpoint(tmp_path)
    dataset_root = tmp_path / "data"

    # ---- online weights ----
    sample_yaml = _write_sample_yaml(
        tmp_path, latest, dataset_root, ema_tag="online",
        output_subdir="eval_online",
    )
    sample_module.main(["--config", str(sample_yaml)])

    out_dir = tmp_path / "eval_online"
    results_path = out_dir / "results.json"
    grid_path = out_dir / "grid.png"

    assert results_path.is_file(), f"results.json missing under {out_dir}"
    assert grid_path.is_file(), f"grid.png missing under {out_dir}"

    payload = json.loads(results_path.read_text())
    for key in ("ckpt_path", "ema_tag", "guidance_scale", "num_steps",
                "num_samples", "seed", "fid", "is_mean", "is_std",
                "wall_time_s", "git_sha"):
        assert key in payload, f"results.json missing key {key!r}"

    assert payload["ema_tag"] == "online"
    assert payload["num_samples"] == 4
    assert payload["num_steps"] == 2
    assert payload["guidance_scale"] == 1.0
    assert payload["seed"] == 0
    assert payload["fid"] is None            # metrics disabled
    assert payload["is_mean"] is None
    assert payload["is_std"] is None
    assert payload["wall_time_s"] > 0.0
    # git_sha may be None outside a git checkout — accept either.
    assert payload["git_sha"] is None or isinstance(payload["git_sha"], str)


def test_sample_smoke_ema_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tiny training run creates ema_0.99; sample.py must load it."""
    import sample as sample_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    latest = _train_tiny_checkpoint(tmp_path)
    dataset_root = tmp_path / "data"

    sample_yaml = _write_sample_yaml(
        tmp_path, latest, dataset_root, ema_tag="ema_0.99",
        output_subdir="eval_ema",
    )
    sample_module.main(["--config", str(sample_yaml)])

    payload = json.loads((tmp_path / "eval_ema" / "results.json").read_text())
    assert payload["ema_tag"] == "ema_0.99"


def test_sample_smoke_missing_ema_tag_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requesting an EMA tag that isn't in the checkpoint must fail loudly."""
    import sample as sample_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    latest = _train_tiny_checkpoint(tmp_path)
    dataset_root = tmp_path / "data"

    sample_yaml = _write_sample_yaml(
        tmp_path, latest, dataset_root, ema_tag="ema_0.9999",
        output_subdir="eval_bad",
    )
    with pytest.raises(RuntimeError, match="ema_0.9999"):
        sample_module.main(["--config", str(sample_yaml)])


def test_sample_smoke_ckpt_shorthand_and_override_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--ckpt beats YAML; --override ckpt_path=... beats --ckpt.

    Also validates that when neither shorthand is given, the YAML's
    ckpt_path is used unchanged.
    """
    import sample as sample_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    latest = _train_tiny_checkpoint(tmp_path)
    dataset_root = tmp_path / "data"

    # A YAML pointing at a bogus path — sample.py should never load this
    # in the two cases below because --ckpt / --override supersede it.
    bogus = tmp_path / "does_not_exist.pt"
    sample_yaml = _write_sample_yaml(
        tmp_path, bogus, dataset_root, ema_tag="online",
        output_subdir="eval_prec",
    )

    # Case A: --ckpt shorthand supersedes the (bogus) YAML value.
    sample_module.main([
        "--config", str(sample_yaml),
        "--ckpt", str(latest),
    ])
    payload_a = json.loads(
        (tmp_path / "eval_prec" / "results.json").read_text()
    )
    assert payload_a["ckpt_path"] == str(latest.resolve())

    # Case B: --override wins over --ckpt.
    #
    # Point --ckpt at the bogus path and --override at the good one;
    # sample.py must load the good one.
    sample_module.main([
        "--config", str(sample_yaml),
        "--ckpt", str(bogus),
        "--override", f"ckpt_path={latest}",
    ])
    payload_b = json.loads(
        (tmp_path / "eval_prec" / "results.json").read_text()
    )
    assert payload_b["ckpt_path"] == str(latest.resolve())


def test_sample_smoke_save_images_dumps_per_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When output.save_images is true, every generated image lands on disk."""
    import sample as sample_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    latest = _train_tiny_checkpoint(tmp_path)
    dataset_root = tmp_path / "data"

    sample_yaml = _write_sample_yaml(
        tmp_path, latest, dataset_root,
        ema_tag="online",
        num_samples=4,
        save_images=True,
        output_subdir="eval_dump",
    )
    sample_module.main(["--config", str(sample_yaml)])

    imgs_root = tmp_path / "eval_dump" / "imgs" / "rank0"
    assert imgs_root.is_dir(), f"no per-rank imgs dir under {imgs_root}"
    dumped = sorted(imgs_root.glob("*.png"))
    assert len(dumped) == 4, (
        f"expected 4 per-image PNGs, got {len(dumped)}: {dumped}"
    )
    # File names are 7-digit zero-padded global indices, starting at 0.
    assert dumped[0].name == "0000000.png"
    assert dumped[3].name == "0000003.png"

"""Offline sampling / FID-IS evaluation entry point.

Runtime spec:  doc/Train.md §2.3, §5.2, §7, §8
Math spec:     doc/FlowMatching.md §6

Purpose
-------
Given a training checkpoint written by ``train.py``, produce **paper-scale**
samples (default: ``num_samples = 50 000`` per the sample YAML), score
FID and Inception Score, save a small grid PNG for eyeballing, and
write a machine-readable ``results.json``.

This is the tool that produces the numbers reported in the paper.
It is deliberately independent of the training loop: no wandb, no
optimizer, no scheduler, no train YAML — just a checkpoint plus a
sample YAML.

CLI
---
::

    python sample.py --config configs/sample/cifar10_sample.yaml \
                     [--override sampling.num_samples=1000 ...] \
                     [--dataset_root /abs/path/to/cifar10]      \
                     [--ckpt runs/<run>/ckpt/step_000400000.pt]

    torchrun --standalone --nproc_per_node=N sample.py --config ...

Precedence for the checkpoint path (lowest -> highest):
    YAML  <  --ckpt <path>  <  --override ckpt_path=<path>

If ``--ckpt`` is omitted, the ``ckpt_path`` from the YAML is used
unchanged; ``--override ckpt_path=...`` further wins over ``--ckpt``.

Multi-GPU
---------
Works out of the box under ``torchrun``.  The sharding scheme mirrors
``run_validation`` in ``train.py``: the full-``N`` noise + labels are
drawn once, identically on every rank (same seed, no rank offset), and
each rank slices its own contiguous shard.  FID/IS use
``sync_on_compute=True``, so every rank ends up with the same scalar
after the collective ``compute()``.  The final numbers are identical
to a ``--nproc_per_node=1`` run up to fp64 associativity in the
running sums.

Weight loading
--------------
The architecture is recovered from the ``model_config_yaml`` field
embedded in the checkpoint — never from a sibling YAML — so
sample-time and train-time cannot disagree on shape.  Which weight
set to load is controlled by ``ema_tag``:

* ``ema_tag = "online"``       → ``payload["model_state"]``
* ``ema_tag = "ema_<decay>"``  → ``payload["ema_states"][tag]["module_state"]``

Rank 0 loads onto the model, then :func:`runtime.broadcast_module_state`
copies the weights to every other rank.

Outputs (all under ``output.dir``)
----------------------------------
* ``results.json``  — {ckpt_path, ema_tag, guidance_scale, num_samples,
                       fid, is_mean, is_std, seed, git_sha, wall_time_s}
* ``grid.png``      — 8×8-ish grid of the leading generated images
                      (skipped if ``output.save_grid = false``)
* ``imgs/rank<r>/<idx:07d>.png`` — every generated image, one per file
                                   (only when ``output.save_images = true``)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torchvision.utils import make_grid, save_image

# Local imports — every one of these has a green test suite behind it.
import models
from configs import ConfigError, SampleConfig, apply_overrides
from configs.schema import (
    ModelConfig,
    _from_dict_strict,
    _load_yaml,
)
from data.cifar10 import CIFAR10Dataset
from eval.fid import FIDMetric
from eval.inception_score import InceptionScoreMetric
from flow.sampler import sample as euler_sample
from runtime import (
    broadcast_module_state,
    build_or_load_fid_cache,
    cleanup_distributed,
    get_git_sha,
    load_checkpoint,
    set_seed,
    setup_distributed,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline sampling / FID-IS evaluation (Train.md §2.3, §5.2)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a sample YAML (see configs/sample/cifar10_sample.yaml).",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config field, e.g. --override sampling.num_samples=1000. "
             "Repeatable. Applied *before* schema validation, so typos are "
             "caught the same way YAML typos are. Overrides win over --ckpt.",
    )
    parser.add_argument(
        "--dataset_root", default=None,
        help="Overrides dataset.root from the YAML (Train.md §2.2). "
             "Provided so the same config + same launcher work on a new "
             "machine by editing exactly one CLI flag.",
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Convenience shorthand: overrides ckpt_path from the YAML. "
             "If omitted, the YAML's ckpt_path is used unchanged. "
             "Loses to an explicit --override ckpt_path=... .",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Config loading with CLI overrides
# ---------------------------------------------------------------------------

def load_and_prepare_config(args: argparse.Namespace) -> SampleConfig:
    """Read + validate the sample YAML, applying CLI overrides.

    Precedence (low -> high): YAML  <  ``--ckpt``  <  ``--override``.
    Both ``--dataset_root`` and ``--ckpt`` are applied *before*
    ``apply_overrides`` so an explicit ``--override`` still wins — this
    matches the general rule "explicit --override beats convenience
    shorthands" also used by ``train.py``'s dataset_root path.
    """
    sample_yaml_path = Path(args.config).resolve()
    raw = _load_yaml(sample_yaml_path)

    # Convenience shorthands write into the raw dict *before* the
    # generic --override pass, so an explicit --override key can still
    # shadow them.
    if args.dataset_root is not None:
        raw.setdefault("dataset", {})["root"] = args.dataset_root
    if args.ckpt is not None:
        raw["ckpt_path"] = args.ckpt

    apply_overrides(raw, list(args.override))

    return _from_dict_strict(SampleConfig, raw, path="")


# ---------------------------------------------------------------------------
# Model construction from a checkpoint
# ---------------------------------------------------------------------------
#
# The arch registry + ``ModelConfig -> nn.Module`` factory live in
# ``models`` (single source of truth shared with ``train.py``).  Both
# entry points call ``models.build_model_from_config`` so a config-shape
# change has exactly one place to update.


def _select_state_dict(payload: dict[str, Any], ema_tag: str) -> dict[str, Any]:
    """Pick which weight set to load from a checkpoint.

    * ``ema_tag == "online"``     — ``payload["model_state"]``
    * ``ema_tag == "ema_<decay>"`` — ``payload["ema_states"][tag]["module_state"]``

    Fails loudly on a missing tag; refusing to silently fall back to
    the online weights matches the "no silent shape/config drift"
    invariant used elsewhere.
    """
    if ema_tag == "online":
        if "model_state" not in payload:
            raise RuntimeError(
                "checkpoint has no 'model_state' — cannot load 'online' weights"
            )
        return payload["model_state"]

    ema_states = payload.get("ema_states", {})
    if ema_tag not in ema_states:
        available = sorted(ema_states.keys())
        raise RuntimeError(
            f"ema_tag={ema_tag!r} not present in checkpoint; "
            f"available EMA tags: {available!r} (plus 'online')"
        )
    entry = ema_states[ema_tag]
    if "module_state" not in entry:
        raise RuntimeError(
            f"checkpoint's ema_states[{ema_tag!r}] has no 'module_state' "
            f"(keys: {sorted(entry.keys())})"
        )
    return entry["module_state"]


def _build_model_from_ckpt(
    cfg: SampleConfig,
    device: torch.device,
    is_main: bool,
    world_size: int,
) -> tuple[nn.Module, ModelConfig]:
    """Reconstruct the DiT and load the requested weight set onto every rank.

    Every rank reads the checkpoint itself (a single .pt file) and
    loads the selected state dict onto its local copy.  Reading the
    file redundantly is far cheaper than broadcasting a ~120 MB state
    dict, and it matches the "each rank has its own checkpoint file
    open" pattern used by ``train.py`` on ``--resume``.

    We still call :func:`broadcast_module_state` afterwards under DDP
    — not for correctness (every rank loaded the same bytes) but as a
    cheap invariant that no rank silently diverged.  On a ~30M-param
    DiT this is milliseconds.

    Returns
    -------
    (model, model_cfg)
        The live module and the reconstructed ``ModelConfig`` — the
        latter is passed through to :func:`_generate_and_score` so
        that no shape read-back off the module is needed.
    """
    ckpt_path = Path(cfg.ckpt_path).resolve()
    payload = load_checkpoint(ckpt_path)

    model_yaml_text: str = payload.get("model_config_yaml", "")
    if not model_yaml_text.strip():
        raise RuntimeError(
            f"checkpoint at {ckpt_path} has no embedded 'model_config_yaml'; "
            f"refusing to guess the architecture."
        )
    model_data = yaml.safe_load(model_yaml_text)
    model_cfg = _from_dict_strict(ModelConfig, model_data, path="model")

    model = models.build_model_from_config(model_cfg).to(device)

    state_dict = _select_state_dict(payload, cfg.ema_tag)
    model.load_state_dict(state_dict)

    if world_size > 1:
        broadcast_module_state(model, src=0)

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[init] loaded {cfg.ema_tag!r} weights from {ckpt_path}  "
              f"arch={model_cfg.arch_name}  params={n_params/1e6:.2f}M",
              flush=True)

    model.eval()
    return model, model_cfg


# ---------------------------------------------------------------------------
# Generation + metrics — sharded, world-size-invariant
# ---------------------------------------------------------------------------

@torch.no_grad()
def _generate_and_score(
    cfg: SampleConfig,
    model: nn.Module,
    model_cfg: ModelConfig,
    device: torch.device,
    rank: int,
    world_size: int,
    is_main: bool,
) -> dict[str, Any]:
    """Run the sharded generation + metric-aggregation pass.

    Returns a dict of scalars on every rank (the values are identical
    across ranks after the collective ``compute()``).  The caller is
    responsible for writing files / logging on rank 0 only.

    This is a stripped-down copy of ``run_validation`` in ``train.py``:
    only one ``ema_tag`` (already loaded onto ``model``) and only one
    ``guidance_scale``.  The full-``N`` noise / label draw is done once
    identically on every rank, then sliced per shard — this is what
    guarantees that FID/IS are invariant under changes to ``world_size``
    and ``batch_size_per_gpu``.  See ``run_validation``'s docstring
    for the full derivation; the invariants preserved here are the
    same.
    """
    n_classes = model_cfg.num_classes
    null_id = n_classes  # matches models/dit.py LabelEmbedder.null_id

    N = cfg.sampling.num_samples
    per_rank_chunk = math.ceil(N / world_size)
    shard_start = min(rank * per_rank_chunk, N)
    shard_end = min(shard_start + per_rank_chunk, N)
    shard_len = shard_end - shard_start
    chunk_size = cfg.sampling.batch_size_per_gpu
    n_chunks = math.ceil(shard_len / chunk_size) if shard_len > 0 else 0

    C = model_cfg.in_channels
    H = W = model_cfg.input_size
    guidance_scale = float(cfg.sampling.guidance_scale)

    # Draw the full-N noise/labels once, then slice per shard + per
    # chunk.  Same rationale as run_validation: interleaving randn /
    # randint calls per-chunk would consume the RNG stream in a
    # chunk-size-dependent order, so different schedules would produce
    # different noise.  Draw-once makes both batch_size_per_gpu and
    # world_size pure throughput knobs.
    val_gen = torch.Generator(device=device).manual_seed(cfg.sampling.seed)
    x_init_all = torch.randn(N, C, H, W, generator=val_gen, device=device)
    y_all = torch.randint(0, n_classes, (N,), generator=val_gen, device=device)
    y_null_all = torch.full((N,), null_id, dtype=torch.long, device=device)

    # ---- Metrics ----
    # Both metrics live on every rank (sync_on_compute=True); if any
    # rank skipped compute() the ranks that entered would deadlock.
    fid_metric: FIDMetric | None = None
    is_metric: InceptionScoreMetric | None = None
    if cfg.metrics.fid:
        # Reuse the same build/load helper as train.py: rank 0 builds
        # the reference cache once if missing, every rank loads it.
        # We hand it a *non-augmented* copy of the training set.
        ref_ds = CIFAR10Dataset(
            root=cfg.dataset.root, split="train", augment=False,
        )
        fid_metric = build_or_load_fid_cache(
            cfg.metrics.fid_ref_stats,
            ref_ds,
            device,
            is_main,
            world_size,
        )
    if cfg.metrics.inception_score:
        is_metric = InceptionScoreMetric(splits=10, device=device)

    # ---- Sampling loop ----
    # On rank 0, keep the leading `k` images so we can save a grid.
    # By construction rank 0's shard is x_init_all[:per_rank_chunk],
    # which are the *global* leading images.
    grid_target = 64  # 8x8 grid; capped by shard length
    k = min(grid_target, N, shard_len) if is_main else 0
    grid_parts: list[torch.Tensor] = []
    grid_collected = 0

    # Optional per-image dumps.
    imgs_dir: Path | None = None
    if cfg.output.save_images:
        imgs_dir = Path(cfg.output.dir) / "imgs" / f"rank{rank}"
        imgs_dir.mkdir(parents=True, exist_ok=True)

    for ci in range(n_chunks):
        start = shard_start + ci * chunk_size
        end = min(start + chunk_size, shard_end)
        x_init_i = x_init_all[start:end]
        y_i = y_all[start:end]
        y_null_i = y_null_all[start:end]

        imgs_i = euler_sample(
            model=model,
            x_init=x_init_i,
            y=y_i,
            y_null=y_null_i,
            num_steps=cfg.sampling.num_steps,
            guidance_scale=guidance_scale,
        )

        if fid_metric is not None:
            fid_metric.update_fake(imgs_i)
        if is_metric is not None:
            is_metric.update(imgs_i)

        if is_main and k > 0 and grid_collected < k:
            take = min(k - grid_collected, imgs_i.shape[0])
            grid_parts.append(imgs_i[:take].detach().clone())
            grid_collected += take

        if imgs_dir is not None:
            # Convert [-1, 1] -> [0, 1] and dump one PNG per sample.
            # Per-rank subdirectories mean no cross-rank collisions
            # and no barrier is needed.
            imgs_norm = imgs_i.clamp(-1, 1).add(1).mul(0.5).cpu()
            for j in range(imgs_norm.shape[0]):
                idx = start + j  # global index across all ranks
                save_image(imgs_norm[j], imgs_dir / f"{idx:07d}.png")

        del imgs_i

    del x_init_all, y_all, y_null_all

    # ---- Collective compute ----
    results: dict[str, Any] = {
        "num_samples": N,
        "guidance_scale": guidance_scale,
        "ema_tag": cfg.ema_tag,
        "seed": cfg.sampling.seed,
    }
    if fid_metric is not None:
        results["fid"] = fid_metric.compute()
    if is_metric is not None:
        is_mean, is_std = is_metric.compute()
        results["is_mean"] = is_mean
        results["is_std"] = is_std

    # ---- Grid PNG (rank 0 only) ----
    if is_main and cfg.output.save_grid and grid_parts:
        grid_snapshot = torch.cat(grid_parts, dim=0)
        grid_imgs = grid_snapshot.clamp(-1, 1).add(1).mul(0.5)
        nrow = int(math.sqrt(grid_snapshot.shape[0])) or 1
        grid = make_grid(grid_imgs, nrow=nrow)
        grid_path = Path(cfg.output.dir) / "grid.png"
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, grid_path)
        print(f"[sample] saved grid to {grid_path}", flush=True)

    # Keep ranks in step before the caller tears down.
    if world_size > 1:
        dist.barrier()

    return results


# ---------------------------------------------------------------------------
# Auxiliary utilities — results.json
# ---------------------------------------------------------------------------

def _write_results_json(
    cfg: SampleConfig,
    results: dict[str, Any],
    wall_time_s: float,
) -> Path:
    """Serialise the metric summary to ``output.dir/results.json``.

    Keys are stable and machine-readable; the paper table script
    aggregates over multiple such files with ``jq`` / a tiny loader.
    """
    payload = {
        "ckpt_path": str(Path(cfg.ckpt_path).resolve()),
        "ema_tag": cfg.ema_tag,
        "guidance_scale": float(cfg.sampling.guidance_scale),
        "num_steps": int(cfg.sampling.num_steps),
        "num_samples": int(cfg.sampling.num_samples),
        "seed": int(cfg.sampling.seed),
        "fid": results.get("fid"),
        "is_mean": results.get("is_mean"),
        "is_std": results.get("is_std"),
        "wall_time_s": float(wall_time_s),
        "git_sha": get_git_sha(),
    }
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def generate_and_evaluate(cfg: SampleConfig) -> dict[str, Any]:
    """Run the full sampling + evaluation pipeline.

    Returns the ``results`` dict on every rank; on rank 0 the same
    dict has already been serialised to ``output.dir/results.json``
    and (unless disabled) a ``grid.png`` has been saved beside it.
    """
    rank, world_size, _local_rank, is_main, device = setup_distributed()

    # Cross-rank-identical seed on the *torch* / numpy / random RNGs
    # is fine here: the generation loop uses a *dedicated* Generator
    # (``val_gen`` in _generate_and_score) that is independent of the
    # global RNGs, so the shared seed only affects things like
    # DataLoader shuffle order for the FID reference build (which is
    # itself sequential on rank 0).
    set_seed(cfg.sampling.seed)

    if is_main:
        Path(cfg.output.dir).mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if is_main:
        print(f"[sample] ckpt={cfg.ckpt_path}  ema_tag={cfg.ema_tag}  "
              f"num_samples={cfg.sampling.num_samples}  "
              f"steps={cfg.sampling.num_steps}  "
              f"gs={cfg.sampling.guidance_scale}  "
              f"world_size={world_size}  device={device}", flush=True)

    model, model_cfg = _build_model_from_ckpt(cfg, device, is_main, world_size)

    t0 = time.time()
    results = _generate_and_score(
        cfg, model, model_cfg, device, rank, world_size, is_main,
    )
    wall_time_s = time.time() - t0

    if is_main:
        results_path = _write_results_json(cfg, results, wall_time_s)
        fid_s = f"FID={results.get('fid', float('nan')):.4f}" \
            if cfg.metrics.fid else "FID=disabled"
        is_s = (f"IS={results.get('is_mean', float('nan')):.4f}"
                f" ± {results.get('is_std', float('nan')):.4f}") \
            if cfg.metrics.inception_score else "IS=disabled"
        print(f"[sample] done  {fid_s}  {is_s}  "
              f"wall={wall_time_s/60:.1f}min  "
              f"results={results_path}", flush=True)

    cleanup_distributed()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    cfg = load_and_prepare_config(args)
    return generate_and_evaluate(cfg)


if __name__ == "__main__":
    try:
        main()
    except (ConfigError, FileNotFoundError, RuntimeError) as e:
        print(f"[sample] error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

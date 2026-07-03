"""Training entry point — flow-matching DiT on class-conditional images.

Runtime spec:  doc/Train.md
Math spec:     doc/FlowMatching.md

CLI
---
::

    python train.py --config configs/train/cifar10_train.yaml \
                    [--override optim.lr=2.0e-4 ...]           \
                    [--dataset_root /abs/path/to/cifar10]      \
                    [--resume runs/<run_name>/ckpt/latest.pt]

The single-GPU form ``python train.py ...`` is fully supported (WORLD_SIZE=1
is auto-inferred).  The multi-GPU form is::

    torchrun --standalone --nproc_per_node=N train.py --config ...

Only rank 0 owns the EMA copies, writes checkpoints, runs validation,
computes metrics, and talks to wandb (Train.md §4.2, §5, §6, §7).

Cross-entry-point helpers (distributed setup, RNG snapshot/restore,
EMA-shadow broadcast, FID reference-cache build/load) live in the
``runtime/`` package and are shared with ``sample.py``.  Anything that
is still specific to training (the training loop itself, wandb setup,
checkpoint save/load, ``ResumableDistributedSampler``) stays here.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid, save_image

# Local imports — every one of these has a green test suite behind it.
import models
from configs import (
    ConfigError,
    TrainConfig,
    apply_overrides,
    load_train_config,
)
from configs.schema import _from_dict_strict, _load_yaml  # noqa: F401  (used only for reload after overrides)
from data.cifar10 import CIFAR10Dataset
from eval.fid import FIDMetric
from eval.inception_score import InceptionScoreMetric
from flow.ema import EMA
from flow.interpolant import interpolant, velocity_gt
from flow.loss import flow_matching_loss
from flow.sampler import sample as euler_sample
from runtime import (
    CHECKPOINT_VERSION,
    broadcast_module_state,
    build_or_load_fid_cache,
    cleanup_distributed,
    get_git_sha,
    load_checkpoint,
    restore_rng,
    set_seed,
    setup_distributed,
    snapshot_rng,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flow-matching DiT training entry point (Train.md §1)",
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to a train YAML (see configs/train/cifar10_train.yaml).",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config field, e.g. --override optim.lr=2.0e-4. "
             "Repeatable. Applied *before* schema validation, so typos are "
             "caught the same way YAML typos are.",
    )
    parser.add_argument(
        "--dataset_root", default=None,
        help="Overrides dataset.root from the YAML (Train.md §2.2). "
             "Provided so the same config + same launcher work on a new "
             "machine by editing exactly one CLI flag.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint .pt file to resume from. If omitted, "
             "training starts from scratch.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Distributed / device setup and cross-rank weight broadcast are provided
# by ``runtime.dist_utils``; RNG seeding + snapshot/restore live in
# ``runtime.rng``; the FID reference-cache build/load helper lives in
# ``runtime.fid_cache``.  All three are re-exported from the ``runtime``
# package (imported at the top of this file) and shared with ``sample.py``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model / optimiser / schedule builders
# ---------------------------------------------------------------------------

# The arch registry + ``ModelConfig -> nn.Module`` factory now live in
# ``models`` (single source of truth shared with ``sample.py``).  The
# alias below keeps ``train.build_model`` reachable for tests and any
# external callers that import it from this module.
build_model = models.build_model_from_config


def build_optimizer(cfg_optim, params) -> torch.optim.Optimizer:
    if cfg_optim.name != "adamw":  # schema already restricts this
        raise ConfigError(f"unsupported optim.name={cfg_optim.name!r}")
    return torch.optim.AdamW(
        params,
        lr=cfg_optim.lr,
        betas=cfg_optim.betas,
        weight_decay=cfg_optim.weight_decay,
    )


def build_lr_scheduler(
    cfg_optim,
    total_steps: int,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear-warmup then constant/cosine, as a LambdaLR over global step."""
    warmup = max(1, int(cfg_optim.warmup_steps))

    if cfg_optim.lr_schedule == "constant":
        def _lr(step: int) -> float:
            if step < cfg_optim.warmup_steps:
                return float(step) / float(warmup)
            return 1.0
    elif cfg_optim.lr_schedule == "cosine":
        def _lr(step: int) -> float:
            if step < cfg_optim.warmup_steps:
                return float(step) / float(warmup)
            # Cosine over the post-warmup range, ending at 0.
            progress = (step - cfg_optim.warmup_steps) / max(
                1, total_steps - cfg_optim.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:  # pragma: no cover — schema guards
        raise ConfigError(f"unsupported lr_schedule={cfg_optim.lr_schedule!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr)


# ---------------------------------------------------------------------------
# Resumable distributed sampler
# ---------------------------------------------------------------------------

class ResumableDistributedSampler(DistributedSampler):
    """DistributedSampler with a step-level skip for --resume.

    When training resumes at ``start_step``, we want each rank to
    pick up at exactly the sample it would have consumed next.  We
    achieve this by:

    1. Setting the sampler's epoch to ``start_step * batch_size //
       len(dataset)`` — the "epoch" the resumed step falls in — which
       makes ``__iter__`` produce the same shuffled-per-epoch order it
       would have on a non-crashed run (DistributedSampler's shuffle
       is deterministic in ``(seed, epoch)``).
    2. Skipping the first ``skip`` samples of that epoch, where ``skip``
       is the residual of the position within the epoch.

    The training loop is responsible for calling
    :meth:`set_epoch` (which zeroes the skip) at each new epoch
    boundary and for calling :meth:`set_start_step` exactly once, right
    after loading a resume checkpoint.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._skip = 0

    def set_start_step(self, start_step: int, batch_size_per_rank: int) -> None:
        """Initialise from a resumed step; must be called before iterating."""
        if start_step <= 0:
            self._skip = 0
            self.set_epoch(0)
            return
        # num_samples is the per-rank length of one epoch.
        per_rank_epoch = self.num_samples
        pos = start_step * batch_size_per_rank            # sample index (per rank)
        epoch = pos // max(1, per_rank_epoch)
        # Order matters: set_epoch() zeroes ``_skip``, so set the epoch
        # *first* and then install the residual skip.
        self.set_epoch(int(epoch))
        self._skip = pos - epoch * per_rank_epoch

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        self._skip = 0

    def __iter__(self):
        it = super().__iter__()
        if self._skip:
            for _ in range(self._skip):
                next(it, None)
            # Skip only fires once — subsequent epochs restart naturally.
            self._skip = 0
        yield from it


# ---------------------------------------------------------------------------
# Checkpoint I/O — Train.md §3.1
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: nn.Module,          # unwrapped (i.e. .module under DDP)
    ema_dict: dict[str, EMA],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler_state: Any | None,
    train_yaml: str,
    model_yaml: str,
    git_sha: str | None,
) -> None:
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "ema_states": {tag: ema.state_dict() for tag, ema in ema_dict.items()},
        "optim_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler_state,
        "rng_state": snapshot_rng(),
        "config_yaml": train_yaml,
        "model_config_yaml": model_yaml,
        "git_sha": git_sha,
        "version": CHECKPOINT_VERSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on POSIX

    # Update the `latest.pt` symlink in the same directory.
    latest = path.parent / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)


# ---------------------------------------------------------------------------
# wandb setup — rank 0 only
# ---------------------------------------------------------------------------

def init_wandb(cfg: TrainConfig, run_dir: Path):
    """Return an initialised wandb run, or None if disabled/unavailable."""
    wcfg = cfg.logging.wandb
    if not wcfg.enabled or wcfg.mode == "disabled":
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[wandb] package not installed; continuing without wandb logging.",
              file=sys.stderr)
        return None

    token_path = Path(wcfg.token_path)
    if token_path.is_file():
        try:
            wandb.login(key=token_path.read_text().strip(), verify=False)
        except Exception as e:  # pragma: no cover — best-effort
            print(f"[wandb] wandb.login failed ({e}); "
                  "continuing without wandb.", file=sys.stderr)
            return None
    else:
        print(f"[wandb] token file {token_path} not found; "
              "continuing without wandb.", file=sys.stderr)
        return None

    return wandb.init(
        project=wcfg.project,
        entity=wcfg.entity,
        name=cfg.logging.run_name,
        dir=str(run_dir),
        mode=wcfg.mode,
        config={
            "train_yaml": cfg,          # dataclass; wandb will str-ify
        },
    )


# ---------------------------------------------------------------------------
# Validation — Train.md §7
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    step: int,
    online_net: nn.Module,
    ema_dict: dict[str, EMA],
    cfg: TrainConfig,
    device: torch.device,
    run_dir: Path,
    fid_metric: FIDMetric | None,
    is_metric: InceptionScoreMetric | None,
    wandb_run: Any,
    rank: int = 0,
    world_size: int = 1,
    is_main: bool = True,
) -> None:
    """Run validation, sharded across all ranks under DDP.

    ``online_net`` is the trainable network (the *online* copy in the
    online-vs-EMA sense; see doc/Train.md §5).  It must be the
    **unwrapped** ``nn.Module`` — callers strip DDP.  Its state is
    snapshotted at entry and restored at exit; each EMA shadow (owned by
    rank 0) is copied onto it in turn and then broadcast to all ranks
    for evaluation.

    Sharding — each rank generates a contiguous slice of the ``N`` total
    samples.  With ``chunk = ceil(N / world_size)`` and per-rank range
    ``[rank*chunk : (rank+1)*chunk]`` (last rank clamped at ``N``), the
    partition is disjoint and covers exactly ``N`` samples.  The full-N
    noise/label tensors are drawn identically on every rank (same seed,
    no rank offset), then sliced — this preserves bit-equivalence with
    the ``world_size=1`` run.

    Chunking within each rank uses ``validation.batch_size_per_gpu`` as
    the per-rank chunk size (peak VRAM bounded by the chunk, not by the
    per-rank shard).

    Metric aggregation — both :class:`FIDMetric` and
    :class:`InceptionScoreMetric` are constructed with
    ``sync_on_compute=True``, so ``compute()`` all-reduces the running
    sums across ranks before deriving the scalar.  Every rank ends up
    with the same value; only rank 0 logs it.

    For each ``(ema_tag, guidance_scale)`` pair we save a wandb grid PNG
    from the leading ``visual_log_count`` images (drawn from rank 0's
    slice, which is the *global* leading slice by construction) and log
    every scalar with the tag ``val/<name>_<ema>_gs<scale>`` — rank 0
    only.

    Reproducibility invariants preserved by chunking + sharding:
      * The RNG is re-seeded to ``cfg.train.seed + step`` at the top of
        every ``(tag, gs)`` pair, identically on every rank, so each
        pair starts from an identical noise stream — differences across
        pairs are attributable to the pair, not to noise or to the
        world size.
      * All noise and labels for the full ``N`` are drawn *once* at the
        top of every pair, before sharding + chunking, and then sliced.
        This is critical: interleaving ``randn``/``randint`` calls
        per-chunk would consume the RNG stream in a chunk-size-dependent
        order (``randint`` uses rejection sampling internally, and
        Box-Muller for ``randn`` caches an odd leftover), so different
        schedules would produce different noise.  Draw-once makes both
        ``batch_size_per_gpu`` and ``world_size`` pure throughput knobs
        with no effect on the numerical output.  Cost: one full-``N``
        noise tensor per pair per rank, freed at the end of the pair.
      * FID/IS ``compute()`` on the accumulated cross-rank + chunk-wise
        state equals a single-shot compute up to fp64 associativity in
        the running sums.

    The string tag ``"online"`` used below is deliberate: it names the
    *entry* in ``ema_tags`` that means "live training weights", and
    matches the ``ema_tag`` field of the sample YAML.
    """
    val = cfg.validation
    model_cfg = cfg.model
    assert model_cfg is not None

    online_net.eval()
    online_state = copy.deepcopy(online_net.state_dict())

    # {tag: module_to_evaluate}.  Tag "online" reuses `online_net` in-place;
    # each `ema_<d>` temporarily copies its shadow onto `online_net` for
    # evaluation, then we restore `online_state` after the pair-loop.
    ema_tags = ["online"] + [f"ema_{d}" for d in cfg.ema.decays]

    N = val.num_samples
    per_rank_chunk = math.ceil(N / world_size)
    shard_start = min(rank * per_rank_chunk, N)
    shard_end = min(shard_start + per_rank_chunk, N)
    shard_len = shard_end - shard_start
    chunk_size = val.batch_size_per_gpu
    n_chunks = math.ceil(shard_len / chunk_size) if shard_len > 0 else 0
    C = model_cfg.in_channels
    H = W = model_cfg.input_size
    n_classes = model_cfg.num_classes
    null_id = n_classes  # LabelEmbedder.null_id (see models/dit.py)

    for tag in ema_tags:
        # Swap weights for this EMA tag.  EMA shadows live on rank 0
        # only (see train() below), so for ema_* tags we copy on rank 0
        # and broadcast to all ranks.  The "online" tag needs no
        # broadcast: DDP's construction + step-time all-reduce keep
        # every rank's `online_net` in sync.
        if tag == "online":
            pass  # `online_net` already holds the live training weights
        else:
            if is_main:
                ema_dict[tag].copy_to(online_net)
            if world_size > 1:
                broadcast_module_state(online_net, src=0)
        online_net.eval()

        for s in val.guidance_scales:
            # Fresh RNG per pair, identical across ranks — this is what
            # guarantees pair-invariant *and* world-size-invariant noise
            # (see docstring). Re-seed *inside* the (tag, gs) loop and
            # do *not* fold rank in: every rank draws the same full-N
            # tensor, then slices its own shard out of it.
            val_gen = torch.Generator(device=device).manual_seed(
                cfg.train.seed + step
            )

            # Draw the full-N noise/labels once, then slice per shard
            # + per chunk.  See docstring for why this must not be
            # per-chunk or rank-offset.
            x_init_all = torch.randn(N, C, H, W, generator=val_gen, device=device)
            y_all = torch.randint(0, n_classes, (N,), generator=val_gen, device=device)
            y_null_all = torch.full((N,), null_id, dtype=torch.long, device=device)

            # Reset per-pair fake-side accumulators on every rank; the
            # real-side FID cache is untouched.  IS has no reference
            # distribution, just a single accumulator.
            if val.metrics.fid and fid_metric is not None:
                fid_metric.reset_fake()
            if val.metrics.inception_score and is_metric is not None:
                is_metric.reset()

            # Leading-`k` visual snapshot — collected on rank 0 only,
            # from its own shard.  Because rank 0's shard is
            # ``x_init_all[:per_rank_chunk]``, the leading images of
            # rank 0's output are the *global* leading images.  The
            # grid has at most ``rank-0 shard length`` images: with a
            # very small ``num_samples`` and a very large
            # ``world_size`` this can shrink the grid below
            # ``visual_log_count``.  In the training regime (num_samples
            # in the thousands, world_size <= 8) this cap is inert.
            k = min(val.visual_log_count, N, shard_len) if is_main else 0
            grid_parts: list[torch.Tensor] = []
            grid_collected = 0

            for ci in range(n_chunks):
                start = shard_start + ci * chunk_size
                end = min(start + chunk_size, shard_end)
                x_init_i = x_init_all[start:end]
                y_i = y_all[start:end]
                y_null_i = y_null_all[start:end]

                imgs_i = euler_sample(
                    model=online_net,
                    x_init=x_init_i,
                    y=y_i,
                    y_null=y_null_i,
                    num_steps=val.sampler.num_steps,
                    guidance_scale=float(s),
                )

                if val.metrics.fid and fid_metric is not None:
                    fid_metric.update_fake(imgs_i)
                if val.metrics.inception_score and is_metric is not None:
                    is_metric.update(imgs_i)

                # Accumulate the leading `k` images across chunks on
                # rank 0.  For the common case ``k <= chunk_size`` this
                # is one clone from chunk 0; for ``k > chunk_size`` we
                # fill across multiple chunks. Cost: k * C * H * W
                # floats — tiny.
                if is_main and k > 0 and grid_collected < k:
                    take = min(k - grid_collected, imgs_i.shape[0])
                    grid_parts.append(imgs_i[:take].detach().clone())
                    grid_collected += take

                # Free the chunk's outputs before the next iteration.
                del imgs_i

            # Free the full-N noise buffers before the next (tag, gs) pair
            # allocates its own.
            del x_init_all, y_all, y_null_all

            # ``compute()`` all-reduces internally (sync_on_compute=True),
            # so every rank must call it — otherwise the ranks that
            # skipped the call would leave the collective hanging.
            metrics: dict[str, float] = {}
            if val.metrics.fid and fid_metric is not None:
                fid_val = fid_metric.compute()
                if is_main:
                    metrics[f"val/fid_{tag}_gs{s}"] = fid_val
            if val.metrics.inception_score and is_metric is not None:
                is_mean, is_std = is_metric.compute()
                if is_main:
                    metrics[f"val/is_mean_{tag}_gs{s}"] = is_mean
                    metrics[f"val/is_std_{tag}_gs{s}"] = is_std

            # Grid PNG on disk + wandb (rank 0 only, from its own shard).
            if is_main and grid_parts:
                grid_snapshot = torch.cat(grid_parts, dim=0)
                grid_imgs = grid_snapshot.clamp(-1, 1).add(1).mul(0.5)
                grid = make_grid(grid_imgs, nrow=int(math.sqrt(k)) or 1)
                grid_path = (run_dir / "samples"
                             / f"step{step:09d}_{tag}_gs{s}.png")
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(grid, grid_path)
                if wandb_run is not None:
                    import wandb  # type: ignore
                    metrics[f"val/grid_{tag}_gs{s}"] = wandb.Image(str(grid_path))

            # Terminal line — Train.md §6.3.  Rank 0 only.
            if is_main:
                fid_s = f"FID={metrics.get(f'val/fid_{tag}_gs{s}', float('nan')):.2f} " \
                    if val.metrics.fid else ""
                is_s = f"IS={metrics.get(f'val/is_mean_{tag}_gs{s}', float('nan')):.2f}" \
                    if val.metrics.inception_score else ""
                print(f"[val   step {step:09d}  gs={s} ema={tag}  "
                      f"{fid_s}{is_s}  ({N} samples across {world_size} rank(s), "
                      f"chunks of <= {chunk_size})]", flush=True)

                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)

            # Keep ranks in step before we mutate metric state or
            # broadcast the next EMA tag onto `online_net`.
            if world_size > 1:
                dist.barrier()

    # Restore the live training weights from the snapshot taken at entry.
    # Every rank must restore — the online weights were mutated on all
    # ranks whenever an ema_* tag was broadcast.
    online_net.load_state_dict(online_state)
    online_net.train()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def format_eta(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def train(cfg: TrainConfig, resume_path: str | None) -> None:
    rank, world_size, local_rank, is_main, device = setup_distributed()

    # Per-rank seeding: same seed across ranks would give identical noise
    # in each forward pass; we want disjoint noise per rank.
    set_seed(cfg.train.seed + rank)

    # ---- Run directory (rank 0 only, then barrier) ----
    run_dir = Path(cfg.logging.out_dir) / cfg.logging.run_name
    if is_main:
        (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
        (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    # ---- Freeze YAML text into strings for the checkpoint ----
    train_yaml_text = Path(cfg_path_global).read_text() if cfg_path_global else ""
    model_yaml_text = Path(model_yaml_path_global).read_text() if model_yaml_path_global else ""
    git_sha = get_git_sha()

    # ---- Model ----
    # ``online_net`` is the trainable network — the *online* copy in the
    # online-vs-EMA sense (see doc/Train.md §5 and flow/ema.py).  It is the
    # sole target of every optimiser step; the EMA shadows below track it.
    online_net = build_model(cfg.model).to(device)
    if is_main:
        n_params = sum(p.numel() for p in online_net.parameters())
        print(f"[init] model={cfg.model.arch_name}  params={n_params/1e6:.2f}M", flush=True)

    # ---- EMA copies (rank 0 only) ----
    ema_dict: dict[str, EMA] = {}
    if is_main:
        for d in cfg.ema.decays:
            tag = f"ema_{d}"
            ema_dict[tag] = EMA(online_net, decay=float(d))
            ema_dict[tag].module.to(device)

    # ---- DDP wrap ----
    if world_size > 1:
        online_net = DDP(
            online_net,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    # `online_module` is the *unwrapped* nn.Module beneath any DDP wrap;
    # used wherever we need the raw parameters/buffers (EMA update, ckpt
    # save, resume-time load_state_dict).
    online_module: nn.Module = online_net.module if isinstance(online_net, DDP) else online_net

    # ---- Optimiser + scheduler ----
    optimizer = build_optimizer(cfg.optim, online_net.parameters())
    scheduler = build_lr_scheduler(cfg.optim, cfg.train.total_steps, optimizer)

    # ---- Data ----
    train_ds = CIFAR10Dataset(
        root=cfg.dataset.root,
        split="train",
        augment=cfg.dataset.augment,
    )
    sampler = ResumableDistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.train.seed,
        drop_last=True,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size_per_gpu,
        sampler=sampler,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory and device.type == "cuda",
        drop_last=True,
        persistent_workers=cfg.dataset.num_workers > 0,
    )

    # ---- FID reference cache (rank 0 builds; all ranks load) ----
    # Metrics are constructed on **every** rank because ``compute()`` is
    # a synchronising collective (sync_on_compute=True): if any rank
    # skipped the call, the ranks that entered it would deadlock.  The
    # real-side FID cache is populated once by rank 0 in
    # build_or_load_fid_cache and read by every rank.
    fid_metric: FIDMetric | None = None
    is_metric: InceptionScoreMetric | None = None
    if cfg.validation.metrics.fid:
        # Use a non-augmented copy for FID reference statistics —
        # we want the *distribution* of clean training images.
        ref_ds = CIFAR10Dataset(
            root=cfg.dataset.root, split="train", augment=False,
        )
        fid_metric = build_or_load_fid_cache(
            cfg.validation.fid_ref_stats,
            ref_ds,
            device,
            is_main,
            world_size,
        )
    if cfg.validation.metrics.inception_score:
        is_metric = InceptionScoreMetric(splits=10, device=device)

    # ---- Resume ----
    start_step = 0
    if resume_path is not None:
        payload = load_checkpoint(Path(resume_path))
        # Guard: refuse to silently change architecture mid-run.
        if payload["model_config_yaml"].strip() != model_yaml_text.strip():
            raise RuntimeError(
                f"resume checkpoint's model config does not match the "
                f"on-disk model config; refusing to load. "
                f"(resume path: {resume_path})"
            )
        online_module.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optim_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if is_main:
            for tag, ema in ema_dict.items():
                if tag in payload["ema_states"]:
                    ema.load_state_dict(payload["ema_states"][tag])
                    ema.module.to(device)
        restore_rng(payload["rng_state"])
        start_step = int(payload["step"])
        sampler.set_start_step(start_step, cfg.train.batch_size_per_gpu)
        if is_main:
            print(f"[resume] restored from {resume_path} at step {start_step}",
                  flush=True)

    # ---- wandb ----
    wandb_run = init_wandb(cfg, run_dir) if is_main else None

    # ---- AMP context ----
    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[cfg.train.amp_dtype]
    def autocast_ctx():
        if amp_dtype is torch.float32 or device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    # ---- Training loop ----
    online_net.train()
    step = start_step
    epoch = sampler.epoch  # DistributedSampler stores this
    grad_accum = cfg.train.grad_accum_steps
    log_interval = cfg.train.log_interval

    # Rolling stats for terminal / wandb scalars.  The ``log_`` prefix
    # marks these as logging-only state: they never feed back into
    # training (loss.backward, optimiser.step, EMA update, checkpoint,
    # etc. all use the raw per-step values).  They are also unrelated
    # to the model-weight EMAs tracked in ``ema_dict`` — the shared
    # word "smoothing" refers to a completely different mechanism.
    log_smoothed_loss = None
    log_smoothed_iter_time = None
    log_smoothing_alpha = 0.02

    total_bs_per_gpu = cfg.train.batch_size_per_gpu

    if is_main:
        print(f"[train] start step={step}/{cfg.train.total_steps}  "
              f"world_size={world_size}  device={device}", flush=True)

    train_start_time = time.time()
    step_start_time = time.time()
    accum_counter = 0
    optimizer.zero_grad(set_to_none=True)

    while step < cfg.train.total_steps:
        for x_1, y in loader:
            if step >= cfg.train.total_steps:
                break

            x_1 = x_1.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            B = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            t = torch.rand(B, device=device, dtype=x_1.dtype)
            x_t = interpolant(x_0, x_1, t)
            v_gt = velocity_gt(x_0, x_1)

            # DDP no-sync until the final micro-batch of the accumulation
            # window — halves gradient traffic when grad_accum > 1.
            is_last_micro = (accum_counter + 1) == grad_accum
            sync_ctx = (online_net.no_sync() if isinstance(online_net, DDP) and not is_last_micro
                        else nullcontext())

            with sync_ctx, autocast_ctx():
                # `online_net` is DDP or the raw module; both accept (x, t, y).
                v_pred = online_net(x_t, t, y)
                loss = flow_matching_loss(v_pred, v_gt) / grad_accum

            loss.backward()
            accum_counter += 1

            if not is_last_micro:
                continue
            accum_counter = 0

            # Grad clipping.
            grad_norm_pre = None
            if cfg.optim.grad_clip is not None:
                params = online_net.parameters()
                grad_norm_pre = torch.nn.utils.clip_grad_norm_(
                    params, max_norm=cfg.optim.grad_clip
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA (rank 0 only).
            if is_main:
                for ema in ema_dict.values():
                    ema.update(online_module)

            step += 1

            # Rolling metrics.
            with torch.no_grad():
                loss_val = float(loss.detach() * grad_accum)   # undo the /=
            iter_time = time.time() - step_start_time
            step_start_time = time.time()
            log_smoothed_loss = loss_val if log_smoothed_loss is None else \
                log_smoothed_loss + log_smoothing_alpha * (loss_val - log_smoothed_loss)
            log_smoothed_iter_time = iter_time if log_smoothed_iter_time is None else \
                log_smoothed_iter_time + log_smoothing_alpha * (iter_time - log_smoothed_iter_time)

            # Terminal + wandb logging.
            if is_main and (step % log_interval == 0 or step == cfg.train.total_steps):
                its_per_s = 1.0 / max(1e-9, log_smoothed_iter_time)
                remaining = (cfg.train.total_steps - step) * log_smoothed_iter_time
                lr = scheduler.get_last_lr()[0]
                gn = float(grad_norm_pre.item()) if grad_norm_pre is not None else float("nan")
                samples_per_sec = its_per_s * total_bs_per_gpu * world_size
                print(
                    f"[step {step:09d}/{cfg.train.total_steps}  ep {sampler.epoch}  "
                    f"{its_per_s:.2f} it/s  loss={log_smoothed_loss:.4f}  "
                    f"lr={lr:.2e}  gn={gn:.2f}  eta={format_eta(remaining)}]",
                    flush=True,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": log_smoothed_loss,
                            "train/lr": lr,
                            "train/grad_norm": gn,
                            "train/sec_per_step": log_smoothed_iter_time,
                            "train/samples_per_sec": samples_per_sec,
                            "train/epoch": sampler.epoch,
                        },
                        step=step,
                    )

            # Validation — every rank participates.  The metrics are
            # collective (sync_on_compute=True), and each rank generates
            # its own shard of the ``num_samples`` images (see
            # run_validation for the sharding scheme).  Only rank 0
            # writes files / logs to wandb.
            if (cfg.validation.interval > 0
                    and step % cfg.validation.interval == 0):
                run_validation(
                    step=step,
                    online_net=online_module,
                    ema_dict=ema_dict,
                    cfg=cfg,
                    device=device,
                    run_dir=run_dir,
                    fid_metric=fid_metric,
                    is_metric=is_metric,
                    wandb_run=wandb_run,
                    rank=rank,
                    world_size=world_size,
                    is_main=is_main,
                )
                # Undo any lingering eval() from run_validation; DDP
                # wants train() active.
                online_net.train()
                step_start_time = time.time()

            # Checkpoint (rank 0 only).
            if is_main and step % cfg.train.ckpt_interval == 0:
                save_checkpoint(
                    run_dir / "ckpt" / f"step_{step:09d}.pt",
                    step=step,
                    model=online_module,
                    ema_dict=ema_dict,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler_state=None,
                    train_yaml=train_yaml_text,
                    model_yaml=model_yaml_text,
                    git_sha=git_sha,
                )
                print(f"[ckpt] step {step:09d} saved", flush=True)

        # End of one dataloader epoch — advance the sampler so the next
        # epoch reshuffles deterministically.
        sampler.set_epoch(sampler.epoch + 1)

    # ---- Final validation + checkpoint ----
    # Validation is collective; every rank enters.  Only rank 0 saves
    # the checkpoint.
    run_validation(
        step=step,
        online_net=online_module,
        ema_dict=ema_dict,
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        fid_metric=fid_metric,
        is_metric=is_metric,
        wandb_run=wandb_run,
        rank=rank,
        world_size=world_size,
        is_main=is_main,
    )
    if is_main:
        save_checkpoint(
            run_dir / "ckpt" / f"step_{step:09d}.pt",
            step=step,
            model=online_module,
            ema_dict=ema_dict,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler_state=None,
            train_yaml=train_yaml_text,
            model_yaml=model_yaml_text,
            git_sha=git_sha,
        )
        elapsed = time.time() - train_start_time
        print(f"[train] done in {elapsed/3600:.2f}h", flush=True)
        if wandb_run is not None:
            wandb_run.finish()

    cleanup_distributed()


# ---------------------------------------------------------------------------
# Config loading with CLI overrides + dataset_root override
# ---------------------------------------------------------------------------

# Global paths captured at config-load time so `train()` can freeze the
# YAML texts into checkpoints without re-parsing arg paths.
cfg_path_global: str | None = None
model_yaml_path_global: str | None = None


def load_and_prepare_config(args: argparse.Namespace) -> TrainConfig:
    """Read + validate the train YAML, applying CLI overrides."""
    global cfg_path_global, model_yaml_path_global

    train_yaml_path = Path(args.config).resolve()
    cfg_path_global = str(train_yaml_path)

    # Apply overrides *before* schema validation so typos in --override
    # are caught the same way YAML typos are.
    raw = _load_yaml(train_yaml_path)
    if args.dataset_root is not None:
        raw.setdefault("dataset", {})["root"] = args.dataset_root
    apply_overrides(raw, list(args.override))

    # Roundtrip: apply_overrides mutates the raw dict, but load_train_config
    # takes a filesystem path.  Write to a tmp file only if we mutated.
    # Easier alternative: re-implement the resolve step here by hand.
    if "model_config" not in raw:
        raise ConfigError(f"{train_yaml_path}: missing required field 'model_config'")
    model_path_raw = raw["model_config"]
    model_yaml_path = (train_yaml_path.parent / model_path_raw).resolve()
    model_yaml_path_global = str(model_yaml_path)

    from configs.schema import ModelConfig
    model_data = _load_yaml(model_yaml_path)
    model_cfg = _from_dict_strict(ModelConfig, model_data, path="model")
    cfg = _from_dict_strict(TrainConfig, {**raw, "model": None}, path="")
    object.__setattr__(cfg, "model", model_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_and_prepare_config(args)
    train(cfg, resume_path=args.resume)


if __name__ == "__main__":
    main()

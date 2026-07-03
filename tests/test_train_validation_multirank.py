"""Multi-rank ``run_validation`` — sharding + weight broadcast regression.

Rationale
---------
``run_validation`` in :mod:`train` shards the ``num_samples`` generation
budget across ranks: with ``world_size=W`` each rank draws the *same*
full-N noise tensor (identical seeds, no rank offset) and processes a
contiguous slice ``x_all[r*ceil(N/W) : (r+1)*ceil(N/W)]``.  For that to
be numerically identical to a ``world_size=1`` run:

1.  The full-N noise must be drawn identically on every rank
    (no rank folded into the seed).
2.  The leading-``k`` grid — collected on rank 0 — must be the leading
    ``k`` of ``x_all``, since rank 0's shard *is* the leading shard.
3.  For ``ema_*`` tags, rank 0 copies its EMA shadow onto ``online_net``
    and broadcasts every parameter and buffer to the other ranks — so
    every rank samples with the same weights.

This test spawns ``world_size=2`` on CPU with the ``gloo`` backend and
asserts the *grid tensor* passed to :func:`torchvision.utils.save_image`
by rank 0 equals the single-rank grid captured by the existing
``test_run_validation_chunk_invariant`` fixture, bit-for-bit.  It runs
FID/IS off (they download Inception on first use, which is inappropriate
in an offline test); torchmetrics' own test suite covers the
``sync_on_compute=True`` path.

The multi-rank path exercises:

*   :func:`train._broadcast_module_state` (the ema_* tag broadcast).
*   The sharded chunk loop in :func:`train.run_validation`.
*   The dist.barrier at the end of each ``(tag, gs)`` pair.

Cost: ~5s process spawn on a warm host — comparable to the existing
smoke test.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Reuse the fixture-class count and model config from the existing smoke.
NUM_FIXTURE_CLASSES = 4


def _make_val_cfg(batch_size_per_gpu: int, num_samples: int, decays: tuple = ()):
    """Build a minimal cfg-like object mirroring tests/test_train_smoke.py.

    We include a non-empty ``ema.decays`` here so the test also exercises
    the EMA-copy + weight-broadcast path — that's the codepath most
    likely to diverge across ranks if the broadcast is wrong.
    """
    return SimpleNamespace(
        validation=SimpleNamespace(
            num_samples=num_samples,
            batch_size_per_gpu=batch_size_per_gpu,
            # Cap at 3 so rank 0's shard (=3 out of 6 with WS=2) can
            # supply a full grid; single-rank will produce the same
            # 3-image grid.  This avoids an artefact of a very small
            # test-scale ``num_samples``: at WS=1 rank 0 could log all
            # 6, but at WS=2 rank 0's shard only holds 3.  The
            # per-shard cap is documented in run_validation.
            visual_log_count=3,
            guidance_scales=(1.0,),
            metrics=SimpleNamespace(fid=False, inception_score=False),
            sampler=SimpleNamespace(num_steps=2),
            fid_ref_stats=None,
            interval=1,
        ),
        model=SimpleNamespace(
            arch_name="DiT_S_2",
            input_size=8,
            in_channels=3,
            patch_size=2,
            num_classes=NUM_FIXTURE_CLASSES,
            class_dropout_prob=0.1,
        ),
        ema=SimpleNamespace(decays=decays),
        train=SimpleNamespace(seed=0),
    )


# ---------------------------------------------------------------------------
# Worker entrypoint (runs inside every spawned process)
# ---------------------------------------------------------------------------

def _worker(
    rank: int,
    world_size: int,
    ref_state_bytes: bytes,          # pickled state_dict — mp can't pass tensors directly
    ema_state_bytes: bytes | None,   # pickled EMA state_dict, None if no ema in test
    tmp_path_str: str,
    result_queue: mp.Queue,
    port: int,
    num_samples: int,
    batch_size_per_gpu: int,
    with_ema: bool,
) -> None:
    """Rank-``rank`` worker: init dist, run validation, ship the grid back."""
    import io

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import train as train_module
        from configs.schema import ModelConfig
        from flow.ema import EMA

        model_cfg = ModelConfig(
            arch_name="DiT_S_2",
            input_size=8,
            in_channels=3,
            patch_size=2,
            num_classes=NUM_FIXTURE_CLASSES,
            class_dropout_prob=0.1,
        )
        net = train_module.build_model(model_cfg)
        net.load_state_dict(
            torch.load(io.BytesIO(ref_state_bytes), map_location="cpu",
                       weights_only=False)
        )

        # EMA shadows live on rank 0 only, matching the training loop's
        # topology.  Every other rank has an empty ema_dict.
        ema_dict: dict[str, EMA] = {}
        if with_ema and rank == 0:
            assert ema_state_bytes is not None
            ema = EMA(net, decay=0.99)
            ema.load_state_dict(
                torch.load(io.BytesIO(ema_state_bytes), map_location="cpu",
                           weights_only=False)
            )
            ema_dict["ema_0.99"] = ema

        cfg_like = _make_val_cfg(
            batch_size_per_gpu=batch_size_per_gpu,
            num_samples=num_samples,
            decays=(0.99,) if with_ema else (),
        )

        captured: dict[str, torch.Tensor] = {}
        real_save = train_module.save_image

        def _capture_save_image(tensor, path, *args, **kwargs):
            captured.setdefault("grids", []).append(tensor.detach().clone())
            return real_save(tensor, path, *args, **kwargs)

        # Only rank 0 ever calls save_image inside run_validation, so the
        # capture on other ranks is a harmless no-op.
        train_module.save_image = _capture_save_image
        try:
            train_module.run_validation(
                step=0,
                online_net=net,
                ema_dict=ema_dict,
                cfg=cfg_like,
                device=torch.device("cpu"),
                run_dir=Path(tmp_path_str),
                fid_metric=None,
                is_metric=None,
                wandb_run=None,
                rank=rank,
                world_size=world_size,
                is_main=(rank == 0),
            )
        finally:
            train_module.save_image = real_save

        if rank == 0:
            grids = captured.get("grids", [])
            # Push a list of grid tensors (one per (tag, gs) pair) back
            # to the parent for bit-identity comparison.  Serialise via
            # torch.save into a bytes buffer — mp.Queue can't ship
            # tensors on some backends.
            buf = io.BytesIO()
            torch.save(grids, buf)
            result_queue.put(("ok", buf.getvalue()))
        else:
            # Non-zero ranks push a sentinel so the parent's queue
            # drain always terminates — otherwise it would block
            # waiting for a message that never comes.
            result_queue.put(("done", b""))
    except BaseException as e:  # pragma: no cover — surfaces failures cleanly
        import traceback
        result_queue.put(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Baseline (world_size=1) captured in-process
# ---------------------------------------------------------------------------

def _run_single_rank(
    ref_state,
    ema_state: dict | None,
    tmp_path: Path,
    num_samples: int,
    batch_size_per_gpu: int,
) -> list[torch.Tensor]:
    """Run ``run_validation`` on the current process with world_size=1
    and return the list of grid tensors it emitted."""
    import train as train_module
    from configs.schema import ModelConfig
    from flow.ema import EMA

    model_cfg = ModelConfig(
        arch_name="DiT_S_2",
        input_size=8,
        in_channels=3,
        patch_size=2,
        num_classes=NUM_FIXTURE_CLASSES,
        class_dropout_prob=0.1,
    )
    net = train_module.build_model(model_cfg)
    net.load_state_dict(ref_state)

    ema_dict: dict[str, EMA] = {}
    if ema_state is not None:
        ema = EMA(net, decay=0.99)
        ema.load_state_dict(ema_state)
        ema_dict["ema_0.99"] = ema

    cfg_like = _make_val_cfg(
        batch_size_per_gpu=batch_size_per_gpu,
        num_samples=num_samples,
        decays=(0.99,) if ema_state is not None else (),
    )

    captured: list[torch.Tensor] = []
    real_save = train_module.save_image

    def _capture(tensor, path, *args, **kwargs):
        captured.append(tensor.detach().clone())
        return real_save(tensor, path, *args, **kwargs)

    train_module.save_image = _capture
    try:
        train_module.run_validation(
            step=0,
            online_net=net,
            ema_dict=ema_dict,
            cfg=cfg_like,
            device=torch.device("cpu"),
            run_dir=tmp_path,
            fid_metric=None,
            is_metric=None,
            wandb_run=None,
            rank=0,
            world_size=1,
            is_main=True,
        )
    finally:
        train_module.save_image = real_save
    return captured


# ---------------------------------------------------------------------------
# Test: world_size=2 on CPU (gloo) matches world_size=1 bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("with_ema", [False, True], ids=["online_only", "with_ema"])
def test_run_validation_multirank_matches_single_rank(
    tmp_path: Path,
    with_ema: bool,
) -> None:
    """world_size=2 must produce the same rank-0 grids as world_size=1.

    * ``num_samples=6`` with ``batch_size_per_gpu=2`` and ``world_size=2``
      partitions as [0..3, 3..6] with 2 chunks of 2 + a 1-chunk tail on
      each rank — the shard math is non-trivial so this catches sharding
      off-by-ones.
    * The ``with_ema=True`` case additionally exercises the EMA
      copy_to → broadcast path in :func:`train._broadcast_module_state`.
    """
    import io
    import train as train_module   # noqa: F401  (ensures import errors surface here)
    from flow.ema import EMA
    from configs.schema import ModelConfig

    # Build a fixed reference model + EMA snapshot up front, so both
    # the single-rank baseline and the multi-rank workers run with
    # identical weights.
    model_cfg = ModelConfig(
        arch_name="DiT_S_2",
        input_size=8,
        in_channels=3,
        patch_size=2,
        num_classes=NUM_FIXTURE_CLASSES,
        class_dropout_prob=0.1,
    )
    torch.manual_seed(0)
    net = train_module.build_model(model_cfg)
    ref_state = copy.deepcopy(net.state_dict())

    ema_state = None
    if with_ema:
        ema = EMA(net, decay=0.99)
        # Perturb the EMA slightly so its weights differ from `net`;
        # otherwise "ema_0.99" and "online" would produce the same
        # samples and a broken broadcast could silently pass.
        with torch.no_grad():
            for p in ema.module.parameters():
                p.mul_(0.5)
        ema_state = ema.state_dict()

    # ---- Baseline: world_size=1 in-process ----
    baseline_grids = _run_single_rank(
        ref_state=ref_state,
        ema_state=ema_state,
        tmp_path=tmp_path / "single",
        num_samples=6,
        batch_size_per_gpu=2,
    )
    # One grid per (tag, gs) pair.  With ema.decays=() → 1 tag, else 2.
    expected_grids = 2 if with_ema else 1
    assert len(baseline_grids) == expected_grids, (
        f"single-rank baseline emitted {len(baseline_grids)} grids, "
        f"expected {expected_grids}"
    )

    # ---- Multi-rank: world_size=2 via mp.spawn (gloo) ----
    # Serialise weights for the workers.  torch.save→BytesIO is enough:
    # mp.Queue can carry bytes cleanly on all start methods.
    ref_buf = io.BytesIO()
    torch.save(ref_state, ref_buf)

    ema_buf_bytes: bytes | None = None
    if ema_state is not None:
        ema_buf = io.BytesIO()
        torch.save(ema_state, ema_buf)
        ema_buf_bytes = ema_buf.getvalue()

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()

    # Pick a port unlikely to collide with other tests running in
    # parallel.  ``0`` would let the OS choose but gloo needs an
    # explicit MASTER_PORT — we use a fixed non-privileged port and
    # rely on the test not being run twice concurrently on one host.
    port = 29500 + (os.getpid() % 1000)

    procs = []
    for r in range(2):
        p = ctx.Process(
            target=_worker,
            args=(
                r, 2,
                ref_buf.getvalue(),
                ema_buf_bytes,
                str(tmp_path / f"multi_rank{r}"),
                result_queue,
                port,
                6,           # num_samples
                2,           # batch_size_per_gpu
                with_ema,
            ),
        )
        p.start()
        procs.append(p)

    # Collect one result per worker.  Rank 0 pushes the grids under
    # tag "ok", rank 1 pushes a "done" sentinel.  Any rank pushes
    # "err" on exception.
    tag = None
    payload: bytes | str = b""
    got_rank0 = False
    for _ in range(2):
        try:
            entry = result_queue.get(timeout=120)
        except Exception:
            break
        if entry[0] == "err":
            for p in procs:
                p.terminate()
            pytest.fail(f"worker raised: {entry[1]}")
        if entry[0] == "ok":
            tag, payload = entry
            got_rank0 = True

    for p in procs:
        p.join(timeout=30)
        assert not p.is_alive(), "worker did not terminate"
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"

    assert got_rank0 and tag == "ok" and payload, "no grids received from rank 0"
    import io as _io
    multi_grids = torch.load(_io.BytesIO(payload), map_location="cpu",
                             weights_only=False)

    assert len(multi_grids) == len(baseline_grids), (
        f"multi-rank rank-0 emitted {len(multi_grids)} grids, "
        f"expected {len(baseline_grids)} (one per (tag, gs) pair)"
    )
    for i, (g_single, g_multi) in enumerate(zip(baseline_grids, multi_grids)):
        assert torch.equal(g_single, g_multi), (
            f"grid {i} differs between world_size=1 and world_size=2 — "
            f"sharding/broadcast has diverged the numerical output. "
            f"max abs diff = "
            f"{(g_single.float() - g_multi.float()).abs().max().item()}"
        )

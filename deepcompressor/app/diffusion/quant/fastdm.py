# -*- coding: utf-8 -*-
"""FastDM block-wise AdaRound calibration for diffusion models.

This is the diffusers-side driver: ``fastdm_diffusion(model, config)`` is
intended to be called from ``ptq.py`` between ``smooth_diffusion`` and
``quantize_diffusion_weights``. It runs gradient-based reconstruction
(AdaRound) on every transformer/UNet block detected by
``DiffusionModelStruct``, then bakes the learned rounding back into the
underlying ``nn.Linear`` / ``nn.Conv*`` weights so the static qtbench
weight calibration that follows operates on AdaRound-friendly tensors.

The full algorithm mirrors ``Fast_DM_PTQ.quant.calibration.cali_model``:

1. Walk the qdiff cache and assemble per-sample timesteps.
2. Optionally run a one-shot TIB (time-embedding) reconstruction.
3. For each progressive loop (controlled by ``progressive`` /
   ``timestep_group``):
     a. Capture FP inputs/outputs of every target block via forward hooks.
     b. Per block, call :func:`reconstruct_module` over the cached IO.
4. Bake AdaRound into ``module.weight`` and return the alpha state dict.
"""

from __future__ import annotations

import logging
import os
import typing as tp
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from deepcompressor.calib.recon import (
    AdaRoundQuantizer,
    LossRecorder,
    ReconLossKind,
    ReconState,
    RtnLowRankParametrize,
    StopForwardException,
    attach_adaround,
    attach_lowrank,
    build_lowrank_branch_state,
    detach_adaround,
    detach_lowrank,
    iter_adaround_quantizers,
    iter_lowrank_parametrizes,
    make_widths_fixedK,
    make_widths_from_adjacent_dist,
    plan_loop_sizes,
    reconstruct_module,
    reconstruct_module_lowrank,
    reconstruct_tib,
)
from deepcompressor.utils import tools

from ..nn.struct import DiffusionBlockStruct, DiffusionModelStruct
from .config import DiffusionQuantConfig
from .fastdm_data import FastDmSample, extract_timesteps, iter_fastdm_samples
from .fastdm_tib import build_tib_bundle

__all__ = ["fastdm_diffusion"]

logger = tools.logging.getLogger(__name__)


_PARAMETRIZED_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


@dataclass
class _BlockTarget:
    name: str
    module: nn.Module


@dataclass
class _BlockCache:
    inputs: list[tuple[torch.Tensor, ...]]
    input_kwargs: list[dict[str, tp.Any]]
    outputs: list[torch.Tensor]


def _detach_cpu_tree(value: tp.Any) -> tp.Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _detach_cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [_detach_cpu_tree(v) for v in value]
    return value


def _move_tree(value: tp.Any, device: torch.device) -> tp.Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_tree(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tree(v, device) for v in value)
    if isinstance(value, list):
        return [_move_tree(v, device) for v in value]
    return value


def _same_non_tensor(values: list[tp.Any]) -> bool:
    first = values[0]
    return all(v == first for v in values[1:])


def _merge_per_sample_values(values: list[tp.Any]) -> tp.Any:
    first = values[0]
    if torch.is_tensor(first) and all(torch.is_tensor(v) for v in values):
        tensors = tp.cast(list[torch.Tensor], values)
        if all(t.ndim == 0 for t in tensors):
            return torch.stack(tensors, dim=0)
        if all(t.ndim > 0 and t.shape[0] == 1 and t.shape[1:] == tensors[0].shape[1:] for t in tensors):
            return torch.cat(tensors, dim=0)
        if all(torch.equal(t, tensors[0]) for t in tensors[1:]):
            return tensors[0]
        return tensors
    if isinstance(first, dict) and all(isinstance(v, dict) for v in values):
        keys = set(first)
        for v in values[1:]:
            keys &= set(v)
        return {k: _merge_per_sample_values([v[k] for v in values]) for k in keys}
    if isinstance(first, tuple) and all(isinstance(v, tuple) and len(v) == len(first) for v in values):
        return tuple(_merge_per_sample_values([v[i] for v in values]) for i in range(len(first)))
    if isinstance(first, list) and all(isinstance(v, list) and len(v) == len(first) for v in values):
        return [_merge_per_sample_values([v[i] for v in values]) for i in range(len(first))]
    if _same_non_tensor(values):
        return first
    return values


def _merge_per_sample_kwargs(kwargs_per_sample: list[dict[str, tp.Any]]) -> dict[str, tp.Any] | None:
    if not kwargs_per_sample:
        return None
    keys = set(kwargs_per_sample[0])
    for kwargs in kwargs_per_sample[1:]:
        keys &= set(kwargs)
    return {k: _merge_per_sample_values([kwargs[k] for kwargs in kwargs_per_sample]) for k in keys}


class _CaptureHook:
    """Forward hook that snapshots a single block's inputs/outputs per call."""

    def __init__(self, sink: _BlockCache) -> None:
        self.sink = sink

    def __call__(self, module: nn.Module, inputs: tuple, kwargs: dict, output) -> None:
        ipts = tuple(_detach_cpu_tree(t) for t in inputs)
        self.sink.inputs.append(ipts)
        self.sink.input_kwargs.append(_detach_cpu_tree(kwargs))
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        if torch.is_tensor(out):
            self.sink.outputs.append(out.detach().cpu())
        else:
            raise RuntimeError(
                f"fastdm: block '{module.__class__.__name__}' returned non-tensor output of type {type(output)}"
            )


def _collect_target_blocks(model_struct: DiffusionModelStruct) -> list[_BlockTarget]:
    """List every transformer/UNet block we should run AdaRound over."""
    targets: list[_BlockTarget] = []
    for block_struct in model_struct.block_structs:
        if not isinstance(block_struct, DiffusionBlockStruct):
            continue
        if block_struct.module is None:
            continue
        # Only blocks with at least one parametrizable child are useful.
        has_target_child = any(
            isinstance(sub, _PARAMETRIZED_TYPES) for sub in block_struct.module.modules()
        )
        if not has_target_child:
            continue
        targets.append(_BlockTarget(name=block_struct.name, module=block_struct.module))
    return targets


def _capture_fp_io(
    model: nn.Module,
    targets: list[_BlockTarget],
    samples: list[FastDmSample],
) -> dict[str, _BlockCache]:
    """Run every sample through the FP model and capture each target's IO."""
    caches: dict[str, _BlockCache] = {
        t.name: _BlockCache(inputs=[], input_kwargs=[], outputs=[]) for t in targets
    }
    handles = []
    for t in targets:
        h = t.module.register_forward_hook(_CaptureHook(caches[t.name]), with_kwargs=True)
        handles.append(h)
    try:
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            for sample in samples:
                try:
                    args = tuple(_move_tree(arg, device) for arg in sample.args)
                    kwargs = {k: _move_tree(v, device) for k, v in sample.kwargs.items()}
                    model(*args, **kwargs)
                except StopForwardException:
                    pass
    finally:
        for h in handles:
            h.remove()
    return caches


def _stack_block_io(
    cache: _BlockCache,
    keep_gpu: bool,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict | None]:
    """Concatenate per-sample inputs/outputs along dim 0 for the given block."""
    if not cache.inputs or not cache.outputs:
        raise RuntimeError("fastdm: block cache is empty -- did the forward hook fire?")
    n_args = len(cache.inputs[0])
    cat_inputs: list[torch.Tensor] = []
    for j in range(n_args):
        per_arg = [t[j] for t in cache.inputs if torch.is_tensor(t[j])]
        if not per_arg:
            raise RuntimeError(
                f"fastdm: positional input #{j} is non-tensor -- not supported by AdaRound"
            )
        cat_inputs.append(torch.cat(per_arg, dim=0))
    cat_outputs = torch.cat(cache.outputs, dim=0)

    if keep_gpu:
        cat_inputs = [t.to(device, non_blocking=True) for t in cat_inputs]
        cat_outputs = cat_outputs.to(device, non_blocking=True)
    return tuple(cat_inputs), cat_outputs, _merge_per_sample_kwargs(cache.input_kwargs)


def _resolve_progressive_indices(
    cfg,
    timesteps: torch.Tensor,
) -> tuple[list[torch.Tensor], list[str]]:
    """Compute per-loop selection indices and human-readable labels.

    Returns ``([indices_loop_0, indices_loop_1, ...], [label_0, label_1, ...])``.
    A single full-set loop is returned if progressive scheduling is disabled.
    """
    n_total = timesteps.numel()
    if not cfg.progressive.is_progressive:
        return [torch.arange(n_total, dtype=torch.long)], ["full"]
    direction = cfg.progressive.direction

    t_sorted, order = torch.sort(timesteps)
    uniq, counts = torch.unique_consecutive(t_sorted, return_counts=True)
    U = int(uniq.numel())

    if cfg.timestep_group.enable and cfg.timestep_group.mode == "adaptive":
        if cfg.timestep_group.feature_dist_path and os.path.exists(
            cfg.timestep_group.feature_dist_path
        ):
            adj = np.load(cfg.timestep_group.feature_dist_path)
            widths = make_widths_from_adjacent_dist(
                adj, K=cfg.timestep_group.num_groups, mode="adaptive"
            )
        else:
            logger.warning(
                "fastdm: adaptive grouping requested but feature_dist_path is missing; "
                "falling back to uniform grouping"
            )
            widths = make_widths_fixedK(
                U, min(cfg.timestep_group.num_groups, U), mode="gradual"
            )
    elif cfg.timestep_group.enable:
        widths = make_widths_fixedK(
            U, min(cfg.timestep_group.num_groups, U), mode="gradual"
        )
    else:
        # No grouping: each unique timestep is its own bin.
        widths = torch.ones(U, dtype=torch.int64)

    n_list = plan_loop_sizes(uniq, counts, n_total, direction, widths)
    if not n_list:
        return [torch.arange(n_total, dtype=torch.long)], ["full"]

    # Convert per-loop sizes back into indices into the *original* sample order.
    # ``order`` maps sorted positions -> original indices; for the reverse
    # progressive direction the k-th loop sees the last n_list[k] entries of
    # the sorted order; for the forward direction the first n_list[k].
    indices: list[torch.Tensor] = []
    labels: list[str] = []
    for k, n in enumerate(n_list):
        if direction == "reverse":
            sel = order[n_total - n : n_total]
            t_lo = float(t_sorted[n_total - n].item())
            t_hi = float(t_sorted[n_total - 1].item())
        else:
            sel = order[:n]
            t_lo = float(t_sorted[0].item())
            t_hi = float(t_sorted[n - 1].item())
        indices.append(sel.to(torch.long))
        labels.append(f"k={k+1}/{len(n_list)}|t∈[{t_lo:g},{t_hi:g}]|N={n}")
    return indices, labels


def _maybe_save_loss_curves(recorder: LossRecorder | None, out_dir: str) -> None:
    if recorder is None or not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    for block_key, store in recorder.store.items():
        if not store.get("err"):
            continue
        safe = block_key.replace(os.sep, "_")
        np.save(os.path.join(out_dir, f"{safe}.err.npy"), np.asarray(store["err"]))
        np.save(os.path.join(out_dir, f"{safe}.rec.npy"), np.asarray(store["rec"]))
        np.save(os.path.join(out_dir, f"{safe}.round.npy"), np.asarray(store["round"]))


def _bake_alpha_state(targets: list[_BlockTarget]) -> dict[str, torch.Tensor]:
    """After AdaRound: collect each module's alpha tensor for cache I/O."""
    state: dict[str, torch.Tensor] = {}
    for t in targets:
        for name, sub in t.module.named_modules():
            for adar in iter_adaround_quantizers(sub):
                if adar.alpha is None:
                    continue
                state[f"{t.name}.{name}.alpha"] = adar.alpha.detach().cpu().clone()
                # One AdaRound per module in our setup; break early.
                break
    return state


def _apply_alpha_state(
    targets: list[_BlockTarget],
    state: dict[str, torch.Tensor],
) -> None:
    """Re-attach AdaRound from a cached alpha state dict and bake the weights."""
    for t in targets:
        for name, sub in t.module.named_modules():
            if not isinstance(sub, _PARAMETRIZED_TYPES):
                continue
            key = f"{t.name}.{name}.alpha"
            if key not in state:
                continue
            attach_adaround(sub, bits=4)  # bits is overwritten below if alpha shape mismatches
            adar = next(iter(iter_adaround_quantizers(sub)), None)
            if adar is not None and adar.alpha is not None and adar.alpha.shape == state[key].shape:
                with torch.no_grad():
                    adar.alpha.copy_(state[key].to(adar.alpha.device))
                adar.freeze_to_hard()
                detach_adaround(sub)


@torch.enable_grad()
def fastdm_diffusion(  # noqa: C901
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Run FastDM block reconstruction on a diffusion backbone.

    Dispatches on ``config.fastdm.recipe``:

    * ``"adaround"`` (default): trains AdaRound rounding offsets (the
      original Fast_DM_PTQ behaviour). Returns ``{block.sub.alpha:
      tensor}``.
    * ``"lowrank"``: trains a SVDQuant-style low-rank branch per
      Linear/Conv to compensate for fixed-RTN quantization error. The
      trained branches are written into ``cache.path.branch`` so the
      downstream ``quantize_diffusion_weights`` consumes them in place of
      its own analytic SVD. Returns ``{block.sub.<a|b>:tensor}`` plus a
      ``__recipe__`` marker.

    Args:
        model: A ``DiffusionModelStruct`` or a raw ``nn.Module`` that
            ``DiffusionModelStruct.construct`` accepts.
        config: The full ``DiffusionQuantConfig``; ``config.fastdm`` holds
            the recipe + scheduling settings, ``config.calib`` points at
            the qdiff cache directory.
        cache: Optional pre-computed state (from a previous run); when
            provided the function bypasses optimisation and only re-applies
            the saved state to the model.

    Returns:
        Recipe-specific cache dict suitable for ``torch.save`` and reuse.
    """
    if config.fastdm is None or not config.fastdm.is_enabled():
        raise RuntimeError(
            "fastdm_diffusion called without an enabled fastdm config"
        )
    fcfg = config.fastdm

    if not isinstance(model, DiffusionModelStruct):
        model_struct = DiffusionModelStruct.construct(model)
    else:
        model_struct = model
    raw_model = model_struct.module
    assert isinstance(raw_model, nn.Module)

    targets = _collect_target_blocks(model_struct)
    if not targets:
        logger.warning("fastdm: no target blocks found, skipping calibration")
        return {}
    logger.info("fastdm: %d target blocks identified", len(targets))

    if cache is not None:
        recipe_in_cache = cache.get("__recipe__", "adaround")
        if isinstance(recipe_in_cache, bytes):
            recipe_in_cache = recipe_in_cache.decode()
        if recipe_in_cache != fcfg.recipe:
            logger.warning(
                "fastdm: cached state was produced with recipe %r but config asks for %r; "
                "ignoring cache and re-running calibration",
                recipe_in_cache,
                fcfg.recipe,
            )
        else:
            if fcfg.recipe == "lowrank":
                logger.info(
                    "fastdm[lowrank]: cache reload is a no-op; downstream "
                    "quantize_diffusion_weights will reload branches from "
                    "cache.path.branch"
                )
                return cache
            logger.info("fastdm: applying cached alpha state to %d blocks", len(targets))
            _apply_alpha_state(targets, cache)
            return cache

    # ---- load qdiff calibration cache as FastDM samples ----
    loader = config.calib.build_loader()
    samples: list[FastDmSample] = list(iter_fastdm_samples(loader))
    if not samples:
        logger.warning("fastdm: empty calibration set, skipping calibration")
        return {}
    logger.info("fastdm: %d calibration samples loaded", len(samples))
    timesteps = extract_timesteps(samples)
    device = next(raw_model.parameters()).device

    # ---- optional TIB pass (AdaRound recipe only) ----
    if fcfg.recipe == "adaround" and fcfg.tib.enable:
        tib = build_tib_bundle(raw_model)
        if tib is not None:
            logger.info("fastdm: running TIB reconstruction (%d iters)", fcfg.tib.iters)
            try:
                # Build a (T,) tensor of timestep values for TIB calibration.
                t_tensor = timesteps.to(device).to(torch.float32)
                cali_tib = (t_tensor,)

                def _tib_forward(_model, batch):
                    _model(*batch)

                reconstruct_tib(
                    tib,
                    cali_tib,
                    forward_fn=_tib_forward,
                    bits=fcfg.adaround.bits,
                    symmetric=fcfg.adaround.symmetric,
                    iters=min(fcfg.tib.iters, max(1, len(samples))),
                    batch_size=min(fcfg.progressive.batch_size, len(samples)),
                    lr=fcfg.tib.lr,
                    weight=fcfg.adaround.weight_loss_w,
                    b_range=fcfg.adaround.b_range,
                    warmup=fcfg.adaround.warmup,
                    decay_start=fcfg.adaround.decay_start,
                    p=fcfg.adaround.p,
                    log_label="tib",
                )
                # TIB modules are referenced inside transformer blocks; bake
                # their AdaRound now so subsequent block recon sees the
                # final dequantized weights.
                for member in tib.members():
                    for sub in member.modules():
                        if isinstance(sub, _PARAMETRIZED_TYPES) and parametrize.is_parametrized(
                            sub, "weight"
                        ):
                            detach_adaround(sub)
            except Exception:
                logger.exception("fastdm: TIB reconstruction failed; continuing without it")

    # ---- progressive scheduling ----
    indices_per_loop, labels = _resolve_progressive_indices(fcfg, timesteps)
    num_loops = len(indices_per_loop)
    logger.info("fastdm: %d progressive loop(s): %s", num_loops, labels)

    # ---- capture FP IO across the full calibration set ONCE ----
    logger.info("fastdm: capturing FP block IO across %d samples", len(samples))
    block_caches = _capture_fp_io(raw_model, targets, samples)

    # ---- per-loop block reconstruction ----
    state_registry = ReconState()
    recorder = LossRecorder() if fcfg.log_loss_curves else None

    keep_gpu = fcfg.keep_gpu
    for loop_idx in range(num_loops):
        sel = indices_per_loop[loop_idx]
        is_first = loop_idx == 0
        is_last = loop_idx == num_loops - 1
        logger.info(
            "fastdm: progressive loop %d/%d (%s)",
            loop_idx + 1,
            num_loops,
            labels[loop_idx],
        )
        for t in targets:
            cache_block = block_caches[t.name]
            cat_in, cat_out, cap_kw = _stack_block_io(cache_block, keep_gpu, device)
            try:
                if fcfg.recipe == "lowrank":
                    reconstruct_module_lowrank(
                        t.module,
                        cached_inputs=cat_in,
                        cached_outputs=cat_out,
                        captured_kwargs=cap_kw,
                        state_registry=state_registry,
                        rank=fcfg.lowrank.rank,
                        bits=fcfg.lowrank.bits,
                        symmetric=fcfg.lowrank.symmetric,
                        ste_rtn=fcfg.lowrank.ste_rtn,
                        epoch=fcfg.progressive.epoch_per_loop,
                        batch_size=fcfg.progressive.batch_size,
                        lr=fcfg.lowrank.lr,
                        weight_decay=fcfg.lowrank.weight_decay,
                        is_first_loop=is_first,
                        is_last_loop=is_last,
                        log_label=t.name,
                        loss_recorder=recorder,
                        block_key=t.name,
                        select_indices=sel,
                    )
                else:
                    reconstruct_module(
                        t.module,
                        cached_inputs=cat_in,
                        cached_outputs=cat_out,
                        captured_kwargs=cap_kw,
                        state_registry=state_registry,
                        bits=fcfg.adaround.bits,
                        symmetric=fcfg.adaround.symmetric,
                        epoch=fcfg.progressive.epoch_per_loop,
                        batch_size=fcfg.progressive.batch_size,
                        weight=fcfg.adaround.block_loss_w,
                        rec_loss_kind=ReconLossKind.MSE,
                        b_range=fcfg.adaround.b_range,
                        warmup=fcfg.adaround.warmup,
                        decay_start=fcfg.adaround.decay_start,
                        p=fcfg.adaround.p,
                        is_first_loop=is_first,
                        is_last_loop=is_last,
                        log_label=t.name,
                        loss_recorder=recorder,
                        block_key=t.name,
                        select_indices=sel,
                    )
            except Exception:
                logger.exception("fastdm: block reconstruction failed for %s", t.name)
                # Don't abort the whole run for one bad block -- detach
                # whatever parametrize was attached so the model stays usable.
                attached = [
                    sub
                    for sub in list(t.module.modules())
                    if isinstance(sub, _PARAMETRIZED_TYPES)
                    and parametrize.is_parametrized(sub, "weight")
                ]
                if fcfg.recipe == "lowrank":
                    for sub in attached:
                        detach_lowrank(sub)
                else:
                    for sub in attached:
                        detach_adaround(sub)

    if fcfg.recipe == "lowrank":
        # ---- harvest trained branches and detach parametrize ----
        state: dict[str, torch.Tensor] = {"__recipe__": "lowrank"}
        branch_state_dict: dict[str, dict[str, torch.Tensor]] = {}
        for t in targets:
            sub_state = build_lowrank_branch_state(t.module, target_name=t.name)
            branch_state_dict.update(sub_state)
            # Snapshot the parametrized children before mutating the tree
            # via detach_lowrank (which removes parametrizations and the
            # branch sub-modules they hold).
            attached = [
                (sub_name, sub)
                for sub_name, sub in list(t.module.named_modules())
                if isinstance(sub, _PARAMETRIZED_TYPES)
                and parametrize.is_parametrized(sub, "weight")
            ]
            for sub_name, sub in attached:
                full = f"{t.name}.{sub_name}" if sub_name else t.name
                if full in sub_state:
                    for pname, pval in sub_state[full].items():
                        state[f"{full}.{pname}"] = pval
                detach_lowrank(sub)

        # ---- write branch_state_dict to cache.path.branch so the
        #      downstream quantize_diffusion_weights consumes it in place
        #      of its own SVD pass.
        try:
            from ..cache.config import DiffusionPtqCacheConfig  # noqa: F401
        except Exception:  # pragma: no cover - defensive
            pass
        # The caller (ptq.py) holds the cache config; we surface the
        # branch dict via the returned state so ptq.py can route it. To
        # keep the call signature unchanged we stash the dict under a
        # well-known key.
        state["__branch_state_dict__"] = branch_state_dict  # type: ignore[assignment]

        if recorder is not None:
            _maybe_save_loss_curves(recorder, fcfg.loss_curve_dir)

        logger.info(
            "fastdm[lowrank]: calibration done; %d branches in cache",
            len(branch_state_dict),
        )
        return state

    # ---- AdaRound: bake the learned rounding into module.weight ----
    state = _bake_alpha_state(targets)
    state["__recipe__"] = "adaround"  # type: ignore[assignment]
    for t in targets:
        for sub in t.module.modules():
            if isinstance(sub, _PARAMETRIZED_TYPES) and parametrize.is_parametrized(
                sub, "weight"
            ):
                detach_adaround(sub)

    if recorder is not None:
        _maybe_save_loss_curves(recorder, fcfg.loss_curve_dir)

    logger.info("fastdm: calibration done; %d alpha tensors in cache", len(state) - 1)
    return state

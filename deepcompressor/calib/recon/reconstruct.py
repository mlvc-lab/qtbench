# -*- coding: utf-8 -*-
"""Block-wise AdaRound reconstruction for FastDM calibration.

Ported from ``Fast_DM_PTQ/quant/reconstruction.py`` and ``data_utill.py``.
The qtbench port:

* Replaces ``QuantLayer.wqtizer`` with PyTorch parametrizations attached to
  the original ``nn.Linear``/``nn.Conv*`` modules. ``alpha`` is the only
  learnable tensor; weights are kept frozen.
* Drops the ``linklink`` distributed wrapper in favour of a thin
  ``torch.distributed`` shim that no-ops when DDP is uninitialised.
* Holds per-module optimiser state in an external ``ReconState`` registry
  rather than monkey-patching attributes onto the user's modules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.utils.parametrize as parametrize
from torch import nn

from .adaround import AdaRoundQuantizer, RMODE, reset_adam_momentum
from .loss import ReconLoss, ReconLossKind, ReconLossTimeEmbedding

__all__ = [
    "ReconState",
    "attach_adaround",
    "detach_adaround",
    "iter_adaround_quantizers",
    "capture_module_io",
    "capture_module_grad",
    "reconstruct_module",
    "reconstruct_tib",
    "freeze_soft_targets",
    "StopForwardException",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helper
# ---------------------------------------------------------------------------


def _maybe_allreduce(params: Iterable[nn.Parameter]) -> None:
    """All-reduce gradients across ranks if a process group is initialised."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    for p in params:
        if p.grad is None:
            continue
        dist.all_reduce(p.grad)
        p.grad.div_(world_size)


# ---------------------------------------------------------------------------
# AdaRound attachment via parametrize
# ---------------------------------------------------------------------------


_PARAMETRIZED_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def attach_adaround(
    module: nn.Module,
    bits: int,
    symmetric: bool = False,
    always_zero: bool = False,
) -> AdaRoundQuantizer | None:
    """Attach an AdaRound parametrization to ``module.weight``.

    Returns the new ``AdaRoundQuantizer`` (or ``None`` if the module is
    already wrapped, in which case the existing one is left untouched).
    """
    if not isinstance(module, _PARAMETRIZED_TYPES):
        return None
    if parametrize.is_parametrized(module, "weight"):
        existing = next(
            (m for m in module.parametrizations.weight if isinstance(m, AdaRoundQuantizer)),
            None,
        )
        return existing

    weight = module.weight.data
    quantizer = AdaRoundQuantizer(
        weight=weight,
        bits=bits,
        symmetric=symmetric,
        always_zero=always_zero,
        rmode=RMODE.LEARNED_HARD_SIGMOID,
    )
    quantizer.soft_tgt = True
    parametrize.register_parametrization(module, "weight", quantizer, unsafe=True)
    return quantizer


def detach_adaround(module: nn.Module) -> None:
    """Bake the dequantized weight back into ``module.weight`` and remove the parametrization."""
    if not isinstance(module, _PARAMETRIZED_TYPES):
        return
    if not parametrize.is_parametrized(module, "weight"):
        return
    parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


def iter_adaround_quantizers(root: nn.Module) -> Iterable[AdaRoundQuantizer]:
    """Yield every ``AdaRoundQuantizer`` attached anywhere under ``root``."""
    for sub in root.modules():
        if isinstance(sub, AdaRoundQuantizer):
            yield sub


def freeze_soft_targets(root: nn.Module) -> None:
    """Switch every AdaRound under ``root`` to its hard-rounding regime."""
    for adar in iter_adaround_quantizers(root):
        adar.soft_tgt = False


def _select_kwargs_tree(value, idx: torch.Tensor, n_total: int, device: torch.device):
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == n_total:
            value = value.index_select(0, idx.to(value.device))
        return value.to(device)
    if isinstance(value, dict):
        return {k: _select_kwargs_tree(v, idx, n_total, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_select_kwargs_tree(v, idx, n_total, device) for v in value)
    if isinstance(value, list):
        return [_select_kwargs_tree(v, idx, n_total, device) for v in value]
    return value


# ---------------------------------------------------------------------------
# Per-module reconstructor state
# ---------------------------------------------------------------------------


@dataclass
class _ModuleState:
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    opt_params: list[nn.Parameter] = field(default_factory=list)
    loss: ReconLoss | None = None


@dataclass
class ReconState:
    """Cross-loop persistent state for AdaRound reconstruction."""

    by_module: dict[int, _ModuleState] = field(default_factory=dict)

    def get(self, module: nn.Module) -> _ModuleState:
        key = id(module)
        if key not in self.by_module:
            self.by_module[key] = _ModuleState()
        return self.by_module[key]


# ---------------------------------------------------------------------------
# IO capture
# ---------------------------------------------------------------------------


class StopForwardException(Exception):
    """Raised by an IO-capture hook to short-circuit the model forward pass."""


class _DataSaverHook:
    def __init__(self, store_input: bool, store_output: bool, stop_forward: bool) -> None:
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store: tuple | None = None
        self.input_kwargs: dict | None = None
        self.output_store: torch.Tensor | tuple | None = None

    def __call__(self, module: nn.Module, inputs: tuple, kwargs: dict, output) -> None:
        if self.store_input:
            self.input_store = inputs
            self.input_kwargs = kwargs
        if self.store_output:
            self.output_store = output
        if self.stop_forward:
            raise StopForwardException


class _GradSaverHook:
    def __init__(self) -> None:
        self.grad_out: torch.Tensor | None = None

    def __call__(self, module: nn.Module, grad_input, grad_output) -> None:
        self.grad_out = grad_output[0]


def _move_quant_state(model: nn.Module, target: nn.Module, *, model_quant: bool, target_quant: bool) -> None:
    """Toggle AdaRound soft/hard rounding on the model and target.

    For pure FP forward passes (``*_quant=False``) we disable AdaRound on
    the model by removing its parametrization effect. For qtbench's
    parametrize-based AdaRound, the simplest toggle is per-module:
    enable/disable parametrization output. Since ``AdaRoundQuantizer``
    always emits a quantized weight, we swap the parametrization on/off
    via the ``parametrize`` helper.
    """
    # The qtbench port keeps AdaRound parametrizations active throughout
    # reconstruction; the source's per-block "set_quant_state" toggling
    # is unnecessary because we cache FP outputs *before* attaching any
    # AdaRound parametrizations (see ``capture_module_io``).
    return None


def capture_module_io(
    model: nn.Module,
    target: nn.Module,
    cali_data: tuple[torch.Tensor, ...],
    *,
    forward_fn,
    batch_size: int,
    keep_gpu: bool,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | tuple[torch.Tensor, ...], dict | None]:
    """Run ``model`` over ``cali_data`` and capture inputs/outputs of ``target``.

    ``forward_fn(model, batch)`` is invoked once per batch; its return value
    is ignored. The function must run a forward pass that traverses
    ``target`` exactly once per sample.
    """
    device = next(model.parameters()).device
    saver = _DataSaverHook(store_input=True, store_output=True, stop_forward=True)
    handle = target.register_forward_hook(saver, with_kwargs=True)

    cached_inputs: list[list[torch.Tensor]] | None = None
    cached_outputs: list[list[torch.Tensor]] | None = None
    captured_kwargs: dict | None = None

    n_samples = cali_data[0].shape[0]
    model.eval()
    try:
        for i in range(0, n_samples, batch_size):
            batch = tuple(t[i : i + batch_size] for t in cali_data)
            try:
                with torch.no_grad():
                    forward_fn(model, batch)
            except StopForwardException:
                pass
            ipts = saver.input_store or ()
            opts = saver.output_store
            if captured_kwargs is None and saver.input_kwargs is not None:
                captured_kwargs = saver.input_kwargs
            ipts = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in ipts)
            if cached_inputs is None:
                cached_inputs = [[] for _ in ipts]
            for j, x in enumerate(ipts):
                if torch.is_tensor(x):
                    cached_inputs[j].append(x)
            if isinstance(opts, torch.Tensor):
                opts_tuple = (opts.detach().cpu(),)
            else:
                opts_tuple = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in opts)
            if cached_outputs is None:
                cached_outputs = [[] for _ in opts_tuple]
            for j, x in enumerate(opts_tuple):
                if torch.is_tensor(x):
                    cached_outputs[j].append(x)
    finally:
        handle.remove()

    if cached_inputs is None or cached_outputs is None:
        raise RuntimeError(
            "capture_module_io did not record any IO for the target module."
        )

    cat_inputs = tuple(torch.cat(ts, dim=0) for ts in cached_inputs)
    cat_outputs_list = [torch.cat(ts, dim=0) for ts in cached_outputs]
    cat_outputs = cat_outputs_list[0] if len(cat_outputs_list) == 1 else tuple(cat_outputs_list)

    if keep_gpu:
        cat_inputs = tuple(t.to(device, non_blocking=True) for t in cat_inputs)
        if isinstance(cat_outputs, torch.Tensor):
            cat_outputs = cat_outputs.to(device, non_blocking=True)
        else:
            cat_outputs = tuple(t.to(device, non_blocking=True) for t in cat_outputs)
    return cat_inputs, cat_outputs, captured_kwargs


def capture_module_grad(
    model: nn.Module,
    target: nn.Module,
    cali_data: tuple[torch.Tensor, ...],
    *,
    forward_fn,
    batch_size: int,
    keep_gpu: bool,
) -> torch.Tensor:
    """Capture absolute gradients flowing into ``target`` (Fisher proxy)."""
    device = next(model.parameters()).device
    saver = _GradSaverHook()
    handle = target.register_full_backward_hook(saver)

    cached: list[torch.Tensor] = []
    n_samples = cali_data[0].shape[0]
    try:
        for i in range(0, n_samples, batch_size):
            batch = tuple(t[i : i + batch_size] for t in cali_data)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                forward_fn(model, batch)
            cached.append(saver.grad_out.detach().cpu())  # type: ignore[union-attr]
    finally:
        handle.remove()

    out = torch.cat(cached, dim=0).abs() + 1.0
    if keep_gpu:
        out = out.to(device, non_blocking=True)
    return out


# ---------------------------------------------------------------------------
# Reconstruction loops
# ---------------------------------------------------------------------------


def _build_loss(
    target: nn.Module,
    *,
    state: _ModuleState,
    rec_loss_kind: ReconLossKind,
    weight: float,
    max_count: int,
    b_range: tuple,
    decay_start: float,
    warmup: float,
    p: float,
    log_label: str,
) -> ReconLoss:
    if state.loss is not None:
        return state.loss

    def _soft_targets() -> Iterable[torch.Tensor]:
        return [adar.get_soft_tgt() for adar in iter_adaround_quantizers(target)]

    state.loss = ReconLoss(
        soft_targets_fn=_soft_targets,
        round_loss=ReconLossKind.RELAXATION,
        w=weight,
        rec_loss=rec_loss_kind,
        max_count=max_count,
        b_range=b_range,
        decay_start=decay_start,
        warmup=warmup,
        p=p,
        log_label=log_label,
    )
    return state.loss


def reconstruct_module(
    target: nn.Module,
    *,
    cached_inputs: tuple[torch.Tensor, ...],
    cached_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    captured_kwargs: dict | None = None,
    cached_grads: torch.Tensor | None = None,
    state_registry: ReconState,
    bits: int,
    symmetric: bool = False,
    always_zero: bool = False,
    epoch: int = 1,
    batch_size: int = 32,
    weight: float = 0.01,
    rec_loss_kind: ReconLossKind = ReconLossKind.MSE,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    decay_start: float = 0.0,
    p: float = 2.0,
    is_first_loop: bool = False,
    is_last_loop: bool = False,
    max_count: int | None = None,
    log_label: str = "",
    loss_recorder=None,
    block_key: str | None = None,
    select_indices: torch.Tensor | None = None,
) -> None:
    """Run AdaRound reconstruction over ``target`` using pre-cached IO.

    On the first loop this attaches ``AdaRoundQuantizer`` parametrizations
    to every Linear/Conv child of ``target`` and builds the optimiser. On
    later loops the existing optimiser is reused (with momentum reset).
    On the last loop AdaRound switches to its hard rounding regime.

    ``cached_inputs`` and ``cached_outputs`` are tensors stacked along
    dim 0 across all calibration samples. Tensor values in
    ``captured_kwargs`` whose leading dimension matches the sample count
    are indexed with the same mini-batch indices as the cached inputs;
    static tensor kwargs are moved to the target device unchanged.

    ``select_indices`` optionally restricts the reconstruction to a
    subset of samples (used by the progressive/timestep-grouping path).
    """
    state = state_registry.get(target)

    if is_first_loop and state.optimizer is None:
        for sub in target.modules():
            if isinstance(sub, _PARAMETRIZED_TYPES) and not parametrize.is_parametrized(
                sub, "weight"
            ):
                attach_adaround(sub, bits=bits, symmetric=symmetric, always_zero=always_zero)
        opt_params = [adar.alpha for adar in iter_adaround_quantizers(target) if adar.alpha is not None]
        if not opt_params:
            logger.info("[%s] no AdaRound params found, skipping reconstruction", log_label)
            return
        state.opt_params = opt_params
        state.optimizer = torch.optim.Adam(opt_params)
        state.scheduler = None
    elif state.optimizer is not None:
        reset_adam_momentum(state.optimizer, state.opt_params)

    if state.optimizer is None:
        return

    n_total = cached_inputs[0].size(0)
    if select_indices is not None:
        active = select_indices.to(cached_inputs[0].device)
    else:
        active = torch.arange(n_total, device=cached_inputs[0].device)
    n = int(active.numel())
    if n == 0:
        return

    loss_fn = _build_loss(
        target,
        state=state,
        rec_loss_kind=rec_loss_kind,
        weight=weight,
        max_count=max_count if max_count is not None else epoch * max(1, n // batch_size),
        b_range=b_range,
        decay_start=decay_start,
        warmup=warmup,
        p=p,
        log_label=log_label,
    )

    drop_last = n >= batch_size
    device = next(target.parameters()).device

    for _ep in range(epoch):
        perm_local = torch.randperm(n, device=active.device)
        perm = active.index_select(0, perm_local)
        rng = (
            range(0, n - (n % batch_size), batch_size)
            if drop_last
            else range(0, n, batch_size)
        )
        for start in rng:
            end = min(start + batch_size, n)
            idx = perm[start:end]
            if drop_last and idx.numel() < batch_size:
                continue

            cur_inputs = tuple(t.index_select(0, idx.to(t.device)).to(device) for t in cached_inputs)
            if isinstance(cached_outputs, torch.Tensor):
                cur_outputs = cached_outputs.index_select(0, idx.to(cached_outputs.device)).to(device)
            else:
                cur_outputs = tuple(
                    o.index_select(0, idx.to(o.device)).to(device) for o in cached_outputs
                )
            cur_grads = (
                cached_grads.index_select(0, idx.to(cached_grads.device)).to(device)
                if cached_grads is not None
                else None
            )

            state.optimizer.zero_grad(set_to_none=True)
            kw = (
                _select_kwargs_tree(captured_kwargs, idx, n_total, device)
                if captured_kwargs is not None
                else {}
            )
            out_quant = target(*cur_inputs, **kw)
            if isinstance(out_quant, tuple):
                out_quant = out_quant[0]

            err, rec_loss, round_loss = loss_fn(out_quant, cur_outputs, cur_grads)
            err.backward()
            _maybe_allreduce(state.opt_params)
            state.optimizer.step()
            if state.scheduler is not None:
                state.scheduler.step()

            if loss_recorder is not None and block_key is not None:
                loss_recorder.add(block_key, err, rec_loss, round_loss)

    torch.cuda.empty_cache()
    if is_last_loop:
        freeze_soft_targets(target)


def reconstruct_tib(
    tib: nn.Module,
    cali_data: tuple[torch.Tensor, ...],
    *,
    forward_fn,
    bits: int,
    symmetric: bool = False,
    always_zero: bool = False,
    iters: int = 20000,
    batch_size: int = 32,
    lr: float = 4e-5,
    weight: float = 0.01,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    decay_start: float = 0.0,
    p: float = 2.0,
    log_label: str = "tib",
) -> None:
    """One-shot AdaRound reconstruction for the temporal information block.

    ``tib`` is expected to be a single ``nn.Module`` whose forward returns
    a tuple of per-consumer time-embedding tensors (matching the order
    used during IO capture).
    """
    for sub in tib.modules():
        if isinstance(sub, _PARAMETRIZED_TYPES) and not parametrize.is_parametrized(
            sub, "weight"
        ):
            attach_adaround(sub, bits=bits, symmetric=symmetric, always_zero=always_zero)

    opt_params = [adar.alpha for adar in iter_adaround_quantizers(tib) if adar.alpha is not None]
    if not opt_params:
        logger.info("[%s] no AdaRound params found, skipping TIB reconstruction", log_label)
        return

    optimizer = torch.optim.Adam(opt_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)

    def _soft_targets() -> Iterable[torch.Tensor]:
        return [adar.get_soft_tgt() for adar in iter_adaround_quantizers(tib)]

    loss_fn = ReconLossTimeEmbedding(
        soft_targets_fn=_soft_targets,
        round_loss=ReconLossKind.RELAXATION,
        w=weight,
        rec_loss=ReconLossKind.MSE,
        max_count=iters,
        b_range=b_range,
        decay_start=decay_start,
        warmup=warmup,
        p=p,
        log_label=log_label,
    )

    # Capture FP outputs by detaching parametrizations temporarily.
    parametrize_was_active: list[tuple[nn.Module, AdaRoundQuantizer]] = []
    for sub in tib.modules():
        if isinstance(sub, _PARAMETRIZED_TYPES) and parametrize.is_parametrized(sub, "weight"):
            adar = next(
                (m for m in sub.parametrizations.weight if isinstance(m, AdaRoundQuantizer)),
                None,
            )
            if adar is not None:
                parametrize_was_active.append((sub, adar))

    # Disable AdaRound during FP capture by temporarily forcing soft_tgt=False
    # AND substituting the floor result with a simple round so the output
    # equals the original weight. Easier: remove parametrizations, capture, re-attach.
    state_dicts: dict[int, dict] = {}
    for sub, adar in parametrize_was_active:
        state_dicts[id(sub)] = {"adar": adar, "alpha": adar.alpha.detach().clone() if adar.alpha is not None else None}
        parametrize.remove_parametrizations(sub, "weight", leave_parametrized=False)

    cached_inputs, cached_outputs, captured_kwargs = capture_module_io(
        tib, tib, cali_data, forward_fn=forward_fn, batch_size=batch_size, keep_gpu=True
    )

    # Re-attach AdaRound parametrizations on the same modules.
    for sub, _ in parametrize_was_active:
        adar = state_dicts[id(sub)]["adar"]
        alpha_state = state_dicts[id(sub)]["alpha"]
        # Re-init quantizer with current weight; re-load saved alpha if present.
        new_quant = AdaRoundQuantizer(
            weight=sub.weight.data,
            bits=bits,
            symmetric=symmetric,
            always_zero=always_zero,
            rmode=RMODE.LEARNED_HARD_SIGMOID,
        )
        new_quant.soft_tgt = True
        if alpha_state is not None and new_quant.alpha is not None and new_quant.alpha.shape == alpha_state.shape:
            new_quant.alpha.data.copy_(alpha_state)
        parametrize.register_parametrization(sub, "weight", new_quant, unsafe=True)

    # Refresh opt_params: AdaRound modules have been replaced.
    opt_params = [adar.alpha for adar in iter_adaround_quantizers(tib) if adar.alpha is not None]
    optimizer = torch.optim.Adam(opt_params)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)

    if isinstance(cached_outputs, torch.Tensor):
        cached_outputs_seq: tuple[torch.Tensor, ...] = (cached_outputs,)
    else:
        cached_outputs_seq = cached_outputs

    n = cached_inputs[0].size(0)
    device = next(tib.parameters()).device
    for _ in range(iters):
        idx = torch.randperm(n, device=cached_inputs[0].device)[:batch_size]
        cur_inputs = tuple(t.index_select(0, idx.to(t.device)).to(device) for t in cached_inputs)
        cur_outputs = tuple(
            o.index_select(0, idx.to(o.device)).to(device) for o in cached_outputs_seq
        )
        optimizer.zero_grad(set_to_none=True)
        kw = captured_kwargs or {}
        out = tib(*cur_inputs, **kw)
        if isinstance(out, torch.Tensor):
            out = (out,)
        err = loss_fn(out, cur_outputs)
        err.backward()
        _maybe_allreduce(opt_params)
        optimizer.step()
        scheduler.step()

    torch.cuda.empty_cache()
    freeze_soft_targets(tib)

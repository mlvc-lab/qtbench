# -*- coding: utf-8 -*-
"""LoRA-style low-rank branch trainer for FastDM block reconstruction.

Companion to :mod:`reconstruct.py`. Where ``reconstruct_module`` trains
AdaRound rounding offsets, ``reconstruct_module_lowrank`` trains a rank-r
``LowRankBranch`` per ``nn.Linear``/``nn.Conv*`` such that the block
forward

    F.linear(x, RTN(W - b @ a)) + b(a(x))

matches the captured FP block output. The underlying ``module.weight``
stays frozen; only ``branch.{a,b}`` are optimised. Quantization is
simulated via min-max RTN with a straight-through estimator so gradients
flow into the branch.

This module is reused by ``app/diffusion/quant/fastdm.py`` via the
``recipe="lowrank"`` dispatch.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

from ...nn.patch.lowrank import LowRankBranch
from ...utils.hooks import AccumBranchHook
from .adaround import compute_minmax_scale_zero, reset_adam_momentum
from .reconstruct import _maybe_allreduce, _select_kwargs_tree

__all__ = [
    "RtnLowRankParametrize",
    "attach_lowrank",
    "detach_lowrank",
    "iter_lowrank_parametrizes",
    "build_lowrank_branch_state",
    "reconstruct_module_lowrank",
]

logger = logging.getLogger(__name__)

_PARAMETRIZED_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with straight-through gradient."""
    return x + (torch.round(x) - x).detach()


class _Fp32TrainingLowRankBranch(LowRankBranch):
    """LowRankBranch whose forward bridges fp16/fp32 dtypes.

    During FastDM training the branch parameters live in fp32 (Adam's
    eps underflows in fp16) while the host transformer is fp16. The hook
    feeds raw fp16 activations into ``branch(x)``; without bridging, the
    matmul errors with ``mat1 dtype != mat2 dtype``. We cast input to
    fp32, run the rank-r matmul there, and cast the result back to the
    input dtype before returning so the additive hook sees the same
    dtype the upstream module produced.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor | None:
        if self.a is None:
            return None
        in_dtype = input.dtype
        out = super().forward(input.to(torch.float32))
        if out is None:
            return None
        return out.to(in_dtype)


class RtnLowRankParametrize(nn.Module):
    """Parametrize on ``module.weight`` returning ``RTN(W - branch.eff())``.

    The min-max ``delta``/``zero`` are computed once from the *original*
    weight (i.e. the SVDQuant residual after smoothing) and frozen --
    matches a fixed RTN quantizer. The branch effective weight is
    subtracted before rounding so the branch absorbs the residual the
    rounding cannot represent. The STE makes the round differentiable
    with respect to ``branch.{a,b}``.

    All numeric work happens in fp32: even when the host module is fp16,
    Adam over branch parameters in fp16 underflows ``eps`` (default 1e-8)
    and quickly produces NaNs. The forward casts the inputs to fp32, does
    the round/clamp there, and returns a tensor in the original weight
    dtype so ``F.linear``/``F.conv`` is unaffected.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bits: int,
        branch: LowRankBranch,
        symmetric: bool = False,
        ste_rtn: bool = True,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.symmetric = bool(symmetric)
        self.ste_rtn = bool(ste_rtn)
        self.branch = branch
        delta, zero = compute_minmax_scale_zero(
            weight.detach().to(torch.float32), bits=self.bits, symmetric=self.symmetric
        )
        self.register_buffer("delta", delta.to(torch.float32))
        self.register_buffer("zero", zero.to(torch.float32))
        self._weight_shape = tuple(weight.shape)
        if symmetric:
            self.qmin = -(1 << (self.bits - 1)) + 1
            self.qmax = (1 << (self.bits - 1)) - 1
        else:
            self.qmin = 0
            self.qmax = (1 << self.bits) - 1

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        weight32 = weight.to(torch.float32)
        eff = self.branch.get_effective_weight()
        if eff is None:
            residual = weight32
        else:
            residual = weight32 - eff.view(weight.shape).to(torch.float32)
        scaled = residual / self.delta + self.zero
        if self.ste_rtn:
            q = _ste_round(scaled)
        else:
            q = torch.round(scaled)
        q = q.clamp(self.qmin, self.qmax)
        out = (q - self.zero) * self.delta
        return out.to(weight.dtype)


def attach_lowrank(
    module: nn.Module,
    *,
    rank: int,
    bits: int,
    symmetric: bool = False,
    ste_rtn: bool = True,
) -> tuple[RtnLowRankParametrize, AccumBranchHook] | None:
    """Attach an RTN-low-rank parametrize and a branch forward hook.

    ``module.weight`` is frozen (``requires_grad=False``); ``branch.{a,b}``
    become the only trainable tensors. The branch is SVD-initialised on
    the residual ``W - RTN(W)`` so training starts from the analytic
    SVDQuant point. Returns ``(parametrize, branch_hook)`` for cleanup, or
    ``None`` if the module is not parametrizable.

    If a parametrize already exists on ``weight``, this is a no-op and
    returns the existing instances.
    """
    if not isinstance(module, _PARAMETRIZED_TYPES):
        return None
    # LowRankBranch requires a flattenable (2D-equivalent) weight; skip
    # spatial convolutions where the rank-r decomposition would discard
    # kernel structure.
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.weight.shape[2:].numel() != 1:
        return None
    if parametrize.is_parametrized(module, "weight"):
        existing = next(
            (m for m in module.parametrizations.weight if isinstance(m, RtnLowRankParametrize)),
            None,
        )
        if existing is None:
            return None
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, AccumBranchHook) and hook.branch is existing.branch:
                return existing, hook
        return existing, None  # type: ignore[return-value]

    weight = module.weight.data
    flat = weight.reshape(weight.shape[0], -1)
    out_features, in_features = flat.shape

    # Compute the SVDQuant residual W - RTN(W) entirely in fp32 -- 4-bit
    # quant of an fp16 weight produces a residual close to the fp16
    # epsilon, where SVD numerics get unreliable.
    weight32 = weight.to(torch.float32)
    delta, zero = compute_minmax_scale_zero(weight32, bits=bits, symmetric=symmetric)
    if symmetric:
        q = torch.round(weight32 / delta + zero).clamp(
            -(1 << (bits - 1)) + 1, (1 << (bits - 1)) - 1
        )
    else:
        q = torch.round(weight32 / delta + zero).clamp(0, (1 << bits) - 1)
    rtn_w = (q - zero) * delta
    residual = (weight32 - rtn_w).to(weight.device)
    branch = _Fp32TrainingLowRankBranch(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        weight=residual.reshape(out_features, in_features),
    )
    # Branch parameters live in fp32 so Adam's eps doesn't underflow even
    # when the host module is fp16. The branch is folded back into the
    # final low-rank cache (``branch.state_dict()``) at fp32 precision and
    # the downstream loader casts to the host dtype when re-installing.
    branch.to(device=weight.device, dtype=torch.float32)
    if branch.a is not None:
        branch.a.weight.requires_grad_(True)
    if branch.b is not None and not isinstance(branch.b, nn.Identity):
        branch.b.weight.requires_grad_(True)

    module.weight.requires_grad_(False)
    parametrize_ = RtnLowRankParametrize(
        weight=weight, bits=bits, branch=branch, symmetric=symmetric, ste_rtn=ste_rtn
    )
    parametrize.register_parametrization(module, "weight", parametrize_, unsafe=True)

    branch_hook = branch.as_hook()
    branch_hook.register(module)
    return parametrize_, branch_hook


def detach_lowrank(module: nn.Module) -> None:
    """Remove the RTN-low-rank parametrize and the branch hook.

    Leaves ``module.weight`` unchanged (the original FP weight) so the
    downstream ``quantize_diffusion_weights`` step starts from a clean
    state and re-installs its own branch from the trained ``state_dict``.

    ``AccumBranchHook`` registers as both a pre-hook (caches the input)
    and a post-hook (adds branch(input) to the output), so we must drop
    matching entries from *both* hook dicts -- otherwise a stale post
    hook fires during inference with ``self.tensor = None`` and the
    forward errors out.
    """
    if not isinstance(module, _PARAMETRIZED_TYPES):
        return
    if parametrize.is_parametrized(module, "weight"):
        # leave_parametrized=False: restore the *original* weight buffer.
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=False)
    for hook_dict in (module._forward_pre_hooks, module._forward_hooks):
        to_drop = [
            hid
            for hid, hook in hook_dict.items()
            if isinstance(hook, AccumBranchHook)
            and isinstance(hook.branch, LowRankBranch)
        ]
        for hid in to_drop:
            del hook_dict[hid]
    module.weight.requires_grad_(False)


def iter_lowrank_parametrizes(root: nn.Module) -> Iterable[RtnLowRankParametrize]:
    """Yield every ``RtnLowRankParametrize`` attached anywhere under ``root``."""
    for sub in root.modules():
        if isinstance(sub, RtnLowRankParametrize):
            yield sub


def build_lowrank_branch_state(
    target_root: nn.Module,
    target_name: str,
    module_name_map: dict[int, str] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Serialize trained branches in the SVDQuant ``branch_state_dict`` format.

    Walks ``target_root`` for every ``RtnLowRankParametrize``, looks up
    the parent module's name (qualified relative to the root model), and
    emits ``{full_module_name: branch.state_dict()}`` -- the format
    consumed by ``calibrate_diffusion_block_low_rank_branch``'s reload
    path.
    """
    out: dict[str, dict[str, torch.Tensor]] = {}
    for sub_name, sub in target_root.named_modules():
        if not isinstance(sub, _PARAMETRIZED_TYPES):
            continue
        if not parametrize.is_parametrized(sub, "weight"):
            continue
        para = next(
            (m for m in sub.parametrizations.weight if isinstance(m, RtnLowRankParametrize)),
            None,
        )
        if para is None:
            continue
        full = f"{target_name}.{sub_name}" if sub_name else target_name
        if module_name_map is not None and id(sub) in module_name_map:
            full = module_name_map[id(sub)]
        branch = para.branch
        sd = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
        out[full] = sd
    return out


def reconstruct_module_lowrank(  # noqa: C901
    target: nn.Module,
    *,
    cached_inputs: tuple[torch.Tensor, ...],
    cached_outputs: torch.Tensor | tuple[torch.Tensor, ...],
    captured_kwargs: dict | None = None,
    state_registry,
    rank: int,
    bits: int,
    symmetric: bool = False,
    ste_rtn: bool = True,
    epoch: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    is_first_loop: bool = False,
    is_last_loop: bool = False,
    log_label: str = "",
    loss_recorder=None,
    block_key: str | None = None,
    select_indices: torch.Tensor | None = None,
) -> None:
    """Train ``branch.{a,b}`` over a target block using cached FP IO.

    Mirrors :func:`reconstruct_module` but optimises the branch parameters
    with an MSE block-output loss. The first loop attaches one
    ``RtnLowRankParametrize`` per parametrizable child of ``target`` and
    builds the optimiser; later loops reuse the same optimiser (with
    momentum reset).
    """
    state = state_registry.get(target)

    if is_first_loop and state.optimizer is None:
        # Snapshot the target Linears/Convs *before* attaching anything --
        # the branch we attach also contains nn.Linear children, and a
        # naive ``target.modules()`` walk would recurse into them.
        pending = [
            sub
            for sub in list(target.modules())
            if isinstance(sub, _PARAMETRIZED_TYPES)
            and not parametrize.is_parametrized(sub, "weight")
        ]
        for sub in pending:
            attach_lowrank(
                sub, rank=rank, bits=bits, symmetric=symmetric, ste_rtn=ste_rtn
            )
        opt_params: list[nn.Parameter] = []
        for para in iter_lowrank_parametrizes(target):
            if para.branch.a is not None and isinstance(para.branch.a, nn.Linear):
                opt_params.append(para.branch.a.weight)
            if (
                para.branch.b is not None
                and isinstance(para.branch.b, nn.Linear)
            ):
                opt_params.append(para.branch.b.weight)
        if not opt_params:
            logger.info("[%s] no LowRank params found, skipping reconstruction", log_label)
            return
        state.opt_params = opt_params
        # eps=1e-7 is the minimum representable in fp16 mantissa space.
        # Even though branch params are fp32 here, fp16 grad inputs can
        # round Adam's m/v statistics down to 0; this keeps the divisor
        # safely above the underflow boundary.
        state.optimizer = torch.optim.Adam(
            opt_params, lr=lr, weight_decay=weight_decay, eps=1e-7
        )
        state.scheduler = None
        state.loss = None  # unused for lowrank recipe
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

            cur_inputs = tuple(
                t.index_select(0, idx.to(t.device)).to(device) for t in cached_inputs
            )
            if isinstance(cached_outputs, torch.Tensor):
                cur_outputs = cached_outputs.index_select(
                    0, idx.to(cached_outputs.device)
                ).to(device)
            else:
                cur_outputs = tuple(
                    o.index_select(0, idx.to(o.device)).to(device) for o in cached_outputs
                )

            state.optimizer.zero_grad(set_to_none=True)
            kw = (
                _select_kwargs_tree(captured_kwargs, idx, n_total, device)
                if captured_kwargs is not None
                else {}
            )
            out = target(*cur_inputs, **kw)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(cur_outputs, tuple):
                cur_outputs = cur_outputs[0]

            # Compute MSE in fp32 so a fp16 block output doesn't lose
            # precision when squaring + reducing.
            err = (out.to(torch.float32) - cur_outputs.to(torch.float32)).pow(2).mean()
            if not torch.isfinite(err):
                logger.warning(
                    "[%s] non-finite block-recon loss; skipping step", log_label
                )
                continue
            err.backward()
            _maybe_allreduce(state.opt_params)
            torch.nn.utils.clip_grad_norm_(state.opt_params, max_norm=1.0)
            state.optimizer.step()
            if state.scheduler is not None:
                state.scheduler.step()

            if loss_recorder is not None and block_key is not None:
                loss_recorder.add(
                    block_key,
                    err.detach(),
                    err.detach(),
                    torch.zeros_like(err.detach()),
                )

    torch.cuda.empty_cache()
    if is_last_loop:
        # No soft/hard switch needed -- the STE round is the same regime
        # train and test. Caller is responsible for detach.
        pass

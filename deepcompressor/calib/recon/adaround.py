# -*- coding: utf-8 -*-
"""AdaRound learnable-rounding quantizer.

Ported from ``Fast_DM_PTQ/quant/adaptive_rounding.py``. The quantizer wraps a
single weight tensor with a learnable hard-sigmoid rounding offset; ``alpha``
is the trainable parameter, ``delta``/``zero`` are static (computed once from
the unquantized weight via min-max).
"""

from __future__ import annotations

import enum

import torch
from torch import nn

__all__ = ["RMODE", "AdaRoundQuantizer", "compute_minmax_scale_zero", "reset_adam_momentum"]


class RMODE(enum.Enum):
    LEARNED_ROUND_SIGMOID = enum.auto()
    NEAREST = enum.auto()
    NEAREST_STE = enum.auto()
    STOCHASTIC = enum.auto()
    LEARNED_HARD_SIGMOID = enum.auto()


def compute_minmax_scale_zero(
    x: torch.Tensor,
    bits: int,
    symmetric: bool = False,
    always_zero: bool = False,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-tensor (delta, zero_point) using the min-max scaler.

    Mirrors ``Fast_DM_PTQ.quant.quant_layer.minmax`` so the AdaRound init is
    bit-identical to the source repo.
    """
    level = 2**bits
    x_min = min(x.min().item(), 0.0)
    x_max = max(x.max().item(), 0.0)
    if symmetric:
        m = max(abs(x_min), x_max)
        x_min, x_max = -m, m
        delta = (x_max - x_min) / (level - 2)
    elif always_zero:
        delta = x_max / (level - 1)
    else:
        delta = (x_max - x_min) / (level - 1)

    if delta < eps:
        delta = eps

    delta_t = torch.tensor(float(delta), dtype=x.dtype, device=x.device)
    if symmetric or always_zero:
        zero_t = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    else:
        zero_t = torch.round(torch.tensor(-x_min / delta, dtype=x.dtype, device=x.device))
    return delta_t, zero_t


class AdaRoundQuantizer(nn.Module):
    """Learnable-rounding quantizer wrapping a single weight tensor.

    ``soft_tgt=True`` returns the differentiable soft rounding (sigmoid of
    alpha clamped through a relaxation window); ``soft_tgt=False`` switches
    to the hard ``alpha >= 0`` rounding used at inference.
    """

    GAMMA = -0.1
    ZETA = 1.1

    def __init__(
        self,
        weight: torch.Tensor,
        bits: int,
        symmetric: bool = False,
        always_zero: bool = False,
        rmode: RMODE = RMODE.LEARNED_HARD_SIGMOID,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.level = 2**self.bits
        self.symmetric = bool(symmetric)
        self.always_zero = bool(always_zero)
        self.rmode = rmode
        self.soft_tgt = False

        delta, zero = compute_minmax_scale_zero(
            weight, bits=self.bits, symmetric=self.symmetric, always_zero=self.always_zero
        )
        # Buffers so they move with .to(device) and load via state_dict.
        self.register_buffer("delta", delta)
        self.register_buffer("zero_point", zero)
        self.alpha: nn.Parameter | None = None
        self.init_alpha(weight)

    def init_alpha(self, weight: torch.Tensor) -> None:
        if self.rmode != RMODE.LEARNED_HARD_SIGMOID:
            raise NotImplementedError(self.rmode)
        with torch.no_grad():
            w_fp32 = weight.detach().to(torch.float32)
            delta = self.delta.to(weight.device, dtype=torch.float32)
            rest = (w_fp32 / delta) - torch.floor(w_fp32 / delta)
            rest = torch.clamp(rest, min=1e-6, max=1 - 1e-6)
            denom = (self.ZETA - self.GAMMA) / (rest - self.GAMMA) - 1
            denom = torch.clamp(denom, min=1e-12)
            alpha = -torch.log(denom)
        # Keep alpha in fp32 for optimizer numerical stability.
        self.alpha = nn.Parameter(alpha)

    def get_soft_tgt(self) -> torch.Tensor:
        assert self.alpha is not None
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.ZETA - self.GAMMA) + self.GAMMA, 0, 1
        )

    def _qmin_qmax(self) -> tuple[int, int]:
        if self.symmetric and not self.always_zero:
            return (-self.level // 2, self.level // 2 - 1)
        return (0, self.level - 1)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        # Gradient flows only through alpha; weight itself is treated as a constant.
        out_dtype = weight.dtype
        w = weight.detach().to(torch.float32)
        delta = self.delta.to(w.device, dtype=torch.float32)
        zero = self.zero_point.to(w.device, dtype=torch.float32)

        with torch.no_grad():
            x_floor = torch.floor(w / delta)

        if self.rmode == RMODE.NEAREST:
            x_int = torch.round(w / delta)
        elif self.rmode == RMODE.STOCHASTIC:
            x_int = x_floor + torch.bernoulli((w / delta) - x_floor)
        elif self.rmode == RMODE.LEARNED_HARD_SIGMOID:
            assert self.alpha is not None
            if self.soft_tgt:
                x_int = x_floor + self.get_soft_tgt()
            else:
                x_int = x_floor + (self.alpha.detach() >= 0).to(torch.float32)
        else:
            raise NotImplementedError(self.rmode)

        nb, pb = self._qmin_qmax()
        x_q = torch.clamp(x_int + zero, nb, pb)
        out = delta * (x_q - zero)
        return out.to(out_dtype)

    def freeze_to_hard(self) -> None:
        """Switch to the hard rounding regime used at inference."""
        self.soft_tgt = False

    def extra_repr(self) -> str:
        return (
            f"bits={self.bits}, symmetric={self.symmetric}, "
            f"always_zero={self.always_zero}, rmode={self.rmode.name}, "
            f"soft_tgt={self.soft_tgt}"
        )


def reset_adam_momentum(
    optimizer: torch.optim.Optimizer | None,
    params: list[nn.Parameter] | None = None,
) -> None:
    """Zero-out Adam's first/second moments for the given params (or all)."""
    if optimizer is None:
        return
    if params is None:
        for group in optimizer.param_groups:
            for p in group.get("params", []):
                _zero_adam_state(optimizer.state.get(p, None))
        return
    for p in params:
        _zero_adam_state(optimizer.state.get(p, None))


def _zero_adam_state(state: dict | None) -> None:
    if state is None:
        return
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        buf = state.get(key)
        if buf is not None:
            buf.zero_()

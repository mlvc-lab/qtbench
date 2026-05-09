# -*- coding: utf-8 -*-
"""Loss functions used by FastDM block reconstruction.

Ported from ``Fast_DM_PTQ/quant/reconstruction_util.py`` with the per-module
soft-target lookup decoupled into a callable so the loss is independent of
any particular ``QuantLayer``/``QuantBlock`` class hierarchy.
"""

from __future__ import annotations

import enum
import logging
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import torch

__all__ = [
    "ReconLossKind",
    "ReconLossReduction",
    "lp_loss",
    "LinearTempDecay",
    "ReconLoss",
    "ReconLossTimeEmbedding",
    "LossRecorder",
]

logger = logging.getLogger(__name__)
PRINT_FREQ = 1000


class ReconLossKind(enum.Enum):
    RELAXATION = enum.auto()
    MSE = enum.auto()
    FISHER_DIAG = enum.auto()
    FISHER_FULL = enum.auto()
    NONE = enum.auto()


class ReconLossReduction(enum.Enum):
    NONE = enum.auto()
    ALL = enum.auto()


def lp_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    p: float = 2.0,
    reduction: ReconLossReduction = ReconLossReduction.NONE,
) -> torch.Tensor:
    """Element-wise L_p reconstruction loss matching the source repo."""
    if reduction == ReconLossReduction.NONE:
        return (pred - tgt).abs().pow(p).sum(1).mean()
    if reduction == ReconLossReduction.ALL:
        return (pred - tgt).abs().pow(p).mean()
    raise NotImplementedError(reduction)


class LinearTempDecay:
    """Linearly decay the rounding temperature ``b`` over a budget of steps."""

    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: float = 10.0,
        end_b: float = 2.0,
    ) -> None:
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = float(start_b)
        self.end_b = float(end_b)

    def __call__(self, t: int) -> float:
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, 1.0 - rel_t)


SoftTargetsFn = Callable[[], Iterable[torch.Tensor]]
"""Returns the AdaRound soft-target tensors for every learnable rounding param."""


class ReconLoss:
    """Combined reconstruction + relaxation rounding loss for one module.

    The ``soft_targets_fn`` callable returns each ``AdaRoundQuantizer``'s
    ``get_soft_tgt()`` tensor for the module being optimised; passing a
    callable rather than a module reference keeps the loss independent of
    any particular quantizer-block class hierarchy.
    """

    def __init__(
        self,
        soft_targets_fn: SoftTargetsFn,
        round_loss: ReconLossKind = ReconLossKind.RELAXATION,
        w: float = 1.0,
        rec_loss: ReconLossKind = ReconLossKind.MSE,
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
        log_label: str = "",
    ) -> None:
        self.soft_targets_fn = soft_targets_fn
        self.round_loss = round_loss
        self.w = float(w)
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = float(p)
        self.temp_decay = LinearTempDecay(
            t_max=max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0
        self.log_label = log_label

    def __call__(
        self,
        pred: torch.Tensor,
        tgt: torch.Tensor,
        grad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float | torch.Tensor]:
        self.count += 1

        if self.rec_loss == ReconLossKind.MSE:
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == ReconLossKind.FISHER_DIAG:
            assert grad is not None
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == ReconLossKind.FISHER_FULL:
            assert grad is not None
            a = (pred - tgt).abs()
            g = grad.abs()
            batch_dotprod = torch.sum(a * g, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * g).mean() / 100
        else:
            raise ValueError(f"Unsupported rec_loss kind: {self.rec_loss}")

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == ReconLossKind.NONE:
            b = 0
            round_loss: float | torch.Tensor = 0.0
        elif self.round_loss == ReconLossKind.RELAXATION:
            round_loss = 0.0
            for soft_tgt in self.soft_targets_fn():
                round_loss = round_loss + self.w * (
                    1 - ((soft_tgt - 0.5).abs() * 2).pow(b)
                ).sum()
        else:
            raise NotImplementedError(self.round_loss)

        total_loss = rec_loss + round_loss
        if self.count % PRINT_FREQ == 0:
            label = f"[{self.log_label}] " if self.log_label else ""
            logger.info(
                "%sTotal loss: %.8f (rec: %.8f, round: %.8f) b=%.2f count=%d",
                label,
                float(total_loss),
                float(rec_loss),
                float(round_loss) if torch.is_tensor(round_loss) else float(round_loss),
                b,
                self.count,
            )
        return total_loss, rec_loss, round_loss


class ReconLossTimeEmbedding:
    """Multi-output reconstruction loss for the temporal information block.

    ``preds`` and ``tgts`` are sequences of tensors (one per consumer of
    the time embedding); the rec loss is summed across them.
    """

    def __init__(
        self,
        soft_targets_fn: SoftTargetsFn,
        round_loss: ReconLossKind = ReconLossKind.RELAXATION,
        w: float = 1.0,
        rec_loss: ReconLossKind = ReconLossKind.MSE,
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
        log_label: str = "tib",
    ) -> None:
        self.soft_targets_fn = soft_targets_fn
        self.round_loss = round_loss
        self.w = float(w)
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = float(p)
        self.temp_decay = LinearTempDecay(
            t_max=max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0
        self.log_label = log_label

    def __call__(
        self,
        preds: Sequence[torch.Tensor],
        tgts: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        self.count += 1
        if not preds:
            raise ValueError("preds is empty")
        device = preds[0].device
        rec_loss = torch.zeros((), device=device, dtype=preds[0].dtype)
        for pred, tgt in zip(preds, tgts, strict=True):
            if self.rec_loss == ReconLossKind.MSE:
                rec_loss = rec_loss + lp_loss(pred, tgt, p=self.p)
            else:
                raise ValueError(f"Unsupported rec_loss kind: {self.rec_loss}")

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == ReconLossKind.NONE:
            b = 0
            round_loss: float | torch.Tensor = 0.0
        elif self.round_loss == ReconLossKind.RELAXATION:
            round_loss = 0.0
            for soft_tgt in self.soft_targets_fn():
                round_loss = round_loss + self.w * (
                    1 - ((soft_tgt - 0.5).abs() * 2).pow(b)
                ).sum()
        else:
            raise NotImplementedError(self.round_loss)

        total_loss = rec_loss + round_loss
        if self.count % PRINT_FREQ == 0:
            logger.info(
                "[%s] Total loss: %.8f (rec: %.8f, round: %.8f) b=%.2f count=%d",
                self.log_label,
                float(total_loss),
                float(rec_loss),
                float(round_loss) if torch.is_tensor(round_loss) else float(round_loss),
                b,
                self.count,
            )
        return total_loss


class LossRecorder:
    """Per-block running history of (err, rec, round) loss components."""

    def __init__(self) -> None:
        self.store: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"err": [], "rec": [], "round": []}
        )

    @torch.no_grad()
    def add(
        self,
        block_key: str,
        err: torch.Tensor | float,
        rec: torch.Tensor | float,
        rnd: torch.Tensor | float,
    ) -> None:
        def to_float(x: torch.Tensor | float) -> float:
            return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)

        bucket = self.store[block_key]
        bucket["err"].append(to_float(err))
        bucket["rec"].append(to_float(rec))
        bucket["round"].append(to_float(rnd))

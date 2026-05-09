# -*- coding: utf-8 -*-
"""Temporal-information-block (TIB) detection for diffusers backbones.

Fast_DM_PTQ defines a ``QuantTemporalInformationBlock`` that bundles the
LDM time-embedding MLP together with every per-block ``time_emb_proj``;
its forward returns the tuple of per-consumer ``temb`` projections so a
single AdaRound pass can calibrate the entire time pathway.

For the diffusers-based qtbench pipeline the analogous bundle is
architecture-specific (UNet vs DiT vs SD3 vs Flux). Robust auto-detection
of every variant is hard, so this module provides a conservative
detector that returns a bundle for the well-known UNet2DConditionModel
case and falls back to ``None`` otherwise. The ``fastdm_diffusion``
driver logs a warning and skips TIB reconstruction in the fallback path
-- the rest of the AdaRound pipeline still runs.
"""

from __future__ import annotations

import logging
import typing as tp

import torch
from torch import nn

__all__ = ["FastDmTibBundle", "build_tib_bundle"]

logger = logging.getLogger(__name__)


_TIB_NAME_HINTS = ("time_embed", "time_embedder", "t_embedder", "time_proj", "time_text_embed")


def _resolve_time_embed(model: nn.Module) -> nn.Module | None:
    """Return the top-level time-embedding module if we can find one."""
    for name in _TIB_NAME_HINTS:
        if hasattr(model, name):
            sub = getattr(model, name)
            if isinstance(sub, nn.Module):
                return sub
    return None


class FastDmTibBundle(nn.Module):
    """A nn.Module wrapper exposing all TIB consumers as one forward call.

    The bundle holds *references* to the existing model modules; it does
    not copy parameters. Its forward takes the timestep tensor and returns
    a tuple containing:

      0. the output of the top-level time-embedding MLP, and
      1..N. the outputs of every detected per-block ``time_emb_proj``
            (UNet ``ResnetBlock2D.time_emb_proj``).

    This matches the structure of Fast_DM_PTQ's
    ``QuantTemporalInformationBlock.forward``.
    """

    def __init__(self, time_embed: nn.Module, temb_projs: list[nn.Linear]) -> None:
        super().__init__()
        self.time_embed = time_embed
        self.temb_projs = nn.ModuleList(temb_projs)

    def members(self) -> tp.Iterable[nn.Module]:
        """Yield every module that participates in the TIB forward."""
        yield self.time_embed
        for p in self.temb_projs:
            yield p

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, ...]:
        emb = self.time_embed(t)
        outs: list[torch.Tensor] = [emb]
        for proj in self.temb_projs:
            outs.append(proj(emb))
        return tuple(outs)


def build_tib_bundle(model: nn.Module) -> FastDmTibBundle | None:
    """Construct a ``FastDmTibBundle`` from ``model`` if possible.

    Currently supports diffusers UNet-style backbones that expose
    ``time_embed: nn.Module`` and contain ``ResnetBlock2D``-style children
    with a ``time_emb_proj`` Linear. Returns ``None`` if either the
    top-level time embedder or any per-block projection cannot be located.
    """
    time_embed = _resolve_time_embed(model)
    if time_embed is None:
        logger.warning(
            "fastdm: no top-level time-embedding module found; TIB skipped"
        )
        return None

    temb_projs: list[nn.Linear] = []
    for sub in model.modules():
        proj = getattr(sub, "time_emb_proj", None)
        if isinstance(proj, nn.Linear):
            temb_projs.append(proj)

    if not temb_projs:
        logger.warning(
            "fastdm: no ResnetBlock2D.time_emb_proj projections found; TIB skipped"
        )
        return None

    logger.info(
        "fastdm: TIB bundle assembled (1 time_embed + %d time_emb_proj projections)",
        len(temb_projs),
    )
    return FastDmTibBundle(time_embed=time_embed, temb_projs=temb_projs)

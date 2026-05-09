# -*- coding: utf-8 -*-
"""PTQ4DiT-format calibration dataset loader.

Reads the single ``.pt`` produced by
``examples/diffusion/scripts/collect_dit_ptq4dit.py`` (which mirrors
``PTQ4DiT/get_calibration_set.py``) and adapts it to qtbench's
``DiffusionCalibCacheLoader`` interface.

Two key behaviours come from PTQ4DiT and are implemented here:

1. **Strided per-timestep subset selection** -- ``cali_st`` evenly-spaced
   timesteps × ``cali_n`` samples per timestep, giving a calibration
   set of size ``cali_st * cali_n`` (default 25 × 64 = 1600 samples).
2. **Conditional/null reorder** -- inside each timestep slice the
   non-null (class<1000) samples are placed before the null
   (class==1000) ones, matching ``quant_sample.py:113-117``.

Each yielded ``ModuleForwardInput`` contains the latent in
``args[0]`` and the timestep + class label in ``input_kwargs`` -- the
shape that ``DiTTransformer2DModel.forward`` expects.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

import torch

from deepcompressor.data.cache import ModuleForwardInput

__all__ = [
    "PTQ4DiTCalibLoaderConfig",
    "PTQ4DiTCalibLoader",
    "load_ptq4dit_blob",
]

logger = logging.getLogger(__name__)


@dataclass
class PTQ4DiTCalibLoaderConfig:
    """Settings that pick a PTQ4DiT subset out of the collected blob.

    ``cali_st`` and ``cali_n`` follow the source repo's CLI flags: the
    final calibration set has ``cali_st * cali_n`` samples, drawn as
    ``cali_n`` samples from each of ``cali_st`` evenly-spaced
    timesteps. ``batch_size`` is the per-iteration batch the rest of
    qtbench's calibration code consumes.
    """

    path: str
    cali_st: int = 25
    cali_n: int = 64
    batch_size: int = 16
    reorder_nulls_last: bool = True


def load_ptq4dit_blob(path: str) -> dict[str, torch.Tensor]:
    """Load and validate a PTQ4DiT-format calibration ``.pt``."""
    blob = torch.load(path, map_location="cpu")
    for key in ("xs", "ts", "y"):
        if key not in blob:
            raise KeyError(f"PTQ4DiT calibration blob missing '{key}': {list(blob.keys())}")
        if blob[key].ndim < 2:
            raise ValueError(
                f"PTQ4DiT calibration blob '{key}' has shape {tuple(blob[key].shape)}; "
                "expected [num_steps, num_samples, ...]"
            )
    return blob


def _stride_steps(num_steps: int, cali_st: int) -> list[int]:
    """Pick ``cali_st`` evenly-spaced timestep indices out of ``num_steps``."""
    if cali_st >= num_steps:
        return list(range(num_steps))
    stride = num_steps // cali_st
    return [i * stride for i in range(cali_st)]


def _reorder_normal_then_null(
    xs: torch.Tensor,
    ts: torch.Tensor,
    ys: torch.Tensor,
    null_label: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place class<1000 samples first, class==1000 (null) samples last."""
    normal_mask = ys != null_label
    normal_idx = normal_mask.nonzero(as_tuple=False).flatten()
    null_idx = (~normal_mask).nonzero(as_tuple=False).flatten()
    order = torch.cat([normal_idx, null_idx], dim=0)
    return xs.index_select(0, order), ts.index_select(0, order), ys.index_select(0, order)


class PTQ4DiTCalibLoader:
    """Iterator that yields ``ModuleForwardInput`` slices of the blob."""

    def __init__(self, config: PTQ4DiTCalibLoaderConfig) -> None:
        self.config = config
        blob = load_ptq4dit_blob(config.path)
        num_steps = blob["xs"].shape[0]
        step_idxs = _stride_steps(num_steps, config.cali_st)
        n = config.cali_n

        slices_x: list[torch.Tensor] = []
        slices_t: list[torch.Tensor] = []
        slices_y: list[torch.Tensor] = []
        for s in step_idxs:
            slices_x.append(blob["xs"][s, :n])
            slices_t.append(blob["ts"][s, :n])
            slices_y.append(blob["y"][s, :n])
        xs = torch.cat(slices_x, dim=0)
        ts = torch.cat(slices_t, dim=0)
        ys = torch.cat(slices_y, dim=0)

        if config.reorder_nulls_last:
            xs, ts, ys = _reorder_normal_then_null(xs, ts, ys)

        self.xs = xs.contiguous()
        self.ts = ts.contiguous()
        self.ys = ys.contiguous()
        logger.info(
            "PTQ4DiTCalibLoader ready: %d samples (cali_st=%d, cali_n=%d) from %d source steps; "
            "xs=%s ts=%s y=%s",
            self.xs.shape[0],
            config.cali_st,
            config.cali_n,
            num_steps,
            tuple(self.xs.shape),
            tuple(self.ts.shape),
            tuple(self.ys.shape),
        )

    def __len__(self) -> int:
        return self.xs.shape[0]

    def init_subset(self, n_per_t: int = 4) -> "PTQ4DiTCalibLoader":
        """Return a smaller loader with ``n_per_t`` samples per unique timestep.

        Mirrors ``quant_sample.py:104-118`` -- used for the 3-pass scale
        initialization where you want a *small* but timestep-diverse
        subset, not the full reconstruction set.
        """
        idxs: list[int] = []
        seen: dict[int, int] = {}
        for i in range(self.ts.shape[0]):
            t = int(self.ts[i].item())
            if seen.get(t, 0) >= n_per_t:
                continue
            idxs.append(i)
            seen[t] = seen.get(t, 0) + 1
        sub_idx = torch.tensor(idxs, dtype=torch.long)
        sub = PTQ4DiTCalibLoader.__new__(PTQ4DiTCalibLoader)
        sub.config = self.config
        sub.xs = self.xs.index_select(0, sub_idx).contiguous()
        sub.ts = self.ts.index_select(0, sub_idx).contiguous()
        sub.ys = self.ts.new_empty((0,))  # placeholder
        sub.ys = self.ys.index_select(0, sub_idx).contiguous()
        if self.config.reorder_nulls_last:
            sub.xs, sub.ts, sub.ys = _reorder_normal_then_null(sub.xs, sub.ts, sub.ys)
        return sub

    def iter_samples(self) -> tp.Generator[ModuleForwardInput, None, None]:
        """Yield ``ModuleForwardInput`` batches matching DiT's transformer signature."""
        n = self.xs.shape[0]
        bs = max(1, min(self.config.batch_size, n))
        for start in range(0, n, bs):
            end = min(start + bs, n)
            x_batch = self.xs[start:end]
            t_batch = self.ts[start:end]
            y_batch = self.ys[start:end]
            yield ModuleForwardInput(
                args=[x_batch],
                kwargs={
                    "timestep": t_batch,
                    "class_labels": y_batch,
                },
            )

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the stacked ``(xs, ts, ys)`` tensors -- handy for fastdm."""
        return self.xs, self.ts, self.ys

# -*- coding: utf-8 -*-
"""Adapters that turn qdiff calibration caches into FastDM-friendly samples.

Each qdiff cache file (saved by ``deepcompressor.app.diffusion.dataset.collect``)
holds ``input_args`` / ``input_kwargs`` / ``outputs`` for a single
``(prompt, denoising_step, guidance)`` call into the diffusion transformer.
FastDM needs the sample's *timestep* per item (so it can group / progressively
schedule the calibration set) plus ready-to-go positional args and kwargs to
re-run the model end-to-end.
"""

from __future__ import annotations

import logging
import typing as tp
from dataclasses import dataclass

import torch

from ..dataset.calib import DiffusionCalibCacheLoader, DiffusionCalibDataset

__all__ = ["FastDmSample", "iter_fastdm_samples", "extract_timesteps", "TIMESTEP_KWARG_KEYS"]

logger = logging.getLogger(__name__)


# diffusers exposes the per-call timestep under a few different keys depending
# on the pipeline; the qdiff collector stores whichever keyword the model's
# forward accepts.
TIMESTEP_KWARG_KEYS = ("timestep", "timesteps", "t")


@dataclass
class FastDmSample:
    """A single qdiff calibration sample reshaped for FastDM."""

    args: list[tp.Any]
    kwargs: dict[str, tp.Any]
    timestep: float | int


def _scalar_timestep(value: tp.Any) -> float | None:
    """Best-effort extraction of a scalar timestep from a cached value."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        v = value.detach().reshape(-1)[0]
        if v.is_floating_point():
            return float(v)
        return int(v)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_timesteps(samples: tp.Iterable[FastDmSample]) -> torch.Tensor:
    """Return a 1D tensor of per-sample timesteps."""
    ts = [s.timestep for s in samples]
    return torch.tensor(ts, dtype=torch.float64)


def iter_fastdm_samples(
    loader_or_dataset: DiffusionCalibCacheLoader | DiffusionCalibDataset,
) -> tp.Generator[FastDmSample, None, None]:
    """Yield ``FastDmSample`` per cache entry in iteration order.

    The function pulls items from ``DiffusionCalibDataset.data`` directly so
    we get one cache file per iteration (matching the source FastDM repo's
    sample-at-a-time scheduling). The caller can re-batch downstream.
    """
    if isinstance(loader_or_dataset, DiffusionCalibCacheLoader):
        dataset: DiffusionCalibDataset = loader_or_dataset.dataset
    else:
        dataset = loader_or_dataset

    for entry in dataset.data:
        kwargs = dict(entry.get("input_kwargs", {}))
        args = list(entry.get("input_args", []))
        ts: float | int | None = None
        for k in TIMESTEP_KWARG_KEYS:
            if k in kwargs:
                ts = _scalar_timestep(kwargs[k])
                if ts is not None:
                    break
        if ts is None:
            # Some collectors record the step index rather than the
            # actual timestep; fall back to whichever exists.
            ts = float(entry.get("step", 0))
        yield FastDmSample(args=args, kwargs=kwargs, timestep=ts)

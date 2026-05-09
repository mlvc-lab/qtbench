# -*- coding: utf-8 -*-
"""Reconstruction-style (gradient-based) calibration utilities."""

from .adaround import (
    RMODE,
    AdaRoundQuantizer,
    compute_minmax_scale_zero,
    reset_adam_momentum,
)
from .loss import (
    LinearTempDecay,
    LossRecorder,
    ReconLoss,
    ReconLossKind,
    ReconLossReduction,
    ReconLossTimeEmbedding,
    lp_loss,
)
from .lowrank_recon import (
    RtnLowRankParametrize,
    attach_lowrank,
    build_lowrank_branch_state,
    detach_lowrank,
    iter_lowrank_parametrizes,
    reconstruct_module_lowrank,
)
from .reconstruct import (
    ReconState,
    StopForwardException,
    attach_adaround,
    capture_module_grad,
    capture_module_io,
    detach_adaround,
    freeze_soft_targets,
    iter_adaround_quantizers,
    reconstruct_module,
    reconstruct_tib,
)
from .schedule import (
    dp_distance_segments_from_adj,
    make_widths_fixedK,
    make_widths_from_adjacent_dist,
    plan_loop_sizes,
    steps_per_epoch_like,
)

__all__ = [
    "RMODE",
    "AdaRoundQuantizer",
    "LinearTempDecay",
    "LossRecorder",
    "ReconLoss",
    "ReconLossKind",
    "ReconLossReduction",
    "ReconLossTimeEmbedding",
    "ReconState",
    "RtnLowRankParametrize",
    "StopForwardException",
    "attach_adaround",
    "attach_lowrank",
    "build_lowrank_branch_state",
    "capture_module_grad",
    "capture_module_io",
    "compute_minmax_scale_zero",
    "detach_adaround",
    "detach_lowrank",
    "dp_distance_segments_from_adj",
    "freeze_soft_targets",
    "iter_adaround_quantizers",
    "iter_lowrank_parametrizes",
    "lp_loss",
    "make_widths_fixedK",
    "make_widths_from_adjacent_dist",
    "plan_loop_sizes",
    "reconstruct_module",
    "reconstruct_module_lowrank",
    "reconstruct_tib",
    "reset_adam_momentum",
    "steps_per_epoch_like",
]

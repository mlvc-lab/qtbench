# -*- coding: utf-8 -*-
"""FastDM reconstruction calibration configurations."""

from dataclasses import dataclass, field

from omniconfig import configclass

__all__ = [
    "AdaRoundConfig",
    "LowRankRecipeConfig",
    "ProgressiveConfig",
    "TimestepGroupConfig",
    "TibReconConfig",
    "FastDmCalibConfig",
]


@configclass
@dataclass
class AdaRoundConfig:
    """AdaRound learnable-rounding configuration.

    Args:
        enable (`bool`, default `True`): Whether AdaRound is active. Disabling
            short-circuits the entire FastDM calibration path.
        rmode (`str`, default `"learned_hard_sigmoid"`): Only the
            ``learned_hard_sigmoid`` regime is implemented in this port.
        weight_loss_w (`float`, default `0.001`): Per-layer relaxation weight.
        block_loss_w (`float`, default `0.01`): Per-block relaxation weight.
        b_range_start (`int`, default `20`): Initial rounding temperature.
        b_range_end (`int`, default `2`): Final rounding temperature.
        warmup (`float`, default `0.0`): Fraction of iterations spent on
            pure reconstruction loss before the rounding penalty kicks in.
        decay_start (`float`, default `0.0`): Relative offset of the
            rounding-temperature decay onset.
        p (`float`, default `2.0`): Lp-norm exponent for the rec loss.
        bits (`int`, default `4`): Bit-width of the AdaRound quantizer.
        symmetric (`bool`, default `False`): Whether to use symmetric int range.
    """

    enable: bool = True
    rmode: str = "learned_hard_sigmoid"
    weight_loss_w: float = 0.001
    block_loss_w: float = 0.01
    b_range_start: int = 20
    b_range_end: int = 2
    warmup: float = 0.0
    decay_start: float = 0.0
    p: float = 2.0
    bits: int = 4
    symmetric: bool = False

    @property
    def b_range(self) -> tuple:
        return (self.b_range_start, self.b_range_end)


@configclass
@dataclass
class ProgressiveConfig:
    """Progressive timestep scheduling.

    Args:
        direction (`str`, default `"reverse"`): One of ``"reverse"``
            (cumulative from noisy end), ``"forward"`` (cumulative from
            clean end), or ``"none"`` (single full-set loop).
        epoch_per_loop (`int`, default `1`): Optimisation epochs per loop.
        batch_size (`int`, default `32`): Reconstruction batch size.
    """

    direction: str = "reverse"
    epoch_per_loop: int = 1
    batch_size: int = 32

    @property
    def is_progressive(self) -> bool:
        return self.direction.lower() not in ("", "none", "null", "off")


@configclass
@dataclass
class TimestepGroupConfig:
    """Adaptive timestep grouping.

    Args:
        enable (`bool`, default `True`): When false, every unique timestep
            forms its own progressive bin.
        num_groups (`int`, default `5`): Target number of timestep bins.
        mode (`str`, default `"adaptive"`): ``"adaptive"`` enables the
            DP-based distance-aware partitioning; ``"uniform"`` falls back
            to even splits and ignores ``feature_dist_path``.
        feature_dist_path (`str`, default `""`): Path to a ``*.npy``
            holding the precomputed adjacent-distance vector. Required for
            ``mode="adaptive"``.
    """

    enable: bool = True
    num_groups: int = 5
    mode: str = "adaptive"
    feature_dist_path: str = ""


@configclass
@dataclass
class LowRankRecipeConfig:
    """Trained-LoRA error-compensation recipe for FastDM.

    Replaces SVDQuant's analytic SVD low-rank branch with an Adam-trained
    one driven by FastDM's progressive timestep schedule. The block forward
    during training is ``RTN(W - b@a) @ x + b(a(x))`` -- an STE-rounded
    residual with the rank-r branch absorbing the quantization error. Only
    ``branch.{a,b}`` are trainable; the underlying weight stays frozen.

    Args:
        enable (`bool`, default `False`): Master toggle for the recipe.
        rank (`int`, default `32`): Branch rank.
        bits (`int`, default `4`): Bit-width of the per-step RTN simulator.
        symmetric (`bool`, default `False`): RTN range symmetry.
        lr (`float`, default `1e-3`): Adam learning rate for ``a, b``.
        weight_decay (`float`, default `0.0`): Adam weight decay.
        ste_rtn (`bool`, default `True`): Whether the residual rounding is
            wrapped in a straight-through estimator. Set false to make the
            branch see the unrounded residual (debug only).
    """

    enable: bool = False
    rank: int = 32
    bits: int = 4
    symmetric: bool = False
    lr: float = 1e-3
    weight_decay: float = 0.0
    ste_rtn: bool = True


@configclass
@dataclass
class TibReconConfig:
    """Temporal-information-block reconstruction.

    Args:
        enable (`bool`, default `True`): Whether to run the one-shot TIB
            pass before per-block reconstruction.
        iters (`int`, default `20000`): Adam iterations for TIB.
        lr (`float`, default `4e-5`): TIB learning rate (AdaRound alpha).
    """

    enable: bool = True
    iters: int = 20000
    lr: float = 4e-5


@configclass
@dataclass
class FastDmCalibConfig:
    """Top-level FastDM calibration configuration.

    Two recipes share the temporal-grouping + progressive scheduling
    machinery:

    * ``recipe="adaround"`` (default): trains AdaRound rounding offsets on
      the model weights -- the original Fast_DM_PTQ behaviour.
    * ``recipe="lowrank"``: trains a LoRA-style low-rank branch to
      compensate for fixed-RTN quantization error -- a SVDQuant-style
      error-compensation pass that is independent of the underlying
      rounding scheme.

    Only the active recipe's settings are consulted. The progressive,
    timestep-group, keep_gpu, and logging settings apply to both.

    Args:
        recipe (`str`, default `"adaround"`): Which recipe to run; one of
            ``"adaround"`` or ``"lowrank"``.
        adaround (`AdaRoundConfig`): AdaRound parameters.
        lowrank (`LowRankRecipeConfig`): LoRA-style branch parameters.
        progressive (`ProgressiveConfig`): Progressive scheduling.
        timestep_group (`TimestepGroupConfig`): Timestep grouping.
        tib (`TibReconConfig`): TIB reconstruction (AdaRound recipe only).
        keep_gpu (`bool`, default `True`): Keep cached IO on GPU when memory permits.
        log_loss_curves (`bool`, default `False`): Record per-block loss
            curves into ``LossRecorder`` for later plotting.
        loss_curve_dir (`str`, default `""`): Directory for written curves.
        activation_running_stat (`bool`, default `False`): Use the
            momentum-based running-stat update for per-bucket activation
            calibration.
        activation_interval (`int`, default `128`): Number of samples per
            activation calibration bucket.
    """

    recipe: str = "adaround"
    adaround: AdaRoundConfig = field(default_factory=AdaRoundConfig)
    lowrank: LowRankRecipeConfig = field(default_factory=LowRankRecipeConfig)
    progressive: ProgressiveConfig = field(default_factory=ProgressiveConfig)
    timestep_group: TimestepGroupConfig = field(default_factory=TimestepGroupConfig)
    tib: TibReconConfig = field(default_factory=TibReconConfig)
    keep_gpu: bool = True
    log_loss_curves: bool = False
    loss_curve_dir: str = ""
    activation_running_stat: bool = False
    activation_interval: int = 128

    def __post_init__(self) -> None:
        recipe = (self.recipe or "adaround").lower()
        if recipe not in ("adaround", "lowrank"):
            raise ValueError(
                f"FastDmCalibConfig.recipe must be 'adaround' or 'lowrank', got {self.recipe!r}"
            )
        self.recipe = recipe

    def is_enabled(self) -> bool:
        if self.recipe == "lowrank":
            return self.lowrank.enable
        return self.adaround.enable

    def generate_dirnames(self, *, prefix: str = "", **kwargs):
        """Directory-name fragments used by the cache-path generator."""
        names = []
        if self.recipe == "lowrank":
            lr_cfg = self.lowrank
            names.append(
                f"lora.r{lr_cfg.rank}.b{lr_cfg.bits}.lr{lr_cfg.lr:g}"
            )
        else:
            ar = self.adaround
            names.append(f"adar.b{ar.bits}.w{ar.weight_loss_w:.4g}.B{ar.block_loss_w:.4g}")
        if self.progressive.is_progressive:
            names.append(
                f"prog.{self.progressive.direction}.e{self.progressive.epoch_per_loop}"
            )
        else:
            names.append(f"prog.full.e{self.progressive.epoch_per_loop}")
        if self.timestep_group.enable:
            names.append(f"tg.{self.timestep_group.mode}.K{self.timestep_group.num_groups}")
        if self.recipe == "adaround" and self.tib.enable:
            names.append(f"tib.i{self.tib.iters}.lr{self.tib.lr:g}")
        if prefix:
            names = [f"{prefix}.{name}" for name in names]
        return names

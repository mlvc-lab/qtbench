# -*- coding: utf-8 -*-
"""LZS Quantization kernel."""

from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Union

import torch
import torch.nn.functional as F
from omniconfig import configclass
from torch import nn

from ...data.dtype import QuantDataType
from ...data.range import QuantRange
from ...data.zero import ZeroPointDomain
from ...utils.common import num2str
from ...utils.tools import logging
from ..config.kernel import BaseQuantKernel, BaseQuantKernelConfig
from ..impl.ste import ste
from .rtn import rtn_quantize

__all__ = ["QuantLzsConfig", "QuantLzsKernel", "lzs_quantize"]


@configclass
@dataclass
class QuantLzsConfig(BaseQuantKernelConfig):
    """
    Configuration for LZS quantization.

    Args:
        bits (`int`, *optional*, defaults to `4`):
            The target number of bits. Defaults to 4.
        base (`int`, *optional*, defaults to `8`):
            The target number of base bits. Defaults to 8.
        group_size (`int` or `torch.Tensor`, *optional*, defaults to `16`):
            The group size for quantization. Defaults to 16.
    """

    bits: int = 4
    base: int = 8
    group_size: int = 16

    @property
    def name(self) -> str:
        return "LZS"

    def build(self) -> "QuantLzsKernel":
        return QuantLzsKernel(self)

    def generate_dirnames(self, *, prefix: str = "", **kwargs) -> list[str]:
        """Generate the directory names of the configuration.

        Args:
            prefix (`str`, *optional*, defaults to `""`):
                The prefix for the directory names.

        Returns:
            `list[str]`:
                The directory names.
        """
        name = f"{num2str(self.base)}2{num2str(self.bits)}.g{num2str(self.group_size)}"
        return [f"{prefix}.{name}" if prefix else name]


class QuantLzsKernel(BaseQuantKernel):
    """LZS Quantization kernel."""

    def __init__(self, config: "QuantLzsConfig"):
        self.config = config

    def quantize(
        self,
        tensor: torch.Tensor,
        *,
        view_shape: torch.Size,
        quant_dtype: QuantDataType,
        zero_domain: ZeroPointDomain | None,
        scale: torch.Tensor,
        zero: torch.Tensor,
        quant_range: QuantRange | None = None,
        use_lzs: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Quantize the tensor."""
        qtensor = rtn_quantize(
            tensor,
            view_shape=view_shape,
            quant_dtype=quant_dtype,
            zero_domain=zero_domain,
            scale=scale,
            zero=zero,
            quant_range=quant_range,
        )
        if not use_lzs:
            return qtensor
        qtensor = lzs_quantize(
            qtensor,
            lzs_config=self.config,
            is_signed=quant_dtype.signed,
        )
        return qtensor


def lzs_quantize(
    x: torch.Tensor,
    *,
    lzs_config: QuantLzsConfig,
    is_signed: bool = True,
) -> torch.Tensor:
    """
    Applies LZS quantization.
    """

    bits = lzs_config.bits
    base = lzs_config.base
    group_size = lzs_config.group_size

    original_shape = x.shape
    original_dtype = x.dtype

    x_min, x_max = x.min().item(), x.max().item()

    sign = torch.sign(x)
    raw_x = torch.abs(x).detach().clone()

    group_size = int(group_size)
    assert group_size > 0, "Group size must be positive"

    raw_x_flat = raw_x.flatten()
    original_length = raw_x_flat.numel()

    if original_length == 0:
        return x

    num_groups = (original_length + group_size - 1) // group_size
    padded_length = num_groups * group_size
    vacant_num = padded_length - original_length
    if vacant_num > 0:
        raw_x_flat = F.pad(raw_x_flat, (0, vacant_num), "constant", 0)

    raw_x_groups = raw_x_flat.view(-1, group_size)
    max_dim1, _ = raw_x_groups.max(dim=1)

    for bit in range(bits, base):
        mul_xth = 2 ** (bit - 1) if is_signed else (2**bit)
        round_value = 2 ** (bit + 1 - bits)
        outlier_id_mask = torch.bitwise_and(max_dim1 >= mul_xth, max_dim1 < mul_xth * 2)

        if outlier_id_mask.any():
            outlier_id_expanded = outlier_id_mask.unsqueeze(-1)
            groups_to_process = raw_x_groups
            processed_values = (
                ste(groups_to_process / round_value, torch.round) * round_value
            )
            raw_x_groups = torch.where(
                outlier_id_expanded, processed_values, raw_x_groups
            )

    raw_x_groups = torch.clamp(raw_x_groups, 0, 2**base - 1)
    processed_flat = raw_x_groups.flatten()
    if vacant_num > 0:
        processed_flat = processed_flat[:-vacant_num]
    processed_abs = processed_flat.view(original_shape)

    decompressed_x = processed_abs * sign
    decompressed_x = (
        ste(decompressed_x, torch.round).clamp(x_min, x_max).to(original_dtype)
    )

    return decompressed_x


def lzs_quantize_signed(
    x: torch.Tensor,
    *,
    lzs_config: QuantLzsConfig,
) -> torch.Tensor:
    """
    Applies LZS quantization.
    """

    # sign + magnitude
    original_shape = x.shape
    original_dtype = x.dtype
    sign = torch.sign(x)
    a = torch.abs(x).clamp_max_(127)  # keep in INT8-like magnitude range

    # MSB per element using frexp: MSB = exp for a>0 else 0
    _, exp = torch.frexp(a)
    msb = torch.where(a > 0, exp, torch.zeros_like(exp))  # same shape as a

    # Block-wise max MSB (shared across last dim = block_size)
    msb_blk = msb.amax(dim=-1, keepdim=True)

    # Shared FLAG per block: max(MSB) - 3, clamped at 0
    flag = (msb_blk - 3).clamp_min_(0)

    # scale = 2^flag (shared per block)
    scale = torch.ldexp(torch.ones_like(a), flag)

    # quantize each element using shared scale
    q = torch.round(a / scale).clamp_(0, 7) * scale
    q = q.view(original_shape).to(original_dtype)
    return sign * q


def lzs_quantize_unsigned(
    x: torch.Tensor,
    *,
    lzs_config: QuantLzsConfig,
) -> torch.Tensor:
    """
    Applies LZS quantization.
    """
    """
    Real per-block LZS (shared flag per block).
    Expects x shaped [..., block_size]. Computes a single flag per block from max MSB in the block.

    For bits=4, we keep a 4-bit mantissa [0..15] and a shared power-of-two shift per block.
    """

    original_shape = x.shape
    original_dtype = x.dtype
    sign = torch.sign(x)

    x = x.view(-1, lzs_config.group_size)  # [..., block_size]
    x = torch.clamp(x, min=0, max=255)

    # MSB per element using frexp: MSB = exp for a>0 else 0
    _, exp = torch.frexp(x)
    msb = torch.where(x > 0, exp, torch.zeros_like(exp))  # same shape as a

    # Block-wise max MSB (shared across last dim = block_size)
    msb_blk = msb.amax(dim=-1, keepdim=True)

    # Shared FLAG per block: max(MSB) - 4, clamped at 0
    flag = (msb_blk - 4).clamp_min_(0)

    # scale = 2^flag (shared per block)
    scale = torch.ldexp(torch.ones_like(x), flag)

    # quantize each element using shared scale
    q = torch.round(x / scale).clamp_(0, 15) * scale
    q = q.view(original_shape).to(original_dtype)
    return sign * q

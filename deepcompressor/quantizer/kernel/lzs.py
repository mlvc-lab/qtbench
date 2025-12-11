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
        return lzs_quantize(
            qtensor,
            lzs_config=self.config,
        )


def lzs_quantize(
    x: torch.Tensor,
    *,
    lzs_config: QuantLzsConfig,
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
        mul_xth = 2 ** (bit - 1)  # if is_processed_symmetric else (2**bit)
        round_value = 2 ** (bit + 1 - bits)
        outlier_id_mask = torch.bitwise_and(max_dim1 >= mul_xth, max_dim1 < mul_xth * 2)

        if outlier_id_mask.any():
            outlier_id_expanded = outlier_id_mask.unsqueeze(-1)
            groups_to_process = raw_x_groups
            processed_values = ste(groups_to_process / round_value, torch.round) * round_value
            raw_x_groups = torch.where(outlier_id_expanded, processed_values, raw_x_groups)

    raw_x_groups = torch.clamp(raw_x_groups, 0, 2**base - 1)
    processed_flat = raw_x_groups.flatten()
    if vacant_num > 0:
        processed_flat = processed_flat[:-vacant_num]
    processed_abs = processed_flat.view(original_shape)

    decompressed_x = processed_abs * sign
    decompressed_x = ste(decompressed_x, torch.round).clamp(x_min, x_max).to(original_dtype)

    return decompressed_x

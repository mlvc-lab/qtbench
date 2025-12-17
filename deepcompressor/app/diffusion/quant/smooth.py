# -*- coding: utf-8 -*-
"""Diffusion smooth quantization module."""

import typing as tp
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from deepcompressor.calib.smooth import (
    ActivationSmoother,
    TimestepActivationSmoother,
    TimestepContext,
    smooth_linear_modules,
)
from deepcompressor.data.cache import IOTensorsCache, TensorCache, TensorsCache
from deepcompressor.quantizer import Quantizer
from deepcompressor.utils import tools
from deepcompressor.utils.hooks import KeyedInputPackager

from ..nn.struct import (
    DiffusionAttentionStruct,
    DiffusionBlockStruct,
    DiffusionFeedForwardStruct,
    DiffusionModelStruct,
    DiffusionTransformerBlockStruct,
)
from .config import DiffusionQuantConfig
from .utils import get_needs_inputs_fn, wrap_joint_attn

__all__ = ["smooth_diffusion"]

_TIMESTEP_CONTEXT: TimestepContext | None = None


def _get_timestep_context() -> TimestepContext | None:
    return _TIMESTEP_CONTEXT


def _set_timestep_context(ctx: TimestepContext | None) -> None:
    global _TIMESTEP_CONTEXT
    _TIMESTEP_CONTEXT = ctx


def _normalize_timestep_value(timestep: tp.Any) -> tp.Any:
    if timestep is None:
        return None
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 0:
            return None
        return timestep.flatten()[0].item()
    return timestep


def _format_cache_key(base_key: str, timestep: tp.Any | None) -> str:
    ts = _normalize_timestep_value(timestep)
    if ts is None:
        return base_key
    return f"t{ts}:{base_key}"


def _slice_tensor_cache(cache: TensorCache, start: int, end: int) -> TensorCache:
    data = [x[start:end].clone() for x in cache.data]
    return TensorCache(
        data=data,
        channels_dim=cache.channels_dim,
        reshape=cache.reshape,
        num_cached=min(cache.num_cached, end - start),
        num_total=end - start,
        num_samples=end - start,
        orig_device=cache.orig_device,
    )


def _slice_io_tensors_cache(cache: IOTensorsCache, start: int, end: int) -> IOTensorsCache:
    inputs = None
    if cache.inputs is not None:
        inputs = TensorsCache(OrderedDict({k: _slice_tensor_cache(v, start, end) for k, v in cache.inputs.items()}))
    outputs = _slice_tensor_cache(cache.outputs, start, end) if cache.outputs is not None else None
    return IOTensorsCache(inputs=inputs, outputs=outputs)


def _register_timestep_activation_smoother(
    target: nn.Module | tp.Sequence[nn.Module],
    timestep: tp.Any,
    scale: torch.Tensor,
    *,
    channels_dim: int,
    input_packager=None,
    output_packager=None,
    upscale: bool = False,
    develop_dtype: torch.dtype | None = None,
) -> None:
    targets = target if isinstance(target, (list, tuple)) else [target]
    ctx = _get_timestep_context()
    for tgt in targets:
        smoother: TimestepActivationSmoother | None = getattr(tgt, "_dc_timestep_smoother", None)
        if smoother is None:
            smoother = TimestepActivationSmoother(
                smooth_scale=scale.detach().cpu(),
                channels_dim=channels_dim,
                upscale=upscale,
                develop_dtype=develop_dtype,
                input_packager=input_packager,
                output_packager=output_packager,
                timestep_context=ctx,
            )
            smoother.as_hook().register(tgt)
            setattr(tgt, "_dc_timestep_smoother", smoother)
        smoother.register_scale(timestep, scale.detach().cpu())


@torch.inference_mode()
def smooth_diffusion_attention(
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # attention qk
    if config.smooth.enabled_attn:
        logger.debug("- %s.k", attn.name)
        raise NotImplementedError("Not implemented yet")
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_qkv_proj(
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # region qkv projection
    module_key = attn.qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.qkv_proj", attn.name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        prevs = None
        if config.smooth.proj.fuse_when_possible and attn.parent.norm_type.startswith("layer_norm"):
            if not hasattr(attn.parent.module, "pos_embed") or attn.parent.module.pos_embed is None:
                prevs = attn.parent.pre_attn_norms[attn.idx]
                assert isinstance(prevs, nn.LayerNorm)
        cache_key = _format_cache_key(attn.q_proj_name, timestep)
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.q_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=attn,
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            # we need to register forward pre hook to smooth inputs
            if attn.module.group_norm is None and attn.module.spatial_norm is None:
                target_module = attn.module
                input_packager = KeyedInputPackager(attn.module, [0])
            else:
                target_module = attn.qkv_proj
                input_packager = None
            if per_timestep:
                _register_timestep_activation_smoother(
                    target_module,
                    timestep,
                    smooth_cache[cache_key],
                    channels_dim=-1,
                    input_packager=input_packager,
                    develop_dtype=config.develop_dtype,
                )
            else:
                ActivationSmoother(
                    smooth_cache[cache_key],
                    channels_dim=-1,
                    input_packager=input_packager,
                ).as_hook().register(target_module)
        for m in attn.qkv_proj:
            m.in_smooth_cache_key = cache_key
    # endregion
    if attn.is_self_attn():
        return smooth_cache
    # region additional qkv projection
    module_key = attn.add_qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    needs_quant = needs_quant and attn.add_k_proj is not None
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s add_qkv_proj", attn.name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        prevs = None
        pre_attn_add_norm = attn.parent.pre_attn_add_norms[attn.idx]
        if isinstance(pre_attn_add_norm, nn.LayerNorm) and config.smooth.proj.fuse_when_possible:
            prevs = pre_attn_add_norm
        cache_key = _format_cache_key(attn.add_k_proj_name, timestep)
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.add_qkv_proj,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_k_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=wrap_joint_attn(attn, indexes=1) if attn.is_joint_attn() else attn,
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            # we need to register forward pre hook to smooth inputs
            if per_timestep:
                _register_timestep_activation_smoother(
                    attn.add_qkv_proj,
                    timestep,
                    smooth_cache[cache_key],
                    channels_dim=-1,
                    develop_dtype=config.develop_dtype,
                )
            else:
                ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(attn.add_qkv_proj)
        for m in attn.add_qkv_proj:
            m.in_smooth_cache_key = cache_key
    # endregion
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_out_proj(  # noqa: C901
    attn: DiffusionAttentionStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    module_keys = []
    per_timestep = getattr(config.smooth, "per_timestep", False)
    for module_key in (attn.out_proj_key, attn.add_out_proj_key) if attn.is_joint_attn() else (attn.out_proj_key,):
        needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
        needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
        if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
            module_keys.append(module_key)
    if not module_keys:
        return smooth_cache
    exclusive = False
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        exclusive = config.wgts.low_rank.exclusive
        config.wgts.low_rank.exclusive = True
    fuse_smooth = not attn.config.linear_attn and config.smooth.proj.fuse_when_possible
    prevs = [attn.v_proj, attn.add_v_proj] if fuse_smooth else None
    if len(module_keys) == 1 and module_keys[0] == attn.out_proj_key:
        logger.debug("- %s.out_proj", attn.name)
        module_key = attn.out_proj_key
        cache_key = _format_cache_key(attn.o_proj_name, timestep)
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.o_proj,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_module=attn.o_proj,
            extra_modules=[attn.add_o_proj] if attn.is_joint_attn() else None,
            develop_dtype=config.develop_dtype,
        )
    elif len(module_keys) == 1 and module_keys[0] == attn.add_out_proj_key:
        assert attn.is_joint_attn()
        logger.debug("- %s.add_out_proj", attn.name)
        module_key = attn.add_out_proj_key
        cache_key = _format_cache_key(attn.add_o_proj_name, timestep)
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            attn.add_o_proj,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.add_o_proj_name].inputs if block_cache else None,
            eval_module=attn.add_o_proj,
            extra_modules=[attn.o_proj],
            develop_dtype=config.develop_dtype,
        )
    else:
        assert attn.is_joint_attn()
        logger.debug("- %s.out_proj + %s.add_out_proj", attn.name, attn.name)
        module_key = attn.out_proj_key
        cache_key = _format_cache_key(attn.o_proj_name, timestep)
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            [attn.o_proj, attn.add_o_proj],
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.o_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[attn.name].inputs if block_cache else None,
            eval_module=wrap_joint_attn(attn, indexes=(0, 1)),
            eval_kwargs=attn.filter_kwargs(block_kwargs),
            develop_dtype=config.develop_dtype,
        )
    if config.enabled_wgts and config.wgts.enabled_low_rank:
        config.wgts.low_rank.exclusive = exclusive
    if fuse_smooth:
        for prev in prevs:
            if prev is not None:
                prev.out_smooth_cache_key = cache_key
    else:
        for o_proj in [attn.o_proj, attn.add_o_proj]:
            if o_proj is not None:
                if per_timestep:
                    _register_timestep_activation_smoother(
                        o_proj, timestep, smooth_cache[cache_key], channels_dim=-1, develop_dtype=config.develop_dtype
                    )
                else:
                    ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(o_proj)
    attn.o_proj.in_smooth_cache_key = cache_key
    if attn.add_o_proj is not None:
        attn.add_o_proj.in_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_up_proj(
    pre_ffn_norm: nn.Module,
    ffn: DiffusionFeedForwardStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    assert len(ffn.up_projs) == 1
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # ffn up projection
    module_key = ffn.up_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.up_proj", ffn.name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        prevs = None
        if config.smooth.proj.fuse_when_possible and isinstance(pre_ffn_norm, nn.LayerNorm):
            if ffn.parent.norm_type in ["ada_norm", "layer_norm"]:
                prevs = pre_ffn_norm
        cache_key = _format_cache_key(ffn.up_proj_name, timestep)
        channels_dim = -1 if isinstance(ffn.down_proj, nn.Linear) else 1
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            prevs,
            ffn.up_projs,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=channels_dim, key=module_key),
            inputs=block_cache[ffn.up_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[ffn.up_proj_name].inputs if block_cache else None,
            eval_module=ffn.up_proj,
            develop_dtype=config.develop_dtype,
        )
        if prevs is None:
            if per_timestep:
                _register_timestep_activation_smoother(
                    ffn.up_proj,
                    timestep,
                    smooth_cache[cache_key],
                    channels_dim=channels_dim,
                    develop_dtype=config.develop_dtype,
                )
            else:
                ActivationSmoother(smooth_cache[cache_key], channels_dim=channels_dim).as_hook().register(ffn.up_proj)
        for proj in ffn.up_projs:
            proj.in_smooth_cache_key = cache_key
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_down_proj(
    ffn: DiffusionFeedForwardStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # ffn down projection
    module_key = ffn.down_proj_key.upper()
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.down_proj", ffn.name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        cache_key = _format_cache_key(ffn.down_proj_name, timestep)
        config_ipts = config.unsigned_ipts if getattr(ffn.down_proj, "unsigned", False) else config.ipts
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        channels_dim = -1 if isinstance(ffn.down_proj, nn.Linear) else 1
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            ffn.down_proj,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config_ipts, channels_dim=channels_dim, key=module_key),
            inputs=block_cache[ffn.down_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[ffn.down_proj_name].inputs if block_cache else None,
            eval_module=ffn.down_proj,
            develop_dtype=config.develop_dtype,
        )
        ffn.down_proj.in_smooth_cache_key = cache_key
        if per_timestep:
            _register_timestep_activation_smoother(
                ffn.down_proj,
                timestep,
                smooth_cache[cache_key],
                channels_dim=channels_dim,
                develop_dtype=config.develop_dtype,
            )
        else:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=channels_dim).as_hook().register(ffn.down_proj)
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_parallel_qkv_up_proj(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    assert block.parallel
    assert len(block.ffn_struct.up_projs) == 1
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    # region qkv proj + up proj
    attn, ffn = block.attn_structs[0], block.ffn_struct
    module_key = attn.qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- %s.qkv_proj + %s.up_proj", attn.name, ffn.name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        cache_key = _format_cache_key(attn.q_proj_name, timestep)
        modules = attn.qkv_proj + ffn.up_projs
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            modules,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.q_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[block.name].inputs if block_cache else None,
            eval_module=block,
            eval_kwargs=block_kwargs,
            splits=[len(attn.qkv_proj)],
            develop_dtype=config.develop_dtype,
        )
        if per_timestep:
            _register_timestep_activation_smoother(
                modules, timestep, smooth_cache[cache_key], channels_dim=-1, develop_dtype=config.develop_dtype
            )
        else:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(modules)
        for m in modules:
            m.in_smooth_cache_key = cache_key
    # endregion
    # region add qkv proj + add up proj
    if attn.is_self_attn():
        if block.add_ffn_struct is not None:
            smooth_cache = smooth_diffusion_up_proj(
                pre_ffn_norm=block.pre_add_ffn_norm,
                ffn=block.add_ffn_struct,
                config=config,
                smooth_cache=smooth_cache,
                block_cache=block_cache,
                timestep=timestep,
            )
        return smooth_cache
    module_key = attn.add_qkv_proj_key
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        add_ffn = block.add_ffn_struct
        per_timestep = getattr(config.smooth, "per_timestep", False)
        cache_key = _format_cache_key(attn.add_k_proj_name, timestep)
        modules = attn.add_qkv_proj
        if add_ffn is None:
            logger.debug("- %s.add_qkv_proj", attn.name)
        else:
            logger.debug("- %s.add_qkv_proj + %s.up_proj", attn.name, add_ffn.name)
            modules = modules + add_ffn.up_projs
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            modules,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key, low_rank=config.wgts.low_rank),
            input_quantizer=Quantizer(config.ipts, channels_dim=-1, key=module_key),
            inputs=block_cache[attn.add_k_proj_name].inputs if block_cache else None,
            eval_inputs=block_cache[block.name].inputs if block_cache else None,
            eval_module=block,
            eval_kwargs=block_kwargs,
            develop_dtype=config.develop_dtype,
        )
        if per_timestep:
            _register_timestep_activation_smoother(
                modules, timestep, smooth_cache[cache_key], channels_dim=-1, develop_dtype=config.develop_dtype
            )
        else:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=-1).as_hook().register(modules)
        for m in modules:
            m.in_smooth_cache_key = cache_key
    # endregion
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_sequential_transformer_block(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    assert not block.parallel
    for attn in block.attn_structs:
        smooth_cache = smooth_diffusion_attention(
            attn=attn,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            block_kwargs=block_kwargs,
            timestep=timestep,
        )
        smooth_cache = smooth_diffusion_qkv_proj(
            attn=attn,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            block_kwargs=block_kwargs,
            timestep=timestep,
        )
        smooth_cache = smooth_diffusion_out_proj(
            attn=attn,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            block_kwargs=block_kwargs,
            timestep=timestep,
        )
    if block.ffn_struct is not None:
        smooth_cache = smooth_diffusion_up_proj(
            pre_ffn_norm=block.pre_ffn_norm,
            ffn=block.ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            timestep=timestep,
        )
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache, timestep=timestep
        )
    if block.add_ffn_struct is not None:
        smooth_cache = smooth_diffusion_up_proj(
            pre_ffn_norm=block.pre_add_ffn_norm,
            ffn=block.add_ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            timestep=timestep,
        )
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.add_ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            timestep=timestep,
        )
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_parallel_transformer_block(
    block: DiffusionTransformerBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    block_cache: dict[str, IOTensorsCache] | None = None,
    block_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    assert block.parallel
    assert block.ffn_struct is not None
    for attn in block.attn_structs:
        smooth_cache = smooth_diffusion_attention(
            attn=attn,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            block_kwargs=block_kwargs,
            timestep=timestep,
        )
        if attn.idx == 0:
            smooth_cache = smooth_diffusion_parallel_qkv_up_proj(
                block=block,
                config=config,
                smooth_cache=smooth_cache,
                block_cache=block_cache,
                block_kwargs=block_kwargs,
                timestep=timestep,
            )
        else:
            smooth_cache = smooth_diffusion_qkv_proj(
                attn=attn,
                config=config,
                smooth_cache=smooth_cache,
                block_cache=block_cache,
                block_kwargs=block_kwargs,
                timestep=timestep,
            )
        smooth_cache = smooth_diffusion_out_proj(
            attn=attn,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            block_kwargs=block_kwargs,
            timestep=timestep,
        )
    smooth_cache = smooth_diffusion_down_proj(
        ffn=block.ffn_struct, config=config, smooth_cache=smooth_cache, block_cache=block_cache, timestep=timestep
    )
    if block.add_ffn_struct is not None:
        smooth_cache = smooth_diffusion_down_proj(
            ffn=block.add_ffn_struct,
            config=config,
            smooth_cache=smooth_cache,
            block_cache=block_cache,
            timestep=timestep,
        )
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_module(
    module_key: str,
    module_name: str,
    module: nn.Linear | nn.Conv2d,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    timestep: tp.Any | None = None,
) -> dict[str, torch.Tensor]:
    assert isinstance(module, (nn.Linear, nn.Conv2d))
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
    needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
    if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
        logger.debug("- Smoothing Module %s", module_name)
        per_timestep = getattr(config.smooth, "per_timestep", False)
        tools.logging.Formatter.indent_inc()
        logger.debug("- %s", module_name)
        cache_key = _format_cache_key(module_name, timestep)
        channels_dim = -1 if isinstance(module, nn.Linear) else 1
        config_wgts = config.wgts
        if config.enabled_extra_wgts and config.extra_wgts.is_enabled_for(module_key):
            config_wgts = config.extra_wgts
        smooth_cache[cache_key] = smooth_linear_modules(
            None,
            module,
            scale=smooth_cache.get(cache_key, None),
            apply_weight=not per_timestep,
            config=config.smooth.proj,
            weight_quantizer=Quantizer(config_wgts, key=module_key),
            input_quantizer=Quantizer(config.ipts, channels_dim=channels_dim, key=module_key),
            inputs=layer_cache[module_name].inputs if layer_cache else None,
            eval_inputs=layer_cache[module_name].inputs if layer_cache else None,
            eval_module=module,
            develop_dtype=config.develop_dtype,
        )
        if per_timestep:
            _register_timestep_activation_smoother(
                module,
                timestep,
                smooth_cache[cache_key],
                channels_dim=channels_dim,
                develop_dtype=config.develop_dtype,
            )
        else:
            ActivationSmoother(smooth_cache[cache_key], channels_dim=channels_dim).as_hook().register(module)
        module.in_smooth_cache_key = cache_key
        tools.logging.Formatter.indent_dec()
    else:
        logger.debug("- Skipping Module %s", module_name)
    return smooth_cache


@torch.inference_mode()
def smooth_diffusion_layer(
    layer: DiffusionBlockStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor],
    layer_cache: dict[str, IOTensorsCache] | None = None,
    layer_kwargs: dict[str, tp.Any] | None = None,
    timestep: tp.Any | None = None,
) -> None:
    """Smooth a single diffusion model block.

    Args:
        layer (`DiffusionBlockStruct`):
            The diffusion block.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`):
            The smoothing scales cache.
        layer_cache (`dict[str, IOTensorsCache]`, *optional*):
            The layer cache.
        layer_kwargs (`dict[str, tp.Any]`, *optional*):
            The layer keyword arguments.
    """
    logger = tools.logging.getLogger(f"{__name__}.SmoothQuant")
    logger.debug("- Smoothing Diffusion Block %s", layer.name)
    tools.logging.Formatter.indent_inc()
    layer_cache = layer_cache or {}
    layer_kwargs = layer_kwargs or {}
    # We skip resnets since we currently cannot scale the Swish function
    visited: set[str] = set()
    for module_key, module_name, module, parent, _ in layer.named_key_modules():
        if isinstance(parent, (DiffusionAttentionStruct, DiffusionFeedForwardStruct)):
            block = parent.parent
            assert isinstance(block, DiffusionTransformerBlockStruct)
            if block.name not in visited:
                logger.debug("- Smoothing Transformer Block %s", block.name)
                visited.add(block.name)
                tools.logging.Formatter.indent_inc()
                if block.parallel:
                    smooth_cache = smooth_diffusion_parallel_transformer_block(
                        block=block,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                        block_kwargs=layer_kwargs,
                        timestep=timestep,
                    )
                else:
                    smooth_cache = smooth_diffusion_sequential_transformer_block(
                        block=block,
                        config=config,
                        smooth_cache=smooth_cache,
                        block_cache=layer_cache,
                        block_kwargs=layer_kwargs,
                        timestep=timestep,
                    )
                tools.logging.Formatter.indent_dec()
        elif isinstance(module, (nn.Linear, nn.Conv2d)):
            smooth_cache = smooth_diffusion_module(
                module_key=module_key,
                module_name=module_name,
                module=module,
                config=config,
                smooth_cache=smooth_cache,
                layer_cache=layer_cache,
                timestep=timestep,
            )
        else:
            needs_quant = config.enabled_wgts and config.wgts.is_enabled_for(module_key)
            needs_quant = needs_quant or (config.enabled_ipts and config.ipts.is_enabled_for(module_key))
            if needs_quant and config.smooth.enabled_proj and config.smooth.proj.is_enabled_for(module_key):
                raise NotImplementedError(f"Module {module_name} is not supported for smoothing")
            logger.debug("- Skipping Module %s", module_name)
    tools.logging.Formatter.indent_dec()


@torch.inference_mode()
def smooth_diffusion(
    model: nn.Module | DiffusionModelStruct,
    config: DiffusionQuantConfig,
    smooth_cache: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Smooth the diffusion model.

    Args:
        model (`nn.Module` or `DiffusionModelStruct`):
            The diffusion model.
        config (`DiffusionQuantConfig`):
            The quantization configuration.
        smooth_cache (`dict[str, torch.Tensor]`, *optional*):
            The smoothing scales cache.

    Returns:
        `dict[str, torch.Tensor]`:
            The smoothing scales cache.
    """
    if not isinstance(model, DiffusionModelStruct):
        model = DiffusionModelStruct.construct(model)
    assert isinstance(model, DiffusionModelStruct)
    smooth_cache = smooth_cache or {}
    per_timestep = config.enabled_smooth and getattr(config.smooth, "per_timestep", False)
    if per_timestep:
        ctx = TimestepContext()
        _set_timestep_context(ctx)

        def _capture_timestep(module, args, kwargs):
            ts = kwargs.get("timestep", None)
            ctx.set(ts)

        model.module.register_forward_pre_hook(_capture_timestep, with_kwargs=True)
    if config.smooth.enabled_proj:
        if smooth_cache:
            assert smooth_cache.get("proj.fuse_when_possible", True) == config.smooth.proj.fuse_when_possible
    if config.smooth.enabled_attn:
        if smooth_cache:
            assert smooth_cache.get("attn.fuse_when_possible", True) == config.smooth.attn.fuse_when_possible
    if not smooth_cache:
        with tools.logging.redirect_tqdm():
            for _, (layer, layer_cache, layer_kwargs) in tqdm(
                config.calib.build_loader().iter_layer_activations(
                    model,
                    needs_inputs_fn=get_needs_inputs_fn(model, config),
                    skip_pre_modules=True,
                    skip_post_modules=True,
                ),
                desc="smoothing",
                leave=False,
                total=model.num_blocks,
                dynamic_ncols=True,
            ):
                if per_timestep:
                    timestep_batches = layer_kwargs.get("timestep_batches", [])
                    batch_size = layer_kwargs.get("batch_size", config.calib.batch_size)
                    base_kwargs = {k: v for k, v in layer_kwargs.items() if k not in ("timestep_batches", "batch_size")}
                    for batch_idx, ts in enumerate(timestep_batches):
                        ts_norm = _normalize_timestep_value(ts)
                        start = batch_idx * batch_size
                        end = start + (ts.shape[0] if isinstance(ts, torch.Tensor) else batch_size)
                        sliced_cache = {k: _slice_io_tensors_cache(v, start, end) for k, v in layer_cache.items()}
                        step_kwargs = dict(base_kwargs)
                        step_kwargs["timestep"] = ts
                        smooth_diffusion_layer(
                            layer=layer,
                            config=config,
                            smooth_cache=smooth_cache,
                            layer_cache=sliced_cache,
                            layer_kwargs=step_kwargs,
                            timestep=ts_norm,
                        )
                else:
                    smooth_diffusion_layer(
                        layer=layer,
                        config=config,
                        smooth_cache=smooth_cache,
                        layer_cache=layer_cache,
                        layer_kwargs=layer_kwargs,
                        timestep=layer_kwargs.get("timestep", None),
                    )
    else:
        for layer in model.block_structs:
            smooth_diffusion_layer(layer=layer, config=config, smooth_cache=smooth_cache)
    if config.smooth.enabled_proj:
        smooth_cache.setdefault("proj.fuse_when_possible", config.smooth.proj.fuse_when_possible)
    if config.smooth.enabled_attn:
        smooth_cache.setdefault("attn.fuse_when_possible", config.smooth.attn.fuse_when_possible)
    return smooth_cache

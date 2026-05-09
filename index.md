# Repository Index

This repository is a fork of MIT Han Lab's DeepCompressor toolbox. It provides PyTorch-based post-training quantization (PTQ) pipelines for large language models and diffusion models, plus calibration routines, quantizer primitives, model-structure adapters, and deployment converters for TinyChat, QServe, and Nunchaku.

The current package name in `pyproject.toml` is `deepcompressor-toolkit`. The Python package is `deepcompressor`. The diffusion metric dependency `image_reward` is currently declared as a normal wildcard package dependency rather than a Git dependency.

## Top-Level Layout

```text
deepcompressor/
  app/
    llm/                 LLM PTQ pipeline, model builders, quantization flows, evaluation.
    diffusion/           Diffusion PTQ pipeline, pipeline builders, quantization flows, datasets, evaluation.
  backend/               Conversion utilities for deployment runtimes.
  calib/                 Search-based calibration, range search, smoothing, rotation, reordering, low-rank branches.
  data/                  Quantization data types, ranges, scales, tensor/cache wrappers.
  dataset/               Generic activation-cache actions and calibration cache loaders.
  nn/                    Shared model-structure abstractions and module patches.
  quantizer/             Quantizer configs, kernels, tensor quantizer implementation.
  utils/                 Config helpers, hooks, logging, math, misc utilities.
  csrc/                  CUDA extension source for quantization kernels.
examples/
  llm/                   LLM YAML configs, scripts, usage documentation.
  diffusion/             Diffusion YAML configs, prompt files, scripts, usage documentation.
assets/                  README and paper assets.
docs/                    Additional method notes, currently QuaRTZ.
```

## Primary User Workflows

### LLM PTQ

Main module:

```bash
python -m deepcompressor.app.llm.ptq \
  examples/llm/configs/qoq-gchn.yaml \
  --model-name llama-2-7b \
  --model-path /path/to/model
```

High-level flow:

1. `LlmPtqRunConfig.get_parser()` loads YAML/CLI config through `omniconfig`.
2. `LlmPtqRunConfig.__post_init__()` normalizes scale dtypes, GPU count, batch size, cache paths, output paths, and seeds.
3. `deepcompressor.app.llm.ptq.main()` builds the Hugging Face model/tokenizer through `LlmModelConfig.build()`.
4. `deepcompressor.app.llm.ptq.ptq()` constructs `LlmModelStruct`, decides which stages are enabled, and runs:
   - `rotate_llm()`
   - `reorder_llm()`
   - `smooth_llm()`
   - `quantize_llm_weights()`
   - `quantize_llm_activations()`
5. `main()` optionally evaluates with the configured LLM evaluators.

Important files:

- `deepcompressor/app/llm/ptq.py`
- `deepcompressor/app/llm/config.py`
- `deepcompressor/app/llm/model/config.py`
- `deepcompressor/app/llm/quant/config.py`
- `deepcompressor/app/llm/quant/weight.py`
- `deepcompressor/app/llm/quant/activation.py`
- `deepcompressor/app/llm/nn/struct.py`
- `examples/llm/configs/*.yaml`

### Diffusion PTQ

Main module:

```bash
python -m deepcompressor.app.diffusion.ptq \
  examples/diffusion/configs/model/flux.1-schnell.yaml \
  examples/diffusion/configs/svdquant/int4.yaml
```

High-level flow:

1. `DiffusionPtqRunConfig.get_parser()` loads YAML/CLI config.
2. `DiffusionPtqRunConfig.__post_init__()` resolves dtype-dependent calibration paths, reference roots, cache paths, output paths, and seeds.
3. `deepcompressor.app.diffusion.ptq.main()` builds a Diffusers pipeline with `DiffusionPipelineConfig.build()`.
4. It optionally quantizes text encoders by calling the LLM PTQ path.
5. It constructs a `DiffusionModelStruct` for the denoising model and calls `deepcompressor.app.diffusion.ptq.ptq()`.
6. `ptq()` runs:
   - `rotate_diffusion()`
   - `smooth_diffusion()`
   - `fastdm_diffusion()` when `quant.fastdm.adaround.enable` is true
   - `quantize_diffusion_weights()`
   - `quantize_diffusion_activations()`
7. `main()` optionally generates images and evaluates image metrics.

Important files:

- `deepcompressor/app/diffusion/ptq.py`
- `deepcompressor/app/diffusion/config.py`
- `deepcompressor/app/diffusion/pipeline/config.py`
- `deepcompressor/app/diffusion/quant/config.py`
- `deepcompressor/app/diffusion/quant/fastdm.py`
- `deepcompressor/app/diffusion/quant/fastdm_data.py`
- `deepcompressor/app/diffusion/quant/fastdm_tib.py`
- `deepcompressor/app/diffusion/quant/weight.py`
- `deepcompressor/app/diffusion/quant/activation.py`
- `deepcompressor/app/diffusion/nn/struct.py`
- `deepcompressor/calib/recon/`
- `examples/diffusion/configs/**/*.yaml`

## High-Impact Pipeline Functions

### `deepcompressor.app.llm.ptq.ptq()`

This is the central LLM quantization driver. It accepts a Hugging Face `PreTrainedModel` or `LlmModelStruct`, tokenizer, `LlmQuantConfig`, optional cache config, optional load/save directories, and save options.

Key responsibilities:

- Converts raw models to `LlmModelStruct`.
- Computes stage flags from `config.enabled_wgts`, `enabled_ipts`, `enabled_opts`, `enabled_rotation`, `enabled_reorder`, and `enabled_smooth`.
- Resolves cache files: `rotation.pt`, `reorder.pt`, `smooth.pt`, `wgts.pt`, `acts.pt`, `model.pt`.
- Loads existing quantized model checkpoints when possible.
- Saves or symlinks generated caches depending on `copy_on_save`.
- Preserves original weights when activation calibration needs an unquantized reference.
- Calls the actual algorithm stages in the correct order.

This function has high blast radius: changes here affect all LLM PTQ runs, cache compatibility, saved model layout, and evaluation behavior.

### `deepcompressor.app.diffusion.ptq.ptq()`

This is the central diffusion quantization driver. It accepts a `DiffusionModelStruct`, `DiffusionQuantConfig`, cache config, load/save directories, and extraction/save flags.

Key responsibilities:

- Converts raw modules to `DiffusionModelStruct`.
- Resolves diffusion cache files: `smooth.pt`, `branch.pt`, `wgts.pt`, `acts.pt`, `fastdm.pt`, `model.pt`.
- Handles low-rank branch checkpoint requirements when loading saved diffusion models.
- Runs rotation, smoothing, optional FastDM AdaRound, weight quantization, and activation quantization.
- Saves quantizer state, low-rank branch state, FastDM alpha state, scales, and model state.

This function has high blast radius for SVDQuant/QuaRTZ/FastDM-style workflows and deployment checkpoint compatibility.

### `quantize_llm_weights()` and `quantize_llm_layer_weights()`

Location: `deepcompressor/app/llm/quant/weight.py`

These functions calibrate and apply weight quantizers layer by layer.

Important behavior:

- Use calibration activations only when GPTQ or range calibration requires them.
- Iterate through `LlmTransformerBlockStruct` layers.
- Build `LlmWeightQuantizer` per module key.
- Calibrate dynamic range with `LlmWeightQuantizer.calibrate_dynamic_range()`.
- Quantize `nn.Linear.weight` in place.
- Optionally return a scale/zero state dict for saved model conversion.

### `quantize_llm_activations()` and `quantize_llm_layer_activations()`

Location: `deepcompressor/app/llm/quant/activation.py`

These functions calibrate activation quantizers and register quantization hooks.

Important behavior:

- Collect/cache input and output activations when static/range calibration requires them.
- Build activation quantizers for attention Q/K/V, attention output projection, FFN up projections, and FFN down projections.
- Share quantizers across related projections where appropriate, such as Q/K/V input projection branches.
- Register `ProcessHook`s on target modules so quantization occurs during forward calls.

### `quantize_diffusion_weights()`

Location: `deepcompressor/app/diffusion/quant/weight.py`

This is the diffusion weight quantization orchestrator.

Important behavior:

- Optionally calibrates low-rank branches before quantizing weights.
- Computes whether pre/post modules can be skipped based on quantizer skip keys.
- Calibrates weight quantizer state dicts when they are not already loaded.
- Applies quantization block by block.
- Returns weight quantizer cache, low-rank branch cache, and optional scale state.

High-impact helpers in the same file:

- `calibrate_diffusion_block_low_rank_branch()`: constructs and registers `LowRankBranch` modules for grouped or individual projections.
- `update_diffusion_block_weight_quantizer_state_dict()`: calibrates per-module weight quantizer states.
- `quantize_diffusion_block_weights()`: performs in-place module weight replacement.
- `load_diffusion_weights_state_dict()`: loads saved quantized diffusion state and reconstructs low-rank branch hooks first.

### `quantize_diffusion_activations()`

Location: `deepcompressor/app/diffusion/quant/activation.py`

This function calibrates diffusion activation quantizers and registers hooks across the model.

Important behavior:

- Groups related attention and FFN modules in parallel transformer blocks so shared input quantizers can be calibrated once.
- Handles `nn.Linear` and `nn.Conv2d` modules with different channel dimensions.
- Supports unsigned input quantizer variants through `config.unsigned_ipts`.
- Registers input/output hooks on all named key modules that have calibrated quantizers.

### `fastdm_diffusion()`

Location: `deepcompressor/app/diffusion/quant/fastdm.py`

This is the new diffusion FastDM AdaRound driver. It runs after smoothing and before static weight quantization when `DiffusionQuantConfig.enabled_fastdm` is true.

Important behavior:

- Collects target transformer/UNet blocks from `DiffusionModelStruct.block_structs`.
- Converts qdiff calibration caches into `FastDmSample` objects through `iter_fastdm_samples()`.
- Extracts timesteps with `extract_timesteps()` and builds progressive timestep loops.
- Optionally runs temporal-information-block reconstruction through `build_tib_bundle()` and `reconstruct_tib()`.
- Captures full-precision block inputs/outputs once, then calls `reconstruct_module()` per block and progressive loop.
- Moves cached calibration sample args/kwargs onto the denoiser device before FP capture, so CPU caches can be replayed against CUDA models.
- Captures block forward kwargs per calibration sample through `_CaptureHook`, then merges batched tensors such as `timestep`, `class_labels`, and masks while preserving static kwargs such as shared rotary embeddings.
- Bakes learned AdaRound rounding back into module weights and returns an alpha cache mapping `"<block>.<sub_module>.alpha"` to tensors.
- Reloads an existing `fastdm.pt` alpha cache without re-optimizing when available.

Supporting files:

- `deepcompressor/app/diffusion/quant/fastdm_data.py`: qdiff cache to FastDM sample adapters.
- `deepcompressor/app/diffusion/quant/fastdm_tib.py`: conservative TIB detection for UNet-style time embedding paths. `FastDmTibBundle` registers the existing time-embedding and `time_emb_proj` modules so `reconstruct_tib()` can find AdaRound targets through `.modules()`.
- `deepcompressor/calib/recon/`: AdaRound parametrization, reconstruction losses, capture helpers, and progressive scheduling.

## Quantizer Core

### `Quantizer`

Location: `deepcompressor/quantizer/processor.py`

`Quantizer` combines the low-level quantization implementation with the hook processor interface. It is used directly by LLM and diffusion quantizer adapters.

High-impact methods:

- `is_enabled_low_rank()`: checks whether low-rank branch quantization applies to the current key.
- `process()`: implements the `BaseTensorProcessor` contract for hook usage.
- `quantize()`: resolves default attributes, specializes key-enabled kernels, and delegates to `QuantizerImpl.quantize()`.
- `update()`: builds or refreshes `QuantInfo`.
- `quantize_with_low_rank()`: quantizes one or more tensors and optionally creates low-rank compensation branches.
- `state_dict()` / `load_state_dict()`: serialize and restore scale, zero point, range, and shape metadata.

### `QuantizerImpl`

Location: `deepcompressor/quantizer/impl/base.py`

This is the low-level implementation that performs multi-step quantization.

High-impact behavior:

- Reshapes tensors by channel dimension before quantization.
- Calls `update()` to construct `QuantInfo` from config and tensor shape.
- Supports scale-based and dynamic-range-based quantization.
- Quantizes intermediate scales through `QuantScale`.
- Uses `QuantRtnKernel` by default, or a supplied kernel such as GPTQ/LZS.
- Returns `QuantTensor` with dequantized data, optional integer-like quantized data, scale state, zero point, and view shape.

### Quantizer Configs

Shared configs:

- `BaseQuantizerConfig`
- `QuantizerConfig`
- `ProgressiveQuantizerConfig`
- `DecomposedQuantizerConfig`

Location: `deepcompressor/quantizer/config/base.py`

Important behavior:

- `QuantizerConfig` defines `dtype`, `zero_point`, `group_shapes`, and `scale_dtypes`.
- `ProgressiveQuantizerConfig` supports intermediate quantization dtypes and levels.
- `decompose()` converts progressive configurations into one or more simple quantization steps.
- `generate_dirnames()` encodes effective bits, dtype name, and group-shape naming for cache/output paths.

Application-specific configs:

- `LlmQuantizerConfig`, `LlmWeightQuantizerConfig`, `LlmActivationQuantizerConfig`, `LlmModuleQuantizerConfig`
- `DiffusionQuantizerConfig`, `DiffusionWeightQuantizerConfig`, `DiffusionActivationQuantizerConfig`, `DiffusionExtraWeightQuantizerConfig`, `DiffusionModuleQuantizerConfig`

The app-specific versions add skip lists, static activation options, GPTQ/LZS kernels, low-rank branch config, dynamic range calibration config, and cache/output naming.

### Quantizer Kernels

Location: `deepcompressor/quantizer/kernel/`

- `rtn.py`: round-to-nearest kernel and `rtn_quantize()`.
- `gptq.py`: GPTQ kernel/config and `gptq_quantize()`.
- `lzs.py`: LZS kernel/config and signed/unsigned LZS quantization helpers.

Kernel configs specialize `BaseQuantKernelConfig` and build `BaseQuantKernel` implementations. `QuantizerImpl` invokes the selected kernel after scale and zero-point setup.

## Calibration System

### `SearchBasedCalibrator`

Location: `deepcompressor/calib/search.py`

This is the base class for range search, smoothing, reordering, and low-rank calibration.

Important responsibilities:

- Parses calibration arguments for weights, inputs, outputs, modules, original references, and evaluation modules.
- Chooses the objective:
  - `TensorError`
  - `ProductsError`
  - `OutputsError`
- Chooses granularity:
  - layer
  - channel group
  - group
- Repartitions activation caches to control calibration batch/element sizes.
- Temporarily patches weights and registers hooks while evaluating candidates.
- Runs the ask/tell loop through `_ask()`, `_tell()`, and `get_best()`.

If calibration behavior changes unexpectedly, inspect this class before changing individual algorithms.

### `DynamicRangeCalibrator` and `calibrate_dynamic_range()`

Location: `deepcompressor/calib/range.py`

These functions search or compute dynamic ranges for weight/input/output quantizers.

Important behavior:

- Supports direct ratio-based ranges when search is disabled.
- Searches clamp ranges or scale ratios depending on `static` and `allow_scale`.
- Handles progressive quantization by calibrating each decomposed quantization step.
- Returns one `DynamicRange` per quantization step, or `None` when defaults are enough.

### Smoothing

Location: `deepcompressor/calib/smooth.py`

Important functions/classes:

- `ActivationSmoother`: hook processor that applies smoothing scales to activations.
- `SmoothCalibrator`: search-based calibrator for smooth scales.
- `SmoothLinearCalibrator`: specializes smoothing for linear projection pairs.
- `SmoothAttentionCalibrator`: specializes smoothing for attention tensors.
- `smooth_linear_modules()`: applies smoothing to connected modules.
- `smooth_attention()`: applies smoothing to attention projections.
- `smooth_upscale_param()`, `smooth_downscale_param()`, `convert_smooth_upscale_to_downscale()`: mutate or convert parameter-side smoothing factors.

Application entry points:

- `smooth_llm()` and `smooth_llm_layer()` in `deepcompressor/app/llm/quant/smooth.py`
- `smooth_diffusion()` and block-level helpers in `deepcompressor/app/diffusion/quant/smooth.py`

### Rotation

Location: `deepcompressor/calib/rotate.py`

Important functions/classes:

- `RMSNorm`: local RMSNorm implementation used during norm transformations.
- `HadamardTransformHook`: hook for Hadamard activation transforms.
- `get_rotation_matrix()`: builds random or Hadamard-compatible rotation matrices.
- `rotate_in_channels()` / `rotate_out_channels()`: apply rotations to parameters.
- `transform_norm_and_linear()`: transforms norm-linear chains.
- `transform_layer_norm_to_rms_norm()` and `transform_rms_norm_and_linear()`: norm-specific rewrites.

Application entry points:

- `rotate_llm()` in `deepcompressor/app/llm/quant/rotate.py`
- `rotate_diffusion()` in `deepcompressor/app/diffusion/quant/rotate.py`

### Reordering

Location: `deepcompressor/calib/reorder.py`

Important functions/classes:

- `ChannelReorderer`: tensor processor that applies channel index permutations.
- `get_channel_metric()`, `update_channel_metric()`: compute reorder metrics.
- `init_channel_index_from_metric()`: initialize channel order from a metric.
- `ChannelOrderCalibrator`: search-based channel order calibrator.

Application entry point:

- `reorder_llm()` and `reorder_llm_layer()` in `deepcompressor/app/llm/quant/reorder.py`

### Low-Rank Branch Calibration

Location: `deepcompressor/calib/lowrank.py`

Important class:

- `QuantLowRankCalibrator`: search-based calibrator that fits `LowRankBranch` compensation modules, used heavily by diffusion SVDQuant.

Related module:

- `LowRankBranch` in `deepcompressor/nn/patch/lowrank.py`: an `nn.Module` branch with `a` and `b` projections that can be registered as a hook.

### Reconstruction / AdaRound

Location: `deepcompressor/calib/recon/`

This package provides reconstruction-style, gradient-based calibration utilities used by the diffusion FastDM path.

Important files/classes/functions:

- `adaround.py`: `RMODE`, `AdaRoundQuantizer`, `compute_minmax_scale_zero()`, `reset_adam_momentum()`.
- `reconstruct.py`: `ReconState`, `attach_adaround()`, `detach_adaround()`, `iter_adaround_quantizers()`, `capture_module_io()`, `capture_module_grad()`, `reconstruct_module()`, `reconstruct_tib()`.
- `loss.py`: `ReconLoss`, `ReconLossKind`, `ReconLossReduction`, `ReconLossTimeEmbedding`, `LinearTempDecay`, `LossRecorder`.
- `schedule.py`: `dp_distance_segments_from_adj()`, `make_widths_from_adjacent_dist()`, `make_widths_fixedK()`, `plan_loop_sizes()`, `steps_per_epoch_like()`.

Config classes are exported from `deepcompressor/calib/config/recon.py` and `deepcompressor/calib/config/__init__.py`:

- `AdaRoundConfig`: bit width, symmetric mode, rounding-temperature schedule, and reconstruction penalty weights.
- `ProgressiveConfig`: progressive direction, epochs per loop, and batch size.
- `TimestepGroupConfig`: adaptive/uniform timestep grouping and optional adjacent-distance file.
- `TibReconConfig`: one-shot temporal-information-block reconstruction settings.
- `FastDmCalibConfig`: top-level FastDM config used by `DiffusionQuantConfig.fastdm`.

The current FastDM implementation attaches AdaRound through `torch.nn.utils.parametrize`, optimizes only the AdaRound `alpha` tensors, then removes parametrizations with quantized weights baked into the original modules.

`reconstruct_module()` expects cached positional inputs and outputs stacked along batch dimension 0. Its `captured_kwargs` path indexes tensor kwargs whose leading dimension matches the cached sample count with the same mini-batch indices as `cached_inputs`; tensors that do not look sample-batched are moved to the target device unchanged. This is important for DiT and class-conditional reconstruction because block kwargs can carry per-sample `timestep`, `class_labels`, attention masks, or other conditioning tensors.

## Model Structure Abstractions

The quantization code does not traverse raw model names ad hoc. It first converts models into structure objects that expose consistent keys and module groups.

### Shared Base Classes

Location: `deepcompressor/nn/struct/`

- `BaseModuleStruct`: base dataclass for wrappers around `nn.Module`; supports registered construction factories through `construct()`.
- `AttentionStruct`: exposes Q/K/V/add-QKV/out projection modules, keys, names, and attention type helpers.
- `FeedForwardStruct`: exposes FFN projection modules.
- `TransformerBlockStruct`: groups attention and FFN structures inside one block.
- `BaseTransformerStruct`: iterable transformer container abstraction.

High-impact API:

- `named_key_modules()`: yields `(module_key, module_name, module, parent_struct, field_name)` and is the core traversal API for quantization.
- `iter_attention_structs()` and `iter_transformer_block_structs()`: used by transforms and quantization passes.
- `get_default_keys()`: gives canonical skip/config keys.

### LLM Structures

Location: `deepcompressor/app/llm/nn/struct.py`

Important classes:

- `LlmModelStruct`: wraps a full causal language model.
- `LlmTransformerStruct`: wraps model backbone layers.
- `LlmTransformerBlockStruct`: wraps one transformer layer.
- `LlmSelfAttentionStruct`: wraps self-attention projections.
- `LlmFeedForwardStruct`: wraps FFN/up/down projections and expert structures.

High-impact methods:

- `LlmModelStruct.construct()`: creates the structure wrapper from a raw model.
- `LlmModelStruct.named_key_modules()`: drives LLM weight/activation quantization target selection.
- `LlmModelStruct.get_iter_layer_activations_args()`: tells calibration loaders how to iterate layer activations.

### Diffusion Structures

Location: `deepcompressor/app/diffusion/nn/struct.py`

Important classes:

- `DiffusionModelStruct`: abstract base for denoising-model wrappers.
- `UNetStruct`: wrapper for UNet-style diffusion models.
- `DiTStruct`: wrapper for diffusion transformers. It handles SD3, PixArt, Sana, FLUX, and patch-based Diffusers `Transformer2DModel` instances used by plain `DiTPipeline` / `facebook/DiT-*` checkpoints.
- `FluxStruct`: FLUX-specific diffusion transformer wrapper.
- `DiffusionTransformerBlockStruct`: wrapper for transformer blocks, including parallel/joint attention variants.
- `DiffusionAttentionStruct`: wrapper for self/cross/joint attention projections.
- `DiffusionFeedForwardStruct`: wrapper for diffusion FFN projections.
- `DiffusionResnetStruct`: wrapper for ResNet blocks.

High-impact methods:

- `DiffusionModelStruct.construct()`: selects the correct structure type for a pipeline/model.
- `get_named_layers()`: returns pre modules, blocks, and post modules for calibration/quantization.
- `get_prev_module_keys()` / `get_post_module_keys()`: used to decide pre/post skip behavior.
- `_get_default_key_map()` and `_simplify_keys()`: define compact key aliases used in cache and output names.

Plain Diffusers DiT support is implemented in `DiTStruct._default_construct()`:

- `DiTPipeline` is included in `DIT_PIPELINE_CLS`, so `DiffusionModelStruct.construct(pipeline)` unwraps `pipeline.transformer`.
- Patch-based `Transformer2DModel` maps `pos_embed` to `input_embed` and `transformer_blocks` to the block list.
- Diffusers DiT has no top-level text embedder and no standalone top-level time embedder in the same shape as SD3/PixArt/Sana, so `time_embed` and `text_embed` are `None` for this branch.
- Output modules map to `norm_out` plus either `proj_out` when present or a synthetic `ModuleDict` over `proj_out_1` and `proj_out_2`.

## Data and Tensor Metadata

### Quantization Data Types

Location: `deepcompressor/data/dtype.py`

`QuantDataType` defines integer, floating-point, exponent-only, and codebook-backed low-bit dtypes. It tracks bit width, sign, exponent/mantissa layout, min/max values, name registration, and conversion helpers.

`QDType` is a metaclass-backed registry-like facade for common quantized dtypes.

This layer affects every quantizer because dtype objects determine quant ranges, scale behavior, effective bits, and output/cache naming.

### Ranges

Location: `deepcompressor/data/range.py`

Important classes:

- `RangeBound`: optional min/max bound.
- `QuantRange`: quantized-value range, with dtype intersection helpers.
- `LogQuantRange`: log2-space range for exponent-only float formats.
- `ProtectiveQuantRange`: computes safe outer ranges for progressive integer quantization.
- `DynamicRange`: measured or ratio-based tensor dynamic range.

`DynamicRange.measure()` and `DynamicRange.construct()` are central to static quantization and calibration.

### Scales and Quantized Tensors

Locations:

- `deepcompressor/data/scale.py`
- `deepcompressor/quantizer/impl/scale.py`
- `deepcompressor/data/tensor.py`

Important classes:

- `QuantScale`: hierarchical scale tensor container.
- `QuantScaleInfo`: computes scale shapes, dtypes, and quantized scale values.
- `QuantTensor`: wrapper returned by quantization; exposes `data` for dequantized tensor and `qdata` for quantized tensor.

### Activation Caches

Location: `deepcompressor/data/cache.py`

Important classes:

- `ModuleForwardInput`: stores args/kwargs for replaying module forwards.
- `TensorCache`: stores tensor batches plus channel dimension, reshape function, original device, and sample counts.
- `TensorsCache`: ordered collection of `TensorCache` objects.
- `IOTensorsCache`: paired input/output caches for one module.

High-impact methods:

- `TensorCache.repartition()`: limits calibration batch/element size and standardizes/reshapes data for GEMM-style objectives.
- `TensorsCache.extract()`: rebuilds `ModuleForwardInput` for evaluation modules.

## Hook System

Locations:

- `deepcompressor/utils/hooks/hook.py`
- `deepcompressor/utils/hooks/processor.py`
- `deepcompressor/utils/hooks/packager.py`
- `deepcompressor/dataset/action.py`

Important classes:

- `Hook`: base wrapper around PyTorch pre/post forward hooks with activation/removal lifecycle.
- `IOHook`: hook base that unpacks and repacks selected inputs/outputs.
- `BaseTensorProcessor`: interface implemented by tensor processors such as `Quantizer`.
- `ProcessHook`: turns a `BaseTensorProcessor` into a module input/output hook.
- `BaseInputPackager`, `SimpleInputPackager`, `KeyedInputPackager`: select and repack module inputs.
- `BaseOutputPackager`, `SimpleOutputPackager`, `KeyedOutputPackager`: select and repack module outputs.
- `CacheAction`, `ConcatCacheAction`, `CacheHook`: collect activation tensors for calibration.

Hooks are how activation quantization is actually applied after calibration. They are also how calibration temporarily applies candidate quantizers while measuring error.

## App-Specific Configuration

### LLM Configs

Location: `deepcompressor/app/llm/config.py`

`LlmPtqRunConfig` is the top-level run config. It owns:

- cache paths: `LlmCacheConfig`
- output paths: `OutputConfig`
- model config: `LlmModelConfig`
- evaluation config: `LlmEvalConfig`
- quantization config: `LlmQuantConfig`
- seed, skip/load/save flags

`LlmQuantConfig` extends `LlmModuleQuantizerConfig` with calibration dataset config, rotation, reorder, smoothing, and development dtype. Its `__post_init__()` normalizes skip interactions between rotation/reordering and validates static activation group shapes.

Important properties:

- `enabled_smooth`
- `enabled_smooth_proj`
- `enabled_smooth_attn`
- `enabled_reorder`
- `enabled_rotation`
- `needs_acts_quantizer_cache`

### Diffusion Configs

Location: `deepcompressor/app/diffusion/config.py`

`DiffusionPtqRunConfig` is the top-level run config. It owns:

- cache paths: `DiffusionPtqCacheConfig`
- output paths: `OutputConfig`
- Diffusers pipeline config: `DiffusionPipelineConfig`
- evaluation config: `DiffusionEvalConfig`
- denoising model quant config: `DiffusionQuantConfig`
- optional text encoder quant config: `LlmQuantConfig`
- seed, skip/load/save/extract flags

`DiffusionQuantConfig` extends `DiffusionModuleQuantizerConfig` with calibration dataset config, rotation, smoothing, FastDM AdaRound reconstruction, and development dtype. It also derives `unsigned_ipts` for activation quantization when unsigned activations are allowed.

Important properties:

- `enabled_rotation`
- `enabled_smooth`
- `enabled_smooth_proj`
- `enabled_smooth_attn`
- `enabled_fastdm`
- `needs_acts_quantizer_cache`

## Diffusion Pipeline Builder

Location: `deepcompressor/app/diffusion/pipeline/config.py`

Important classes/functions:

- `DiffusionPipelineConfig`: builds Diffusers pipelines by name/path/dtype/device.
- `LoRAConfig`: describes LoRA path, weight name, and alpha.
- `DiffusionPipelineConfig.register_pipeline_factory()`: registers custom builders.
- `DiffusionPipelineConfig.register_text_extractor()`: registers custom text encoder extractors.
- `DiffusionPipelineConfig.extract_text_encoders()`: returns text encoder/tokenizer pairs for optional LLM quantization.
- `DiffusionPipelineConfig.load_lora()`: loads and attaches LoRA branches, including special handling for patched linears and low-rank hooks.

This file also invokes diffusion model patch helpers:

- `replace_fused_linear_with_concat_linear()`
- `replace_up_block_conv_with_concat_conv()`
- `shift_input_activations()`

## Evaluation

### LLM

Locations:

- `deepcompressor/app/llm/eval/config.py`
- `deepcompressor/app/llm/eval/base.py`
- `deepcompressor/app/llm/eval/custom.py`
- `deepcompressor/app/llm/eval/lm_eval.py`
- `deepcompressor/app/llm/eval/longbench/`

Important classes:

- `LlmEvalConfig`: selects evaluator tasks and generation/evaluation options.
- `LlmEvaluatorBase`: evaluator interface.
- `LlmCustomEvaluator`: custom perplexity/evaluation path.
- `LmevalEvaluator`: lm-evaluation-harness integration.
- `LongbenchEvaluator` and `LongbenchScorer`: LongBench integration and scoring.

### Diffusion

Locations:

- `deepcompressor/app/diffusion/eval/config.py`
- `deepcompressor/app/diffusion/eval/metrics/`
- `deepcompressor/app/diffusion/eval_metrics.py`

Important functions:

- `compute_image_metrics()`: dispatches image metrics.
- `compute_fid()`: FID calculation.
- `compute_image_multimodal_metrics()`: multimodal text-image metrics.
- `compute_image_reward()`: ImageReward scoring.
- `compute_image_similarity_metrics()`: similarity metrics.

## Dataset and Calibration Data

Generic cache framework:

- `deepcompressor/dataset/action.py`
- `deepcompressor/dataset/cache.py`
- `deepcompressor/dataset/config.py`

LLM calibration:

- `deepcompressor/app/llm/quant/dataset.py`
- `LlmCalibDataLoaderConfig`
- `LlmCalibDataset`
- `LlmCalibCacheLoader`

Diffusion calibration:

- `deepcompressor/app/diffusion/dataset/calib.py`
- `DiffusionCalibCacheLoaderConfig`
- `DiffusionCalibDataset`
- `DiffusionConcatCacheAction`
- `DiffusionCalibCacheLoader`

Diffusion dataset collection:

- `deepcompressor/app/diffusion/dataset/collect/calib.py`
- `deepcompressor/app/diffusion/dataset/collect/utils.py`

`CollectHook` in `collect/utils.py` normalizes denoiser forward inputs into cache entries. It supports:

- `UNet2DConditionModel`, including timestep tensor normalization and expansion to batch size.
- PixArt, Sana, and FLUX transformer models by caching `hidden_states` as the positional replay input.
- Patch-based Diffusers `Transformer2DModel` for plain DiT class-to-image collection, preserving kwargs such as `timestep`, `class_labels`, and `return_dict`.

Bundled dataset builders:

- `deepcompressor/app/diffusion/dataset/data/COCO/`
- `deepcompressor/app/diffusion/dataset/data/DCI/`
- `deepcompressor/app/diffusion/dataset/data/MJHQ/`

## Backend Conversion

Backends convert saved quantized checkpoints into runtime-specific layouts.

### TinyChat

Locations:

- `deepcompressor/backend/tinychat/convert.py`
- `deepcompressor/backend/tinychat/utils.py`
- `deepcompressor/backend/tinychat/linear.py`

Important APIs:

- `convert_to_tinychat_w4x16y16_linear_state_dict()`
- `convert_to_tinychat_state_dict()`
- `convert_to_tinychat_w4x16y16_linear_weight()`
- `W4Linear`

### QServe

Locations:

- `deepcompressor/backend/qserve/convert.py`
- `deepcompressor/backend/qserve/utils.py`

Important APIs:

- `convert_to_qserve_w4x8y16_linear_state_dict()`
- `convert_to_qserve_w8x8y16_linear_state_dict()`
- `convert_to_qserve_state_dict()`
- `QServePacker`
- `convert_to_qserve_w4x8y16_linear_weight()`
- `convert_to_qserve_w8x8y16_linear_weight()`

### Nunchaku

Locations:

- `deepcompressor/backend/nunchaku/convert.py`
- `deepcompressor/backend/nunchaku/convert_lora.py`
- `deepcompressor/backend/nunchaku/utils.py`

Important APIs:

- `convert_to_nunchaku_w4x4y16_linear_state_dict()`
- `convert_to_nunchaku_transformer_block_state_dict()`
- `convert_to_nunchaku_flux_state_dicts()`
- `convert_to_nunchaku_flux_lowrank_dict()`
- `NunchakuWeightPacker`
- `convert_to_nunchaku_w4x4y16_linear_weight()`
- `convert_to_nunchaku_w4x16_linear_weight()`

## Native/CUDA Components

Locations:

- `deepcompressor/csrc/load.py`
- `deepcompressor/csrc/load.pyi`
- `deepcompressor/csrc/pybind.cpp`
- `deepcompressor/csrc/quantize/quantize.cu`
- `deepcompressor/csrc/quantize/quantize.h`
- `deepcompressor/backend/tinychat/csrc/`

These files provide native quantization and TinyChat packing/runtime helpers. The Python code generally loads these lazily through `deepcompressor.csrc.load`.

## Key Naming and Skip Semantics

Quantization configs use module keys rather than raw module names. Structure wrappers map raw model modules to stable keys such as attention Q/K/V, output projection, up projection, down projection, residual/pre/post modules, and diffusion-specific aliases.

Important APIs:

- `named_key_modules()` on model/block/attention/FFN structs.
- `SkipBasedConfig` and `IncludeBasedConfig` in `deepcompressor/utils/config/base.py`.
- `DiffusionQuantCacheConfig.simplify_path()` and `DiffusionModelStruct._get_default_key_map()`.
- `generate_dirnames()`, `generate_calib_dirname()`, and `generate_default_dirname()` on quant configs.

When adding a new model family or module type, first make sure the relevant struct emits correct keys. The quantization passes depend on these keys for skip lists, cache names, low-rank branch grouping, and activation hook placement.

## Cache and Output Conventions

LLM quantization cache files:

- `rotation.pt`
- `reorder.pt`
- `smooth.pt`
- `wgts.pt`
- `acts.pt`
- `model.pt`
- `scale.pt` when saving scale metadata

Diffusion quantization cache files:

- `smooth.pt`
- `branch.pt`
- `wgts.pt`
- `acts.pt`
- `fastdm.pt`
- `model.pt`
- `scale.pt` when saving scale metadata

Cache directories are generated from:

- quantized dtype/group/scale settings
- calibration settings
- smoothing/reorder/rotation options
- low-rank settings
- FastDM AdaRound/progressive/timestep-group/TIB settings
- model or pipeline name
- seed for LLM cache generation
- diffusion calibration dataset identity

The PTQ drivers can symlink existing cache files into an output directory when `copy_on_save=False`.

## Common Extension Points

Add a new LLM model family:

1. Add or adjust construction logic in `LlmModelStruct` and related structs.
2. Ensure `named_key_modules()` emits canonical keys.
3. Add model config handling in `LlmModelConfig` if needed.
4. Add example YAML under `examples/llm/configs/`.

Add a new diffusion model family:

1. Add a `DiffusionModelStruct` subclass or construction branch.
2. Implement block iteration, pre/post module keys, and named key modules.
3. Register a pipeline factory or text extractor in `DiffusionPipelineConfig` if the default builder is insufficient.
4. Add model YAML under `examples/diffusion/configs/model/`.

For Diffusers class-to-image DiT models, also verify that:

- `DiffusionPipelineConfig.build()` returns a pipeline class included in `DIT_PIPELINE_CLS`.
- `CollectHook` accepts the denoiser class used during calibration collection.
- FastDM sample replay moves all cached args/kwargs to the denoiser device before forward capture.
- Any block-level kwargs that vary by sample are captured and sliced during reconstruction rather than reused from the first sample.

Add a new quantization kernel:

1. Implement a `BaseQuantKernel` and `BaseQuantKernelConfig` under `deepcompressor/quantizer/kernel/`.
2. Add config fields to the relevant app quantizer config.
3. Update app quantizer adapter `__post_init__()` to select the kernel.
4. Ensure state dict/cache naming includes kernel settings if they affect results.

Add a new calibration method:

1. Subclass `SearchBasedCalibrator` if it searches candidate parameters.
2. Define config under `deepcompressor/calib/config/`.
3. Wire the method into app-level quantization flow.
4. Add cache serialization if results are expensive to recompute.

Add or tune FastDM reconstruction:

1. Update `FastDmCalibConfig` in `deepcompressor/calib/config/recon.py` for new knobs.
2. Keep cache path naming in `DiffusionQuantConfig.generate_cache_dirpath()` and `generate_default_dirname()` in sync.
3. Update `fastdm_diffusion()` for driver behavior and `deepcompressor/calib/recon/` for core AdaRound/reconstruction behavior.
4. Add or adjust presets under `examples/diffusion/configs/fastdm/`.

## Known Caveats in Current Checkout

- The worktree is dirty. The FastDM/AdaRound behavior documented above comes from current uncommitted working-tree changes, including:
  - modified `deepcompressor/app/diffusion/cache/config.py`
  - modified `deepcompressor/app/diffusion/dataset/collect/calib.py`
  - modified `deepcompressor/app/diffusion/dataset/collect/utils.py`
  - modified `deepcompressor/app/diffusion/eval/config.py`
  - modified `deepcompressor/app/diffusion/nn/struct.py`
  - modified `deepcompressor/app/diffusion/pipeline/config.py`
  - modified `deepcompressor/app/diffusion/ptq.py`
  - modified `deepcompressor/app/diffusion/quant/__init__.py`
  - modified `deepcompressor/app/diffusion/quant/config.py`
  - modified `deepcompressor/calib/__init__.py`
  - modified `deepcompressor/calib/config/__init__.py`
  - modified `examples/diffusion/README.md`
  - modified `examples/diffusion/configs/__default__.yaml`
  - modified `pyproject.toml`
  - untracked `deepcompressor/app/diffusion/quant/fastdm.py`
  - untracked `deepcompressor/app/diffusion/quant/fastdm_data.py`
  - untracked `deepcompressor/app/diffusion/quant/fastdm_tib.py`
  - untracked `deepcompressor/calib/config/recon.py`
  - untracked `deepcompressor/calib/recon/`
  - untracked `examples/diffusion/configs/collect/imagenet.yaml`
  - untracked `examples/diffusion/configs/fastdm/`
  - untracked `examples/diffusion/configs/model/dit-xl-2-256.yaml`
  - untracked `examples/diffusion/prompts/imagenet.yaml`
  - untracked `examples/diffusion/scripts/compare_dit_fid.py`
  - untracked `review.md`
- `deepcompressor/app/diffusion/ptq.py` imports `ActivationExtractor` and `ActivationModifier` from `.investigate`, but no `deepcompressor/app/diffusion/investigate.py` file is present in the current repository file listing. Importing `deepcompressor.app.diffusion.ptq` may fail until that module is restored or the import is guarded/removed.
- Diffusion output activation quantization is explicitly not supported in `DiffusionModuleQuantizerConfig.__post_init__()`.

## Useful Commands

Install from source:

```bash
conda env create -f environment.yml
poetry install
```

Run LLM PTQ help:

```bash
python -m deepcompressor.app.llm.ptq -h
```

Run diffusion PTQ help:

```bash
python -m deepcompressor.app.diffusion.ptq -h
```

Collect diffusion calibration data:

```bash
python -m deepcompressor.app.diffusion.dataset.collect.calib \
  examples/diffusion/configs/model/flux.1-schnell.yaml \
  examples/diffusion/configs/collect/qdiff.yaml
```

Run FastDM-only diffusion INT4 AdaRound:

```bash
python -m deepcompressor.app.diffusion.ptq \
  examples/diffusion/configs/model/flux.1-schnell.yaml \
  examples/diffusion/configs/fastdm/int4.yaml \
  --eval-benchmarks MJHQ \
  --eval-num-samples 1024
```

Run SVDQuant plus FastDM:

```bash
python -m deepcompressor.app.diffusion.ptq \
  examples/diffusion/configs/model/flux.1-schnell.yaml \
  examples/diffusion/configs/svdquant/int4.yaml \
  examples/diffusion/configs/fastdm/int4-svdq.yaml \
  --eval-benchmarks MJHQ \
  --eval-num-samples 1024
```

Convert saved LLM checkpoint to TinyChat:

```bash
python -m deepcompressor.backend.tinychat.convert \
  --model-name MODEL_NAME \
  --quant-path /path/to/quantized-model \
  --output-root /path/to/output-root
```

Convert saved LLM checkpoint to QServe:

```bash
python -m deepcompressor.backend.qserve.convert \
  --model-path /path/to/hf-model \
  --quant-path /path/to/quantized-model \
  --weight-bits 4 \
  --output-root /path/to/output-root
```

Convert saved diffusion checkpoint to Nunchaku:

```bash
python -m deepcompressor.backend.nunchaku.convert \
  --quant-path /path/to/checkpoint-dir \
  --output-root /path/to/output-root \
  --model-name MODEL_NAME
```

## Mental Model for Future Tasks

Most future changes fit into one of five layers:

1. **Config layer**: `omniconfig` dataclasses decide what is enabled and name caches/outputs.
2. **Structure layer**: `LlmModelStruct` and `DiffusionModelStruct` expose stable module keys and block traversal.
3. **Calibration layer**: activation caches plus `SearchBasedCalibrator` choose ranges, scales, orderings, or low-rank branches.
4. **Reconstruction layer**: FastDM/AdaRound uses cached diffusion block IO and `deepcompressor/calib/recon/` to learn rounding alpha tensors.
5. **Quantizer layer**: `Quantizer` and `QuantizerImpl` turn tensors into dequantized fake-quant tensors and optional serialized scales.
6. **Application driver layer**: `ptq.py` files coordinate stages, cache loading/saving, and evaluation.

When debugging accuracy, start with calibration data, FastDM alpha state if enabled, and quantizer state. When debugging missing modules or skipped modules, start with structure wrappers and key maps. When debugging saved checkpoints or deployment conversion, start with scale/zero serialization and backend packers.

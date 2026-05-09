# FastDM Calibration Pipeline

This document describes the actual data flow of FastDM block-wise
calibration as implemented in `deepcompressor.app.diffusion`. Every
step below names the concrete function or class that runs it, and the
shapes / objects that move between them. Two recipes share the same
pipeline:

- `recipe: adaround` -- trains AdaRound rounding offsets on the model
  weights (the original `Fast_DM_PTQ` algorithm).
- `recipe: lowrank` -- trains a LoRA-style `LowRankBranch` per
  `nn.Linear`/`nn.Conv*` to compensate for fixed-RTN quantization
  error (a SVDQuant-style residual absorber).

## 1. Where it is invoked

`fastdm_diffusion(model, config, cache=None)` is called from
`deepcompressor/app/diffusion/ptq.py` between `smooth_diffusion` and
`quantize_diffusion_weights`:

```python
# deepcompressor/app/diffusion/ptq.py
if quant_wgts and config.enabled_fastdm:
    fastdm_cache = fastdm_diffusion(model, config)         # train
    # ... or, if cached:
    fastdm_diffusion(model, config, cache=fastdm_cache)    # reload
```

Inputs:

- `model`: a `DiffusionModelStruct` or a raw `nn.Module` accepted by
  `DiffusionModelStruct.construct(...)`. The driver re-wraps with
  `DiffusionModelStruct.construct(model)` if needed and reads
  `model_struct.module` (the underlying denoiser) plus
  `model_struct.block_structs` (per-block wrappers).
- `config`: the full `DiffusionQuantConfig`. The fields actually read
  are `config.fastdm` (a `FastDmCalibConfig`) and `config.calib` (the
  `DiffusionCalibCacheLoaderConfig`, used to point at the qdiff cache).
- `cache`: an optional pre-computed state dict from a prior run. If
  present, the function bypasses optimisation and only re-applies the
  saved state to `model`.

Output: a recipe-specific dict suitable for `torch.save`.

| Recipe     | Returned dict                                                                                               |
| ---------- | ----------------------------------------------------------------------------------------------------------- |
| `adaround` | `{f"{block}.{sub}.alpha": tensor, "__recipe__": "adaround"}`                                                |
| `lowrank`  | `{f"{block}.{sub}.<a|b>.weight": tensor, "__recipe__": "lowrank", "__branch_state_dict__": {...}}`         |

The driver in `ptq.py` pops `"__branch_state_dict__"` and writes it to
`cache.path.branch` so the downstream `quantize_diffusion_weights`
consumes it in place of running its own analytic SVD pass.

## 2. Locating the target blocks

`_collect_target_blocks(model_struct)` walks
`model_struct.block_structs`, keeps every `DiffusionBlockStruct` whose
underlying `module` contains at least one `nn.Linear`/`nn.Conv*`, and
returns a list of `_BlockTarget(name=block_struct.name,
module=block_struct.module)` entries. These are the units AdaRound /
the LoRA branch are attached to and trained over.

Models that already share the abstraction:

- `UNetStruct` -- UNet down/mid/up blocks.
- `DiTStruct` / `FluxStruct` -- transformer blocks (handles SD3,
  PixArt, Sana, Flux, plain diffusers `Transformer2DModel`).

## 3. Loading the qdiff calibration cache

```python
loader = config.calib.build_loader()
samples: list[FastDmSample] = list(iter_fastdm_samples(loader))
timesteps = extract_timesteps(samples)        # 1D float64, len == n_samples
```

`iter_fastdm_samples` (see `quant/fastdm_data.py`) iterates the
`DiffusionCalibDataset.data` underlying the loader and emits one
`FastDmSample(args, kwargs, timestep)` per cache entry -- the same
`(args, kwargs)` the calibration collector recorded when running the
denoiser forward. `timestep` is best-effort extracted from the
`timestep` / `timesteps` / `t` kwarg, with a fallback to the recorded
`step` index.

`extract_timesteps` returns a `torch.Tensor` of dtype `float64` and
shape `(n_samples,)`, used both for progressive scheduling and the
TIB pass.

## 4. Optional one-shot TIB pass (AdaRound recipe only)

When `recipe == "adaround"` and `fastdm.tib.enable` is set:

1. `build_tib_bundle(raw_model)` (see `quant/fastdm_tib.py`) returns a
   `FastDmTibBundle` whose `forward(t)` calls the top-level time
   embedder and every `ResnetBlock2D.time_emb_proj`. Returns `None` on
   non-UNet backbones (DiT/Flux/SD3) and the driver logs a warning.
2. `reconstruct_tib(tib, (timesteps,), forward_fn=...,
   bits=fastdm.adaround.bits, ...)` runs an Adam loop over the
   AdaRound `alpha` of every member.
3. After TIB converges, `detach_adaround(sub)` is called on each
   parametrized member so subsequent block-level recon sees the
   final dequantized time-embedding weights.

## 5. Progressive timestep scheduling

`_resolve_progressive_indices(fastdm, timesteps)` returns
`(indices_per_loop, labels)`:

- If `fastdm.progressive.is_progressive` is false, a single `[arange]`
  loop covers the full set.
- Otherwise it sorts unique timesteps, optionally bins them into
  `K = fastdm.timestep_group.num_groups` groups via
  `make_widths_from_adjacent_dist(adj, K, mode="adaptive")` (when
  `mode="adaptive"` and `feature_dist_path` exists) or
  `make_widths_fixedK(U, K, mode="gradual")` (uniform fallback).
- `plan_loop_sizes(uniq, counts, n_total, direction, widths)` plans
  cumulative loop sizes; each entry is converted back into indices
  into the original sample order. `direction = "reverse"` accumulates
  from the noisy end; `"forward"` from the clean end.

The driver logs each loop's label, e.g. `"k=3/5|t∈[12,24]|N=2400"`.

## 6. Capturing FP block IO once

```python
block_caches = _capture_fp_io(raw_model, targets, samples)
# {block_name: _BlockCache(inputs=[(t0, t1, ...), ...],
#                          input_kwargs=[{...}, ...],
#                          outputs=[tensor, ...])}
```

`_CaptureHook` is registered on every target block with
`with_kwargs=True`. For each `FastDmSample` the driver moves
`sample.args` / `sample.kwargs` onto the denoiser device with
`_move_tree(...)` and runs `model(*args, **kwargs)` under
`torch.no_grad()`. The hook detaches each `(inputs, kwargs, output)`
to CPU and pushes them into the per-block cache. Non-tensor outputs
raise; tuple outputs are flattened to their first element.

`_stack_block_io(cache, keep_gpu, device)` then concatenates one
block's per-sample tensors along dim 0, returning
`(cat_inputs, cat_outputs, captured_kwargs)`. If `keep_gpu=True` the
stacked tensors are moved back onto the device. `captured_kwargs` is
built by `_merge_per_sample_kwargs`, which stacks/cats tensor values
that look sample-batched and keeps everything else as a single shared
value.

## 7. Per-loop block reconstruction

```python
state_registry = ReconState()
for loop_idx, sel in enumerate(indices_per_loop):
    is_first = loop_idx == 0
    is_last  = loop_idx == num_loops - 1
    for t in targets:
        cat_in, cat_out, cap_kw = _stack_block_io(block_caches[t.name], keep_gpu, device)
        if fastdm.recipe == "lowrank":
            reconstruct_module_lowrank(t.module, cached_inputs=cat_in,
                cached_outputs=cat_out, captured_kwargs=cap_kw,
                state_registry=state_registry, rank=fastdm.lowrank.rank,
                bits=fastdm.lowrank.bits, ..., select_indices=sel)
        else:
            reconstruct_module(t.module, cached_inputs=cat_in,
                cached_outputs=cat_out, captured_kwargs=cap_kw,
                state_registry=state_registry, bits=fastdm.adaround.bits,
                ..., is_first_loop=is_first, is_last_loop=is_last,
                select_indices=sel)
```

`ReconState` (in `calib/recon/reconstruct.py`) is the cross-loop
registry that holds each block's optimiser, optimised parameter list,
and `ReconLoss`. `is_first_loop` triggers attachment + optimiser
construction; later loops reuse the same optimiser and reset the Adam
momentum via `reset_adam_momentum(...)`. `is_last_loop` switches
AdaRound to its hard regime (`freeze_soft_targets`) for the final pass.

`select_indices` indexes into the dim-0 stacked IO so each loop sees
only its progressive subset. `_select_kwargs_tree(captured_kwargs,
idx, n_total, device)` slices kwargs whose leading dimension matches
`n_total` and moves the rest onto the device unchanged.

### 7a. AdaRound recipe (`reconstruct_module`)

Per first loop, `attach_adaround(sub, bits, symmetric, always_zero)`
registers an `AdaRoundQuantizer` parametrize on every parametrizable
child of the block; only the `alpha` tensors are optimised. The loss
combines an Lp reconstruction term (`MSE` by default) with the
relaxation rounding penalty defined in `ReconLoss`, with the rounding
temperature decayed from `b_range_start` to `b_range_end` after the
`warmup` / `decay_start` fractions.

### 7b. LoRA recipe (`reconstruct_module_lowrank`)

Per first loop, `attach_lowrank(sub, rank, bits, symmetric, ste_rtn)`:

1. Computes the analytic `RTN(W)` residual entirely in fp32 via
   `compute_minmax_scale_zero(weight32, bits, symmetric)`.
2. Builds `_Fp32TrainingLowRankBranch(in_features, out_features,
   rank, weight=residual)` -- a `LowRankBranch` whose forward bridges
   fp16/fp32 dtypes so Adam's `eps` does not underflow on fp16
   models.
3. Registers `RtnLowRankParametrize(weight=W, bits, branch,
   symmetric, ste_rtn)` on `module.weight`; its `forward(W)` returns
   `RTN(W - branch.eff()) ` with a straight-through round.
4. Registers `branch.as_hook()` (an `AccumBranchHook`) so the block
   forward becomes `F.linear(x, RTN(W - b@a)) + b(a(x))`.
5. Freezes `module.weight` and exposes only `branch.a.weight` /
   `branch.b.weight` to Adam (`eps=1e-7`).

The training loss is plain block-output MSE in fp32:

```python
err = (target(*cur_inputs, **kw) - cur_outputs).pow(2).mean()
```

Optimisation walks `select_indices` in random mini-batches of
`progressive.batch_size`; per-step gradients are clipped with
`clip_grad_norm_(opt_params, max_norm=1.0)` and (if running under
DDP) all-reduced via `_maybe_allreduce`.

## 8. Baking the result back into the model

After all loops finish:

### 8a. AdaRound

`_bake_alpha_state(targets)` walks every target block's
`iter_adaround_quantizers(sub)` and snapshots `adar.alpha` to a CPU
fp32 dict keyed `f"{block}.{sub_name}.alpha"`. Then `detach_adaround`
removes the parametrize with `leave_parametrized=True`, so
`module.weight` now stores the dequantized AdaRound-rounded tensor
that the downstream static weight pass sees.

The returned dict carries a `"__recipe__": "adaround"` marker.

### 8b. LoRA

`build_lowrank_branch_state(t.module, target_name=t.name)` walks every
`RtnLowRankParametrize` under each block, looks up the parent module,
and emits `{full_module_name: branch.state_dict()}` -- the same format
consumed by `calibrate_diffusion_block_low_rank_branch`'s reload path.

The dict is stashed under `state["__branch_state_dict__"]` plus a
flattened mirror under `state[f"{full}.<a|b>.weight"]` for inspection.
`detach_lowrank(sub)` restores the original weight buffer (so the
downstream static weight pass starts from a clean state) and removes
the `AccumBranchHook` from both `_forward_pre_hooks` and
`_forward_hooks`.

The driver in `ptq.py` then `cache.path.branch <- branch_state_dict`
so `quantize_diffusion_weights` re-installs the trained branches via
`load_diffusion_weights_state_dict`'s low-rank path.

## 9. Cache files and reload behaviour

| File                                 | Path field          | Recipe(s)         | Contents                                                                  |
| ------------------------------------ | ------------------- | ----------------- | ------------------------------------------------------------------------- |
| `fastdm.pt`                          | `cache.path.fastdm` | adaround, lowrank | The dict returned by `fastdm_diffusion`.                                  |
| `branch.pt`                          | `cache.path.branch` | lowrank only      | `{module_name: branch.state_dict()}` ready for the SVDQuant load path.    |

On reload (`fastdm_diffusion(..., cache=loaded_dict)`):

- `recipe == "adaround"`: `_apply_alpha_state(targets, loaded_dict)`
  re-attaches AdaRound, copies `alpha` from the cache, and bakes the
  rounded weights back via `detach_adaround`. No optimisation runs.
- `recipe == "lowrank"`: the function returns the cached dict
  unchanged. The actual branches are reloaded by the downstream
  weight quantizer from `branch.pt`.

The cache directory name is generated by
`FastDmCalibConfig.generate_dirnames(...)` and includes:

- the recipe and either AdaRound bits + loss weights or LoRA rank +
  bits + lr;
- the progressive direction and `epoch_per_loop`;
- the timestep-group mode and `K`;
- the TIB iters / lr (AdaRound recipe only).

## 10. Quick reference -- function map

| Step                          | Function                                                         | Module                                     |
| ----------------------------- | ---------------------------------------------------------------- | ------------------------------------------ |
| Driver entry                  | `fastdm_diffusion`                                               | `app/diffusion/quant/fastdm.py`            |
| Block discovery               | `_collect_target_blocks`                                         | same                                       |
| Sample iterator               | `iter_fastdm_samples`, `extract_timesteps`                       | `app/diffusion/quant/fastdm_data.py`       |
| TIB bundle builder            | `build_tib_bundle`, `FastDmTibBundle`                            | `app/diffusion/quant/fastdm_tib.py`        |
| Progressive scheduler         | `_resolve_progressive_indices` -> `make_widths_*`, `plan_loop_sizes` | `calib/recon/schedule.py`              |
| FP IO capture                 | `_capture_fp_io`, `_CaptureHook`, `_stack_block_io`              | `app/diffusion/quant/fastdm.py`            |
| AdaRound attach / detach      | `attach_adaround`, `detach_adaround`, `iter_adaround_quantizers` | `calib/recon/reconstruct.py`               |
| AdaRound block recon          | `reconstruct_module`, `reconstruct_tib`                          | `calib/recon/reconstruct.py`               |
| LoRA attach / detach          | `attach_lowrank`, `detach_lowrank`, `iter_lowrank_parametrizes`  | `calib/recon/lowrank_recon.py`             |
| LoRA block recon              | `reconstruct_module_lowrank`                                     | `calib/recon/lowrank_recon.py`             |
| LoRA branch parametrize       | `RtnLowRankParametrize`                                          | `calib/recon/lowrank_recon.py`             |
| AdaRound bake                 | `_bake_alpha_state`                                              | `app/diffusion/quant/fastdm.py`            |
| LoRA branch harvest           | `build_lowrank_branch_state`                                     | `calib/recon/lowrank_recon.py`             |
| Reload (AdaRound)             | `_apply_alpha_state`                                             | `app/diffusion/quant/fastdm.py`            |
| Cache routing                 | branch.pt write in `ptq.py`                                      | `app/diffusion/ptq.py`                     |

#!/usr/bin/env python3
"""Collect a PTQ4DiT-style calibration dataset for DiT-XL/2.

Mirrors `PTQ4DiT/get_calibration_set.py`:
  - Iterates over all 1000 ImageNet classes.
  - Generates ``--n-per-class`` samples per class (default 2, matching the
    paper's 2x recipe). Each sample is run through the full denoising
    trajectory at ``--num-steps`` steps with classifier-free guidance
    (CFG=1.5 by default), so each call captures two trajectories per
    sample (one conditional + one null).
  - Hooks the transformer's ``forward`` to capture the noisy latent
    ``xs``, the timestep ``ts``, and the class label ``y`` at every
    denoising step, in the *CFG-doubled* batch order produced by
    ``diffusers.DiTPipeline``.
  - Saves a single ``.pt`` containing tensors of shape
    ``[num_steps, num_total_trajectories, ...]`` matching the format
    consumed by ``PTQ4DiTCalibLoader``.

This is the "data scale" half of bringing qdiff in line with PTQ4DiT.
The runtime calibration / strided subset selection / 3-pass scale init
lives in the loader and the PTQ pipeline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from diffusers import DiTPipeline


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt",
        default="facebook/DiT-XL-2-256",
        help="DiT pipeline path or HF hub id (default: facebook/DiT-XL-2-256)",
    )
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument(
        "--n-per-class",
        type=int,
        default=2,
        help="Samples per ImageNet class (PTQ4DiT default: 2)",
    )
    p.add_argument(
        "--batch-classes",
        type=int,
        default=8,
        help="Classes per pipeline call (each call processes batch_classes*n_per_class samples, "
             "doubled by CFG inside the pipeline)",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .pt path (e.g. calib/imagenet_DiT-256_sample4000_50steps_allst.pt).",
    )
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]

    pipe = DiTPipeline.from_pretrained(args.ckpt, torch_dtype=dtype).to(device)
    pipe.set_progress_bar_config(disable=True)
    transformer = pipe.transformer

    captured_x: list[torch.Tensor] = []
    captured_t: list[torch.Tensor] = []
    captured_y: list[torch.Tensor] = []

    def _hook(module, inputs, kwargs, output):
        # hidden_states is the first positional arg; timestep/class_labels are kwargs
        x = inputs[0] if inputs else kwargs["hidden_states"]
        t = kwargs["timestep"]
        y = kwargs["class_labels"]
        captured_x.append(x.detach().to("cpu"))
        captured_t.append(t.detach().to("cpu"))
        captured_y.append(y.detach().to("cpu"))

    handle = transformer.register_forward_hook(_hook, with_kwargs=True)

    classes_per_call = args.batch_classes
    n_per_class = args.n_per_class
    total_classes = args.num_classes
    print(f"[collect-ptq4dit] generating {total_classes} classes × {n_per_class} samples × CFG-2 "
          f"@ {args.num_steps} steps, {classes_per_call} classes/call",
          flush=True)

    try:
        for cls_start in range(0, total_classes, classes_per_call):
            cls_end = min(cls_start + classes_per_call, total_classes)
            cls_block = list(range(cls_start, cls_end))
            # PTQ4DiT runs n_per_class trajectories per class -> repeat each label.
            class_labels = [c for c in cls_block for _ in range(n_per_class)]

            generators = [
                torch.Generator(device=device).manual_seed(
                    args.seed + cls_start * n_per_class + i
                )
                for i in range(len(class_labels))
            ]
            _ = pipe(
                class_labels=class_labels,
                guidance_scale=args.cfg_scale,
                num_inference_steps=args.num_steps,
                generator=generators,
            )
            if cls_end % (classes_per_call * 8) == 0 or cls_end == total_classes:
                print(f"[collect-ptq4dit] classes processed: {cls_end}/{total_classes}", flush=True)
    finally:
        handle.remove()

    # captured_* contains one tensor per *transformer call*, i.e. per
    # (pipeline_call, denoising_step). Each tensor has the CFG-doubled
    # batch dim. We need to fold this into [num_steps, total_traj, ...]:
    #   total_pipe_calls = num_classes / classes_per_call
    #   each call -> num_steps captures, in order
    num_steps = args.num_steps
    n_calls = len(captured_x) // num_steps
    assert len(captured_x) == n_calls * num_steps, (
        f"unexpected capture count: got {len(captured_x)} captures for "
        f"{n_calls} pipeline calls × {num_steps} steps"
    )
    print(f"[collect-ptq4dit] captured {len(captured_x)} forward calls "
          f"({n_calls} pipeline calls × {num_steps} steps); regrouping...",
          flush=True)

    # group by step
    xs_by_step: list[list[torch.Tensor]] = [[] for _ in range(num_steps)]
    ts_by_step: list[list[torch.Tensor]] = [[] for _ in range(num_steps)]
    ys_by_step: list[list[torch.Tensor]] = [[] for _ in range(num_steps)]
    for call_idx in range(n_calls):
        for step_idx in range(num_steps):
            i = call_idx * num_steps + step_idx
            xs_by_step[step_idx].append(captured_x[i])
            ts_by_step[step_idx].append(captured_t[i])
            ys_by_step[step_idx].append(captured_y[i])

    xs = torch.stack([torch.cat(xs_by_step[s], dim=0) for s in range(num_steps)], dim=0)
    ts = torch.stack([torch.cat(ts_by_step[s], dim=0) for s in range(num_steps)], dim=0)
    y = torch.stack([torch.cat(ys_by_step[s], dim=0) for s in range(num_steps)], dim=0)
    print(f"[collect-ptq4dit] xs={tuple(xs.shape)}, ts={tuple(ts.shape)}, y={tuple(y.shape)}", flush=True)

    blob = {"xs": xs.contiguous(), "ts": ts.contiguous(), "y": y.contiguous()}
    torch.save(blob, args.out)
    sz_mb = args.out.stat().st_size / 1e6
    print(f"[collect-ptq4dit] saved {args.out}  ({sz_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

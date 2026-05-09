#!/usr/bin/env python3
"""Compare FID of DiT-XL/2 (256x256, 50 steps) outputs under two calibration
methods, both at W4A32:

  qdiff   -- the baseline static path: 4-bit weight quantization with the
             existing min-max + grid-search range calibrator. No smoothing,
             no FastDM.
  fastdm  -- 4-bit weight quantization with FastDM AdaRound block-wise
             reconstruction (configs/fastdm/int4-dit.yaml).

Both methods leave activations in FP32 (W4A32). The script:
  1. Programmatically runs PTQ for each method *in-process* so the
     quantization hooks installed by qtbench's pipeline stay live for
     sample generation.
  2. Generates ``--num-samples`` class-balanced ImageNet samples
     (round-robin over classes 0-999) using diffusers' ``DiTPipeline``
     directly -- bypasses the eval module which is text-to-image only.
  3. Computes FID against an ImageNet-256 reference via ``pytorch-fid``
     and prints a side-by-side comparison.

Reference statistics: this script expects the *standard* ImageNet-256
reference -- typically ``VIRTUAL_imagenet256_labeled.npz`` (50K samples)
or a directory of 50K JPEGs. Pass via ``--imagenet-ref``.

Requires: ``pip install pytorch-fid``.

Example:

    cd examples/diffusion
    python scripts/compare_dit_fid.py \\
        --work-root /scratch/dit-fid \\
        --imagenet-ref /data/imagenet/VIRTUAL_imagenet256_labeled.npz \\
        --num-samples 10000 --batch-size 8
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import torch

from deepcompressor.app.diffusion.config import DiffusionPtqRunConfig
from deepcompressor.app.diffusion.nn.struct import DiffusionModelStruct
from deepcompressor.app.diffusion.ptq import ptq
from deepcompressor.utils import tools


EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
MODEL_CFG = EXAMPLE_ROOT / "configs" / "model" / "dit-xl-2-256.yaml"
FASTDM_CFG = EXAMPLE_ROOT / "configs" / "fastdm" / "int4-dit.yaml"
QDIFF_BASELINE_CFG = EXAMPLE_ROOT / "configs" / "fastdm" / "int4-dit-baseline.yaml"


# --------------------------------------------------------------------------- #
# PTQ orchestration                                                            #
# --------------------------------------------------------------------------- #


def _build_argv(method, work_root):
    """Compose the CLI arg list that ``DiffusionPtqRunConfig`` would parse."""
    argv = [str(MODEL_CFG)]
    if method == "fastdm":
        # int4-dit.yaml already pins wgts.dtype=sint4 and turns FastDM on.
        argv.append(str(FASTDM_CFG))
    elif method == "qdiff":
        # Baseline INT4 weight-only stub: same wgts.dtype as fastdm but no
        # AdaRound, no smoothing. Lets us compare static range calibration
        # vs FastDM on identical W4A32 settings.
        argv.append(str(QDIFF_BASELINE_CFG))
    else:
        raise ValueError("unknown method: {!r}".format(method))
    argv += [
        "--skip-eval", "true",
        "--skip-gen", "true",  # we generate ourselves below
        "--output-root", str(work_root),
        "--output-dirname", method,
    ]
    return argv


def run_ptq_inline(method: str, work_root: Path, logger: logging.Logger):
    """Run PTQ in-process and return ``(pipeline, config)``.

    Mirrors ``deepcompressor.app.diffusion.ptq.main`` but stops short of the
    eval/generate phase so we can do class-conditional generation ourselves.
    """
    logger.info("=== PTQ: %s ===", method)
    parser = DiffusionPtqRunConfig.get_parser()
    config, _, unused_cfgs, unused_args, unknown = parser.parse_known_args(
        _build_argv(method, work_root)
    )
    assert isinstance(config, DiffusionPtqRunConfig)
    if unknown:
        raise RuntimeError(f"unknown CLI args: {unknown}")
    if unused_cfgs:
        logger.warning("unused configs: %s", unused_cfgs)

    config.output.lock()
    try:
        config.dump(path=config.output.get_running_job_path("config.yaml"))
        tools.logging.setup(
            path=config.output.get_running_job_path("run.log"),
            level=tools.logging.INFO,
        )

        pipeline = config.pipeline.build()
        model = DiffusionModelStruct.construct(pipeline)
        cache_dirpath = str(Path(config.output.running_job_dirpath) / "cache")
        ptq(
            model,
            config.quant,
            cache=config.cache,
            load_dirpath=config.load_from,
            save_dirpath=cache_dirpath,
            copy_on_save=config.copy_on_save,
            save_model=False,
        )
    except Exception:
        config.output.unlock(error=True)
        raise
    return pipeline, config


# --------------------------------------------------------------------------- #
# Class-conditional sample generation                                          #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def generate_samples(
    pipeline,
    out_dir: Path,
    num_samples: int,
    num_steps: int,
    guidance_scale: float = 4.0,
    batch_size: int = 4,
    seed: int = 0,
) -> None:
    """Generate ``num_samples`` images, round-robin across ImageNet classes.

    Output filenames are ``<idx>_cls<class>.png`` so a per-class FID would
    also be computable from the same directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline.set_progress_bar_config(disable=True)
    classes = [(i % 1000) for i in range(num_samples)]
    device = pipeline.device

    for i in range(0, num_samples, batch_size):
        batch_cls = classes[i : i + batch_size]
        gens = [
            torch.Generator(device=device).manual_seed(seed + i + j)
            for j in range(len(batch_cls))
        ]
        out = pipeline(
            class_labels=batch_cls,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=gens,
        )
        for j, img in enumerate(out.images):
            img.save(out_dir / f"{i + j:06d}_cls{batch_cls[j]:04d}.png")
        if (i // batch_size) % 10 == 0:
            print(f"  generated {min(i + batch_size, num_samples)} / {num_samples}", flush=True)


# --------------------------------------------------------------------------- #
# FID computation                                                              #
# --------------------------------------------------------------------------- #


def compute_fid(gen_dir: Path, reference: Path, device: str = "cuda") -> float:
    """Run ``python -m pytorch_fid <ref> <gen>`` and parse the FID line."""
    cmd = [
        sys.executable, "-m", "pytorch_fid",
        str(reference), str(gen_dir),
        "--device", device,
    ]
    print(f"$ {' '.join(cmd)}", flush=True)
    out = subprocess.check_output(cmd, text=True)
    print(out)
    for line in out.splitlines():
        if line.startswith("FID:"):
            return float(line.split(":", 1)[1].strip())
    raise RuntimeError(f"could not parse FID from pytorch_fid output:\n{out}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--work-root", type=Path, required=True,
                   help="Output root for PTQ runs and generated samples.")
    p.add_argument("--imagenet-ref", type=Path, required=True,
                   help="Path to ImageNet-256 reference image dir or *.npz stats.")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--methods", default="qdiff,fastdm",
                   help="Comma-separated subset of {qdiff,fastdm}.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-generation", action="store_true",
                   help="Reuse samples in <work-root>/<method>/samples/ (e.g. for FID retries).")
    p.add_argument("--skip-fid", action="store_true",
                   help="Run PTQ + generation only; print sample dirs.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger = logging.getLogger("compare_dit_fid")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sample_dirs: dict[str, Path] = {}

    for method in methods:
        sample_dir = args.work_root / method / "samples"
        sample_dirs[method] = sample_dir
        if args.skip_generation and sample_dir.exists() and any(sample_dir.iterdir()):
            logger.info("[%s] reusing existing samples in %s", method, sample_dir)
            continue
        pipeline, _ = run_ptq_inline(method, args.work_root, logger)
        try:
            generate_samples(
                pipeline,
                sample_dir,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                batch_size=args.batch_size,
                seed=args.seed,
            )
        finally:
            del pipeline
            torch.cuda.empty_cache()
        logger.info("[%s] %d samples written to %s", method, args.num_samples, sample_dir)

    if args.skip_fid:
        for method, d in sample_dirs.items():
            print(f"  {method:8s}  samples -> {d}")
        return 0

    fids: dict[str, float] = {}
    for method in methods:
        fids[method] = compute_fid(sample_dirs[method], args.imagenet_ref)

    print("\n=== FID Comparison: DiT-XL/2 256x256, "
          f"{args.num_steps} steps, W4A32, N={args.num_samples} ===")
    for method in methods:
        print(f"  {method:8s}  FID = {fids[method]:.3f}")
    if {"qdiff", "fastdm"}.issubset(set(methods)):
        delta = fids["fastdm"] - fids["qdiff"]
        marker = "fastdm better" if delta < 0 else "qdiff better" if delta > 0 else "tie"
        print(f"  delta(fastdm - qdiff) = {delta:+.3f}  ({marker})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

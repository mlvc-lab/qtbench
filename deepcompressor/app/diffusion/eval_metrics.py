from .eval.metrics import compute_image_metrics

if __name__ == "__main__":
    gen_root = "qtbench/examples/diffusion/final/diffusion/sdxl/sdxl/w.4-x.4-y.32/w.sfp4_e2m1_all-x.sfp4_e2m1_all-y.fp32/w.v16.sfp8_e4m3_nan.tsnr.fp32-x.v16.sfp8_e4m3_nan-y.tnsr.fp32/w.static/shift-skip.x.[[w]].w.[aa+e+fa+rs+rtp+s+tan+tn+tpi+tpo]-qdiff.128-t50.g5-s5000.RUNNING/run-260128.182007.RUNNING"
    ref_root = "qtbench/examples/diffusion/baselines/torch.float32/sdxl/euler50-g5"
    result = compute_image_metrics(
        gen_root=gen_root,
        benchmarks=("COCO"),
        max_dataset_size=3776,
        ref_root=ref_root,
        gt_stats_root="benchmarks/stats",
        gt_metrics=("clip_iqa", "clip_score", "image_reward", "fid"),
        ref_metrics=("psnr", "lpips", "ssim", "fid"),
    )
    print("Final results:")
    print(result)

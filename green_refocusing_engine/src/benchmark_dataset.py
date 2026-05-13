"""Unified dataset-scale benchmark runner for baseline vs Green method."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .blur_engine import apply_spatially_varying_blur
from .dataset_loader import NYUDataset
from .depth_estimator import DepthEstimator
from .energy_profiler import EnergyProfiler
from .evaluation import mse, psnr, save_unified_benchmark_csv, ssim
from .neural_baseline import repeated_neural_refocusing


def run_dataset_benchmark(
    rgb_dir: str,
    depth_dir: str,
    output_root: str,
    max_images: int = 50,
    focal_planes: list[float] | None = None,
    sigma_max: float = 8.0,
    num_sigma_levels: int = 12,
    alpha: float = 12.0,
) -> None:
    focal_planes = focal_planes or [0.1, 0.3, 0.5, 0.7, 0.9]
    sigma_levels = np.linspace(0.0, sigma_max, num_sigma_levels)

    ds = NYUDataset(rgb_dir=rgb_dir, depth_dir=depth_dir)
    n = len(ds) if max_images <= 0 else min(max_images, len(ds))

    estimator = DepthEstimator(model_name="LiheYoung/depth-anything-base-hf")
    profiler = EnergyProfiler()

    rows = []
    for i in tqdm(range(n), desc="Benchmarking dataset"):
        sample = ds[i]
        image, name = sample["rgb"], sample["name"]

        try:
            def _green_block():
                depth = estimator.infer_depth(image)[0].depth
                return [apply_spatially_varying_blur(depth, image, f, alpha, sigma_levels, sigma_max).image for f in focal_planes]

            green_outputs, gprof = profiler.profile_block(_green_block)
            g_mid = green_outputs[len(green_outputs) // 2]

            bres = repeated_neural_refocusing(image, focal_planes, estimator, alpha, sigma_levels, sigma_max, profiler)
            b_mid = bres.outputs[len(bres.outputs) // 2]

            row = {
                "image": name,
                "PSNR_original": psnr(image, b_mid),
                "PSNR_GreenComputing": psnr(image, g_mid),
                "SSIM_original": ssim(image, b_mid),
                "SSIM_GreenComputing": ssim(image, g_mid),
                "MSE_original": mse(image, b_mid),
                "MSE_GreenComputing": mse(image, g_mid),
                "runtime_original": bres.profile.runtime_s,
                "runtime_GreenComputing": gprof.runtime_s,
                "fps_original": len(focal_planes) / bres.profile.runtime_s,
                "fps_GreenComputing": len(focal_planes) / gprof.runtime_s,
                "gpu_memory_original": bres.profile.gpu_mem_mb,
                "gpu_memory_GreenComputing": gprof.gpu_mem_mb,
                "energy_original": bres.profile.energy_kwh,
                "energy_GreenComputing": gprof.energy_kwh,
                "status": "completed",
            }
        except Exception as exc:
            row = {
                "image": name,
                "PSNR_original": np.nan,
                "PSNR_GreenComputing": np.nan,
                "SSIM_original": np.nan,
                "SSIM_GreenComputing": np.nan,
                "MSE_original": np.nan,
                "MSE_GreenComputing": np.nan,
                "runtime_original": np.nan,
                "runtime_GreenComputing": np.nan,
                "fps_original": np.nan,
                "fps_GreenComputing": np.nan,
                "gpu_memory_original": np.nan,
                "gpu_memory_GreenComputing": np.nan,
                "energy_original": np.nan,
                "energy_GreenComputing": np.nan,
                "status": f"failed:{exc}",
            }
        rows.append(row)

    out_root = Path(output_root)
    df = save_unified_benchmark_csv(rows, out_root / "metrics" / "final_benchmark_comparison.csv")
    avg = df[df["image"] == "AVERAGE"].iloc[0]

    speedup = avg["runtime_original"] / max(avg["runtime_GreenComputing"], 1e-12)
    energy_reduction = ((avg["energy_original"] - avg["energy_GreenComputing"]) / max(avg["energy_original"], 1e-12)) * 100

    print("=" * 50)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Average Baseline PSNR: {avg['PSNR_original']:.4f}")
    print(f"Average Green PSNR: {avg['PSNR_GreenComputing']:.4f}")
    print(f"Average Baseline Runtime: {avg['runtime_original']:.4f}")
    print(f"Average Green Runtime: {avg['runtime_GreenComputing']:.4f}")
    print(f"Average Speedup: {speedup:.2f}x")
    print(f"Average Energy Reduction: {energy_reduction:.2f}%")
    print("=" * 50)

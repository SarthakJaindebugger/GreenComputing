"""Dataset-scale benchmark runner for Green vs Baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .blur_engine import apply_spatially_varying_blur
from .dataset_loader import NYUDataset
from .depth_estimator import DepthEstimator
from .energy_profiler import EnergyProfiler
from .evaluation import final_comparison_table, psnr, save_results_csv, ssim
from .neural_baseline import repeated_neural_refocusing
from .visualization import plot_metric_comparison, plot_quality_efficiency_tradeoff


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

    green_rows, base_rows = [], []
    for i in tqdm(range(n), desc="Benchmarking dataset"):
        sample = ds[i]
        image, name = sample["rgb"], sample["name"]

        try:
            # Green: compute once, filter many
            def _green_block():
                depth = estimator.infer_depth(image)[0].depth
                return [apply_spatially_varying_blur(depth, image, f, alpha, sigma_levels, sigma_max).image for f in focal_planes]

            green_outputs, gprof = profiler.profile_block(_green_block)
            for fp, out in zip(focal_planes, green_outputs):
                green_rows.append(
                    {
                        "image_name": name,
                        "focal_plane": fp,
                        "PSNR": psnr(image, out),
                        "SSIM": ssim(image, out),
                        "runtime_sec": gprof.runtime_s / len(focal_planes),
                        "fps": len(focal_planes) / gprof.runtime_s,
                        "gpu_memory_mb": gprof.gpu_mem_mb,
                        "energy_kwh": gprof.energy_kwh / len(focal_planes),
                        "co2_kg": gprof.co2_kg / len(focal_planes),
                        "status": "completed",
                    }
                )

            # Baseline: repeated inference
            bres = repeated_neural_refocusing(image, focal_planes, estimator, alpha, sigma_levels, sigma_max, profiler)
            for fp, out in zip(focal_planes, bres.outputs):
                base_rows.append(
                    {
                        "image_name": name,
                        "focal_plane": fp,
                        "PSNR": psnr(image, out),
                        "SSIM": ssim(image, out),
                        "runtime_sec": bres.profile.runtime_s / len(focal_planes),
                        "fps": len(focal_planes) / bres.profile.runtime_s,
                        "gpu_memory_mb": bres.profile.gpu_mem_mb,
                        "energy_kwh": bres.profile.energy_kwh / len(focal_planes),
                        "co2_kg": bres.profile.co2_kg / len(focal_planes),
                        "status": "completed",
                    }
                )
        except Exception as exc:
            green_rows.append({"image_name": name, "focal_plane": -1, "PSNR": np.nan, "SSIM": np.nan, "runtime_sec": np.nan, "fps": np.nan, "gpu_memory_mb": np.nan, "energy_kwh": np.nan, "co2_kg": np.nan, "status": f"failed:{exc}"})
            base_rows.append({"image_name": name, "focal_plane": -1, "PSNR": np.nan, "SSIM": np.nan, "runtime_sec": np.nan, "fps": np.nan, "gpu_memory_mb": np.nan, "energy_kwh": np.nan, "co2_kg": np.nan, "status": f"failed:{exc}"})

    out_root = Path(output_root)
    green_df = save_results_csv(green_rows, out_root / "metrics" / "green_results.csv")
    base_df = save_results_csv(base_rows, out_root / "metrics" / "baseline_results.csv")
    summary = final_comparison_table(green_df[green_df.status.str.startswith("completed")], base_df[base_df.status.str.startswith("completed")], out_root / "metrics" / "final_comparison.csv")

    plot_metric_comparison(summary, "Avg_Runtime", out_root / "plots" / "runtime_comparison.png")
    plot_metric_comparison(summary, "Avg_Energy", out_root / "plots" / "energy_comparison.png")
    plot_metric_comparison(summary, "Avg_FPS", out_root / "plots" / "fps_comparison.png")
    plot_metric_comparison(summary, "Avg_PSNR", out_root / "plots" / "psnr_comparison.png")
    plot_metric_comparison(summary, "Avg_SSIM", out_root / "plots" / "ssim_comparison.png")
    plot_quality_efficiency_tradeoff(summary, out_root / "plots" / "quality_efficiency_tradeoff.png")

    g, b = summary.iloc[0], summary.iloc[1]
    speedup = b.Avg_Runtime / max(g.Avg_Runtime, 1e-8)
    energy_reduction = ((b.Avg_Energy - g.Avg_Energy) / max(b.Avg_Energy, 1e-12)) * 100
    memory_reduction = ((b.Avg_GPU_Memory - g.Avg_GPU_Memory) / max(b.Avg_GPU_Memory, 1e-12)) * 100
    print("=" * 50)
    print("FINAL DATASET BENCHMARK")
    print("=" * 50)
    print("GREEN METHOD:")
    print(f"Avg PSNR: {g.Avg_PSNR:.4f}\nAvg SSIM: {g.Avg_SSIM:.4f}\nAvg Runtime: {g.Avg_Runtime:.4f} sec\nAvg FPS: {g.Avg_FPS:.4f}\nAvg GPU Memory: {g.Avg_GPU_Memory:.2f} MB\nAvg Energy: {g.Avg_Energy:.6f} kWh")
    print("\nBASELINE METHOD:")
    print(f"Avg PSNR: {b.Avg_PSNR:.4f}\nAvg SSIM: {b.Avg_SSIM:.4f}\nAvg Runtime: {b.Avg_Runtime:.4f} sec\nAvg FPS: {b.Avg_FPS:.4f}\nAvg GPU Memory: {b.Avg_GPU_Memory:.2f} MB\nAvg Energy: {b.Avg_Energy:.6f} kWh")
    print("=" * 50)
    print("EFFICIENCY GAIN")
    print("=" * 50)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Energy Reduction: {energy_reduction:.2f}%")
    print(f"Memory Reduction: {memory_reduction:.2f}%")

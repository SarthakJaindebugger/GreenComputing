from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .blur_engine import apply_spatially_varying_blur
    from .dataset_loader import NYUDataset
    from .depth_estimator import DepthEstimator
    from .evaluation import evaluate_green_vs_baseline
    from .neural_baseline import repeated_neural_refocusing
    from .utils import load_image_rgb, save_image_rgb, set_seed
    from .visualization import make_focus_sweep_gif, plot_bars, save_depth_and_sigma
except ImportError:
    # Fallback for direct script execution style: `python src/main.py`
    from src.blur_engine import apply_spatially_varying_blur
    from src.dataset_loader import NYUDataset
    from src.depth_estimator import DepthEstimator
    from src.evaluation import evaluate_green_vs_baseline
    from src.neural_baseline import repeated_neural_refocusing
    from src.utils import load_image_rgb, save_image_rgb, set_seed
    from src.visualization import make_focus_sweep_gif, plot_bars, save_depth_and_sigma


def run(args):
    set_seed(42)
    estimator = DepthEstimator()
    focus_values = np.linspace(0.1, 0.9, 5).tolist()
    sigma_levels = np.linspace(0.0, args.sigma_max, args.num_sigma_levels)

    if args.image_path:
        image = load_image_rgb(args.image_path)
        name = Path(args.image_path).stem
    else:
        ds = NYUDataset(rgb_dir=args.rgb_dir, depth_dir=args.depth_dir, resize=(args.width, args.height))
        sample = ds[0]
        image, name = sample["rgb"], sample["name"]

    depth = estimator.infer_depth(image)[0].depth
    green_results = [
        apply_spatially_varying_blur(depth, image, f, alpha=args.alpha, sigma_levels=sigma_levels, sigma_max=args.sigma_max)
        for f in focus_values
    ]
    green_outputs = [r.image for r in green_results]
    baseline = repeated_neural_refocusing(
        image, focus_values, estimator, alpha=args.alpha, sigma_levels=sigma_levels, sigma_max=args.sigma_max
    )

    out_root = Path(args.output_root)
    save_image_rgb(out_root / "refocused_images" / f"{name}_green.png", green_outputs[len(green_outputs) // 2])
    save_image_rgb(out_root / "comparisons" / f"{name}_baseline.png", baseline.outputs[len(baseline.outputs) // 2])
    save_depth_and_sigma(depth, green_results[len(green_results) // 2].sigma_map, out_root / "depth_maps")
    make_focus_sweep_gif(green_outputs, out_root / "refocused_images" / f"{name}_sweep.gif")

    df = evaluate_green_vs_baseline(green_outputs, baseline.outputs, image, out_root / "metrics" / "quality_metrics.csv")
    for metric in ["PSNR", "SSIM", "MSE"]:
        plot_bars(df, metric, out_root / "plots" / f"{metric.lower()}_comparison.png")
    print(df)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Green Refocusing Engine runner")
    p.add_argument("--image_path", type=str, default="", help="Optional single image path")
    p.add_argument("--rgb_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images")
    p.add_argument("--depth_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images")
    p.add_argument("--sigma_max", type=float, default=8.0)
    p.add_argument("--num_sigma_levels", type=int, default=12)
    p.add_argument("--alpha", type=float, default=12.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--output_root", type=str, default="outputs")
    run(p.parse_args())

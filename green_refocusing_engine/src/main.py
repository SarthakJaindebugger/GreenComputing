from __future__ import annotations

import argparse

from .benchmark_dataset import run_dataset_benchmark


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Green Refocusing Engine Dataset Benchmark")
    p.add_argument("--rgb_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images")
    p.add_argument("--depth_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images")
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--max_images", type=int, default=50)
    p.add_argument("--sigma_max", type=float, default=8.0)
    p.add_argument("--num_sigma_levels", type=int, default=12)
    p.add_argument("--alpha", type=float, default=12.0)
    args = p.parse_args()

    run_dataset_benchmark(
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        output_root=args.output_root,
        max_images=args.max_images,
        sigma_max=args.sigma_max,
        num_sigma_levels=args.num_sigma_levels,
        alpha=args.alpha,
    )

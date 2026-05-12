"""One-click Python runner (no bash required)."""

from __future__ import annotations

from src.main import run


class Args:
    image_path = ""
    rgb_dir = "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images"
    depth_dir = "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/depth_images"
    sigma_max = 8.0
    num_sigma_levels = 12
    alpha = 12.0
    width = 640
    height = 480
    output_root = "outputs"


if __name__ == "__main__":
    run(Args())

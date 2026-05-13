"""Self-contained runner for full-dataset Green refocusing evaluation.

Run:
python run_pipeline.py --rgb_dir <folder> --output_root outputs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import pipeline


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image_rgb(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def save_image_rgb(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    bgr = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def depth_model(device: int = -1):
    return pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=device)


def infer_depth(pipe, image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    out = pipe(pil)
    d = np.array(out["depth"], dtype=np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def generate_gaussian_kernel(sigma: float) -> np.ndarray:
    if sigma <= 1e-8:
        return np.array([[1.0]], dtype=np.float32)
    r = int(3.0 * sigma + 0.5)
    x = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(x, x)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).astype(np.float32)


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pad = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -kh // 2, axis=0)
    pad = np.roll(pad, -kw // 2, axis=1)
    return fft2(pad)


def fft_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    otf = psf_to_otf(generate_gaussian_kernel(sigma), image.shape[:2])
    out = np.zeros_like(image)
    for c in range(3):
        out[..., c] = np.real(ifft2(fft2(image[..., c]) * otf))
    return np.clip(out, 0, 1)


def apply_spatially_varying_blur(
    image: np.ndarray,
    depth: np.ndarray,
    focus: float,
    alpha: float,
    sigma_levels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_map = np.clip(alpha * np.abs(depth - focus), 0, float(sigma_levels[-1]))
    pyr = np.stack([fft_gaussian_blur(image, float(s)) for s in sigma_levels], axis=0)
    idx_hi = np.searchsorted(sigma_levels, sigma_map, side="left")
    idx_hi = np.clip(idx_hi, 1, len(sigma_levels) - 1)
    idx_lo = idx_hi - 1
    s_lo, s_hi = sigma_levels[idx_lo], sigma_levels[idx_hi]
    w = (sigma_map - s_lo) / (s_hi - s_lo + 1e-8)
    lo = np.take_along_axis(pyr, idx_lo[None, ..., None], axis=0).squeeze(0)
    hi = np.take_along_axis(pyr, idx_hi[None, ..., None], axis=0).squeeze(0)
    return np.clip((1 - w[..., None]) * lo + w[..., None] * hi, 0, 1), sigma_map


def process_one_image(
    image_path: Path,
    pipe,
    output_root: Path,
    sigma_levels: np.ndarray,
    alpha: float,
    focus_values: np.ndarray,
    save_artifacts: bool,
) -> dict:
    t0 = time.perf_counter()
    image = load_image_rgb(str(image_path))
    depth = infer_depth(pipe, image)

    outputs = []
    sigma_map = None
    for f in focus_values:
        ref, sigma_map = apply_spatially_varying_blur(image, depth, float(f), alpha, sigma_levels)
        outputs.append(ref)

    mid = outputs[len(outputs) // 2]
    psnr = float(peak_signal_noise_ratio(image, mid, data_range=1.0))
    ssim = float(structural_similarity(image, mid, channel_axis=2, data_range=1.0))
    mse = float(np.mean((image - mid) ** 2))
    runtime_s = time.perf_counter() - t0

    name = image_path.stem
    if save_artifacts:
        save_image_rgb(output_root / "refocused_images" / f"{name}_green.png", mid)
        ensure_dir(output_root / "refocused_images")
        frames = [(np.clip(o, 0, 1) * 255).astype(np.uint8) for o in outputs]
        imageio.mimsave(output_root / "refocused_images" / f"{name}_sweep.gif", frames, duration=0.25)
        ensure_dir(output_root / "depth_maps")
        plt.imsave(output_root / "depth_maps" / f"{name}_depth.png", depth, cmap="magma")
        if sigma_map is not None:
            plt.imsave(output_root / "depth_maps" / f"{name}_sigma.png", sigma_map, cmap="viridis")

    return {
        "image": name,
        "PSNR_vs_original": psnr,
        "SSIM_vs_original": ssim,
        "MSE_vs_original": mse,
        "runtime_s": runtime_s,
        "status": "completed",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images")
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--sigma_max", type=float, default=8.0)
    p.add_argument("--num_sigma_levels", type=int, default=12)
    p.add_argument("--alpha", type=float, default=12.0)
    p.add_argument("--num_images", type=int, default=0, help="0 means all images")
    p.add_argument("--save_artifacts", action="store_true")
    args = p.parse_args()

    rgb_dir = Path(args.rgb_dir)
    image_paths = sorted([*rgb_dir.glob("*.png"), *rgb_dir.glob("*.jpg"), *rgb_dir.glob("*.jpeg")])
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found in {rgb_dir}")

    if args.num_images > 0:
        image_paths = image_paths[: args.num_images]

    output_root = Path(args.output_root)
    ensure_dir(output_root / "metrics")

    pipe = depth_model(device=-1)
    sigma_levels = np.linspace(0.0, args.sigma_max, args.num_sigma_levels)
    focus_values = np.linspace(0.1, 0.9, 5)

    rows = []
    for pth in image_paths:
        row = process_one_image(
            image_path=pth,
            pipe=pipe,
            output_root=output_root,
            sigma_levels=sigma_levels,
            alpha=args.alpha,
            focus_values=focus_values,
            save_artifacts=args.save_artifacts,
        )
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    summary = {
        "image": "__AVERAGE__",
        "PSNR_vs_original": float(df["PSNR_vs_original"].mean()),
        "SSIM_vs_original": float(df["SSIM_vs_original"].mean()),
        "MSE_vs_original": float(df["MSE_vs_original"].mean()),
        "runtime_s": float(df["runtime_s"].mean()),
        "status": "summary",
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    out_csv = output_root / "metrics" / "final_dataset_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved final CSV: {out_csv}")


if __name__ == "__main__":
    main()

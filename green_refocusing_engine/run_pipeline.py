"""Self-contained runner to avoid cross-file import issues.

Run:
python run_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
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
    out = pipe((np.clip(image, 0, 1) * 255).astype(np.uint8))
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


def apply_spatially_varying_blur(image: np.ndarray, depth: np.ndarray, focus: float, alpha: float, sigma_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir", type=str, default="/Users/sarthakjain/Desktop/ML Projects/GreenComputing/nyu_data/rbg_images")
    p.add_argument("--image_path", type=str, default="")
    p.add_argument("--output_root", type=str, default="outputs")
    p.add_argument("--sigma_max", type=float, default=8.0)
    p.add_argument("--num_sigma_levels", type=int, default=12)
    p.add_argument("--alpha", type=float, default=12.0)
    args = p.parse_args()

    if args.image_path:
        image_path = args.image_path
    else:
        rgb_dir = Path(args.rgb_dir)
        candidates = sorted([*rgb_dir.glob("*.png"), *rgb_dir.glob("*.jpg"), *rgb_dir.glob("*.jpeg")])
        if not candidates:
            raise FileNotFoundError(f"No RGB images found in {rgb_dir}")
        image_path = str(candidates[0])

    image = load_image_rgb(image_path)
    name = Path(image_path).stem
    pipe = depth_model(device=-1)
    depth = infer_depth(pipe, image)

    focus_values = np.linspace(0.1, 0.9, 5)
    sigma_levels = np.linspace(0.0, args.sigma_max, args.num_sigma_levels)
    outputs = []
    sigma_map = None
    for f in focus_values:
        ref, sigma_map = apply_spatially_varying_blur(image, depth, float(f), args.alpha, sigma_levels)
        outputs.append(ref)

    out_root = Path(args.output_root)
    save_image_rgb(out_root / "refocused_images" / f"{name}_green.png", outputs[len(outputs) // 2])
    frames = [(np.clip(o, 0, 1) * 255).astype(np.uint8) for o in outputs]
    ensure_dir(out_root / "refocused_images")
    imageio.mimsave(out_root / "refocused_images" / f"{name}_sweep.gif", frames, duration=0.25)

    ensure_dir(out_root / "depth_maps")
    plt.imsave(out_root / "depth_maps" / f"{name}_depth.png", depth, cmap="magma")
    if sigma_map is not None:
        plt.imsave(out_root / "depth_maps" / f"{name}_sigma.png", sigma_map, cmap="viridis")

    psnr = float(peak_signal_noise_ratio(image, outputs[len(outputs) // 2], data_range=1.0))
    ssim = float(structural_similarity(image, outputs[len(outputs) // 2], channel_axis=2, data_range=1.0))
    print({"image": name, "PSNR": psnr, "SSIM": ssim, "status": "completed"})


if __name__ == "__main__":
    main()

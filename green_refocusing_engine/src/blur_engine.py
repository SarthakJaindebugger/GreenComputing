"""FFT-based refocusing engine.

Using convolution theorem: spatial convolution equals element-wise multiplication in Fourier domain.
This reduces complexity for large kernels and makes runtime nearly independent of kernel radius.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import fft2, ifft2

try:
    from pypher.pypher import psf2otf
except Exception:  # optional dependency path issues
    psf2otf = None


def generate_gaussian_kernel(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 1e-8:
        return np.array([[1.0]], dtype=np.float32)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def psf_to_otf(psf: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if psf2otf is not None:
        return psf2otf(psf, shape)
    padded = np.zeros(shape, dtype=np.float32)
    kh, kw = psf.shape
    padded[:kh, :kw] = psf
    padded = np.roll(padded, -kh // 2, axis=0)
    padded = np.roll(padded, -kw // 2, axis=1)
    return fft2(padded)


def fft_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel = generate_gaussian_kernel(sigma)
    h, w = image.shape[:2]
    otf = psf_to_otf(kernel, (h, w))
    out = np.zeros_like(image)
    for c in range(image.shape[2]):
        out[..., c] = np.real(ifft2(fft2(image[..., c]) * otf))
    return np.clip(out, 0, 1)


def precompute_blur_pyramid(image: np.ndarray, sigma_levels: np.ndarray) -> list[np.ndarray]:
    return [fft_gaussian_blur(image, float(s)) for s in sigma_levels]


def compute_sigma_map(depth: np.ndarray, focal_depth: float, alpha: float, sigma_max: float) -> np.ndarray:
    sigma = alpha * np.abs(depth - focal_depth)
    return np.clip(sigma, 0.0, sigma_max)


@dataclass
class BlurSynthesisResult:
    image: np.ndarray
    sigma_map: np.ndarray


def apply_spatially_varying_blur(
    depth: np.ndarray,
    image: np.ndarray,
    focal_depth: float,
    alpha: float,
    sigma_levels: np.ndarray,
    sigma_max: float,
) -> BlurSynthesisResult:
    sigma_map = compute_sigma_map(depth, focal_depth, alpha, sigma_max)
    pyramid = np.stack(precompute_blur_pyramid(image, sigma_levels), axis=0)

    idx_hi = np.searchsorted(sigma_levels, sigma_map, side="left")
    idx_hi = np.clip(idx_hi, 1, len(sigma_levels) - 1)
    idx_lo = idx_hi - 1

    s_lo = sigma_levels[idx_lo]
    s_hi = sigma_levels[idx_hi]
    w = (sigma_map - s_lo) / (s_hi - s_lo + 1e-8)

    lo = np.take_along_axis(pyramid, idx_lo[None, ..., None], axis=0).squeeze(0)
    hi = np.take_along_axis(pyramid, idx_hi[None, ..., None], axis=0).squeeze(0)
    out = (1.0 - w[..., None]) * lo + w[..., None] * hi
    return BlurSynthesisResult(image=np.clip(out, 0, 1), sigma_map=sigma_map)

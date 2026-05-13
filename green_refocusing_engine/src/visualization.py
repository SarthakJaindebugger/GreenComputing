"""Visualization utilities for plots and focus sweep outputs."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_bars(df: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(df["Method"], df[metric], color=["#2ca02c", "#d62728"])
    plt.ylabel(metric)
    plt.title(f"{metric} comparison")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_depth_and_sigma(depth: np.ndarray, sigma_map: np.ndarray, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arr, name, cmap in [(depth, "depth_map.png", "magma"), (sigma_map, "sigma_map.png", "viridis")]:
        plt.figure(figsize=(6, 4))
        plt.imshow(arr, cmap=cmap)
        plt.axis("off")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=220)
        plt.close()


def make_focus_sweep_gif(images: list[np.ndarray], out_path: str | Path, fps: int = 4) -> None:
    frames = [(np.clip(im, 0, 1) * 255).astype(np.uint8) for im in images]
    imageio.mimsave(out_path, frames, duration=1.0 / fps)

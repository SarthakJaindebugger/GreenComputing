"""Evaluation and aggregation helpers for full-dataset benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(x, y, data_range=1.0))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    return float(structural_similarity(x, y, channel_axis=2, data_range=1.0))


def save_results_csv(rows: list[dict], out_csv: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize_method(df: pd.DataFrame, method_name: str) -> dict:
    return {
        "Method": method_name,
        "Avg_PSNR": float(df["PSNR"].mean()),
        "Avg_SSIM": float(df["SSIM"].mean()),
        "Avg_Runtime": float(df["runtime_sec"].mean()),
        "Avg_FPS": float(df["fps"].mean()),
        "Avg_GPU_Memory": float(df["gpu_memory_mb"].mean()),
        "Avg_Energy": float(df["energy_kwh"].mean()),
    }


def final_comparison_table(green_df: pd.DataFrame, baseline_df: pd.DataFrame, out_csv: str | Path) -> pd.DataFrame:
    table = pd.DataFrame(
        [
            summarize_method(green_df, "Green FFT Hybrid"),
            summarize_method(baseline_df, "Baseline Neural"),
        ]
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    return table

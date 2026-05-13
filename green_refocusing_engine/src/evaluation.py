"""Evaluation and unified benchmark CSV helpers."""

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


REQUIRED_COLUMNS = [
    "image",
    "PSNR_original",
    "PSNR_GreenComputing",
    "SSIM_original",
    "SSIM_GreenComputing",
    "MSE_original",
    "MSE_GreenComputing",
    "runtime_original",
    "runtime_GreenComputing",
    "fps_original",
    "fps_GreenComputing",
    "gpu_memory_original",
    "gpu_memory_GreenComputing",
    "energy_original",
    "energy_GreenComputing",
    "status",
]


def save_unified_benchmark_csv(rows: list[dict], out_csv: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[REQUIRED_COLUMNS]

    completed = df[df["status"] == "completed"].copy()
    avg_row = {"image": "AVERAGE", "status": "summary"}
    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in {"image", "status"}]
    for c in numeric_cols:
        avg_row[c] = float(completed[c].mean()) if len(completed) else np.nan

    out_df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df

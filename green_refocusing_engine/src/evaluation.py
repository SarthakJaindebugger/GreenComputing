"""Evaluation metrics and benchmarking."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    from .energy_profiler import EnergyProfiler
except ImportError:
    from energy_profiler import EnergyProfiler


def mse(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((x - y) ** 2))


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(x, y, data_range=1.0))


def ssim(x: np.ndarray, y: np.ndarray) -> float:
    return float(structural_similarity(x, y, channel_axis=2, data_range=1.0))


def evaluate_green_vs_baseline(green_outputs, base_outputs, reference, out_csv: str | Path) -> pd.DataFrame:
    rows = []
    for method, outs in [("Green FFT Hybrid", green_outputs), ("Baseline Neural", base_outputs)]:
        vals = [
            {"MSE": mse(o, reference), "PSNR": psnr(o, reference), "SSIM": ssim(o, reference)}
            for o in outs
        ]
        agg = {k: float(np.mean([v[k] for v in vals])) for k in vals[0].keys()}
        rows.append({"Method": method, **agg})
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

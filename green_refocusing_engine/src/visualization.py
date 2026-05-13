"""Plotting utilities for dataset-scale benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric_comparison(summary_df: pd.DataFrame, metric: str, out_path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(summary_df["Method"], summary_df[metric], color=["#2ca02c", "#d62728"])
    plt.ylabel(metric)
    plt.title(f"{metric} comparison")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_quality_efficiency_tradeoff(summary_df: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(summary_df["Avg_Runtime"], summary_df["Avg_PSNR"], s=120)
    for _, r in summary_df.iterrows():
        plt.annotate(r["Method"], (r["Avg_Runtime"], r["Avg_PSNR"]))
    plt.xlabel("Avg Runtime (s)")
    plt.ylabel("Avg PSNR")
    plt.title("Quality vs Efficiency")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

"""Heavy baseline: repeated depth inference at every focal plane."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blur_engine import apply_spatially_varying_blur
from .depth_estimator import DepthEstimator
from .energy_profiler import EnergyProfiler, ProfileStats


@dataclass
class BaselineResult:
    outputs: list[np.ndarray]
    profile: ProfileStats


def repeated_neural_refocusing(
    image: np.ndarray,
    focus_values: list[float],
    estimator: DepthEstimator,
    alpha: float,
    sigma_levels: np.ndarray,
    sigma_max: float,
    profiler: EnergyProfiler,
) -> BaselineResult:
    def _run():
        outs = []
        for f in focus_values:
            depth = estimator.infer_depth(image)[0].depth
            outs.append(apply_spatially_varying_blur(depth, image, f, alpha, sigma_levels, sigma_max).image)
        return outs

    outs, prof = profiler.profile_block(_run)
    return BaselineResult(outputs=outs, profile=prof)

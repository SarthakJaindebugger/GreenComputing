"""Heavy baseline: repeated depth inference per focus setting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blur_engine import apply_spatially_varying_blur
from depth_estimator import DepthEstimator


@dataclass
class BaselineRun:
    outputs: list[np.ndarray]
    total_runtime_s: float


def repeated_neural_refocusing(
    image: np.ndarray,
    focus_values: list[float],
    estimator: DepthEstimator,
    alpha: float,
    sigma_levels: np.ndarray,
    sigma_max: float,
) -> BaselineRun:
    outputs = []
    total = 0.0
    for f in focus_values:
        depth_result = estimator.infer_depth(image)[0]
        total += depth_result.latency_s
        res = apply_spatially_varying_blur(depth_result.depth, image, f, alpha, sigma_levels, sigma_max)
        outputs.append(res.image)
    return BaselineRun(outputs=outputs, total_runtime_s=total)

"""Depth estimation module using Hugging Face transformers Depth Anything.

This module intentionally uses transformers-only loading (no local dev repo imports).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


@dataclass
class DepthInferenceResult:
    depth: np.ndarray
    latency_s: float


class DepthEstimator:
    def __init__(
        self,
        model_name: str = "LiheYoung/depth-anything-base-hf",
        device: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = self._resolve_pipeline_device(device)
        self.model = self.initialize_model(model_name)

    def _resolve_pipeline_device(self, device: int | None) -> int:
        if device is not None:
            return device
        # transformers pipeline device index is CUDA-centric.
        # On Mac/MPS, fallback to CPU for compatibility.
        return 0 if torch.cuda.is_available() else -1

    def initialize_model(self, model_name: str):
        return pipeline(task="depth-estimation", model=model_name, device=self.device)

    @torch.inference_mode()
    def infer_depth(self, images: Iterable[np.ndarray] | np.ndarray) -> list[DepthInferenceResult]:
        if isinstance(images, np.ndarray):
            images = [images]
        images = list(images)

        t0 = time.perf_counter()
        depths = []
        for image in images:
            pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            result = self.model(pil)
            depth = np.array(result["depth"], dtype=np.float32)
            depths.append(self.normalize_depth_map(depth))
        elapsed = time.perf_counter() - t0
        latency = elapsed / max(1, len(depths))
        return [DepthInferenceResult(depth=d, latency_s=latency) for d in depths]

    def normalize_depth_map(self, depth: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        return (depth - depth.min()) / (depth.max() - depth.min() + eps)

    def save_depth_visualization(self, depth: np.ndarray, out_path: str | Path) -> None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.imshow(depth, cmap="magma")
        plt.colorbar(label="Relative depth")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=220)
        plt.close()

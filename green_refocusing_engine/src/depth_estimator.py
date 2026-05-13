"""Depth estimation module using Depth Anything.

Monocular depth estimation predicts a relative depth field D(x, y) from one RGB image.
The output preserves ordinal geometry and local structure but not absolute metric scale.
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
from torchvision import transforms

from depth_anything.dpt import DepthAnything


@dataclass
class DepthInferenceResult:
    depth: np.ndarray
    latency_s: float


class DepthEstimator:
    def __init__(self, model_name: str = "LiheYoung/depth_anything_vits14", device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.initialize_model(model_name)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def initialize_model(self, model_name: str) -> torch.nn.Module:
        model = DepthAnything.from_pretrained(model_name).to(self.device).eval()
        return model

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        return tensor

    @torch.inference_mode()
    def infer_depth(self, images: Iterable[np.ndarray] | np.ndarray) -> list[DepthInferenceResult]:
        if isinstance(images, np.ndarray):
            images = [images]
        batch = torch.cat([self.preprocess_image(im) for im in images], dim=0)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = self.model(batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        pred_np = pred.squeeze(1).detach().cpu().numpy()
        return [DepthInferenceResult(depth=self.normalize_depth_map(d), latency_s=latency / len(pred_np)) for d in pred_np]

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

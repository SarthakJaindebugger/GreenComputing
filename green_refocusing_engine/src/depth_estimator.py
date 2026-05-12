"""Depth estimation module with local Depth-Anything repo support.

Import priority:
1) local cloned Depth-Anything repo path,
2) installed `depth_anything` package,
3) transformers fallback.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

LOCAL_DEPTH_ANYTHING_ROOT = Path(
    os.getenv(
        "DEPTH_ANYTHING_LOCAL_REPO",
        "/Users/sarthakjain/Desktop/ML Projects/GreenComputing/Depth-Anything",
    )
)
if LOCAL_DEPTH_ANYTHING_ROOT.exists():
    sys.path.insert(0, str(LOCAL_DEPTH_ANYTHING_ROOT))

DEPTH_ANYTHING_IMPORT_ERROR: Exception | None = None
try:
    from depth_anything.dpt import DepthAnything  # type: ignore
except Exception as exc:  # noqa: BLE001
    DepthAnything = None
    DEPTH_ANYTHING_IMPORT_ERROR = exc

TRANSFORMERS_IMPORT_ERROR: Exception | None = None
try:
    from transformers import pipeline
except Exception as exc:  # noqa: BLE001
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = exc


@dataclass
class DepthInferenceResult:
    depth: np.ndarray
    latency_s: float


class DepthEstimator:
    def __init__(self, model_name: str = "LiheYoung/depth_anything_vits14", device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.backend = ""
        self.model = self.initialize_model(model_name)
        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def initialize_model(self, model_name: str):
        if DepthAnything is not None:
            self.backend = "depth_anything"
            return DepthAnything.from_pretrained(model_name).to(self.device).eval()

        if pipeline is not None:
            self.backend = "transformers"
            device_idx = 0 if self.device.type == "cuda" else -1
            return pipeline("depth-estimation", model=model_name, device=device_idx)

        raise ImportError(
            "No depth backend available.\n"
            f"Tried local repo: {LOCAL_DEPTH_ANYTHING_ROOT}\n"
            "Install one of:\n"
            "1) ensure cloned repo path contains `depth_anything/` package\n"
            "2) pip install git+https://github.com/LiheYoung/Depth-Anything.git\n"
            "3) pip install transformers\n"
            f"depth_anything import error: {DEPTH_ANYTHING_IMPORT_ERROR}\n"
            f"transformers import error: {TRANSFORMERS_IMPORT_ERROR}"
        )

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
        tensor = self.preprocess(pil).unsqueeze(0).to(self.device)
        return tensor

    @torch.inference_mode()
    def infer_depth(self, images: Iterable[np.ndarray] | np.ndarray) -> list[DepthInferenceResult]:
        if isinstance(images, np.ndarray):
            images = [images]
        images = list(images)

        if self.backend == "depth_anything":
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

        t0 = time.perf_counter()
        results = []
        for im in images:
            pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
            out = self.model(pil)
            depth_img = np.array(out["depth"], dtype=np.float32)
            results.append(self.normalize_depth_map(depth_img))
        latency = (time.perf_counter() - t0) / max(1, len(images))
        return [DepthInferenceResult(depth=d, latency_s=latency) for d in results]

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

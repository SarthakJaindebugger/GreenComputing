"""Utility helpers for reproducible computational imaging experiments."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_rgb(path: str | Path, resize_to: Tuple[int, int] | None = None) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if resize_to is not None:
        image = cv2.resize(image, resize_to, interpolation=cv2.INTER_AREA)
    return image


def save_image_rgb(path: str | Path, image: np.ndarray) -> None:
    ensure_dir(Path(path).parent)
    clipped = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(clipped, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def normalize_minmax(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min, x_max = float(x.min()), float(x.max())
    return (x - x_min) / (x_max - x_min + eps)


def to_torch_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

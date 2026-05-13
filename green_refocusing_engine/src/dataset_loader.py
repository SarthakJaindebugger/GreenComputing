"""NYU Depth V2 loader utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NYUDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "test", resize: tuple[int, int] = (640, 480), transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.split = split
        self.resize = resize
        self.transform = transform
        self.rgb_files = sorted((self.root / split / "rgb").glob("*.png"))
        self.depth_files = sorted((self.root / split / "depth").glob("*.png"))
        if len(self.rgb_files) != len(self.depth_files):
            raise ValueError("RGB/depth file count mismatch.")

    def __len__(self) -> int:
        return len(self.rgb_files)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        rgb = cv2.cvtColor(cv2.imread(str(self.rgb_files[idx])), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(self.depth_files[idx]), cv2.IMREAD_UNCHANGED)
        rgb = cv2.resize(rgb, self.resize, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        depth = cv2.resize(depth, self.resize, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        sample = {"rgb": rgb, "depth": depth, "name": self.rgb_files[idx].stem}
        return self.transform(sample) if self.transform else sample


def create_dataloader(root: str | Path, split: str = "test", batch_size: int = 4, num_workers: int = 2) -> DataLoader:
    return DataLoader(NYUDataset(root=root, split=split), batch_size=batch_size, shuffle=False, num_workers=num_workers)

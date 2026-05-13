"""NYU Depth V2-style dataset loader.

Supports either:
- root/test/rgb + root/test/depth layout, or
- explicit rgb_dir + depth_dir layout (e.g., user's local folders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class NYUDataset(Dataset):
    def __init__(
        self,
        root: str | Path | None = None,
        split: str = "test",
        resize: tuple[int, int] = (640, 480),
        transform: Callable | None = None,
        rgb_dir: str | Path | None = None,
        depth_dir: str | Path | None = None,
    ) -> None:
        self.resize = resize
        self.transform = transform

        if rgb_dir and depth_dir:
            rgb_path = Path(rgb_dir)
            depth_path = Path(depth_dir)
        elif root is not None:
            base = Path(root)
            rgb_path = base / split / "rgb"
            depth_path = base / split / "depth"
        else:
            raise ValueError("Provide either (root) or (rgb_dir and depth_dir).")

        # also accept common typo folder name rbg_images
        if not rgb_path.exists() and rgb_dir:
            alt = Path(str(rgb_dir).replace("rbg_", "rgb_"))
            if alt.exists():
                rgb_path = alt

        self.rgb_files = sorted([*rgb_path.glob("*.png"), *rgb_path.glob("*.jpg"), *rgb_path.glob("*.jpeg")])
        self.depth_files = sorted([*depth_path.glob("*.png"), *depth_path.glob("*.jpg")])

        if len(self.rgb_files) == 0 or len(self.depth_files) == 0:
            raise FileNotFoundError(f"No files found. RGB dir: {rgb_path}, Depth dir: {depth_path}")
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


def create_dataloader(
    root: str | Path | None = None,
    split: str = "test",
    batch_size: int = 4,
    num_workers: int = 2,
    rgb_dir: str | Path | None = None,
    depth_dir: str | Path | None = None,
) -> DataLoader:
    dataset = NYUDataset(root=root, split=split, rgb_dir=rgb_dir, depth_dir=depth_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

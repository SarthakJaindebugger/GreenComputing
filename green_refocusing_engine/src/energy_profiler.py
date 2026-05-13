"""Energy and runtime profiling utilities."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

try:
    import pynvml
except Exception:
    pynvml = None


@dataclass
class ProfileStats:
    runtime_s: float
    gpu_mem_mb: float
    gpu_util_percent: float
    energy_j_estimate: float
    co2_g_estimate: float


class EnergyProfiler:
    def __init__(self, assumed_gpu_power_w: float = 140.0, co2_per_kwh_g: float = 400.0) -> None:
        self.assumed_gpu_power_w = assumed_gpu_power_w
        self.co2_per_kwh_g = co2_per_kwh_g

    def profile_block(self, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        runtime = time.perf_counter() - t0
        mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        util = self._gpu_utilization()
        energy_j = runtime * self.assumed_gpu_power_w * max(0.15, util / 100.0)
        co2_g = (energy_j / 3_600_000.0) * self.co2_per_kwh_g
        return out, ProfileStats(runtime, mem_mb, util, energy_j, co2_g)

    def _gpu_utilization(self) -> float:
        if pynvml is None:
            return 0.0
        try:
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            pynvml.nvmlShutdown()
            return float(util)
        except Exception:
            return 0.0

    @staticmethod
    def append_csv(path: str | Path, row: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                w.writeheader()
            w.writerow(row)

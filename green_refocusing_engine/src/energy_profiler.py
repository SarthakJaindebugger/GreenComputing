"""Energy and resource profiling utilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class ProfileStats:
    runtime_s: float
    fps: float
    gpu_mem_mb: float
    gpu_util_percent: float
    energy_kwh: float
    co2_kg: float


class EnergyProfiler:
    def __init__(self, assumed_power_w: float = 140.0, co2_kg_per_kwh: float = 0.4) -> None:
        self.assumed_power_w = assumed_power_w
        self.co2_kg_per_kwh = co2_kg_per_kwh

    def profile_block(self, fn, *args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtime = max(1e-8, time.perf_counter() - t0)
        fps = 1.0 / runtime
        gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
        util = self._gpu_utilization()
        util_ratio = util / 100.0 if util > 0 else 0.15
        energy_kwh = (self.assumed_power_w * util_ratio * runtime) / 3_600_000.0
        co2_kg = energy_kwh * self.co2_kg_per_kwh
        return out, ProfileStats(runtime, fps, gpu_mem, util, energy_kwh, co2_kg)

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

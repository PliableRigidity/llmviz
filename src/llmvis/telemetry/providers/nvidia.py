"""NVIDIA telemetry provider using pynvml.

Falls back gracefully to CPU-only snapshot if pynvml is not installed or
the NVML init fails (e.g. no NVIDIA driver present).
"""

from __future__ import annotations

import logging

import psutil

from llmvis.telemetry.providers.base import BaseTelemetryProvider
from llmvis.telemetry.snapshot import SystemSnapshot

logger = logging.getLogger(__name__)

_nvml_ok: bool | None = None  # None = not yet tried


def _ensure_nvml() -> bool:
    global _nvml_ok
    if _nvml_ok is not None:
        return _nvml_ok
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_ok = True
        logger.info("pynvml initialised — NVIDIA telemetry active")
    except ImportError:
        logger.info("pynvml not installed — no NVIDIA telemetry (pip install llmvis[nvidia])")
        _nvml_ok = False
    except Exception as exc:
        logger.info("pynvml init failed: %s", exc)
        _nvml_ok = False
    return _nvml_ok


class NvidiaTelemetryProvider(BaseTelemetryProvider):
    """Telemetry for systems with one or more NVIDIA GPUs."""

    def __init__(self, gpu_index: int = 0, device_name: str = "") -> None:
        self._gpu_index = gpu_index
        self._device_name = device_name or "NVIDIA GPU"

    def snapshot(self) -> SystemSnapshot:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)

        gpu_pct: float | None = None
        gpu_mem_used: int | None = None
        gpu_mem_total: int | None = None

        if _ensure_nvml():
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                vram = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_pct = float(util.gpu)
                gpu_mem_used = int(vram.used)
                gpu_mem_total = int(vram.total)
                if not self._device_name or self._device_name == "NVIDIA GPU":
                    name = pynvml.nvmlDeviceGetName(handle)
                    self._device_name = name.decode() if isinstance(name, bytes) else name
            except Exception as exc:
                logger.debug("NVML snapshot error: %s", exc)

        return SystemSnapshot(
            cpu_percent=cpu,
            ram_used_bytes=mem.used,
            ram_total_bytes=mem.total,
            gpu_percent=gpu_pct,
            gpu_mem_used_bytes=gpu_mem_used,
            gpu_mem_total_bytes=gpu_mem_total,
            unified_mem_used_bytes=None,
            unified_mem_total_bytes=None,
            mlx_active_bytes=None,
            backend_label=self._device_name,
            memory_architecture="Discrete CPU/GPU",
        )

    def close(self) -> None:
        global _nvml_ok
        if _nvml_ok:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
        _nvml_ok = None

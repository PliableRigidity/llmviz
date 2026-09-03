"""GPU telemetry with graceful fallback.

Tries pynvml (NVIDIA) first, then falls back to no GPU data.
AMD and Apple Silicon stubs are provided for future implementation.
"""

from __future__ import annotations

import logging

from llmvis.core.events import GpuStatsEvent

logger = logging.getLogger(__name__)

_nvml_available = False
_nvml_initialized = False


def _try_init_nvml() -> bool:
    global _nvml_available, _nvml_initialized
    if _nvml_initialized:
        return _nvml_available
    _nvml_initialized = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_available = True
        logger.info("pynvml initialized — NVIDIA GPU telemetry available")
    except ImportError:
        logger.info("pynvml not installed — install llmvis[nvidia] for GPU telemetry")
    except Exception as exc:
        logger.info("pynvml init failed: %s — GPU telemetry unavailable", exc)
    return _nvml_available


async def get_gpu_stats() -> list[GpuStatsEvent]:
    """Return GPU stats. Returns empty list if GPU telemetry unavailable."""
    if not _try_init_nvml():
        return []

    try:
        import pynvml
        count = pynvml.nvmlDeviceGetCount()
        events = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                ))
            except Exception:
                temp = None

            vram_total = mem.total
            vram_used = mem.used
            vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0.0

            events.append(GpuStatsEvent(
                source="gpu_telemetry",
                gpu_index=i,
                gpu_name=name,
                gpu_util_percent=float(util.gpu),
                vram_used_bytes=vram_used,
                vram_total_bytes=vram_total,
                vram_percent=vram_pct,
                temperature_c=temp,
            ))
        return events
    except Exception as exc:
        logger.debug("GPU stats fetch error: %s", exc)
        return []


def gpu_telemetry_available() -> bool:
    return _try_init_nvml()


def gpu_backend_name() -> str:
    if _try_init_nvml():
        return "NVIDIA (pynvml)"
    return "None"

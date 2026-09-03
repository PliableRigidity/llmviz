"""CPU-only telemetry provider — universal fallback using psutil."""

from __future__ import annotations

import psutil

from llmvis.telemetry.providers.base import BaseTelemetryProvider
from llmvis.telemetry.snapshot import SystemSnapshot


class SystemTelemetryProvider(BaseTelemetryProvider):
    """CPU + RAM only. No GPU telemetry. Works everywhere psutil works."""

    def __init__(self, backend_label: str = "CPU") -> None:
        self._backend_label = backend_label

    def snapshot(self) -> SystemSnapshot:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        return SystemSnapshot(
            cpu_percent=cpu,
            ram_used_bytes=mem.used,
            ram_total_bytes=mem.total,
            gpu_percent=None,
            gpu_mem_used_bytes=None,
            gpu_mem_total_bytes=None,
            unified_mem_used_bytes=None,
            unified_mem_total_bytes=None,
            mlx_active_bytes=None,
            backend_label=self._backend_label,
            memory_architecture="CPU-only",
        )

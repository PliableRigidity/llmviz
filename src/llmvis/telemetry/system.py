"""System telemetry via psutil."""

from __future__ import annotations

import logging

import psutil

from llmvis.core.events import SystemStatsEvent
from llmvis.telemetry.base import BaseTelemetry, GpuStatsEvent

logger = logging.getLogger(__name__)


def _find_ollama_proc() -> psutil.Process | None:
    """Find the main ollama server process."""
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline).lower()
            if "ollama" in name and "serve" in cmdline_str:
                return proc
            # On Windows, the process might just be named ollama with no subcommand
            if name.startswith("ollama") and "app" not in name:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class SystemTelemetry(BaseTelemetry):
    def __init__(self) -> None:
        self._ollama_proc: psutil.Process | None = None
        self._proc_refresh_counter = 0

    async def get_system_stats(self) -> SystemStatsEvent:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()

        ollama_cpu = 0.0
        ollama_ram = 0

        # Refresh process handle periodically
        self._proc_refresh_counter += 1
        if self._proc_refresh_counter >= 10 or self._ollama_proc is None:
            self._ollama_proc = _find_ollama_proc()
            self._proc_refresh_counter = 0

        if self._ollama_proc is not None:
            try:
                ollama_cpu = self._ollama_proc.cpu_percent(interval=None)
                ollama_ram = self._ollama_proc.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._ollama_proc = None

        return SystemStatsEvent(
            source="system_telemetry",
            cpu_percent=cpu,
            ram_used_bytes=mem.used,
            ram_total_bytes=mem.total,
            ram_percent=mem.percent,
            ollama_cpu_percent=ollama_cpu,
            ollama_ram_bytes=ollama_ram,
        )

    async def get_gpu_stats(self) -> list[GpuStatsEvent]:
        # Implemented in gpu.py
        return []

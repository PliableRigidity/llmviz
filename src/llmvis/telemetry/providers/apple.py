"""Apple Silicon telemetry provider.

Key architectural facts:
- Apple Silicon uses UNIFIED MEMORY — there is no separate VRAM pool.
- GPU utilisation via `powermetrics` requires sudo; we do NOT use it.
- MLX exposes `mlx.core.metal.get_active_memory()` for active Metal allocation
  within the unified pool. We surface this as `mlx_active_bytes`, clearly
  labelled as an accounting figure, NOT a separate physical pool.
- If MLX is not installed the provider still works; mlx_active_bytes is None.
"""

from __future__ import annotations

import logging
import subprocess

import psutil

from llmvis.telemetry.providers.base import BaseTelemetryProvider
from llmvis.telemetry.snapshot import SystemSnapshot

logger = logging.getLogger(__name__)


def _get_chip_name() -> str:
    """Read the Apple chip model from sysctl with a short timeout."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        name = result.stdout.strip()
        if name:
            return name
    except Exception:
        pass
    return "Apple Silicon"


def _mlx_active_bytes() -> int | None:
    """Return the current MLX Metal active allocation in bytes, or None."""
    try:
        import mlx.core as mx  # type: ignore[import]
        return int(mx.metal.get_active_memory())
    except (ImportError, AttributeError, Exception):
        return None


class AppleTelemetryProvider(BaseTelemetryProvider):
    """Telemetry for Apple Silicon (M1/M2/M3/M4) systems.

    Reports unified memory rather than separate RAM + VRAM figures.
    GPU utilisation is intentionally omitted (powermetrics requires sudo).
    """

    def __init__(self, chip_name: str = "") -> None:
        self._chip_name = chip_name or _get_chip_name()

    def snapshot(self) -> SystemSnapshot:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        mlx_bytes = _mlx_active_bytes()

        return SystemSnapshot(
            cpu_percent=cpu,
            ram_used_bytes=mem.used,
            ram_total_bytes=mem.total,
            # No discrete GPU on Apple Silicon
            gpu_percent=None,
            gpu_mem_used_bytes=None,
            gpu_mem_total_bytes=None,
            # Unified memory is the same pool as RAM
            unified_mem_used_bytes=mem.used,
            unified_mem_total_bytes=mem.total,
            mlx_active_bytes=mlx_bytes,
            backend_label=f"{self._chip_name} / Metal",
            memory_architecture="Unified Memory",
        )

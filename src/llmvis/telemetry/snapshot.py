"""SystemSnapshot — unified hardware telemetry dataclass.

All telemetry providers return a SystemSnapshot so the TUI does not need
platform-specific code paths.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SystemSnapshot:
    """Current hardware telemetry, normalised across all platforms."""

    cpu_percent: float
    ram_used_bytes: int
    ram_total_bytes: int

    # NVIDIA discrete GPU (None on CPU-only or Apple Silicon)
    gpu_percent: float | None
    gpu_mem_used_bytes: int | None
    gpu_mem_total_bytes: int | None

    # Apple Silicon unified memory (None on discrete-GPU or CPU-only systems)
    unified_mem_used_bytes: int | None
    unified_mem_total_bytes: int | None

    # Optional MLX Metal allocation (within unified memory, not a separate pool)
    mlx_active_bytes: int | None

    # Human-readable labels for the TUI
    backend_label: str          # e.g. "NVIDIA RTX 4090", "Apple M3 Pro / Metal", "CPU"
    memory_architecture: str    # "Discrete CPU/GPU", "Unified Memory", "CPU-only"

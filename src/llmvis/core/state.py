"""Application state for LLMVis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from llmvis.core.events import ModelInfo


class OllamaStatus(Enum):
    UNKNOWN = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    DISCONNECTED = auto()
    RECONNECTING = auto()


class InferenceStatus(Enum):
    IDLE = auto()
    POSSIBLY_ACTIVE = auto()   # estimated from resource usage


@dataclass
class SystemStats:
    cpu_percent: float = 0.0
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0
    ram_percent: float = 0.0
    ollama_cpu_percent: float = 0.0
    ollama_ram_bytes: int = 0


@dataclass
class GpuStats:
    available: bool = False
    gpu_index: int = 0
    gpu_name: str = ""
    gpu_util_percent: float = 0.0
    vram_used_bytes: int = 0
    vram_total_bytes: int = 0
    vram_percent: float = 0.0
    temperature_c: float | None = None


@dataclass
class AppState:
    """Central application state. Mutated by the event loop, read by TUI."""

    # Connection
    ollama_status: OllamaStatus = OllamaStatus.UNKNOWN
    ollama_host: str = "http://localhost:11434"
    ollama_version: str = ""
    connection_error: str = ""

    # Loaded models (from /api/ps)
    loaded_models: list[ModelInfo] = field(default_factory=list)

    # Installed models (from /api/tags)
    installed_models: list[ModelInfo] = field(default_factory=list)

    # Which loaded model appears most active (heuristic)
    active_model: ModelInfo | None = None

    # Inference state
    inference_status: InferenceStatus = InferenceStatus.IDLE
    inference_note: str = ""   # explains what we can/cannot observe

    # Resource history for spike detection (ring buffer, last N samples)
    cpu_history: list[float] = field(default_factory=list)
    gpu_history: list[float] = field(default_factory=list)

    # Current stats
    system: SystemStats = field(default_factory=SystemStats)
    gpu: GpuStats = field(default_factory=GpuStats)

    # Uptime / session
    session_start: float = 0.0


def make_initial_state(host: str = "http://localhost:11434") -> AppState:
    import time
    return AppState(
        ollama_host=host,
        session_start=time.monotonic(),
        inference_note=(
            "Stock Ollama does not expose another client's token stream. "
            "Activity is estimated from system resource usage."
        ),
    )

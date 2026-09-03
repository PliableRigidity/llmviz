"""Normalized event types for LLMVis.

V1 events are emitted by the Ollama adapter using publicly available data.
Future events (marked FUTURE) are defined here so adapters can emit them
when deeper instrumentation (llama.cpp hooks, PyTorch forward hooks, etc.)
becomes available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class EventType(Enum):
    # Connection
    OLLAMA_CONNECTED = auto()
    OLLAMA_DISCONNECTED = auto()
    OLLAMA_RECONNECTING = auto()

    # Model lifecycle
    MODEL_DETECTED = auto()
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()

    # Inference lifecycle (observable via stock Ollama)
    INFERENCE_POSSIBLY_STARTED = auto()   # estimated from resource spike
    INFERENCE_POSSIBLY_ENDED = auto()     # estimated

    # Deep inference events (FUTURE — require instrumented backend)
    PREFILL_START = auto()
    PREFILL_END = auto()
    DECODE_START = auto()
    TOKEN_GENERATED = auto()
    KV_CACHE_UPDATE = auto()
    LOGITS_READY = auto()
    TOKEN_SAMPLED = auto()
    LAYER_ENTER = auto()
    LAYER_EXIT = auto()
    ACTIVATION_STATS = auto()

    # Agent events (FUTURE)
    AGENT_STEP = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    MEMORY_LOOKUP = auto()
    MEMORY_RESULT = auto()
    RAG_QUERY = auto()
    RAG_RESULT = auto()
    CONTEXT_UPDATE = auto()

    # Telemetry
    SYSTEM_STATS_UPDATE = auto()
    GPU_STATS_UPDATE = auto()


@dataclass
class LLMVisEvent:
    """Base event. All events carry a monotonic timestamp."""
    type: EventType
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "unknown"


@dataclass
class OllamaConnectedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.OLLAMA_CONNECTED, init=False)
    version: str = ""
    host: str = ""


@dataclass
class OllamaDisconnectedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.OLLAMA_DISCONNECTED, init=False)
    reason: str = ""


@dataclass
class OllamaReconnectingEvent(LLMVisEvent):
    type: EventType = field(default=EventType.OLLAMA_RECONNECTING, init=False)
    attempt: int = 0


@dataclass
class ModelInfo:
    """Normalized model metadata from Ollama /api/tags and /api/ps."""
    name: str
    family: str = ""
    families: list[str] = field(default_factory=list)
    parameter_size: str = ""       # e.g. "3.1B"
    quantization_level: str = ""   # e.g. "Q4_K_M"
    format: str = ""               # e.g. "gguf"
    size_bytes: int = 0            # on-disk size
    size_vram_bytes: int = 0       # VRAM usage when loaded (0 if not loaded)
    context_length: int | None = None
    embedding_length: int | None = None
    capabilities: list[str] = field(default_factory=list)
    digest: str = ""
    expires_at: str | None = None   # ISO8601 when model auto-unloads


@dataclass
class ModelDetectedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.MODEL_DETECTED, init=False)
    model: ModelInfo = field(default_factory=lambda: ModelInfo(name=""))


@dataclass
class ModelLoadedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.MODEL_LOADED, init=False)
    model: ModelInfo = field(default_factory=lambda: ModelInfo(name=""))


@dataclass
class ModelUnloadedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.MODEL_UNLOADED, init=False)
    model_name: str = ""


@dataclass
class InferencePossiblyStartedEvent(LLMVisEvent):
    """Emitted when resource usage suggests inference has begun.

    This is an ESTIMATE based on CPU/GPU activity change.
    Stock Ollama does not expose another client's inference state.
    """
    type: EventType = field(default=EventType.INFERENCE_POSSIBLY_STARTED, init=False)
    confidence: str = "estimated"   # always "estimated" for stock Ollama


@dataclass
class InferencePossiblyEndedEvent(LLMVisEvent):
    """Emitted when resource usage returns to baseline after a spike."""
    type: EventType = field(default=EventType.INFERENCE_POSSIBLY_ENDED, init=False)
    confidence: str = "estimated"


@dataclass
class SystemStatsEvent(LLMVisEvent):
    type: EventType = field(default=EventType.SYSTEM_STATS_UPDATE, init=False)
    cpu_percent: float = 0.0
    ram_used_bytes: int = 0
    ram_total_bytes: int = 0
    ram_percent: float = 0.0
    ollama_cpu_percent: float = 0.0
    ollama_ram_bytes: int = 0


@dataclass
class GpuStatsEvent(LLMVisEvent):
    type: EventType = field(default=EventType.GPU_STATS_UPDATE, init=False)
    gpu_index: int = 0
    gpu_name: str = ""
    gpu_util_percent: float = 0.0
    vram_used_bytes: int = 0
    vram_total_bytes: int = 0
    vram_percent: float = 0.0
    temperature_c: float | None = None


# ── Future / placeholder types (not emitted in V1) ──────────────────────────

@dataclass
class TokenGeneratedEvent(LLMVisEvent):
    """FUTURE: Requires instrumented backend (e.g. llama.cpp hooks).
    Stock Ollama does not expose per-token events from another client's session.
    """
    type: EventType = field(default=EventType.TOKEN_GENERATED, init=False)
    token_id: int = 0
    token_text: str = ""
    logprob: float | None = None


@dataclass
class LayerEvent(LLMVisEvent):
    """FUTURE: Requires forward-hook instrumentation."""
    type: EventType = field(default=EventType.LAYER_ENTER, init=False)
    layer_index: int = 0
    layer_type: str = ""   # "attention", "mlp", etc.


@dataclass
class ActivationStatsEvent(LLMVisEvent):
    """FUTURE: Per-layer activation statistics from forward hooks."""
    type: EventType = field(default=EventType.ACTIVATION_STATS, init=False)
    layer_index: int = 0
    mean: float = 0.0
    std: float = 0.0
    max_abs: float = 0.0

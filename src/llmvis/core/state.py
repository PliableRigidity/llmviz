"""Application state for LLMVis."""

from __future__ import annotations

from collections import deque
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


class InferencePhase(Enum):
    IDLE = auto()
    LOADING = auto()
    TOKENIZING = auto()
    PREFILLING = auto()
    DECODING = auto()
    COMPLETE = auto()


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
class LayerSummary:
    """Summary statistics for one transformer layer during one token generation. All values computed from actual tensors."""
    layer_index: int
    hidden_state_rms: float
    hidden_state_mean: float
    hidden_state_std: float
    delta_from_prev: float
    exec_time_ms: float
    mlp_output_rms: float | None = None
    mlp_sparsity: float | None = None


@dataclass
class TokenRecord:
    """Complete record of one generated token. Stored in bounded ring buffer."""
    index: int
    text: str
    token_id: int
    latency_ms: float
    layer_summaries: list  # list[LayerSummary]
    top_logits: list       # list[tuple[str, float]] = (token_text, probability)
    kv_seq_len: int
    kv_bytes: int
    temperature: float = 1.0
    top_k_param: int = 0
    top_p_param: float = 1.0


@dataclass
class ModelArchInfo:
    """Discovered model architecture — populated from model.config at load time."""
    model_id: str = ""
    num_layers: int = 0
    num_attention_heads: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    head_dim: int = 0
    dtype: str = ""
    device: str = ""


@dataclass
class DeepState:
    """State for deep instrumentation mode. Added to AppState when using InstrumentedTransformersAdapter."""
    arch: ModelArchInfo = field(default_factory=ModelArchInfo)
    phase: InferencePhase = InferencePhase.IDLE

    prefill_token_count: int = 0
    prefill_duration_ms: float = 0.0
    decode_token_count: int = 0
    decode_tokens_per_sec: float = 0.0

    current_layer: int = -1           # layer currently executing (-1 = none)
    current_token_index: int = 0
    current_layer_summaries: list = field(default_factory=list)  # list[LayerSummary] for current token
    current_kv_seq_len: int = 0
    current_kv_bytes: int = 0
    current_top_logits: list = field(default_factory=list)  # list[tuple[str, float]]

    token_history: object = field(default_factory=lambda: deque(maxlen=100))  # deque[TokenRecord]
    selected_token_idx: int | None = None  # for keyboard navigation

    prompt_text: str = ""
    response_text: str = ""

    total_inference_time_ms: float = 0.0
    telemetry_mode: str = "standard"

    is_generating: bool = False
    model_loaded: bool = False
    load_error: str = ""

    # Pause state: when paused the TUI stops refreshing but inference continues
    visualization_paused: bool = False

    # KV cache tensor shapes from the most recent KvCacheUpdateEvent
    last_k_shape: list = field(default_factory=list)   # e.g. [1, num_heads, seq, head_dim]
    last_v_shape: list = field(default_factory=list)
    last_kv_dtype: str = ""


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

    # Deep instrumentation mode state (None when not in deep mode)
    deep: DeepState | None = field(default=None)


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


def make_deep_state() -> DeepState:
    return DeepState()

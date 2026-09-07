"""Normalized event types for LLMVis.

V1 events are emitted by the Ollama adapter using publicly available data.
V2 events are emitted by InstrumentedTransformersAdapter (PyTorch + HuggingFace)
and carry real measured data from forward hooks, logits, and KV cache tensors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class EventType(Enum):
    # ── Connection ──────────────────────────────────────────────────────────
    OLLAMA_CONNECTED = auto()
    OLLAMA_DISCONNECTED = auto()
    OLLAMA_RECONNECTING = auto()

    # ── Model lifecycle ──────────────────────────────────────────────────────
    MODEL_DETECTED = auto()
    MODEL_LOADED = auto()
    MODEL_UNLOADED = auto()

    # ── Inference lifecycle (observable via stock Ollama) ────────────────────
    INFERENCE_POSSIBLY_STARTED = auto()   # estimated from resource spike
    INFERENCE_POSSIBLY_ENDED = auto()     # estimated

    # ── Deep inference lifecycle (V2 — InstrumentedTransformersAdapter) ──────
    INFERENCE_START = auto()              # generation session started (exact)
    INFERENCE_END = auto()               # generation session ended (exact)
    PROMPT_RECEIVED = auto()             # prompt text received by adapter
    TOKENIZATION_COMPLETE = auto()       # tokenizer finished, token count known
    PREFILL_START = auto()               # prefill phase beginning
    PREFILL_END = auto()                 # prefill phase done, timing known
    DECODE_START = auto()                # decode loop starting
    TOKEN_START = auto()                 # starting to generate one token
    TOKEN_END = auto()                   # one token fully generated, latency known
    TOKEN_GENERATED = auto()             # token text available (compat)

    # ── Layer-level events (V2) ──────────────────────────────────────────────
    LAYER_ENTER = auto()                 # kept for legacy compat
    LAYER_STATS = auto()                 # layer statistics from forward hook
    LAYER_EXIT = auto()                  # kept for legacy compat
    MLP_STATS = auto()                   # MLP sub-layer statistics
    ATTENTION_STATS = auto()             # attention sub-layer statistics

    # ── Logits and sampling (V2) ─────────────────────────────────────────────
    LOGITS_READY = auto()               # top-N logit candidates before sampling
    TOKEN_SAMPLED = auto()              # sampling complete, token chosen

    # ── KV cache (V2) ────────────────────────────────────────────────────────
    KV_CACHE_UPDATE = auto()            # KV cache updated after a token

    # ── Model architecture (V2) ──────────────────────────────────────────────
    MODEL_ARCHITECTURE = auto()         # emitted once when model is loaded

    # ── Activation stats (legacy compat) ────────────────────────────────────
    ACTIVATION_STATS = auto()

    # ── Agent events (FUTURE) ────────────────────────────────────────────────
    AGENT_STEP = auto()
    TOOL_CALL = auto()
    TOOL_RESULT = auto()
    MEMORY_LOOKUP = auto()
    MEMORY_RESULT = auto()
    RAG_QUERY = auto()
    RAG_RESULT = auto()
    CONTEXT_UPDATE = auto()

    # ── Telemetry ────────────────────────────────────────────────────────────
    SYSTEM_STATS_UPDATE = auto()
    GPU_STATS_UPDATE = auto()

    # ── IPC (instrument/run two-terminal model) ───────────────────────────────
    SESSION_INFO = auto()          # server sends on client connect
    BACKEND_CONNECTED = auto()     # client emits when TCP connection established
    BACKEND_DISCONNECTED = auto()  # client emits when connection lost


@dataclass
class LLMVisEvent:
    """Base event. All events carry a monotonic timestamp and optional sequence info."""
    type: EventType
    timestamp: float = field(default_factory=time.monotonic)
    source: str = "unknown"
    session_id: str = ""        # set by adapter; empty means not tracked
    sequence_num: int = 0       # monotonically increasing within a session; 0 = unset


# ── V1: Connection events ────────────────────────────────────────────────────

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


# ── V1: Model metadata ───────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """Normalized model metadata from Ollama /api/tags and /api/ps."""
    name: str
    family: str = ""
    families: list[str] = field(default_factory=list)
    parameter_size: str = ""
    quantization_level: str = ""
    format: str = ""
    size_bytes: int = 0
    size_vram_bytes: int = 0
    context_length: int | None = None
    embedding_length: int | None = None
    capabilities: list[str] = field(default_factory=list)
    digest: str = ""
    expires_at: str | None = None


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


# ── V1: Inference estimation events ─────────────────────────────────────────

@dataclass
class InferencePossiblyStartedEvent(LLMVisEvent):
    """Estimated from resource spike. Stock Ollama does not expose real inference state."""
    type: EventType = field(default=EventType.INFERENCE_POSSIBLY_STARTED, init=False)
    confidence: str = "estimated"


@dataclass
class InferencePossiblyEndedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.INFERENCE_POSSIBLY_ENDED, init=False)
    confidence: str = "estimated"


# ── V1: Telemetry events ─────────────────────────────────────────────────────

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


# ── V2: Shared value types ───────────────────────────────────────────────────

@dataclass
class TokenCandidate:
    """One vocabulary candidate before sampling. All values from actual model logits."""
    token_id: int
    token_text: str
    logit: float              # raw logit from LM head
    raw_probability: float    # softmax BEFORE temperature/filtering
    probability: float        # softmax AFTER temperature + top-k/p filtering


@dataclass
class SamplerConfig:
    """Sampling parameters in effect during a generation session."""
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    seed: int | None = None


# ── V2: Model architecture ───────────────────────────────────────────────────

@dataclass
class ModelArchitectureEvent(LLMVisEvent):
    """Emitted once when a model is loaded. Fields read from model.config."""
    type: EventType = field(default=EventType.MODEL_ARCHITECTURE, init=False)
    model_id: str = ""
    num_layers: int = 0
    num_attention_heads: int = 0
    hidden_size: int = 0
    vocab_size: int = 0
    head_dim: int = 0
    dtype: str = ""
    device: str = ""


# ── V2: Inference lifecycle ──────────────────────────────────────────────────

@dataclass
class InferenceStartEvent(LLMVisEvent):
    type: EventType = field(default=EventType.INFERENCE_START, init=False)
    model_id: str = ""
    prompt_preview: str = ""


@dataclass
class InferenceEndEvent(LLMVisEvent):
    type: EventType = field(default=EventType.INFERENCE_END, init=False)
    model_id: str = ""
    total_tokens: int = 0
    total_time_ms: float = 0.0


@dataclass
class PromptReceivedEvent(LLMVisEvent):
    type: EventType = field(default=EventType.PROMPT_RECEIVED, init=False)
    text: str = ""


@dataclass
class TokenizationCompleteEvent(LLMVisEvent):
    """token_count is measured from actual tokenizer output."""
    type: EventType = field(default=EventType.TOKENIZATION_COMPLETE, init=False)
    token_count: int = 0
    token_ids: list[int] = field(default_factory=list)
    duration_ms: float = 0.0


@dataclass
class PrefillStartEvent(LLMVisEvent):
    type: EventType = field(default=EventType.PREFILL_START, init=False)
    token_count: int = 0


@dataclass
class PrefillEndEvent(LLMVisEvent):
    """duration_ms is measured wall-clock time of the prefill forward pass."""
    type: EventType = field(default=EventType.PREFILL_END, init=False)
    token_count: int = 0
    duration_ms: float = 0.0


@dataclass
class DecodeStartEvent(LLMVisEvent):
    type: EventType = field(default=EventType.DECODE_START, init=False)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass
class TokenStartEvent(LLMVisEvent):
    type: EventType = field(default=EventType.TOKEN_START, init=False)
    token_index: int = 0


# ── V2: Layer-level events ───────────────────────────────────────────────────

@dataclass
class LayerStatsEvent(LLMVisEvent):
    """Per-layer statistics collected from a registered forward hook.

    hidden_state_rms, mean, std: computed from the layer's OUTPUT tensor.
    delta_from_prev: RMS of (output_hidden - input_hidden), the residual change.
    exec_time_ms: wall-clock time for this layer's forward pass.
    All values are measured from actual tensors, not inferred or estimated.
    """
    type: EventType = field(default=EventType.LAYER_STATS, init=False)
    token_index: int = 0
    layer_index: int = 0
    hidden_state_rms: float = 0.0
    hidden_state_mean: float = 0.0
    hidden_state_std: float = 0.0
    delta_from_prev: float = 0.0
    exec_time_ms: float = 0.0


@dataclass
class MLPStatsEvent(LLMVisEvent):
    """MLP sub-layer output statistics. Only emitted in 'deep' telemetry mode."""
    type: EventType = field(default=EventType.MLP_STATS, init=False)
    token_index: int = 0
    layer_index: int = 0
    output_rms: float = 0.0
    sparsity: float = 0.0   # fraction of |x| < 0.01


@dataclass
class AttentionStatsEvent(LLMVisEvent):
    """Attention weight statistics. Only emitted with --attention flag."""
    type: EventType = field(default=EventType.ATTENTION_STATS, init=False)
    token_index: int = 0
    layer_index: int = 0
    per_head_entropy: list[float] = field(default_factory=list)
    max_weight: float = 0.0
    mean_weight: float = 0.0


# ── V2: Logit and sampling events ────────────────────────────────────────────

@dataclass
class LogitsReadyEvent(LLMVisEvent):
    """Top-N candidates from the LM head before sampling.

    top_candidates sorted by probability descending.
    All logits and probabilities are from the actual model output tensor.
    """
    type: EventType = field(default=EventType.LOGITS_READY, init=False)
    token_index: int = 0
    top_candidates: list[TokenCandidate] = field(default_factory=list)
    vocab_size: int = 0


@dataclass
class TokenSampledEvent(LLMVisEvent):
    type: EventType = field(default=EventType.TOKEN_SAMPLED, init=False)
    token_index: int = 0
    token_id: int = 0
    token_text: str = ""
    logprob: float = 0.0
    sampler: SamplerConfig = field(default_factory=SamplerConfig)


# ── V2: KV cache event ───────────────────────────────────────────────────────

@dataclass
class KvCacheUpdateEvent(LLMVisEvent):
    """KV cache state after one token is appended.

    measured_bytes = actual sum of tensor.nbytes for all K and V tensors.
    k_shape / v_shape are read from the actual tensors.
    """
    type: EventType = field(default=EventType.KV_CACHE_UPDATE, init=False)
    token_index: int = 0
    seq_len_before: int = 0
    seq_len_after: int = 0
    num_layers: int = 0
    k_shape: list[int] = field(default_factory=list)
    v_shape: list[int] = field(default_factory=list)
    dtype: str = ""
    measured_bytes: int = 0


# ── V2: Token end event ──────────────────────────────────────────────────────

@dataclass
class TokenEndEvent(LLMVisEvent):
    """Emitted when one token's full generation cycle is complete."""
    type: EventType = field(default=EventType.TOKEN_END, init=False)
    token_index: int = 0
    token_text: str = ""
    token_id: int = 0
    latency_ms: float = 0.0   # measured wall-clock from TokenStart to now


# ── V1+V2 compat: TokenGeneratedEvent ────────────────────────────────────────

@dataclass
class TokenGeneratedEvent(LLMVisEvent):
    """Token generated. V1: future placeholder. V2: real data from adapter."""
    type: EventType = field(default=EventType.TOKEN_GENERATED, init=False)
    token_id: int = 0
    token_text: str = ""
    logprob: float | None = None
    latency_ms: float = 0.0


# ── Legacy: Layer and activation events (kept for test compatibility) ─────────

@dataclass
class LayerEvent(LLMVisEvent):
    """Legacy. Use LayerStatsEvent for V2 instrumentation."""
    type: EventType = field(default=EventType.LAYER_ENTER, init=False)
    layer_index: int = 0
    layer_type: str = ""


@dataclass
class ActivationStatsEvent(LLMVisEvent):
    """Legacy. Use LayerStatsEvent for V2 instrumentation."""
    type: EventType = field(default=EventType.ACTIVATION_STATS, init=False)
    layer_index: int = 0
    mean: float = 0.0
    std: float = 0.0
    max_abs: float = 0.0


# ── IPC events (instrument/run two-terminal model) ───────────────────────────

@dataclass
class SessionInfoEvent(LLMVisEvent):
    """Sent by TelemetryServer to each client immediately on connect."""
    type: EventType = field(default=EventType.SESSION_INFO, init=False)
    server_session_id: str = ""
    model_id: str = ""
    backend: str = ""
    telemetry_mode: str = ""
    platform_name: str = ""
    port: int = 0


@dataclass
class BackendConnectedEvent(LLMVisEvent):
    """Emitted by ClientAdapter when a TCP connection to the server is established."""
    type: EventType = field(default=EventType.BACKEND_CONNECTED, init=False)
    host: str = "localhost"
    port: int = 0


@dataclass
class BackendDisconnectedEvent(LLMVisEvent):
    """Emitted by ClientAdapter when the TCP connection to the server is lost."""
    type: EventType = field(default=EventType.BACKEND_DISCONNECTED, init=False)
    host: str = "localhost"
    port: int = 0
    reason: str = ""


# ── Serialization helpers (for record/replay) ─────────────────────────────────

import dataclasses
import json


_EVENT_CLASSES: dict[str, type] = {
    "OLLAMA_CONNECTED": OllamaConnectedEvent,
    "OLLAMA_DISCONNECTED": OllamaDisconnectedEvent,
    "OLLAMA_RECONNECTING": OllamaReconnectingEvent,
    "MODEL_ARCHITECTURE": ModelArchitectureEvent,
    "INFERENCE_START": InferenceStartEvent,
    "INFERENCE_END": InferenceEndEvent,
    "PROMPT_RECEIVED": PromptReceivedEvent,
    "TOKENIZATION_COMPLETE": TokenizationCompleteEvent,
    "PREFILL_START": PrefillStartEvent,
    "PREFILL_END": PrefillEndEvent,
    "DECODE_START": DecodeStartEvent,
    "TOKEN_START": TokenStartEvent,
    "LAYER_STATS": LayerStatsEvent,
    "MLP_STATS": MLPStatsEvent,
    "ATTENTION_STATS": AttentionStatsEvent,
    "LOGITS_READY": LogitsReadyEvent,
    "TOKEN_SAMPLED": TokenSampledEvent,
    "KV_CACHE_UPDATE": KvCacheUpdateEvent,
    "TOKEN_END": TokenEndEvent,
    "TOKEN_GENERATED": TokenGeneratedEvent,
    "INFERENCE_POSSIBLY_STARTED": InferencePossiblyStartedEvent,
    "INFERENCE_POSSIBLY_ENDED": InferencePossiblyEndedEvent,
    "SYSTEM_STATS_UPDATE": SystemStatsEvent,
    "GPU_STATS_UPDATE": GpuStatsEvent,
    "SESSION_INFO": SessionInfoEvent,
    "BACKEND_CONNECTED": BackendConnectedEvent,
    "BACKEND_DISCONNECTED": BackendDisconnectedEvent,
}


def event_to_dict(evt: LLMVisEvent) -> dict:
    """Serialize an event to a JSON-compatible dict. Does not include raw tensors."""
    d = dataclasses.asdict(evt)
    d["_type"] = evt.type.name
    # Remove the Enum field — _type is the canonical serialization for round-trips
    d.pop("type", None)
    return d


def _json_default(obj: object) -> object:
    """Fallback serializer for json.dumps — handles Enum values."""
    if isinstance(obj, Enum):
        return obj.name
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def event_to_jsonl(evt: LLMVisEvent) -> str:
    """Serialize an event to a single JSONL line."""
    return json.dumps(event_to_dict(evt), ensure_ascii=False, default=_json_default)


def event_from_dict(d: dict) -> LLMVisEvent | None:
    """Deserialize an event from a dict produced by event_to_dict().

    Returns None for unknown event types (forward compat).
    """
    type_name = d.get("_type", "")
    cls = _EVENT_CLASSES.get(type_name)
    if cls is None:
        return None

    # Remove meta keys that aren't constructor params
    d = dict(d)
    d.pop("_type", None)
    d.pop("type", None)  # managed by dataclass field default

    # Reconstruct nested objects
    if type_name == "LOGITS_READY" and "top_candidates" in d:
        d["top_candidates"] = [
            TokenCandidate(**tc) for tc in d["top_candidates"]
        ]
    if type_name in ("TOKEN_SAMPLED", "DECODE_START") and "sampler" in d and isinstance(d["sampler"], dict):
        d["sampler"] = SamplerConfig(**d["sampler"])

    try:
        return cls(**d)
    except Exception:
        return None


def event_from_jsonl(line: str) -> LLMVisEvent | None:
    """Deserialize an event from a JSONL line. Returns None on parse error."""
    try:
        return event_from_dict(json.loads(line))
    except Exception:
        return None

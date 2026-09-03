"""Tests for event dataclasses."""

from __future__ import annotations

from llmvis.core.events import (
    EventType,
    OllamaConnectedEvent,
    OllamaDisconnectedEvent,
    ModelInfo,
    ModelLoadedEvent,
    ModelUnloadedEvent,
    SystemStatsEvent,
    GpuStatsEvent,
    InferencePossiblyStartedEvent,
    TokenGeneratedEvent,
)


def test_ollama_connected_event():
    evt = OllamaConnectedEvent(source="test", version="0.33.2", host="http://localhost:11434")
    assert evt.type == EventType.OLLAMA_CONNECTED
    assert evt.version == "0.33.2"
    assert evt.host == "http://localhost:11434"
    assert evt.timestamp > 0


def test_ollama_disconnected_event():
    evt = OllamaDisconnectedEvent(source="test", reason="connection refused")
    assert evt.type == EventType.OLLAMA_DISCONNECTED
    assert evt.reason == "connection refused"


def test_model_info_defaults():
    m = ModelInfo(name="qwen2.5:3b")
    assert m.name == "qwen2.5:3b"
    assert m.family == ""
    assert m.size_vram_bytes == 0
    assert m.context_length is None


def test_model_loaded_event():
    info = ModelInfo(
        name="qwen2.5:3b",
        family="qwen2",
        parameter_size="3.1B",
        quantization_level="Q4_K_M",
        context_length=4096,
    )
    evt = ModelLoadedEvent(source="test", model=info)
    assert evt.type == EventType.MODEL_LOADED
    assert evt.model.name == "qwen2.5:3b"
    assert evt.model.context_length == 4096


def test_model_unloaded_event():
    evt = ModelUnloadedEvent(source="test", model_name="qwen2.5:3b")
    assert evt.type == EventType.MODEL_UNLOADED
    assert evt.model_name == "qwen2.5:3b"


def test_system_stats_event():
    evt = SystemStatsEvent(
        source="test",
        cpu_percent=45.5,
        ram_used_bytes=4_000_000_000,
        ram_total_bytes=16_000_000_000,
        ram_percent=25.0,
    )
    assert evt.type == EventType.SYSTEM_STATS_UPDATE
    assert evt.cpu_percent == 45.5


def test_gpu_stats_event():
    evt = GpuStatsEvent(
        source="test",
        gpu_name="RTX 4090",
        gpu_util_percent=72.0,
        vram_used_bytes=8_000_000_000,
        vram_total_bytes=16_000_000_000,
        vram_percent=50.0,
        temperature_c=75.0,
    )
    assert evt.type == EventType.GPU_STATS_UPDATE
    assert evt.gpu_name == "RTX 4090"
    assert evt.temperature_c == 75.0


def test_inference_possibly_started_event():
    evt = InferencePossiblyStartedEvent(source="test")
    assert evt.type == EventType.INFERENCE_POSSIBLY_STARTED
    assert evt.confidence == "estimated"


def test_token_generated_is_future():
    """TokenGeneratedEvent is defined but clearly marked as future/V2."""
    evt = TokenGeneratedEvent(source="future_backend", token_id=1234, token_text="hello")
    assert evt.type == EventType.TOKEN_GENERATED
    assert evt.token_text == "hello"


def test_event_timestamps_monotonic():
    e1 = OllamaConnectedEvent(source="t")
    e2 = OllamaConnectedEvent(source="t")
    assert e2.timestamp >= e1.timestamp

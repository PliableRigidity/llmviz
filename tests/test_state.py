"""Tests for application state."""

from __future__ import annotations

from llmvis.core.state import (
    AppState, OllamaStatus, InferenceStatus, SystemStats, GpuStats,
    make_initial_state,
)
from llmvis.core.events import ModelInfo


def test_initial_state():
    state = make_initial_state()
    assert state.ollama_status == OllamaStatus.UNKNOWN
    assert state.loaded_models == []
    assert state.active_model is None
    assert state.inference_status == InferenceStatus.IDLE
    assert state.session_start > 0


def test_initial_state_custom_host():
    state = make_initial_state("http://192.168.1.10:11434")
    assert state.ollama_host == "http://192.168.1.10:11434"


def test_state_model_assignment():
    state = make_initial_state()
    m = ModelInfo(name="llama3.2:3b", family="llama", parameter_size="3.2B")
    state.active_model = m
    assert state.active_model.name == "llama3.2:3b"


def test_state_transitions():
    state = make_initial_state()
    assert state.ollama_status == OllamaStatus.UNKNOWN

    state.ollama_status = OllamaStatus.CONNECTED
    assert state.ollama_status == OllamaStatus.CONNECTED

    state.ollama_status = OllamaStatus.DISCONNECTED
    assert state.ollama_status == OllamaStatus.DISCONNECTED

    state.ollama_status = OllamaStatus.RECONNECTING
    assert state.ollama_status == OllamaStatus.RECONNECTING


def test_inference_status_transitions():
    state = make_initial_state()
    assert state.inference_status == InferenceStatus.IDLE
    state.inference_status = InferenceStatus.POSSIBLY_ACTIVE
    assert state.inference_status == InferenceStatus.POSSIBLY_ACTIVE
    state.inference_status = InferenceStatus.IDLE
    assert state.inference_status == InferenceStatus.IDLE


def test_gpu_stats_defaults():
    g = GpuStats()
    assert g.available is False
    assert g.gpu_name == ""
    assert g.temperature_c is None


def test_system_stats_defaults():
    s = SystemStats()
    assert s.cpu_percent == 0.0
    assert s.ram_total_bytes == 0

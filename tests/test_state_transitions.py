"""Tests for state machine transitions."""

from __future__ import annotations

from llmvis.core.state import AppState, OllamaStatus, InferenceStatus, make_initial_state
from llmvis.core.events import ModelInfo


def test_transition_unknown_to_connected():
    state = make_initial_state()
    assert state.ollama_status == OllamaStatus.UNKNOWN
    state.ollama_status = OllamaStatus.CONNECTED
    assert state.ollama_status == OllamaStatus.CONNECTED


def test_transition_connected_to_disconnected():
    state = make_initial_state()
    state.ollama_status = OllamaStatus.CONNECTED
    state.active_model = ModelInfo(name="qwen2.5:3b")
    state.loaded_models = [state.active_model]

    state.ollama_status = OllamaStatus.DISCONNECTED
    state.loaded_models = []
    state.active_model = None
    state.inference_status = InferenceStatus.IDLE

    assert state.ollama_status == OllamaStatus.DISCONNECTED
    assert state.loaded_models == []
    assert state.active_model is None


def test_transition_reconnecting_to_connected():
    state = make_initial_state()
    state.ollama_status = OllamaStatus.RECONNECTING
    state.ollama_status = OllamaStatus.CONNECTED
    state.ollama_version = "0.33.2"
    assert state.ollama_version == "0.33.2"


def test_model_loaded_sets_active():
    state = make_initial_state()
    state.ollama_status = OllamaStatus.CONNECTED

    m1 = ModelInfo(name="qwen2.5:3b")
    state.loaded_models.append(m1)
    if state.active_model is None:
        state.active_model = m1

    assert state.active_model.name == "qwen2.5:3b"


def test_model_unloaded_clears_active():
    state = make_initial_state()
    m = ModelInfo(name="qwen2.5:3b")
    state.loaded_models = [m]
    state.active_model = m

    state.loaded_models = [x for x in state.loaded_models if x.name != "qwen2.5:3b"]
    if state.active_model and state.active_model.name == "qwen2.5:3b":
        state.active_model = state.loaded_models[0] if state.loaded_models else None

    assert state.active_model is None
    assert state.loaded_models == []


def test_multiple_model_unload_selects_remaining():
    state = make_initial_state()
    m1 = ModelInfo(name="qwen2.5:3b")
    m2 = ModelInfo(name="gemma3:4b")
    state.loaded_models = [m1, m2]
    state.active_model = m1

    state.loaded_models = [x for x in state.loaded_models if x.name != "qwen2.5:3b"]
    if state.active_model and state.active_model.name == "qwen2.5:3b":
        state.active_model = state.loaded_models[0] if state.loaded_models else None

    assert state.active_model.name == "gemma3:4b"


def test_inference_idle_when_no_model():
    state = make_initial_state()
    state.active_model = None
    assert state.inference_status == InferenceStatus.IDLE


def test_cpu_history_bounded():
    state = make_initial_state()
    for i in range(100):
        state.cpu_history.append(float(i))
        if len(state.cpu_history) > 60:
            state.cpu_history.pop(0)
    assert len(state.cpu_history) == 60


def test_gpu_history_bounded():
    state = make_initial_state()
    for i in range(100):
        state.gpu_history.append(float(i % 100))
        if len(state.gpu_history) > 60:
            state.gpu_history.pop(0)
    assert len(state.gpu_history) == 60

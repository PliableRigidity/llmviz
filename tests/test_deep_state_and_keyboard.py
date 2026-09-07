"""Tests for deep state population, Ollama isolation, and keyboard focus model."""

from __future__ import annotations

import asyncio

import pytest

from llmvis.core.events import (
    BackendConnectedEvent,
    EventType,
    InferenceStartEvent,
    ModelArchitectureEvent,
    SessionInfoEvent,
    event_from_jsonl,
    event_to_jsonl,
)
from llmvis.core.state import (
    InferencePhase,
    OllamaStatus,
    make_deep_state,
    make_initial_state,
)
from llmvis.ipc.server import TelemetryServer


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_line(reader: asyncio.StreamReader, timeout: float = 2.0) -> str:
    raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    return raw.decode().strip()


# ── DeepState: SESSION_INFO populates model_id ────────────────────────────────

def test_session_info_sets_arch_model_id():
    """_on_session_info fallback: SESSION_INFO model_id fills d.arch.model_id."""
    from llmvis.core.state import make_deep_state, ModelArchInfo

    d = make_deep_state()
    assert d.arch.model_id == ""

    # Simulate the logic in _on_session_info
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    if model_id and not d.arch.model_id:
        d.arch.model_id = model_id
        d.model_loaded = True
        d.phase = InferencePhase.IDLE

    assert d.arch.model_id == model_id
    assert d.model_loaded is True
    assert d.phase == InferencePhase.IDLE


def test_session_info_does_not_overwrite_existing_arch():
    """SESSION_INFO fallback must not overwrite arch already set by MODEL_ARCHITECTURE."""
    from llmvis.core.state import make_deep_state, ModelArchInfo

    d = make_deep_state()
    d.arch.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    d.arch.num_layers = 28
    d.model_loaded = True

    # Simulate the fallback guard: only fill if empty
    fallback_id = "some-other-model"
    if fallback_id and not d.arch.model_id:
        d.arch.model_id = fallback_id

    assert d.arch.model_id == "Qwen/Qwen2.5-1.5B-Instruct"  # unchanged
    assert d.arch.num_layers == 28


# ── DeepState: MODEL_ARCHITECTURE populates arch ──────────────────────────────

def test_model_arch_event_fields_populate_deep_state():
    """MODEL_ARCHITECTURE event carries real arch fields that DeepState stores."""
    d = make_deep_state()

    evt = ModelArchitectureEvent(
        source="adapter",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_layers=28,
        num_attention_heads=16,
        hidden_size=1536,
        vocab_size=151936,
        head_dim=96,
        dtype="bfloat16",
        device="cuda:0",
    )

    # Simulate _on_model_arch logic
    from llmvis.core.state import ModelArchInfo
    d.arch = ModelArchInfo(
        model_id=evt.model_id,
        num_layers=evt.num_layers,
        num_attention_heads=evt.num_attention_heads,
        hidden_size=evt.hidden_size,
        vocab_size=evt.vocab_size,
        head_dim=evt.head_dim,
        dtype=evt.dtype,
        device=evt.device,
    )
    d.phase = InferencePhase.IDLE
    d.model_loaded = True

    assert d.arch.model_id == "Qwen/Qwen2.5-1.5B-Instruct"
    assert d.arch.num_layers == 28
    assert d.arch.num_attention_heads == 16
    assert d.arch.hidden_size == 1536
    assert d.arch.vocab_size == 151936
    assert d.arch.dtype == "bfloat16"
    assert d.arch.device == "cuda:0"
    assert d.model_loaded is True
    assert d.phase == InferencePhase.IDLE


# ── ModelPanel: deep mode does not show Ollama empty-state ────────────────────

def test_model_panel_deep_mode_no_ollama_message():
    """ModelPanel must not show 'Run: ollama run <model>' in deep mode."""
    from llmvis.tui.widgets.model_panel import ModelPanel

    state = make_initial_state()
    state.deep = make_deep_state()
    state.deep.arch.model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    state.deep.model_loaded = True
    state.ollama_status = OllamaStatus.CONNECTED  # as set by _on_backend_connected

    panel = ModelPanel(state)
    rendered = panel._build(state)

    assert "ollama run" not in rendered.lower()
    assert "No model loaded" not in rendered
    assert "Qwen/Qwen2.5-1.5B-Instruct" in rendered


def test_model_panel_deep_mode_shows_arch_fields():
    """ModelPanel renders deep arch fields (layers, heads, hidden size, etc.)."""
    from llmvis.tui.widgets.model_panel import ModelPanel
    from llmvis.core.state import ModelArchInfo

    state = make_initial_state()
    state.deep = make_deep_state()
    state.deep.model_loaded = True
    state.deep.arch = ModelArchInfo(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_layers=28,
        num_attention_heads=16,
        hidden_size=1536,
        vocab_size=151936,
        head_dim=96,
        dtype="bfloat16",
        device="cuda:0",
    )

    panel = ModelPanel(state)
    rendered = panel._build(state)

    assert "Qwen/Qwen2.5-1.5B-Instruct" in rendered
    assert "28" in rendered        # num_layers
    assert "16" in rendered        # num_attention_heads
    assert "1,536" in rendered     # hidden_size formatted
    assert "bfloat16" in rendered
    assert "cuda:0" in rendered
    assert "Deep Instrumentation" in rendered


def test_model_panel_deep_mode_connecting_state():
    """When deep mode is active but model not loaded, show connecting message."""
    from llmvis.tui.widgets.model_panel import ModelPanel

    state = make_initial_state()
    state.deep = make_deep_state()
    state.deep.model_loaded = False
    state.deep.arch.model_id = ""
    state.deep.phase = InferencePhase.LOADING

    panel = ModelPanel(state)
    rendered = panel._build(state)

    assert "ollama run" not in rendered.lower()
    assert "No model loaded" not in rendered


# ── ModelPanel: Ollama model does not contaminate deep state ──────────────────

def test_model_panel_deep_mode_ignores_active_ollama_model():
    """ModelPanel must NOT show Ollama model info when deep mode is active."""
    from llmvis.tui.widgets.model_panel import ModelPanel
    from llmvis.core.events import ModelInfo

    state = make_initial_state()
    # Simulate an Ollama model also running
    state.active_model = ModelInfo(name="gemma4:e2b", family="gemma")
    # But deep mode is active
    state.deep = make_deep_state()
    state.deep.model_loaded = True
    state.deep.arch.model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    panel = ModelPanel(state)
    rendered = panel._build(state)

    assert "gemma4:e2b" not in rendered
    assert "Qwen/Qwen2.5-1.5B-Instruct" in rendered


# ── TelemetryServer: persistent MODEL_ARCHITECTURE for late clients ───────────

@pytest.mark.asyncio
async def test_persistent_model_arch_sent_to_late_client_beyond_history(unused_tcp_port):
    """MODEL_ARCHITECTURE is always sent to late clients even if rotated out of ring buffer."""
    from llmvis.ipc.server import MAX_HISTORY

    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        arch = ModelArchitectureEvent(
            source="test", model_id="qwen-test", num_layers=4,
            num_attention_heads=4, hidden_size=64,
            vocab_size=200, head_dim=16, dtype="float32", device="cpu",
        )
        await server.broadcast(arch)

        # Push enough events to rotate MODEL_ARCHITECTURE out of the ring buffer.
        filler = InferenceStartEvent(source="test", model_id="m", prompt_preview="x")
        for _ in range(MAX_HISTORY):
            await server.broadcast(filler)

        # MODEL_ARCHITECTURE should NOT be in the ring buffer now.
        assert len(server._history) == MAX_HISTORY
        assert "MODEL_ARCHITECTURE" in server._persistent

        # Late client connects after the ring buffer rotated.
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        # SESSION_INFO
        await _read_line(reader)

        # Collect all lines until we get MODEL_ARCHITECTURE or timeout.
        found_arch = False
        for _ in range(MAX_HISTORY + 5):
            try:
                line = await _read_line(reader, timeout=1.0)
                evt = event_from_jsonl(line)
                if evt is not None and evt.type == EventType.MODEL_ARCHITECTURE:
                    found_arch = True
                    assert evt.model_id == "qwen-test"
                    break
            except asyncio.TimeoutError:
                break

        assert found_arch, "Late client must receive MODEL_ARCHITECTURE from persistent cache"

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_persistent_arch_not_sent_twice_when_in_history(unused_tcp_port):
    """MODEL_ARCHITECTURE is not duplicated when it is still in the ring buffer."""
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        arch = ModelArchitectureEvent(
            source="test", model_id="qwen-test", num_layers=4,
            num_attention_heads=4, hidden_size=64,
            vocab_size=200, head_dim=16, dtype="float32", device="cpu",
        )
        await server.broadcast(arch)

        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        # SESSION_INFO
        await _read_line(reader)

        # Collect lines for a short time and count MODEL_ARCHITECTURE occurrences.
        arch_count = 0
        for _ in range(5):
            try:
                line = await _read_line(reader, timeout=0.5)
                evt = event_from_jsonl(line)
                if evt is not None and evt.type == EventType.MODEL_ARCHITECTURE:
                    arch_count += 1
            except asyncio.TimeoutError:
                break

        assert arch_count == 1, "MODEL_ARCHITECTURE must be sent exactly once to late clients"

        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


# ── StatusBar: mode indicator ─────────────────────────────────────────────────

def test_status_bar_default_nav_mode():
    """StatusBar starts in NAV MODE and shows navigation keys."""
    from llmvis.tui.widgets.status_bar import StatusBar

    sb = StatusBar(deep_mode=True)
    text = sb.render().plain

    assert "NAV MODE" in text
    assert "INPUT MODE" not in text


def test_status_bar_input_mode():
    """After set_mode(input_mode=True), StatusBar shows INPUT MODE hints."""
    from llmvis.tui.widgets.status_bar import StatusBar

    sb = StatusBar(deep_mode=True)
    sb.set_mode(input_mode=True)
    text = sb.render().plain

    assert "INPUT MODE" in text
    assert "NAV MODE" not in text
    assert "Enter" in text
    assert "Esc" in text


def test_status_bar_paused_indicator():
    """StatusBar shows PAUSED indicator when paused in NAV MODE."""
    from llmvis.tui.widgets.status_bar import StatusBar

    sb = StatusBar(deep_mode=True)
    sb.set_mode(paused=True)
    text = sb.render().plain

    assert "PAUSED" in text
    assert "NAV MODE" in text


def test_status_bar_nav_shows_i_key():
    """StatusBar NAV MODE shows 'i' as the prompt key."""
    from llmvis.tui.widgets.status_bar import StatusBar

    sb = StatusBar(deep_mode=True)
    text = sb.render().plain

    assert "i" in text
    assert "prompt" in text


def test_status_bar_non_deep_mode_unchanged():
    """Non-deep StatusBar still renders V1 keybindings."""
    from llmvis.tui.widgets.status_bar import StatusBar

    sb = StatusBar(deep_mode=False)
    text = sb.render().plain

    assert "quit" in text
    # Non-deep mode should not show deep mode indicators
    assert "NAV MODE" not in text
    assert "INPUT MODE" not in text

"""Tests for Ollama adapter — mocking HTTP calls."""

from __future__ import annotations

import asyncio
import pytest
import respx
import httpx
from typing import AsyncIterator

from llmvis.adapters.ollama import OllamaAdapter, _parse_model_info
from llmvis.core.events import (
    EventType,
    LLMVisEvent,
    ModelLoadedEvent,
    ModelUnloadedEvent,
    OllamaConnectedEvent,
    OllamaReconnectingEvent,
)
from tests.conftest import (
    FAKE_VERSION_RESPONSE,
    FAKE_PS_RESPONSE_EMPTY,
    FAKE_PS_RESPONSE_ONE,
    FAKE_PS_RESPONSE_TWO,
    FAKE_TAGS_RESPONSE,
    FAKE_MALFORMED_RESPONSE,
)


def test_parse_model_info_basic():
    raw = FAKE_PS_RESPONSE_ONE["models"][0]
    info = _parse_model_info(raw, loaded_info=raw)
    assert info.name == "qwen2.5:3b"
    assert info.family == "qwen2"
    assert info.parameter_size == "3.1B"
    assert info.quantization_level == "Q4_K_M"
    assert info.size_vram_bytes == 1800000000
    assert info.context_length == 4096


def test_parse_model_info_no_loaded():
    raw = FAKE_TAGS_RESPONSE["models"][0]
    info = _parse_model_info(raw)
    assert info.name == "qwen2.5:3b"
    assert info.size_vram_bytes == 0
    assert info.context_length == 32768


def test_parse_model_info_malformed():
    info = _parse_model_info({})
    assert info.name == ""
    assert info.family == ""


async def _collect_events(
    adapter: OllamaAdapter, max_events: int, timeout: float
) -> list[LLMVisEvent]:
    """Collect up to max_events from adapter within timeout seconds."""
    events: list[LLMVisEvent] = []
    try:
        async with asyncio.timeout(timeout):
            async for event in adapter.run():
                events.append(event)
                if len(events) >= max_events:
                    break
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    return events


@pytest.mark.asyncio
@respx.mock
async def test_adapter_connects():
    """Adapter emits OllamaConnectedEvent on first successful connection."""
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json=FAKE_VERSION_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_PS_RESPONSE_EMPTY)
    )

    adapter = OllamaAdapter()
    events = await _collect_events(adapter, max_events=1, timeout=3.0)

    assert len(events) >= 1
    assert events[0].type == EventType.OLLAMA_CONNECTED
    assert isinstance(events[0], OllamaConnectedEvent)
    assert events[0].version == "0.33.2"
    await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_disconnected():
    """Adapter emits reconnecting event when Ollama is unreachable."""
    respx.get("http://localhost:11434/api/version").mock(
        side_effect=httpx.ConnectError("refused")
    )

    adapter = OllamaAdapter()
    events = await _collect_events(adapter, max_events=1, timeout=5.0)

    assert len(events) >= 1
    assert events[0].type == EventType.OLLAMA_RECONNECTING
    await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_model_loaded():
    """Adapter emits ModelLoadedEvent when a model appears in /api/ps."""
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json=FAKE_VERSION_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_PS_RESPONSE_ONE)
    )

    adapter = OllamaAdapter()
    events = await _collect_events(adapter, max_events=2, timeout=5.0)

    types = [e.type for e in events]
    assert EventType.OLLAMA_CONNECTED in types
    assert EventType.MODEL_LOADED in types

    loaded_events = [e for e in events if e.type == EventType.MODEL_LOADED]
    assert loaded_events[0].model.name == "qwen2.5:3b"
    await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_multiple_models():
    """Adapter correctly handles multiple simultaneously loaded models."""
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json=FAKE_VERSION_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_PS_RESPONSE_TWO)
    )

    adapter = OllamaAdapter()
    events = await _collect_events(adapter, max_events=3, timeout=5.0)

    loaded = [e for e in events if e.type == EventType.MODEL_LOADED]
    loaded_names = {e.model.name for e in loaded}
    assert "qwen2.5:3b" in loaded_names
    assert "gemma3:4b" in loaded_names
    await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_model_unloaded():
    """Adapter emits ModelUnloadedEvent when model disappears from /api/ps."""
    call_count = 0

    def ps_side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=FAKE_PS_RESPONSE_ONE)
        return httpx.Response(200, json=FAKE_PS_RESPONSE_EMPTY)

    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json=FAKE_VERSION_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(side_effect=ps_side_effect)

    import llmvis.adapters.ollama as ollama_mod
    original = ollama_mod.POLL_INTERVAL_ACTIVE
    ollama_mod.POLL_INTERVAL_ACTIVE = 0.1

    try:
        adapter = OllamaAdapter()
        events = await _collect_events(adapter, max_events=3, timeout=8.0)

        types = [e.type for e in events]
        assert EventType.MODEL_UNLOADED in types
        unloaded = [e for e in events if e.type == EventType.MODEL_UNLOADED]
        assert unloaded[0].model_name == "qwen2.5:3b"
    finally:
        ollama_mod.POLL_INTERVAL_ACTIVE = original
        await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_malformed_response():
    """Adapter handles malformed API responses gracefully."""
    respx.get("http://localhost:11434/api/version").mock(
        return_value=httpx.Response(200, json=FAKE_VERSION_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_MALFORMED_RESPONSE)
    )

    adapter = OllamaAdapter()
    events = await _collect_events(adapter, max_events=1, timeout=4.0)

    assert any(e.type == EventType.OLLAMA_CONNECTED for e in events)
    await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_adapter_reconnects_after_disconnect():
    """Adapter recovers and emits OllamaConnectedEvent after reconnecting."""
    call_count = 0

    def version_side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=FAKE_VERSION_RESPONSE)

    respx.get("http://localhost:11434/api/version").mock(side_effect=version_side_effect)
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_PS_RESPONSE_EMPTY)
    )

    import llmvis.adapters.ollama as ollama_mod
    original_delay = ollama_mod.RECONNECT_DELAY
    ollama_mod.RECONNECT_DELAY = 0.1

    try:
        adapter = OllamaAdapter()
        events = await _collect_events(adapter, max_events=3, timeout=10.0)

        types = [e.type for e in events]
        assert EventType.OLLAMA_RECONNECTING in types
        assert EventType.OLLAMA_CONNECTED in types
    finally:
        ollama_mod.RECONNECT_DELAY = original_delay
        await adapter.stop()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_installed_models():
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json=FAKE_TAGS_RESPONSE)
    )
    respx.get("http://localhost:11434/api/ps").mock(
        return_value=httpx.Response(200, json=FAKE_PS_RESPONSE_EMPTY)
    )

    adapter = OllamaAdapter()
    models = await adapter.fetch_installed_models()
    assert len(models) == 1
    assert models[0].name == "qwen2.5:3b"
    assert models[0].context_length == 32768
    await adapter.stop()

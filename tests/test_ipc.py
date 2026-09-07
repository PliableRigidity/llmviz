"""Tests for TelemetryServer, ClientAdapter, and probe_source."""

from __future__ import annotations

import asyncio
import json

import pytest

from llmvis.core.events import (
    BackendConnectedEvent,
    BackendDisconnectedEvent,
    EventType,
    InferenceStartEvent,
    ModelArchitectureEvent,
    SessionInfoEvent,
    event_from_jsonl,
    event_to_jsonl,
)
from llmvis.ipc.server import TelemetryServer


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read_line(reader: asyncio.StreamReader, timeout: float = 2.0) -> str:
    raw = await asyncio.wait_for(reader.readline(), timeout=timeout)
    return raw.decode().strip()


# ── TelemetryServer ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_server_start_stop(unused_tcp_port):
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    assert server.client_count == 0
    await server.stop()


@pytest.mark.asyncio
async def test_session_info_on_connect(unused_tcp_port):
    server = TelemetryServer(
        port=unused_tcp_port,
        session_id="abc123",
        model_id="test-model",
        backend="transformers",
        telemetry_mode="standard",
        platform_name="Linux",
    )
    await server.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        line = await _read_line(reader)
        evt = event_from_jsonl(line)
        assert evt is not None
        assert evt.type == EventType.SESSION_INFO
        assert evt.server_session_id == "abc123"
        assert evt.model_id == "test-model"
        assert evt.backend == "transformers"
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_broadcast_reaches_client(unused_tcp_port):
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        # Drain the session info line
        await _read_line(reader)

        # Broadcast an event
        evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="hello")
        await server.broadcast(evt)

        line = await _read_line(reader)
        received = event_from_jsonl(line)
        assert received is not None
        assert received.type == EventType.INFERENCE_START
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_history_replayed_to_late_client(unused_tcp_port):
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        # Broadcast before any client connects
        arch = ModelArchitectureEvent(
            source="test", model_id="m", num_layers=4,
            num_attention_heads=4, hidden_size=64,
            vocab_size=200, head_dim=16, dtype="float32", device="cpu",
        )
        await server.broadcast(arch)

        # Late-connecting client should receive history
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        # session info
        await _read_line(reader)
        # history replay
        line = await _read_line(reader)
        received = event_from_jsonl(line)
        assert received is not None
        assert received.type == EventType.MODEL_ARCHITECTURE
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_prompt_command_triggers_callback(unused_tcp_port):
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    received_prompts = []
    server.add_prompt_callback(received_prompts.append)
    await server.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        await _read_line(reader)  # session info

        cmd = json.dumps({"_cmd": "prompt", "text": "hello world"}) + "\n"
        writer.write(cmd.encode())
        await writer.drain()
        await asyncio.sleep(0.1)

        assert received_prompts == ["hello world"]
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_multiple_clients_receive_broadcast(unused_tcp_port):
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        r1, w1 = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        r2, w2 = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
        await _read_line(r1)  # session info for client 1
        await _read_line(r2)  # session info for client 2

        await asyncio.sleep(0.05)  # let server register both clients

        evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="x")
        await server.broadcast(evt)

        l1 = await _read_line(r1)
        l2 = await _read_line(r2)
        assert event_from_jsonl(l1).type == EventType.INFERENCE_START
        assert event_from_jsonl(l2).type == EventType.INFERENCE_START

        for w in (w1, w2):
            w.close()
            await w.wait_closed()
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_history_capped_at_max(unused_tcp_port):
    from llmvis.ipc.server import MAX_HISTORY
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="x")
        for _ in range(MAX_HISTORY + 10):
            await server.broadcast(evt)
        assert len(server._history) == MAX_HISTORY
    finally:
        await server.stop()


# ── ClientAdapter ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_client_adapter_connect_and_receive(unused_tcp_port):
    from llmvis.adapters.client_adapter import ClientAdapter

    server = TelemetryServer(port=unused_tcp_port, session_id="s1", model_id="m")
    await server.start()

    # Collect events from ClientAdapter
    events = []
    adapter = ClientAdapter(host="127.0.0.1", port=unused_tcp_port)

    async def _run() -> None:
        async for evt in adapter.run():
            events.append(evt)
            if len(events) >= 2:  # BackendConnected + SessionInfo
                await adapter.stop()
                break

    task = asyncio.create_task(_run())
    await asyncio.wait_for(task, timeout=5.0)

    assert any(e.type == EventType.BACKEND_CONNECTED for e in events)
    assert any(e.type == EventType.SESSION_INFO for e in events)
    await server.stop()


@pytest.mark.asyncio
async def test_client_adapter_detects_disconnect(unused_tcp_port):
    from llmvis.adapters.client_adapter import ClientAdapter

    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()

    events = []
    adapter = ClientAdapter(host="127.0.0.1", port=unused_tcp_port)
    connected = asyncio.Event()

    async def _run() -> None:
        async for evt in adapter.run():
            events.append(evt)
            if evt.type == EventType.SESSION_INFO:
                connected.set()
            if evt.type == EventType.BACKEND_DISCONNECTED:
                await adapter.stop()
                break

    task = asyncio.create_task(_run())
    # Wait for connection, then stop server — server.stop() closes client
    # writers which triggers EOF in the adapter's reader loop.
    await asyncio.wait_for(connected.wait(), timeout=5.0)
    await server.stop()
    await asyncio.wait_for(task, timeout=10.0)

    assert any(e.type == EventType.BACKEND_DISCONNECTED for e in events)


@pytest.mark.asyncio
async def test_client_adapter_submit_prompt(unused_tcp_port):
    from llmvis.adapters.client_adapter import ClientAdapter

    received = []
    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    server.add_prompt_callback(received.append)
    await server.start()

    adapter = ClientAdapter(host="127.0.0.1", port=unused_tcp_port)

    async def _run() -> None:
        async for evt in adapter.run():
            if evt.type == EventType.SESSION_INFO:
                await adapter.submit_prompt("test prompt")
                await asyncio.sleep(0.2)
                await adapter.stop()
                break

    task = asyncio.create_task(_run())
    await asyncio.wait_for(task, timeout=5.0)
    await server.stop()

    assert "test prompt" in received


@pytest.mark.asyncio
async def test_client_adapter_retries_until_server_starts(unused_tcp_port):
    """ClientAdapter should eventually connect once the server starts."""
    from llmvis.adapters.client_adapter import ClientAdapter

    events = []
    adapter = ClientAdapter(host="127.0.0.1", port=unused_tcp_port)

    async def _run() -> None:
        async for evt in adapter.run():
            events.append(evt)
            if evt.type == EventType.SESSION_INFO:
                await adapter.stop()
                break

    task = asyncio.create_task(_run())

    # Start server after a short delay
    await asyncio.sleep(0.3)
    server = TelemetryServer(port=unused_tcp_port, session_id="s2")
    await server.start()

    await asyncio.wait_for(task, timeout=10.0)
    await server.stop()

    assert any(e.type == EventType.SESSION_INFO for e in events)


# ── Event round-trips for new types ──────────────────────────────────────────

def test_session_info_event_round_trip():
    evt = SessionInfoEvent(
        source="server",
        server_session_id="abc",
        model_id="m",
        backend="transformers",
        telemetry_mode="standard",
        platform_name="Linux",
        port=7654,
    )
    line = event_to_jsonl(evt)
    restored = event_from_jsonl(line)
    assert restored is not None
    assert restored.type == EventType.SESSION_INFO
    assert restored.server_session_id == "abc"
    assert restored.port == 7654


def test_backend_connected_event_round_trip():
    evt = BackendConnectedEvent(source="ca", host="127.0.0.1", port=7654)
    restored = event_from_jsonl(event_to_jsonl(evt))
    assert restored is not None
    assert restored.type == EventType.BACKEND_CONNECTED
    assert restored.port == 7654


def test_backend_disconnected_event_round_trip():
    evt = BackendDisconnectedEvent(
        source="ca", host="127.0.0.1", port=7654, reason="connection refused"
    )
    restored = event_from_jsonl(event_to_jsonl(evt))
    assert restored is not None
    assert restored.type == EventType.BACKEND_DISCONNECTED
    assert restored.reason == "connection refused"


# ── probe_source ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_probe_source_finds_deep_server(unused_tcp_port):
    from llmvis.tui.waiting_app import _probe_source

    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        result = await _probe_source("http://localhost:11434", unused_tcp_port)
        assert result == "deep"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_probe_source_returns_none_when_nothing_available(unused_tcp_port):
    from llmvis.tui.waiting_app import _probe_source
    # Port is guaranteed free (unused_tcp_port fixture), Ollama not running in test env
    result = await _probe_source("http://localhost:1", unused_tcp_port)
    assert result is None


@pytest.mark.asyncio
async def test_probe_source_deep_takes_priority_over_ollama(unused_tcp_port):
    """Deep session takes priority over Ollama when both are available."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from llmvis.tui.waiting_app import _probe_source

    server = TelemetryServer(port=unused_tcp_port, session_id="s1")
    await server.start()
    try:
        # Even if Ollama would be reachable, deep must win
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_client_cls.return_value = mock_client

            result = await _probe_source("http://localhost:11434", unused_tcp_port)
            assert result == "deep", "Deep session must take priority over Ollama"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_probe_source_falls_back_to_ollama_when_no_deep(unused_tcp_port):
    """Falls back to 'ollama' when deep is not running but Ollama is reachable."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from llmvis.tui.waiting_app import _probe_source

    # unused_tcp_port has no server — deep probe will fail
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
        mock_client_cls.return_value = mock_client

        result = await _probe_source("http://localhost:11434", unused_tcp_port)
        assert result == "ollama", "Must fall back to Ollama when deep is not running"

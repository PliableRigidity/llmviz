"""ClientAdapter — wraps a TCP connection to TelemetryServer as a BaseAdapter.

Connects to a running TelemetryServer on localhost:<port>, reads JSONL events,
and yields them as LLMVisEvent instances. Auto-reconnects on disconnect.
Emits BackendConnectedEvent / BackendDisconnectedEvent to signal connection state.

To submit a prompt, call submit_prompt(text) — it sends the command over TCP.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from llmvis.adapters.base import BaseAdapter
from llmvis.core.events import (
    BackendConnectedEvent,
    BackendDisconnectedEvent,
    LLMVisEvent,
    event_from_jsonl,
)

logger = logging.getLogger(__name__)

RECONNECT_DELAY = 2.0  # seconds between reconnect attempts


class ClientAdapter(BaseAdapter):
    """Connects to a TelemetryServer via TCP and yields events as a BaseAdapter."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7654) -> None:
        self._host = host
        self._port = port
        self._stop_event = asyncio.Event()
        self._writer: asyncio.StreamWriter | None = None
        self._writer_lock = asyncio.Lock()

    async def run(self) -> AsyncIterator[LLMVisEvent]:
        while not self._stop_event.is_set():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self._host, self._port), timeout=1.0
                )
                async with self._writer_lock:
                    self._writer = writer

                yield BackendConnectedEvent(
                    source="client_adapter", host=self._host, port=self._port
                )
                logger.info("Connected to TelemetryServer at %s:%d", self._host, self._port)

                try:
                    while not self._stop_event.is_set():
                        try:
                            raw = await asyncio.wait_for(reader.readline(), timeout=5.0)
                        except asyncio.TimeoutError:
                            continue
                        if not raw:
                            break
                        line = raw.decode().strip()
                        if not line:
                            continue
                        evt = event_from_jsonl(line)
                        if evt is not None:
                            yield evt
                finally:
                    async with self._writer_lock:
                        self._writer = None
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

                if not self._stop_event.is_set():
                    yield BackendDisconnectedEvent(
                        source="client_adapter",
                        host=self._host,
                        port=self._port,
                        reason="connection closed",
                    )
                    logger.info(
                        "Disconnected from TelemetryServer; retrying in %.1fs", RECONNECT_DELAY
                    )

            except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                if not self._stop_event.is_set():
                    yield BackendDisconnectedEvent(
                        source="client_adapter",
                        host=self._host,
                        port=self._port,
                        reason="connection refused",
                    )
            except Exception as exc:
                logger.debug("ClientAdapter error: %s", exc)
                if not self._stop_event.is_set():
                    yield BackendDisconnectedEvent(
                        source="client_adapter",
                        host=self._host,
                        port=self._port,
                        reason=str(exc),
                    )

            if not self._stop_event.is_set():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(self._stop_event.wait()), timeout=RECONNECT_DELAY
                    )
                except asyncio.TimeoutError:
                    pass

    async def submit_prompt(self, text: str) -> None:
        async with self._writer_lock:
            writer = self._writer
        if writer is None:
            logger.warning("submit_prompt called but not connected to server")
            return
        try:
            cmd = json.dumps({"_cmd": "prompt", "text": text}) + "\n"
            writer.write(cmd.encode())
            await writer.drain()
        except Exception as exc:
            logger.warning("Failed to send prompt: %s", exc)

    async def stop(self) -> None:
        self._stop_event.set()
        async with self._writer_lock:
            writer = self._writer
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

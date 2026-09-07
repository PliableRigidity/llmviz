"""TelemetryServer — broadcasts LLMVis events over TCP to connected visualizers.

Protocol: line-delimited JSONL (one event per line).
  Server→Client: events as JSONL lines, sent to all connected clients.
  Client→Server: {"_cmd": "prompt", "text": "..."} to submit a prompt.

On each new client connection the server immediately sends:
  1. A SessionInfoEvent describing the current session.
  2. Up to MAX_HISTORY recent events (for late-connecting visualizers).
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Callable

from llmvis.core.events import (
    LLMVisEvent,
    SessionInfoEvent,
    event_to_jsonl,
)

logger = logging.getLogger(__name__)

DEFAULT_PORT = 7654
MAX_HISTORY = 100


class TelemetryServer:
    """TCP server that broadcasts telemetry events to all connected visualizers."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        session_id: str = "",
        model_id: str = "",
        backend: str = "",
        telemetry_mode: str = "standard",
        platform_name: str = "",
    ) -> None:
        self._port = port
        self._session_id = session_id
        self._model_id = model_id
        self._backend = backend
        self._telemetry_mode = telemetry_mode
        self._platform_name = platform_name

        self._history: deque[str] = deque(maxlen=MAX_HISTORY)
        # Persistent events always replayed to late-connecting clients regardless
        # of whether the general ring-buffer has rotated them out.
        self._persistent: dict[str, str] = {}  # EventType.name -> JSONL line
        self._clients: set[asyncio.StreamWriter] = set()
        self._lock = asyncio.Lock()
        self._server: asyncio.AbstractServer | None = None
        self._prompt_callbacks: list[Callable[[str], None]] = []

    @property
    def port(self) -> int:
        return self._port

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self) -> None:
        # Bind to 127.0.0.1 explicitly rather than "localhost" to avoid
        # IPv6/IPv4 ambiguity on Windows (localhost can resolve to ::1 first).
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", self._port
        )
        logger.info("TelemetryServer listening on 127.0.0.1:%d", self._port)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
        # Close client connections BEFORE await wait_closed().
        # In Python 3.11+ Server.wait_closed() waits for all handler tasks to
        # finish. Handlers are blocked reading commands from clients, so we must
        # signal them to exit by closing the writers first.
        async with self._lock:
            for writer in list(self._clients):
                try:
                    writer.close()
                except Exception:
                    pass
            self._clients.clear()
        if self._server is not None:
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=3.0)
            except (asyncio.TimeoutError, Exception):
                pass

    async def broadcast(self, evt: LLMVisEvent) -> None:
        """Send event to all connected clients and add it to the history cache."""
        line = event_to_jsonl(evt) + "\n"
        # Persist MODEL_ARCHITECTURE so late clients always receive it even if
        # it has rotated out of the general ring-buffer.
        _PERSISTENT_TYPES = {"MODEL_ARCHITECTURE"}
        if evt.type.name in _PERSISTENT_TYPES:
            self._persistent[evt.type.name] = line
        self._history.append(line)
        async with self._lock:
            dead = set()
            for writer in self._clients:
                try:
                    writer.write(line.encode())
                    await writer.drain()
                except Exception:
                    dead.add(writer)
            self._clients -= dead

    def add_prompt_callback(self, cb: Callable[[str], None]) -> None:
        """Register a callback to receive prompt text submitted by visualizer clients."""
        self._prompt_callbacks.append(cb)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername", ("?", 0))
        logger.info("Client connected from %s:%s", *peer)

        async with self._lock:
            self._clients.add(writer)

        try:
            # Send session info immediately
            session_info = SessionInfoEvent(
                source="server",
                server_session_id=self._session_id,
                model_id=self._model_id,
                backend=self._backend,
                telemetry_mode=self._telemetry_mode,
                platform_name=self._platform_name,
                port=self._port,
            )
            writer.write((event_to_jsonl(session_info) + "\n").encode())
            await writer.drain()

            # Send any persistent metadata events (e.g. MODEL_ARCHITECTURE) that
            # must always reach late clients, even if the ring-buffer has rotated.
            # Skip types already present in the history snapshot to avoid doubles.
            history_snapshot = list(self._history)
            history_types = set()
            import json as _json
            for _line in history_snapshot:
                try:
                    history_types.add(_json.loads(_line).get("_type", ""))
                except Exception:
                    pass
            for evt_type_name, pline in self._persistent.items():
                if evt_type_name not in history_types:
                    try:
                        writer.write(pline.encode())
                        await writer.drain()
                    except Exception:
                        break

            # Replay history so late-connecting clients catch up
            for line in history_snapshot:
                try:
                    writer.write(line.encode())
                    await writer.drain()
                except Exception:
                    break

            # Read commands from client
            while True:
                try:
                    raw = await asyncio.wait_for(reader.readline(), timeout=30.0)
                except asyncio.TimeoutError:
                    continue
                if not raw:
                    break
                try:
                    import json
                    cmd = json.loads(raw.decode().strip())
                    if cmd.get("_cmd") == "prompt":
                        text = cmd.get("text", "")
                        if text:
                            for cb in self._prompt_callbacks:
                                cb(text)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Client handler error: %s", exc)
        finally:
            async with self._lock:
                self._clients.discard(writer)
            try:
                writer.close()
            except Exception:
                pass
            logger.info("Client disconnected from %s:%s", *peer)

"""Ollama adapter — polls public Ollama HTTP API and emits normalized events.

Observable with stock Ollama:
  - Connection state via GET /api/version
  - Loaded model list via GET /api/ps
  - Model metadata via GET /api/tags
  - Inference activity: ESTIMATED from CPU/GPU usage change only

Not observable without instrumented backend:
  - Another client's token stream
  - Prefill vs. decode phases
  - Per-token timing, probabilities, logits
  - Layer activations, KV-cache state
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx

from llmvis.adapters.base import BaseAdapter
from llmvis.core.events import (
    LLMVisEvent,
    ModelInfo,
    ModelLoadedEvent,
    ModelUnloadedEvent,
    OllamaConnectedEvent,
    OllamaDisconnectedEvent,
    OllamaReconnectingEvent,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_IDLE = 2.0      # seconds between polls when no model loaded
POLL_INTERVAL_ACTIVE = 1.0    # seconds when model is loaded
RECONNECT_DELAY = 3.0
MAX_RECONNECT_DELAY = 30.0


def _parse_model_info(raw: dict, loaded_info: dict | None = None) -> ModelInfo:
    """Parse a model entry from /api/tags or /api/ps into ModelInfo."""
    details = raw.get("details", {})
    families = details.get("families") or []
    if isinstance(families, str):
        families = [families]

    size_vram = 0
    context_length_loaded = None
    expires_at = None
    if loaded_info:
        size_vram = loaded_info.get("size_vram", 0)
        context_length_loaded = loaded_info.get("context_length")
        expires_at = loaded_info.get("expires_at")

    # context_length from details (installed) or loaded
    ctx = details.get("context_length") or context_length_loaded

    return ModelInfo(
        name=raw.get("name", raw.get("model", "")),
        family=details.get("family", ""),
        families=families,
        parameter_size=details.get("parameter_size", ""),
        quantization_level=details.get("quantization_level", ""),
        format=details.get("format", ""),
        size_bytes=raw.get("size", 0),
        size_vram_bytes=size_vram,
        context_length=ctx,
        embedding_length=details.get("embedding_length"),
        capabilities=raw.get("capabilities", []),
        digest=raw.get("digest", ""),
        expires_at=expires_at,
    )


class OllamaAdapter(BaseAdapter):
    """Polls Ollama's public API and emits LLMVisEvents."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self.host = host.rstrip("/")
        self._stop_event = asyncio.Event()
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._loaded_model_names: set[str] = set()
        self._version = ""

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.host,
                timeout=httpx.Timeout(5.0),
            )
        return self._client

    async def _check_connection(self) -> tuple[bool, str]:
        """Returns (connected, version)."""
        try:
            client = await self._get_client()
            resp = await client.get("/api/version")
            resp.raise_for_status()
            version = resp.json().get("version", "")
            return True, version
        except Exception as exc:
            logger.debug("Ollama connection check failed: %s", exc)
            return False, ""

    async def _fetch_ps(self) -> list[dict]:
        """Fetch currently loaded models from /api/ps."""
        try:
            client = await self._get_client()
            resp = await client.get("/api/ps")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return []

    async def _fetch_tags(self) -> list[dict]:
        """Fetch installed models from /api/tags."""
        try:
            client = await self._get_client()
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])
        except Exception:
            return []

    async def run(self) -> AsyncIterator[LLMVisEvent]:
        reconnect_delay = RECONNECT_DELAY
        attempt = 0

        while not self._stop_event.is_set():
            connected, version = await self._check_connection()

            if not connected:
                if self._connected:
                    # Was connected, now lost
                    self._connected = False
                    self._loaded_model_names.clear()
                    yield OllamaDisconnectedEvent(source="ollama_adapter", reason="connection lost")

                attempt += 1
                yield OllamaReconnectingEvent(source="ollama_adapter", attempt=attempt)

                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=reconnect_delay
                    )
                except asyncio.TimeoutError:
                    pass

                reconnect_delay = min(reconnect_delay * 1.5, MAX_RECONNECT_DELAY)
                continue

            # Connected
            reconnect_delay = RECONNECT_DELAY
            attempt = 0

            if not self._connected:
                self._connected = True
                self._version = version
                yield OllamaConnectedEvent(
                    source="ollama_adapter", version=version, host=self.host
                )

            # Poll loaded models
            ps_models = await self._fetch_ps()
            current_names = {m.get("name", m.get("model", "")) for m in ps_models}

            # Build lookup for loaded model details (size_vram, context_length, expires_at)
            ps_by_name = {m.get("name", m.get("model", "")): m for m in ps_models}

            # Detect newly loaded models
            new_names = current_names - self._loaded_model_names
            for name in new_names:
                raw = ps_by_name[name]
                info = _parse_model_info(raw, loaded_info=raw)
                yield ModelLoadedEvent(source="ollama_adapter", model=info)

            # Detect unloaded models
            gone_names = self._loaded_model_names - current_names
            for name in gone_names:
                yield ModelUnloadedEvent(source="ollama_adapter", model_name=name)

            self._loaded_model_names = current_names

            interval = POLL_INTERVAL_ACTIVE if current_names else POLL_INTERVAL_IDLE
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def fetch_installed_models(self) -> list[ModelInfo]:
        tags = await self._fetch_tags()
        ps_models = await self._fetch_ps()
        ps_by_name = {m.get("name", m.get("model", "")): m for m in ps_models}
        result = []
        for raw in tags:
            name = raw.get("name", raw.get("model", ""))
            loaded = ps_by_name.get(name)
            result.append(_parse_model_info(raw, loaded_info=loaded))
        return result

    async def fetch_loaded_models(self) -> list[ModelInfo]:
        ps_models = await self._fetch_ps()
        return [_parse_model_info(m, loaded_info=m) for m in ps_models]

    async def stop(self) -> None:
        self._stop_event.set()
        if self._client:
            await self._client.aclose()
            self._client = None

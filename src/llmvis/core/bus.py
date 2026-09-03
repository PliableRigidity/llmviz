"""Simple async event bus."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable

from llmvis.core.events import EventType, LLMVisEvent

Handler = Callable[[LLMVisEvent], Awaitable[None]]

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)
        self._catch_all: list[Handler] = []
        self._queue: asyncio.Queue[LLMVisEvent] = asyncio.Queue()

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        self._catch_all.append(handler)

    async def publish(self, event: LLMVisEvent) -> None:
        await self._queue.put(event)

    async def dispatch_forever(self) -> None:
        while True:
            event = await self._queue.get()
            handlers = self._handlers.get(event.type, []) + self._catch_all
            for h in handlers:
                try:
                    await h(event)
                except Exception as exc:
                    logger.error("Event handler error for %s: %s", event.type, exc)
            self._queue.task_done()

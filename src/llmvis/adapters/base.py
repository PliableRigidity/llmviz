"""Abstract adapter interface."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from llmvis.core.events import LLMVisEvent


class BaseAdapter(abc.ABC):
    """Produces normalized LLMVisEvents from a backend."""

    @abc.abstractmethod
    async def run(self) -> AsyncIterator[LLMVisEvent]:
        """Yield events until stopped."""
        ...  # pragma: no cover

    @abc.abstractmethod
    async def stop(self) -> None:
        ...  # pragma: no cover

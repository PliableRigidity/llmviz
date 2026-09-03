"""Abstract telemetry interface."""

from __future__ import annotations

import abc

from llmvis.core.events import GpuStatsEvent, SystemStatsEvent


class BaseTelemetry(abc.ABC):
    @abc.abstractmethod
    async def get_system_stats(self) -> SystemStatsEvent:
        ...  # pragma: no cover

    @abc.abstractmethod
    async def get_gpu_stats(self) -> list[GpuStatsEvent]:
        ...  # pragma: no cover

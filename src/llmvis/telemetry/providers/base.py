"""Abstract telemetry provider for platform-aware hardware snapshots."""

from __future__ import annotations

import abc

from llmvis.telemetry.snapshot import SystemSnapshot


class BaseTelemetryProvider(abc.ABC):
    """Returns a SystemSnapshot on demand. Each provider targets one platform."""

    @abc.abstractmethod
    def snapshot(self) -> SystemSnapshot:
        """Collect and return a current hardware snapshot."""
        ...  # pragma: no cover

    def close(self) -> None:
        """Release any resources held by the provider (e.g. pynvml handles)."""

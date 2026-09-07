"""ReplayAdapter — plays back a recorded LLMVis telemetry session.

Records are JSONL files written by InstrumentedTransformersAdapter with --record.
Replay faithfully reproduces the event sequence and timing, making the TUI
behave exactly as it would during live inference.

This is also used by `llmvis demo` to replay the bundled demo trace.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path

from llmvis.adapters.base import BaseAdapter
from llmvis.core.events import LLMVisEvent, event_from_jsonl

logger = logging.getLogger(__name__)


class ReplayAdapter(BaseAdapter):
    """Replays a recorded telemetry session from a JSONL file.

    Parameters
    ----------
    path:
        Path to the JSONL recording produced by ``--record``.
    speed:
        Playback speed multiplier. 1.0 = real time, 2.0 = 2× faster, 0 = no delays.
    loop:
        If True, repeat the recording indefinitely (for demo mode).
    """

    def __init__(
        self,
        path: str | Path,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        self._path = Path(path)
        self._speed = max(speed, 0.0)
        self._loop = loop
        self._stop_event: asyncio.Event = asyncio.Event()

    async def run(self) -> AsyncIterator[LLMVisEvent]:  # type: ignore[override]
        while True:
            async for evt in self._replay_once():
                yield evt
            if not self._loop or self._stop_event.is_set():
                break
            # Brief pause between loops in demo mode
            await asyncio.sleep(2.0)

    async def _replay_once(self) -> AsyncIterator[LLMVisEvent]:
        events: list[LLMVisEvent] = []
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.error("Cannot read replay file %s: %s", self._path, exc)
            return

        for line in lines:
            line = line.strip()
            if not line:
                continue
            evt = event_from_jsonl(line)
            if evt is not None:
                events.append(evt)

        if not events:
            logger.warning("Replay file %s contained no valid events", self._path)
            return

        logger.info("Replaying %d events from %s", len(events), self._path)

        prev_ts: float | None = None
        for evt in events:
            if self._stop_event.is_set():
                return

            if prev_ts is not None and self._speed > 0:
                delay = (evt.timestamp - prev_ts) / self._speed
                if delay > 0:
                    await asyncio.sleep(min(delay, 2.0))  # cap at 2s to avoid stalls

            prev_ts = evt.timestamp
            yield evt

    async def stop(self) -> None:
        self._stop_event.set()

    # replay has no prompt queue — submitting a prompt is a no-op
    async def submit_prompt(self, text: str) -> None:
        logger.debug("Replay mode: prompt '%s' ignored", text)

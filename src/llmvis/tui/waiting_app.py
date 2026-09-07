"""WaitingApp — minimal Textual screen shown when no telemetry source is available.

Probes for a deep session (TCP on deep_port) and Ollama every 2 seconds.
Exits with return value 'deep', 'ollama', or None (if the user quits).
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from llmvis.ipc.server import DEFAULT_PORT

PROBE_INTERVAL = 2.0  # seconds between discovery probes


class WaitingApp(App):
    """Waiting screen: probes for available telemetry sources every 2 seconds."""

    CSS = """
    Screen {
        background: #0d1117;
        align: center middle;
    }
    #status {
        color: #8b949e;
        text-align: center;
        width: auto;
        height: auto;
        padding: 2 4;
    }
    """

    BINDINGS = [
        Binding("q", "quit_none", "Quit", priority=True),
        Binding("ctrl+c", "quit_none", "Quit", show=False, priority=True),
    ]

    def __init__(self, ollama_host: str = "http://localhost:11434", deep_port: int = DEFAULT_PORT) -> None:
        super().__init__()
        self._ollama_host = ollama_host
        self._deep_port = deep_port
        self._source: str | None = None
        self._dots = 0

    def compose(self) -> ComposeResult:
        yield Static(
            "LLMVis — waiting for a telemetry source",
            id="status",
        )

    async def on_mount(self) -> None:
        self._task = asyncio.create_task(self._probe_loop())

    async def _probe_loop(self) -> None:
        while True:
            source = await _probe_source(self._ollama_host, self._deep_port)
            if source is not None:
                self._source = source
                self.exit(source)
                return
            self._dots = (self._dots + 1) % 4
            dots = "." * self._dots
            try:
                self.query_one("#status", Static).update(
                    f"LLMVis — waiting for a telemetry source{dots}\n\n"
                    f"[dim]Probing deep session (port {self._deep_port}) and Ollama ({self._ollama_host})[/dim]\n"
                    "[dim]Start  Terminal 1: llmvis instrument <model>[/dim]\n"
                    "[dim]   or  Terminal 1: ollama run <model>[/dim]"
                )
            except Exception:
                pass
            await asyncio.sleep(PROBE_INTERVAL)

    def action_quit_none(self) -> None:
        if hasattr(self, "_task"):
            self._task.cancel()
        self.exit(None)


async def _probe_source(ollama_host: str, deep_port: int) -> str | None:
    """Fast probe: returns 'deep', 'ollama', or None. Times out quickly.

    Uses 127.0.0.1 explicitly to avoid IPv6/IPv4 resolution ambiguity on
    Windows where 'localhost' may resolve to ::1 first.
    """
    # Prefer deep session over Ollama
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", deep_port), timeout=0.5
        )
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        return "deep"
    except Exception:
        pass

    try:
        import httpx
        async with httpx.AsyncClient(timeout=0.5) as c:
            await c.get(f"{ollama_host}/api/version")
        return "ollama"
    except Exception:
        pass

    return None


def probe_source(ollama_host: str = "http://localhost:11434", deep_port: int = DEFAULT_PORT) -> str | None:
    """Synchronous wrapper for _probe_source (used by CLI)."""
    return asyncio.run(_probe_source(ollama_host, deep_port))

"""Bottom prompt input bar for deep instrumentation mode."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, Static


class PromptInput(Static):
    """Bottom prompt input bar for deep instrumentation mode."""

    CSS = """
    PromptInput {
        height: 3;
        border-top: solid #30363d;
        background: #161b22;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Enter prompt and press Enter...", id="prompt-field")

    def clear(self) -> None:
        self.query_one(Input).value = ""

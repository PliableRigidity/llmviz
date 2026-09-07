"""LLM pipeline diagram panel."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class PipelinePanel(Static):
    """Shows the LLM inference pipeline as a static educational diagram."""

    CSS = """
    PipelinePanel {
        border: solid #30363d;
        padding: 0 1;
        height: 5;
        background: #0d1117;
    }
    """

    def render(self) -> Text:
        return Text.from_markup(
            "[bold cyan]─ MODEL PIPELINE ──────────────────────────────────────────────────────[/bold cyan]\n"
            "\n"
            "  [dim]Prompt[/dim] → [white]Tokenize[/white] → [white]Prefill[/white]"
            " → [white]Decode[/white] → [white]Sample[/white] → [white]Output[/white]\n"
            "  [dim]  │           │           │            │          │[/dim]\n"
            "  [dim]  text    token ids    KV cache    next tok   detokenize[/dim]"
        )

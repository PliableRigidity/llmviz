"""Bottom status/keybinding bar."""

from __future__ import annotations

from textual.widgets import Static


class StatusBar(Static):
    """Bottom bar showing key bindings."""

    CSS = """
    StatusBar {
        height: 1;
        background: #161b22;
        border-top: solid #30363d;
        padding: 0 1;
        color: #8b949e;
    }
    """

    def render(self) -> str:
        return (
            "[dim]q[/dim] quit  "
            "[dim]c[/dim] concepts  "
            "[dim]?[/dim] help  "
            "[dim]1[/dim] overview  "
            "[dim]2[/dim] system  "
            "[dim]esc[/dim] close overlay"
        )

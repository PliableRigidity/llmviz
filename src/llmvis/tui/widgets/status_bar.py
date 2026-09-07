"""Bottom status/keybinding bar."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static


class StatusBar(Static):
    """Bottom bar showing key bindings.

    Pass ``deep_mode=True`` when used in the deep instrumentation TUI so that
    the bar shows the correct keybindings for that mode instead of V1 bindings.
    """

    CSS = """
    StatusBar {
        height: 1;
        background: #161b22;
        border-top: solid #30363d;
        padding: 0 1;
        color: #8b949e;
    }
    """

    def __init__(self, deep_mode: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self._deep_mode = deep_mode
        self._input_mode: bool = False
        self._paused: bool = False

    def set_mode(self, input_mode: bool = False, paused: bool = False) -> None:
        """Update the displayed mode indicator and refresh."""
        self._input_mode = input_mode
        self._paused = paused
        self.refresh()

    def render(self) -> Text:
        if self._deep_mode:
            if self._input_mode:
                return Text.from_markup(
                    "[bold yellow][INPUT MODE][/bold yellow]  "
                    "[dim]Enter[/dim] submit  "
                    "[dim]Esc[/dim] return to nav"
                )
            pause_str = "  [bold yellow]PAUSED[/bold yellow]" if self._paused else ""
            return Text.from_markup(
                "[bold green][NAV MODE][/bold green]  "
                "[dim]q[/dim] quit  "
                "[dim]1-6[/dim] tabs  "
                "[dim]←→[/dim] history  "
                "[dim]l[/dim] live  "
                "[dim]space[/dim] pause  "
                "[dim]i[/dim] prompt  "
                "[dim]?[/dim] help"
                + pause_str
            )
        return Text.from_markup(
            "[dim]q[/dim] quit  "
            "[dim]c[/dim] concepts  "
            "[dim]?[/dim] help  "
            "[dim]1[/dim] overview  "
            "[dim]2[/dim] system  "
            "[dim]esc[/dim] close overlay"
        )

"""Token history screen — shows per-token latency and KV cache growth."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, TokenRecord


_MAX_DISPLAY = 20
_HEADER = "─ TOKEN HISTORY ──────────────────────────────────────────────────"


def _fmt_token_text(text: str, max_len: int = 12) -> str:
    """Escape and truncate a token string for display."""
    # Replace newlines / tabs with visible stand-ins
    visible = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    if len(visible) > max_len:
        visible = visible[: max_len - 1] + "…"
    return f"'{visible}'"


class TokensScreen(Static):
    """TOKENS tab — generated token history with per-token latency.

    Displays the last _MAX_DISPLAY tokens from state.deep.token_history.
    The currently selected token (state.deep.selected_token_idx) is
    highlighted with a leading arrow.  Navigation: left/right arrow keys
    (handled by the parent app, which updates selected_token_idx).
    """

    DEFAULT_CSS = """
    TokensScreen {
        background: #0d1117;
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    def _build(self, state: AppState) -> str:
        lines: list[str] = []
        lines.append(f"[bold cyan]{_HEADER}[/bold cyan]")
        lines.append("")

        deep = state.deep
        if deep is None:
            lines.append("  [dim]Deep instrumentation not active.[/dim]")
            lines.append("  [dim]Use the InstrumentedTransformersAdapter to enable token telemetry.[/dim]")
            lines.append("")
            lines.append(self._educational_note())
            return "\n".join(lines)

        history: list[TokenRecord] = list(deep.token_history)

        if not history:
            if deep.is_generating:
                lines.append("  [dim]Waiting for first token...[/dim]")
            else:
                lines.append("  [dim]No tokens generated yet.[/dim]")
                lines.append("  [dim]Start a prompt to see token history here.[/dim]")
            lines.append("")
            lines.append(self._educational_note())
            return "\n".join(lines)

        total = len(history)
        selected_idx = deep.selected_token_idx  # may be None

        # Show last _MAX_DISPLAY tokens; if fewer exist, show all.
        window = history[-_MAX_DISPLAY:]
        window_start_abs = total - len(window)  # absolute index of window[0]

        for rel, record in enumerate(window):
            abs_idx = window_start_abs + rel
            is_selected = (selected_idx is not None and selected_idx == record.index)

            marker = "[bold yellow]▶[/bold yellow]" if is_selected else " "
            idx_str = f"Token {record.index:<4}"
            tok_str = f"{_fmt_token_text(record.text):<14}"
            lat_str = f"{record.latency_ms:>6.1f} ms"
            seq_str = f"seq={record.kv_seq_len}"

            if is_selected:
                row = (
                    f"{marker} [bold white]{idx_str}[/bold white]"
                    f"  [bold green]{tok_str}[/bold green]"
                    f"  [bold yellow]{lat_str}[/bold yellow]"
                    f"  [dim]{seq_str}[/dim]"
                )
            else:
                row = (
                    f"{marker} [dim]{idx_str}[/dim]"
                    f"  [white]{tok_str}[/white]"
                    f"  [cyan]{lat_str}[/cyan]"
                    f"  [dim]{seq_str}[/dim]"
                )
            lines.append(f"  {row}")

        # Summary footer
        lines.append("")
        lines.append("  " + "─" * 64)

        avg_latency = (
            sum(r.latency_ms for r in history) / len(history) if history else 0.0
        )
        lines.append(
            f"  [dim]Total tokens:[/dim] [white]{total}[/white]"
            f"   [dim]Avg latency:[/dim] [white]{avg_latency:.1f} ms[/white]"
        )

        if total > _MAX_DISPLAY:
            lines.append(
                f"  [dim](showing last {_MAX_DISPLAY} of {total} tokens)[/dim]"
            )

        lines.append("")

        # Navigation hint
        if selected_idx is not None:
            lines.append(
                "  [dim]← → navigate token history   selected: "
                f"[/dim][bold white]Token {selected_idx}[/bold white]"
            )
        else:
            lines.append("  [dim]← → to navigate token history[/dim]")

        lines.append("")
        lines.append(self._educational_note())

        return "\n".join(lines)

    @staticmethod
    def _educational_note() -> str:
        return (
            "  [dim]Note: Token latency = time from previous forward pass completion "
            "to sample selection (measured)[/dim]"
        )

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

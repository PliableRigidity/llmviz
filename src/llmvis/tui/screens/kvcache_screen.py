"""KV Cache tab screen for LLMVis deep telemetry mode."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, DeepState


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n >= 1_073_741_824:
        return f"{n / 1_073_741_824:.1f} GB"
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f} MB"
    if n >= 1_024:
        return f"{n / 1_024:.1f} KB"
    return f"{n} B"


def _progress_bar(current: int, total: int, width: int = 16) -> str:
    """Return a filled/empty Unicode block progress bar string."""
    if total <= 0:
        fraction = 0.0
    else:
        fraction = min(1.0, current / total)
    filled = round(fraction * width)
    empty = width - filled
    return "█" * filled + "░" * empty


class KvCacheScreen(Static):
    """KV CACHE tab — shows live KV cache state from deep instrumentation.

    Displayed when using InstrumentedTransformersAdapter (deep telemetry mode).
    Falls back to an idle/educational state when no KV data is available.
    """

    DEFAULT_CSS = """
    KvCacheScreen {
        background: #0d1117;
        height: auto;
        padding: 0;
    }
    """

    # Context-window maximum used when arch info is not yet available.
    _FALLBACK_MAX_TOKENS = 4096

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    # ------------------------------------------------------------------
    # Textual rendering
    # ------------------------------------------------------------------

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    # ------------------------------------------------------------------
    # State update (called by parent on each refresh cycle)
    # ------------------------------------------------------------------

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

    # ------------------------------------------------------------------
    # Content builder
    # ------------------------------------------------------------------

    def _build(self, state: AppState) -> str:
        lines: list[str] = []
        lines.append(
            "[bold cyan]"
            "─ KV CACHE "
            "──────────"
            "──────────"
            "──────────"
            "──────────"
            "──────────"
            "───────"
            "[/bold cyan]"
        )
        lines.append("")

        deep: DeepState | None = state.deep
        has_kv = (
            deep is not None
            and deep.current_kv_seq_len > 0
        )

        if has_kv:
            self._build_active(lines, deep)  # type: ignore[arg-type]
        else:
            self._build_idle(lines, deep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Active state (KV data available)
    # ------------------------------------------------------------------

    def _build_active(self, lines: list[str], deep: DeepState) -> None:
        arch = deep.arch
        seq_len = deep.current_kv_seq_len
        seq_len_before = seq_len - 1 if seq_len > 0 else 0

        # Maximum context window: not stored on arch, use fallback.
        max_tokens: int = self._FALLBACK_MAX_TOKENS
        num_layers = arch.num_layers if (arch is not None and arch.num_layers > 0) else 0

        # ── Sequence length progress bar ──────────────────────────────────
        bar = _progress_bar(seq_len, max_tokens, width=16)
        lines.append(
            f"  [dim]Tokens[/dim]  "
            f"[green]\[{bar}][/green]"
            f"  [white]{seq_len:,}[/white] [dim]/ {max_tokens:,}[/dim]"
        )
        lines.append("")

        # ── Sequence transition ───────────────────────────────────────────
        if deep.is_generating and seq_len_before >= 0:
            lines.append(
                f"  [dim]Sequence:[/dim]  "
                f"[white]{seq_len_before}[/white] "
                f"[dim]→[/dim] "
                f"[bold white]{seq_len}[/bold white]"
            )
        else:
            lines.append(
                f"  [dim]Sequence:[/dim]  [white]{seq_len}[/white]"
            )

        # ── Layer count ───────────────────────────────────────────────────
        if num_layers > 0:
            lines.append(f"  [dim]Layers:  [/dim]  [white]{num_layers}[/white]")
        else:
            lines.append("  [dim]Layers:    —[/dim]")

        lines.append("")

        # ── K/V shape ─────────────────────────────────────────────────────
        # DeepState does not cache k_shape directly; it is carried per token
        # in token_history. Pull from the most recent record if available.
        k_shape_str = self._get_k_shape_str(deep)
        if k_shape_str:
            lines.append(
                f"  [dim]K/V shape:[/dim]  [white]{k_shape_str}[/white]"
            )
            lines.append("")

        # ── Memory ───────────────────────────────────────────────────────
        lines.append("  [dim]Memory:[/dim]")
        measured = deep.current_kv_bytes
        if measured > 0:
            lines.append(
                f"    [dim]Measured:[/dim]  "
                f"[bold yellow]{_fmt_bytes(measured)}[/bold yellow]"
            )
            lines.append(
                "    [dim](from actual tensor byte sizes, not estimated)[/dim]"
            )
        else:
            lines.append("    [dim]Measured:  —[/dim]")

        lines.append("")

        # ── Educational footer ────────────────────────────────────────────
        self._add_educational(lines)

    # ------------------------------------------------------------------
    # Idle / no-data state
    # ------------------------------------------------------------------

    def _build_idle(self, lines: list[str], deep: DeepState | None) -> None:
        if deep is None:
            lines.append("  [dim]Deep telemetry not active.[/dim]")
            lines.append("")
            lines.append(
                "  [dim]KV cache data is only available when using[/dim]"
            )
            lines.append(
                "  [dim]InstrumentedTransformersAdapter (deep mode).[/dim]"
            )
        else:
            # Deep mode is active but no tokens generated yet.
            lines.append("  [dim]Waiting for first token...[/dim]")
            lines.append("")
            lines.append(
                "  [dim]KV cache will appear here once generation starts.[/dim]"
            )

        lines.append("")
        self._add_educational(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_k_shape_str(deep: DeepState) -> str:
        """Return K/V shape string from the actual KvCacheUpdateEvent data."""
        k = deep.last_k_shape
        v = deep.last_v_shape
        dtype = deep.last_kv_dtype
        if k and v:
            return f"K {k}  V {v}" + (f"  [{dtype}]" if dtype else "")
        return ""

    @staticmethod
    def _add_educational(lines: list[str]) -> None:
        lines.append(
            "[dim]KV cache stores attention keys and values from prior tokens.[/dim]"
        )
        lines.append(
            "[dim]It grows linearly with sequence length and is discarded "
            "when the session ends.[/dim]"
        )

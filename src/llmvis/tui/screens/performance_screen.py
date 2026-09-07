"""Performance metrics screen — prefill and decode timing panel."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, DeepState, InferencePhase


class PerformanceScreen(Static):
    """Shows measured prefill and decode performance metrics."""

    CSS = """
    PerformanceScreen {
        border: solid #30363d;
        padding: 0 1;
        height: auto;
        min-height: 16;
        background: #0d1117;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    def _build(self, state: AppState) -> str:
        lines = [
            "[bold cyan]─ PERFORMANCE ────────────────────────────────────────────────────[/bold cyan]"
        ]
        lines.append("")

        deep = state.deep
        if deep is None:
            lines.append("  [dim]Performance data requires deep instrumentation mode.[/dim]")
            lines.append("  [dim]Use the InstrumentedTransformersAdapter to enable metrics.[/dim]")
            lines.append("")
            lines.append("[dim]Prefill: all prompt tokens processed in parallel.[/dim]")
            lines.append("[dim]Decode: one token per forward pass. These timings are measured, not estimated.[/dim]")
            return "\n".join(lines)

        # --- PREFILL ---
        lines.append("  [bold]PREFILL[/bold]")

        prefill_tokens = deep.prefill_token_count
        prefill_ms = deep.prefill_duration_ms

        bar = self._progress_bar(1.0 if prefill_tokens > 0 else 0.0, width=30)
        pct = "100%" if prefill_tokens > 0 else "  0%"
        lines.append(f"  [{bar}]  {pct}")
        lines.append(f"  [{prefill_tokens}] prompt tokens")

        if prefill_tokens > 0 and prefill_ms > 0.0:
            tok_per_s = prefill_tokens / (prefill_ms / 1000.0)
            lines.append(f"  {prefill_ms:.1f} ms  →  {tok_per_s:.1f} tok/s")
        elif prefill_tokens > 0:
            lines.append("  [dim]timing not available[/dim]")
        else:
            lines.append("  [dim]waiting for prefill...[/dim]")

        lines.append("")

        # --- DECODE ---
        lines.append("  [bold]DECODE[/bold]")

        decode_tokens = deep.decode_token_count
        decode_tok_per_s = deep.decode_tokens_per_sec

        lines.append(f"  Tokens generated: [{decode_tokens}]")

        token_history = list(deep.token_history)
        if token_history:
            latencies = [rec.latency_ms for rec in token_history]
            avg_latency = sum(latencies) / len(latencies)
            lines.append(f"  Avg latency:      [{avg_latency:.1f}] ms/tok")
        else:
            lines.append("  Avg latency:      [dim]—[/dim]")

        if decode_tok_per_s > 0.0:
            lines.append(f"  Throughput:       [{decode_tok_per_s:.1f}] tok/s")
        else:
            lines.append("  Throughput:       [dim]—[/dim]")

        # Recent latency sparkline (last 5 tokens)
        if token_history:
            recent_records = token_history[-5:]
            recent_parts = "  ".join(f"{rec.latency_ms:.0f}ms" for rec in recent_records)
            lines.append(f"  Recent: {recent_parts}")

        # Total inference time
        total_ms = deep.total_inference_time_ms
        if total_ms > 0.0:
            lines.append(f"  Total inference time: [{total_ms:.1f}] ms")
        else:
            lines.append("  Total inference time: [dim]—[/dim]")

        lines.append("")
        lines.append("[dim]Prefill: all prompt tokens processed in parallel.[/dim]")
        lines.append("[dim]Decode: one token per forward pass. These timings are measured, not estimated.[/dim]")

        return "\n".join(lines)

    def _progress_bar(self, fraction: float, width: int = 30) -> str:
        """Return a filled bar string of the given width."""
        fraction = max(0.0, min(1.0, fraction))
        filled = int(round(fraction * width))
        return "█" * filled + "░" * (width - filled)

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

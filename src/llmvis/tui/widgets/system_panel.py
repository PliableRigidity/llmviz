"""System resource panel (CPU, RAM, GPU, VRAM)."""

from __future__ import annotations

from textual.widgets import Static

from llmvis.core.state import AppState


def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _bar(pct: float, width: int = 14) -> str:
    """Render a Unicode block progress bar."""
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100 * width)
    empty = width - filled

    if pct >= 90:
        color = "red"
    elif pct >= 70:
        color = "yellow"
    else:
        color = "green"

    filled_str = "█" * filled
    empty_str = "░" * empty
    return f"[{color}]{filled_str}[/{color}][dim]{empty_str}[/dim]"


class SystemPanel(Static):
    """Displays CPU, RAM, GPU, and VRAM metrics."""

    CSS = """
    SystemPanel {
        border: solid #30363d;
        padding: 0 1;
        height: auto;
        min-height: 12;
        background: #0d1117;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def render(self) -> str:
        return self._build(self._state)

    def _build(self, state: AppState) -> str:
        lines = ["[bold cyan]─ SYSTEM ────────────────────────────[/bold cyan]"]
        lines.append("")
        s = state.system
        g = state.gpu

        def metric_row(label: str, pct: float, used_str: str, note: str = "") -> str:
            bar = _bar(pct)
            pct_str = f"{pct:5.1f}%"
            note_part = f" [dim]{note}[/dim]" if note else ""
            return f"  [dim]{label:<6}[/dim] {bar} {pct_str}  [dim]{used_str}[/dim]{note_part}"

        lines.append(metric_row("CPU", s.cpu_percent, "", "system"))

        if s.ram_total_bytes:
            ram_str = f"{_fmt_bytes(s.ram_used_bytes)} / {_fmt_bytes(s.ram_total_bytes)}"
            lines.append(metric_row("RAM", s.ram_percent, ram_str, "measured"))
        else:
            lines.append("  [dim]RAM    unavailable[/dim]")

        if s.ollama_ram_bytes:
            lines.append(
                f"  [dim]Ollama[/dim] CPU [white]{s.ollama_cpu_percent:5.1f}%[/white]"
                f"  RAM [white]{_fmt_bytes(s.ollama_ram_bytes)}[/white]"
            )

        lines.append("")

        if g.available:
            lines.append(f"  [dim]GPU   {g.gpu_name}[/dim]")
            lines.append(metric_row("GPU", g.gpu_util_percent, "", "measured"))
            if g.vram_total_bytes:
                vram_str = f"{_fmt_bytes(g.vram_used_bytes)} / {_fmt_bytes(g.vram_total_bytes)}"
                lines.append(metric_row("VRAM", g.vram_percent, vram_str, "measured"))
            if g.temperature_c is not None:
                lines.append(f"  [dim]Temp  [/dim] [white]{g.temperature_c:.0f}°C[/white]")
        else:
            lines.append("  [dim]GPU    No GPU telemetry[/dim]")
            lines.append("  [dim]       pip install llmvis[nvidia][/dim]")

        return "\n".join(lines)

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

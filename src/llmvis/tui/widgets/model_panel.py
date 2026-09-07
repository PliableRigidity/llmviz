"""Model information panel."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, InferencePhase, OllamaStatus


def _fmt_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


class ModelPanel(Static):
    """Displays active model metadata."""

    CSS = """
    ModelPanel {
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

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    def _build(self, state: AppState) -> str:
        lines = ["[bold cyan]─ MODEL ─────────────────────────────[/bold cyan]"]

        # Deep instrumentation mode: show real arch info instead of Ollama state.
        if state.deep is not None:
            return self._build_deep(state, lines)

        # Stock Ollama observer mode.
        if not state.active_model:
            if state.ollama_status == OllamaStatus.CONNECTED:
                lines.append("")
                lines.append("  [dim]No model loaded.[/dim]")
                lines.append("  [dim]Run: ollama run <model>[/dim]")
                if state.installed_models:
                    lines.append("")
                    lines.append(f"  [dim]{len(state.installed_models)} model(s) installed.[/dim]")
            elif state.ollama_status == OllamaStatus.DISCONNECTED:
                lines.append("")
                lines.append("  [dim]Ollama not detected.[/dim]")
                lines.append("  [dim]Start Ollama to continue.[/dim]")
            elif state.ollama_status == OllamaStatus.RECONNECTING:
                lines.append("")
                lines.append("  [yellow]Reconnecting to Ollama...[/yellow]")
            else:
                lines.append("")
                lines.append("  [dim]Connecting...[/dim]")
            return "\n".join(lines)

        m = state.active_model
        lines.append("")

        def row(label: str, value: str, note: str = "") -> str:
            note_str = f" [dim]{note}[/dim]" if note else ""
            return f"  [dim]{label:<14}[/dim] [white]{value}[/white]{note_str}"

        lines.append(row("Model", m.name))
        if m.family:
            lines.append(row("Family", m.family))
        if m.parameter_size:
            lines.append(row("Parameters", m.parameter_size))
        if m.quantization_level:
            lines.append(row("Quantization", m.quantization_level))
        if m.format:
            lines.append(row("Format", m.format.upper()))
        if m.context_length:
            lines.append(row("Context", f"{m.context_length:,} tokens"))
        if m.embedding_length:
            lines.append(row("Embedding", f"{m.embedding_length:,} dims"))
        if m.size_bytes:
            lines.append(row("Disk size", _fmt_bytes(m.size_bytes)))
        if m.size_vram_bytes:
            lines.append(row("VRAM usage", _fmt_bytes(m.size_vram_bytes), "measured"))
        if m.capabilities:
            lines.append(row("Capabilities", ", ".join(m.capabilities)))

        if len(state.loaded_models) > 1:
            lines.append("")
            lines.append(f"  [dim]+{len(state.loaded_models)-1} other model(s) loaded[/dim]")

        return "\n".join(lines)

    def _build_deep(self, state: AppState, lines: list[str]) -> str:
        """Render deep instrumentation model info from DeepState.arch."""
        d = state.deep
        arch = d.arch

        lines.append("")

        def row(label: str, value: str, note: str = "") -> str:
            note_str = f" [dim]{note}[/dim]" if note else ""
            return f"  [dim]{label:<14}[/dim] [white]{value}[/white]{note_str}"

        # Show loading / connecting state when no arch info yet.
        if not d.model_loaded and not arch.model_id:
            if d.load_error:
                lines.append(f"  [red]Error: {d.load_error}[/red]")
            elif d.phase == InferencePhase.LOADING:
                lines.append("  [yellow]Loading model...[/yellow]")
            else:
                lines.append("  [dim]Connecting to deep backend...[/dim]")
            return "\n".join(lines)

        if arch.model_id:
            lines.append(row("Model", arch.model_id))

        device = arch.device.upper() if arch.device else ""
        if device:
            lines.append(row("Backend", f"PyTorch / {device}"))
            lines.append(row("Device", arch.device))
        else:
            lines.append(row("Backend", "PyTorch"))

        if arch.num_layers:
            lines.append(row("Layers", str(arch.num_layers)))
        if arch.num_attention_heads:
            lines.append(row("Heads", str(arch.num_attention_heads)))
        if arch.hidden_size:
            lines.append(row("Hidden size", f"{arch.hidden_size:,}"))
        if arch.vocab_size:
            lines.append(row("Vocab size", f"{arch.vocab_size:,}"))
        if arch.dtype:
            lines.append(row("dtype", arch.dtype))

        lines.append(row("Mode", "Deep Instrumentation"))

        return "\n".join(lines)

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

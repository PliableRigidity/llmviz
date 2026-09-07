"""Inference state panel — honest about what is and isn't observable."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, InferenceStatus, OllamaStatus


class InferencePanel(Static):
    """Shows inference status with honest data/estimate labeling."""

    CSS = """
    InferencePanel {
        border: solid #30363d;
        padding: 0 1;
        height: auto;
        min-height: 10;
        background: #0d1117;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    def _build(self, state: AppState) -> str:
        lines = ["[bold cyan]─ INFERENCE ─────────────────────────────────────────[/bold cyan]"]
        lines.append("")

        if state.ollama_status != OllamaStatus.CONNECTED:
            lines.append("  [dim]Waiting for Ollama connection...[/dim]")
            lines.append("")
            self._add_deep_telemetry_note(lines)
            return "\n".join(lines)

        if not state.active_model:
            lines.append("  [dim]No model loaded.[/dim]")
            lines.append("  [dim]Load a model with: ollama run <model>[/dim]")
            lines.append("")
            self._add_deep_telemetry_note(lines)
            return "\n".join(lines)

        match state.inference_status:
            case InferenceStatus.POSSIBLY_ACTIVE:
                status = "[bold green]ACTIVITY DETECTED[/bold green] [dim](estimated)[/dim]"
                detail = "GPU utilization elevated — inference possibly active in another terminal."
            case _:
                status = "[bold blue]IDLE[/bold blue]"
                detail = "Waiting for inference activity..."

        lines.append(f"  Status:  {status}")
        lines.append(f"  [dim]{detail}[/dim]")
        lines.append("")

        g = state.gpu
        if g.available and state.gpu_history:
            recent = state.gpu_history[-20:]
            avg = sum(recent) / len(recent)
            peak = max(recent)
            lines.append(
                f"  [dim]GPU util  avg [/dim][white]{avg:5.1f}%[/white]"
                f"  [dim]peak [/dim][white]{peak:5.1f}%[/white]"
                f"  [dim](estimated)[/dim]"
            )

        lines.append("")
        lines.append(
            "  [dim]Note: Stock Ollama does not expose another client's token stream.[/dim]"
        )
        lines.append(
            "  [dim]Activity is inferred from GPU/CPU resource usage only.[/dim]"
        )
        lines.append("")
        self._add_deep_telemetry_note(lines)

        return "\n".join(lines)

    def _add_deep_telemetry_note(self, lines: list[str]) -> None:
        lines.append("  [dim]DEEP MODEL TELEMETRY — Unavailable with stock Ollama[/dim]")
        lines.append(
            "  [dim]Future adapters can expose: layers · activations · logits · KV cache[/dim]"
        )

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

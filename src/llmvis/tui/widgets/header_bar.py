"""Top header bar showing connection status and active model."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, OllamaStatus


class HeaderBar(Static):
    """Top status bar: connection + active model summary."""

    CSS = """
    HeaderBar {
        height: 3;
        background: #161b22;
        border-bottom: solid #30363d;
        padding: 0 1;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def render(self) -> Text:
        return Text.from_markup(self._build_text(self._state))

    def _build_text(self, state: AppState) -> str:
        if state.deep is not None:
            return self._build_deep_text(state)
        return self._build_ollama_text(state)

    def _build_ollama_text(self, state: AppState) -> str:
        match state.ollama_status:
            case OllamaStatus.CONNECTED:
                conn = "[bold green]● CONNECTED[/bold green]"
            case OllamaStatus.DISCONNECTED:
                conn = "[bold red]● DISCONNECTED[/bold red]"
            case OllamaStatus.RECONNECTING:
                conn = "[bold yellow]◌ RECONNECTING[/bold yellow]"
            case _:
                conn = "[dim]● CONNECTING...[/dim]"

        ver = f"[dim]v{state.ollama_version}[/dim]" if state.ollama_version else ""

        model_str = ""
        if state.active_model:
            m = state.active_model
            parts = [m.name]
            if m.quantization_level:
                parts.append(m.quantization_level)
            if m.context_length:
                parts.append(f"ctx {m.context_length:,}")
            model_str = " [dim]│[/dim] " + " [dim]|[/dim] ".join(parts)
        elif state.ollama_status == OllamaStatus.CONNECTED:
            model_str = " [dim]│ No model loaded[/dim]"
        elif state.ollama_status == OllamaStatus.DISCONNECTED:
            model_str = f" [dim]│ {state.connection_error or 'Start Ollama to connect'}[/dim]"

        line1 = f"[bold cyan]LLMVIS[/bold cyan]  Ollama {conn} {ver}{model_str}"
        line2 = f"[dim]  {state.ollama_host}[/dim]"
        return f"{line1}\n{line2}"

    def _build_deep_text(self, state: AppState) -> str:
        match state.ollama_status:
            case OllamaStatus.CONNECTED:
                conn = "[bold green]● CONNECTED[/bold green]"
            case OllamaStatus.DISCONNECTED:
                conn = "[bold red]● DISCONNECTED[/bold red]"
            case OllamaStatus.RECONNECTING:
                conn = "[bold yellow]◌ RECONNECTING[/bold yellow]"
            case _:
                conn = "[dim]● CONNECTING...[/dim]"

        d = state.deep
        arch = d.arch if d else None
        model_id = arch.model_id if arch and arch.model_id else ""
        device = arch.device.upper() if arch and arch.device else ""

        if device:
            backend = f"PyTorch / {device}"
        else:
            backend = "Deep Instrumentation"

        line1 = f"[bold cyan]LLMVIS[/bold cyan]  Deep Instrumentation {conn}"
        if model_id:
            line2 = f"[dim]  {backend}  │  {model_id}[/dim]"
        else:
            line2 = f"[dim]  {backend}[/dim]"
        return f"{line1}\n{line2}"

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

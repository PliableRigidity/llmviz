from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.state import AppState, DeepState, InferencePhase, LayerSummary

_HEADER = "─ TRANSFORMER ─────────────────────────────────────────────────────"

_BAR_CHARS = "█"
_BAR_EMPTY = "░"
_BAR_WIDTH = 12
_RMS_MAX = 2.0

_MAX_LAYERS_DISPLAYED = 32


def _build_bar(rms: float, delta: float) -> tuple[str, str]:
    """Return (bar_string, color) based on RMS magnitude and delta."""
    filled = min(_BAR_WIDTH, int(round((rms / _RMS_MAX) * _BAR_WIDTH)))
    bar = _BAR_CHARS * filled + _BAR_EMPTY * (_BAR_WIDTH - filled)
    if delta < 0.1:
        color = "green"
    elif delta < 0.3:
        color = "yellow"
    else:
        color = "cyan"
    return bar, color


def _phase_label(phase: InferencePhase) -> str:
    return phase.name  # e.g. "PREFILLING", "DECODING"


class TransformerScreen(Static):
    DEFAULT_CSS = """
    TransformerScreen {
        border: solid #30363d;
        background: #0d1117;
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._state: AppState | None = None

    def render(self) -> Text:
        lines: list[str] = []

        lines.append(f"[bold white]{_HEADER}[/bold white]")
        lines.append("")

        deep: DeepState | None = self._state.deep if self._state is not None else None

        if deep is None:
            lines.append("  [dim]No deep instrumentation active.[/dim]")
            lines.append("  [dim]Run: llmvis instrument <model_id>[/dim]")
            lines.append("")
            lines.append(
                "  [dim]When active this screen shows live layer-by-layer hidden-state[/dim]"
            )
            lines.append(
                "  [dim]RMS magnitudes, inter-layer deltas, and per-layer execution time[/dim]"
            )
            lines.append(
                "  [dim]measured directly from model tensors during generation.[/dim]"
            )
            return Text.from_markup("\n".join(lines))

        # Phase header
        phase_label = _phase_label(deep.phase)
        lines.append(f"  Phase: [bold]{phase_label}[/bold]")

        if deep.phase == InferencePhase.DECODING:
            lines.append(f"  Token: [bold]{deep.current_token_index}[/bold]")

        lines.append("")

        num_layers = deep.arch.num_layers if deep.arch.num_layers > 0 else 0
        displayed = min(num_layers, _MAX_LAYERS_DISPLAYED)

        # Build a fast lookup: layer_index -> LayerSummary for current token
        summary_map: dict[int, LayerSummary] = {}
        for s in deep.current_layer_summaries:
            summary_map[s.layer_index] = s

        for i in range(displayed):
            # Determine status indicator
            if i == deep.current_layer:
                indicator = "[bold yellow]▶[/bold yellow]"
            elif deep.current_layer >= 0 and i < deep.current_layer:
                indicator = "[green]✓[/green]"
            else:
                indicator = "[dim]○[/dim]"

            layer_label = f"L{i + 1:02d}"

            summary: LayerSummary | None = summary_map.get(i)

            if summary is None:
                # Layer not yet reached for this token — grey placeholder
                placeholder_bar = _BAR_EMPTY * _BAR_WIDTH
                line = (
                    f"  {indicator} [dim]{layer_label}[/dim] "
                    f"[dim]{placeholder_bar}[/dim]  "
                    f"[dim]RMS --.-  Δ +-.--  --.--ms[/dim]"
                )
            else:
                bar, color = _build_bar(summary.hidden_state_rms, summary.delta_from_prev)
                delta_sign = "+" if summary.delta_from_prev >= 0 else ""
                rms_str = f"{summary.hidden_state_rms:.2f}"
                delta_str = f"{delta_sign}{summary.delta_from_prev:.2f}"
                time_str = f"{summary.exec_time_ms:.2f}"

                extras = ""
                if summary.mlp_output_rms is not None:
                    extras = f"  [dim]MLP {summary.mlp_output_rms:.2f}[/dim]"

                line = (
                    f"  {indicator} {layer_label} "
                    f"[{color}]{bar}[/{color}]  "
                    f"RMS [bold]{rms_str}[/bold]  "
                    f"Δ [bold]{delta_str}[/bold]  "
                    f"[dim]{time_str}ms[/dim]"
                    f"{extras}"
                )

            lines.append(line)

        if num_layers > _MAX_LAYERS_DISPLAYED:
            remaining = num_layers - _MAX_LAYERS_DISPLAYED
            lines.append(f"  [dim]... and {remaining} more layers (display capped at {_MAX_LAYERS_DISPLAYED})[/dim]")

        lines.append("")
        lines.append(
            "[dim]Bars show hidden-state RMS magnitude "
            "(measured from actual tensors, not inferred)[/dim]"
        )

        return Text.from_markup("\n".join(lines))

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

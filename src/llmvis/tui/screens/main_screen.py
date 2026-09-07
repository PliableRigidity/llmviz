"""Main view for LLMVis (mounted as a regular widget inside the app)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical

from llmvis.core.state import AppState
from llmvis.tui.widgets.header_bar import HeaderBar
from llmvis.tui.widgets.inference_panel import InferencePanel
from llmvis.tui.widgets.model_panel import ModelPanel
from llmvis.tui.widgets.pipeline_panel import PipelinePanel
from llmvis.tui.widgets.status_bar import StatusBar
from llmvis.tui.widgets.system_panel import SystemPanel


class MainScreen(Vertical):
    """Primary LLMVis layout — a Vertical container, not a Textual Screen.

    Mounted directly in App.compose() so query_one(MainScreen) finds it.
    ConceptScreen is the only true Textual Screen (pushed as a modal overlay).
    """

    DEFAULT_CSS = """
    MainScreen {
        background: #0d1117;
        height: 1fr;
    }

    #body {
        height: 1fr;
        overflow-y: auto;
    }

    #mid-row {
        layout: horizontal;
        height: auto;
        min-height: 12;
    }

    #model-panel {
        width: 1fr;
        min-width: 30;
    }

    #system-panel {
        width: 1fr;
        min-width: 30;
    }

    #inference-panel {
        height: auto;
        min-height: 10;
    }

    #pipeline-panel {
        height: 5;
    }
    """

    def __init__(self, state: AppState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield HeaderBar(self._state, id="header")
        with Vertical(id="body"):
            with Horizontal(id="mid-row"):
                yield ModelPanel(self._state, id="model-panel")
                yield SystemPanel(self._state, id="system-panel")
            yield InferencePanel(self._state, id="inference-panel")
            yield PipelinePanel(id="pipeline-panel")
        yield StatusBar(id="status-bar")

    def refresh_state(self, state: AppState) -> None:
        self._state = state
        self._update_widgets(state)

    def _update_widgets(self, state: AppState) -> None:
        try:
            self.query_one(HeaderBar).update_state(state)
            self.query_one(ModelPanel).update_state(state)
            self.query_one(SystemPanel).update_state(state)
            self.query_one(InferencePanel).update_state(state)
        except Exception:
            pass

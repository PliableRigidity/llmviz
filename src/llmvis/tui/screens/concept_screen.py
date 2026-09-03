"""Educational concept overlay screen."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, Static

from llmvis.educational.concepts import CONCEPTS, get_concept


class ConceptScreen(ModalScreen):
    """Modal overlay showing educational concept explanations."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
    ]

    CSS = """
    ConceptScreen {
        align: center middle;
    }

    #concept-container {
        width: 70;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        background: #161b22;
        border: solid #30363d;
        padding: 1 2;
    }

    #concept-title {
        color: #58a6ff;
        text-style: bold;
        margin-bottom: 1;
    }

    #concept-body {
        color: #c9d1d9;
    }
    """

    def __init__(self, initial_key: str = "context_window") -> None:
        super().__init__()
        self._key = initial_key

    def compose(self) -> ComposeResult:
        concept = get_concept(self._key) or list(CONCEPTS.values())[0]
        with Vertical(id="concept-container"):
            yield Label(f"  {concept.name}", id="concept-title")
            yield Static(concept.full, id="concept-body")
            yield Label("\n[dim]Press Escape or Q to close[/dim]")

    def action_dismiss(self) -> None:
        self.dismiss()

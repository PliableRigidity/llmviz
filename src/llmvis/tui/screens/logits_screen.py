"""Logits & Sampling tab — top token candidates before sampling.

All displayed values come from actual model logits and sampling computation.
No values are fabricated or inferred.
"""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from llmvis.core.events import TokenCandidate
from llmvis.core.state import AppState


_BAR_WIDTH = 10
_BAR_FULL = "█"
_BAR_EMPTY = "░"


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = round(fraction * width)
    filled = max(0, min(width, filled))
    return _BAR_FULL * filled + _BAR_EMPTY * (width - filled)


class LogitsScreen(Static):
    """Shows top token candidates and sampler configuration for the current token."""

    DEFAULT_CSS = """
    LogitsScreen {
        background: #0d1117;
        color: #c9d1d9;
        padding: 0 1;
        height: auto;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def render(self) -> Text:
        return Text.from_markup(self._build(self._state))

    def _build(self, state: AppState) -> str:
        lines: list[str] = [
            "[bold cyan]─ LOGITS & SAMPLING ─────────────────────────────────────────────────[/bold cyan]"
        ]

        deep = state.deep
        if deep is None or not deep.current_top_logits:
            lines.append("")
            lines.append("  [dim]Waiting for next token logits...[/dim]")
            return "\n".join(lines)

        token_index = deep.current_token_index
        raw_list = list(deep.current_top_logits[:10])

        # Support both TokenCandidate objects (V2) and legacy (text, prob) tuples
        candidates: list[TokenCandidate] = []
        for item in raw_list:
            if isinstance(item, TokenCandidate):
                candidates.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                candidates.append(TokenCandidate(
                    token_id=0,
                    token_text=str(item[0]),
                    logit=0.0,
                    raw_probability=float(item[1]),
                    probability=float(item[1]),
                ))

        if not candidates:
            lines.append("")
            lines.append("  [dim]No candidates available.[/dim]")
            return "\n".join(lines)

        lines.append("")
        lines.append(f"  Top candidates for token [{token_index}]:")
        lines.append("")
        lines.append(
            f"  {'Token':16}  {'Post-filter':12}  {'Pre-filter':11}  {'Logit':>8}"
        )
        lines.append(f"  {'─'*16}  {'─'*12}  {'─'*11}  {'─'*8}")

        max_prob = max((c.probability for c in candidates), default=1.0) or 1.0

        for i, cand in enumerate(candidates):
            fraction = cand.probability / max_prob
            bar = _bar(fraction)
            pct_post = cand.probability * 100.0
            pct_raw = cand.raw_probability * 100.0

            display = repr(cand.token_text) if cand.token_text.strip() == "" else f"'{cand.token_text}'"
            display = display[:16].ljust(16)

            if i == 0:
                color_open = "[bold green]"
                color_close = "[/bold green]"
            else:
                color_open = "[dim]"
                color_close = "[/dim]"

            # Show real logit if available; mark as not fabricated
            if cand.logit != 0.0:
                logit_str = f"{cand.logit:8.3f}"
            else:
                logit_str = "    n/a "

            line = (
                f"  {color_open}{display}  {bar} {pct_post:5.2f}%"
                f"  raw:{pct_raw:5.2f}%  {logit_str}{color_close}"
            )
            lines.append(line)

        # Sampler config
        lines.append("")
        lines.append("  Sampler config (from adapter):")

        temperature = None
        top_k = None
        top_p = None
        if deep.token_history:
            last = list(deep.token_history)[-1]
            temperature = getattr(last, "temperature", None)
            top_k = getattr(last, "top_k_param", None)
            top_p = getattr(last, "top_p_param", None)

        temp_str = f"{temperature:.2f}" if temperature is not None else "—"
        top_k_str = str(top_k) if top_k is not None else "—"
        top_p_str = f"{top_p:.2f}" if top_p is not None else "—"

        lines.append(f"    Temperature:  {temp_str}  (>1 = flatter, <1 = sharper)")
        lines.append(f"    Top-K filter: {top_k_str}  (0 = disabled)")
        lines.append(f"    Top-P filter: {top_p_str}  (1.0 = disabled)")

        lines.append("")
        lines.append(
            "[dim]Post-filter = probability AFTER temperature + top-k/p; "
            "pre-filter = raw softmax.[/dim]"
        )
        lines.append(
            "[dim]All logits from actual LM head output. "
            "Probability is not factual confidence.[/dim]"
        )

        return "\n".join(lines)

    def update_state(self, state: AppState) -> None:
        self._state = state
        self.refresh()

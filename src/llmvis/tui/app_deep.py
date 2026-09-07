"""LLMVis deep instrumentation Textual application.

Runs when `llmvis instrument <model_id>` is used. Loads a local HuggingFace
model, instruments it with PyTorch forward hooks, and visualizes actual runtime
data in a 6-tab TUI.

Keybindings:
    1   Overview (system + model info)
    2   Transformer (live layer traversal)
    3   Tokens (generation history)
    4   Logits (top candidates before sampling)
    5   KV Cache (sequence length + memory)
    6   Performance (prefill/decode timing)
    left   Select previous token in history
    right  Select next token in history
    q      Quit
"""

from __future__ import annotations

import asyncio
import logging

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.widgets import ContentSwitcher, Input, Label, Static

from llmvis.adapters.base import BaseAdapter
from llmvis.core.bus import EventBus
from llmvis.core.events import (
    BackendConnectedEvent,
    BackendDisconnectedEvent,
    EventType,
    GpuStatsEvent,
    InferenceEndEvent,
    InferenceStartEvent,
    KvCacheUpdateEvent,
    LayerStatsEvent,
    LogitsReadyEvent,
    ModelArchitectureEvent,
    PrefillEndEvent,
    PrefillStartEvent,
    SessionInfoEvent,
    SystemStatsEvent,
    TokenCandidate,
    TokenEndEvent,
    TokenSampledEvent,
    TokenStartEvent,
)
from llmvis.core.state import (
    AppState,
    InferencePhase,
    LayerSummary,
    ModelArchInfo,
    OllamaStatus,
    TokenRecord,
    make_deep_state,
    make_initial_state,
)
from llmvis.telemetry.gpu import get_gpu_stats
from llmvis.telemetry.system import SystemTelemetry
from llmvis.tui.screens.kvcache_screen import KvCacheScreen
from llmvis.tui.screens.logits_screen import LogitsScreen
from llmvis.tui.screens.performance_screen import PerformanceScreen
from llmvis.tui.screens.tokens_screen import TokensScreen
from llmvis.tui.screens.transformer_screen import TransformerScreen
from llmvis.tui.widgets.header_bar import HeaderBar
from llmvis.tui.widgets.model_panel import ModelPanel
from llmvis.tui.widgets.status_bar import StatusBar
from llmvis.tui.widgets.system_panel import SystemPanel

logger = logging.getLogger(__name__)

TELEMETRY_INTERVAL = 1.5


class TabBar(Static):
    """Simple horizontal tab indicator bar."""

    DEFAULT_CSS = """
    TabBar {
        height: 1;
        background: #161b22;
        color: #8b949e;
        padding: 0 1;
    }
    """

    def __init__(self, active: str = "1", **kwargs) -> None:
        super().__init__(**kwargs)
        self._active = active

    def set_active(self, tab: str) -> None:
        self._active = tab
        self.refresh()

    def render(self) -> Text:
        tabs = [
            ("1", "Overview"),
            ("2", "Transformer"),
            ("3", "Tokens"),
            ("4", "Logits"),
            ("5", "KV Cache"),
            ("6", "Performance"),
        ]
        parts = []
        for key, label in tabs:
            display = f"{key} {label}"
            if key == self._active:
                parts.append(f"[bold white on #1f2937] {display} [/bold white on #1f2937]")
            else:
                parts.append(f"[dim] {display} [/dim]")
        return Text.from_markup("  ".join(parts))


class OverviewPanel(Static):
    """Overview tab: system stats + model metadata side by side."""

    DEFAULT_CSS = """
    OverviewPanel {
        background: #0d1117;
        height: 1fr;
    }
    """

    def __init__(self, state: AppState, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield ModelPanel(self._state, id="ov-model")
            yield SystemPanel(self._state, id="ov-system")

    def update_state(self, state: AppState) -> None:
        self._state = state
        try:
            self.query_one(ModelPanel).update_state(state)
            self.query_one(SystemPanel).update_state(state)
        except Exception:
            pass


class LLMVisDeepApp(App):
    """LLMVis V2 — deep instrumentation mode.

    Connects to InstrumentedTransformersAdapter which runs a local HuggingFace
    model with PyTorch forward hooks and emits real runtime telemetry.
    """

    CSS = """
    Screen {
        background: #0d1117;
    }
    ContentSwitcher {
        height: 1fr;
    }
    #prompt-area {
        height: 3;
        background: #161b22;
        border-top: solid #30363d;
        padding: 0 1;
    }
    #prompt-label {
        height: 1;
        color: #8b949e;
    }
    #prompt-input {
        height: 1;
        background: #0d1117;
        border: none;
        color: #c9d1d9;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        # Navigation bindings — only active when Input does NOT have focus.
        # Using priority=False (default) so they don't steal characters typed
        # in the prompt input during INPUT MODE.
        Binding("1", "tab_1", "Overview"),
        Binding("2", "tab_2", "Transformer"),
        Binding("3", "tab_3", "Tokens"),
        Binding("4", "tab_4", "Logits"),
        Binding("5", "tab_5", "KV Cache"),
        Binding("6", "tab_6", "Performance"),
        Binding("left", "prev_token", "Prev", show=False),
        Binding("right", "next_token", "Next", show=False),
        Binding("l", "live_mode", "Live", show=False),
        Binding("space", "toggle_pause", "Pause", show=False),
        Binding("question_mark", "show_help", "Help", show=False),
        # i — enter INPUT MODE (focus the prompt field).
        # priority=True so it fires even when focus is somewhere unexpected.
        Binding("i", "focus_input", "Prompt [i]", priority=True, show=False),
    ]

    def __init__(
        self,
        adapter: BaseAdapter,
        model_id: str = "",
        telemetry_mode: str = "standard",
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._model_id = model_id
        self._telemetry_mode = telemetry_mode

        self.state: AppState = make_initial_state()
        self.state.deep = make_deep_state()
        self.state.deep.telemetry_mode = telemetry_mode
        self.state.deep.phase = InferencePhase.LOADING

        self.bus = EventBus()
        self._telemetry = SystemTelemetry()
        self._tasks: list[asyncio.Task] = []
        self._active_tab = "1"

        # Per-token staging buffers
        self._staging_layers: list[LayerSummary] = []
        self._staging_candidates: list[TokenCandidate] = []  # full candidates with real logits
        self._staging_kv_seq: int = 0
        self._staging_kv_bytes: int = 0
        self._staging_token_id: int = 0
        self._staging_token_text: str = ""
        self._staging_sampler_temp: float = 1.0
        self._staging_sampler_top_k: int = 0
        self._staging_sampler_top_p: float = 1.0

        # Pause state
        self._visualization_paused: bool = False

        # Keyboard mode: False = NAV MODE (default), True = INPUT MODE
        self._input_mode: bool = False

    def compose(self) -> ComposeResult:
        yield HeaderBar(self.state, id="header")
        yield TabBar(active="1", id="tab-bar")
        with ContentSwitcher(initial="tab-overview"):
            yield OverviewPanel(self.state, id="tab-overview")
            yield TransformerScreen(id="tab-transformer")
            yield TokensScreen(self.state, id="tab-tokens")
            yield LogitsScreen(self.state, id="tab-logits")
            yield KvCacheScreen(self.state, id="tab-kvcache")
            yield PerformanceScreen(self.state, id="tab-performance")
        with Vertical(id="prompt-area"):
            yield Label("  Prompt: [dim]press [bold]i[/bold] to type, Enter to send, Esc to cancel[/dim]", id="prompt-label", markup=True)
            yield Input(
                placeholder="Type your prompt here...",
                id="prompt-input",
            )
        yield StatusBar(deep_mode=True, id="status-bar")

    async def on_mount(self) -> None:
        # Start in NAV MODE: remove focus from the prompt input so that
        # number keys (1–6) and other shortcuts work immediately.
        try:
            self.query_one("#prompt-input", Input).blur()
        except Exception:
            pass

        # Subscribe to all deep instrumentation events
        self.bus.subscribe(EventType.MODEL_ARCHITECTURE, self._on_model_arch)
        self.bus.subscribe(EventType.INFERENCE_START, self._on_inference_start)
        self.bus.subscribe(EventType.INFERENCE_END, self._on_inference_end)
        self.bus.subscribe(EventType.PREFILL_START, self._on_prefill_start)
        self.bus.subscribe(EventType.PREFILL_END, self._on_prefill_end)
        self.bus.subscribe(EventType.TOKEN_START, self._on_token_start)
        self.bus.subscribe(EventType.LAYER_STATS, self._on_layer_stats)
        self.bus.subscribe(EventType.LOGITS_READY, self._on_logits_ready)
        self.bus.subscribe(EventType.TOKEN_SAMPLED, self._on_token_sampled)
        self.bus.subscribe(EventType.KV_CACHE_UPDATE, self._on_kv_cache_update)
        self.bus.subscribe(EventType.TOKEN_END, self._on_token_end)
        self.bus.subscribe(EventType.SYSTEM_STATS_UPDATE, self._on_system_stats)
        self.bus.subscribe(EventType.GPU_STATS_UPDATE, self._on_gpu_stats)
        self.bus.subscribe(EventType.SESSION_INFO, self._on_session_info)
        self.bus.subscribe(EventType.BACKEND_CONNECTED, self._on_backend_connected)
        self.bus.subscribe(EventType.BACKEND_DISCONNECTED, self._on_backend_disconnected)

        self._tasks = [
            asyncio.create_task(self.bus.dispatch_forever(), name="bus"),
            asyncio.create_task(self._adapter_loop(), name="adapter"),
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
        ]

        # Initialize all screens with current state and update mode indicator.
        self._refresh_all()
        self._update_status_bar()

    async def _adapter_loop(self) -> None:
        async for event in self._adapter.run():
            await self.bus.publish(event)

    async def _telemetry_loop(self) -> None:
        import psutil
        psutil.cpu_percent(interval=None)
        while True:
            await asyncio.sleep(TELEMETRY_INTERVAL)
            try:
                sys_event = await self._telemetry.get_system_stats()
                await self.bus.publish(sys_event)
                gpu_events = await get_gpu_stats()
                for ge in gpu_events:
                    await self.bus.publish(ge)
            except Exception as exc:
                logger.debug("Telemetry error: %s", exc)

    # ── Event handlers ────────────────────────────────────────────────────────

    async def _on_model_arch(self, event: ModelArchitectureEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.arch = ModelArchInfo(
            model_id=event.model_id,
            num_layers=event.num_layers,
            num_attention_heads=event.num_attention_heads,
            hidden_size=event.hidden_size,
            vocab_size=event.vocab_size,
            head_dim=event.head_dim,
            dtype=event.dtype,
            device=event.device,
        )
        d.phase = InferencePhase.IDLE
        d.model_loaded = True
        self._refresh_all()

    async def _on_inference_start(self, event: InferenceStartEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.is_generating = True
        d.phase = InferencePhase.PREFILLING
        d.prompt_text = event.prompt_preview
        d.response_text = ""
        d.current_layer_summaries = []
        d.current_layer = -1
        self._staging_layers.clear()
        self._staging_candidates.clear()
        self._refresh_all()

    async def _on_inference_end(self, event: InferenceEndEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.is_generating = False
        d.phase = InferencePhase.COMPLETE
        d.total_inference_time_ms = event.total_time_ms
        if event.total_tokens > 0 and event.total_time_ms > 0:
            d.decode_tokens_per_sec = (
                event.total_tokens / (event.total_time_ms / 1000.0)
            )
        self._refresh_all()

    async def _on_prefill_start(self, event: PrefillStartEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.phase = InferencePhase.PREFILLING
        d.prefill_token_count = event.token_count
        self._refresh_all()

    async def _on_prefill_end(self, event: PrefillEndEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.prefill_duration_ms = event.duration_ms
        d.prefill_token_count = event.token_count
        self._refresh_all()

    async def _on_token_start(self, event: TokenStartEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        d.phase = InferencePhase.DECODING
        d.current_token_index = event.token_index
        d.current_layer = -1
        d.current_layer_summaries = []
        self._staging_layers.clear()
        self._staging_candidates.clear()
        self._refresh_transformer()

    async def _on_layer_stats(self, event: LayerStatsEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        summary = LayerSummary(
            layer_index=event.layer_index,
            hidden_state_rms=event.hidden_state_rms,
            hidden_state_mean=event.hidden_state_mean,
            hidden_state_std=event.hidden_state_std,
            delta_from_prev=event.delta_from_prev,
            exec_time_ms=event.exec_time_ms,
        )
        self._staging_layers.append(summary)
        d.current_layer = event.layer_index
        d.current_layer_summaries = list(self._staging_layers)
        self._refresh_transformer()

    async def _on_logits_ready(self, event: LogitsReadyEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        # Store full TokenCandidate objects so screens can show real logit values
        self._staging_candidates = list(event.top_candidates)
        d.current_top_logits = list(self._staging_candidates)
        if not self._visualization_paused:
            self._refresh_logits()

    async def _on_token_sampled(self, event: TokenSampledEvent) -> None:
        self._staging_token_id = event.token_id
        self._staging_token_text = event.token_text
        self._staging_sampler_temp = event.sampler.temperature
        self._staging_sampler_top_k = event.sampler.top_k
        self._staging_sampler_top_p = event.sampler.top_p

    async def _on_kv_cache_update(self, event: KvCacheUpdateEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        self._staging_kv_seq = event.seq_len_after
        self._staging_kv_bytes = event.measured_bytes
        d.current_kv_seq_len = event.seq_len_after
        d.current_kv_bytes = event.measured_bytes
        d.last_k_shape = event.k_shape
        d.last_v_shape = event.v_shape
        d.last_kv_dtype = event.dtype
        self._refresh_kvcache()

    async def _on_token_end(self, event: TokenEndEvent) -> None:
        d = self.state.deep
        if d is None:
            return
        record = TokenRecord(
            index=event.token_index,
            text=event.token_text,
            token_id=event.token_id,
            latency_ms=event.latency_ms,
            layer_summaries=list(self._staging_layers),
            top_logits=list(self._staging_candidates),  # full TokenCandidate objects
            kv_seq_len=self._staging_kv_seq,
            kv_bytes=self._staging_kv_bytes,
            temperature=self._staging_sampler_temp,
            top_k_param=self._staging_sampler_top_k,
            top_p_param=self._staging_sampler_top_p,
        )
        d.token_history.append(record)
        d.decode_token_count = len(list(d.token_history))
        d.response_text += event.token_text

        # Update live indicator — always point to the latest token unless user navigated
        if d.selected_token_idx is None:
            d._is_live = True  # type: ignore[attr-defined]

        # Recompute tokens/sec from ring buffer
        history = list(d.token_history)
        if len(history) >= 2:
            total_lat = sum(r.latency_ms for r in history)
            if total_lat > 0:
                d.decode_tokens_per_sec = len(history) / (total_lat / 1000.0)

        if not self._visualization_paused:
            self._refresh_all()

    async def _on_session_info(self, event: SessionInfoEvent) -> None:
        if event.model_id:
            self._model_id = event.model_id
        d = self.state.deep
        if d is not None:
            if event.telemetry_mode:
                d.telemetry_mode = event.telemetry_mode
            # Use SESSION_INFO model_id as an early fallback so the Overview
            # shows the model name immediately, before MODEL_ARCHITECTURE replay.
            if event.model_id and not d.arch.model_id:
                d.arch.model_id = event.model_id
                d.model_loaded = True
                d.phase = InferencePhase.IDLE
        self._refresh_all()

    async def _on_backend_connected(self, event: BackendConnectedEvent) -> None:
        # Use CONNECTED so the header bar shows the green indicator.
        self.state.ollama_status = OllamaStatus.CONNECTED
        d = self.state.deep
        if d is not None:
            d.load_error = ""
            # Only reset phase if model hasn't been identified yet.
            if not d.model_loaded:
                d.phase = InferencePhase.LOADING
        self._refresh_all()

    async def _on_backend_disconnected(self, event: BackendDisconnectedEvent) -> None:
        self.state.ollama_status = OllamaStatus.RECONNECTING
        d = self.state.deep
        if d is not None:
            d.load_error = f"Reconnecting to telemetry server (port {event.port})..."
        self._refresh_all()

    async def _on_system_stats(self, event: SystemStatsEvent) -> None:
        self.state.system.cpu_percent = event.cpu_percent
        self.state.system.ram_used_bytes = event.ram_used_bytes
        self.state.system.ram_total_bytes = event.ram_total_bytes
        self.state.system.ram_percent = event.ram_percent
        self.state.system.ollama_cpu_percent = event.ollama_cpu_percent
        self.state.system.ollama_ram_bytes = event.ollama_ram_bytes
        self._refresh_overview()

    async def _on_gpu_stats(self, event: GpuStatsEvent) -> None:
        self.state.gpu.available = True
        self.state.gpu.gpu_name = event.gpu_name
        self.state.gpu.gpu_util_percent = event.gpu_util_percent
        self.state.gpu.vram_used_bytes = event.vram_used_bytes
        self.state.gpu.vram_total_bytes = event.vram_total_bytes
        self.state.gpu.vram_percent = event.vram_percent
        self.state.gpu.temperature_c = event.temperature_c
        self._refresh_overview()

    # ── Input mode (NAV MODE ↔ INPUT MODE) ───────────────────────────────────

    def action_focus_input(self) -> None:
        """i — enter INPUT MODE and focus the prompt field."""
        self._input_mode = True
        try:
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            pass
        self._update_status_bar()

    def on_key(self, event: Key) -> None:
        """Handle Escape in INPUT MODE to return to NAV MODE."""
        if event.key == "escape" and self._input_mode:
            self._input_mode = False
            try:
                self.set_focus(None)
            except Exception:
                pass
            event.prevent_default()
            event.stop()
            self._update_status_bar()

    # ── Prompt submission ─────────────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.value = ""
        # Return to NAV MODE after submitting a prompt.
        self._input_mode = False
        try:
            self.set_focus(None)
        except Exception:
            pass
        if self.state.deep:
            self.state.deep.is_generating = True
            self.state.deep.phase = InferencePhase.PREFILLING
        await self._adapter.submit_prompt(prompt)
        self._refresh_all()
        self._update_status_bar()

    # ── Tab actions ───────────────────────────────────────────────────────────

    def _switch_tab(self, tab_id: str, tab_num: str) -> None:
        self._active_tab = tab_num
        try:
            self.query_one(ContentSwitcher).current = tab_id
            self.query_one(TabBar).set_active(tab_num)
        except Exception:
            pass
        self._refresh_all()

    def action_tab_1(self) -> None:
        self._switch_tab("tab-overview", "1")

    def action_tab_2(self) -> None:
        self._switch_tab("tab-transformer", "2")

    def action_tab_3(self) -> None:
        self._switch_tab("tab-tokens", "3")

    def action_tab_4(self) -> None:
        self._switch_tab("tab-logits", "4")

    def action_tab_5(self) -> None:
        self._switch_tab("tab-kvcache", "5")

    def action_tab_6(self) -> None:
        self._switch_tab("tab-performance", "6")

    # ── Token navigation ──────────────────────────────────────────────────────

    def action_prev_token(self) -> None:
        d = self.state.deep
        if d is None:
            return
        history = list(d.token_history)
        if not history:
            return
        if d.selected_token_idx is None:
            d.selected_token_idx = history[-1].index
        else:
            idx = d.selected_token_idx
            pos = next((i for i, r in enumerate(history) if r.index == idx), -1)
            if pos > 0:
                d.selected_token_idx = history[pos - 1].index
        self._refresh_tokens()

    def action_next_token(self) -> None:
        d = self.state.deep
        if d is None:
            return
        history = list(d.token_history)
        if not history:
            return
        if d.selected_token_idx is None:
            d.selected_token_idx = history[0].index
        else:
            idx = d.selected_token_idx
            pos = next((i for i, r in enumerate(history) if r.index == idx), -1)
            if 0 <= pos < len(history) - 1:
                d.selected_token_idx = history[pos + 1].index
        self._refresh_tokens()

    # ── Quit ──────────────────────────────────────────────────────────────────

    def action_quit(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Schedule adapter shutdown without creating a dangling task on exit
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(self._adapter.stop())
        except Exception:
            pass
        self.exit()

    def action_toggle_pause(self) -> None:
        """Space: pause/resume visualization (inference continues)."""
        self._visualization_paused = not self._visualization_paused
        d = self.state.deep
        if d is not None:
            d.visualization_paused = self._visualization_paused
        if not self._visualization_paused:
            # Resumed — do a full refresh to catch up
            self._refresh_all()
        self._update_status_bar()

    def action_live_mode(self) -> None:
        """L: return to live mode (deselect token history)."""
        d = self.state.deep
        if d is not None:
            d.selected_token_idx = None
            d._is_live = True  # type: ignore[attr-defined]
        if self._visualization_paused:
            self._visualization_paused = False
            if d is not None:
                d.visualization_paused = False
        self._refresh_tokens()
        self._update_status_bar()

    def action_show_help(self) -> None:
        """?: show keyboard help overlay."""
        help_text = (
            "[bold]LLMVis Deep Mode — Keyboard Reference[/bold]\n\n"
            "  [bold cyan]1–6[/bold cyan]    Switch tabs\n"
            "  [bold cyan]←/→[/bold cyan]    Browse token history\n"
            "  [bold cyan]L[/bold cyan]      Return to live (latest) token\n"
            "  [bold cyan]Space[/bold cyan]  Pause/resume visualization\n"
            "        (inference continues; display freezes)\n"
            "  [bold cyan]Enter[/bold cyan]  Submit prompt (in prompt box)\n"
            "  [bold cyan]?[/bold cyan]      Show this help\n"
            "  [bold cyan]Q[/bold cyan]      Quit\n\n"
            "[dim]Tabs:[/dim]\n"
            "  1 Overview    — model info + system stats\n"
            "  2 Transformer — live layer traversal (measured RMS)\n"
            "  3 Tokens      — token history with latency + KV\n"
            "  4 Logits      — top candidates before sampling\n"
            "  5 KV Cache    — sequence length + memory (measured)\n"
            "  6 Performance — prefill/decode timing\n\n"
            "[dim]Educational notes:[/dim]\n"
            "  Layer RMS bars show numerical magnitude only.\n"
            "  Higher RMS does not mean 'more important.'\n"
            "  Token probabilities are not factual confidence.\n"
            "  KV memory is measured from actual tensor shapes.\n\n"
            "  Press [bold cyan]?[/bold cyan] or [bold cyan]Q[/bold cyan] to close."
        )
        from textual.widgets import Markdown
        from textual.screen import ModalScreen

        class HelpScreen(ModalScreen):
            BINDINGS = [
                Binding("q", "dismiss", show=False),
                Binding("question_mark", "dismiss", show=False),
                Binding("escape", "dismiss", show=False),
            ]
            CSS = """
            HelpScreen {
                align: center middle;
            }
            #help-dialog {
                background: #161b22;
                border: solid #30363d;
                padding: 2 4;
                width: 60;
                height: auto;
                max-height: 80%;
            }
            """
            def compose(self) -> ComposeResult:
                from textual.containers import Vertical
                from textual.widgets import Static
                with Vertical(id="help-dialog"):
                    yield Static(help_text, markup=True)

            def action_dismiss(self) -> None:
                self.app.pop_screen()

        self.push_screen(HelpScreen())

    def _update_status_bar(self) -> None:
        """Update the status bar to reflect current mode and pause state."""
        try:
            sb = self.query_one(StatusBar)
            sb.set_mode(input_mode=self._input_mode, paused=self._visualization_paused)
        except Exception:
            pass

    # ── Refresh helpers ───────────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._refresh_overview()
        self._refresh_transformer()
        self._refresh_tokens()
        self._refresh_logits()
        self._refresh_kvcache()
        self._refresh_performance()

    def _refresh_overview(self) -> None:
        try:
            self.query_one(OverviewPanel).update_state(self.state)
        except Exception:
            pass

    def _refresh_transformer(self) -> None:
        try:
            self.query_one(TransformerScreen).update_state(self.state)
        except Exception:
            pass

    def _refresh_tokens(self) -> None:
        try:
            self.query_one(TokensScreen).update_state(self.state)
        except Exception:
            pass

    def _refresh_logits(self) -> None:
        try:
            self.query_one(LogitsScreen).update_state(self.state)
        except Exception:
            pass

    def _refresh_kvcache(self) -> None:
        try:
            self.query_one(KvCacheScreen).update_state(self.state)
        except Exception:
            pass

    def _refresh_performance(self) -> None:
        try:
            self.query_one(PerformanceScreen).update_state(self.state)
        except Exception:
            pass

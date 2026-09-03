"""Main LLMVis Textual application."""

from __future__ import annotations

import asyncio
import logging

from textual.app import App, ComposeResult
from textual.binding import Binding

from llmvis.adapters.ollama import OllamaAdapter
from llmvis.core.bus import EventBus
from llmvis.core.events import (
    EventType,
    GpuStatsEvent,
    ModelLoadedEvent,
    ModelUnloadedEvent,
    OllamaConnectedEvent,
    OllamaDisconnectedEvent,
    OllamaReconnectingEvent,
    SystemStatsEvent,
)
from llmvis.core.state import AppState, InferenceStatus, OllamaStatus, make_initial_state
from llmvis.telemetry.gpu import get_gpu_stats
from llmvis.telemetry.system import SystemTelemetry
from llmvis.tui.screens.concept_screen import ConceptScreen
from llmvis.tui.screens.main_screen import MainScreen

logger = logging.getLogger(__name__)

TELEMETRY_INTERVAL = 1.5
INFERENCE_SPIKE_THRESHOLD = 15.0
INFERENCE_COOLDOWN_SAMPLES = 4
HISTORY_LEN = 60


class LLMVisApp(App):
    """LLMVis — Live visualizer and educational debugger for local LLM inference."""

    CSS = """
    Screen {
        background: #0d1117;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("?", "show_help", "Help"),
        Binding("c", "show_concepts", "Concepts"),
        Binding("1", "tab_overview", "Overview"),
        Binding("2", "tab_system", "System"),
        Binding("escape", "close_overlay", "Close", show=False),
    ]

    def __init__(self, host: str = "http://localhost:11434") -> None:
        super().__init__()
        self.state: AppState = make_initial_state(host)
        self.bus = EventBus()
        self._adapter = OllamaAdapter(host=host)
        self._telemetry = SystemTelemetry()
        self._tasks: list[asyncio.Task] = []
        self._inference_below_threshold_count = 0
        self._gpu_baseline: float | None = None

    def compose(self) -> ComposeResult:
        yield MainScreen(self.state)

    async def on_mount(self) -> None:
        self.bus.subscribe(EventType.OLLAMA_CONNECTED, self._on_ollama_connected)
        self.bus.subscribe(EventType.OLLAMA_DISCONNECTED, self._on_ollama_disconnected)
        self.bus.subscribe(EventType.OLLAMA_RECONNECTING, self._on_ollama_reconnecting)
        self.bus.subscribe(EventType.MODEL_LOADED, self._on_model_loaded)
        self.bus.subscribe(EventType.MODEL_UNLOADED, self._on_model_unloaded)
        self.bus.subscribe(EventType.SYSTEM_STATS_UPDATE, self._on_system_stats)
        self.bus.subscribe(EventType.GPU_STATS_UPDATE, self._on_gpu_stats)

        self._tasks = [
            asyncio.create_task(self.bus.dispatch_forever(), name="bus"),
            asyncio.create_task(self._adapter_loop(), name="adapter"),
            asyncio.create_task(self._telemetry_loop(), name="telemetry"),
            asyncio.create_task(self._installed_models_loop(), name="models"),
        ]

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

    async def _installed_models_loop(self) -> None:
        while True:
            if self.state.ollama_status == OllamaStatus.CONNECTED:
                try:
                    models = await self._adapter.fetch_installed_models()
                    self.state.installed_models = models
                    loaded = await self._adapter.fetch_loaded_models()
                    self.state.loaded_models = loaded
                    if loaded and not self.state.active_model:
                        self.state.active_model = loaded[0]
                    elif loaded and self.state.active_model:
                        active_name = self.state.active_model.name
                        for m in loaded:
                            if m.name == active_name:
                                self.state.active_model = m
                                break
                        else:
                            self.state.active_model = loaded[0]
                    elif not loaded:
                        self.state.active_model = None
                    self._refresh_main_screen()
                except Exception as exc:
                    logger.debug("Model list refresh error: %s", exc)
            await asyncio.sleep(5.0)

    async def _on_ollama_connected(self, event: OllamaConnectedEvent) -> None:
        self.state.ollama_status = OllamaStatus.CONNECTED
        self.state.ollama_version = event.version
        self.state.connection_error = ""
        self._refresh_main_screen()

    async def _on_ollama_disconnected(self, event: OllamaDisconnectedEvent) -> None:
        self.state.ollama_status = OllamaStatus.DISCONNECTED
        self.state.connection_error = event.reason
        self.state.loaded_models = []
        self.state.active_model = None
        self.state.inference_status = InferenceStatus.IDLE
        self._refresh_main_screen()

    async def _on_ollama_reconnecting(self, event: OllamaReconnectingEvent) -> None:
        self.state.ollama_status = OllamaStatus.RECONNECTING
        self._refresh_main_screen()

    async def _on_model_loaded(self, event: ModelLoadedEvent) -> None:
        name = event.model.name
        existing_names = {m.name for m in self.state.loaded_models}
        if name not in existing_names:
            self.state.loaded_models.append(event.model)
        if self.state.active_model is None:
            self.state.active_model = event.model
        self._refresh_main_screen()

    async def _on_model_unloaded(self, event: ModelUnloadedEvent) -> None:
        self.state.loaded_models = [
            m for m in self.state.loaded_models if m.name != event.model_name
        ]
        if self.state.active_model and self.state.active_model.name == event.model_name:
            self.state.active_model = (
                self.state.loaded_models[0] if self.state.loaded_models else None
            )
        self.state.inference_status = InferenceStatus.IDLE
        self._refresh_main_screen()

    async def _on_system_stats(self, event: SystemStatsEvent) -> None:
        self.state.system.cpu_percent = event.cpu_percent
        self.state.system.ram_used_bytes = event.ram_used_bytes
        self.state.system.ram_total_bytes = event.ram_total_bytes
        self.state.system.ram_percent = event.ram_percent
        self.state.system.ollama_cpu_percent = event.ollama_cpu_percent
        self.state.system.ollama_ram_bytes = event.ollama_ram_bytes

        self.state.cpu_history.append(event.cpu_percent)
        if len(self.state.cpu_history) > HISTORY_LEN:
            self.state.cpu_history.pop(0)

        self._refresh_main_screen()

    async def _on_gpu_stats(self, event: GpuStatsEvent) -> None:
        self.state.gpu.available = True
        self.state.gpu.gpu_index = event.gpu_index
        self.state.gpu.gpu_name = event.gpu_name
        self.state.gpu.gpu_util_percent = event.gpu_util_percent
        self.state.gpu.vram_used_bytes = event.vram_used_bytes
        self.state.gpu.vram_total_bytes = event.vram_total_bytes
        self.state.gpu.vram_percent = event.vram_percent
        self.state.gpu.temperature_c = event.temperature_c

        self.state.gpu_history.append(event.gpu_util_percent)
        if len(self.state.gpu_history) > HISTORY_LEN:
            self.state.gpu_history.pop(0)

        if self.state.active_model and self.state.ollama_status == OllamaStatus.CONNECTED:
            self._update_inference_estimate(event.gpu_util_percent)

        self._refresh_main_screen()

    def _update_inference_estimate(self, gpu_pct: float) -> None:
        """Estimate inference state from GPU utilization.

        This is an approximation. Stock Ollama does not expose another
        client's inference state.
        """
        history = self.state.gpu_history
        if len(history) >= 5:
            low_samples = [h for h in history[-10:] if h < 10.0]
            if len(low_samples) >= 3:
                self._gpu_baseline = sum(low_samples) / len(low_samples)

        baseline = self._gpu_baseline or 5.0
        delta = gpu_pct - baseline

        if delta >= INFERENCE_SPIKE_THRESHOLD:
            self._inference_below_threshold_count = 0
            if self.state.inference_status != InferenceStatus.POSSIBLY_ACTIVE:
                self.state.inference_status = InferenceStatus.POSSIBLY_ACTIVE
        else:
            self._inference_below_threshold_count += 1
            if (
                self.state.inference_status == InferenceStatus.POSSIBLY_ACTIVE
                and self._inference_below_threshold_count >= INFERENCE_COOLDOWN_SAMPLES
            ):
                self.state.inference_status = InferenceStatus.IDLE

    def _refresh_main_screen(self) -> None:
        try:
            screen = self.screen
            if hasattr(screen, "refresh_state"):
                screen.refresh_state(self.state)
        except Exception:
            pass

    def action_quit(self) -> None:
        for task in self._tasks:
            task.cancel()
        self.exit()

    def action_show_help(self) -> None:
        self.push_screen(ConceptScreen("deep_telemetry"))

    def action_show_concepts(self) -> None:
        self.push_screen(ConceptScreen("kv_cache"))

    def action_tab_overview(self) -> None:
        pass

    def action_tab_system(self) -> None:
        pass

    def action_close_overlay(self) -> None:
        if len(self.screen_stack) > 1:
            self.pop_screen()

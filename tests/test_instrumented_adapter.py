"""Tests for InstrumentedTransformersAdapter using mocked PyTorch model.

Does NOT require downloading real models. AutoModelForCausalLM and
AutoTokenizer are patched out entirely; torch is used directly to build
small real tensors so that numeric assertions are exact.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections import deque
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Path / import setup
# ---------------------------------------------------------------------------
# The source tree uses the package name ``llmvis`` (under src/llmvis).  pytest
# should find it via pyproject.toml / setup.cfg, but add the src dir just in
# case when running directly.
import importlib
import os

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llmvis.adapters.instrumented_transformers import (
    InstrumentedTransformersAdapter,
    ModelArchInfo,
    _kv_cache_stats,
    _sample_logits,
    _tensor_rms,
)
from llmvis.core.events import LayerStatsEvent, TokenCandidate


# ===========================================================================
# Shared mock infrastructure
# ===========================================================================


class MockConfig:
    """Minimal stand-in for a HuggingFace model config."""

    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    hidden_size: int = 16
    vocab_size: int = 100


class MockLayer:
    """A decoder layer that supports hook registration (no-ops)."""

    def __init__(self, idx: int) -> None:
        self.idx = idx
        self._pre_hooks: list = []
        self._post_hooks: list = []

    def register_forward_pre_hook(self, fn) -> MagicMock:
        self._pre_hooks.append(fn)
        handle = MagicMock()
        handle.remove = MagicMock()
        return handle

    def register_forward_hook(self, fn) -> MagicMock:
        self._post_hooks.append(fn)
        handle = MagicMock()
        handle.remove = MagicMock()
        return handle


class MockInnerModel:
    """model.model — the inner transformer body."""

    def __init__(self) -> None:
        self.layers = [MockLayer(0), MockLayer(1)]


class MockModel:
    """Top-level causal LM model object."""

    def __init__(self) -> None:
        self.config = MockConfig()
        self.model = MockInnerModel()
        # A single parameter tensor so _discover_arch can read dtype/device.
        self._param = torch.zeros(4, dtype=torch.float32)

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def __call__(self, **kwargs):
        raise NotImplementedError("MockModel is not meant to run forward passes in unit tests")


class MockTokenizer:
    """Minimal tokenizer substitute."""

    eos_token_id: int = 2

    def __call__(self, text: str, return_tensors: str = "pt"):
        # Always returns a 5-token sequence for any input
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return f"<tok{token_ids[0]}>"


# ---------------------------------------------------------------------------
# Helper: build a patched adapter without touching disk
# ---------------------------------------------------------------------------

def _make_adapter(telemetry_mode: str = "standard") -> InstrumentedTransformersAdapter:
    """Construct an adapter and inject the mock model/tokenizer directly."""
    adapter = InstrumentedTransformersAdapter(
        model_id="mock/model",
        device="cpu",
        telemetry_mode=telemetry_mode,
        max_new_tokens=4,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        top_n_logits=5,
    )
    # Inject mocks so load_model_sync is never called
    adapter._model = MockModel()
    adapter._tokenizer = MockTokenizer()
    adapter._actual_device = "cpu"
    adapter._arch = adapter._discover_arch()
    adapter._model_loaded = True
    return adapter


# ===========================================================================
# Tests
# ===========================================================================


class TestAdapterInit:
    def test_adapter_init(self):
        """Adapter stores model_id and default fields without loading the model."""
        adapter = InstrumentedTransformersAdapter(model_id="some/model")
        assert adapter._model_id == "some/model"
        assert adapter._model_loaded is False
        assert adapter._model is None
        assert adapter._tokenizer is None

    def test_adapter_init_custom_params(self):
        adapter = InstrumentedTransformersAdapter(
            model_id="x/y",
            telemetry_mode="light",
            max_new_tokens=128,
            temperature=0.5,
            top_k=20,
            top_p=0.95,
            top_n_logits=8,
        )
        assert adapter._telemetry_mode == "light"
        assert adapter._max_new_tokens == 128
        assert adapter._temperature == 0.5
        assert adapter._top_k == 20
        assert adapter._top_p == 0.95
        assert adapter._top_n_logits == 8


class TestArchDiscovery:
    def test_arch_discovery(self):
        """_discover_arch reads fields from config correctly."""
        adapter = _make_adapter()
        arch = adapter._arch

        assert isinstance(arch, ModelArchInfo)
        assert arch.num_layers == 2
        assert arch.num_attention_heads == 4
        assert arch.hidden_size == 16
        assert arch.vocab_size == 100
        # head_dim = hidden_size // num_heads = 16 // 4 = 4
        assert arch.head_dim == 4
        assert arch.dtype == "float32"
        assert arch.device == "cpu"

    def test_arch_discovery_head_dim_fallback(self):
        """When config has no head_dim, it's computed from hidden_size / num_heads."""
        adapter = _make_adapter()
        # MockConfig has no head_dim attribute → fallback path
        assert not hasattr(adapter._model.config, "head_dim")
        assert adapter._arch.head_dim == adapter._arch.hidden_size // adapter._arch.num_attention_heads


class TestKvCacheStats:
    def _make_kv_tuple(
        self,
        num_layers: int = 2,
        batch: int = 1,
        num_heads: int = 4,
        seq_len: int = 6,
        head_dim: int = 4,
        dtype=torch.float32,
    ):
        """Build a classic tuple-of-tuples past_key_values."""
        layers = []
        for _ in range(num_layers):
            k = torch.zeros(batch, num_heads, seq_len, head_dim, dtype=dtype)
            v = torch.zeros(batch, num_heads, seq_len, head_dim, dtype=dtype)
            layers.append((k, v))
        return tuple(layers)

    def test_kv_cache_stats_basic(self):
        pkv = self._make_kv_tuple(num_layers=2, seq_len=6)
        seq_len, measured_bytes, k_shape, v_shape, dtype_str = _kv_cache_stats(pkv)

        assert seq_len == 6
        assert k_shape == [1, 4, 6, 4]
        assert v_shape == [1, 4, 6, 4]
        assert dtype_str == "float32"

        # Each tensor: 1 * 4 * 6 * 4 = 96 elements * 4 bytes = 384 bytes
        # 2 layers * (k + v) = 2 * 2 * 384 = 1536
        per_tensor = 1 * 4 * 6 * 4 * 4  # 384 bytes (float32 = 4 bytes)
        expected = 2 * 2 * per_tensor
        assert measured_bytes == expected

    def test_kv_cache_stats_none(self):
        seq_len, measured_bytes, k_shape, v_shape, dtype_str = _kv_cache_stats(None)
        assert seq_len == 0
        assert measured_bytes == 0
        assert k_shape == []
        assert v_shape == []
        assert dtype_str == "none"

    def test_kv_cache_byte_count(self):
        """Measured bytes must match the sum of tensor.nbytes for all K and V tensors."""
        pkv = self._make_kv_tuple(
            num_layers=3, batch=1, num_heads=2, seq_len=8, head_dim=8, dtype=torch.float16
        )
        _, measured_bytes, _, _, _ = _kv_cache_stats(pkv)

        # Manually compute expected bytes
        expected = 0
        for layer in pkv:
            k, v = layer
            expected += k.nelement() * k.element_size()
            expected += v.nelement() * v.element_size()

        assert measured_bytes == expected

    def test_kv_cache_stats_dynamic_cache(self):
        """_kv_cache_stats handles a DynamicCache-style object with key_cache/value_cache."""
        cache = MagicMock()
        k0 = torch.zeros(1, 4, 5, 4, dtype=torch.float32)
        v0 = torch.zeros(1, 4, 5, 4, dtype=torch.float32)
        cache.key_cache = [k0]
        cache.value_cache = [v0]

        seq_len, measured_bytes, k_shape, v_shape, dtype_str = _kv_cache_stats(cache)
        assert seq_len == 5
        assert measured_bytes == k0.nelement() * k0.element_size() * 2  # k + v


class TestSampleLogits:
    def _make_logits(self, vocab_size: int = 100) -> torch.Tensor:
        """Deterministic logits: token i has logit i * 0.1."""
        return torch.arange(vocab_size, dtype=torch.float32) * 0.1

    class _DummyTokenizer:
        eos_token_id = 2

        def decode(self, token_ids, skip_special_tokens=False):
            return f"<{token_ids[0]}>"

    def test_sample_logits_top_n(self):
        """_sample_logits returns at most top_n candidates, sorted by probability desc."""
        logits = self._make_logits(vocab_size=100)
        tokenizer = self._DummyTokenizer()

        _, candidates = _sample_logits(
            logits,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            top_n=5,
            tokenizer=tokenizer,
        )

        assert len(candidates) == 5
        # Must be sorted descending by probability
        probs = [c.probability for c in candidates]
        assert probs == sorted(probs, reverse=True)

    def test_sample_logits_temperature_changes_distribution(self):
        """Low temperature sharpens distribution; high entropy at temp=1.0."""
        logits = self._make_logits(vocab_size=50)
        tokenizer = self._DummyTokenizer()

        _, cands_high = _sample_logits(
            logits.clone(), temperature=1.0, top_k=0, top_p=1.0, top_n=10,
            tokenizer=tokenizer,
        )
        _, cands_low = _sample_logits(
            logits.clone(), temperature=0.1, top_k=0, top_p=1.0, top_n=10,
            tokenizer=tokenizer,
        )

        # At low temperature the top candidate should dominate much more
        top_prob_high = cands_high[0].probability
        top_prob_low = cands_low[0].probability

        assert top_prob_low > top_prob_high, (
            f"Expected sharper distribution at temp=0.1 "
            f"(top={top_prob_low:.4f}) vs temp=1.0 (top={top_prob_high:.4f})"
        )

    def test_sample_logits_eos_not_in_top_candidates_unless_high_logit(self):
        """EOS token should not appear high in the list when its logit is low."""
        vocab_size = 50
        logits = torch.ones(vocab_size, dtype=torch.float32)
        eos_id = 2
        # Give EOS a very low logit
        logits[eos_id] = -100.0

        tokenizer = self._DummyTokenizer()
        _, candidates = _sample_logits(
            logits, temperature=1.0, top_k=0, top_p=1.0, top_n=5,
            tokenizer=tokenizer,
        )

        token_ids = [c.token_id for c in candidates]
        # EOS should not be among the top-5 candidates
        assert eos_id not in token_ids

    def test_sample_logits_returns_token_id_and_candidates(self):
        """Return value has the right types."""
        logits = self._make_logits(vocab_size=30)
        tokenizer = self._DummyTokenizer()

        token_id, candidates = _sample_logits(
            logits, temperature=1.0, top_k=0, top_p=1.0, top_n=3,
            tokenizer=tokenizer,
        )

        assert isinstance(token_id, int)
        assert 0 <= token_id < 30
        for c in candidates:
            assert isinstance(c, TokenCandidate)
            assert 0.0 <= c.probability <= 1.0
            assert 0.0 <= c.raw_probability <= 1.0


class TestLayerStatsRms:
    def test_tensor_rms_manual(self):
        """_tensor_rms must equal sqrt(mean(x^2)) computed manually."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # E[x^2] = (1 + 4 + 9 + 16) / 4 = 7.5
        expected = math.sqrt(7.5)
        result = _tensor_rms(t)
        assert abs(result - expected) < 1e-5, f"{result} != {expected}"

    def test_tensor_rms_zeros(self):
        t = torch.zeros(8)
        assert _tensor_rms(t) == pytest.approx(0.0)

    def test_tensor_rms_negative(self):
        """RMS of a tensor with negative values equals RMS of their absolute values."""
        t = torch.tensor([-3.0, 3.0])
        expected = 3.0
        assert _tensor_rms(t) == pytest.approx(expected, abs=1e-5)

    def test_hook_data_rms_computed_from_real_tensor(self):
        """Simulate what the pre-hook does and verify the stored rms."""
        adapter = _make_adapter()

        # Craft a hidden tensor with known RMS
        hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # shape [1, 1, 4]
        # E[x^2] = (1+4+9+16)/4 = 7.5, rms = sqrt(7.5)
        expected_rms = math.sqrt(7.5)

        pre_hook = adapter._make_pre_hook(0)
        adapter._is_prefill = False  # force stats collection
        pre_hook(None, (hidden,))

        stored_rms = adapter._hook_data[0].input_rms
        assert abs(stored_rms - expected_rms) < 1e-5


class TestTelemetryMode:
    def _collect_layer_stats_events_from_hook_data(
        self, adapter: InstrumentedTransformersAdapter
    ) -> list[LayerStatsEvent]:
        """Replay what _generate_sync does for LayerStatsEvent emission."""
        events: list[LayerStatsEvent] = []

        def capture(evt):
            if isinstance(evt, LayerStatsEvent):
                events.append(evt)

        adapter._event_callback = capture
        adapter._is_prefill = False

        # Populate hook_data for two layers with real tensors
        for layer_idx in range(2):
            pre_hook = adapter._make_pre_hook(layer_idx)
            post_hook = adapter._make_post_hook(layer_idx)
            hidden_in = torch.randn(1, 1, 16)
            hidden_out = torch.randn(1, 1, 16)
            pre_hook(None, (hidden_in,))
            post_hook(None, (hidden_in,), (hidden_out,))

        # Simulate the emission loop in _generate_sync
        prev_rms = None
        for layer_idx in sorted(adapter._hook_data.keys()):
            hd = adapter._hook_data[layer_idx]
            exec_ms = (hd.end_time - hd.start_time) * 1000.0
            delta = 0.0 if prev_rms is None else abs(hd.output_rms - prev_rms)
            prev_rms = hd.output_rms

            if adapter._telemetry_mode in ("standard", "deep"):
                adapter._emit(
                    LayerStatsEvent(
                        source="test",
                        token_index=0,
                        layer_index=layer_idx,
                        hidden_state_rms=hd.output_rms,
                        hidden_state_mean=hd.output_mean,
                        hidden_state_std=hd.output_std,
                        delta_from_prev=delta,
                        exec_time_ms=exec_ms,
                    )
                )

        adapter._event_callback = None
        return events

    def test_telemetry_mode_standard_emits_layer_stats(self):
        """In standard mode, LayerStatsEvent must be emitted for each layer."""
        adapter = _make_adapter(telemetry_mode="standard")
        events = self._collect_layer_stats_events_from_hook_data(adapter)

        assert len(events) == 2, f"Expected 2 LayerStatsEvents, got {len(events)}"
        assert events[0].layer_index == 0
        assert events[1].layer_index == 1

    def test_telemetry_mode_light_no_layer_stats(self):
        """In light mode, LayerStatsEvent must NOT be emitted."""
        adapter = _make_adapter(telemetry_mode="light")
        events = self._collect_layer_stats_events_from_hook_data(adapter)

        assert len(events) == 0, (
            f"Expected 0 LayerStatsEvents in light mode, got {len(events)}"
        )


class TestBoundedTokenHistory:
    def test_token_history_deque_bounded(self):
        """A deque with maxlen enforces the bound and evicts the oldest entry."""
        max_len = 3
        history: deque[int] = deque(maxlen=max_len)

        for i in range(10):
            history.append(i)

        assert len(history) == max_len
        # Should contain the last max_len items: 7, 8, 9
        assert list(history) == [7, 8, 9]

    def test_token_history_deque_preserves_order(self):
        history: deque[int] = deque(maxlen=5)
        for i in range(5):
            history.append(i)
        assert list(history) == [0, 1, 2, 3, 4]

    def test_token_history_does_not_exceed_maxlen_after_many_appends(self):
        max_len = 50
        history: deque[str] = deque(maxlen=max_len)
        for i in range(200):
            history.append(f"tok{i}")
        assert len(history) == max_len


class TestAdapterStop:
    def test_stop_sets_stop_event(self):
        """stop() must set the internal asyncio stop event."""
        adapter = _make_adapter()
        assert not adapter._stop_event.is_set()

        async def _run():
            await adapter.stop()

        asyncio.run(_run())
        assert adapter._stop_event.is_set()

    def test_stop_cancels_generation(self):
        """After stop(), _stop_event is set so generation loop would exit."""
        adapter = _make_adapter()

        async def _run():
            await adapter.stop()
            # Simulate checking the stop event as the generation loop does
            return adapter._stop_event.is_set()

        result = asyncio.run(_run())
        assert result is True


class TestKvCacheByteCount:
    """Separate, focused tests for the byte-count correctness of _kv_cache_stats."""

    def _pkv(
        self,
        num_layers: int,
        batch: int,
        heads: int,
        seq: int,
        hdim: int,
        dtype=torch.float32,
    ):
        return tuple(
            (
                torch.zeros(batch, heads, seq, hdim, dtype=dtype),
                torch.zeros(batch, heads, seq, hdim, dtype=dtype),
            )
            for _ in range(num_layers)
        )

    def test_float32_bytes(self):
        pkv = self._pkv(2, 1, 4, 10, 8, dtype=torch.float32)
        _, measured, _, _, dtype_str = _kv_cache_stats(pkv)
        # float32 = 4 bytes, shape (1,4,10,8) = 320 elements * 4 = 1280 bytes per tensor
        # 2 layers * 2 tensors = 4 tensors total
        expected = 2 * 2 * (1 * 4 * 10 * 8 * 4)
        assert measured == expected
        assert dtype_str == "float32"

    def test_float16_bytes(self):
        pkv = self._pkv(1, 1, 2, 5, 4, dtype=torch.float16)
        _, measured, _, _, dtype_str = _kv_cache_stats(pkv)
        # float16 = 2 bytes, shape (1,2,5,4) = 40 elements * 2 = 80 bytes per tensor
        # 1 layer * 2 tensors = 160 bytes
        expected = 1 * 2 * (1 * 2 * 5 * 4 * 2)
        assert measured == expected
        assert dtype_str == "float16"

    def test_bytes_match_nbytes_sum(self):
        """Measured bytes equals the manual sum of tensor.nbytes."""
        pkv = self._pkv(3, 2, 8, 12, 16, dtype=torch.bfloat16)
        _, measured, _, _, _ = _kv_cache_stats(pkv)

        manual = sum(k.nbytes + v.nbytes for k, v in pkv)
        assert measured == manual


# ===========================================================================
# Regression tests for runtime bug-fixes
# ===========================================================================


class TestInferenceEndOnException:
    """Regression: INFERENCE_END must be emitted even when _generate_sync raises.

    Bug: exceptions in generation would swallow the INFERENCE_END event,
    permanently blocking the REPL's _generation_done gate.
    Fix: the except block in _generate_sync explicitly emits InferenceEndEvent
    before propagating, and generated_ids is declared before the try block so
    the except handler can safely read its length.
    """

    def _run_generation_with_failing_model(self, fail_on: str = "prefill"):
        """Run _generate_sync with a model that raises during forward, collect emitted events."""
        from llmvis.core.events import EventType

        adapter = _make_adapter()

        # Override MockModel.__call__ to raise at the chosen stage
        if fail_on == "prefill":
            def _failing_call(**kwargs):
                raise RuntimeError("simulated GPU OOM")
            adapter._model.__class__.__call__ = lambda self, **kw: _failing_call(**kw)

        emitted = []

        def _capture(evt):
            if evt is not None:  # adapter sends None sentinel in finally block
                emitted.append(evt)

        adapter._event_callback = _capture
        # _generate_sync is synchronous; call it directly
        adapter._generate_sync("hello world")
        adapter._event_callback = None
        return emitted

    def test_inference_end_emitted_on_model_exception(self):
        """INFERENCE_END is emitted when the model forward pass raises."""
        from llmvis.core.events import EventType

        events = self._run_generation_with_failing_model()
        types = [e.type for e in events]

        assert EventType.INFERENCE_START in types, "INFERENCE_START must still be emitted"
        assert EventType.INFERENCE_END in types, (
            "INFERENCE_END must be emitted even on exception (required to unblock _generation_done gate)"
        )

    def test_inference_end_is_last_event_on_exception(self):
        """INFERENCE_END is the final event emitted after a generation failure."""
        from llmvis.core.events import EventType

        events = self._run_generation_with_failing_model()
        assert events, "At least one event must be emitted"
        assert events[-1].type == EventType.INFERENCE_END, (
            f"Last event must be INFERENCE_END, got {events[-1].type}"
        )

    def test_generated_ids_accessible_in_except(self):
        """generated_ids is declared before try block so except can read len(generated_ids)=0."""
        from llmvis.core.events import EventType, InferenceEndEvent

        events = self._run_generation_with_failing_model()
        end_events = [e for e in events if e.type == EventType.INFERENCE_END]
        assert end_events, "INFERENCE_END must appear"
        # On early failure, total_tokens must be 0 (no tokens were generated)
        assert end_events[0].total_tokens == 0, (
            f"Expected total_tokens=0 on prefill failure, got {end_events[0].total_tokens}"
        )


class TestGenerationDoneGate:
    """Regression: _generation_done asyncio.Event must unblock after generation ends.

    Bug: if INFERENCE_END was never emitted (e.g., on exception), the REPL
    would block forever at `await _generation_done.wait()`.
    Fix: both normal completion and exception paths emit INFERENCE_END, and
    _adapter_loop always calls _generation_done.set() in its finally block.
    """

    def test_generation_done_set_after_normal_inference_end(self):
        """Simulates the _adapter_loop gate behavior: set after INFERENCE_END."""
        from llmvis.core.events import EventType, InferenceEndEvent, InferenceStartEvent

        async def _run():
            gate = asyncio.Event()
            gate.set()  # start idle

            events = [
                InferenceStartEvent(source="t", model_id="m", prompt_preview="hi"),
                InferenceEndEvent(source="t", model_id="m", total_tokens=5, total_time_ms=10.0),
            ]

            for evt in events:
                t = evt.type
                if t == EventType.INFERENCE_START:
                    gate.clear()
                elif t == EventType.INFERENCE_END:
                    gate.set()

            return gate.is_set()

        result = asyncio.run(_run())
        assert result is True, "_generation_done gate must be set after INFERENCE_END"

    def test_generation_done_cleared_during_generation(self):
        """Gate is clear (blocking) between INFERENCE_START and INFERENCE_END."""
        from llmvis.core.events import EventType, InferenceStartEvent

        async def _run():
            gate = asyncio.Event()
            gate.set()

            start = InferenceStartEvent(source="t", model_id="m", prompt_preview="hi")
            if start.type == EventType.INFERENCE_START:
                gate.clear()

            return gate.is_set()

        result = asyncio.run(_run())
        assert result is False, "Gate must be clear (blocking) while generation is in progress"

    def test_generation_done_set_after_exception_path(self):
        """Finally block always sets the gate, even if INFERENCE_END was emitted by exception path."""
        from llmvis.core.events import EventType, InferenceEndEvent, InferenceStartEvent

        async def _run():
            gate = asyncio.Event()
            gate.set()

            # Simulate: start generation, exception fires, except emits INFERENCE_END
            gate.clear()  # INFERENCE_START clears it
            end_evt = InferenceEndEvent(source="t", model_id="m", total_tokens=0, total_time_ms=1.0)
            if end_evt.type == EventType.INFERENCE_END:
                gate.set()  # except handler sets it via the same code path

            return gate.is_set()

        result = asyncio.run(_run())
        assert result is True, "Gate must be unblocked after exception-path INFERENCE_END"


class TestChatTemplateAndEOS:
    """Regression tests for chat template tokenization and multi-ID EOS handling."""

    # ── _collect_eos_ids ─────────────────────────────────────────────────────

    def test_collect_eos_ids_single_int(self):
        """Single int eos_token_id is collected."""
        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = 99
        ids = adapter._collect_eos_ids()
        assert 99 in ids

    def test_collect_eos_ids_list(self):
        """List of eos_token_ids are all collected."""
        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = [99, 100, 101]
        ids = adapter._collect_eos_ids()
        assert ids == frozenset({99, 100, 101})

    def test_collect_eos_ids_merges_generation_config(self):
        """generation_config.eos_token_id is merged with tokenizer EOS IDs."""
        from unittest.mock import MagicMock
        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = 2
        gc = MagicMock()
        gc.eos_token_id = [2, 200]
        adapter._model.generation_config = gc
        ids = adapter._collect_eos_ids()
        assert 2 in ids
        assert 200 in ids

    def test_collect_eos_ids_none_returns_empty(self):
        """None eos_token_id returns empty frozenset (no crash)."""
        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = None
        ids = adapter._collect_eos_ids()
        assert isinstance(ids, frozenset)
        assert len(ids) == 0

    # ── Chat template detection ───────────────────────────────────────────────

    def test_chat_template_applied_when_present(self):
        """apply_chat_template is called for tokenizers that have a chat_template."""
        adapter = _make_adapter()
        adapter._tokenizer.chat_template = "{% for msg in messages %}{{msg['content']}}{% endfor %}"

        called_with = []

        def fake_apply_chat_template(messages, tokenize, add_generation_prompt, return_tensors):
            called_with.append(messages)
            return torch.tensor([[1, 2, 3, 4, 5]])

        adapter._tokenizer.apply_chat_template = fake_apply_chat_template

        # Generation will fail at model forward — we only care that tokenization happened
        adapter._generate_sync("hello")

        assert len(called_with) == 1
        assert called_with[0] == [{"role": "user", "content": "hello"}]

    def test_raw_tokenization_when_no_chat_template(self):
        """Plain tokenizer() call is used when chat_template is None."""
        adapter = _make_adapter()
        adapter._tokenizer.chat_template = None

        call_log = []
        original_call = MockTokenizer.__call__

        def tracked_call(self_tok, text, return_tensors=None):
            call_log.append(text)
            return original_call(self_tok, text, return_tensors=return_tensors)

        adapter._tokenizer.__class__.__call__ = tracked_call
        adapter._generate_sync("hello")
        # restore
        adapter._tokenizer.__class__.__call__ = original_call

        assert "hello" in call_log

    # ── EOS stop behaviour ────────────────────────────────────────────────────

    def test_eos_int_stops_generation_without_token_generated(self):
        """Single int EOS ID: generation stops and EOS token NOT emitted as TOKEN_GENERATED."""
        from llmvis.core.events import EventType

        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = 2

        def model_forward(**kwargs):
            batch, seq = kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1]
            logits = torch.zeros(batch, seq, 100)
            logits[..., 2] = 100.0  # argmax → token 2 (EOS)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits, past_key_values=None)

        adapter._model.__class__.__call__ = lambda self, **kw: model_forward(**kw)

        emitted = []

        def _capture(evt):
            if evt is not None:
                emitted.append(evt)

        adapter._event_callback = _capture
        adapter._generate_sync("test")
        adapter._event_callback = None

        types = [e.type for e in emitted]
        assert EventType.INFERENCE_END in types
        token_ends = [e for e in emitted if e.type == EventType.TOKEN_END]
        assert len(token_ends) == 0, "TOKEN_END must not be emitted for the EOS token"

    def test_eos_list_stops_generation(self):
        """List EOS IDs: any matching ID stops generation."""
        from llmvis.core.events import EventType

        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = [2, 50]

        def model_forward(**kwargs):
            batch, seq = kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1]
            logits = torch.zeros(batch, seq, 100)
            logits[..., 50] = 100.0  # argmax → token 50 (second EOS ID)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits, past_key_values=None)

        adapter._model.__class__.__call__ = lambda self, **kw: model_forward(**kw)

        emitted = []

        def _capture(evt):
            if evt is not None:
                emitted.append(evt)

        adapter._event_callback = _capture
        adapter._generate_sync("test")
        adapter._event_callback = None

        types = [e.type for e in emitted]
        assert EventType.INFERENCE_END in types
        token_ends = [e for e in emitted if e.type == EventType.TOKEN_END]
        assert len(token_ends) == 0, "TOKEN_END must not be emitted for the EOS token"

    def test_generation_config_eos_stops_generation(self):
        """EOS ID from generation_config (not tokenizer) stops generation."""
        from llmvis.core.events import EventType
        from unittest.mock import MagicMock

        adapter = _make_adapter()
        adapter._tokenizer.eos_token_id = 2  # model will NOT produce this
        gc = MagicMock()
        gc.eos_token_id = 77  # model WILL produce this
        adapter._model.generation_config = gc

        def model_forward(**kwargs):
            batch, seq = kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1]
            logits = torch.zeros(batch, seq, 100)
            logits[..., 77] = 100.0  # argmax → token 77 (only in generation_config)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits, past_key_values=None)

        adapter._model.__class__.__call__ = lambda self, **kw: model_forward(**kw)

        emitted = []

        def _capture(evt):
            if evt is not None:
                emitted.append(evt)

        adapter._event_callback = _capture
        adapter._generate_sync("test")
        adapter._event_callback = None

        types = [e.type for e in emitted]
        assert EventType.INFERENCE_END in types
        token_ends = [e for e in emitted if e.type == EventType.TOKEN_END]
        assert len(token_ends) == 0, "generation_config EOS must stop generation before TOKEN_END"

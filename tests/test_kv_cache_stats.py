"""Regression tests for _kv_cache_stats() and the KV Cache tab.

Covers:
- Transformers 5.x DynamicCache (.layers / DynamicLayer.keys / .values)
- Transformers 4.38–4.45 DynamicCache (.key_cache / .value_cache)
- Classic tuple-of-tuples
- KvCacheUpdateEvent JSONL round-trip
- DeepState update from event
- KvCacheScreen renders active vs idle correctly
"""

from __future__ import annotations

import pytest

from llmvis.adapters.instrumented_transformers import _kv_cache_stats
from llmvis.core.events import (
    EventType,
    KvCacheUpdateEvent,
    event_from_jsonl,
    event_to_jsonl,
)
from llmvis.core.state import (
    DeepState,
    ModelArchInfo,
    make_deep_state,
    make_initial_state,
)


# ── Minimal duck-typed mocks (no torch dependency) ────────────────────────────

class _MockShape(tuple):
    pass


class _MockTensor:
    """Minimal tensor mock exposing .shape, .dtype, .numel(), .element_size()."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str = "torch.float32",
        element_size: int = 4,
    ) -> None:
        self.shape = _MockShape(shape)
        self.dtype = dtype
        self._element_size = element_size

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def element_size(self) -> int:
        return self._element_size


class _MockLayer5x:
    """Mocks Transformers 5.x DynamicLayer with .keys and .values."""

    def __init__(self, k: _MockTensor, v: _MockTensor) -> None:
        self.keys = k
        self.values = v


class _MockDynamicCache5x:
    """Mocks Transformers 5.x DynamicCache with .layers and .get_seq_length()."""

    def __init__(self, layers: list[_MockLayer5x], seq_len: int | None = None) -> None:
        self.layers = layers
        _auto = layers[0].keys.shape[2] if layers and len(layers[0].keys.shape) >= 3 else 0
        self._seq_len = seq_len if seq_len is not None else _auto

    def get_seq_length(self) -> int:
        return self._seq_len

    def __iter__(self):
        for layer in self.layers:
            yield (layer.keys, layer.values)


class _MockDynamicCache4x:
    """Mocks Transformers 4.38–4.45 DynamicCache with .key_cache/.value_cache."""

    def __init__(
        self, key_tensors: list[_MockTensor], val_tensors: list[_MockTensor]
    ) -> None:
        self.key_cache = key_tensors
        self.value_cache = val_tensors


def _kv(
    seq_len: int = 10,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    dtype: str = "torch.float32",
    element_size: int = 4,
) -> tuple[_MockTensor, _MockTensor]:
    k = _MockTensor((1, num_kv_heads, seq_len, head_dim), dtype, element_size)
    v = _MockTensor((1, num_kv_heads, seq_len, head_dim), dtype, element_size)
    return k, v


# ── None / empty ──────────────────────────────────────────────────────────────

def test_none_returns_zeros():
    seq_len, b, k_shape, v_shape, dtype = _kv_cache_stats(None)
    assert seq_len == 0
    assert b == 0
    assert k_shape == []
    assert v_shape == []
    assert dtype == "none"


def test_5x_empty_layers_returns_zeros():
    cache = _MockDynamicCache5x([])
    seq_len, b, k_shape, v_shape, dtype = _kv_cache_stats(cache)
    assert seq_len == 0 and b == 0 and k_shape == []


# ── Transformers 5.x DynamicCache ─────────────────────────────────────────────

def test_5x_seq_len_uses_get_seq_length():
    k, v = _kv(seq_len=23)
    cache = _MockDynamicCache5x([_MockLayer5x(k, v)] * 4, seq_len=23)
    seq_len, _, _, _, _ = _kv_cache_stats(cache)
    assert seq_len == 23


def test_5x_seq_len_fallback_to_shape_dim2():
    k, v = _kv(seq_len=17)
    layer = _MockLayer5x(k, v)

    class _NoSeqLen:
        layers = [layer]

        def __iter__(self):
            yield (k, v)

    seq_len, _, _, _, _ = _kv_cache_stats(_NoSeqLen())
    assert seq_len == 17


def test_5x_k_v_shape():
    k, v = _kv(seq_len=15, num_kv_heads=2, head_dim=128)
    cache = _MockDynamicCache5x([_MockLayer5x(k, v)])
    _, _, k_shape, v_shape, _ = _kv_cache_stats(cache)
    assert k_shape == [1, 2, 15, 128]
    assert v_shape == [1, 2, 15, 128]


def test_5x_dtype_stripped():
    k, v = _kv(dtype="torch.bfloat16")
    cache = _MockDynamicCache5x([_MockLayer5x(k, v)])
    _, _, _, _, dtype = _kv_cache_stats(cache)
    assert dtype == "bfloat16"


def test_5x_byte_count_all_layers():
    # 4 layers, float32 (4 bytes), shape [1, 2, 10, 128]
    k, v = _kv(seq_len=10, num_kv_heads=2, head_dim=128, element_size=4)
    n = 4
    cache = _MockDynamicCache5x([_MockLayer5x(k, v) for _ in range(n)], seq_len=10)
    _, total_bytes, _, _, _ = _kv_cache_stats(cache)
    # numel = 1*2*10*128 = 2560, 4 bytes each = 10240 per tensor, 2 tensors × 4 layers
    assert total_bytes == n * 2 * (1 * 2 * 10 * 128 * 4)


def test_5x_correct_result_for_qwen_dims():
    # Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim=128, bfloat16 (2 bytes)
    k, v = _kv(seq_len=42, num_kv_heads=2, head_dim=128, dtype="torch.bfloat16", element_size=2)
    n_layers = 28
    cache = _MockDynamicCache5x([_MockLayer5x(k, v) for _ in range(n_layers)], seq_len=42)
    seq_len, total_bytes, k_shape, v_shape, dtype = _kv_cache_stats(cache)
    assert seq_len == 42
    assert k_shape == [1, 2, 42, 128]
    assert dtype == "bfloat16"
    expected_bytes = n_layers * 2 * (1 * 2 * 42 * 128 * 2)
    assert total_bytes == expected_bytes


# ── Transformers 4.38–4.45: .key_cache / .value_cache ────────────────────────

def test_4x_seq_len():
    k, v = _kv(seq_len=30)
    cache = _MockDynamicCache4x([k], [v])
    seq_len, _, _, _, _ = _kv_cache_stats(cache)
    assert seq_len == 30


def test_4x_shape():
    k, v = _kv(seq_len=30, num_kv_heads=8, head_dim=64)
    cache = _MockDynamicCache4x([k] * 3, [v] * 3)
    _, _, k_shape, v_shape, _ = _kv_cache_stats(cache)
    assert k_shape == [1, 8, 30, 64]
    assert v_shape == [1, 8, 30, 64]


def test_4x_dtype_stripped():
    k, v = _kv(dtype="torch.float16")
    cache = _MockDynamicCache4x([k], [v])
    _, _, _, _, dtype = _kv_cache_stats(cache)
    assert dtype == "float16"


def test_4x_byte_count():
    k, v = _kv(seq_len=5, num_kv_heads=4, head_dim=64, element_size=2)
    n = 6
    cache = _MockDynamicCache4x([k] * n, [v] * n)
    _, total_bytes, _, _, _ = _kv_cache_stats(cache)
    assert total_bytes == n * 2 * (1 * 4 * 5 * 64 * 2)


def test_4x_empty_key_cache_returns_zeros():
    cache = _MockDynamicCache4x([], [])
    seq_len, b, k_shape, v_shape, dtype = _kv_cache_stats(cache)
    assert seq_len == 0 and b == 0 and k_shape == []


# ── Classic tuple-of-tuples ───────────────────────────────────────────────────

def test_tuple_seq_len():
    k, v = _kv(seq_len=12)
    seq_len, _, _, _, _ = _kv_cache_stats(((k, v), (k, v)))
    assert seq_len == 12


def test_tuple_shapes():
    k, v = _kv(seq_len=7, num_kv_heads=4, head_dim=96)
    _, _, k_shape, v_shape, _ = _kv_cache_stats(((k, v),))
    assert k_shape == [1, 4, 7, 96]
    assert v_shape == [1, 4, 7, 96]


def test_tuple_dtype():
    k, v = _kv(dtype="torch.float32")
    _, _, _, _, dtype = _kv_cache_stats(((k, v),))
    assert dtype == "float32"


def test_tuple_byte_count():
    k, v = _kv(seq_len=8, num_kv_heads=2, head_dim=64, element_size=4)
    n = 2
    _, total_bytes, _, _, _ = _kv_cache_stats(tuple((k, v) for _ in range(n)))
    assert total_bytes == n * 2 * (1 * 2 * 8 * 64 * 4)


# ── KvCacheUpdateEvent JSONL round-trip ───────────────────────────────────────

def test_kv_cache_update_event_jsonl_roundtrip():
    evt = KvCacheUpdateEvent(
        source="adapter",
        seq_len_before=0,
        seq_len_after=23,
        k_shape=[1, 2, 23, 128],
        v_shape=[1, 2, 23, 128],
        dtype="bfloat16",
        measured_bytes=1_507_328,
        num_layers=28,
    )
    line = event_to_jsonl(evt)
    recovered = event_from_jsonl(line)

    assert recovered is not None
    assert recovered.type == EventType.KV_CACHE_UPDATE
    assert recovered.seq_len_after == 23
    assert recovered.k_shape == [1, 2, 23, 128]
    assert recovered.v_shape == [1, 2, 23, 128]
    assert recovered.dtype == "bfloat16"
    assert recovered.measured_bytes == 1_507_328
    assert recovered.num_layers == 28


def test_kv_cache_event_seq_len_zero_roundtrip():
    evt = KvCacheUpdateEvent(source="test", seq_len_before=0, seq_len_after=0)
    line = event_to_jsonl(evt)
    recovered = event_from_jsonl(line)
    assert recovered is not None
    assert recovered.seq_len_after == 0


# ── DeepState update from KvCacheUpdateEvent ──────────────────────────────────

def test_deep_state_update_from_kv_event():
    d = make_deep_state()
    evt = KvCacheUpdateEvent(
        source="adapter",
        seq_len_before=0,
        seq_len_after=42,
        k_shape=[1, 2, 42, 128],
        v_shape=[1, 2, 42, 128],
        dtype="float16",
        measured_bytes=999_000,
        num_layers=28,
    )
    # Simulate _on_kv_cache_update in app_deep.py
    d.current_kv_seq_len = evt.seq_len_after
    d.current_kv_bytes = evt.measured_bytes
    d.last_k_shape = evt.k_shape
    d.last_v_shape = evt.v_shape
    d.last_kv_dtype = evt.dtype

    assert d.current_kv_seq_len == 42
    assert d.current_kv_bytes == 999_000
    assert d.last_k_shape == [1, 2, 42, 128]
    assert d.last_v_shape == [1, 2, 42, 128]
    assert d.last_kv_dtype == "float16"


def test_deep_state_starts_with_zero_kv_seq_len():
    d = make_deep_state()
    assert d.current_kv_seq_len == 0
    assert d.current_kv_bytes == 0
    assert d.last_k_shape == []
    assert d.last_kv_dtype == ""


# ── KvCacheScreen renders correctly ───────────────────────────────────────────

def _make_state_with_kv(seq_len: int) -> "AppState":  # noqa: F821
    state = make_initial_state()
    state.deep = make_deep_state()
    state.deep.arch = ModelArchInfo(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_layers=28,
        num_attention_heads=12,
        hidden_size=1536,
        vocab_size=151936,
        head_dim=128,
        dtype="bfloat16",
        device="cuda:0",
    )
    state.deep.current_kv_seq_len = seq_len
    state.deep.current_kv_bytes = seq_len * 28 * 2 * 128 * 2  # rough estimate for non-zero
    state.deep.last_k_shape = [1, 2, seq_len, 128] if seq_len else []
    state.deep.last_v_shape = [1, 2, seq_len, 128] if seq_len else []
    state.deep.last_kv_dtype = "bfloat16" if seq_len else ""
    return state


def test_kv_screen_idle_shows_waiting_message():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = _make_state_with_kv(seq_len=0)
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "Waiting for first token" in rendered


def test_kv_screen_active_hides_waiting_message():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = _make_state_with_kv(seq_len=23)
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "Waiting for first token" not in rendered


def test_kv_screen_active_shows_seq_len():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = _make_state_with_kv(seq_len=23)
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "23" in rendered


def test_kv_screen_active_shows_layer_count():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = _make_state_with_kv(seq_len=10)
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "28" in rendered  # num_layers from arch


def test_kv_screen_active_shows_shape():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = _make_state_with_kv(seq_len=15)
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "128" in rendered  # head_dim in shape string
    assert "bfloat16" in rendered


def test_kv_screen_no_deep_shows_not_active_msg():
    from llmvis.tui.screens.kvcache_screen import KvCacheScreen

    state = make_initial_state()
    # deep is None — stock Ollama mode
    screen = KvCacheScreen(state)
    rendered = screen._build(state)

    assert "Waiting for first token" not in rendered
    assert "deep telemetry not active" in rendered.lower()

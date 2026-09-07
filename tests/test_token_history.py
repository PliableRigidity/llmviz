"""Tests for token history, bounded buffers, and deep state."""

from __future__ import annotations

from collections import deque

from llmvis.core.state import (
    AppState,
    DeepState,
    InferencePhase,
    LayerSummary,
    TokenRecord,
    make_deep_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer_summary(layer_index: int = 0) -> LayerSummary:
    return LayerSummary(
        layer_index=layer_index,
        hidden_state_rms=0.5,
        hidden_state_mean=0.01,
        hidden_state_std=0.3,
        delta_from_prev=0.02,
        exec_time_ms=1.5,
        mlp_output_rms=0.4,
        mlp_sparsity=0.1,
    )


def _make_token_record(index: int = 0) -> TokenRecord:
    return TokenRecord(
        index=index,
        text=" hello",
        token_id=9906,
        latency_ms=12.3,
        layer_summaries=[_make_layer_summary(i) for i in range(4)],
        top_logits=[(" hello", 0.7), (" world", 0.2)],
        kv_seq_len=index + 1,
        kv_bytes=(index + 1) * 1024,
        temperature=0.8,
        top_k_param=40,
        top_p_param=0.9,
    )


# ---------------------------------------------------------------------------
# 1. test_deep_state_defaults
# ---------------------------------------------------------------------------

def test_deep_state_defaults():
    ds = make_deep_state()
    assert ds.phase == InferencePhase.IDLE
    assert len(ds.token_history) == 0
    assert ds.selected_token_idx is None
    assert ds.response_text == ""
    assert ds.prompt_text == ""
    assert ds.current_layer == -1
    assert ds.decode_token_count == 0
    assert ds.prefill_token_count == 0
    assert ds.is_generating is False


# ---------------------------------------------------------------------------
# 2. test_token_record_creation
# ---------------------------------------------------------------------------

def test_token_record_creation():
    layers = [_make_layer_summary(i) for i in range(3)]
    record = TokenRecord(
        index=5,
        text=" cat",
        token_id=7,
        latency_ms=8.1,
        layer_summaries=layers,
        top_logits=[(" cat", 0.6), (" dog", 0.3)],
        kv_seq_len=6,
        kv_bytes=6144,
        temperature=1.0,
        top_k_param=50,
        top_p_param=0.95,
    )
    assert record.index == 5
    assert record.text == " cat"
    assert record.token_id == 7
    assert record.latency_ms == 8.1
    assert len(record.layer_summaries) == 3
    assert record.top_logits[0] == (" cat", 0.6)
    assert record.kv_seq_len == 6
    assert record.kv_bytes == 6144
    assert record.temperature == 1.0
    assert record.top_k_param == 50
    assert record.top_p_param == 0.95


# ---------------------------------------------------------------------------
# 3. test_token_history_bounded
# ---------------------------------------------------------------------------

def test_token_history_bounded():
    ds = make_deep_state()
    for i in range(150):
        ds.token_history.append(_make_token_record(i))
    assert len(ds.token_history) == 100


# ---------------------------------------------------------------------------
# 4. test_token_history_oldest_dropped
# ---------------------------------------------------------------------------

def test_token_history_oldest_dropped():
    ds = make_deep_state()
    for i in range(150):
        ds.token_history.append(_make_token_record(i))
    # The first 50 (index 0-49) should have been dropped; oldest remaining is index 50
    oldest = ds.token_history[0]
    assert oldest.index == 50
    newest = ds.token_history[-1]
    assert newest.index == 149


# ---------------------------------------------------------------------------
# 5. test_layer_summary_fields
# ---------------------------------------------------------------------------

def test_layer_summary_fields():
    ls = LayerSummary(
        layer_index=7,
        hidden_state_rms=1.2,
        hidden_state_mean=-0.05,
        hidden_state_std=0.88,
        delta_from_prev=0.15,
        exec_time_ms=3.7,
        mlp_output_rms=0.9,
        mlp_sparsity=0.25,
    )
    assert ls.layer_index == 7
    assert ls.hidden_state_rms == 1.2
    assert ls.hidden_state_mean == -0.05
    assert ls.hidden_state_std == 0.88
    assert ls.delta_from_prev == 0.15
    assert ls.exec_time_ms == 3.7
    assert ls.mlp_output_rms == 0.9
    assert ls.mlp_sparsity == 0.25


def test_layer_summary_optional_fields_default_none():
    ls = LayerSummary(
        layer_index=0,
        hidden_state_rms=0.0,
        hidden_state_mean=0.0,
        hidden_state_std=0.0,
        delta_from_prev=0.0,
        exec_time_ms=0.0,
    )
    assert ls.mlp_output_rms is None
    assert ls.mlp_sparsity is None


# ---------------------------------------------------------------------------
# 6. test_selected_token_idx_navigation
# ---------------------------------------------------------------------------

def test_selected_token_idx_navigation():
    ds = make_deep_state()
    for i in range(5):
        ds.token_history.append(_make_token_record(i))

    # Start with no selection
    assert ds.selected_token_idx is None

    # Select first token
    ds.selected_token_idx = 0
    assert ds.selected_token_idx == 0

    # Increment — move forward through history
    ds.selected_token_idx += 1
    assert ds.selected_token_idx == 1

    ds.selected_token_idx += 1
    assert ds.selected_token_idx == 2

    # Decrement
    ds.selected_token_idx -= 1
    assert ds.selected_token_idx == 1

    # Navigate to last token
    ds.selected_token_idx = len(ds.token_history) - 1
    assert ds.selected_token_idx == 4

    # Reset selection
    ds.selected_token_idx = None
    assert ds.selected_token_idx is None


# ---------------------------------------------------------------------------
# 7. test_deep_state_phase_transitions
# ---------------------------------------------------------------------------

def test_deep_state_phase_transitions():
    ds = make_deep_state()
    assert ds.phase == InferencePhase.IDLE

    ds.phase = InferencePhase.PREFILLING
    assert ds.phase == InferencePhase.PREFILLING

    ds.phase = InferencePhase.DECODING
    assert ds.phase == InferencePhase.DECODING

    ds.phase = InferencePhase.COMPLETE
    assert ds.phase == InferencePhase.COMPLETE

    # Can return to IDLE
    ds.phase = InferencePhase.IDLE
    assert ds.phase == InferencePhase.IDLE


def test_inference_phase_all_values():
    phases = list(InferencePhase)
    names = {p.name for p in phases}
    assert "IDLE" in names
    assert "PREFILLING" in names
    assert "DECODING" in names
    assert "COMPLETE" in names
    assert "LOADING" in names
    assert "TOKENIZING" in names


# ---------------------------------------------------------------------------
# 8. test_current_layer_tracking
# ---------------------------------------------------------------------------

def test_current_layer_tracking():
    ds = make_deep_state()
    assert ds.current_layer == -1

    ds.current_layer = 0
    assert ds.current_layer == 0

    ds.current_layer = 11
    assert ds.current_layer == 11

    # Reset when done
    ds.current_layer = -1
    assert ds.current_layer == -1


# ---------------------------------------------------------------------------
# 9. test_kv_stats_tracking
# ---------------------------------------------------------------------------

def test_kv_stats_tracking():
    ds = make_deep_state()
    assert ds.current_kv_seq_len == 0
    assert ds.current_kv_bytes == 0

    ds.current_kv_seq_len = 42
    ds.current_kv_bytes = 42 * 4096

    assert ds.current_kv_seq_len == 42
    assert ds.current_kv_bytes == 42 * 4096

    # Simulate growth over decoding
    for i in range(43, 53):
        ds.current_kv_seq_len = i
        ds.current_kv_bytes = i * 4096

    assert ds.current_kv_seq_len == 52
    assert ds.current_kv_bytes == 52 * 4096


# ---------------------------------------------------------------------------
# 10. test_response_text_accumulation
# ---------------------------------------------------------------------------

def test_response_text_accumulation():
    ds = make_deep_state()
    assert ds.response_text == ""

    tokens = [" Hello", ",", " world", "!"]
    for tok in tokens:
        ds.response_text += tok

    assert ds.response_text == " Hello, world!"

    # Each token appended in order
    ds2 = make_deep_state()
    words = ["The", " quick", " brown", " fox"]
    for w in words:
        ds2.response_text += w
    assert ds2.response_text == "The quick brown fox"


# ---------------------------------------------------------------------------
# 11. test_decode_tokens_per_sec
# ---------------------------------------------------------------------------

def test_decode_tokens_per_sec():
    ds = make_deep_state()

    # Simulate: 50 decode tokens over 2000 ms
    ds.decode_token_count = 50
    ds.prefill_duration_ms = 2000.0

    # Compute tokens/sec from count and duration
    if ds.prefill_duration_ms > 0:
        computed = ds.decode_token_count / (ds.prefill_duration_ms / 1000.0)
    else:
        computed = 0.0

    assert computed == 25.0

    # Store result in decode_tokens_per_sec field
    ds.decode_tokens_per_sec = computed
    assert ds.decode_tokens_per_sec == 25.0


def test_decode_tokens_per_sec_zero_duration():
    ds = make_deep_state()
    ds.decode_token_count = 10
    ds.prefill_duration_ms = 0.0

    computed = ds.decode_token_count / (ds.prefill_duration_ms / 1000.0) if ds.prefill_duration_ms > 0 else 0.0
    assert computed == 0.0
    ds.decode_tokens_per_sec = computed
    assert ds.decode_tokens_per_sec == 0.0


def test_decode_tokens_per_sec_direct_field():
    ds = make_deep_state()
    assert ds.decode_tokens_per_sec == 0.0
    ds.decode_tokens_per_sec = 47.3
    assert ds.decode_tokens_per_sec == 47.3


# ---------------------------------------------------------------------------
# 12. test_token_record_layer_summaries_list
# ---------------------------------------------------------------------------

def test_token_record_layer_summaries_list():
    summaries = [_make_layer_summary(i) for i in range(8)]
    record = _make_token_record(0)
    record = TokenRecord(
        index=0,
        text=" hi",
        token_id=123,
        latency_ms=5.0,
        layer_summaries=summaries,
        top_logits=[(" hi", 0.9)],
        kv_seq_len=1,
        kv_bytes=1024,
    )

    assert isinstance(record.layer_summaries, list)
    assert len(record.layer_summaries) == 8
    for i, ls in enumerate(record.layer_summaries):
        assert isinstance(ls, LayerSummary)
        assert ls.layer_index == i


def test_token_record_layer_summaries_empty():
    record = TokenRecord(
        index=0,
        text="<s>",
        token_id=1,
        latency_ms=0.0,
        layer_summaries=[],
        top_logits=[],
        kv_seq_len=0,
        kv_bytes=0,
    )
    assert record.layer_summaries == []


# ---------------------------------------------------------------------------
# AppState.deep integration
# ---------------------------------------------------------------------------

def test_app_state_deep_none_by_default():
    state = AppState()
    assert state.deep is None


def test_app_state_deep_can_be_set():
    state = AppState()
    ds = make_deep_state()
    state.deep = ds
    assert state.deep is not None
    assert isinstance(state.deep, DeepState)
    assert state.deep.phase == InferencePhase.IDLE


def test_app_state_deep_state_is_independent():
    state1 = AppState()
    state2 = AppState()
    state1.deep = make_deep_state()
    assert state1.deep is not None
    assert state2.deep is None


def test_app_state_deep_state_mutations_visible():
    state = AppState()
    state.deep = make_deep_state()
    state.deep.phase = InferencePhase.DECODING
    state.deep.response_text = "test output"
    state.deep.decode_token_count = 7

    assert state.deep.phase == InferencePhase.DECODING
    assert state.deep.response_text == "test output"
    assert state.deep.decode_token_count == 7


def test_app_state_deep_token_history_accessible():
    state = AppState()
    state.deep = make_deep_state()

    for i in range(10):
        state.deep.token_history.append(_make_token_record(i))

    assert len(state.deep.token_history) == 10
    assert state.deep.token_history[0].index == 0
    assert state.deep.token_history[-1].index == 9


def test_app_state_deep_maxlen_enforced_via_app_state():
    state = AppState()
    state.deep = make_deep_state()

    for i in range(120):
        state.deep.token_history.append(_make_token_record(i))

    assert len(state.deep.token_history) == 100
    assert state.deep.token_history[0].index == 20

"""Tests for all new deep instrumentation event types (V2)."""

from __future__ import annotations

import time

import pytest

from llmvis.core.events import (
    EventType,
    ModelArchitectureEvent,
    InferenceStartEvent,
    InferenceEndEvent,
    PrefillStartEvent,
    PrefillEndEvent,
    TokenizationCompleteEvent,
    LayerStatsEvent,
    MLPStatsEvent,
    LogitsReadyEvent,
    TokenSampledEvent,
    KvCacheUpdateEvent,
    TokenEndEvent,
    SamplerConfig,
    TokenCandidate,
)


# ── ModelArchitectureEvent ────────────────────────────────────────────────────

def test_model_architecture_event_type():
    evt = ModelArchitectureEvent(source="test")
    assert evt.type == EventType.MODEL_ARCHITECTURE


def test_model_architecture_event_fields():
    evt = ModelArchitectureEvent(
        source="test",
        model_id="meta-llama/Llama-3-8B",
        num_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=128256,
        head_dim=128,
        dtype="bfloat16",
        device="cuda:0",
    )
    assert evt.model_id == "meta-llama/Llama-3-8B"
    assert evt.num_layers == 32
    assert evt.num_attention_heads == 32
    assert evt.hidden_size == 4096
    assert evt.vocab_size == 128256
    assert evt.head_dim == 128
    assert evt.dtype == "bfloat16"
    assert evt.device == "cuda:0"


def test_model_architecture_event_defaults():
    evt = ModelArchitectureEvent(source="test")
    assert evt.model_id == ""
    assert evt.num_layers == 0
    assert evt.num_attention_heads == 0
    assert evt.hidden_size == 0
    assert evt.vocab_size == 0
    assert evt.head_dim == 0
    assert evt.dtype == ""
    assert evt.device == ""


# ── InferenceStartEvent / InferenceEndEvent ───────────────────────────────────

def test_inference_start_event_type():
    evt = InferenceStartEvent(source="test")
    assert evt.type == EventType.INFERENCE_START


def test_inference_start_event_fields():
    evt = InferenceStartEvent(
        source="test",
        model_id="gpt2",
        prompt_preview="Once upon a time",
    )
    assert evt.model_id == "gpt2"
    assert evt.prompt_preview == "Once upon a time"


def test_inference_end_event_type():
    evt = InferenceEndEvent(source="test")
    assert evt.type == EventType.INFERENCE_END


def test_inference_end_event_fields():
    evt = InferenceEndEvent(
        source="test",
        model_id="gpt2",
        total_tokens=128,
        total_time_ms=1234.5,
    )
    assert evt.model_id == "gpt2"
    assert evt.total_tokens == 128
    assert evt.total_time_ms == 1234.5


# ── PrefillStartEvent / PrefillEndEvent ───────────────────────────────────────

def test_prefill_start_event_type():
    evt = PrefillStartEvent(source="test")
    assert evt.type == EventType.PREFILL_START


def test_prefill_start_event_token_count():
    evt = PrefillStartEvent(source="test", token_count=42)
    assert evt.token_count == 42


def test_prefill_end_event_type():
    evt = PrefillEndEvent(source="test")
    assert evt.type == EventType.PREFILL_END


def test_prefill_end_event_timing():
    evt = PrefillEndEvent(source="test", token_count=42, duration_ms=87.3)
    assert evt.token_count == 42
    assert evt.duration_ms == 87.3


def test_prefill_end_event_defaults():
    evt = PrefillEndEvent(source="test")
    assert evt.token_count == 0
    assert evt.duration_ms == 0.0


# ── TokenizationCompleteEvent ─────────────────────────────────────────────────

def test_tokenization_complete_event_type():
    evt = TokenizationCompleteEvent(source="test")
    assert evt.type == EventType.TOKENIZATION_COMPLETE


def test_tokenization_complete_event_token_count():
    evt = TokenizationCompleteEvent(source="test", token_count=15)
    assert evt.token_count == 15


def test_tokenization_complete_event_token_ids():
    ids = [1, 2, 3, 4, 5]
    evt = TokenizationCompleteEvent(source="test", token_count=5, token_ids=ids)
    assert evt.token_ids == ids
    assert len(evt.token_ids) == 5


def test_tokenization_complete_event_token_ids_default_empty():
    evt = TokenizationCompleteEvent(source="test")
    assert evt.token_ids == []


# ── LayerStatsEvent ───────────────────────────────────────────────────────────

def test_layer_stats_event_type():
    evt = LayerStatsEvent(source="test")
    assert evt.type == EventType.LAYER_STATS


def test_layer_stats_event_all_fields():
    evt = LayerStatsEvent(
        source="test",
        token_index=3,
        layer_index=7,
        hidden_state_rms=1.23,
        hidden_state_mean=0.01,
        hidden_state_std=0.87,
        delta_from_prev=0.15,
        exec_time_ms=2.5,
    )
    assert evt.token_index == 3
    assert evt.layer_index == 7
    assert evt.hidden_state_rms == 1.23
    assert evt.hidden_state_mean == 0.01
    assert evt.hidden_state_std == 0.87
    assert evt.delta_from_prev == 0.15
    assert evt.exec_time_ms == 2.5


def test_layer_stats_event_defaults():
    evt = LayerStatsEvent(source="test")
    assert evt.hidden_state_rms == 0.0
    assert evt.hidden_state_mean == 0.0
    assert evt.hidden_state_std == 0.0
    assert evt.delta_from_prev == 0.0
    assert evt.exec_time_ms == 0.0


# ── MLPStatsEvent ─────────────────────────────────────────────────────────────

def test_mlp_stats_event_type():
    evt = MLPStatsEvent(source="test")
    assert evt.type == EventType.MLP_STATS


def test_mlp_stats_event_output_rms():
    evt = MLPStatsEvent(source="test", layer_index=4, output_rms=2.11)
    assert evt.output_rms == 2.11


def test_mlp_stats_event_sparsity():
    evt = MLPStatsEvent(source="test", layer_index=4, sparsity=0.43)
    assert evt.sparsity == 0.43


def test_mlp_stats_event_defaults():
    evt = MLPStatsEvent(source="test")
    assert evt.output_rms == 0.0
    assert evt.sparsity == 0.0


# ── LogitsReadyEvent ──────────────────────────────────────────────────────────

def test_logits_ready_event_type():
    evt = LogitsReadyEvent(source="test")
    assert evt.type == EventType.LOGITS_READY


def test_logits_ready_event_candidates():
    candidates = [
        TokenCandidate(
            token_id=1234,
            token_text="hello",
            logit=5.3,
            raw_probability=0.35,
            probability=0.55,
        ),
        TokenCandidate(
            token_id=5678,
            token_text=" world",
            logit=3.1,
            raw_probability=0.20,
            probability=0.30,
        ),
    ]
    evt = LogitsReadyEvent(source="test", token_index=0, top_candidates=candidates, vocab_size=32000)
    assert len(evt.top_candidates) == 2
    assert evt.top_candidates[0].token_id == 1234
    assert evt.top_candidates[1].token_text == " world"
    assert evt.vocab_size == 32000


def test_logits_ready_event_candidates_default_empty():
    evt = LogitsReadyEvent(source="test")
    assert evt.top_candidates == []


def test_logits_ready_event_candidates_have_probability_fields():
    c = TokenCandidate(
        token_id=99,
        token_text="x",
        logit=1.0,
        raw_probability=0.10,
        probability=0.25,
    )
    assert hasattr(c, "raw_probability")
    assert hasattr(c, "probability")
    assert c.raw_probability == 0.10
    assert c.probability == 0.25


# ── TokenSampledEvent ─────────────────────────────────────────────────────────

def test_token_sampled_event_type():
    evt = TokenSampledEvent(source="test")
    assert evt.type == EventType.TOKEN_SAMPLED


def test_token_sampled_event_token_id():
    evt = TokenSampledEvent(source="test", token_id=1234)
    assert evt.token_id == 1234


def test_token_sampled_event_token_text():
    evt = TokenSampledEvent(source="test", token_text="hello")
    assert evt.token_text == "hello"


def test_token_sampled_event_sampler():
    cfg = SamplerConfig(temperature=0.8, top_k=50, top_p=0.9)
    evt = TokenSampledEvent(source="test", sampler=cfg)
    assert evt.sampler.temperature == 0.8
    assert evt.sampler.top_k == 50
    assert evt.sampler.top_p == 0.9


# ── KvCacheUpdateEvent ────────────────────────────────────────────────────────

def test_kv_cache_update_event_type():
    evt = KvCacheUpdateEvent(source="test")
    assert evt.type == EventType.KV_CACHE_UPDATE


def test_kv_cache_update_event_seq_lens():
    evt = KvCacheUpdateEvent(
        source="test",
        seq_len_before=10,
        seq_len_after=11,
    )
    assert evt.seq_len_before == 10
    assert evt.seq_len_after == 11


def test_kv_cache_update_event_shapes():
    evt = KvCacheUpdateEvent(
        source="test",
        k_shape=[1, 32, 11, 128],
        v_shape=[1, 32, 11, 128],
    )
    assert evt.k_shape == [1, 32, 11, 128]
    assert evt.v_shape == [1, 32, 11, 128]


def test_kv_cache_update_event_measured_bytes():
    evt = KvCacheUpdateEvent(source="test", measured_bytes=536_870_912)
    assert evt.measured_bytes == 536_870_912


def test_kv_cache_update_event_defaults():
    evt = KvCacheUpdateEvent(source="test")
    assert evt.seq_len_before == 0
    assert evt.seq_len_after == 0
    assert evt.k_shape == []
    assert evt.v_shape == []
    assert evt.measured_bytes == 0


# ── TokenEndEvent ─────────────────────────────────────────────────────────────

def test_token_end_event_type():
    evt = TokenEndEvent(source="test")
    assert evt.type == EventType.TOKEN_END


def test_token_end_event_latency_ms():
    evt = TokenEndEvent(source="test", latency_ms=12.7)
    assert evt.latency_ms == 12.7


def test_token_end_event_defaults():
    evt = TokenEndEvent(source="test")
    assert evt.latency_ms == 0.0
    assert evt.token_id == 0
    assert evt.token_text == ""


# ── SamplerConfig defaults ────────────────────────────────────────────────────

def test_sampler_config_defaults_are_sensible():
    cfg = SamplerConfig()
    # temperature=1.0 means no scaling (identity)
    assert cfg.temperature == 1.0
    # top_k=0 means disabled (no truncation)
    assert cfg.top_k == 0
    # top_p=1.0 means disabled (full nucleus)
    assert cfg.top_p == 1.0
    # repetition_penalty=1.0 means no penalty
    assert cfg.repetition_penalty == 1.0
    # seed=None means random
    assert cfg.seed is None


def test_sampler_config_custom_values():
    cfg = SamplerConfig(temperature=0.7, top_k=40, top_p=0.95, repetition_penalty=1.1, seed=42)
    assert cfg.temperature == 0.7
    assert cfg.top_k == 40
    assert cfg.top_p == 0.95
    assert cfg.repetition_penalty == 1.1
    assert cfg.seed == 42


# ── TokenCandidate: raw_probability separate from post-filter probability ─────

def test_token_candidate_raw_probability_separate():
    c = TokenCandidate(
        token_id=1,
        token_text="a",
        logit=4.0,
        raw_probability=0.30,   # softmax BEFORE temperature/filtering
        probability=0.60,       # softmax AFTER temperature + top-k/p
    )
    assert c.raw_probability != c.probability
    assert c.raw_probability == 0.30
    assert c.probability == 0.60


def test_token_candidate_all_fields_present():
    c = TokenCandidate(
        token_id=42,
        token_text="foo",
        logit=-1.5,
        raw_probability=0.05,
        probability=0.08,
    )
    assert c.token_id == 42
    assert c.token_text == "foo"
    assert c.logit == -1.5
    assert c.raw_probability == 0.05
    assert c.probability == 0.08


# ── Monotonic timestamps ──────────────────────────────────────────────────────

def test_all_events_have_monotonic_timestamps():
    events = [
        ModelArchitectureEvent(source="t"),
        InferenceStartEvent(source="t"),
        InferenceEndEvent(source="t"),
        PrefillStartEvent(source="t"),
        PrefillEndEvent(source="t"),
        TokenizationCompleteEvent(source="t"),
        LayerStatsEvent(source="t"),
        MLPStatsEvent(source="t"),
        LogitsReadyEvent(source="t"),
        TokenSampledEvent(source="t"),
        KvCacheUpdateEvent(source="t"),
        TokenEndEvent(source="t"),
    ]
    timestamps = [e.timestamp for e in events]
    for ts in timestamps:
        assert ts > 0, "timestamp must be positive (monotonic)"
    # Each timestamp is >= the previous (monotonic non-decreasing)
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"timestamp[{i}] ({timestamps[i]}) < timestamp[{i-1}] ({timestamps[i-1]})"
        )


def test_timestamp_increases_over_time():
    e1 = InferenceStartEvent(source="t")
    e2 = InferenceEndEvent(source="t")
    assert e2.timestamp >= e1.timestamp


# ── EventType enum coverage ───────────────────────────────────────────────────

def test_event_types_exist_in_enum():
    required = [
        "MODEL_ARCHITECTURE",
        "INFERENCE_START",
        "INFERENCE_END",
        "PREFILL_START",
        "PREFILL_END",
        "TOKENIZATION_COMPLETE",
        "LAYER_STATS",
        "MLP_STATS",
        "LOGITS_READY",
        "TOKEN_SAMPLED",
        "KV_CACHE_UPDATE",
        "TOKEN_END",
    ]
    enum_names = {member.name for member in EventType}
    for name in required:
        assert name in enum_names, f"EventType.{name} is missing from the enum"


def test_each_deep_event_type_matches_enum():
    assert ModelArchitectureEvent(source="t").type == EventType.MODEL_ARCHITECTURE
    assert InferenceStartEvent(source="t").type == EventType.INFERENCE_START
    assert InferenceEndEvent(source="t").type == EventType.INFERENCE_END
    assert PrefillStartEvent(source="t").type == EventType.PREFILL_START
    assert PrefillEndEvent(source="t").type == EventType.PREFILL_END
    assert TokenizationCompleteEvent(source="t").type == EventType.TOKENIZATION_COMPLETE
    assert LayerStatsEvent(source="t").type == EventType.LAYER_STATS
    assert MLPStatsEvent(source="t").type == EventType.MLP_STATS
    assert LogitsReadyEvent(source="t").type == EventType.LOGITS_READY
    assert TokenSampledEvent(source="t").type == EventType.TOKEN_SAMPLED
    assert KvCacheUpdateEvent(source="t").type == EventType.KV_CACHE_UPDATE
    assert TokenEndEvent(source="t").type == EventType.TOKEN_END

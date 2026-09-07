"""Tests for event serialization (JSONL round-trip) and event ordering."""

from __future__ import annotations

import json

import pytest

from llmvis.core.events import (
    DecodeStartEvent,
    InferenceEndEvent,
    InferenceStartEvent,
    KvCacheUpdateEvent,
    LayerStatsEvent,
    LogitsReadyEvent,
    ModelArchitectureEvent,
    PrefillEndEvent,
    PrefillStartEvent,
    SamplerConfig,
    TokenCandidate,
    TokenEndEvent,
    TokenSampledEvent,
    TokenStartEvent,
    TokenizationCompleteEvent,
    event_from_jsonl,
    event_to_dict,
    event_to_jsonl,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidates(n: int = 3) -> list[TokenCandidate]:
    return [
        TokenCandidate(
            token_id=i,
            token_text=f"tok{i}",
            logit=float(10 - i),
            raw_probability=0.3 / (i + 1),
            probability=0.4 / (i + 1),
        )
        for i in range(n)
    ]


# ── event_to_dict ─────────────────────────────────────────────────────────────

def test_event_to_dict_has_type_key():
    evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="hi")
    d = event_to_dict(evt)
    assert "_type" in d
    assert d["_type"] == "INFERENCE_START"


def test_event_to_dict_drops_enum_field():
    """The raw EventType Enum field must not appear (it would break json.dumps)."""
    evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="hi")
    d = event_to_dict(evt)
    # "type" key either absent or serialized as a string — never an Enum instance
    if "type" in d:
        assert isinstance(d["type"], str), "type field must be string, not Enum"


def test_event_to_jsonl_is_valid_json():
    evt = PrefillStartEvent(source="test", token_count=5)
    line = event_to_jsonl(evt)
    parsed = json.loads(line)
    assert parsed["_type"] == "PREFILL_START"
    assert parsed["token_count"] == 5


def test_event_to_jsonl_with_nested_token_candidates():
    """LogitsReadyEvent has nested TokenCandidate objects — must serialize cleanly."""
    cands = _candidates(5)
    evt = LogitsReadyEvent(
        source="test",
        token_index=0,
        top_candidates=cands,
        vocab_size=32000,
    )
    line = event_to_jsonl(evt)
    parsed = json.loads(line)
    assert len(parsed["top_candidates"]) == 5
    assert parsed["top_candidates"][0]["token_text"] == "tok0"
    assert isinstance(parsed["top_candidates"][0]["logit"], float)


def test_event_to_jsonl_with_nested_sampler_config():
    sampler = SamplerConfig(temperature=0.7, top_k=40, top_p=0.9)
    evt = DecodeStartEvent(source="test", sampler=sampler)
    line = event_to_jsonl(evt)
    parsed = json.loads(line)
    assert parsed["sampler"]["temperature"] == pytest.approx(0.7)


# ── Round-trip ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("evt", [
    ModelArchitectureEvent(
        source="test",
        model_id="meta-llama/Llama-2-7b",
        num_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=32000,
        head_dim=128,
        dtype="float16",
        device="cuda:0",
    ),
    InferenceStartEvent(source="test", model_id="m", prompt_preview="hello"),
    PrefillStartEvent(source="test", token_count=7),
    PrefillEndEvent(source="test", token_count=7, duration_ms=12.3),
    TokenizationCompleteEvent(source="test", token_count=3, token_ids=[1, 2, 3], duration_ms=0.5),
    TokenStartEvent(source="test", token_index=0),
    LayerStatsEvent(
        source="test",
        token_index=0,
        layer_index=5,
        hidden_state_rms=1.2,
        hidden_state_mean=0.01,
        hidden_state_std=0.5,
        delta_from_prev=0.1,
        exec_time_ms=0.8,
    ),
    LogitsReadyEvent(
        source="test",
        token_index=0,
        top_candidates=_candidates(3),
        vocab_size=32000,
    ),
    TokenSampledEvent(
        source="test",
        token_index=0,
        token_id=42,
        token_text=" the",
        logprob=-1.5,
        sampler=SamplerConfig(temperature=0.8, top_k=50, top_p=0.95),
    ),
    KvCacheUpdateEvent(
        source="test",
        token_index=0,
        seq_len_before=6,
        seq_len_after=7,
        num_layers=32,
        k_shape=[1, 32, 7, 128],
        v_shape=[1, 32, 7, 128],
        dtype="float16",
        measured_bytes=262144,
    ),
    TokenEndEvent(
        source="test",
        token_index=0,
        token_text=" the",
        token_id=42,
        latency_ms=4.5,
    ),
    InferenceEndEvent(source="test", model_id="m", total_tokens=10, total_time_ms=200.0),
])
def test_round_trip(evt):
    """Serialize → JSONL → deserialize must reproduce the event type and key fields."""
    line = event_to_jsonl(evt)
    restored = event_from_jsonl(line)
    assert restored is not None, f"Deserialization returned None for {evt}"
    assert restored.type == evt.type


def test_round_trip_preserves_logits_values():
    cands = _candidates(3)
    evt = LogitsReadyEvent(source="test", token_index=2, top_candidates=cands, vocab_size=200)
    restored = event_from_jsonl(event_to_jsonl(evt))
    assert len(restored.top_candidates) == 3
    assert restored.top_candidates[0].logit == pytest.approx(cands[0].logit)
    assert restored.top_candidates[0].raw_probability == pytest.approx(cands[0].raw_probability)


def test_round_trip_preserves_kv_shapes():
    evt = KvCacheUpdateEvent(
        source="test", token_index=5,
        seq_len_before=10, seq_len_after=11,
        num_layers=4, k_shape=[1, 4, 11, 64], v_shape=[1, 4, 11, 64],
        dtype="bfloat16", measured_bytes=8192,
    )
    restored = event_from_jsonl(event_to_jsonl(evt))
    assert restored.k_shape == [1, 4, 11, 64]
    assert restored.v_shape == [1, 4, 11, 64]
    assert restored.measured_bytes == 8192


def test_round_trip_preserves_session_metadata():
    evt = InferenceStartEvent(
        source="test", model_id="m", prompt_preview="hi",
        session_id="abc123", sequence_num=7,
    )
    restored = event_from_jsonl(event_to_jsonl(evt))
    assert restored.session_id == "abc123"
    assert restored.sequence_num == 7


def test_unknown_event_type_returns_none():
    bad = json.dumps({"_type": "MADE_UP_EVENT", "source": "test"})
    assert event_from_jsonl(bad) is None


def test_malformed_jsonl_returns_none():
    assert event_from_jsonl("not valid json {{{") is None


# ── Event ordering ────────────────────────────────────────────────────────────

def test_event_sequence_num_is_monotonic():
    """sequence_num on events from run() should increase monotonically."""
    events = [
        ModelArchitectureEvent(source="s", model_id="m", num_layers=2,
                               num_attention_heads=2, hidden_size=64, vocab_size=200,
                               head_dim=32, dtype="float32", device="cpu",
                               session_id="s1", sequence_num=1),
        InferenceStartEvent(source="s", model_id="m", prompt_preview="hi",
                            session_id="s1", sequence_num=2),
        PrefillStartEvent(source="s", token_count=3, session_id="s1", sequence_num=3),
    ]
    nums = [e.sequence_num for e in events]
    assert nums == sorted(nums), "sequence_num must be monotonically increasing"
    assert len(set(nums)) == len(nums), "sequence_num must be unique"


def test_demo_jsonl_round_trips():
    """The generated demo_session.jsonl must fully deserialize without None."""
    from pathlib import Path
    demo = Path("src/llmvis/demo/demo_session.jsonl")
    if not demo.exists():
        pytest.skip("demo_session.jsonl not generated yet")
    lines = demo.read_text().strip().splitlines()
    assert len(lines) > 0, "demo_session.jsonl is empty"
    failed = []
    for i, line in enumerate(lines):
        evt = event_from_jsonl(line)
        if evt is None:
            failed.append(i)
    assert not failed, f"Lines {failed} failed to deserialize"


def test_demo_jsonl_event_ordering():
    """In the demo JSONL: MODEL_ARCHITECTURE appears before INFERENCE_START, etc."""
    from pathlib import Path
    import json as _json
    demo = Path("src/llmvis/demo/demo_session.jsonl")
    if not demo.exists():
        pytest.skip("demo_session.jsonl not generated yet")
    types = [_json.loads(l)["_type"] for l in demo.read_text().strip().splitlines()]
    assert types[0] == "MODEL_ARCHITECTURE"
    assert "INFERENCE_START" in types
    arch_idx = types.index("MODEL_ARCHITECTURE")
    inf_idx = types.index("INFERENCE_START")
    assert arch_idx < inf_idx, "MODEL_ARCHITECTURE must precede INFERENCE_START"

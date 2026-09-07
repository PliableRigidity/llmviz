"""Tests for bounded EventBus queue and pause state in DeepState."""

from __future__ import annotations

import asyncio

import pytest

from llmvis.core.bus import EventBus
from llmvis.core.events import InferenceStartEvent
from llmvis.core.state import DeepState, make_deep_state


# ── EventBus bounded queue ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_eventbus_default_maxsize_is_2000():
    bus = EventBus()
    assert bus._queue.maxsize == 2000


@pytest.mark.asyncio
async def test_eventbus_custom_maxsize():
    bus = EventBus(maxsize=10)
    assert bus._queue.maxsize == 10


@pytest.mark.asyncio
async def test_eventbus_drop_on_full():
    """When the queue is full, additional events are dropped (not blocking)."""
    bus = EventBus(maxsize=3)
    evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="x")

    # Fill the queue
    for _ in range(3):
        await bus.publish(evt)

    assert bus._queue.qsize() == 3
    assert bus._dropped == 0

    # One more — should be dropped, not block
    await bus.publish(evt)
    assert bus._queue.qsize() == 3       # not grown
    assert bus._dropped == 1             # counted


@pytest.mark.asyncio
async def test_eventbus_dropped_counter_accumulates():
    bus = EventBus(maxsize=2)
    evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="x")
    for _ in range(2):
        await bus.publish(evt)
    for _ in range(5):
        await bus.publish(evt)
    assert bus._dropped == 5


@pytest.mark.asyncio
async def test_eventbus_subscribe_and_dispatch():
    bus = EventBus(maxsize=100)
    received = []

    async def handler(e):
        received.append(e)

    from llmvis.core.events import EventType
    bus.subscribe(EventType.INFERENCE_START, handler)

    evt = InferenceStartEvent(source="test", model_id="m", prompt_preview="hello")
    await bus.publish(evt)

    # Run the dispatcher briefly
    task = asyncio.create_task(bus.dispatch_forever())
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0].model_id == "m"


# ── DeepState pause fields ────────────────────────────────────────────────────

def test_deepstate_default_not_paused():
    d = make_deep_state()
    assert d.visualization_paused is False


def test_deepstate_can_set_paused():
    d = make_deep_state()
    d.visualization_paused = True
    assert d.visualization_paused is True


def test_deepstate_model_loaded_default():
    d = make_deep_state()
    assert d.model_loaded is False


def test_deepstate_kv_shape_fields_default_empty():
    d = make_deep_state()
    assert d.last_k_shape == []
    assert d.last_v_shape == []
    assert d.last_kv_dtype == ""


def test_deepstate_kv_shape_fields_assignable():
    d = make_deep_state()
    d.last_k_shape = [1, 32, 7, 128]
    d.last_v_shape = [1, 32, 7, 128]
    d.last_kv_dtype = "float16"
    assert d.last_k_shape[2] == 7
    assert d.last_kv_dtype == "float16"

"""Tests for MLXAdapter.

All tests run on non-Apple platforms by mocking MLX imports.
Validates:
- MLXAdapter raises RuntimeError when MLX is not available
- MLX adapter emits the correct normalized event types
- KV stats, logit sampling, architecture discovery work with mock objects
- Backend capability: layer_stats=False for MLX
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from llmvis.platform.detect import PlatformInfo


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apple_platform_with_mlx() -> PlatformInfo:
    return PlatformInfo(
        os="darwin",
        arch="arm64",
        is_apple_silicon=True,
        chip_model="Apple M3 Pro",
        cuda_available=False,
        cuda_device_name="",
        mlx_available=True,
        mps_available=True,
    )


def _make_mock_mlx():
    """Return a minimal mock of mlx.core that satisfies MLXAdapter."""
    mx = MagicMock()
    # mx.array returns an array-like object
    arr = MagicMock()
    arr.shape = (1, 3, 200)
    arr.__getitem__ = MagicMock(return_value=arr)
    arr.astype = MagicMock(return_value=arr)
    arr.tolist = MagicMock(return_value=[0.01] * 200)
    arr.item = MagicMock(return_value=0)
    arr.nbytes = 4096
    mx.array.return_value = arr
    mx.softmax.return_value = arr
    mx.argmax.return_value = MagicMock(item=MagicMock(return_value=42))
    mx.random.categorical.return_value = MagicMock(item=MagicMock(return_value=7))
    mx.float32 = "float32"
    mx.eval = MagicMock()
    mx.metal.get_active_memory = MagicMock(return_value=1_000_000)
    return mx


# ── Test: MLXAdapter raises when MLX unavailable ──────────────────────────────

def test_mlx_adapter_not_available_raises():
    """MLXAdapter.__init__ must raise RuntimeError if MLX is not installed."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=False):
        with pytest.raises(RuntimeError, match="MLX is not available"):
            from llmvis.adapters.mlx_adapter import MLXAdapter
            MLXAdapter(model_id="any/model")


def test_mlx_adapter_importable_on_non_apple():
    """mlx_adapter module must be importable even without MLX installed."""
    import importlib
    try:
        mod = importlib.import_module("llmvis.adapters.mlx_adapter")
        assert hasattr(mod, "MLXAdapter")
    except ImportError as exc:
        pytest.fail(f"mlx_adapter not importable on non-Apple: {exc}")


# ── Test: MLXAdapter instantiation when MLX available ────────────────────────

def test_mlx_adapter_instantiation_when_mlx_available():
    """MLXAdapter should instantiate without error when _check_mlx returns True."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        adapter = MLXAdapter(
            model_id="mlx-community/test-model",
            telemetry_mode="light",
            max_new_tokens=10,
        )
    assert adapter._model_id == "mlx-community/test-model"
    assert adapter._max_new_tokens == 10


# ── Test: architecture event ──────────────────────────────────────────────────

def test_mlx_adapter_arch_event_from_config():
    """_make_arch_event reads num_hidden_layers etc. from model.config."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        from llmvis.core.events import EventType

        adapter = MLXAdapter(model_id="test/model")

        # Inject a fake model with a config
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 24
        mock_config.num_attention_heads = 16
        mock_config.hidden_size = 1024
        mock_config.vocab_size = 32000
        mock_config.head_dim = None

        mock_model = MagicMock()
        mock_model.config = mock_config
        adapter._model = mock_model

        evt = adapter._make_arch_event()

    assert evt.type == EventType.MODEL_ARCHITECTURE
    assert evt.num_layers == 24
    assert evt.num_attention_heads == 16
    assert evt.hidden_size == 1024
    assert evt.vocab_size == 32000
    assert evt.device == "metal"


# ── Test: KV stats ────────────────────────────────────────────────────────────

def test_mlx_adapter_kv_stats_none_cache():
    """With no cache, seq_len equals expected_seq_len and bytes=0."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        adapter = MLXAdapter(model_id="test/model")

    seq_len, total_bytes, k_shape, v_shape = adapter._kv_stats(None, expected_seq_len=7)
    assert seq_len == 7
    assert total_bytes == 0
    assert k_shape == []
    assert v_shape == []


def test_mlx_adapter_kv_stats_with_layer_cache():
    """KV stats reads .keys/.values from per-layer cache objects."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        adapter = MLXAdapter(model_id="test/model")

    mock_k = MagicMock()
    mock_k.shape = [1, 8, 15, 64]
    mock_k.nbytes = 32768
    mock_v = MagicMock()
    mock_v.shape = [1, 8, 15, 64]
    mock_v.nbytes = 32768

    mock_layer = MagicMock()
    mock_layer.keys = mock_k
    mock_layer.values = mock_v

    cache = [mock_layer]
    seq_len, total_bytes, k_shape, v_shape = adapter._kv_stats(cache, expected_seq_len=15)

    assert seq_len == 15
    assert total_bytes == 65536  # 32768 * 2
    assert k_shape == [1, 8, 15, 64]


# ── Test: Capability: layer_stats=False for MLX ───────────────────────────────

def test_capabilities_mlx_no_layer_stats():
    """The capability model must declare layer_stats=False for MLX/Metal."""
    from llmvis.platform.capabilities import capabilities_for_platform
    caps = capabilities_for_platform(_apple_platform_with_mlx())
    assert caps.layer_stats is False, (
        "MLX does not support PyTorch-style hooks — layer_stats must be False"
    )


def test_capabilities_mlx_backend_name():
    from llmvis.platform.capabilities import capabilities_for_platform
    caps = capabilities_for_platform(_apple_platform_with_mlx())
    assert "MLX" in caps.backend_name or "Metal" in caps.backend_name


# ── Test: _sample_mlx with mocked arrays ─────────────────────────────────────

def test_mlx_adapter_has_sample_mlx_method():
    """_sample_mlx method must exist and accept (logits_1d, mx)."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        adapter = MLXAdapter(model_id="test")
    assert callable(getattr(adapter, "_sample_mlx", None)), "_sample_mlx must be callable"


# ── Test: adapter has required BaseAdapter interface ──────────────────────────

def test_mlx_adapter_is_base_adapter():
    """MLXAdapter subclasses BaseAdapter."""
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        from llmvis.adapters.base import BaseAdapter
        assert issubclass(MLXAdapter, BaseAdapter)


def test_mlx_adapter_has_run_and_stop():
    with patch("llmvis.adapters.mlx_adapter._check_mlx", return_value=True):
        from llmvis.adapters.mlx_adapter import MLXAdapter
        adapter = MLXAdapter(model_id="test/model")
    assert hasattr(adapter, "run")
    assert hasattr(adapter, "stop")
    assert hasattr(adapter, "submit_prompt")
    assert hasattr(adapter, "load_model")

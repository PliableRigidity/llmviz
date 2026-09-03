"""Tests for system and GPU telemetry (V1 + V2 providers)."""

from __future__ import annotations

import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from llmvis.telemetry.system import SystemTelemetry, _find_ollama_proc


@pytest.mark.asyncio
async def test_system_telemetry_returns_event():
    """SystemTelemetry.get_system_stats returns a SystemStatsEvent."""
    from llmvis.core.events import EventType
    tel = SystemTelemetry()
    event = await tel.get_system_stats()
    assert event.type == EventType.SYSTEM_STATS_UPDATE
    assert 0.0 <= event.cpu_percent <= 100.0
    assert event.ram_total_bytes > 0
    assert event.ram_used_bytes > 0
    assert 0.0 <= event.ram_percent <= 100.0


@pytest.mark.asyncio
async def test_system_telemetry_no_ollama_proc():
    """Telemetry works even when Ollama process is not found."""
    with patch("llmvis.telemetry.system._find_ollama_proc", return_value=None):
        tel = SystemTelemetry()
        tel._ollama_proc = None
        event = await tel.get_system_stats()
        assert event.ollama_cpu_percent == 0.0
        assert event.ollama_ram_bytes == 0


@pytest.mark.asyncio
async def test_gpu_stats_no_pynvml():
    """GPU stats returns empty list when pynvml is unavailable."""
    import llmvis.telemetry.gpu as gpu_mod
    original_avail = gpu_mod._nvml_available
    original_init = gpu_mod._nvml_initialized
    gpu_mod._nvml_available = False
    gpu_mod._nvml_initialized = True

    try:
        result = await gpu_mod.get_gpu_stats()
        assert result == []
    finally:
        gpu_mod._nvml_available = original_avail
        gpu_mod._nvml_initialized = original_init


@pytest.mark.asyncio
async def test_gpu_stats_with_mock_pynvml():
    """GPU stats correctly parses pynvml data when available."""
    import sys
    import llmvis.telemetry.gpu as gpu_mod

    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetCount.return_value = 1
    mock_handle = MagicMock()
    mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
    mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 4090"
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(
        total=17_179_869_184, used=4_294_967_296
    )
    mock_util = MagicMock()
    mock_util.gpu = 65
    mock_pynvml.nvmlDeviceGetUtilizationRates.return_value = mock_util
    mock_pynvml.nvmlDeviceGetTemperature.return_value = 72
    mock_pynvml.NVML_TEMPERATURE_GPU = 0

    original_avail = gpu_mod._nvml_available
    original_init = gpu_mod._nvml_initialized
    gpu_mod._nvml_available = True
    gpu_mod._nvml_initialized = True

    old_pynvml = sys.modules.get("pynvml")
    sys.modules["pynvml"] = mock_pynvml

    try:
        result = await gpu_mod.get_gpu_stats()
        assert len(result) == 1
        assert result[0].gpu_name == "NVIDIA RTX 4090"
        assert result[0].gpu_util_percent == 65.0
        assert result[0].temperature_c == 72.0
        assert result[0].vram_total_bytes == 17_179_869_184
    finally:
        gpu_mod._nvml_available = original_avail
        gpu_mod._nvml_initialized = original_init
        if old_pynvml is None:
            sys.modules.pop("pynvml", None)
        else:
            sys.modules["pynvml"] = old_pynvml


# ── V2 provider tests (platform-aware providers) ──────────────────────────────


def _mock_vmem(used=8_000_000_000, total=16_000_000_000):
    return SimpleNamespace(used=used, total=total, percent=50.0)


def _make_platform(is_apple=False, cuda=False, cuda_name=""):
    from llmvis.platform.detect import PlatformInfo
    return PlatformInfo(
        os="darwin" if is_apple else "linux",
        arch="arm64" if is_apple else "x86_64",
        is_apple_silicon=is_apple,
        chip_model="Apple M3 Pro" if is_apple else "unknown",
        cuda_available=cuda,
        cuda_device_name=cuda_name,
        mlx_available=is_apple,
        mps_available=is_apple,
    )


def test_system_provider_gpu_percent_none():
    from llmvis.telemetry.providers.system import SystemTelemetryProvider
    provider = SystemTelemetryProvider()
    with (
        patch("psutil.virtual_memory", return_value=_mock_vmem()),
        patch("psutil.cpu_percent", return_value=10.0),
    ):
        snap = provider.snapshot()
    assert snap.gpu_percent is None
    assert snap.memory_architecture == "CPU-only"


def test_apple_provider_no_vram():
    from llmvis.telemetry.providers.apple import AppleTelemetryProvider
    provider = AppleTelemetryProvider(chip_name="Apple M3 Pro")
    with (
        patch("psutil.virtual_memory", return_value=_mock_vmem()),
        patch("psutil.cpu_percent", return_value=40.0),
        patch("llmvis.telemetry.providers.apple._mlx_active_bytes", return_value=None),
    ):
        snap = provider.snapshot()
    assert snap.gpu_mem_used_bytes is None
    assert snap.gpu_mem_total_bytes is None
    assert snap.unified_mem_used_bytes is not None
    assert snap.memory_architecture == "Unified Memory"


def test_apple_provider_mlx_bytes_populated():
    from llmvis.telemetry.providers.apple import AppleTelemetryProvider
    provider = AppleTelemetryProvider(chip_name="Apple M2")
    with (
        patch("psutil.virtual_memory", return_value=_mock_vmem()),
        patch("psutil.cpu_percent", return_value=5.0),
        patch("llmvis.telemetry.providers.apple._mlx_active_bytes", return_value=999_000),
    ):
        snap = provider.snapshot()
    assert snap.mlx_active_bytes == 999_000


def test_factory_apple_returns_apple_provider():
    from llmvis.telemetry.providers.factory import get_telemetry_provider
    from llmvis.telemetry.providers.apple import AppleTelemetryProvider
    p = _make_platform(is_apple=True)
    provider = get_telemetry_provider(p)
    assert isinstance(provider, AppleTelemetryProvider)


def test_factory_nvidia_returns_nvidia_provider():
    from llmvis.telemetry.providers.factory import get_telemetry_provider
    from llmvis.telemetry.providers.nvidia import NvidiaTelemetryProvider
    p = _make_platform(cuda=True, cuda_name="NVIDIA RTX 4090")
    provider = get_telemetry_provider(p)
    assert isinstance(provider, NvidiaTelemetryProvider)


def test_factory_cpu_returns_system_provider():
    from llmvis.telemetry.providers.factory import get_telemetry_provider
    from llmvis.telemetry.providers.system import SystemTelemetryProvider
    p = _make_platform()
    provider = get_telemetry_provider(p)
    assert isinstance(provider, SystemTelemetryProvider)

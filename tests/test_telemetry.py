"""Tests for system and GPU telemetry."""

from __future__ import annotations

import pytest
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

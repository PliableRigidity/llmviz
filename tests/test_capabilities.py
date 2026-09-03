"""Tests for BackendCapabilities and capabilities_for_platform."""

from __future__ import annotations

import pytest

from llmvis.platform.detect import PlatformInfo
from llmvis.platform.capabilities import BackendCapabilities, capabilities_for_platform


def _apple_silicon_with_mlx() -> PlatformInfo:
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


def _apple_silicon_no_mlx() -> PlatformInfo:
    return PlatformInfo(
        os="darwin",
        arch="arm64",
        is_apple_silicon=True,
        chip_model="Apple M1",
        cuda_available=False,
        cuda_device_name="",
        mlx_available=False,
        mps_available=True,
    )


def _nvidia_linux() -> PlatformInfo:
    return PlatformInfo(
        os="linux",
        arch="x86_64",
        is_apple_silicon=False,
        chip_model="NVIDIA RTX 4090",
        cuda_available=True,
        cuda_device_name="NVIDIA GeForce RTX 4090",
        mlx_available=False,
        mps_available=False,
    )


def _cpu_only() -> PlatformInfo:
    return PlatformInfo(
        os="linux",
        arch="x86_64",
        is_apple_silicon=False,
        chip_model="unknown",
        cuda_available=False,
        cuda_device_name="",
        mlx_available=False,
        mps_available=False,
    )


# ── Apple Silicon ──────────────────────────────────────────────────────────────

def test_capabilities_for_apple_silicon_mlx():
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    assert "MLX" in caps.backend_name or "Metal" in caps.backend_name
    assert caps.unified_memory is True
    assert caps.dedicated_vram is False
    assert caps.memory_architecture == "Unified Memory"


def test_unified_memory_capability():
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    assert caps.unified_memory is True
    assert caps.dedicated_vram is False


def test_apple_silicon_no_gpu_utilization():
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    # powermetrics requires sudo — GPU util should be False
    assert caps.gpu_utilization is False


def test_apple_silicon_mps_fallback():
    """Apple Silicon with MPS but without MLX → PyTorch/MPS."""
    caps = capabilities_for_platform(_apple_silicon_no_mlx())
    assert "MPS" in caps.backend_name
    assert caps.unified_memory is True
    assert caps.dedicated_vram is False
    assert caps.layer_stats is True   # PyTorch hooks work on MPS


def test_apple_silicon_has_token_stream():
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    assert caps.token_stream is True


def test_apple_silicon_has_logits():
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    assert caps.logits is True


def test_apple_silicon_no_layer_stats_when_mlx():
    """MLX does not have PyTorch-style hooks — layer_stats must be False."""
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    assert caps.layer_stats is False


# ── NVIDIA ────────────────────────────────────────────────────────────────────

def test_capabilities_for_nvidia():
    caps = capabilities_for_platform(_nvidia_linux())
    assert "CUDA" in caps.backend_name
    assert caps.dedicated_vram is True
    assert caps.unified_memory is False
    assert caps.gpu_utilization is True
    assert caps.memory_architecture == "Discrete CPU/GPU"


def test_nvidia_capability_layer_stats():
    caps = capabilities_for_platform(_nvidia_linux())
    assert caps.layer_stats is True
    assert caps.attention_stats is True


def test_nvidia_capability_no_unified_memory():
    caps = capabilities_for_platform(_nvidia_linux())
    assert caps.unified_memory is False


# ── CPU-only ──────────────────────────────────────────────────────────────────

def test_cpu_only_capabilities():
    caps = capabilities_for_platform(_cpu_only())
    assert "CPU" in caps.backend_name
    assert caps.gpu_utilization is False
    assert caps.dedicated_vram is False
    assert caps.unified_memory is False
    assert caps.memory_architecture == "CPU-only"


def test_tui_no_vram_on_unified_memory():
    """TUI must not display VRAM panel when dedicated_vram is False."""
    caps = capabilities_for_platform(_apple_silicon_with_mlx())
    # The TUI checks capabilities.dedicated_vram before rendering VRAM widgets
    assert caps.dedicated_vram is False, (
        "dedicated_vram must be False on Apple Silicon so the TUI omits VRAM panels"
    )


# ── BackendCapabilities fields ─────────────────────────────────────────────────

def test_backend_capabilities_all_fields_present():
    caps = BackendCapabilities(
        backend_name="PyTorch/CUDA",
        inference_runtime="PyTorch",
        device_label="NVIDIA RTX 4090",
        token_stream=True,
        logits=True,
        kv_cache=True,
        layer_stats=True,
        attention_stats=True,
        gpu_utilization=True,
        dedicated_vram=True,
        unified_memory=False,
        memory_breakdown=True,
        memory_architecture="Discrete CPU/GPU",
    )
    assert caps.backend_name == "PyTorch/CUDA"
    assert caps.inference_runtime == "PyTorch"
    assert caps.device_label == "NVIDIA RTX 4090"
    assert caps.memory_architecture == "Discrete CPU/GPU"

"""Tests for platform detection.

All tests run on Windows (and any other non-Apple platform) by mocking
the platform-specific functions. No Apple Silicon or MLX hardware required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmvis.platform.detect import PlatformInfo, detect_platform


# ── Apple Silicon detection ───────────────────────────────────────────────────

def test_apple_silicon_detected():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("llmvis.platform.detect._check_mlx", return_value=True),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=True),
        patch("llmvis.platform.detect._detect_chip_model_macos", return_value="Apple M3 Pro"),
    ):
        info = detect_platform()

    assert info.is_apple_silicon is True
    assert info.os == "darwin"
    assert info.arch == "arm64"
    assert info.mlx_available is True
    assert info.mps_available is True
    assert "M3" in info.chip_model


def test_unsupported_intel_mac():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="x86_64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    assert info.is_apple_silicon is False
    assert info.os == "darwin"
    assert info.arch == "x86_64"


def test_linux_nvidia_detected():
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(True, "NVIDIA GeForce RTX 4090")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    assert info.is_apple_silicon is False
    assert info.os == "linux"
    assert info.cuda_available is True
    assert "NVIDIA" in info.cuda_device_name


def test_windows_cuda_detected():
    with (
        patch("platform.system", return_value="Windows"),
        patch("platform.machine", return_value="AMD64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(True, "NVIDIA GeForce RTX 4090 Laptop GPU")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    assert info.os == "windows"
    assert info.is_apple_silicon is False
    assert info.cuda_available is True


def test_cpu_only_linux():
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="x86_64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    assert info.is_apple_silicon is False
    assert info.cuda_available is False
    assert info.mlx_available is False


def test_mlx_available_true():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("llmvis.platform.detect._check_mlx", return_value=True),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=True),
        patch("llmvis.platform.detect._detect_chip_model_macos", return_value="Apple M2"),
    ):
        info = detect_platform()

    assert info.mlx_available is True


def test_mlx_unavailable():
    with (
        patch("platform.system", return_value="Darwin"),
        patch("platform.machine", return_value="arm64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
        patch("llmvis.platform.detect._detect_chip_model_macos", return_value="Apple M1"),
    ):
        info = detect_platform()

    assert info.mlx_available is False


def test_platform_info_fields_complete():
    """PlatformInfo dataclass has all required fields."""
    info = PlatformInfo(
        os="linux",
        arch="x86_64",
        is_apple_silicon=False,
        chip_model="unknown",
        cuda_available=True,
        cuda_device_name="NVIDIA A100",
        mlx_available=False,
        mps_available=False,
    )
    assert info.os == "linux"
    assert info.cuda_device_name == "NVIDIA A100"


def test_aarch64_mapped_to_arm64():
    """Linux aarch64 (e.g. Raspberry Pi) is normalised to arm64."""
    with (
        patch("platform.system", return_value="Linux"),
        patch("platform.machine", return_value="aarch64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    # aarch64 Linux is arm64 arch but NOT Apple Silicon
    assert info.arch == "arm64"
    assert info.is_apple_silicon is False


def test_amd64_mapped_to_x86_64():
    with (
        patch("platform.system", return_value="Windows"),
        patch("platform.machine", return_value="AMD64"),
        patch("llmvis.platform.detect._check_mlx", return_value=False),
        patch("llmvis.platform.detect._check_cuda", return_value=(False, "")),
        patch("llmvis.platform.detect._check_mps", return_value=False),
    ):
        info = detect_platform()

    assert info.arch == "x86_64"

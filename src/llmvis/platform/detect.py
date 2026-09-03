"""Platform detection for LLMVis.

Determines the host OS, CPU architecture, and available compute backends
without importing heavy ML libraries at module load time.
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Snapshot of the host platform and available compute backends."""

    os: str                  # "darwin", "linux", "windows"
    arch: str                # "arm64", "x86_64", "amd64"
    is_apple_silicon: bool   # True only on macOS arm64
    chip_model: str          # "Apple M3 Pro", "unknown", or CUDA device name
    cuda_available: bool
    cuda_device_name: str    # e.g. "NVIDIA GeForce RTX 4090"
    mlx_available: bool      # True if `import mlx.core` succeeds
    mps_available: bool      # True if PyTorch MPS is available


def _detect_chip_model_macos() -> str:
    """Attempt to read the chip model string via sysctl on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        chip = result.stdout.strip()
        if chip:
            return chip
    except Exception:
        pass

    # Fallback: try system_profiler for Apple Silicon brand
    try:
        result = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "Chip" in line or "Processor Name" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    return parts[1].strip()
    except Exception:
        pass

    return "Apple Silicon"


def _check_mlx() -> bool:
    """Return True if MLX core is importable (Apple Silicon only in practice)."""
    try:
        import importlib
        importlib.import_module("mlx.core")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _check_cuda() -> tuple[bool, str]:
    """Return (cuda_available, device_name) using torch if available."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return True, name
        return False, ""
    except (ImportError, Exception):
        return False, ""


def _check_mps() -> bool:
    """Return True if PyTorch MPS backend is available."""
    try:
        import torch
        return bool(torch.backends.mps.is_available())
    except (ImportError, AttributeError):
        return False


def detect_platform() -> PlatformInfo:
    """Detect the current platform and available compute backends."""
    system = platform.system().lower()
    # Normalize: "windows", "darwin", "linux"
    os_name = {
        "windows": "windows",
        "darwin": "darwin",
        "linux": "linux",
    }.get(system, system)

    machine = platform.machine().lower()
    # Normalize common architecture strings
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine

    is_apple_silicon = (os_name == "darwin" and arch == "arm64")

    chip_model = "unknown"
    if is_apple_silicon:
        chip_model = _detect_chip_model_macos()
    elif os_name == "darwin":
        chip_model = "Intel Mac"

    cuda_available, cuda_device_name = _check_cuda()
    if cuda_available and not chip_model or chip_model == "unknown":
        chip_model = cuda_device_name

    mlx_available = _check_mlx()
    mps_available = _check_mps()

    return PlatformInfo(
        os=os_name,
        arch=arch,
        is_apple_silicon=is_apple_silicon,
        chip_model=chip_model,
        cuda_available=cuda_available,
        cuda_device_name=cuda_device_name,
        mlx_available=mlx_available,
        mps_available=mps_available,
    )

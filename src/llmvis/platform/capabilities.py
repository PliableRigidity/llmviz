"""Backend capability model for LLMVis.

Maps a detected platform to a normalized capability description so TUI panels
can render accurately without platform-specific conditionals scattered everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

from llmvis.platform.detect import PlatformInfo


@dataclass
class BackendCapabilities:
    """What a compute backend can expose to LLMVis."""

    backend_name: str        # "PyTorch/CUDA", "MLX/Metal", "PyTorch/MPS", "PyTorch/CPU"
    inference_runtime: str   # "PyTorch", "MLX"
    device_label: str        # Human-readable: "NVIDIA RTX 4090", "Apple M3 Pro", "CPU"

    # Per-token telemetry
    token_stream: bool
    logits: bool
    kv_cache: bool
    layer_stats: bool
    attention_stats: bool

    # Hardware telemetry
    gpu_utilization: bool
    dedicated_vram: bool     # False on Apple Silicon (unified memory)
    unified_memory: bool     # True on Apple Silicon
    memory_breakdown: bool   # Can we report model/kv/other breakdown?

    # Architecture specifics
    memory_architecture: str  # "Discrete CPU/GPU", "Unified Memory", "CPU-only"


def capabilities_for_platform(p: PlatformInfo) -> BackendCapabilities:
    """Return the best-available backend capabilities for the detected platform."""

    if p.is_apple_silicon and p.mlx_available:
        return BackendCapabilities(
            backend_name="MLX/Metal",
            inference_runtime="MLX",
            device_label=p.chip_model or "Apple Silicon",
            token_stream=True,
            logits=True,
            kv_cache=True,
            layer_stats=False,     # MLX has no PyTorch-style forward hooks
            attention_stats=False,
            gpu_utilization=False,  # powermetrics requires sudo
            dedicated_vram=False,
            unified_memory=True,
            memory_breakdown=True,  # MLX can report active Metal allocation
            memory_architecture="Unified Memory",
        )

    if p.is_apple_silicon and p.mps_available:
        return BackendCapabilities(
            backend_name="PyTorch/MPS",
            inference_runtime="PyTorch",
            device_label=p.chip_model or "Apple Silicon",
            token_stream=True,
            logits=True,
            kv_cache=True,
            layer_stats=True,
            attention_stats=True,
            gpu_utilization=False,
            dedicated_vram=False,
            unified_memory=True,
            memory_breakdown=False,
            memory_architecture="Unified Memory",
        )

    if p.is_apple_silicon:
        # Apple Silicon but neither MLX nor MPS — CPU fallback
        return BackendCapabilities(
            backend_name="PyTorch/CPU",
            inference_runtime="PyTorch",
            device_label=p.chip_model or "Apple Silicon",
            token_stream=True,
            logits=True,
            kv_cache=True,
            layer_stats=True,
            attention_stats=True,
            gpu_utilization=False,
            dedicated_vram=False,
            unified_memory=True,
            memory_breakdown=False,
            memory_architecture="Unified Memory",
        )

    if p.cuda_available:
        return BackendCapabilities(
            backend_name="PyTorch/CUDA",
            inference_runtime="PyTorch",
            device_label=p.cuda_device_name or "NVIDIA GPU",
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

    # CPU-only fallback
    return BackendCapabilities(
        backend_name="PyTorch/CPU",
        inference_runtime="PyTorch",
        device_label="CPU",
        token_stream=True,
        logits=True,
        kv_cache=True,
        layer_stats=True,
        attention_stats=True,
        gpu_utilization=False,
        dedicated_vram=False,
        unified_memory=False,
        memory_breakdown=False,
        memory_architecture="CPU-only",
    )

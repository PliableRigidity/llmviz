"""Platform detection and capability model for LLMVis."""

from llmvis.platform.detect import PlatformInfo, detect_platform
from llmvis.platform.capabilities import BackendCapabilities, capabilities_for_platform

__all__ = [
    "PlatformInfo",
    "detect_platform",
    "BackendCapabilities",
    "capabilities_for_platform",
]

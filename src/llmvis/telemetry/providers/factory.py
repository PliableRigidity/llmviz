"""Factory: pick the best telemetry provider for the detected platform."""

from __future__ import annotations

from llmvis.platform.detect import PlatformInfo
from llmvis.telemetry.providers.base import BaseTelemetryProvider


def get_telemetry_provider(p: PlatformInfo) -> BaseTelemetryProvider:
    """Return the most capable telemetry provider for the given platform."""
    if p.is_apple_silicon:
        from llmvis.telemetry.providers.apple import AppleTelemetryProvider
        return AppleTelemetryProvider(chip_name=p.chip_model)

    if p.cuda_available:
        from llmvis.telemetry.providers.nvidia import NvidiaTelemetryProvider
        return NvidiaTelemetryProvider(device_name=p.cuda_device_name)

    from llmvis.telemetry.providers.system import SystemTelemetryProvider
    return SystemTelemetryProvider()

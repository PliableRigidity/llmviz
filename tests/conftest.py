"""Shared test fixtures."""

from __future__ import annotations

import socket

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def unused_tcp_port() -> int:
    """Return a free TCP port on localhost."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


FAKE_VERSION_RESPONSE = {"version": "0.33.2"}

FAKE_PS_RESPONSE_EMPTY = {"models": []}

FAKE_PS_RESPONSE_ONE = {
    "models": [
        {
            "name": "qwen2.5:3b",
            "model": "qwen2.5:3b",
            "size": 1929912432,
            "digest": "abc123",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "3.1B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": "2026-09-03T17:00:00Z",
            "size_vram": 1800000000,
            "context_length": 4096,
        }
    ]
}

FAKE_PS_RESPONSE_TWO = {
    "models": [
        {
            "name": "qwen2.5:3b",
            "model": "qwen2.5:3b",
            "size": 1929912432,
            "digest": "abc123",
            "details": {
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "3.1B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": "2026-09-03T17:00:00Z",
            "size_vram": 1800000000,
            "context_length": 4096,
        },
        {
            "name": "gemma3:4b",
            "model": "gemma3:4b",
            "size": 3338801804,
            "digest": "def456",
            "details": {
                "format": "gguf",
                "family": "gemma3",
                "families": ["gemma3"],
                "parameter_size": "4.3B",
                "quantization_level": "Q4_K_M",
            },
            "expires_at": "2026-09-03T17:00:00Z",
            "size_vram": 3200000000,
            "context_length": 8192,
        },
    ]
}

FAKE_TAGS_RESPONSE = {
    "models": [
        {
            "name": "qwen2.5:3b",
            "model": "qwen2.5:3b",
            "size": 1929912432,
            "digest": "abc123",
            "details": {
                "format": "gguf",
                "family": "qwen2",
                "families": ["qwen2"],
                "parameter_size": "3.1B",
                "quantization_level": "Q4_K_M",
                "context_length": 32768,
                "embedding_length": 2048,
            },
            "capabilities": ["completion", "tools"],
        }
    ]
}

FAKE_MALFORMED_RESPONSE = {"unexpected_key": [{"bad": "data"}]}

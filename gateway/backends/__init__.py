"""Provider adapters. `build_backends` turns config entries into live objects."""

from __future__ import annotations

from gateway.backends.anthropic import AnthropicBackend
from gateway.backends.base import Backend
from gateway.backends.mock import MockBackend
from gateway.backends.openai_compat import OpenAICompatBackend
from gateway.config import BackendConfig

__all__ = ["Backend", "AnthropicBackend", "MockBackend", "OpenAICompatBackend", "build_backends"]


def build_backends(configs: list[BackendConfig]) -> dict[str, Backend]:
    factories = {"anthropic": AnthropicBackend, "openai": OpenAICompatBackend,
                 "vllm": OpenAICompatBackend, "mock": MockBackend}
    return {c.name: factories[c.type](c) for c in configs}

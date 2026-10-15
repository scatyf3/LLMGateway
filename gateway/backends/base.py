"""The one interface every provider adapter implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from gateway.config import BackendConfig
from gateway.errors import BackendError


class Backend(ABC):
    """Takes and returns OpenAI-format dicts, whatever the provider speaks.

    Errors must be raised as `BackendError` so the router can decide on
    fallback without knowing provider-specific exception types.
    """

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.name = cfg.name

    @property
    def available(self) -> bool:
        """False when unconfigured (e.g. no API key). The router skips it silently."""
        return self.cfg.enabled

    @abstractmethod
    async def chat(self, model: str, body: dict) -> dict:
        """Non-streaming chat completion -> `chat.completion` dict."""

    @abstractmethod
    def chat_stream(self, model: str, body: dict) -> AsyncIterator[dict]:
        """Streaming chat completion -> async iterator of `chat.completion.chunk` dicts.

        Must raise BackendError before yielding anything if the request fails
        up front: that is what lets the router fall back before the first byte.
        """

    async def embed(self, model: str, body: dict) -> dict:
        raise BackendError(self.name, 400, f"{self.cfg.type} backend does not support embeddings",
                           retryable=False)

    async def aclose(self) -> None:
        return None

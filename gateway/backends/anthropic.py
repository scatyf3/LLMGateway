"""Anthropic backend: OpenAI-shaped in and out, official SDK in the middle."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import anthropic

from gateway.backends.anthropic_translate import (
    StreamTranslator, TranslationError, from_anthropic_message, to_anthropic_request,
)
from gateway.backends.base import Backend
from gateway.config import BackendConfig
from gateway.errors import BackendError, is_retryable_status


def _as_dict(obj: Any) -> dict:
    return obj if isinstance(obj, dict) else obj.model_dump()


class AnthropicBackend(Backend):
    def __init__(self, cfg: BackendConfig, client: Any = None):
        super().__init__(cfg)
        api_key = os.environ.get(cfg.api_key_env or "ANTHROPIC_API_KEY", "")
        # max_retries=0: the gateway's router owns retries/fallback. SDK retries
        # would silently multiply tail latency before we ever get to fall back.
        self.client = client or (
            anthropic.AsyncAnthropic(api_key=api_key, base_url=cfg.base_url,
                                     timeout=cfg.timeout_s, max_retries=0)
            if api_key else None)

    @property
    def available(self) -> bool:
        return self.cfg.enabled and self.client is not None

    def _request(self, body: dict, model: str) -> dict:
        try:
            return to_anthropic_request(body, model)
        except TranslationError as e:
            raise BackendError(self.name, 400, str(e), retryable=False) from None

    def _map_error(self, e: anthropic.APIError) -> BackendError:
        if isinstance(e, anthropic.APITimeoutError):
            return BackendError.timeout(self.name)
        if isinstance(e, anthropic.APIConnectionError):
            return BackendError.connection(self.name, str(e))
        status = getattr(e, "status_code", 500)
        return BackendError(self.name, status, getattr(e, "message", str(e)),
                            retryable=is_retryable_status(status))

    async def chat(self, model: str, body: dict) -> dict:
        req = self._request(body, model)
        try:
            msg = await self.client.messages.create(**req)
        except anthropic.APIError as e:
            raise self._map_error(e) from None
        return from_anthropic_message(_as_dict(msg), model)

    async def chat_stream(self, model: str, body: dict) -> AsyncIterator[dict]:
        req = self._request(body, model)
        translator = StreamTranslator(model)
        try:
            async with self.client.messages.stream(**req) as stream:
                async for event in stream:
                    for chunk in translator.feed(_as_dict(event)):
                        yield chunk
        except anthropic.APIError as e:
            raise self._map_error(e) from None

    async def aclose(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            await self.client.close()

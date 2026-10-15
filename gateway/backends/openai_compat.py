"""OpenAI and vLLM backends: both speak the OpenAI API, so this is a pass-through.

The body is forwarded as-is (only `model` is rewritten), so OpenAI features we
don't explicitly handle - tools, response_format, logprobs, seed - still work.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from gateway.backends.base import Backend
from gateway.config import BackendConfig
from gateway.errors import BackendError, is_retryable_status
from gateway.openai_format import parse_sse


class OpenAICompatBackend(Backend):
    def __init__(self, cfg: BackendConfig, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__(cfg)
        self.api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # One pooled client per backend: connection reuse matters for p99.
        self.client = httpx.AsyncClient(
            base_url=(cfg.base_url or "https://api.openai.com/v1").rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s, connect=min(5.0, cfg.timeout_s)),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            transport=transport,
        )

    @property
    def available(self) -> bool:
        # vLLM needs no key; OpenAI does.
        return self.cfg.enabled and (self.cfg.type == "vllm" or bool(self.api_key))

    async def _post(self, path: str, body: dict) -> dict:
        try:
            resp = await self.client.post(path, json=body)
        except httpx.TimeoutException:
            raise BackendError.timeout(self.name) from None
        except httpx.TransportError as e:
            raise BackendError.connection(self.name, repr(e)) from None
        self._raise_for_status(resp.status_code, resp.text)
        return resp.json()

    def _raise_for_status(self, status: int, text: str) -> None:
        if status >= 400:
            raise BackendError(self.name, status, text[:500], retryable=is_retryable_status(status))

    async def chat(self, model: str, body: dict) -> dict:
        return await self._post("/chat/completions", {**body, "model": model, "stream": False})

    async def chat_stream(self, model: str, body: dict) -> AsyncIterator[dict]:
        # Always ask for usage so billing uses real token counts; the pipeline
        # drops the usage chunk again if the client didn't ask for it.
        payload = {**body, "model": model, "stream": True,
                   "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    self._raise_for_status(resp.status_code, (await resp.aread()).decode(errors="replace"))
                async for chunk in parse_sse(resp.aiter_lines()):
                    yield chunk
        except httpx.TimeoutException:
            raise BackendError.timeout(self.name) from None
        except httpx.TransportError as e:
            raise BackendError.connection(self.name, repr(e)) from None

    async def embed(self, model: str, body: dict) -> dict:
        return await self._post("/embeddings", {**body, "model": model})

    async def aclose(self) -> None:
        await self.client.aclose()

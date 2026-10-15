"""Shared fixtures: an in-memory Redis, fast mock backends, and an ASGI client.
Nothing here touches the network or Docker."""

from __future__ import annotations

import hashlib
from typing import AsyncIterator

import fakeredis
import httpx
import pytest

from gateway.app import create_app
from gateway.backends import Backend, build_backends
from gateway.config import BackendConfig, GatewayConfig
from gateway.errors import BackendError

TEST_KEY = "sk-test"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def make_config(**overrides) -> GatewayConfig:
    fast_mock = {"ttft_ms": 0, "tokens_per_s": 100000, "output_tokens": 12}
    raw = {
        "backends": [
            {"name": "m1", "type": "mock", "mock": fast_mock},
            {"name": "m2", "type": "mock", "mock": fast_mock},
        ],
        "aliases": {
            "fast": [{"backend": "m1", "model": "mock-small"}, {"backend": "m2", "model": "mock-small"}],
            "smart": [{"backend": "m1", "model": "mock-large"}],
            "local": [{"backend": "m2", "model": "mock-local"}],
            "embed": [{"backend": "m1", "model": "mock-embed"}],
        },
        "default_alias": "fast",
        "rules": [{"name": "long-to-local", "min_prompt_tokens": 1000, "aliases": ["fast"], "route_to": "local"}],
        "tiers": {"pro": {"allowed_aliases": ["*"], "default_alias": "fast"},
                  "free": {"allowed_aliases": ["fast", "embed"], "default_alias": "fast"}},
        "prices": {"mock-small": {"input": 1.0, "output": 2.0}, "mock-large": {"input": 10.0, "output": 20.0}},
        "api_keys": [
            {"id": "demo", "secret_sha256": _sha(TEST_KEY), "tier": "pro", "rpm": 1000, "tpm": 10**7,
             "monthly_budget_usd": 100, "admin": True},
            {"id": "limited", "secret_sha256": _sha("sk-limited"), "tier": "pro", "rpm": 2, "tpm": 10**7,
             "monthly_budget_usd": 100},
            {"id": "broke", "secret_sha256": _sha("sk-broke"), "tier": "pro", "rpm": 1000, "tpm": 10**7,
             "monthly_budget_usd": 0.00001},
            {"id": "free", "secret_sha256": _sha("sk-free"), "tier": "free", "rpm": 1000, "tpm": 10**7,
             "monthly_budget_usd": 100},
        ],
        "cache": {"semantic": {"threshold": 0.95, "embedding_alias": "embed"}},
    }
    raw.update(overrides)
    return GatewayConfig.model_validate(raw)


@pytest.fixture
def cfg() -> GatewayConfig:
    return make_config()


@pytest.fixture
def redis():
    return fakeredis.FakeAsyncRedis()


@pytest.fixture
def backends(cfg):
    return build_backends(cfg.backends)


@pytest.fixture
def app(cfg, redis, backends):
    return create_app(cfg, redis, backends)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw",
                                 headers={"Authorization": f"Bearer {TEST_KEY}"}) as c:
        yield c


def chat_body(text: str = "hello", **kw) -> dict:
    return {"model": "fast", "temperature": 0, "messages": [{"role": "user", "content": text}], **kw}


class ScriptedBackend(Backend):
    """A backend whose behavior is set per test: fail, hang, or stream-then-fail."""

    def __init__(self, name: str, *, fail_status: int | None = None, hang_s: float = 0.0,
                 fail_mid_stream: bool = False, timeout_s: float = 1.0):
        super().__init__(BackendConfig(name=name, type="mock", timeout_s=timeout_s))
        self.fail_status = fail_status
        self.hang_s = hang_s
        self.fail_mid_stream = fail_mid_stream
        self.calls = 0

    async def _gate(self) -> None:
        import asyncio
        self.calls += 1
        if self.hang_s:
            await asyncio.sleep(self.hang_s)
        if self.fail_status:
            raise BackendError(self.name, self.fail_status, "scripted failure",
                               retryable=self.fail_status >= 500 or self.fail_status == 429)

    async def chat(self, model: str, body: dict) -> dict:
        from gateway.openai_format import make_completion, make_usage
        await self._gate()
        return make_completion(model, f"from {self.name}", usage=make_usage(10, 5))

    async def chat_stream(self, model: str, body: dict) -> AsyncIterator[dict]:
        from gateway.openai_format import make_chunk, make_usage
        await self._gate()
        yield make_chunk("x", model, {"role": "assistant", "content": ""})
        yield make_chunk("x", model, {"content": f"from {self.name}"})
        if self.fail_mid_stream:
            raise BackendError(self.name, 500, "died mid-stream", retryable=True)
        yield make_chunk("x", model, finish_reason="stop")
        yield make_chunk("x", model, usage=make_usage(10, 5))

"""Routing: alias + rules + tier -> ordered chain; then try the chain with fallback.

Fallback policy in one place:
  * retryable failure (5xx, 429, timeout, connection) -> try the next backend
  * non-retryable (most 4xx) -> return it; another provider won't fix a bad request
  * streaming: fall back only until the first chunk arrives. After the client
    has received bytes, switching backends would splice two different answers
    into one response, so a mid-stream failure is surfaced as an SSE error.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, TypeVar

from gateway import metrics
from gateway.backends.base import Backend
from gateway.config import GatewayConfig, Target, Tier
from gateway.errors import BackendError, GatewayError

log = logging.getLogger("gateway.router")
T = TypeVar("T")


@dataclass
class Route:
    requested: str
    alias: str
    chain: list[Target]
    reason: str  # why this alias: "alias", "rule:<name>", "tier"


@dataclass
class Served:
    target: Target
    fallbacks: int
    backend_seconds: float  # time spent waiting on backends, incl. failed attempts


class Router:
    def __init__(self, cfg: GatewayConfig, backends: dict[str, Backend]):
        self.cfg = cfg
        self.backends = backends

    def resolve(self, requested: str | None, prompt_tokens: int, tier: Tier) -> Route:
        alias = requested or self.cfg.default_alias
        if alias not in self.cfg.aliases:
            raise GatewayError(404, f"unknown model {alias!r}; GET /v1/models lists aliases",
                               code="model_not_found")
        original, reasons = alias, []
        for rule in self.cfg.rules:
            if prompt_tokens >= rule.min_prompt_tokens and (rule.aliases is None or alias in rule.aliases):
                alias = rule.route_to
                reasons.append(f"rule:{rule.name}")
                break
        if not tier.allows(alias):
            alias = tier.default_alias
            reasons.append("tier")
        return Route(original, alias, self.cfg.aliases[alias], ",".join(reasons) or "alias")

    async def _run(self, route: Route, call: Callable[[Backend, Target], Awaitable[T]]) -> tuple[T, Served]:
        candidates = [(self.backends[t.backend], t) for t in route.chain if self.backends[t.backend].available]
        if not candidates:
            raise GatewayError(503, f"no backend configured for {route.alias!r}", "server_error", "no_backend")
        loop = asyncio.get_running_loop()
        started = loop.time()
        last: BackendError | None = None
        for i, (backend, target) in enumerate(candidates):
            try:
                result = await asyncio.wait_for(call(backend, target), backend.cfg.timeout_s)
            except asyncio.TimeoutError:
                err = BackendError.timeout(backend.name)
            except BackendError as e:
                err = e
            else:
                if i:
                    metrics.FALLBACK_REQUESTS.labels("success").inc()
                return result, Served(target, i, loop.time() - started)

            last = err
            if not err.retryable:
                if i:
                    metrics.FALLBACK_REQUESTS.labels("failed").inc()
                raise GatewayError(err.status, f"{err.backend}: {err.message}", code="upstream_error")
            if i + 1 < len(candidates):
                metrics.FALLBACKS.labels(route.alias, backend.name, err.reason).inc()
                log.warning("fallback alias=%s from=%s reason=%s detail=%s",
                            route.alias, backend.name, err.reason, err.message[:200])
        if len(candidates) > 1:
            metrics.FALLBACK_REQUESTS.labels("failed").inc()
        raise GatewayError(502, f"all backends failed for {route.alias!r}; last: {last}",
                           "server_error", "all_backends_failed")

    async def chat(self, route: Route, body: dict) -> tuple[dict, Served]:
        return await self._run(route, lambda b, t: b.chat(t.model, body))

    async def embed(self, route: Route, body: dict) -> tuple[dict, Served]:
        return await self._run(route, lambda b, t: b.embed(t.model, body))

    async def stream(self, route: Route, body: dict) -> tuple[AsyncIterator[dict], Served]:
        """Returns a stream whose first chunk has already arrived (so fallback is over)."""
        return await self._run(route, lambda b, t: _open_stream(b, t, body))


async def _open_stream(backend: Backend, target: Target, body: dict) -> AsyncIterator[dict]:
    agen = backend.chat_stream(target.model, body)
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        raise BackendError(backend.name, 502, "empty stream", retryable=True) from None
    except BaseException:  # incl. CancelledError from the router's timeout
        await agen.aclose()
        raise
    return _prepend(first, agen)


async def _prepend(first: dict, agen: AsyncIterator[dict]) -> AsyncIterator[dict]:
    try:
        yield first
        async for chunk in agen:
            yield chunk
    finally:
        await agen.aclose()

"""The per-request pipeline, top to bottom:

    rate limit -> budget -> route -> exact cache -> semantic cache
      -> router (with fallback) -> [response sent] -> accounting -> cache write

(Auth happens in the HTTP layer just before.) Accounting and cache writes run
as a background task after the response is flushed, so they never add latency. Reading `chat_completions`
should be enough to explain the whole gateway; helpers below it hold the details.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from gateway import metrics
from gateway.accounting import Accountant, cost_usd
from gateway.cache import ExactCache, SemanticCache, is_cacheable, namespace, request_fingerprint
from gateway.config import ApiKey, GatewayConfig
from gateway.errors import BackendError
from gateway.limits import Limiter
from gateway.openai_format import (
    ChunkAccumulator, chunks_from_completion, estimate_prompt_tokens, estimate_tokens,
    last_user_text, new_id, sse,
)
from gateway.router import Route, Router, Served

log = logging.getLogger("gateway.pipeline")

_CACHEABLE_FINISH = {"stop", "length", "tool_calls"}


@dataclass
class Gateway:
    """Everything a request needs, built once at startup."""
    cfg: GatewayConfig
    router: Router
    limiter: Limiter
    accountant: Accountant
    exact: ExactCache | None
    semantic: SemanticCache | None


@dataclass
class RequestCtx:
    key: ApiKey
    endpoint: str
    t0: float = field(default_factory=time.perf_counter)
    route: Route | None = None
    cache: str = "miss"
    backend: str = "cache"
    model: str = "-"
    cost: float = 0.0
    backend_s: float = 0.0
    embed_s: float = 0.0  # semantic-cache embedding call: a model call, so not "overhead"
    fallbacks: int = 0
    overhead_frozen: float | None = None  # fixed when headers are built, so metric == header

    def served(self, s: Served) -> None:
        self.backend, self.model = s.target.backend, s.target.model
        self.backend_s, self.fallbacks = s.backend_seconds, s.fallbacks

    def overhead_s(self) -> float:
        if self.overhead_frozen is not None:
            return self.overhead_frozen
        return max(0.0, time.perf_counter() - self.t0 - self.backend_s - self.embed_s)

    def headers(self, with_cost: bool = True) -> dict[str, str]:
        """Built right before the response starts; freezes the overhead measurement.
        Streaming misses omit the cost (unknown until the last chunk; see /admin/usage)."""
        self.overhead_frozen = self.overhead_s()
        h = {"x-gateway-backend": self.backend, "x-gateway-model": self.model,
             "x-gateway-cache": self.cache, "x-gateway-overhead-ms": f"{self.overhead_frozen * 1000:.2f}"}
        if with_cost:
            h["x-gateway-cost-usd"] = f"{self.cost:.8f}"
        if self.route:
            h["x-gateway-alias"] = self.route.alias
            h["x-gateway-route"] = self.route.reason
        if self.embed_s:
            h["x-gateway-semantic-ms"] = f"{self.embed_s * 1000:.2f}"
        if self.fallbacks:
            h["x-gateway-fallbacks"] = str(self.fallbacks)
        return h

    def observe(self, status: str) -> None:
        metrics.REQUESTS.labels(self.endpoint, self.backend, self.model, status, self.cache).inc()
        metrics.LATENCY.labels(self.endpoint, self.backend, self.cache).observe(time.perf_counter() - self.t0)
        metrics.OVERHEAD.labels(self.endpoint, self.cache).observe(self.overhead_s())


@dataclass
class CacheSlot:
    """Where a miss should be written back once the response is complete."""
    fingerprint: str | None = None
    sem_ns: str | None = None
    sem_text: str = ""
    sem_vec: np.ndarray | None = None


# --------------------------------------------------------------------------- chat


async def chat_completions(gw: Gateway, key: ApiKey, body: dict, cache_header: str | None):
    ctx = RequestCtx(key, "chat")
    prompt_tokens = estimate_prompt_tokens(body)

    await gw.limiter.check_rate(key, prompt_tokens)
    await gw.limiter.check_budget(key)
    ctx.route = gw.router.resolve(body.get("model"), prompt_tokens, gw.cfg.tiers[key.tier])

    hit, slot = await _cache_lookup(gw, ctx, body, cache_header)
    if hit is not None:
        return await _serve_hit(gw, ctx, body, hit)

    if body.get("stream"):
        return await _stream_miss(gw, ctx, body, slot, prompt_tokens)

    response, served = await gw.router.chat(ctx.route, body)
    ctx.served(served)
    ctx.cost = cost_usd(gw.cfg.prices, ctx.model, response.get("usage"))

    async def after_response() -> None:
        # Billing and cache writes run after the body is flushed: off the latency path.
        await _finish(gw, ctx, response.get("usage"), "200")
        await _cache_write(gw, slot, response, ctx.cost)
        ctx.observe("200")

    return JSONResponse(response, headers=ctx.headers(), background=BackgroundTask(after_response))


async def _cache_lookup(gw: Gateway, ctx: RequestCtx, body: dict,
                        cache_header: str | None) -> tuple[dict | None, CacheSlot]:
    slot = CacheSlot()
    if not is_cacheable(body, gw.cfg.cache, cache_header):
        return None, slot
    alias = ctx.route.alias
    try:
        if gw.exact:
            slot.fingerprint = request_fingerprint(alias, body)
            if (entry := await gw.exact.get(slot.fingerprint)) is not None:
                metrics.CACHE_LOOKUPS.labels("exact", "hit").inc()
                ctx.cache = "exact"
                return entry, slot
            metrics.CACHE_LOOKUPS.labels("exact", "miss").inc()
        if gw.semantic and (ns := namespace(alias, body)):
            slot.sem_ns, slot.sem_text = ns, last_user_text(body["messages"])
            t = time.perf_counter()
            entry, slot.sem_vec = await gw.semantic.lookup(ns, slot.sem_text)
            ctx.embed_s = time.perf_counter() - t
            metrics.SEMANTIC_LOOKUP.observe(ctx.embed_s)
            metrics.CACHE_LOOKUPS.labels("semantic", "hit" if entry else "miss").inc()
            if entry is not None:
                ctx.cache = "semantic"
                return entry, slot
    except Exception:  # a cache must never fail a request
        log.exception("cache lookup failed; treating as miss")
    return None, slot


async def _cache_write(gw: Gateway, slot: CacheSlot, response: dict, cost: float) -> None:
    if response["choices"][0].get("finish_reason") not in _CACHEABLE_FINISH:
        return
    try:
        if gw.exact and slot.fingerprint:
            await gw.exact.put(slot.fingerprint, response, cost)
        if gw.semantic and slot.sem_ns and slot.sem_vec is not None:
            await gw.semantic.store(slot.sem_ns, slot.sem_text, slot.sem_vec, response, cost)
    except Exception:
        log.exception("cache write failed")


async def _serve_hit(gw: Gateway, ctx: RequestCtx, body: dict, entry: dict):
    response = {**entry["response"], "id": new_id(), "created": int(time.time())}
    ctx.model = response.get("model", "-")
    headers = ctx.headers()

    async def after_response() -> None:
        try:
            await gw.accountant.record(ctx.key.id, ctx.model, response.get("usage"), ctx.cache,
                                       saved_usd=entry.get("cost_usd", 0.0))
        except Exception:
            log.exception("accounting failed for cache hit")
        ctx.observe("200")

    if not body.get("stream"):
        return JSONResponse(response, headers=headers, background=BackgroundTask(after_response))
    metrics.TTFT.labels("cache", ctx.cache).observe(time.perf_counter() - ctx.t0)
    include_usage = _wants_usage(body)

    async def replay() -> AsyncIterator[bytes]:
        for chunk in chunks_from_completion(response):
            if chunk.get("usage") is not None and not include_usage:
                continue
            yield sse(chunk)
        yield sse("[DONE]")

    return StreamingResponse(replay(), media_type="text/event-stream", headers=headers,
                             background=BackgroundTask(after_response))


async def _stream_miss(gw: Gateway, ctx: RequestCtx, body: dict, slot: CacheSlot, prompt_tokens: int):
    # Fallback happens inside router.stream, before any byte reaches the client.
    stream, served = await gw.router.stream(ctx.route, body)
    ctx.served(served)
    headers = ctx.headers(with_cost=False)  # overhead here = pre-first-byte gateway time
    include_usage = _wants_usage(body)

    async def relay() -> AsyncIterator[bytes]:
        acc = ChunkAccumulator()
        status = "200"
        try:
            async for chunk in stream:
                had_content = acc.saw_content
                acc.add(chunk)
                if acc.saw_content and not had_content:
                    metrics.TTFT.labels(ctx.backend, "miss").observe(time.perf_counter() - ctx.t0)
                if chunk.get("usage") and not chunk.get("choices") and not include_usage:
                    continue
                yield sse(chunk)
        except BackendError as e:
            # Too late to fall back: the client already has part of an answer.
            status = "502"
            yield sse({"error": {"message": f"upstream failed mid-stream: {e}", "type": "server_error",
                                 "code": "upstream_stream_error"}})
        finally:
            await stream.aclose()
            response = acc.to_completion(prompt_tokens)
            if status == "200" and acc.finish_reason is None:
                status = "499"  # client went away before the stream finished
            await _finish(gw, ctx, response["usage"], status)
            if status == "200":
                await _cache_write(gw, slot, response, ctx.cost)
            ctx.observe(status)
        yield sse("[DONE]")

    return StreamingResponse(relay(), media_type="text/event-stream", headers=headers)


async def _finish(gw: Gateway, ctx: RequestCtx, usage: dict | None, status: str) -> None:
    """Bill the key and pay back output tokens into the TPM bucket."""
    try:
        ctx.cost = await gw.accountant.record(ctx.key.id, ctx.model, usage, ctx.cache)
        await gw.limiter.debit_output_tokens(ctx.key, (usage or {}).get("completion_tokens", 0))
    except Exception:
        log.exception("accounting failed for key=%s status=%s", ctx.key.id, status)


def _wants_usage(body: dict) -> bool:
    return bool((body.get("stream_options") or {}).get("include_usage"))


# --------------------------------------------------------------------------- embeddings


async def embeddings(gw: Gateway, key: ApiKey, body: dict) -> JSONResponse:
    ctx = RequestCtx(key, "embeddings")
    inputs = body.get("input")
    texts = [inputs] if isinstance(inputs, str) else [t for t in (inputs or []) if isinstance(t, str)]
    await gw.limiter.check_rate(key, sum(estimate_tokens(t) for t in texts))
    await gw.limiter.check_budget(key)
    # prompt_tokens=0: length rules are about chat prompts, never reroute embeddings.
    ctx.route = gw.router.resolve(body.get("model"), 0, gw.cfg.tiers[key.tier])
    response, served = await gw.router.embed(ctx.route, body)
    ctx.served(served)
    usage = {"prompt_tokens": (response.get("usage") or {}).get("prompt_tokens", 0), "completion_tokens": 0}
    ctx.cost = cost_usd(gw.cfg.prices, ctx.model, usage)

    async def after_response() -> None:
        await _finish(gw, ctx, usage, "200")
        ctx.observe("200")

    return JSONResponse(response, headers=ctx.headers(), background=BackgroundTask(after_response))

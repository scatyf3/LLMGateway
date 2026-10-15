"""Semantic cache: reuse an answer when a new prompt *means* the same thing.

Design, chosen for explainability over scale:
  * Embed only the last user message. Everything else that shapes the answer -
    alias, system prompt, earlier turns, max_tokens/stop/response_format - goes
    into a *namespace* hash, so two
    requests with different system prompts or histories can never collide, no
    matter how similar their last messages are.
  * Per namespace, keep the N most recent (vector, response) pairs in Redis and
    do brute-force cosine similarity in numpy. With N=2000 x 1536 dims that's a
    ~12 MB matrix-vector product: sub-millisecond, and trivially correct. The
    scale-up path is a real ANN index (Redis Stack HNSW or pgvector).
  * Strict threshold (0.95 default) and a structured log line for every hit, so
    false hits can be audited and the threshold tuned on a labeled set.
  * Skipped when tools are present: "same question" doesn't imply "same tool
    calls needed", and replaying a tool call or a stale answer is worse than a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Awaitable, Callable

import numpy as np
from redis.asyncio import Redis

from gateway.cache.exact import ANSWER_FIELDS
from gateway.config import SemanticCacheConfig
from gateway.openai_format import message_text

log = logging.getLogger("gateway.semantic_cache")

Embedder = Callable[[str], Awaitable[np.ndarray]]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def namespace(alias: str, body: dict) -> str | None:
    """Namespace hash, or None if this request isn't eligible for semantic caching."""
    messages = body.get("messages") or []
    if body.get("tools") or not messages or messages[-1].get("role") != "user":
        return None
    context = [{"role": m.get("role"), "content": message_text(m.get("content")),
                "tool_calls": m.get("tool_calls"), "tool_call_id": m.get("tool_call_id")}
               for m in messages[:-1]]
    params = {f: body[f] for f in ANSWER_FIELDS if body.get(f) is not None}
    blob = json.dumps({"alias": alias, "context": context, "params": params}, sort_keys=True)
    return _sha(blob)


class SemanticCache:
    def __init__(self, redis: Redis, cfg: SemanticCacheConfig, embed: Embedder):
        self.redis = redis
        self.cfg = cfg
        self.embed = embed

    def _keys(self, ns: str) -> tuple[str, str, str]:
        return f"sem:{ns}:order", f"sem:{ns}:vec", f"sem:{ns}:resp"

    async def lookup(self, ns: str, text: str) -> tuple[dict | None, np.ndarray]:
        """Returns (cache entry or None, query vector). The vector is reused by `store`."""
        query = await self.embed(text)
        query = query / (np.linalg.norm(query) or 1.0)
        order, vecs, resps = self._keys(ns)
        ids = await self.redis.lrange(order, 0, -1)
        if not ids:
            return None, query
        raw = await self.redis.hmget(vecs, ids)
        width = query.astype(np.float32).nbytes
        pairs = [(i, v) for i, v in zip(ids, raw) if v is not None and len(v) == width]
        if not pairs:
            return None, query
        matrix = np.frombuffer(b"".join(v for _, v in pairs), dtype=np.float32).reshape(len(pairs), -1)
        sims = matrix @ query.astype(np.float32)
        best = int(np.argmax(sims))
        similarity = float(sims[best])
        if similarity < self.cfg.threshold:
            return None, query
        entry_raw = await self.redis.hget(resps, pairs[best][0])
        if entry_raw is None:
            return None, query
        entry = json.loads(entry_raw)
        log.info(json.dumps({"event": "semantic_cache_hit", "namespace": ns,
                             "similarity": round(similarity, 4), "threshold": self.cfg.threshold,
                             "query_sha": _sha(text), "matched_sha": entry.get("prompt_sha")}))
        return entry, query

    async def store(self, ns: str, text: str, query: np.ndarray, response: dict, cost_usd: float) -> None:
        order, vecs, resps = self._keys(ns)
        entry_id = _sha(text)
        entry = json.dumps({"response": response, "cost_usd": cost_usd, "prompt_sha": entry_id})
        pipe = self.redis.pipeline(transaction=True)
        pipe.hset(vecs, entry_id, query.astype(np.float32).tobytes())
        pipe.hset(resps, entry_id, entry)
        pipe.lrem(order, 0, entry_id)
        pipe.lpush(order, entry_id)
        for k in (order, vecs, resps):
            pipe.expire(k, self.cfg.ttl_s)
        await pipe.execute()
        # Cap per-namespace size: evict the oldest entries.
        excess = await self.redis.llen(order) - self.cfg.max_entries
        if excess > 0:
            evicted = [await self.redis.rpop(order) for _ in range(excess)]
            evicted = [e for e in evicted if e]
            if evicted:
                await self.redis.hdel(vecs, *evicted)
                await self.redis.hdel(resps, *evicted)

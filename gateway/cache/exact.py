"""Exact-match response cache.

Key = sha256 over a *normalized* request: only fields that change the answer
(messages, tools, sampling params...), JSON with sorted keys, string content
stripped of surrounding whitespace. Fields that don't change the answer
(`stream`, `stream_options`, `user`, `metadata`) are excluded, so a streamed
and a non-streamed request share one entry.

Requests with tools ARE cached: tool definitions and tool results are part of
the key, so a hit means byte-identical inputs. (The semantic cache, by
contrast, skips tool requests - see semantic.py.)
"""

from __future__ import annotations

import hashlib
import json

from redis.asyncio import Redis

from gateway.config import CacheConfig

ANSWER_FIELDS = ("tools", "tool_choice", "parallel_tool_calls", "temperature", "top_p",
                 "max_tokens", "max_completion_tokens", "stop", "response_format", "seed", "n",
                 "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs")
_MESSAGE_FIELDS = ("role", "content", "name", "tool_calls", "tool_call_id")


def _norm_message(m: dict) -> dict:
    out = {k: m[k] for k in _MESSAGE_FIELDS if m.get(k) is not None}
    if isinstance(out.get("content"), str):
        out["content"] = out["content"].strip()
    return out


def request_fingerprint(alias: str, body: dict) -> str:
    norm = {"alias": alias, "messages": [_norm_message(m) for m in body.get("messages", [])]}
    norm.update({f: body[f] for f in ANSWER_FIELDS if body.get(f) is not None})
    blob = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def is_cacheable(body: dict, cfg: CacheConfig, header: str | None) -> bool:
    """Deterministic requests only, unless the client opts in/out via `x-gateway-cache`."""
    if header == "skip":
        return False
    if header == "allow":
        return True
    if (body.get("n") or 1) != 1:
        return False
    return not cfg.require_zero_temperature or body.get("temperature") == 0


class ExactCache:
    def __init__(self, redis: Redis, ttl_s: int):
        self.redis = redis
        self.ttl_s = ttl_s

    async def get(self, fingerprint: str) -> dict | None:
        raw = await self.redis.get(f"exact:{fingerprint}")
        return json.loads(raw) if raw else None

    async def put(self, fingerprint: str, response: dict, cost_usd: float) -> None:
        entry = json.dumps({"response": response, "cost_usd": cost_usd})
        await self.redis.set(f"exact:{fingerprint}", entry, ex=self.ttl_s)

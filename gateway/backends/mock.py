"""In-process fake LLM with realistic timing and failure injection.

It lets the compose demo, the tests and the load test run with zero API keys,
and lets the load test "cut" a provider mid-run (`down`) to measure fallback.
Timing model: sleep TTFT, then emit tokens at `tokens_per_s` - the same two
numbers you'd use to describe a real serving system's latency.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from typing import AsyncIterator

import numpy as np

from gateway.backends.base import Backend
from gateway.errors import BackendError
from gateway.openai_format import (
    estimate_prompt_tokens, last_user_text, make_chunk, make_completion, make_usage, new_id,
)

EMBED_DIM = 256
_WORDS = ("the gateway routes each request to a backend and caches the answer so repeated "
          "prompts are cheap while latency stays low under load").split()


def hashing_embedding(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic bag-of-words embedding via the hashing trick.

    Lexical, not semantic: "capital of France" vs "capital of Germany" score
    high. That's a feature for the demo - it shows why the threshold must be
    strict and tuned with a real embedding model on a labeled set.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist() if norm else vec.tolist()


class MockBackend(Backend):
    def __init__(self, cfg, rng: random.Random | None = None):
        super().__init__(cfg)
        self.opts = cfg.mock
        self.rng = rng or random.Random()

    @property
    def down(self) -> bool:
        return self.opts.down

    def set_down(self, down: bool) -> None:
        self.opts.down = down

    def _maybe_fail(self) -> None:
        if self.opts.down:
            raise BackendError.connection(self.name, "mock backend is down")
        if self.opts.error_rate and self.rng.random() < self.opts.error_rate:
            raise BackendError(self.name, 500, "injected mock failure", retryable=True)

    def _reply(self, body: dict) -> tuple[str | None, list[dict] | None]:
        """Text answer, or a tool call when tools are offered and none were answered yet."""
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        if tools and body.get("tool_choice") != "none" and messages and messages[-1].get("role") == "user":
            fn = tools[0]["function"]
            props = (fn.get("parameters") or {}).get("properties") or {}
            args = {k: "mock" for k, v in props.items() if v.get("type", "string") == "string"}
            return None, [{"id": new_id("call"), "type": "function",
                           "function": {"name": fn["name"], "arguments": json.dumps(args)}}]
        prompt = last_user_text(messages)[:60]
        n = min(int(body.get("max_completion_tokens") or body.get("max_tokens") or self.opts.output_tokens),
                self.opts.output_tokens)
        seed = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        filler = [_WORDS[(seed + i) % len(_WORDS)] for i in range(max(0, n - 4))]
        return " ".join([f"[{self.name}] re: {prompt!r}."] + filler), None

    async def chat(self, model: str, body: dict) -> dict:
        self._maybe_fail()
        text, tool_calls = self._reply(body)
        out_tokens = len((text or "").split()) or 8
        await asyncio.sleep((self.opts.ttft_ms / 1000) + out_tokens / self.opts.tokens_per_s)
        usage = make_usage(estimate_prompt_tokens(body), out_tokens)
        finish = "tool_calls" if tool_calls else "stop"
        return make_completion(model, text, finish, tool_calls, usage)

    async def chat_stream(self, model: str, body: dict) -> AsyncIterator[dict]:
        self._maybe_fail()
        text, tool_calls = self._reply(body)
        id_ = new_id()
        await asyncio.sleep(self.opts.ttft_ms / 1000)
        yield make_chunk(id_, model, {"role": "assistant", "content": ""})
        words = (text or "").split(" ") if text else []
        for i, w in enumerate(words):
            yield make_chunk(id_, model, {"content": w if i == 0 else " " + w})
            await asyncio.sleep(1 / self.opts.tokens_per_s)
        for idx, tc in enumerate(tool_calls or []):
            yield make_chunk(id_, model, {"tool_calls": [{"index": idx, **tc}]})
        yield make_chunk(id_, model, finish_reason="tool_calls" if tool_calls else "stop")
        yield make_chunk(id_, model, usage=make_usage(estimate_prompt_tokens(body), len(words) or 8))

    async def embed(self, model: str, body: dict) -> dict:
        self._maybe_fail()
        inputs = body.get("input")
        inputs = [inputs] if isinstance(inputs, str) else list(inputs or [])
        if inputs and not isinstance(inputs[0], str):
            raise BackendError(self.name, 400, "mock embeddings accept strings only", retryable=False)
        await asyncio.sleep(self.opts.ttft_ms / 4000)
        data = [{"object": "embedding", "index": i, "embedding": hashing_embedding(t)}
                for i, t in enumerate(inputs)]
        tokens = sum(max(1, len(t) // 4) for t in inputs)
        return {"object": "list", "data": data, "model": model,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens}}

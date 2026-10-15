"""OpenAI wire-format helpers: the gateway's internal lingua franca.

Every backend speaks OpenAI chat-completion dicts to the rest of the gateway,
so caching, accounting and streaming code only has to understand one shape.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Iterable


def new_id(prefix: str = "chatcmpl") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def message_text(content: Any) -> str:
    """Flatten OpenAI message content (str or list of parts) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def estimate_tokens(text: str) -> int:
    """~4 chars/token. Good enough for routing rules and rate limits; real
    usage numbers from the backend are used for billing whenever available."""
    return max(1, len(text) // 4) if text else 0


def estimate_prompt_tokens(body: dict) -> int:
    msgs = body.get("messages") or []
    text = "".join(message_text(m.get("content")) for m in msgs)
    tools = json.dumps(body.get("tools")) if body.get("tools") else ""
    return estimate_tokens(text + tools) + 4 * len(msgs)


def last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return message_text(m.get("content"))
    return ""


def make_usage(prompt: int, completion: int, cached: int = 0) -> dict:
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if cached:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def make_completion(model: str, content: str | None, finish_reason: str = "stop",
                    tool_calls: list[dict] | None = None, usage: dict | None = None,
                    id_: str | None = None) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": id_ or new_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage or make_usage(0, 0),
    }


def make_chunk(id_: str, model: str, delta: dict | None = None,
               finish_reason: str | None = None, usage: dict | None = None) -> dict:
    """One `chat.completion.chunk`. A usage-only chunk has empty `choices`,
    exactly like OpenAI's `stream_options.include_usage` final chunk."""
    chunk: dict[str, Any] = {"id": id_, "object": "chat.completion.chunk",
                             "created": int(time.time()), "model": model}
    if usage is not None and delta is None:
        chunk["choices"] = []
        chunk["usage"] = usage
    else:
        chunk["choices"] = [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}]
    return chunk


def chunks_from_completion(resp: dict, words_per_chunk: int = 4) -> Iterable[dict]:
    """Replay a stored completion as stream chunks (cache hits on stream=True)."""
    id_, model = resp.get("id", new_id()), resp.get("model", "")
    choice = resp["choices"][0]
    msg = choice["message"]
    yield make_chunk(id_, model, {"role": "assistant", "content": ""})
    words = (msg.get("content") or "").split(" ")
    for i in range(0, len(words), words_per_chunk):
        piece = " ".join(words[i:i + words_per_chunk])
        if i + words_per_chunk < len(words):
            piece += " "
        if piece:
            yield make_chunk(id_, model, {"content": piece})
    for idx, tc in enumerate(msg.get("tool_calls") or []):
        yield make_chunk(id_, model, {"tool_calls": [{"index": idx, **tc}]})
    yield make_chunk(id_, model, finish_reason=choice.get("finish_reason", "stop"))
    yield make_chunk(id_, model, usage=resp.get("usage") or make_usage(0, 0))


class ChunkAccumulator:
    """Rebuild a full completion from stream chunks, for cache writes and billing."""

    def __init__(self) -> None:
        self.id = ""
        self.model = ""
        self.text: list[str] = []
        self.tool_calls: dict[int, dict] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.saw_content = False

    def add(self, chunk: dict) -> None:
        self.id = self.id or chunk.get("id", "")
        self.model = self.model or chunk.get("model", "")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text.append(delta["content"])
                self.saw_content = True
            for tc in delta.get("tool_calls") or []:
                self.saw_content = True
                slot = self.tool_calls.setdefault(
                    tc.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

    @property
    def content(self) -> str:
        return "".join(self.text)

    def to_completion(self, prompt_tokens_estimate: int) -> dict:
        usage = self.usage or make_usage(prompt_tokens_estimate, estimate_tokens(self.content))
        tool_calls = [self.tool_calls[i] for i in sorted(self.tool_calls)]
        return make_completion(self.model, self.content if self.text or not tool_calls else None,
                               self.finish_reason or "stop", tool_calls or None, usage, self.id or None)


def sse(data: dict | str) -> bytes:
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    return f"data: {payload}\n\n".encode()


async def parse_sse(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """Parse an OpenAI-style SSE stream (`data: {...}` lines, `[DONE]` sentinel)."""
    async for line in lines:
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        if data:
            yield json.loads(data)

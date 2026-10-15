"""OpenAI Chat Completions <-> Anthropic Messages translation (pure functions).

Kept free of I/O so every mapping rule is unit-testable with plain dicts.
The main structural differences being bridged:
  * OpenAI puts system prompts in `messages`; Anthropic has a top-level `system`.
  * OpenAI tool calls live in `assistant.tool_calls` with JSON-string arguments;
    Anthropic uses `tool_use` content blocks with a parsed `input` object.
  * OpenAI has one `role: tool` message per result; Anthropic wants all results
    for a turn as `tool_result` blocks inside ONE user message.
  * Streams: Anthropic emits typed block events; OpenAI emits flat deltas.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from gateway.openai_format import make_chunk, make_completion, make_usage, message_text, new_id

log = logging.getLogger("gateway.anthropic")

DEFAULT_MAX_TOKENS = 16000

# Claude 5.x models reject temperature/top_p/top_k with a 400, so we drop them.
_NO_SAMPLING_PARAMS = re.compile(r"^claude-[a-z]+-5")

FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class TranslationError(ValueError):
    """The OpenAI request can't be expressed for Anthropic (-> HTTP 400)."""


# --------------------------------------------------------------------------- request


def to_anthropic_request(body: dict, model: str) -> dict:
    """Build kwargs for `client.messages.create/stream` from an OpenAI body."""
    messages = body.get("messages") or []
    if not messages:
        raise TranslationError("messages must not be empty")
    if messages[-1].get("role") == "assistant":
        raise TranslationError("assistant prefill (last message from the assistant) "
                               "is not supported by current Claude models; end with a user turn")
    if (body.get("n") or 1) != 1:
        raise TranslationError("n > 1 is not supported for Anthropic backends")

    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "developer"):
            system_parts.append(message_text(m.get("content")))
        elif role == "user":
            out.append({"role": "user", "content": _user_content(m.get("content"))})
        elif role == "assistant":
            blocks = _assistant_blocks(m)
            if blocks:  # Anthropic rejects empty assistant content
                out.append({"role": "assistant", "content": blocks})
        elif role == "tool":
            _append_tool_result(out, m)
        else:
            raise TranslationError(f"unsupported message role: {role!r}")

    req: dict[str, Any] = {
        "model": model,
        "max_tokens": body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "messages": out,
    }
    if system_parts:
        req["system"] = "\n\n".join(p for p in system_parts if p)
    if stop := body.get("stop"):
        req["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if not _NO_SAMPLING_PARAMS.match(model):
        if body.get("temperature") is not None:
            req["temperature"] = min(float(body["temperature"]), 1.0)  # Anthropic range is 0..1
        if body.get("top_p") is not None:
            req["top_p"] = body["top_p"]
    if tools := body.get("tools"):
        req["tools"] = [_tool(t) for t in tools]
        req["tool_choice"] = _tool_choice(body.get("tool_choice"), body.get("parallel_tool_calls"))
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema" and (schema := (fmt.get("json_schema") or {}).get("schema")):
        req["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
    # Deliberately NOT sent: `thinking` (5.x models reject "disabled"; default is fine).
    return req


def _user_content(content: Any) -> str | list[dict]:
    if content is None or isinstance(content, str):
        return content or ""
    blocks = []
    for part in content:
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            blocks.append(_image_block((part.get("image_url") or {}).get("url", "")))
    return blocks


def _image_block(url: str) -> dict:
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        media_type = header[5:].split(";")[0]
        return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _assistant_blocks(m: dict) -> list[dict]:
    blocks: list[dict] = []
    if text := message_text(m.get("content")):
        blocks.append({"type": "text", "text": text})
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError as e:
            raise TranslationError(f"tool call {tc.get('id')} has invalid JSON arguments: {e}") from None
        blocks.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": args})
    return blocks


def _append_tool_result(out: list[dict], m: dict) -> None:
    """Consecutive `role: tool` messages become ONE user message of tool_result blocks."""
    block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
             "content": message_text(m.get("content"))}
    prev = out[-1] if out else None
    if prev and prev["role"] == "user" and isinstance(prev["content"], list) and prev["content"] \
            and all(b.get("type") == "tool_result" for b in prev["content"]):
        prev["content"].append(block)
    else:
        out.append({"role": "user", "content": [block]})


def _tool(t: dict) -> dict:
    fn = t.get("function") or {}
    tool = {"name": fn["name"], "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}}
    if fn.get("strict"):
        tool["strict"] = True
    return tool


def _tool_choice(choice: Any, parallel: bool | None) -> dict:
    """Only "auto" and "none" map cleanly.

    "required" and a specific function would be Anthropic `any` / `tool`, which
    current Claude models (Opus/Sonnet 5.5) reject with a 400. We degrade them
    to `auto` and log it; callers who need a guaranteed call should say so in
    the prompt (and check for a tool call in the response).
    """
    if choice == "none":
        return {"type": "none"}
    if choice not in (None, "auto"):
        log.warning("tool_choice %r not supported on Claude 5.x; using auto", choice)
    tc: dict[str, Any] = {"type": "auto"}
    if parallel is False:
        tc["disable_parallel_tool_use"] = True
    return tc


# --------------------------------------------------------------------------- response


def _usage(u: dict | None) -> dict:
    u = u or {}
    cached = u.get("cache_read_input_tokens") or 0
    prompt = (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + cached
    return make_usage(prompt, u.get("output_tokens") or 0, cached)


def from_anthropic_message(msg: dict, model: str) -> dict:
    """Anthropic `Message` dict -> OpenAI `chat.completion` dict. Thinking blocks are dropped."""
    content = msg.get("content") or []
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    tool_calls = [
        {"id": b["id"], "type": "function",
         "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
        for b in content if b.get("type") == "tool_use"
    ]
    return make_completion(
        msg.get("model") or model,
        text if text or not tool_calls else None,
        FINISH_REASONS.get(msg.get("stop_reason") or "end_turn", "stop"),
        tool_calls or None,
        _usage(msg.get("usage")),
        id_=msg.get("id"),
    )


class StreamTranslator:
    """Anthropic stream events -> OpenAI `chat.completion.chunk` dicts.

    Anthropic numbers content blocks across text/thinking/tool_use; OpenAI
    numbers only tool calls. So we keep a map from block index to tool-call
    index and drop thinking deltas entirely.
    """

    def __init__(self, model: str):
        self.id = new_id()
        self.model = model
        self.tool_index: dict[int, int] = {}
        self.usage: dict[str, int] = {}
        self.stop_reason: str | None = None

    def _chunk(self, delta: dict) -> dict:
        return make_chunk(self.id, self.model, delta)

    def feed(self, event: dict) -> list[dict]:
        etype = event.get("type")
        if etype == "message_start":
            msg = event.get("message") or {}
            self.id = msg.get("id") or self.id
            self.model = msg.get("model") or self.model
            self.usage.update({k: v for k, v in (msg.get("usage") or {}).items() if isinstance(v, int)})
            return [self._chunk({"role": "assistant", "content": ""})]
        if etype == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") != "tool_use":
                return []
            idx = self.tool_index[event["index"]] = len(self.tool_index)
            return [self._chunk({"tool_calls": [{"index": idx, "id": block["id"], "type": "function",
                                                 "function": {"name": block["name"], "arguments": ""}}]})]
        if etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return [self._chunk({"content": delta.get("text", "")})]
            if delta.get("type") == "input_json_delta" and event.get("index") in self.tool_index:
                return [self._chunk({"tool_calls": [{"index": self.tool_index[event["index"]],
                                                     "function": {"arguments": delta.get("partial_json", "")}}]})]
            return []  # thinking_delta, signature_delta, citations_delta
        if etype == "message_delta":
            self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason
            # message_delta usage is cumulative; ignore None fields.
            self.usage.update({k: v for k, v in (event.get("usage") or {}).items() if isinstance(v, int)})
            return []
        if etype == "message_stop":
            finish = FINISH_REASONS.get(self.stop_reason or "end_turn", "stop")
            return [make_chunk(self.id, self.model, finish_reason=finish),
                    make_chunk(self.id, self.model, usage=_usage(self.usage))]
        return []  # ping, content_block_stop, SDK helper events (text, input_json, ...)

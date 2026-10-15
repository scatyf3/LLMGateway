"""AnthropicBackend with a monkeypatched SDK client: checks the SDK calls we
make and that SDK-typed objects/errors are handled, without any network."""

import anthropic
import httpx2
import pytest
from anthropic.types import Message, RawContentBlockDeltaEvent, RawMessageStartEvent

from gateway.backends.anthropic import AnthropicBackend
from gateway.config import BackendConfig
from gateway.errors import BackendError
from tests.test_anthropic_translate import _stream_events


class FakeStream:
    def __init__(self, events):
        self.events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()


class FakeMessages:
    def __init__(self, response=None, events=None, error=None):
        self.response, self.events, self.error = response, events, error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.error:
            raise self.error
        return self.response

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        if self.error:
            raise self.error
        return FakeStream(self.events)


class FakeClient:
    def __init__(self, **kw):
        self.messages = FakeMessages(**kw)


def _backend(client) -> AnthropicBackend:
    return AnthropicBackend(BackendConfig(name="anthropic", type="anthropic"), client=client)


async def test_chat_uses_sdk_message_object():
    msg = Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "Paris."}], "stop_reason": "end_turn",
        "stop_sequence": None, "usage": {"input_tokens": 12, "output_tokens": 3}})
    client = FakeClient(response=msg)
    out = await _backend(client).chat("claude-haiku-5-5", {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Capital of France?"}],
        "temperature": 0})
    kind, kwargs = client.messages.calls[0]
    assert kind == "create" and kwargs["system"] == "Be brief." and "temperature" not in kwargs
    assert out["choices"][0]["message"]["content"] == "Paris."
    assert out["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}


async def test_stream_accepts_sdk_typed_events():
    events = [
        RawMessageStartEvent.model_validate({"type": "message_start", "message": {
            "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-haiku-5-5", "content": [],
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 0}}}),
        *_stream_events()[4:6],
        RawContentBlockDeltaEvent.model_validate({"type": "content_block_delta", "index": 1,
                                                   "delta": {"type": "text_delta", "text": "!"}}),
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
        {"type": "message_stop"},
    ]
    client = FakeClient(events=events)
    chunks = [c async for c in _backend(client).chat_stream("claude-haiku-5-5",
                                                            {"messages": [{"role": "user", "content": "hi"}]})]
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == "Hel!"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert client.messages.calls[0][0] == "stream"


@pytest.mark.parametrize("status,retryable", [(500, True), (529, True), (429, True), (400, False), (401, False)])
async def test_sdk_status_errors_map_to_backend_errors(status, retryable):
    resp = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    err = anthropic.APIStatusError("boom", response=resp, body=None)
    with pytest.raises(BackendError) as ei:
        await _backend(FakeClient(error=err)).chat("claude-haiku-5-5", {"messages": [{"role": "user", "content": "x"}]})
    assert ei.value.status == status and ei.value.retryable is retryable


async def test_timeout_and_prefill_errors():
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    with pytest.raises(BackendError) as ei:
        await _backend(FakeClient(error=anthropic.APITimeoutError(request=req))).chat(
            "claude-haiku-5-5", {"messages": [{"role": "user", "content": "x"}]})
    assert ei.value.reason == "timeout" and ei.value.retryable

    with pytest.raises(BackendError) as ei:
        await _backend(FakeClient()).chat("claude-haiku-5-5", {"messages": [
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "prefill"}]})
    assert ei.value.status == 400 and not ei.value.retryable


async def test_embeddings_unsupported():
    with pytest.raises(BackendError) as ei:
        await _backend(FakeClient()).embed("claude-haiku-5-5", {"input": "x"})
    assert ei.value.status == 400


def test_unconfigured_backend_is_unavailable(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = AnthropicBackend(BackendConfig(name="anthropic", type="anthropic", api_key_env="ANTHROPIC_API_KEY"))
    assert not b.available

import json

import pytest

from gateway.backends.anthropic_translate import (
    StreamTranslator, TranslationError, from_anthropic_message, to_anthropic_request,
)
from gateway.openai_format import ChunkAccumulator

TOOLS = [{"type": "function", "function": {
    "name": "search_docs", "description": "Search the docs",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}}]


def test_system_and_developer_messages_become_top_level_system():
    req = to_anthropic_request({"messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "developer", "content": "Cite sources."},
        {"role": "user", "content": "hi"},
    ]}, "claude-haiku-5-5")
    assert req["system"] == "You are terse.\n\nCite sources."
    assert req["messages"] == [{"role": "user", "content": "hi"}]
    assert req["max_tokens"] == 16000
    assert "thinking" not in req


def test_tool_calls_and_results_round_trip_with_merging():
    body = {"messages": [
        {"role": "user", "content": "find two things"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "search_docs", "arguments": '{"q": "a"}'}},
            {"id": "call_2", "type": "function", "function": {"name": "search_docs", "arguments": '{"q": "b"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "result a"},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "result b"}]},
    ], "tools": TOOLS}
    # Last message is a tool result -> fine (it's a user turn on the Anthropic side).
    req = to_anthropic_request(body, "claude-sonnet-5-5")
    assistant, results = req["messages"][1], req["messages"][2]
    assert assistant == {"role": "assistant", "content": [
        {"type": "tool_use", "id": "call_1", "name": "search_docs", "input": {"q": "a"}},
        {"type": "tool_use", "id": "call_2", "name": "search_docs", "input": {"q": "b"}},
    ]}
    assert results["role"] == "user"
    assert results["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "result a"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "result b"},
    ]
    assert len(req["messages"]) == 3  # two tool messages merged into one user turn
    assert req["tools"] == [{"name": "search_docs", "description": "Search the docs",
                             "input_schema": TOOLS[0]["function"]["parameters"]}]
    assert req["tool_choice"] == {"type": "auto"}


@pytest.mark.parametrize("choice,expected", [
    ("auto", {"type": "auto"}),
    ("none", {"type": "none"}),
    ("required", {"type": "auto"}),  # unsupported on Claude 5.x -> degraded
    ({"type": "function", "function": {"name": "search_docs"}}, {"type": "auto"}),
])
def test_tool_choice_mapping(choice, expected):
    req = to_anthropic_request({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS,
                                "tool_choice": choice}, "claude-opus-5-5")
    assert req["tool_choice"] == expected


def test_parallel_tool_calls_false_disables_parallel_use():
    req = to_anthropic_request({"messages": [{"role": "user", "content": "x"}], "tools": TOOLS,
                                "parallel_tool_calls": False}, "claude-opus-5-5")
    assert req["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_sampling_params_dropped_for_claude_5_kept_for_older():
    body = {"messages": [{"role": "user", "content": "x"}], "temperature": 0.2, "top_p": 0.9,
            "max_completion_tokens": 50, "stop": "END"}
    new = to_anthropic_request(body, "claude-sonnet-5-5")
    assert "temperature" not in new and "top_p" not in new
    assert new["max_tokens"] == 50 and new["stop_sequences"] == ["END"]
    old = to_anthropic_request(body, "claude-3-5-haiku-latest")
    assert old["temperature"] == 0.2 and old["top_p"] == 0.9


def test_assistant_prefill_rejected():
    with pytest.raises(TranslationError, match="prefill"):
        to_anthropic_request({"messages": [{"role": "user", "content": "x"},
                                           {"role": "assistant", "content": "Sure,"}]}, "claude-haiku-5-5")


def test_image_parts():
    req = to_anthropic_request({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]}, "claude-haiku-5-5")
    assert req["messages"][0]["content"][1] == {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}


def test_response_mapping_text_tools_finish_usage():
    msg = {"id": "msg_1", "model": "claude-haiku-5-5", "stop_reason": "tool_use", "content": [
        {"type": "thinking", "thinking": "hmm", "signature": "s"},
        {"type": "text", "text": "Let me search."},
        {"type": "tool_use", "id": "toolu_1", "name": "search_docs", "input": {"q": "kv cache"}},
    ], "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50,
                 "cache_creation_input_tokens": 0}}
    out = from_anthropic_message(msg, "claude-haiku-5-5")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == "Let me search."
    assert choice["message"]["tool_calls"] == [{"id": "toolu_1", "type": "function", "function": {
        "name": "search_docs", "arguments": json.dumps({"q": "kv cache"})}}]
    assert out["usage"]["prompt_tokens"] == 150
    assert out["usage"]["completion_tokens"] == 20
    assert out["usage"]["prompt_tokens_details"] == {"cached_tokens": 50}


@pytest.mark.parametrize("stop,finish", [("end_turn", "stop"), ("max_tokens", "length"),
                                         ("refusal", "content_filter"), ("stop_sequence", "stop")])
def test_finish_reason_mapping(stop, finish):
    msg = {"content": [{"type": "text", "text": "x"}], "stop_reason": stop, "usage": {}}
    assert from_anthropic_message(msg, "m")["choices"][0]["finish_reason"] == finish


def _stream_events():
    return [
        {"type": "message_start", "message": {"id": "msg_9", "model": "claude-haiku-5-5",
                                              "usage": {"input_tokens": 30, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "secret"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "text", "text": "Hel", "snapshot": "Hel"},  # SDK helper event: must be ignored
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2,
         "content_block": {"type": "tool_use", "id": "toolu_7", "name": "search_docs", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"q": '}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"kv"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 17}},
        {"type": "message_stop"},
    ]


def test_stream_translation_text_and_tool_deltas():
    tr = StreamTranslator("claude-haiku-5-5")
    chunks = [c for ev in _stream_events() for c in tr.feed(ev)]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    texts = [c["choices"][0]["delta"].get("content") for c in chunks if c["choices"]]
    assert "secret" not in "".join(t or "" for t in texts)
    tool_start = next(c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls"))
    tc = tool_start["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0  # OpenAI tool index, not Anthropic block index (2)
    assert tc["id"] == "toolu_7" and tc["function"] == {"name": "search_docs", "arguments": ""}
    assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == 17
    assert chunks[-1]["usage"]["prompt_tokens"] == 30

    acc = ChunkAccumulator()
    for c in chunks:
        acc.add(c)
    full = acc.to_completion(0)
    assert full["choices"][0]["message"]["content"] == "Hello"
    assert full["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"q": "kv"}'
    assert all(c["id"] == "msg_9" for c in chunks)

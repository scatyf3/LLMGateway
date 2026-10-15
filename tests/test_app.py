"""End-to-end through FastAPI (httpx ASGITransport) against the in-process mock backends."""

import json

import httpx
import pytest
import respx

from gateway.app import create_app
from gateway.backends import build_backends
from tests.conftest import TEST_KEY, chat_body, make_config


def _sse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[6:]))
    return out


async def test_healthz_and_models(client):
    assert (await client.get("/healthz")).json()["ok"] is True
    ids = [m["id"] for m in (await client.get("/v1/models")).json()["data"]]
    assert {"fast", "smart", "local", "embed"} <= set(ids)


async def test_auth_required(client):
    r = await client.post("/v1/chat/completions", json=chat_body(), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


async def test_bad_request_is_openai_shaped_400(client):
    r = await client.post("/v1/chat/completions", json={"model": "fast", "messages": []})
    assert r.status_code == 400 and "error" in r.json()


async def test_non_stream_then_exact_cache_hit(client):
    r1 = await client.post("/v1/chat/completions", json=chat_body("What is paged attention?"))
    assert r1.status_code == 200
    assert r1.headers["x-gateway-cache"] == "miss" and r1.headers["x-gateway-backend"] == "m1"
    assert float(r1.headers["x-gateway-cost-usd"]) > 0
    assert float(r1.headers["x-gateway-overhead-ms"]) >= 0
    body1 = r1.json()
    assert body1["object"] == "chat.completion" and body1["usage"]["completion_tokens"] > 0

    r2 = await client.post("/v1/chat/completions", json=chat_body("What is paged attention?"))
    assert r2.headers["x-gateway-cache"] == "exact" and r2.headers["x-gateway-backend"] == "cache"
    assert float(r2.headers["x-gateway-cost-usd"]) == 0
    assert r2.json()["choices"][0]["message"] == body1["choices"][0]["message"]


async def test_semantic_hit_and_logged(client, caplog):
    await client.post("/v1/chat/completions", json=chat_body("Explain speculative decoding"))
    with caplog.at_level("INFO", logger="gateway.semantic_cache"):
        r = await client.post("/v1/chat/completions", json=chat_body("explain speculative decoding?"))
    assert r.headers["x-gateway-cache"] == "semantic"
    assert any("semantic_cache_hit" in rec.message for rec in caplog.records)


async def test_nonzero_temperature_not_cached(client):
    body = {**chat_body("random please"), "temperature": 0.9}
    await client.post("/v1/chat/completions", json=body)
    r = await client.post("/v1/chat/completions", json=body)
    assert r.headers["x-gateway-cache"] == "miss"


async def test_streaming_miss_then_streamed_cache_replay(client):
    body = chat_body("Tell me about TTFT", stream=True, stream_options={"include_usage": True})
    r = await client.post("/v1/chat/completions", json=body)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")
    chunks = _sse(r.text)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert "Tell me about TTFT" in text
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0

    r2 = await client.post("/v1/chat/completions", json={**body, "stream_options": None})
    assert r2.headers["x-gateway-cache"] == "exact"
    chunks2 = _sse(r2.text)
    text2 = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks2 if c["choices"])
    assert text2 == text
    assert all(c["choices"] for c in chunks2)  # usage chunk dropped: client didn't ask for it


async def test_tool_calling_through_gateway(client):
    tools = [{"type": "function", "function": {"name": "search_docs", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}}}}]
    r = await client.post("/v1/chat/completions", json=chat_body("find docs", tools=tools))
    msg = r.json()["choices"][0]
    assert msg["finish_reason"] == "tool_calls"
    call = msg["message"]["tool_calls"][0]
    follow = chat_body("find docs", tools=tools)
    follow["messages"] += [{"role": "assistant", "content": None, "tool_calls": [call]},
                           {"role": "tool", "tool_call_id": call["id"], "content": "doc text"}]
    r2 = await client.post("/v1/chat/completions", json=follow)
    assert r2.json()["choices"][0]["finish_reason"] == "stop"


async def test_streaming_tool_call(client):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}}}}}]
    r = await client.post("/v1/chat/completions", json=chat_body("x", tools=tools, stream=True))
    chunks = _sse(r.text)
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert any(d.get("tool_calls") for d in deltas)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_embeddings(client):
    r = await client.post("/v1/embeddings", json={"model": "embed", "input": ["a", "b"]})
    data = r.json()["data"]
    assert r.status_code == 200 and len(data) == 2 and len(data[0]["embedding"]) == 256


async def test_admin_toggle_down_causes_fallback(client):
    r = await client.post("/admin/backends/m1", json={"down": True})
    assert r.json()["down"] is True
    resp = await client.post("/v1/chat/completions", json=chat_body("anything", temperature=0.5))
    assert resp.headers["x-gateway-backend"] == "m2" and resp.headers["x-gateway-fallbacks"] == "1"
    await client.post("/admin/backends/m1", json={"down": False})


async def test_long_prompt_routed_to_local(client):
    r = await client.post("/v1/chat/completions", json=chat_body("word " * 5000))
    assert r.headers["x-gateway-alias"] == "local" and r.headers["x-gateway-backend"] == "m2"


async def test_rate_limit_429(client):
    h = {"Authorization": "Bearer sk-limited"}
    codes = [(await client.post("/v1/chat/completions", json=chat_body(f"q{i}"), headers=h)).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


async def test_budget_402(client):
    h = {"Authorization": "Bearer sk-broke"}
    assert (await client.post("/v1/chat/completions", json=chat_body("one"), headers=h)).status_code == 200
    r = await client.post("/v1/chat/completions", json=chat_body("two"), headers=h)
    assert r.status_code == 402 and r.json()["error"]["code"] == "budget_exceeded"


async def test_usage_accounting_and_admin(client):
    await client.post("/v1/chat/completions", json=chat_body("count me", model="smart"))
    usage = (await client.get("/admin/usage")).json()["keys"]["demo"]
    assert usage["requests"] >= 1 and usage["cost_usd"] > 0 and usage["completion_tokens"] > 0
    r = await client.get("/admin/usage", headers={"Authorization": "Bearer sk-free"})
    assert r.status_code == 403


async def test_metrics_exposed(client):
    await client.post("/v1/chat/completions", json=chat_body("metrics"))
    text = (await client.get("/metrics")).text
    assert "gateway_requests_total" in text and "gateway_overhead_seconds_bucket" in text


@respx.mock
async def test_openai_backend_passthrough_stream_and_nonstream(monkeypatch, redis):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-upstream")
    cfg = make_config(
        backends=[{"name": "openai", "type": "openai", "base_url": "http://upstream/v1",
                   "api_key_env": "OPENAI_API_KEY"},
                  {"name": "m1", "type": "mock"}, {"name": "m2", "type": "mock"}],
        aliases={"fast": [{"backend": "openai", "model": "gpt-4.1-mini"}],
                 "local": [{"backend": "m2", "model": "x"}], "embed": [{"backend": "openai", "model": "emb"}]})
    app = create_app(cfg, redis, build_backends(cfg.backends))
    completion = {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-4.1-mini",
                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi!"},
                               "finish_reason": "stop"}],
                  "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}
    sse_body = (
        'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
        'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-4.1-mini","choices":[{"index":0,"delta":{"content":"yo"},"finish_reason":"stop"}]}\n\n'
        'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-4.1-mini","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}\n\n'
        "data: [DONE]\n\n")

    def handler(request: httpx.Request):
        sent = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer sk-upstream"
        assert sent["model"] == "gpt-4.1-mini"
        if sent.get("stream"):
            assert sent["stream_options"] == {"include_usage": True}
            return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completion)

    respx.post("http://upstream/v1/chat/completions").mock(side_effect=handler)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw",
                                 headers={"Authorization": f"Bearer {TEST_KEY}"}) as c:
        r = await c.post("/v1/chat/completions", json=chat_body("hi"))
        assert r.json()["choices"][0]["message"]["content"] == "hi!"
        assert r.headers["x-gateway-backend"] == "openai"
        r = await c.post("/v1/chat/completions", json=chat_body("hey", stream=True))
        chunks = _sse(r.text)
        assert [c["choices"][0]["delta"].get("content") for c in chunks] == ["", "yo"]  # usage chunk hidden


@respx.mock
async def test_openai_5xx_falls_back_to_mock(monkeypatch, redis):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-upstream")
    cfg = make_config(
        backends=[{"name": "openai", "type": "openai", "base_url": "http://upstream/v1",
                   "api_key_env": "OPENAI_API_KEY"},
                  {"name": "m1", "type": "mock", "mock": {"ttft_ms": 0, "tokens_per_s": 100000}},
                  {"name": "m2", "type": "mock"}],
        aliases={"fast": [{"backend": "openai", "model": "gpt-4.1-mini"}, {"backend": "m1", "model": "mock-small"}],
                 "local": [{"backend": "m2", "model": "x"}], "embed": [{"backend": "m1", "model": "e"}]})
    app = create_app(cfg, redis, build_backends(cfg.backends))
    respx.post("http://upstream/v1/chat/completions").mock(return_value=httpx.Response(503, text="overloaded"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw",
                                 headers={"Authorization": f"Bearer {TEST_KEY}"}) as c:
        r = await c.post("/v1/chat/completions", json=chat_body("hi", stream=True))
        assert r.status_code == 200 and r.headers["x-gateway-backend"] == "m1"
        assert r.headers["x-gateway-fallbacks"] == "1"


@pytest.mark.parametrize("path", ["/admin/usage", "/admin/backends"])
async def test_admin_requires_auth(client, path):
    r = await client.get(path, headers={"Authorization": ""})
    assert r.status_code == 401

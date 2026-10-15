import numpy as np

from gateway.backends.mock import hashing_embedding
from gateway.cache import ExactCache, SemanticCache, is_cacheable, namespace, request_fingerprint
from gateway.config import CacheConfig, SemanticCacheConfig
from gateway.openai_format import make_completion
from tests.conftest import chat_body


def test_fingerprint_ignores_stream_and_key_order_but_not_content():
    a = chat_body("hi", stream=True, user="u1")
    b = {"messages": [{"content": "  hi ", "role": "user"}], "temperature": 0, "model": "fast"}
    assert request_fingerprint("fast", a) == request_fingerprint("fast", b)
    assert request_fingerprint("fast", a) != request_fingerprint("smart", a)
    assert request_fingerprint("fast", a) != request_fingerprint("fast", chat_body("hi!"))
    assert request_fingerprint("fast", a) != request_fingerprint("fast", chat_body("hi", max_tokens=5))


def test_cacheability_policy():
    cfg = CacheConfig()
    assert is_cacheable(chat_body(), cfg, None)
    assert not is_cacheable({**chat_body(), "temperature": 0.7}, cfg, None)
    assert is_cacheable({**chat_body(), "temperature": 0.7}, cfg, "allow")
    assert not is_cacheable(chat_body(), cfg, "skip")


async def test_exact_cache_roundtrip(redis):
    cache = ExactCache(redis, ttl_s=60)
    resp = make_completion("m", "answer")
    await cache.put("abc", resp, 0.01)
    assert (await cache.get("abc"))["response"]["choices"][0]["message"]["content"] == "answer"
    assert await cache.get("missing") is None


async def _embed(text: str) -> np.ndarray:
    return np.asarray(hashing_embedding(text), dtype=np.float32)


def _sem(redis, **kw) -> SemanticCache:
    return SemanticCache(redis, SemanticCacheConfig(threshold=0.95, **kw), _embed)


async def test_semantic_hit_on_near_duplicate_and_miss_on_different(redis):
    cache = _sem(redis)
    ns = namespace("fast", chat_body("What is a KV cache?"))
    _, vec = await cache.lookup(ns, "What is a KV cache?")
    await cache.store(ns, "What is a KV cache?", vec, make_completion("m", "cached answer"), 0.02)

    hit, _ = await cache.lookup(ns, "what is a kv cache")
    assert hit and hit["response"]["choices"][0]["message"]["content"] == "cached answer"
    assert hit["cost_usd"] == 0.02
    miss, _ = await cache.lookup(ns, "How does continuous batching work in vLLM?")
    assert miss is None


async def test_semantic_namespace_isolation():
    sys_a = {"messages": [{"role": "system", "content": "You are a pirate."}, {"role": "user", "content": "hi"}]}
    sys_b = {"messages": [{"role": "system", "content": "You are a lawyer."}, {"role": "user", "content": "hi"}]}
    history = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
                            {"role": "user", "content": "hi"}]}
    plain = {"messages": [{"role": "user", "content": "hi"}]}
    spaces = {namespace("fast", b) for b in (sys_a, sys_b, history, plain)}
    assert len(spaces) == 4
    assert namespace("fast", plain) != namespace("smart", plain)
    assert namespace("fast", plain) != namespace("fast", {**plain, "max_tokens": 10})


async def test_semantic_skips_tool_requests():
    assert namespace("fast", {"messages": [{"role": "user", "content": "x"}], "tools": [{"type": "function"}]}) is None
    assert namespace("fast", {"messages": [{"role": "user", "content": "x"},
                                           {"role": "tool", "tool_call_id": "1", "content": "r"}]}) is None


async def test_semantic_cache_evicts_oldest(redis):
    cache = _sem(redis, max_entries=2)
    ns = "n1"
    for text in ("alpha one", "beta two", "gamma three"):
        _, vec = await cache.lookup(ns, text)
        await cache.store(ns, text, vec, make_completion("m", text), 0.0)
    assert await redis.llen(f"sem:{ns}:order") == 2
    assert (await cache.lookup(ns, "alpha one"))[0] is None
    assert (await cache.lookup(ns, "gamma three"))[0] is not None

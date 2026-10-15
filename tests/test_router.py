import pytest

from gateway.config import Target, Tier
from gateway.errors import GatewayError
from gateway.router import Route, Router
from tests.conftest import ScriptedBackend, make_config

PRO = Tier(allowed_aliases=["*"], default_alias="fast")


def _router(*backends):
    return Router(make_config(), {b.name: b for b in backends})


def _route(*names):
    return Route("fast", "fast", [Target(backend=n, model="m") for n in names], "alias")


async def test_fallback_on_5xx():
    bad, good = ScriptedBackend("a", fail_status=500), ScriptedBackend("b")
    resp, served = await _router(bad, good).chat(_route("a", "b"), {"messages": []})
    assert served.target.backend == "b" and served.fallbacks == 1
    assert resp["choices"][0]["message"]["content"] == "from b"


async def test_fallback_on_timeout():
    slow, good = ScriptedBackend("a", hang_s=5, timeout_s=0.05), ScriptedBackend("b")
    _, served = await _router(slow, good).chat(_route("a", "b"), {"messages": []})
    assert served.target.backend == "b"


async def test_fallback_on_429():
    _, served = await _router(ScriptedBackend("a", fail_status=429), ScriptedBackend("b")).chat(
        _route("a", "b"), {"messages": []})
    assert served.target.backend == "b"


async def test_no_fallback_on_client_error():
    bad, good = ScriptedBackend("a", fail_status=400), ScriptedBackend("b")
    with pytest.raises(GatewayError) as ei:
        await _router(bad, good).chat(_route("a", "b"), {"messages": []})
    assert ei.value.status == 400 and good.calls == 0


async def test_all_backends_fail_is_502():
    with pytest.raises(GatewayError) as ei:
        await _router(ScriptedBackend("a", fail_status=500), ScriptedBackend("b", fail_status=503)).chat(
            _route("a", "b"), {"messages": []})
    assert ei.value.status == 502


async def test_unavailable_backends_skipped_silently():
    a = ScriptedBackend("a")
    a.cfg.enabled = False
    _, served = await _router(a, ScriptedBackend("b")).chat(_route("a", "b"), {"messages": []})
    assert served.target.backend == "b" and served.fallbacks == 0 and a.calls == 0


async def test_stream_falls_back_before_first_byte():
    stream, served = await _router(ScriptedBackend("a", fail_status=502), ScriptedBackend("b")).stream(
        _route("a", "b"), {"messages": []})
    chunks = [c async for c in stream]
    assert served.target.backend == "b"
    assert "from b" in "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])


async def test_stream_does_not_fall_back_after_first_byte():
    dies, spare = ScriptedBackend("a", fail_mid_stream=True), ScriptedBackend("b")
    stream, served = await _router(dies, spare).stream(_route("a", "b"), {"messages": []})
    assert served.target.backend == "a"
    got = []
    with pytest.raises(Exception, match="mid-stream"):
        async for c in stream:
            got.append(c)
    assert len(got) == 2 and spare.calls == 0  # the client keeps one coherent (partial) answer


def test_rule_routes_long_prompts_to_local():
    r = Router(make_config(), {})
    assert r.resolve("fast", 50, PRO).alias == "fast"
    long = r.resolve("fast", 5000, PRO)
    assert long.alias == "local" and long.reason == "rule:long-to-local"
    assert r.resolve("smart", 5000, PRO).alias == "smart"  # rule only applies to listed aliases


def test_tier_downgrades_disallowed_alias():
    r = Router(make_config(), {})
    route = r.resolve("smart", 10, Tier(allowed_aliases=["fast"], default_alias="fast"))
    assert route.alias == "fast" and route.reason == "tier"


def test_unknown_alias_404():
    with pytest.raises(GatewayError) as ei:
        Router(make_config(), {}).resolve("gpt-9", 10, PRO)
    assert ei.value.status == 404

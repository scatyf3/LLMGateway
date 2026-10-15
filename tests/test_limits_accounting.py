import pytest

from gateway.accounting import Accountant, cost_usd
from gateway.config import ApiKey, Price
from gateway.errors import GatewayError
from gateway.limits import Limiter

PRICES = {"m": Price(input=1.0, output=2.0)}


def test_cost_formula():
    assert cost_usd(PRICES, "m", {"prompt_tokens": 1_000_000, "completion_tokens": 500_000}) == pytest.approx(2.0)
    assert cost_usd(PRICES, "unknown", {"prompt_tokens": 10}) == 0.0


async def test_rpm_bucket_blocks_then_refills(redis):
    lim, key = Limiter(redis), ApiKey(id="k", rpm=2, tpm=10**6)
    await lim.check_rate(key, 10, now=1000.0)
    await lim.check_rate(key, 10, now=1000.0)
    with pytest.raises(GatewayError) as ei:
        await lim.check_rate(key, 10, now=1000.0)
    assert ei.value.status == 429 and ei.value.code == "rate_limit_rpm"
    assert int(ei.value.headers["retry-after"]) >= 1
    await lim.check_rate(key, 10, now=1030.0)  # 30s at 2/min refills one request


async def test_tpm_bucket_and_post_paid_output_tokens(redis):
    lim, key = Limiter(redis), ApiKey(id="k", rpm=1000, tpm=1000)
    await lim.check_rate(key, 600, now=0.0)
    await lim.debit_output_tokens(key, 400)  # bucket now ~0
    with pytest.raises(GatewayError) as ei:
        await lim.check_rate(key, 100, now=0.0)
    assert ei.value.code == "rate_limit_tpm"


async def test_rejected_request_consumes_nothing(redis):
    """Both buckets are checked before either is debited (atomic in one Lua script)."""
    lim, key = Limiter(redis), ApiKey(id="k", rpm=5, tpm=100)
    await lim.check_rate(key, 90, now=0.0)
    with pytest.raises(GatewayError) as ei:
        await lim.check_rate(key, 50, now=0.0)
    assert ei.value.code == "rate_limit_tpm"
    assert float(await redis.hget("rl:k:rpm", "tokens")) == pytest.approx(4.0)  # only the first request


async def test_budget_exceeded_is_402(redis):
    acct, lim = Accountant(redis, PRICES), Limiter(redis)
    key = ApiKey(id="k", monthly_budget_usd=1.0)
    await lim.check_budget(key)
    charged = await acct.record("k", "m", {"prompt_tokens": 1_000_000, "completion_tokens": 0}, "miss")
    assert charged == pytest.approx(1.0)
    with pytest.raises(GatewayError) as ei:
        await lim.check_budget(key)
    assert ei.value.status == 402 and ei.value.code == "budget_exceeded"


async def test_cache_hits_are_free_and_tracked_as_savings(redis):
    acct = Accountant(redis, PRICES)
    charged = await acct.record("k", "m", {"prompt_tokens": 1000, "completion_tokens": 1000}, "exact", saved_usd=0.5)
    usage = await acct.usage("k")
    assert charged == 0.0 and usage.get("cost_usd", 0.0) == 0.0
    assert usage["saved_usd"] == pytest.approx(0.5) and usage["cache_hits_exact"] == 1

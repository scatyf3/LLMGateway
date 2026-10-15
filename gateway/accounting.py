"""Cost and token accounting per key: Redis for budgets/admin, Prometheus for dashboards."""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis

from gateway import metrics
from gateway.config import Price
from gateway.limits import usage_key


def cost_usd(prices: dict[str, Price], model: str, usage: dict | None) -> float:
    """Unknown models cost 0 (and are visible as such on the dashboard) rather than failing a request."""
    price = prices.get(model)
    if not price or not usage:
        return 0.0
    return (usage.get("prompt_tokens", 0) * price.input
            + usage.get("completion_tokens", 0) * price.output) / 1_000_000


@dataclass
class Accountant:
    redis: Redis
    prices: dict[str, Price]

    async def record(self, key_id: str, model: str, usage: dict | None, cache: str,
                     saved_usd: float = 0.0) -> float:
        """Record one finished request; returns the cost charged to the key."""
        usage = usage or {}
        charged = 0.0 if cache != "miss" else cost_usd(self.prices, model, usage)
        hkey = usage_key(key_id)
        pipe = self.redis.pipeline(transaction=False)
        pipe.hincrby(hkey, "requests", 1)
        if cache == "miss":
            pipe.hincrbyfloat(hkey, "cost_usd", charged)
            pipe.hincrby(hkey, "prompt_tokens", usage.get("prompt_tokens", 0))
            pipe.hincrby(hkey, "completion_tokens", usage.get("completion_tokens", 0))
        else:
            pipe.hincrby(hkey, f"cache_hits_{cache}", 1)
            pipe.hincrbyfloat(hkey, "saved_usd", saved_usd)
        pipe.expire(hkey, 60 * 60 * 24 * 100)
        await pipe.execute()

        if cache == "miss":
            metrics.TOKENS.labels(key_id, model, "input").inc(usage.get("prompt_tokens", 0))
            metrics.TOKENS.labels(key_id, model, "output").inc(usage.get("completion_tokens", 0))
            metrics.COST.labels(key_id, model).inc(charged)
        else:
            metrics.COST_SAVED.labels(key_id, cache).inc(saved_usd)
        return charged

    async def usage(self, key_id: str) -> dict[str, float]:
        raw = await self.redis.hgetall(usage_key(key_id))
        return {k.decode() if isinstance(k, bytes) else k: float(v) for k, v in raw.items()}

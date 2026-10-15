"""Per-key rate limits (token buckets) and monthly budgets, both in Redis.

Token bucket, not fixed window: a fixed window lets a client send 2x its limit
across a window boundary; a bucket refills continuously and allows bursts up
to capacity. Both buckets (requests and tokens) are checked and debited in ONE
Lua script so the check-then-debit is atomic across gateway replicas.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis

from gateway import metrics
from gateway.config import ApiKey
from gateway.errors import GatewayError

# KEYS: bucket keys. ARGV: now, then (capacity, refill_per_s, cost) per key.
# Returns {allowed, retry_after_ms, index_of_limiting_bucket}.
_TAKE = """
local now = tonumber(ARGV[1])
local n = #KEYS
local tokens = {}
for i = 1, n do
  local cap = tonumber(ARGV[2 + (i-1)*3])
  local rate = tonumber(ARGV[3 + (i-1)*3])
  local cost = tonumber(ARGV[4 + (i-1)*3])
  local b = redis.call('HMGET', KEYS[i], 'tokens', 'ts')
  local t = tonumber(b[1]) or cap
  local ts = tonumber(b[2]) or now
  t = math.min(cap, t + (now - ts) * rate)
  if t < cost then
    return {0, math.ceil((cost - t) / rate * 1000), i}
  end
  tokens[i] = t - cost
end
for i = 1, n do
  local cap = tonumber(ARGV[2 + (i-1)*3])
  local rate = tonumber(ARGV[3 + (i-1)*3])
  redis.call('HSET', KEYS[i], 'tokens', tokens[i], 'ts', now)
  redis.call('EXPIRE', KEYS[i], math.ceil(cap / rate) + 60)
end
return {1, 0, 0}
"""

# Post-paid debit for output tokens (unknown until the response is done).
# Allowed to go negative: the next request waits until the debt is refilled.
_DEBIT = """
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
if b[1] then redis.call('HSET', KEYS[1], 'tokens', tonumber(b[1]) - tonumber(ARGV[1])) end
return 1
"""


def month() -> str:
    return time.strftime("%Y-%m", time.gmtime())


def usage_key(key_id: str) -> str:
    return f"usage:{key_id}:{month()}"


@dataclass
class Limiter:
    redis: Redis

    async def check_rate(self, key: ApiKey, prompt_tokens: int, now: float | None = None) -> None:
        """Take 1 request + estimated prompt tokens, or raise 429 with Retry-After."""
        now = time.time() if now is None else now
        keys = [f"rl:{key.id}:rpm", f"rl:{key.id}:tpm"]
        args = [now, key.rpm, key.rpm / 60, 1, key.tpm, key.tpm / 60, min(prompt_tokens, key.tpm)]
        allowed, retry_ms, which = await self.redis.eval(_TAKE, len(keys), *keys, *args)
        if not allowed:
            limit = "rpm" if int(which) == 1 else "tpm"
            metrics.RATE_LIMITED.labels(key.id, limit).inc()
            retry_s = max(1, int(retry_ms) // 1000 + 1)
            raise GatewayError(429, f"rate limit exceeded ({limit}) for key {key.id!r}",
                               "rate_limit_error", f"rate_limit_{limit}",
                               headers={"retry-after": str(retry_s)})

    async def debit_output_tokens(self, key: ApiKey, tokens: int) -> None:
        if tokens > 0:
            await self.redis.eval(_DEBIT, 1, f"rl:{key.id}:tpm", tokens)

    async def check_budget(self, key: ApiKey) -> None:
        """402, not 429: OpenAI SDKs auto-retry 429s, which would just hammer a
        key that cannot succeed until the month rolls over or the budget is raised.
        Soft limit: checked before the request, so one request can overshoot."""
        spent = float(await self.redis.hget(usage_key(key.id), "cost_usd") or 0)
        if spent >= key.monthly_budget_usd:
            metrics.RATE_LIMITED.labels(key.id, "budget").inc()
            raise GatewayError(402, f"monthly budget of ${key.monthly_budget_usd:.2f} exhausted "
                               f"for key {key.id!r} (spent ${spent:.4f})",
                               "insufficient_quota", "budget_exceeded")

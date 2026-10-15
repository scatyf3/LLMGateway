"""Redis connection. `fakeredis://` gives an in-process Redis (with Lua) so
`uvicorn gateway.main:app` works on a laptop without running a server."""

from __future__ import annotations

from redis.asyncio import Redis


def connect(url: str) -> Redis:
    if url.startswith("fakeredis://"):
        try:
            import fakeredis
        except ImportError as e:  # pragma: no cover
            raise RuntimeError('fakeredis:// needs `pip install -e ".[local]"`') from e
        return fakeredis.FakeAsyncRedis()
    return Redis.from_url(url)

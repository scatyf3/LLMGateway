"""HTTP layer: OpenAI-compatible routes, admin routes, wiring.

`create_app` takes optional pre-built pieces so tests can inject fakeredis and
fake backends without monkeypatching globals.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from redis.asyncio import Redis

from gateway import pipeline
from gateway.accounting import Accountant
from gateway.auth import KeyStore
from gateway.backends import Backend, MockBackend, build_backends
from gateway.cache import ExactCache, SemanticCache
from gateway.config import ApiKey, GatewayConfig, Settings, load_config
from gateway.errors import GatewayError
from gateway.limits import Limiter, month
from gateway.redis_client import connect
from gateway.router import Route, Router


class ChatRequest(BaseModel):
    """Validate only what the gateway itself needs; pass every other field through."""
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict] = Field(min_length=1)
    stream: bool = False


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: str | list


class BackendToggle(BaseModel):
    down: bool | None = None
    error_rate: float | None = None


def _body(req: BaseModel) -> dict:
    """Drop top-level nulls only; nested values (e.g. `content: null` next to
    tool_calls) must reach the backend untouched."""
    return {k: v for k, v in req.model_dump().items() if v is not None}


def _make_embedder(router: Router, cfg: GatewayConfig):
    alias = cfg.cache.semantic.embedding_alias
    route = Route(alias, alias, cfg.aliases[alias], "semantic-cache")

    async def embed(text: str) -> np.ndarray:
        resp, _ = await router.embed(route, {"input": text})
        return np.asarray(resp["data"][0]["embedding"], dtype=np.float32)
    return embed


def build_gateway(cfg: GatewayConfig, redis: Redis, backends: dict[str, Backend]) -> pipeline.Gateway:
    router = Router(cfg, backends)
    sem_cfg = cfg.cache.semantic
    semantic = (SemanticCache(redis, sem_cfg, _make_embedder(router, cfg))
                if sem_cfg.enabled and sem_cfg.embedding_alias in cfg.aliases else None)
    return pipeline.Gateway(
        cfg=cfg, router=router, limiter=Limiter(redis), accountant=Accountant(redis, cfg.prices),
        exact=ExactCache(redis, cfg.cache.exact.ttl_s) if cfg.cache.exact.enabled else None,
        semantic=semantic,
    )


def create_app(cfg: GatewayConfig | None = None, redis: Redis | None = None,
               backends: dict[str, Backend] | None = None) -> FastAPI:
    settings = Settings()
    cfg = cfg or load_config(settings.config)
    redis = redis if redis is not None else connect(settings.redis_url)
    backends = backends or build_backends(cfg.backends)
    gw = build_gateway(cfg, redis, backends)
    keys = KeyStore(cfg.api_keys)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for b in backends.values():
            await b.aclose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gw

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, e: GatewayError):
        return JSONResponse(e.to_body(), status_code=e.status, headers=e.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, e: RequestValidationError):
        err = GatewayError(400, f"invalid request: {e.errors()[0].get('msg')} at "
                                f"{'.'.join(map(str, e.errors()[0].get('loc', [])))}")
        return JSONResponse(err.to_body(), status_code=400)

    def api_key(authorization: str | None = Header(default=None)) -> ApiKey:
        return keys.authenticate(authorization)

    def admin_key(key: ApiKey = Depends(api_key)) -> ApiKey:
        if not key.admin:
            raise GatewayError(403, "admin key required", "permission_error")
        return key

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest, key: ApiKey = Depends(api_key),
                               x_gateway_cache: str | None = Header(default=None)):
        return await pipeline.chat_completions(gw, key, _body(req), x_gateway_cache)

    @app.post("/v1/embeddings")
    async def embeddings(req: EmbeddingRequest, key: ApiKey = Depends(api_key)):
        return await pipeline.embeddings(gw, key, _body(req))

    @app.get("/v1/models")
    async def models(_: ApiKey = Depends(api_key)):
        data = [{"id": alias, "object": "model", "owned_by": "llm-gateway",
                 "chain": [f"{t.backend}/{t.model}" for t in chain]}
                for alias, chain in cfg.aliases.items()]
        return {"object": "list", "data": data}

    @app.get("/healthz")
    async def healthz():
        try:
            redis_ok = bool(await redis.ping())
        except Exception:
            redis_ok = False
        available = [n for n, b in backends.items() if b.available]
        return JSONResponse({"ok": redis_ok, "redis": redis_ok, "backends_available": available},
                            status_code=200 if redis_ok else 503)

    @app.get("/metrics")
    async def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/usage")
    async def usage(_: ApiKey = Depends(admin_key)):
        out = {}
        for k in cfg.api_keys:
            u = await gw.accountant.usage(k.id)
            spent = u.get("cost_usd", 0.0)
            out[k.id] = {**u, "tier": k.tier, "monthly_budget_usd": k.monthly_budget_usd,
                         "budget_remaining_usd": max(0.0, k.monthly_budget_usd - spent)}
        return {"month": month(), "keys": out}

    @app.get("/admin/backends")
    async def list_backends(_: ApiKey = Depends(admin_key)):
        return {n: {"type": b.cfg.type, "available": b.available,
                    **({"down": b.down, "error_rate": b.opts.error_rate} if isinstance(b, MockBackend) else {})}
                for n, b in backends.items()}

    @app.post("/admin/backends/{name}")
    async def toggle_backend(name: str, toggle: BackendToggle, _: ApiKey = Depends(admin_key)):
        """Failure injection for mock backends (used by the load test to cut a provider)."""
        b = backends.get(name)
        if not isinstance(b, MockBackend):
            raise GatewayError(404, f"no mock backend named {name!r}", code="not_found")
        if toggle.down is not None:
            b.set_down(toggle.down)
        if toggle.error_rate is not None:
            b.opts.error_rate = toggle.error_rate
        return {"name": name, "down": b.down, "error_rate": b.opts.error_rate}

    return app

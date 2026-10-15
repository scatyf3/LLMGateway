"""Configuration: process settings from env, routing/limits/prices from YAML.

Two layers on purpose: deployment knobs (Redis URL, config path) come from the
environment, while everything an operator reasons about (backends, aliases,
keys, prices) lives in one readable YAML file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    config: str = "config/gateway.yaml"
    # "fakeredis://" runs an in-process Redis so local dev needs no server.
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"


class MockOptions(BaseModel):
    ttft_ms: float = 100.0
    tokens_per_s: float = 100.0
    output_tokens: int = 48
    error_rate: float = 0.0
    down: bool = False


class BackendConfig(BaseModel):
    name: str
    type: Literal["anthropic", "openai", "vllm", "mock"]
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_s: float = 30.0
    enabled: bool = True
    mock: MockOptions = Field(default_factory=MockOptions)


class Target(BaseModel):
    backend: str
    model: str


class Rule(BaseModel):
    name: str
    min_prompt_tokens: int
    route_to: str
    aliases: list[str] | None = None  # None = applies to every alias


class Tier(BaseModel):
    allowed_aliases: list[str] = ["*"]
    default_alias: str

    def allows(self, alias: str) -> bool:
        return "*" in self.allowed_aliases or alias in self.allowed_aliases


class Price(BaseModel):
    input: float   # USD per 1M input tokens
    output: float  # USD per 1M output tokens


class ApiKey(BaseModel):
    id: str
    secret_sha256: str | None = None
    secret_env: str | None = None
    tier: str = "free"
    rpm: int = 60
    tpm: int = 100_000
    monthly_budget_usd: float = 10.0
    admin: bool = False


class ExactCacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 3600


class SemanticCacheConfig(BaseModel):
    enabled: bool = True
    threshold: float = 0.95
    max_entries: int = 2000
    ttl_s: int = 3600
    embedding_alias: str = "embed"


class CacheConfig(BaseModel):
    exact: ExactCacheConfig = Field(default_factory=ExactCacheConfig)
    semantic: SemanticCacheConfig = Field(default_factory=SemanticCacheConfig)
    require_zero_temperature: bool = True


class GatewayConfig(BaseModel):
    backends: list[BackendConfig]
    aliases: dict[str, list[Target]]
    default_alias: str = "fast"
    rules: list[Rule] = []
    tiers: dict[str, Tier]
    prices: dict[str, Price] = {}
    api_keys: list[ApiKey] = []
    cache: CacheConfig = Field(default_factory=CacheConfig)

    @model_validator(mode="after")
    def _check_references(self) -> "GatewayConfig":
        """Fail at startup, not at 3am, if the YAML points at something undefined."""
        names = {b.name for b in self.backends}
        for alias, chain in self.aliases.items():
            for t in chain:
                if t.backend not in names:
                    raise ValueError(f"alias {alias!r} uses unknown backend {t.backend!r}")
        for rule in self.rules:
            if rule.route_to not in self.aliases:
                raise ValueError(f"rule {rule.name!r} routes to unknown alias {rule.route_to!r}")
        for tier_name, tier in self.tiers.items():
            if tier.default_alias not in self.aliases:
                raise ValueError(f"tier {tier_name!r} default_alias {tier.default_alias!r} is not an alias")
        for key in self.api_keys:
            if key.tier not in self.tiers:
                raise ValueError(f"api key {key.id!r} has unknown tier {key.tier!r}")
        return self


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1)) or (m.group(2) or ""), text)


def load_config(path: str | Path) -> GatewayConfig:
    raw = _expand_env(Path(path).read_text(encoding="utf-8"))
    return GatewayConfig.model_validate(yaml.safe_load(raw))

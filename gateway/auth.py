"""API-key authentication.

Keys are indexed by sha256(secret): the config never has to hold plaintext
secrets, and lookup is a dict hit rather than a loop of string compares.
"""

from __future__ import annotations

import hashlib
import logging
import os

from gateway.config import ApiKey
from gateway.errors import GatewayError

log = logging.getLogger("gateway.auth")


def sha256_hex(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class KeyStore:
    def __init__(self, keys: list[ApiKey]):
        self.keys = keys
        self._by_digest: dict[str, ApiKey] = {}
        for k in keys:
            digest = k.secret_sha256
            if not digest and k.secret_env and os.environ.get(k.secret_env):
                digest = sha256_hex(os.environ[k.secret_env])
            if digest:
                self._by_digest[digest.lower()] = k
            else:
                log.warning("api key %r has no secret configured; it cannot authenticate", k.id)

    def authenticate(self, authorization: str | None) -> ApiKey:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise GatewayError(401, "missing bearer API key", "authentication_error", "invalid_api_key")
        key = self._by_digest.get(sha256_hex(token.strip()))
        if key is None:
            raise GatewayError(401, "invalid API key", "authentication_error", "invalid_api_key")
        return key

"""Error types. All client-facing errors use OpenAI's `{"error": {...}}` shape
so existing OpenAI SDKs surface them unchanged."""

from __future__ import annotations


class GatewayError(Exception):
    """An error returned to the client with a specific HTTP status."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: str | None = None, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type_
        self.code = code
        self.headers = headers or {}

    def to_body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


class BackendError(Exception):
    """A backend call failed. `retryable` decides whether the router falls back.

    Retryable = the next backend might succeed (5xx, 429, timeout, connection
    error). Non-retryable = the request itself is bad (most 4xx), so trying
    another provider would only hide the bug and burn latency.
    """

    def __init__(self, backend: str, status: int, message: str, retryable: bool,
                 reason: str | None = None):
        super().__init__(f"{backend}: {status} {message}")
        self.backend = backend
        self.status = status
        self.message = message
        self.retryable = retryable
        # Low-cardinality label for the fallback metric.
        self.reason = reason or ("rate_limited" if status == 429
                                 else "5xx" if status >= 500 else str(status))

    @classmethod
    def timeout(cls, backend: str) -> "BackendError":
        return cls(backend, 504, "timed out", retryable=True, reason="timeout")

    @classmethod
    def connection(cls, backend: str, detail: str) -> "BackendError":
        return cls(backend, 503, f"connection error: {detail}", retryable=True, reason="connection")


def is_retryable_status(status: int) -> bool:
    return status in (408, 409, 429) or status >= 500

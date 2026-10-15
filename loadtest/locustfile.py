"""Locust load test: replay a trace against the gateway, measuring TTFT client-side.

    locust -f loadtest/locustfile.py --host http://127.0.0.1:8080 --headless -u 20 -r 20 -t 60s --csv out/step

Each user sends PER_USER_RPS requests/s (constant throughput), so `-u N` ~= N req/s.
Besides the normal HTTP rows, two custom metric rows are emitted per request so
that Locust's CSV directly contains the percentiles the report needs:
  * TTFT,     name "<backend>|<cache>"  - first content delta, client-side
  * OVERHEAD_US, name "<cache>"         - x-gateway-overhead-ms, recorded in MICROseconds
                                          (Locust stores integer "ms", which would round sub-ms to 0)
Env: TRACE, GATEWAY_KEY, STREAM (1/0), PER_USER_RPS.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time

from locust import HttpUser, constant_throughput, events, task

sys.path.insert(0, os.path.dirname(__file__))
from prompt_trace import load_trace  # noqa: E402

TRACE = load_trace(os.environ.get("TRACE", "loadtest/data/trace.jsonl"))
STREAM = os.environ.get("STREAM", "1") == "1"
KEY = os.environ.get("GATEWAY_KEY", "sk-gw-demo")
_cursor = itertools.count()
_lock = threading.Lock()


def _next_record() -> dict:
    """All users walk the trace in order, so duplicates arrive in trace order."""
    with _lock:
        return TRACE[next(_cursor) % len(TRACE)]


def _metric(kind: str, name: str, ms: float) -> None:
    events.request.fire(request_type=kind, name=name, response_time=ms, response_length=0,
                        exception=None, context={})


class GatewayUser(HttpUser):
    wait_time = constant_throughput(float(os.environ.get("PER_USER_RPS", "1")))

    @task
    def chat(self) -> None:
        rec = _next_record()
        body = {"model": rec["model"], "messages": rec["messages"], "temperature": 0,
                "max_tokens": rec.get("max_tokens", 128), "stream": STREAM}
        headers = {"Authorization": f"Bearer {KEY}"}
        start = time.perf_counter()
        with self.client.post("/v1/chat/completions", json=body, headers=headers, stream=STREAM,
                              catch_response=True, name=f"chat {rec['model']}") as resp:
            if resp.status_code != 200:
                resp.failure(f"{resp.status_code}: {resp.text[:200]}")
                return
            ttft_ms = None
            if STREAM:
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: ") or line == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        resp.failure(f"mid-stream error: {chunk['error'].get('message')}")
                        return
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if ttft_ms is None and (delta.get("content") or delta.get("tool_calls")):
                        ttft_ms = (time.perf_counter() - start) * 1000
            else:
                resp.content  # drain
            resp.success()
        backend = resp.headers.get("x-gateway-backend", "?")
        cache = resp.headers.get("x-gateway-cache", "?")
        if ttft_ms is not None:
            _metric("TTFT", f"{backend}|{cache}", ttft_ms)
        if "x-gateway-overhead-ms" in resp.headers:
            _metric("OVERHEAD_US", cache, float(resp.headers["x-gateway-overhead-ms"]) * 1000)

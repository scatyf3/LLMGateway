"""Prometheus metrics. Labels are kept low-cardinality (key ids, not users;
backend/model from config, not free-form) so the TSDB stays small."""

from prometheus_client import Counter, Histogram, disable_created_metrics

disable_created_metrics()  # drop the *_created series; they double the scrape size for no insight

_LATENCY_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
_OVERHEAD_BUCKETS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)

REQUESTS = Counter("gateway_requests_total", "Requests handled",
                   ["endpoint", "backend", "model", "status", "cache"])
LATENCY = Histogram("gateway_request_latency_seconds", "End-to-end request latency",
                    ["endpoint", "backend", "cache"], buckets=_LATENCY_BUCKETS)
TTFT = Histogram("gateway_ttft_seconds", "Time to first content byte (streaming)",
                 ["backend", "cache"], buckets=_LATENCY_BUCKETS)
OVERHEAD = Histogram("gateway_overhead_seconds", "Time spent in the gateway itself (not waiting on a backend)",
                     ["endpoint", "cache"], buckets=_OVERHEAD_BUCKETS)
SEMANTIC_LOOKUP = Histogram("gateway_semantic_lookup_seconds",
                            "Semantic cache lookup incl. the embedding call", buckets=_LATENCY_BUCKETS)
TOKENS = Counter("gateway_tokens_total", "Tokens processed", ["key", "model", "direction"])
COST = Counter("gateway_cost_usd_total", "Spend in USD", ["key", "model"])
COST_SAVED = Counter("gateway_cost_saved_usd_total", "Spend avoided by cache hits", ["key", "cache"])
CACHE_LOOKUPS = Counter("gateway_cache_lookups_total", "Cache lookups", ["type", "result"])
FALLBACKS = Counter("gateway_fallbacks_total", "Fallback hops (one per failed attempt)",
                    ["alias", "from_backend", "reason"])
FALLBACK_REQUESTS = Counter("gateway_fallback_requests_total",
                            "Requests that needed at least one fallback, by final outcome", ["outcome"])
RATE_LIMITED = Counter("gateway_rate_limited_total", "Requests rejected by limits", ["key", "limit"])

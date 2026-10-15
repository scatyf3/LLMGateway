# Load-test report

**Template: no measurements yet.** Run the load test (README, "Load test") and regenerate with
`python loadtest/report.py --results loadtest/results/<run>`. Never hand-enter numbers.

## Setup

- Hardware / where the gateway ran: TO FILL
- Backends enabled (mock / Anthropic / OpenAI / vLLM + GPU): TO FILL
- Trace: TO FILL
- Fallback experiment: TO FILL

## Throughput and end-to-end latency (client-side, busiest route)

| target req/s | requests | achieved req/s | failures | p50 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
| TO FILL | | | | | |

## Gateway overhead per request (x-gateway-overhead-ms: time not spent waiting on a model)

| req/s | cache | n | p50 ms | p99 ms |
|---:|---|---:|---:|---:|
| TO FILL | | | | |

## TTFT per backend (client-side, streaming)

| req/s | backend | cache | n | p50 ms | p99 ms |
|---:|---|---|---:|---:|---:|
| TO FILL | | | | | |

## Cache hit rate and cost saved

| req/s | chat requests | exact hits | semantic hits | hit rate | $ spent | $ saved | saved / (spent+saved) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| TO FILL | | | | | | | |

Mock-backend prices are pretend numbers (see `config/gateway.yaml`), so $ figures are only meaningful
with real providers. Semantic false hits on this trace: TO FILL (audit `semantic_cache_hit` log lines).

## Fallback when a provider is cut mid-test

| req/s | requests needing fallback | succeeded | success rate | failed attempts by backend |
|---:|---:|---:|---:|---|
| TO FILL | | | | |

## vLLM prefix caching: TTFT on repeated long system prompts

TO FILL: run `loadtest/prefix_cache_experiment.py` against vLLM and pass `--prefix <file>`.

## Takeaways

TO FILL

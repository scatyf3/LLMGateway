"""Turn a run_steps.py results directory into loadtest/REPORT.md tables.

    python loadtest/report.py --results loadtest/results/<run> [--prefix loadtest/results/prefix_cache.md]
    python loadtest/report.py            # no results: (re)write the blank "TO FILL" template

Numbers come only from the files in the results dir; anything not measured
stays "TO FILL". Per-step metrics are deltas between consecutive /metrics scrapes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

HERE = Path(__file__).resolve().parent
TODO = "TO FILL"


def read_metrics(path: Path) -> dict[tuple, float]:
    """{(sample_name, frozenset(labels)): value} for gateway_* samples."""
    out = {}
    if not path.exists():
        return out
    for fam in text_string_to_metric_families(path.read_text(encoding="utf-8")):
        for s in fam.samples:
            if s.name.startswith("gateway_"):
                out[(s.name, frozenset(s.labels.items()))] = s.value
    return out


def total(m: dict, name: str, **match) -> float:
    return sum(v for (n, labels), v in m.items()
               if n == name and all(dict(labels).get(k) == want for k, want in match.items()))


def group(m: dict, name: str, label: str) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for (n, labels), v in m.items():
        if n == name:
            out[dict(labels).get(label, "?")] += v
    return out


def diff(after: dict, before: dict) -> dict:
    return {k: v - before.get(k, 0.0) for k, v in after.items()}


def read_stats(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ms(row: dict | None, col: str, scale: float = 1.0) -> str:
    return f"{float(row[col]) * scale:.{3 if scale < 1 else 1}f}" if row and row.get(col) not in (None, "", "N/A") else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="run_steps.py output dir; omit to write the blank template")
    ap.add_argument("--prefix", help="markdown table written by prefix_cache_experiment.py")
    ap.add_argument("--out", default=str(HERE / "REPORT.md"))
    args = ap.parse_args()
    res = Path(args.results) if args.results else None
    meta = json.loads((res / "meta.json").read_text()) if res and (res / "meta.json").exists() else {}
    rates = [] if res is None else sorted(int(re.match(r"rate_(\d+)_stats\.csv", p.name).group(1)) for p in res.glob("rate_*_stats.csv"))

    load_rows, overhead_rows, ttft_rows, cache_rows, fb_rows = [], [], [], [], []
    prev = read_metrics(res / "baseline.metrics.txt") if res else {}
    for rate in rates:
        stats = read_stats(res / f"rate_{rate}_stats.csv")
        http = [r for r in stats if r["Type"] == "POST"]
        reqs = sum(int(r["Request Count"]) for r in http)
        fails = sum(int(r["Failure Count"]) for r in http)
        rps = sum(float(r["Requests/s"]) for r in http)
        main_http = max(http, key=lambda r: int(r["Request Count"]), default=None)
        load_rows.append(f"| {rate} | {reqs} | {rps:.1f} | {fails} ({fails / reqs:.1%}) | "
                         f"{ms(main_http, '50%')} | {ms(main_http, '99%')} |" if reqs else f"| {rate} | 0 | - | - | - | - |")
        for r in stats:
            if r["Type"] == "OVERHEAD_US":
                overhead_rows.append(f"| {rate} | {r['Name']} | {r['Request Count']} | "
                                     f"{ms(r, '50%', 1e-3)} | {ms(r, '99%', 1e-3)} |")
            if r["Type"] == "TTFT":
                backend, _, cache = r["Name"].partition("|")
                ttft_rows.append(f"| {rate} | {backend} | {cache} | {r['Request Count']} | {ms(r, '50%')} | {ms(r, '99%')} |")

        cur = read_metrics(res / f"rate_{rate}.metrics.txt")
        d = diff(cur, prev)
        prev = cur
        chat = total(d, "gateway_requests_total", endpoint="chat")
        by_cache = group({k: v for k, v in d.items() if dict(k[1]).get("endpoint") == "chat"},
                         "gateway_requests_total", "cache")
        hits = by_cache.get("exact", 0) + by_cache.get("semantic", 0)
        spent = total(d, "gateway_cost_usd_total")
        saved = total(d, "gateway_cost_saved_usd_total")
        cache_rows.append(f"| {rate} | {chat:.0f} | {by_cache.get('exact', 0):.0f} | {by_cache.get('semantic', 0):.0f} | "
                          f"{(hits / chat if chat else 0):.1%} | ${spent:.4f} | ${saved:.4f} | "
                          f"{(saved / (spent + saved) if spent + saved else 0):.1%} |")
        ok = total(d, "gateway_fallback_requests_total", outcome="success")
        bad = total(d, "gateway_fallback_requests_total", outcome="failed")
        hops = group(d, "gateway_fallbacks_total", "from_backend")
        hop_txt = ", ".join(f"{b}: {v:.0f}" for b, v in hops.items() if v) or "-"
        fb_rows.append(f"| {rate} | {ok + bad:.0f} | {ok:.0f} | {(ok / (ok + bad) if ok + bad else 0):.1%} | {hop_txt} |")

    kill = (f"`{meta['kill_backend']}` cut at +{meta['kill_after']}s for {meta['kill_for']}s "
            f"in the {meta.get('kill_at_rate') or (rates[-1] if rates else '?')} req/s step"
            if meta.get("kill_backend") else TODO if res is None else "no provider was cut in this run")
    prefix = Path(args.prefix).read_text(encoding="utf-8") if args.prefix and Path(args.prefix).exists() else \
        f"{TODO}: run `loadtest/prefix_cache_experiment.py` against vLLM and pass `--prefix <file>`."

    def table(header: str, rows: list[str]) -> str:
        cols = header.splitlines()[0].count("|") - 1
        return header + ("\n".join(rows) if rows else f"| {TODO} |" + " |" * (cols - 1))

    source = (f"Generated by `loadtest/report.py` from `{res.as_posix()}`. Every number below comes from that\n"
              "run's Locust CSVs and gateway `/metrics` scrapes; nothing is hand-entered."
              if res else "**Template: no measurements yet.** Run the load test (README, \"Load test\") and "
              "regenerate with\n`python loadtest/report.py --results loadtest/results/<run>`. Never hand-enter numbers.")
    trace = (f"`{meta.get('trace')}`, streaming={bool(meta.get('stream', 1))}, "
             f"step length {meta.get('step_seconds')}s" if meta else TODO)
    report = f"""# Load-test report

{source}

## Setup

- Hardware / where the gateway ran: {TODO}
- Backends enabled (mock / Anthropic / OpenAI / vLLM + GPU): {TODO}
- Trace: {trace}
- Fallback experiment: {kill}

## Throughput and end-to-end latency (client-side, busiest route)

{table('''| target req/s | requests | achieved req/s | failures | p50 ms | p99 ms |
|---:|---:|---:|---:|---:|---:|
''', load_rows)}

## Gateway overhead per request (x-gateway-overhead-ms: time not spent waiting on a model)

{table('''| req/s | cache | n | p50 ms | p99 ms |
|---:|---|---:|---:|---:|
''', overhead_rows)}

## TTFT per backend (client-side, streaming)

{table('''| req/s | backend | cache | n | p50 ms | p99 ms |
|---:|---|---|---:|---:|---:|
''', ttft_rows)}

## Cache hit rate and cost saved

{table('''| req/s | chat requests | exact hits | semantic hits | hit rate | $ spent | $ saved | saved / (spent+saved) |
|---:|---:|---:|---:|---:|---:|---:|---:|
''', cache_rows)}

Mock-backend prices are pretend numbers (see `config/gateway.yaml`), so $ figures are only meaningful
with real providers. Semantic false hits on this trace: {TODO} (audit `semantic_cache_hit` log lines).

## Fallback when a provider is cut mid-test

{table('''| req/s | requests needing fallback | succeeded | success rate | failed attempts by backend |
|---:|---:|---:|---:|---|
''', fb_rows)}

## vLLM prefix caching: TTFT on repeated long system prompts

{prefix}

## Takeaways

{TODO}
"""
    Path(args.out).write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

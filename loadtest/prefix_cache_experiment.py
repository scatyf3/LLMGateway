"""vLLM prefix-caching experiment: TTFT for requests sharing a long system prompt.

Two arms in one run, no server restart needed:
  * shared  - every request has the SAME long system prompt (only the question
              differs). With --enable-prefix-caching the KV blocks for the
              prompt are reused, so requests after the first skip most prefill.
  * unique  - a random nonce at the START of the system prompt makes every
              prefix different, so nothing can be reused (= caching "off").
Prefill cost is ~linear in uncached prompt tokens, so the TTFT gap between arms
is the prefix-cache win. Requests run one at a time to isolate TTFT from queueing.
Optionally rerun against a server started with --no-enable-prefix-caching
(`--label off`) to confirm "shared" then looks like "unique".

    # straight at vLLM
    python loadtest/prefix_cache_experiment.py --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-1.5B-Instruct --n 30 --prefix-words 3000
    # through the gateway (gateway caches bypassed with x-gateway-cache: skip)
    python loadtest/prefix_cache_experiment.py --base-url http://localhost:8080/v1 --model local --api-key sk-gw-demo
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import httpx
import numpy as np

QUESTIONS = ["Summarize section {i}.", "What does section {i} say about caching?",
             "List the key terms in section {i}.", "Is section {i} about latency?"]


def long_prompt(words: int, seed: int = 7) -> str:
    rng = random.Random(seed)
    vocab = ("request token cache batch prefill decode latency throughput memory block "
             "scheduler queue router backend replica gpu kernel attention").split()
    paras = [f"Section {i + 1}. " + " ".join(rng.choice(vocab) for _ in range(50)) + "."
             for i in range(max(1, words // 50))]
    return "You are a documentation assistant. Answer only from the manual below.\n\n" + "\n\n".join(paras)


def ttft_ms(client: httpx.Client, model: str, system: str, question: str) -> float:
    body = {"model": model, "stream": True, "max_tokens": 16, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": question}]}
    start = time.perf_counter()
    first: float | None = None
    with client.stream("POST", "/chat/completions", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():  # read to the end so the connection is reused
            if first is not None or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = (json.loads(line[6:]).get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                first = (time.perf_counter() - start) * 1000
    return first if first is not None else (time.perf_counter() - start) * 1000


def summarize(label: str, xs: list[float]) -> str:
    a = np.asarray(xs)
    return (f"| {label} | {len(a)} | {np.percentile(a, 50):.1f} | {np.percentile(a, 90):.1f} | "
            f"{np.percentile(a, 99):.1f} | {a.mean():.1f} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--prefix-words", type=int, default=3000)
    ap.add_argument("--label", default="on", help="server prefix-caching setting, for the table")
    ap.add_argument("--out", default="loadtest/results/prefix_cache.md")
    args = ap.parse_args()

    headers = {"x-gateway-cache": "skip"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    system = long_prompt(args.prefix_words)
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=120) as c:
        ttft_ms(c, args.model, "warmup", "hi")  # load model / CUDA graphs outside the measurement
        shared = [ttft_ms(c, args.model, system, QUESTIONS[i % 4].format(i=i + 1)) for i in range(args.n)]
        unique = [ttft_ms(c, args.model, f"[session {uuid.uuid4()}]\n" + system, QUESTIONS[i % 4].format(i=i + 1))
                  for i in range(args.n)]

    warm = shared[1:]
    speedup = np.percentile(unique, 50) / np.percentile(warm, 50) if warm else float("nan")
    table = "\n".join([
        f"Server prefix caching: **{args.label}**; model `{args.model}`; ~{args.prefix_words} words of shared "
        f"system prompt; {args.n} sequential requests per arm; max_tokens=16.\n",
        "| arm | n | TTFT p50 ms | p90 ms | p99 ms | mean ms |",
        "|---|---:|---:|---:|---:|---:|",
        f"| shared prefix, first (cold) | 1 | {shared[0]:.1f} | - | - | {shared[0]:.1f} |",
        summarize("shared prefix, repeats (warm)", warm),
        summarize("unique prefix (no reuse possible)", unique),
        f"\nMedian TTFT speedup from prefix reuse (unique p50 / warm p50): **{speedup:.2f}x**",
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

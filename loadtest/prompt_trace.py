"""Build a request trace (JSONL, one OpenAI chat request per line) for the load test.

    # bundled synthetic trace: repeated system prompts, exact + near duplicates
    python loadtest/prompt_trace.py synthetic --n 3000 --out loadtest/data/trace.jsonl
    # real prompts: ShareGPT JSON (e.g. ShareGPT_V3_unfiltered_cleaned_split.json, download yourself)
    python loadtest/prompt_trace.py sharegpt --path ShareGPT_V3.json --n 3000 --out loadtest/data/trace.jsonl

Each line: {"model": alias, "messages": [...], "max_tokens": int, "kind": "unique|dup|near_dup|long"}.
`kind` is ground truth for the synthetic set, so cache hit rates can be sanity-checked.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPTS = [
    "You are a concise assistant. Answer in at most three sentences.",
    "You are a senior Python engineer. Prefer short code examples.",
    "You are a helpful assistant for an internal documentation site. Cite the section you used.",
]
TOPICS = ["KV cache", "continuous batching", "speculative decoding", "paged attention", "quantization",
          "tensor parallelism", "prefix caching", "LoRA adapters", "rate limiting", "consistent hashing",
          "Redis eviction", "HTTP/2 multiplexing", "TCP slow start", "B-trees", "bloom filters",
          "vector databases", "BM25", "cosine similarity", "gradient checkpointing", "FlashAttention"]
TEMPLATES = ["What is {t}?", "Explain {t} to a new engineer.", "What are the trade-offs of {t}?",
             "When should I avoid {t}?", "Give a one-paragraph overview of {t}.", "How does {t} affect latency?",
             "Compare {t} with the most common alternative.", "List three pitfalls of {t}."]
ALIASES = [("fast", 0.7), ("smart", 0.2), ("cheap", 0.1)]


def long_context(rng: random.Random, words: int) -> str:
    """A fixed pseudo-document: stands in for RAG context shared across requests."""
    vocab = [w.lower() for t in TOPICS for w in t.split()] + ["the", "system", "request", "token", "cache"]
    paras = []
    for i in range(words // 60):
        paras.append(f"Section {i + 1}. " + " ".join(rng.choice(vocab) for _ in range(60)) + ".")
    return "\n\n".join(paras)


def _near_dup(text: str, rng: random.Random) -> str:
    variants = [text.lower(), text.rstrip("?."), text + " Please.", "Quick question: " + text, text.replace(" a ", " the ")]
    return rng.choice(variants)


def _pick_alias(rng: random.Random) -> str:
    r, acc = rng.random(), 0.0
    for alias, p in ALIASES:
        acc += p
        if r < acc:
            return alias
    return ALIASES[0][0]


def synthetic(n: int, seed: int = 0, dup_rate: float = 0.15, near_dup_rate: float = 0.10,
              long_rate: float = 0.10, long_words: int = 2000) -> list[dict]:
    rng = random.Random(seed)
    doc_system = SYSTEM_PROMPTS[2] + "\n\nDOCUMENTATION:\n" + long_context(random.Random(42), long_words)
    out: list[dict] = []
    for _ in range(n):
        r = rng.random()
        if out and r < dup_rate:
            out.append({**rng.choice(out), "kind": "dup"})
            continue
        if out and r < dup_rate + near_dup_rate:
            base = rng.choice([x for x in out if x["kind"] != "long"] or out)
            msgs = [*base["messages"][:-1], {"role": "user", "content": _near_dup(base["messages"][-1]["content"], rng)}]
            out.append({**base, "messages": msgs, "kind": "near_dup"})
            continue
        question = rng.choice(TEMPLATES).format(t=rng.choice(TOPICS)) + f" (case {rng.randint(1, 10_000)})"
        if r < dup_rate + near_dup_rate + long_rate:
            # Shared long system prompt: exercises the long-prompt rule + vLLM prefix caching.
            out.append({"model": "fast", "max_tokens": 64, "kind": "long",
                        "messages": [{"role": "system", "content": doc_system}, {"role": "user", "content": question}]})
            continue
        out.append({"model": _pick_alias(rng), "max_tokens": 128, "kind": "unique",
                    "messages": [{"role": "system", "content": rng.choice(SYSTEM_PROMPTS[:2])},
                                 {"role": "user", "content": question}]})
    return out


def sharegpt(path: str, n: int, seed: int = 0, max_chars: int = 8000) -> list[dict]:
    """First human turn of each ShareGPT conversation (natural duplicates included)."""
    rng = random.Random(seed)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rng.shuffle(data)
    out = []
    for conv in data:
        turns = conv.get("conversations") or []
        first = next((t.get("value", "") for t in turns if t.get("from") == "human"), "")
        if not first.strip() or len(first) > max_chars:
            continue
        out.append({"model": _pick_alias(rng), "max_tokens": 128, "kind": "sharegpt",
                    "messages": [{"role": "user", "content": first}]})
        if len(out) >= n:
            break
    return out


def load_trace(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dup-rate", type=float, default=0.15)
    s.add_argument("--near-dup-rate", type=float, default=0.10)
    s.add_argument("--long-rate", type=float, default=0.10)
    g = sub.add_parser("sharegpt")
    g.add_argument("--path", required=True)
    g.add_argument("--n", type=int, default=3000)
    g.add_argument("--seed", type=int, default=0)
    for p in (s, g):
        p.add_argument("--out", default="loadtest/data/trace.jsonl")
    args = ap.parse_args()

    if args.cmd == "synthetic":
        rows = synthetic(args.n, args.seed, args.dup_rate, args.near_dup_rate, args.long_rate)
    else:
        rows = sharegpt(args.path, args.n, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(rows)} requests to {args.out}: {kinds}")


if __name__ == "__main__":
    main()

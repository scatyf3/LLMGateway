"""Tune the semantic-cache similarity threshold on a labeled set of prompt pairs.

Input: JSONL lines {"a": str, "b": str, "same_answer": bool}. A cache hit on a
pair labeled false is a *false hit* - the user gets someone else's answer.

    # with the embedding model the gateway actually uses (through the gateway):
    python scripts/tune_semantic_threshold.py --pairs scripts/data/semantic_pairs_sample.jsonl \
        --embed-url http://localhost:8080/v1 --model embed --api-key sk-gw-demo
    # offline, with the mock hashing embedder (shows why lexical similarity is dangerous):
    python scripts/tune_semantic_threshold.py --pairs scripts/data/semantic_pairs_sample.jsonl --embedder hashing

The bundled sample (40 pairs) is a STARTER set: hand-written, heavy on hard
negatives. Replace or extend it with pairs from your own traffic before
trusting the recommended threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_pairs(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_texts(texts: list[str], args) -> np.ndarray:
    if args.embedder == "hashing":
        from gateway.backends.mock import hashing_embedding
        return np.asarray([hashing_embedding(t) for t in texts], dtype=np.float32)
    import httpx
    vecs = []
    with httpx.Client(base_url=args.embed_url, timeout=60,
                      headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}) as c:
        for i in range(0, len(texts), 64):
            r = c.post("/embeddings", json={"model": args.model, "input": texts[i:i + 64]})
            r.raise_for_status()
            vecs += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
    return np.asarray(vecs, dtype=np.float32)


def sweep(sims: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    rows = []
    for t in thresholds:
        hit = sims >= t
        tp, fp = int((hit & labels).sum()), int((hit & ~labels).sum())
        fn, tn = int((~hit & labels).sum()), int((~hit & ~labels).sum())
        rows.append({
            "threshold": round(float(t), 3), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_hit_rate": fp / (tp + fp) if tp + fp else 0.0,   # share of hits that are wrong
            "fpr": fp / (fp + tn) if fp + tn else 0.0,              # share of negatives that hit
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--embedder", choices=["http", "hashing"], default="http")
    ap.add_argument("--embed-url", default="http://localhost:8080/v1")
    ap.add_argument("--model", default="embed")
    ap.add_argument("--api-key", default="sk-gw-demo")
    ap.add_argument("--max-false-hit-rate", type=float, default=0.0,
                    help="pick the lowest threshold whose false-hit rate is at most this")
    ap.add_argument("--show-pairs", action="store_true", help="print each pair's similarity")
    args = ap.parse_args()

    pairs = load_pairs(args.pairs)
    vecs = embed_texts([p["a"] for p in pairs] + [p["b"] for p in pairs], args)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    a, b = vecs[:len(pairs)], vecs[len(pairs):]
    sims = (a * b).sum(axis=1)
    labels = np.asarray([bool(p["same_answer"]) for p in pairs])

    if args.show_pairs:
        for p, s in sorted(zip(pairs, sims), key=lambda x: -x[1]):
            print(f"{s:.3f}  {'SAME' if p['same_answer'] else 'DIFF'}  {p['a']!r} | {p['b']!r}")
        print()

    rows = sweep(sims, labels, np.arange(0.80, 1.0001, 0.01))
    print(f"{len(pairs)} pairs ({labels.sum()} same / {(~labels).sum()} different), embedder={args.embedder}\n")
    print("| threshold | hits | false hits | precision | recall | false-hit rate | FPR |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['threshold']:.2f} | {r['tp'] + r['fp']} | {r['fp']} | {r['precision']:.3f} | "
              f"{r['recall']:.3f} | {r['false_hit_rate']:.3f} | {r['fpr']:.3f} |")
    ok = [r for r in rows if r["false_hit_rate"] <= args.max_false_hit_rate and r["tp"] > 0]
    if ok:
        best = ok[0]
        print(f"\nLowest threshold with false-hit rate <= {args.max_false_hit_rate}: "
              f"{best['threshold']:.2f} (recall {best['recall']:.3f})")
    else:
        print(f"\nNo threshold reaches false-hit rate <= {args.max_false_hit_rate} with any hits.")


if __name__ == "__main__":
    main()

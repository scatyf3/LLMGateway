"""Run the Locust test at rising request rates, snapshotting gateway metrics per step.

    python loadtest/run_steps.py --host http://127.0.0.1:8080 --rates 5 10 20 40 --step-seconds 60
    # fallback experiment: cut mock-primary 20s into the step at 20 req/s, for 20s
    python loadtest/run_steps.py --rates 20 --kill-backend mock-primary --kill-at-rate 20 \
        --kill-after 20 --kill-for 20

Outputs to --out (default loadtest/results/<timestamp>/):
  rate_<R>_stats.csv            Locust stats (HTTP rows + TTFT/OVERHEAD custom rows)
  rate_<R>.metrics.txt          gateway /metrics scraped right after the step
  baseline.metrics.txt          /metrics before the first step
  usage.json, meta.json         /admin/usage at the end; run parameters
Then: python loadtest/report.py --results <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent


def scrape(host: str, path: Path) -> None:
    path.write_text(httpx.get(f"{host}/metrics", timeout=10).text, encoding="utf-8")


def toggle(host: str, key: str, backend: str, down: bool) -> None:
    r = httpx.post(f"{host}/admin/backends/{backend}", json={"down": down},
                   headers={"Authorization": f"Bearer {key}"}, timeout=10)
    r.raise_for_status()
    print(f"[{time.strftime('%X')}] {backend} down={down}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and each
    # new connection can stall ~2s, which shows up as a fake p99.
    ap.add_argument("--host", default="http://127.0.0.1:8080")
    ap.add_argument("--key", default=os.environ.get("GATEWAY_KEY", "sk-gw-demo"))
    ap.add_argument("--trace", default="loadtest/data/trace.jsonl")
    ap.add_argument("--rates", type=int, nargs="+", default=[5, 10, 20, 40])
    ap.add_argument("--step-seconds", type=int, default=60)
    ap.add_argument("--stream", type=int, default=1)
    ap.add_argument("--kill-backend", help="mock backend to cut mid-test, e.g. mock-primary")
    ap.add_argument("--kill-at-rate", type=int, help="which step to cut it in (default: last)")
    ap.add_argument("--kill-after", type=float, default=20.0, help="seconds into the step")
    ap.add_argument("--kill-for", type=float, default=20.0, help="seconds to keep it down")
    ap.add_argument("--out")
    args = ap.parse_args()

    out = Path(args.out or HERE / "results" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(vars(args) | {"started": time.time()}, indent=2), encoding="utf-8")
    scrape(args.host, out / "baseline.metrics.txt")
    kill_rate = args.kill_at_rate or args.rates[-1]

    for rate in args.rates:
        print(f"=== step: {rate} req/s for {args.step_seconds}s ===", flush=True)
        timers = []
        if args.kill_backend and rate == kill_rate:
            timers = [threading.Timer(args.kill_after, toggle, (args.host, args.key, args.kill_backend, True)),
                      threading.Timer(args.kill_after + args.kill_for, toggle,
                                      (args.host, args.key, args.kill_backend, False))]
            for t in timers:
                t.start()
        env = {**os.environ, "TRACE": args.trace, "GATEWAY_KEY": args.key, "STREAM": str(args.stream),
               "PER_USER_RPS": "1"}
        cmd = [sys.executable, "-m", "locust", "-f", str(HERE / "locustfile.py"), "--host", args.host,
               "--headless", "-u", str(rate), "-r", str(rate), "-t", f"{args.step_seconds}s",
               "--csv", str(out / f"rate_{rate}"), "--only-summary"]
        subprocess.run(cmd, env=env, check=False)
        for t in timers:
            t.join()
        scrape(args.host, out / f"rate_{rate}.metrics.txt")

    usage = httpx.get(f"{args.host}/admin/usage", headers={"Authorization": f"Bearer {args.key}"}, timeout=10)
    (out / "usage.json").write_text(usage.text, encoding="utf-8")
    print(f"\nresults in {out}\nnext: python loadtest/report.py --results {out}")


if __name__ == "__main__":
    main()

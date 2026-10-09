"""Measure single-request latency of a running API server.

Usage (with the server running):
    python scripts/benchmark.py
    python scripts/benchmark.py --url http://127.0.0.1:8000/evaluate --requests 500
"""

import argparse
import json
import statistics
import time
import urllib.request

SAMPLE_TEXTS = [
    "I love this!",
    "This is the worst day ever",
    "just got home from work, so tired",
    "@friend thanks for the birthday wishes, you made my day",
    "can't believe the game got cancelled again",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000/evaluate")
    parser.add_argument("--requests", type=int, default=300)
    args = parser.parse_args()

    latencies_ms = []
    for i in range(args.requests):
        body = json.dumps({"text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]}).encode()
        request = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request) as response:
            response.read()
        latencies_ms.append((time.perf_counter() - started) * 1000)

    quantiles = statistics.quantiles(latencies_ms, n=100)
    print(f"{args.requests} sequential requests to {args.url}")
    print(f"p50 = {quantiles[49]:.1f} ms, p95 = {quantiles[94]:.1f} ms, max = {max(latencies_ms):.1f} ms")


if __name__ == "__main__":
    main()

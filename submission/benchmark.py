"""Measure cold initialization and single-turn response latency locally."""

from __future__ import annotations

import argparse
import json
import statistics
import time

try:
    from .agent import Agent
except ImportError:
    from agent import Agent


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")

    started = time.perf_counter()
    agent = Agent(args.catalog)
    initialization_seconds = time.perf_counter() - started

    latencies: list[float] = []
    for index in range(args.samples):
        session_id = f"latency_{index}"
        agent.reset(session_id, {"preference_tags": ["comfort", "durability"]})
        started = time.perf_counter()
        agent.respond(
            session_id,
            "I'm looking for Women Dresses, but I'm still exploring.",
            1,
            10,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)

    print(json.dumps({
        "samples": len(latencies),
        "initialization_seconds": round(initialization_seconds, 6),
        "turn_latency_ms": {
            "mean": round(statistics.fmean(latencies), 6),
            "p50": round(statistics.median(latencies), 6),
            "p95": round(percentile(latencies, .95), 6),
            "maximum": round(max(latencies), 6),
        },
    }, indent=2))


if __name__ == "__main__":
    main()

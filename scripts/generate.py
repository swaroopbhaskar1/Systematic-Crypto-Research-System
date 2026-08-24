"""Generate hypothesis JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from cq.ai.generate import generate_hypotheses


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hypotheses")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("research/hypotheses"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60.0) as client:
        hypotheses = generate_hypotheses(
            client=client,
            calls=args.calls,
            batch_size=args.batch_size,
            dropped_log=Path("research/log/dropped.jsonl"),
        )
    for hypothesis in hypotheses:
        path = args.output / f"{hypothesis.id}.json"
        path.write_text(hypothesis.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

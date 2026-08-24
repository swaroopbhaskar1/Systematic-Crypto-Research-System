"""Critique hypothesis JSON files."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from cq.ai.critique import critique_hypothesis
from cq.research.schema import Hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description="Critique hypotheses")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    hypothesis = Hypothesis.model_validate_json(args.path.read_text(encoding="utf-8"))
    with httpx.Client(timeout=60.0) as client:
        verdict = critique_hypothesis(hypothesis, client=client)
    out = args.path.with_suffix(".verdict.json")
    out.write_text(verdict.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

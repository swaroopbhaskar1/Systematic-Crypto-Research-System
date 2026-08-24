"""Run a compiled hypothesis through the backtester."""

from __future__ import annotations

import argparse
from pathlib import Path

from cq.backtest.engine import run
from cq.grammar.compile import compile_signal
from cq.research.schema import Hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one backtest")
    parser.add_argument("hypothesis", type=Path)
    args = parser.parse_args()
    hypothesis = Hypothesis.model_validate_json(
        args.hypothesis.read_text(encoding="utf-8")
    )
    signal = compile_signal(hypothesis)
    raise SystemExit(
        "panel loading is a data-layer concern; pass a prepared panel via cq.backtest.engine.run "
        f"for {hypothesis.id} compiled={signal.__class__.__name__}"
    )


if __name__ == "__main__":
    main()

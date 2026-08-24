"""Thin CLI entrypoint: backtest one hypothesis against the Parquet store."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from cq.backtest.pipeline import (
    DEFAULT_TIMEFRAME,
    BacktestRequest,
    default_asof,
    run_from_store,
)
from cq.data.panel import MarketType
from cq.research.log import DEFAULT_LOG_PATH
from cq.research.report import render_backtest_result

DEFAULT_COSTS = Path("config") / "costs.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one backtest")
    parser.add_argument("hypothesis", type=Path)
    parser.add_argument("--store", type=Path, default=Path("data") / "panel")
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--market", choices=("spot", "perp"), default="perp")
    parser.add_argument("--asof", type=int, default=None)
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--count-log", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="skip the counting log; every real test must be counted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = BacktestRequest(
        hypothesis_path=args.hypothesis,
        store_root=args.store,
        costs_path=args.costs,
        timeframe=args.timeframe,
        market_type=cast(MarketType, args.market),
        asof=args.asof if args.asof is not None else default_asof(),
        starting_equity=args.equity,
        count_log=None if args.no_count else args.count_log,
    )
    hypothesis, result = run_from_store(request)
    print(render_backtest_result(result, hypothesis_id=hypothesis.id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Render a counting-aware research report."""

from __future__ import annotations

import argparse
from pathlib import Path

from cq.research.log import test_count
from cq.research.report import (
    ResearchReport,
    adjusted_sharpe_threshold,
    render_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the counting report")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--gross-sharpe", type=float, default=None)
    parser.add_argument("--net-sharpe", type=float, default=None)
    args = parser.parse_args()
    count = test_count(args.log)
    if count < 1:
        raise SystemExit("counting log is empty")
    report = ResearchReport(
        test_count=count,
        naive_sharpe_threshold=1.0,
        adjusted_sharpe_threshold=adjusted_sharpe_threshold(count),
        candidates_above_adjusted=0,
        gross_sharpe=args.gross_sharpe,
        net_sharpe=args.net_sharpe,
    )
    print(render_report(report))


if __name__ == "__main__":
    main()

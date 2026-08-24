"""Store-to-result orchestration for the backtest entrypoint.

The repository layout requires CLI scripts to hold no logic, so the wiring
that turns a hypothesis file and a Parquet store into a
:class:`~cq.backtest.engine.BacktestResult` lives here rather than in
``scripts/``.  This module is deliberately the only place that performs I/O
on behalf of a backtest: the engine, cost model, and metrics layers stay
pure and therefore stay testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cq.backtest.costs import CostConfig
from cq.backtest.engine import BacktestResult, run
from cq.data.panel import MarketType, Panel, add_execution_features
from cq.data.store import ParquetStore
from cq.grammar.compile import compile_signal
from cq.research.log import record
from cq.research.schema import Hypothesis
from cq.research.splits import assert_research_timestamps

DEFAULT_TIMEFRAME = "1d"


@dataclass(frozen=True)
class BacktestRequest:
    """Everything one backtest run needs, resolved from the command line."""

    hypothesis_path: Path
    store_root: Path
    costs_path: Path
    timeframe: str
    market_type: MarketType
    asof: int
    starting_equity: float
    count_log: Path | None


def load_panel(request: BacktestRequest) -> Panel:
    """Read the store as of a timestamp and derive execution features.

    The as-of filter is what makes this point-in-time: only revisions that
    had been published by ``asof`` are visible, so a later correction to a
    historical bar cannot leak into a backtest dated before the correction.
    """

    frame = ParquetStore(request.store_root).read(
        timeframe=request.timeframe,
        query_ts=request.asof,
    )
    if frame.empty:
        raise ValueError(
            f"no data in {request.store_root} for timeframe {request.timeframe!r} "
            f"as of {request.asof}"
        )
    scoped = frame.loc[frame["market_type"] == request.market_type].copy()
    if scoped.empty:
        raise ValueError(f"store contains no {request.market_type} rows")
    featured = add_execution_features(scoped)
    if not bool(featured["in_universe"].any()):
        raise ValueError(
            "no bar has enough trailing history to model execution; "
            "ingest more history before backtesting"
        )
    return Panel.from_long(
        featured.drop(columns=["asof"]),
        market_type=request.market_type,
    )


def run_from_store(request: BacktestRequest) -> tuple[Hypothesis, BacktestResult]:
    """Compile, backtest, and count one hypothesis against stored data.

    The counting log is written before the result is returned, so a run
    cannot be inspected and then quietly left uncounted.  Recording raises on
    a duplicate rule identity, which is the intended behavior: retesting the
    same rules under a new name must not buy another look at the data.
    """

    hypothesis = Hypothesis.model_validate_json(
        request.hypothesis_path.read_text(encoding="utf-8")
    )
    panel = load_panel(request)
    assert_research_timestamps(panel.field("close").index)
    cost_model = CostConfig.from_yaml(request.costs_path).cost_model(
        request.market_type
    )
    result = run(
        panel,
        compile_signal(hypothesis),
        starting_equity=request.starting_equity,
        cost_model=cost_model,
        hypothesis_id=hypothesis.id,
    )
    if request.count_log is not None:
        record(hypothesis, result, "tested", path=request.count_log)
    return hypothesis, result


def default_asof() -> int:
    """Return the current time in Unix milliseconds."""
    return int(pd.Timestamp.utcnow().value // 1_000_000)

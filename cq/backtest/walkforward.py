"""Rolling walk-forward evaluation that cannot touch holdout data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cq.backtest.costs import CostModel
from cq.backtest.engine import BacktestResult, Signal, run
from cq.backtest.metrics import annualized_sharpe
from cq.data.panel import Panel
from cq.research.splits import (
    DEV_END,
    WALKFWD_END,
    as_utc,
    as_utc_ms,
    assert_research_timestamps,
)

TRAIN_MONTHS = 12
TEST_MONTHS = 3
STEP_MONTHS = 3


@dataclass(frozen=True)
class WalkForwardWindow:
    """One rolling train/test split that ends before the holdout."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WindowBacktest:
    """Full-window engine output plus the test-period equity slice."""

    window: WalkForwardWindow
    result: BacktestResult
    equity: pd.Series
    test_equity: pd.Series
    test_gross_equity: pd.Series


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-window metrics plus headline and leave-best-out Sharpe."""

    windows: tuple[WalkForwardWindow, ...]
    window_results: tuple[WindowBacktest, ...]
    window_metrics: tuple[dict[str, dict[str, float]], ...]
    headline_sharpe_gross: float
    headline_sharpe_net: float
    sharpe_ex_best_window: float
    sharpe_dispersion: float


def generate_windows(
    dev_end: pd.Timestamp = DEV_END,
    walkfwd_end: pd.Timestamp = WALKFWD_END,
) -> tuple[WalkForwardWindow, ...]:
    """Emit 12-month train / 3-month test windows stepping by 3 months."""
    cursor = as_utc(dev_end) + pd.Timedelta(days=1)
    end = as_utc(walkfwd_end)
    windows: list[WalkForwardWindow] = []
    while True:
        test_end = cursor + pd.DateOffset(months=TEST_MONTHS) - pd.Timedelta(days=1)
        test_end = as_utc(pd.Timestamp(test_end))
        if test_end > end:
            break
        train_end = cursor - pd.Timedelta(days=1)
        train_start = as_utc(
            pd.Timestamp(train_end - pd.DateOffset(months=TRAIN_MONTHS) + pd.Timedelta(days=1))
        )
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=cursor,
                test_end=test_end,
            )
        )
        cursor = as_utc(pd.Timestamp(cursor + pd.DateOffset(months=STEP_MONTHS)))
    if not windows:
        raise ValueError("no walk-forward windows fit in the requested span")
    return tuple(windows)


def walk_forward(
    panel: Panel,
    signal: Signal,
    *,
    cost_model: CostModel | None = None,
    starting_equity: float = 100_000.0,
) -> WalkForwardResult:
    """Run the same signal on successive OOS windows, never the holdout."""
    assert_research_timestamps(panel.field("close").index)
    windows = generate_windows()
    window_results = tuple(
        _run_window(
            panel,
            signal,
            window,
            cost_model=cost_model,
            starting_equity=starting_equity,
        )
        for window in windows
    )
    metrics = tuple(_window_metrics(item) for item in window_results)
    net_sharpes = tuple(row["net"]["sharpe"] for row in metrics)
    return WalkForwardResult(
        windows=windows,
        window_results=window_results,
        window_metrics=metrics,
        headline_sharpe_gross=_concatenated_sharpe(window_results, gross=True),
        headline_sharpe_net=_concatenated_sharpe(window_results, gross=False),
        sharpe_ex_best_window=_sharpe_excluding_best(window_results, net_sharpes),
        sharpe_dispersion=_sample_std(net_sharpes),
    )


def _run_window(
    panel: Panel,
    signal: Signal,
    window: WalkForwardWindow,
    *,
    cost_model: CostModel | None,
    starting_equity: float,
) -> WindowBacktest:
    sliced = panel.slice(as_utc_ms(window.train_start), as_utc_ms(window.test_end))
    result = run(
        sliced,
        signal,
        starting_equity=starting_equity,
        cost_model=cost_model,
    )
    test_start = as_utc_ms(window.test_start)
    test_end = as_utc_ms(window.test_end)
    test_equity = result.equity.loc[
        (result.equity.index >= test_start) & (result.equity.index <= test_end)
    ]
    test_gross = result.gross_equity.loc[
        (result.gross_equity.index >= test_start)
        & (result.gross_equity.index <= test_end)
    ]
    if test_equity.empty:
        raise ValueError("walk-forward window has no test observations")
    return WindowBacktest(
        window=window,
        result=result,
        equity=result.equity,
        test_equity=test_equity,
        test_gross_equity=test_gross,
    )


def _window_metrics(item: WindowBacktest) -> dict[str, dict[str, float]]:
    return {
        "gross": {"sharpe": _equity_sharpe(item.test_gross_equity)},
        "net": {"sharpe": _equity_sharpe(item.test_equity)},
    }


def _equity_sharpe(equity: pd.Series) -> float:
    returns = equity.pct_change(fill_method=None).iloc[1:]
    return annualized_sharpe(float(value) for value in returns)


def _period_returns(item: WindowBacktest, *, gross: bool) -> tuple[float, ...]:
    equity = item.test_gross_equity if gross else item.test_equity
    returns = equity.pct_change(fill_method=None).iloc[1:]
    return tuple(float(value) for value in returns)


def _concatenated_sharpe(
    windows: tuple[WindowBacktest, ...],
    *,
    gross: bool,
) -> float:
    values = [value for item in windows for value in _period_returns(item, gross=gross)]
    return annualized_sharpe(values)


def _sharpe_excluding_best(
    windows: tuple[WindowBacktest, ...],
    sharpes: tuple[float, ...],
) -> float:
    if len(windows) < 2:
        raise ValueError("sharpe_ex_best_window requires at least two windows")
    best = max(range(len(sharpes)), key=lambda index: sharpes[index])
    retained = tuple(
        item for index, item in enumerate(windows) if index != best
    )
    return _concatenated_sharpe(retained, gross=False)


def _sample_std(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        raise ValueError("dispersion requires at least two windows")
    mean = sum(values) / len(values)
    squared = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return squared**0.5

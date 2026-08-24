"""Walk-forward windows must not leak holdout or future test data."""

import pandas as pd
import pytest
from fixtures.research import DEV_END, WALKFWD_END, make_split_panel, utc_ms
from fixtures.signals import ConstantWeightSignal

from cq.backtest.costs import CostModel
from cq.backtest.engine import DEFAULT_SPREAD_BPS_BY_DECILE
from cq.research.holdout import HoldoutLockedError


FRICTIONLESS = CostModel(
    taker_bps=0.0,
    impact_coefficient=0.0,
    spread_bps_by_decile=DEFAULT_SPREAD_BPS_BY_DECILE,
)


def test_generated_windows_are_12m_train_3m_test_step_3m() -> None:
    from cq.backtest.walkforward import generate_windows

    windows = generate_windows(DEV_END, WALKFWD_END)
    assert [window.test_start for window in windows] == [
        pd.Timestamp("2024-07-01", tz="UTC"),
        pd.Timestamp("2024-10-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
    ]
    assert [window.test_end for window in windows] == [
        pd.Timestamp("2024-09-30", tz="UTC"),
        pd.Timestamp("2024-12-31", tz="UTC"),
        pd.Timestamp("2025-03-31", tz="UTC"),
        WALKFWD_END,
    ]
    first = windows[0]
    assert first.train_start == pd.Timestamp("2023-07-01", tz="UTC")
    assert first.train_end == DEV_END
    assert all(window.train_end < window.test_start for window in windows)


def test_windows_never_enter_holdout() -> None:
    from cq.backtest.walkforward import generate_windows

    windows = generate_windows(DEV_END, WALKFWD_END)
    holdout_start = WALKFWD_END + pd.Timedelta(days=1)
    assert all(window.test_end < holdout_start for window in windows)


def test_walkforward_refuses_panel_timestamps_after_walkfwd_end() -> None:
    from cq.backtest.walkforward import walk_forward

    panel = make_split_panel(end="2025-12-31")
    with pytest.raises(HoldoutLockedError):
        walk_forward(panel, ConstantWeightSignal(0.5), cost_model=FRICTIONLESS)


def test_each_window_backtest_uses_only_data_through_test_end() -> None:
    from cq.backtest.walkforward import generate_windows, walk_forward

    panel = make_split_panel(end="2025-06-30")
    result = walk_forward(
        panel,
        ConstantWeightSignal(0.5),
        cost_model=FRICTIONLESS,
        starting_equity=1_000.0,
    )
    windows = generate_windows(DEV_END, WALKFWD_END)
    assert len(result.window_results) == len(windows)
    for window, window_result in zip(windows, result.window_results, strict=True):
        equity_index = window_result.equity.index
        assert int(equity_index.min()) >= utc_ms(window.train_start)
        assert int(equity_index.max()) <= utc_ms(window.test_end)
        metric_index = window_result.test_equity.index
        assert int(metric_index.min()) >= utc_ms(window.test_start)
        assert int(metric_index.max()) <= utc_ms(window.test_end)


def test_walkforward_reports_dispersion_and_sharpe_ex_best_window() -> None:
    from cq.backtest.walkforward import walk_forward

    panel = make_split_panel(end="2025-06-30")
    result = walk_forward(
        panel,
        ConstantWeightSignal(0.5),
        cost_model=FRICTIONLESS,
        starting_equity=1_000.0,
    )
    assert result.headline_sharpe_gross is not None
    assert result.headline_sharpe_net is not None
    assert result.sharpe_ex_best_window is not None
    assert result.sharpe_dispersion >= 0.0
    assert "gross" in result.window_metrics[0]
    assert "net" in result.window_metrics[0]
    sharpes = [row["net"]["sharpe"] for row in result.window_metrics]
    assert len(sharpes) == 4

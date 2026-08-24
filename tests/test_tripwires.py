"""Mechanical tripwires have no discretionary override path."""

import pandas as pd
import pytest

from cq.monitor.tripwires import Action, evaluate


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index)


def test_halve_when_rolling_90d_sharpe_is_below_half_oos() -> None:
    live = _series([0.001] * 80 + [-0.02] * 10)
    assert evaluate("s1", live, oos_sharpe=2.0) is Action.HALVE


def test_cut_after_a_further_90d_below_threshold() -> None:
    live = _series([-0.02] * 180)
    assert evaluate("s1", live, oos_sharpe=2.0, prior_halve_bars=90) is Action.CUT


def test_halt_on_slippage_or_reconciliation_mismatch() -> None:
    live = _series([0.01] * 120)
    assert (
        evaluate(
            "s1",
            live,
            oos_sharpe=1.0,
            slippage_ratio_30d=1.6,
        )
        is Action.HALT
    )
    assert (
        evaluate(
            "s1",
            live,
            oos_sharpe=1.0,
            reconciliation_mismatch=True,
        )
        is Action.HALT
    )


def test_no_override_argument_exists() -> None:
    import inspect

    parameters = inspect.signature(evaluate).parameters
    assert "override" not in parameters
    assert "discretion" not in parameters

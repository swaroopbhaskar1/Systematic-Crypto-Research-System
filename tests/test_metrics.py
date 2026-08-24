import math

import pandas as pd
import pytest

from cq.backtest.metrics import (
    annualized_sharpe,
    annualized_sortino,
    average_holding_period,
    calmar_ratio,
    cost_drag_percent,
    gross_net_returns,
    gross_return,
    group_values_by_regime,
    hit_rate,
    max_drawdown,
    metric_by_regime,
    net_return,
    turnover,
)


def test_daily_risk_metrics_have_known_answers() -> None:
    returns = [0.01, -0.02, 0.03, -0.01]
    mean = 0.0025
    sample_deviation = math.sqrt(sum((value - mean) ** 2 for value in returns) / 3)
    downside_deviation = math.sqrt((0.02**2 + 0.01**2) / 4)

    assert annualized_sharpe(returns) == pytest.approx(
        math.sqrt(365.0) * mean / sample_deviation
    )
    assert annualized_sortino(returns) == pytest.approx(
        math.sqrt(365.0) * mean / downside_deviation
    )


def test_drawdown_and_calmar_have_known_answers() -> None:
    equity = [100.0, 120.0, 90.0, 108.0, 80.0]
    annualized_return = (80.0 / 100.0) ** (365.0 / 4.0) - 1.0

    assert max_drawdown(equity) == pytest.approx(1.0 - 80.0 / 120.0)
    assert calmar_ratio(equity) == pytest.approx(
        annualized_return / (1.0 - 80.0 / 120.0)
    )


def test_turnover_is_average_daily_one_way_weight_change() -> None:
    weights = pd.DataFrame(
        [[0.5, -0.5], [0.25, -0.25], [-0.25, 0.25]],
        columns=["long", "short"],
    )

    assert turnover(weights) == pytest.approx((0.25 + 0.5) / 2.0)


def test_trade_metrics_have_known_answers() -> None:
    entries = pd.to_datetime(["2024-01-01", "2024-01-05"], utc=True)
    exits = pd.to_datetime(["2024-01-04", "2024-01-13"], utc=True)

    assert hit_rate([0.10, 0.0, -0.20, 0.05]) == pytest.approx(0.5)
    assert average_holding_period(entries, exits) == pytest.approx(5.5)
    assert average_holding_period([0], [86_400_000]) == pytest.approx(1.0)


def test_gross_and_net_are_reported_together_with_cost_drag() -> None:
    gross_equity = [100.0, 110.0]
    net_equity = [100.0, 108.0]

    assert gross_return(gross_equity) == pytest.approx(0.10)
    assert net_return(net_equity) == pytest.approx(0.08)
    comparison = gross_net_returns(gross_equity, net_equity)
    assert comparison.gross == pytest.approx(0.10)
    assert comparison.net == pytest.approx(0.08)
    assert comparison.cost_drag_percent == pytest.approx(20.0)
    assert cost_drag_percent(comparison.gross, comparison.net) == pytest.approx(20.0)
    assert cost_drag_percent(-0.10, -0.12) == pytest.approx(-20.0)


def test_regime_helpers_preserve_alignment_and_show_gross_and_net() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    gross = pd.Series([0.01, -0.02, 0.03, 0.04], index=index)
    net = pd.Series([0.005, -0.025, 0.02, 0.03], index=index)
    regimes = pd.Series(["bull", "bear", "bull", "bull"], index=index)

    assert group_values_by_regime(gross, regimes) == {
        "bull": (0.01, 0.03, 0.04),
        "bear": (-0.02,),
    }
    grouped = metric_by_regime(gross, net, regimes, hit_rate)
    assert list(grouped.index) == ["bull", "bear"]
    assert list(grouped.columns) == ["gross", "net"]
    assert grouped.loc["bull", "gross"] == 1.0
    assert grouped.loc["bull", "net"] == 1.0
    assert grouped.loc["bear", "gross"] == 0.0
    assert grouped.loc["bear", "net"] == 0.0


@pytest.mark.parametrize(
    "function, values",
    [
        (annualized_sharpe, []),
        (annualized_sharpe, [0.01]),
        (annualized_sortino, []),
        (max_drawdown, []),
        (calmar_ratio, [100.0]),
        (gross_return, []),
        (net_return, [100.0]),
        (hit_rate, []),
    ],
)
def test_empty_or_insufficient_inputs_fail_loudly(
    function: object, values: list[float]
) -> None:
    with pytest.raises(ValueError):
        function(values)  # type: ignore[operator]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_numeric_inputs_fail_loudly(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        annualized_sharpe([0.01, value])
    with pytest.raises(ValueError, match="finite"):
        max_drawdown([100.0, value])
    with pytest.raises(ValueError, match="finite"):
        hit_rate([value])


def test_undefined_denominators_and_no_trades_fail_loudly() -> None:
    with pytest.raises(ValueError, match="variance"):
        annualized_sharpe([0.01, 0.01])
    with pytest.raises(ValueError, match="downside"):
        annualized_sortino([0.01, 0.02])
    with pytest.raises(ValueError, match="drawdown"):
        calmar_ratio([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="gross return"):
        cost_drag_percent(0.0, -0.01)
    with pytest.raises(ValueError, match="trades"):
        hit_rate([])
    with pytest.raises(ValueError, match="trades"):
        average_holding_period([], [])


def test_turnover_and_regime_helpers_reject_invalid_shapes_or_alignment() -> None:
    with pytest.raises(ValueError, match="observations"):
        turnover(pd.DataFrame([[0.5, -0.5]]))
    with pytest.raises(ValueError, match="finite"):
        turnover(pd.DataFrame([[0.5], [float("nan")]]))

    values = pd.Series([0.01, 0.02], index=[0, 1])
    misaligned = pd.Series(["bull", "bear"], index=[1, 2])
    with pytest.raises(ValueError, match="index"):
        group_values_by_regime(values, misaligned)


def test_holding_period_rejects_unclosed_or_reversed_trades() -> None:
    entry = pd.to_datetime(["2024-01-02"], utc=True)
    with pytest.raises(ValueError, match="same number"):
        average_holding_period(entry, [])
    with pytest.raises(ValueError, match="exit"):
        average_holding_period(
            entry,
            pd.to_datetime(["2024-01-01"], utc=True),
        )

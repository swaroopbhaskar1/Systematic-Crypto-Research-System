"""Deterministic performance metrics for daily crypto backtests.

All numeric inputs must be finite.  Undefined statistics raise ``ValueError``
instead of silently dropping observations or substituting zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

ANNUALIZATION_DAYS = 365


@dataclass(frozen=True)
class GrossNetReturns:
    """Total gross and net returns, reported under one explicit contract."""

    gross: float
    net: float
    cost_drag_percent: float


def _finite_values(
    values: Iterable[float],
    *,
    name: str,
    minimum: int,
) -> tuple[float, ...]:
    observed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain real numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        observed.append(number)
    if len(observed) < minimum:
        raise ValueError(f"{name} requires at least {minimum} observations")
    return tuple(observed)


def _mean(values: tuple[float, ...]) -> float:
    return math.fsum(values) / len(values)


def annualized_sharpe(returns: Iterable[float]) -> float:
    """Return the zero-risk-free Sharpe ratio using sample volatility.

    Daily observations are annualized by ``sqrt(365)``.  At least two returns
    and nonzero sample variance are required.
    """

    observed = _finite_values(returns, name="returns", minimum=2)
    average = _mean(observed)
    variance = math.fsum((value - average) ** 2 for value in observed) / (
        len(observed) - 1
    )
    if variance == 0.0:
        raise ValueError("Sharpe ratio is undefined for zero variance")
    return math.sqrt(ANNUALIZATION_DAYS) * average / math.sqrt(variance)


def annualized_sortino(returns: Iterable[float]) -> float:
    """Return Sortino using a zero target and population downside deviation.

    The downside deviation denominator is
    ``sqrt(mean(min(return, 0) ** 2))`` over every observation.  A series with
    no negative return has zero downside and therefore no finite Sortino.
    """

    observed = _finite_values(returns, name="returns", minimum=1)
    downside_variance = math.fsum(min(value, 0.0) ** 2 for value in observed) / len(
        observed
    )
    if downside_variance == 0.0:
        raise ValueError("Sortino ratio is undefined with zero downside")
    return (
        math.sqrt(ANNUALIZATION_DAYS) * _mean(observed) / math.sqrt(downside_variance)
    )


def _equity_values(
    equity: Iterable[float],
    *,
    minimum: int,
) -> tuple[float, ...]:
    observed = _finite_values(equity, name="equity", minimum=minimum)
    if any(value <= 0.0 for value in observed):
        raise ValueError("equity must be strictly positive")
    return observed


def max_drawdown(equity: Iterable[float]) -> float:
    """Return peak-to-trough maximum drawdown as a nonnegative fraction."""

    observed = _equity_values(equity, minimum=1)
    peak = observed[0]
    largest = 0.0
    for value in observed[1:]:
        peak = max(peak, value)
        largest = max(largest, 1.0 - value / peak)
    return largest


def calmar_ratio(equity: Iterable[float]) -> float:
    """Return annualized compounded return divided by maximum drawdown.

    Each adjacent equity pair represents one daily return, so ``n`` equity
    observations span ``n - 1`` days.  A path with no drawdown has no finite
    Calmar ratio and raises ``ValueError``.
    """

    observed = _equity_values(equity, minimum=2)
    drawdown = max_drawdown(observed)
    if drawdown == 0.0:
        raise ValueError("Calmar ratio is undefined with zero drawdown")
    try:
        annualized = (observed[-1] / observed[0]) ** (
            ANNUALIZATION_DAYS / (len(observed) - 1)
        ) - 1.0
    except OverflowError as error:
        raise ValueError("annualized return is not finite") from error
    if not math.isfinite(annualized):
        raise ValueError("annualized return is not finite")
    return annualized / drawdown


def turnover(weights: pd.DataFrame) -> float:
    """Return average daily one-way turnover from portfolio weights.

    The first row establishes the initial portfolio and is not counted as a
    rebalance.  Each later row contributes half the sum of absolute changes.
    """

    if len(weights.index) < 2:
        raise ValueError("turnover requires at least 2 observations")
    if len(weights.columns) == 0:
        raise ValueError("turnover requires at least one asset")
    if any(
        is_bool_dtype(dtype) or not is_numeric_dtype(dtype) for dtype in weights.dtypes
    ):
        raise ValueError("weights must contain real numbers")
    numeric = weights.to_numpy(dtype=float, copy=True)
    if not bool(pd.notna(numeric).all()) or not all(
        math.isfinite(float(value)) for value in numeric.flat
    ):
        raise ValueError("weights must contain only finite values")
    changes = weights.diff().iloc[1:].abs().sum(axis="columns") / 2.0
    result = float(changes.mean())
    if not math.isfinite(result):
        raise ValueError("turnover is not finite")
    return result


def hit_rate(trade_returns: Iterable[float]) -> float:
    """Return winning trades divided by all trades; zero is not a win."""

    observed = _finite_values(trade_returns, name="trades", minimum=1)
    return sum(value > 0.0 for value in observed) / len(observed)


def average_holding_period(
    entry_times: Iterable[datetime | pd.Timestamp | Integral],
    exit_times: Iterable[datetime | pd.Timestamp | Integral],
) -> float:
    """Return the arithmetic mean completed-trade holding period in days.

    Integral timestamps use the backtest layer's Unix-millisecond convention.
    """

    entries = tuple(_timestamp(value) for value in entry_times)
    exits = tuple(_timestamp(value) for value in exit_times)
    if not entries and not exits:
        raise ValueError("average holding period requires completed trades")
    if len(entries) != len(exits):
        raise ValueError("entry and exit times must have the same number of trades")

    durations: list[float] = []
    for entry, exit_ in zip(entries, exits, strict=True):
        if pd.isna(entry) or pd.isna(exit_):
            raise ValueError("entry and exit times must be valid timestamps")
        try:
            duration = (exit_ - entry).total_seconds() / 86_400.0
        except TypeError as error:
            raise ValueError(
                "entry and exit timestamps must use compatible timezones"
            ) from error
        if duration < 0.0:
            raise ValueError("exit time must not precede entry time")
        durations.append(duration)
    return math.fsum(durations) / len(durations)


def _timestamp(value: object) -> pd.Timestamp:
    if isinstance(value, bool):
        raise TypeError("timestamps must be datetimes or Unix milliseconds")
    if isinstance(value, Integral):
        return pd.Timestamp(int(value), unit="ms", tz="UTC")
    if not isinstance(value, (datetime, pd.Timestamp)):
        raise TypeError("timestamps must be datetimes or Unix milliseconds")
    return pd.Timestamp(value)


def _total_return(equity: tuple[float, ...]) -> float:
    return equity[-1] / equity[0] - 1.0


def gross_return(gross_equity: Iterable[float]) -> float:
    """Return the total frictionless return from a gross equity path."""

    return _total_return(_equity_values(gross_equity, minimum=2))


def net_return(net_equity: Iterable[float]) -> float:
    """Return the total realized return from a net equity path."""

    return _total_return(_equity_values(net_equity, minimum=2))


def cost_drag_percent(gross: float, net: float) -> float:
    """Return ``(gross - net) / gross * 100``.

    The signed denominator reports the literal percentage of gross return, so
    a cost reduction to an already negative return is negative.  Zero gross
    return is undefined.
    """

    observed = _finite_values(
        (gross, net),
        name="gross and net returns",
        minimum=2,
    )
    gross_value, net_value = observed
    if gross_value == 0.0:
        raise ValueError("cost drag is undefined for zero gross return")
    return (gross_value - net_value) / gross_value * 100.0


def gross_net_returns(
    gross_equity: Iterable[float],
    net_equity: Iterable[float],
) -> GrossNetReturns:
    """Return gross, net, and cost drag together rather than in isolation."""

    gross_values = _equity_values(gross_equity, minimum=2)
    net_values = _equity_values(net_equity, minimum=2)
    if len(gross_values) != len(net_values):
        raise ValueError("gross and net equity must have the same observations")
    gross = _total_return(gross_values)
    net = _total_return(net_values)
    return GrossNetReturns(
        gross=gross,
        net=net,
        cost_drag_percent=cost_drag_percent(gross, net),
    )


def group_values_by_regime(
    values: pd.Series,
    regimes: pd.Series,
) -> dict[str, tuple[float, ...]]:
    """Group aligned finite observations by regime in first-seen order."""

    if len(values.index) == 0:
        raise ValueError("regime grouping requires observations")
    if not values.index.equals(regimes.index):
        raise ValueError("values and regimes must have the same index")
    observed = _finite_values(values, name="values", minimum=1)
    labels = tuple(regimes)
    grouped: dict[str, list[float]] = {}
    for value, label in zip(observed, labels, strict=True):
        if not isinstance(label, str):
            raise TypeError("regime labels must be strings")
        if not label:
            raise ValueError("regime labels must not be empty")
        grouped.setdefault(label, []).append(value)
    return {label: tuple(group) for label, group in grouped.items()}


def metric_by_regime(
    gross_values: pd.Series,
    net_values: pd.Series,
    regimes: pd.Series,
    metric: Callable[[Iterable[float]], float],
) -> pd.DataFrame:
    """Apply one metric per regime with adjacent gross and net columns."""

    if not gross_values.index.equals(net_values.index):
        raise ValueError("gross and net values must have the same index")
    gross_groups = group_values_by_regime(gross_values, regimes)
    net_groups = group_values_by_regime(net_values, regimes)
    rows = {
        label: {
            "gross": metric(gross_group),
            "net": metric(net_groups[label]),
        }
        for label, gross_group in gross_groups.items()
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "regime"
    return result.loc[:, ["gross", "net"]]

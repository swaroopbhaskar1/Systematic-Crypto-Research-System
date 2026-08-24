"""Deterministic, point-in-time-safe portfolio backtesting."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from numbers import Integral, Real
from typing import Protocol, TypedDict, cast

import numpy as np
import pandas as pd

from cq.backtest import costs
from cq.backtest import metrics as stats
from cq.backtest.metrics import RegimeConfig
from cq.data.panel import Panel

MAX_PARTICIPATION = costs.MAX_PARTICIPATION
Bar = costs.Bar
Side = costs.Side
CostModel = costs.CostModel
executable_notional = costs.executable_notional
fill_price = costs.fill_price

DEFAULT_DELISTING_HAIRCUT = 0.20
# Closing quantities are derived by dividing a notional by a price, so a
# flat position can land a few ulps away from exactly zero.
_FLAT_TOLERANCE = 1e-12
DEFAULT_SPREAD_BPS_BY_DECILE = {
    1: 20.0,
    2: 17.5,
    3: 15.0,
    4: 12.5,
    5: 10.0,
    6: 8.0,
    7: 6.5,
    8: 5.0,
    9: 3.5,
    10: 2.0,
}
TRADE_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "quantity",
    "price",
    "cost",
    "requested_notional",
    "executed_notional",
    "fill_pct",
    "capped",
    "reason",
)


class Signal(Protocol):
    """A strategy that emits unlagged target portfolio weights."""

    def compute(self, panel: Panel) -> pd.DataFrame:
        """Return target weights aligned exactly to the panel."""
        ...


class BacktestMetrics(TypedDict, total=False):
    """Metric values populated by the metrics layer.

    Every key is optional because several statistics are genuinely undefined
    on some equity paths: Sharpe needs nonzero variance, Sortino needs a
    losing bar, Calmar needs a drawdown, cost drag needs a nonzero gross
    return, and the trade statistics need a completed round trip.  An
    undefined statistic is omitted.  Reporting it as ``0.0`` would assert a
    measurement that was never made.

    Gross and net are reported side by side because the gap between them is
    the most informative number in the output.
    """

    gross_return: float
    net_return: float
    cost_drag_percent: float
    gross_sharpe: float
    net_sharpe: float
    gross_sortino: float
    net_sortino: float
    gross_max_drawdown: float
    net_max_drawdown: float
    gross_calmar: float
    net_calmar: float
    turnover: float
    hit_rate: float
    avg_holding_period_days: float
    n_round_trips: int
    n_bars: int


@dataclass(frozen=True, slots=True)
class RoundTrip:
    """One completed position, opened from flat and closed back to flat."""

    symbol: str
    entry_timestamp: pd.Timestamp | int
    exit_timestamp: pd.Timestamp | int
    return_fraction: float


@dataclass(slots=True)
class _OpenLot:
    """Average-cost state for a position that has not returned to flat."""

    quantity: float
    basis: float
    entry_timestamp: pd.Timestamp | int
    realized: float
    closed_basis: float


@dataclass(frozen=True)
class BacktestResult:
    """Complete deterministic output of one backtest run."""

    equity: pd.Series
    gross_equity: pd.Series
    cash: pd.Series
    positions: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame
    metrics: BacktestMetrics
    regime_metrics: dict[str, BacktestMetrics]
    pct_bars_capped: float
    n_trades: int
    hypothesis_id: str | None
    data_span: pd.Timedelta
    config_hash: str

    @property
    def net_returns(self) -> pd.Series:
        """Per-bar net returns after all execution costs."""
        result = self.equity.pct_change(fill_method=None)
        result.name = "net_returns"
        return result

    @property
    def gross_returns(self) -> pd.Series:
        """Per-bar frictionless returns using the executed quantities."""
        result = self.gross_equity.pct_change(fill_method=None)
        result.name = "gross_returns"
        return result

    @property
    def returns(self) -> pd.Series:
        """Alias for net returns."""
        return self.net_returns


@dataclass(frozen=True)
class _MarketValues:
    open: float
    close: float
    quote_volume: float
    adv: float
    volatility: float
    liquidity_decile: int


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _positive_or_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _liquidity_decile(value: object) -> int:
    number = _positive_float(value, "liquidity_decile")
    decile = int(number)
    if number != decile or not 1 <= decile <= 10:
        raise ValueError("liquidity_decile must be an integer from 1 through 10")
    return decile


def _market_values(
    timestamp: object,
    symbol: str,
    fields: dict[str, pd.DataFrame],
) -> _MarketValues:
    return _MarketValues(
        open=_positive_float(fields["open"].at[timestamp, symbol], "open"),
        close=_positive_float(fields["close"].at[timestamp, symbol], "close"),
        quote_volume=_positive_float(
            fields["quote_volume"].at[timestamp, symbol],
            "quote_volume",
        ),
        adv=_positive_float(fields["adv"].at[timestamp, symbol], "adv"),
        volatility=_positive_float(
            fields["volatility"].at[timestamp, symbol],
            "volatility",
        ),
        liquidity_decile=_liquidity_decile(
            fields["liquidity_decile"].at[timestamp, symbol]
        ),
    )


def _default_cost_model() -> CostModel:
    return CostModel(
        taker_bps=0.0,
        impact_coefficient=0.0,
        spread_bps_by_decile=DEFAULT_SPREAD_BPS_BY_DECILE,
    )


def _validate_run_configuration(
    starting_equity: float,
    max_participation: float,
    delisting_haircut: float,
) -> None:
    _positive_float(starting_equity, "starting_equity")
    if (
        not math.isfinite(max_participation)
        or max_participation <= 0.0
        or max_participation > 0.05
    ):
        raise ValueError("max_participation must be in (0, 0.05]")
    if (
        not math.isfinite(delisting_haircut)
        or delisting_haircut < 0.0
        or delisting_haircut >= 1.0
    ):
        raise ValueError("delisting_haircut must be in [0, 1)")


def _validate_weights(
    weights: pd.DataFrame,
    expected: pd.DataFrame,
    emission: pd.DataFrame,
) -> None:
    """Require finite target weights everywhere the engine can act.

    A non-finite weight on a cell the engine can trade is a signal bug and
    raises.  Outside the emission mask the value is never read: the grammar
    compiler deliberately emits NaN for symbols absent from the universe so
    that "no data" stays distinguishable from "deliberately flat", and the
    engine cannot hold a position in a symbol it cannot trade.
    """

    if not weights.index.equals(expected.index):
        raise ValueError("signal weights must have the panel index")
    if not weights.columns.equals(expected.columns):
        raise ValueError("signal weights must have the panel columns")
    actionable = emission.to_numpy(dtype=bool)
    values = weights.to_numpy(dtype=float)[actionable]
    if not np.isfinite(values).all():
        raise ValueError("signal weights must be finite where the engine trades")


def _emission_mask(universe: pd.DataFrame) -> pd.DataFrame:
    mask = pd.DataFrame(
        False,
        index=universe.index,
        columns=universe.columns,
        dtype=bool,
    )
    if len(mask.index) > 1:
        mask.iloc[1:] = universe.iloc[:-1].to_numpy(dtype=bool)
    return mask & universe


def _data_span(index: pd.Index) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(0)
    difference = index[-1] - index[0]
    if isinstance(difference, pd.Timedelta):
        return difference
    if isinstance(difference, timedelta):
        return pd.Timedelta(difference)
    if isinstance(difference, np.timedelta64):
        nanoseconds = int(difference / np.timedelta64(1, "ns"))
        return pd.Timedelta(nanoseconds, unit="ns")
    numeric_difference = float(cast(float, difference))
    return pd.to_timedelta(numeric_difference, unit="ms")


def _config_hash(
    starting_equity: float,
    cost_model: CostModel,
    max_participation: float,
    delisting_haircut: float,
    regime_config: RegimeConfig,
) -> str:
    configuration = {
        "starting_equity": starting_equity,
        "max_participation": max_participation,
        "delisting_haircut": delisting_haircut,
        "taker_bps": cost_model.taker_bps,
        "impact_coefficient": cost_model.impact_coefficient,
        "spread_bps_by_decile": {
            decile: cost_model.spread_bps(decile) for decile in range(1, 11)
        },
        "regime": {
            "volatility_window": regime_config.volatility_window,
            "drawdown_window": regime_config.drawdown_window,
            "trend_window": regime_config.trend_window,
            "crash_drawdown": regime_config.crash_drawdown,
            "high_volatility": regime_config.high_volatility,
            "trend_strength": regime_config.trend_strength,
        },
    }
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="object"),
            "symbol": pd.Series(dtype="object"),
            "side": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"),
            "price": pd.Series(dtype="float64"),
            "cost": pd.Series(dtype="float64"),
            "requested_notional": pd.Series(dtype="float64"),
            "executed_notional": pd.Series(dtype="float64"),
            "fill_pct": pd.Series(dtype="float64"),
            "capped": pd.Series(dtype="bool"),
            "reason": pd.Series(dtype="object"),
        },
        columns=TRADE_COLUMNS,
    )


def _trade_record(
    *,
    timestamp: object,
    symbol: str,
    side: Side,
    quantity: float,
    price: float,
    cost: float,
    requested_notional: float,
    executed_notional: float,
    capped: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "symbol": symbol,
        "side": side.value,
        "quantity": quantity,
        "price": price,
        "cost": cost,
        "requested_notional": requested_notional,
        "executed_notional": executed_notional,
        "fill_pct": executed_notional / requested_notional,
        "capped": capped,
        "reason": reason,
    }


def _bar_timestamp(value: object) -> pd.Timestamp | int:
    if isinstance(value, pd.Timestamp):
        return value
    if not isinstance(value, bool) and isinstance(value, Integral):
        return int(value)
    raise TypeError("bar timestamps must be timestamps or Unix milliseconds")


def _record_float(record: Mapping[str, object], key: str) -> float:
    return _positive_or_negative_float(record[key], key)


def _signed_quantity(record: Mapping[str, object]) -> float:
    quantity = _positive_float(record["quantity"], "quantity")
    return quantity if record["side"] == Side.BUY.value else -quantity


def round_trips(records: Sequence[Mapping[str, object]]) -> tuple[RoundTrip, ...]:
    """Reconstruct completed round trips from the executed trade records.

    Positions use average-cost accounting and a round trip is recorded only
    when the position returns to flat, so an open position at the end of the
    run contributes nothing.  Counting it would report an unrealized mark as
    a realized outcome.  Execution costs are inside the basis and the
    proceeds, so the reported return is net.
    """

    lots: dict[str, _OpenLot] = {}
    completed: list[RoundTrip] = []
    for record in records:
        symbol = str(record["symbol"])
        delta = _signed_quantity(record)
        cash_out = delta * _record_float(record, "price") + _record_float(
            record,
            "cost",
        )
        timestamp = _bar_timestamp(record["timestamp"])
        lot = lots.get(symbol)
        if lot is None:
            lots[symbol] = _OpenLot(delta, cash_out, timestamp, 0.0, 0.0)
            continue
        trip = _apply_fill(lots, symbol, lot, delta, cash_out, timestamp)
        if trip is not None:
            completed.append(trip)
    return tuple(completed)


def _apply_fill(
    lots: dict[str, _OpenLot],
    symbol: str,
    lot: _OpenLot,
    delta: float,
    cash_out: float,
    timestamp: pd.Timestamp | int,
) -> RoundTrip | None:
    if delta * lot.quantity > 0.0:
        lot.quantity += delta
        lot.basis += cash_out
        return None
    closing = min(abs(delta), abs(lot.quantity))
    basis_closed = lot.basis * closing / abs(lot.quantity)
    cash_closed = cash_out * closing / abs(delta)
    lot.realized += -cash_closed - basis_closed
    lot.closed_basis += abs(basis_closed)
    remaining = lot.quantity + delta
    if abs(remaining) > abs(lot.quantity) * _FLAT_TOLERANCE:
        lot.quantity = remaining
        lot.basis -= basis_closed
        return None
    del lots[symbol]
    _reopen(lots, symbol, delta, cash_out, closing, timestamp)
    return RoundTrip(
        symbol=symbol,
        entry_timestamp=lot.entry_timestamp,
        exit_timestamp=timestamp,
        return_fraction=lot.realized / lot.closed_basis,
    )


def _reopen(
    lots: dict[str, _OpenLot],
    symbol: str,
    delta: float,
    cash_out: float,
    closing: float,
    timestamp: pd.Timestamp | int,
) -> None:
    """Open the leftover of a fill that reversed the position's sign."""
    leftover = abs(delta) - closing
    if leftover <= abs(delta) * _FLAT_TOLERANCE:
        return
    direction = 1.0 if delta > 0.0 else -1.0
    share = leftover / abs(delta)
    lots[symbol] = _OpenLot(
        direction * leftover,
        cash_out * share,
        timestamp,
        0.0,
        0.0,
    )


def _optional(compute: Callable[[], float]) -> float | None:
    """Return the statistic, or ``None`` when it is genuinely undefined.

    Only ``ValueError`` is treated as "undefined"; the metrics layer raises it
    deliberately for zero variance, zero downside, zero drawdown, and zero
    gross return.  A ``TypeError`` is a programming error and propagates.
    """

    try:
        return compute()
    except ValueError:
        return None


def _bar_returns(equity: pd.Series) -> tuple[float, ...]:
    changes = equity.pct_change(fill_method=None).iloc[1:]
    return tuple(float(value) for value in changes)


def _total_return_metrics(
    gross: tuple[float, ...],
    net: tuple[float, ...],
) -> BacktestMetrics:
    result: BacktestMetrics = {}
    total_gross = _optional(lambda: stats.gross_return(gross))
    total_net = _optional(lambda: stats.net_return(net))
    if total_gross is not None:
        result["gross_return"] = total_gross
    if total_net is not None:
        result["net_return"] = total_net
    if total_gross is None or total_net is None:
        return result
    gross_value, net_value = total_gross, total_net
    drag = _optional(lambda: stats.cost_drag_percent(gross_value, net_value))
    if drag is not None:
        result["cost_drag_percent"] = drag
    return result


def _ratio_metrics(
    gross_returns: tuple[float, ...],
    net_returns: tuple[float, ...],
) -> BacktestMetrics:
    result: BacktestMetrics = {}
    gross_sharpe = _optional(lambda: stats.annualized_sharpe(gross_returns))
    net_sharpe = _optional(lambda: stats.annualized_sharpe(net_returns))
    gross_sortino = _optional(lambda: stats.annualized_sortino(gross_returns))
    net_sortino = _optional(lambda: stats.annualized_sortino(net_returns))
    if gross_sharpe is not None:
        result["gross_sharpe"] = gross_sharpe
    if net_sharpe is not None:
        result["net_sharpe"] = net_sharpe
    if gross_sortino is not None:
        result["gross_sortino"] = gross_sortino
    if net_sortino is not None:
        result["net_sortino"] = net_sortino
    return result


def _drawdown_metrics(
    gross: tuple[float, ...],
    net: tuple[float, ...],
) -> BacktestMetrics:
    result: BacktestMetrics = {}
    gross_drawdown = _optional(lambda: stats.max_drawdown(gross))
    net_drawdown = _optional(lambda: stats.max_drawdown(net))
    gross_calmar = _optional(lambda: stats.calmar_ratio(gross))
    net_calmar = _optional(lambda: stats.calmar_ratio(net))
    if gross_drawdown is not None:
        result["gross_max_drawdown"] = gross_drawdown
    if net_drawdown is not None:
        result["net_max_drawdown"] = net_drawdown
    if gross_calmar is not None:
        result["gross_calmar"] = gross_calmar
    if net_calmar is not None:
        result["net_calmar"] = net_calmar
    return result


def _trade_metrics(
    weights: pd.DataFrame,
    trips: tuple[RoundTrip, ...],
) -> BacktestMetrics:
    result: BacktestMetrics = {"n_round_trips": len(trips)}
    average = _optional(lambda: stats.turnover(weights))
    if average is not None:
        result["turnover"] = average
    if not trips:
        return result
    returns = tuple(trip.return_fraction for trip in trips)
    wins = _optional(lambda: stats.hit_rate(returns))
    if wins is not None:
        result["hit_rate"] = wins
    holding = _optional(
        lambda: stats.average_holding_period(
            tuple(trip.entry_timestamp for trip in trips),
            tuple(trip.exit_timestamp for trip in trips),
        )
    )
    if holding is not None:
        result["avg_holding_period_days"] = holding
    return result


def _result_metrics(
    gross_equity: pd.Series,
    net_equity: pd.Series,
    weights: pd.DataFrame,
    trips: tuple[RoundTrip, ...],
) -> BacktestMetrics:
    gross = tuple(float(value) for value in gross_equity)
    net = tuple(float(value) for value in net_equity)
    result: BacktestMetrics = {"n_bars": len(net)}
    result.update(_total_return_metrics(gross, net))
    result.update(_ratio_metrics(_bar_returns(gross_equity), _bar_returns(net_equity)))
    result.update(_drawdown_metrics(gross, net))
    result.update(_trade_metrics(weights, trips))
    return result


def _regime_bucket(
    gross_values: tuple[float, ...],
    net_values: tuple[float, ...],
) -> BacktestMetrics:
    """Summarize one regime.

    Drawdown and Calmar are deliberately absent: a regime's bars are not
    contiguous, so a peak-to-trough path across them describes a sequence
    that never occurred and would understate the real crash drawdown.
    """

    result: BacktestMetrics = {"n_bars": len(net_values)}
    result.update(_total_return_metrics_from_bars(gross_values, net_values))
    result.update(_ratio_metrics(gross_values, net_values))
    return result


def _total_return_metrics_from_bars(
    gross_values: tuple[float, ...],
    net_values: tuple[float, ...],
) -> BacktestMetrics:
    result: BacktestMetrics = {}
    total_gross = _optional(lambda: stats.compounded_return(gross_values))
    total_net = _optional(lambda: stats.compounded_return(net_values))
    if total_gross is not None:
        result["gross_return"] = total_gross
    if total_net is not None:
        result["net_return"] = total_net
    if total_gross is None or total_net is None:
        return result
    gross_value, net_value = total_gross, total_net
    drag = _optional(lambda: stats.cost_drag_percent(gross_value, net_value))
    if drag is not None:
        result["cost_drag_percent"] = drag
    return result


def _regime_metrics(
    gross_equity: pd.Series,
    net_equity: pd.Series,
    close: pd.DataFrame,
    universe: pd.DataFrame,
    config: RegimeConfig,
) -> dict[str, BacktestMetrics]:
    """Attribute per-bar performance to the regime knowable at that bar."""
    labels = _regime_labels(close, universe, config)
    if labels.empty:
        return {}
    gross_bars = gross_equity.pct_change(fill_method=None).loc[labels.index]
    net_bars = net_equity.pct_change(fill_method=None).loc[labels.index]
    measurable = gross_bars.notna() & net_bars.notna()
    if not bool(measurable.any()):
        return {}
    gross_groups = stats.group_values_by_regime(
        gross_bars.loc[measurable],
        labels.loc[measurable],
    )
    net_groups = stats.group_values_by_regime(
        net_bars.loc[measurable],
        labels.loc[measurable],
    )
    return {
        label: _regime_bucket(values, net_groups[label])
        for label, values in gross_groups.items()
    }


def _regime_labels(
    close: pd.DataFrame,
    universe: pd.DataFrame,
    config: RegimeConfig,
) -> pd.Series:
    """Classify market regimes, or return nothing when history is too short.

    A short run genuinely has no regime attribution.  Widening the window to
    manufacture one would label bars using statistics nobody could compute.
    """

    if len(close.index) < 2:
        return pd.Series(dtype="object")
    returns = stats.market_returns(close, universe)
    needed = max(config.volatility_window, config.trend_window)
    if len(returns) < needed:
        return pd.Series(dtype="object")
    return stats.classify_regimes(returns, config)


def run(
    panel: Panel,
    signal: Signal,
    *,
    starting_equity: float = 100_000.0,
    cost_model: CostModel | None = None,
    max_participation: float = MAX_PARTICIPATION,
    delisting_haircut: float = DEFAULT_DELISTING_HAIRCUT,
    hypothesis_id: str | None = None,
    regime_config: RegimeConfig | None = None,
) -> BacktestResult:
    """Run a target-weight signal with next-open execution."""
    _validate_run_configuration(
        starting_equity,
        max_participation,
        delisting_haircut,
    )
    regimes = regime_config if regime_config is not None else RegimeConfig()
    model = cost_model if cost_model is not None else _default_cost_model()
    close = panel.field("close")
    if close.empty:
        raise ValueError("panel must contain at least one bar and symbol")

    weights = signal.compute(panel).shift(1)
    universe = panel.universe_mask()
    emission = _emission_mask(universe)
    _validate_weights(weights, close, emission)
    weights = weights.where(emission, 0.0)
    fields = {
        name: panel.field(name)
        for name in (
            "open",
            "close",
            "quote_volume",
            "adv",
            "volatility",
            "liquidity_decile",
        )
    }
    index = close.index
    symbols = tuple(str(column) for column in close.columns)
    positions = pd.DataFrame(0.0, index=index, columns=close.columns)
    fills = pd.DataFrame(0.0, index=index, columns=close.columns)
    equity = pd.Series(index=index, dtype=float, name="equity")
    gross_equity = pd.Series(index=index, dtype=float, name="gross_equity")
    cash = pd.Series(index=index, dtype=float, name="cash")
    for result in (positions, fills):
        result.index.name = None
        result.columns.name = None
    for result in (equity, gross_equity, cash):
        result.index.name = None

    current_positions = {symbol: 0.0 for symbol in symbols}
    last_values: dict[str, _MarketValues] = {}
    net_cash = float(starting_equity)
    gross_cash = float(starting_equity)
    prior_net_equity = float(starting_equity)
    trade_records: list[dict[str, object]] = []
    capped_bars = 0

    for timestamp in index:
        bar_capped = False
        available_values: dict[str, _MarketValues] = {}
        for symbol in symbols:
            if bool(universe.at[timestamp, symbol]):
                market = _market_values(timestamp, symbol, fields)
                available_values[symbol] = market
                last_values[symbol] = market

        for symbol in symbols:
            quantity = current_positions[symbol]
            if quantity == 0.0 or bool(universe.at[timestamp, symbol]):
                continue
            if symbol not in last_values:
                raise ValueError(
                    f"missing last valid execution data for delisted {symbol}"
                )
            last = last_values[symbol]
            side = Side.SELL if quantity > 0.0 else Side.BUY
            base_price = last.close
            direction = -1.0 if side is Side.SELL else 1.0
            forced_price = base_price * (
                1.0 + direction * delisting_haircut
            )
            absolute_quantity = abs(quantity)
            notional = absolute_quantity * base_price
            trade_cost = model.trade_cost(
                notional=notional,
                adv=last.adv,
                volatility=last.volatility,
                liquidity_decile=last.liquidity_decile,
            )
            delta = -quantity
            net_cash -= delta * forced_price + trade_cost.total_deducted
            gross_cash -= delta * base_price
            current_positions[symbol] = 0.0
            fills.at[timestamp, symbol] = delta
            trade_records.append(
                _trade_record(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=absolute_quantity,
                    price=forced_price,
                    cost=trade_cost.total_deducted,
                    requested_notional=notional,
                    executed_notional=notional,
                    capped=False,
                    reason="delisting",
                )
            )

        for symbol in symbols:
            if not bool(universe.at[timestamp, symbol]):
                continue
            market = available_values[symbol]
            target_notional = (
                _positive_or_negative_float(
                    weights.at[timestamp, symbol],
                    "weight",
                )
                * prior_net_equity
            )
            target_quantity = target_notional / market.open
            desired_delta = target_quantity - current_positions[symbol]
            requested_notional = abs(desired_delta) * market.open
            if requested_notional == 0.0:
                continue
            executed_notional = costs.executable_notional(
                requested_notional,
                market.quote_volume,
                max_participation,
            )
            capped = executed_notional < requested_notional
            bar_capped = bar_capped or capped
            absolute_quantity = executed_notional / market.open
            side = Side.BUY if desired_delta > 0.0 else Side.SELL
            delta = absolute_quantity if side is Side.BUY else -absolute_quantity
            price = costs.fill_price(
                side,
                Bar(
                    open=market.open,
                    close=market.close,
                    quote_volume=market.quote_volume,
                ),
                model.spread_bps(market.liquidity_decile),
            )
            trade_cost = model.trade_cost(
                notional=executed_notional,
                adv=market.adv,
                volatility=market.volatility,
                liquidity_decile=market.liquidity_decile,
            )
            net_cash -= delta * price + trade_cost.total_deducted
            gross_cash -= delta * market.open
            current_positions[symbol] += delta
            fills.at[timestamp, symbol] = delta
            trade_records.append(
                _trade_record(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=absolute_quantity,
                    price=price,
                    cost=trade_cost.total_deducted,
                    requested_notional=requested_notional,
                    executed_notional=executed_notional,
                    capped=capped,
                    reason="rebalance",
                )
            )

        marked_value = 0.0
        for symbol in symbols:
            quantity = current_positions[symbol]
            positions.at[timestamp, symbol] = quantity
            if quantity != 0.0:
                if symbol not in available_values:
                    raise ValueError(f"missing close data for held symbol {symbol}")
                marked_value += quantity * available_values[symbol].close
        net_equity = net_cash + marked_value
        gross_value = gross_cash + marked_value
        if not math.isfinite(net_equity) or not math.isfinite(gross_value):
            raise ValueError("backtest produced nonfinite equity")
        cash.at[timestamp] = net_cash
        equity.at[timestamp] = net_equity
        gross_equity.at[timestamp] = gross_value
        prior_net_equity = net_equity
        if bar_capped:
            capped_bars += 1

    trades = (
        pd.DataFrame.from_records(trade_records, columns=TRADE_COLUMNS)
        if trade_records
        else _empty_trades()
    )
    return BacktestResult(
        equity=equity,
        gross_equity=gross_equity,
        cash=cash,
        positions=positions,
        fills=fills,
        trades=trades,
        metrics=_result_metrics(
            gross_equity,
            equity,
            weights,
            round_trips(trade_records),
        ),
        regime_metrics=_regime_metrics(
            gross_equity,
            equity,
            close,
            universe,
            regimes,
        ),
        pct_bars_capped=capped_bars / len(index),
        n_trades=len(trades),
        hypothesis_id=hypothesis_id,
        data_span=_data_span(index),
        config_hash=_config_hash(
            starting_equity,
            model,
            max_participation,
            delisting_haircut,
            regimes,
        ),
    )


def run_backtest(
    panel: Panel,
    signal: Signal,
    *,
    starting_equity: float = 100_000.0,
    cost_model: CostModel | None = None,
    max_participation: float = MAX_PARTICIPATION,
    delisting_haircut: float = DEFAULT_DELISTING_HAIRCUT,
    hypothesis_id: str | None = None,
    regime_config: RegimeConfig | None = None,
) -> BacktestResult:
    """Compatibility entry point for :func:`run`."""
    return run(
        panel,
        signal,
        starting_equity=starting_equity,
        cost_model=cost_model,
        max_participation=max_participation,
        delisting_haircut=delisting_haircut,
        hypothesis_id=hypothesis_id,
        regime_config=regime_config,
    )

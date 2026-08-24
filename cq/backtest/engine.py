"""Deterministic, point-in-time-safe portfolio backtesting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import timedelta
from numbers import Real
from typing import Protocol, TypedDict, cast

import numpy as np
import pandas as pd

from cq.backtest import costs
from cq.data.panel import Panel

MAX_PARTICIPATION = costs.MAX_PARTICIPATION
Bar = costs.Bar
Side = costs.Side
CostModel = costs.CostModel
executable_notional = costs.executable_notional
fill_price = costs.fill_price

DEFAULT_DELISTING_HAIRCUT = 0.20
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
    """Metric values populated by the metrics layer."""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float


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


def _validate_weights(weights: pd.DataFrame, expected: pd.DataFrame) -> None:
    if not weights.index.equals(expected.index):
        raise ValueError("signal weights must have the panel index")
    if not weights.columns.equals(expected.columns):
        raise ValueError("signal weights must have the panel columns")
    values = weights.iloc[1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("signal weights must be finite")


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


def run(
    panel: Panel,
    signal: Signal,
    *,
    starting_equity: float = 100_000.0,
    cost_model: CostModel | None = None,
    max_participation: float = MAX_PARTICIPATION,
    delisting_haircut: float = DEFAULT_DELISTING_HAIRCUT,
    hypothesis_id: str | None = None,
) -> BacktestResult:
    """Run a target-weight signal with next-open execution."""
    _validate_run_configuration(
        starting_equity,
        max_participation,
        delisting_haircut,
    )
    model = cost_model if cost_model is not None else _default_cost_model()
    close = panel.field("close")
    if close.empty:
        raise ValueError("panel must contain at least one bar and symbol")

    weights = signal.compute(panel).shift(1)
    _validate_weights(weights, close)
    weights.iloc[0, :] = 0.0

    universe = panel.universe_mask()
    weights = weights.where(_emission_mask(universe), 0.0)
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
        metrics=BacktestMetrics(),
        regime_metrics={},
        pct_bars_capped=capped_bars / len(index),
        n_trades=len(trades),
        hypothesis_id=hypothesis_id,
        data_span=_data_span(index),
        config_hash=_config_hash(
            starting_equity,
            model,
            max_participation,
            delisting_haircut,
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
    )

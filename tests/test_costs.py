import math
import random
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.signals import ScheduledWeightSignal

from cq.backtest.costs import CostModel
from cq.backtest.engine import (
    MAX_PARTICIPATION,
    Bar,
    Side,
    executable_notional,
    fill_price,
    run_backtest,
)
from cq.data.panel import Panel

DAY_MS = 86_400_000
COSTS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "costs.yaml"

SPREAD_BPS_BY_DECILE = {
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


def cost_model(
    *,
    taker_bps: float = 5.0,
    impact_coefficient: float = 0.20,
    spreads: dict[int, float] | None = None,
) -> CostModel:
    return CostModel(
        taker_bps=taker_bps,
        impact_coefficient=impact_coefficient,
        spread_bps_by_decile=spreads or SPREAD_BPS_BY_DECILE,
    )


def zero_fee_impact_model() -> CostModel:
    """Keep a positive quoted spread but disable deducted fee and impact."""
    return CostModel(
        taker_bps=0.0,
        impact_coefficient=0.0,
        spread_bps_by_decile=SPREAD_BPS_BY_DECILE,
    )


def test_side_is_an_explicit_buy_sell_contract() -> None:
    assert Side.BUY.value == "buy"
    assert Side.SELL.value == "sell"
    assert set(Side) == {Side.BUY, Side.SELL}


def test_fill_price_is_always_worse_than_bar_open_for_seeded_fuzz() -> None:
    rng = random.Random(20260824)

    for _ in range(10_000):
        open_price = 10 ** rng.uniform(-6.0, 6.0)
        # Deliberately unrelated to open: close must never become a midprice path.
        close_price = 10 ** rng.uniform(-6.0, 6.0)
        quote_volume = 10 ** rng.uniform(0.0, 12.0)
        spread_bps = rng.uniform(0.0001, 500.0)
        side = rng.choice((Side.BUY, Side.SELL))
        bar = Bar(
            open=open_price,
            close=close_price,
            quote_volume=quote_volume,
        )

        price = fill_price(side, bar, spread_bps)
        half_spread_fraction = spread_bps / 20_000.0
        direction = 1.0 if side is Side.BUY else -1.0
        expected = open_price * (1.0 + direction * half_spread_fraction)

        if side is Side.BUY:
            assert price > open_price
        else:
            assert price < open_price
        assert price == pytest.approx(expected, rel=1e-15, abs=0.0)


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        (Side.BUY, 100.05),
        (Side.SELL, 99.95),
    ],
)
def test_fill_price_applies_exactly_the_worse_half_spread(
    side: Side, expected: float
) -> None:
    bar = Bar(open=100.0, close=151.0, quote_volume=1_000_000.0)

    assert fill_price(side, bar, spread_bps=10.0) == pytest.approx(expected)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_fill_price_has_no_close_or_midprice_path(side: Side) -> None:
    low_close = Bar(open=100.0, close=1.0, quote_volume=1_000_000.0)
    high_close = Bar(open=100.0, close=10_000.0, quote_volume=1_000_000.0)

    assert fill_price(side, low_close, 12.0) == fill_price(side, high_close, 12.0)


def test_default_max_participation_is_two_percent() -> None:
    assert MAX_PARTICIPATION == pytest.approx(0.02)
    assert executable_notional(1_000_000.0, 10_000.0) == pytest.approx(200.0)


@pytest.mark.parametrize(
    ("desired_notional", "quote_volume", "participation"),
    [
        (100.0, 10_000.0, 0.02),
        (200.0, 10_000.0, 0.02),
        (49.0, 1_000.0, 0.05),
    ],
)
def test_participation_cap_returns_minimum_of_desired_and_cap(
    desired_notional: float,
    quote_volume: float,
    participation: float,
) -> None:
    assert executable_notional(
        desired_notional,
        quote_volume,
        participation,
    ) == pytest.approx(
        min(desired_notional, participation * quote_volume)
    )


@pytest.mark.parametrize("participation", [0.001, 0.02, 0.05])
def test_participation_cap_binds_and_never_exceeds_quote_volume_share(
    participation: float,
) -> None:
    quote_volume = 75_000.0

    capped = executable_notional(
        desired_notional=10.0 * quote_volume,
        quote_volume=quote_volume,
        max_participation=participation,
    )

    assert capped == pytest.approx(participation * quote_volume)
    assert capped <= participation * quote_volume


@pytest.mark.parametrize("participation", [0.0500000001, 0.051, 1.0])
def test_participation_above_five_percent_is_rejected(
    participation: float,
) -> None:
    with pytest.raises(ValueError, match="participation"):
        executable_notional(1_000.0, 10_000.0, participation)


def test_square_root_impact_makes_larger_trades_costlier_per_unit() -> None:
    model = cost_model()
    inputs = {
        "adv": 10_000_000.0,
        "volatility": 0.04,
        "liquidity_decile": 5,
    }

    small = model.trade_cost(notional=100_000.0, **inputs)
    large = model.trade_cost(notional=200_000.0, **inputs)

    assert large.total_deducted > small.total_deducted
    assert (
        large.total_deducted / 200_000.0
        > small.total_deducted / 100_000.0
    )
    assert large.impact / 200_000.0 > small.impact / 100_000.0


def test_cost_components_follow_half_spread_fee_and_sqrt_impact_formula() -> None:
    model = cost_model(taker_bps=5.0, impact_coefficient=0.20)
    notional = 100_000.0
    adv = 10_000_000.0
    volatility = 0.04

    costs = model.trade_cost(
        notional=notional,
        adv=adv,
        volatility=volatility,
        liquidity_decile=5,
    )

    expected_spread = notional * 10.0 / 20_000.0
    expected_fee = notional * 5.0 / 10_000.0
    expected_impact = (
        notional
        * 0.20
        * volatility
        * math.sqrt(notional / adv)
    )
    assert costs.spread == pytest.approx(expected_spread)
    assert costs.taker_fee == pytest.approx(expected_fee)
    assert costs.impact == pytest.approx(expected_impact)
    assert costs.total_deducted == pytest.approx(
        expected_fee + expected_impact
    )


def test_spread_is_analytical_only_when_worse_fill_price_is_used() -> None:
    """Spread changes cash through fill price and is never deducted twice."""
    model = cost_model(taker_bps=5.0, impact_coefficient=0.20)
    costs = model.trade_cost(
        notional=100_000.0,
        adv=10_000_000.0,
        volatility=0.04,
        liquidity_decile=5,
    )

    assert costs.spread > 0.0
    assert costs.total_deducted == pytest.approx(
        costs.taker_fee + costs.impact
    )
    assert costs.total_deducted != pytest.approx(
        costs.spread + costs.taker_fee + costs.impact
    )


def test_lower_liquidity_deciles_have_strictly_wider_spreads() -> None:
    model = cost_model()
    configured = [model.spread_bps(decile) for decile in range(1, 11)]

    assert configured == [SPREAD_BPS_BY_DECILE[index] for index in range(1, 11)]
    assert all(
        lower_liquidity > higher_liquidity
        for lower_liquidity, higher_liquidity in pairwise(configured)
    )


@pytest.mark.parametrize(
    ("market", "taker_bps"),
    [
        ("spot", 10.0),
        ("perp", 5.0),
    ],
)
def test_from_yaml_parses_actual_market_taker_fees(
    market: str, taker_bps: float
) -> None:
    # costs.yaml intentionally contains only maker/taker fees.  Spread and
    # impact assumptions are supplied explicitly rather than invented by the
    # parser.
    model = CostModel.from_yaml(
        COSTS_CONFIG,
        market_type=market,
        impact_coefficient=0.20,
        spread_bps_by_decile=SPREAD_BPS_BY_DECILE,
    )

    costs = model.trade_cost(
        notional=25_000.0,
        adv=1_000_000.0,
        volatility=0.03,
        liquidity_decile=5,
    )

    assert model.taker_bps == taker_bps, market
    assert costs.taker_fee == pytest.approx(25_000.0 * taker_bps / 10_000.0)


@pytest.mark.parametrize("market_type", ["cash", "futures", "", None])
def test_from_yaml_rejects_unknown_market_types(
    market_type: str | None,
) -> None:
    with pytest.raises(ValueError, match="market"):
        CostModel.from_yaml(
            COSTS_CONFIG,
            market_type=market_type,
            impact_coefficient=0.20,
            spread_bps_by_decile=SPREAD_BPS_BY_DECILE,
        )


@pytest.mark.parametrize(
    ("open_price", "close_price", "quote_volume"),
    [
        (0.0, 100.0, 1_000.0),
        (-1.0, 100.0, 1_000.0),
        (100.0, 0.0, 1_000.0),
        (100.0, -1.0, 1_000.0),
        (100.0, 100.0, 0.0),
        (100.0, 100.0, -1.0),
        (float("nan"), 100.0, 1_000.0),
        (float("inf"), 100.0, 1_000.0),
        (100.0, float("nan"), 1_000.0),
        (100.0, 100.0, float("inf")),
    ],
)
def test_bar_rejects_nonpositive_or_nonfinite_market_inputs(
    open_price: float, close_price: float, quote_volume: float
) -> None:
    with pytest.raises(ValueError):
        Bar(
            open=open_price,
            close=close_price,
            quote_volume=quote_volume,
        )


@pytest.mark.parametrize(
    ("desired_notional", "quote_volume", "participation"),
    [
        (0.0, 1_000.0, 0.02),
        (-1.0, 1_000.0, 0.02),
        (100.0, 0.0, 0.02),
        (100.0, -1.0, 0.02),
        (100.0, 1_000.0, 0.0),
        (100.0, 1_000.0, -0.01),
        (float("nan"), 1_000.0, 0.02),
        (float("inf"), 1_000.0, 0.02),
        (100.0, float("nan"), 0.02),
        (100.0, float("inf"), 0.02),
        (100.0, 1_000.0, float("nan")),
        (100.0, 1_000.0, float("inf")),
    ],
)
def test_executable_notional_rejects_nonpositive_or_nonfinite_inputs(
    desired_notional: float,
    quote_volume: float,
    participation: float,
) -> None:
    with pytest.raises(ValueError):
        executable_notional(
            desired_notional,
            quote_volume,
            max_participation=participation,
        )


@pytest.mark.parametrize(
    "spread_bps",
    [0.0, -0.0001, float("nan"), float("inf")],
)
def test_fill_price_rejects_invalid_spread(spread_bps: float) -> None:
    bar = Bar(open=100.0, close=101.0, quote_volume=1_000.0)

    with pytest.raises(ValueError, match="spread"):
        fill_price(Side.BUY, bar, spread_bps)


@pytest.mark.parametrize(
    ("taker_bps", "impact_coefficient", "spreads"),
    [
        (-1.0, 0.2, SPREAD_BPS_BY_DECILE),
        (5.0, -0.1, SPREAD_BPS_BY_DECILE),
        (5.0, 0.2, {**SPREAD_BPS_BY_DECILE, 1: 0.0}),
        (5.0, 0.2, {**SPREAD_BPS_BY_DECILE, 1: -1.0}),
        (float("nan"), 0.2, SPREAD_BPS_BY_DECILE),
        (float("inf"), 0.2, SPREAD_BPS_BY_DECILE),
        (5.0, float("nan"), SPREAD_BPS_BY_DECILE),
        (5.0, float("inf"), SPREAD_BPS_BY_DECILE),
    ],
)
def test_cost_model_rejects_negative_or_nonfinite_configuration(
    taker_bps: float,
    impact_coefficient: float,
    spreads: dict[int, float],
) -> None:
    with pytest.raises(ValueError):
        cost_model(
            taker_bps=taker_bps,
            impact_coefficient=impact_coefficient,
            spreads=spreads,
        )


def test_cost_model_accepts_zero_fee_and_impact_configuration() -> None:
    costs = zero_fee_impact_model().trade_cost(
        notional=100.0,
        adv=10_000.0,
        volatility=0.01,
        liquidity_decile=10,
    )

    assert costs.spread == pytest.approx(0.01)
    assert costs.taker_fee == 0.0
    assert costs.impact == 0.0
    assert costs.total_deducted == 0.0


@pytest.mark.parametrize(
    "spreads",
    [
        {key: value for key, value in SPREAD_BPS_BY_DECILE.items() if key != 10},
        {**SPREAD_BPS_BY_DECILE, 11: 1.0},
        {str(key): value for key, value in SPREAD_BPS_BY_DECILE.items()},
        {**SPREAD_BPS_BY_DECILE, 1: float("nan")},
        {**SPREAD_BPS_BY_DECILE, 1: float("inf")},
        {**SPREAD_BPS_BY_DECILE, 1: SPREAD_BPS_BY_DECILE[2]},
        {**SPREAD_BPS_BY_DECILE, 2: SPREAD_BPS_BY_DECILE[1] + 1.0},
    ],
)
def test_cost_model_rejects_invalid_decile_configuration(
    spreads: dict[object, float],
) -> None:
    with pytest.raises(ValueError, match="decile|spread"):
        CostModel(
            taker_bps=5.0,
            impact_coefficient=0.20,
            spread_bps_by_decile=spreads,
        )


@pytest.mark.parametrize(
    ("notional", "adv", "volatility"),
    [
        (0.0, 1_000_000.0, 0.03),
        (-1.0, 1_000_000.0, 0.03),
        (10_000.0, 0.0, 0.03),
        (10_000.0, -1.0, 0.03),
        (10_000.0, 1_000_000.0, 0.0),
        (10_000.0, 1_000_000.0, -0.01),
        (float("nan"), 1_000_000.0, 0.03),
        (float("inf"), 1_000_000.0, 0.03),
        (10_000.0, float("nan"), 0.03),
        (10_000.0, float("inf"), 0.03),
        (10_000.0, 1_000_000.0, float("nan")),
        (10_000.0, 1_000_000.0, float("inf")),
    ],
)
def test_trade_cost_rejects_nonpositive_or_nonfinite_inputs(
    notional: float, adv: float, volatility: float
) -> None:
    with pytest.raises(ValueError):
        cost_model().trade_cost(
            notional=notional,
            adv=adv,
            volatility=volatility,
            liquidity_decile=5,
        )


@pytest.mark.parametrize("liquidity_decile", [0, 11])
def test_trade_cost_rejects_deciles_outside_one_through_ten(
    liquidity_decile: int,
) -> None:
    with pytest.raises(ValueError, match="decile"):
        cost_model().trade_cost(
            notional=10_000.0,
            adv=1_000_000.0,
            volatility=0.03,
            liquidity_decile=liquidity_decile,
        )


def test_tax_is_excluded_from_trade_costs() -> None:
    costs = cost_model().trade_cost(
        notional=100_000.0,
        adv=10_000_000.0,
        volatility=0.04,
        liquidity_decile=5,
    )

    assert not hasattr(costs, "tax")
    assert not hasattr(costs, "taxes")
    assert costs.total_deducted == pytest.approx(
        costs.taker_fee + costs.impact
    )


def test_costs_are_deterministic_for_identical_inputs() -> None:
    model = cost_model()
    inputs = {
        "notional": 123_456.78,
        "adv": 9_876_543.21,
        "volatility": 0.037,
        "liquidity_decile": 7,
    }

    first = model.trade_cost(**inputs)
    second = model.trade_cost(**inputs)

    assert first == second


def _integration_panel(
    quote_volumes: list[float],
    *,
    symbols: tuple[str, ...] = ("CAPUSDT",),
    membership: dict[str, list[bool]] | None = None,
) -> Panel:
    rows: list[dict[str, object]] = []
    masks = membership or {
        symbol: [True] * len(quote_volumes) for symbol in symbols
    }
    for symbol in symbols:
        for bar, quote_volume in enumerate(quote_volumes):
            rows.append(
                {
                    "ts": bar * DAY_MS,
                    "symbol": symbol,
                    "market_type": "spot",
                    "open": 10.0 if symbol == "CAPUSDT" else 100.0,
                    "close": 10.0 if symbol == "CAPUSDT" else 100.0,
                    "quote_volume": quote_volume,
                    "adv": 1_000_000.0,
                    "volatility": 0.01,
                    "liquidity_decile": 10,
                    "in_universe": masks[symbol][bar],
                }
            )
    return Panel.from_long(pd.DataFrame(rows), market_type="spot")


def test_engine_reports_nonbinding_and_binding_volume_cap_fills() -> None:
    panel = _integration_panel([100_000.0, 100_000.0, 1_000.0, 100_000.0])
    close = panel.field("close")
    max_participation = 0.01
    weights = pd.DataFrame(
        [0.50, 1.00, 1.00, 1.00],
        index=close.index,
        columns=close.columns,
    )

    result = run_backtest(
        panel=panel,
        signal=ScheduledWeightSignal(weights),
        starting_equity=1_000.0,
        cost_model=zero_fee_impact_model(),
        max_participation=max_participation,
    )
    trades = result.trades.set_index("timestamp")
    first = trades.loc[DAY_MS]
    capped = trades.loc[2 * DAY_MS]

    first_requested = 0.50 * 1_000.0
    first_cap = max_participation * 100_000.0
    assert first["side"] == Side.BUY.value
    assert not bool(first["capped"])
    assert first["requested_notional"] == pytest.approx(first_requested, abs=1e-9)
    assert first["executed_notional"] == pytest.approx(
        min(first_requested, first_cap),
        abs=1e-9,
    )
    assert first["fill_pct"] == pytest.approx(1.0)

    # The first buy pays only the 1 bp half-spread: 500 * .0001 = .05.
    # Thus prior net equity is 999.95 and the uncapped rebalance requests
    # 1.00 * 999.95 - 50 tokens * $10 = $499.95 independently of the report.
    requested = 999.95 - 50.0 * 10.0
    cap = max_participation * 1_000.0
    default_cap = MAX_PARTICIPATION * 1_000.0
    expected_executed = min(requested, cap)

    assert capped["side"] == Side.BUY.value
    assert bool(capped["capped"])
    assert capped["requested_notional"] == pytest.approx(requested, abs=1e-9)
    assert capped["executed_notional"] == pytest.approx(
        expected_executed,
        abs=1e-9,
    )
    assert 0.0 < capped["executed_notional"] <= cap
    assert capped["executed_notional"] < capped["requested_notional"]
    assert capped["fill_pct"] == pytest.approx(
        expected_executed / requested,
        abs=1e-9,
    )
    assert capped["quantity"] * 10.0 == pytest.approx(cap, abs=1e-9)
    position_change = (
        result.positions.loc[2 * DAY_MS, "CAPUSDT"]
        - result.positions.loc[DAY_MS, "CAPUSDT"]
    )
    assert position_change > 0.0
    assert position_change * 10.0 == pytest.approx(cap, abs=1e-9)
    assert cap == pytest.approx(10.0)
    assert default_cap == pytest.approx(20.0)
    assert capped["executed_notional"] != pytest.approx(default_cap)


def test_delisting_forces_haircut_liquidation_and_blocks_future_listing() -> None:
    symbols = ("OLDUSDT", "FUTUREUSDT")
    membership = {
        "OLDUSDT": [True, True, False, False, False],
        "FUTUREUSDT": [False, False, True, True, True],
    }
    panel = _integration_panel(
        [1_000_000.0] * 5,
        symbols=symbols,
        membership=membership,
    )
    close = panel.field("close")
    weights = pd.DataFrame(
        {
            "OLDUSDT": [0.50, 0.50, 0.00, 0.00, 0.00],
            "FUTUREUSDT": [1.00, 0.00, 0.00, 0.00, 0.00],
        },
        index=close.index,
    ).loc[:, close.columns]

    result = run_backtest(
        panel=panel,
        signal=ScheduledWeightSignal(weights),
        starting_equity=1_000.0,
        cost_model=zero_fee_impact_model(),
        delisting_haircut=0.20,
    )
    old_trades = result.trades.loc[result.trades["symbol"] == "OLDUSDT"]
    liquidation = old_trades.loc[old_trades["reason"] == "delisting"]

    # At bar 1 the shifted .50 weight buys exactly 5 tokens at a 100.01 fill:
    # cash = 1000 - 5*100.01 = 499.95 and equity = 999.95.  Panel masks OLD's
    # bar-2 prices, so the engine must retain the last valid $100 terminal
    # price.  The 20% haircut is the complete forced-liquidation price
    # adjustment: no additional spread is applied, and this zero-fee,
    # zero-impact model deducts no trade cost.  Sale proceeds are exactly $400,
    # terminal cash/equity is $899.95, and the haircut loses exactly $100.
    assert np.isnan(panel.field("close").loc[2 * DAY_MS, "OLDUSDT"])
    assert len(liquidation) == 1
    forced = liquidation.iloc[0]
    assert forced["timestamp"] == 2 * DAY_MS
    assert forced["side"] == Side.SELL.value
    assert forced["price"] == pytest.approx(80.0, abs=1e-9)
    assert forced["quantity"] == pytest.approx(5.0, abs=1e-9)
    assert forced["cost"] == pytest.approx(0.0, abs=1e-9)
    assert forced["quantity"] * forced["price"] == pytest.approx(
        400.0,
        abs=1e-9,
    )
    assert result.cash.loc[DAY_MS] == pytest.approx(499.95, abs=1e-9)
    assert result.equity.loc[DAY_MS] == pytest.approx(999.95, abs=1e-9)
    assert result.cash.loc[2 * DAY_MS] == pytest.approx(899.95, abs=1e-9)
    assert result.equity.loc[2 * DAY_MS] == pytest.approx(899.95, abs=1e-9)
    assert (
        result.equity.loc[DAY_MS] - result.equity.loc[2 * DAY_MS]
    ) == pytest.approx(100.0, abs=1e-9)
    assert result.positions.loc[2 * DAY_MS, "OLDUSDT"] == pytest.approx(
        0.0,
        abs=1e-9,
    )
    assert result.positions.iloc[-1]["OLDUSDT"] == pytest.approx(
        0.0,
        abs=1e-9,
    )

    # Targets for an unavailable future listing are ignored, not queued.
    assert not (result.trades["symbol"] == "FUTUREUSDT").any()
    assert (result.positions["FUTUREUSDT"] == 0.0).all()

"""Survivorship-bias contracts for the universe registry and the engine.

Every test here defends one property: a backtest must see the tokens that
died, price their death, and never see a token before it existed.
"""

from datetime import UTC, datetime, timedelta
from typing import Final, TypeAlias

import numpy as np
import pandas as pd
import pytest
from fixtures.signals import ScheduledWeightSignal

from cq.backtest.costs import CostModel, Side
from cq.backtest.engine import run
from cq.data.panel import Panel
from cq.data.universe import Listing, UniverseRegistry

DAY_MS: Final[int] = 86_400_000
MICROSECOND: Final[timedelta] = timedelta(microseconds=1)
QUOTE_VOLUME: Final[float] = 10_000_000.0
ADV: Final[float] = 1_000_000.0
VOLATILITY: Final[float] = 0.01
LIQUIDITY_DECILE: Final[int] = 10

# Decile 10 quotes a 2 bp full spread, so exactly one adverse half-spread of
# 1 bp is paid on every rebalance fill: a $100.00 open buys at $100.01.
SPREAD_BPS_BY_DECILE: Final[dict[int, float]] = {
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

# (open, close, in_universe) for one symbol on one bar.
BarSpec: TypeAlias = tuple[float, float, bool]

DEAD_SYMBOL: Final[str] = "DEADUSDT"
ZOMBIE_SYMBOL: Final[str] = "ZOMBIEUSDT"

# Hand-built 2021-2024 listing history.  Four terminal classifications plus
# two survivors, so "today's universe" is provably a strict subset.
ZERO_EVIDENCE: Final[str] = "https://issuer.example/terra-classic-collapse"


def at(
    year: int,
    month: int = 1,
    day: int = 1,
) -> datetime:
    """Return a timezone-aware UTC midnight, as the registry requires."""
    return datetime(year, month, day, tzinfo=UTC)


HISTORY: Final[tuple[Listing, ...]] = (
    Listing("BTCUSDT", "binance", "spot", at(2019, 1, 1)),
    Listing(
        "BTTUSDT",
        "binance",
        "spot",
        at(2021, 2, 1),
        at(2022, 1, 20),
        "delisted",
    ),
    Listing(
        "LUNAUSDT",
        "binance",
        "spot",
        at(2021, 3, 1),
        at(2022, 5, 13),
        "zero",
        zero_evidence=ZERO_EVIDENCE,
    ),
    Listing(
        "SRMUSDT",
        "ftx",
        "spot",
        at(2021, 1, 1),
        at(2022, 11, 11),
        "exchange_failure",
    ),
    Listing(
        "MATICUSDT",
        "binance",
        "spot",
        at(2021, 6, 1),
        at(2024, 9, 1),
        "migrated",
        successor="POLUSDT",
    ),
    Listing("POLUSDT", "binance", "spot", at(2024, 9, 1)),
)
DEAD_LISTINGS: Final[tuple[Listing, ...]] = tuple(
    listing for listing in HISTORY if listing.delisted_at is not None
)


def bar_row(
    *,
    timestamp: int,
    symbol: str,
    spec: BarSpec,
) -> dict[str, object]:
    """Return one long-format panel row with all engine-required fields."""
    open_price, close_price, in_universe = spec
    return {
        "ts": timestamp,
        "symbol": symbol,
        "market_type": "spot",
        "open": open_price,
        "close": close_price,
        "quote_volume": QUOTE_VOLUME,
        "adv": ADV,
        "volatility": VOLATILITY,
        "liquidity_decile": LIQUIDITY_DECILE,
        "in_universe": in_universe,
    }


def build_panel(
    bars: dict[str, list[BarSpec]],
    *,
    timestamps: tuple[int, ...] | None = None,
) -> Panel:
    """Build a spot panel from per-symbol ``(open, close, in_universe)`` bars."""
    lengths = {len(specs) for specs in bars.values()}
    if len(lengths) != 1:
        raise ValueError("every symbol must supply the same number of bars")
    bar_count = lengths.pop()
    stamps = timestamps or tuple(index * DAY_MS for index in range(bar_count))
    rows = [
        bar_row(timestamp=stamp, symbol=symbol, spec=spec)
        for symbol, specs in bars.items()
        for stamp, spec in zip(stamps, specs, strict=True)
    ]
    return Panel.from_long(pd.DataFrame(rows), market_type="spot")


def constant_weight_signal(panel: Panel, weight: float) -> ScheduledWeightSignal:
    """Demand ``weight`` in every symbol on every bar, forever.

    The signal never stops asking for the dead token.  Refusing it is the
    engine's job, not the signal's.
    """
    close = panel.field("close")
    return ScheduledWeightSignal(
        pd.DataFrame(
            weight,
            index=close.index,
            columns=close.columns,
            dtype=float,
        )
    )


def zero_fee_model() -> CostModel:
    """Quote a real spread but deduct no fee or impact.

    Isolating the spread keeps the delisting arithmetic hand-checkable: the
    only frictions are one 1 bp entry half-spread and the haircut itself.
    """
    return CostModel(
        taker_bps=0.0,
        impact_coefficient=0.0,
        spread_bps_by_decile=SPREAD_BPS_BY_DECILE,
    )


def test_delisted_tokens_present() -> None:
    """A historical universe must contain the dead, not only the survivors.

    Defends the property that a 2021-2024 registry enumerates tokens with
    ``delisted_at`` set, that each was a genuine member before its exit and a
    non-member after it, and that all four terminal classifications are
    representable.  A dataset holding only today's tickers passes every other
    test in the suite while silently inflating every result.
    """
    registry = UniverseRegistry(HISTORY)

    assert len(DEAD_LISTINGS) == 4
    assert {listing.delist_reason for listing in DEAD_LISTINGS} == {
        "delisted",
        "zero",
        "migrated",
        "exchange_failure",
    }

    for listing in DEAD_LISTINGS:
        delisted_at = listing.delisted_at
        assert delisted_at is not None
        alive = registry.universe_at(delisted_at - timedelta(days=1), "spot")
        assert listing.symbol in alive, listing.symbol
        for later in (delisted_at, delisted_at + timedelta(days=1)):
            assert listing.symbol not in registry.universe_at(later, "spot")

    # Mid-history the universe is dominated by tokens that no longer exist.
    assert registry.universe_at(at(2022, 1, 1), "spot") == frozenset(
        {"BTCUSDT", "BTTUSDT", "LUNAUSDT", "SRMUSDT", "MATICUSDT"}
    )

    # The survivors-only view loses every dead token: that gap is the bias.
    survivors = registry.universe_at(at(2024, 12, 31), "spot")
    assert survivors == frozenset({"BTCUSDT", "POLUSDT"})
    dead_symbols = frozenset(listing.symbol for listing in DEAD_LISTINGS)
    assert dead_symbols.isdisjoint(survivors)
    assert len(dead_symbols) == 4

    # A zero and a migration are economically opposite and must not be
    # conflated.  The migration hands over to a successor with no coverage
    # gap and no double count; the zero hands over to nothing.
    handover = registry.universe_at(at(2024, 9, 1), "spot")
    assert "MATICUSDT" not in handover
    assert "POLUSDT" in handover
    migrated = next(
        listing for listing in DEAD_LISTINGS if listing.delist_reason == "migrated"
    )
    zeroed = next(
        listing for listing in DEAD_LISTINGS if listing.delist_reason == "zero"
    )
    assert migrated.successor == "POLUSDT"
    assert zeroed.successor is None
    assert "LUNAUSDT" not in registry.universe_at(at(2022, 5, 13), "spot")


@pytest.mark.parametrize(
    (
        "weight",
        "haircut",
        "expected_side",
        "expected_forced_price",
        "expected_equity_before",
        "expected_equity_after",
        "expected_gross_equity",
    ),
    [
        (0.5, 0.00, Side.SELL, 110.0, 1049.95, 1049.95, 1050.0),
        (0.5, 0.20, Side.SELL, 88.0, 1049.95, 939.95, 1050.0),
        (0.5, 0.35, Side.SELL, 71.5, 1049.95, 857.45, 1050.0),
        (0.5, 0.50, Side.SELL, 55.0, 1049.95, 774.95, 1050.0),
        (-0.5, 0.00, Side.BUY, 110.0, 949.95, 949.95, 950.0),
        (-0.5, 0.20, Side.BUY, 132.0, 949.95, 839.95, 950.0),
        (-0.5, 0.35, Side.BUY, 148.5, 949.95, 757.45, 950.0),
        (-0.5, 0.50, Side.BUY, 165.0, 949.95, 674.95, 950.0),
    ],
)
def test_delisted_position_realizes_loss(
    weight: float,
    haircut: float,
    expected_side: Side,
    expected_forced_price: float,
    expected_equity_before: float,
    expected_equity_after: float,
    expected_gross_equity: float,
) -> None:
    """Holding a token through its delisting must cost exactly the haircut.

    Defends the property that the equity curve pays for the death instead of
    skipping it.  Every number below is computed by hand from the paper
    convention, never from the engine's own formula:

      bar 0: no lagged weight, so equity is the $1,000 starting equity.
      bar 1: target notional = 0.5 * 1000 = $500 at a $100.00 open, so the
             position is exactly 5 tokens.  The decile-10 half-spread is
             1 bp, so a long fills at $100.01 and a short at $99.99; either
             way the round-trip entry costs 5 * $0.01 = $0.05.
             long  cash = 1000 - 5*100.01 = 499.95, marked 5*110 = 550.00,
                   equity = 1049.95.
             short cash = 1000 + 5*99.99  = 1499.95, marked -550.00,
                   equity =  949.95.
      bar 2: the token leaves the universe.  Its bar-2 prices are masked to
             NaN, so the only lawful terminal price is the last valid close
             of $110.  The forced fill is 110*(1 - h) for the long and
             110*(1 + h) for the short - adverse in both directions, because
             a short in a dead token must not be a windfall.
             realized haircut = 5 * 110 * h = 550h, identical for both sides.
             long  h=0.20: fill $88.00,  equity 1049.95 - 110.00 = 939.95.
             short h=0.20: fill $132.00, equity  949.95 - 110.00 = 839.95.
      Gross equity uses the unhaircut $110 close, so it is flat across the
      delisting bar for every haircut.  The whole loss lives in the net-gross
      gap, which is 550h + the $0.05 entry spread.
    """
    panel = build_panel(
        {
            DEAD_SYMBOL: [
                (100.0, 100.0, True),
                (100.0, 110.0, True),
                # Prices the vendor still publishes but the universe denies.
                (50.0, 45.0, False),
                (40.0, 35.0, False),
            ]
        }
    )
    assert np.isnan(panel.field("close").loc[2 * DAY_MS, DEAD_SYMBOL])

    result = run(
        panel,
        constant_weight_signal(panel, weight),
        starting_equity=1_000.0,
        cost_model=zero_fee_model(),
        delisting_haircut=haircut,
    )

    equity_before = result.equity.loc[DAY_MS]
    equity_after = result.equity.loc[2 * DAY_MS]
    assert equity_before == pytest.approx(expected_equity_before, abs=1e-9)
    assert equity_after == pytest.approx(expected_equity_after, abs=1e-9)
    assert result.equity.loc[3 * DAY_MS] == pytest.approx(
        expected_equity_after, abs=1e-9
    )
    if haircut > 0.0:
        assert equity_after < equity_before
    else:
        assert equity_after == pytest.approx(equity_before, abs=1e-12)
    assert (equity_before - equity_after) == pytest.approx(550.0 * haircut, abs=1e-9)

    # Gross is frictionless, so it must not absorb the haircut.
    assert result.gross_equity.loc[DAY_MS] == pytest.approx(
        expected_gross_equity, abs=1e-9
    )
    assert result.gross_equity.loc[2 * DAY_MS] == pytest.approx(
        expected_gross_equity, abs=1e-9
    )
    assert (
        result.gross_equity.loc[2 * DAY_MS] - result.equity.loc[2 * DAY_MS]
    ) == pytest.approx(550.0 * haircut + 0.05, abs=1e-9)

    liquidations = result.trades.loc[result.trades["reason"] == "delisting"]
    assert len(liquidations) == 1
    forced = liquidations.iloc[0]
    assert forced["symbol"] == DEAD_SYMBOL
    assert forced["timestamp"] == 2 * DAY_MS
    assert forced["side"] == expected_side.value
    assert forced["quantity"] == pytest.approx(5.0, abs=1e-9)
    assert forced["price"] == pytest.approx(expected_forced_price, abs=1e-9)
    assert forced["cost"] == pytest.approx(0.0, abs=1e-9)
    assert forced["requested_notional"] == pytest.approx(550.0, abs=1e-9)
    assert forced["executed_notional"] == pytest.approx(550.0, abs=1e-9)
    assert forced["fill_pct"] == pytest.approx(1.0, abs=1e-12)
    assert not bool(forced["capped"])

    # The forced fill is adverse on both sides, never a windfall.
    if haircut > 0.0 and expected_side is Side.SELL:
        assert forced["price"] < 110.0
    elif haircut > 0.0:
        assert forced["price"] > 110.0

    assert result.n_trades == 2
    assert list(result.trades["reason"]) == ["rebalance", "delisting"]
    expected_fill = -5.0 if expected_side is Side.SELL else 5.0
    assert result.fills.loc[2 * DAY_MS, DEAD_SYMBOL] == pytest.approx(
        expected_fill, abs=1e-9
    )

    # The position is gone and stays gone even though the signal keeps
    # demanding it on every later bar.
    assert result.positions.loc[2 * DAY_MS, DEAD_SYMBOL] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.positions.loc[3 * DAY_MS, DEAD_SYMBOL] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.fills.loc[3 * DAY_MS, DEAD_SYMBOL] == pytest.approx(0.0, abs=1e-12)
    assert result.trades["timestamp"].max() == 2 * DAY_MS


def test_universe_at_excludes_future_listings() -> None:
    """A token cannot be traded before it existed, not even by one microsecond.

    Defends the point-in-time boundary convention of ``universe_at``, which
    this test pins down explicitly so that changing it breaks the suite:
    membership is the half-open interval ``[listed_at, delisted_at)``.  A
    token is a member exactly at ``listed_at`` and is already gone exactly at
    ``delisted_at``.
    """
    listed_at = at(2023, 1, 1)
    delisted_at = at(2022, 6, 1)
    registry = UniverseRegistry(
        (
            Listing(
                "OLDUSDT",
                "binance",
                "spot",
                at(2021, 1, 1),
                delisted_at,
                "delisted",
            ),
            Listing("FUTUREUSDT", "binance", "spot", listed_at),
        )
    )

    assert registry.universe_at(at(2022, 1, 1), "spot") == frozenset({"OLDUSDT"})
    assert "FUTUREUSDT" not in registry.universe_at(at(2022, 1, 1), "spot")
    assert "FUTUREUSDT" not in registry.universe_at(at(2022, 12, 31), "spot")

    # Lower bound is inclusive.
    assert "FUTUREUSDT" in registry.universe_at(listed_at, "spot")
    assert "FUTUREUSDT" not in registry.universe_at(listed_at - MICROSECOND, "spot")

    # Upper bound is exclusive.
    assert "OLDUSDT" in registry.universe_at(delisted_at - MICROSECOND, "spot")
    assert "OLDUSDT" not in registry.universe_at(delisted_at, "spot")


def test_panel_membership_is_point_in_time_not_data_presence() -> None:
    """Having a price row for 2023 does not make a token investable in 2023.

    Defends the join between the registry and the panel.  A vendor dump
    backfills a token's full history the day it lists, so a panel that treats
    "a row exists" as "the token is investable" reproduces exactly the
    survivorship error the registry exists to prevent.  Membership here is
    generated from ``universe_at`` per bar, and the panel must mask the
    backfilled prices to NaN and keep the engine flat until the token is
    genuinely tradable.
    """
    start = pd.Timestamp("2023-01-01", tz="UTC")
    stamps = tuple(
        int((start + pd.Timedelta(days=index)).value // 1_000_000)
        for index in range(5)
    )
    late_listed_at = (start + pd.Timedelta(days=3)).to_pydatetime()
    registry = UniverseRegistry(
        (
            Listing("LIVEUSDT", "binance", "spot", start.to_pydatetime()),
            Listing("LATEUSDT", "binance", "spot", late_listed_at),
        )
    )
    membership = {
        symbol: [
            symbol
            in registry.universe_at(
                (start + pd.Timedelta(days=index)).to_pydatetime(),
                "spot",
            )
            for index in range(5)
        ]
        for symbol in ("LIVEUSDT", "LATEUSDT")
    }
    assert membership["LIVEUSDT"] == [True, True, True, True, True]
    assert membership["LATEUSDT"] == [False, False, False, True, True]

    # The vendor supplies a complete backfilled series for both symbols.
    panel = build_panel(
        {
            "LIVEUSDT": [
                (100.0, 100.0, membership["LIVEUSDT"][index]) for index in range(5)
            ],
            "LATEUSDT": [
                (50.0, 50.0, membership["LATEUSDT"][index]) for index in range(5)
            ],
        },
        timestamps=stamps,
    )
    close = panel.field("close")
    assert bool(np.isnan(close.loc[stamps[0], "LATEUSDT"]))
    assert bool(np.isnan(close.loc[stamps[2], "LATEUSDT"]))
    assert close.loc[stamps[3], "LATEUSDT"] == pytest.approx(50.0, abs=1e-12)
    assert list(panel.universe_mask().loc[:, "LATEUSDT"]) == [
        False,
        False,
        False,
        True,
        True,
    ]

    result = run(
        panel,
        constant_weight_signal(panel, 0.25),
        starting_equity=1_000.0,
        cost_model=zero_fee_model(),
    )
    late_trades = result.trades.loc[result.trades["symbol"] == "LATEUSDT"]

    # A newly listed token has no prior in-universe bar, so the single
    # execution lag makes its first tradable bar the one after listing.
    assert list(late_trades["timestamp"]) == [stamps[4]]
    for index in range(4):
        assert result.positions.loc[stamps[index], "LATEUSDT"] == pytest.approx(
            0.0, abs=1e-12
        )
        assert result.fills.loc[stamps[index], "LATEUSDT"] == pytest.approx(
            0.0, abs=1e-12
        )
    assert result.positions.loc[stamps[4], "LATEUSDT"] > 0.0


def test_relisted_symbol_does_not_resurrect_the_liquidated_position() -> None:
    """A token that returns to the universe must return flat, not still held.

    This is the survivorship bug that hides behind a correct delisting path:
    an engine that "pauses" a symbol while it is out of the universe rather
    than closing it will hand the position back when the symbol reappears,
    quietly refunding the haircut and paying the strategy for a gap it never
    carried risk through.  The realized loss must survive the round trip, and
    the re-entry must be sized off post-loss equity from a zero base.

    Hand-computed, starting equity $1,000, 20% haircut, 1 bp half-spread:
      bar 1: buy 5 tokens at $100.01, cash 499.95, marked 5*110 = 550.00,
             equity 1049.95.
      bar 2: out of universe, forced sale of 5 at 110*0.80 = $88.00,
             proceeds $440.00, cash and equity 939.95.
      bar 3: back in the universe, but the prior bar was not, so the single
             execution lag forbids a trade: equity stays exactly 939.95.
      bar 4: first lawful re-entry.  Target notional 0.5 * 939.95 = 469.975
             at a $100.00 open is 4.69975 tokens, filled at $100.01 for
             470.0219975, leaving cash 469.9280025.  Marked at $105 the
             position is worth 493.47375, so equity is 963.4017525.
    """
    panel = build_panel(
        {
            ZOMBIE_SYMBOL: [
                (100.0, 100.0, True),
                (100.0, 110.0, True),
                (50.0, 45.0, False),
                (90.0, 95.0, True),
                (100.0, 105.0, True),
            ]
        }
    )

    result = run(
        panel,
        constant_weight_signal(panel, 0.5),
        starting_equity=1_000.0,
        cost_model=zero_fee_model(),
        delisting_haircut=0.20,
    )

    assert result.equity.loc[DAY_MS] == pytest.approx(1049.95, abs=1e-9)
    assert result.equity.loc[2 * DAY_MS] == pytest.approx(939.95, abs=1e-9)
    assert result.equity.loc[3 * DAY_MS] == pytest.approx(939.95, abs=1e-9)
    assert result.equity.loc[4 * DAY_MS] == pytest.approx(963.4017525, abs=1e-9)

    assert result.positions.loc[2 * DAY_MS, ZOMBIE_SYMBOL] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.positions.loc[3 * DAY_MS, ZOMBIE_SYMBOL] == pytest.approx(
        0.0, abs=1e-12
    )
    assert result.positions.loc[4 * DAY_MS, ZOMBIE_SYMBOL] == pytest.approx(
        4.69975, abs=1e-9
    )

    assert list(result.trades["reason"]) == ["rebalance", "delisting", "rebalance"]
    assert list(result.trades["timestamp"]) == [DAY_MS, 2 * DAY_MS, 4 * DAY_MS]
    reentry = result.trades.iloc[2]
    assert reentry["side"] == Side.BUY.value
    assert reentry["quantity"] == pytest.approx(4.69975, abs=1e-9)
    assert reentry["price"] == pytest.approx(100.01, abs=1e-9)
    assert result.n_trades == 3

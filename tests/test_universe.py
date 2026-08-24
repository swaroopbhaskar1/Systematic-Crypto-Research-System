from datetime import UTC, datetime

import pytest

from cq.data.universe import Listing, UniverseRegistry


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_spot_and_perp_are_separate_and_market_type_is_required() -> None:
    registry = UniverseRegistry(
        [
            Listing("BTCUSDT", "binance", "spot", dt(2018), None, None, None),
            Listing("BTCUSDT", "binance", "perp", dt(2020), None, None, None),
        ]
    )

    assert registry.universe_at(dt(2019), "spot") == frozenset({"BTCUSDT"})
    assert registry.universe_at(dt(2019), "perp") == frozenset()
    with pytest.raises(TypeError):
        registry.universe_at(dt(2024))  # type: ignore[call-arg]


def test_future_listings_are_excluded_and_delisted_are_historical() -> None:
    registry = UniverseRegistry(
        [
            Listing(
                "OLDUSDT",
                "binance",
                "spot",
                dt(2019),
                dt(2021),
                "delisted",
                None,
            ),
            Listing("NEWUSDT", "binance", "spot", dt(2025), None, None, None),
        ]
    )

    assert registry.universe_at(dt(2020), "spot") == frozenset({"OLDUSDT"})
    assert registry.universe_at(dt(2022), "spot") == frozenset()
    assert registry.universe_at(dt(2026), "spot") == frozenset({"NEWUSDT"})


def test_zero_requires_explicit_evidence_and_delisting_is_not_zero() -> None:
    delisted = Listing(
        "VENUEEXITUSDT",
        "binance",
        "spot",
        dt(2020),
        dt(2021),
        "delisted",
        None,
    )
    assert delisted.delist_reason == "delisted"

    with pytest.raises(ValueError, match="evidence"):
        Listing(
            "GONEUSDT",
            "binance",
            "spot",
            dt(2020),
            dt(2021),
            "zero",
            None,
        )

    zero = Listing(
        "ZEROUSDT",
        "binance",
        "spot",
        dt(2020),
        dt(2021),
        "zero",
        None,
        zero_evidence="https://example.test/issuer-liquidation",
    )
    assert zero.delist_reason == "zero"

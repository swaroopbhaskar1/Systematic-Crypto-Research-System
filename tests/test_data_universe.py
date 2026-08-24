from dataclasses import fields
from datetime import UTC, datetime

import pytest

from cq.data.universe import Listing, UniverseRegistry


def at(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def test_listing_has_only_the_domain_fields() -> None:
    assert tuple(field.name for field in fields(Listing)) == (
        "symbol",
        "exchange",
        "market_type",
        "listed_at",
        "delisted_at",
        "delist_reason",
        "successor",
    )


def test_spot_and_perp_histories_are_separate() -> None:
    registry = UniverseRegistry(
        (
            Listing("BTCUSDT", "binance", "spot", at(2018)),
            Listing("BTCUSDT", "binance", "perp", at(2020)),
        )
    )

    assert registry.universe_at(at(2019), "spot") == frozenset({"BTCUSDT"})
    assert registry.universe_at(at(2019), "perp") == frozenset()
    with pytest.raises(TypeError):
        registry.universe_at(at(2024))  # type: ignore[call-arg]


def test_future_listing_is_excluded_and_delisted_symbol_is_retained_before_exit() -> (
    None
):
    registry = UniverseRegistry(
        (
            Listing(
                "OLDUSDT",
                "binance",
                "spot",
                at(2019),
                at(2021),
                "delisted",
            ),
            Listing("NEWUSDT", "binance", "spot", at(2025)),
        )
    )

    assert registry.universe_at(at(2020), "spot") == frozenset({"OLDUSDT"})
    assert registry.universe_at(at(2021), "spot") == frozenset()
    assert registry.universe_at(at(2024), "spot") == frozenset()


def test_zero_requires_explicit_evidence_and_is_not_inferred_from_delisting() -> None:
    delisted = Listing(
        "VENUEEXITUSDT",
        "binance",
        "spot",
        at(2020),
        at(2021),
        "delisted",
    )
    assert delisted.delist_reason == "delisted"

    with pytest.raises(ValueError, match="explicit evidence"):
        Listing(
            "GONEUSDT",
            "binance",
            "spot",
            at(2020),
            at(2021),
            "zero",
        )

    confirmed = Listing(
        "ZEROUSDT",
        "binance",
        "spot",
        at(2020),
        at(2021),
        "zero",
        zero_evidence="https://issuer.example/liquidation",
    )
    assert confirmed.delist_reason == "zero"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symbol": ""}, "non-empty"),
        ({"market_type": "option"}, "market type"),
        ({"delist_reason": "unknown"}, "delist reason"),
        ({"listed_at": datetime(2020, 1, 1)}, "listed_at"),  # noqa: DTZ001
        (
            {"delisted_at": at(2021), "delist_reason": None},
            "set together",
        ),
        ({"delist_reason": "delisted"}, "set together"),
        (
            {
                "delisted_at": at(2021),
                "delist_reason": "delisted",
                "zero_evidence": "not applicable",
            },
            "only valid",
        ),
        (
            {"delisted_at": at(2021), "delist_reason": "migrated"},
            "successor",
        ),
        ({"successor": "NEWUSDT"}, "only valid"),
        (
            {
                "delisted_at": datetime(2021, 1, 1),  # noqa: DTZ001
                "delist_reason": "delisted",
            },
            "delisted_at",
        ),
        (
            {"delisted_at": at(2019), "delist_reason": "delisted"},
            "precede",
        ),
    ],
)
def test_listing_rejects_inconsistent_domain_states(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "symbol": "BTCUSDT",
        "exchange": "binance",
        "market_type": "spot",
        "listed_at": at(2020),
        "delisted_at": None,
        "delist_reason": None,
        "successor": None,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        Listing(**values)  # type: ignore[arg-type]


def test_registry_rejects_duplicate_identity_and_invalid_queries() -> None:
    listing = Listing("BTCUSDT", "binance", "spot", at(2020))
    with pytest.raises(ValueError, match="duplicate"):
        UniverseRegistry([listing, listing])

    registry = UniverseRegistry([listing])
    with pytest.raises(ValueError, match="timezone-aware"):
        registry.universe_at(datetime(2021, 1, 1), "spot")  # noqa: DTZ001
    with pytest.raises(ValueError, match="market type"):
        registry.universe_at(at(2021), "option")  # type: ignore[arg-type]

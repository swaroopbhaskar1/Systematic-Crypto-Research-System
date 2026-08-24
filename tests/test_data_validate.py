from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
import pytest

from cq.data.validate import (
    DataValidationError,
    funding_coverage_count,
    validate_frame,
)

DAY_MS = 86_400_000
START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)


@dataclass(frozen=True)
class Bounds:
    symbol: str
    market_type: Literal["spot", "perp"]
    listed_at: datetime
    delisted_at: datetime | None


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": START + offset,
                "symbol": "BTCUSDT",
                "market_type": "perp",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 10.0,
                "quote_volume": 1_000.0,
                "funding_8h": 0.001,
                "in_universe": True,
                "asof": START + 2 * DAY_MS,
            }
            for offset in (0, DAY_MS)
        ]
    )


def bounds(
    *,
    listed_day: int = 1,
    delisted_day: int | None = 3,
) -> Bounds:
    return Bounds(
        symbol="BTCUSDT",
        market_type="perp",
        listed_at=datetime(2024, 1, listed_day, tzinfo=UTC),
        delisted_at=(
            datetime(2024, 1, delisted_day, tzinfo=UTC)
            if delisted_day is not None
            else None
        ),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("duplicate", "duplicate"),
        ("internal_gap", "gap"),
        ("boundary_gap", "gap"),
        ("zero_price", "price"),
        ("missing_price", "price"),
        ("negative_volume", "volume"),
        ("negative_quote_volume", "volume"),
        ("funding_at_limit", "funding"),
    ],
)
def test_invalid_data_fails_loudly(mutate: str, message: str) -> None:
    data = frame()
    listing = bounds()
    if mutate == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif mutate == "internal_gap":
        data.loc[1, "ts"] = START + 2 * DAY_MS
        listing = bounds(delisted_day=4)
    elif mutate == "boundary_gap":
        data = data.iloc[[1]].reset_index(drop=True)
    elif mutate == "zero_price":
        data.loc[0, "close"] = 0.0
    elif mutate == "missing_price":
        data.loc[0, "open"] = None
    elif mutate == "negative_volume":
        data.loc[0, "volume"] = -1.0
    elif mutate == "negative_quote_volume":
        data.loc[0, "quote_volume"] = -1.0
    elif mutate == "funding_at_limit":
        data.loc[0, "funding_8h"] = -0.05

    with pytest.raises(DataValidationError, match=message):
        validate_frame(data, listings=[listing], timeframe="1d")


def test_survivorship_warning_and_ratio_use_listing_bounds() -> None:
    live = [
        Bounds(
            symbol=f"TOKEN{index}USDT",
            market_type="spot",
            listed_at=datetime(2020, 1, 1, tzinfo=UTC),
            delisted_at=None,
        )
        for index in range(20)
    ]

    with pytest.warns(RuntimeWarning, match="survivorship"):
        report = validate_frame(
            pd.DataFrame(columns=frame().columns),
            listings=live,
            timeframe="1d",
        )

    assert report.live_symbol_count == 20
    assert report.delisted_symbol_count == 0
    assert report.delisted_live_ratio == 0.0


def test_validation_rejects_unsupported_schema_and_missing_delisted_data() -> None:
    with pytest.raises(DataValidationError, match="unsupported timeframe"):
        validate_frame(frame(), listings=[], timeframe="5m")

    with pytest.raises(DataValidationError, match="missing columns"):
        validate_frame(
            frame().drop(columns="close"),
            listings=[],
            timeframe="1d",
        )

    with pytest.raises(DataValidationError, match="no bars"):
        validate_frame(
            frame().iloc[0:0],
            listings=[bounds()],
            timeframe="1d",
        )


def test_validation_checks_listing_bound_timezone() -> None:
    naive = Bounds(
        symbol="BTCUSDT",
        market_type="perp",
        listed_at=datetime(2024, 1, 1),  # noqa: DTZ001
        delisted_at=datetime(2024, 1, 2, tzinfo=UTC),
    )

    with pytest.raises(DataValidationError, match="timezone-aware"):
        validate_frame(frame(), listings=[naive], timeframe="1d")


def test_live_and_short_lived_listing_reports_are_valid() -> None:
    with pytest.warns(RuntimeWarning, match="survivorship"):
        live_report = validate_frame(
            frame(), listings=[bounds(delisted_day=None)], timeframe="1d"
        )
    short_lived = Bounds(
        symbol="BTCUSDT",
        market_type="perp",
        listed_at=datetime(2024, 1, 1, 12, tzinfo=UTC),
        delisted_at=datetime(2024, 1, 1, 13, tzinfo=UTC),
    )
    short_data = frame().iloc[[0]].copy()
    short_data["in_universe"] = False
    short_report = validate_frame(short_data, listings=[short_lived], timeframe="1d")

    assert live_report.live_symbol_count == 1
    assert short_report.delisted_symbol_count == 1
    assert short_report.delisted_live_ratio == float("inf")


def test_empty_funding_history_has_zero_contiguous_coverage() -> None:
    assert funding_coverage_count(pd.Series(dtype="float64")) == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("market_type", "option", "market type"),
        ("asof", START - 1, "asof"),
        ("asof", None, "asof"),
        ("in_universe", None, "in_universe"),
    ],
)
def test_validation_rejects_ambiguous_pit_rows(
    column: str, value: object, message: str
) -> None:
    data = frame()
    if value is None:
        data[column] = data[column].astype("object")
    data.loc[0, column] = value

    with pytest.raises(DataValidationError, match=message):
        validate_frame(data, listings=[], timeframe="1d")


def test_validation_rejects_spot_funding_values() -> None:
    data = frame()
    data["market_type"] = "spot"

    with pytest.raises(DataValidationError, match="spot.*funding"):
        validate_frame(data, listings=[], timeframe="1d")


def test_delisting_bound_is_exclusive_for_gap_validation() -> None:
    data = frame().iloc[[0]].reset_index(drop=True)

    report = validate_frame(data, listings=[bounds(delisted_day=2)], timeframe="1d")

    assert report.delisted_symbol_count == 1


def test_validation_rejects_membership_at_delisting_timestamp() -> None:
    data = frame()

    with pytest.raises(DataValidationError, match="outside listing bounds"):
        validate_frame(data, listings=[bounds(delisted_day=2)], timeframe="1d")

from datetime import UTC, datetime

import pandas as pd
import pytest

from cq.data.universe import Listing
from cq.data.validate import (
    DataValidationError,
    funding_coverage_count,
    validate_frame,
)

DAY_MS = 86_400_000
EIGHT_HOURS_MS = 28_800_000
START = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": START,
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
                "asof": START + DAY_MS,
            },
            {
                "ts": START + DAY_MS,
                "symbol": "BTCUSDT",
                "market_type": "perp",
                "open": 101.0,
                "high": 103.0,
                "low": 100.0,
                "close": 102.0,
                "volume": 11.0,
                "quote_volume": 1_100.0,
                "funding_8h": 0.002,
                "in_universe": True,
                "asof": START + 2 * DAY_MS,
            },
        ]
    )


def listing() -> Listing:
    return Listing(
        "BTCUSDT",
        "binance",
        "perp",
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
        "delisted",
        None,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate", "duplicate"),
        ("gap", "gap"),
        ("price", "nonpositive"),
        ("volume", "negative volume"),
        ("funding", "funding"),
    ],
)
def test_invalid_market_data_fails(mutation: str, message: str) -> None:
    data = valid_frame()
    listings = [listing()]
    if mutation == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif mutation == "gap":
        data.loc[1, "ts"] = START + 2 * DAY_MS
        listings = [
            Listing(
                "BTCUSDT",
                "binance",
                "perp",
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 4, tzinfo=UTC),
                "delisted",
                None,
            )
        ]
    elif mutation == "price":
        data.loc[0, "close"] = 0.0
    elif mutation == "volume":
        data.loc[0, "volume"] = -1.0
    elif mutation == "funding":
        data.loc[0, "funding_8h"] = 0.051

    with pytest.raises(DataValidationError, match=message):
        validate_frame(data, listings=listings, timeframe="1d")


def test_survivorship_ratio_below_five_percent_warns_and_is_reported() -> None:
    live = [
        Listing(
            f"TOKEN{i}USDT",
            "binance",
            "spot",
            datetime(2020, 1, 1, tzinfo=UTC),
            None,
            None,
            None,
        )
        for i in range(20)
    ]

    with pytest.warns(RuntimeWarning, match="survivorship"):
        report = validate_frame(
            pd.DataFrame(columns=valid_frame().columns),
            listings=live,
            timeframe="1d",
        )

    assert report.delisted_live_ratio == 0.0


def test_funding_coverage_requires_270_contiguous_eight_hour_bars() -> None:
    timestamps = pd.Series(
        [START + i * EIGHT_HOURS_MS for i in range(270)], dtype="int64"
    )
    assert funding_coverage_count(timestamps) == 270

    broken = timestamps.drop(index=135).reset_index(drop=True)
    assert funding_coverage_count(broken) < 270

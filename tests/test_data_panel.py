from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from cq.data.panel import Panel


def timestamp(day: int) -> int:
    return int(datetime(2024, 1, day, tzinfo=UTC).timestamp() * 1000)


def long_panel(include_future: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "ts": timestamp(1),
            "symbol": "BTCUSDT",
            "market_type": "spot",
            "close": 100.0,
            "volume": 10.0,
            "in_universe": True,
            "asof": timestamp(2),
        },
        {
            "ts": timestamp(2),
            "symbol": "BTCUSDT",
            "market_type": "spot",
            "close": 101.0,
            "volume": 11.0,
            "in_universe": True,
            "asof": timestamp(3),
        },
    ]
    if include_future:
        rows.extend(
            [
                {
                    "ts": timestamp(3),
                    "symbol": "BTCUSDT",
                    "market_type": "spot",
                    "close": 102.0,
                    "volume": 12.0,
                    "in_universe": True,
                    "asof": timestamp(4),
                },
                {
                    "ts": timestamp(3),
                    "symbol": "FUTUREUSDT",
                    "market_type": "spot",
                    "close": 1.0,
                    "volume": 2.0,
                    "in_universe": True,
                    "asof": timestamp(4),
                },
            ]
        )
    return pd.DataFrame(rows)


def test_panel_is_wide_market_scoped_and_exposes_fields() -> None:
    panel = Panel.from_long(long_panel(False), market_type="spot")

    assert panel.market_type == "spot"
    assert panel.data.columns.names == ["field", "symbol"]
    assert panel.field("close").loc[timestamp(1), "BTCUSDT"] == 100.0
    assert bool(panel.universe_mask().loc[timestamp(2), "BTCUSDT"])


def test_historical_slice_is_unchanged_by_future_listings() -> None:
    original = Panel.from_long(long_panel(False), market_type="spot")
    extended = Panel.from_long(long_panel(True), market_type="spot")

    historical = extended.slice(timestamp(1), timestamp(2))

    pd.testing.assert_frame_equal(
        historical.data,
        original.slice(timestamp(1), timestamp(2)).data,
    )
    if "FUTUREUSDT" in historical.symbols:
        assert historical.field("close")["FUTUREUSDT"].isna().all()
        assert not historical.universe_mask()["FUTUREUSDT"].any()

    assert np.isnan(extended.field("close").loc[timestamp(1), "FUTUREUSDT"])
    assert not bool(extended.universe_mask().loc[timestamp(1), "FUTUREUSDT"])


def test_out_of_universe_values_are_masked_not_backfilled() -> None:
    data = long_panel(True)
    prelisting = pd.DataFrame(
        [
            {
                "ts": timestamp(1),
                "symbol": "FUTUREUSDT",
                "market_type": "spot",
                "close": 0.5,
                "volume": 1.0,
                "in_universe": False,
                "asof": timestamp(2),
            }
        ]
    )
    panel = Panel.from_long(
        pd.concat([data, prelisting], ignore_index=True),
        market_type="spot",
    )

    assert np.isnan(panel.field("close").loc[timestamp(1), "FUTUREUSDT"])
    assert not bool(panel.universe_mask().loc[timestamp(1), "FUTUREUSDT"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        ("missing_key", "missing panel columns"),
        ("missing_membership", "in_universe"),
        ("other_market", "other market types"),
        ("duplicate", "duplicate panel logical key"),
    ],
)
def test_panel_rejects_ambiguous_long_data(mutate: str, message: str) -> None:
    data = long_panel(False)
    if mutate == "missing_key":
        data = data.drop(columns="symbol")
    elif mutate == "missing_membership":
        data = data.drop(columns="in_universe")
    elif mutate == "other_market":
        data.loc[0, "market_type"] = "perp"
    elif mutate == "duplicate":
        data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=message):
        Panel.from_long(data, market_type="spot")


def test_panel_rejects_reversed_slice_and_unknown_field() -> None:
    panel = Panel.from_long(long_panel(False), market_type="spot")

    with pytest.raises(ValueError, match="end"):
        panel.slice(timestamp(2), timestamp(1))
    with pytest.raises(KeyError, match="unknown"):
        panel.field("unknown")

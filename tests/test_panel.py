from datetime import UTC, datetime

import numpy as np
import pandas as pd

from cq.data.panel import Panel


def ms(day: int) -> int:
    return int(datetime(2024, 1, day, tzinfo=UTC).timestamp() * 1000)


def rows(include_future: bool) -> pd.DataFrame:
    values: list[dict[str, object]] = [
        {
            "ts": ms(1),
            "symbol": "BTCUSDT",
            "market_type": "spot",
            "close": 100.0,
            "volume": 10.0,
            "in_universe": True,
        },
        {
            "ts": ms(2),
            "symbol": "BTCUSDT",
            "market_type": "spot",
            "close": 101.0,
            "volume": 11.0,
            "in_universe": True,
        },
    ]
    if include_future:
        values.extend(
            [
                {
                    "ts": ms(3),
                    "symbol": "BTCUSDT",
                    "market_type": "spot",
                    "close": 102.0,
                    "volume": 12.0,
                    "in_universe": True,
                },
                {
                    "ts": ms(3),
                    "symbol": "FUTUREUSDT",
                    "market_type": "spot",
                    "close": 1.0,
                    "volume": 2.0,
                    "in_universe": True,
                },
            ]
        )
    return pd.DataFrame(values)


def test_panel_slice_is_future_invariant() -> None:
    original = Panel.from_long(rows(False), market_type="spot")
    extended = Panel.from_long(rows(True), market_type="spot")

    pd.testing.assert_frame_equal(
        original.slice(ms(1), ms(2)).data,
        extended.slice(ms(1), ms(2)).data,
    )


def test_future_symbols_are_absent_or_nan_and_membership_false() -> None:
    panel = Panel.from_long(rows(True), market_type="spot")
    historical = panel.slice(ms(1), ms(2))

    if "FUTUREUSDT" in historical.symbols:
        assert historical.field("close")["FUTUREUSDT"].isna().all()
        assert not historical.universe_mask()["FUTUREUSDT"].any()

    assert panel.data.columns.names == ["field", "symbol"]
    assert bool(panel.universe_mask().loc[ms(1), "BTCUSDT"])
    assert np.isnan(panel.field("close").loc[ms(1), "FUTUREUSDT"])

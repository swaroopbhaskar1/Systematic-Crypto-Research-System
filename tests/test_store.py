from pathlib import Path

import pandas as pd

from cq.data.store import ParquetStore

TS = 1_700_000_000_000


def frame(close: float, asof: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts": TS,
                "symbol": "BTCUSDT",
                "market_type": "spot",
                "open": 99.0,
                "high": 102.0,
                "low": 98.0,
                "close": close,
                "volume": 10.0,
                "quote_volume": 1_000.0,
                "funding_8h": None,
                "in_universe": True,
                "asof": asof,
            }
        ]
    )


def test_writes_are_immutable_append_only_and_have_unique_names(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    first = store.write(frame(100.0, TS + 10), timeframe="1d")
    first_bytes = first.read_bytes()
    second = store.write(frame(100.0, TS + 10), timeframe="1d")

    assert first != second
    assert first.exists() and second.exists()
    assert first.read_bytes() == first_bytes
    assert "timeframe=1d" in str(first)
    assert "year=2023" in str(first)


def test_read_selects_latest_revision_eligible_at_query_time(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    store.write(frame(100.0, TS + 10), timeframe="1d")
    store.write(frame(101.0, TS + 20), timeframe="1d")

    before = store.read(timeframe="1d", query_ts=TS + 15)
    after = store.read(timeframe="1d", query_ts=TS + 25)

    assert before["close"].tolist() == [100.0]
    assert after["close"].tolist() == [101.0]
    assert before["asof"].max() <= TS + 15
    assert after["asof"].max() <= TS + 25

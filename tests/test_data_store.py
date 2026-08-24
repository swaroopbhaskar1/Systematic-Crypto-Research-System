from pathlib import Path

import pandas as pd
import pytest

from cq.data.store import ParquetStore

TS = 1_700_000_000_000


def market_frame(
    close: float,
    asof: int,
    *,
    mark_price: float | None = None,
) -> pd.DataFrame:
    row: dict[str, object] = {
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
    if mark_price is not None:
        row["mark_price"] = mark_price
    return pd.DataFrame([row])


def test_append_never_overwrites_an_existing_partition_file(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path)
    first = store.write(market_frame(100.0, TS + 10), timeframe="1d")
    original = first.read_bytes()
    second = store.write(market_frame(100.0, TS + 10), timeframe="1d")

    assert first != second
    assert first.read_bytes() == original
    assert first.parent == tmp_path / "timeframe=1d" / "year=2023"
    assert first.name.startswith("part-")
    assert first.suffix == ".parquet"
    assert len(tuple(first.parent.glob("part-*.parquet"))) == 2


def test_corrections_append_and_reads_are_point_in_time(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path)
    store.write(market_frame(100.0, TS + 10), timeframe="1d")
    store.write(market_frame(101.0, TS + 20), timeframe="1d")

    before_correction = store.read(timeframe="1d", query_ts=TS + 15)
    after_correction = store.read(timeframe="1d", query_ts=TS + 20)
    before_any_knowledge = store.read(timeframe="1d", query_ts=TS + 9)

    assert before_correction[["close", "asof"]].values.tolist() == [
        [100.0, float(TS + 10)]
    ]
    assert after_correction[["close", "asof"]].values.tolist() == [
        [101.0, float(TS + 20)]
    ]
    assert before_any_knowledge.empty


def test_optional_availability_driven_fields_round_trip(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path)
    store.write(
        market_frame(100.0, TS + 10, mark_price=100.5),
        timeframe="1d",
    )

    result = store.read(timeframe="1d", query_ts=TS + 10)

    assert result.loc[0, "mark_price"] == 100.5


def test_empty_store_read_is_schema_stable_and_requires_timeframe(
    tmp_path: Path,
) -> None:
    store = ParquetStore(tmp_path)

    result = store.read(timeframe="1d", query_ts=10)

    assert result.empty
    assert result.columns.tolist() == [
        "ts",
        "symbol",
        "market_type",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "funding_8h",
        "in_universe",
        "asof",
    ]
    with pytest.raises(ValueError, match="timeframe"):
        store.read(timeframe="", query_ts=10)


@pytest.mark.parametrize(
    ("mutate", "timeframe", "message"),
    [
        ("none", "", "timeframe"),
        ("missing", "1d", "missing store columns"),
        ("reserved", "1d", "reserved store columns"),
        ("empty", "1d", "empty frame"),
        ("duplicate", "1d", "duplicate logical revision"),
        ("years", "1d", "one UTC year"),
    ],
)
def test_store_rejects_invalid_appends(
    tmp_path: Path, mutate: str, timeframe: str, message: str
) -> None:
    data = market_frame(100.0, TS + 10)
    if mutate == "missing":
        data = data.drop(columns="close")
    elif mutate == "reserved":
        data["_file_order"] = 0
    elif mutate == "empty":
        data = data.iloc[0:0]
    elif mutate == "duplicate":
        data = pd.concat([data, data], ignore_index=True)
    elif mutate == "years":
        other_year = data.copy()
        other_year["ts"] = 1_735_689_600_000
        other_year["asof"] = 1_735_689_600_001
        data = pd.concat([data, other_year], ignore_index=True)

    with pytest.raises(ValueError, match=message):
        ParquetStore(tmp_path).write(data, timeframe=timeframe)


def test_failed_parquet_write_removes_reserved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(_self: pd.DataFrame, _handle: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="disk full"):
        ParquetStore(tmp_path).write(market_frame(100.0, TS + 10), timeframe="1d")

    assert not tuple(tmp_path.rglob("part-*.parquet"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("market_type", "option", "market type"),
        ("asof", TS - 1, "asof"),
        ("asof", None, "asof"),
    ],
)
def test_store_rejects_invalid_market_identity_and_asof(
    tmp_path: Path, column: str, value: object, message: str
) -> None:
    data = market_frame(100.0, TS + 1)
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        ParquetStore(tmp_path).write(data, timeframe="1d")

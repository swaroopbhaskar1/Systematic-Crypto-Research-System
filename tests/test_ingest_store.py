"""Adversarial tests for archive-driven ingestion into the Parquet store.

Every test here is offline.  The Binance archive is served by an
``httpx.MockTransport`` that answers S3 listings and object downloads from an
in-memory key/value map, so a test that reaches the real network would fail
rather than quietly succeed.
"""

import csv
import io
import json
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

import httpx
import pandas as pd
import pytest
import scripts.ingest as ingest_cli

from cq.data import ingest_store
from cq.data.ingest import (
    BINANCE_ARCHIVE_URL,
    BINANCE_S3_URL,
    BINANCE_SPOT_MARKET_DATA_URL,
    BinanceArchiveClient,
    fetch_ohlcv_archives,
    kline_archive_key,
    monthly_kline_archive_key,
    plan_kline_archives,
    raw_cache_key,
)
from cq.data.ingest_store import (
    IngestSummary,
    SymbolCoverage,
    aggregate_funding,
    build_archive_universe,
    coverage_listing,
    ingest_to_store,
    resolve_symbols,
    summary_json,
)
from cq.data.panel import Panel, add_execution_features
from cq.data.store import REQUIRED_COLUMNS, ParquetStore
from cq.data.universe import Listing, MarketType, UniverseRegistry

DAY_MS = 86_400_000
EIGHT_HOURS_MS = 28_800_000


def day_ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def days_between(first: date, last: date) -> tuple[date, ...]:
    count = (last - first).days + 1
    return tuple(first + timedelta(days=offset) for offset in range(count))


def month_days(year: int, month: int) -> tuple[date, ...]:
    first = date(year, month, 1)
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return days_between(first, last)


def _zipped(rows: Sequence[Sequence[str]], name: str) -> bytes:
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(name, text.getvalue())
    return payload.getvalue()


def kline_zip(days: Sequence[date], *, base: float = 100.0) -> bytes:
    rows: list[list[str]] = []
    for offset, day in enumerate(days):
        opened = day_ms(day)
        price = base + offset
        rows.append(
            [
                str(opened),
                str(price),
                str(price + 1.0),
                str(price - 1.0),
                str(price + 0.5),
                "10",
                str(opened + DAY_MS - 1),
                str(1_000.0 + offset),
                "5",
                "5",
                "500",
                "0",
            ]
        )
    return _zipped(rows, "klines.csv")


def funding_zip(prints: Mapping[int, float]) -> bytes:
    rows: list[list[str]] = [
        ["calc_time", "funding_interval_hours", "last_funding_rate"]
    ]
    rows.extend(
        [str(timestamp), "8", repr(rate)] for timestamp, rate in sorted(prints.items())
    )
    return _zipped(rows, "funding.csv")


def eight_hourly(day: date, rates: Sequence[float]) -> dict[int, float]:
    opened = day_ms(day)
    return {
        opened + index * EIGHT_HOURS_MS: rate for index, rate in enumerate(rates)
    }


class FakeArchive:
    """An in-memory Binance archive served over ``httpx.MockTransport``."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.requests: list[str] = []
        self.downloads: list[str] = []
        self.missing: set[str] = set()
        self.malformed_listing_prefixes: set[str] = set()
        self.trading_symbols: tuple[str, ...] = ()

    def add(self, key: str, payload: bytes) -> None:
        self.objects[key] = payload

    def add_monthly_klines(
        self,
        market_type: MarketType,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
        *,
        base: float = 100.0,
    ) -> str:
        key = monthly_kline_archive_key(
            market_type, symbol, timeframe, date(year, month, 1)
        )
        self.add(key, kline_zip(month_days(year, month), base=base))
        return key

    def add_daily_klines(
        self,
        market_type: MarketType,
        symbol: str,
        timeframe: str,
        day: date,
        *,
        base: float = 100.0,
    ) -> str:
        key = kline_archive_key(market_type, symbol, timeframe, day)
        self.add(key, kline_zip((day,), base=base))
        return key

    def add_funding(self, symbol: str, month: date, prints: Mapping[int, float]) -> str:
        key = (
            f"data/futures/um/monthly/fundingRate/{symbol}/"
            f"{symbol}-fundingRate-{month:%Y-%m}.zip"
        )
        self.add(key, funding_zip(prints))
        return key

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self, cache_root: Path) -> BinanceArchiveClient:
        return BinanceArchiveClient(
            transport=self.transport(),
            cache_root=cache_root,
            attempts=1,
            initial_delay=0.0,
            sleep=lambda _seconds: None,
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url.startswith(BINANCE_S3_URL):
            return httpx.Response(200, content=self._listing(request))
        if url.startswith(BINANCE_SPOT_MARKET_DATA_URL):
            return httpx.Response(200, json=self._exchange_info())
        if not url.startswith(f"{BINANCE_ARCHIVE_URL}/"):
            raise AssertionError(f"unexpected network call: {url}")
        key = unquote(url[len(BINANCE_ARCHIVE_URL) + 1 :])
        self.downloads.append(key)
        payload = self.objects.get(key)
        if payload is None or key in self.missing:
            return httpx.Response(404)
        return httpx.Response(200, content=payload)

    def _listing(self, request: httpx.Request) -> bytes:
        prefix = request.url.params["prefix"]
        delimiter = request.url.params.get("delimiter")
        if prefix in self.malformed_listing_prefixes:
            return b"<ListBucketResult><unclosed>"
        matched = sorted(key for key in self.objects if key.startswith(prefix))
        if delimiter is None:
            body = "".join(
                f"<Contents><Key>{key}</Key><Size>{len(self.objects[key])}</Size>"
                "</Contents>"
                for key in matched
            )
        else:
            children = sorted(
                {
                    f"{prefix}{key[len(prefix) :].split('/', 1)[0]}/"
                    for key in matched
                    if "/" in key[len(prefix) :]
                }
            )
            body = "".join(
                f"<CommonPrefixes><Prefix>{child}</Prefix></CommonPrefixes>"
                for child in children
            )
        return (
            "<ListBucketResult><IsTruncated>false</IsTruncated>"
            f"{body}</ListBucketResult>"
        ).encode()

    def _exchange_info(self) -> dict[str, object]:
        return {
            "symbols": [
                {"symbol": symbol, "quoteAsset": "USDT", "status": "TRADING"}
                for symbol in self.trading_symbols
            ]
        }


def spot_archive(tmp_path: Path) -> tuple[FakeArchive, BinanceArchiveClient]:
    archive = FakeArchive()
    return archive, archive.client(tmp_path / "cache")


# --------------------------------------------------------------------------
# Monthly / daily archive planning and the seam between them
# --------------------------------------------------------------------------


def test_whole_months_use_monthly_archives_and_partial_edges_use_daily() -> None:
    planned = plan_kline_archives(
        "perp",
        "BTCUSDT",
        "1d",
        day_ms(date(2024, 1, 30)),
        day_ms(date(2024, 3, 3)),
    )

    keys = [item.key for item in planned]
    periods = [item.period for item in planned]

    assert keys[:2] == [
        "data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-30.zip",
        "data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-01-31.zip",
    ]
    assert keys[2] == (
        "data/futures/um/monthly/klines/BTCUSDT/1d/BTCUSDT-1d-2024-02.zip"
    )
    assert keys[3:] == [
        "data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-03-01.zip",
        "data/futures/um/daily/klines/BTCUSDT/1d/BTCUSDT-1d-2024-03-02.zip",
    ]
    assert periods == ["daily", "daily", "monthly", "daily", "daily"]


def test_monthly_and_daily_seam_produces_no_duplicate_or_missing_bar(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)
    for day in days_between(date(2024, 2, 1), date(2024, 2, 15)):
        archive.add_daily_klines("spot", "BTCUSDT", "1d", day, base=200.0)

    with client:
        frame = fetch_ohlcv_archives(
            "binance",
            "BTCUSDT",
            "1d",
            day_ms(date(2024, 1, 1)),
            day_ms(date(2024, 2, 16)),
            "spot",
            archive_client=client,
        )

    expected = [
        day_ms(day) for day in days_between(date(2024, 1, 1), date(2024, 2, 15))
    ]
    assert frame["ts"].tolist() == expected
    assert not frame["ts"].duplicated().any()
    assert len(archive.downloads) == 16


def test_monthly_archive_cache_key_never_collides_with_the_first_daily_key(
    tmp_path: Path,
) -> None:
    monthly = raw_cache_key(
        "binance", "spot", "BTCUSDT", "1d", date(2024, 1, 1), period="monthly"
    )
    daily = raw_cache_key("binance", "spot", "BTCUSDT", "1d", date(2024, 1, 1))

    assert monthly != daily
    assert "2024-01" in str(monthly)
    assert "2024-01-01" in str(daily)

    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)
    with client:
        first = fetch_ohlcv_archives(
            "binance",
            "BTCUSDT",
            "1d",
            day_ms(date(2024, 1, 1)),
            day_ms(date(2024, 2, 1)),
            "spot",
            archive_client=client,
        )
        second = fetch_ohlcv_archives(
            "binance",
            "BTCUSDT",
            "1d",
            day_ms(date(2024, 1, 1)),
            day_ms(date(2024, 2, 1)),
            "spot",
            archive_client=client,
        )

    pd.testing.assert_frame_equal(first, second)
    assert len(archive.downloads) == 1


# --------------------------------------------------------------------------
# Survivorship: the universe is built from archive listings
# --------------------------------------------------------------------------


def _three_symbol_archive(tmp_path: Path) -> tuple[FakeArchive, BinanceArchiveClient]:
    archive, client = spot_archive(tmp_path)
    for month in (1, 2, 3):
        archive.add_monthly_klines("spot", "LIVEUSDT", "1d", 2024, month)
    for month in (1, 2):
        archive.add_monthly_klines("spot", "DEADUSDT", "1d", 2024, month)
    for month in (2, 3):
        archive.add_monthly_klines("spot", "LATEUSDT", "1d", 2024, month)
    archive.trading_symbols = ("LIVEUSDT", "LATEUSDT")
    return archive, client


def test_universe_comes_from_archive_listings_not_live_trading_symbols(
    tmp_path: Path,
) -> None:
    archive, client = _three_symbol_archive(tmp_path)

    with client:
        registry = build_archive_universe("spot", end=date(2024, 4, 1), client=client)

    members = registry.universe_at(datetime(2024, 1, 15, tzinfo=UTC), "spot")

    assert "DEADUSDT" in members
    assert "DEADUSDT" not in archive.trading_symbols
    assert not any(
        url.startswith(BINANCE_SPOT_MARKET_DATA_URL) for url in archive.requests
    )


def test_archive_that_stops_mid_range_delists_and_leaves_the_universe(
    tmp_path: Path,
) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        registry = build_archive_universe("spot", end=date(2024, 4, 1), client=client)

    before = registry.universe_at(datetime(2024, 2, 20, tzinfo=UTC), "spot")
    after = registry.universe_at(datetime(2024, 3, 20, tzinfo=UTC), "spot")

    assert "DEADUSDT" in before
    assert "DEADUSDT" not in after
    assert "LIVEUSDT" in after


def test_archive_that_starts_mid_range_is_absent_from_the_earlier_universe(
    tmp_path: Path,
) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        registry = build_archive_universe("spot", end=date(2024, 4, 1), client=client)

    assert "LATEUSDT" not in registry.universe_at(
        datetime(2024, 1, 31, 23, 59, tzinfo=UTC), "spot"
    )
    assert "LATEUSDT" in registry.universe_at(datetime(2024, 2, 1, tzinfo=UTC), "spot")


def test_delist_reason_is_only_asserted_where_the_archive_substantiates_it(
    tmp_path: Path,
) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        listings = ingest_store.archive_listings(
            "spot", end=date(2024, 4, 1), client=client
        )

    by_symbol = {listing.symbol: listing for listing in listings}

    assert by_symbol["DEADUSDT"].delisted_at == datetime(2024, 3, 1, tzinfo=UTC)
    assert by_symbol["DEADUSDT"].delist_reason == "delisted"
    assert by_symbol["DEADUSDT"].successor is None
    assert by_symbol["LIVEUSDT"].delisted_at is None
    assert by_symbol["LIVEUSDT"].delist_reason is None
    assert by_symbol["LATEUSDT"].listed_at == datetime(2024, 2, 1, tzinfo=UTC)


def test_listing_bounds_override_a_fully_priced_bar() -> None:
    """A complete-looking bar outside its listing window is still not tradable."""
    registry = UniverseRegistry(
        [
            Listing(
                symbol="WINDOWUSDT",
                exchange="binance",
                market_type="spot",
                listed_at=datetime(2024, 2, 1, tzinfo=UTC),
                delisted_at=datetime(2024, 3, 1, tzinfo=UTC),
                delist_reason="delisted",
            )
        ]
    )
    bars = pd.DataFrame(
        {
            "ts": [
                day_ms(date(2024, 1, 15)),
                day_ms(date(2024, 2, 15)),
                day_ms(date(2024, 3, 15)),
            ],
            "symbol": "WINDOWUSDT",
            "market_type": "spot",
            "open": [10.0, 10.0, 10.0],
            "high": [11.0, 11.0, 11.0],
            "low": [9.0, 9.0, 9.0],
            "close": [10.5, 10.5, 10.5],
        }
    )

    mask = ingest_store.membership_mask(bars, registry, "spot")

    assert mask.tolist() == [False, True, False]


def test_mislabelled_archive_cannot_smuggle_a_bar_into_an_earlier_universe(
    tmp_path: Path,
) -> None:
    """An archive whose contents predate its own key must not extend history."""
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "SLIPUSDT", "1d", 2024, 2)
    archive.add_daily_klines("spot", "SLIPUSDT", "1d", date(2024, 3, 6))
    archive.add(
        kline_archive_key("spot", "SLIPUSDT", "1d", date(2024, 3, 5)),
        kline_zip((date(2024, 1, 20),)),
    )

    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 3, 15),
            client=client,
            asof=day_ms(date(2024, 4, 2)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 4, 2))
    )
    smuggled = frame.loc[frame["ts"] == day_ms(date(2024, 1, 20))]

    assert len(smuggled) == 1
    assert not bool(smuggled["in_universe"].iloc[0])
    assert bool(frame.loc[frame["ts"] >= day_ms(date(2024, 2, 1)), "in_universe"].all())


def test_survivorship_membership_is_written_to_the_store(tmp_path: Path) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            client=client,
            asof=day_ms(date(2024, 4, 2)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 4, 2))
    )
    dead = frame.loc[frame["symbol"] == "DEADUSDT"]
    late = frame.loc[frame["symbol"] == "LATEUSDT"]

    assert dead["ts"].max() == day_ms(date(2024, 2, 29))
    assert bool(dead["in_universe"].all())
    assert late["ts"].min() == day_ms(date(2024, 2, 1))
    assert not (late["ts"] < day_ms(date(2024, 2, 1))).any()


# --------------------------------------------------------------------------
# Funding join
# --------------------------------------------------------------------------


def _perp_archive(
    tmp_path: Path, prints: Mapping[int, float]
) -> tuple[FakeArchive, BinanceArchiveClient]:
    archive, client = spot_archive(tmp_path)
    for day in days_between(date(2024, 1, 1), date(2024, 1, 3)):
        archive.add_daily_klines("perp", "PERPUSDT", "1d", day)
    archive.add_funding("PERPUSDT", date(2024, 1, 1), prints)
    return archive, client


def _perp_rows(tmp_path: Path, prints: Mapping[int, float]) -> pd.DataFrame:
    _, client = _perp_archive(tmp_path, prints)
    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="perp",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 1, 4),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )
    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 2, 1))
    )
    return frame.set_index("ts").sort_index()


def test_funding_uses_only_prints_published_inside_the_bar(tmp_path: Path) -> None:
    prints = {
        **eight_hourly(date(2024, 1, 1), (0.001, 0.002, 0.003)),
        **eight_hourly(date(2024, 1, 3), (0.004,)),
    }

    rows = _perp_rows(tmp_path, prints)

    assert rows.loc[day_ms(date(2024, 1, 1)), "funding_8h"] == pytest.approx(0.002)
    assert pd.isna(rows.loc[day_ms(date(2024, 1, 2)), "funding_8h"])
    assert rows.loc[day_ms(date(2024, 1, 3)), "funding_8h"] == pytest.approx(0.004)


def test_bar_without_a_funding_print_is_nan_and_out_of_universe(
    tmp_path: Path,
) -> None:
    prints = {
        **eight_hourly(date(2024, 1, 1), (0.001, 0.002, 0.003)),
        **eight_hourly(date(2024, 1, 3), (0.004,)),
    }

    rows = _perp_rows(tmp_path, prints)

    assert not bool(rows.loc[day_ms(date(2024, 1, 2)), "in_universe"])
    assert bool(rows.loc[day_ms(date(2024, 1, 1)), "in_universe"])
    assert bool(rows.loc[day_ms(date(2024, 1, 3)), "in_universe"])


def test_zero_funding_is_preserved_and_never_confused_with_missing(
    tmp_path: Path,
) -> None:
    prints = {
        **eight_hourly(date(2024, 1, 1), (0.0, 0.0, 0.0)),
        **eight_hourly(date(2024, 1, 2), (0.0, 0.0, 0.0)),
        **eight_hourly(date(2024, 1, 3), (0.0, 0.0, 0.0)),
    }

    rows = _perp_rows(tmp_path, prints)

    assert rows["funding_8h"].tolist() == [0.0, 0.0, 0.0]
    assert rows["funding_8h"].notna().all()
    assert bool(rows["in_universe"].all())


def test_spot_rows_carry_nan_funding_without_leaving_the_universe(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)

    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 2, 1))
    )

    assert frame["funding_8h"].isna().all()
    assert bool(frame["in_universe"].all())


# --------------------------------------------------------------------------
# Store contract, partitioning, and idempotency
# --------------------------------------------------------------------------


def _long_history_archive(tmp_path: Path) -> tuple[FakeArchive, BinanceArchiveClient]:
    archive, client = spot_archive(tmp_path)
    for symbol, base in (("AAAUSDT", 100.0), ("BBBUSDT", 50.0)):
        for year, month in ((2023, 11), (2023, 12), (2024, 1), (2024, 2)):
            archive.add_monthly_klines(
                "spot", symbol, "1d", year, month, base=base
            )
    return archive, client


def test_written_frame_round_trips_through_store_features_and_panel(
    tmp_path: Path,
) -> None:
    _, client = _long_history_archive(tmp_path)
    asof = day_ms(date(2024, 3, 2))

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2023, 11, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=asof,
        )

    frame = ParquetStore(tmp_path / "store").read(timeframe="1d", query_ts=asof)

    assert list(REQUIRED_COLUMNS) == frame.columns.tolist()
    assert summary.bars_written == len(frame)

    featured = add_execution_features(frame)
    panel = Panel.from_long(featured.drop(columns=["asof"]), market_type="spot")

    assert set(panel.symbols) == {"AAAUSDT", "BBBUSDT"}
    assert bool(panel.universe_mask().to_numpy().any())


def test_multi_year_range_writes_exactly_one_file_per_year(tmp_path: Path) -> None:
    _, client = _long_history_archive(tmp_path)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2023, 11, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=day_ms(date(2024, 3, 2)),
        )

    partitions = sorted(
        path.name
        for path in (tmp_path / "store" / "timeframe=1d").iterdir()
        if path.is_dir()
    )

    assert partitions == ["year=2023", "year=2024"]
    for partition in partitions:
        files = list(
            (tmp_path / "store" / "timeframe=1d" / partition).glob("part-*.parquet")
        )
        assert len(files) == 1
    assert len(summary.files_written) == 2


def test_reingesting_the_same_range_does_not_duplicate_logical_rows(
    tmp_path: Path,
) -> None:
    archive, client = _long_history_archive(tmp_path)
    store_root = tmp_path / "store"

    with client:
        first = ingest_to_store(
            store_root,
            market_type="spot",
            timeframe="1d",
            start=date(2023, 11, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=day_ms(date(2024, 3, 2)),
        )
        downloads_after_first = len(archive.downloads)
        second = ingest_to_store(
            store_root,
            market_type="spot",
            timeframe="1d",
            start=date(2023, 11, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=day_ms(date(2024, 3, 3)),
        )

    frame = ParquetStore(store_root).read(
        timeframe="1d", query_ts=day_ms(date(2024, 3, 3))
    )

    assert first.bars_written == second.bars_written
    assert len(frame) == first.bars_written
    assert not frame.duplicated(["ts", "symbol", "market_type"]).any()
    assert frame["asof"].nunique() == 1
    assert int(frame["asof"].iloc[0]) == day_ms(date(2024, 3, 3))
    assert len(archive.downloads) == downloads_after_first


def test_one_asof_per_run_makes_a_reingest_a_distinguishable_revision(
    tmp_path: Path,
) -> None:
    _, client = _long_history_archive(tmp_path)
    asof = day_ms(date(2024, 3, 2))

    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2023, 11, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=asof,
        )

    frame = ParquetStore(tmp_path / "store").read(timeframe="1d", query_ts=asof)

    assert frame["asof"].unique().tolist() == [asof]
    assert ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=asof - 1
    ).empty


def test_default_asof_is_the_ingest_wall_clock(tmp_path: Path) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)
    before = int(datetime.now(UTC).timestamp() * 1000)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            client=client,
        )

    assert before <= summary.asof <= int(datetime.now(UTC).timestamp() * 1000)


def test_missing_days_are_absent_rather_than_fabricated(tmp_path: Path) -> None:
    archive, client = spot_archive(tmp_path)
    for day in days_between(date(2024, 1, 1), date(2024, 1, 10)):
        if day == date(2024, 1, 5):
            continue
        archive.add_daily_klines("spot", "GAPUSDT", "1d", day)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 1, 11),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 2, 1))
    )

    assert day_ms(date(2024, 1, 5)) not in frame["ts"].tolist()
    assert len(frame) == 9
    assert summary.failures == ()


# --------------------------------------------------------------------------
# Resilience: one bad symbol must not abort a multi-hundred-symbol run
# --------------------------------------------------------------------------


def test_corrupt_symbol_archive_is_reported_and_does_not_abort_the_run(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "GOODUSDT", "1d", 2024, 1)
    archive.add(
        monthly_kline_archive_key("spot", "BADUSDT", "1d", date(2024, 1, 1)),
        b"this is not a zip file",
    )

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 2, 1))
    )
    reasons = {failure.symbol: failure.reason for failure in summary.failures}

    assert set(frame["symbol"].unique()) == {"GOODUSDT"}
    assert summary.symbols_attempted == 2
    assert summary.symbols_with_data == 1
    assert "BADUSDT" in reasons
    assert "ZIP" in reasons["BADUSDT"]


def test_missing_listed_archive_is_reported_and_does_not_abort_the_run(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "GOODUSDT", "1d", 2024, 1)
    vanished = archive.add_monthly_klines("spot", "GONEUSDT", "1d", 2024, 1)
    archive.missing.add(vanished)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    reasons = {failure.symbol: failure.reason for failure in summary.failures}

    assert summary.symbols_with_data == 1
    assert "GONEUSDT" in reasons
    assert "404" in reasons["GONEUSDT"]
    assert summary.bars_written == 31


def test_symbol_without_any_archive_is_reported_rather_than_dropped_silently(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "GOODUSDT", "1d", 2024, 1)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            symbols=("GOODUSDT", "GHOSTUSDT"),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    reasons = {failure.symbol: failure.reason for failure in summary.failures}

    assert "GHOSTUSDT" in reasons
    assert "no archive" in reasons["GHOSTUSDT"]


def test_unreadable_symbol_listing_is_reported_and_does_not_abort_the_run(
    tmp_path: Path,
) -> None:
    archive = FakeArchive()
    archive.add_monthly_klines("spot", "GOODUSDT", "1d", 2024, 1)
    archive.add_monthly_klines("spot", "XMLUSDT", "1d", 2024, 1)
    archive.malformed_listing_prefixes.add("data/spot/monthly/klines/XMLUSDT/1d/")
    client = archive.client(tmp_path / "cache")

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 2, 1),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    reasons = {failure.symbol: failure.reason for failure in summary.failures}

    assert summary.symbols_with_data == 1
    assert summary.bars_written == 31
    assert "malformed" in reasons["XMLUSDT"]


def test_perp_without_any_funding_archive_is_priced_but_never_tradable(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    for day in days_between(date(2024, 1, 1), date(2024, 1, 3)):
        archive.add_daily_klines("perp", "NOFUNDUSDT", "1d", day)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="perp",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 1, 4),
            client=client,
            asof=day_ms(date(2024, 2, 1)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 2, 1))
    )

    assert summary.bars_written == 3
    assert frame["funding_8h"].isna().all()
    assert not bool(frame["in_universe"].any())


def test_ingest_closes_the_client_it_creates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)
    closed: list[bool] = []
    monkeypatch.setattr(client, "close", lambda: closed.append(True))
    monkeypatch.setattr(
        ingest_store, "BinanceArchiveClient", lambda **_kwargs: client
    )

    summary = ingest_to_store(
        tmp_path / "store",
        market_type="spot",
        timeframe="1d",
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        asof=day_ms(date(2024, 2, 1)),
    )

    assert summary.bars_written == 31
    assert closed == [True]


def test_summary_reports_attempted_counts_and_timestamp_bounds(
    tmp_path: Path,
) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            client=client,
            asof=day_ms(date(2024, 4, 2)),
        )

    assert isinstance(summary, IngestSummary)
    assert summary.symbols_attempted == 3
    assert summary.symbols_with_data == 3
    assert summary.first_ts == day_ms(date(2024, 1, 1))
    assert summary.last_ts == day_ms(date(2024, 3, 31))
    assert summary.bars_written == 91 + 60 + 60
    assert json.loads(summary_json(summary))["symbols_with_data"] == 3


def test_symbol_limit_bounds_a_first_run(tmp_path: Path) -> None:
    _, client = _three_symbol_archive(tmp_path)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 4, 1),
            limit=1,
            client=client,
            asof=day_ms(date(2024, 4, 2)),
        )

    frame = ParquetStore(tmp_path / "store").read(
        timeframe="1d", query_ts=day_ms(date(2024, 4, 2))
    )

    assert summary.symbols_attempted == 1
    assert set(frame["symbol"].unique()) == {"DEADUSDT"}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"market_type": "option"}, "market type"),
        ({"timeframe": "5m"}, "unsupported timeframe"),
        ({"end": date(2024, 1, 1)}, "end must be later"),
        ({"limit": 0}, "limit"),
    ],
)
def test_ingest_rejects_invalid_requests(
    tmp_path: Path, kwargs: dict[str, object], message: str
) -> None:
    request: dict[str, object] = {
        "market_type": "spot",
        "timeframe": "1d",
        "start": date(2024, 1, 1),
        "end": date(2024, 2, 1),
    }
    request.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ingest_to_store(tmp_path / "store", **request)  # type: ignore[arg-type]


def test_partial_symbol_failure_drops_the_whole_symbol_rather_than_half_of_it(
    tmp_path: Path,
) -> None:
    """Half a history written as if whole is a silent lie; the symbol is dropped."""
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "HALFUSDT", "1d", 2024, 1)
    archive.add(
        monthly_kline_archive_key("spot", "HALFUSDT", "1d", date(2024, 2, 1)),
        b"truncated",
    )

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2024, 1, 1),
            end=date(2024, 3, 1),
            client=client,
            asof=day_ms(date(2024, 3, 1)),
        )

    assert summary.bars_written == 0
    assert summary.symbols_with_data == 0
    assert [failure.symbol for failure in summary.failures] == ["HALFUSDT"]


def test_funding_archives_outside_the_range_are_never_downloaded(
    tmp_path: Path,
) -> None:
    archive, client = spot_archive(tmp_path)
    for day in days_between(date(2024, 6, 1), date(2024, 6, 3)):
        archive.add_daily_klines("perp", "PERPUSDT", "1d", day)
    stale = archive.add_funding(
        "PERPUSDT", date(2024, 1, 1), eight_hourly(date(2024, 1, 1), (0.001,))
    )
    wanted = archive.add_funding(
        "PERPUSDT", date(2024, 6, 1), eight_hourly(date(2024, 6, 1), (0.001,))
    )

    with client:
        ingest_to_store(
            tmp_path / "store",
            market_type="perp",
            timeframe="1d",
            start=date(2024, 6, 1),
            end=date(2024, 6, 4),
            client=client,
            asof=day_ms(date(2024, 7, 1)),
        )

    assert wanted in archive.downloads
    assert stale not in archive.downloads


def test_funding_aggregation_is_a_mean_and_is_empty_safe() -> None:
    opened = day_ms(date(2024, 1, 1))
    prints = pd.DataFrame(
        {
            "ts": [
                opened,
                opened + EIGHT_HOURS_MS,
                opened + 2 * EIGHT_HOURS_MS,
                opened + DAY_MS,
            ],
            "funding_8h": [0.001, 0.002, 0.006, -0.004],
        }
    )

    daily = aggregate_funding(prints, timeframe="1d")
    empty = aggregate_funding(
        pd.DataFrame({"ts": [], "funding_8h": []}), timeframe="1d"
    )

    assert daily["ts"].tolist() == [opened, opened + DAY_MS]
    assert daily["funding_8h"].tolist() == pytest.approx([0.003, -0.004])
    assert empty.empty
    with pytest.raises(ValueError, match="unsupported timeframe"):
        aggregate_funding(prints, timeframe="5m")


@pytest.mark.parametrize(
    ("symbols", "limit", "message"),
    [
        (("BTCUSDT",), 0, "limit"),
        ((), None, "at least one"),
    ],
)
def test_symbol_resolution_rejects_impossible_selections(
    symbols: tuple[str, ...], limit: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_symbols(
            None,  # type: ignore[arg-type]
            "spot",
            symbols,
            limit,
        )


def test_listing_cannot_be_derived_without_archive_evidence() -> None:
    empty = SymbolCoverage("GHOSTUSDT", frozenset(), None, None)

    with pytest.raises(ValueError, match="no archive coverage"):
        coverage_listing(
            empty, exchange="binance", market_type="spot", end=date(2024, 1, 1)
        )


def test_empty_ingest_writes_nothing_and_says_so(tmp_path: Path) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "BTCUSDT", "1d", 2024, 1)

    with client:
        summary = ingest_to_store(
            tmp_path / "store",
            market_type="spot",
            timeframe="1d",
            start=date(2025, 1, 1),
            end=date(2025, 2, 1),
            client=client,
            asof=day_ms(date(2025, 2, 1)),
        )

    assert summary.bars_written == 0
    assert summary.first_ts is None
    assert summary.last_ts is None
    assert summary.files_written == ()


# --------------------------------------------------------------------------
# CLI surface
# --------------------------------------------------------------------------


def test_store_cli_ingests_the_requested_slice_and_prints_the_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "AAAUSDT", "1d", 2024, 1)
    archive.add_monthly_klines("spot", "BBBUSDT", "1d", 2024, 1)

    class Injected:
        def __init__(self, **_kwargs: object) -> None:
            self._client = client

        def __enter__(self) -> BinanceArchiveClient:
            return self._client

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(ingest_store, "BinanceArchiveClient", Injected)

    exit_code = ingest_store.main(
        [
            "--store",
            str(tmp_path / "store"),
            "--market",
            "spot",
            "--timeframe",
            "1d",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
            "--symbols",
            "AAAUSDT,BBBUSDT",
            "--limit",
            "1",
        ]
    )

    rendered = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert rendered["symbols_attempted"] == 1
    assert rendered["bars_written"] == 31
    assert rendered["failures"] == []


def test_store_cli_exits_nonzero_when_a_symbol_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    archive, client = spot_archive(tmp_path)
    archive.add_monthly_klines("spot", "AAAUSDT", "1d", 2024, 1)
    archive.add(
        monthly_kline_archive_key("spot", "BADUSDT", "1d", date(2024, 1, 1)),
        b"not a zip",
    )

    class Injected:
        def __init__(self, **_kwargs: object) -> None:
            self._client = client

        def __enter__(self) -> BinanceArchiveClient:
            return self._client

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(ingest_store, "BinanceArchiveClient", Injected)

    exit_code = ingest_store.main(
        [
            "--store",
            str(tmp_path / "store"),
            "--market",
            "spot",
            "--start",
            "2024-01-01",
            "--end",
            "2024-02-01",
        ]
    )

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["failures"][0]["symbol"] == "BADUSDT"


def test_scripts_entrypoint_dispatches_both_subcommands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, tuple[str, ...]]] = []

    def fake_gate(argv: Sequence[str] | None = None) -> int:
        seen.append(("gate", tuple(argv or ())))
        return 0

    def fake_store(argv: Sequence[str] | None = None) -> int:
        seen.append(("store", tuple(argv or ())))
        return 3

    monkeypatch.setattr(ingest_cli, "archive_gate_main", fake_gate)
    monkeypatch.setattr(ingest_cli, "archive_store_main", fake_store)

    assert ingest_cli.main(["gate", "--workers", "2"]) == 0
    assert ingest_cli.main(["store", "--market", "perp"]) == 3
    assert seen == [
        ("gate", ("--workers", "2")),
        ("store", ("--market", "perp")),
    ]


def test_fake_archive_rejects_any_call_outside_the_binance_archive(
    tmp_path: Path,
) -> None:
    """The harness itself must fail loudly if a test escapes the mock."""
    archive, _ = spot_archive(tmp_path)
    http = httpx.Client(transport=archive.transport())

    with pytest.raises(AssertionError, match="unexpected network call"):
        http.get("https://example.invalid/whatever")

    assert archive.requests == ["https://example.invalid/whatever"]

import csv
import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from cq.data import ingest
from cq.data.ingest import (
    BinanceArchiveClient,
    archive_root,
    call_with_backoff,
    fetch_ohlcv,
    kline_archive_url,
    raw_cache_key,
    read_funding_archive,
    read_kline_archive,
    validate_bar_gaps,
)


def zipped_rows(rows: list[list[str]]) -> bytes:
    data = io.StringIO()
    csv.writer(data, lineterminator="\n").writerows(rows)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("bars.csv", data.getvalue())
    return payload.getvalue()


def kline_row(ts: int) -> list[str]:
    return [
        str(ts),
        "1",
        "2",
        "0.5",
        "1.5",
        "10",
        str(ts + 3_599_999),
        "15",
        "3",
        "6",
        "8",
        "0",
    ]


def test_archive_urls_keep_spot_and_perpetual_histories_separate() -> None:
    day = date(2024, 1, 2)

    spot = kline_archive_url("spot", "BTCUSDT", "1h", day)
    perp = kline_archive_url("perp", "BTCUSDT", "1h", day)

    assert "/data/spot/daily/klines/BTCUSDT/1h/" in spot
    assert "/data/futures/um/daily/klines/BTCUSDT/1h/" in perp
    assert spot.endswith("BTCUSDT-1h-2024-01-02.zip")
    assert spot != perp


def test_s3_symbol_enumeration_is_paginated_deduplicated_and_sorted(
    tmp_path: Path,
) -> None:
    pages = [
        b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
        <IsTruncated>true</IsTruncated>
        <NextContinuationToken>next token</NextContinuationToken>
        <CommonPrefixes><Prefix>data/spot/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>
        </ListBucketResult>""",
        b"""<ListBucketResult>
        <IsTruncated>false</IsTruncated>
        <CommonPrefixes><Prefix>data/spot/monthly/klines/BTCUSDT/</Prefix></CommonPrefixes>
        <CommonPrefixes><Prefix>data/spot/monthly/klines/ETHUSDT/</Prefix></CommonPrefixes>
        </ListBucketResult>""",
    ]
    seen_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.append(dict(request.url.params))
        return httpx.Response(200, content=pages[len(seen_params) - 1])

    http = httpx.Client(transport=httpx.MockTransport(handler))
    with BinanceArchiveClient(client=http, cache_root=tmp_path) as archive:
        assert archive.symbols("spot") == ("BTCUSDT", "ETHUSDT")

    assert seen_params[1]["continuation-token"] == "next token"


def test_fetch_ohlcv_caches_raw_response_by_full_identity(tmp_path: Path) -> None:
    payload = zipped_rows([kline_row(1_704_153_600_000)])
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=payload)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    archive = BinanceArchiveClient(client=http, cache_root=tmp_path)
    bounds = (
        datetime(2024, 1, 2, tzinfo=UTC),
        datetime(2024, 1, 3, tzinfo=UTC),
    )
    first = fetch_ohlcv(
        "binance", "BTCUSDT", "1h", *bounds, "spot", archive_client=archive
    )
    second = fetch_ohlcv(
        "binance", "BTCUSDT", "1h", *bounds, "spot", archive_client=archive
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(requests) == 1
    assert raw_cache_key(
        "binance", "spot", "BTCUSDT", "1h", date(2024, 1, 2)
    ) != raw_cache_key("binance", "perp", "BTCUSDT", "1h", date(2024, 1, 2))


def test_fetch_ohlcv_rejects_gap_larger_than_one_bar(tmp_path: Path) -> None:
    payload = zipped_rows([kline_row(1_704_153_600_000), kline_row(1_704_164_400_000)])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    archive = BinanceArchiveClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        cache_root=tmp_path,
    )
    with pytest.raises(ValueError, match="gap"):
        fetch_ohlcv(
            "binance",
            "BTCUSDT",
            "1h",
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 3, tzinfo=UTC),
            "spot",
            archive_client=archive,
        )


def test_archive_download_retries_server_error_then_uses_cache(
    tmp_path: Path,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=b"archive")

    key = "data/spot/daily/klines/BTCUSDT/1h/file.zip"
    with BinanceArchiveClient(
        transport=httpx.MockTransport(handler),
        cache_root=tmp_path,
        attempts=2,
        initial_delay=0.25,
        sleep=sleeps.append,
    ) as archive:
        assert archive.download(key) == b"archive"
        assert archive.download(key) == b"archive"

    assert attempts == 2
    assert sleeps == [0.25]


def test_archive_metadata_parses_object_details(tmp_path: Path) -> None:
    payload = b"""<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <IsTruncated>false</IsTruncated>
    <Contents>
      <Key>data/spot/file.zip</Key>
      <LastModified>2024-01-02T03:04:05Z</LastModified>
      <ETag>abc123</ETag>
      <Size>42</Size>
    </Contents>
    </ListBucketResult>"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    with BinanceArchiveClient(
        transport=httpx.MockTransport(handler), cache_root=tmp_path
    ) as archive:
        objects = archive.list_metadata("data/spot/")

    assert len(objects) == 1
    assert objects[0].key == "data/spot/file.zip"
    assert objects[0].size == 42
    assert objects[0].last_modified == datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    assert objects[0].etag == "abc123"


def test_archive_readers_normalize_headers_timestamps_and_order() -> None:
    base_ms = 1_704_153_600_000
    klines = read_kline_archive(
        zipped_rows(
            [
                [
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_base",
                    "taker_quote",
                    "unused",
                ],
                kline_row(base_ms * 1000),
            ]
        )
    )
    funding = read_funding_archive(
        zipped_rows(
            [
                [
                    "calc_time",
                    "funding_interval_hours",
                    "last_funding_rate",
                ],
                [str((base_ms + 28_800_000) * 1000), "8", "0.002"],
                [str(base_ms * 1000), "8", "0.001"],
            ]
        )
    )

    assert klines["ts"].tolist() == [base_ms]
    assert klines["close"].tolist() == [1.5]
    assert funding.to_dict("records") == [
        {"ts": base_ms, "funding_8h": 0.001},
        {"ts": base_ms + 28_800_000, "funding_8h": 0.002},
    ]


def test_spot_timestamp_switchover_normalizes_ms_and_microseconds() -> None:
    before_switch_ms = 1_735_603_200_000
    after_switch_ms = 1_735_689_600_000

    result = read_kline_archive(
        zipped_rows(
            [
                kline_row(before_switch_ms),
                kline_row(after_switch_ms * 1000),
            ]
        )
    )

    assert result["ts"].tolist() == [before_switch_ms, after_switch_ms]


def test_funding_reader_filters_cadence_canonicalizes_and_deduplicates() -> None:
    boundary = 1_704_153_600_000
    result = read_funding_archive(
        zipped_rows(
            [
                [
                    "calc_time",
                    "funding_interval_hours",
                    "last_funding_rate",
                ],
                [str(boundary + 47), "8", "0.001"],
                [str(boundary + 47), "8", "0.001"],
                [str(boundary + 3_600_000), "1", "0.009"],
                [str(boundary + 28_800_000 - 31), "8", "0.002"],
            ]
        )
    )

    assert result.to_dict("records") == [
        {"ts": boundary, "funding_8h": 0.001},
        {"ts": boundary + 28_800_000, "funding_8h": 0.002},
    ]


def test_funding_reader_requires_explicit_interval_evidence() -> None:
    payload = zipped_rows(
        [
            ["calc_time", "last_funding_rate"],
            ["1704153600000", "0.001"],
        ]
    )

    with pytest.raises(ValueError, match="funding_interval_hours"):
        read_funding_archive(payload)


def test_call_with_backoff_retries_only_declared_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.TransportError("temporary")
        return "ok"

    monkeypatch.setattr(ingest.time, "sleep", sleeps.append)

    assert call_with_backoff(operation, attempts=3, initial_delay=0.5) == "ok"
    assert calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: archive_root("spot", data_type="funding"), "only for perps"),
        (
            lambda: archive_root("perp", data_type="funding", period="daily"),
            "enumerated monthly",
        ),
        (
            lambda: validate_bar_gaps(pd.DataFrame(), timeframe="1h"),
            "missing ts",
        ),
        (
            lambda: validate_bar_gaps(pd.DataFrame({"ts": []}), timeframe="5m"),
            "unsupported timeframe",
        ),
        (
            lambda: call_with_backoff(lambda: None, attempts=0),
            "attempts",
        ),
    ],
)
def test_public_ingestion_validation_errors(operation: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation()  # type: ignore[operator]

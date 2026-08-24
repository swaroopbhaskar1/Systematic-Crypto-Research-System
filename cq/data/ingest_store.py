"""Archive-to-store ingestion: the orchestration the M1 data layer was missing.

``cq.data.ingest`` knows how to name, download, and decode one Binance archive
object.  This module is what turns those primitives into a populated
:class:`~cq.data.store.ParquetStore`: it enumerates symbols, derives
point-in-time listing bounds, downloads a date range, joins perpetual funding,
and writes one immutable file per UTC year.

Two decisions here are survivorship-critical and are stated explicitly because
getting either wrong produces a backtest that looks plausible and is wrong.

First, the symbol universe is enumerated from the *archive*, never from the
exchange's live symbol list.  ``BinanceArchiveClient.spot_trading_symbols``
returns only what is ``TRADING`` today; selecting history with it would study
survivors.  The archive keeps publishing a dead symbol's history forever, so
listing the archive is the only enumeration that sees the failures.

Second, nothing here ever invents a bar or a funding rate.  A day with no
archive produces no row.  A bar with no funding print carries NaN and leaves
the universe.  A zero funding rate is a real observation and is preserved as
zero; imputing zero for "missing" would manufacture carry that never existed.
"""

import argparse
import json
import re
from calendar import monthrange
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pandas as pd

from cq.data.ingest import (
    TIMEFRAME_MS,
    BinanceArchiveClient,
    fetch_ohlcv_archives,
    funding_prefix,
    kline_prefix,
    read_funding_archive,
)
from cq.data.store import REQUIRED_COLUMNS, ParquetStore
from cq.data.universe import Listing, MarketType, UniverseRegistry

DAY_MS = 86_400_000
DEFAULT_CACHE_ROOT = Path("data/archive-cache")
DEFAULT_TIMEFRAME = "1d"
DEFAULT_EXCHANGE = "binance"
PRICE_COLUMNS = ("open", "high", "low", "close")
_DAILY_TOKEN = re.compile(r"-(\d{4})-(\d{2})-(\d{2})\.zip$")
_MONTHLY_TOKEN = re.compile(r"-(\d{4})-(\d{2})\.zip$")
# A symbol fails the way real archives fail: a vanished or unreadable object
# raises httpx.HTTPError, a truncated or malformed payload raises ValueError.
# Nothing broader is caught, so a disk or programming fault still aborts.
_SYMBOL_ERRORS = (httpx.HTTPError, ValueError)


@dataclass(frozen=True, slots=True)
class SymbolCoverage:
    """Which archive objects exist for one symbol, and the days they span."""

    symbol: str
    keys: frozenset[str]
    first_day: date | None
    last_day: date | None


@dataclass(frozen=True, slots=True)
class SymbolFailure:
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class IngestSummary:
    """What one ingest run actually did, including what it failed to do."""

    market_type: MarketType
    timeframe: str
    start: date
    end: date
    asof: int
    symbols_attempted: int
    symbols_with_data: int
    bars_written: int
    first_ts: int | None
    last_ts: int | None
    files_written: tuple[str, ...]
    failures: tuple[SymbolFailure, ...]


def list_symbol_coverage(
    client: BinanceArchiveClient,
    market_type: MarketType,
    symbol: str,
    timeframe: str,
) -> SymbolCoverage:
    """List every kline archive published for one symbol.

    Both the monthly and the daily prefix are read.  Monthly archives are the
    bulk of the history; the daily prefix carries the trailing partial month
    that has not been rolled up yet, and is what dates a very recent listing
    or a very recent delisting to better than month precision.
    """

    keys: set[str] = set()
    for period in ("monthly", "daily"):
        prefix = kline_prefix(market_type, symbol, timeframe, period=period)
        keys.update(key for key in client.list_objects(prefix) if key.endswith(".zip"))
    spans = tuple(
        span for key in sorted(keys) if (span := _archive_span(key)) is not None
    )
    if not spans:
        return SymbolCoverage(symbol, frozenset(keys), None, None)
    return SymbolCoverage(
        symbol,
        frozenset(keys),
        min(span[0] for span in spans),
        max(span[1] for span in spans),
    )


def coverage_listing(
    coverage: SymbolCoverage,
    *,
    exchange: str,
    market_type: MarketType,
    end: date,
) -> Listing:
    """Turn archive coverage into a point-in-time listing record.

    ``listed_at`` is the first day covered by the symbol's earliest archive.
    ``delisted_at`` is the first day of the month *after* the symbol's last
    archive, because the monthly archive is the unit in which the venue stops
    publishing; dating the exit at month granularity can only ever extend a
    listing, never truncate real history, and membership additionally requires
    a real bar so the extension grants nothing.

    The only reason this can substantiate is ``"delisted"`` — the archive
    proves publication stopped, and nothing more.  ``"zero"`` needs a terminal
    price observation and ``"migrated"`` needs a named successor; neither is
    inferable from a key listing, so neither is ever guessed here.
    """

    if coverage.first_day is None or coverage.last_day is None:
        raise ValueError(f"no archive coverage for {coverage.symbol}")
    listed_at = _midnight(coverage.first_day)
    stopped_at = _midnight(_next_month_start(coverage.last_day))
    if stopped_at >= _midnight(end):
        return Listing(
            symbol=coverage.symbol,
            exchange=exchange,
            market_type=market_type,
            listed_at=listed_at,
        )
    return Listing(
        symbol=coverage.symbol,
        exchange=exchange,
        market_type=market_type,
        listed_at=listed_at,
        delisted_at=stopped_at,
        delist_reason="delisted",
    )


def archive_listings(
    market_type: MarketType,
    *,
    end: date,
    client: BinanceArchiveClient,
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> tuple[Listing, ...]:
    """Derive listing records for every symbol the archive publishes."""
    selected = resolve_symbols(client, market_type, symbols, limit)
    coverages, _ = _collect_coverages(client, market_type, timeframe, selected)
    return _listings(coverages, exchange=exchange, market_type=market_type, end=end)


def build_archive_universe(
    market_type: MarketType,
    *,
    end: date,
    client: BinanceArchiveClient,
    timeframe: str = DEFAULT_TIMEFRAME,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    exchange: str = DEFAULT_EXCHANGE,
) -> UniverseRegistry:
    """Build the point-in-time universe from archive listings alone."""
    return UniverseRegistry(
        archive_listings(
            market_type,
            end=end,
            client=client,
            timeframe=timeframe,
            symbols=symbols,
            limit=limit,
            exchange=exchange,
        )
    )


def resolve_symbols(
    client: BinanceArchiveClient,
    market_type: MarketType,
    symbols: Sequence[str] | None,
    limit: int | None,
) -> tuple[str, ...]:
    """Choose the symbols to ingest, defaulting to the whole archive.

    The default enumerates the archive's monthly *and* daily kline roots.
    Neither root filters by current trading status, so a delisted symbol is
    selected exactly like a live one.
    """

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if symbols is not None:
        chosen = tuple(sorted({symbol for symbol in symbols if symbol}))
        if not chosen:
            raise ValueError("symbols must contain at least one name")
    else:
        chosen = tuple(
            sorted(
                {
                    *client.symbols(market_type, period="monthly"),
                    *client.symbols(market_type, period="daily"),
                }
            )
        )
    return chosen if limit is None else chosen[:limit]


def aggregate_funding(prints: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    """Reduce eight-hourly funding prints to one value per bar.

    Binance prints funding every eight hours, so a daily bar contains three
    prints.  A print stamped ``t`` is attributed to the bar
    ``[bar_ts, bar_ts + interval)`` that contains it.  Bar timestamps from the
    kline archive are bar OPEN times, so that window holds only funding that
    was published while the bar was forming: no lookahead.

    The three prints are combined with an arithmetic MEAN, giving the bar's
    average eight-hour rate.  Mean is the convention because carry accrues per
    print rather than per bar, so the average print is the quantity a daily
    position actually pays; a sum would silently change units to per-day and a
    last-print rule would discard two thirds of the observation.

    A bar with no print is simply absent from the result.  It is left to the
    caller to carry NaN and drop the bar from the universe, because a zero
    funding rate is a real and common observation and must never be produced
    by the absence of data.
    """

    interval = _interval_ms(timeframe)
    if prints.empty:
        return _empty_funding()
    bar_ts = (prints["ts"].astype("int64") // interval) * interval
    grouped = prints.groupby(bar_ts, sort=True)["funding_8h"].mean()
    return pd.DataFrame(
        {
            "ts": pd.Index(grouped.index).astype("int64"),
            "funding_8h": grouped.to_numpy(dtype="float64"),
        }
    )


def membership_mask(
    frame: pd.DataFrame, registry: UniverseRegistry, market_type: MarketType
) -> "pd.Series[bool]":
    """Mark a bar tradable only where listing, price, and funding all agree.

    The listing term comes from :meth:`UniverseRegistry.universe_at`, which is
    the only legal way to enumerate symbols at a timestamp.  It is what makes
    the survivorship contract binding: a bar that exists in the archive but
    falls outside its symbol's listing window is out of universe regardless of
    how complete its prices look.  Price and funding availability can only
    ever take membership away from a listed symbol, never grant it.
    """

    members = {
        timestamp: registry.universe_at(_from_ms(timestamp), market_type)
        for timestamp in sorted({int(value) for value in frame["ts"]})
    }
    listed = pd.Series(
        [
            symbol in members[int(timestamp)]
            for timestamp, symbol in zip(frame["ts"], frame["symbol"], strict=True)
        ],
        index=frame.index,
        dtype=bool,
    )
    priced = frame.loc[:, list(PRICE_COLUMNS)].notna().all(axis=1)
    if market_type == "spot":
        return listed & priced
    return listed & priced & frame["funding_8h"].notna()


def ingest_to_store(
    store_root: Path,
    *,
    market_type: MarketType,
    timeframe: str,
    start: date,
    end: date,
    symbols: Sequence[str] | None = None,
    limit: int | None = None,
    exchange: str = DEFAULT_EXCHANGE,
    client: BinanceArchiveClient | None = None,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    asof: int | None = None,
) -> IngestSummary:
    """Ingest ``[start, end)`` for one market type and write it to the store.

    ``end`` is exclusive.  Every row of one run carries the same ``asof``, the
    ingest wall clock in Unix milliseconds, so a later re-ingest lands as a
    distinguishable revision that a point-in-time read can choose to ignore
    rather than as a silent overwrite.

    A symbol that fails — a vanished object, a truncated ZIP, a malformed CSV
    — is recorded in the returned summary and the run continues, so one bad
    archive cannot abort a several-hundred-symbol run.  The whole symbol is
    dropped rather than the failing month, because writing the readable half
    of a history would leave a hole that looks exactly like a real trading
    halt.  Nothing is swallowed: the summary is the report, and the CLI turns
    a non-empty failure list into a non-zero exit status.
    """

    _validate_request(market_type, timeframe, start, end, limit)
    resolved_asof = asof if asof is not None else _now_ms()
    since_ms, until_ms = _midnight_ms(start), _midnight_ms(end)
    owned = client is None
    active = client or BinanceArchiveClient(cache_root=cache_root)
    try:
        selected = resolve_symbols(active, market_type, symbols, limit)
        coverages, failures = _collect_coverages(
            active, market_type, timeframe, selected
        )
        registry = UniverseRegistry(
            _listings(coverages, exchange=exchange, market_type=market_type, end=end)
        )
        frames, fetch_failures = _collect_frames(
            active,
            coverages,
            exchange=exchange,
            market_type=market_type,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
        )
    finally:
        if owned:
            active.close()

    all_failures = tuple(
        sorted((*failures, *fetch_failures), key=lambda item: item.symbol)
    )
    if not frames:
        return _summary(
            market_type=market_type,
            timeframe=timeframe,
            start=start,
            end=end,
            asof=resolved_asof,
            attempted=len(selected),
            frame=None,
            paths=(),
            failures=all_failures,
        )
    frame = _finalize(
        pd.concat(frames, ignore_index=True), registry, market_type, resolved_asof
    )
    paths = _write_by_year(ParquetStore(store_root), frame, timeframe)
    return _summary(
        market_type=market_type,
        timeframe=timeframe,
        start=start,
        end=end,
        asof=resolved_asof,
        attempted=len(selected),
        frame=frame,
        paths=paths,
        failures=all_failures,
    )


def summary_json(summary: IngestSummary) -> str:
    return json.dumps(asdict(summary), indent=2, sort_keys=True, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest Binance archives into the immutable Parquet store"
    )
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--market", choices=["spot", "perp"], required=True)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="exclusive end date (YYYY-MM-DD)",
    )
    parser.add_argument("--symbols", help="comma-separated symbols; default is all")
    parser.add_argument("--limit", type=int, help="cap the number of symbols")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args(argv)

    with BinanceArchiveClient(cache_root=args.cache_root) as client:
        summary = ingest_to_store(
            args.store,
            market_type=args.market,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            symbols=_split_symbols(args.symbols),
            limit=args.limit,
            client=client,
        )
    print(summary_json(summary))
    return 1 if summary.failures else 0


def _collect_coverages(
    client: BinanceArchiveClient,
    market_type: MarketType,
    timeframe: str,
    symbols: Sequence[str],
) -> tuple[tuple[SymbolCoverage, ...], tuple[SymbolFailure, ...]]:
    coverages: list[SymbolCoverage] = []
    failures: list[SymbolFailure] = []
    for symbol in symbols:
        try:
            coverage = list_symbol_coverage(client, market_type, symbol, timeframe)
        except _SYMBOL_ERRORS as error:
            failures.append(SymbolFailure(symbol, _reason(error)))
            continue
        if coverage.first_day is None:
            failures.append(SymbolFailure(symbol, "no archive published for symbol"))
            continue
        coverages.append(coverage)
    return tuple(coverages), tuple(failures)


def _collect_frames(
    client: BinanceArchiveClient,
    coverages: Sequence[SymbolCoverage],
    *,
    exchange: str,
    market_type: MarketType,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> tuple[tuple[pd.DataFrame, ...], tuple[SymbolFailure, ...]]:
    frames: list[pd.DataFrame] = []
    failures: list[SymbolFailure] = []
    for coverage in coverages:
        try:
            frame = _symbol_frame(
                client,
                coverage,
                exchange=exchange,
                market_type=market_type,
                timeframe=timeframe,
                since_ms=since_ms,
                until_ms=until_ms,
            )
        except _SYMBOL_ERRORS as error:
            failures.append(SymbolFailure(coverage.symbol, _reason(error)))
            continue
        if not frame.empty:
            frames.append(frame)
    return tuple(frames), tuple(failures)


def _symbol_frame(
    client: BinanceArchiveClient,
    coverage: SymbolCoverage,
    *,
    exchange: str,
    market_type: MarketType,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    bars = fetch_ohlcv_archives(
        exchange,
        coverage.symbol,
        timeframe,
        since_ms,
        until_ms,
        market_type,
        archive_client=client,
        available_keys=coverage.keys,
    )
    if bars.empty:
        return bars
    frame = bars.copy()
    frame["symbol"] = coverage.symbol
    frame["market_type"] = market_type
    if market_type != "perp":
        frame["funding_8h"] = pd.Series(
            float("nan"), index=frame.index, dtype="float64"
        )
        return frame
    funding = aggregate_funding(
        _funding_prints(client, coverage.symbol, since_ms, until_ms),
        timeframe=timeframe,
    )
    return frame.merge(funding, on="ts", how="left")


def _funding_prints(
    client: BinanceArchiveClient,
    symbol: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key in sorted(client.list_objects(funding_prefix(symbol))):
        span = _archive_span(key) if key.endswith(".zip") else None
        if span is None or not _overlaps(span, since_ms, until_ms):
            continue
        frames.append(read_funding_archive(client.download(key)))
    if not frames:
        return _empty_funding()
    prints = pd.concat(frames, ignore_index=True)
    within = prints.loc[(prints["ts"] >= since_ms) & (prints["ts"] < until_ms)]
    return within.drop_duplicates("ts", keep="last").reset_index(drop=True)


def _finalize(
    frame: pd.DataFrame,
    registry: UniverseRegistry,
    market_type: MarketType,
    asof: int,
) -> pd.DataFrame:
    result = frame.sort_values(["ts", "symbol"], kind="stable").reset_index(drop=True)
    result["in_universe"] = membership_mask(result, registry, market_type)
    result["asof"] = asof
    return result.loc[:, list(REQUIRED_COLUMNS)]


def _write_by_year(
    store: ParquetStore, frame: pd.DataFrame, timeframe: str
) -> tuple[str, ...]:
    years = pd.to_datetime(frame["ts"], unit="ms", utc=True).dt.year
    written: list[str] = []
    for year in sorted({int(value) for value in years.unique()}):
        group = frame.loc[years == year].reset_index(drop=True)
        written.append(str(store.write(group, timeframe=timeframe)))
    return tuple(written)


def _listings(
    coverages: Iterable[SymbolCoverage],
    *,
    exchange: str,
    market_type: MarketType,
    end: date,
) -> tuple[Listing, ...]:
    return tuple(
        coverage_listing(
            coverage, exchange=exchange, market_type=market_type, end=end
        )
        for coverage in coverages
    )


def _summary(
    *,
    market_type: MarketType,
    timeframe: str,
    start: date,
    end: date,
    asof: int,
    attempted: int,
    frame: pd.DataFrame | None,
    paths: tuple[str, ...],
    failures: tuple[SymbolFailure, ...],
) -> IngestSummary:
    if frame is None or frame.empty:
        return IngestSummary(
            market_type=market_type,
            timeframe=timeframe,
            start=start,
            end=end,
            asof=asof,
            symbols_attempted=attempted,
            symbols_with_data=0,
            bars_written=0,
            first_ts=None,
            last_ts=None,
            files_written=paths,
            failures=failures,
        )
    return IngestSummary(
        market_type=market_type,
        timeframe=timeframe,
        start=start,
        end=end,
        asof=asof,
        symbols_attempted=attempted,
        symbols_with_data=int(frame["symbol"].nunique()),
        bars_written=len(frame),
        first_ts=int(frame["ts"].min()),
        last_ts=int(frame["ts"].max()),
        files_written=paths,
        failures=failures,
    )


def _validate_request(
    market_type: MarketType,
    timeframe: str,
    start: date,
    end: date,
    limit: int | None,
) -> None:
    if market_type not in ("spot", "perp"):
        raise ValueError(f"unsupported market type: {market_type}")
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if end <= start:
        raise ValueError("end must be later than start")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")


def _archive_span(key: str) -> tuple[date, date] | None:
    daily = _DAILY_TOKEN.search(key)
    if daily is not None:
        day = date(int(daily.group(1)), int(daily.group(2)), int(daily.group(3)))
        return day, day
    monthly = _MONTHLY_TOKEN.search(key)
    if monthly is None:
        return None
    year, month = int(monthly.group(1)), int(monthly.group(2))
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _overlaps(span: tuple[date, date], since_ms: int, until_ms: int) -> bool:
    """Test whether an archive's covered days intersect ``[since, until)``."""
    first, last = span
    return _midnight_ms(first) < until_ms and _midnight_ms(last) + DAY_MS > since_ms


def _next_month_start(day: date) -> date:
    return date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def _midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _midnight_ms(day: date) -> int:
    return int(_midnight(day).timestamp() * 1000)


def _from_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _interval_ms(timeframe: str) -> int:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return TIMEFRAME_MS[timeframe]


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="int64"),
            "funding_8h": pd.Series(dtype="float64"),
        }
    )


def _split_symbols(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _reason(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


if __name__ == "__main__":
    raise SystemExit(main())

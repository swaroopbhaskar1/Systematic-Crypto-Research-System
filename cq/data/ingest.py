"""Deterministic ingestion from Binance's public data archive."""

import io
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Self, TypeAlias, TypeVar, cast
from urllib.parse import quote
from xml.etree import ElementTree

import httpx
import pandas as pd

from cq.data.universe import MarketType

BINANCE_ARCHIVE_URL = "https://data.binance.vision"
BINANCE_S3_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
BINANCE_SPOT_MARKET_DATA_URL = "https://data-api.binance.vision/api/v3/exchangeInfo"
TIMEFRAME_MS = {"1h": 3_600_000, "1d": 86_400_000}
EIGHT_HOURS_MS = 28_800_000
FUNDING_JITTER_TOLERANCE_MS = 1_000
OHLCV_COLUMNS = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
]
ArchiveDataType: TypeAlias = Literal["klines", "funding"]
ArchivePeriod: TypeAlias = Literal["daily", "monthly"]
CacheDate: TypeAlias = date | datetime | int
T = TypeVar("T")
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class DataAvailability:
    dataset: str
    available_for_deep_history: bool
    maximum_history_days: int | None
    reason: str
    evidence_url: str
    checked_at: date

    @classmethod
    def binance_open_interest(cls) -> "DataAvailability":
        return cls(
            dataset="binance_um_open_interest",
            available_for_deep_history=False,
            maximum_history_days=30,
            reason=(
                "The Binance REST open-interest endpoint retains only 30 days. "
                "Binance Vision metrics has deep open-interest history, but a "
                "separate metrics adapter is not implemented in M1; open "
                "interest is explicitly unavailable and is never persisted as "
                "a sparse historical column."
            ),
            evidence_url=(
                "https://developers.binance.com/docs/derivatives/usds-margined-"
                "futures/market-data/rest-api/Open-Interest-Statistics"
            ),
            checked_at=date(2026, 8, 24),
        )


@dataclass(frozen=True, slots=True)
class ArchiveObject:
    key: str
    size: int
    last_modified: datetime | None
    etag: str | None


def raw_cache_key(
    exchange: str,
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    archive_date: CacheDate,
    until: datetime | int | None = None,
) -> Path:
    """Build a cache path containing every raw-response identity dimension."""
    _validate_market_and_timeframe(market_type, timeframe)
    day, suffix = _cache_date_and_suffix(archive_date, until)
    return (
        Path(f"exchange={_safe_component(exchange.lower())}")
        / f"market_type={market_type}"
        / f"symbol={_safe_component(symbol)}"
        / f"timeframe={_safe_component(timeframe)}"
        / f"date={day.isoformat()}{suffix}.zip"
    )


def fetch_ohlcv(
    exchange: str,
    symbol: str,
    timeframe: str,
    since: datetime | int,
    until: datetime | int,
    market_type: MarketType,
    *,
    cache_root: Path = Path("data/raw-cache"),
    archive_client: "BinanceArchiveClient | None" = None,
) -> pd.DataFrame:
    """Fetch Binance daily archives and reject malformed or gapped bars."""
    _validate_fetch_request(exchange, market_type, timeframe)
    since_ms, until_ms = _epoch_ms(since), _epoch_ms(until)
    if until_ms <= since_ms:
        raise ValueError("until must be later than since")
    owned = archive_client is None
    client = archive_client or BinanceArchiveClient(cache_root=cache_root)
    try:
        frames = _download_daily_klines(
            client, exchange, market_type, symbol, timeframe, since_ms, until_ms
        )
    finally:
        if owned:
            client.close()
    return _combine_ohlcv(frames, timeframe, since_ms, until_ms)


def validate_bar_gaps(frame: pd.DataFrame, *, timeframe: str) -> None:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    if "ts" not in frame:
        raise ValueError("missing ts column")
    timestamps = frame["ts"].astype("int64").sort_values()
    if timestamps.diff().dropna().gt(TIMEFRAME_MS[timeframe]).any():
        raise ValueError(f"bar gap exceeds one {timeframe} interval")


def call_with_backoff(
    operation: Callable[[], T],
    *,
    transient_errors: tuple[type[Exception], ...] = (httpx.TransportError,),
    attempts: int = 5,
    initial_delay: float = 1.0,
) -> T:
    """Retry declared network failures with exponential backoff."""
    if attempts < 1 or initial_delay < 0:
        raise ValueError("attempts must be positive and delay non-negative")
    for attempt in range(attempts):
        try:
            return operation()
        except transient_errors:
            if attempt + 1 == attempts:
                raise
            time.sleep(initial_delay * (2**attempt))
    raise RuntimeError("unreachable retry state")


class BinanceArchiveClient:
    """Metadata-first client for Binance's public S3 archive."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        cache_root: Path = Path("data/archive-cache"),
        attempts: int = 5,
        initial_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("provide client or transport, not both")
        self._owned_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout, follow_redirects=True, transport=transport
        )
        self._cache_root = cache_root
        self._attempts = attempts
        self._initial_delay = initial_delay
        self._sleep = sleep

    def close(self) -> None:
        if self._owned_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: object,
    ) -> None:
        self.close()

    def list_metadata(self, prefix: str) -> tuple[ArchiveObject, ...]:
        objects, _ = self._list(prefix=prefix, delimiter=None)
        return _unique_objects(objects)

    def list_prefixes(self, prefix: str) -> tuple[str, ...]:
        _, prefixes = self._list(prefix=prefix, delimiter="/")
        return tuple(sorted(set(prefixes)))

    def list_objects(self, prefix: str) -> tuple[str, ...]:
        return tuple(item.key for item in self.list_metadata(prefix))

    def spot_trading_symbols(self) -> frozenset[str]:
        response = self._get(BINANCE_SPOT_MARKET_DATA_URL)
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise TypeError("spot exchange info must be an object")
        exchange_info = cast(Mapping[str, object], payload)
        raw_symbols = exchange_info.get("symbols")
        if not isinstance(raw_symbols, list):
            raise TypeError("spot exchange info omitted symbols")
        symbols: set[str] = set()
        for item in cast(list[object], raw_symbols):
            if not isinstance(item, dict):
                raise TypeError("spot exchange info contains malformed symbol")
            fields = cast(Mapping[str, object], item)
            symbol = fields.get("symbol")
            quote_asset = fields.get("quoteAsset")
            status = fields.get("status")
            if (
                not isinstance(symbol, str)
                or not isinstance(quote_asset, str)
                or not isinstance(status, str)
            ):
                raise TypeError("spot exchange info symbol fields must be strings")
            if quote_asset == "USDT" and status == "TRADING":
                symbols.add(symbol)
        if not symbols:
            raise RuntimeError("spot exchange info contained no trading USDT pairs")
        return frozenset(symbols)

    def symbol_prefixes(
        self,
        market_type: MarketType,
        data_type: ArchiveDataType = "klines",
        *,
        period: ArchivePeriod = "monthly",
    ) -> tuple[str, ...]:
        return self.list_prefixes(
            archive_root(market_type, data_type=data_type, period=period)
        )

    def symbols(
        self,
        market_type: MarketType,
        data_type: ArchiveDataType = "klines",
        *,
        period: ArchivePeriod = "monthly",
    ) -> tuple[str, ...]:
        root = archive_root(market_type, data_type=data_type, period=period)
        names = (_symbol_from_prefix(root, item) for item in self.list_prefixes(root))
        return tuple(sorted({name for name in names if name is not None}))

    def download(self, key: str, *, cache_key: Path | None = None) -> bytes:
        _validate_archive_key(key)
        relative = cache_key or Path(key)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe cache key: {relative}")
        cache_path = self._cache_root / relative
        if cache_path.exists():
            return cache_path.read_bytes()
        response = self._get(f"{BINANCE_ARCHIVE_URL}/{quote(key, safe='/')}")
        return _write_cache_once(cache_path, response.content)

    def _get(
        self, url: str, *, params: httpx.QueryParams | None = None
    ) -> httpx.Response:
        return _http_get_with_backoff(
            self._client,
            url,
            params=params,
            attempts=self._attempts,
            initial_delay=self._initial_delay,
            sleep=self._sleep,
        )

    def _list(
        self, *, prefix: str, delimiter: str | None
    ) -> tuple[list[ArchiveObject], list[str]]:
        objects: list[ArchiveObject] = []
        prefixes: list[str] = []
        continuation: str | None = None
        seen_tokens: set[str] = set()
        while True:
            page = self._listing_page(prefix, delimiter, continuation)
            objects.extend(page.objects)
            prefixes.extend(page.prefixes)
            if not page.truncated:
                return objects, prefixes
            continuation = _next_token(page.next_token, seen_tokens)

    def _listing_page(
        self,
        prefix: str,
        delimiter: str | None,
        continuation: str | None,
    ) -> "_ListingPage":
        params: dict[str, str] = {"list-type": "2", "prefix": prefix}
        if delimiter is not None:
            params["delimiter"] = delimiter
        if continuation is not None:
            params["continuation-token"] = continuation
        response = self._get(BINANCE_S3_URL, params=httpx.QueryParams(params))
        return _parse_listing(response.content)


@dataclass(frozen=True, slots=True)
class _ListingPage:
    objects: tuple[ArchiveObject, ...]
    prefixes: tuple[str, ...]
    truncated: bool
    next_token: str | None


def archive_root(
    market_type: MarketType,
    *,
    data_type: ArchiveDataType,
    period: ArchivePeriod = "monthly",
) -> str:
    _validate_market_type(market_type)
    if data_type == "funding":
        if market_type != "perp":
            raise ValueError("funding archives exist only for perps")
        if period != "monthly":
            raise ValueError("funding archives are enumerated monthly")
        return "data/futures/um/monthly/fundingRate/"
    market_root = "spot" if market_type == "spot" else "futures/um"
    return f"data/{market_root}/{period}/klines/"


def kline_prefix(
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    *,
    period: ArchivePeriod = "monthly",
) -> str:
    _validate_market_and_timeframe(market_type, timeframe)
    _safe_component(symbol)
    return (
        f"{archive_root(market_type, data_type='klines', period=period)}"
        f"{symbol}/{timeframe}/"
    )


def kline_archive_key(
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    archive_date: date,
) -> str:
    prefix = kline_prefix(market_type, symbol, timeframe, period="daily")
    return f"{prefix}{symbol}-{timeframe}-{archive_date.isoformat()}.zip"


def kline_archive_url(
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    archive_date: date,
) -> str:
    key = kline_archive_key(market_type, symbol, timeframe, archive_date)
    return f"{BINANCE_ARCHIVE_URL}/{quote(key, safe='/')}"


def funding_prefix(symbol: str) -> str:
    _safe_component(symbol)
    return f"{archive_root('perp', data_type='funding')}{symbol}/"


def funding_archive_url(symbol: str, month: date) -> str:
    key = f"{funding_prefix(symbol)}{symbol}-fundingRate-{month:%Y-%m}.zip"
    return f"{BINANCE_ARCHIVE_URL}/{quote(key, safe='/')}"


def read_kline_archive(payload: bytes) -> pd.DataFrame:
    """Decode one Binance kline ZIP into the OHLCV schema."""
    raw = _read_first_zip_csv(payload, header=None)
    if raw.empty:
        return _empty_ohlcv()
    if raw.shape[1] < 8:
        raise ValueError("kline archive row has fewer than eight columns")
    if pd.isna(pd.to_numeric(raw.iloc[0, 0], errors="coerce")):
        raw = raw.iloc[1:]
    selected = raw.iloc[:, [0, 1, 2, 3, 4, 5, 7]].copy()
    selected.columns = OHLCV_COLUMNS
    for column in OHLCV_COLUMNS:
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if selected.isna().any().any():
        raise ValueError("kline archive contains missing values")
    selected["ts"] = _timestamps_to_ms(selected["ts"])
    dtypes = {column: "float64" for column in OHLCV_COLUMNS if column != "ts"}
    return selected.astype({"ts": "int64", **dtypes}).reset_index(drop=True)


def read_funding_archive(payload: bytes) -> pd.DataFrame:
    """Decode one Binance funding-rate ZIP to ts/funding_8h rows."""
    raw = _read_first_zip_csv(payload, header=0)
    normalized = {str(column).lower(): column for column in raw.columns}
    timestamp = _first_present(normalized, ("calc_time", "fundingtime", "funding_time"))
    interval = _first_present(normalized, ("funding_interval_hours",))
    rate = _first_present(
        normalized, ("last_funding_rate", "fundingrate", "funding_rate")
    )
    result = raw.loc[:, [timestamp, interval, rate]].copy()
    result.columns = ["ts", "funding_interval_hours", "funding_8h"]
    result["ts"] = _timestamps_to_ms(result["ts"])
    result["funding_interval_hours"] = pd.to_numeric(
        result["funding_interval_hours"], errors="raise"
    )
    result["funding_8h"] = pd.to_numeric(result["funding_8h"], errors="raise").astype(
        "float64"
    )
    if result.isna().any().any():
        raise ValueError("funding archive contains missing values")
    result = result.loc[result["funding_interval_hours"] == 8].copy()
    if result.empty:
        return _empty_funding()
    result["ts"] = _canonical_funding_timestamps(result["ts"])
    result = result.sort_values("ts", kind="stable")
    result = result.drop_duplicates("ts", keep="last")
    return result.loc[:, ["ts", "funding_8h"]].reset_index(drop=True)


def _download_daily_klines(
    client: BinanceArchiveClient,
    exchange: str,
    market_type: MarketType,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for day in _days_covering(since_ms, until_ms):
        key = kline_archive_key(market_type, symbol, timeframe, day)
        cache_key = raw_cache_key(exchange, market_type, symbol, timeframe, day)
        frames.append(read_kline_archive(client.download(key, cache_key=cache_key)))
    return frames


def _combine_ohlcv(
    frames: list[pd.DataFrame],
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    if not frames:
        return _empty_ohlcv()
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[(frame["ts"] >= since_ms) & (frame["ts"] < until_ms)]
    frame = frame.sort_values("ts").reset_index(drop=True)
    _reject_conflicting_duplicates(frame)
    frame = frame.drop_duplicates("ts", keep="first").reset_index(drop=True)
    validate_bar_gaps(frame, timeframe=timeframe)
    return frame.loc[:, OHLCV_COLUMNS]


def _reject_conflicting_duplicates(frame: pd.DataFrame) -> None:
    duplicated = frame.loc[frame.duplicated("ts", keep=False)]
    for _, group in duplicated.groupby("ts", sort=False):
        if len(group.drop_duplicates()) != 1:
            raise ValueError("conflicting duplicate bars in archive")


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="int64"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "quote_volume": pd.Series(dtype="float64"),
        }
    )


def _empty_funding() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="int64"),
            "funding_8h": pd.Series(dtype="float64"),
        }
    )


def _canonical_funding_timestamps(timestamps: pd.Series) -> pd.Series:
    values = timestamps.astype("int64")
    canonical = ((values + EIGHT_HOURS_MS // 2) // EIGHT_HOURS_MS) * EIGHT_HOURS_MS
    if (values - canonical).abs().gt(FUNDING_JITTER_TOLERANCE_MS).any():
        raise ValueError("8-hour funding timestamp is outside jitter tolerance")
    return canonical.astype("int64")


def _days_covering(since_ms: int, until_ms: int) -> tuple[date, ...]:
    first = datetime.fromtimestamp(since_ms / 1000, tz=UTC).date()
    last = datetime.fromtimestamp((until_ms - 1) / 1000, tz=UTC).date()
    count = (last - first).days + 1
    return tuple(first + timedelta(days=offset) for offset in range(count))


def _parse_listing(payload: bytes) -> _ListingPage:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ValueError("malformed S3 archive listing") from error
    objects = tuple(_parse_archive_object(item) for item in _elements(root, "Contents"))
    prefixes = tuple(
        value
        for item in _elements(root, "CommonPrefixes")
        if (value := _child_text(item, "Prefix")) is not None
    )
    truncated = (_child_text(root, "IsTruncated") or "false").lower() == "true"
    return _ListingPage(
        objects,
        prefixes,
        truncated,
        _child_text(root, "NextContinuationToken"),
    )


def _parse_archive_object(element: ElementTree.Element) -> ArchiveObject:
    key = _required_child_text(element, "Key")
    size_text = _required_child_text(element, "Size")
    modified_text = _child_text(element, "LastModified")
    etag = _child_text(element, "ETag")
    try:
        size = int(size_text)
    except ValueError as error:
        raise ValueError(f"invalid S3 object size for {key!r}") from error
    modified = _parse_s3_datetime(modified_text) if modified_text else None
    return ArchiveObject(key, size, modified, etag)


def _elements(
    root: ElementTree.Element, local_name: str
) -> Iterable[ElementTree.Element]:
    return (
        element for element in root.iter() if _local_name(element.tag) == local_name
    )


def _child_text(element: ElementTree.Element, local_name: str) -> str | None:
    for child in element:
        if _local_name(child.tag) == local_name:
            return child.text
    return None


def _required_child_text(element: ElementTree.Element, local_name: str) -> str:
    value = _child_text(element, local_name)
    if value is None:
        raise ValueError(f"S3 listing object omitted {local_name}")
    return value


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _parse_s3_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("S3 last-modified timestamp is timezone-naive")
    return parsed


def _next_token(token: str | None, seen: set[str]) -> str:
    if token is None:
        raise RuntimeError("truncated S3 listing omitted continuation token")
    if token in seen:
        raise RuntimeError("S3 listing repeated its continuation token")
    seen.add(token)
    return token


def _unique_objects(objects: Iterable[ArchiveObject]) -> tuple[ArchiveObject, ...]:
    by_key: dict[str, ArchiveObject] = {}
    for item in objects:
        previous = by_key.setdefault(item.key, item)
        if previous != item:
            raise ValueError(f"conflicting S3 metadata for {item.key!r}")
    return tuple(by_key[key] for key in sorted(by_key))


def _symbol_from_prefix(root: str, prefix: str) -> str | None:
    if not prefix.startswith(root):
        raise ValueError(f"archive returned prefix outside request: {prefix}")
    remainder = prefix[len(root) :].rstrip("/")
    if not remainder or "/" in remainder:
        return None
    return remainder


def _http_get_with_backoff(
    client: httpx.Client,
    url: str,
    *,
    params: httpx.QueryParams | None = None,
    attempts: int = 5,
    initial_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    if attempts < 1 or initial_delay < 0:
        raise ValueError("attempts must be positive and delay non-negative")
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
        except httpx.TransportError:
            if attempt + 1 == attempts:
                raise
        else:
            if response.status_code != 429 and response.status_code < 500:
                response.raise_for_status()
                return response
            if attempt + 1 == attempts:
                response.raise_for_status()
        sleep(initial_delay * (2**attempt))
    raise RuntimeError("unreachable HTTP retry state")


def _read_first_zip_csv(payload: bytes, *, header: int | None) -> pd.DataFrame:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("archive response is not a valid ZIP") from error
    with archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
        if len(names) != 1:
            raise ValueError("archive must contain exactly one data file")
        with archive.open(names[0]) as data_file:
            return pd.read_csv(data_file, header=header)


def _first_present(columns: Mapping[str, V], candidates: Iterable[str]) -> V:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    raise ValueError(f"archive missing columns: {tuple(candidates)}")


def _timestamps_to_ms(values: pd.Series) -> pd.Series:
    timestamps = pd.to_numeric(values, errors="raise").astype("int64")
    return timestamps.where(timestamps < 100_000_000_000_000, timestamps // 1000)


def _write_cache_once(path: Path, payload: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        return path.read_bytes()
    return payload


def _cache_date_and_suffix(
    value: CacheDate, until: datetime | int | None
) -> tuple[date, str]:
    if isinstance(value, datetime):
        start_ms = _epoch_ms(value)
        day = value.astimezone(UTC).date()
    elif isinstance(value, int):
        start_ms = value
        day = datetime.fromtimestamp(value / 1000, tz=UTC).date()
    else:
        start_ms = None
        day = value
    if until is None:
        return day, ""
    end_ms = _epoch_ms(until)
    end_day = datetime.fromtimestamp(end_ms / 1000, tz=UTC).date()
    return day, f"_to={end_day.isoformat()}_start={start_ms}_end={end_ms}"


def _epoch_ms(value: datetime | int) -> int:
    if isinstance(value, int):
        return value
    if value.tzinfo is None:
        raise ValueError("datetime bounds must be timezone-aware")
    return int(value.timestamp() * 1000)


def _validate_fetch_request(
    exchange: str, market_type: MarketType, timeframe: str
) -> None:
    if exchange.lower() != "binance":
        raise ValueError("public archive ingestion currently supports Binance")
    _safe_component(exchange)
    _validate_market_and_timeframe(market_type, timeframe)


def _validate_market_and_timeframe(market_type: MarketType, timeframe: str) -> None:
    _validate_market_type(market_type)
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")


def _validate_market_type(market_type: MarketType) -> None:
    if market_type not in ("spot", "perp"):
        raise ValueError(f"unsupported market type: {market_type}")


def _validate_archive_key(key: str) -> None:
    if key.startswith("/") or ".." in Path(key).parts:
        raise ValueError(f"unsafe archive key: {key!r}")


def _safe_component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe cache component: {value!r}")
    return value

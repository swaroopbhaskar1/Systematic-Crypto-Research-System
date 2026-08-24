"""Real Binance archive inventory gate for the M1 data layer."""

import argparse
import json
import re
from calendar import monthrange
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from itertools import repeat
from pathlib import Path

import pandas as pd

from cq.data.ingest import (
    BinanceArchiveClient,
    funding_prefix,
    kline_prefix,
    read_funding_archive,
)
from cq.data.universe import MarketType
from cq.data.validate import funding_coverage_count

TOKEN_THRESHOLD = 150
DELISTED_THRESHOLD = 50
FUNDED_PERP_THRESHOLD = 100
FUNDING_BAR_THRESHOLD = 270
QUOTE_ASSETS = (
    "FDUSD",
    "TUSD",
    "USDT",
    "USDC",
    "BUSD",
    "BTC",
    "ETH",
    "BNB",
)
SPOT_KEY = re.compile(r"^data/spot/(?:monthly|daily)/klines/([^/]+)/(?:1d|1h)/")
PERP_KEY = re.compile(r"^data/futures/um/(?:monthly|daily)/klines/([^/]+)/(?:1d|1h)/")
FUNDING_KEY = re.compile(r"^data/futures/um/monthly/fundingRate/([^/]+)/")
DATE_TOKEN = re.compile(r"-(\d{4})-(\d{2})(?:-(\d{2}))?\.zip$")


def _empty_last_archive_dates() -> dict[str, date]:
    return {}


@dataclass(frozen=True, slots=True)
class ArchiveInventory:
    spot_symbols: frozenset[str]
    perp_symbols: frozenset[str]
    funding_symbols: frozenset[str]
    delisted_base_assets: frozenset[str]
    funding_contiguous_counts: Mapping[str, int]
    last_archive_dates: Mapping[str, date] = field(
        default_factory=_empty_last_archive_dates
    )


@dataclass(frozen=True, slots=True)
class ArchiveGateResult:
    spot_symbol_count: int
    perp_symbol_count: int
    funding_symbol_count: int
    token_count: int
    delisted_count: int
    funded_perp_count: int
    archive_latest_date: str | None
    open_interest_disposition: str
    passed: bool
    failures: tuple[str, ...]


def classify_archive_inventory(
    keys: Iterable[str], *, archive_latest: date
) -> ArchiveInventory:
    """Classify real S3 keys and infer venue exits from last archive dates."""
    spot: set[str] = set()
    perps: set[str] = set()
    funding: set[str] = set()
    last_dates: dict[str, date] = {}
    for key in sorted(set(keys)):
        match = SPOT_KEY.match(key)
        if match is not None:
            symbol = match.group(1)
            spot.add(symbol)
            object_date = _archive_date(key)
            if object_date is not None:
                last_dates[symbol] = max(
                    last_dates.get(symbol, object_date), object_date
                )
            continue
        match = PERP_KEY.match(key)
        if match is not None:
            perps.add(match.group(1))
            continue
        match = FUNDING_KEY.match(key)
        if match is not None:
            symbol = match.group(1)
            perps.add(symbol)
            funding.add(symbol)
    delisted = frozenset(
        _base_asset(symbol)
        for symbol, last_date in last_dates.items()
        if last_date < archive_latest
    )
    return ArchiveInventory(
        spot_symbols=frozenset(spot),
        perp_symbols=frozenset(perps),
        funding_symbols=frozenset(funding),
        delisted_base_assets=delisted,
        funding_contiguous_counts={},
        last_archive_dates=last_dates,
    )


def evaluate_archive_gate(inventory: ArchiveInventory) -> ArchiveGateResult:
    token_count = len({_base_asset(symbol) for symbol in inventory.spot_symbols})
    latest = (
        max(inventory.last_archive_dates.values()).isoformat()
        if inventory.last_archive_dates
        else None
    )
    funded_perp_count = sum(
        1
        for symbol in inventory.perp_symbols
        if inventory.funding_contiguous_counts.get(symbol, 0) >= FUNDING_BAR_THRESHOLD
    )
    counts = {
        "token_count": (token_count, TOKEN_THRESHOLD),
        "delisted_count": (
            len(inventory.delisted_base_assets),
            DELISTED_THRESHOLD,
        ),
        "funded_perp_count": (
            funded_perp_count,
            FUNDED_PERP_THRESHOLD,
        ),
    }
    failures = tuple(
        name for name, (actual, required) in counts.items() if actual < required
    )
    return ArchiveGateResult(
        spot_symbol_count=len(inventory.spot_symbols),
        perp_symbol_count=len(inventory.perp_symbols),
        funding_symbol_count=len(inventory.funding_symbols),
        token_count=token_count,
        delisted_count=len(inventory.delisted_base_assets),
        funded_perp_count=funded_perp_count,
        archive_latest_date=latest,
        open_interest_disposition="unavailable",
        passed=not failures,
        failures=failures,
    )


def collect_real_archive_inventory(
    client: BinanceArchiveClient, *, workers: int = 16
) -> ArchiveInventory:
    """Inspect USDT metadata and only download five funding ZIPs per perp."""
    if workers < 1:
        raise ValueError("workers must be positive")
    spot_symbols = _usdt_symbols(
        _symbols_from_prefixes(
            client.symbol_prefixes("spot", "klines", period="monthly")
        )
    )
    perp_symbols = _usdt_symbols(
        _symbols_from_prefixes(
            client.symbol_prefixes("perp", "klines", period="monthly")
        )
    )
    funding_symbols = _usdt_symbols(
        _symbols_from_prefixes(client.symbol_prefixes("perp", "funding"))
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        spot_object_groups = tuple(
            executor.map(
                _list_kline_objects,
                repeat(client),
                repeat("spot"),
                spot_symbols,
            )
        )
        funding_object_groups = tuple(
            executor.map(
                client.list_objects,
                (funding_prefix(symbol) for symbol in funding_symbols),
            )
        )
    spot_keys = tuple(key for group in spot_object_groups for key in group)
    funding_keys = tuple(key for group in funding_object_groups for key in group)
    observed_dates = tuple(
        object_date
        for object_date in (_archive_date(key) for key in spot_keys)
        if object_date is not None
    )
    if not observed_dates:
        raise RuntimeError("spot archive inventory returned no dated objects")
    inventory = classify_archive_inventory(
        (*spot_keys, *funding_keys), archive_latest=max(observed_dates)
    )
    trading_spot_symbols = client.spot_trading_symbols()
    inventory = replace(
        inventory,
        spot_symbols=frozenset(spot_symbols),
        perp_symbols=frozenset((*perp_symbols, *funding_symbols)),
        funding_symbols=frozenset(funding_symbols),
        delisted_base_assets=frozenset(
            _base_asset(symbol)
            for symbol in spot_symbols
            if symbol not in trading_spot_symbols
        ),
    )
    funding_counts = _download_funding_counts(
        client,
        funding_symbols,
        funding_object_groups,
        workers=workers,
    )
    return replace(inventory, funding_contiguous_counts=funding_counts)


def _list_kline_objects(
    client: BinanceArchiveClient,
    market_type: MarketType,
    symbol: str,
) -> tuple[str, ...]:
    monthly = client.list_objects(
        kline_prefix(market_type, symbol, "1d", period="monthly")
    )
    daily = client.list_objects(kline_prefix(market_type, symbol, "1d", period="daily"))
    return tuple(sorted({*monthly, *daily}))


def result_json(result: ArchiveGateResult) -> str:
    return json.dumps(asdict(result), indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate real Binance archive M1 inventory thresholds"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with BinanceArchiveClient() as client:
        inventory = collect_real_archive_inventory(client, workers=args.workers)
    result = evaluate_archive_gate(inventory)
    rendered = result_json(result)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if result.passed else 1


def _download_funding_counts(
    client: BinanceArchiveClient,
    symbols: tuple[str, ...],
    object_groups: tuple[tuple[str, ...], ...],
    *,
    workers: int,
) -> dict[str, int]:
    selected_by_symbol = {
        symbol: tuple(
            key for key in sorted(key for key in keys if key.endswith(".zip"))[:5]
        )
        for symbol, keys in zip(symbols, object_groups, strict=True)
    }
    selected = tuple(
        (symbol, key) for symbol in symbols for key in selected_by_symbol[symbol]
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        payloads = tuple(executor.map(client.download, (key for _, key in selected)))
    timestamps: dict[str, list[pd.Series]] = {symbol: [] for symbol in symbols}
    for (symbol, _), payload in zip(selected, payloads, strict=True):
        timestamps[symbol].append(read_funding_archive(payload)["ts"])
    return {
        symbol: funding_coverage_count(
            pd.concat(series, ignore_index=True) if series else pd.Series(dtype="int64")
        )
        for symbol, series in timestamps.items()
    }


def _symbols_from_prefixes(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    symbols = {prefix.rstrip("/").rsplit("/", maxsplit=1)[-1] for prefix in prefixes}
    if not symbols:
        raise RuntimeError("archive symbol prefix listing was empty")
    return tuple(sorted(symbols))


def _usdt_symbols(symbols: tuple[str, ...]) -> tuple[str, ...]:
    filtered = tuple(symbol for symbol in symbols if symbol.endswith("USDT"))
    if not filtered:
        raise RuntimeError("archive inventory contained no USDT symbols")
    return filtered


def _archive_date(key: str) -> date | None:
    match = DATE_TOKEN.search(key)
    if match is None:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    day_text = match.group(3)
    day = int(day_text) if day_text is not None else monthrange(year, month)[1]
    return date(year, month, day)


def _base_asset(symbol: str) -> str:
    for quote_asset in QUOTE_ASSETS:
        if symbol.endswith(quote_asset) and len(symbol) > len(quote_asset):
            return symbol[: -len(quote_asset)]
    return symbol


if __name__ == "__main__":
    raise SystemExit(main())

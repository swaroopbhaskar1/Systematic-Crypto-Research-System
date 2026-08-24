"""Fail-fast validation for point-in-time market data."""

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Protocol

import numpy as np
import pandas as pd

EIGHT_HOURS_MS = 8 * 60 * 60 * 1000
TIMEFRAME_MS = {"1h": 60 * 60 * 1000, "1d": 24 * 60 * 60 * 1000}
LOGICAL_KEY = ["ts", "symbol", "market_type"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


class DataValidationError(ValueError):
    """Raised when market data violates a hard M1 invariant."""


class ListingBounds(Protocol):
    """Structural input required for gap and survivorship validation."""

    @property
    def symbol(self) -> str: ...

    @property
    def market_type(self) -> str: ...

    @property
    def listed_at(self) -> datetime: ...

    @property
    def delisted_at(self) -> datetime | None: ...


@dataclass(frozen=True, slots=True)
class ValidationReport:
    row_count: int
    live_symbol_count: int
    delisted_symbol_count: int
    delisted_live_ratio: float


def validate_frame(
    frame: pd.DataFrame,
    *,
    listings: Iterable[ListingBounds],
    timeframe: str,
) -> ValidationReport:
    """Validate a long market-data frame and report survivorship risk."""
    if timeframe not in TIMEFRAME_MS:
        raise DataValidationError(f"unsupported timeframe: {timeframe}")
    materialized = tuple(listings)
    _validate_required_columns(frame)
    _validate_duplicate_keys(frame)
    _validate_values(frame)
    _validate_membership_bounds(frame, materialized)
    _validate_gaps(frame, materialized, TIMEFRAME_MS[timeframe])
    return _survivorship_report(frame, materialized)


def funding_coverage_count(timestamps: pd.Series) -> int:
    """Return the longest exact contiguous run of eight-hour timestamps."""
    values = sorted(
        {int(value) for value in timestamps.dropna().astype("int64").tolist()}
    )
    if not values:
        return 0
    longest = 1
    current = 1
    for previous, current_value in pairwise(values):
        if current_value - previous == EIGHT_HOURS_MS:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def _validate_required_columns(frame: pd.DataFrame) -> None:
    required = set(LOGICAL_KEY + PRICE_COLUMNS)
    required.update({"volume", "quote_volume", "funding_8h", "in_universe", "asof"})
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"missing columns: {sorted(missing)}")


def _validate_duplicate_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(LOGICAL_KEY).any():
        raise DataValidationError("duplicate logical key")


def _validate_values(frame: pd.DataFrame) -> None:
    invalid_market_types = set(frame["market_type"].dropna().unique()).difference(
        {"spot", "perp"}
    )
    if frame["market_type"].isna().any() or invalid_market_types:
        raise DataValidationError(
            f"unsupported market type values: {sorted(invalid_market_types)}"
        )
    if (
        frame["in_universe"].isna().any()
        or not frame["in_universe"]
        .map(lambda value: isinstance(value, (bool, np.bool_)))
        .all()
    ):
        raise DataValidationError("in_universe must contain explicit booleans")
    if frame[["ts", "asof"]].isna().any().any():
        raise DataValidationError("ts and asof must be present")
    try:
        timestamps = frame["ts"].astype("int64")
        asof = frame["asof"].astype("int64")
    except (TypeError, ValueError) as error:
        raise DataValidationError("ts and asof must be integer timestamps") from error
    if (asof < timestamps).any():
        raise DataValidationError("asof cannot precede ts")
    spot_funding = frame.loc[frame["market_type"] == "spot", "funding_8h"]
    if spot_funding.notna().any():
        raise DataValidationError("spot rows cannot contain funding")

    membership = frame["in_universe"].astype(bool)
    prices = frame.loc[membership, PRICE_COLUMNS]
    if prices.isna().any().any():
        raise DataValidationError("missing in-universe price")
    if (prices <= 0).any().any():
        raise DataValidationError("nonpositive in-universe price")
    if (frame[["volume", "quote_volume"]].dropna() < 0).any().any():
        raise DataValidationError("negative volume")
    funding = frame["funding_8h"].dropna().astype(float)
    if (funding.abs() >= 0.05).any():
        raise DataValidationError("funding outside absolute 0.05 bound")


def _validate_gaps(
    frame: pd.DataFrame,
    listings: tuple[ListingBounds, ...],
    interval_ms: int,
) -> None:
    groups = {
        (str(symbol), str(market_type)): group
        for (symbol, market_type), group in frame.groupby(
            ["symbol", "market_type"], sort=True
        )
    }
    for listing in listings:
        key = (listing.symbol, listing.market_type)
        group = groups.get(key)
        if group is None:
            if listing.delisted_at is not None:
                raise DataValidationError(
                    f"gap detected for {listing.symbol} {listing.market_type}: no bars"
                )
            continue
        observed = set(group.loc[group["in_universe"], "ts"].astype("int64").tolist())
        expected = _expected_timestamps(group, listing, interval_ms)
        missing = expected.difference(observed)
        if missing:
            raise DataValidationError(
                f"gap detected for {listing.symbol} {listing.market_type}: "
                f"{len(missing)} missing bar(s)"
            )


def _validate_membership_bounds(
    frame: pd.DataFrame, listings: tuple[ListingBounds, ...]
) -> None:
    intervals: dict[tuple[str, str], list[tuple[int, int | None]]] = {}
    for listing in listings:
        key = (listing.symbol, listing.market_type)
        end = (
            _datetime_ms(listing.delisted_at)
            if listing.delisted_at is not None
            else None
        )
        intervals.setdefault(key, []).append((_datetime_ms(listing.listed_at), end))
    active = frame.loc[frame["in_universe"], ["ts", "symbol", "market_type"]]
    for timestamp, symbol, market_type in active.itertuples(index=False, name=None):
        bounds = intervals.get((str(symbol), str(market_type)))
        if bounds is None:
            continue
        ts = int(timestamp)
        if not any(start <= ts and (end is None or ts < end) for start, end in bounds):
            raise DataValidationError(
                f"in-universe row for {symbol} {market_type} is outside listing bounds"
            )


def _expected_timestamps(
    group: pd.DataFrame,
    listing: ListingBounds,
    interval_ms: int,
) -> set[int]:
    start = _ceil_to_interval(_datetime_ms(listing.listed_at), interval_ms)
    if listing.delisted_at is None:
        end = int(group["ts"].max())
    else:
        end = _floor_to_interval(_datetime_ms(listing.delisted_at) - 1, interval_ms)
    if end < start:
        return set()
    return set(range(start, end + 1, interval_ms))


def _datetime_ms(value: datetime) -> int:
    if value.tzinfo is None:
        raise DataValidationError("listing bounds must be timezone-aware")
    return int(value.timestamp() * 1000)


def _ceil_to_interval(value: int, interval: int) -> int:
    return ((value + interval - 1) // interval) * interval


def _floor_to_interval(value: int, interval: int) -> int:
    return (value // interval) * interval


def _survivorship_report(
    frame: pd.DataFrame, listings: tuple[ListingBounds, ...]
) -> ValidationReport:
    delisted = {
        (listing.symbol, listing.market_type)
        for listing in listings
        if listing.delisted_at is not None
    }
    live = {
        (listing.symbol, listing.market_type)
        for listing in listings
        if listing.delisted_at is None
    }
    ratio = float("inf") if not live and delisted else len(delisted) / max(len(live), 1)
    if ratio < 0.05:
        warnings.warn(
            "survivorship warning: distinct delisted/live ratio "
            f"is {ratio:.2%}, below 5%",
            RuntimeWarning,
            stacklevel=2,
        )
    return ValidationReport(
        row_count=len(frame),
        live_symbol_count=len(live),
        delisted_symbol_count=len(delisted),
        delisted_live_ratio=ratio,
    )

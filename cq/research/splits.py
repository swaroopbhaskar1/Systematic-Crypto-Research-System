"""Canonical development, walk-forward, and holdout time splits."""

from collections.abc import Iterable

import pandas as pd

from cq.research.holdout import HoldoutLockedError

DEV_END = pd.Timestamp("2024-06-30", tz="UTC")
WALKFWD_END = pd.Timestamp("2025-06-30", tz="UTC")
HOLDOUT_START = WALKFWD_END + pd.Timedelta(days=1)


def as_utc(stamp: pd.Timestamp | str) -> pd.Timestamp:
    """Return a UTC timestamp, localizing naive values."""
    parsed = pd.Timestamp(stamp)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def as_utc_ms(stamp: pd.Timestamp | str) -> int:
    """Return Unix milliseconds for a UTC timestamp."""
    return int(as_utc(stamp).value // 1_000_000)


def from_utc_ms(value: object) -> pd.Timestamp:
    """Interpret a panel index value as UTC milliseconds."""
    return pd.Timestamp(int(value), unit="ms", tz="UTC")


def assert_research_timestamps(timestamps: Iterable[object]) -> None:
    """Reject any timestamp that falls in the locked holdout period."""
    for value in timestamps:
        stamp = value if isinstance(value, pd.Timestamp) else from_utc_ms(value)
        if as_utc(stamp) >= HOLDOUT_START:
            raise HoldoutLockedError("holdout timestamps are locked")

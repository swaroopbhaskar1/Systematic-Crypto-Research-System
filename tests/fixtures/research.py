"""Deterministic fixtures for validation-harness tests."""

from dataclasses import dataclass

import pandas as pd

from cq.data.panel import Panel

DAY_MS = 86_400_000
DEV_END = pd.Timestamp("2024-06-30", tz="UTC")
WALKFWD_END = pd.Timestamp("2025-06-30", tz="UTC")


@dataclass(frozen=True)
class CountableHypothesis:
    """Minimal hypothesis fields hashed by the counting log."""

    id: str
    entry_rule: str
    exit_rule: str
    universe_filter: str
    direction: str


def utc_ms(stamp: pd.Timestamp) -> int:
    """Return Unix milliseconds for a UTC timestamp."""
    aware = stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")
    return int(aware.value // 1_000_000)


def daily_index(start: str, end: str) -> pd.DatetimeIndex:
    """Return an inclusive UTC daily index."""
    return pd.date_range(start, end, freq="D", tz="UTC")


def make_split_panel(
    start: str = "2023-07-01",
    end: str = "2025-12-31",
    *,
    symbols: tuple[str, ...] = ("AAAUSDT", "BBBUSDT"),
) -> Panel:
    """Build a daily panel covering development, walk-forward, and holdout."""
    stamps = daily_index(start, end)
    rows: list[dict[str, object]] = []
    for offset, stamp in enumerate(stamps):
        for symbol_index, symbol in enumerate(symbols):
            close = 100.0 + symbol_index * 10.0 + (offset % 7)
            rows.append(
                {
                    "ts": utc_ms(stamp),
                    "symbol": symbol,
                    "market_type": "perp",
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000.0,
                    "quote_volume": 1e12,
                    "funding_8h": 0.0001,
                    "adv": 1e12,
                    "volatility": 0.01,
                    "liquidity_decile": 10,
                    "in_universe": True,
                }
            )
    return Panel.from_long(pd.DataFrame(rows), market_type="perp")

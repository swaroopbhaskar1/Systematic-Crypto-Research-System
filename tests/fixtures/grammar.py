"""Shared M3 grammar contracts and deterministic market-data fixtures."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cq.data.panel import Panel

DAY_MS = 86_400_000
HISTORY_PERIODS = 8
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")


@dataclass(frozen=True)
class GrammarHypothesis:
    """Minimal hypothesis-shaped object accepted by the M3 compiler."""

    entry_rule: str
    exit_rule: str
    data_required: tuple[str, ...]
    mode: str = "long_only"


def make_grammar_panel(
    periods: int = HISTORY_PERIODS,
    *,
    extreme_future: bool = False,
) -> Panel:
    """Build a panel with changing membership and every grammar column."""
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for period in range(periods):
            close = _close_value(symbol_index, period)
            if extreme_future and period >= HISTORY_PERIODS:
                close = (symbol_index + 1) * 1_000_000.0 * (period + 1)
            in_universe = not (
                (symbol == "DDDUSDT" and period in {0, 1, 5})
                or (symbol == "CCCUSDT" and period == 4)
            )
            rows.append(
                {
                    "ts": period * DAY_MS,
                    "symbol": symbol,
                    "market_type": "perp",
                    "open": close - 0.25,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 100.0 + 10.0 * symbol_index + period,
                    "quote_volume": close * (100.0 + period),
                    "funding_8h": (
                        (-1.0 if symbol_index % 2 else 1.0)
                        * (period + 1)
                        / 10_000.0
                    ),
                    "open_interest": (
                        1_000.0 + 100.0 * symbol_index + 5.0 * period
                    ),
                    "in_universe": in_universe,
                }
            )
    return Panel.from_long(pd.DataFrame(rows), market_type="perp")


def _close_value(symbol_index: int, period: int) -> float:
    if symbol_index == 0:
        return 10.0 + period
    if symbol_index == 1:
        return 20.0 - period
    if symbol_index == 2:
        return 14.0 + (1.0 if period % 2 else -1.0)
    return 8.0 + 2.0 * period


SAMPLE_HYPOTHESES = (
    GrammarHypothesis(
        "lag(close, 1) < close",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "close > roll_mean(close, 3)",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "roll_std(close, 3) > 0",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "roll_z(close, 3) > 0",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "close >= roll_pct(close, 3, 0.5)",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "close >= roll_min(close, 3)",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "close <= roll_max(close, 3)",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "pct_change(close, 1) > 0",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "xs_rank(volume) >= 0.5",
        "volume < 0",
        ("volume",),
    ),
    GrammarHypothesis(
        "xs_z(open_interest) > 0",
        "open_interest < 0",
        ("open_interest",),
    ),
    GrammarHypothesis(
        "abs(funding_8h) > 0",
        "funding_8h == 0",
        ("funding_8h",),
    ),
    GrammarHypothesis(
        "sign(pct_change(close, 1)) > 0",
        "close < 0",
        ("close",),
    ),
    GrammarHypothesis(
        "log(quote_volume) > 0",
        "quote_volume <= 0",
        ("quote_volume",),
    ),
    GrammarHypothesis(
        "clip(funding_8h, -0.0005, 0.0005) >= -0.0005",
        "funding_8h == 0",
        ("funding_8h",),
    ),
    GrammarHypothesis(
        "(high - low) / open > 0",
        "open <= 0",
        ("high", "low", "open"),
    ),
    GrammarHypothesis(
        "(close > open) and (volume > 0)",
        "close <= 0",
        ("close", "open", "volume"),
    ),
    GrammarHypothesis(
        "(close > open) or (funding_8h > 0)",
        "close <= 0",
        ("close", "funding_8h", "open"),
    ),
    GrammarHypothesis(
        "not (close < open)",
        "close <= 0",
        ("close", "open"),
    ),
    GrammarHypothesis(
        "xs_rank(roll_z(close, 3)) > 0.5",
        "close < 0",
        ("close",),
        mode="market_neutral",
    ),
    GrammarHypothesis(
        "roll_mean(volume, 2) > lag(volume, 1)",
        "volume <= 0",
        ("volume",),
        mode="short_only",
    ),
)


def assert_aligned(result: pd.DataFrame, panel: Panel) -> None:
    """Assert the common shape contract for evaluated expressions and weights."""
    expected = panel.field("close")
    assert result.index.equals(expected.index)
    assert result.columns.equals(expected.columns)
    assert result.columns.name == "symbol"
    assert result.shape == expected.shape


def finite_cells(frame: pd.DataFrame) -> np.ndarray:
    """Return finite numeric cells, ignoring unavailable NaN entries."""
    values = frame.to_numpy(dtype=float)
    return values[np.isfinite(values)]


def has_finite_cells(frame: pd.DataFrame) -> bool:
    """Return whether any cell is a finite number."""
    return bool(np.isfinite(frame.to_numpy(dtype=float)).any())

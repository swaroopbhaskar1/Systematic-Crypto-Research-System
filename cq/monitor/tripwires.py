"""Mechanical strategy tripwires. No discretionary override path."""

from __future__ import annotations

from enum import Enum

import pandas as pd

from cq.backtest.metrics import annualized_sharpe

HALVE_RATIO = 0.5
SLIPPAGE_HALT_RATIO = 1.5
LOOKBACK_BARS = 90


class Action(str, Enum):
    """Pre-committed monitoring actions."""

    NONE = "none"
    HALVE = "halve"
    CUT = "cut"
    HALT = "halt"


def evaluate(
    strategy_id: str,
    live: pd.Series,
    oos_sharpe: float,
    *,
    prior_halve_bars: int = 0,
    slippage_ratio_30d: float = 1.0,
    reconciliation_mismatch: bool = False,
) -> Action:
    """Return HALT, CUT, HALVE, or NONE from mechanical rules only."""
    if not strategy_id:
        raise ValueError("strategy_id is required")
    if reconciliation_mismatch or slippage_ratio_30d > SLIPPAGE_HALT_RATIO:
        return Action.HALT
    sharpe = _rolling_sharpe(live)
    threshold = HALVE_RATIO * oos_sharpe
    if sharpe < threshold and prior_halve_bars >= LOOKBACK_BARS:
        return Action.CUT
    if sharpe < threshold:
        return Action.HALVE
    return Action.NONE


def _rolling_sharpe(live: pd.Series) -> float:
    if live.empty:
        raise ValueError("live returns are required")
    window = live.iloc[-LOOKBACK_BARS:] if len(live) > LOOKBACK_BARS else live
    try:
        return annualized_sharpe(float(value) for value in window)
    except ValueError:
        return float("-inf")

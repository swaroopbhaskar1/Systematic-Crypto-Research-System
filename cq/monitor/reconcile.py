"""Intended versus actual position reconciliation."""

from __future__ import annotations


def positions_mismatch(
    intended: dict[str, float],
    actual: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return True when any symbol's actual position disagrees."""
    symbols = set(intended) | set(actual)
    for symbol in symbols:
        if abs(intended.get(symbol, 0.0) - actual.get(symbol, 0.0)) > tolerance:
            return True
    return False

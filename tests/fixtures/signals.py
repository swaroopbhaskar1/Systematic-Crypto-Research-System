"""Deterministic target-weight signals for backtest boundary tests only.

``Signal.compute`` always returns unlagged portfolio weights, never token
quantities.  Conversion to quantities belongs exclusively to the engine.
These M2 fixtures intentionally permit partial weights so individual
arithmetic and timing contracts can be isolated before the portfolio
compiler enforces long-only rows summing to one and market-neutral rows
netting to zero.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from cq.data.panel import Panel


class Signal(Protocol):
    """Minimal interface accepted by the backtest engine."""

    def compute(self, panel: Panel) -> pd.DataFrame:
        """Return unlagged target weights indexed and columned like the panel."""
        ...


def _tradable(panel: Panel, close: pd.DataFrame) -> pd.DataFrame:
    return panel.universe_mask() & close.notna()


@dataclass(frozen=True)
class ScheduledWeightSignal:
    """Return an explicit target-weight schedule aligned to the panel."""

    weights: pd.DataFrame

    def compute(self, panel: Panel) -> pd.DataFrame:
        expected = panel.field("close")
        assert self.weights.index.equals(expected.index)
        assert self.weights.columns.equals(expected.columns)
        return self.weights.copy()


@dataclass(frozen=True)
class ConstantWeightSignal:
    """Emit one explicit target weight per tradable symbol."""

    weight: float = 1.0

    def compute(self, panel: Panel) -> pd.DataFrame:
        close = panel.field("close").astype(float)
        weights = pd.DataFrame(
            self.weight,
            index=close.index,
            columns=close.columns,
            dtype=float,
        )
        return weights.where(_tradable(panel, close), 0.0)


@dataclass(frozen=True)
class TrailingMomentumSignal:
    """Emit signed weights using only current and trailing closes.

    This is deliberately a signal fixture, not a portfolio compiler: with one
    symbol its target weight is -1, 0, or +1.  The engine applies the only
    execution lag and converts that target weight to a token quantity.
    """

    lookback: int = 20

    def __post_init__(self) -> None:
        if self.lookback < 1:
            raise ValueError("lookback must be positive")

    def compute(self, panel: Panel) -> pd.DataFrame:
        close = panel.field("close").astype(float)
        trailing = close.shift(self.lookback)
        weights = close.gt(trailing).astype(float) - close.lt(trailing).astype(float)
        return weights.where(_tradable(panel, close) & trailing.notna(), 0.0)


@dataclass(frozen=True)
class CheatingNextCloseSignal:
    """Emit signed target weights using the unknowable close at ``t + 1``.

    This fixture is a vulnerability control.  The M2 engine is not expected to
    sandbox or reject arbitrary signal code; extension-invariance and the
    impossible performance of this signal demonstrate the lookahead.
    """

    def compute(self, panel: Panel) -> pd.DataFrame:
        close = panel.field("close").astype(float)
        next_close = pd.DataFrame(
            float("nan"),
            index=close.index,
            columns=close.columns,
        )
        if len(close.index) > 1:
            next_close.iloc[:-1] = close.iloc[1:].to_numpy()

        weights = np.sign(next_close - close)
        return weights.where(
            _tradable(panel, close) & next_close.notna(),
            0.0,
        )

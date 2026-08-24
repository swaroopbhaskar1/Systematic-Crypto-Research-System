"""Regime-conditional strategy correlation and effective breadth.

Correlations converge to 1 in crashes.  Strategies that look independent for
two years all lose together on the bad day, so diversification is weakest
exactly when it is needed.  An unconditional correlation matrix is therefore a
risk model that lies; everything here is measured conditional on a regime, and
a sample too thin to support an estimate raises instead of returning a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MINIMUM_REGIME_OBSERVATIONS = 30
ABSOLUTE_MINIMUM_OBSERVATIONS = 10
SYMMETRY_TOLERANCE = 1e-12
UNIT_DIAGONAL_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CorrelationConvergence:
    """How much of the measured diversification survives a stressed regime."""

    strategies: int
    stressed_observations: int
    unconditional_mean_correlation: float
    stressed_mean_correlation: float
    convergence: float


def _require_return_panel(strategy_returns: pd.DataFrame) -> None:
    """Reject anything that is not a finite panel of at least two strategies."""

    if len(strategy_returns.columns) < 2:
        raise ValueError("correlation requires at least two strategies")
    if len(strategy_returns.index) == 0:
        raise ValueError("strategy_returns must not be empty")
    for dtype in strategy_returns.dtypes:
        if is_bool_dtype(dtype) or not is_numeric_dtype(dtype):
            raise TypeError("strategy_returns must contain real numbers")
    values: npt.NDArray[np.float64] = strategy_returns.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("strategy_returns must contain only finite values")


def _require_regime_labels(
    regime_labels: pd.Series,
    *,
    index: pd.Index,
) -> None:
    """Reject labels that are misaligned, non-string, or empty."""

    if not regime_labels.index.equals(index):
        raise ValueError("regime_labels must share the strategy returns index")
    for label in regime_labels:
        if not isinstance(label, str):
            raise TypeError("regime labels must be strings")
        if not label:
            raise ValueError("regime labels must not be empty")


def _require_minimum(min_observations: int) -> int:
    """Return a validated observation floor, refusing an implausibly low one."""

    if isinstance(min_observations, bool) or not isinstance(min_observations, Integral):
        raise TypeError("min_observations must be an integer")
    minimum = int(min_observations)
    if minimum < ABSOLUTE_MINIMUM_OBSERVATIONS:
        raise ValueError(
            "min_observations must be at least "
            f"{ABSOLUTE_MINIMUM_OBSERVATIONS} observations"
        )
    return minimum


def _correlation_matrix(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    """Return the pairwise Pearson correlation, raising on a degenerate one."""

    if bool((frame.std(ddof=1) == 0.0).any()):
        raise ValueError(f"{label} contains a strategy with zero variance")
    matrix = frame.corr(method="pearson")
    if not bool(np.isfinite(matrix.to_numpy(dtype=np.float64)).all()):
        raise ValueError(f"{label} produced a non-finite correlation")
    return matrix


def _mean_pairwise(matrix: pd.DataFrame) -> float:
    """Return the mean of the strictly upper-triangular correlations."""

    values: npt.NDArray[np.float64] = matrix.to_numpy(dtype=np.float64)
    rows, columns = np.triu_indices(values.shape[0], k=1)
    return float(np.mean(values[rows, columns]))


def _regime_rows(
    strategy_returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    regime: str,
    minimum: int,
) -> pd.DataFrame:
    """Return only the rows in ``regime``, raising if the sample is too thin."""

    if not regime:
        raise ValueError("regime must be a non-empty string")
    selected: pd.DataFrame = strategy_returns.loc[
        (regime_labels == regime).to_numpy(dtype=bool)
    ]
    count = len(selected.index)
    if count < minimum:
        raise ValueError(
            f"regime {regime!r} has {count} observations, "
            f"below the required {minimum}"
        )
    return selected


def conditional_correlation(
    strategy_returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    regime: str,
    min_observations: int = MINIMUM_REGIME_OBSERVATIONS,
) -> pd.DataFrame:
    """Pairwise correlation using only the rows labelled ``regime``.

    ``regime_labels`` must be indexed identically to ``strategy_returns``.
    Fewer than ``min_observations`` rows in the regime raises: a correlation
    estimated from a handful of observations is noise presented as a risk
    estimate, and its standard error alone is roughly ``1 / sqrt(n - 3)``.  The
    default floor of 30 keeps that error under about 0.2.  ``min_observations``
    itself may not be set below ``ABSOLUTE_MINIMUM_OBSERVATIONS``.
    """

    _require_return_panel(strategy_returns)
    _require_regime_labels(regime_labels, index=strategy_returns.index)
    minimum = _require_minimum(min_observations)
    selected = _regime_rows(
        strategy_returns,
        regime_labels,
        regime=regime,
        minimum=minimum,
    )
    return _correlation_matrix(selected, label=f"regime {regime!r}")


def correlation_convergence(
    strategy_returns: pd.DataFrame,
    regime_labels: pd.Series,
    *,
    stressed_regime: str,
    min_observations: int = MINIMUM_REGIME_OBSERVATIONS,
) -> CorrelationConvergence:
    """Compare unconditional and stressed-regime mean pairwise correlation.

    ``convergence`` is ``stressed - unconditional``.  A positive value is the
    amount of diversification that disappears in the stressed regime, which is
    the number that says the risk model is more optimistic than reality.  Both
    samples must clear ``min_observations``.
    """

    _require_return_panel(strategy_returns)
    _require_regime_labels(regime_labels, index=strategy_returns.index)
    minimum = _require_minimum(min_observations)
    if len(strategy_returns.index) < minimum:
        raise ValueError(
            f"the full sample has {len(strategy_returns.index)} observations, "
            f"below the required {minimum}"
        )
    stressed_rows = _regime_rows(
        strategy_returns,
        regime_labels,
        regime=stressed_regime,
        minimum=minimum,
    )
    unconditional = _mean_pairwise(
        _correlation_matrix(strategy_returns, label="the full sample")
    )
    stressed = _mean_pairwise(
        _correlation_matrix(stressed_rows, label=f"regime {stressed_regime!r}")
    )
    return CorrelationConvergence(
        strategies=len(strategy_returns.columns),
        stressed_observations=len(stressed_rows.index),
        unconditional_mean_correlation=unconditional,
        stressed_mean_correlation=stressed,
        convergence=stressed - unconditional,
    )


def _require_correlation_matrix(correlation: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return a validated square, symmetric, unit-diagonal correlation matrix."""

    if len(correlation.index) == 0 or len(correlation.columns) == 0:
        raise ValueError("correlation must not be empty")
    if len(correlation.index) != len(correlation.columns):
        raise ValueError("correlation must be a square matrix")
    if not correlation.index.equals(correlation.columns):
        raise ValueError("correlation must use the same labels on both axes")
    values: npt.NDArray[np.float64] = correlation.to_numpy(dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("correlation must contain only finite values")
    if not bool(np.allclose(values, values.T, atol=SYMMETRY_TOLERANCE, rtol=0.0)):
        raise ValueError("correlation must be symmetric")
    if float(np.abs(np.diag(values) - 1.0).max()) > UNIT_DIAGONAL_TOLERANCE:
        raise ValueError("correlation must have a unit diagonal")
    if float(np.abs(values).max()) > 1.0 + UNIT_DIAGONAL_TOLERANCE:
        raise ValueError("correlation entries must lie in [-1, 1]")
    return values


def effective_breadth(correlation: pd.DataFrame) -> float:
    """Return ``N / (1 + (N - 1) * rho)`` for ``N`` strategies.

    The Fundamental Law gives ``Information Ratio ~ IC * sqrt(Breadth)``, and
    correlated strategies do not supply independent bets.  ``rho`` is the mean
    off-diagonal correlation, so breadth is ``N`` when ``rho`` is 0 and 1 when
    ``rho`` is 1.  A single strategy has breadth 1.  A ``rho`` at or below
    ``-1 / (N - 1)`` makes the denominator non-positive and raises rather than
    reporting unbounded breadth.
    """

    values = _require_correlation_matrix(correlation)
    size = values.shape[0]
    if size == 1:
        return 1.0
    rho = _mean_pairwise(correlation)
    denominator = 1.0 + float(size - 1) * rho
    if denominator <= 0.0:
        raise ValueError("effective breadth is undefined for this correlation")
    breadth = float(size) / denominator
    if not math.isfinite(breadth):
        raise ValueError("effective breadth is undefined for this correlation")
    return breadth

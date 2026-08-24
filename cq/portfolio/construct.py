"""Combining strategy signals into target weights.

This module deliberately contains **no portfolio optimizer**.  Per the build
spec's out-of-scope list, strategies are combined equal-weight until there is
out-of-sample evidence to weight on; mean-variance, risk parity, and any other
covariance-inverting allocator are forbidden here because a covariance matrix
estimated from a short crypto sample is mostly noise.

All inputs must be finite.  Missing weights raise rather than being imputed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

DEFAULT_DECILE_FRACTION = 0.1
MAXIMUM_LEG_FRACTION = 0.5
TARGET_GROSS_EXPOSURE = 1.0
NEUTRALITY_TOLERANCE = 1e-12


def _require_numeric_frame(frame: pd.DataFrame, *, name: str) -> None:
    """Reject an empty frame or one that does not hold real numbers."""

    if len(frame.index) == 0 or len(frame.columns) == 0:
        raise ValueError(f"{name} must not be empty")
    for dtype in frame.dtypes:
        if is_bool_dtype(dtype) or not is_numeric_dtype(dtype):
            raise TypeError(f"{name} must contain real numbers")


def _finite_matrix(frame: pd.DataFrame, *, name: str) -> npt.NDArray[np.float64]:
    """Return the frame as a float matrix, rejecting NaN and infinities."""

    _require_numeric_frame(frame, name=name)
    values: npt.NDArray[np.float64] = frame.to_numpy(dtype=np.float64, copy=True)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _require_fraction(value: float, *, name: str, upper: float) -> float:
    """Return ``value`` as a float in ``(0, upper]``, rejecting bools."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if not 0.0 < number <= upper:
        raise ValueError(f"{name} must lie in (0, {upper}]")
    return number


def combine_equal_weight(signals: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average per-strategy target weights, one equal share per strategy.

    Every frame must share an identical index and identical columns in the same
    order, and contain only finite values; a missing weight raises rather than
    being read as zero.  This is the only across-strategy combination rule the
    spec permits until out-of-sample evidence justifies another.
    """

    if len(signals) == 0:
        raise ValueError("signals must contain at least one strategy")
    reference = signals[0]
    _require_numeric_frame(reference, name="signals[0]")
    total = np.zeros((len(reference.index), len(reference.columns)), dtype=np.float64)
    for position, frame in enumerate(signals):
        values = _finite_matrix(frame, name=f"signals[{position}]")
        if not frame.index.equals(reference.index):
            raise ValueError("every signal frame must share an identical index")
        if not frame.columns.equals(reference.columns):
            raise ValueError("every signal frame must share identical columns")
        total += values
    return pd.DataFrame(
        total / float(len(signals)),
        index=reference.index,
        columns=reference.columns,
    )


def apply_position_cap(weights: pd.DataFrame, *, cap: float) -> pd.DataFrame:
    """Clip every weight to ``cap`` times its row's gross exposure.

    Convention, chosen for determinism: the reference gross exposure is the
    row's gross **before** clipping, and the row is **not** re-normalized
    afterwards.  A binding cap therefore lowers gross exposure instead of
    pushing the freed exposure into the remaining names.  Re-normalizing would
    immediately re-violate the cap and require an iterative fixed point, so the
    single pass is preferred.  Signs are preserved and all-zero rows are
    returned unchanged.
    """

    values = _finite_matrix(weights, name="weights")
    limit = _require_fraction(cap, name="cap", upper=1.0)
    ceiling = limit * np.abs(values).sum(axis=1, keepdims=True)
    clipped = np.minimum(np.abs(values), ceiling) * np.sign(values)
    return pd.DataFrame(clipped, index=weights.index, columns=weights.columns)


def _neutral_row(
    row: npt.NDArray[np.float64],
    *,
    leg_fraction: float,
) -> npt.NDArray[np.float64]:
    """Return dollar-matched weights for one cross-section.

    NaN entries mean "no data" and are excluded from the ranking rather than
    imputed.  Ties are broken by column order via a stable sort.  A row that
    cannot fill both legs returns all zeros rather than a partial book.
    """

    weights = np.zeros_like(row)
    ranked = np.flatnonzero(~np.isnan(row))
    leg = int(math.floor(len(ranked) * leg_fraction))
    if leg < 1 or 2 * leg > len(ranked):
        return weights
    order = ranked[np.argsort(row[ranked], kind="stable")]
    magnitude = TARGET_GROSS_EXPOSURE / 2.0 / float(leg)
    weights[order[:leg]] = -magnitude
    weights[order[len(ranked) - leg :]] = magnitude
    return weights


def cross_sectional_neutral(
    scores: pd.DataFrame,
    *,
    decile_fraction: float = DEFAULT_DECILE_FRACTION,
    per_position_cap: float,
) -> pd.DataFrame:
    """Long the top fraction and short the bottom fraction of each row.

    Each leg holds ``floor(finite_scores * decile_fraction)`` names at equal
    magnitude, so the row is dollar-matched to ``0.0`` within
    ``NEUTRALITY_TOLERANCE`` and BTC direction cancels.  Gross exposure is
    ``TARGET_GROSS_EXPOSURE`` before ``apply_position_cap`` is applied with
    ``per_position_cap``; because both legs carry equal magnitudes the cap
    cannot break neutrality.  Rows too thin to fill both legs are all zero.
    ``NaN`` means "no score" and is excluded from the ranking, never imputed.
    """

    _require_numeric_frame(scores, name="scores")
    leg_fraction = _require_fraction(
        decile_fraction,
        name="decile_fraction",
        upper=MAXIMUM_LEG_FRACTION,
    )
    _require_fraction(per_position_cap, name="per_position_cap", upper=1.0)
    values: npt.NDArray[np.float64] = scores.to_numpy(dtype=np.float64, copy=True)
    if bool(np.isinf(values).any()):
        raise ValueError("scores must contain only finite values or NaN")
    rows = [_neutral_row(row, leg_fraction=leg_fraction) for row in values]
    weights = pd.DataFrame(
        np.vstack(rows),
        index=scores.index,
        columns=scores.columns,
    )
    capped = apply_position_cap(weights, cap=per_position_cap)
    _assert_dollar_matched(capped)
    return capped


def _assert_dollar_matched(weights: pd.DataFrame) -> None:
    """Raise if any row's net exposure drifts beyond ``NEUTRALITY_TOLERANCE``."""

    worst = float(weights.sum(axis="columns").abs().max())
    if worst > NEUTRALITY_TOLERANCE:
        raise AssertionError(
            f"cross-sectional weights are not dollar-matched: net {worst:.3e}"
        )

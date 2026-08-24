"""Within-strategy position sizing and across-strategy weight limits.

Sizing is roughly half the system: a real edge sized badly still loses money.
Every function here refuses to invent a number.  A token with no trailing
volatility gets no position rather than a default volatility, because a
defaulted volatility is exactly how a thin-data token acquires an enormous
weight.  Windows are trailing only; there is no way to request a centered one.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

ANNUALIZATION_DAYS = 365
HALF_KELLY = 0.5
MAXIMUM_GROSS_EXPOSURE = 1.0
DEFAULT_MAX_FRACTIONAL_CHANGE = 0.25
MINIMUM_VOLATILITY_WINDOW = 2


def _require_numeric_frame(frame: pd.DataFrame, *, name: str) -> None:
    """Reject an empty frame or one that does not hold real numbers."""

    if len(frame.index) == 0 or len(frame.columns) == 0:
        raise ValueError(f"{name} must not be empty")
    for dtype in frame.dtypes:
        if is_bool_dtype(dtype) or not is_numeric_dtype(dtype):
            raise TypeError(f"{name} must contain real numbers")


def _matrix(frame: pd.DataFrame, *, name: str) -> npt.NDArray[np.float64]:
    """Return the frame as a float matrix, rejecting infinities but not NaN."""

    _require_numeric_frame(frame, name=name)
    values: npt.NDArray[np.float64] = frame.to_numpy(dtype=np.float64, copy=True)
    if bool(np.isinf(values).any()):
        raise ValueError(f"{name} must contain only finite values or NaN")
    return values


def _finite_matrix(frame: pd.DataFrame, *, name: str) -> npt.NDArray[np.float64]:
    """Return the frame as a float matrix, rejecting NaN and infinities."""

    values = _matrix(frame, name=name)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    return values


def _require_positive(value: float, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, rejecting bools."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return number


def _require_aligned(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Raise unless both frames carry the same index and the same columns."""

    if not left.index.equals(right.index):
        raise ValueError("frames must share an identical index")
    if not left.columns.equals(right.columns):
        raise ValueError("frames must share identical columns")


def volatility_scaled_weights(
    raw_weights: pd.DataFrame,
    trailing_volatility: pd.DataFrame,
    *,
    target_volatility: float,
) -> pd.DataFrame:
    """Scale weights by ``target_volatility / trailing_volatility``.

    Risk contribution is then roughly constant across tokens and across time,
    so a 60%-vol token and a 200%-vol token do not carry the same exposure.
    A nonzero raw weight whose trailing volatility is missing, non-positive, or
    infinite raises: no default volatility is ever substituted.  A zero raw
    weight stays zero even where volatility is missing.  The result is not
    capped or de-levered; compose with :func:`enforce_no_leverage`.
    """

    weights = _finite_matrix(raw_weights, name="raw_weights")
    volatility = _matrix(trailing_volatility, name="trailing volatility")
    _require_aligned(raw_weights, trailing_volatility)
    target = _require_positive(target_volatility, name="target_volatility")
    active = weights != 0.0
    usable = np.isfinite(volatility) & (volatility > 0.0)
    if bool((active & ~usable).any()):
        raise ValueError(
            "trailing volatility must be finite and positive wherever a "
            "weight is nonzero"
        )
    scaled = np.divide(
        weights * target,
        volatility,
        out=np.zeros_like(weights),
        where=active,
    )
    return pd.DataFrame(
        scaled,
        index=raw_weights.index,
        columns=raw_weights.columns,
    )


def trailing_realized_volatility(
    returns: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Return trailing annualized realized volatility over ``window`` bars.

    Sample standard deviation (``ddof=1``) of the trailing window, annualized
    by ``sqrt(365)``.  ``min_periods`` equals ``window``, so a partial or
    NaN-punctured window yields ``NaN`` rather than a fabricated number.  The
    signature exposes no ``center`` parameter, so a centered window cannot be
    requested; the value at ``T`` depends only on bars at or before ``T``.
    """

    _require_numeric_frame(returns, name="returns")
    if isinstance(window, bool) or not isinstance(window, Integral):
        raise TypeError("window must be an integer")
    size = int(window)
    if size < MINIMUM_VOLATILITY_WINDOW:
        raise ValueError(
            f"window must be at least {MINIMUM_VOLATILITY_WINDOW} observations"
        )
    if size > len(returns.index):
        raise ValueError("window must not exceed the available history")
    _matrix(returns, name="returns")
    rolling = returns.rolling(window=size, min_periods=size, center=False)
    return rolling.std(ddof=1) * math.sqrt(ANNUALIZATION_DAYS)


def _kelly_fraction(fraction: float) -> float:
    """Return a validated Kelly fraction in ``(0, 0.5]``."""

    if isinstance(fraction, bool) or not isinstance(fraction, Real):
        raise TypeError("fraction must be a real number")
    share = float(fraction)
    if not math.isfinite(share):
        raise ValueError("fraction must be finite")
    if share <= 0.0:
        raise ValueError("fraction must be strictly positive")
    if share > HALF_KELLY:
        raise ValueError("fraction must be half-Kelly or less (at most 0.5)")
    return share


def fractional_kelly_scale(edge_scale: float, *, fraction: float = HALF_KELLY) -> float:
    """Return ``edge_scale * fraction`` for a fraction of at most half-Kelly.

    Full Kelly assumes the edge is known exactly, and it never is; over-betting
    a real edge is the common way people with genuine edge go broke.  A
    ``fraction`` above 0.5 raises.  A negative ``edge_scale`` also raises: a
    negative Kelly fraction means "do not take this bet", and silently
    returning it would flip position signs instead.
    """

    if isinstance(edge_scale, bool) or not isinstance(edge_scale, Real):
        raise TypeError("edge_scale must be a real number")
    scale = float(edge_scale)
    if not math.isfinite(scale):
        raise ValueError("edge_scale must be finite")
    if scale < 0.0:
        raise ValueError("edge_scale must not be negative")
    return scale * _kelly_fraction(fraction)


def enforce_no_leverage(weights: pd.DataFrame) -> pd.DataFrame:
    """Scale any row with gross exposure above 1.0 down to gross 1.0.

    Relative sizes and signs within the row are preserved.  Rows already at or
    below gross 1.0 pass through untouched — this de-levers, it never levers up
    an under-invested row.  No leverage, not at this stage.
    """

    values = _finite_matrix(weights, name="weights")
    gross = np.abs(values).sum(axis=1, keepdims=True)
    scale = np.divide(
        MAXIMUM_GROSS_EXPOSURE,
        gross,
        out=np.ones_like(gross),
        where=gross > MAXIMUM_GROSS_EXPOSURE,
    )
    return pd.DataFrame(
        values * scale,
        index=weights.index,
        columns=weights.columns,
    )


def _require_finite_series(series: pd.Series, *, name: str) -> npt.NDArray[np.float64]:
    """Return the series as a finite float vector, rejecting bools and NaN."""

    if len(series.index) == 0:
        raise ValueError(f"{name} must not be empty")
    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise TypeError(f"{name} must contain real numbers")
    values: npt.NDArray[np.float64] = series.to_numpy(dtype=np.float64, copy=True)
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")
    return values


def constrain_weight_change(
    previous: pd.Series,
    proposed: pd.Series,
    *,
    max_fractional_change: float = DEFAULT_MAX_FRACTIONAL_CHANGE,
) -> pd.Series:
    """Cap how far each across-strategy allocation weight may move in one step.

    A strategy earns weight by performing forward, so a lucky month must not be
    able to dominate allocation.  Each name may move at most
    ``max_fractional_change`` times a reference magnitude.

    Two conventions the spec leaves open, chosen here and fixed:

    * The reference magnitude is ``abs(previous)`` for that name.  Where the
      previous weight is exactly zero the fraction is undefined, so the
      reference becomes the mean absolute previous weight — a strategy new to
      the book is phased in rather than admitted at full size.
    * If every previous weight is zero there is nothing to constrain against
      and ``proposed`` passes through unchanged: this is the first allocation.

    The result is a pure clamp and is deliberately not re-normalized to sum to
    one; re-normalizing would silently reintroduce the move the cap removed.
    """

    prior = _require_finite_series(previous, name="previous")
    target = _require_finite_series(proposed, name="proposed")
    if not previous.index.equals(proposed.index):
        raise ValueError("previous and proposed must share an identical index")
    limit = _require_positive(max_fractional_change, name="max_fractional_change")
    if limit > MAXIMUM_GROSS_EXPOSURE:
        raise ValueError("max_fractional_change must not exceed 1.0")
    magnitudes = np.abs(prior)
    fallback = float(magnitudes.mean())
    if fallback == 0.0:
        return proposed.copy()
    allowed = limit * np.where(magnitudes > 0.0, magnitudes, fallback)
    return pd.Series(
        np.clip(target, prior - allowed, prior + allowed),
        index=proposed.index,
        name=proposed.name,
    )

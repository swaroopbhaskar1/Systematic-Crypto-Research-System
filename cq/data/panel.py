"""Market-type-scoped wide panel representation."""

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from pandas.core.window import Rolling

MarketType: TypeAlias = Literal["spot", "perp"]

KEY_COLUMNS = frozenset({"ts", "symbol", "market_type"})
MEMBERSHIP_COLUMN = "in_universe"

ADV_WINDOW = 30
VOLATILITY_WINDOW = 30
DECILE_COUNT = 10
FEATURE_COLUMNS = ("adv", "volatility", "liquidity_decile")
_FEATURE_SOURCE_COLUMNS = ("ts", "symbol", "close", "quote_volume")


@dataclass(frozen=True)
class Panel:
    """Wide data indexed by bar timestamp and columns ``(field, symbol)``."""

    _data: pd.DataFrame
    market_type: MarketType

    @classmethod
    def from_long(cls, frame: pd.DataFrame, *, market_type: MarketType) -> "Panel":
        missing = KEY_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"missing panel columns: {sorted(missing)}")
        if MEMBERSHIP_COLUMN not in frame.columns:
            raise ValueError("missing panel columns: ['in_universe']")
        scoped = frame.loc[frame["market_type"] == market_type].copy()
        other_types = set(frame["market_type"].dropna().unique()).difference(
            {market_type}
        )
        if other_types:
            raise ValueError(
                f"frame contains other market types: {sorted(other_types)}"
            )
        if scoped.duplicated(["ts", "symbol"]).any():
            raise ValueError("duplicate panel logical key")

        value_columns = sorted(set(scoped.columns).difference(KEY_COLUMNS))
        if not value_columns:
            raise ValueError("panel requires at least one value field")
        membership = scoped.loc[:, MEMBERSHIP_COLUMN]
        membership = membership.fillna(False).astype(bool)
        masked_columns = [
            column for column in value_columns if column != MEMBERSHIP_COLUMN
        ]
        scoped.loc[:, MEMBERSHIP_COLUMN] = membership
        scoped.loc[~membership, masked_columns] = pd.NA
        wide = scoped.pivot(index="ts", columns="symbol", values=value_columns)
        if not isinstance(wide.columns, pd.MultiIndex):
            raise TypeError("panel columns must have field and symbol levels")
        wide.columns = wide.columns.set_names(["field", "symbol"])
        wide = wide.sort_index().sort_index(axis=1)
        return cls(wide, market_type)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @property
    def symbols(self) -> tuple[str, ...]:
        values: Iterable[object] = self._data.columns.get_level_values(
            "symbol"
        ).unique()
        return tuple(str(value) for value in values)

    def slice(self, start: int, end: int) -> "Panel":
        if end < start:
            raise ValueError("slice end cannot precede start")
        sliced = self._data.loc[
            (self._data.index >= start) & (self._data.index <= end)
        ].copy()
        membership = (
            sliced.xs(MEMBERSHIP_COLUMN, axis=1, level="field")
            .fillna(False)
            .astype(bool)
        )
        retained = tuple(
            symbol
            for symbol in self.symbols
            if membership.loc[:, [symbol]].to_numpy(dtype=bool).any()
        )
        sliced = sliced.loc[
            :,
            sliced.columns.get_level_values("symbol").isin(retained),
        ]
        return Panel(sliced, self.market_type)

    def field(self, name: str) -> pd.DataFrame:
        if name not in self._data.columns.get_level_values("field"):
            raise KeyError(name)
        result = self._data.xs(name, axis=1, level="field", drop_level=True).copy()
        if not isinstance(result, pd.DataFrame):
            raise TypeError("panel field selection must be two-dimensional")
        result.columns.name = "symbol"
        return result

    def universe_mask(self) -> pd.DataFrame:
        return self.field(MEMBERSHIP_COLUMN).fillna(False).astype(bool)


def add_execution_features(
    frame: pd.DataFrame,
    *,
    adv_window: int = ADV_WINDOW,
    volatility_window: int = VOLATILITY_WINDOW,
) -> pd.DataFrame:
    """Derive the average volume, volatility, and liquidity decile columns.

    Every feature reads a trailing window that closes on the *previous* bar,
    because a trade executing at this bar's open cannot know this bar's own
    completed volume or return.  The volume cap is a separate rule and
    deliberately uses the current bar per the execution spec.

    A bar whose features are not computable leaves the universe rather than
    receiving an imputed value: a default volume or volatility would hand a
    brand-new or halted listing the execution profile of a liquid one.  This
    function may only remove tradability, never grant it.
    """

    _validate_feature_frame(frame, adv_window, volatility_window)
    result = frame.copy()
    grouped = result.sort_values(["symbol", "ts"], kind="stable")
    result["adv"] = _trailing_mean(grouped, "quote_volume", adv_window)
    result["volatility"] = _trailing_volatility(grouped, volatility_window)
    tradable = (
        result[MEMBERSHIP_COLUMN].fillna(False).astype(bool)
        & result["close"].notna()
        & _positive(result["adv"])
        & _positive(result["volatility"])
    )
    result["liquidity_decile"] = _liquidity_deciles(result, tradable)
    result[MEMBERSHIP_COLUMN] = tradable & result["liquidity_decile"].notna()
    result["liquidity_decile"] = (
        result["liquidity_decile"].where(result[MEMBERSHIP_COLUMN]).astype("Int64")
    )
    return result


def _validate_feature_frame(
    frame: pd.DataFrame,
    adv_window: int,
    volatility_window: int,
) -> None:
    _validated_window(adv_window, name="adv_window")
    _validated_window(volatility_window, name="volatility_window")
    missing = [
        column
        for column in (*_FEATURE_SOURCE_COLUMNS, MEMBERSHIP_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if frame.empty:
        raise ValueError("cannot derive features from an empty frame")
    if frame.duplicated(["ts", "symbol"]).any():
        raise ValueError("duplicate logical key in feature frame")
    existing = [column for column in FEATURE_COLUMNS if column in frame.columns]
    if existing:
        raise ValueError(f"features already present: {existing}")


def _validated_window(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    window = int(value)
    if window < 2:
        raise ValueError(f"{name} must span at least 2 bars")
    return window


def _exclusive_trailing(series: "pd.Series[float]", window: int) -> Rolling:
    """Return a trailing window that ends on the *previous* bar.

    ``closed="left"`` drops the current observation from the window, which is
    what makes these features knowable at the bar that uses them.  It is used
    in preference to lagging the result because the execution lag lives in
    exactly one place in this codebase, and that place is the engine.
    """

    return series.rolling(window, min_periods=window, closed="left")


def _trailing_mean(
    grouped: pd.DataFrame,
    column: str,
    window: int,
) -> "pd.Series[float]":
    values = grouped.groupby("symbol", sort=False)[column]
    rolled = values.transform(
        lambda series: _exclusive_trailing(series, window).mean()
    )
    return rolled.reindex(grouped.index).sort_index()


def _trailing_volatility(
    grouped: pd.DataFrame,
    window: int,
) -> "pd.Series[float]":
    closes = grouped.groupby("symbol", sort=False)["close"]
    rolled = closes.transform(
        lambda series: _exclusive_trailing(
            series.pct_change(fill_method=None),
            window,
        ).std()
    )
    return rolled.reindex(grouped.index).sort_index()


def _positive(values: "pd.Series[float]") -> "pd.Series[bool]":
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & np.isfinite(numeric.fillna(0.0)) & (numeric > 0.0)


def _liquidity_deciles(
    frame: pd.DataFrame,
    tradable: "pd.Series[bool]",
) -> "pd.Series[float]":
    """Rank each bar's tradable cross-section into ten liquidity buckets.

    Decile 1 is the least liquid and carries the widest spread.  Ranking uses
    only the symbols priced in that bar, so the assignment for a bar cannot
    move when later bars arrive.
    """

    ranks = (
        frame["adv"]
        .where(tradable)
        .groupby(frame["ts"], sort=False)
        .rank(method="first", pct=True, ascending=True)
    )
    scaled = np.ceil(ranks.to_numpy(dtype=float) * DECILE_COUNT)
    bounded = np.clip(scaled, 1.0, float(DECILE_COUNT))
    return pd.Series(bounded, index=frame.index).where(ranks.notna())

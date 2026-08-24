"""Market-type-scoped wide panel representation."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import pandas as pd

MarketType: TypeAlias = Literal["spot", "perp"]

KEY_COLUMNS = frozenset({"ts", "symbol", "market_type"})
MEMBERSHIP_COLUMN = "in_universe"


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

"""Immutable, append-only Parquet market data store."""

import hashlib
from pathlib import Path
from typing import BinaryIO

import pandas as pd

REQUIRED_COLUMNS = (
    "ts",
    "symbol",
    "market_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "funding_8h",
    "in_universe",
    "asof",
)
LOGICAL_KEY = ["ts", "symbol", "market_type"]
INTERNAL_COLUMNS = frozenset({"_file_order"})


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, frame: pd.DataFrame, *, timeframe: str) -> Path:
        """Append one immutable file to one timeframe/year partition."""
        self._validate_write(frame, timeframe)
        stored_columns: list[str] = list(REQUIRED_COLUMNS)
        stored_columns.extend(sorted(set(frame.columns).difference(REQUIRED_COLUMNS)))
        canonical = frame.loc[:, stored_columns].sort_values(
            LOGICAL_KEY + ["asof"], kind="stable"
        )
        years = pd.to_datetime(canonical["ts"], unit="ms", utc=True).dt.year
        unique_years = tuple(int(year) for year in years.unique())
        if len(unique_years) != 1:
            raise ValueError("each write must contain exactly one UTC year")

        partition = self.root / f"timeframe={timeframe}" / f"year={unique_years[0]}"
        partition.mkdir(parents=True, exist_ok=True)
        digest = self._content_digest(canonical)
        destination, handle = self._reserve(partition, digest)
        try:
            with handle:
                canonical.to_parquet(handle, index=False, engine="pyarrow")
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return destination

    def read(self, *, timeframe: str, query_ts: int) -> pd.DataFrame:
        """Read the latest revision whose as-of timestamp is eligible."""
        if not timeframe:
            raise ValueError("timeframe is required")
        partition_root = self.root / f"timeframe={timeframe}"
        files = sorted(partition_root.glob("year=*/part-*.parquet"))
        if not files:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        pieces: list[pd.DataFrame] = []
        for order, path in enumerate(files):
            piece = pd.read_parquet(path)
            piece["_file_order"] = order
            pieces.append(piece)
        combined = pd.concat(pieces, ignore_index=True)
        eligible = combined.loc[combined["asof"] <= query_ts].copy()
        if eligible.empty:
            return pd.DataFrame(columns=_public_columns(combined.columns))
        eligible.sort_values(
            LOGICAL_KEY + ["asof", "_file_order"],
            kind="stable",
            inplace=True,
        )
        latest = eligible.drop_duplicates(LOGICAL_KEY, keep="last")
        return (
            latest.loc[:, _public_columns(combined.columns)]
            .sort_values(LOGICAL_KEY, kind="stable")
            .reset_index(drop=True)
        )

    @staticmethod
    def _validate_write(frame: pd.DataFrame, timeframe: str) -> None:
        if not timeframe:
            raise ValueError("timeframe is required")
        missing = set(REQUIRED_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"missing store columns: {sorted(missing)}")
        reserved = INTERNAL_COLUMNS.intersection(frame.columns)
        if reserved:
            raise ValueError(f"reserved store columns: {sorted(reserved)}")
        if frame.empty:
            raise ValueError("cannot append an empty frame")
        if frame.duplicated(LOGICAL_KEY + ["asof"]).any():
            raise ValueError("duplicate logical revision in write")
        market_types = set(frame["market_type"].dropna().unique())
        invalid_market_types = market_types.difference({"spot", "perp"})
        if frame["market_type"].isna().any() or invalid_market_types:
            raise ValueError(
                f"unsupported market type values: {sorted(invalid_market_types)}"
            )
        if frame[["ts", "asof"]].isna().any().any():
            raise ValueError("ts and asof must be present")
        try:
            timestamps = frame["ts"].astype("int64")
            asof = frame["asof"].astype("int64")
        except (TypeError, ValueError) as error:
            raise ValueError("ts and asof must be integer timestamps") from error
        if (asof < timestamps).any():
            raise ValueError("asof cannot precede ts")

    @staticmethod
    def _content_digest(frame: pd.DataFrame) -> str:
        row_hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
        metadata = "|".join(
            f"{column}:{frame[column].dtype}" for column in frame.columns
        ).encode()
        row_bytes = row_hashes.to_numpy(dtype="uint64").tobytes()
        return hashlib.sha256(metadata + row_bytes).hexdigest()[:20]

    @staticmethod
    def _reserve(partition: Path, digest: str) -> tuple[Path, BinaryIO]:
        ordinal = 0
        while True:
            destination = partition / f"part-{digest}-{ordinal:04d}.parquet"
            try:
                return destination, destination.open("xb")
            except FileExistsError:
                ordinal += 1


def _public_columns(columns: pd.Index) -> list[str]:
    names = {str(column) for column in columns}
    ordered: list[str] = list(REQUIRED_COLUMNS)
    ordered.extend(sorted(names.difference(REQUIRED_COLUMNS, INTERNAL_COLUMNS)))
    return ordered

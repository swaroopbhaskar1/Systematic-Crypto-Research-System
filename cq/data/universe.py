"""Point-in-time universe membership with explicit terminal classifications."""

from collections.abc import Iterable
from dataclasses import InitVar, dataclass
from datetime import datetime
from typing import Literal, TypeAlias

MarketType: TypeAlias = Literal["spot", "perp"]
DelistReason: TypeAlias = Literal["delisted", "zero", "migrated", "exchange_failure"]


@dataclass(frozen=True, slots=True)
class Listing:
    symbol: str
    exchange: str
    market_type: MarketType
    listed_at: datetime
    delisted_at: datetime | None = None
    delist_reason: DelistReason | None = None
    successor: str | None = None
    zero_evidence: InitVar[str | None] = None

    def __post_init__(self, zero_evidence: str | None) -> None:
        if not self.symbol or not self.exchange:
            raise ValueError("symbol and exchange must be non-empty")
        if self.market_type not in ("spot", "perp"):
            raise ValueError(f"unsupported market type: {self.market_type}")
        if self.delist_reason not in (
            None,
            "delisted",
            "zero",
            "migrated",
            "exchange_failure",
        ):
            raise ValueError(f"unsupported delist reason: {self.delist_reason}")
        if self.listed_at.tzinfo is None:
            raise ValueError("listed_at must be timezone-aware")
        self._validate_delisting()
        if (self.delisted_at is None) != (self.delist_reason is None):
            raise ValueError("delisted_at and delist_reason must be set together")
        if self.delist_reason == "zero" and (
            zero_evidence is None or not zero_evidence.strip()
        ):
            raise ValueError("zero classification requires explicit evidence")
        if self.delist_reason != "zero" and zero_evidence is not None:
            raise ValueError("zero evidence is only valid for zero classifications")
        if self.delist_reason == "migrated" and not self.successor:
            raise ValueError("migrated listings require a successor")
        if self.delist_reason != "migrated" and self.successor is not None:
            raise ValueError("successor is only valid for migrated listings")

    def _validate_delisting(self) -> None:
        if self.delisted_at is None:
            return
        if self.delisted_at.tzinfo is None:
            raise ValueError("delisted_at must be timezone-aware")
        if self.delisted_at < self.listed_at:
            raise ValueError("delisted_at cannot precede listed_at")


class UniverseRegistry:
    """Immutable listing registry with one public point-in-time enumeration."""

    def __init__(self, listings: Iterable[Listing]) -> None:
        materialized = tuple(listings)
        identities = [
            (
                listing.symbol,
                listing.exchange,
                listing.market_type,
                listing.listed_at,
            )
            for listing in materialized
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate listing identity")
        self._listings = materialized

    def universe_at(self, ts: datetime, market_type: MarketType) -> frozenset[str]:
        """Return symbols active at ``ts`` for exactly one market type."""
        if ts.tzinfo is None:
            raise ValueError("ts must be timezone-aware")
        if market_type not in ("spot", "perp"):
            raise ValueError(f"unsupported market type: {market_type}")
        return frozenset(
            listing.symbol
            for listing in self._listings
            if listing.market_type == market_type
            and listing.listed_at <= ts
            and (listing.delisted_at is None or ts < listing.delisted_at)
        )

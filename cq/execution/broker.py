"""Broker protocol shared by paper and unimplemented live adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, TypeAlias

OrderId: TypeAlias = str


class OrderType(str, Enum):
    """Only limit orders are permitted."""

    LIMIT = "LIMIT"


@dataclass(frozen=True)
class Order:
    """A single parent or child order."""

    symbol: str
    side: str
    quantity: float
    type: OrderType | str
    limit_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _limit_type(self.type))
        if self.quantity <= 0.0 or self.limit_price <= 0.0:
            raise ValueError("quantity and limit_price must be positive")
        if self.side not in {"buy", "sell"}:
            raise ValueError("side must be buy or sell")


class Broker(Protocol):
    """Execution adapter. Strategy code must not branch on paper vs live."""

    def submit(self, order: Order) -> OrderId: ...

    def cancel(self, oid: OrderId) -> None: ...

    def positions(self) -> dict[str, float]: ...

    def open_orders(self) -> list[Order]: ...


def _limit_type(value: OrderType | str) -> OrderType:
    if value is OrderType.LIMIT or value == OrderType.LIMIT.value:
        return OrderType.LIMIT
    raise AssertionError("Order.type == LIMIT is required")

"""Limit-order slicing and a hard liquidity veto."""

from __future__ import annotations

from dataclasses import dataclass

from cq.execution.broker import Order, OrderType

LIQUIDITY_VETO = "LIQUIDITY_VETO"
DEFAULT_CHILDREN = 4
DEFAULT_WINDOW_HOURS = 4.0
VETO_PARTICIPATION = 0.01


@dataclass(frozen=True)
class ChildOrder(Order):
    """A time-sliced child limit order."""

    delay_hours: float = 0.0


def slice_order(
    order: Order,
    trailing_1h_quote_volume: float,
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    n_children: int = DEFAULT_CHILDREN,
) -> list[ChildOrder]:
    """Split a parent limit into 3-4 children, or skip as a liquidity veto."""
    if order.type is not OrderType.LIMIT:
        raise AssertionError("Order.type == LIMIT is required")
    if trailing_1h_quote_volume <= 0.0:
        raise ValueError("trailing_1h_quote_volume must be positive")
    if n_children not in {3, 4}:
        raise ValueError("n_children must be 3 or 4")
    notional = order.quantity * order.limit_price
    if notional > VETO_PARTICIPATION * trailing_1h_quote_volume:
        return []
    widths = _child_quantities(order.quantity, n_children)
    step = window_hours / (n_children - 1)
    return [
        ChildOrder(
            symbol=order.symbol,
            side=order.side,
            quantity=quantity,
            type=OrderType.LIMIT,
            limit_price=order.limit_price,
            delay_hours=index * step,
        )
        for index, quantity in enumerate(widths)
    ]


def _child_quantities(total: float, count: int) -> tuple[float, ...]:
    base = total / count
    values = [base] * count
    values[-1] = total - base * (count - 1)
    return tuple(values)

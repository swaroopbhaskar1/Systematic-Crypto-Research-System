"""Limit-only slicing with a liquidity veto."""

import pytest

from cq.execution.broker import Order, OrderType
from cq.execution.slicing import LIQUIDITY_VETO, slice_order


def test_slice_order_rejects_non_limit_orders() -> None:
    with pytest.raises(AssertionError):
        slice_order(
            Order(
                symbol="BTCUSDT",
                side="buy",
                quantity=1.0,
                type="MARKET",
                limit_price=1.0,
            ),
            trailing_1h_quote_volume=1_000_000.0,
        )


def test_default_slice_creates_three_to_four_child_limits() -> None:
    order = Order(
        symbol="BTCUSDT",
        side="sell",
        quantity=8.0,
        type=OrderType.LIMIT,
        limit_price=50.0,
    )
    children = slice_order(order, trailing_1h_quote_volume=1_000_000.0)
    assert 3 <= len(children) <= 4
    assert all(child.type is OrderType.LIMIT for child in children)
    assert sum(child.quantity for child in children) == pytest.approx(8.0)
    delays = [child.delay_hours for child in children]
    assert delays == sorted(delays)
    assert delays[-1] <= 4.0


def test_liquidity_veto_skips_oversized_orders() -> None:
    order = Order(
        symbol="TINYUSDT",
        side="buy",
        quantity=100.0,
        type=OrderType.LIMIT,
        limit_price=10.0,
    )
    result = slice_order(order, trailing_1h_quote_volume=1_000.0)
    assert result == []
    assert LIQUIDITY_VETO == "LIQUIDITY_VETO"

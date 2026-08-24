"""Broker protocol and paper-vs-live isolation."""

import pytest

from cq.execution.broker import Order, OrderType
from cq.execution.live import LiveBroker
from cq.execution.paper import PaperBroker


def test_orders_must_be_limit_type() -> None:
    with pytest.raises(AssertionError):
        Order(
            symbol="BTCUSDT",
            side="buy",
            quantity=1.0,
            type="MARKET",
            limit_price=100.0,
        )
    order = Order(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        type=OrderType.LIMIT,
        limit_price=100.0,
    )
    assert order.type == OrderType.LIMIT


def test_live_broker_is_unimplemented() -> None:
    broker = LiveBroker()
    with pytest.raises(NotImplementedError):
        broker.submit(
            Order(
                symbol="BTCUSDT",
                side="buy",
                quantity=1.0,
                type=OrderType.LIMIT,
                limit_price=100.0,
            )
        )


def test_paper_and_live_share_the_broker_protocol() -> None:
    assert callable(PaperBroker.submit)
    assert callable(LiveBroker.submit)
    assert callable(PaperBroker.positions)
    assert callable(LiveBroker.positions)

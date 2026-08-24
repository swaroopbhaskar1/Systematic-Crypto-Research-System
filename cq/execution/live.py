"""Live execution remains an explicit stub."""

from typing import NoReturn

from cq.execution.broker import Order, OrderId


def execute_live(*_args: object, **_kwargs: object) -> NoReturn:
    """Reject every attempt to perform live execution."""
    raise NotImplementedError("live execution is intentionally not implemented")


class LiveBroker:
    """Live venue adapter. Intentionally unimplemented."""

    def submit(self, order: Order) -> OrderId:
        raise NotImplementedError("live execution is intentionally not implemented")

    def cancel(self, oid: OrderId) -> None:
        raise NotImplementedError("live execution is intentionally not implemented")

    def positions(self) -> dict[str, float]:
        raise NotImplementedError("live execution is intentionally not implemented")

    def open_orders(self) -> list[Order]:
        raise NotImplementedError("live execution is intentionally not implemented")

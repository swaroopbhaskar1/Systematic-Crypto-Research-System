"""Paper broker: live ticks, simulated fills, ledger-first recovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from cq.execution.broker import Order, OrderId, OrderType

_HALF_SPREAD = 20_000.0


@dataclass(frozen=True)
class Tick:
    """Top-of-book snapshot used for paper fills."""

    symbol: str
    bid: float
    ask: float
    quote_volume: float


@dataclass
class _WorkingOrder:
    identity: OrderId
    order: Order
    remaining: float
    seen_decision_tick: bool


class PaperBroker:
    """Simulate fills on the next tick and persist every event."""

    def __init__(
        self,
        *,
        ledger_path: Path,
        spread_bps: float,
        max_participation: float,
    ) -> None:
        if spread_bps < 0.0 or max_participation <= 0.0:
            raise ValueError("spread and participation must be valid")
        self._ledger_path = ledger_path
        self._spread_bps = spread_bps
        self._max_participation = max_participation
        self._positions: dict[str, float] = {}
        self._orders: dict[OrderId, _WorkingOrder] = {}
        self._last_tick: dict[str, Tick] = {}
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_ledger(
        cls,
        *,
        ledger_path: Path,
        spread_bps: float,
        max_participation: float,
    ) -> "PaperBroker":
        broker = cls(
            ledger_path=ledger_path,
            spread_bps=spread_bps,
            max_participation=max_participation,
        )
        broker._replay()
        return broker

    def submit(self, order: Order) -> OrderId:
        if order.type is not OrderType.LIMIT:
            raise AssertionError("Order.type == LIMIT is required")
        identity = uuid4().hex
        self._orders[identity] = _WorkingOrder(
            identity=identity,
            order=order,
            remaining=order.quantity,
            seen_decision_tick=False,
        )
        self._append(
            {
                "event": "intent",
                "order_id": identity,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "book": _tick_payload(self._last_tick.get(order.symbol)),
            }
        )
        return identity

    def cancel(self, oid: OrderId) -> None:
        self._orders.pop(oid, None)

    def positions(self) -> dict[str, float]:
        return dict(self._positions)

    def open_orders(self) -> list[Order]:
        return [item.order for item in self._orders.values()]

    def on_tick(self, tick: Tick) -> None:
        self._validate_tick(tick)
        previous = self._last_tick.get(tick.symbol)
        self._last_tick[tick.symbol] = tick
        self._append({"event": "book", "book": _tick_payload(tick)})
        if previous is None:
            self._mark_decision_ticks(tick.symbol)
            return
        self._fill_symbol(tick)

    def run_ticks(self, ticks: list[Tick]) -> None:
        for tick in ticks:
            self.on_tick(tick)

    def _mark_decision_ticks(self, symbol: str) -> None:
        for item in self._orders.values():
            if item.order.symbol == symbol:
                item.seen_decision_tick = True

    def _fill_symbol(self, tick: Tick) -> None:
        for identity, item in list(self._orders.items()):
            if item.order.symbol != tick.symbol:
                continue
            if not item.seen_decision_tick:
                item.seen_decision_tick = True
                continue
            self._execute(identity, item, tick)

    def _execute(self, identity: OrderId, item: _WorkingOrder, tick: Tick) -> None:
        price = _worse_price(item.order.side, tick, self._spread_bps)
        cap_notional = self._max_participation * tick.quote_volume
        remaining_notional = item.remaining * price
        executed_notional = min(remaining_notional, cap_notional)
        quantity = executed_notional / price
        if quantity <= 0.0:
            return
        signed = quantity if item.order.side == "buy" else -quantity
        self._positions[item.order.symbol] = (
            self._positions.get(item.order.symbol, 0.0) + signed
        )
        item.remaining -= quantity
        if item.remaining <= 1e-12:
            del self._orders[identity]
        self._append(
            {
                "event": "fill",
                "order_id": identity,
                "symbol": item.order.symbol,
                "side": item.order.side,
                "quantity": quantity,
                "price": price,
                "capped": executed_notional < remaining_notional,
                "modelled_price": item.order.limit_price,
                "slippage": price - item.order.limit_price
                if item.order.side == "buy"
                else item.order.limit_price - price,
            }
        )

    def _replay(self) -> None:
        if not self._ledger_path.exists():
            return
        for raw in self._ledger_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(raw)
            if record.get("event") != "fill":
                continue
            symbol = str(record["symbol"])
            quantity = float(record["quantity"])
            signed = quantity if record["side"] == "buy" else -quantity
            self._positions[symbol] = self._positions.get(symbol, 0.0) + signed

    def _append(self, record: dict[str, object]) -> None:
        with self._ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _validate_tick(self, tick: Tick) -> None:
        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.quote_volume < 0.0:
            raise ValueError("tick prices must be positive")
        if tick.ask < tick.bid:
            raise ValueError("tick ask cannot be below bid")


def _worse_price(side: str, tick: Tick, spread_bps: float) -> float:
    raw = tick.ask if side == "buy" else tick.bid
    half = raw * spread_bps / _HALF_SPREAD
    return raw + half if side == "buy" else raw - half


def _tick_payload(tick: Tick | None) -> dict[str, object] | None:
    if tick is None:
        return None
    return {
        "ask": tick.ask,
        "bid": tick.bid,
        "quote_volume": tick.quote_volume,
        "symbol": tick.symbol,
    }

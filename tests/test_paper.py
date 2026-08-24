"""Paper fills use the next tick, worse side of spread, and a durable ledger."""

import json
from pathlib import Path

import pytest

from cq.execution.broker import Order, OrderType
from cq.execution.paper import PaperBroker, Tick


def _order() -> Order:
    return Order(
        symbol="ETHUSDT",
        side="buy",
        quantity=2.0,
        type=OrderType.LIMIT,
        limit_price=101.0,
    )


def test_paper_fill_uses_next_tick_worse_side_and_volume_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    broker = PaperBroker(
        ledger_path=ledger,
        spread_bps=20.0,
        max_participation=0.02,
    )
    broker.submit(_order())
    broker.on_tick(Tick(symbol="ETHUSDT", bid=99.0, ask=101.0, quote_volume=10.0))
    # Decision tick is observed; fill waits for the next tick.
    assert broker.positions() == {}
    broker.on_tick(Tick(symbol="ETHUSDT", bid=100.0, ask=102.0, quote_volume=10.0))
    positions = broker.positions()
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    fill = next(record for record in records if record["event"] == "fill")
    assert fill["capped"] is True
    assert fill["price"] >= 102.0
    assert positions["ETHUSDT"] == pytest.approx(0.2 / fill["price"])


def test_paper_restart_reconciles_from_ledger_not_memory(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first = PaperBroker(ledger_path=ledger, spread_bps=0.0, max_participation=1.0)
    first.submit(_order())
    first.on_tick(Tick(symbol="ETHUSDT", bid=100.0, ask=100.0, quote_volume=1_000.0))
    first.on_tick(Tick(symbol="ETHUSDT", bid=100.0, ask=100.0, quote_volume=1_000.0))
    assert first.positions()["ETHUSDT"] == 2.0

    restarted = PaperBroker.from_ledger(
        ledger_path=ledger,
        spread_bps=0.0,
        max_participation=1.0,
    )
    assert restarted.positions() == first.positions()
    assert restarted.open_orders() == []


def test_simulated_unattended_day_writes_a_complete_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    broker = PaperBroker(ledger_path=ledger, spread_bps=10.0, max_participation=0.05)
    broker.submit(_order())
    ticks = [
        Tick(symbol="ETHUSDT", bid=100.0 + hour / 100.0, ask=100.2 + hour / 100.0, quote_volume=1_000.0)
        for hour in range(24)
    ]
    broker.run_ticks(ticks)
    assert ledger.exists()
    events = [json.loads(line)["event"] for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert "intent" in events
    assert "fill" in events
    assert broker.positions()["ETHUSDT"] > 0.0

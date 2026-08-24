"""Generator emits schema-valid JSON and never silently repairs invalid rows."""

import json
from pathlib import Path

import pytest

from cq.ai.generate import generate_hypotheses
from cq.research.log import content_hash
from cq.research.schema import Hypothesis


VALID_PAYLOAD = {
    "id": "funding_extreme_001",
    "mechanism_class": "carry",
    "forced_participant": "perpetual longs paying funding at the 95th percentile",
    "why_forced": (
        "Holders of leveraged perpetual positions must pay funding every 8h "
        "regardless of price view; exiting requires crossing the spread."
    ),
    "universe_filter": "top 100 by 30d median quote volume, excluding stablecoins",
    "entry_rule": "funding_8h > roll_pct(funding_8h, 270, 0.95)",
    "exit_rule": "funding_8h < roll_pct(funding_8h, 270, 0.50)",
    "direction": "market_neutral",
    "holding_bars": 9,
    "testable_prediction": "mean forward 3d return of the paid side exceeds costs",
    "expected_capacity_usd": 50_000.0,
    "competition_risk": "medium",
    "data_required": ["funding_8h"],
}


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, payloads: list[object]) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payloads.pop(0))


def _completion(text: str) -> dict[str, object]:
    return {"content": [{"type": "text", "text": text}]}


def test_generator_keeps_valid_json_and_drops_invalid_without_repair(
    tmp_path: Path,
) -> None:
    invalid = dict(VALID_PAYLOAD)
    invalid["id"] = "bad_lookahead"
    invalid["entry_rule"] = "lead(close, 1) > close"
    client = FakeClient(
        [
            _completion(json.dumps([VALID_PAYLOAD, invalid])),
        ]
    )
    dropped = tmp_path / "dropped.jsonl"
    hypotheses = generate_hypotheses(
        client=client,
        calls=1,
        batch_size=2,
        dropped_log=dropped,
        api_key="test-key",
    )
    assert [item.id for item in hypotheses] == ["funding_extreme_001"]
    assert "lead(close" in dropped.read_text(encoding="utf-8")
    assert client.calls[0]["json"]["temperature"] == 1.0


def test_generator_runs_independent_calls_and_deduplicates_by_content_hash() -> None:
    clone = dict(VALID_PAYLOAD)
    clone["id"] = "funding_extreme_002"
    client = FakeClient(
        [
            _completion(json.dumps([VALID_PAYLOAD])),
            _completion(json.dumps([clone])),
        ]
    )
    hypotheses = generate_hypotheses(
        client=client,
        calls=2,
        batch_size=1,
        api_key="test-key",
    )
    assert len(hypotheses) == 1
    assert content_hash(hypotheses[0]) == content_hash(
        Hypothesis.model_validate(VALID_PAYLOAD)
    )
    assert len(client.calls) == 2

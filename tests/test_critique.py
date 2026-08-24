"""Critic returns binary verdicts with closed kill reasons and no scores."""

import json

import pytest
from pydantic import ValidationError

from cq.ai.critique import Verdict, critique_hypothesis
from cq.research.schema import Hypothesis


HYPOTHESIS = Hypothesis.model_validate(
    {
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
)


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object]) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payload)


def test_verdict_schema_rejects_numeric_scores() -> None:
    with pytest.raises(ValidationError):
        Verdict.model_validate(
            {
                "id": "funding_extreme_001",
                "verdict": "PASS",
                "kill_reason": None,
                "flags": [],
                "score": 8,
            }
        )


def test_critic_accepts_only_enumerated_kill_reasons() -> None:
    with pytest.raises(ValidationError):
        Verdict.model_validate(
            {
                "id": "x",
                "verdict": "KILL",
                "kill_reason": "sounds_unconvincing",
                "flags": [],
            }
        )


def test_critic_returns_binary_verdict_from_fresh_payload() -> None:
    client = FakeClient(
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "id": "funding_extreme_001",
                            "verdict": "PASS",
                            "kill_reason": None,
                            "flags": ["capacity_uncertain"],
                        }
                    ),
                }
            ]
        }
    )
    verdict = critique_hypothesis(HYPOTHESIS, client=client, api_key="test-key")
    assert verdict.verdict == "PASS"
    assert verdict.kill_reason is None
    assert verdict.flags == ["capacity_uncertain"]
    user_text = client.calls[0]["json"]["messages"][0]["content"]
    assert "because" not in user_text.lower() or "generator" not in user_text.lower()
    assert HYPOTHESIS.entry_rule in user_text
    assert "score" not in Verdict.model_fields

"""Hypothesis schema must reject unparseable or incomplete claims."""

import pytest
from pydantic import ValidationError

from cq.research.schema import Hypothesis


VALID = {
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


def test_valid_hypothesis_parses_and_is_immutable() -> None:
    hypothesis = Hypothesis.model_validate(VALID)
    assert hypothesis.direction == "market_neutral"
    with pytest.raises(ValidationError):
        hypothesis.id = "mutated"


def test_unparseable_rules_are_rejected_at_construction() -> None:
    payload = dict(VALID)
    payload["entry_rule"] = "lead(close, 1) > close"
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(payload)


def test_short_forced_fields_are_rejected() -> None:
    payload = dict(VALID)
    payload["forced_participant"] = "shorts"
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(payload)
    payload = dict(VALID)
    payload["why_forced"] = "because"
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(payload)


def test_data_required_must_match_rule_columns() -> None:
    payload = dict(VALID)
    payload["data_required"] = ["close"]
    with pytest.raises(ValidationError):
        Hypothesis.model_validate(payload)


def test_first_hypothesis_file_is_schema_valid() -> None:
    from pathlib import Path

    from cq.grammar.compile import compile_signal

    path = Path("research/hypotheses/funding_extreme_001.json")
    hypothesis = Hypothesis.model_validate_json(path.read_text(encoding="utf-8"))
    assert hypothesis.id == "funding_extreme_001"
    signal = compile_signal(hypothesis)
    assert callable(signal.compute)

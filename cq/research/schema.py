"""Pydantic hypothesis contract. Unparseable rules never reach the log."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cq.grammar.ast import GrammarError, ValueKind
from cq.grammar.parser import ALLOWED_COLUMNS, parse_expression

MechanismClass = Literal["carry", "mean_reversion", "momentum", "event"]
Direction = Literal["long_only", "market_neutral", "short_only"]
CompetitionRisk = Literal["low", "medium", "high"]


class Hypothesis(BaseModel):
    """A falsifiable, grammar-parseable research claim."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_min_length=1)

    id: str
    mechanism_class: MechanismClass
    forced_participant: str = Field(min_length=20)
    why_forced: str = Field(min_length=40)
    universe_filter: str
    entry_rule: str
    exit_rule: str
    direction: Direction
    holding_bars: int = Field(gt=0)
    testable_prediction: str
    expected_capacity_usd: float = Field(gt=0.0)
    competition_risk: CompetitionRisk
    data_required: list[str]

    @field_validator("data_required")
    @classmethod
    def _unique_known_columns(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("data_required columns must be unique")
        unknown = [column for column in value if column not in ALLOWED_COLUMNS]
        if unknown:
            raise ValueError(f"unknown data_required columns: {unknown}")
        return value

    @model_validator(mode="after")
    def _rules_must_parse_and_match_columns(self) -> "Hypothesis":
        required = _boolean_rule_columns(self.entry_rule) | _boolean_rule_columns(
            self.exit_rule
        )
        if set(self.data_required) != required:
            raise ValueError("data_required must exactly match rule columns")
        return self


def _boolean_rule_columns(rule: str) -> set[str]:
    try:
        parsed = parse_expression(rule)
    except GrammarError as error:
        raise ValueError(str(error)) from error
    if parsed.kind is not ValueKind.BOOLEAN:
        raise ValueError("entry and exit rules must be boolean expressions")
    return set(parsed.required_columns)

"""Compile hypothesis-shaped objects into deterministic target-weight signals."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, assert_never, runtime_checkable

import pandas as pd

from cq.data.panel import Panel
from cq.grammar.ast import Expression, GrammarError, ValueKind
from cq.grammar.ops import evaluate
from cq.grammar.parser import ALLOWED_COLUMNS, parse_expression


class _Direction(Enum):
    LONG_ONLY = "long_only"
    MARKET_NEUTRAL = "market_neutral"
    SHORT_ONLY = "short_only"


@runtime_checkable
class _Hypothesis(Protocol):
    entry_rule: str
    exit_rule: str
    data_required: tuple[str, ...]
    mode: str


@dataclass(frozen=True, slots=True)
class CompiledSignal:
    """An immutable signal compiled from validated grammar expressions."""

    _entry: Expression
    _exit: Expression
    _required_columns: frozenset[str]
    _direction: _Direction

    def compute(self, panel: Panel) -> pd.DataFrame:
        """Emit unshifted same-row target weights aligned to the panel."""

        _validate_panel_fields(panel, self._required_columns)
        universe = panel.universe_mask()
        entry = evaluate(self._entry, panel).fillna(False).astype(bool)
        exit_ = evaluate(self._exit, panel).fillna(False).astype(bool)
        eligible = universe & ~exit_
        selected = eligible & entry
        return _target_weights(selected, eligible, universe, self._direction)


def _validate_panel_fields(panel: Panel, required: frozenset[str]) -> None:
    for column in required:
        try:
            panel.field(column)
        except KeyError as error:
            raise GrammarError(f"panel is missing required field {column!r}") from error


def _target_weights(
    selected: pd.DataFrame,
    eligible: pd.DataFrame,
    universe: pd.DataFrame,
    direction: _Direction,
) -> pd.DataFrame:
    match direction:
        case _Direction.LONG_ONLY:
            weights = _normalize(selected, 1.0)
        case _Direction.SHORT_ONLY:
            weights = _normalize(selected, -1.0)
        case _Direction.MARKET_NEUTRAL:
            weights = _neutral_weights(selected, eligible)
        case _ as unreachable:
            assert_never(unreachable)
    return weights.where(universe)


def _normalize(selected: pd.DataFrame, total: float) -> pd.DataFrame:
    counts = selected.sum(axis=1)
    weights = selected.astype(float).div(counts.replace(0, pd.NA), axis=0)
    return weights.mul(total).fillna(0.0)


def _neutral_weights(
    selected: pd.DataFrame,
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    unselected = eligible & ~selected
    long_counts = selected.sum(axis=1)
    short_counts = unselected.sum(axis=1)
    both_sides = (long_counts > 0) & (short_counts > 0)
    longs = _normalize(selected, 0.5).where(both_sides, 0.0, axis=0)
    shorts = _normalize(unselected, -0.5).where(both_sides, 0.0, axis=0)
    return longs + shorts


def _validated_hypothesis(
    hypothesis: object,
) -> tuple[str, str, tuple[str, ...], _Direction]:
    if not isinstance(hypothesis, _Hypothesis):
        raise GrammarError("hypothesis is missing required grammar fields")
    if not isinstance(hypothesis.entry_rule, str):
        raise GrammarError("entry_rule must be a string")
    if not isinstance(hypothesis.exit_rule, str):
        raise GrammarError("exit_rule must be a string")
    data_required = _validated_data_required(hypothesis.data_required)
    try:
        direction = _Direction(hypothesis.mode)
    except (TypeError, ValueError) as error:
        raise GrammarError("mode is not a supported direction") from error
    return hypothesis.entry_rule, hypothesis.exit_rule, data_required, direction


def _validated_data_required(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(column, str) for column in value
    ):
        raise GrammarError("data_required must be a tuple of column names")
    columns = tuple(value)
    if len(columns) != len(set(columns)):
        raise GrammarError("data_required columns must be unique")
    if not set(columns).issubset(ALLOWED_COLUMNS):
        raise GrammarError("data_required contains an unknown column")
    return columns


def compile_signal(hypothesis: object) -> CompiledSignal:
    """Compile the minimal hypothesis protocol without importing a schema."""

    entry_rule, exit_rule, data_required, direction = _validated_hypothesis(
        hypothesis
    )
    entry = parse_expression(entry_rule)
    exit_ = parse_expression(exit_rule)
    if entry.kind is not ValueKind.BOOLEAN or exit_.kind is not ValueKind.BOOLEAN:
        raise GrammarError("entry and exit rules must be boolean expressions")
    required = entry.required_columns | exit_.required_columns
    if frozenset(data_required) != required:
        raise GrammarError("data_required must exactly match rule columns")
    return CompiledSignal(entry, exit_, required, direction)

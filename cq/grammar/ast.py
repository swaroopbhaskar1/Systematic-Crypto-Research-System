"""Typed, immutable nodes for the restricted research expression grammar."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class GrammarError(ValueError):
    """Raised when source or hypothesis data violates the research grammar."""


class ValueKind(Enum):
    """Static expression value categories."""

    NUMBER = "number"
    VECTOR = "vector"
    BOOLEAN = "boolean"


class UnaryOperator(Enum):
    """Whitelisted unary operators."""

    POSITIVE = "+"
    NEGATIVE = "-"
    NOT = "not"


class BinaryOperator(Enum):
    """Whitelisted binary operators."""

    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    GREATER = ">"
    GREATER_EQUAL = ">="
    LESS = "<"
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    AND = "and"
    OR = "or"


class Function(Enum):
    """Whitelisted point-in-time-safe functions."""

    LAG = "lag"
    ROLL_MEAN = "roll_mean"
    ROLL_STD = "roll_std"
    ROLL_Z = "roll_z"
    ROLL_PCT = "roll_pct"
    ROLL_MIN = "roll_min"
    ROLL_MAX = "roll_max"
    PCT_CHANGE = "pct_change"
    XS_RANK = "xs_rank"
    XS_Z = "xs_z"
    ABS = "abs"
    SIGN = "sign"
    LOG = "log"
    CLIP = "clip"


class Expression(ABC):
    """Base type for every valid grammar expression."""

    __slots__ = ()

    @property
    @abstractmethod
    def kind(self) -> ValueKind:
        """Return the statically validated result kind."""

    @property
    @abstractmethod
    def required_columns(self) -> frozenset[str]:
        """Return all market-data columns referenced by this subtree."""


@dataclass(frozen=True, slots=True)
class Number(Expression):
    """A finite numeric literal."""

    value: float
    is_integer_literal: bool

    @property
    def kind(self) -> ValueKind:
        return ValueKind.NUMBER

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True, slots=True)
class Column(Expression):
    """A whitelisted market-data column."""

    name: str

    @property
    def kind(self) -> ValueKind:
        return ValueKind.VECTOR

    @property
    def required_columns(self) -> frozenset[str]:
        return frozenset((self.name,))


@dataclass(frozen=True, slots=True)
class Unary(Expression):
    """A validated unary operation."""

    operator: UnaryOperator
    operand: Expression
    result_kind: ValueKind

    @property
    def kind(self) -> ValueKind:
        return self.result_kind

    @property
    def required_columns(self) -> frozenset[str]:
        return self.operand.required_columns


@dataclass(frozen=True, slots=True)
class Binary(Expression):
    """A validated binary operation."""

    operator: BinaryOperator
    left: Expression
    right: Expression
    result_kind: ValueKind

    @property
    def kind(self) -> ValueKind:
        return self.result_kind

    @property
    def required_columns(self) -> frozenset[str]:
        return self.left.required_columns | self.right.required_columns


@dataclass(frozen=True, slots=True)
class Call(Expression):
    """A validated call to one whitelisted function."""

    function: Function
    arguments: tuple[Expression, ...]

    @property
    def kind(self) -> ValueKind:
        return ValueKind.VECTOR

    @property
    def required_columns(self) -> frozenset[str]:
        required: frozenset[str] = frozenset()
        for argument in self.arguments:
            required |= argument.required_columns
        return required

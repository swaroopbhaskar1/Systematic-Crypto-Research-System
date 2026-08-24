"""Tokenizer and recursive-descent parser for the restricted grammar."""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from cq.grammar.ast import (
    Binary,
    BinaryOperator,
    Call,
    Column,
    Expression,
    Function,
    GrammarError,
    Number,
    Unary,
    UnaryOperator,
    ValueKind,
)

ALLOWED_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "funding_8h",
        "open_interest",
    }
)
_NUMBER_PATTERN = re.compile(
    r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TWO_CHARACTER_OPERATORS = frozenset({">=", "<=", "==", "!="})
_SINGLE_CHARACTER_TOKENS = frozenset("+-*/><(),=")


class _TokenKind(Enum):
    NUMBER = "number"
    IDENTIFIER = "identifier"
    SYMBOL = "symbol"
    END = "end"


@dataclass(frozen=True, slots=True)
class _Token:
    kind: _TokenKind
    text: str
    position: int


def _tokenize(source: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    position = 0
    while position < len(source):
        character = source[position]
        if character.isspace():
            position += 1
            continue
        token = _token_at(source, position)
        tokens.append(token)
        position += len(token.text)
    tokens.append(_Token(_TokenKind.END, "", len(source)))
    return tuple(tokens)


def _token_at(source: str, position: int) -> _Token:
    number = _NUMBER_PATTERN.match(source, position)
    if number is not None:
        return _Token(_TokenKind.NUMBER, number.group(), position)
    identifier = _IDENTIFIER_PATTERN.match(source, position)
    if identifier is not None:
        return _Token(_TokenKind.IDENTIFIER, identifier.group(), position)
    pair = source[position : position + 2]
    if pair in _TWO_CHARACTER_OPERATORS:
        return _Token(_TokenKind.SYMBOL, pair, position)
    character = source[position]
    if character in _SINGLE_CHARACTER_TOKENS:
        return _Token(_TokenKind.SYMBOL, character, position)
    raise GrammarError(f"unsupported token at position {position}")


class _Parser:
    def __init__(self, tokens: tuple[_Token, ...]) -> None:
        self._tokens = tokens
        self._position = 0

    @property
    def _current(self) -> _Token:
        return self._tokens[self._position]

    def parse(self) -> Expression:
        if self._current.kind is _TokenKind.END:
            raise GrammarError("expression cannot be empty")
        expression = self._parse_or()
        if self._current.text:
            raise GrammarError(f"unexpected token {self._current.text!r}")
        if expression.kind is ValueKind.NUMBER:
            raise GrammarError("expression must reference market data")
        return expression

    def _accept(self, text: str) -> bool:
        if self._current.text != text:
            return False
        self._position += 1
        return True

    def _expect(self, text: str) -> None:
        if not self._accept(text):
            raise GrammarError(f"expected {text!r} at position {self._current.position}")

    def _parse_or(self) -> Expression:
        left = self._parse_and()
        while self._accept("or"):
            right = self._parse_and()
            left = _boolean_binary(BinaryOperator.OR, left, right)
        return left

    def _parse_and(self) -> Expression:
        left = self._parse_not()
        while self._accept("and"):
            right = self._parse_not()
            left = _boolean_binary(BinaryOperator.AND, left, right)
        return left

    def _parse_not(self) -> Expression:
        if not self._accept("not"):
            return self._parse_comparison()
        operand = self._parse_not()
        if operand.kind is not ValueKind.BOOLEAN:
            raise GrammarError("'not' requires a boolean expression")
        return Unary(UnaryOperator.NOT, operand, ValueKind.BOOLEAN)

    def _parse_comparison(self) -> Expression:
        left = self._parse_sum()
        operator = _comparison_operator(self._current.text)
        if operator is None:
            return left
        self._position += 1
        right = self._parse_sum()
        return _comparison(operator, left, right)

    def _parse_sum(self) -> Expression:
        left = self._parse_product()
        while self._current.text in {"+", "-"}:
            text = self._current.text
            self._position += 1
            operator = (
                BinaryOperator.ADD if text == "+" else BinaryOperator.SUBTRACT
            )
            left = _numeric_binary(operator, left, self._parse_product())
        return left

    def _parse_product(self) -> Expression:
        left = self._parse_unary()
        while self._current.text in {"*", "/"}:
            text = self._current.text
            self._position += 1
            operator = (
                BinaryOperator.MULTIPLY if text == "*" else BinaryOperator.DIVIDE
            )
            left = _numeric_binary(operator, left, self._parse_unary())
        return left

    def _parse_unary(self) -> Expression:
        if self._current.text not in {"+", "-"}:
            return self._parse_primary()
        text = self._current.text
        self._position += 1
        operand = self._parse_unary()
        if operand.kind is ValueKind.BOOLEAN:
            raise GrammarError("numeric unary operators require numeric operands")
        operator = (
            UnaryOperator.POSITIVE if text == "+" else UnaryOperator.NEGATIVE
        )
        return Unary(operator, operand, operand.kind)

    def _parse_primary(self) -> Expression:
        token = self._current
        if token.kind is _TokenKind.NUMBER:
            self._position += 1
            return _number(token)
        if token.kind is _TokenKind.IDENTIFIER:
            self._position += 1
            return self._parse_identifier(token.text)
        if self._accept("("):
            expression = self._parse_or()
            self._expect(")")
            return expression
        raise GrammarError(f"expected expression at position {token.position}")

    def _parse_identifier(self, name: str) -> Expression:
        if not self._accept("("):
            if name not in ALLOWED_COLUMNS:
                raise GrammarError(f"unknown identifier {name!r}")
            return Column(name)
        try:
            function = Function(name)
        except ValueError as error:
            raise GrammarError(f"unknown function {name!r}") from error
        arguments = self._parse_arguments()
        return _validated_call(function, arguments)

    def _parse_arguments(self) -> tuple[Expression, ...]:
        if self._accept(")"):
            return ()
        arguments = [self._parse_or()]
        while self._accept(","):
            arguments.append(self._parse_or())
        self._expect(")")
        return tuple(arguments)


def _number(token: _Token) -> Number:
    value = float(token.text)
    if not math.isfinite(value):
        raise GrammarError("numeric literals must be finite")
    is_integer = "." not in token.text and "e" not in token.text.lower()
    return Number(value, is_integer)


def _comparison_operator(text: str) -> BinaryOperator | None:
    operators = {
        ">": BinaryOperator.GREATER,
        ">=": BinaryOperator.GREATER_EQUAL,
        "<": BinaryOperator.LESS,
        "<=": BinaryOperator.LESS_EQUAL,
        "==": BinaryOperator.EQUAL,
        "!=": BinaryOperator.NOT_EQUAL,
    }
    return operators.get(text)


def _numeric_binary(
    operator: BinaryOperator,
    left: Expression,
    right: Expression,
) -> Binary:
    if ValueKind.BOOLEAN in {left.kind, right.kind}:
        raise GrammarError("arithmetic operators require numeric operands")
    kind = (
        ValueKind.VECTOR
        if ValueKind.VECTOR in {left.kind, right.kind}
        else ValueKind.NUMBER
    )
    return Binary(operator, left, right, kind)


def _comparison(
    operator: BinaryOperator,
    left: Expression,
    right: Expression,
) -> Binary:
    if ValueKind.BOOLEAN in {left.kind, right.kind}:
        raise GrammarError("comparisons require numeric operands")
    if ValueKind.VECTOR not in {left.kind, right.kind}:
        raise GrammarError("comparisons must reference market data")
    return Binary(operator, left, right, ValueKind.BOOLEAN)


def _boolean_binary(
    operator: BinaryOperator,
    left: Expression,
    right: Expression,
) -> Binary:
    if left.kind is not ValueKind.BOOLEAN or right.kind is not ValueKind.BOOLEAN:
        raise GrammarError("boolean operators require boolean operands")
    return Binary(operator, left, right, ValueKind.BOOLEAN)


def _validated_call(
    function: Function,
    arguments: tuple[Expression, ...],
) -> Call:
    validators: dict[Function, Callable[[tuple[Expression, ...]], None]] = {
        Function.LAG: _validate_window_call,
        Function.ROLL_MEAN: _validate_window_call,
        Function.ROLL_STD: _validate_window_call,
        Function.ROLL_Z: _validate_window_call,
        Function.ROLL_PCT: _validate_quantile_call,
        Function.ROLL_MIN: _validate_window_call,
        Function.ROLL_MAX: _validate_window_call,
        Function.PCT_CHANGE: _validate_window_call,
        Function.XS_RANK: _validate_vector_call,
        Function.XS_Z: _validate_vector_call,
        Function.ABS: _validate_vector_call,
        Function.SIGN: _validate_vector_call,
        Function.LOG: _validate_vector_call,
        Function.CLIP: _validate_clip_call,
    }
    validators[function](arguments)
    return Call(function, arguments)


def _require_arity(arguments: tuple[Expression, ...], expected: int) -> None:
    if len(arguments) != expected:
        raise GrammarError(f"function requires exactly {expected} arguments")


def _require_vector(argument: Expression) -> None:
    if argument.kind is not ValueKind.VECTOR:
        raise GrammarError("function requires a numeric vector argument")


def _validate_vector_call(arguments: tuple[Expression, ...]) -> None:
    _require_arity(arguments, 1)
    _require_vector(arguments[0])


def _validate_window_call(arguments: tuple[Expression, ...]) -> None:
    _require_arity(arguments, 2)
    _require_vector(arguments[0])
    value, is_integer = _literal_value(arguments[1])
    if not is_integer or value < 1:
        raise GrammarError("window or period must be a positive integer literal")


def _validate_quantile_call(arguments: tuple[Expression, ...]) -> None:
    _require_arity(arguments, 3)
    _require_vector(arguments[0])
    window, is_integer = _literal_value(arguments[1])
    quantile, _ = _literal_value(arguments[2])
    if not is_integer or window < 1:
        raise GrammarError("window must be a positive integer literal")
    if not 0.0 <= quantile <= 1.0:
        raise GrammarError("rolling quantile must be in [0, 1]")


def _validate_clip_call(arguments: tuple[Expression, ...]) -> None:
    _require_arity(arguments, 3)
    _require_vector(arguments[0])
    lower, _ = _literal_value(arguments[1])
    upper, _ = _literal_value(arguments[2])
    if lower >= upper:
        raise GrammarError("clip bounds must be strictly ordered")


def _literal_value(expression: Expression) -> tuple[float, bool]:
    if isinstance(expression, Number):
        return expression.value, expression.is_integer_literal
    if isinstance(expression, Unary) and expression.operator in {
        UnaryOperator.POSITIVE,
        UnaryOperator.NEGATIVE,
    }:
        value, is_integer = _literal_value(expression.operand)
        multiplier = -1.0 if expression.operator is UnaryOperator.NEGATIVE else 1.0
        return multiplier * value, is_integer
    raise GrammarError("argument must be a numeric literal")


def parse_expression(source: object) -> Expression:
    """Parse one expression without executing Python or reading market data."""

    if not isinstance(source, str):
        raise GrammarError("expression source must be a string")
    return _Parser(_tokenize(source)).parse()

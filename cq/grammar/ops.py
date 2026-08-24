"""Deterministic evaluation of validated grammar expressions."""

from typing import TypeAlias, assert_never, cast

import numpy as np
import pandas as pd

from cq.data.panel import Panel
from cq.grammar.ast import (
    Binary,
    BinaryOperator,
    Call,
    Column,
    Expression,
    Function,
    Number,
    Unary,
    UnaryOperator,
)

_Value: TypeAlias = pd.DataFrame | float


def evaluate(expression: Expression, panel: Panel) -> pd.DataFrame:
    """Evaluate an expression against a panel without mutating either input."""

    result = _evaluate(expression, panel)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("a complete expression must evaluate to a data frame")
    return result


def _evaluate(expression: Expression, panel: Panel) -> _Value:
    if isinstance(expression, Number):
        return expression.value
    if isinstance(expression, Column):
        return panel.field(expression.name)
    if isinstance(expression, Unary):
        return _evaluate_unary(expression, panel)
    if isinstance(expression, Binary):
        return _evaluate_binary(expression, panel)
    if isinstance(expression, Call):
        return _evaluate_call(expression, panel)
    raise TypeError(f"unsupported expression type: {type(expression).__name__}")


def _evaluate_unary(expression: Unary, panel: Panel) -> _Value:
    operand = _evaluate(expression.operand, panel)
    match expression.operator:
        case UnaryOperator.POSITIVE:
            return operand
        case UnaryOperator.NEGATIVE:
            return -operand
        case UnaryOperator.NOT:
            return ~cast(pd.DataFrame, operand)
        case _ as unreachable:
            assert_never(unreachable)


def _evaluate_binary(expression: Binary, panel: Panel) -> _Value:
    left = _evaluate(expression.left, panel)
    right = _evaluate(expression.right, panel)
    match expression.operator:
        case BinaryOperator.ADD:
            return left + right
        case BinaryOperator.SUBTRACT:
            return left - right
        case BinaryOperator.MULTIPLY:
            return left * right
        case BinaryOperator.DIVIDE:
            return left / right
        case BinaryOperator.GREATER:
            return _numeric_value(left) > _numeric_value(right)
        case BinaryOperator.GREATER_EQUAL:
            return _numeric_value(left) >= _numeric_value(right)
        case BinaryOperator.LESS:
            return _numeric_value(left) < _numeric_value(right)
        case BinaryOperator.LESS_EQUAL:
            return _numeric_value(left) <= _numeric_value(right)
        case BinaryOperator.EQUAL:
            return _numeric_value(left) == _numeric_value(right)
        case BinaryOperator.NOT_EQUAL:
            return _numeric_value(left) != _numeric_value(right)
        case BinaryOperator.AND:
            return cast(pd.DataFrame, left) & cast(pd.DataFrame, right)
        case BinaryOperator.OR:
            return cast(pd.DataFrame, left) | cast(pd.DataFrame, right)
        case _ as unreachable:
            assert_never(unreachable)


def _evaluate_call(expression: Call, panel: Panel) -> pd.DataFrame:
    source = cast(pd.DataFrame, _evaluate(expression.arguments[0], panel))
    match expression.function:
        case Function.LAG:
            return source.shift(_integer_argument(expression, panel, 1))
        case Function.ROLL_MEAN:
            return _rolling(source, expression, panel).mean()
        case Function.ROLL_STD:
            return _rolling(source, expression, panel).std()
        case Function.ROLL_Z:
            rolling = _rolling(source, expression, panel)
            return (source - rolling.mean()) / rolling.std()
        case Function.ROLL_PCT:
            quantile = _number_argument(expression, panel, 2)
            return _rolling(source, expression, panel).quantile(quantile)
        case Function.ROLL_MIN:
            return _rolling(source, expression, panel).min()
        case Function.ROLL_MAX:
            return _rolling(source, expression, panel).max()
        case Function.PCT_CHANGE:
            periods = _integer_argument(expression, panel, 1)
            return source.pct_change(periods=periods, fill_method=None)
        case Function.XS_RANK:
            return _xs_rank(source, panel)
        case Function.XS_Z:
            return _xs_z(source, panel)
        case Function.ABS:
            return source.abs()
        case Function.SIGN:
            return cast(pd.DataFrame, np.sign(_numeric_frame(source)))
        case Function.LOG:
            return cast(pd.DataFrame, np.log(_numeric_frame(source)))
        case Function.CLIP:
            lower = _number_argument(expression, panel, 1)
            upper = _number_argument(expression, panel, 2)
            return source.clip(lower, upper)
        case _ as unreachable:
            assert_never(unreachable)


def _rolling(
    source: pd.DataFrame,
    expression: Call,
    panel: Panel,
) -> pd.core.window.rolling.Rolling:
    window = _integer_argument(expression, panel, 1)
    return source.rolling(window=window, min_periods=window, center=False)


def _number_argument(expression: Call, panel: Panel, index: int) -> float:
    result = _evaluate(expression.arguments[index], panel)
    if isinstance(result, pd.DataFrame):
        raise TypeError("validated literal unexpectedly evaluated to a vector")
    return result


def _integer_argument(expression: Call, panel: Panel, index: int) -> int:
    return int(_number_argument(expression, panel, index))


def _numeric_value(value: _Value) -> _Value:
    return _numeric_frame(value) if isinstance(value, pd.DataFrame) else value


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float)


def _xs_rank(source: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    universe = panel.universe_mask()
    eligible = source.where(universe)
    return eligible.rank(axis=1, method="average", pct=True).where(universe)


def _xs_z(source: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    universe = panel.universe_mask()
    eligible = _numeric_frame(source).where(universe)
    mean = eligible.mean(axis=1)
    standard_deviation = eligible.std(axis=1, ddof=0)
    result = eligible.sub(mean, axis=0).div(standard_deviation, axis=0)
    return result.astype(object).where(universe)

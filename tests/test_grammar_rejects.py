"""Adversarial rejection contracts for the restricted research grammar."""

import pytest
from cq.grammar.parser import parse_expression

from cq.grammar import GrammarError


@pytest.mark.parametrize(
    "source",
    [
        "price",
        "returns",
        "adv",
        "volatility",
        "in_universe",
        "Close",
        "OPEN",
        "true",
        "null",
    ],
)
def test_unknown_identifiers_are_grammar_errors(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "mean(close, 3)",
        "rolling_mean(close, 3)",
        "rank(close)",
        "zscore(close)",
        "exp(close)",
        "sqrt(close)",
        "where(close > 0, close, 0)",
        "__import__('os')",
    ],
)
def test_unknown_functions_are_grammar_errors(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "lead(close, 1)",
        "shift(close, 1)",
        "shift(close, -1)",
        "future_close",
        "future_return(close, 1)",
        "future_value(close)",
        "close.shift(-1)",
    ],
)
def test_future_looking_constructs_are_rejected(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "lag(close, 0)",
        "lag(close, -1)",
        "lag(close, 1.0)",
        "lag(close, 1.5)",
        "lag(close, True)",
        "lag(close, '1')",
        "lag(close, volume)",
    ],
)
def test_lag_requires_a_positive_integer_literal(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "roll_mean(close, 0)",
        "roll_std(close, -2)",
        "roll_z(close, 2.5)",
        "roll_pct(close, 0, 0.5)",
        "roll_min(close, True)",
        "roll_max(close, volume)",
        "pct_change(close, 0)",
        "pct_change(close, -1)",
        "pct_change(close, 1.5)",
    ],
)
def test_windows_and_periods_must_be_positive_integer_literals(
    source: str,
) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "roll_pct(close, 3, -0.01)",
        "roll_pct(close, 3, 1.01)",
        "roll_pct(close, 3, volume)",
        "roll_pct(close, 3, '0.5')",
    ],
)
def test_rolling_quantile_must_be_a_unit_interval_literal(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "clip(close, 1, 1)",
        "clip(close, 2, 1)",
        "clip(close, volume, 1)",
        "clip(close, 0, high)",
        "clip(close, '0', 1)",
    ],
)
def test_clip_requires_ordered_numeric_literal_bounds(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "lag(close)",
        "lag(close, 1, 2)",
        "roll_mean(close)",
        "roll_std(close, 2, 3)",
        "roll_pct(close, 3)",
        "roll_pct(close, 3, 0.5, 1)",
        "pct_change(close)",
        "xs_rank(close, 1)",
        "xs_z()",
        "abs(close, open)",
        "sign()",
        "log(close, 10)",
        "clip(close, 0)",
        "clip(close, 0, 1, 2)",
    ],
)
def test_function_arity_is_validated(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "roll_mean(1, 3)",
        "xs_rank(1)",
        "xs_z(0.5)",
        "abs(True)",
        "sign('close')",
        "log(False)",
        "clip(1, 0, 2)",
        "lag(close > open, 1)",
    ],
)
def test_functions_validate_vector_and_numeric_argument_types(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "lag(close, n=1)",
        "roll_mean(close, window=3)",
        "roll_mean(close, 3, center=True)",
        "roll_std(close, 3, min_periods=1)",
        "roll_pct(close, 3, q=0.5)",
        "clip(close, lower=0, upper=1)",
    ],
)
def test_all_keyword_arguments_including_center_are_rejected(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "close.__class__",
        "close.real",
        "close[0]",
        "close['AAAUSDT']",
        "[close]",
        "{'x': close}",
        "(close, open)",
        "'close'",
        "b'close'",
        "lambda: close",
        "close if volume > 0 else open",
        "(x := close)",
        "[x for x in close]",
        "close @ open",
        "close ** 2",
        "close // 2",
        "close % 2",
        "~close",
    ],
)
def test_unsafe_or_out_of_grammar_python_syntax_is_rejected(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)


@pytest.mark.parametrize(
    "source",
    [
        "",
        " ",
        "close +",
        "(close",
        "close >",
        "and close",
        "close and",
        "roll_mean(close, 3",
        "close < open < high",
        "close === open",
        "1e309 + close",
    ],
)
def test_malformed_or_nonfinite_expressions_are_rejected(source: str) -> None:
    with pytest.raises(GrammarError):
        parse_expression(source)

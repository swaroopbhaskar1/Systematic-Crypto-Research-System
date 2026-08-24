"""Public AST and parser contracts without prescribing node internals."""

import pytest
from cq.grammar.ast import Expression
from cq.grammar.parser import parse_expression

ALLOWED_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "funding_8h",
    "open_interest",
)


@pytest.mark.parametrize("column", ALLOWED_COLUMNS)
def test_each_allowed_column_parses_to_an_expression(column: str) -> None:
    parsed = parse_expression(column)

    assert isinstance(parsed, Expression)
    assert parsed.required_columns == frozenset({column})


@pytest.mark.parametrize(
    ("source", "required"),
    [
        ("close + open", {"close", "open"}),
        ("high - low * 2", {"high", "low"}),
        ("(high - low) / (open + 1)", {"high", "low", "open"}),
        ("-funding_8h + +volume", {"funding_8h", "volume"}),
        ("close > open", {"close", "open"}),
        ("close >= open", {"close", "open"}),
        ("close < high", {"close", "high"}),
        ("close <= high", {"close", "high"}),
        ("close == open", {"close", "open"}),
        ("close != open", {"close", "open"}),
        (
            "(close > open) and (volume > roll_mean(volume, 3))",
            {"close", "open", "volume"},
        ),
        (
            "(close > open) or not (funding_8h < 0)",
            {"close", "open", "funding_8h"},
        ),
    ],
)
def test_arithmetic_comparisons_and_boolean_conditions_parse(
    source: str,
    required: set[str],
) -> None:
    parsed = parse_expression(source)

    assert isinstance(parsed, Expression)
    assert parsed.required_columns == frozenset(required)


@pytest.mark.parametrize(
    ("source", "required"),
    [
        ("lag(close, 2)", {"close"}),
        ("roll_mean(volume, 20)", {"volume"}),
        ("roll_std(quote_volume, 5)", {"quote_volume"}),
        ("roll_z(funding_8h, 8)", {"funding_8h"}),
        ("roll_pct(open_interest, 10, 0.75)", {"open_interest"}),
        ("roll_min(low, 4)", {"low"}),
        ("roll_max(high, 4)", {"high"}),
        ("pct_change(close, 1)", {"close"}),
        ("xs_rank(volume)", {"volume"}),
        ("xs_z(open_interest)", {"open_interest"}),
        ("abs(funding_8h)", {"funding_8h"}),
        ("sign(close - open)", {"close", "open"}),
        ("log(quote_volume)", {"quote_volume"}),
        ("clip(funding_8h, -0.01, 0.01)", {"funding_8h"}),
    ],
)
def test_every_allowed_function_parses(
    source: str,
    required: set[str],
) -> None:
    parsed = parse_expression(source)

    assert isinstance(parsed, Expression)
    assert parsed.required_columns == frozenset(required)


def test_nested_calls_collect_unique_columns_recursively() -> None:
    parsed = parse_expression(
        "xs_rank(roll_z(close / lag(open, 1), 5)) "
        "+ clip(funding_8h, -0.01, 0.01)"
    )

    assert parsed.required_columns == frozenset(
        {"close", "open", "funding_8h"}
    )


def test_parser_does_not_evaluate_or_read_market_data() -> None:
    parsed = parse_expression(
        "(roll_mean(close, 3) > lag(open, 1)) and (volume > 0)"
    )

    assert isinstance(parsed, Expression)
    assert parsed.required_columns == frozenset({"close", "open", "volume"})

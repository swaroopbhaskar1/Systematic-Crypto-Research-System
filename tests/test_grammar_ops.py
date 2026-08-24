"""Numerical and point-in-time contracts for grammar operations."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from fixtures.grammar import (
    DAY_MS,
    HISTORY_PERIODS,
    assert_aligned,
    make_grammar_panel,
)

from cq.data.panel import Panel
from cq.grammar.ops import evaluate
from cq.grammar.parser import parse_expression


@pytest.fixture
def panel() -> Panel:
    return make_grammar_panel()


def _evaluate(source: str, panel: Panel) -> pd.DataFrame:
    result = evaluate(parse_expression(source), panel)
    assert isinstance(result, pd.DataFrame)
    assert_aligned(result, panel)
    return result


def test_arithmetic_obeys_precedence_parentheses_and_unary_operators(
    panel: Panel,
) -> None:
    actual = _evaluate("-(high - low) / (open + 1) + +funding_8h", panel)
    expected = (
        -(panel.field("high") - panel.field("low"))
        / (panel.field("open") + 1.0)
        + panel.field("funding_8h")
    )

    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("source", "expected_factory"),
    [
        ("close > open", lambda p: p.field("close") > p.field("open")),
        ("close >= open", lambda p: p.field("close") >= p.field("open")),
        ("low < high", lambda p: p.field("low") < p.field("high")),
        ("low <= high", lambda p: p.field("low") <= p.field("high")),
        ("close == open", lambda p: p.field("close") == p.field("open")),
        ("close != open", lambda p: p.field("close") != p.field("open")),
    ],
)
def test_comparisons_are_elementwise(
    panel: Panel,
    source: str,
    expected_factory: Callable[[Panel], pd.DataFrame],
) -> None:
    actual = _evaluate(source, panel)
    expected = expected_factory(panel)
    mask = panel.universe_mask()

    pd.testing.assert_frame_equal(actual.where(mask), expected.where(mask))


def test_and_or_not_are_elementwise_boolean_operations(panel: Panel) -> None:
    close = panel.field("close")
    open_ = panel.field("open")
    funding = panel.field("funding_8h")
    mask = panel.universe_mask()

    conjunction = _evaluate(
        "(close > open) and not (funding_8h < 0)",
        panel,
    )
    disjunction = _evaluate(
        "(close < open) or (funding_8h > 0)",
        panel,
    )

    expected_conjunction = (close > open_) & ~(funding < 0)
    expected_disjunction = (close < open_) | (funding > 0)
    pd.testing.assert_frame_equal(
        conjunction.where(mask),
        expected_conjunction.where(mask),
    )
    pd.testing.assert_frame_equal(
        disjunction.where(mask),
        expected_disjunction.where(mask),
    )


def test_lag_uses_only_strictly_prior_rows(panel: Panel) -> None:
    actual = _evaluate("lag(close, 2)", panel)
    expected = panel.field("close").shift(2)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.iloc[:2].isna().all().all()


@pytest.mark.parametrize(
    ("source", "method", "method_args"),
    [
        ("roll_mean(close, 3)", "mean", ()),
        ("roll_std(close, 3)", "std", ()),
        ("roll_pct(close, 3, 0.5)", "quantile", (0.5,)),
        ("roll_min(close, 3)", "min", ()),
        ("roll_max(close, 3)", "max", ()),
    ],
)
def test_rolling_ops_are_trailing_and_require_a_complete_window(
    panel: Panel,
    source: str,
    method: str,
    method_args: tuple[float, ...],
) -> None:
    close = panel.field("close")
    rolling = close.rolling(window=3, min_periods=3, center=False)
    expected = getattr(rolling, method)(*method_args)

    actual = _evaluate(source, panel)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual["AAAUSDT"].iloc[:2].isna().all()
    assert actual["AAAUSDT"].iloc[2:].notna().all()
    assert pd.isna(actual.loc[5 * DAY_MS, "DDDUSDT"])


def test_roll_z_uses_the_same_complete_trailing_window(panel: Panel) -> None:
    close = panel.field("close")
    rolling = close.rolling(window=3, min_periods=3, center=False)
    expected = (close - rolling.mean()) / rolling.std()

    actual = _evaluate("roll_z(close, 3)", panel)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual["AAAUSDT"].iloc[:2].isna().all()


def test_pct_change_uses_the_requested_trailing_period(panel: Panel) -> None:
    actual = _evaluate("pct_change(close, 2)", panel)
    expected = panel.field("close").pct_change(periods=2, fill_method=None)

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.iloc[:2].isna().all().all()


@pytest.mark.parametrize(
    ("source", "expected_factory"),
    [
        ("abs(funding_8h)", lambda p: p.field("funding_8h").abs()),
        (
            "sign(funding_8h)",
            lambda p: np.sign(p.field("funding_8h").astype(float)),
        ),
        (
            "log(quote_volume)",
            lambda p: np.log(p.field("quote_volume").astype(float)),
        ),
        (
            "clip(funding_8h, -0.0003, 0.0003)",
            lambda p: p.field("funding_8h").clip(-0.0003, 0.0003),
        ),
    ],
)
def test_elementwise_functions_match_independent_pandas_oracles(
    panel: Panel,
    source: str,
    expected_factory: Callable[[Panel], pd.DataFrame],
) -> None:
    actual = _evaluate(source, panel)
    expected = expected_factory(panel)

    pd.testing.assert_frame_equal(actual, expected)


def test_xs_rank_uses_only_same_timestamp_eligible_symbols(panel: Panel) -> None:
    source = panel.field("close").where(panel.universe_mask())
    expected = source.rank(axis=1, method="average", pct=True).where(
        panel.universe_mask()
    )

    actual = _evaluate("xs_rank(close)", panel)

    pd.testing.assert_frame_equal(actual, expected)
    unavailable = ~panel.universe_mask()
    assert actual.where(unavailable).stack().empty


def test_xs_z_uses_only_same_timestamp_eligible_symbols(panel: Panel) -> None:
    source = panel.field("open_interest").where(panel.universe_mask())
    row_mean = source.mean(axis=1)
    row_std = source.std(axis=1, ddof=0)
    expected = source.sub(row_mean, axis=0).div(row_std, axis=0)
    expected = expected.where(panel.universe_mask())

    actual = _evaluate("xs_z(open_interest)", panel)

    pd.testing.assert_frame_equal(actual, expected)
    unavailable = ~panel.universe_mask()
    assert actual.where(unavailable).stack().empty


@pytest.mark.parametrize(
    "source",
    [
        "lag(close, 2)",
        "roll_mean(close, 3)",
        "roll_std(close, 3)",
        "roll_z(close, 3)",
        "roll_pct(close, 3, 0.75)",
        "roll_min(close, 3)",
        "roll_max(close, 3)",
        "pct_change(close, 2)",
        "xs_rank(close)",
        "xs_z(close)",
    ],
)
def test_operations_are_future_extension_invariant(source: str) -> None:
    historical = make_grammar_panel()
    extended = make_grammar_panel(
        HISTORY_PERIODS + 3,
        extreme_future=True,
    )

    expected = _evaluate(source, historical)
    actual = _evaluate(source, extended).loc[expected.index, expected.columns]

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_evaluation_does_not_mutate_the_panel(panel: Panel) -> None:
    before = panel.data

    _evaluate(
        "xs_rank(roll_z(close, 3) + abs(funding_8h))",
        panel,
    )

    pd.testing.assert_frame_equal(panel.data, before, check_exact=True)

"""Research reports are malformed without the exact test count."""

from pathlib import Path
from statistics import NormalDist

import pytest

from cq.backtest.costs import CostConfig
from cq.research.report import (
    ResearchReport,
    adjusted_sharpe_threshold,
    apply_short_term_tax,
    render_report,
)

COSTS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "costs.yaml"


def _z_one_sided(probability: float) -> float:
    return NormalDist().inv_cdf(1.0 - probability)


def test_bonferroni_threshold_matches_one_sided_normal_formula() -> None:
    naive = 1.0
    count = 247
    expected = naive * _z_one_sided(0.05 / count) / _z_one_sided(0.05)
    assert adjusted_sharpe_threshold(count) == pytest.approx(expected)
    assert 2.0 < expected < 2.6


def test_report_requires_test_count() -> None:
    with pytest.raises(TypeError):
        ResearchReport(
            naive_sharpe_threshold=1.0,
            adjusted_sharpe_threshold=2.4,
            candidates_above_adjusted=0,
            gross_sharpe=1.8,
            net_sharpe=1.1,
        )


def test_report_is_malformed_when_test_count_is_not_a_positive_int() -> None:
    with pytest.raises(ValueError):
        ResearchReport(
            test_count=0,
            naive_sharpe_threshold=1.0,
            adjusted_sharpe_threshold=1.0,
            candidates_above_adjusted=0,
            gross_sharpe=None,
            net_sharpe=None,
        )


def test_render_report_includes_count_and_gross_net_side_by_side() -> None:
    report = ResearchReport(
        test_count=247,
        naive_sharpe_threshold=1.0,
        adjusted_sharpe_threshold=adjusted_sharpe_threshold(247),
        candidates_above_adjusted=1,
        gross_sharpe=2.8,
        net_sharpe=2.5,
    )
    text = render_report(report)
    assert "Tests to date: 247" in text
    assert "Naive Sharpe threshold" in text
    assert "Adjusted threshold" in text
    assert "Candidates above adjusted threshold:" in text
    assert text.split("Candidates above adjusted threshold:")[1].strip().startswith("1")
    assert "Gross Sharpe: 2.8" in text
    assert "Net Sharpe: 2.5" in text


# --------------------------------------------------------------------------
# Reporting-only after-tax figures.
#
# Convention under test: tax is charged period-by-period on positive net
# returns only, with no loss carry-forward, carry-back, or offset.  A losing
# period passes through untaxed and generates no credit.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("net_return", "rate", "expected"),
    [
        (0.20, 0.35, 0.13),
        (1.00, 0.35, 0.65),
        (0.20, 0.00, 0.20),
        (0.0, 0.35, 0.0),
        (-0.10, 0.35, -0.10),
        (-1.00, 0.35, -1.00),
        (-0.0, 0.35, 0.0),
    ],
)
def test_short_term_tax_hits_gains_only_and_never_credits_losses(
    net_return: float, rate: float, expected: float
) -> None:
    assert apply_short_term_tax(net_return, rate) == pytest.approx(expected)


def test_losses_are_not_carried_into_a_later_periods_tax() -> None:
    """No carry: a prior loss does not shelter a later gain in this report."""
    losing = apply_short_term_tax(-0.50, 0.35)
    winning = apply_short_term_tax(0.50, 0.35)

    assert losing == pytest.approx(-0.50)
    assert winning == pytest.approx(0.325)
    assert losing + winning == pytest.approx(-0.175)


@pytest.mark.parametrize("rate", [1.0, 1.5, -0.01, float("nan"), float("inf")])
def test_apply_short_term_tax_rejects_rates_outside_zero_to_one(rate: float) -> None:
    with pytest.raises(ValueError, match="short_term_rate"):
        apply_short_term_tax(0.20, rate)


@pytest.mark.parametrize("net_return", [float("nan"), float("inf"), float("-inf")])
def test_apply_short_term_tax_rejects_nonfinite_returns(net_return: float) -> None:
    with pytest.raises(ValueError, match="net_return"):
        apply_short_term_tax(net_return, 0.35)


def test_apply_short_term_tax_rejects_bool_inputs() -> None:
    with pytest.raises(TypeError):
        apply_short_term_tax(True, 0.35)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        apply_short_term_tax(0.2, True)  # type: ignore[arg-type]


def _report_with_returns(
    *,
    gross_return: float | None = 0.40,
    net_return: float | None = 0.20,
    short_term_tax_rate: float | None = 0.35,
) -> ResearchReport:
    return ResearchReport(
        test_count=247,
        naive_sharpe_threshold=1.0,
        adjusted_sharpe_threshold=adjusted_sharpe_threshold(247),
        candidates_above_adjusted=1,
        gross_sharpe=2.8,
        net_sharpe=2.5,
        gross_return=gross_return,
        net_return=net_return,
        short_term_tax_rate=short_term_tax_rate,
    )


def test_after_tax_return_uses_the_configured_short_term_rate() -> None:
    report = _report_with_returns()

    assert report.after_tax_return == pytest.approx(0.13)


def test_after_tax_return_is_unavailable_without_a_net_return_or_rate() -> None:
    assert _report_with_returns(net_return=None).after_tax_return is None
    assert _report_with_returns(short_term_tax_rate=None).after_tax_return is None


def test_after_tax_return_equals_net_return_when_the_period_lost_money() -> None:
    report = _report_with_returns(gross_return=-0.05, net_return=-0.12)

    assert report.after_tax_return == pytest.approx(-0.12)


def test_report_rejects_tax_rates_outside_zero_to_one() -> None:
    for rate in (1.0, -0.1):
        with pytest.raises(ValueError, match="short_term_tax_rate"):
            _report_with_returns(short_term_tax_rate=rate)


def test_report_rejects_nonfinite_returns() -> None:
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="return"):
            _report_with_returns(net_return=value)
        with pytest.raises(ValueError, match="return"):
            _report_with_returns(gross_return=value)


def test_render_shows_gross_net_and_after_tax_as_three_labelled_columns() -> None:
    text = render_report(_report_with_returns())

    assert "Gross Sharpe: 2.8" in text
    assert "Net Sharpe: 2.5" in text
    assert "Gross return: 40.00%" in text
    assert "Net return: 20.00%" in text
    assert "After-tax return (reporting only): 13.00%" in text
    assert text.index("Gross return:") < text.index("Net return:")
    assert text.index("Net return:") < text.index("After-tax return")


def test_render_never_taxes_a_sharpe_ratio() -> None:
    text = render_report(_report_with_returns())

    assert "After-tax Sharpe" not in text
    assert "Net Sharpe: 2.5" in text


def test_render_marks_after_tax_unavailable_when_inputs_are_missing() -> None:
    text = render_report(
        _report_with_returns(
            gross_return=None, net_return=None, short_term_tax_rate=None
        )
    )

    assert "Gross return: n/a" in text
    assert "Net return: n/a" in text
    assert "After-tax return (reporting only): n/a" in text


def test_after_tax_figure_is_driven_by_the_repo_cost_config() -> None:
    rate = CostConfig.from_yaml(COSTS_CONFIG).short_term_rate
    report = _report_with_returns(short_term_tax_rate=rate)

    assert rate == pytest.approx(0.35)
    assert report.after_tax_return == pytest.approx(0.20 * (1.0 - 0.35))
    assert "After-tax return (reporting only): 13.00%" in render_report(report)

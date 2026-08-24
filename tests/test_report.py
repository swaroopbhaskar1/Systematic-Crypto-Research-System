"""Research reports are malformed without the exact test count."""

from statistics import NormalDist

import pytest

from cq.research.report import ResearchReport, adjusted_sharpe_threshold, render_report


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

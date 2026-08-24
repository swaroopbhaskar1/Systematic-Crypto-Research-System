"""Counting-aware research reporting. Test count is a required field."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

NAIVE_SHARPE_THRESHOLD = 1.0
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class ResearchReport:
    """A research summary that cannot omit the multiple-testing count."""

    test_count: int
    naive_sharpe_threshold: float
    adjusted_sharpe_threshold: float
    candidates_above_adjusted: int
    gross_sharpe: float | None
    net_sharpe: float | None

    def __post_init__(self) -> None:
        if isinstance(self.test_count, bool) or self.test_count < 1:
            raise ValueError("test_count must be a positive integer")
        if self.candidates_above_adjusted < 0:
            raise ValueError("candidates_above_adjusted cannot be negative")
        if self.naive_sharpe_threshold <= 0.0:
            raise ValueError("naive_sharpe_threshold must be positive")
        if self.adjusted_sharpe_threshold < self.naive_sharpe_threshold:
            raise ValueError("adjusted threshold cannot be below the naive bar")


def adjusted_sharpe_threshold(
    n_tests: int,
    naive: float = NAIVE_SHARPE_THRESHOLD,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """Bonferroni-adjust a one-test Sharpe bar for ``n_tests`` looks."""
    if isinstance(n_tests, bool) or n_tests < 1:
        raise ValueError("n_tests must be a positive integer")
    if naive <= 0.0 or not 0.0 < alpha < 1.0:
        raise ValueError("naive threshold and alpha must be positive")
    return naive * _z_one_sided(alpha / n_tests) / _z_one_sided(alpha)


def render_report(report: ResearchReport) -> str:
    """Render the counting summary with gross and net Sharpe side by side."""
    gross = _format_sharpe(report.gross_sharpe)
    net = _format_sharpe(report.net_sharpe)
    return "\n".join(
        (
            f"Tests to date: {report.test_count}",
            (
                "Naive Sharpe threshold (p<0.05, 1 test):   "
                f"{report.naive_sharpe_threshold:.1f}"
            ),
            (
                f"Adjusted threshold (Bonferroni, {report.test_count}):      "
                f"{report.adjusted_sharpe_threshold:.1f}"
            ),
            (
                "Candidates above adjusted threshold:       "
                f"{report.candidates_above_adjusted}"
            ),
            f"Gross Sharpe: {gross}",
            f"Net Sharpe: {net}",
        )
    )


def _z_one_sided(probability: float) -> float:
    return NormalDist().inv_cdf(1.0 - probability)


def _format_sharpe(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"

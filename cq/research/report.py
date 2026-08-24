"""Counting-aware research reporting. Test count is a required field."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from statistics import NormalDist

from cq.backtest.engine import BacktestResult

NAIVE_SHARPE_THRESHOLD = 1.0
DEFAULT_ALPHA = 0.05
_PAIRED_METRICS = (
    ("Total return", "return"),
    ("Sharpe (annualized)", "sharpe"),
    ("Sortino", "sortino"),
    ("Max drawdown", "max_drawdown"),
    ("Calmar", "calmar"),
)


@dataclass(frozen=True)
class ResearchReport:
    """A research summary that cannot omit the multiple-testing count.

    ``short_term_tax_rate`` is a reporting-only input, normally sourced from
    ``CostConfig.short_term_rate``.  It never participates in backtest costs.
    """

    test_count: int
    naive_sharpe_threshold: float
    adjusted_sharpe_threshold: float
    candidates_above_adjusted: int
    gross_sharpe: float | None
    net_sharpe: float | None
    gross_return: float | None = None
    net_return: float | None = None
    short_term_tax_rate: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.test_count, bool) or self.test_count < 1:
            raise ValueError("test_count must be a positive integer")
        if self.candidates_above_adjusted < 0:
            raise ValueError("candidates_above_adjusted cannot be negative")
        if self.naive_sharpe_threshold <= 0.0:
            raise ValueError("naive_sharpe_threshold must be positive")
        if self.adjusted_sharpe_threshold < self.naive_sharpe_threshold:
            raise ValueError("adjusted threshold cannot be below the naive bar")
        _validate_optional("gross_return", self.gross_return)
        _validate_optional("net_return", self.net_return)
        if self.short_term_tax_rate is not None:
            _validated_rate("short_term_tax_rate", self.short_term_tax_rate)

    @property
    def after_tax_return(self) -> float | None:
        """Reporting-only after-tax net return, or ``None`` if not computable."""
        if self.net_return is None or self.short_term_tax_rate is None:
            return None
        return apply_short_term_tax(self.net_return, self.short_term_tax_rate)


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


def apply_short_term_tax(net_return: float, short_term_rate: float) -> float:
    """Tax a period's net return at the short-term rate, gains only.

    Reporting-only.  Losses pass through untaxed and generate no credit: this
    report applies no loss carry-forward, carry-back, or cross-period offset,
    so a losing period never shelters another period's gain.
    """
    period_return = _finite("net_return", net_return)
    rate = _validated_rate("short_term_rate", short_term_rate)
    if period_return <= 0.0:
        return period_return
    return period_return * (1.0 - rate)


def render_backtest_result(
    result: BacktestResult,
    *,
    hypothesis_id: str,
) -> str:
    """Render one backtest as a CLI table with gross and net side by side.

    A metric the engine omitted is rendered as ``n/a``.  It is not rendered
    as zero, because "we did not measure this" and "this measured zero" are
    different claims and only one of them is true.
    """

    sections = (
        (f"Hypothesis: {hypothesis_id}",),
        _paired_lines(result.metrics),
        _execution_lines(result),
        _regime_lines(result.regime_metrics),
    )
    return "\n".join(line for section in sections for line in section)


def _paired_lines(metrics: Mapping[str, float | int]) -> tuple[str, ...]:
    header = f"{'Metric':<24}{'Gross':>14}{'Net':>14}"
    rows = [
        _paired_row(label, metrics.get(f"gross_{key}"), metrics.get(f"net_{key}"))
        for label, key in _PAIRED_METRICS
    ]
    return ("", header, "-" * len(header), *rows)


def _paired_row(label: str, gross: object, net: object) -> str:
    return f"{label:<24}{_format_metric(gross):>14}{_format_metric(net):>14}"


def _execution_lines(result: BacktestResult) -> tuple[str, ...]:
    metrics = result.metrics
    return (
        "",
        f"Cost drag (% of gross):   {_format_metric(metrics.get('cost_drag_percent'))}",
        f"Turnover (daily one-way): {_format_metric(metrics.get('turnover'))}",
        f"Hit rate:                 {_format_metric(metrics.get('hit_rate'))}",
        (
            "Avg holding (days):       "
            f"{_format_metric(metrics.get('avg_holding_period_days'))}"
        ),
        f"Round trips:              {metrics.get('n_round_trips', 0)}",
        f"Trades:                   {result.n_trades}",
        f"Bars capped (%):          {result.pct_bars_capped * 100.0:.2f}",
    )


def _regime_lines(
    regimes: Mapping[str, Mapping[str, float | int]],
) -> tuple[str, ...]:
    if not regimes:
        return ("", "Regime attribution: n/a (insufficient trailing history)")
    header = f"{'Regime':<14}{'Bars':>6}{'Gross ret':>14}{'Net ret':>14}{'Net Sharpe':>14}"
    rows = [
        (
            f"{label:<14}{int(bucket.get('n_bars', 0)):>6}"
            f"{_format_metric(bucket.get('gross_return')):>14}"
            f"{_format_metric(bucket.get('net_return')):>14}"
            f"{_format_metric(bucket.get('net_sharpe')):>14}"
        )
        for label, bucket in sorted(regimes.items())
    ]
    return ("", header, "-" * len(header), *rows)


def _format_metric(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("metric values must be real numbers")
    return f"{float(value):.4f}"


def render_report(report: ResearchReport) -> str:
    """Render counting, Sharpe, and gross/net/after-tax return columns."""
    return "\n".join((*_counting_lines(report), *_performance_lines(report)))


def _counting_lines(report: ResearchReport) -> tuple[str, ...]:
    return (
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
    )


def _performance_lines(report: ResearchReport) -> tuple[str, ...]:
    return (
        f"Gross Sharpe: {_format_sharpe(report.gross_sharpe)}",
        f"Net Sharpe: {_format_sharpe(report.net_sharpe)}",
        f"Gross return: {_format_return(report.gross_return)}",
        f"Net return: {_format_return(report.net_return)}",
        (
            "After-tax return (reporting only): "
            f"{_format_return(report.after_tax_return)}"
        ),
    )


def _z_one_sided(probability: float) -> float:
    return NormalDist().inv_cdf(1.0 - probability)


def _format_sharpe(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _format_return(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2%}"


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validated_rate(name: str, value: object) -> float:
    rate = _finite(name, value)
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return rate


def _validate_optional(name: str, value: float | None) -> None:
    if value is not None:
        _finite(name, value)

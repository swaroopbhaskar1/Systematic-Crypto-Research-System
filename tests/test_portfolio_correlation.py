import dataclasses

import numpy as np
import pandas as pd
import pytest

from cq.portfolio.correlation import (
    MINIMUM_REGIME_OBSERVATIONS,
    CorrelationConvergence,
    conditional_correlation,
    correlation_convergence,
    effective_breadth,
)

TOLERANCE = 1e-12


def _regime_panel(calm_rows: int, stressed_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    """Two strategies: exactly uncorrelated when calm, identical when stressed.

    ``calm_rows`` must be a multiple of four so the calm-regime correlation is
    exactly zero rather than merely small.
    """

    if calm_rows % 4 != 0:
        raise ValueError("calm_rows must be a multiple of four")
    calm_alpha = [0.01 if row % 2 == 0 else -0.01 for row in range(calm_rows)]
    calm_beta = [0.01 if row % 4 < 2 else -0.01 for row in range(calm_rows)]
    shock = [0.01 if row % 2 == 0 else -0.01 for row in range(stressed_rows)]
    index = pd.RangeIndex(calm_rows + stressed_rows)
    frame = pd.DataFrame(
        {"alpha": calm_alpha + shock, "beta": calm_beta + shock},
        index=index,
    )
    labels = pd.Series(
        ["calm"] * calm_rows + ["stressed"] * stressed_rows,
        index=index,
    )
    return frame, labels


def _correlation_frame(values: list[list[float]]) -> pd.DataFrame:
    names = [f"s{position}" for position in range(len(values))]
    return pd.DataFrame(values, index=names, columns=names)


def test_conditional_correlation_uses_only_the_requested_regime() -> None:
    returns, labels = _regime_panel(40, 40)

    calm = conditional_correlation(returns, labels, regime="calm")
    stressed = conditional_correlation(returns, labels, regime="stressed")

    assert float(calm.loc["alpha", "beta"]) == pytest.approx(0.0, abs=TOLERANCE)
    assert float(stressed.loc["alpha", "beta"]) == pytest.approx(1.0, abs=TOLERANCE)


def test_conditional_correlation_ignores_rows_outside_the_regime() -> None:
    returns, labels = _regime_panel(40, 40)
    poisoned = returns.copy()
    poisoned.loc[0:39, "beta"] = poisoned.loc[0:39, "alpha"] * -1.0

    stressed = conditional_correlation(poisoned, labels, regime="stressed")

    assert float(stressed.loc["alpha", "beta"]) == pytest.approx(1.0, abs=TOLERANCE)


def test_conditional_correlation_is_symmetric_with_a_unit_diagonal() -> None:
    returns, labels = _regime_panel(40, 40)

    matrix = conditional_correlation(returns, labels, regime="calm")

    assert list(matrix.index) == ["alpha", "beta"]
    assert float(matrix.loc["alpha", "alpha"]) == pytest.approx(1.0)
    assert float(matrix.loc["beta", "beta"]) == pytest.approx(1.0)
    assert float(matrix.loc["alpha", "beta"]) == pytest.approx(
        float(matrix.loc["beta", "alpha"])
    )


def test_conditional_correlation_accepts_exactly_the_minimum_observations() -> None:
    returns, labels = _regime_panel(40, MINIMUM_REGIME_OBSERVATIONS)

    matrix = conditional_correlation(returns, labels, regime="stressed")

    assert float(matrix.loc["alpha", "beta"]) == pytest.approx(1.0, abs=TOLERANCE)


def test_conditional_correlation_rejects_one_observation_below_the_minimum() -> None:
    returns, labels = _regime_panel(40, MINIMUM_REGIME_OBSERVATIONS - 1)

    with pytest.raises(ValueError, match="observations"):
        conditional_correlation(returns, labels, regime="stressed")


def test_conditional_correlation_rejects_an_absent_regime() -> None:
    returns, labels = _regime_panel(40, 40)

    with pytest.raises(ValueError, match="observations"):
        conditional_correlation(returns, labels, regime="euphoric")


def test_conditional_correlation_rejects_a_tiny_sample_even_when_asked_nicely() -> None:
    returns, labels = _regime_panel(40, 3)

    with pytest.raises(ValueError, match="min_observations"):
        conditional_correlation(
            returns,
            labels,
            regime="stressed",
            min_observations=1,
        )


def test_conditional_correlation_rejects_misaligned_labels() -> None:
    returns, labels = _regime_panel(40, 40)
    shifted = pd.Series(list(labels), index=pd.RangeIndex(1, len(labels) + 1))

    with pytest.raises(ValueError, match="index"):
        conditional_correlation(returns, shifted, regime="stressed")


def test_conditional_correlation_rejects_missing_returns() -> None:
    returns, labels = _regime_panel(40, 40)
    returns.iloc[45, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        conditional_correlation(returns, labels, regime="stressed")


def test_conditional_correlation_rejects_a_constant_strategy() -> None:
    returns, labels = _regime_panel(40, 40)
    returns["flat"] = 0.0

    with pytest.raises(ValueError, match="variance"):
        conditional_correlation(returns, labels, regime="stressed")


def test_conditional_correlation_rejects_a_single_strategy() -> None:
    returns, labels = _regime_panel(40, 40)

    with pytest.raises(ValueError, match="two strategies"):
        conditional_correlation(returns[["alpha"]], labels, regime="stressed")


def test_conditional_correlation_rejects_non_string_labels() -> None:
    returns, _ = _regime_panel(40, 40)
    numeric = pd.Series([0] * 40 + [1] * 40, index=returns.index)

    with pytest.raises(TypeError, match="strings"):
        conditional_correlation(returns, numeric, regime="stressed")


def test_correlation_convergence_detects_a_crash_that_erases_diversification() -> None:
    returns, labels = _regime_panel(40, 40)

    report = correlation_convergence(returns, labels, stressed_regime="stressed")

    assert report.unconditional_mean_correlation == pytest.approx(0.5, abs=TOLERANCE)
    assert report.stressed_mean_correlation == pytest.approx(1.0, abs=TOLERANCE)
    assert report.convergence == pytest.approx(0.5, abs=TOLERANCE)
    assert report.strategies == 2
    assert report.stressed_observations == 40


def test_correlation_convergence_is_positive_when_diversification_fails() -> None:
    returns, labels = _regime_panel(40, 40)

    report = correlation_convergence(returns, labels, stressed_regime="stressed")

    assert report.stressed_mean_correlation > report.unconditional_mean_correlation
    assert report.convergence > 0.0


def test_correlation_convergence_report_is_frozen() -> None:
    returns, labels = _regime_panel(40, 40)

    report = correlation_convergence(returns, labels, stressed_regime="stressed")

    assert dataclasses.is_dataclass(report)
    assert isinstance(report, CorrelationConvergence)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.convergence = 0.0  # pyright: ignore[reportAttributeAccessIssue]


def test_correlation_convergence_rejects_a_thin_stressed_sample() -> None:
    returns, labels = _regime_panel(40, MINIMUM_REGIME_OBSERVATIONS - 1)

    with pytest.raises(ValueError, match="observations"):
        correlation_convergence(returns, labels, stressed_regime="stressed")


def test_correlation_convergence_rejects_misaligned_labels() -> None:
    returns, labels = _regime_panel(40, 40)
    shifted = pd.Series(list(labels), index=pd.RangeIndex(1, len(labels) + 1))

    with pytest.raises(ValueError, match="index"):
        correlation_convergence(returns, shifted, stressed_regime="stressed")


def test_effective_breadth_equals_strategy_count_when_uncorrelated() -> None:
    identity = _correlation_frame(np.eye(4).tolist())

    assert effective_breadth(identity) == pytest.approx(4.0)


def test_effective_breadth_collapses_to_one_when_perfectly_correlated() -> None:
    ones = _correlation_frame(np.ones((4, 4)).tolist())

    assert effective_breadth(ones) == pytest.approx(1.0)


def test_effective_breadth_has_a_hand_computed_intermediate_value() -> None:
    matrix = _correlation_frame(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ]
    )

    assert effective_breadth(matrix) == pytest.approx(1.5)


def test_effective_breadth_uses_average_pairwise_correlation() -> None:
    matrix = _correlation_frame(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert effective_breadth(matrix) == pytest.approx(3.0 / (1.0 + 2.0 * 0.3))


def test_effective_breadth_of_a_single_strategy_is_one() -> None:
    matrix = _correlation_frame([[1.0]])

    assert effective_breadth(matrix) == pytest.approx(1.0)


def test_effective_breadth_rejects_a_non_square_matrix() -> None:
    matrix = pd.DataFrame([[1.0, 0.2, 0.3], [0.2, 1.0, 0.4]])

    with pytest.raises(ValueError, match="square"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_an_asymmetric_matrix() -> None:
    matrix = _correlation_frame([[1.0, 0.2], [0.4, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_a_non_unit_diagonal() -> None:
    matrix = _correlation_frame([[1.0, 0.2], [0.2, 0.9]])

    with pytest.raises(ValueError, match="diagonal"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_mismatched_labels() -> None:
    matrix = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["a", "b"], columns=["a", "c"])

    with pytest.raises(ValueError, match="labels"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_missing_correlations() -> None:
    matrix = _correlation_frame([[1.0, float("nan")], [float("nan"), 1.0]])

    with pytest.raises(ValueError, match="finite"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_out_of_range_correlations() -> None:
    matrix = _correlation_frame([[1.0, 1.4], [1.4, 1.0]])

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_a_degenerate_denominator() -> None:
    matrix = _correlation_frame([[1.0, -1.0], [-1.0, 1.0]])

    with pytest.raises(ValueError, match="undefined"):
        effective_breadth(matrix)


def test_effective_breadth_rejects_an_empty_matrix() -> None:
    matrix = pd.DataFrame(dtype=float)

    with pytest.raises(ValueError, match="empty"):
        effective_breadth(matrix)

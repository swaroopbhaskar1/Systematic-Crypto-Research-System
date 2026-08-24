import inspect
import math

import numpy as np
import pandas as pd
import pytest

from cq.portfolio.sizing import (
    constrain_weight_change,
    enforce_no_leverage,
    fractional_kelly_scale,
    trailing_realized_volatility,
    volatility_scaled_weights,
)

ANNUALIZATION = math.sqrt(365.0)


def _sample_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [0.01, -0.01, 0.02, 0.00, -0.03], "b": [0.02, 0.02, -0.01, 0.03, 0.01]},
        index=pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
    )


def test_volatility_scaled_weights_have_hand_computed_values() -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], columns=["a", "b"])

    scaled = volatility_scaled_weights(raw, volatility, target_volatility=0.4)

    assert list(scaled.iloc[0]) == pytest.approx([0.5, -1.0])


def test_volatility_scaled_weights_equalize_risk_contribution() -> None:
    raw = pd.DataFrame([[1.0, 1.0]], columns=["calm", "wild"])
    volatility = pd.DataFrame([[0.60, 2.00]], columns=["calm", "wild"])

    scaled = volatility_scaled_weights(raw, volatility, target_volatility=0.30)

    risk = [
        float(scaled.iloc[0, column]) * float(volatility.iloc[0, column])
        for column in range(2)
    ]
    assert risk[0] == pytest.approx(risk[1])
    assert float(scaled.iloc[0]["wild"]) < float(scaled.iloc[0]["calm"])


def test_volatility_scaled_weights_refuse_to_impute_missing_volatility() -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, float("nan")]], columns=["a", "b"])

    with pytest.raises(ValueError, match="trailing volatility"):
        volatility_scaled_weights(raw, volatility, target_volatility=0.4)


def test_volatility_scaled_weights_tolerate_missing_volatility_at_zero_weight() -> None:
    raw = pd.DataFrame([[1.0, 0.0]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, float("nan")]], columns=["a", "b"])

    scaled = volatility_scaled_weights(raw, volatility, target_volatility=0.4)

    assert list(scaled.iloc[0]) == pytest.approx([0.5, 0.0])


@pytest.mark.parametrize("bad_volatility", [0.0, -0.5, float("inf")])
def test_volatility_scaled_weights_reject_non_positive_volatility(
    bad_volatility: float,
) -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, bad_volatility]], columns=["a", "b"])

    with pytest.raises(ValueError, match="trailing volatility"):
        volatility_scaled_weights(raw, volatility, target_volatility=0.4)


def test_volatility_scaled_weights_reject_missing_raw_weights() -> None:
    raw = pd.DataFrame([[1.0, float("nan")]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], columns=["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        volatility_scaled_weights(raw, volatility, target_volatility=0.4)


def test_volatility_scaled_weights_reject_misaligned_frames() -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], columns=["a", "c"])

    with pytest.raises(ValueError, match="columns"):
        volatility_scaled_weights(raw, volatility, target_volatility=0.4)


def test_volatility_scaled_weights_reject_misaligned_index() -> None:
    raw = pd.DataFrame([[1.0, -0.5]], index=[0], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], index=[1], columns=["a", "b"])

    with pytest.raises(ValueError, match="index"):
        volatility_scaled_weights(raw, volatility, target_volatility=0.4)


@pytest.mark.parametrize("target", [0.0, -0.2, float("nan")])
def test_volatility_scaled_weights_reject_bad_target(target: float) -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], columns=["a", "b"])

    with pytest.raises(ValueError):
        volatility_scaled_weights(raw, volatility, target_volatility=target)


def test_volatility_scaled_weights_reject_boolean_target() -> None:
    raw = pd.DataFrame([[1.0, -0.5]], columns=["a", "b"])
    volatility = pd.DataFrame([[0.8, 0.2]], columns=["a", "b"])

    with pytest.raises(TypeError, match="real number"):
        volatility_scaled_weights(
            raw,
            volatility,
            target_volatility=True,  # pyright: ignore[reportArgumentType]
        )


def test_trailing_realized_volatility_has_a_hand_computed_value() -> None:
    returns = _sample_returns()
    window = [0.01, -0.01, 0.02]
    mean = sum(window) / 3.0
    expected = (
        math.sqrt(sum((value - mean) ** 2 for value in window) / 2.0) * ANNUALIZATION
    )

    volatility = trailing_realized_volatility(returns, window=3)

    assert float(volatility.iloc[2]["a"]) == pytest.approx(expected)


def test_trailing_realized_volatility_leaves_partial_windows_missing() -> None:
    returns = _sample_returns()

    volatility = trailing_realized_volatility(returns, window=3)

    assert bool(volatility.iloc[0].isna().all())
    assert bool(volatility.iloc[1].isna().all())
    assert not bool(volatility.iloc[2].isna().any())


def test_trailing_realized_volatility_is_unchanged_by_future_rows() -> None:
    returns = _sample_returns()
    rng = np.random.default_rng(7)
    future = pd.DataFrame(
        rng.normal(scale=0.05, size=(100, 2)),
        columns=["a", "b"],
        index=pd.date_range("2024-01-06", periods=100, freq="D", tz="UTC"),
    )
    extended = pd.concat([returns, future])

    original = trailing_realized_volatility(returns, window=3)
    lengthened = trailing_realized_volatility(extended, window=3)

    pd.testing.assert_frame_equal(lengthened.loc[original.index], original)


def test_trailing_realized_volatility_cannot_be_centered() -> None:
    parameters = inspect.signature(trailing_realized_volatility).parameters

    assert "center" not in parameters
    assert list(parameters) == ["returns", "window"]

    with pytest.raises(TypeError):
        trailing_realized_volatility(
            _sample_returns(),
            window=3,
            center=True,  # pyright: ignore[reportCallIssue]
        )


@pytest.mark.parametrize("window", [-3, 0, 1])
def test_trailing_realized_volatility_rejects_degenerate_windows(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        trailing_realized_volatility(_sample_returns(), window=window)


def test_trailing_realized_volatility_rejects_boolean_window() -> None:
    with pytest.raises(TypeError, match="integer"):
        trailing_realized_volatility(
            _sample_returns(),
            window=True,  # pyright: ignore[reportArgumentType]
        )


def test_trailing_realized_volatility_rejects_window_longer_than_history() -> None:
    with pytest.raises(ValueError, match="window"):
        trailing_realized_volatility(_sample_returns(), window=6)


def test_trailing_realized_volatility_rejects_infinite_returns() -> None:
    returns = _sample_returns()
    returns.iloc[1, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        trailing_realized_volatility(returns, window=3)


def test_trailing_realized_volatility_propagates_missing_returns() -> None:
    returns = _sample_returns()
    returns.iloc[1, 0] = float("nan")

    volatility = trailing_realized_volatility(returns, window=3)

    assert bool(volatility["a"].iloc[:4].isna().all())
    assert not bool(pd.isna(volatility["a"].iloc[4]))


def test_fractional_kelly_scale_halves_by_default() -> None:
    assert fractional_kelly_scale(0.4) == pytest.approx(0.2)


def test_fractional_kelly_scale_accepts_smaller_fractions() -> None:
    assert fractional_kelly_scale(0.4, fraction=0.25) == pytest.approx(0.1)


def test_fractional_kelly_scale_allows_the_half_kelly_boundary() -> None:
    assert fractional_kelly_scale(1.0, fraction=0.5) == pytest.approx(0.5)


@pytest.mark.parametrize("fraction", [0.5000001, 0.75, 1.0, 2.0])
def test_fractional_kelly_scale_rejects_more_than_half_kelly(fraction: float) -> None:
    with pytest.raises(ValueError, match="half-Kelly"):
        fractional_kelly_scale(0.4, fraction=fraction)


@pytest.mark.parametrize("fraction", [0.0, -0.25, float("nan")])
def test_fractional_kelly_scale_rejects_non_positive_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        fractional_kelly_scale(0.4, fraction=fraction)


def test_fractional_kelly_scale_rejects_negative_edge() -> None:
    with pytest.raises(ValueError, match="edge_scale"):
        fractional_kelly_scale(-0.4)


def test_fractional_kelly_scale_rejects_boolean_edge() -> None:
    with pytest.raises(TypeError, match="real number"):
        fractional_kelly_scale(True)  # pyright: ignore[reportArgumentType]


def test_enforce_no_leverage_scales_gross_down_to_one() -> None:
    weights = pd.DataFrame([[0.6, -0.8]], columns=["a", "b"])

    constrained = enforce_no_leverage(weights)

    assert list(constrained.iloc[0]) == pytest.approx([3.0 / 7.0, -4.0 / 7.0])
    assert float(constrained.iloc[0].abs().sum()) == pytest.approx(1.0, abs=1e-12)


def test_enforce_no_leverage_passes_through_unlevered_rows() -> None:
    weights = pd.DataFrame([[0.3, -0.2], [0.5, -0.5]], columns=["a", "b"])

    constrained = enforce_no_leverage(weights)

    pd.testing.assert_frame_equal(constrained, weights)


def test_enforce_no_leverage_scales_only_the_offending_row() -> None:
    weights = pd.DataFrame([[0.3, -0.2], [1.0, -1.0]], columns=["a", "b"])

    constrained = enforce_no_leverage(weights)

    assert list(constrained.iloc[0]) == pytest.approx([0.3, -0.2])
    assert list(constrained.iloc[1]) == pytest.approx([0.5, -0.5])


def test_enforce_no_leverage_preserves_relative_sizes_and_signs() -> None:
    weights = pd.DataFrame([[2.0, -1.0, 1.0]], columns=["a", "b", "c"])

    constrained = enforce_no_leverage(weights)

    assert list(constrained.iloc[0]) == pytest.approx([0.5, -0.25, 0.25])


def test_enforce_no_leverage_rejects_non_finite_weights() -> None:
    weights = pd.DataFrame([[0.6, float("nan")]], columns=["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        enforce_no_leverage(weights)


def test_constrain_weight_change_caps_moves_at_a_quarter() -> None:
    previous = pd.Series([0.4, 0.4, 0.2], index=["mom", "carry", "revert"])
    proposed = pd.Series([0.8, 0.1, 0.2], index=["mom", "carry", "revert"])

    constrained = constrain_weight_change(previous, proposed)

    assert list(constrained) == pytest.approx([0.5, 0.3, 0.2])


def test_constrain_weight_change_lets_small_moves_through() -> None:
    previous = pd.Series([0.4, 0.6], index=["mom", "carry"])
    proposed = pd.Series([0.45, 0.55], index=["mom", "carry"])

    constrained = constrain_weight_change(previous, proposed)

    assert list(constrained) == pytest.approx([0.45, 0.55])


def test_constrain_weight_change_admits_a_new_strategy_slowly() -> None:
    previous = pd.Series([0.5, 0.5, 0.0], index=["mom", "carry", "events"])
    proposed = pd.Series([0.5, 0.5, 0.4], index=["mom", "carry", "events"])

    constrained = constrain_weight_change(previous, proposed)

    assert float(constrained["events"]) == pytest.approx(0.25 * (1.0 / 3.0))
    assert float(constrained["events"]) < float(proposed["events"])


def test_constrain_weight_change_passes_through_a_first_allocation() -> None:
    previous = pd.Series([0.0, 0.0], index=["mom", "carry"])
    proposed = pd.Series([0.5, 0.5], index=["mom", "carry"])

    constrained = constrain_weight_change(previous, proposed)

    assert list(constrained) == pytest.approx([0.5, 0.5])


def test_constrain_weight_change_honours_a_custom_limit() -> None:
    previous = pd.Series([0.4], index=["mom"])
    proposed = pd.Series([1.0], index=["mom"])

    constrained = constrain_weight_change(previous, proposed, max_fractional_change=0.1)

    assert float(constrained.iloc[0]) == pytest.approx(0.44)


def test_constrain_weight_change_rejects_misaligned_index() -> None:
    previous = pd.Series([0.4, 0.6], index=["mom", "carry"])
    proposed = pd.Series([0.4, 0.6], index=["mom", "events"])

    with pytest.raises(ValueError, match="index"):
        constrain_weight_change(previous, proposed)


def test_constrain_weight_change_rejects_non_finite_weights() -> None:
    previous = pd.Series([0.4, 0.6], index=["mom", "carry"])
    proposed = pd.Series([0.4, float("nan")], index=["mom", "carry"])

    with pytest.raises(ValueError, match="finite"):
        constrain_weight_change(previous, proposed)


@pytest.mark.parametrize("limit", [0.0, -0.25, 1.5])
def test_constrain_weight_change_rejects_bad_limit(limit: float) -> None:
    previous = pd.Series([0.4], index=["mom"])
    proposed = pd.Series([0.5], index=["mom"])

    with pytest.raises(ValueError):
        constrain_weight_change(previous, proposed, max_fractional_change=limit)

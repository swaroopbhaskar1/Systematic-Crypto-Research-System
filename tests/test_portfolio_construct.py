import math

import numpy as np
import pandas as pd
import pytest

import cq.portfolio.construct as construct
from cq.portfolio.construct import (
    apply_position_cap,
    combine_equal_weight,
    cross_sectional_neutral,
)

NEUTRALITY_TOLERANCE = 1e-12


def _ramp_scores(columns: int, rows: int = 1) -> pd.DataFrame:
    """Strictly increasing scores across columns so the extremes are known."""

    values = [[float(column) for column in range(columns)] for _ in range(rows)]
    return pd.DataFrame(
        values,
        index=pd.RangeIndex(rows),
        columns=[f"t{column}" for column in range(columns)],
    )


def test_combine_equal_weight_averages_with_hand_computed_values() -> None:
    first = pd.DataFrame([[0.2, 0.4], [-0.1, 0.0]], columns=["a", "b"])
    second = pd.DataFrame([[0.4, 0.0], [0.3, -0.2]], columns=["a", "b"])
    third = pd.DataFrame([[0.6, -0.1], [0.1, 0.2]], columns=["a", "b"])

    combined = combine_equal_weight([first, second, third])

    assert combined.iloc[0, 0] == pytest.approx(0.4)
    assert combined.iloc[0, 1] == pytest.approx(0.1)
    assert combined.iloc[1, 0] == pytest.approx(0.1)
    assert combined.iloc[1, 1] == pytest.approx(0.0)


def test_combine_equal_weight_gives_every_strategy_the_same_share() -> None:
    loud = pd.DataFrame([[1.0, -1.0]], columns=["a", "b"])
    quiet = pd.DataFrame([[0.0, 0.0]], columns=["a", "b"])

    combined = combine_equal_weight([loud, quiet, quiet, quiet])

    assert combined.iloc[0, 0] == pytest.approx(0.25)
    assert combined.iloc[0, 1] == pytest.approx(-0.25)


def test_combine_equal_weight_rejects_mismatched_columns() -> None:
    first = pd.DataFrame([[0.2, 0.4]], columns=["a", "b"])
    second = pd.DataFrame([[0.2, 0.4]], columns=["a", "c"])

    with pytest.raises(ValueError, match="columns"):
        combine_equal_weight([first, second])


def test_combine_equal_weight_rejects_mismatched_index() -> None:
    first = pd.DataFrame([[0.2, 0.4]], index=[0], columns=["a", "b"])
    second = pd.DataFrame([[0.2, 0.4]], index=[1], columns=["a", "b"])

    with pytest.raises(ValueError, match="index"):
        combine_equal_weight([first, second])


def test_combine_equal_weight_rejects_reordered_columns() -> None:
    first = pd.DataFrame([[0.2, 0.4]], columns=["a", "b"])
    second = pd.DataFrame([[0.4, 0.2]], columns=["b", "a"])

    with pytest.raises(ValueError, match="columns"):
        combine_equal_weight([first, second])


def test_combine_equal_weight_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        combine_equal_weight([])


def test_combine_equal_weight_rejects_empty_frames() -> None:
    empty = pd.DataFrame(index=pd.RangeIndex(0), columns=["a"], dtype=float)

    with pytest.raises(ValueError, match="empty"):
        combine_equal_weight([empty])


def test_combine_equal_weight_refuses_to_impute_missing_weights() -> None:
    first = pd.DataFrame([[0.2, float("nan")]], columns=["a", "b"])
    second = pd.DataFrame([[0.4, 0.4]], columns=["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        combine_equal_weight([first, second])


def test_combine_equal_weight_rejects_infinite_weights() -> None:
    first = pd.DataFrame([[0.2, float("inf")]], columns=["a", "b"])
    second = pd.DataFrame([[0.4, 0.4]], columns=["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        combine_equal_weight([first, second])


def test_combine_equal_weight_rejects_boolean_frames() -> None:
    flags = pd.DataFrame([[True, False]], columns=["a", "b"])

    with pytest.raises(TypeError, match="real numbers"):
        combine_equal_weight([flags])


def test_cross_sectional_neutral_is_dollar_matched_to_1e_12() -> None:
    rng = np.random.default_rng(20240824)
    scores = pd.DataFrame(
        rng.normal(size=(25, 37)),
        columns=[f"t{column}" for column in range(37)],
    )

    weights = cross_sectional_neutral(scores, per_position_cap=0.2)

    row_sums = weights.sum(axis="columns").abs()
    assert float(row_sums.max()) <= NEUTRALITY_TOLERANCE


def test_cross_sectional_neutral_longs_top_and_shorts_bottom_decile() -> None:
    scores = _ramp_scores(20)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.1,
        per_position_cap=1.0,
    )

    row = weights.iloc[0]
    assert float(row["t0"]) == pytest.approx(-0.25)
    assert float(row["t1"]) == pytest.approx(-0.25)
    assert float(row["t18"]) == pytest.approx(0.25)
    assert float(row["t19"]) == pytest.approx(0.25)
    assert float(row.iloc[2:18].abs().sum()) == 0.0
    assert float(row.abs().sum()) == pytest.approx(1.0)


def test_cross_sectional_neutral_honours_a_larger_fraction() -> None:
    scores = _ramp_scores(4)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.5,
        per_position_cap=1.0,
    )

    assert list(weights.iloc[0]) == pytest.approx([-0.25, -0.25, 0.25, 0.25])


def test_cross_sectional_neutral_excludes_nan_from_ranking() -> None:
    scores = _ramp_scores(12)
    scores.iloc[0, 0] = float("nan")
    scores.iloc[0, 11] = float("nan")

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.1,
        per_position_cap=1.0,
    )

    row = weights.iloc[0]
    assert float(row["t0"]) == 0.0
    assert float(row["t11"]) == 0.0
    assert float(row["t1"]) == pytest.approx(-0.5)
    assert float(row["t10"]) == pytest.approx(0.5)
    assert float(row.sum()) == pytest.approx(0.0, abs=NEUTRALITY_TOLERANCE)


def test_cross_sectional_neutral_zeroes_rows_with_too_few_scores() -> None:
    scores = _ramp_scores(9)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.1,
        per_position_cap=1.0,
    )

    assert float(weights.iloc[0].abs().sum()) == 0.0


def test_cross_sectional_neutral_zeroes_a_single_name_cross_section() -> None:
    scores = _ramp_scores(1)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.5,
        per_position_cap=1.0,
    )

    assert float(weights.iloc[0].abs().sum()) == 0.0


def test_cross_sectional_neutral_handles_the_minimum_viable_cross_section() -> None:
    scores = _ramp_scores(2)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.5,
        per_position_cap=1.0,
    )

    assert list(weights.iloc[0]) == pytest.approx([-0.5, 0.5])


def test_cross_sectional_neutral_never_emits_an_unbalanced_row() -> None:
    scores = pd.DataFrame(
        [
            [float(column) for column in range(10)],
            [float("nan")] * 6 + [1.0, 2.0, 3.0, 4.0],
            [float("nan")] * 10,
        ],
        columns=[f"t{column}" for column in range(10)],
    )

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.1,
        per_position_cap=1.0,
    )

    longs = (weights > 0.0).sum(axis="columns")
    shorts = (weights < 0.0).sum(axis="columns")
    assert list(longs) == list(shorts)
    assert float(weights.iloc[1].abs().sum()) == 0.0
    assert float(weights.iloc[2].abs().sum()) == 0.0


def test_cross_sectional_neutral_ranks_only_within_the_row() -> None:
    single = _ramp_scores(10)
    paired = pd.concat([single, single.iloc[[0]] * -1.0], ignore_index=True)

    from_single = cross_sectional_neutral(single, per_position_cap=1.0)
    from_paired = cross_sectional_neutral(paired, per_position_cap=1.0)

    pd.testing.assert_series_equal(
        from_single.iloc[0],
        from_paired.iloc[0],
        check_names=False,
    )


def test_cross_sectional_neutral_breaks_ties_by_column_order() -> None:
    scores = pd.DataFrame(
        [[5.0] * 10],
        columns=[f"t{column}" for column in range(10)],
    )

    weights = cross_sectional_neutral(scores, per_position_cap=1.0)

    assert float(weights.iloc[0]["t0"]) == pytest.approx(-0.5)
    assert float(weights.iloc[0]["t9"]) == pytest.approx(0.5)
    assert float(weights.iloc[0].abs().sum()) == pytest.approx(1.0)


def test_cross_sectional_neutral_applies_the_position_cap() -> None:
    scores = _ramp_scores(10)

    weights = cross_sectional_neutral(
        scores,
        decile_fraction=0.1,
        per_position_cap=0.2,
    )

    row = weights.iloc[0]
    assert float(row["t0"]) == pytest.approx(-0.2)
    assert float(row["t9"]) == pytest.approx(0.2)
    assert float(row.sum()) == pytest.approx(0.0, abs=NEUTRALITY_TOLERANCE)
    assert float(row.abs().sum()) == pytest.approx(0.4)


def test_cross_sectional_neutral_rejects_infinite_scores() -> None:
    scores = _ramp_scores(10)
    scores.iloc[0, 3] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        cross_sectional_neutral(scores, per_position_cap=1.0)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 0.6, 1.0, float("nan")])
def test_cross_sectional_neutral_rejects_bad_decile_fraction(fraction: float) -> None:
    scores = _ramp_scores(10)

    with pytest.raises(ValueError):
        cross_sectional_neutral(
            scores,
            decile_fraction=fraction,
            per_position_cap=1.0,
        )


def test_cross_sectional_neutral_rejects_boolean_decile_fraction() -> None:
    scores = _ramp_scores(10)

    with pytest.raises(TypeError, match="real number"):
        cross_sectional_neutral(
            scores,
            decile_fraction=True,  # pyright: ignore[reportArgumentType]
            per_position_cap=1.0,
        )


@pytest.mark.parametrize("cap", [0.0, -0.2, 1.5])
def test_cross_sectional_neutral_rejects_bad_cap(cap: float) -> None:
    scores = _ramp_scores(10)

    with pytest.raises(ValueError):
        cross_sectional_neutral(scores, per_position_cap=cap)


def test_apply_position_cap_binds_and_preserves_sign() -> None:
    weights = pd.DataFrame([[0.6, -0.3, 0.1]], columns=["a", "b", "c"])

    capped = apply_position_cap(weights, cap=0.25)

    assert list(capped.iloc[0]) == pytest.approx([0.25, -0.25, 0.1])


def test_apply_position_cap_does_not_renormalize() -> None:
    weights = pd.DataFrame([[0.6, -0.3, 0.1]], columns=["a", "b", "c"])

    capped = apply_position_cap(weights, cap=0.25)

    assert float(weights.iloc[0].abs().sum()) == pytest.approx(1.0)
    assert float(capped.iloc[0].abs().sum()) == pytest.approx(0.6)


def test_apply_position_cap_uses_the_precap_gross_as_reference() -> None:
    weights = pd.DataFrame([[1.2, -0.6, 0.2]], columns=["a", "b", "c"])

    capped = apply_position_cap(weights, cap=0.5)

    assert list(capped.iloc[0]) == pytest.approx([1.0, -0.6, 0.2])


def test_apply_position_cap_is_a_no_op_when_slack() -> None:
    weights = pd.DataFrame([[0.3, -0.3, 0.4]], columns=["a", "b", "c"])

    capped = apply_position_cap(weights, cap=0.5)

    pd.testing.assert_frame_equal(capped, weights)


def test_apply_position_cap_caps_each_row_independently() -> None:
    weights = pd.DataFrame(
        [[0.9, -0.1], [0.5, -0.5]],
        columns=["a", "b"],
    )

    capped = apply_position_cap(weights, cap=0.6)

    assert list(capped.iloc[0]) == pytest.approx([0.6, -0.1])
    assert list(capped.iloc[1]) == pytest.approx([0.5, -0.5])


def test_apply_position_cap_leaves_all_zero_rows_unchanged() -> None:
    weights = pd.DataFrame([[0.0, 0.0]], columns=["a", "b"])

    capped = apply_position_cap(weights, cap=0.2)

    assert list(capped.iloc[0]) == [0.0, 0.0]


def test_apply_position_cap_preserves_dollar_neutrality() -> None:
    weights = pd.DataFrame([[0.5, -0.5]], columns=["a", "b"])

    capped = apply_position_cap(weights, cap=0.1)

    assert float(capped.iloc[0].sum()) == pytest.approx(
        0.0, abs=NEUTRALITY_TOLERANCE
    )


def test_apply_position_cap_rejects_non_finite_weights() -> None:
    weights = pd.DataFrame([[0.5, math.nan]], columns=["a", "b"])

    with pytest.raises(ValueError, match="finite"):
        apply_position_cap(weights, cap=0.2)


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.01])
def test_apply_position_cap_rejects_invalid_cap(cap: float) -> None:
    weights = pd.DataFrame([[0.5, -0.5]], columns=["a", "b"])

    with pytest.raises(ValueError):
        apply_position_cap(weights, cap=cap)


def test_portfolio_construction_exposes_no_optimizer() -> None:
    forbidden = (
        "mean_variance",
        "risk_parity",
        "optimize",
        "optimise",
        "kelly_weights",
    )

    assert [name for name in dir(construct) if name in forbidden] == []

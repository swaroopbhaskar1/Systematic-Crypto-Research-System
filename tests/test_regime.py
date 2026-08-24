"""Adversarial tests for point-in-time regime tagging.

Regime attribution is only trustworthy if the label attached to a bar was
knowable at that bar.  A classifier that peeks forward makes every crash
bucket look survivable, because it labels the crash using the recovery.
"""

import math

import pandas as pd
import pytest

from cq.backtest.metrics import (
    REGIME_CRASH,
    REGIME_HIGH_VOL,
    REGIME_LOW_VOL,
    REGIME_TRENDING,
    RegimeConfig,
    classify_regimes,
    compounded_return,
    market_returns,
)

SMALL = RegimeConfig(
    volatility_window=3,
    drawdown_window=4,
    trend_window=3,
    crash_drawdown=0.10,
    high_volatility=1.0,
    trend_strength=1.0,
)


def series(values: list[float]) -> pd.Series:
    """Return a daily UTC-indexed return series starting 2024-01-01."""
    index = pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


class TestClassifyRegimes:
    def test_warmup_bars_carry_no_label(self) -> None:
        """Insufficient trailing history must yield no label, not a guess.

        Imputing "low_vol" for the warmup would silently attribute the first
        month of every backtest to a regime nobody measured.
        """
        result = classify_regimes(series([0.01, -0.01, 0.01, -0.01]), SMALL)
        assert len(result) == 2
        assert result.index[0] == pd.Timestamp("2024-01-03", tz="UTC")

    def test_crash_outranks_high_volatility(self) -> None:
        """A drawdown past the threshold is a crash even when vol is extreme.

        Wealth path 1.0 -> 0.95 -> 0.9025 -> 0.857375 is a 14.2625% drawdown
        from the trailing peak of 1.0, above the 10% crash threshold.
        """
        result = classify_regimes(series([-0.05, -0.05, -0.05]), SMALL)
        assert list(result) == [REGIME_CRASH]

    def test_high_volatility_when_drawdown_is_shallow(self) -> None:
        """Daily sample sigma 0.06 annualizes to 1.1463, above the 1.0 gate.

        The trailing peak is 1.06 and wealth ends at 0.9964, a 6% drawdown,
        which stays under the 10% crash threshold.
        """
        result = classify_regimes(series([0.06, 0.0, -0.06]), SMALL)
        assert list(result) == [REGIME_HIGH_VOL]

    def test_trending_when_move_dominates_noise(self) -> None:
        """Sum 0.06 against sigma*sqrt(3)=0.0086603 scores 6.93, above 1.0."""
        result = classify_regimes(series([0.02, 0.015, 0.025]), SMALL)
        assert list(result) == [REGIME_TRENDING]

    def test_low_vol_is_the_documented_residual(self) -> None:
        """Chop that neither crashes, spikes, nor trends lands in low_vol.

        Sum 0.01 against sigma*sqrt(3)=0.02 scores 0.5, below the 1.0 gate.
        """
        result = classify_regimes(series([0.01, -0.01, 0.01]), SMALL)
        assert list(result) == [REGIME_LOW_VOL]

    def test_zero_volatility_with_no_move_is_low_vol(self) -> None:
        """A dead-flat window must not divide by zero to reach trending."""
        result = classify_regimes(series([0.0, 0.0, 0.0]), SMALL)
        assert list(result) == [REGIME_LOW_VOL]

    def test_zero_volatility_with_a_move_is_trending(self) -> None:
        """Identical nonzero returns are pure trend with no dispersion."""
        result = classify_regimes(series([0.01, 0.01, 0.01]), SMALL)
        assert list(result) == [REGIME_TRENDING]

    def test_labels_are_unchanged_by_future_returns(self) -> None:
        """The strongest test here: extend the series and relabel.

        Every label at a timestamp present in the short series must be
        byte-identical in the long one.  Any centered window, negative shift,
        or full-sample statistic breaks this immediately.
        """
        history = [0.02, -0.03, 0.01, 0.04, -0.06, 0.05, 0.0, -0.01, 0.03]
        future = [0.5, -0.4, 0.3, -0.2] * 25
        short = classify_regimes(series(history), SMALL)
        long = classify_regimes(series(history + future), SMALL)
        assert len(short) == len(history) - 2
        assert list(short) == list(long.loc[short.index])

    def test_every_label_is_one_of_the_four_documented_regimes(self) -> None:
        """No classifier path may invent a fifth label."""
        history = [0.02, -0.03, 0.01, 0.04, -0.06, 0.05, 0.0, -0.01, 0.03]
        result = classify_regimes(series(history), SMALL)
        allowed = {REGIME_CRASH, REGIME_HIGH_VOL, REGIME_TRENDING, REGIME_LOW_VOL}
        assert set(result).issubset(allowed)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_returns_are_rejected(self, bad: float) -> None:
        """A NaN return must raise, never be dropped or filled."""
        with pytest.raises(ValueError):
            classify_regimes(series([0.01, bad, 0.01]), SMALL)

    def test_series_shorter_than_the_window_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            classify_regimes(series([0.01, 0.02]), SMALL)

    def test_unsorted_index_is_rejected(self) -> None:
        """Out-of-order bars make every trailing window a lie."""
        reversed_index = series([0.01, 0.02, 0.03]).iloc[::-1]
        with pytest.raises(ValueError):
            classify_regimes(reversed_index, SMALL)


class TestRegimeConfig:
    @pytest.mark.parametrize("window", [0, 1, -3])
    def test_windows_must_span_at_least_two_bars(self, window: int) -> None:
        with pytest.raises(ValueError):
            RegimeConfig(volatility_window=window)

    @pytest.mark.parametrize("threshold", [0.0, 1.0, 1.5, -0.1])
    def test_crash_drawdown_must_be_a_proper_fraction(
        self, threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            RegimeConfig(crash_drawdown=threshold)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan")])
    def test_thresholds_must_be_positive_and_finite(
        self, threshold: float
    ) -> None:
        with pytest.raises(ValueError):
            RegimeConfig(high_volatility=threshold)

    def test_defaults_match_the_documented_contract(self) -> None:
        config = RegimeConfig()
        assert config.volatility_window == 30
        assert config.drawdown_window == 90
        assert config.trend_window == 30
        assert config.crash_drawdown == 0.20
        assert config.high_volatility == 1.0
        assert config.trend_strength == 1.0


class TestMarketReturns:
    def frame(self, values: dict[str, list[float | None]]) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
        return pd.DataFrame(values, index=index, dtype=float)

    def test_equal_weight_mean_of_in_universe_returns(self) -> None:
        """Hand-computed: (0.10 + 0.20) / 2 = 0.15, then (-0.10 + 0.0) / 2."""
        close = self.frame({"A": [100.0, 110.0, 99.0], "B": [50.0, 60.0, 60.0]})
        universe = pd.DataFrame(True, index=close.index, columns=close.columns)
        result = market_returns(close, universe)
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(0.15, abs=1e-12)
        assert result.iloc[1] == pytest.approx(-0.05, abs=1e-12)

    def test_symbol_absent_at_either_end_is_excluded(self) -> None:
        """A return needs two consecutive in-universe observations.

        Pricing a listing bar against a price that did not exist manufactures
        a return out of nothing, which is the survivorship bug in miniature.
        """
        close = self.frame({"A": [100.0, 110.0, 121.0], "B": [None, 60.0, 66.0]})
        universe = pd.DataFrame(
            [[True, False], [True, True], [True, True]],
            index=close.index,
            columns=close.columns,
        )
        result = market_returns(close, universe)
        assert result.iloc[0] == pytest.approx(0.10, abs=1e-12)
        assert result.iloc[1] == pytest.approx(0.10, abs=1e-12)

    def test_bar_with_no_tradable_symbol_is_absent(self) -> None:
        """No imputation: a bar with nothing to measure carries no return."""
        close = self.frame({"A": [100.0, None, 121.0]})
        universe = pd.DataFrame(
            [[True], [False], [True]],
            index=close.index,
            columns=close.columns,
        )
        result = market_returns(close, universe)
        assert result.empty

    def test_mismatched_universe_shape_is_rejected(self) -> None:
        close = self.frame({"A": [100.0, 110.0, 121.0]})
        universe = pd.DataFrame(True, index=close.index, columns=["B"])
        with pytest.raises(ValueError):
            market_returns(close, universe)


class TestCompoundedReturn:
    def test_hand_computed_compounding(self) -> None:
        """1.10 * 0.90 * 1.05 - 1 = 0.0395 exactly."""
        assert compounded_return([0.10, -0.10, 0.05]) == pytest.approx(
            0.0395, abs=1e-12
        )

    def test_total_wipeout_is_representable(self) -> None:
        """A token going to zero is a real -100%, not an error."""
        assert compounded_return([0.10, -1.0]) == pytest.approx(-1.0, abs=1e-12)

    def test_return_below_negative_one_is_rejected(self) -> None:
        """Losing more than the position without leverage is impossible."""
        with pytest.raises(ValueError):
            compounded_return([-1.5])

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            compounded_return([])

    def test_nonfinite_input_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            compounded_return([0.01, math.nan])

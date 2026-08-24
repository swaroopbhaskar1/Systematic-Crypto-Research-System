"""Tests that the engine actually reports the metrics the spec requires.

Spec 7.3 names Sharpe, Sortino, max drawdown, Calmar, turnover, hit rate,
average holding period, and cost drag as a percentage of gross return, and
requires gross and net side by side.  An empty metrics dict is a silent
failure: the backtest looks like it ran and reports nothing that could
contradict the hypothesis.

The other property defended here is that an *undefined* metric is omitted
rather than reported as zero.  Zero Sharpe and undefined Sharpe are different
claims, and substituting one for the other flatters a flat strategy.
"""

import pandas as pd
import pytest

from cq.backtest import metrics as metric_functions
from cq.backtest.engine import BacktestResult, run
from cq.backtest.metrics import (
    REGIME_CRASH,
    REGIME_HIGH_VOL,
    REGIME_LOW_VOL,
    REGIME_TRENDING,
    RegimeConfig,
)
from cq.data.panel import Panel

from tests.fixtures.signals import ConstantWeightSignal, ScheduledWeightSignal

SYMBOL = "AAAUSDT"
QUOTE_VOLUME = 10_000_000.0
STARTING_EQUITY = 100_000.0
SMALL_REGIMES = RegimeConfig(
    volatility_window=3,
    drawdown_window=4,
    trend_window=3,
    crash_drawdown=0.10,
    high_volatility=1.0,
    trend_strength=1.0,
)
REGIME_LABELS = frozenset(
    {REGIME_CRASH, REGIME_HIGH_VOL, REGIME_TRENDING, REGIME_LOW_VOL}
)


def timestamps(count: int) -> list[int]:
    """Return Unix-millisecond daily bar stamps, the engine's convention."""
    stamps = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    return [int(stamp.value // 1_000_000) for stamp in stamps]


def build_panel(bars: list[tuple[float, float]]) -> Panel:
    """Build a one-symbol spot panel from explicit (open, close) bars."""
    rows = [
        {
            "ts": stamp,
            "symbol": SYMBOL,
            "market_type": "spot",
            "open": open_,
            "close": close,
            "quote_volume": QUOTE_VOLUME,
            "adv": 1_000_000.0,
            "volatility": 0.01,
            "liquidity_decile": 10,
            "in_universe": True,
        }
        for stamp, (open_, close) in zip(timestamps(len(bars)), bars, strict=True)
    ]
    return Panel.from_long(pd.DataFrame(rows), market_type="spot")


def scheduled(panel: Panel, raw_weights: list[float]) -> ScheduledWeightSignal:
    """Wrap an explicit unlagged weight schedule for the single symbol."""
    close = panel.field("close")
    frame = pd.DataFrame(
        {SYMBOL: raw_weights},
        index=close.index,
        dtype=float,
    )
    return ScheduledWeightSignal(frame.loc[:, close.columns])


def single_round_trip() -> BacktestResult:
    """Buy 1000 units at 100.01 on bar 1, sell at 109.989 on bar 2.

    Decile 10 carries a 2 bps spread, so the adverse half spread is 1 bp.
    Taker fee and impact coefficient are zero in the engine default model,
    which keeps this arithmetic exact.
    """
    panel = build_panel(
        [
            (100.0, 100.0),
            (100.0, 110.0),
            (110.0, 110.0),
            (110.0, 110.0),
            (110.0, 110.0),
        ]
    )
    return run(
        panel,
        scheduled(panel, [1.0, 0.0, 0.0, 0.0, 0.0]),
        starting_equity=STARTING_EQUITY,
    )


class TestGrossAndNetSideBySide:
    def test_gross_and_net_returns_are_both_reported(self) -> None:
        """Spec 7.3: the gross-net gap is the most informative number here."""
        result = single_round_trip()
        assert result.metrics["gross_return"] == pytest.approx(0.10, abs=1e-12)
        assert result.metrics["net_return"] == pytest.approx(0.09979, abs=1e-12)

    def test_cost_drag_is_the_hand_computed_percentage_of_gross(self) -> None:
        """(0.10 - 0.09979) / 0.10 * 100 = 0.21 percent of gross return."""
        result = single_round_trip()
        assert result.metrics["cost_drag_percent"] == pytest.approx(
            0.21, abs=1e-9
        )

    def test_net_never_flatters_gross_when_costs_are_paid(self) -> None:
        result = single_round_trip()
        assert result.metrics["net_return"] < result.metrics["gross_return"]

    def test_risk_metrics_are_reported_for_both_paths(self) -> None:
        """Reporting only net Sharpe hides how much of the edge costs ate."""
        result = single_round_trip()
        for key in (
            "gross_sharpe",
            "net_sharpe",
            "gross_max_drawdown",
            "net_max_drawdown",
        ):
            assert key in result.metrics


class TestMetricsMatchTheMetricsLayer:
    def test_sharpe_matches_the_tested_metric_function(self) -> None:
        """Wiring test: the engine must not reimplement the formula."""
        result = single_round_trip()
        expected = metric_functions.annualized_sharpe(
            float(value) for value in result.net_returns.iloc[1:]
        )
        assert result.metrics["net_sharpe"] == pytest.approx(expected, abs=1e-12)

    def test_max_drawdown_is_the_hand_computed_fraction(self) -> None:
        """Peak 109990 to trough 109979 is a drawdown of 11 / 109990."""
        result = single_round_trip()
        assert result.metrics["net_max_drawdown"] == pytest.approx(
            11.0 / 109_990.0, abs=1e-15
        )

    def test_turnover_is_the_hand_computed_average(self) -> None:
        """Weights 0,1,0,0,0 give per-bar moves 0.5, 0.5, 0, 0; mean 0.25."""
        result = single_round_trip()
        assert result.metrics["turnover"] == pytest.approx(0.25, abs=1e-12)

    def test_bar_count_is_reported(self) -> None:
        result = single_round_trip()
        assert result.metrics["n_bars"] == 5


class TestRoundTrips:
    def test_winning_round_trip_is_measured_net_of_costs(self) -> None:
        """Basis 100010 against proceeds 109989 is a 9979 gain."""
        result = single_round_trip()
        assert result.metrics["n_round_trips"] == 1
        assert result.metrics["hit_rate"] == pytest.approx(1.0, abs=1e-12)
        assert result.metrics["avg_holding_period_days"] == pytest.approx(
            1.0, abs=1e-12
        )

    def test_losing_round_trip_is_not_counted_as_a_win(self) -> None:
        """Basis 100010 against proceeds 89991 is a 10019 loss."""
        panel = build_panel(
            [(100.0, 100.0), (100.0, 90.0), (90.0, 90.0), (90.0, 90.0)]
        )
        result = run(
            panel,
            scheduled(panel, [1.0, 0.0, 0.0, 0.0]),
            starting_equity=STARTING_EQUITY,
        )
        assert result.metrics["n_round_trips"] == 1
        assert result.metrics["hit_rate"] == pytest.approx(0.0, abs=1e-12)

    def test_profitable_short_is_a_win_despite_a_negative_basis(self) -> None:
        """Short 1000 at 99.99, cover at 90.009: a 9981 gain on 99990 basis.

        Average-cost accounting on a short holds a negative cash basis.  A
        naive division by that signed basis flips the sign of the trade and
        reports a winning short as a loss.
        """
        panel = build_panel(
            [(100.0, 100.0), (100.0, 90.0), (90.0, 90.0), (90.0, 90.0)]
        )
        result = run(
            panel,
            scheduled(panel, [-1.0, 0.0, 0.0, 0.0]),
            starting_equity=STARTING_EQUITY,
        )
        assert result.metrics["n_round_trips"] == 1
        assert result.metrics["hit_rate"] == pytest.approx(1.0, abs=1e-12)

    def test_two_round_trips_give_a_fractional_hit_rate(self) -> None:
        """One winner then one loser must report exactly 0.5, not 1.0."""
        panel = build_panel(
            [
                (100.0, 100.0),
                (100.0, 110.0),
                (110.0, 110.0),
                (110.0, 110.0),
                (110.0, 100.0),
                (100.0, 100.0),
                (100.0, 100.0),
            ]
        )
        result = run(
            panel,
            scheduled(panel, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            starting_equity=STARTING_EQUITY,
        )
        assert result.metrics["n_round_trips"] == 2
        assert result.metrics["hit_rate"] == pytest.approx(0.5, abs=1e-12)
        assert result.metrics["avg_holding_period_days"] == pytest.approx(
            1.0, abs=1e-12
        )

    def test_position_still_open_at_the_end_is_not_a_round_trip(self) -> None:
        """Counting an open position as closed invents an unrealized win."""
        panel = build_panel(
            [(100.0, 100.0), (100.0, 110.0), (110.0, 120.0), (120.0, 130.0)]
        )
        result = run(
            panel,
            scheduled(panel, [1.0, 1.0, 1.0, 1.0]),
            starting_equity=STARTING_EQUITY,
        )
        assert result.metrics["n_round_trips"] == 0
        assert "hit_rate" not in result.metrics


class TestUndefinedMetricsAreOmitted:
    def flat(self) -> BacktestResult:
        panel = build_panel([(100.0, 100.0)] * 5)
        return run(
            panel,
            ConstantWeightSignal(weight=0.0),
            starting_equity=STARTING_EQUITY,
        )

    @pytest.mark.parametrize(
        "key",
        [
            "net_sharpe",
            "gross_sharpe",
            "net_sortino",
            "net_calmar",
            "cost_drag_percent",
            "hit_rate",
            "avg_holding_period_days",
        ],
    )
    def test_undefined_metric_is_absent_rather_than_zero(
        self, key: str
    ) -> None:
        """A flat equity path has no Sharpe.  Reporting 0.0 asserts a fact.

        Zero variance, zero downside, zero drawdown, and zero gross return
        each make a statistic undefined.  The honest report omits the key.
        """
        assert key not in self.flat().metrics

    def test_defined_metrics_are_still_reported_on_a_flat_path(self) -> None:
        """Omission must be surgical, not a blanket bail-out."""
        result = self.flat()
        assert result.metrics["net_return"] == pytest.approx(0.0, abs=1e-12)
        assert result.metrics["net_max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert result.metrics["turnover"] == pytest.approx(0.0, abs=1e-12)
        assert result.metrics["n_round_trips"] == 0


class TestRegimeMetrics:
    def volatile_panel(self) -> Panel:
        closes = [
            100.0,
            106.0,
            106.0,
            99.6,
            104.6,
            94.1,
            98.8,
            98.8,
            97.8,
            100.7,
            95.7,
            98.5,
        ]
        return build_panel([(close, close) for close in closes])

    def test_regimes_are_recorded_for_every_eligible_bar(self) -> None:
        """Spec 7.2 step 8 requires a regime tag, and 7.3 requires the split.

        Eleven market returns with a three-bar volatility window leaves nine
        labelled bars; the two warmup bars must not be attributed anywhere.
        """
        panel = self.volatile_panel()
        result = run(
            panel,
            ConstantWeightSignal(weight=0.5),
            starting_equity=STARTING_EQUITY,
            regime_config=SMALL_REGIMES,
        )
        assert result.regime_metrics
        assert set(result.regime_metrics).issubset(REGIME_LABELS)
        total = sum(
            int(bucket["n_bars"]) for bucket in result.regime_metrics.values()
        )
        assert total == 9

    def test_each_regime_reports_gross_and_net(self) -> None:
        panel = self.volatile_panel()
        result = run(
            panel,
            ConstantWeightSignal(weight=0.5),
            starting_equity=STARTING_EQUITY,
            regime_config=SMALL_REGIMES,
        )
        for bucket in result.regime_metrics.values():
            assert "gross_return" in bucket
            assert "net_return" in bucket

    def test_drawdown_is_not_reported_per_regime(self) -> None:
        """Regime rows are non-contiguous, so a peak-to-trough path is fiction.

        Reporting a drawdown over scattered bars would understate the real
        crash drawdown, which is exactly the number a reader wants.
        """
        panel = self.volatile_panel()
        result = run(
            panel,
            ConstantWeightSignal(weight=0.5),
            starting_equity=STARTING_EQUITY,
            regime_config=SMALL_REGIMES,
        )
        for bucket in result.regime_metrics.values():
            assert "net_max_drawdown" not in bucket
            assert "net_calmar" not in bucket

    def test_short_history_yields_no_regime_attribution(self) -> None:
        """With the default 30-bar window a five-bar run must claim nothing."""
        result = single_round_trip()
        assert result.regime_metrics == {}

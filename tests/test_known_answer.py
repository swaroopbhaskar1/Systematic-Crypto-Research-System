from dataclasses import dataclass

import pandas as pd
import pytest
from cq.backtest.costs import CostModel
from cq.backtest.engine import run_backtest

from cq.data.panel import Panel

BAR_TIMESTAMPS = pd.date_range(
    "2024-01-01", periods=10, freq="D", tz="UTC"
).asi8 // 1_000_000
SYMBOLS = ("AAAUSDT", "BBBUSDT")


@dataclass(frozen=True)
class FixedSignal:
    """A test-only signal whose values are unlagged target weights."""

    weights: pd.DataFrame

    def compute(self, panel: Panel) -> pd.DataFrame:
        close = panel.field("close")
        assert self.weights.index.equals(close.index)
        assert self.weights.columns.equals(close.columns)
        return self.weights.copy()


@pytest.fixture
def daily_panel() -> Panel:
    # Each tuple is (open, high, low, close, quote_volume, in_universe).
    # All bars stay in-universe so this arithmetic oracle is independent of
    # the separate delisting and future-listing integration contract.
    bars: dict[str, list[tuple[float, float, float, float, float, bool]]] = {
        "AAAUSDT": [
            (100.0, 102.0, 99.0, 100.0, 100_000.0, True),
            (100.0, 106.0, 99.0, 104.0, 120_000.0, True),
            (105.0, 108.0, 103.0, 106.0, 110_000.0, True),
            (110.0, 112.0, 107.0, 108.0, 130_000.0, True),
            (107.0, 109.0, 101.0, 103.0, 125_000.0, True),
            (100.0, 102.0, 96.0, 98.0, 115_000.0, True),
            (99.0, 101.0, 97.0, 100.0, 105_000.0, True),
            (101.0, 104.0, 100.0, 103.0, 140_000.0, True),
            (103.0, 105.0, 101.0, 102.0, 135_000.0, True),
            (102.0, 106.0, 101.0, 105.0, 150_000.0, True),
        ],
        "BBBUSDT": [
            (49.0, 51.0, 48.0, 50.0, 200_000.0, True),
            (50.0, 51.0, 49.0, 50.0, 200_000.0, True),
            (50.0, 53.0, 49.0, 52.0, 210_000.0, True),
            (53.0, 55.0, 52.0, 54.0, 220_000.0, True),
            (55.0, 57.0, 54.0, 56.0, 230_000.0, True),
            (56.0, 58.0, 55.0, 57.0, 240_000.0, True),
            (57.0, 59.0, 56.0, 58.0, 250_000.0, True),
            (58.0, 60.0, 57.0, 59.0, 260_000.0, True),
            (59.0, 61.0, 58.0, 60.0, 270_000.0, True),
            (60.0, 62.0, 59.0, 61.0, 280_000.0, True),
        ],
    }
    # ADV is hand-written so each expected trade is exactly 1% of ADV. With
    # volatility 1% and impact coefficient 10%, impact is 1 bp of open-price
    # notional. Non-trade-bar ADV values are immaterial but explicit.
    adv = {
        "AAAUSDT": [
            10_000.0,
            10_000.0,
            10_500.0,
            10_000.0,
            10_000.0,
            10_033.003609109437,
            9_932.673573018341,
            10_000.0,
            10_000.0,
            9_917.362031734125,
        ],
        "BBBUSDT": [
            10_000.0,
            10_000.0,
            10_000.0,
            20_038.09,
            20_794.244339622644,
            10_000.0,
            10_000.0,
            20_025.776880347064,
            20_371.048895525462,
            9_917.362031734125,
        ],
    }
    rows: list[dict[str, object]] = []
    for symbol in SYMBOLS:
        for ts, bar, bar_adv in zip(
            BAR_TIMESTAMPS, bars[symbol], adv[symbol], strict=True
        ):
            open_, high, low, close, quote_volume, mask = bar
            rows.append(
                {
                    "ts": ts,
                    "symbol": symbol,
                    "market_type": "spot",
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "quote_volume": quote_volume,
                    "adv": bar_adv,
                    "volatility": 0.01,
                    "liquidity_decile": 5,
                    "in_universe": mask,
                }
            )
    return Panel.from_long(pd.DataFrame(rows), market_type="spot")


@pytest.fixture
def signal() -> FixedSignal:
    # These are weights, never quantities.  The engine shifts once and uses:
    # target_notional[t + 1] = weight[t] * net_equity[t]
    # target_quantity[t + 1] = target_notional[t + 1] / open[t + 1].
    # Partial weights intentionally isolate arithmetic before the future
    # compiler enforces sum-one long-only or sum-zero market-neutral rows.
    return FixedSignal(
        pd.DataFrame(
            [
                (0.10, 0.00),
                (0.00, 0.00),
                (0.00, 0.20),
                (0.00, 0.00),
                (-0.10, 0.00),
                (0.00, 0.00),
                (0.00, -0.20),
                (0.00, 0.00),
                (0.10, -0.10),
                (0.00, 0.00),
            ],
            index=BAR_TIMESTAMPS,
            columns=SYMBOLS,
            dtype=float,
        )
    )


def test_known_answer_next_open_costs_positions_and_equity(
    daily_panel: Panel, signal: FixedSignal
) -> None:
    # Independent paper convention (all dollar values):
    #
    # q*[t] = weight[t - 1] * net_equity[t - 1] / open[t].
    # N = abs(q* - q) * open.  ADV is fixed at 100*N on every trade bar.
    # fill(buy/sell) = open * (1 +/- .01) for the 200-bp full spread.
    # fee = .005*N; impact = N*.10*.01*sqrt(N/ADV) = .0001*N.
    # trade.cost = fee + impact = .0051*N.  The spread is already paid
    # through cash at the worse fill and is not deducted from cash again.
    #
    # Examples:
    # t1: qAAA=.10*1000/100=1; net=1000 + 1*(104-100)
    #     - 1 spread - .510 fee/impact = 1002.490.
    # t3: qBBB=.20*1001.9045/53=3.780771698113208;
    #     net=1001.9045 + q*(54-53) - 2.003809 - 1.02194259
    #     = 1002.659520108113.
    # t9: qAAA=.10*991.736203173412/102=.972290395268051 and
    #     qBBB=-.10*991.736203173412/60=-1.652893671955687.
    #
    # Gross uses these same quantities and frictionless open fills.  Net uses
    # worse-side fills plus fee/impact.  The minimum participation cap is
    # $2,000, well above the largest requested notional ($207.95).
    cost_model = CostModel(
        taker_bps=50.0,
        impact_coefficient=0.10,
        spread_bps_by_decile={
            1: 300.0,
            2: 275.0,
            3: 250.0,
            4: 225.0,
            5: 200.0,
            6: 175.0,
            7: 150.0,
            8: 125.0,
            9: 100.0,
            10: 75.0,
        },
    )

    result = run_backtest(
        panel=daily_panel,
        signal=signal,
        starting_equity=1_000.0,
        cost_model=cost_model,
    )

    expected_equity = pd.Series(
        [
            1000.0,
            1002.49,
            1001.9045000000001,
            1002.6595201081134,
            1003.3003609109435,
            1003.7919780877897,
            1001.2888440173531,
            994.8122315566367,
            991.7362031734124,
            990.0051373536771,
        ],
        index=BAR_TIMESTAMPS,
        name="equity",
    )
    expected_gross_equity = pd.Series(
        [
            1000.0,
            1004.0,
            1005.0,
            1008.7807716981133,
            1012.5615433962264,
            1014.5681441180483,
            1013.5648437571374,
            1010.1121236053533,
            1010.1121236053533,
            1011.3761011192017,
        ],
        index=BAR_TIMESTAMPS,
        name="gross_equity",
    )
    pd.testing.assert_series_equal(
        result.equity,
        expected_equity,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )
    pd.testing.assert_series_equal(
        result.gross_equity,
        expected_gross_equity,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )
    expected_net_returns = pd.Series(
        [
            float("nan"),
            0.00249,
            -0.000584045726142,
            0.000753584905660,
            0.000639140994503,
            0.00049,
            -0.002493678097732,
            -0.006468275862069,
            -0.003092069322882,
            -0.001745490196078,
        ],
        index=BAR_TIMESTAMPS,
        name="net_returns",
    )
    expected_gross_returns = pd.Series(
        [
            float("nan"),
            0.004,
            0.000996015936255,
            0.003761961888670,
            0.003747862572508,
            0.001981707418091,
            -0.000988894010449,
            -0.003406511357463,
            0.0,
            0.001251323971182,
        ],
        index=BAR_TIMESTAMPS,
        name="gross_returns",
    )
    pd.testing.assert_series_equal(
        result.net_returns,
        expected_net_returns,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )
    pd.testing.assert_series_equal(
        result.gross_returns,
        expected_gross_returns,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )

    expected_positions = pd.DataFrame(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
            (0.0, 3.780771698113208),
            (0.0, 0.0),
            (-1.003300360910944, 0.0),
            (0.0, 0.0),
            (0.0, -3.452720151783977),
            (0.0, 0.0),
            (0.972290395268051, -1.652893671955687),
        ],
        index=BAR_TIMESTAMPS,
        columns=SYMBOLS,
    )
    pd.testing.assert_frame_equal(
        result.positions,
        expected_positions,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )

    expected_trades = pd.DataFrame(
        {
            "timestamp": [
                BAR_TIMESTAMPS[1],
                BAR_TIMESTAMPS[2],
                BAR_TIMESTAMPS[3],
                BAR_TIMESTAMPS[4],
                BAR_TIMESTAMPS[5],
                BAR_TIMESTAMPS[6],
                BAR_TIMESTAMPS[7],
                BAR_TIMESTAMPS[8],
                BAR_TIMESTAMPS[9],
                BAR_TIMESTAMPS[9],
            ],
            "symbol": [
                "AAAUSDT",
                "AAAUSDT",
                "BBBUSDT",
                "BBBUSDT",
                "AAAUSDT",
                "AAAUSDT",
                "BBBUSDT",
                "BBBUSDT",
                "AAAUSDT",
                "BBBUSDT",
            ],
            "side": [
                "buy",
                "sell",
                "buy",
                "sell",
                "sell",
                "buy",
                "sell",
                "buy",
                "buy",
                "sell",
            ],
            "quantity": [
                1.0,
                1.0,
                3.780771698113208,
                3.780771698113208,
                1.003300360910944,
                1.003300360910944,
                3.452720151783977,
                3.452720151783977,
                0.972290395268051,
                1.652893671955687,
            ],
            "price": [
                101.0,
                103.95,
                53.53,
                54.45,
                99.0,
                99.99,
                57.42,
                59.59,
                103.02,
                59.4,
            ],
            "cost": [
                0.51,
                0.5355,
                1.02194259,
                1.060506461320755,
                0.511683184064581,
                0.506566352223935,
                1.0213146208977,
                1.038923493671799,
                0.50578546361844,
                0.50578546361844,
            ],
        }
    )
    pd.testing.assert_frame_equal(
        result.trades.loc[:, expected_trades.columns].reset_index(drop=True),
        expected_trades,
        check_exact=False,
        rtol=0.0,
        atol=1e-9,
    )

    assert result.n_trades == 10
    assert result.data_span == pd.Timedelta(days=9)
    assert result.gross_equity.iloc[-1] - result.equity.iloc[-1] == pytest.approx(
        21.37096376552477, abs=1e-9
    )
    assert result.trades["cost"].sum() == pytest.approx(
        7.21800762941565,
        abs=1e-9,
    )
    # The remaining 14.15295613610912 is the implied half-spread paid once
    # through fill prices: 21.37096376552477 - 7.21800762941565.
    assert (
        result.gross_equity.iloc[-1]
        - result.equity.iloc[-1]
        - result.trades["cost"].sum()
    ) == pytest.approx(14.15295613610912, abs=1e-9)

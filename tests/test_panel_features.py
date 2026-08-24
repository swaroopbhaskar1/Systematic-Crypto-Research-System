"""Adversarial tests for derived execution features.

The store holds OHLCV.  The engine needs average daily volume, trailing
volatility, and a liquidity decile to price execution.  Deriving those is
where lookahead is easiest to introduce and hardest to see: an ADV that
includes the current bar's own volume means the backtest sized a trade using
the volume that trade helped create.
"""

import pandas as pd
import pytest

from cq.data.panel import add_execution_features

SYMBOL = "AAAUSDT"


def timestamps(count: int) -> list[int]:
    """Return Unix-millisecond daily bar stamps."""
    stamps = pd.date_range("2024-01-01", periods=count, freq="D", tz="UTC")
    return [int(stamp.value // 1_000_000) for stamp in stamps]


def long_frame(
    closes: dict[str, list[float]],
    volumes: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Build a store-shaped long frame for the given per-symbol closes."""
    count = len(next(iter(closes.values())))
    stamps = timestamps(count)
    rows: list[dict[str, object]] = []
    for symbol, series in closes.items():
        volume = volumes[symbol] if volumes is not None else [1_000_000.0] * count
        for stamp, close, quote_volume in zip(stamps, series, volume, strict=True):
            rows.append(
                {
                    "ts": stamp,
                    "symbol": symbol,
                    "market_type": "spot",
                    "open": close,
                    "close": close,
                    "quote_volume": quote_volume,
                    "in_universe": True,
                }
            )
    return pd.DataFrame(rows)


def cell(frame: pd.DataFrame, symbol: str, position: int, column: str) -> object:
    """Return one derived value by symbol and bar position."""
    stamps = sorted(set(int(value) for value in frame["ts"]))
    match = frame.loc[
        (frame["symbol"] == symbol) & (frame["ts"] == stamps[position]),
        column,
    ]
    return match.iloc[0]


class TestAverageDailyVolume:
    def test_adv_excludes_the_current_bar(self) -> None:
        """ADV at bar 3 is the mean of bars 0, 1, and 2: exactly 200.

        Including bar 3's own volume would let a trade be sized against the
        liquidity it is about to consume.
        """
        volumes = {SYMBOL: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]}
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        result = add_execution_features(
            long_frame(closes, volumes),
            adv_window=3,
            volatility_window=3,
        )
        assert cell(result, SYMBOL, 3, "adv") == pytest.approx(200.0, abs=1e-12)
        assert cell(result, SYMBOL, 4, "adv") == pytest.approx(300.0, abs=1e-12)
        assert cell(result, SYMBOL, 5, "adv") == pytest.approx(400.0, abs=1e-12)

    def test_bars_without_a_full_window_are_not_tradable(self) -> None:
        """No imputation: an unmodellable bar leaves the universe.

        Substituting a default ADV would give a brand-new listing the
        execution profile of an established one.
        """
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        result = add_execution_features(
            long_frame(closes),
            adv_window=3,
            volatility_window=3,
        )
        for position in range(4):
            assert not bool(cell(result, SYMBOL, position, "in_universe"))
        assert bool(cell(result, SYMBOL, 4, "in_universe"))


class TestTrailingVolatility:
    def test_volatility_is_the_hand_computed_trailing_deviation(self) -> None:
        """Returns 0.10, -0.10, 0.10 have sample deviation 0.11547005383792516.

        The value lands on bar 4 because the window closes at bar 3.
        """
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        result = add_execution_features(
            long_frame(closes),
            adv_window=3,
            volatility_window=3,
        )
        assert cell(result, SYMBOL, 4, "volatility") == pytest.approx(
            0.11547005383792516, abs=1e-15
        )

    def test_a_flat_window_is_not_tradable(self) -> None:
        """Zero volatility makes the impact model degenerate, so stand aside.

        A constant price is either a stablecoin or a halted market.  Modelling
        it as having zero impact would report free liquidity.
        """
        closes = {SYMBOL: [100.0] * 6}
        result = add_execution_features(
            long_frame(closes),
            adv_window=3,
            volatility_window=3,
        )
        assert not bool(cell(result, SYMBOL, 5, "in_universe"))


class TestLiquidityDecile:
    def frame(self) -> pd.DataFrame:
        closes = {
            f"S{index:02d}USDT": [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]
            for index in range(10)
        }
        volumes = {
            symbol: [1_000.0 * (index + 1)] * 6
            for index, symbol in enumerate(sorted(closes))
        }
        return add_execution_features(
            long_frame(closes, volumes),
            adv_window=3,
            volatility_window=3,
        )

    def test_least_liquid_symbol_takes_decile_one(self) -> None:
        """Decile 1 carries the widest spread, so it must be the thinnest name.

        Inverting this mapping would charge the most liquid names the widest
        spread and quietly make small-cap strategies look cheap to trade.
        """
        result = self.frame()
        assert int(cell(result, "S00USDT", 4, "liquidity_decile")) == 1
        assert int(cell(result, "S09USDT", 4, "liquidity_decile")) == 10

    def test_deciles_span_the_full_range_exactly_once(self) -> None:
        result = self.frame()
        stamps = sorted(set(int(value) for value in result["ts"]))
        bar = result.loc[
            (result["ts"] == stamps[4]) & result["in_universe"],
            "liquidity_decile",
        ]
        assert sorted(int(value) for value in bar) == list(range(1, 11))

    def test_decile_is_an_integer_in_range(self) -> None:
        result = self.frame()
        deciles = result.loc[result["in_universe"], "liquidity_decile"]
        assert not deciles.empty
        for value in deciles:
            assert int(value) == value
            assert 1 <= int(value) <= 10


class TestNoLookahead:
    def base_closes(self) -> dict[str, list[float]]:
        return {
            "AAAUSDT": [100.0, 110.0, 99.0, 108.9, 100.0, 105.0, 102.0, 99.0],
            "BBBUSDT": [50.0, 52.0, 49.0, 51.0, 50.0, 53.0, 52.0, 50.0],
        }

    def test_features_at_t_are_unchanged_by_future_bars(self) -> None:
        """The strongest test here: derive twice and compare the overlap.

        Any centered window, negative shift, or whole-sample normalization
        changes an early value once later bars exist.
        """
        closes = self.base_closes()
        extended = {
            symbol: series + [value * 3.0 for value in series]
            for symbol, series in closes.items()
        }
        short = add_execution_features(
            long_frame(closes),
            adv_window=3,
            volatility_window=3,
        )
        long = add_execution_features(
            long_frame(extended),
            adv_window=3,
            volatility_window=3,
        )
        keys = ["ts", "symbol"]
        columns = keys + ["adv", "volatility", "liquidity_decile", "in_universe"]
        overlap = long.loc[long["ts"].isin(set(short["ts"])), columns]
        pd.testing.assert_frame_equal(
            short.loc[:, columns].sort_values(keys).reset_index(drop=True),
            overlap.sort_values(keys).reset_index(drop=True),
        )

    def test_missing_close_is_never_filled(self) -> None:
        """A gap must stay a gap and must not become a tradable bar."""
        closes = self.base_closes()
        closes["BBBUSDT"][3] = float("nan")
        result = add_execution_features(
            long_frame(closes),
            adv_window=3,
            volatility_window=3,
        )
        assert pd.isna(cell(result, "BBBUSDT", 3, "close"))
        assert not bool(cell(result, "BBBUSDT", 3, "in_universe"))


class TestUniversePreservation:
    def test_a_symbol_already_out_of_universe_stays_out(self) -> None:
        """Feature derivation may only remove tradability, never grant it."""
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        frame = long_frame(closes)
        frame.loc[frame["ts"] == timestamps(6)[4], "in_universe"] = False
        result = add_execution_features(
            frame,
            adv_window=3,
            volatility_window=3,
        )
        assert not bool(cell(result, SYMBOL, 4, "in_universe"))

    def test_required_columns_are_enforced(self) -> None:
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        frame = long_frame(closes).drop(columns=["quote_volume"])
        with pytest.raises(ValueError):
            add_execution_features(frame, adv_window=3, volatility_window=3)

    @pytest.mark.parametrize("window", [0, 1, -2])
    def test_windows_must_span_at_least_two_bars(self, window: int) -> None:
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        with pytest.raises(ValueError):
            add_execution_features(long_frame(closes), adv_window=window)

    def test_duplicate_logical_key_is_rejected(self) -> None:
        """Two rows for one bar would silently double a symbol's volume."""
        closes = {SYMBOL: [100.0, 110.0, 99.0, 108.9, 100.0, 105.0]}
        frame = long_frame(closes)
        doubled = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError):
            add_execution_features(doubled, adv_window=3, volatility_window=3)

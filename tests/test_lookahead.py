"""Adversarial lookahead tests for the M2 backtest boundary.

Each synthetic bar has an IID intrabar return and is gapless:
``open[t] == close[t - 1]``.  A bar's close becomes observable only at that
bar's end.  Therefore an honest signal computed at ``t`` may use ``close[t]``
and is filled at ``open[t + 1]`` after the engine's one-bar shift.  The
cheating control reads ``close[t + 1]`` at ``t``; because
``open[t + 1] == close[t]``, it knows the sign of the entire next intrabar
return before the fill and earns impossible returns.  M2 deliberately detects
that vulnerability; it does not claim to sandbox arbitrary signal code.
"""

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from fixtures.signals import (
    CheatingNextCloseSignal,
    ConstantWeightSignal,
    ScheduledWeightSignal,
    Signal,
    TrailingMomentumSignal,
)

from cq.data.panel import Panel

DAY_MS = 86_400_000
NOISE_SEED = 20_260_824
NOISE_BARS = 8_192
ANNUALIZATION = 365
STARTING_EQUITY = 1_000.0
HISTORY_BARS = 400
EXTENSION_BARS = 100
PRODUCTION_ROOT = Path(__file__).resolve().parents[1] / "cq"


def _panel_from_returns(returns: np.ndarray) -> Panel:
    """Build gapless bars whose supplied values are genuine intrabar returns."""
    if returns.ndim != 1 or len(returns) < 2:
        raise ValueError("returns must be a one-dimensional multi-bar array")
    if not np.isfinite(returns).all() or (returns <= -1.0).any():
        raise ValueError("returns must be finite and greater than -1")

    open_ = np.empty(len(returns), dtype=float)
    open_[0] = 100.0
    close = 100.0 * np.cumprod(1.0 + returns)
    open_[1:] = close[:-1]
    frame = pd.DataFrame(
        {
            "ts": np.arange(len(close), dtype=np.int64) * DAY_MS,
            "symbol": "NOISEUSDT",
            "market_type": "spot",
            "open": open_,
            "close": close,
            "quote_volume": 1_000_000_000.0,
            "adv": 1_000_000_000.0,
            "volatility": 0.01,
            "liquidity_decile": 10,
            "in_universe": True,
        }
    )
    return Panel.from_long(frame, market_type="spot")


def _iid_noise_panel(bar_count: int = NOISE_BARS) -> Panel:
    rng = np.random.default_rng(NOISE_SEED)
    returns = rng.normal(loc=0.0, scale=0.01, size=bar_count)
    return _panel_from_returns(returns)


def test_seeded_iid_panel_is_gapless_complete_and_intrabar() -> None:
    bar_count = 500
    panel = _iid_noise_panel(bar_count)
    open_ = panel.field("open")["NOISEUSDT"]
    close = panel.field("close")["NOISEUSDT"]
    actual_returns = (
        close.to_numpy(dtype=float) / open_.to_numpy(dtype=float) - 1.0
    )
    expected_returns = np.random.default_rng(NOISE_SEED).normal(
        loc=0.0,
        scale=0.01,
        size=bar_count,
    )

    np.testing.assert_allclose(
        open_.iloc[1:].to_numpy(dtype=float),
        close.iloc[:-1].to_numpy(dtype=float),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        actual_returns,
        expected_returns,
        rtol=0.0,
        atol=1e-15,
    )
    assert (panel.field("quote_volume") > 0.0).all().all()
    assert panel.universe_mask().all().all()


@dataclass(frozen=True)
class NonFiniteWeightSignal:
    value: float

    def compute(self, panel: Panel) -> pd.DataFrame:
        close = panel.field("close")
        return pd.DataFrame(self.value, index=close.index, columns=close.columns)


def _engine_module() -> ModuleType:
    return importlib.import_module("cq.backtest.engine")


def _run_backtest(panel: Panel, signal: Signal) -> object:
    engine = _engine_module()
    run_backtest: Callable[..., object] = engine.run_backtest
    return run_backtest(
        panel=panel,
        signal=signal,
        starting_equity=STARTING_EQUITY,
    )


def _result_frame(result: object, name: str) -> pd.DataFrame:
    value = getattr(result, name)
    assert isinstance(value, pd.DataFrame), f"result.{name} must be a DataFrame"
    return value


def _result_series(result: object, name: str) -> pd.Series:
    value = getattr(result, name)
    assert isinstance(value, pd.Series), f"result.{name} must be a Series"
    return value


def _annualized_sharpe(returns: pd.Series) -> float:
    observed = returns.dropna()
    assert len(observed) >= NOISE_BARS - 2
    volatility = float(observed.std(ddof=1))
    assert volatility > 0.0
    return float(np.sqrt(ANNUALIZATION) * observed.mean() / volatility)


def _assert_byte_and_value_identical(
    historical: pd.DataFrame,
    extended_history: pd.DataFrame,
) -> None:
    pd.testing.assert_frame_equal(
        historical,
        extended_history,
        check_exact=True,
        check_dtype=True,
    )
    assert historical.to_numpy().tobytes() == extended_history.to_numpy().tobytes()


def _through_timestamp(value: pd.Series | pd.DataFrame, end: int) -> object:
    if isinstance(value, pd.DataFrame) and "timestamp" in value.columns:
        return value.loc[value["timestamp"] <= end].reset_index(drop=True)
    return value.loc[value.index <= end]


@pytest.mark.parametrize(
    "signal",
    [
        ConstantWeightSignal(weight=0.75),
        TrailingMomentumSignal(lookback=20),
    ],
    ids=["constant", "trailing-momentum"],
)
def test_honest_signal_weights_are_future_extension_invariant(
    signal: Signal,
) -> None:
    panel = _iid_noise_panel(HISTORY_BARS + EXTENSION_BARS)
    historical = panel.slice(0, (HISTORY_BARS - 1) * DAY_MS)
    extended = panel.slice(
        0,
        (HISTORY_BARS + EXTENSION_BARS - 1) * DAY_MS,
    )

    historical_weights = signal.compute(historical)
    extended_history = signal.compute(extended).loc[historical_weights.index]

    assert len(historical_weights) == HISTORY_BARS
    _assert_byte_and_value_identical(historical_weights, extended_history)


def test_cheating_next_close_signal_violates_future_extension_invariance() -> None:
    panel = _iid_noise_panel(HISTORY_BARS + EXTENSION_BARS)
    historical = panel.slice(0, (HISTORY_BARS - 1) * DAY_MS)
    extended = panel.slice(
        0,
        (HISTORY_BARS + EXTENSION_BARS - 1) * DAY_MS,
    )
    signal = CheatingNextCloseSignal()

    historical_weights = signal.compute(historical)
    extended_history = signal.compute(extended).loc[historical_weights.index]
    changed = historical_weights.ne(extended_history).any(axis=1)

    assert changed.sum() == 1
    assert changed.index[changed][0] == (HISTORY_BARS - 1) * DAY_MS
    assert historical_weights.iloc[-1, 0] == 0.0
    assert abs(extended_history.iloc[-1, 0]) == 1.0
    assert historical_weights.to_numpy().tobytes() != extended_history.to_numpy().tobytes()


def test_honest_engine_outputs_are_future_extension_invariant() -> None:
    """Adding 100 future bars cannot alter any honest result through T."""
    panel = _iid_noise_panel(HISTORY_BARS + EXTENSION_BARS)
    historical = panel.slice(0, (HISTORY_BARS - 1) * DAY_MS)
    extended = panel.slice(
        0,
        (HISTORY_BARS + EXTENSION_BARS - 1) * DAY_MS,
    )
    signal = TrailingMomentumSignal(lookback=20)

    historical_result = _run_backtest(historical, signal)
    extended_result = _run_backtest(extended, signal)
    end = (HISTORY_BARS - 1) * DAY_MS

    for name in ("equity", "gross_equity", "net_returns", "gross_returns"):
        expected = _result_series(historical_result, name)
        actual = _through_timestamp(_result_series(extended_result, name), end)
        pd.testing.assert_series_equal(actual, expected, check_exact=True)
    for name in ("positions", "fills", "trades"):
        expected = _result_frame(historical_result, name)
        actual = _through_timestamp(_result_frame(extended_result, name), end)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_production_contains_exactly_one_canonical_execution_shift() -> None:
    sources = [
        path.read_text(encoding="utf-8")
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
    ]
    source = "\n".join(sources)

    assert source.count(".shift(1)") == 1
    numeric_shifts = re.findall(
        r"\.shift\s*\(\s*(?:periods\s*=\s*)?[+-]?\d+\s*\)",
        source,
    )
    assert numeric_shifts == [".shift(1)"], (
        "production must contain one canonical one-bar delay and no alternate "
        "positive/negative numeric shifts"
    )


def test_shifted_weight_is_converted_to_quantity_at_next_open() -> None:
    panel = _panel_from_returns(np.zeros(3))
    signal = ConstantWeightSignal(weight=0.5)

    result = _run_backtest(panel, signal)
    positions = _result_frame(result, "positions")
    fills = _result_frame(result, "fills")

    # weight .5 * prior net equity 1000 / next open 100 = 5 tokens.
    assert fills.iloc[0, 0] == 0.0
    assert positions.iloc[0, 0] == 0.0
    assert fills.iloc[1, 0] == pytest.approx(5.0)
    assert positions.iloc[1, 0] == pytest.approx(5.0)


@pytest.mark.parametrize("signal_row", [0, 1, 2])
def test_weight_emission_never_fills_until_following_bar(signal_row: int) -> None:
    panel = _panel_from_returns(np.zeros(5))
    close = panel.field("close")
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    weights.iloc[signal_row, 0] = 0.25
    signal = ScheduledWeightSignal(weights)

    result = _run_backtest(panel, signal)
    positions = _result_frame(result, "positions")
    trades = _result_frame(result, "trades")
    signal_ts = int(weights.index[signal_row])
    fill_ts = int(weights.index[signal_row + 1])

    assert positions.iloc[signal_row, 0] == 0.0
    assert not (trades["timestamp"] == signal_ts).any()
    buy = trades.loc[
        (trades["timestamp"] == fill_ts) & (trades["side"] == "buy")
    ]
    assert len(buy) == 1
    assert buy.iloc[0]["quantity"] == pytest.approx(2.5)
    assert positions.iloc[signal_row + 1, 0] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "signal",
    [
        ConstantWeightSignal(),
        TrailingMomentumSignal(lookback=20),
    ],
    ids=["constant", "trailing-momentum"],
)
def test_honest_strategies_have_near_zero_sharpe_on_seeded_iid_noise(
    signal: Signal,
) -> None:
    result = _run_backtest(_iid_noise_panel(), signal)

    sharpe = _annualized_sharpe(_result_series(result, "gross_returns"))

    assert abs(sharpe) < 1.0


def test_cheating_signal_earns_impossible_returns_on_seeded_iid_noise() -> None:
    result = _run_backtest(_iid_noise_panel(), CheatingNextCloseSignal())
    gross_returns = _result_series(result, "gross_returns")
    observed = gross_returns.dropna()

    assert len(observed) >= NOISE_BARS - 2
    assert (observed.iloc[1:] >= 0.0).all()
    assert observed.iloc[1:].gt(0.0).mean() > 0.999
    assert _annualized_sharpe(gross_returns) > 15.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_engine_rejects_nonfinite_target_weights(value: float) -> None:
    with pytest.raises(ValueError, match="finite|weight"):
        _run_backtest(_panel_from_returns(np.zeros(3)), NonFiniteWeightSignal(value))

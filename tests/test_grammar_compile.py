"""End-to-end contracts for compiling hypothesis-shaped objects to signals."""

import builtins
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from cq.grammar.compiler import compile_signal
from fixtures.grammar import (
    HISTORY_PERIODS,
    SAMPLE_HYPOTHESES,
    GrammarHypothesis,
    assert_aligned,
    finite_cells,
    has_finite_cells,
    make_grammar_panel,
)

from cq.data.panel import Panel
from cq.grammar import GrammarError

RESEARCH_ROOT = Path(__file__).resolve().parents[1] / "cq" / "research"


@pytest.fixture
def panel() -> Panel:
    return make_grammar_panel()


def _weights(hypothesis: object, panel: Panel) -> pd.DataFrame:
    signal = compile_signal(hypothesis)
    assert callable(getattr(signal, "compute", None))
    weights = signal.compute(panel)
    assert isinstance(weights, pd.DataFrame)
    assert_aligned(weights, panel)
    return weights


def test_compiler_accepts_a_minimal_hypothesis_protocol(panel: Panel) -> None:
    hypothesis = SimpleNamespace(
        entry_rule="close > open",
        exit_rule="volume <= 0",
        data_required=("close", "open", "volume"),
        mode="long_only",
    )

    weights = _weights(hypothesis, panel)

    assert weights.notna().eq(panel.universe_mask()).all().all()


@pytest.mark.parametrize(
    ("mode", "expected_sum"),
    [
        ("long_only", 1.0),
        ("short_only", -1.0),
    ],
)
def test_directional_rows_are_normalized_when_any_symbol_is_selected(
    panel: Panel,
    mode: str,
    expected_sum: float,
) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="xs_rank(close) > 0.5",
        exit_rule="close < 0",
        data_required=("close",),
        mode=mode,
    )

    weights = _weights(hypothesis, panel)
    available = weights.where(panel.universe_mask())
    selected_rows = available.abs().sum(axis=1) > 0

    assert selected_rows.any()
    np.testing.assert_allclose(
        available.loc[selected_rows].sum(axis=1),
        expected_sum,
        rtol=0.0,
        atol=1e-12,
    )
    if mode == "long_only":
        assert (finite_cells(available) >= 0.0).all()
    else:
        assert (finite_cells(available) <= 0.0).all()


def test_market_neutral_rows_have_zero_net_and_both_sides(panel: Panel) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="xs_rank(close) > 0.5",
        exit_rule="close < 0",
        data_required=("close",),
        mode="market_neutral",
    )

    weights = _weights(hypothesis, panel).where(panel.universe_mask())
    active_rows = weights.abs().sum(axis=1) > 0

    assert active_rows.any()
    np.testing.assert_allclose(
        weights.loc[active_rows].sum(axis=1),
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    assert weights.loc[active_rows].gt(0.0).any(axis=1).all()
    assert weights.loc[active_rows].lt(0.0).any(axis=1).all()


@pytest.mark.parametrize("mode", ["long_only", "market_neutral", "short_only"])
def test_unavailable_symbols_are_nan_not_zero(panel: Panel, mode: str) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="close > 0",
        exit_rule="close < 0",
        data_required=("close",),
        mode=mode,
    )

    weights = _weights(hypothesis, panel)
    unavailable = ~panel.universe_mask()

    assert unavailable.any().any()
    assert not has_finite_cells(weights.where(unavailable))
    assert (weights.notna() | unavailable).all().all()


def test_exit_rule_vetoes_an_entry_on_the_same_row(panel: Panel) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="close > 0",
        exit_rule="close > 0",
        data_required=("close",),
    )

    weights = _weights(hypothesis, panel)
    available = weights.where(panel.universe_mask())

    assert (finite_cells(available) == 0.0).all()


def test_signal_emits_current_row_weights_without_an_execution_shift(
    panel: Panel,
) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="close > 16",
        exit_rule="close < 0",
        data_required=("close",),
    )

    weights = _weights(hypothesis, panel)
    mask = panel.universe_mask()
    expected_selected = (panel.field("close") > 16).where(mask)
    actual_selected = weights.gt(0.0).where(mask)

    pd.testing.assert_frame_equal(actual_selected, expected_selected)


@pytest.mark.parametrize(
    ("entry_rule", "exit_rule", "data_required"),
    [
        ("close > open", "close < 0", ("close",)),
        ("close > 0", "volume < 0", ("close",)),
        ("close > 0", "close < 0", ("close", "volume")),
        ("close > 0", "close < 0", ("price",)),
        ("close > 0", "close < 0", ("close", "close")),
    ],
)
def test_data_required_must_exactly_match_unique_rule_columns(
    entry_rule: str,
    exit_rule: str,
    data_required: tuple[str, ...],
) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule=entry_rule,
        exit_rule=exit_rule,
        data_required=data_required,
    )

    with pytest.raises(GrammarError):
        compile_signal(hypothesis)


def test_missing_required_panel_field_fails_at_compute_time() -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="open_interest > 0",
        exit_rule="open_interest < 0",
        data_required=("open_interest",),
    )
    panel = make_grammar_panel()
    without_open_interest = Panel(
        panel.data.drop(columns="open_interest", level="field"),
        panel.market_type,
    )

    signal = compile_signal(hypothesis)
    with pytest.raises(GrammarError):
        signal.compute(without_open_interest)


@pytest.mark.parametrize(
    "hypothesis",
    [
        SimpleNamespace(
            exit_rule="close < 0",
            data_required=("close",),
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule="close > 0",
            data_required=("close",),
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule="close > 0",
            exit_rule="close < 0",
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule=1,
            exit_rule="close < 0",
            data_required=("close",),
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule="close > 0",
            exit_rule=None,
            data_required=("close",),
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule="close > 0",
            exit_rule="close < 0",
            data_required="close",
            mode="long_only",
        ),
        SimpleNamespace(
            entry_rule="close > 0",
            exit_rule="close < 0",
            data_required=("close",),
            mode="leveraged",
        ),
    ],
)
def test_malformed_hypothesis_objects_are_rejected(hypothesis: object) -> None:
    with pytest.raises(GrammarError):
        compile_signal(hypothesis)


def test_malformed_entry_and_exit_rules_are_both_compiled() -> None:
    bad_entry = GrammarHypothesis(
        entry_rule="close >>> 1",
        exit_rule="close < 0",
        data_required=("close",),
    )
    bad_exit = GrammarHypothesis(
        entry_rule="close > 0",
        exit_rule="lead(close, 1) > 0",
        data_required=("close",),
    )

    with pytest.raises(GrammarError):
        compile_signal(bad_entry)
    with pytest.raises(GrammarError):
        compile_signal(bad_exit)


def test_compilation_and_compute_are_deterministic_and_do_no_io(
    panel: Panel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hypothesis = GrammarHypothesis(
        entry_rule="xs_rank(roll_z(close, 3)) > 0.5",
        exit_rule="volume <= 0",
        data_required=("close", "volume"),
        mode="market_neutral",
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("grammar compilation and signals must not use I/O or RNG")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)

    first = _weights(hypothesis, panel)
    second = _weights(hypothesis, panel)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_research_package_has_no_handwritten_signal_implementations() -> None:
    offenders = [
        path.relative_to(RESEARCH_ROOT)
        for path in RESEARCH_ROOT.rglob("*.py")
        if "def compute(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_compiled_signal_is_future_extension_invariant() -> None:
    hypothesis = GrammarHypothesis(
        entry_rule=(
            "(xs_rank(roll_z(close, 3)) > 0.5) "
            "and (lag(volume, 1) > 0)"
        ),
        exit_rule="funding_8h == 0",
        data_required=("close", "funding_8h", "volume"),
        mode="market_neutral",
    )
    historical = make_grammar_panel()
    extended = make_grammar_panel(
        HISTORY_PERIODS + 3,
        extreme_future=True,
    )

    expected = _weights(hypothesis, historical)
    actual = _weights(hypothesis, extended).loc[
        expected.index,
        expected.columns,
    ]

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize(
    "hypothesis",
    SAMPLE_HYPOTHESES,
    ids=lambda hypothesis: hypothesis.entry_rule,
)
def test_twenty_representative_hypotheses_compile_and_run(
    panel: Panel,
    hypothesis: GrammarHypothesis,
) -> None:
    weights = _weights(hypothesis, panel)
    available = weights.where(panel.universe_mask())
    active_rows = available.abs().sum(axis=1) > 0

    assert len(SAMPLE_HYPOTHESES) == 20
    assert np.isfinite(finite_cells(available)).all()
    if hypothesis.mode == "long_only":
        np.testing.assert_allclose(
            available.loc[active_rows].sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    elif hypothesis.mode == "short_only":
        np.testing.assert_allclose(
            available.loc[active_rows].sum(axis=1),
            -1.0,
            rtol=0.0,
            atol=1e-12,
        )
    else:
        np.testing.assert_allclose(
            available.loc[active_rows].sum(axis=1),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )

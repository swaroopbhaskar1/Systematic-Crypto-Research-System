"""Every hypothesis test is counted once, including abandoned work."""

from pathlib import Path

import pytest
from fixtures.research import CountableHypothesis

from cq.backtest.engine import BacktestResult


HYPOTHESIS = CountableHypothesis(
    id="funding_extreme_001",
    entry_rule="funding_8h > roll_pct(funding_8h, 270, 0.95)",
    exit_rule="funding_8h < roll_pct(funding_8h, 270, 0.50)",
    universe_filter="top 100 by 30d median quote volume, excluding stablecoins",
    direction="market_neutral",
)


def test_content_hash_ignores_hypothesis_id() -> None:
    from cq.research.log import content_hash

    renamed = CountableHypothesis(
        id="totally_new_name",
        entry_rule=HYPOTHESIS.entry_rule,
        exit_rule=HYPOTHESIS.exit_rule,
        universe_filter=HYPOTHESIS.universe_filter,
        direction=HYPOTHESIS.direction,
    )
    assert content_hash(HYPOTHESIS) == content_hash(renamed)
    mutated = CountableHypothesis(
        id=HYPOTHESIS.id,
        entry_rule=HYPOTHESIS.entry_rule,
        exit_rule="funding_8h < 0",
        universe_filter=HYPOTHESIS.universe_filter,
        direction=HYPOTHESIS.direction,
    )
    assert content_hash(HYPOTHESIS) != content_hash(mutated)


def test_record_refuses_to_double_count_identical_rules(tmp_path: Path) -> None:
    from cq.research.log import DuplicateHypothesisError, record, test_count

    path = tmp_path / "counting.jsonl"
    record(HYPOTHESIS, None, "abandoned", path=path)
    with pytest.raises(DuplicateHypothesisError):
        record(HYPOTHESIS, None, "tested", path=path)
    assert test_count(path) == 1


def test_killed_and_abandoned_outcomes_still_increment_count(
    tmp_path: Path,
) -> None:
    from cq.research.log import record, test_count

    path = tmp_path / "counting.jsonl"
    first = CountableHypothesis(
        id="a",
        entry_rule="close > 0",
        exit_rule="close < 0",
        universe_filter="all",
        direction="long_only",
    )
    second = CountableHypothesis(
        id="b",
        entry_rule="funding_8h > 0",
        exit_rule="funding_8h < 0",
        universe_filter="all",
        direction="short_only",
    )
    record(first, None, "killed_by_critic", path=path)
    record(second, None, "abandoned", path=path)
    assert test_count(path) == 2


def test_counting_log_persists_across_sessions(tmp_path: Path) -> None:
    from cq.research.log import load_records, record, test_count

    path = tmp_path / "counting.jsonl"
    record(HYPOTHESIS, None, "tested", path=path)
    assert test_count(path) == 1
    loaded = load_records(path)
    assert loaded[0]["hypothesis_id"] == HYPOTHESIS.id
    assert loaded[0]["outcome"] == "tested"
    assert loaded[0]["result"] is None


def test_invalid_outcome_is_rejected(tmp_path: Path) -> None:
    from cq.research.log import record

    path = tmp_path / "counting.jsonl"
    with pytest.raises(ValueError):
        record(HYPOTHESIS, None, "looks_promising", path=path)


def test_backtest_result_is_serialized_without_mutating_log_schema(
    tmp_path: Path,
) -> None:
    from cq.research.log import load_records, record

    path = tmp_path / "counting.jsonl"
    result = _empty_result()
    record(HYPOTHESIS, result, "tested", path=path)
    loaded = load_records(path)
    assert loaded[0]["result"]["hypothesis_id"] == result.hypothesis_id
    assert loaded[0]["result"]["n_trades"] == 0


def _empty_result() -> BacktestResult:
    import pandas as pd

    empty = pd.Series(dtype=float)
    frame = pd.DataFrame()
    return BacktestResult(
        equity=empty,
        gross_equity=empty,
        cash=empty,
        positions=frame,
        fills=frame,
        trades=frame,
        metrics={},
        regime_metrics={},
        pct_bars_capped=0.0,
        n_trades=0,
        hypothesis_id="funding_extreme_001",
        data_span=pd.Timedelta(0),
        config_hash="abc",
    )

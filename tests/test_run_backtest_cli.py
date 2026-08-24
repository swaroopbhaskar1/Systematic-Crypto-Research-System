"""End-to-end tests for the backtest entrypoint.

A stubbed entrypoint that raises instead of running is worse than a missing
one: the milestone looks complete and nothing was ever measured.  These
tests drive the real path, store to panel to engine to report, and defend
the two properties that make the CLI safe to run: it cannot silently touch
the holdout, and it cannot silently retest the same rules twice.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from cq.backtest.pipeline import BacktestRequest, run_from_store
from cq.data.store import ParquetStore
from cq.research.holdout import HoldoutLockedError
from cq.research.log import DuplicateHypothesisError, load_records
from cq.research.report import render_backtest_result

COSTS = Path(__file__).resolve().parents[1] / "config" / "costs.yaml"
SYMBOLS = ("AAAUSDT", "BBBUSDT")
BAR_COUNT = 60

HYPOTHESIS = {
    "id": "cli_smoke_001",
    "mechanism_class": "carry",
    "forced_participant": "perpetual longs paying funding at an extreme trailing percentile",
    "why_forced": (
        "Leveraged perpetual longs must pay funding every eight hours regardless "
        "of their price view, so crowded positioning persists past the point the "
        "carry cost justifies."
    ),
    "universe_filter": "deterministic test fixture universe",
    "entry_rule": "funding_8h > roll_pct(funding_8h, 5, 0.80)",
    "exit_rule": "funding_8h < roll_pct(funding_8h, 5, 0.50)",
    "direction": "market_neutral",
    "holding_bars": 3,
    "testable_prediction": "the funding-paying side underperforms after costs",
    "expected_capacity_usd": 50000.0,
    "competition_risk": "medium",
    "data_required": ["funding_8h"],
}


def bars(start: str) -> pd.DataFrame:
    """Build a store-shaped perp frame with varied prices and funding."""
    stamps = pd.date_range(start, periods=BAR_COUNT, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for position, stamp in enumerate(stamps):
            close = 100.0 + symbol_index * 40.0 + (position % 7) * 3.0
            milliseconds = int(stamp.value // 1_000_000)
            rows.append(
                {
                    "ts": milliseconds,
                    "symbol": symbol,
                    "market_type": "perp",
                    "open": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 50_000.0 + position * 10.0,
                    "quote_volume": 8_000_000.0 + symbol_index * 1_000_000.0,
                    "funding_8h": 0.0001 * ((position % 11) - 5),
                    "in_universe": True,
                    "asof": milliseconds,
                }
            )
    return pd.DataFrame(rows)


def build_store(root: Path, start: str = "2024-01-01") -> Path:
    """Write one immutable partition and return the store root."""
    ParquetStore(root).write(bars(start), timeframe="1d")
    return root


def write_hypothesis(path: Path, **overrides: object) -> Path:
    payload = dict(HYPOTHESIS)
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def request_for(tmp_path: Path, **overrides: object) -> BacktestRequest:
    defaults: dict[str, object] = {
        "hypothesis_path": write_hypothesis(tmp_path / "hypothesis.json"),
        "store_root": build_store(tmp_path / "store"),
        "costs_path": COSTS,
        "timeframe": "1d",
        "market_type": "perp",
        "asof": int(pd.Timestamp("2030-01-01", tz="UTC").value // 1_000_000),
        "starting_equity": 100_000.0,
        "count_log": tmp_path / "counting.jsonl",
    }
    defaults.update(overrides)
    return BacktestRequest(**defaults)  # pyright: ignore[reportArgumentType]


class TestPipeline:
    def test_the_pipeline_actually_produces_a_result(self, tmp_path: Path) -> None:
        """The whole path must run: store, features, panel, engine, metrics."""
        hypothesis, result = run_from_store(request_for(tmp_path))
        assert hypothesis.id == "cli_smoke_001"
        assert result.hypothesis_id == "cli_smoke_001"
        assert len(result.equity) == BAR_COUNT
        assert result.metrics["n_bars"] == BAR_COUNT

    def test_gross_and_net_are_both_measured(self, tmp_path: Path) -> None:
        _, result = run_from_store(request_for(tmp_path))
        assert "gross_return" in result.metrics
        assert "net_return" in result.metrics

    def test_costs_come_from_the_config_file(self, tmp_path: Path) -> None:
        """The run must charge the configured perp taker fee, not a default.

        config/costs.yaml sets the perp taker fee to 5 bps and the impact
        coefficient to 0.6, so the config hash must differ from a run using
        the engine's zero-fee fallback model.
        """
        from cq.backtest.costs import CostConfig
        from cq.backtest.engine import run

        request = request_for(tmp_path)
        _, configured = run_from_store(request)
        model = CostConfig.from_yaml(COSTS).cost_model("perp")
        assert model.taker_bps == 5.0
        assert model.impact_coefficient == 0.6
        from cq.backtest.pipeline import load_panel
        from cq.grammar.compile import compile_signal
        from cq.research.schema import Hypothesis

        panel = load_panel(request)
        hypothesis = Hypothesis.model_validate_json(
            request.hypothesis_path.read_text(encoding="utf-8")
        )
        fallback = run(panel, compile_signal(hypothesis))
        assert configured.config_hash != fallback.config_hash


class TestHoldoutProtection:
    def test_holdout_era_bars_are_refused(self, tmp_path: Path) -> None:
        """The CLI must not be a side door into the locked holdout.

        Holdout access is one-shot and gated elsewhere; a research CLI that
        happily backtests post-holdout bars burns that shot silently.
        """
        request = request_for(
            tmp_path,
            store_root=build_store(tmp_path / "holdout_store", start="2025-08-01"),
        )
        with pytest.raises(HoldoutLockedError):
            run_from_store(request)


class TestCounting:
    def test_a_run_is_counted(self, tmp_path: Path) -> None:
        """Every backtest is a test and must appear in the counting log."""
        request = request_for(tmp_path)
        run_from_store(request)
        records = load_records(request.count_log)
        assert len(records) == 1
        assert records[0]["outcome"] == "tested"
        assert records[0]["hypothesis_id"] == "cli_smoke_001"

    def test_identical_rules_under_a_new_name_are_refused(
        self, tmp_path: Path
    ) -> None:
        """Renaming a hypothesis must not buy a fresh look at the data.

        This is the multiple-testing defence.  If the CLI lets a rename slip
        through, the adjusted Sharpe threshold is computed from a count that
        understates how many times the data was actually interrogated.
        """
        request = request_for(tmp_path)
        run_from_store(request)
        renamed = request_for(
            tmp_path,
            hypothesis_path=write_hypothesis(
                tmp_path / "renamed.json",
                id="cli_smoke_002",
            ),
            store_root=request.store_root,
            count_log=request.count_log,
        )
        with pytest.raises(DuplicateHypothesisError):
            run_from_store(renamed)
        assert len(load_records(request.count_log)) == 1

    def test_counting_can_be_skipped_for_a_dry_run(self, tmp_path: Path) -> None:
        """Opting out must be explicit and must write nothing."""
        request = request_for(tmp_path, count_log=None)
        run_from_store(request)
        assert not (tmp_path / "counting.jsonl").exists()


class TestEmptyStore:
    def test_an_empty_store_raises_rather_than_reporting_zero(
        self, tmp_path: Path
    ) -> None:
        """A backtest over no data must fail loudly, not print a flat curve."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError):
            run_from_store(request_for(tmp_path, store_root=empty))

    def test_a_missing_market_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            run_from_store(request_for(tmp_path, market_type="spot"))


class TestRendering:
    def test_report_shows_gross_and_net_side_by_side(self, tmp_path: Path) -> None:
        """Spec 7.3: the gap between gross and net is the headline number."""
        hypothesis, result = run_from_store(request_for(tmp_path))
        rendered = render_backtest_result(result, hypothesis_id=hypothesis.id)
        assert "gross" in rendered.lower()
        assert "net" in rendered.lower()
        assert "cli_smoke_001" in rendered

    def test_report_states_the_capacity_and_trade_counts(
        self, tmp_path: Path
    ) -> None:
        """pct_bars_capped is a finding, so it must always be visible."""
        hypothesis, result = run_from_store(request_for(tmp_path))
        rendered = render_backtest_result(result, hypothesis_id=hypothesis.id)
        assert "capped" in rendered.lower()
        assert "trades" in rendered.lower()

    def test_undefined_metrics_render_as_not_available(
        self, tmp_path: Path
    ) -> None:
        """An omitted metric must read as unmeasured, never as zero."""
        hypothesis, result = run_from_store(request_for(tmp_path))
        rendered = render_backtest_result(result, hypothesis_id=hypothesis.id)
        for line in rendered.splitlines():
            assert "0.00 (undefined)" not in line

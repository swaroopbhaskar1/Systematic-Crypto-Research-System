"""Append-only, content-addressed hypothesis counting log."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Protocol

from cq.backtest.engine import BacktestResult

Outcome = Literal["tested", "abandoned", "killed_by_critic"]
OUTCOMES = frozenset({"tested", "abandoned", "killed_by_critic"})
DEFAULT_LOG_PATH = Path("research") / "log" / "counting.jsonl"


class DuplicateHypothesisError(Exception):
    """Raised when an identical rule set would be counted twice."""


class Countable(Protocol):
    """Fields that identify a hypothesis independently of its display name."""

    id: str
    entry_rule: str
    exit_rule: str
    universe_filter: str
    direction: str


def content_hash(hypothesis: Countable) -> str:
    """Hash the rule identity, ignoring the hypothesis display id."""
    payload = {
        "direction": hypothesis.direction,
        "entry_rule": hypothesis.entry_rule,
        "exit_rule": hypothesis.exit_rule,
        "universe_filter": hypothesis.universe_filter,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record(
    hypothesis: Countable,
    result: BacktestResult | None,
    outcome: str,
    *,
    path: Path | None = None,
) -> None:
    """Append one counted outcome; refuse duplicate rule identities."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unsupported counting outcome: {outcome}")
    log_path = path or DEFAULT_LOG_PATH
    digest = content_hash(hypothesis)
    if digest in _existing_hashes(log_path):
        raise DuplicateHypothesisError("identical hypothesis already counted")
    entry = {
        "content_hash": digest,
        "hypothesis_id": hypothesis.id,
        "outcome": outcome,
        "result": None if result is None else _result_summary(result),
    }
    _append_jsonl(log_path, entry)


def load_records(path: Path | None = None) -> list[dict[str, object]]:
    """Return every counting-log record in append order."""
    log_path = path or DEFAULT_LOG_PATH
    if not log_path.exists():
        return []
    records: list[dict[str, object]] = []
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        if raw:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("counting log entries must be objects")
            records.append(parsed)
    return records


def test_count(path: Path | None = None) -> int:
    """Return the number of counted hypothesis outcomes."""
    return len(load_records(path))


def _existing_hashes(path: Path) -> set[str]:
    hashes: set[str] = set()
    for record in load_records(path):
        digest = record.get("content_hash")
        if not isinstance(digest, str):
            raise ValueError("counting log is missing content_hash")
        hashes.add(digest)
    return hashes


def _result_summary(result: BacktestResult) -> dict[str, object]:
    return {
        "config_hash": result.config_hash,
        "hypothesis_id": result.hypothesis_id,
        "n_trades": result.n_trades,
        "pct_bars_capped": result.pct_bars_capped,
    }


def _append_jsonl(path: Path, entry: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(entry, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
